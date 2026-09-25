"""Misiones diarias, XP y nivel del coleccionista."""
from __future__ import annotations

from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pymongo import ReturnDocument

from app.db import get_db, oid, serialize
from app.schemas_social import MissionOut, MissionProgress, MissionStateOut
from app.security import CurrentUser, get_current_user

router = APIRouter(prefix="/missions", tags=["missions"])

LEVEL_THRESHOLDS = [0, 100, 250, 500, 800, 1200, 1800, 2500, 3500, 5000]

DAILY_MISSION_TEMPLATES = [
    {
        "title": "Escanea una pieza",
        "description": "Usa el escáner para identificar un auto",
        "icon": "📷",
        "xp_reward": 25,
        "type": "scan",
        "target": 1,
    },
    {
        "title": "Actualiza tu wishlist",
        "description": "Agrega una pieza que quieras conseguir",
        "icon": "⭐",
        "xp_reward": 20,
        "type": "wishlist",
        "target": 1,
    },
    {
        "title": "Explora el catálogo",
        "description": "Revisa al menos 5 piezas del catálogo global",
        "icon": "🔍",
        "xp_reward": 15,
        "type": "explore",
        "target": 5,
    },
    {
        "title": "Compara valores",
        "description": "Consulta la valoración de una pieza",
        "icon": "📊",
        "xp_reward": 20,
        "type": "compare",
        "target": 1,
    },
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _today() -> str:
    return date.today().isoformat()


def _level_for(xp: int) -> tuple[int, int, int]:
    """→ (level, xp_for_next, next_threshold)."""
    level = 1
    for i, th in enumerate(LEVEL_THRESHOLDS):
        if xp >= th:
            level = i + 1
    next_th = LEVEL_THRESHOLDS[level] if level < len(LEVEL_THRESHOLDS) else LEVEL_THRESHOLDS[-1]
    return level, max(0, next_th - xp), next_th


def _ensure_daily_missions(db, user_id: str) -> list[dict]:
    today = _today()
    existing = list(db.missions.find({"user_id": user_id, "date": today}))
    if existing:
        return existing
    now = _now()
    docs = []
    for tpl in DAILY_MISSION_TEMPLATES:
        doc = {
            "user_id": user_id,
            "date": today,
            **tpl,
            "completed": False,
            "progress": 0.0,
            "current": 0,
            "created_at": now,
        }
        res = db.missions.insert_one(doc)
        doc["_id"] = res.inserted_id
        docs.append(doc)
    return docs


@router.get("/state", response_model=MissionStateOut)
def mission_state(user: CurrentUser = Depends(get_current_user)) -> MissionStateOut:
    db = get_db()
    missions = _ensure_daily_missions(db, user.id)
    # Progreso de XP persistido en el perfil
    profile = db.users.find_one({"_id": oid(user.id)}) or {}
    total_xp = int(profile.get("total_xp", 0))
    level, xp_next, next_th = _level_for(total_xp)
    return MissionStateOut(
        missions=[MissionOut(**serialize(d)) for d in missions],  # type: ignore[arg-type]
        streak=int(profile.get("streak", 0)),
        longest_streak=int(profile.get("longest_streak", 0)),
        total_xp=total_xp,
        level=level,
        xp_for_next_level=xp_next,
        next_level_threshold=next_th,
        last_active_date=profile.get("last_active_date"),
    )


@router.post("/{mission_id}/progress", response_model=MissionOut)
def increment_progress(
    mission_id: str,
    body: MissionProgress,
    user: CurrentUser = Depends(get_current_user),
) -> MissionOut:
    db = get_db()
    doc = db.missions.find_one({"_id": oid(mission_id), "user_id": user.id})
    if not doc:
        raise HTTPException(status_code=404, detail="Misión no encontrada")
    if doc.get("completed"):
        return MissionOut(**serialize(doc))  # type: ignore[arg-type]

    current = min(doc.get("target", 1), int(doc.get("current", 0)) + body.increment)
    progress = current / max(1, doc.get("target", 1))
    doc = db.missions.find_one_and_update(
        {"_id": oid(mission_id)},
        {"$set": {"current": current, "progress": round(progress, 3)}},
        return_document=ReturnDocument.AFTER,
    )
    return MissionOut(**serialize(doc))  # type: ignore[arg-type]


@router.post("/{mission_id}/claim", response_model=MissionOut)
def claim_mission(
    mission_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> MissionOut:
    db = get_db()
    doc = db.missions.find_one({"_id": oid(mission_id), "user_id": user.id})
    if not doc:
        raise HTTPException(status_code=404, detail="Misión no encontrada")
    if doc.get("completed"):
        raise HTTPException(status_code=409, detail="Misión ya completada")
    if int(doc.get("current", 0)) < int(doc.get("target", 1)):
        raise HTTPException(status_code=409, detail="Misión incompleta")

    doc = db.missions.find_one_and_update(
        {"_id": oid(mission_id)},
        {"$set": {"completed": True, "progress": 1.0}},
        return_document=ReturnDocument.AFTER,
    )
    xp = int(doc.get("xp_reward", 0))
    db.users.update_one(
        {"_id": oid(user.id)},
        {
            "$inc": {"total_xp": xp},
            "$set": {"last_active_date": _today(), "updated_at": _now()},
        },
    )
    return MissionOut(**serialize(doc))  # type: ignore[arg-type]
