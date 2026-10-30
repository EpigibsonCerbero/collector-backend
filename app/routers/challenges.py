"""Retos de colección: se calculan con los DATOS REALES del usuario.

Antes la colección `challenges` no la llenaba nada (lista vacía para todos),
ningún reto se marcaba completado y el XP «reclamado» solo existía en el
teléfono. Ahora el progreso sale del garaje/wishlist/canjes y el XP se
acredita en el servidor una sola vez por reto.
"""
from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.db import as_id, get_db, oid
from app.security import CurrentUser, get_current_user

router = APIRouter(prefix="/challenges", tags=["challenges"])

_RARE = {"chase", "super_treasure_hunt", "treasure_hunt", "limited", "rare", "exclusive"}

# (id, título, descripción, icono, métrica, objetivo, XP)
CHALLENGES = [
    ("garage-10", "Primeras 10 piezas", "Registra 10 piezas en tu garaje", "car-sport-outline", "pieces", 10, 100),
    ("garage-50", "Garaje en serio", "Llega a 50 piezas en tu garaje", "albums-outline", "pieces", 50, 300),
    ("brands-5", "Coleccionista diverso", "Ten piezas de 5 marcas distintas", "globe-outline", "brands", 5, 150),
    ("rare-1", "Cazador de rarezas", "Consigue una pieza Chase o de edición limitada", "flame-outline", "rare", 1, 200),
    ("wishlist-10", "Mapa de búsqueda", "Guarda 10 piezas en tu lista de deseos", "heart-outline", "wishlist", 10, 50),
    ("trade-1", "Primer canje", "Cierra un canje con otro coleccionista", "swap-horizontal-outline", "trades", 1, 150),
]


class ChallengeOut(BaseModel):
    id: str
    title: str
    description: str
    icon: str = ""
    reward_xp: int = 50
    progress: float = 0.0
    target: int = 1
    completed: bool = False
    claimed: bool = False
    ends_at: str | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _metrics(db, user_id: str) -> dict[str, int]:
    items = list(db.user_items.find({"user_id": user_id}, {"item_variant_id": 1, "quantity": 1}))
    pieces = sum(int(i.get("quantity") or 1) for i in items)
    keys = []
    for i in items:
        if i.get("item_variant_id"):
            try:
                keys.append(as_id(str(i["item_variant_id"])))
            except Exception:  # noqa: BLE001
                keys.append(str(i["item_variant_id"]))
    variants = list(db.variants.find({"_id": {"$in": keys}}, {"rarity": 1, "item_master_id": 1})) if keys else []
    rare = sum(1 for v in variants if (v.get("rarity") or "").lower() in _RARE)
    mkeys = []
    for v in variants:
        if v.get("item_master_id"):
            try:
                mkeys.append(as_id(str(v["item_master_id"])))
            except Exception:  # noqa: BLE001
                mkeys.append(str(v["item_master_id"]))
    brands = {
        (m.get("manufacturer") or "").strip().lower()
        for m in (db.items.find({"_id": {"$in": mkeys}}, {"manufacturer": 1}) if mkeys else [])
    } - {""}
    wl = db.user_lists.find_one({"user_id": user_id, "list_type": "wishlist"}, {"_id": 1})
    wishlist = db.user_list_items.count_documents({"user_list_id": str(wl["_id"])}) if wl else 0
    offer_ids = [
        str(o["_id"])
        for o in db.trade_offers.find({"$or": [{"from_user_id": user_id}, {"to_user_id": user_id}]}, {"_id": 1})
    ]
    trades = (
        db.trade_escrows.count_documents({"offer_id": {"$in": offer_ids}, "status": {"$in": ["delivered", "completed"]}})
        if offer_ids
        else 0
    )
    return {"pieces": pieces, "brands": len(brands), "rare": rare, "wishlist": wishlist, "trades": trades}


def _build(db, user_id: str) -> list[ChallengeOut]:
    m = _metrics(db, user_id)
    claimed = {c["challenge_id"] for c in db.challenge_claims.find({"user_id": user_id}, {"challenge_id": 1})}
    out = []
    for cid, title, desc, icon, metric, target, xp in CHALLENGES:
        current = min(target, m.get(metric, 0))
        out.append(
            ChallengeOut(
                id=cid,
                title=title,
                description=desc,
                icon=icon,
                reward_xp=xp,
                progress=float(current),  # CONTEO (la app lo convierte a fracción)
                target=target,
                completed=current >= target,
                claimed=cid in claimed,
            )
        )
    return out


@router.get("", response_model=list[ChallengeOut])
def list_challenges(user: CurrentUser = Depends(get_current_user)) -> list[ChallengeOut]:
    return _build(get_db(), user.id)


@router.post("/{challenge_id}/claim", response_model=ChallengeOut)
def claim_challenge(challenge_id: str, user: CurrentUser = Depends(get_current_user)) -> ChallengeOut:
    db = get_db()
    ch = next((c for c in _build(db, user.id) if c.id == challenge_id), None)
    if not ch:
        raise HTTPException(status_code=404, detail="Reto no encontrado")
    if not ch.completed:
        raise HTTPException(status_code=409, detail="Aún no completas este reto")
    if ch.claimed:
        raise HTTPException(status_code=409, detail="Ya reclamaste este reto")
    try:
        db.challenge_claims.insert_one({"user_id": user.id, "challenge_id": challenge_id, "at": _now()})
    except Exception:  # índice único: doble toque simultáneo
        raise HTTPException(status_code=409, detail="Ya reclamaste este reto")
    db.users.update_one({"_id": oid(user.id)}, {"$inc": {"total_xp": ch.reward_xp}})
    return ch.model_copy(update={"claimed": True})


@router.post("/{challenge_id}/progress")
def update_challenge_progress(challenge_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
    """Compatibilidad con versiones viejas de la app: el progreso ya se calcula solo."""
    return {"ok": True}
