"""Alertas y notificaciones del usuario."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.db import get_db, oid, serialize
from app.security import CurrentUser, get_current_user

router = APIRouter(prefix="/alerts", tags=["alerts"])


class AlertOut(BaseModel):
    id: str
    type: str
    title: str
    message: str
    icon: str = "🔔"
    read: bool = False
    created_at: str


def _now() -> datetime:
    return datetime.now(timezone.utc)


@router.get("", response_model=list[AlertOut])
def list_alerts(
    user: CurrentUser = Depends(get_current_user),
) -> list[AlertOut]:
    db = get_db()
    docs = list(db.alerts.find({"user_id": user.id}).sort("created_at", -1).limit(50))
    return [AlertOut(**serialize(d)) for d in docs]


@router.post("/{alert_id}/read")
def mark_alert_read(
    alert_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    db = get_db()
    db.alerts.update_one(
        {"_id": oid(alert_id), "user_id": user.id},
        {"$set": {"read": True}},
    )
    return {"ok": True}
