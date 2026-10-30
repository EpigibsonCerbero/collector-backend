"""Suscripción Premium y cuota diaria de escaneos.

El estado Premium vive en `subscriptions` (lo escribe el webhook de RevenueCat);
la cuota en `scan_usage` (un documento por usuario y día de México).
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from app.config import get_settings

MX_TZ = ZoneInfo("America/Mexico_City")


def _utc(dt: datetime) -> datetime:
    """Mongo devuelve fechas naive (UTC): normalizar antes de comparar."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _now(now: datetime | None) -> datetime:
    return _utc(now) if now else datetime.now(timezone.utc)


def today_key(now: datetime | None = None) -> str:
    """Día calendario (YYYY-MM-DD) en hora de México."""
    return _now(now).astimezone(MX_TZ).strftime("%Y-%m-%d")


def next_reset(now: datetime | None = None) -> datetime:
    """Próxima medianoche de México, en UTC."""
    local = _now(now).astimezone(MX_TZ)
    nxt = datetime.combine(local.date() + timedelta(days=1), time.min, tzinfo=MX_TZ)
    return nxt.astimezone(timezone.utc)


def _forced_premium(user_id: str) -> bool:
    ids = {x.strip() for x in get_settings().premium_user_ids.split(",") if x.strip()}
    return user_id in ids


def is_premium(db, user_id: str, now: datetime | None = None) -> bool:
    if _forced_premium(user_id):
        return True
    doc = db.subscriptions.find_one({"user_id": user_id})
    if not doc or not doc.get("premium") or doc.get("revoked"):
        return False
    exp = doc.get("expires_at")
    if exp is None:
        return True
    grace = timedelta(days=get_settings().premium_grace_days)
    return _now(now) <= _utc(exp) + grace


def get_status(db, user_id: str) -> dict[str, Any]:
    doc = db.subscriptions.find_one({"user_id": user_id}) or {}
    premium = is_premium(db, user_id)
    usage = db.scan_usage.find_one({"user_id": user_id, "day": today_key()}) or {}
    exp = doc.get("expires_at")
    return {
        "premium": premium,
        "plan": doc.get("period") if premium else None,
        "expires_at": _utc(exp) if exp else None,
        "will_renew": bool(doc.get("will_renew")) if premium else False,
        "scans_used": int(usage.get("count", 0)),
        "scans_limit": None if premium else get_settings().free_daily_scans,
        "resets_at": next_reset(),
    }


def upsert_subscription(
    db,
    user_id: str,
    *,
    premium: bool,
    product_id: str | None,
    period: str | None,
    expires_at: datetime | None,
    will_renew: bool,
    store: str | None,
    event_id: str | None,
    revoked: bool = False,
    event_ts_ms: int | None = None,
) -> None:
    fields: dict[str, Any] = {
        "premium": premium,
        "product_id": product_id,
        "period": period,
        "expires_at": expires_at,
        "will_renew": will_renew,
        "store": store,
        "last_event_id": event_id,
        "revoked": revoked,
        "updated_at": datetime.now(timezone.utc),
    }
    if event_ts_ms is not None:
        fields["last_event_ts_ms"] = event_ts_ms  # guarda de orden del webhook
    db.subscriptions.update_one(
        {"user_id": user_id},
        {"$set": fields},
        upsert=True,
    )


def reserve_scan(db, user_id: str) -> tuple[bool, int]:
    """Reserva un escaneo del día. Devuelve (permitido, usados tras reservar).

    Premium y cuota apagada pasan siempre sin tocar el contador.
    """
    st = get_settings()
    if not st.scan_limit_enabled or is_premium(db, user_id):
        usage = db.scan_usage.find_one({"user_id": user_id, "day": today_key()}) or {}
        return True, int(usage.get("count", 0))
    limit = st.free_daily_scans
    day = today_key()
    try:
        doc = db.scan_usage.find_one_and_update(
            {"user_id": user_id, "day": day, "count": {"$lt": limit}},
            {"$inc": {"count": 1}, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
    except DuplicateKeyError:
        # El documento del día existe y ya no cumple count < limit: tope alcanzado.
        return False, limit
    return True, int(doc["count"])


def release_scan(db, user_id: str) -> None:
    """Devuelve un escaneo reservado (el identify falló). Nunca baja de 0."""
    db.scan_usage.update_one(
        {"user_id": user_id, "day": today_key(), "count": {"$gt": 0}},
        {"$inc": {"count": -1}},
    )
