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


def _doc_premium(doc: dict | None, now: datetime | None = None) -> bool:
    if not doc or not doc.get("premium") or doc.get("revoked"):
        return False
    exp = doc.get("expires_at")
    if exp is None:
        return True
    grace = timedelta(days=get_settings().premium_grace_days)
    return _now(now) <= _utc(exp) + grace


def is_premium(db, user_id: str, now: datetime | None = None) -> bool:
    if _forced_premium(user_id):
        return True
    return _doc_premium(db.subscriptions.find_one({"user_id": user_id}), now)


def premium_state(db, user_id: str, now: datetime | None = None) -> tuple[bool, bool]:
    """(es premium, su vencimiento ya pasó y sigue dentro de la gracia). Una sola lectura."""
    if _forced_premium(user_id):
        return True, False
    doc = db.subscriptions.find_one({"user_id": user_id})
    premium = _doc_premium(doc, now)
    exp = (doc or {}).get("expires_at")
    return premium, bool(premium and exp is not None and _utc(exp) < _now(now))


def get_status(db, user_id: str) -> dict[str, Any]:
    doc = db.subscriptions.find_one({"user_id": user_id}) or {}
    premium = is_premium(db, user_id)
    usage = db.scan_usage.find_one({"user_id": user_id, "day": today_key()}) or {}
    exp = doc.get("expires_at")
    st = get_settings()
    return {
        "premium": premium,
        "plan": doc.get("period") if premium else None,
        "expires_at": _utc(exp) if exp else None,
        "will_renew": bool(doc.get("will_renew")) if premium else False,
        "scans_used": int(usage.get("count", 0)),
        # Sin cuota activa no hay tope que mostrar ni que vender.
        "scans_limit": None if (premium or not st.scan_limit_enabled) else st.free_daily_scans,
        "limit_enabled": bool(st.scan_limit_enabled),
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


def reserve_scan(db, user_id: str, skip_premium_check: bool = False) -> tuple[bool, int]:
    """Reserva un escaneo del día. Devuelve (permitido, usados tras reservar).

    Premium y cuota apagada pasan siempre sin tocar el contador.
    `skip_premium_check`: el llamador ya decidió que no es Premium (evita una lectura).
    """
    st = get_settings()
    if not st.scan_limit_enabled or (not skip_premium_check and is_premium(db, user_id)):
        usage = db.scan_usage.find_one({"user_id": user_id, "day": today_key()}) or {}
        return True, int(usage.get("count", 0))
    limit = st.free_daily_scans
    day = today_key()
    for _ in range(3):
        try:
            doc = db.scan_usage.find_one_and_update(
                {"user_id": user_id, "day": day, "count": {"$lt": limit}},
                {"$inc": {"count": 1}, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            return True, int(doc["count"])
        except DuplicateKeyError:
            # El documento del día ya existe: o llegó al tope, o es la carrera del
            # primer escaneo (otra petición lo creó entre el filtro y el upsert).
            cur = db.scan_usage.find_one({"user_id": user_id, "day": day}) or {}
            if int(cur.get("count", 0)) >= limit:
                return False, limit
    return False, limit


def release_scan(db, user_id: str) -> None:
    """Devuelve un escaneo reservado (el identify falló). Nunca baja de 0."""
    db.scan_usage.update_one(
        {"user_id": user_id, "day": today_key(), "count": {"$gt": 0}},
        {"$inc": {"count": -1}},
    )
