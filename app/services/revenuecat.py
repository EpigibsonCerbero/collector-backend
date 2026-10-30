"""Cliente REST de RevenueCat y traducción de su payload al estado Premium.

Todo es síncrono y fail-open: cualquier fallo devuelve None/False, nunca lanza.
El router lo envuelve en `asyncio.to_thread`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx

from app.config import get_settings
from app.services.subscription import upsert_subscription

API_URL = "https://api.revenuecat.com/v1/subscribers/"
ENTITLEMENT = "premium"


def fetch_subscriber(app_user_id: str) -> dict | None:
    """GET del suscriptor. None si no hay clave, hay error de red o la respuesta no es JSON."""
    key = get_settings().revenuecat_secret_key
    if not key:
        return None
    try:
        r = httpx.get(
            API_URL + quote(app_user_id, safe=""),
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            timeout=5.0,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def parse_date(value: Any) -> datetime | None:
    """ISO-8601 de RevenueCat (con 'Z') → datetime UTC aware."""
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def period_from_product(product_id: str | None, expires_at: datetime | None) -> str | None:
    p = (product_id or "").lower()
    if expires_at is None:
        return "lifetime"
    if "year" in p or "annual" in p:
        return "annual"
    if "month" in p:
        return "monthly"
    return None


def entitlement_from_subscriber(payload: dict) -> dict | None:
    """Estado del entitlement `premium`, o None si falta o está malformado.

    `expires_at=None` solo se produce si el payload trae `expires_date: null`
    (vitalicio); si la clave no existe el payload se considera malformado.
    """
    try:
        sub = payload["subscriber"]
        ent = sub["entitlements"][ENTITLEMENT]
        if "expires_date" not in ent:
            return None
        raw = ent["expires_date"]
        expires_at = parse_date(raw)
        if raw is not None and expires_at is None:
            return None
        product_id = ent.get("product_identifier")
        info = (sub.get("subscriptions") or {}).get(product_id) or {}
        active = expires_at is None or expires_at > datetime.now(timezone.utc)
        will_renew = (
            expires_at is not None
            and active
            and not info.get("unsubscribe_detected_at")
        )
        return {
            "premium": active,
            "product_id": product_id,
            "period": period_from_product(product_id, expires_at),
            "expires_at": expires_at,
            "will_renew": bool(will_renew),
            "store": info.get("store"),
        }
    except (KeyError, TypeError, AttributeError):
        return None


def sync_user(db, user_id: str) -> bool:
    """Consulta RevenueCat y aplica el estado. False si no respondió (estado previo intacto)."""
    payload = fetch_subscriber(user_id)
    if payload is None:
        return False
    ent = entitlement_from_subscriber(payload)
    if ent is None:
        return True  # respondió, pero no hay entitlement utilizable: no se toca nada
    upsert_subscription(db, user_id, event_id=None, **ent)
    return True
