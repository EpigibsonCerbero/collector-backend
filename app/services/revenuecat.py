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
from app.services.subscription import _utc, upsert_subscription

API_URL = "https://api.revenuecat.com/v1/subscribers/"
ENTITLEMENT = "premium"


DEFAULT_TIMEOUT_S = 5.0


def fetch_subscriber(app_user_id: str, timeout: float = DEFAULT_TIMEOUT_S) -> dict | None:
    """GET del suscriptor. None si no hay clave, hay error de red o la respuesta no es JSON.

    `timeout` corto desde el identify (presupuesto de 8 s); 5 s en /sync y webhooks.
    """
    key = get_settings().revenuecat_secret_key
    if not key:
        return None
    try:
        r = httpx.get(
            API_URL + quote(app_user_id, safe=""),
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            timeout=timeout,
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
            # RevenueCat conserva los entitlements vencidos: expires_at + gracia deciden en is_premium.
            "premium": True,
            "product_id": product_id,
            "period": period_from_product(product_id, expires_at),
            "expires_at": expires_at,
            "will_renew": bool(will_renew),
            "store": info.get("store"),
        }
    except (KeyError, TypeError, AttributeError):
        return None


def entitlement_absent(payload: dict) -> bool:
    """True si RevenueCat respondió bien formado y el entitlement `premium` no está."""
    try:
        ents = payload["subscriber"]["entitlements"]
    except (KeyError, TypeError):
        return False
    return isinstance(ents, dict) and ENTITLEMENT not in ents


def sync_user(db, user_id: str, timeout: float = DEFAULT_TIMEOUT_S) -> bool:
    """Consulta RevenueCat y aplica el estado. False si no respondió (estado previo intacto).

    RevenueCat es la fuente de verdad: si responde sin el entitlement `premium`
    (p. ej. tras un TRANSFER a otra cuenta) el usuario deja de ser Premium.
    Un error de red o un payload malformado nunca degrada.
    """
    payload = fetch_subscriber(user_id, timeout=timeout)
    if payload is None:
        return False
    ent = entitlement_from_subscriber(payload)
    if ent is None:
        if entitlement_absent(payload):
            # Se conservan producto, vencimiento y `revoked`; solo se apaga el acceso.
            db.subscriptions.update_one(
                {"user_id": user_id},
                {"$set": {"premium": False, "will_renew": False,
                          "updated_at": datetime.now(timezone.utc)}},
            )
        return True  # malformado: no se toca nada
    doc = db.subscriptions.find_one({"user_id": user_id}) or {}
    if doc.get("revoked"):
        # Tras un reembolso/revocación solo una compra posterior (vencimiento más nuevo) lo repone.
        prev = doc.get("expires_at")
        new = ent["expires_at"]
        if prev is not None and (new is not None and _utc(new) <= _utc(prev)):
            return True
    upsert_subscription(db, user_id, event_id=doc.get("last_event_id"), **ent)
    return True
