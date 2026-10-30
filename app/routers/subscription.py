"""Estado Premium: sync con RevenueCat, consulta y webhook."""
from __future__ import annotations

import asyncio
import hmac
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pymongo.errors import DuplicateKeyError

from app.config import get_settings
from app.db import get_db
from app.ratelimit import hit_or_429
from app.security import CurrentUser, get_current_user
from app.services import revenuecat, subscription

router = APIRouter(prefix="/subscription", tags=["subscription"])
webhook_router = APIRouter(prefix="/webhooks", tags=["subscription"])

ACTIVATE = {"INITIAL_PURCHASE", "RENEWAL", "UNCANCELLATION", "PRODUCT_CHANGE"}
REVOKE_TYPES = {"REFUND", "REVOCATION"}
REVOKE_CANCEL_REASONS = {"CUSTOMER_SUPPORT"}  # reembolso hecho por soporte


@router.post("/sync")
async def sync(user: CurrentUser = Depends(get_current_user)):
    hit_or_429("sub-sync", user.id, limit=5, per_seconds=60)
    db = get_db()

    def _work():
        revenuecat.sync_user(db, user.id)
        return subscription.get_status(db, user.id)

    return await asyncio.to_thread(_work)


@router.get("/status")
def status_(user: CurrentUser = Depends(get_current_user)):
    return subscription.get_status(get_db(), user.id)


def _ms_to_dt(ms) -> datetime | None:
    if isinstance(ms, bool) or not isinstance(ms, (int, float)):
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _apply(db, ev: dict) -> None:
    etype = ev.get("type")
    uid = ev.get("app_user_id")
    if not uid or str(uid).startswith("$RCAnonymousID"):
        return
    ts = ev.get("event_timestamp_ms")
    ts = int(ts) if isinstance(ts, (int, float)) and not isinstance(ts, bool) else None
    doc = db.subscriptions.find_one({"user_id": uid}) or {}
    last = doc.get("last_event_ts_ms")
    if ts is not None and last is not None and ts < last:
        return  # evento más viejo que el último aplicado
    # PRODUCT_CHANGE: el producto nuevo viene en new_product_id.
    product_id = (ev.get("new_product_id") if etype == "PRODUCT_CHANGE" else None) or ev.get("product_id")
    common = dict(
        product_id=product_id,
        store=ev.get("store"),
        event_id=ev.get("id"),
        event_ts_ms=ts,
    )
    revoke = etype in REVOKE_TYPES or (
        etype == "CANCELLATION" and ev.get("cancel_reason") in REVOKE_CANCEL_REASONS
    )
    exp = _ms_to_dt(ev.get("expiration_at_ms"))
    if revoke:
        subscription.upsert_subscription(
            db, uid, premium=False, period=doc.get("period"), expires_at=exp or doc.get("expires_at") or datetime.now(timezone.utc),
            will_renew=False, revoked=True, **common,
        )
    elif etype in ACTIVATE:
        if exp is None:
            return  # sin vencimiento no se concede Premium (None = para siempre)
        subscription.upsert_subscription(
            db, uid, premium=True, period=revenuecat.period_from_product(product_id, exp),
            expires_at=exp, will_renew=True, **common,
        )
    elif etype == "CANCELLATION":
        if doc:  # sigue activo hasta expires_at, solo deja de renovar
            update = {"will_renew": False, "last_event_id": ev.get("id"),
                      "updated_at": datetime.now(timezone.utc)}
            if ts is not None:
                update["last_event_ts_ms"] = ts
            db.subscriptions.update_one({"user_id": uid}, {"$set": update})
    elif etype == "EXPIRATION":
        # No apaga premium: is_premium aplica expires_at + gracia.
        subscription.upsert_subscription(
            db, uid, premium=bool(doc.get("premium", False)), period=doc.get("period"),
            expires_at=exp or doc.get("expires_at") or datetime.now(timezone.utc),
            will_renew=False, **common,
        )
    # BILLING_ISSUE y el resto: se mantiene el estado hasta expires_at


@webhook_router.post("/revenuecat")
async def revenuecat_webhook(request: Request):
    secret = get_settings().revenuecat_webhook_auth
    got = request.headers.get("authorization", "")
    if not secret or not got or not hmac.compare_digest(got.encode(), secret.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "No autorizado")
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return {"ok": True}
    ev = body.get("event") if isinstance(body, dict) else None
    if not isinstance(ev, dict) or not ev.get("id"):
        return {"ok": True}
    db = get_db()

    def _work() -> dict:
        try:
            db.revenuecat_events.insert_one(
                {"event_id": ev["id"], "type": ev.get("type"), "created_at": datetime.now(timezone.utc)}
            )
        except DuplicateKeyError:
            return {"ok": True, "duplicate": True}
        try:
            _apply(db, ev)
        except Exception:
            # Libera el id para que el reintento de RevenueCat pueda procesarlo.
            db.revenuecat_events.delete_one({"event_id": ev["id"]})
            raise
        return {"ok": True}

    return await asyncio.to_thread(_work)
