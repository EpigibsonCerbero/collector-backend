"""Servicio de suscripción y cuota diaria de escaneos. Mongo simulado."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mongomock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app.db as dbmod

_db = mongomock.MongoClient()["sub"]
dbmod.get_db = lambda: _db

from app import ratelimit
from app.config import get_settings
from app.security import create_access_token
from app.services import subscription as sub

UTC = timezone.utc


@pytest.fixture(scope="module", autouse=True)
def _indexes():
    # Sin el índice único mongomock insertaría un 2º documento y el tope nunca se alcanzaría.
    _db.scan_usage.create_index([("user_id", 1), ("day", 1)], unique=True)
    _db.subscriptions.create_index("user_id", unique=True)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    ratelimit._hits.clear()
    _db.scan_usage.delete_many({})
    _db.subscriptions.delete_many({})
    st = get_settings()
    monkeypatch.setattr(st, "scan_limit_enabled", True)
    monkeypatch.setattr(st, "premium_user_ids", "")


def _user(name: str) -> str:
    uid = f"u-{name}"
    create_access_token(uid, name)
    return uid


def _sub(uid, *, expires_at, premium=True, revoked=False):
    sub.upsert_subscription(
        _db, uid, premium=premium, product_id="curio_pro_monthly", period="monthly",
        expires_at=expires_at, will_renew=True, store="PLAY_STORE", event_id="e1", revoked=revoked,
    )


def test_today_key_mexico_boundary():
    # México = UTC-6 (sin horario de verano desde 2022).
    before = datetime(2026, 10, 30, 5, 59, tzinfo=UTC)  # 23:59 del 29 en México
    after = datetime(2026, 10, 30, 6, 1, tzinfo=UTC)    # 00:01 del 30 en México
    assert sub.today_key(before) == "2026-10-29"
    assert sub.today_key(after) == "2026-10-30"
    assert sub.next_reset(before) == datetime(2026, 10, 30, 6, 0, tzinfo=UTC)


def test_reserve_scan_stops_at_20():
    uid = _user("a")
    for i in range(1, 21):
        assert sub.reserve_scan(_db, uid) == (True, i)
    assert sub.reserve_scan(_db, uid) == (False, 20)
    assert _db.scan_usage.count_documents({"user_id": uid}) == 1


def test_reserve_scan_next_day_resets(monkeypatch):
    uid = _user("b")
    for _ in range(20):
        sub.reserve_scan(_db, uid)
    assert sub.reserve_scan(_db, uid)[0] is False
    monkeypatch.setattr(sub, "today_key", lambda now=None: "2099-01-01")
    assert sub.reserve_scan(_db, uid) == (True, 1)


def test_release_scan_decrements_never_below_zero():
    uid = _user("c")
    sub.reserve_scan(_db, uid)
    sub.reserve_scan(_db, uid)
    sub.release_scan(_db, uid)
    assert sub.get_status(_db, uid)["scans_used"] == 1
    sub.release_scan(_db, uid)
    sub.release_scan(_db, uid)
    assert sub.get_status(_db, uid)["scans_used"] == 0


def test_premium_skips_counter():
    uid = _user("d")
    _sub(uid, expires_at=datetime.now(UTC) + timedelta(days=10))
    for _ in range(25):
        assert sub.reserve_scan(_db, uid)[0] is True
    assert _db.scan_usage.count_documents({"user_id": uid}) == 0
    assert sub.get_status(_db, uid)["premium"] is True


def test_is_premium_grace_and_revoked():
    now = datetime.now(UTC)
    uid = _user("e")
    _sub(uid, expires_at=now + timedelta(days=5))
    assert sub.is_premium(_db, uid) is True
    _sub(uid, expires_at=now - timedelta(days=2))
    assert sub.is_premium(_db, uid) is True
    _sub(uid, expires_at=now - timedelta(days=4))
    assert sub.is_premium(_db, uid) is False
    _sub(uid, expires_at=now - timedelta(minutes=1), revoked=True)
    assert sub.is_premium(_db, uid) is False


def test_premium_user_ids_env_forces_premium(monkeypatch):
    uid = _user("f")
    assert sub.is_premium(_db, uid) is False
    monkeypatch.setattr(get_settings(), "premium_user_ids", f"x, {uid}")
    assert sub.is_premium(_db, uid) is True


def test_reserve_scan_disabled_flag_allows_all(monkeypatch):
    monkeypatch.setattr(get_settings(), "scan_limit_enabled", False)
    uid = _user("g")
    for _ in range(30):
        assert sub.reserve_scan(_db, uid)[0] is True
    assert _db.scan_usage.count_documents({"user_id": uid}) == 0


# ───────────────────────── Task 2: RevenueCat (REST, estado, webhook) ─────────────────────────
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import subscription as sub_router
from app.services import revenuecat

_api = FastAPI()
_api.include_router(sub_router.router, prefix="/api/v1")
_api.include_router(sub_router.webhook_router, prefix="/api/v1")
_c = TestClient(_api)
WH = "/api/v1/webhooks/revenuecat"
SECRET = "whsec-test"


@pytest.fixture(scope="module", autouse=True)
def _event_index():
    _db.revenuecat_events.create_index("event_id", unique=True)


@pytest.fixture(autouse=True)
def _clean_events(monkeypatch):
    _db.revenuecat_events.delete_many({})
    monkeypatch.setattr(get_settings(), "revenuecat_webhook_auth", SECRET)


def _ms(dt):
    return int(dt.timestamp() * 1000)


def _payload(expires, product="curio_pro_monthly", unsub=None, ent=True):
    ent_d = {"expires_date": expires, "product_identifier": product, "purchase_date": "2026-10-01T00:00:00Z"}
    return {"subscriber": {
        "entitlements": {"premium": ent_d} if ent else {},
        "subscriptions": {product: {"expires_date": expires, "store": "play_store",
                                    "unsubscribe_detected_at": unsub}},
    }}


def _ev(etype, eid, ts, user="u-w", **kw):
    ev = {"id": eid, "type": etype, "app_user_id": user, "event_timestamp_ms": ts,
          "product_id": "curio_pro_monthly", "store": "PLAY_STORE", **kw}
    return {"event": ev}


def _post(body, auth=SECRET):
    h = {"Authorization": auth} if auth is not None else {}
    return _c.post(WH, json=body, headers=h)


def test_entitlement_from_subscriber_active_and_expired():
    future = (datetime.now(UTC) + timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    past = (datetime.now(UTC) - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    act = revenuecat.entitlement_from_subscriber(_payload(future))
    assert act["premium"] is True and act["period"] == "monthly" and act["will_renew"] is True
    assert act["product_id"] == "curio_pro_monthly" and act["store"] == "play_store"
    assert act["expires_at"] > datetime.now(UTC)
    exp = revenuecat.entitlement_from_subscriber(_payload(past, unsub="2026-09-01T00:00:00Z"))
    assert exp["premium"] is True and exp["will_renew"] is False  # la gracia la decide is_premium
    assert revenuecat.entitlement_from_subscriber(_payload(future, ent=False)) is None
    assert revenuecat.entitlement_from_subscriber({"subscriber": {}}) is None
    # Vitalicio solo si el payload trae expires_date: null de verdad.
    life = revenuecat.entitlement_from_subscriber(_payload(None, product="curio_pro_lifetime"))
    assert life["premium"] is True and life["expires_at"] is None and life["period"] == "lifetime"
    # Entitlement sin la clave expires_date = malformado.
    bad = {"subscriber": {"entitlements": {"premium": {"product_identifier": "x"}}}}
    assert revenuecat.entitlement_from_subscriber(bad) is None


def test_sync_user_updates_state_and_keeps_previous_on_failure():
    uid = "u-sync"
    future = (datetime.now(UTC) + timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with patch.object(revenuecat, "fetch_subscriber", return_value=_payload(future)):
        assert revenuecat.sync_user(_db, uid) is True
    assert sub.is_premium(_db, uid)
    before = _db.subscriptions.find_one({"user_id": uid})
    with patch.object(revenuecat, "fetch_subscriber", return_value=None):
        assert revenuecat.sync_user(_db, uid) is False
    assert _db.subscriptions.find_one({"user_id": uid}) == before
    # Sin entitlement premium o malformado: no escribe premium.
    other = "u-sync2"
    with patch.object(revenuecat, "fetch_subscriber", return_value=_payload(future, ent=False)):
        assert revenuecat.sync_user(_db, other) is True
    assert not sub.is_premium(_db, other)
    assert _db.subscriptions.find_one({"user_id": other}) is None


def test_fetch_subscriber_fail_open_without_key(monkeypatch):
    monkeypatch.setattr(get_settings(), "revenuecat_secret_key", "")
    assert revenuecat.fetch_subscriber("u-x") is None
    monkeypatch.setattr(get_settings(), "revenuecat_secret_key", "sk_x")
    with patch("app.services.revenuecat.httpx.get", side_effect=RuntimeError("boom")):
        assert revenuecat.fetch_subscriber("u-x") is None


def test_status_endpoint_requires_auth():
    assert _c.get("/api/v1/subscription/status").status_code == 401
    assert _c.post("/api/v1/subscription/sync").status_code == 401
    tok = create_access_token("u-st", "st")
    r = _c.get("/api/v1/subscription/status", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200 and r.json()["premium"] is False and r.json()["scans_limit"] == 20
    with patch.object(revenuecat, "fetch_subscriber", return_value=None):
        r = _c.post("/api/v1/subscription/sync", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200 and r.json()["premium"] is False


def test_webhook_rejects_wrong_auth():
    body = _ev("INITIAL_PURCHASE", "e-bad", 1000, expiration_at_ms=_ms(datetime.now(UTC) + timedelta(days=5)))
    assert _post(body, auth="otra").status_code == 401
    assert _post(body, auth=None).status_code == 401
    assert _db.subscriptions.find_one({"user_id": "u-w"}) is None


def test_webhook_rejects_when_secret_unset(monkeypatch):
    monkeypatch.setattr(get_settings(), "revenuecat_webhook_auth", "")
    assert _post(_ev("INITIAL_PURCHASE", "e-x", 1), auth="").status_code == 401
    assert _post(_ev("INITIAL_PURCHASE", "e-x", 1), auth="cualquiera").status_code == 401


def test_webhook_is_idempotent_and_ignores_stale_event():
    uid = "u-w"
    later = datetime.now(UTC) + timedelta(days=30)
    r = _post(_ev("RENEWAL", "e1", 2000, expiration_at_ms=_ms(later)))
    assert r.status_code == 200 and sub.is_premium(_db, uid)
    # Mismo id otra vez: no se reprocesa.
    assert _post(_ev("EXPIRATION", "e1", 2500, expiration_at_ms=1000)).status_code == 200
    assert sub.is_premium(_db, uid)
    assert _db.revenuecat_events.count_documents({"event_id": "e1"}) == 1
    # Evento viejo (timestamp menor) no revierte la expiración posterior.
    assert _post(_ev("EXPIRATION", "e0", 1000, expiration_at_ms=1000)).status_code == 200
    assert sub.is_premium(_db, uid)
    assert _db.subscriptions.count_documents({"user_id": uid}) == 1
    # Uno más nuevo sí aplica.
    assert _post(_ev("EXPIRATION", "e2", 3000, expiration_at_ms=1000)).status_code == 200
    assert not sub.is_premium(_db, uid, now=datetime.now(UTC) + timedelta(days=60))


def test_webhook_cancellation_only_stops_renewal_and_ignored_types_ok():
    uid = "u-w"
    later = datetime.now(UTC) + timedelta(days=30)
    _post(_ev("INITIAL_PURCHASE", "c1", 1000, expiration_at_ms=_ms(later)))
    assert _post(_ev("CANCELLATION", "c2", 2000, cancel_reason="UNSUBSCRIBE")).status_code == 200
    doc = _db.subscriptions.find_one({"user_id": uid})
    assert doc["will_renew"] is False and sub.is_premium(_db, uid)
    assert _post(_ev("BILLING_ISSUE", "c3", 3000)).status_code == 200
    assert sub.is_premium(_db, uid)
    assert _post(_ev("TEST", "c4", 4000)).status_code == 200


def test_webhook_activation_without_expiration_does_not_grant_forever():
    r = _post(_ev("INITIAL_PURCHASE", "n1", 1000, user="u-noexp"))
    assert r.status_code == 200
    assert not sub.is_premium(_db, "u-noexp")


def test_webhook_refund_revokes_without_grace():
    uid = "u-w"
    later = datetime.now(UTC) + timedelta(days=30)
    _post(_ev("INITIAL_PURCHASE", "r1", 1000, expiration_at_ms=_ms(later)))
    assert sub.is_premium(_db, uid)
    assert _post(_ev("CANCELLATION", "r2", 2000, cancel_reason="CUSTOMER_SUPPORT")).status_code == 200
    doc = _db.subscriptions.find_one({"user_id": uid})
    assert doc["revoked"] is True and doc["premium"] is False
    assert not sub.is_premium(_db, uid)


def _days(n):
    return (datetime.now(UTC) + timedelta(days=n)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_grace_window_after_expiration_webhook_and_sync():
    uid = "u-w"
    for ago, expected in ((1, True), (4, False)):
        _db.subscriptions.delete_many({})
        _db.revenuecat_events.delete_many({})
        _post(_ev("RENEWAL", f"g{ago}a", 1000, expiration_at_ms=_ms(datetime.now(UTC) + timedelta(days=30))))
        _post(_ev("EXPIRATION", f"g{ago}b", 2000, expiration_at_ms=_ms(datetime.now(UTC) - timedelta(days=ago))))
        assert sub.is_premium(_db, uid) is expected
        # sync con entitlement ya vencido: mismo resultado
        _db.subscriptions.delete_many({})
        with patch.object(revenuecat, "fetch_subscriber", return_value=_payload(_days(-ago))):
            revenuecat.sync_user(_db, uid)
        assert sub.is_premium(_db, uid) is expected


def test_sync_does_not_restore_revoked_premium():
    uid = "u-w"
    end = _days(30)
    _post(_ev("INITIAL_PURCHASE", "v1", 1000, expiration_at_ms=_ms(revenuecat.parse_date(end))))
    _post(_ev("CANCELLATION", "v2", 2000, cancel_reason="CUSTOMER_SUPPORT"))
    assert not sub.is_premium(_db, uid)
    with patch.object(revenuecat, "fetch_subscriber", return_value=_payload(end)):
        revenuecat.sync_user(_db, uid)
    doc = _db.subscriptions.find_one({"user_id": uid})
    assert doc["revoked"] is True and not sub.is_premium(_db, uid)
    # Una compra posterior (vencimiento más nuevo que el estado revocado) sí repone.
    with patch.object(revenuecat, "fetch_subscriber", return_value=_payload(_days(60))):
        revenuecat.sync_user(_db, uid)
    assert sub.is_premium(_db, uid)


def test_sync_endpoint_is_rate_limited():
    tok = create_access_token("u-rl", "rl")
    h = {"Authorization": f"Bearer {tok}"}
    with patch.object(revenuecat, "fetch_subscriber", return_value=None):
        codes = [_c.post("/api/v1/subscription/sync", headers=h).status_code for _ in range(6)]
    assert codes == [200] * 5 + [429]


def test_product_change_records_new_product():
    later = _ms(datetime.now(UTC) + timedelta(days=30))
    _post(_ev("PRODUCT_CHANGE", "p1", 1000, new_product_id="curio_pro_annual", expiration_at_ms=later))
    doc = _db.subscriptions.find_one({"user_id": "u-w"})
    assert doc["product_id"] == "curio_pro_annual" and doc["period"] == "annual"
