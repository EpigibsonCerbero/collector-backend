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


# ── Task 3: cuota en el identify y freno de registros ───────────────────────
import threading
from unittest.mock import patch as _patch

from pymongo.errors import DuplicateKeyError

from app.routers import auth as auth_router
from app.routers import identify as identify_router
from app.routers.identify import IdentifyOut, IdentifySuggestion

_iapi = FastAPI()
_iapi.include_router(identify_router.router, prefix="/api/v1")
_iapi.include_router(auth_router.router, prefix="/api/v1")
_ic = TestClient(_iapi)
IDENT = "/api/v1/identify/base64"
_IMG = {"image_base64": "aGVsbG8="}


def _hdr(uid):
    return {"Authorization": f"Bearer {create_access_token(uid, uid)}"}


def _ok_result():
    return IdentifyOut(method="c", confidence=0.9, needs_manual=False, item={"id": "v1"})


@pytest.fixture(autouse=True)
def _identify_state():
    identify_router._last_reconcile.clear()
    with _patch.object(identify_router, "_run_identify", side_effect=lambda *a, **k: _ok_result()):
        yield


def _scan(uid):
    return _ic.post(IDENT, json=_IMG, headers=_hdr(uid))


def _used(uid):
    d = _db.scan_usage.find_one({"user_id": uid, "day": sub.today_key()})
    return d["count"] if d else 0


def test_reserve_scan_first_scan_race_loser_still_gets_slot():
    uid = _user("race1")
    real = _db.scan_usage.find_one_and_update

    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise DuplicateKeyError("dup")  # perdió la carrera del primer escaneo
        return real(*a, **k)

    with _patch.object(_db.scan_usage, "find_one_and_update", side_effect=flaky):
        _db.scan_usage.insert_one({"user_id": uid, "day": sub.today_key(), "count": 1})
        assert sub.reserve_scan(_db, uid) == (True, 2)
    assert _used(uid) == 2


def test_reserve_scan_duplicate_at_limit_denies():
    uid = _user("race2")
    _db.scan_usage.insert_one({"user_id": uid, "day": sub.today_key(), "count": 20})
    assert sub.reserve_scan(_db, uid) == (False, 20)


def test_identify_requires_session():
    assert _ic.post(IDENT, json=_IMG).status_code == 401


def test_identify_402_after_20_with_flag_on():
    uid = _user("q20")
    for i in range(20):
        r = _scan(uid)
        assert r.status_code == 200, r.text
        assert r.json()["quota"] == {"used": i + 1, "limit": 20, "premium": False}
    with _patch.object(identify_router.revenuecat, "sync_user", return_value=False):
        r = _scan(uid)
    assert r.status_code == 402
    body = r.json()
    assert isinstance(body["detail"], str)
    assert body["code"] == "scan_limit_reached" and body["limit"] == 20
    assert body["resets_at"] == sub.next_reset().isoformat()
    assert _used(uid) == 20


def test_identify_unlimited_when_flag_off(monkeypatch):
    monkeypatch.setattr(get_settings(), "scan_limit_enabled", False)
    uid = _user("off")
    for _ in range(25):
        r = _scan(uid)
        assert r.status_code == 200
        assert r.json()["quota"] is None
    assert _used(uid) == 0


def test_premium_not_counted():
    uid = _user("prem")
    _sub(uid, expires_at=datetime.now(UTC) + timedelta(days=5))
    for _ in range(22):
        r = _scan(uid)
        assert r.status_code == 200
    assert r.json()["quota"]["premium"] is True
    assert _used(uid) == 0


def test_two_parallel_requests_at_19_only_one_passes():
    uid = _user("par")
    _db.scan_usage.insert_one({"user_id": uid, "day": sub.today_key(), "count": 19})
    hdr = _hdr(uid)
    barrier = threading.Barrier(2)
    codes = []

    def go():
        barrier.wait()
        codes.append(_ic.post(IDENT, json=_IMG, headers=hdr).status_code)

    # Un solo patch fuera de los hilos: dos patch anidados en hilos dejan el mock puesto.
    with _patch.object(identify_router.revenuecat, "sync_user", return_value=False):
        ts = [threading.Thread(target=go) for _ in range(2)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    assert sorted(codes) == [200, 402]
    assert _used(uid) == 20


def test_release_on_exception_and_on_empty_result():
    uid = _user("rel")
    with _patch.object(identify_router, "_run_identify", side_effect=RuntimeError("boom")):
        c = TestClient(_iapi, raise_server_exceptions=False)
        assert c.post(IDENT, json=_IMG, headers=_hdr(uid)).status_code == 500
    assert _used(uid) == 0
    with _patch.object(identify_router, "_run_identify", return_value=IdentifyOut()):
        r = _scan(uid)
        assert r.status_code == 200
    assert _used(uid) == 0


def test_manual_with_suggestions_counts():
    uid = _user("man")
    res = IdentifyOut(suggestions=[IdentifySuggestion(id="s1", name="x")])
    with _patch.object(identify_router, "_run_identify", return_value=res):
        assert _scan(uid).status_code == 200
    assert _used(uid) == 1


def test_402_triggers_single_reconcile_and_grants_if_premium():
    uid = _user("rec")
    _db.scan_usage.insert_one({"user_id": uid, "day": sub.today_key(), "count": 20})
    calls = []

    def fake_sync(db, user_id, **kw):
        calls.append(user_id)
        _sub(user_id, expires_at=datetime.now(UTC) + timedelta(days=5))
        return True

    with _patch.object(identify_router.revenuecat, "sync_user", side_effect=fake_sync):
        r1 = _scan(uid)
        r2 = _scan(uid)
    assert r1.status_code == 200 and r1.json()["quota"]["premium"] is True
    assert r2.status_code == 200
    assert calls == [uid]
    assert _used(uid) == 20


def test_reconcile_rate_limited_to_one_per_60s():
    uid = _user("rec2")
    _db.scan_usage.insert_one({"user_id": uid, "day": sub.today_key(), "count": 20})
    with _patch.object(identify_router.revenuecat, "sync_user", return_value=False) as m:
        assert _scan(uid).status_code == 402
        assert _scan(uid).status_code == 402
    assert m.call_count == 1


def test_reconcile_failure_keeps_free_limit_no_500():
    uid = _user("recfail")
    _db.scan_usage.insert_one({"user_id": uid, "day": sub.today_key(), "count": 20})
    with _patch.object(identify_router.revenuecat, "sync_user", side_effect=RuntimeError("down")):
        assert _scan(uid).status_code == 402


def test_expired_premium_in_grace_triggers_reconcile():
    uid = _user("grace")
    _sub(uid, expires_at=datetime.now(UTC) - timedelta(hours=2))
    with _patch.object(identify_router.revenuecat, "sync_user", return_value=True) as m:
        r = _scan(uid)
    assert r.status_code == 200 and m.call_count == 1


def test_quota_failure_after_reserve_releases_scan():
    uid = _user("leak")
    class Bad:
        item = {"id": "v1"}
        suggestions: list = []

        def __setattr__(self, k, v):  # falla al montar la respuesta, con la reserva ya hecha
            raise RuntimeError("boom")

    with _patch.object(identify_router, "_run_identify", return_value=Bad()):
        c = TestClient(_iapi, raise_server_exceptions=False)
        assert c.post(IDENT, json=_IMG, headers=_hdr(uid)).status_code == 500
    assert _used(uid) == 0


def test_free_scan_does_not_reread_premium_twice():
    uid = _user("reads")
    with _patch.object(sub, "is_premium", wraps=sub.is_premium) as m:
        assert _scan(uid).status_code == 200
    assert m.call_count == 0
    assert _used(uid) == 1


def test_register_blocked_after_5_per_ip():
    ratelimit._hits.clear()
    codes = []
    for i in range(6):
        r = _ic.post("/api/v1/auth/register", json={
            "email": f"reg{i}@example.com", "username": f"reguser{i}", "password": "Passw0rd!x"})
        codes.append(r.status_code)
    assert codes[:5] == [201] * 5 and codes[5] == 429, codes


# ── Ola final de correcciones ───────────────────────────────────────────────
def test_status_flag_off_reports_no_limit(monkeypatch):
    uid = _user("stoff")
    monkeypatch.setattr(get_settings(), "scan_limit_enabled", False)
    st = sub.get_status(_db, uid)
    assert st["scans_limit"] is None and st["limit_enabled"] is False
    monkeypatch.setattr(get_settings(), "scan_limit_enabled", True)
    st = sub.get_status(_db, uid)
    assert st["scans_limit"] == 20 and st["limit_enabled"] is True


def test_webhook_anonymous_id_falls_back_to_original_then_aliases():
    exp = _ms(datetime.now(UTC) + timedelta(days=30))
    _post(_ev("INITIAL_PURCHASE", "an1", 1000, user="$RCAnonymousID:abc",
              original_app_user_id="u-orig", expiration_at_ms=exp))
    assert sub.is_premium(_db, "u-orig")
    _post(_ev("INITIAL_PURCHASE", "an2", 1000, user="$RCAnonymousID:abc",
              original_app_user_id="$RCAnonymousID:abc",
              aliases=["$RCAnonymousID:abc", "u-alias"], expiration_at_ms=exp))
    assert sub.is_premium(_db, "u-alias")
    _post(_ev("INITIAL_PURCHASE", "an3", 1000, user="$RCAnonymousID:zz",
              original_app_user_id="$RCAnonymousID:zz", aliases=["$RCAnonymousID:zz"],
              expiration_at_ms=exp))
    assert _db.subscriptions.count_documents({"user_id": {"$regex": "^\\$RC"}}) == 0


def test_sync_entitlement_absent_downgrades_when_revenuecat_answers():
    uid = "u-gone"
    _sub(uid, expires_at=datetime.now(UTC) + timedelta(days=20))
    assert sub.is_premium(_db, uid)
    # RevenueCat caído: nada cambia.
    with patch.object(revenuecat, "fetch_subscriber", return_value=None):
        assert revenuecat.sync_user(_db, uid) is False
    assert sub.is_premium(_db, uid)
    # Payload malformado: nada cambia.
    with patch.object(revenuecat, "fetch_subscriber", return_value={"subscriber": None}):
        revenuecat.sync_user(_db, uid)
    assert sub.is_premium(_db, uid)
    # Respondió sin el entitlement: deja de ser Premium (conserva el resto).
    with patch.object(revenuecat, "fetch_subscriber", return_value=_payload(_days(20), ent=False)):
        assert revenuecat.sync_user(_db, uid) is True
    doc = _db.subscriptions.find_one({"user_id": uid})
    assert doc["premium"] is False and doc["will_renew"] is False
    assert doc["product_id"] == "curio_pro_monthly" and doc["revoked"] is False
    assert not sub.is_premium(_db, uid)


def test_transfer_revokes_old_account_and_activates_new():
    old, new = "u-old", "u-new"
    _sub(old, expires_at=datetime.now(UTC) + timedelta(days=20))

    def fake(user_id, timeout=5.0):
        return _payload(_days(20), ent=(user_id == new))

    with patch.object(revenuecat, "fetch_subscriber", side_effect=fake):
        r = _post({"event": {"id": "t1", "type": "TRANSFER", "event_timestamp_ms": 5000,
                             "transferred_from": [old, "$RCAnonymousID:q"], "transferred_to": [new]}})
        assert r.status_code == 200
        assert not sub.is_premium(_db, old) and sub.is_premium(_db, new)
        # Idempotente por id de evento.
        _sub(old, expires_at=datetime.now(UTC) + timedelta(days=20))
        _post({"event": {"id": "t1", "type": "TRANSFER", "event_timestamp_ms": 5000,
                         "transferred_from": [old], "transferred_to": [new]}})
    assert sub.is_premium(_db, old)


def test_transfer_respects_stale_guard():
    old = "u-old2"
    _sub(old, expires_at=datetime.now(UTC) + timedelta(days=20))
    _db.subscriptions.update_one({"user_id": old}, {"$set": {"last_event_ts_ms": 9000}})
    with patch.object(revenuecat, "fetch_subscriber", return_value=_payload(_days(20), ent=False)):
        _post({"event": {"id": "t2", "type": "TRANSFER", "event_timestamp_ms": 1000,
                         "transferred_from": [old], "transferred_to": []}})
    assert sub.is_premium(_db, old)


def test_expiration_after_refund_keeps_revoked_flag():
    uid = "u-w"
    end = datetime.now(UTC) + timedelta(days=30)
    _post(_ev("INITIAL_PURCHASE", "x1", 1000, expiration_at_ms=_ms(end)))
    _post(_ev("REFUND", "x2", 2000))
    exp_at = datetime.now(UTC) - timedelta(days=1)
    _post(_ev("EXPIRATION", "x3", 3000, expiration_at_ms=_ms(exp_at)))
    assert _db.subscriptions.find_one({"user_id": uid})["revoked"] is True
    # Un sync con el mismo vencimiento (viejo) no repone Premium ni da gracia.
    with patch.object(revenuecat, "fetch_subscriber",
                      return_value=_payload(exp_at.strftime("%Y-%m-%dT%H:%M:%SZ"))):
        revenuecat.sync_user(_db, uid)
    assert not sub.is_premium(_db, uid)


def test_identify_reconcile_uses_short_timeout():
    uid = _user("tmo")
    _db.scan_usage.insert_one({"user_id": uid, "day": sub.today_key(), "count": 20})
    with _patch.object(identify_router.revenuecat, "sync_user", return_value=False) as m:
        assert _scan(uid).status_code == 402
    assert m.call_args.kwargs.get("timeout") == identify_router._RECONCILE_TIMEOUT_S
    assert identify_router._RECONCILE_TIMEOUT_S <= 2.0


def test_fetch_subscriber_passes_timeout(monkeypatch):
    monkeypatch.setattr(get_settings(), "revenuecat_secret_key", "sk_x")
    with patch("app.services.revenuecat.httpx.get", side_effect=RuntimeError("x")) as g:
        revenuecat.fetch_subscriber("u-x", timeout=2.0)
        assert g.call_args.kwargs["timeout"] == 2.0
        revenuecat.fetch_subscriber("u-x")
        assert g.call_args.kwargs["timeout"] == 5.0


def test_free_scan_is_logged(caplog):
    import logging
    uid = _user("logme")
    with caplog.at_level(logging.INFO, logger="app.routers.identify"):
        assert _scan(uid).status_code == 200
    assert f"scan_free user={uid} used=1/20" in caplog.text
