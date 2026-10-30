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
