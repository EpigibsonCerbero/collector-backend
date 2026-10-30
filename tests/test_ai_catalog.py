"""Recibo de escaneo del autocompletado de catálogo. Mongo simulado."""
import asyncio
import sys
from datetime import timezone
from pathlib import Path

import mongomock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app.db as dbmod

_db = mongomock.MongoClient()["aicat"]
dbmod.get_db = lambda: _db

from app import ratelimit
from app.config import get_settings
from app.routers import identify as idn
from app.security import CurrentUser

IMG = b"fake-image-bytes"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    ratelimit._hits.clear()
    _db.identify_receipts.delete_many({})
    _db.scan_usage.delete_many({})
    st = get_settings()
    monkeypatch.setattr(st, "scan_limit_enabled", False)
    monkeypatch.setattr(st, "ai_catalog_enabled", True)


def _fake(manual=True, item=None, top=0.4, twins=False):
    def run(data, brand):
        out = idn.IdentifyOut(needs_manual=manual, item=item)
        out.debug["top_score"] = top
        out.debug["twins"] = twins
        return out
    return run


def _go():
    return asyncio.run(idn._identify_guarded(None, IMG, None, CurrentUser({"id": "u1"})))


def test_image_hash_stable():
    assert idn.image_hash(IMG) == idn.image_hash(bytes(IMG))
    assert len(idn.image_hash(IMG)) == 64
    assert idn.image_hash(IMG) != idn.image_hash(b"otra")


@pytest.mark.parametrize("quota", [False, True])
def test_receipt_written_only_when_manual_and_enabled(monkeypatch, quota):
    monkeypatch.setattr(get_settings(), "scan_limit_enabled", quota)
    monkeypatch.setattr(idn, "_run_identify", _fake(manual=True, top=0.4, twins=True))
    _go()
    docs = list(_db.identify_receipts.find({}))
    assert len(docs) == 1
    d = docs[0]
    assert d["user_id"] == "u1"
    assert d["image_hash"] == idn.image_hash(IMG)
    assert d["top_score"] == 0.4 and d["twins"] is True
    # mongomock devuelve naive; el valor escrito debe ser UTC
    assert d["created_at"].replace(tzinfo=timezone.utc).utcoffset().total_seconds() == 0


def test_receipt_not_written_when_matched(monkeypatch):
    monkeypatch.setattr(idn, "_run_identify", _fake(manual=False, item={"id": "x"}))
    _go()
    assert _db.identify_receipts.count_documents({}) == 0


def test_receipt_not_written_when_disabled(monkeypatch):
    monkeypatch.setattr(get_settings(), "ai_catalog_enabled", False)
    monkeypatch.setattr(idn, "_run_identify", _fake(manual=True))
    _go()
    assert _db.identify_receipts.count_documents({}) == 0


def test_write_receipt_never_raises():
    class Boom:
        def __getattr__(self, n):
            raise RuntimeError("db down")
    idn.write_receipt(Boom(), "u1", IMG, 0.4, False)


def test_receipt_tz_aware_in_doc():
    seen = {}
    class Coll:
        def insert_one(self, d):
            seen.update(d)
    class Db:
        identify_receipts = Coll()
    idn.write_receipt(Db(), "u1", IMG, 0.5, False)
    assert seen["created_at"].tzinfo is not None


def test_indexes_declared():
    ttl = get_settings().ai_catalog_receipt_ttl_s
    r = dbmod.INDEXES["identify_receipts"]
    assert any(o.get("expireAfterSeconds") == ttl for _, o in r)
    u = dbmod.INDEXES["ai_catalog_usage"]
    assert any(o.get("unique") for _, o in u)
