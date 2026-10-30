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


# ---------------------------------------------------------------- Task 2
import json as _json

from app.services import claude_vision as cv


class _Resp:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else _json.dumps(body or {})

    def json(self):
        return self._body


def _ok(payload):
    return _Resp(200, {"content": [{"type": "text", "text": _json.dumps(payload)}]})


@pytest.fixture
def cvenv(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")
    monkeypatch.setattr(cv, "_prepare", lambda b: ("b64", "image/jpeg"))
    cv._cache.clear()
    cv._cat_cache.clear()
    cv._breaker_reset()
    calls = []
    state = {"resp": _ok({})}

    def post(*a, **k):
        calls.append(k)
        return state["resp"]

    monkeypatch.setattr(cv.requests, "post", post)
    yield calls, state
    cv._breaker_reset()


def test_read_for_catalog_parses_json(cvenv):
    calls, st = cvenv
    st["resp"] = _ok({"es_diecast": True, "brand": "Mini GT", "casting": "Skyline",
                      "color": "azul", "year": 1999, "confidence": "alto"})
    r = cv.read_for_catalog(b"img1")
    assert r["es_diecast"] is True and r["brand"] == "Mini GT" and r["year"] == 1999
    assert r["confidence"] == "alto" and r["modelo"] is None and r["toy_number"] is None
    body = calls[0]["json"]
    assert body["temperature"] == 0 and body["max_tokens"] >= 600
    txt = body["messages"][0]["content"][1]["text"]
    assert "nunca inventes" in txt
    cv.read_for_catalog(b"img1")
    assert len(calls) == 1  # caché


def test_read_for_catalog_none_without_key(cvenv, monkeypatch):
    calls, _ = cvenv
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert cv.read_for_catalog(b"img2") is None
    assert calls == []


def test_year_not_in_image_is_null(cvenv):
    _, st = cvenv
    st["resp"] = _ok({"es_diecast": True, "brand": "Hot Wheels", "year": None,
                      "texto": "Copyright 1999"})
    assert cv.read_for_catalog(b"img3")["year"] is None
    st["resp"] = _ok({"es_diecast": True, "year": "1999"})
    assert cv.read_for_catalog(b"img3b")["year"] is None


def test_confidence_coerced(cvenv):
    _, st = cvenv
    st["resp"] = _ok({"es_diecast": True, "confidence": "altísimo"})
    assert cv.read_for_catalog(b"img4")["confidence"] == "bajo"


def test_failures_not_cached_and_unparseable_none(cvenv):
    calls, st = cvenv
    st["resp"] = _Resp(500, {}, "boom")
    assert cv.read_for_catalog(b"img5") is None
    st["resp"] = _Resp(200, {"content": [{"type": "text", "text": "sin json"}]})
    assert cv.read_for_catalog(b"img5") is None
    assert len(calls) == 2


def test_breaker_opens_on_credit_error_and_stops_calls(cvenv):
    calls, st = cvenv
    st["resp"] = _Resp(400, {}, '{"error":{"message":"Your credit balance is too low"}}')
    assert cv.read_for_catalog(b"a") is None
    assert cv.breaker_open() is True
    assert len(calls) == 1
    assert cv.read_for_catalog(b"b") is None
    assert cv.read_packaging_with_claude(b"c") is None
    assert len(calls) == 1


def test_breaker_opens_from_packaging_on_401(cvenv):
    calls, st = cvenv
    st["resp"] = _Resp(401, {}, "unauthorized")
    assert cv.read_packaging_with_claude(b"p") is None
    assert cv.breaker_open() is True
    assert cv.read_for_catalog(b"q") is None
    assert len(calls) == 1


def test_breaker_not_opened_by_other_400(cvenv):
    _, st = cvenv
    st["resp"] = _Resp(400, {}, "invalid image")
    cv.read_for_catalog(b"z")
    assert cv.breaker_open() is False


def test_breaker_closes_after_window(cvenv, monkeypatch):
    calls, st = cvenv
    clock = {"t": 1000.0}
    monkeypatch.setattr(cv, "_now", lambda: clock["t"])
    st["resp"] = _Resp(403, {}, "forbidden")
    cv.read_for_catalog(b"a")
    assert cv.breaker_open() is True
    clock["t"] += 299
    assert cv.breaker_open() is True
    clock["t"] += 2
    assert cv.breaker_open() is False
    st["resp"] = _ok({"es_diecast": True})
    assert cv.read_for_catalog(b"b") is not None
    assert len(calls) == 2


def test_breaker_logs_once(cvenv, capsys):
    _, st = cvenv
    st["resp"] = _Resp(401, {}, "x")
    cv.read_for_catalog(b"a")
    cv.read_for_catalog(b"b")
    cv.read_packaging_with_claude(b"c")
    assert capsys.readouterr().out.count("breaker abierto") == 1
