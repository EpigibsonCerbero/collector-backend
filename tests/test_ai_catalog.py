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
from app.routers import garage as _garage
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
    cv._cat_cache.clear()
    getattr(cv, "_logged", set()).clear()
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


# ── Task 4: índice visual vivo ─────────────────────────────────────────────
import time as _time

import numpy as _np

from app.services import vision as _vis

_DIM = 768


def _unit(seed):
    v = _np.random.default_rng(seed).standard_normal(_DIM).astype(_np.float32)
    return v / _np.linalg.norm(v)


@pytest.fixture
def vidx(tmp_path, monkeypatch):
    """Índice grande sintético (30 vectores) + archivo vivo en tmp."""
    big = tmp_path / "big.json"
    rows = [
        {"id": f"b{i}", "toy": f"MGT{i:05d}", "name": f"Big {i}", "view": 0, "vec": _unit(i).tolist()}
        for i in range(30)
    ]
    big.write_text(_json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(_vis, "DINOV2_INDEX", big)
    monkeypatch.setenv("EMB_LIVE_PATH", str(tmp_path / "live.json"))
    for name, val in (("_index", None), ("_matrix", None), ("_index_sig", None)):
        monkeypatch.setattr(_vis, name, val)
    _vis._live_reset()
    yield tmp_path
    _vis._live_reset()


def _entry(i, view=0):
    return {"id": f"live{i}", "toy": f"AI{i}", "name": f"Live {i}", "view": view}


def test_live_append_visible_in_match_embedding(vidx):
    v = _unit(1000)
    _vis.live_index_append(_entry(1), v.tolist())
    near = v + 0.01 * _unit(1001)
    res = _vis.match_embedding(near / _np.linalg.norm(near))
    assert res[0].id == "live1" and res[0].score > 0.99


def test_live_mv_entry_only_with_include_mv(vidx):
    v = _unit(1002)
    _vis.live_index_append(_entry(2, view=20), v.tolist())
    assert "live2" not in [m.id for m in _vis.match_embedding(v)]
    assert _vis.match_embedding(v, include_mv=True)[0].id == "live2"


def test_live_append_does_not_reload_big_index(vidx, monkeypatch):
    idx, mat = _vis.get_index()
    big_sig = _vis._index_sig
    loads = []
    real = _json.loads
    monkeypatch.setattr(_vis.json, "loads", lambda *a, **k: (loads.append(1), real(*a, **k))[1])
    _vis.live_index_append(_entry(3), _unit(1003).tolist())
    idx2, mat2 = _vis.get_index()
    assert _vis._index_sig == big_sig
    assert _vis._index is idx  # grande intacto, sin reparsear
    assert len(idx2) == len(idx) + 1 and mat2.shape[0] == mat.shape[0] + 1
    assert _vis.get_index()[0] is idx2  # fusionado cacheado
    assert loads == []


def test_live_index_persists_across_reload(vidx):
    v = _unit(1004)
    _vis.live_index_append(_entry(4), v.tolist())
    _vis._live_reset()  # borra el estado en memoria
    assert _vis.match_embedding(v)[0].id == "live4"


def test_live_file_change_reloads_only_live(vidx):
    _vis.get_index()
    other = [{**_entry(5), "vec": _unit(1005).tolist()}]
    (vidx / "live.json").write_text(_json.dumps(other), encoding="utf-8")
    assert _vis.match_embedding(_unit(1005))[0].id == "live5"


def test_live_corrupt_file_does_not_break_index(vidx):
    (vidx / "live.json").write_text("{not json", encoding="utf-8")
    idx, mat = _vis.get_index()
    assert len(idx) == 30


def test_live_append_atomic_on_failure(vidx, monkeypatch):
    _vis.live_index_append(_entry(6), _unit(1006).tolist())
    before = _vis.get_index()
    disk = (vidx / "live.json").read_text()

    def boom(*a, **k):
        raise OSError("disk full")

    with monkeypatch.context() as mp:
        mp.setattr(_vis.os, "replace", boom)
        with pytest.raises(OSError):
            _vis.live_index_append(_entry(7), _unit(1007).tolist())
    after = _vis.get_index()
    assert len(after[0]) == len(before[0]) == 31
    assert (vidx / "live.json").read_text() == disk
    assert [p.name for p in vidx.iterdir() if p.name.startswith("live") and p.name != "live.json"] == []


def test_live_dim_mismatch_rejected(vidx):
    _vis.get_index()
    with pytest.raises(ValueError):
        _vis.live_index_append(_entry(8), [0.1] * 10)
    assert len(_vis.get_index()[0]) == 30
    assert not (vidx / "live.json").exists()


def test_live_vector_is_l2_normalized(vidx):
    v = _unit(1009) * 5.0
    _vis.live_index_append(_entry(9), v.tolist())
    _, mat = _vis.get_index()
    assert abs(float(_np.linalg.norm(mat[-1])) - 1.0) < 1e-5


def test_match_identical_without_live(vidx):
    q = _unit(3)
    res = _vis.match_embedding(q)
    assert res[0].id == "b3" and abs(res[0].score - 1.0) < 1e-5


def test_live_perf_overhead_100_entries(vidx):
    q = _unit(77)
    _vis.match_embedding(q)

    def t():
        s = _time.perf_counter()
        for _ in range(200):
            _vis.match_embedding(q)
        return (_time.perf_counter() - s) / 200 * 1000

    base = t()
    for i in range(100):
        _vis.live_index_append(_entry(100 + i), _unit(2000 + i).tolist())
    _vis.match_embedding(q)
    with_live = t()
    print(f"\nmatch_embedding base={base:.3f}ms live100={with_live:.3f}ms delta={with_live-base:.3f}ms")
    assert with_live - base < 5.0


# ── Task 3: servicio ai_catalog ────────────────────────────────────────────
import io as _io
import uuid as _uuid
from types import SimpleNamespace as _NS

from PIL import Image as _Image

from app.routers import catalog as _catr
from app.routers import uploads as _upl
from app.schemas import ItemVariantOut as _IVO
from app.services import ai_catalog as aic
from app.services import car_segment as _seg
from app.services import dinov2_embed as _emb
from app.services import subscription as _subs

_NSID = _uuid.UUID("a9f73cdf-180a-5bd7-ac02-42c291653666")


def _jpeg(color=(200, 30, 30), size=(64, 48)):
    buf = _io.BytesIO()
    _Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


PHOTO = _jpeg()


def _read(**kw):
    r = {"es_diecast": True, "brand": "Mini GT", "casting": "Nissan Skyline GT-R R34",
         "modelo": None, "color": "azul", "dorsal": None, "livery": "Bayside Blue",
         "texto": "", "toy_number": None, "serie": None, "year": None, "confidence": "alto"}
    r.update(kw)
    return r


@pytest.fixture
def aienv(monkeypatch, tmp_path):
    for coll in ("variants", "items", "ai_catalog_usage", "brands"):
        _db[coll].delete_many({})
    monkeypatch.setattr(_upl, "UPLOAD_DIR", tmp_path)
    crop = _Image.new("RGB", (32, 32), (10, 10, 200))
    state = {"seg": _NS(image=crop), "appends": [], "embed_calls": [],
             "text": [], "match": [], "text_calls": [], "match_calls": [], "truncate": False}

    monkeypatch.setattr(_seg, "isolate_car", lambda im, *a, **k: state["seg"])

    def emb(im):
        state["embed_calls"].append(im)
        return [1.0] + [0.0] * 767 if im is crop else [0.0, 1.0] + [0.0] * 766

    monkeypatch.setattr(_emb, "embed_image", emb)
    def text(q, top_k=8, budget_s=2.0, stats=None):
        state["text_calls"].append((q, budget_s))
        if stats is not None:
            stats["truncated"] = state["truncate"]
        return list(state["text"])

    def match(e, top_k=8, include_mv=False):
        state["match_calls"].append((top_k, include_mv))
        return list(state["match"])

    monkeypatch.setattr(_vis, "search_catalog_by_text", text)
    monkeypatch.setattr(_vis, "match_embedding", match)
    monkeypatch.setattr(_vis, "live_index_append",
                        lambda entry, vec: state["appends"].append((dict(entry), list(vec))))
    state["crop"] = crop
    return state


def _vm(id_, name, score):
    return _vis.VisionMatch(id=id_, toy=None, name=name, score=score, source="ocr")


def _seed_variant(_id="exist1", name="Nissan Skyline GT-R R34", variant_name="Bayside Blue", toy="MGT00100",
                  manufacturer="Mini GT"):
    _db.items.insert_one({"_id": "mexist", "name": name, "manufacturer": manufacturer, "release_year": 2023})
    _db.variants.insert_one({"_id": _id, "item_master_id": "mexist", "variant_name": variant_name,
                             "toy_number": toy, "ai_local_hash": f"h-{_id}"})


def test_gate_rejects_non_diecast():
    assert aic.gate(None, {"top_score": 0.1}) == "no_diecast"
    assert aic.gate(_read(es_diecast=False), {"top_score": 0.1}) == "no_diecast"


def test_gate_rejects_low_confidence():
    assert aic.gate(_read(confidence="medio"), {"top_score": 0.1}) == "low_confidence"
    assert aic.gate(_read(confidence="bajo"), {}) == "low_confidence"
    assert aic.gate(_read(brand=""), {"top_score": 0.1}) == "missing_identity"
    assert aic.gate(_read(casting=None), {"top_score": 0.1}) == "missing_identity"


def test_gate_rejects_twins_and_high_top_score():
    mx = get_settings().ai_catalog_max_top_score
    assert aic.gate(_read(), {"top_score": 0.2, "twins": True}) == "visual_candidate"
    assert aic.gate(_read(), {"top_score": mx, "twins": False}) == "visual_candidate"
    assert aic.gate(_read(), {"top_score": mx - 0.01, "twins": False}) is None
    # top_score desconocido (salida manual temprana) no basta por sí solo
    assert aic.gate(_read(), {"top_score": 0.0}) is None
    assert aic.gate(_read(confidence="medio"), {"top_score": 0.0}) == "low_confidence"


def test_normalize_key_stable():
    a = aic.normalize_key("Mini GT", "Nissan Skyline", "Azúl", "#23", "Bayside, Blue")
    b = aic.normalize_key("mini gt", "skyline nissan", "AZUL", "23", "blue bayside")
    assert a == b and a == " ".join(sorted(a.split()))
    assert "ú" not in a and "#" not in a


def test_toy_number_uses_scraper_id():
    sid = aic.make_source_id(_read(toy_number="MGT00692"))
    assert sid == "minigt-MGT00692"
    assert aic.piece_id(sid) == str(_uuid.uuid5(_NSID, "minigt-MGT00692"))
    assert aic.make_source_id(_read(toy_number="692")) == "minigt-MGT00692"
    sid2 = aic.make_source_id(_read())
    assert sid2.startswith("ai-minigt-") and len(sid2) == len("ai-minigt-") + 12
    assert sid2 == aic.make_source_id(_read(livery="blue bayside"))


def test_find_duplicate_by_id_and_toy(aienv):
    r = _read(toy_number="MGT00100")
    _seed_variant()
    d = aic.find_duplicate(_db, r, None)
    assert d and d["id"] == "exist1" and d["created"] is False
    assert d["item_master"]["name"] == "Nissan Skyline GT-R R34"


def test_find_duplicate_by_text_returns_existing(aienv):
    # el maestro en Mongo no lo encuentra la consulta por casting; el índice sí
    _seed_variant(toy="MGT09999", name="GTR Special")
    aienv["text"] = [_vm("exist1", "Nissan Skyline GT-R R34 Bayside Blue", 0.9)]
    chk = aic.check_duplicate(_db, _read(), None)
    assert chk.outcome == "duplicate" and chk.reason == "text" and chk.piece["id"] == "exist1"
    q, budget = aienv["text_calls"][-1]
    assert "Skyline" in q and "Bayside Blue" in q and budget == aic.DUP_TEXT_BUDGET_S


def test_find_duplicate_text_twin_livery_not_dup(aienv):
    _seed_variant(toy="MGT09999", variant_name="Midnight Purple")
    aienv["text"] = [_vm("exist1", "Nissan Skyline GT-R R34 Midnight Purple", 0.9)]
    assert aic.find_duplicate(_db, _read(), None) is None
    # candidato textual sin variante real en Mongo: no cuenta
    aienv["text"] = [_vm("ghost", "Nissan Skyline GT-R R34 Bayside Blue", 0.9)]
    assert aic.find_duplicate(_db, _read(), None) is None


def test_find_duplicate_by_embedding_returns_existing(aienv):
    _seed_variant(toy="MGT09999")
    aienv["match"] = [_vis.VisionMatch(id="exist1", toy=None, name="x", score=0.91, source="embed")]
    d = aic.find_duplicate(_db, _read(livery="otra", casting="Otro Coche"), [0.1] * 768)
    assert d and d["id"] == "exist1"
    assert aienv["match_calls"][-1] == (3, True)  # índice completo, vistas aisladas incluidas
    aienv["match"] = [_vis.VisionMatch(id="exist1", toy=None, name="x", score=0.89, source="embed")]
    assert aic.find_duplicate(_db, _read(livery="otra", casting="Otro Coche"), [0.1] * 768) is None
    # sin recorte no se consulta el índice visual
    aienv["match"] = [_vis.VisionMatch(id="exist1", toy=None, name="x", score=0.99, source="embed")]
    assert aic.find_duplicate(_db, _read(livery="otra", casting="Otro Coche"), None) is None


def test_create_from_read_fields_and_live_index(aienv):
    out = aic.create_from_read(_db, "u1", _read(toy_number="MGT00692", dorsal="23"), PHOTO)
    assert out["created"] is True and out["outcome"] == "created"
    assert out["id"] == aic.piece_id("minigt-MGT00692")
    doc = _db.variants.find_one({"_id": out["id"]})
    assert doc["origin"] == "ai" and doc["verified"] is False and doc["created_by"] == "u1"
    assert doc["source_confidence"] == "alto" and doc["confirmed_by"] == []
    assert doc["ai_local_hash"] and doc["toy_number"] == "MGT00692"
    assert doc["livery_number"] == "23" and doc["body_color"] == "azul"
    assert out["main_image_url"].startswith("/api/v1/uploads/") and out["images"] == [out["main_image_url"]]
    assert out["item_master"]["manufacturer"] == "Mini GT"
    _IVO(**out)  # contrato de la app intacto
    views = sorted(e["view"] for e, _ in aienv["appends"])
    assert views == [0, 20]
    assert all(e["id"] == out["id"] for e, _ in aienv["appends"])
    v20 = [v for e, v in aienv["appends"] if e["view"] == 20][0]
    assert v20[0] == 1.0  # el de vista 20 es el del recorte


def test_same_piece_two_users_one_row(aienv):
    a = aic.create_from_read(_db, "u1", _read(), PHOTO)
    n_app = len(aienv["appends"])
    b = aic.create_from_read(_db, "u2", _read(), PHOTO)
    assert a["id"] == b["id"] and b["created"] is False
    assert _db.variants.count_documents({}) == 1
    assert len(aienv["appends"]) == n_app  # sin entradas duplicadas en el índice vivo
    assert _db.variants.find_one({})["created_by"] == "u1"


def test_existing_piece_not_duplicated(aienv):
    _seed_variant(toy="MGT09999")
    out = aic.create_from_read(_db, "u1", _read(), PHOTO)
    assert out["id"] == "exist1" and out["created"] is False
    assert out["outcome"] == "duplicate" and out["reason"] == "same_casting_livery"
    assert _db.variants.count_documents({}) == 1 and aienv["appends"] == []


def test_year_and_series_null_when_not_read(aienv):
    out = aic.create_from_read(_db, "u1", _read(), PHOTO)
    doc = _db.variants.find_one({"_id": out["id"]})
    m = _db.items.find_one({"_id": as_oid(doc["item_master_id"])})
    assert m["release_year"] == 0
    assert doc["metadata"].get("year") is None and doc["metadata"].get("series") is None
    out2 = aic.create_from_read(_db, "u1", _read(casting="Honda NSX", year=2024, serie="Mijo"), PHOTO)
    d2 = _db.variants.find_one({"_id": out2["id"]})
    assert d2["metadata"]["year"] == 2024 and d2["metadata"]["series"] == "Mijo"
    assert out2["item_master"]["release_year"] == 2024
    out3 = aic.create_from_read(_db, "u1", _read(casting="Mazda RX7", year=1890), PHOTO)
    assert out3["item_master"]["release_year"] == 0
    assert _db.variants.find_one({"_id": out3["id"]})["metadata"].get("year") is None


def as_oid(v):
    from app.db import as_id
    return as_id(v)


def test_image_is_crop_or_none(aienv):
    out = aic.create_from_read(_db, "u1", _read(), PHOTO)
    saved = list(_upl.UPLOAD_DIR.iterdir())
    assert len(saved) == 1
    with _Image.open(saved[0]) as im:
        assert im.size == (32, 32)  # el recorte, nunca la foto cruda (64x48)
    aienv["seg"] = None
    aienv["appends"].clear()
    out2 = aic.create_from_read(_db, "u1", _read(casting="Honda NSX"), PHOTO)
    assert out2["main_image_url"] is None and out2["images"] == []
    assert [e["view"] for e, _ in aienv["appends"]] == [0]
    assert len(list(_upl.UPLOAD_DIR.iterdir())) == 1


def test_daily_cap_blocks_11th(aienv, monkeypatch):
    monkeypatch.setattr(get_settings(), "ai_catalog_daily_cap", 10)
    for _ in range(10):
        assert aic.daily_cap_reached(_db, "u1") is False
        aic.consume_daily(_db, "u1")
    assert aic.daily_cap_reached(_db, "u1") is True
    assert aic.daily_cap_reached(_db, "u2") is False
    d = _db.ai_catalog_usage.find_one({"user_id": "u1"})
    assert d["day"] == _subs.today_key() and d["count"] == 10


def test_strip_internal_removes_flags(aienv):
    out = aic.create_from_read(_db, "u1", _read(), PHOTO)
    s = aic.strip_internal(dict(out, hidden=False))
    for k in ("origin", "verified", "source_confidence", "confirmed_by", "created_by", "hidden", "created",
              "outcome", "reason"):
        assert k not in s
    assert s["ai_local_hash"] and s["id"] == out["id"]
    assert "origin" in out  # no muta la entrada


def test_upsert_extraction_keeps_create_catalog_item_behavior():
    for coll in ("variants", "items"):
        _db[coll].delete_many({})
    master = {"name": "Datsun 510", "manufacturer": "Mini GT", "release_year": 2026, "category_id": "0",
              "brand_id": None, "series_id": None, "casting_name": None, "designer": None, "scale": "1:64"}
    variant = {"id": "manual-1", "variant_name": "Std", "toy_number": "T1", "metadata": {}}
    v, m = _catr.upsert_catalog_piece(_db, dict(master), dict(variant), "u9")
    assert v["_id"] == "manual-1" and v["created_by"] == "u9" and v["user_submitted"] is True
    assert m["name"] == "Datsun 510" and "brand_id" not in m
    assert v["ai_local_hash"] == "hash-t1"
    # dedup por toy_number + variant_name → misma fila
    v2, _ = _catr.upsert_catalog_piece(_db, dict(master), dict(variant, id="manual-2"), "u8")
    assert v2["_id"] == "manual-1" and _db.variants.count_documents({}) == 1
    # mismo id sin dedup → 409 como antes
    from fastapi import HTTPException as _HE
    with pytest.raises(_HE) as ei:
        _catr.upsert_catalog_piece(_db, dict(master), dict(variant, toy_number="T2"), "u8")
    assert ei.value.status_code == 409


# ── Task 3, ronda 1: tres salidas anti-duplicado, toy# solo Mini GT, búsqueda truncada


def _rows():
    return _db.variants.count_documents({})


def test_two_wordings_same_piece_one_row(aienv):
    a = aic.create_from_read(_db, "u1", _read(livery="Bayside Blue", color="azul"), PHOTO)
    n_app = len(aienv["appends"])
    b = aic.create_from_read(_db, "u2", _read(livery="bayside blue metálico", color="Azul", casting="Skyline GT-R R34"), PHOTO)
    assert a["outcome"] == "created" and b["outcome"] == "duplicate" and b["id"] == a["id"]
    assert _rows() == 1 and len(aienv["appends"]) == n_app


def test_livery_missing_vs_present_is_uncertain_and_writes_nothing(aienv):
    aic.create_from_read(_db, "u1", _read(livery="Bayside Blue", color="azul"), PHOTO)
    n_app, n_files, n_items = len(aienv["appends"]), len(list(_upl.UPLOAD_DIR.iterdir())), _db.items.count_documents({})
    out = aic.create_from_read(_db, "u2", _read(livery=None, color="azul"), PHOTO)
    assert out == {"outcome": "uncertain", "reason": "same_casting_ambiguous", "created": False}
    assert _rows() == 1 and len(aienv["appends"]) == n_app
    assert len(list(_upl.UPLOAD_DIR.iterdir())) == n_files and _db.items.count_documents({}) == n_items


def test_same_casting_clearly_different_livery_or_dorsal_created(aienv):
    aic.create_from_read(_db, "u1", _read(livery="Bayside Blue", color="azul"), PHOTO)
    c = aic.create_from_read(_db, "u1", _read(livery="Midnight Purple", color="morado"), PHOTO)
    assert c["outcome"] == "created" and _rows() == 2
    aic.create_from_read(_db, "u1", _read(casting="Honda NSX", livery="Advan", color="blanco", dorsal="#23"), PHOTO)
    d = aic.create_from_read(_db, "u1", _read(casting="Honda NSX", livery="Advan", color="blanco", dorsal="24"), PHOTO)
    assert d["outcome"] == "created" and _rows() == 4


def test_no_livery_both_same_color_is_duplicate(aienv):
    a = aic.create_from_read(_db, "u1", _read(livery=None, color="rojo"), PHOTO)
    b = aic.create_from_read(_db, "u2", _read(livery=None, color="Red"), PHOTO)
    assert b["outcome"] == "duplicate" and b["id"] == a["id"] and _rows() == 1


def test_brand_spellings_share_slug_and_id():
    ids = {aic.make_source_id(_read(brand=b)) for b in ("Mini GT", "MiniGT", "TSM Mini GT", "mini-gt")}
    assert len(ids) == 1 and ids.pop().startswith("ai-minigt-")
    assert aic.make_source_id(_read(brand="TSM MiniGT", toy_number="692")) == "minigt-MGT00692"


def test_non_minigt_toy_is_metadata_only(aienv):
    sid = aic.make_source_id(_read(brand="Hot Wheels", casting="Twin Mill", toy_number="23"))
    assert sid.startswith("ai-hw-")
    a = aic.create_from_read(_db, "u1", _read(brand="Hot Wheels", casting="Twin Mill", livery="Flames",
                                               color="rojo", toy_number="23"), PHOTO)
    doc = _db.variants.find_one({"_id": a["id"]})
    assert doc.get("toy_number") is None and doc["metadata"]["toy_number"] == "23"
    b = aic.create_from_read(_db, "u1", _read(brand="Hot Wheels", casting="Bone Shaker", livery="Skulls",
                                               color="negro", toy_number="23"), PHOTO)
    assert b["outcome"] == "created" and b["id"] != a["id"] and _rows() == 2


def test_minigt_toy_hit_needs_brand_and_casting(aienv):
    # toy# leído existe pero es de otro casting (lectura dudosa 692↔629): incierto
    _seed_variant(toy="MGT00629", name="Honda NSX")
    out = aic.create_from_read(_db, "u1", _read(toy_number="629"), PHOTO)
    assert out["outcome"] == "uncertain" and out["reason"] == "toy_name_mismatch" and _rows() == 1
    # mismo toy# en otra marca: no es identidad Mini GT → incierto, nunca la devuelve
    _db.variants.delete_many({}); _db.items.delete_many({})
    _seed_variant(toy="MGT00629", manufacturer="Inno64")
    out = aic.create_from_read(_db, "u1", _read(toy_number="629"), PHOTO)
    assert out["outcome"] == "uncertain" and _rows() == 1


def test_truncated_text_search_is_uncertain(aienv, monkeypatch):
    aienv["truncate"] = True
    out = aic.create_from_read(_db, "u1", _read(), PHOTO)
    assert out["outcome"] == "uncertain" and out["reason"] == "text_search_truncated"
    assert _rows() == 0 and aienv["appends"] == []
    aienv["truncate"] = False
    ticks = iter([0.0, aic.DUP_TEXT_BUDGET_S + 0.1])
    monkeypatch.setattr(aic, "_clock", lambda: next(ticks))
    assert aic.check_duplicate(_db, _read(), None).outcome == "uncertain"


def test_text_search_truncation_flag_in_vision(monkeypatch):
    monkeypatch.setattr(_vis, "get_index", lambda: ([{"id": "a", "name": "Nissan Skyline"}], None))
    _vis.search_catalog_by_text("nissan skyline bayside", budget_s=10.0, stats=(st := {}))
    assert st == {"truncated": False}
    clock = iter([0.0] + [100.0] * 100000)
    monkeypatch.setattr(_vis.time, "time", lambda: next(clock))
    _vis.search_catalog_by_text("nissan skyline bayside", budget_s=1.0, stats=(st2 := {}))
    assert st2["truncated"] is True


def test_upsert_dedupe_id_race_returns_existing():
    for coll in ("variants", "items"):
        _db[coll].delete_many({})
    _db.variants.insert_one({"_id": "race1", "item_master_id": "m", "variant_name": "X", "ai_local_hash": "hr"})

    class Coll:
        def __init__(self, c):
            self.c, self.misses = c, 1

        def find_one(self, q, *a, **k):
            if "_id" in q and self.misses:
                self.misses -= 1
                return None  # la búsqueda previa no lo ve: el otro usuario aún no escribía
            return self.c.find_one(q, *a, **k)

        def __getattr__(self, n):
            return getattr(self.c, n)

    class Db:
        variants = Coll(_db.variants)
        items = _db.items

    master = {"name": "N", "manufacturer": "Mini GT", "release_year": 0}
    v, _m, created = _catr._upsert_catalog_piece(Db(), master, {"id": "race1", "variant_name": "Y"}, "u2",
                                                  dedupe_id=True)
    assert created is False and v["_id"] == "race1" and _db.variants.count_documents({}) == 1


# ── Task 3, ronda 2: crear solo ante diferencia CLARA


def _snapshot(aienv):
    return (_rows(), _db.items.count_documents({}), len(aienv["appends"]),
            len(list(_upl.UPLOAD_DIR.iterdir())))


@pytest.mark.parametrize("dorsals", [(None, None), ("23", None), (None, "23"), ("23", "#23")])
def test_strong_livery_color_mismatch_is_duplicate(aienv, dorsals):
    a = aic.create_from_read(_db, "u1", _read(casting="Honda NSX", livery="Advan", color="blanco",
                                               dorsal=dorsals[0]), PHOTO)
    before = _snapshot(aienv)
    r2 = _read(casting="Honda NSX", livery="ADVAN racing", color="rojo", dorsal=dorsals[1])
    assert aic.make_source_id(r2) != aic.make_source_id(_read(casting="Honda NSX", livery="Advan",
                                                              dorsal=dorsals[0]))  # pasa por el veredicto
    b = aic.create_from_read(_db, "u2", r2, PHOTO)
    assert b["outcome"] == "duplicate" and b["reason"] == "same_casting_livery" and b["id"] == a["id"]
    assert _snapshot(aienv) == before


def test_disjoint_livery_color_one_side_is_uncertain(aienv):
    _seed_variant(toy="MGT09999", variant_name="Gulf")  # scrape: sin color guardado
    before = _snapshot(aienv)
    out = aic.create_from_read(_db, "u1", _read(livery="Azul celeste con naranja", color="azul"), PHOTO)
    assert out["outcome"] == "uncertain" and out["reason"] == "same_casting_ambiguous"
    assert _snapshot(aienv) == before


def test_disjoint_livery_colors_differ_is_created(aienv):
    _seed_variant(toy="MGT09999", variant_name="Gulf")
    _db.variants.update_one({"_id": "exist1"}, {"$set": {"body_color": "Red"}})
    out = aic.create_from_read(_db, "u1", _read(livery="Martini", color="blanco"), PHOTO)
    assert out["outcome"] == "created" and _rows() == 2


def test_disjoint_livery_no_colors_dorsal_differs_is_created(aienv):
    _seed_variant(toy="MGT09999", variant_name="Gulf")
    _db.variants.update_one({"_id": "exist1"}, {"$set": {"livery_number": "7"}})
    out = aic.create_from_read(_db, "u1", _read(livery="Martini", color=None, dorsal="#8"), PHOTO)
    assert out["outcome"] == "created" and _rows() == 2


def test_strong_livery_dorsal_conflict_is_created(aienv):
    a = aic.create_from_read(_db, "u1", _read(casting="Honda NSX", livery="Advan", color="blanco", dorsal="23"), PHOTO)
    b = aic.create_from_read(_db, "u1", _read(casting="Honda NSX", livery="Advan", color="rojo", dorsal="#24"), PHOTO)
    assert b["outcome"] == "created" and b["id"] != a["id"] and _rows() == 2


def test_no_livery_dorsal_or_color_differs_is_uncertain(aienv):
    aic.create_from_read(_db, "u1", _read(livery=None, color="rojo", dorsal="5"), PHOTO)
    before = _snapshot(aienv)
    assert aic.create_from_read(_db, "u2", _read(livery=None, color="rojo", dorsal="6"), PHOTO)["outcome"] == "uncertain"
    assert aic.create_from_read(_db, "u2", _read(livery=None, color="verde", dorsal="5"), PHOTO)["outcome"] == "uncertain"
    assert _snapshot(aienv) == before


def test_same_casting_scan_cap_is_uncertain(aienv, monkeypatch):
    monkeypatch.setattr(aic, "SAME_CASTING_SCAN_LIMIT", 2)
    for i in range(2):
        _db.items.insert_one({"_id": f"mcap{i}", "name": f"Nissan Skyline GT-R R34 {i}", "manufacturer": "Inno64",
                              "release_year": 2020})
    before = _snapshot(aienv)
    out = aic.create_from_read(_db, "u1", _read(), PHOTO)
    assert out["outcome"] == "uncertain" and out["reason"] == "same_casting_scan_capped"
    assert _snapshot(aienv) == before


# ── Task 5: endpoint POST /identify/resolve-new ────────────────────────────
import base64 as _b64
import logging as _logging
from datetime import datetime as _dt, timedelta as _td

from fastapi import FastAPI as _FastAPI
from fastapi.testclient import TestClient as _TC

from app.security import get_current_user as _gcu

_api = _FastAPI()
_api.include_router(idn.router, prefix="/api/v1")
_client = _TC(_api)
_URL = "/api/v1/identify/resolve-new"


@pytest.fixture
def rn(aienv, monkeypatch):
    """Endpoint con sesión simulada, IA simulada y pipeline pesado de aienv."""
    who = {"id": "u1"}
    _api.dependency_overrides[_gcu] = lambda: CurrentUser({"id": who["id"], "username": who["id"]})
    calls = []
    state = {"read": _read(), "who": who, "calls": calls}

    def fake_read(data):
        calls.append(data)
        r = state["read"]
        if isinstance(r, Exception):
            raise r
        return dict(r) if r else r

    monkeypatch.setattr(cv, "read_for_catalog", fake_read)
    yield state
    _api.dependency_overrides.clear()


def _post(img=PHOTO, raw=None, headers=None):
    body = {"image_base64": raw if raw is not None else _b64.b64encode(img).decode()}
    return _client.post(_URL, json=body, headers=headers or {})


def _receipt(uid="u1", img=PHOTO, top=0.3, twins=False, age_s=0):
    _db.identify_receipts.insert_one({
        "user_id": uid, "image_hash": idn.image_hash(img), "top_score": top, "twins": twins,
        "created_at": _dt.now(timezone.utc) - _td(seconds=age_s),
    })


def _is_manual(j):
    return j["method"] == "m" and j["needs_manual"] is True and j["item"] is None and j["suggestions"] == []


def test_disabled_returns_manual_without_calling_ai(rn, monkeypatch):
    monkeypatch.setattr(get_settings(), "ai_catalog_enabled", False)
    _receipt()

    class Boom:
        def __getattr__(self, n):
            raise AssertionError("no DB con el flag apagado")

    monkeypatch.setattr(idn, "get_db", lambda: Boom())
    r = _post()
    assert r.status_code == 200 and _is_manual(r.json())
    assert rn["calls"] == [] and _rows() == 0


def test_requires_session_401(rn):
    _api.dependency_overrides.clear()
    _receipt()
    assert _post().status_code == 401
    assert rn["calls"] == []


def test_invalid_image_400(rn):
    _receipt()
    assert _post(raw="%%%no-base64%%%").status_code == 400
    assert _post(raw=_b64.b64encode(b"not an image").decode()).status_code == 400
    assert _post(raw="").status_code == 400
    assert rn["calls"] == []


def test_data_url_prefix_accepted(rn):
    _receipt()
    r = _post(raw="data:image/jpeg;base64," + _b64.b64encode(PHOTO).decode())
    assert r.status_code == 200 and r.json()["method"] == "o"


def test_no_receipt_403_and_no_ai_call(rn):
    r = _post()
    assert r.status_code == 403 and rn["calls"] == [] and _rows() == 0


def test_receipt_for_other_image_403(rn):
    _receipt(img=_jpeg((0, 200, 0)))
    assert _post().status_code == 403
    _receipt(uid="u2")  # misma foto, otro usuario
    assert _post().status_code == 403
    assert rn["calls"] == []


def test_expired_receipt_403(rn):
    ttl = get_settings().ai_catalog_receipt_ttl_s
    _receipt(age_s=ttl + 5)  # el índice TTL aún no lo borró
    assert _post().status_code == 403 and rn["calls"] == []
    _receipt(age_s=ttl - 30)
    assert _post().status_code == 200


def test_success_returns_item_without_internal_flags(rn, caplog):
    _receipt()
    with caplog.at_level(_logging.INFO, logger=idn._log.name):
        r = _post()
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["method"] == "o" and j["needs_manual"] is False and j["confidence"] >= 0.9
    assert j["suggestions"] == []
    item = j["item"]
    for k in ("origin", "verified", "source_confidence", "confirmed_by", "created_by", "hidden",
              "created", "outcome", "reason"):
        assert k not in item
    assert item["id"] == _db.variants.find_one({})["_id"] and _rows() == 1
    assert item["item_master"]["manufacturer"] == "Mini GT"
    _IVO(**item)  # mismo contrato que el item del identify
    assert _db.ai_catalog_usage.find_one({"user_id": "u1"})["count"] == 1
    assert len(rn["calls"]) == 1
    msgs = [rec.getMessage() for rec in caplog.records if "ai_catalog" in rec.getMessage()]
    assert msgs == ["ai_catalog created user=u1"]


def test_receipt_consumed_no_replay(rn):
    _receipt()
    _receipt()  # dos escaneos de la misma foto: ninguno sirve tras el alta
    assert _post().status_code == 200
    assert _db.identify_receipts.count_documents({}) == 0
    assert _post().status_code == 403
    assert len(rn["calls"]) == 1


def test_existing_piece_returned_not_created(rn):
    _seed_variant(toy="MGT09999")
    _receipt()
    r = _post()
    j = r.json()
    assert r.status_code == 200 and j["method"] == "o" and j["needs_manual"] is False
    assert j["item"]["id"] == "exist1" and "outcome" not in j["item"]
    assert _rows() == 1
    assert _db.ai_catalog_usage.count_documents({}) == 0  # no gasta el tope diario
    assert _db.identify_receipts.count_documents({}) == 0  # recibo consumido


def test_uncertain_returns_manual_keeps_receipt(rn, aienv, caplog):
    aienv["truncate"] = True  # búsqueda textual no concluyente → «uncertain» real
    _receipt()
    with caplog.at_level(_logging.INFO, logger=idn._log.name):
        r = _post()
    assert r.status_code == 200 and _is_manual(r.json())
    assert _rows() == 0 and aienv["appends"] == []
    assert _db.identify_receipts.count_documents({}) == 1
    assert _db.ai_catalog_usage.count_documents({}) == 0
    assert "ai_catalog uncertain:text_search_truncated user=u1" in caplog.text


def test_gate_rejection_returns_manual(rn):
    _receipt(twins=True)
    r = _post()
    assert r.status_code == 200 and _is_manual(r.json()) and _rows() == 0
    assert rn["calls"] == []  # barrera del recibo: sin lectura pagada
    rn["read"] = _read(es_diecast=False)
    _receipt(img=_jpeg((1, 2, 3)))
    assert _is_manual(_post(img=_jpeg((1, 2, 3))).json()) and _rows() == 0


def test_cap_reached_returns_manual(rn, monkeypatch):
    monkeypatch.setattr(get_settings(), "ai_catalog_daily_cap", 1)
    aic.consume_daily(_db, "u1")
    _receipt()
    r = _post()
    assert r.status_code == 200 and _is_manual(r.json())
    assert rn["calls"] == [] and _rows() == 0  # sobre el tope no cuesta nada


def test_ai_error_returns_manual(rn):
    _receipt()
    rn["read"] = None
    assert _is_manual(_post().json())
    rn["read"] = RuntimeError("api down")
    r = _post()
    assert r.status_code == 200 and _is_manual(r.json())
    assert _rows() == 0 and _db.identify_receipts.count_documents({}) == 1


def test_rate_limited(rn):
    rn["read"] = None  # cada llamada sale a manual y deja el recibo
    _receipt()
    for _ in range(6):
        assert _post().status_code == 200
    assert _post().status_code == 429
    assert len(rn["calls"]) == 6
    rn["who"]["id"] = "u2"  # el tope es por usuario
    _receipt(uid="u2")
    assert _post().status_code == 200


def test_does_not_consume_extra_scan_quota(rn, monkeypatch):
    monkeypatch.setattr(get_settings(), "scan_limit_enabled", True)
    _db.scan_usage.insert_one({"user_id": "u1", "day": _subs.today_key(), "count": 5})
    called = []
    monkeypatch.setattr(_subs, "reserve_scan", lambda *a, **k: called.append(1) or (True, 6))
    _receipt()
    assert _post().json()["method"] == "o"
    assert _db.scan_usage.find_one({"user_id": "u1"})["count"] == 5 and called == []


# ── Task 5, ronda 1: la lectura no ocupa slot, sin repetición pagada, sin 500 tras el alta


def test_ai_read_does_not_hold_identify_slot(rn, monkeypatch):
    seen = {}
    real_read = cv.read_for_catalog
    real_create = aic.create_from_read

    def read_spy(data):
        seen["read"] = idn._IDENTIFY_SLOTS._value
        return real_read(data)

    def create_spy(*a, **k):
        seen["create"] = idn._IDENTIFY_SLOTS._value
        return real_create(*a, **k)

    monkeypatch.setattr(cv, "read_for_catalog", read_spy)
    monkeypatch.setattr(aic, "create_from_read", create_spy)
    _receipt()
    assert _post().json()["method"] == "o"
    assert seen == {"read": 2, "create": 1}  # solo el alta (CPU) toma el semáforo


def test_replay_after_gate_rejection_makes_no_ai_call(rn):
    # Barrera del recibo (gemelos): ni la primera vez se paga la lectura.
    _receipt(twins=True)
    assert _is_manual(_post().json()) and rn["calls"] == []
    assert _is_manual(_post().json()) and rn["calls"] == []
    # Barrera de la lectura (no es diecast): una lectura y nunca más.
    _db.identify_receipts.delete_many({})
    rn["read"] = _read(es_diecast=False)
    _receipt()
    assert _is_manual(_post().json()) and len(rn["calls"]) == 1
    assert _db.identify_receipts.find_one({})["ai_tried"] is True
    r = _post()
    assert r.status_code == 200 and _is_manual(r.json())
    assert len(rn["calls"]) == 1 and _rows() == 0


def test_replay_after_uncertain_makes_no_ai_call(rn, aienv):
    aienv["truncate"] = True
    _receipt()
    assert _is_manual(_post().json()) and len(rn["calls"]) == 1
    aienv["truncate"] = False  # aunque ahora sí se pudiera crear, el escaneo ya se gastó
    assert _is_manual(_post().json())
    assert len(rn["calls"]) == 1 and _rows() == 0


def test_retry_after_ai_error_calls_ai_again(rn):
    _receipt()
    rn["read"] = RuntimeError("api down")
    assert _is_manual(_post().json())
    assert not _db.identify_receipts.find_one({}).get("ai_tried")
    rn["read"] = _read()
    assert _post().json()["method"] == "o"
    assert len(rn["calls"]) == 2


def test_bookkeeping_failure_after_create_is_not_500(rn, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(aic, "consume_daily", boom)
    _receipt()
    r = _post()
    assert r.status_code == 200 and r.json()["method"] == "o" and _rows() == 1
    assert _db.identify_receipts.count_documents({}) == 0


# ── Task 6: verificación comunitaria ───────────────────────────────────────


def _ai_piece(_id="ai1", created_by="u1", **kw):
    _db.variants.delete_many({})
    d = {"_id": _id, "origin": "ai", "verified": False, "created_by": created_by,
         "confirmed_by": [], "variant_name": "Bayside Blue", "ai_local_hash": "h" + _id}
    d.update(kw)
    _db.variants.insert_one(d)
    return _id


def _v(_id="ai1"):
    return _db.variants.find_one({"_id": _id})


def test_creator_alone_does_not_verify():
    _ai_piece()
    assert aic.confirm(_db, "ai1", "u1") is False
    assert _v()["verified"] is False and _v()["confirmed_by"] == []


def test_second_distinct_user_verifies():
    _ai_piece()
    assert aic.confirm(_db, "ai1", "u2") is True
    assert _v()["verified"] is True and _v()["confirmed_by"] == ["u2"]
    assert aic.confirm(_db, "ai1", "u3") is False  # ya verificada: no vuelve a True
    assert _v()["verified"] is True


def test_same_user_twice_counts_once():
    _ai_piece(created_by=None)
    assert aic.confirm(_db, "ai1", "u2") is False
    assert aic.confirm(_db, "ai1", "u2") is False
    assert _v()["verified"] is False and _v()["confirmed_by"] == ["u2"]
    assert aic.confirm(_db, "ai1", "u3") is True


def test_two_non_creators_verify_without_creator():
    _ai_piece(created_by=None)
    assert aic.confirm(_db, "ai1", "a") is False
    assert aic.confirm(_db, "ai1", "b") is True


def _count_writes(monkeypatch):
    n = []
    for name in ("update_one", "find_one_and_update", "update_many"):
        real = getattr(_db.variants, name)
        monkeypatch.setattr(type(_db.variants), name,
                            lambda self, *a, _r=real, _n=name, **k: (n.append(_n), _r(*a, **k))[1],
                            raising=False)
    return n


def test_confirm_ignores_non_ai_pieces(monkeypatch):
    _db.variants.delete_many({})
    _db.variants.insert_one({"_id": "s1", "variant_name": "x"})
    _db.variants.insert_one({"_id": "s2", "origin": "scrape", "verified": True})
    writes = _count_writes(monkeypatch)
    assert aic.confirm(_db, "s1", "u2") is False and aic.confirm(_db, "s2", "u2") is False
    assert aic.confirm(_db, "missing", "u2") is False
    assert writes == []
    assert "confirmed_by" not in _v("s1") and "confirmed_by" not in _v("s2")


def test_creator_alone_writes_nothing(monkeypatch):
    _ai_piece()
    writes = _count_writes(monkeypatch)
    assert aic.confirm(_db, "ai1", "u1") is False
    assert writes == []


def test_stale_read_does_not_lose_verification(monkeypatch):
    """A y B leen confirmed_by=[] a la vez (sin creador): la 2ª decide con el
    documento fresco del $addToSet y verifica."""
    _ai_piece(created_by=None)
    stale = _v()
    monkeypatch.setattr(aic, "_resolve_variant", lambda db, vid: dict(stale))
    assert aic.confirm(_db, "ai1", "A") is False
    assert aic.confirm(_db, "ai1", "B") is True  # lectura vieja: confirmed_by=[]
    assert _v()["verified"] is True and sorted(_v()["confirmed_by"]) == ["A", "B"]


def test_failed_second_update_self_heals(monkeypatch):
    _ai_piece()
    cls = type(_db.variants)
    real = cls.update_one
    state = {"boom": True}

    def flaky(self, *a, **k):
        if state["boom"]:
            state["boom"] = False
            raise RuntimeError("db blip")
        return real(self, *a, **k)

    monkeypatch.setattr(cls, "update_one", flaky)
    assert aic.confirm(_db, "ai1", "u2") is False  # $addToSet ok, verified falló
    assert _v()["confirmed_by"] == ["u2"] and _v()["verified"] is False
    assert aic.confirm(_db, "ai1", "u2") is True  # misma usuaria reintenta y verifica
    assert _v()["verified"] is True
    assert aic.confirm(_db, "ai1", "u1") is False  # creador sobre pieza ya verificada


def test_creator_call_heals_when_others_confirmed():
    _ai_piece(confirmed_by=["u2"])  # quedó sin verificar por un fallo previo
    assert aic.confirm(_db, "ai1", "u1") is True
    assert _v()["verified"] is True


def test_confirm_never_raises():
    class Boom:
        class variants:
            @staticmethod
            def find_one(*a, **k):
                raise RuntimeError("down")

    assert aic.confirm(Boom, "x", "u") is False


def test_identify_winner_ai_confirms(monkeypatch):
    _ai_piece()

    def run(data, brand):
        idn._note_ai(_v())
        return idn.IdentifyOut(needs_manual=False)

    monkeypatch.setattr(idn, "_run_identify", run)
    asyncio.run(idn._identify_guarded(None, IMG, None, CurrentUser({"id": "u2"})))
    assert _v()["verified"] is True


def test_identify_hook_skips_verified_and_non_ai(monkeypatch):
    _ai_piece(verified=True)
    seen = []
    monkeypatch.setattr(aic, "confirm", lambda *a: seen.append(a))

    def run(data, brand):
        idn._note_ai(_v())
        idn._note_ai({"_id": "s1", "variant_name": "x"})
        return idn.IdentifyOut(needs_manual=False)

    monkeypatch.setattr(idn, "_run_identify", run)
    asyncio.run(idn._identify_guarded(None, IMG, None, CurrentUser({"id": "u2"})))
    assert seen == []


def test_identify_confirm_failure_does_not_break(monkeypatch):
    _ai_piece()
    monkeypatch.setattr(aic, "confirm", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))

    def run(data, brand):
        idn._note_ai(_v())
        return idn.IdentifyOut(needs_manual=False)

    monkeypatch.setattr(idn, "_run_identify", run)
    out = asyncio.run(idn._identify_guarded(None, IMG, None, CurrentUser({"id": "u2"})))
    assert out.needs_manual is False


def test_real_run_identify_toy_hit_confirms_once(monkeypatch):
    """Ruta real de _run_identify (solo `identify_image`, el pipeline pesado, se
    simula) con ganador IA sin verificar por toy number."""
    _db.items.delete_many({})
    _db.items.insert_one({"_id": "m1", "name": "GT-R", "manufacturer": "Mini GT", "release_year": 2023})
    _ai_piece(toy_number="MGT00555", item_master_id="m1", confirmed_by=[])
    res = _vis.VisionResult(toy_number="MGT00555", needs_manual=False, confidence=0.9)
    monkeypatch.setattr(idn, "identify_image", lambda img, brand_hint=None: res)
    seen = []
    real = aic.confirm
    monkeypatch.setattr(aic, "confirm", lambda *a: (seen.append(a), real(*a))[1])
    out = asyncio.run(idn._identify_guarded(None, PHOTO, None, CurrentUser({"id": "u2"})))
    assert out.method == "o" and out.item is not None
    assert seen == [(_db, "ai1", "u2")]
    assert _v()["verified"] is True
    blob = out.model_dump_json()
    for k in ("origin", "verified", "confirmed_by", "created_by"):
        assert k not in blob


@pytest.fixture
def gar(monkeypatch):
    monkeypatch.setattr(_garage, "get_db", lambda: _db)
    app = _FastAPI()
    app.include_router(_garage.router, prefix="/api/v1")
    who = {"id": "u2"}
    app.dependency_overrides[_gcu] = lambda: CurrentUser({"id": who["id"], "username": who["id"]})
    for c in ("collections", "user_items", "user_lists", "user_list_items", "items"):
        _db[c].delete_many({})
    coll = _db.collections.insert_one({"user_id": "u2", "name": "c"}).inserted_id
    return _TC(app), str(coll)


def test_garage_add_confirms_and_does_not_leak(gar):
    client, coll = gar
    _ai_piece(item_master_id="m1")
    r = client.post("/api/v1/garage/items", json={"collection_id": coll, "item_variant_id": "ai1"})
    assert r.status_code == 201
    assert _v()["verified"] is True
    iv = r.json()["item_variant"]
    for k in ("origin", "verified", "confirmed_by", "created_by"):
        assert k not in iv


def test_garage_add_non_ai_no_write_and_confirm_failure_ok(gar, monkeypatch):
    client, coll = gar
    _db.variants.delete_many({})
    _db.variants.insert_one({"_id": "s1", "variant_name": "x", "ai_local_hash": "hs1", "item_master_id": "m1"})
    r = client.post("/api/v1/garage/items", json={"collection_id": coll, "item_variant_id": "s1"})
    assert r.status_code == 201 and "confirmed_by" not in _v("s1")
    _ai_piece(item_master_id="m1")
    monkeypatch.setattr(aic, "confirm", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    r = client.post("/api/v1/garage/items", json={"collection_id": coll, "item_variant_id": "ai1"})
    assert r.status_code == 201


# ── Task 7: herramienta de revisión y piezas ocultas ───────────────────────

import importlib.util as _ilu
from datetime import datetime as _dt7, timedelta as _td7

_spec = _ilu.spec_from_file_location(
    "review_ai_pieces", Path(__file__).resolve().parents[1] / "scripts" / "review_ai_pieces.py")
rev = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(rev)

_NH = {"hidden": {"$ne": True}}


@pytest.fixture
def hid(monkeypatch):
    for coll in ("variants", "items"):
        _db[coll].delete_many({})
    monkeypatch.setattr(_vis, "_hidden", frozenset())
    monkeypatch.setattr(idn, "_NAME_IDX", None)
    yield
    monkeypatch.setattr(_vis, "_hidden", frozenset())


def _p7(_id, minutes=0, **kw):
    d = {"_id": _id, "origin": "ai", "verified": False, "created_by": "u1", "confirmed_by": ["a", "b"][: kw.pop("nconf", 0)],
         "variant_name": f"Pieza {_id}", "ai_local_hash": "h" + _id, "source_confidence": 0.8,
         "created_at": _dt7(2026, 10, 1, tzinfo=timezone.utc) + _td7(minutes=minutes)}
    d.update(kw)
    _db.variants.insert_one(d)


def test_list_pending_filters_and_orders(hid):
    _p7("old", 0, nconf=1)
    _p7("new", 10)
    _p7("ver", 20, verified=True)
    _p7("hid", 30, hidden=True)
    _db.variants.insert_one({"_id": "scrape", "variant_name": "x"})
    rows = rev.list_pending(_db)
    assert [r["id"] for r in rows] == ["new", "old"]
    assert rows[1]["confirmed_by"] == 1 and rows[1]["created_by"] == "u1"
    assert {"name", "brand", "created_at", "source_confidence"} <= set(rows[0])
    assert len(rev.list_pending(_db, limit=1)) == 1


def test_demote_dry_run_does_not_write(hid, capsys):
    _p7("a1")
    res = rev.demote(_db, "a1")
    assert res["applied"] is False and "hidden" not in _db.variants.find_one({"_id": "a1"})
    assert rev.main(["--demote", "a1"], db=_db) == 0
    assert "DRY-RUN" in capsys.readouterr().out
    assert "hidden" not in _db.variants.find_one({"_id": "a1"})


def test_demote_apply_hides_piece(hid, capsys):
    _p7("a1")
    assert rev.main(["--demote", "a1", "--apply"], db=_db) == 0
    out = capsys.readouterr().out
    assert "respaldo" in out.lower()
    assert _db.variants.find_one({"_id": "a1"})["hidden"] is True
    assert _db.variants.count_documents({"_id": "a1"}) == 1  # nada se borra
    assert rev.list_pending(_db) == []


def test_demote_refuses_non_ai_piece(hid, capsys):
    _db.variants.insert_one({"_id": "s1", "variant_name": "x"})
    _db.variants.insert_one({"_id": "s2", "variant_name": "y", "origin": "scrape"})
    for pid in ("s1", "s2", "nope"):
        assert rev.main(["--demote", pid, "--apply"], db=_db) == 1
    assert "hidden" not in _db.variants.find_one({"_id": "s1"})
    assert "hidden" not in _db.variants.find_one({"_id": "s2"})
    assert rev.main(["--apply"], db=_db) == 2


def test_cli_prints_table(hid, capsys):
    _p7("a1")
    assert rev.main([], db=_db) == 0
    assert "a1" in capsys.readouterr().out


def test_catalog_hides_hidden_but_lookup_by_id_works(hid):
    _db.items.insert_one({"_id": "m7", "name": "Skyline", "manufacturer": "Mini GT", "release_year": 2023})
    for i, hd in (("v1", False), ("v2", True)):
        _p7(i, item_master_id="m7", toy_number="MGT00777", hidden=hd or None, variant_name="Skyline " + i)
    _catr._MASTER_CACHE["at"] = 0.0
    resp = _NS(headers={})
    lst = _catr.list_variants(resp, limit=100, offset=0, brand=None, rarity=None, search=None,
                              manufacturer=None, sub_brand=None, scale=None, color=None, decade=None,
                              sort="name", sort_dir="asc")
    assert [v.id for v in lst] == ["v1"] and resp.headers["X-Total-Count"] == "1"
    srch = _catr.list_variants(_NS(headers={}), limit=100, offset=0, brand=None, rarity=None, search="Skyline",
                               manufacturer=None, sub_brand=None, scale=None, color=None, decade=None,
                               sort="name", sort_dir="asc")
    assert [v.id for v in srch] == ["v1"]
    assert [v.id for v in _catr.get_variants_by_toy("MGT00777")] == ["v1"]
    assert _catr.count_variants() == {"total": 1}
    assert _catr.get_variant(_catr_id := "v2").id == "v2"  # por id sigue disponible


def test_identify_resolution_ignores_hidden(hid):
    _p7("h1", hidden=True, toy_number="MGT00555", sku_upc="0123456789012", variant_name="Hidden Skyline GT-R R34 Blue")
    assert idn._find_variant_by_id(_db, "h1") == (None, None)
    assert idn._find_variant_by_toy(_db, "MGT00555") == (None, None)
    assert idn._find_variant_by_upc(_db, "0123456789012") == (None, None)
    assert idn._name_index(_db) == []
    sug = idn._to_suggestions([_vis.VisionMatch(id="h1", toy="MGT00555", name="n", score=0.9, source="embed")], _db)
    assert sug[0].image is None and sug[0].body_color is None
    _p7("v1", toy_number="MGT00556", variant_name="Visible Skyline GT-R R34 Blue")
    assert idn._find_variant_by_id(_db, "v1")[0]["_id"] == "v1"
    assert idn._find_variant_by_toy(_db, "MGT00556")[0]["_id"] == "v1"


def test_best_name_match_skips_hidden_ids_even_if_cached(hid, monkeypatch):
    _p7("v1", toy_number="MGT00556", variant_name="Visible Skyline GT-R R34 Blue")
    text = "MINI GT Visible Skyline GT-R R34 Blue"
    assert idn._best_name_match(_db, text, True)[0] == "v1"
    monkeypatch.setattr(_vis, "_hidden", frozenset({"v1"}))  # índice ya en caché
    assert idn._best_name_match(_db, text, True) is None


def test_vision_matches_skip_hidden_ids(vidx, monkeypatch):
    v = _unit(1000)
    _vis.live_index_append(_entry(1), v.tolist())
    assert _vis.match_embedding(v)[0].id == "live1"
    _vis.set_hidden_ids({"live1"})
    try:
        assert all(m.id != "live1" for m in _vis.match_embedding(v))
        assert all(m.id != "live1" for m in _vis.search_catalog_by_text("Live 1 Big"))
    finally:
        _vis.set_hidden_ids(set())


def test_sync_hidden_feeds_vision(hid, monkeypatch):
    _p7("h1", hidden=True)
    monkeypatch.setitem(idn._HIDDEN_SYNC, "at", 0.0)
    idn._sync_hidden(_db)
    try:
        assert "h1" in _vis.hidden_ids()
    finally:
        _vis.set_hidden_ids(set())
        idn._HIDDEN_SYNC["at"] = 0.0


def test_check_duplicate_hidden_is_uncertain(aienv):
    r = _read(toy_number="MGT00100")
    _seed_variant()
    _db.variants.update_one({"_id": "exist1"}, {"$set": {"hidden": True}})
    # mismo id determinista oculto
    sid = aic.piece_id(aic.make_source_id(r))
    _db.variants.insert_one({"_id": sid, "variant_name": "ai", "hidden": True, "origin": "ai"})
    chk = aic.check_duplicate(_db, r, None)
    assert chk.outcome == "uncertain" and chk.piece is None
    _db.variants.delete_one({"_id": sid})
    chk = aic.check_duplicate(_db, r, None)  # por toy#
    assert chk.outcome == "uncertain" and chk.piece is None
    assert aic.find_duplicate(_db, r, None) is None


def test_check_duplicate_hidden_via_text_and_embedding(aienv):
    _seed_variant(toy="MGT09999")
    _db.variants.update_one({"_id": "exist1"}, {"$set": {"hidden": True}})
    aienv["text"] = [_vm("exist1", "Nissan Skyline GT-R R34 Bayside Blue", 0.9)]
    chk = aic.check_duplicate(_db, _read(), None)
    assert chk.outcome == "uncertain"
    aienv["text"] = []
    aienv["match"] = [_vis.VisionMatch(id="exist1", toy=None, name="x", score=0.95, source="embed")]
    chk = aic.check_duplicate(_db, _read(livery="otra", casting="Otro Coche"), [0.1] * 768)
    assert chk.outcome == "uncertain"


def test_check_duplicate_visible_beats_hidden_twin(aienv):
    _seed_variant(_id="vis1", toy="MGT09998")
    _db.variants.insert_one({"_id": "hid1", "item_master_id": "mexist", "variant_name": "Bayside Blue",
                             "toy_number": "MGT09997", "hidden": True})
    chk = aic.check_duplicate(_db, _read(), None)
    assert chk.outcome == "duplicate" and chk.piece["id"] == "vis1"


# ── Task 7 fix round 1 ─────────────────────────────────────────────────────


def test_inject_race_num_skips_hidden_and_not_suggested(vidx):
    base = {"id": "b0", "toy": "MGT00001", "name": "Dino Rexy #77", "view": 0}
    _vis.live_index_append(base, _unit(2000).tolist())
    _vis.live_index_append({"id": "hid80", "toy": "AI80", "name": "Dino Rexy #80", "view": 0},
                           _unit(2001).tolist())
    pool = [_vis.VisionMatch(id="b0", toy="MGT00001", name="Dino Rexy #77", score=0.9, source="embed")]
    out = _vis.inject_race_num_candidates(list(pool), "texto #80", top_n=6)
    assert sorted(m.id for m in out) == ["b0", "hid80"]  # control: sin ocultar se inyecta
    _vis.set_hidden_ids({"hid80"})
    try:
        out = _vis.inject_race_num_candidates(list(pool), "texto #80", top_n=6)
        assert [m.id for m in out] == ["b0"]
        sug = idn._to_suggestions(out, _db)
        assert all(s.id != "hid80" and "#80" not in (s.name or "") for s in sug)
    finally:
        _vis.set_hidden_ids(set())


def test_sync_hidden_failure_keeps_set_and_does_not_retry(hid, monkeypatch):
    _vis.set_hidden_ids({"keep1"})
    monkeypatch.setitem(idn._HIDDEN_SYNC, "at", 0.0)
    calls = []

    class Boom:
        class variants:
            @staticmethod
            def find(*a, **k):
                calls.append(1)
                raise RuntimeError("mongo caido")

    try:
        idn._sync_hidden(Boom)
        idn._sync_hidden(Boom)  # dentro de la ventana: no consulta otra vez
        assert len(calls) == 1
        assert "keep1" in _vis.hidden_ids()
    finally:
        _vis.set_hidden_ids(set())
        idn._HIDDEN_SYNC["at"] = 0.0


# ── Task 9: respaldo del identify sobre la misma lectura ───────────────────
from app.services.gemini_vision import gemini_to_ocr_text as _g2t

_FULL = {"es_diecast": True, "brand": "Mini GT", "casting": "Nissan Skyline", "modelo": "GT-R R34",
         "livery": "azul con franja", "dorsal": "80", "color": "azul", "texto": "MINI GT 14+",
         "serie": "Japan", "toy_number": "872", "confidence": "alto"}


def test_one_paid_call_per_photo_fallback_then_creation(cvenv):
    calls, st = cvenv
    st["resp"] = _ok(_FULL)
    assert cv.read_packaging_with_claude(b"same") is not None
    assert cv.read_for_catalog(b"same")["es_diecast"] is True
    assert len(calls) == 1


def test_one_paid_call_per_photo_creation_then_fallback(cvenv):
    calls, st = cvenv
    st["resp"] = _ok(_FULL)
    assert cv.read_for_catalog(b"same2") is not None
    assert cv.read_packaging_with_claude(b"same2") is not None
    assert len(calls) == 1


def test_adapter_keys_and_mapping_feed_gemini_to_ocr_text(cvenv):
    _, st = cvenv
    st["resp"] = _ok(_FULL)
    r = cv.read_packaging_with_claude(b"m1")
    assert set(r) == {"brand", "model", "variant", "toy_number", "series", "color", "year",
                      "driver", "full_text", "confidence"}
    assert r["model"] == "GT-R R34" and r["variant"] == "azul con franja 80"
    assert r["series"] == "Japan" and r["full_text"] == "MINI GT 14+" and r["driver"] is None
    assert r["confidence"] == "alto" and r["toy_number"] == "872"
    t = _g2t(r)
    assert "Mini GT" in t and "GT-R R34" in t and "MGT00872" in t


def test_adapter_fallbacks_casting_and_color(cvenv):
    _, st = cvenv
    st["resp"] = _ok({"es_diecast": True, "brand": "Hot Wheels", "casting": "Civic", "color": "rojo"})
    r = cv.read_packaging_with_claude(b"m2")
    assert r["model"] == "Civic" and r["variant"] == "rojo"


def test_adapter_none_when_not_diecast(cvenv):
    _, st = cvenv
    st["resp"] = _ok({"es_diecast": False, "texto": "documento"})
    assert cv.read_packaging_with_claude(b"m3") is None


def test_timeout_not_cached_and_does_not_trip_breaker(cvenv, monkeypatch):
    calls, st = cvenv

    def boom(*a, **k):
        calls.append(k)
        raise cv.requests.Timeout("t")

    monkeypatch.setattr(cv.requests, "post", boom)
    assert cv.read_packaging_with_claude(b"to") is None
    assert cv.breaker_open() is False
    st["resp"] = _ok(_FULL)
    monkeypatch.setattr(cv.requests, "post", lambda *a, **k: (calls.append(k), st["resp"])[1])
    assert cv.read_for_catalog(b"to") is not None
    assert len(calls) == 2


def test_timeouts_adapter_short_catalog_long(cvenv, monkeypatch):
    calls, st = cvenv
    st["resp"] = _ok(_FULL)
    cv.read_packaging_with_claude(b"t1")
    cv.read_for_catalog(b"t2")
    assert calls[0]["timeout"] == cv._TIMEOUT_S
    assert calls[1]["timeout"] == cv._CATALOG_TIMEOUT_S == 12.0
    cv.read_for_catalog(b"t3", timeout_s=2.0)
    assert calls[2]["timeout"] == 2.0


def test_int_dorsal_and_toy_coerced_bool_not(cvenv):
    _, st = cvenv
    st["resp"] = _ok({"es_diecast": True, "dorsal": 80, "toy_number": 872})
    r = cv.read_for_catalog(b"c1")
    assert r["dorsal"] == "80" and r["toy_number"] == "872"
    st["resp"] = _ok({"es_diecast": True, "dorsal": True, "toy_number": False})
    r = cv.read_for_catalog(b"c2")
    assert r["dorsal"] is None and r["toy_number"] is None


def test_no_key_logged_once(cvenv, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    cv.read_for_catalog(b"k1")
    cv.read_for_catalog(b"k2")
    assert capsys.readouterr().out.count("sin ANTHROPIC_API_KEY") == 1


def test_unparseable_logged_once_without_body(cvenv, capsys):
    _, st = cvenv
    st["resp"] = _Resp(200, {"content": [{"type": "text", "text": "SECRETBODY sin json"}]})
    cv.read_for_catalog(b"u1")
    cv.read_for_catalog(b"u2")
    out = capsys.readouterr().out
    assert out.count("respuesta sin JSON") == 1 and "SECRETBODY" not in out


@pytest.mark.parametrize("on", [True, False])
@pytest.mark.parametrize("ident", [True, False])
@pytest.mark.parametrize("toy", [None, "MGT00001"])
@pytest.mark.parametrize("useful", [True, False])
@pytest.mark.parametrize("vok", [True, False])
@pytest.mark.parametrize("fb", [True, False])
def test_fallback_should_run_all_combinations(on, ident, toy, useful, vok, fb):
    old = fb and not ident and toy is None and not useful and not vok
    wide = fb and not ident and toy is None and not vok
    got = _vis._fallback_should_run(fb, ident, toy, useful, vok, widen=on)
    assert got == (wide if on else old)


def test_fallback_flag_default_off(monkeypatch):
    monkeypatch.delenv("VISION_FALLBACK_ON_MANUAL", raising=False)
    assert _vis._fallback_widened() is False
    monkeypatch.setenv("VISION_FALLBACK_ON_MANUAL", "1")
    assert _vis._fallback_widened() is True
    monkeypatch.setenv("VISION_FALLBACK_ON_MANUAL", "0")
    assert _vis._fallback_widened() is False


# ── Revisión final C2: barrera del recibo ANTES de la IA + resolve_hint ─────


def test_gate_split_receipt_only_and_read_only():
    mx = get_settings().ai_catalog_max_top_score
    assert aic.gate_receipt({"top_score": 0.2, "twins": True}) == "visual_candidate"
    assert aic.gate_receipt({"top_score": mx}) == "visual_candidate"
    assert aic.gate_receipt({"top_score": mx - 0.01}) is None
    assert aic.gate_receipt({"top_score": "x"}) is None and aic.gate_receipt(None) is None
    assert aic.gate_read(None) == "no_diecast"
    assert aic.gate_read(_read(confidence="medio")) == "low_confidence"
    assert aic.gate_read(_read(casting=" ")) == "missing_identity"
    assert aic.gate_read(_read()) is None


def test_twins_or_high_receipt_makes_zero_ai_calls(rn, caplog):
    mx = get_settings().ai_catalog_max_top_score
    _receipt(twins=True)
    with caplog.at_level(_logging.INFO, logger=idn._log.name):
        r = _post()
    assert r.status_code == 200 and _is_manual(r.json())
    assert rn["calls"] == [] and _rows() == 0
    assert "ai_catalog rejected:visual_candidate user=u1" in caplog.text
    _db.identify_receipts.delete_many({})
    _receipt(top=mx)
    assert _is_manual(_post().json()) and rn["calls"] == []


@pytest.mark.parametrize("top,twins,hint", [(0.4, False, True), (0.4, True, False), (0.99, False, False)])
def test_identify_resolve_hint_follows_receipt_gate(monkeypatch, top, twins, hint):
    monkeypatch.setattr(idn, "_run_identify", _fake(manual=True, top=top, twins=twins))
    out = _go()
    assert out.debug["resolve_hint"] is hint
    assert _db.identify_receipts.count_documents({}) == 1


def test_identify_resolve_hint_false_when_off_matched_or_unwritten(monkeypatch):
    monkeypatch.setattr(get_settings(), "ai_catalog_enabled", False)
    monkeypatch.setattr(idn, "_run_identify", _fake(manual=True, top=0.3))
    assert _go().debug["resolve_hint"] is False
    monkeypatch.setattr(get_settings(), "ai_catalog_enabled", True)
    monkeypatch.setattr(idn, "_run_identify", _fake(manual=False, item={"id": "x"}))
    assert _go().debug["resolve_hint"] is False
    monkeypatch.setattr(idn, "_run_identify", _fake(manual=True, top=0.3))
    monkeypatch.setattr(idn, "write_receipt", lambda *a, **k: False)
    assert _go().debug["resolve_hint"] is False


def test_identify_resolve_hint_on_base64_route(monkeypatch):
    who = {"id": "u1"}
    _api.dependency_overrides[_gcu] = lambda: CurrentUser({"id": who["id"], "username": who["id"]})
    try:
        monkeypatch.setattr(idn, "_run_identify", _fake(manual=True, top=0.3))
        r = _client.post("/api/v1/identify/base64", json={"image_base64": _b64.b64encode(IMG).decode()})
        assert r.status_code == 200 and r.json()["debug"]["resolve_hint"] is True
    finally:
        _api.dependency_overrides.clear()
