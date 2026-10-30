"""Flujos núcleo de punta a punta (como los usa la app): registro → garaje →
wishlist → estadísticas → catálogo. Mongo simulado: no toca producción.
"""
import sys
from pathlib import Path

import mongomock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app.db as dbmod

_db = mongomock.MongoClient()["core"]
dbmod.get_db = lambda: _db

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import auth, catalog, garage, lists, stats

api = FastAPI()
for r in (auth.router, auth.me_router, catalog.router, garage.router, lists.router, stats.router):
    api.include_router(r, prefix="/api/v1")
c = TestClient(api)
P = "/api/v1"


@pytest.fixture(scope="module")
def s():
    _db.items.insert_one({"_id": "m1", "name": "Nissan Skyline GT-R R34", "manufacturer": "Mini GT", "release_year": 2023})
    for i in range(3):
        _db.variants.insert_one({"_id": f"v{i}", "item_master_id": "m1", "variant_name": f"Bayside Blue {i}",
                                 "toy_number": f"MGT0000{i}", "ai_local_hash": f"h{i}", "main_image_url": "http://x"})
    r = c.post(f"{P}/auth/register", json={"email": "z@x.com", "username": "zeta", "password": "Secreta123!"})
    assert r.status_code == 201, r.text
    h = {"Authorization": f"Bearer {r.json()['access_token']}"}
    r2 = c.post(f"{P}/auth/register", json={"email": "y@x.com", "username": "yeti", "password": "Secreta123!"})
    return {"h": h, "h2": {"Authorization": f"Bearer {r2.json()['access_token']}"}}


def test_register_creates_default_garage(s):
    cols = c.get(f"{P}/collections", headers=s["h"]).json()
    assert len(cols) == 1


def test_garage_crud_and_isolation(s):
    col = c.get(f"{P}/collections", headers=s["h"]).json()[0]["id"]
    r = c.post(f"{P}/garage/items", json={"collection_id": col, "item_variant_id": "v0", "packaging_type": "Box",
                                          "purchase_price": 20}, headers=s["h"])
    assert r.status_code == 201, r.text
    item = r.json()
    assert item["item_variant"]["variant_name"] == "Bayside Blue 0"  # la ficha trae la variante
    # variante inexistente → error claro (antes la app lo guardaba solo en local)
    assert c.post(f"{P}/garage/items", json={"collection_id": col, "item_variant_id": "nope"},
                  headers=s["h"]).status_code == 404
    # empaque fuera del enum → 422 (la app normaliza al crear Y al editar)
    assert c.patch(f"{P}/garage/items/{item['id']}", json={"packaging_type": "Caja"}, headers=s["h"]).status_code == 422
    assert c.patch(f"{P}/garage/items/{item['id']}", json={"quantity": 3}, headers=s["h"]).json()["quantity"] == 3
    # otro usuario no ve, no edita, no borra
    assert c.get(f"{P}/garage/items/{item['id']}", headers=s["h2"]).status_code == 404
    assert c.patch(f"{P}/garage/items/{item['id']}", json={"quantity": 9}, headers=s["h2"]).status_code == 404
    assert c.delete(f"{P}/garage/items/{item['id']}", headers=s["h2"]).status_code == 404
    assert c.post(f"{P}/garage/items", json={"collection_id": col, "item_variant_id": "v1"},
                  headers=s["h2"]).status_code in (403, 404)
    assert len(c.get(f"{P}/garage/items", headers=s["h"]).json()) == 1


def test_wishlist_roundtrip_and_auto_remove(s):
    wl = c.get(f"{P}/lists/wishlist", headers=s["h"]).json()
    assert c.get(f"{P}/lists/wishlist", headers=s["h"]).json()["id"] == wl["id"]  # idempotente
    r = c.post(f"{P}/lists/{wl['id']}/items", json={"item_variant_id": "v2", "priority": 1}, headers=s["h"])
    assert r.status_code == 201, r.text
    items = c.get(f"{P}/lists/{wl['id']}/items", headers=s["h"]).json()
    assert [i["item_variant_id"] for i in items] == ["v2"]
    # al meterla al garaje sale sola de la wishlist
    col = c.get(f"{P}/collections", headers=s["h"]).json()[0]["id"]
    c.post(f"{P}/garage/items", json={"collection_id": col, "item_variant_id": "v2"}, headers=s["h"])
    assert c.get(f"{P}/lists/{wl['id']}/items", headers=s["h"]).json() == []
    # wishlist ajena
    assert c.get(f"{P}/lists/{wl['id']}/items", headers=s["h2"]).status_code in (403, 404)


def test_stats_reflect_garage(s):
    st = c.get(f"{P}/stats/collection", headers=s["h"]).json()
    assert st, st


def test_catalog_search_and_by_toy(s):
    r = c.get(f"{P}/catalog/variants?search=Bayside&limit=10")
    assert r.status_code == 200 and len(r.json()) >= 1
    r = c.get(f"{P}/catalog/variants/by-toy/MGT00001")
    assert r.status_code == 200


def test_manual_piece_roundtrip_keeps_catalog_readable(s):
    """Alta manual (pieza fuera del catálogo) como la manda la app → garaje → listados sin 500."""
    body = {"id": "manual-123", "variant_name": "Edición Estándar", "toy_number": None, "rarity": "regular",
            "main_image_url": None, "metadata": {"user_submitted": True}, "name": "Datsun 510 Pro Street",
            "manufacturer": "Mini GT", "release_year": 2026, "category_id": "0"}
    r = c.post(f"{P}/catalog/variants", json=body, headers=s["h"])
    assert r.status_code == 201, r.text
    vid = r.json()["id"]
    col = c.get(f"{P}/collections", headers=s["h"]).json()[0]["id"]
    assert c.post(f"{P}/garage/items", json={"collection_id": col, "item_variant_id": vid},
                  headers=s["h"]).status_code == 201
    assert c.get(f"{P}/garage/items", headers=s["h"]).status_code == 200
    assert c.get(f"{P}/catalog/variants?limit=50").status_code == 200
    assert c.get(f"{P}/catalog/variants?search=Datsun").status_code == 200
    assert _db.variants.find_one({"_id": vid})["created_by"]


def test_stats_numbers(s):
    st = c.get(f"{P}/stats/collection", headers=s["h"]).json()
    assert st["total_items"] >= 2 and st.get("total_collections") == 1
    # la taxonomía sigue resolviéndose (lote de variantes/maestros)
    assert "Mini GT" in str(st)


def test_garage_batch_equals_single(s):
    from app.routers.garage import _user_item_out, _user_items_out

    docs = list(_db.user_items.find({}))
    assert len(docs) >= 2
    one = [_user_item_out(_db, d).model_dump() for d in docs]
    batch = [x.model_dump() for x in _user_items_out(_db, docs)]
    assert one == batch
    assert all(x["item_variant"] for x in batch)


def test_catalog_search_by_model_name_tokens(s):
    """El nombre del modelo vive en el maestro: antes «Skyline R34» no encontraba nada."""
    _db.items.insert_one({"_id": "m2", "name": "Porsche 911 GT3", "manufacturer": "Mini GT", "release_year": 2022})
    _db.variants.insert_one({"_id": "p1", "item_master_id": "m2", "variant_name": "Shark Blue",
                             "toy_number": "MGT00601", "ai_local_hash": "hp1", "metadata": {"scale": "1/64"}})
    ids = lambda q: {v["id"] for v in c.get(f"{P}/catalog/variants", params={"search": q, "limit": 50}).json()}
    assert {"v0", "v1", "v2"} <= ids("Skyline R34")          # palabras sueltas, en el modelo
    assert ids("skyline bayside 1") == {"v1"}                 # modelo + livery + número
    assert ids("Porsche shark") == {"p1"}
    assert ids("Skyline Porsche") == set()                    # TODAS las palabras deben casar
    assert c.get(f"{P}/catalog/variants", params={"search": "(a+)+$"}).status_code == 200  # sin ReDoS/500
    both = {v["id"] for v in c.get(f"{P}/catalog/variants", params={"search": "Porsche", "scale": "1:64"}).json()}
    assert both == {"p1"}                                     # búsqueda Y escala (intersección)


def test_challenges_from_real_data_and_single_claim(s):
    from app.routers import challenges

    api.include_router(challenges.router, prefix="/api/v1")
    ch = {x["id"]: x for x in c.get(f"{P}/challenges", headers=s["h"]).json()}
    assert ch and ch["garage-10"]["target"] == 10 and ch["garage-10"]["progress"] >= 2
    assert c.post(f"{P}/challenges/garage-10/claim", headers=s["h"]).status_code == 409  # incompleto
    # completar: 10 piezas
    col = c.get(f"{P}/collections", headers=s["h"]).json()[0]["id"]
    item = c.get(f"{P}/garage/items", headers=s["h"]).json()[0]
    c.patch(f"{P}/garage/items/{item['id']}", json={"quantity": 20}, headers=s["h"])
    xp0 = _db.users.find_one({"username": "zeta"}).get("total_xp", 0)
    r = c.post(f"{P}/challenges/garage-10/claim", headers=s["h"])
    assert r.status_code == 200 and r.json()["claimed"] is True
    assert _db.users.find_one({"username": "zeta"})["total_xp"] == xp0 + 100
    assert c.post(f"{P}/challenges/garage-10/claim", headers=s["h"]).status_code == 409  # una sola vez
    # otro usuario: su propio progreso, no el ajeno
    other = {x["id"]: x for x in c.get(f"{P}/challenges", headers=s["h2"]).json()}
    assert other["garage-10"]["claimed"] is False and other["garage-10"]["completed"] is False


def test_catalog_list_includes_master_and_sorts_by_model(s):
    rows = c.get(f"{P}/catalog/variants", params={"limit": 50}).json()
    by_id = {r["id"]: r for r in rows}
    assert by_id["v0"]["item_master"]["name"] == "Nissan Skyline GT-R R34"
    assert by_id["v0"]["item_master"]["manufacturer"] == "Mini GT"
    names = [(r.get("item_master") or {}).get("name") or "" for r in rows]
    assert names == sorted(names)  # orden real por modelo


def test_catalog_list_paging_sort_and_filters(s):
    import app.routers.catalog as cat

    cat._MASTER_CACHE["at"] = 0  # el test añade maestros: refrescar caché
    rows = lambda **p: c.get(f"{P}/catalog/variants", params={"limit": 50, **p})
    r = rows(limit=2)
    total = int(r.headers["x-total-count"])
    assert total >= 5 and len(r.json()) == 2
    allrows = rows(limit=50).json()
    assert [x["id"] for x in rows(limit=2, offset=2).json()] == [x["id"] for x in allrows[2:4]]  # paginación estable
    years = [(x.get("item_master") or {}).get("release_year") for x in rows(sort="year", sort_dir="desc").json()]
    with_y = [y for y in years if y]
    assert with_y == sorted(with_y, reverse=True) and years[: len(with_y)] == with_y  # sin año al final
    assert {x["id"] for x in rows(decade="2020").json()} >= {"v0", "p1"}
    assert rows(decade="1990").json() == [] and rows(decade="1990").headers["x-total-count"] == "0"
    assert {x["id"] for x in rows(manufacturer="Mini GT", search="porsche").json()} == {"p1"}
