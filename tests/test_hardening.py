"""Seguridad y robustez: auth, catálogo, subidas, ranking, notificaciones.

Mongo simulado (mongomock): no toca producción.
Requiere: pip install mongomock pytest httpx pillow pyotp python-multipart
"""
import base64
import io
import sys
from pathlib import Path

import mongomock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app.db as dbmod

_db = mongomock.MongoClient()["h"]
dbmod.get_db = lambda: _db

import os

os.environ.setdefault("UPLOAD_DIR", str(Path(__file__).resolve().parent / "_uploads_tmp"))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.routers import auth, catalog, garage, leaderboard, notifications, uploads

api = FastAPI()
for r in (auth.router, auth.me_router, catalog.router, garage.router, leaderboard.router,
          notifications.router, uploads.router):
    api.include_router(r, prefix="/api/v1")
c = TestClient(api)
P = "/api/v1"


def reg(name, pwd="Secreta123!"):
    r = c.post(f"{P}/auth/register", json={"email": f"{name}@x.com", "username": name, "password": pwd})
    assert r.status_code == 201, r.text
    d = r.json()
    return {"h": {"Authorization": f"Bearer {d['access_token']}"}, "id": d["user"]["id"],
            "refresh": d["refresh_token"], "name": name}


@pytest.fixture(scope="module")
def u():
    return reg("ana"), reg("beto")


def test_login_by_username_case_and_regex_injection(u):
    assert c.post(f"{P}/auth/login", json={"email": "ANA", "password": "Secreta123!"}).status_code == 200
    assert c.post(f"{P}/auth/login", json={"email": ".*", "password": "Secreta123!"}).status_code == 401
    assert c.post(f"{P}/auth/login", json={"email": "an.", "password": "Secreta123!"}).status_code == 401


def test_login_bruteforce_is_limited(u):
    for _ in range(8):
        assert c.post(f"{P}/auth/login", json={"email": "beto", "password": "mala"}).status_code == 401
    r = c.post(f"{P}/auth/login", json={"email": "beto", "password": "Secreta123!"})
    assert r.status_code == 429  # bloqueado aunque la contraseña sea buena
    _db.auth_failures.delete_many({})
    assert c.post(f"{P}/auth/login", json={"email": "beto", "password": "Secreta123!"}).status_code == 200


def test_forgot_and_magic_never_leak_code(u, monkeypatch):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    r = c.post(f"{P}/auth/forgot-password", json={"email": "ana@x.com"}).json()
    assert r["reset_code"] is None and r["delivery"] == "unavailable"
    r = c.post(f"{P}/auth/magic-code", json={"email": "ana@x.com"}).json()
    assert r["code"] is None and r["delivery"] == "unavailable"
    # con correo: se envía, no se devuelve
    sent = []
    monkeypatch.setenv("SMTP_HOST", "smtp.test")
    monkeypatch.setattr(auth, "send_mail", lambda to, subj, text: sent.append((to, text)) or True)
    r = c.post(f"{P}/auth/forgot-password", json={"email": "ana@x.com"}).json()
    assert r["reset_code"] is None and r["delivery"] == "email" and sent and sent[0][0] == "ana@x.com"
    # cuenta inexistente: misma respuesta, sin correo
    sent.clear()
    r = c.post(f"{P}/auth/forgot-password", json={"email": "nadie@x.com"}).json()
    assert r["delivery"] == "email" and not sent


def test_password_change_closes_other_sessions(u):
    a, _ = u
    s2 = c.post(f"{P}/auth/login", json={"email": "ana", "password": "Secreta123!"}).json()
    r = c.post(f"{P}/auth/change-password", json={"current_password": "Secreta123!", "new_password": "Nueva12345!"},
               headers=a["h"])
    assert r.status_code == 200 and r.json()["refresh_token"]
    assert c.post(f"{P}/auth/refresh", json={"refresh_token": s2["refresh_token"]}).status_code == 401
    assert c.post(f"{P}/auth/refresh", json={"refresh_token": r.json()["refresh_token"]}).status_code == 200
    c.post(f"{P}/auth/change-password", json={"current_password": "Nueva12345!", "new_password": "Secreta123!"},
           headers={"Authorization": f"Bearer {r.json()['access_token']}"})


def test_magic_login_respects_2fa(u, monkeypatch):
    import pyotp
    from app.security import hash_password
    from datetime import datetime, timedelta, timezone

    secret = pyotp.random_base32()
    _db.users.update_one({"username": "beto"}, {"$set": {
        "totp_enabled": True, "totp_secret": secret,
        "magic_code": {"code_hash": hash_password("123456"), "attempts": 0,
                       "expires_at": datetime.now(timezone.utc) + timedelta(minutes=5)}}})
    assert c.post(f"{P}/auth/magic-login", json={"email": "beto@x.com", "code": "123456"}).status_code == 401
    r = c.post(f"{P}/auth/magic-login", json={"email": "beto@x.com", "code": "123456",
                                             "totp_code": pyotp.TOTP(secret).now()})
    assert r.status_code == 200
    _db.users.update_one({"username": "beto"}, {"$set": {"totp_enabled": False}})


def test_public_default_admin_is_disabled():
    from app.security import hash_password
    from app.services.seed import seed_admin

    s = get_settings()
    _db.users.insert_one({"email": s.seed_admin_email, "username": "admin",
                          "pwd_hash": hash_password("AdminDev123!"), "is_active": True})
    seed_admin(_db)
    assert _db.users.find_one({"email": s.seed_admin_email})["is_active"] is False
    r = c.post(f"{P}/auth/login", json={"email": s.seed_admin_email, "password": "AdminDev123!"})
    assert r.status_code == 403


def test_catalog_edit_admin_only(u):
    a, b = u
    _db.variants.insert_one({"_id": "v9", "variant_name": "Original", "ai_local_hash": "h9", "item_master_id": "m9"})
    assert c.patch(f"{P}/catalog/variants/v9", json={"variant_name": "Hack"}, headers=a["h"]).status_code == 403
    assert _db.variants.find_one({"_id": "v9"})["variant_name"] == "Original"
    get_settings().admin_user_ids = a["id"]
    try:
        assert c.patch(f"{P}/catalog/variants/v9", json={"variant_name": "Fix"}, headers=a["h"]).status_code == 200
    finally:
        get_settings().admin_user_ids = ""


def test_open_push_endpoints_removed(u):
    a, b = u
    r = c.post(f"{P}/notifications/send", json={"to_user_id": b["id"], "title": "x", "body": "y"}, headers=a["h"])
    assert r.status_code in (404, 405)


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (200, 10, 10)).save(buf, "PNG")
    return buf.getvalue()


def test_uploads_validate_content_and_url_works(u):
    a, _ = u
    fake = base64.b64encode(b"<?php system($_GET['x']); ?>" * 10).decode()
    assert c.post(f"{P}/uploads/image-base64", json={"image_base64": fake}, headers=a["h"]).status_code == 400
    r = c.post(f"{P}/uploads/image-base64", json={"image_base64": base64.b64encode(_png()).decode(),
                                                  "mime": "image/jpeg"}, headers=a["h"])
    assert r.status_code == 201, r.text
    url = r.json()["url"]
    assert url.startswith("/api/v1/uploads/") and url.endswith(".png")  # extensión REAL
    g = c.get(url)
    assert g.status_code == 200 and g.content[:4] == b"\x89PNG"
    assert c.get(f"{P}/uploads/..%2F..%2Fetc%2Fpasswd").status_code in (403, 404)


def test_leaderboard_counts_real_pieces(u):
    a, b = u
    _db.user_items.insert_many([
        {"user_id": a["id"], "item_variant_id": "v9", "quantity": 2},
        {"user_id": a["id"], "item_variant_id": "v9"},
        {"user_id": b["id"], "item_variant_id": "v9"},
    ])
    lb = c.get(f"{P}/leaderboard", headers=a["h"]).json()
    assert [(e["username"], e["pieces_count"]) for e in lb][:2] == [("ana", 3), ("beto", 1)]
    assert lb[0]["is_current_user"] is True and lb[0]["rank"] == 1
    _db.user_items.insert_one({"user_id": b["id"], "item_variant_id": "v8", "purchase_price": 50})
    by_val = c.get(f"{P}/leaderboard?by=value", headers=a["h"]).json()
    assert by_val[0]["username"] == "beto" and by_val[0]["value"] == 50
    by_div = c.get(f"{P}/leaderboard?by=diversity", headers=a["h"]).json()
    assert by_div[0]["username"] == "beto" and by_div[0]["distinct_models"] == 2
