"""Seed de arranque: marcas, categorías y catálogo Mini GT."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymongo.database import Database

from app.config import get_settings
from app.security import hash_password

BRANDS = [
    ("Hot Wheels", "hot-wheels", "United States", "1:64"),
    ("Matchbox", "matchbox", "United Kingdom", "1:64"),
    ("Mini GT", "mini-gt", "Hong Kong", "1:64"),
    ("Inno64", "inno64", "Hong Kong", "1:64"),
    ("Kaido House", "kaido-house", "United States", "1:64"),
    ("Tomica", "tomica", "Japan", "1:64"),
    ("Greenlight", "greenlight", "United States", "1:64"),
    ("Majorette", "majorette", "France", "1:64"),
    ("M2 Machines", "m2-machines", "United States", "1:64"),
    ("Pop Race", "pop-race", "Hong Kong", "1:64"),
    ("Tarmac Works", "tarmac-works", "Hong Kong", "1:64"),
    ("Johnny Lightning", "johnny-lightning", "United States", "1:64"),
]

BODY_COLORS = [
    "Red", "Blue", "Green", "Black", "White", "Yellow", "Orange", "Purple",
    "Silver", "Grey", "Gray", "Gold", "Pink", "Brown", "Turquoise", "Teal",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def seed_admin(db: Database) -> None:
    s = get_settings()
    if db.users.find_one({"email": s.seed_admin_email.lower()}):
        return
    now = _now()
    res = db.users.insert_one(
        {
            "email": s.seed_admin_email.lower(),
            "username": s.seed_admin_username,
            "pwd_hash": hash_password(s.seed_admin_password),
            "avatar_url": None,
            "is_active": True,
            "total_xp": 0,
            "created_at": now,
            "updated_at": now,
        }
    )
    db.collections.insert_one(
        {
            "user_id": str(res.inserted_id),
            "name": "Mi Garaje",
            "description": "Colección principal",
            "is_private": False,
            "created_at": now,
            "updated_at": now,
        }
    )
    print(f"[seed] admin {s.seed_admin_email} creado")


def seed_brands(db: Database) -> None:
    for name, slug, country, scale in BRANDS:
        if not db.brands.find_one({"slug": slug}):
            db.brands.insert_one(
                {
                    "name": name,
                    "slug": slug,
                    "logo_url": None,
                    "country": country,
                    "website_url": None,
                    "default_scale": scale,
                    "created_at": _now(),
                }
            )
    if not db.categories.find_one({"slug": "diecast"}):
        db.categories.insert_one(
            {
                "name": "Diecast 1:64",
                "slug": "diecast",
                "description": "Coches diecast a escala 1:64",
                "created_at": _now(),
            }
        )


def _release_year(name: str) -> int:
    m = re.search(r"\b(19|20)\d{2}\b", name or "")
    return int(m.group(0)) if m else 2024


def _clean(value: Any) -> str:
    return (value or "").strip() if isinstance(value, str) else ""


def normalize_item(it: dict[str, Any]) -> dict[str, Any] | None:
    """Scraped item → documento Mongo (mismos ids que el TS local)."""
    toy = it.get("toy_number")
    name = _clean(it.get("name"))
    if not name or name.startswith("MINIGT.com"):
        return None
    marque = _clean(it.get("marque")) or "Mini GT"
    scale = _clean(it.get("scale"))
    if not re.fullmatch(r"\d+[:/]\d+", scale or ""):
        scale = "1:64"
    images = it.get("images") or []
    img = images[0] if images else ""
    body = None
    for c in BODY_COLORS:
        if c.lower() in name.lower():
            body = c
            break

    metadata: dict[str, Any] = {
        "scale": scale.replace(":", "/") if scale else "1/64",
        "series": marque,
        "marque": marque,
        "status": it.get("status"),
        "url": it.get("url"),
        "source": "minigt.tsm-models.com",
    }

    variant_id = it["id"]
    master_id = f"{it['id']}-master"
    return {
        "_id": variant_id,  # uuid5 determinista del scrape
        "item_master_id": master_id,
        "variant_name": (name.split(" ", 1)[1] if " " in name else name)[:80],
        "toy_number": toy,
        "body_color": body,
        "main_image_url": img,
        "ai_local_hash": f"hash-{(toy or it['id']).lower()}",
        "metadata": metadata,
        "rarity": "regular",
        "base_material": "Metal/Plastic",
        "created_at": datetime(2026, 9, 23, tzinfo=timezone.utc),
        "item_master": {
            "_id": master_id,
            "name": name,
            "manufacturer": "Mini GT",
            "release_year": _release_year(name),
            "scale": "1:64",
            "category_id": None,
            "created_at": datetime(2026, 9, 23, tzinfo=timezone.utc),
        },
    }


def seed_catalog(db: Database, limit: int | None = None) -> None:
    if db.variants.estimated_document_count() > 0:
        print("[seed] catálogo ya cargado, se omite")
        return

    path = Path(get_settings().catalog_seed_path)
    if not path.exists():
        print(f"[seed] catálogo no encontrado en {path}, se omite")
        return

    data = json.loads(path.read_text())
    items = data.get("items") or []
    variants: list[dict] = []
    masters: dict[str, dict] = {}
    for raw in items:
        norm = normalize_item(raw)
        if not norm:
            continue
        master = norm.pop("item_master")
        masters[str(master.pop("_id"))] = master
        variants.append(norm)
        if limit and len(variants) >= limit:
            break

    for mid, master in masters.items():
        master["_id"] = mid
        db.items.insert_one(master)
    if variants:
        db.variants.insert_many(variants, ordered=False)
    print(f"[seed] catálogo: {len(variants)} variantes, {len(masters)} items")


def run_seed(db: Database, catalog_limit: int | None = None) -> None:
    seed_admin(db)
    seed_brands(db)
    seed_catalog(db, limit=catalog_limit)
