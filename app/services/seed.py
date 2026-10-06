"""Seed de arranque: marcas, categorías y catálogo Mini GT + Hot Wheels."""
from __future__ import annotations

import json
import re
import uuid
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
    # Pieza fuera del sitio = descontinuada. NUNCA se excluye del catálogo:
    # en coleccionismo lo viejo/descontinuado vale más (regla de producto).
    if it.get("discontinued"):
        metadata["discontinued"] = True
        if it.get("discontinued_at"):
            metadata["discontinued_at"] = it["discontinued_at"]

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


# ── Hot Wheels / Mattel Shop (paralelo a Mini GT, no lo toca) ──────────────

# namespace = uuid5(URL, "collector-manager/minigt-catalog") — mismo que Mini GT.
_UUID5_NS = uuid.UUID("a9f73cdf-180a-5bd7-ac02-42c291653666")


def _u5(seed: str) -> str:
    return str(uuid.uuid5(_UUID5_NS, seed))


def _hw_source_id(it: dict[str, Any]) -> str:
    if it.get("id") and str(it["id"]).startswith("hw-"):
        return str(it["id"]).lower()
    sku = _clean(it.get("mattel_product_number") or it.get("sku"))
    return f"hw-{sku.lower()}" if sku else f"hw-{it.get('handle', 'unknown')}"


def _hw_rarity(name: str, subtypes: list[str]) -> str:
    blob = f"{name} {' '.join(subtypes)}".lower()
    if "super treasure hunt" in blob or re.search(r"\bsth\b", blob):
        return "super_treasure_hunt"
    if "treasure hunt" in blob or re.search(r"\bth\b", (name or "").lower()):
        return "treasure_hunt"
    if "rlc" in blob:
        return "rlc"
    if "convention" in blob or "sdcc" in blob or "nycc" in blob:
        return "convention"
    if "premium" in blob or "car culture" in blob or "real riders" in blob:
        return "premium"
    if "coleccionista" in blob or "collector" in blob or "adultos" in blob or "adult builder" in blob:
        return "exclusive"
    if "chase" in blob:
        return "chase"
    return "regular"


_HW_BRAND_PREFIX_RE = re.compile(
    r"^(?:Hot Wheels|Matchbox|Monster Trucks|Cars de Disney y Pixar|"
    r"Disney(?:\s+and)?\s+Pixar(?:'s)?\s+Cars|Carros da Disney e Pixar|"
    r"Les Bagnoles|Thomas(?:\s*&\s*Friends)?|Barbie|Polly Pocket|"
    r"Jurassic World|Toy Story|Power Wheels|Little People|Mattel Brick Shop)"
    r"(?:\s+\S+)*?\s+",
    re.I,
)


def _hw_variant_name(name: str) -> str:
    out = _HW_BRAND_PREFIX_RE.sub("", name).strip()
    return (out or name)[:80]


def normalize_hotwheels_item(it: dict[str, Any]) -> dict[str, Any] | None:
    """Scraped Hot Wheels item → documento Mongo (mismos uuid5 que el TS local)."""
    name = _clean(it.get("name"))
    if not name:
        return None
    sid = _hw_source_id(it)
    sku = _clean(it.get("mattel_product_number") or it.get("sku")) or None
    # marca = Mattel (dueño de shop.mattel.com); submarca = Hot Wheels / Matchbox / …
    brand = "Mattel"
    sub_brand = _clean(it.get("sub_brand") or it.get("franchise") or it.get("brand")) or "Hot Wheels"
    scale = _clean(it.get("scale"))
    if not re.fullmatch(r"\d+[:/]\d+", scale or ""):
        scale = "1:64"
    subtypes = [s for s in (_clean(x) for x in (it.get("subtypes") or [])) if s]
    upc = _clean(it.get("upc_ean")) or None
    images = it.get("images") or []
    img = images[0] if images else ""
    body = it.get("body_color")
    if not body:
        for c in BODY_COLORS:
            if c.lower() in name.lower():
                body = c
                break

    variant_id = _u5(sid)
    master_id = _u5(f"{sid}-master")

    metadata: dict[str, Any] = {
        "scale": scale.replace(":", "/") if scale else "1/64",
        "series": _clean(it.get("web_category")) or (subtypes[0] if subtypes else sub_brand),
        "marque": brand,
        "sub_brand": sub_brand,
        "franchise": sub_brand,
        "source": "shop.mattel.com",
        "source_id": sid,
        "super_category": _clean(it.get("super_category")),
        "web_category": _clean(it.get("web_category")),
        "subtypes": subtypes,
        "age_grade": _clean(it.get("age_grade")),
        "upc_ean": upc,
        "price": it.get("price"),
        "price_currency": it.get("price_currency") or "MXN",
        "url": it.get("url"),
        "handle": it.get("handle"),
    }
    # Pieza fuera del sitio = descontinuada. NUNCA se excluye del catálogo
    # (regla de producto: lo descontinuado/viejo vale más para el coleccionista).
    if it.get("discontinued"):
        metadata["discontinued"] = True
        if it.get("discontinued_at"):
            metadata["discontinued_at"] = it["discontinued_at"]

    return {
        "_id": variant_id,
        "item_master_id": master_id,
        "variant_name": _hw_variant_name(name),
        "toy_number": sku,
        "sku_upc": upc,
        "body_color": body,
        "rarity": _hw_rarity(name, subtypes),
        "base_material": "Metal/Plastic",
        "main_image_url": img,
        "ai_local_hash": f"hash-{sid}",
        "metadata": metadata,
        "created_at": datetime(2026, 9, 28, tzinfo=timezone.utc),
        "item_master": {
            "_id": master_id,
            "name": name,
            "manufacturer": brand,
            "sub_brand": sub_brand,
            # solo año real (enrich_hotwheels_filter_fields); 0 = desconocido
            "release_year": it.get("release_year") or 0,
            "scale": scale.replace(":", "/") if scale else "1/64",
            "category_id": None,
            "created_at": datetime(2026, 9, 28, tzinfo=timezone.utc),
        },
    }


def seed_hotwheels_catalog(db: Database, limit: int | None = None) -> None:
    """Carga el catálogo Hot Wheels. Idempotente e independiente del seed Mini GT."""
    if db.variants.find_one({"ai_local_hash": {"$regex": r"^hash-hw-"}}):
        print("[seed] catálogo Hot Wheels ya cargado, se omite")
        return

    path = Path(get_settings().hotwheels_catalog_seed_path)
    if not path.exists():
        print(f"[seed] catálogo Hot Wheels no encontrado en {path}, se omite")
        return

    data = json.loads(path.read_text())
    items = data.get("items") or []
    variants: list[dict] = []
    masters: dict[str, dict] = {}
    seen_sid: set[str] = set()
    for raw in items:
        sid = _hw_source_id(raw)
        if sid in seen_sid:
            continue
        seen_sid.add(sid)
        norm = normalize_hotwheels_item(raw)
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
    print(f"[seed] catálogo Hot Wheels: {len(variants)} variantes, {len(masters)} items")


def seed_diecast_extra_catalog(db: Database, limit: int | None = None) -> None:
    """Carga el catálogo multi-marca extra (Inno64, Tarmac, Kaido…).

    Cualquier pieza y cualquier escala entra — es app de coleccionismo.
    Idempotente e independiente de Mini GT / Hot Wheels.
    """
    if db.variants.find_one({"ai_local_hash": {"$regex": r"^hash-(inno64|tarmac-works|kaidohouse)-"}}):
        print("[seed] catálogo diecast extra ya cargado, se omite")
        return

    path = Path(get_settings().diecast_extra_seed_path)
    if not path.exists():
        print(f"[seed] catálogo diecast extra no encontrado en {path}, se omite")
        return

    data = json.loads(path.read_text())
    items = data.get("items") or []
    variants: list[dict] = []
    masters: dict[str, dict] = {}
    for raw in items:
        norm = dict(raw)
        master = norm.pop("item_master", None)
        # limpiar campos de pipeline que no van a Mongo
        norm.pop("_images", None)
        norm.pop("_raw_sku", None)
        norm.pop("images", None)  # main_image_url ya está
        mid = norm.get("item_master_id")
        if master and mid:
            master = dict(master)
            master["_id"] = mid
            master.pop("id", None)
            masters[mid] = master
        norm["_id"] = norm.pop("id")
        variants.append(norm)
        if limit and len(variants) >= limit:
            break

    for mid, master in masters.items():
        if not db.items.find_one({"_id": mid}):
            db.items.insert_one(master)
    if variants:
        db.variants.insert_many(variants, ordered=False)
    print(f"[seed] catálogo diecast extra: {len(variants)} variantes, {len(masters)} items")


def run_seed(db: Database, catalog_limit: int | None = None) -> None:
    seed_admin(db)
    seed_brands(db)
    seed_catalog(db, limit=catalog_limit)
    seed_hotwheels_catalog(db, limit=catalog_limit)
    seed_diecast_extra_catalog(db, limit=catalog_limit)
