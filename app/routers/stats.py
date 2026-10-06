"""Estadísticas de la colección del usuario."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.db import get_db
from app.security import CurrentUser, get_current_user

router = APIRouter(prefix="/stats", tags=["stats"])


class CollectionStats(BaseModel):
    total_items: int = 0
    total_collections: int = 0
    total_value: float = 0.0
    avg_value: float = 0.0
    unique_variants: int = 0
    duplicates: int = 0
    wishlist_count: int = 0
    tradeable_count: int = 0
    for_sale_count: int = 0
    by_rarity: dict[str, int] = {}
    by_manufacturer: dict[str, int] = {}
    by_sub_brand: dict[str, int] = {}
    by_scale: dict[str, int] = {}
    by_decade: dict[str, int] = {}
    by_condition: dict[str, int] = {}
    rare_count: int = 0  # TH/STH/RLC/chase/limited
    top_pieces: list[dict] = []  # top por valor
    recent_adds: int = 0


_RARE = {
    "treasure_hunt",
    "super_treasure_hunt",
    "rlc",
    "chase",
    "limited_edition",
    "convention",
}


@router.get("/collection", response_model=CollectionStats)
def get_collection_stats(
    user: CurrentUser = Depends(get_current_user),
) -> CollectionStats:
    db = get_db()

    items = list(db.user_items.find({"user_id": user.id}))
    total_items = sum(i.get("quantity", 1) for i in items)
    collections = list(db.collections.find({"user_id": user.id}))

    total_value = 0.0
    by_rarity: dict[str, int] = {}
    by_manufacturer: dict[str, int] = {}
    by_sub_brand: dict[str, int] = {}
    by_scale: dict[str, int] = {}
    by_decade: dict[str, int] = {}
    by_condition: dict[str, int] = {}
    rare_count = 0
    valued: list[dict] = []

    for i in items:
        qty = i.get("quantity", 1) or 1
        price = (i.get("purchase_price") or i.get("sale_price") or 0) * qty
        total_value += price

        cond = i.get("condition") or "Unknown"
        by_condition[cond] = by_condition.get(cond, 0) + qty

        # resolver variante para taxonomía
        vid = i.get("item_variant_id")
        variant = None
        if vid:
            try:
                from app.db import as_id

                variant = db.variants.find_one({"_id": as_id(str(vid))})
            except Exception:  # noqa: BLE001
                variant = db.variants.find_one({"_id": str(vid)})
        meta = (variant or {}).get("metadata") or {}
        master = {}
        mid = (variant or {}).get("item_master_id")
        if mid:
            try:
                from app.db import as_id

                master = db.items.find_one({"_id": as_id(str(mid))}) or {}
            except Exception:  # noqa: BLE001
                master = db.items.find_one({"_id": str(mid)}) or {}

        rarity = (variant or {}).get("rarity") or "regular"
        by_rarity[rarity] = by_rarity.get(rarity, 0) + qty
        if rarity in _RARE:
            rare_count += qty

        mfr = master.get("manufacturer") or meta.get("marque") or "Otro"
        by_manufacturer[mfr] = by_manufacturer.get(mfr, 0) + qty

        sb = master.get("sub_brand") or meta.get("sub_brand") or meta.get("franchise")
        if sb:
            by_sub_brand[sb] = by_sub_brand.get(sb, 0) + qty

        scale = master.get("scale") or meta.get("scale") or "?"
        by_scale[scale] = by_scale.get(scale, 0) + qty

        ry = master.get("release_year") or 0
        decade = f"{(ry // 10) * 10}s" if ry and ry > 0 else "—"
        by_decade[decade] = by_decade.get(decade, 0) + qty

        name = master.get("name") or (variant or {}).get("variant_name") or ""
        valued.append(
            {
                "name": name[:80],
                "toy_number": (variant or {}).get("toy_number"),
                "rarity": rarity,
                "manufacturer": mfr,
                "sub_brand": sb,
                "value": round(price, 2),
                "image_url": (variant or {}).get("main_image_url"),
            }
        )

    valued.sort(key=lambda x: -x["value"])
    variant_ids = [str(i.get("item_variant_id")) for i in items]
    unique_variants = len(set(variant_ids))
    duplicates = total_items - unique_variants

    wishlist = db.lists.find_one({"user_id": user.id, "list_type": "wishlist"})
    wishlist_count = 0
    if wishlist:
        wishlist_count = db.list_items.count_documents({"list_id": str(wishlist["_id"])})

    tradeable_count = sum(1 for i in items if i.get("is_tradeable"))
    for_sale_count = sum(1 for i in items if i.get("is_for_sale"))

    # adds últimos 30 días
    from datetime import datetime, timedelta, timezone

    since = datetime.now(timezone.utc) - timedelta(days=30)
    recent_adds = 0
    for i in items:
        ca = i.get("created_at")
        if isinstance(ca, datetime) and ca >= since:
            recent_adds += i.get("quantity", 1) or 1

    return CollectionStats(
        total_items=total_items,
        total_collections=len(collections),
        total_value=round(total_value, 2),
        avg_value=round(total_value / total_items, 2) if total_items else 0.0,
        unique_variants=unique_variants,
        duplicates=duplicates,
        wishlist_count=wishlist_count,
        tradeable_count=tradeable_count,
        for_sale_count=for_sale_count,
        by_rarity=by_rarity,
        by_manufacturer=by_manufacturer,
        by_sub_brand=by_sub_brand,
        by_scale=by_scale,
        by_decade=by_decade,
        by_condition=by_condition,
        rare_count=rare_count,
        top_pieces=valued[:10],
        recent_adds=recent_adds,
    )
