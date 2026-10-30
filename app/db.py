"""Lazy MongoDB access (pymongo sync client).

Connections open on first use so the app can boot (and tests run) without a
live database. Handlers call into Mongo via `get_db()`; FastAPI runs sync
endpoints in a threadpool so blocking IO is fine.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, TEXT, MongoClient
from pymongo.database import Database

from app.config import get_settings

_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(get_settings().mongo_url, serverSelectionTimeoutMS=5000)
    return _client


def get_db() -> Database:
    return get_client()[get_settings().mongo_db]


def ping() -> bool:
    try:
        get_client().admin.command("ping")
        return True
    except Exception:
        return False


def oid(value: str | ObjectId) -> ObjectId:
    """String → ObjectId, with a clear 400-ish error for bad ids."""
    from fastapi import HTTPException

    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(value)
    except Exception:
        raise HTTPException(status_code=400, detail=f"id inválido: {value}")


def as_id(value: str | ObjectId) -> str | ObjectId:
    """Lookup key for `_id` that may be an ObjectId OR a stable string id.

    The seeded catalog uses uuid5 strings as `_id` (kept 1:1 with the local TS
    catalog); everything else is an ObjectId minted by Mongo.
    """
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(value)
    except Exception:
        return value


def serialize(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    """Mongo doc → JSON-friendly dict (`_id` becomes `id`, datetimes ISO)."""
    if doc is None:
        return None
    out: dict[str, Any] = {}
    for key, value in doc.items():
        if key == "_id":
            out["id"] = str(value)
        elif isinstance(value, ObjectId):
            out[key] = str(value)
        elif isinstance(value, datetime):
            out[key] = value.isoformat()
        elif isinstance(value, date):
            out[key] = value.isoformat()
        else:
            out[key] = value
    return out


def serialize_many(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [serialize(d) for d in docs]  # type: ignore[misc]


INDEXES: dict[str, list[tuple[list[tuple[str, int]], dict[str, Any]]]] = {
    "users": [
        ([("email", ASCENDING)], {"unique": True}),
        ([("username", ASCENDING)], {"unique": True}),
    ],
    "brands": [
        ([("slug", ASCENDING)], {"unique": True}),
        ([("name", ASCENDING)], {"unique": True}),
    ],
    "series": [
        ([("brand_id", ASCENDING), ("name", ASCENDING), ("release_year", ASCENDING)], {}),
    ],
    "categories": [
        ([("slug", ASCENDING)], {"unique": True}),
    ],
    "items": [
        ([("brand_id", ASCENDING)], {}),
        ([("series_id", ASCENDING)], {}),
        ([("release_year", ASCENDING)], {}),
        ([("name", TEXT), ("manufacturer", TEXT), ("casting_name", TEXT)], {}),
    ],
    "variants": [
        ([("ai_local_hash", ASCENDING)], {"unique": True}),
        ([("item_master_id", ASCENDING)], {}),
        ([("sku_upc", ASCENDING)], {}),
        ([("toy_number", ASCENDING)], {}),
        ([("rarity", ASCENDING)], {}),
        ([("name", TEXT), ("variant_name", TEXT), ("toy_number", TEXT)], {}),
    ],
    "collections": [
        ([("user_id", ASCENDING)], {}),
    ],
    "user_items": [
        ([("collection_id", ASCENDING)], {}),
        ([("user_id", ASCENDING)], {}),
        ([("item_variant_id", ASCENDING)], {}),
        ([("is_tradeable", ASCENDING)], {}),
    ],
    "user_lists": [
        ([("user_id", ASCENDING)], {}),
    ],
    "user_list_items": [
        ([("user_list_id", ASCENDING)], {}),
        ([("item_variant_id", ASCENDING)], {}),
    ],
    "item_value_history": [
        ([("item_variant_id", ASCENDING), ("recorded_date", DESCENDING)], {}),
    ],
    "transactions": [
        ([("user_id", ASCENDING), ("created_at", DESCENDING)], {}),
    ],
    "variant_proposals": [
        ([("user_id", ASCENDING), ("created_at", DESCENDING)], {}),
    ],
    "trade_listings": [
        ([("user_id", ASCENDING)], {}),
        ([("variant_id", ASCENDING)], {}),
        ([("type", ASCENDING), ("created_at", DESCENDING)], {}),
    ],
    "trade_offers": [
        ([("from_user_id", ASCENDING), ("status", ASCENDING)], {}),
        ([("to_user_id", ASCENDING), ("status", ASCENDING)], {}),
    ],
    "trade_escrows": [
        ([("offer_id", ASCENDING)], {"unique": True}),
        ([("status", ASCENDING)], {}),
    ],
    "missions": [
        ([("user_id", ASCENDING), ("date", ASCENDING)], {}),
    ],
    "feed_events": [
        ([("created_at", DESCENDING)], {}),
        ([("user_id", ASCENDING)], {}),
    ],
    "friends": [
        ([("user_id", ASCENDING), ("friend_id", ASCENDING)], {"unique": True}),
    ],
    "friend_showcase": [
        ([("friend_id", ASCENDING)], {}),
    ],
    "events": [
        ([("type", ASCENDING), ("date", ASCENDING)], {}),
        ([("user_id", ASCENDING)], {}),
    ],
    "chat_messages": [
        ([("thread_id", ASCENDING), ("timestamp", ASCENDING)], {}),
        ([("thread_id", ASCENDING), ("author_id", ASCENDING), ("client_id", ASCENDING)], {}),
    ],
    # Fallos de login/códigos: caducan solos a la hora (TTL).
    "auth_failures": [
        ([("at", ASCENDING)], {"expireAfterSeconds": 3600}),
        ([("key", ASCENDING), ("at", ASCENDING)], {}),
    ],
    "chat_reads": [
        ([("thread_id", ASCENDING), ("user_id", ASCENDING)], {"unique": True}),
    ],
    "challenge_claims": [
        ([("user_id", ASCENDING), ("challenge_id", ASCENDING)], {"unique": True}),
    ],
    "alerts": [
        ([("user_id", ASCENDING), ("created_at", DESCENDING)], {}),
    ],
    "user_favorites": [
        ([("user_id", ASCENDING), ("kind", ASCENDING), ("event_id", ASCENDING)], {"unique": True}),
    ],
    "subscriptions": [
        ([("user_id", ASCENDING)], {"unique": True}),
    ],
    "scan_usage": [
        ([("user_id", ASCENDING), ("day", ASCENDING)], {"unique": True}),
        ([("created_at", ASCENDING)], {"expireAfterSeconds": 3 * 24 * 3600}),
    ],
}


def ensure_indexes() -> None:
    db = get_db()
    for coll, defs in INDEXES.items():
        for keys, opts in defs:
            # Uno a uno: un índice en conflicto no debe impedir el arranque ni el seed.
            try:
                db[coll].create_index(keys, **opts)
            except Exception as exc:  # noqa: BLE001
                print(f"[indexes] {coll} {keys}: {exc}")
