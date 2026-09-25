"""Garaje: colecciones y piezas del usuario (user_items)."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pymongo import ReturnDocument

from app.db import as_id, get_db, oid, serialize
from app.schemas import (
    CollectionCreate,
    CollectionOut,
    CollectionUpdate,
    UserItemCreate,
    UserItemOut,
    UserItemUpdate,
)
from app.security import CurrentUser, get_current_user

router = APIRouter(tags=["garage"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _collection_out(doc: dict) -> CollectionOut:
    data = serialize(doc)
    assert data is not None
    return CollectionOut(**data)


def _user_item_out(db, doc: dict, with_variant: bool = True) -> UserItemOut:
    data = serialize(doc)
    assert data is not None
    if with_variant and data.get("item_variant_id"):
        from app.routers.catalog import _load_master, _variant_out

        v = db.variants.find_one({"_id": as_id(data["item_variant_id"])})
        if v:
            data["item_variant"] = _variant_out(v, _load_master(db, v))
    return UserItemOut(**data)


def _own_collection(db, collection_id: str, user_id: str) -> dict:
    coll = db.collections.find_one({"_id": oid(collection_id)})
    if not coll:
        raise HTTPException(status_code=404, detail="Colección no encontrada")
    if str(coll["user_id"]) != user_id:
        raise HTTPException(status_code=403, detail="No es tu colección")
    return coll


# ── Colecciones ───────────────────────────────────────────────────────────


@router.get("/collections", response_model=list[CollectionOut])
def list_collections(user: CurrentUser = Depends(get_current_user)) -> list[CollectionOut]:
    docs = list(get_db().collections.find({"user_id": user.id}).sort("created_at", 1))
    return [_collection_out(d) for d in docs]


@router.post("/collections", response_model=CollectionOut, status_code=201)
def create_collection(
    body: CollectionCreate,
    user: CurrentUser = Depends(get_current_user),
) -> CollectionOut:
    db = get_db()
    now = _now()
    doc = body.model_dump()
    doc.update({"user_id": user.id, "created_at": now, "updated_at": now})
    res = db.collections.insert_one(doc)
    doc["_id"] = res.inserted_id
    return _collection_out(doc)


@router.get("/collections/{collection_id}", response_model=CollectionOut)
def get_collection(
    collection_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> CollectionOut:
    coll = _own_collection(get_db(), collection_id, user.id)
    return _collection_out(coll)


@router.patch("/collections/{collection_id}", response_model=CollectionOut)
def update_collection(
    collection_id: str,
    body: CollectionUpdate,
    user: CurrentUser = Depends(get_current_user),
) -> CollectionOut:
    db = get_db()
    _own_collection(db, collection_id, user.id)
    updates = {k: v for k, v in body.model_dump(exclude_none=True).items()}
    updates["updated_at"] = _now()
    doc = db.collections.find_one_and_update(
        {"_id": oid(collection_id)},
        {"$set": updates},
        return_document=ReturnDocument.AFTER,
    )
    return _collection_out(doc)  # type: ignore[arg-type]


@router.delete("/collections/{collection_id}", status_code=204)
def delete_collection(
    collection_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    db = get_db()
    _own_collection(db, collection_id, user.id)
    db.user_items.delete_many({"collection_id": collection_id})
    db.collections.delete_one({"_id": oid(collection_id)})


@router.get("/collections/{collection_id}/items", response_model=list[UserItemOut])
def list_collection_items(
    collection_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> list[UserItemOut]:
    db = get_db()
    _own_collection(db, collection_id, user.id)
    docs = list(db.user_items.find({"collection_id": collection_id}).sort("created_at", -1))
    return [_user_item_out(db, d) for d in docs]


# ── Piezas del garaje ─────────────────────────────────────────────────────


@router.get("/garage/items", response_model=list[UserItemOut])
def list_all_user_items(user: CurrentUser = Depends(get_current_user)) -> list[UserItemOut]:
    docs = list(get_db().user_items.find({"user_id": user.id}).sort("created_at", -1))
    return [_user_item_out(get_db(), d) for d in docs]


@router.post("/garage/items", response_model=UserItemOut, status_code=201)
def add_collectible(
    body: UserItemCreate,
    user: CurrentUser = Depends(get_current_user),
) -> UserItemOut:
    db = get_db()
    _own_collection(db, body.collection_id, user.id)
    if not db.variants.find_one({"_id": as_id(body.item_variant_id)}):
        raise HTTPException(status_code=404, detail="Variante no encontrada")

    doc = body.model_dump()
    doc.update(
        {
            "user_id": user.id,
            "collection_id": body.collection_id,
            "item_variant_id": body.item_variant_id,
            "created_at": _now(),
            "updated_at": _now(),
        }
    )
    res = db.user_items.insert_one(doc)
    doc["_id"] = res.inserted_id

    # Auto-remover de wishlists con auto_remove_on_garage_add
    db.user_list_items.delete_many(
        {
            "item_variant_id": body.item_variant_id,
            "user_list_id": {
                "$in": [
                    str(l["_id"])
                    for l in db.user_lists.find(
                        {"user_id": user.id, "list_type": "wishlist", "auto_remove_on_garage_add": True},
                        {"_id": 1},
                    )
                ]
            },
        }
    )
    return _user_item_out(db, doc)


@router.get("/garage/items/{item_id}", response_model=UserItemOut)
def get_user_item(
    item_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> UserItemOut:
    db = get_db()
    doc = db.user_items.find_one({"_id": as_id(item_id)})
    if not doc or doc.get("user_id") != user.id:
        raise HTTPException(status_code=404, detail="Pieza no encontrada")
    return _user_item_out(db, doc)


@router.patch("/garage/items/{item_id}", response_model=UserItemOut)
def update_user_item(
    item_id: str,
    body: UserItemUpdate,
    user: CurrentUser = Depends(get_current_user),
) -> UserItemOut:
    db = get_db()
    doc = db.user_items.find_one({"_id": as_id(item_id)})
    if not doc or doc.get("user_id") != user.id:
        raise HTTPException(status_code=404, detail="Pieza no encontrada")
    updates = {k: v for k, v in body.model_dump(exclude_none=True).items()}
    updates["updated_at"] = _now()
    doc = db.user_items.find_one_and_update(
        {"_id": as_id(item_id)},
        {"$set": updates},
        return_document=ReturnDocument.AFTER,
    )
    return _user_item_out(db, doc)  # type: ignore[arg-type]


@router.delete("/garage/items/{item_id}", status_code=204)
def delete_user_item(
    item_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    db = get_db()
    doc = db.user_items.find_one({"_id": as_id(item_id)})
    if not doc or doc.get("user_id") != user.id:
        raise HTTPException(status_code=404, detail="Pieza no encontrada")
    db.user_items.delete_one({"_id": as_id(item_id)})
