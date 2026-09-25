"""Listas de usuario: wishlist / trade / sale / custom."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pymongo import ReturnDocument

from app.db import as_id, get_db, oid, serialize
from app.schemas import (
    UserListItemCreate,
    UserListItemOut,
    UserListCreate,
    UserListOut,
)
from app.security import CurrentUser, get_current_user

router = APIRouter(prefix="/lists", tags=["lists"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _list_out(doc: dict) -> UserListOut:
    data = serialize(doc)
    assert data is not None
    return UserListOut(**data)


def _item_out(db, doc: dict, with_variant: bool = True) -> UserListItemOut:
    data = serialize(doc)
    assert data is not None
    if with_variant and data.get("item_variant_id"):
        from app.routers.catalog import _load_master, _variant_out

        v = db.variants.find_one({"_id": as_id(data["item_variant_id"])})
        if v:
            data["item_variant"] = _variant_out(v, _load_master(db, v))
    return UserListItemOut(**data)


def _own_list(db, list_id: str, user_id: str) -> dict:
    doc = db.user_lists.find_one({"_id": oid(list_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Lista no encontrada")
    if str(doc["user_id"]) != user_id:
        raise HTTPException(status_code=403, detail="No es tu lista")
    return doc


@router.get("", response_model=list[UserListOut])
def list_lists(user: CurrentUser = Depends(get_current_user)) -> list[UserListOut]:
    docs = list(get_db().user_lists.find({"user_id": user.id}).sort("created_at", 1))
    return [_list_out(d) for d in docs]


@router.post("", response_model=UserListOut, status_code=201)
def create_list(
    body: UserListCreate,
    user: CurrentUser = Depends(get_current_user),
) -> UserListOut:
    db = get_db()
    now = _now()
    doc = body.model_dump()
    doc.update({"user_id": user.id, "created_at": now, "updated_at": now})
    res = db.user_lists.insert_one(doc)
    doc["_id"] = res.inserted_id
    return _list_out(doc)


@router.get("/wishlist", response_model=UserListOut)
def get_or_create_wishlist(user: CurrentUser = Depends(get_current_user)) -> UserListOut:
    """Devuelve la wishlist del usuario (la crea si no existe)."""
    db = get_db()
    doc = db.user_lists.find_one({"user_id": user.id, "list_type": "wishlist"})
    if not doc:
        now = _now()
        doc = {
            "user_id": user.id,
            "name": "Wishlist",
            "description": "Piezas que quiero conseguir",
            "list_type": "wishlist",
            "is_private": False,
            "auto_remove_on_garage_add": True,
            "created_at": now,
            "updated_at": now,
        }
        res = db.user_lists.insert_one(doc)
        doc["_id"] = res.inserted_id
    return _list_out(doc)


@router.get("/{list_id}/items", response_model=list[UserListItemOut])
def list_items(
    list_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> list[UserListItemOut]:
    db = get_db()
    _own_list(db, list_id, user.id)
    docs = list(db.user_list_items.find({"user_list_id": list_id}).sort("created_at", -1))
    return [_item_out(db, d) for d in docs]


@router.post("/{list_id}/items", response_model=UserListItemOut, status_code=201)
def add_item(
    list_id: str,
    body: UserListItemCreate,
    user: CurrentUser = Depends(get_current_user),
) -> UserListItemOut:
    db = get_db()
    _own_list(db, list_id, user.id)
    if not db.variants.find_one({"_id": as_id(body.item_variant_id)}):
        raise HTTPException(status_code=404, detail="Variante no encontrada")
    if db.user_list_items.find_one(
        {"user_list_id": list_id, "item_variant_id": body.item_variant_id}
    ):
        raise HTTPException(status_code=409, detail="Ya está en la lista")

    doc = body.model_dump()
    doc.update(
        {
            "user_list_id": list_id,
            "item_variant_id": body.item_variant_id,
            "created_at": _now(),
        }
    )
    res = db.user_list_items.insert_one(doc)
    doc["_id"] = res.inserted_id
    return _item_out(db, doc)


@router.delete("/{list_id}/items/{item_id}", status_code=204)
def remove_item(
    list_id: str,
    item_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    db = get_db()
    _own_list(db, list_id, user.id)
    res = db.user_list_items.delete_one({"_id": as_id(item_id), "user_list_id": list_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Elemento no encontrado")
