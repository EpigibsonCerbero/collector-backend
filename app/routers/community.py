"""Comunidad: feed social y amigos."""
from __future__ import annotations

import re
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from pymongo import DESCENDING, ReturnDocument

from app.db import as_id, get_db, oid, serialize
from app.routers.alerts import notify
from app.schemas_social import (
    FeedEventCreate,
    FeedEventOut,
    FeedReaction,
    FriendAdd,
    FriendOut,
    ShowcaseItemOut,
)
from app.security import CurrentUser, get_current_user

feed_router = APIRouter(prefix="/community", tags=["community"])
friends_router = APIRouter(prefix="/friends", tags=["friends"])

_REACTIONS = ("🔥", "❤️", "👑", "🤩")


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Feed ──────────────────────────────────────────────────────────────────


def _feed_out(doc: dict, user_id: str) -> FeedEventOut:
    """La reacción «mía» es POR USUARIO (`reactions_by`), no global del evento."""
    data = serialize(doc)
    assert data is not None
    by = data.pop("reactions_by", None) or {}
    data["user_reaction"] = by.get(user_id)
    return FeedEventOut(**data)


@feed_router.get("/feed", response_model=list[FeedEventOut])
def list_feed(
    user: CurrentUser = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0,
) -> list[FeedEventOut]:
    limit = max(1, min(limit, 100))
    docs = list(
        get_db()
        .feed_events.find({})
        .sort("timestamp", DESCENDING)
        .skip(max(0, offset))
        .limit(limit)
    )
    return [_feed_out(d, user.id) for d in docs]


@feed_router.post("/feed", response_model=FeedEventOut, status_code=201)
def add_event(
    body: FeedEventCreate,
    user: CurrentUser = Depends(get_current_user),
) -> FeedEventOut:
    doc = body.model_dump()
    doc.update(
        {
            "user_id": user.id,
            "username": user.username,
            "avatar": "",
            "timestamp": _now().isoformat(),
            "reactions": {r: 0 for r in _REACTIONS},
            "reactions_by": {},
        }
    )
    res = get_db().feed_events.insert_one(doc)
    doc["_id"] = res.inserted_id
    return _feed_out(doc, user.id)


@feed_router.post("/feed/{event_id}/react", response_model=FeedEventOut)
def react(
    event_id: str,
    body: FeedReaction,
    user: CurrentUser = Depends(get_current_user),
) -> FeedEventOut:
    db = get_db()
    doc = db.feed_events.find_one({"_id": oid(event_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Evento no encontrado")
    reactions = dict(doc.get("reactions") or {})
    by = dict(doc.get("reactions_by") or {})
    prev = by.get(user.id)
    if prev:
        reactions[prev] = max(0, int(reactions.get(prev, 0)) - 1)
    if prev == body.reaction:
        by.pop(user.id, None)
    else:
        reactions[body.reaction] = int(reactions.get(body.reaction, 0)) + 1
        by[user.id] = body.reaction
    doc = db.feed_events.find_one_and_update(
        {"_id": oid(event_id)},
        {"$set": {"reactions": reactions, "reactions_by": by}, "$unset": {"user_reaction": ""}},
        return_document=ReturnDocument.AFTER,
    )
    return _feed_out(doc, user.id)


# ── Amigos ────────────────────────────────────────────────────────────────
# La amistad es entre USUARIOS REALES: `friend_id` es el id del usuario (antes
# era el texto que se tecleaba). Es mutua: agregar crea los dos sentidos.


def _find_user(db, handle: str) -> dict | None:
    h = (handle or "").strip().lstrip("@")
    if not h:
        return None
    u = db.users.find_one({"username": {"$regex": f"^{re.escape(h)}$", "$options": "i"}})
    if not u and ObjectId.is_valid(h):
        u = db.users.find_one({"_id": ObjectId(h)})
    return u


def _public_items_query(db, owner_id: str) -> dict:
    private = [
        str(c["_id"]) for c in db.collections.find({"user_id": owner_id, "is_private": True}, {"_id": 1})
    ]
    q: dict = {"user_id": owner_id}
    if private:
        q["collection_id"] = {"$nin": private}
    return q


def _friend_out(db, doc: dict) -> FriendOut:
    data = serialize(doc)
    assert data is not None
    fid = data.get("friend_id", "")
    if ObjectId.is_valid(fid):
        u = db.users.find_one({"_id": ObjectId(fid)}, {"username": 1, "avatar_url": 1})
        if u:
            data["username"] = u.get("username") or data.get("username", "")
            data["avatar"] = u.get("avatar_url") or data.get("avatar") or ""
        data["pieces_count"] = db.user_items.count_documents(_public_items_query(db, fid))
    return FriendOut(**data)


@friends_router.get("", response_model=list[FriendOut])
def list_friends(user: CurrentUser = Depends(get_current_user)) -> list[FriendOut]:
    db = get_db()
    docs = list(db.friends.find({"user_id": user.id}).sort("added_at", DESCENDING))
    # Las amistades antiguas guardaban el texto tecleado (no un usuario): se ocultan.
    return [_friend_out(db, d) for d in docs if ObjectId.is_valid(str(d.get("friend_id", "")))]


@friends_router.post("", response_model=FriendOut, status_code=201)
def add_friend(
    body: FriendAdd,
    user: CurrentUser = Depends(get_current_user),
) -> FriendOut:
    db = get_db()
    target = _find_user(db, body.username or body.friend_id)
    if not target:
        raise HTTPException(status_code=404, detail="No existe ningún coleccionista con ese usuario")
    fid = str(target["_id"])
    if fid == user.id:
        raise HTTPException(status_code=400, detail="No puedes agregarte a ti mismo")
    if db.friends.find_one({"user_id": user.id, "friend_id": fid}):
        raise HTTPException(status_code=409, detail="Ya son amigos")
    now = _now().isoformat()
    doc = {
        "user_id": user.id,
        "friend_id": fid,
        "username": target["username"],
        "avatar": target.get("avatar_url") or "",
        "location": body.location,
        "specialties": body.specialties,
        "added_at": now,
        "is_online": False,
    }
    res = db.friends.insert_one(doc)
    doc["_id"] = res.inserted_id
    if not db.friends.find_one({"user_id": fid, "friend_id": user.id}):
        db.friends.insert_one(
            {
                "user_id": fid,
                "friend_id": user.id,
                "username": user.username,
                "avatar": "",
                "location": "",
                "specialties": [],
                "added_at": now,
                "is_online": False,
            }
        )
        notify(db, fid, "friend", "Nuevo amigo", f"@{user.username} te agregó a su círculo.", "👥")
    return _friend_out(db, doc)


@friends_router.delete("/{friend_id}", status_code=204)
def remove_friend(
    friend_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    db = get_db()
    doc = db.friends.find_one({"_id": oid(friend_id), "user_id": user.id})
    if not doc:
        raise HTTPException(status_code=404, detail="Amigo no encontrado")
    db.friends.delete_one({"_id": doc["_id"]})
    db.friends.delete_one({"user_id": doc.get("friend_id"), "friend_id": user.id})


@friends_router.get("/{friend_ref}/showcase", response_model=list[ShowcaseItemOut])
def friend_showcase(
    friend_ref: str,
    user: CurrentUser = Depends(get_current_user),
) -> list[ShowcaseItemOut]:
    """Vitrina = garaje REAL del usuario (sin colecciones privadas).

    `friend_ref` puede ser el id de la amistad o el id del usuario (el ranking
    abre vitrinas de gente que aún no es amiga).
    """
    db = get_db()
    owner_id = friend_ref
    if ObjectId.is_valid(friend_ref):
        rel = db.friends.find_one({"_id": ObjectId(friend_ref), "user_id": user.id})
        if rel:
            owner_id = rel["friend_id"]
    if not ObjectId.is_valid(owner_id) or not db.users.find_one({"_id": ObjectId(owner_id)}, {"_id": 1}):
        raise HTTPException(status_code=404, detail="Coleccionista no encontrado")
    items = list(db.user_items.find(_public_items_query(db, owner_id)).sort("created_at", DESCENDING).limit(200))
    # Variantes y maestros en lote (no 2 consultas por pieza).
    vkeys = [as_id(str(it["item_variant_id"])) for it in items if it.get("item_variant_id")]
    variants = {str(v["_id"]): v for v in db.variants.find({"_id": {"$in": vkeys}})} if vkeys else {}
    mkeys = [as_id(str(v["item_master_id"])) for v in variants.values() if v.get("item_master_id")]
    masters = {str(m["_id"]): m for m in db.items.find({"_id": {"$in": mkeys}}, {"name": 1})} if mkeys else {}
    out: list[ShowcaseItemOut] = []
    for it in items:
        v = variants.get(str(it.get("item_variant_id"))) or {}
        master = masters.get(str(v.get("item_master_id"))) or {}
        photo = (it.get("custom_images") or [None])[0]
        if photo and ":::" in photo:
            photo = photo.split(":::", 1)[1]
        if photo and not str(photo).startswith("http"):
            photo = None  # ruta local del teléfono del dueño: nadie más puede verla
        out.append(
            ShowcaseItemOut(
                id=str(it["_id"]),
                friend_id=owner_id,
                variant_id=str(it.get("item_variant_id") or ""),
                variant_name=master.get("name") or v.get("variant_name") or "Pieza",
                livery=v.get("livery_desc") or v.get("variant_name") or "",
                box_number=v.get("toy_number") or "",
                image_url=photo or v.get("main_image_url") or "",
                condition=it.get("condition") or "Mint",
                value=float(it.get("purchase_price") or 0),
                is_tradeable=bool(it.get("is_tradeable")),
            )
        )
    return out
