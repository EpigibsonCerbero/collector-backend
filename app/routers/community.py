"""Comunidad: feed social y amigos."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pymongo import DESCENDING, ReturnDocument

from app.db import get_db, oid, serialize
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


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Feed ──────────────────────────────────────────────────────────────────


@feed_router.get("/feed", response_model=list[FeedEventOut])
def list_feed(
    user: CurrentUser = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0,
) -> list[FeedEventOut]:
    docs = list(
        get_db()
        .feed_events.find({})
        .sort("timestamp", DESCENDING)
        .skip(offset)
        .limit(limit)
    )
    return [FeedEventOut(**serialize(d)) for d in docs]  # type: ignore[arg-type]


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
            "reactions": {"🔥": 0, "❤️": 0, "👑": 0, "🤩": 0},
            "user_reaction": None,
        }
    )
    res = get_db().feed_events.insert_one(doc)
    doc["_id"] = res.inserted_id
    return FeedEventOut(**serialize(doc))  # type: ignore[arg-type]


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
    prev = doc.get("user_reaction")
    if prev == body.reaction:
        reactions[prev] = max(0, int(reactions.get(prev, 0)) - 1)
        new_reaction = None
    else:
        if prev:
            reactions[prev] = max(0, int(reactions.get(prev, 0)) - 1)
        reactions[body.reaction] = int(reactions.get(body.reaction, 0)) + 1
        new_reaction = body.reaction
    doc = db.feed_events.find_one_and_update(
        {"_id": oid(event_id)},
        {"$set": {"reactions": reactions, "user_reaction": new_reaction}},
        return_document=ReturnDocument.AFTER,
    )
    return FeedEventOut(**serialize(doc))  # type: ignore[arg-type]


# ── Amigos ────────────────────────────────────────────────────────────────


@friends_router.get("", response_model=list[FriendOut])
def list_friends(user: CurrentUser = Depends(get_current_user)) -> list[FriendOut]:
    docs = list(get_db().friends.find({"user_id": user.id}).sort("added_at", DESCENDING))
    return [FriendOut(**serialize(d)) for d in docs]  # type: ignore[arg-type]


@friends_router.post("", response_model=FriendOut, status_code=201)
def add_friend(
    body: FriendAdd,
    user: CurrentUser = Depends(get_current_user),
) -> FriendOut:
    db = get_db()
    if body.friend_id == user.id:
        raise HTTPException(status_code=400, detail="No puedes agregarte a ti mismo")
    if db.friends.find_one({"user_id": user.id, "friend_id": body.friend_id}):
        raise HTTPException(status_code=409, detail="Ya son amigos")
    doc = body.model_dump()
    doc.update({"user_id": user.id, "added_at": _now().isoformat(), "is_online": False})
    res = db.friends.insert_one(doc)
    doc["_id"] = res.inserted_id
    return FriendOut(**serialize(doc))  # type: ignore[arg-type]


@friends_router.delete("/{friend_id}", status_code=204)
def remove_friend(
    friend_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    res = get_db().friends.delete_one({"_id": oid(friend_id), "user_id": user.id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Amigo no encontrado")


@friends_router.get("/{friend_id}/showcase", response_model=list[ShowcaseItemOut])
def friend_showcase(friend_id: str) -> list[ShowcaseItemOut]:
    docs = list(get_db().friend_showcase.find({"friend_id": friend_id}))
    return [ShowcaseItemOut(**serialize(d)) for d in docs]  # type: ignore[arg-type]
