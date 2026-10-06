"""Eventos diecast (convenciones, meets, drops) y chat P2P."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Set

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pymongo import DESCENDING, ReturnDocument

from app.db import get_db, oid, serialize
from app.schemas_social import (
    ChatMessageCreate,
    ChatMessageOut,
    EventCreate,
    EventOut,
)
from app.security import CurrentUser, get_current_user

events_router = APIRouter(prefix="/events", tags=["events"])
chat_router = APIRouter(prefix="/chat", tags=["chat"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Eventos ───────────────────────────────────────────────────────────────


@events_router.get("", response_model=list[EventOut])
def list_events(
    user: CurrentUser = Depends(get_current_user),
    type: str | None = None,
) -> list[EventOut]:
    db = get_db()
    query = {"type": type} if type else {}
    docs = list(db.events.find(query).sort("date", 1))
    favs = {
        str(e["event_id"])
        for e in db.user_favorites.find({"user_id": user.id, "kind": "event"}, {"event_id": 1})
    }
    out = []
    for d in docs:
        data = serialize(d)
        assert data is not None
        data["is_favorite"] = data["id"] in favs
        data["is_custom"] = bool(data.get("user_id"))
        out.append(EventOut(**data))
    return out


@events_router.post("", response_model=EventOut, status_code=201)
def add_custom_event(
    body: EventCreate,
    user: CurrentUser = Depends(get_current_user),
) -> EventOut:
    doc = body.model_dump()
    doc.update({"user_id": user.id, "is_favorite": False, "is_custom": True})
    res = get_db().events.insert_one(doc)
    doc["_id"] = res.inserted_id
    return EventOut(**serialize(doc))  # type: ignore[arg-type]


@events_router.post("/{event_id}/favorite", response_model=EventOut)
def toggle_favorite(
    event_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> EventOut:
    db = get_db()
    doc = db.events.find_one({"_id": oid(event_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Evento no encontrado")
    fav = db.user_favorites.find_one({"user_id": user.id, "kind": "event", "event_id": event_id})
    if fav:
        db.user_favorites.delete_one({"_id": fav["_id"]})
        is_fav = False
    else:
        db.user_favorites.insert_one(
            {"user_id": user.id, "kind": "event", "event_id": event_id}
        )
        is_fav = True
    data = serialize(doc)
    assert data is not None
    data["is_favorite"] = is_fav
    data["is_custom"] = bool(data.get("user_id"))
    return EventOut(**data)


@events_router.delete("/{event_id}", status_code=204)
def remove_event(
    event_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    res = get_db().events.delete_one({"_id": oid(event_id), "user_id": user.id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Evento no encontrado o no es tuyo")


# ── Chat P2P ──────────────────────────────────────────────────────────────


@chat_router.get("/{thread_id}", response_model=list[ChatMessageOut])
def get_thread(thread_id: str, user: CurrentUser = Depends(get_current_user)) -> list[ChatMessageOut]:
    docs = list(
        get_db().chat_messages.find({"thread_id": thread_id}).sort("timestamp", 1)
    )
    return [ChatMessageOut(**serialize(d)) for d in docs]  # type: ignore[arg-type]


@chat_router.post("/{thread_id}", response_model=ChatMessageOut, status_code=201)
def add_message(
    thread_id: str,
    body: ChatMessageCreate,
    user: CurrentUser = Depends(get_current_user),
) -> ChatMessageOut:
    doc = body.model_dump()
    doc.update(
        {
            "thread_id": thread_id,
            "timestamp": _now().isoformat(),
            "author_name": body.author_name or (user.username if body.sender == "user" else None),
        }
    )
    res = get_db().chat_messages.insert_one(doc)
    doc["_id"] = res.inserted_id
    return ChatMessageOut(**serialize(doc))  # type: ignore[arg-type]


@chat_router.delete("/{thread_id}", status_code=204)
def clear_thread(thread_id: str, user: CurrentUser = Depends(get_current_user)) -> None:
    get_db().chat_messages.delete_many({"thread_id": thread_id})


# ── WebSocket: tiempo real ───────────────────────────────────────────────


class ChatConnectionManager:
    """Gestiona conexiones WebSocket activas por thread_id."""

    def __init__(self) -> None:
        self._active: Dict[str, Set[WebSocket]] = {}

    async def connect(self, thread_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._active.setdefault(thread_id, set()).add(ws)

    def disconnect(self, thread_id: str, ws: WebSocket) -> None:
        conns = self._active.get(thread_id)
        if conns:
            conns.discard(ws)
            if not conns:
                del self._active[thread_id]

    async def broadcast(self, thread_id: str, message: dict) -> None:
        conns = self._active.get(thread_id, set())
        dead: list[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_text(json.dumps(message, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(thread_id, ws)


chat_ws_manager = ChatConnectionManager()


@chat_router.websocket("/ws/{thread_id}")
async def chat_ws(websocket: WebSocket, thread_id: str) -> None:
    """
    WebSocket para chat en tiempo real.
    Cliente envía: {"text": "...", "type": "text"} o {"action": "ping"}
    Servidor emite: {"type": "message", "data": {...}} o {"type": "ack", ...}
    """
    await chat_ws_manager.connect(thread_id, websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "detail": "JSON inválido"}))
                continue

            action = payload.get("action")
            if action == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            # Nuevo mensaje
            text = payload.get("text")
            if not text:
                await websocket.send_text(json.dumps({"type": "error", "detail": "text requerido"}))
                continue

            doc = {
                "thread_id": thread_id,
                "sender": payload.get("sender", "user"),
                "author_name": payload.get("author_name"),
                "text": text,
                "type": payload.get("type", "text"),
                "photo_url": payload.get("photo_url"),
                "seal_code": payload.get("seal_code"),
                "condition": payload.get("condition"),
                "amount": payload.get("amount"),
                "timestamp": _now().isoformat(),
            }
            res = get_db().chat_messages.insert_one(doc)
            doc["_id"] = res.inserted_id
            msg_out = ChatMessageOut(**serialize(doc))  # type: ignore[arg-type]

            # Broadcast a todos los conectados en este thread
            await chat_ws_manager.broadcast(
                thread_id,
                {"type": "message", "data": msg_out.model_dump()},
            )
    except WebSocketDisconnect:
        pass
    finally:
        chat_ws_manager.disconnect(thread_id, websocket)
