"""Eventos diecast (convenciones, meets, drops) y chat P2P."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Set

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pymongo import DESCENDING, ReturnDocument

from app.db import get_db, oid, serialize
from app.schemas_social import (
    ChatMessageCreate,
    ChatMessageOut,
    EventCreate,
    EventOut,
)
from app.security import CurrentUser, decode_token, get_current_user

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
# Hilos válidos (cualquier otro → 403):
#   dm-<idA>-<idB>    conversación directa; ids de usuario ordenados (idA < idB)
#   trade-<escrowId>  canal de un canje; solo las dos partes de la oferta
# El autor lo fija el servidor con el token (`author_id`); el cliente decide
# «mío / del otro» comparando con su id, nunca con un `sender` que él mismo manda.


def _thread_member(thread_id: str, user_id: str) -> bool:
    if thread_id.startswith("dm-"):
        parts = thread_id[3:].split("-")
        return (
            len(parts) == 2
            and all(ObjectId.is_valid(p) for p in parts)
            and parts[0] != parts[1]
            and parts[0] < parts[1]
            and user_id in parts
        )
    if thread_id.startswith("trade-"):
        escrow_id = thread_id[6:]
        if not ObjectId.is_valid(escrow_id):
            return False
        db = get_db()
        escrow = db.trade_escrows.find_one({"_id": ObjectId(escrow_id)}, {"offer_id": 1})
        if not escrow or not ObjectId.is_valid(str(escrow.get("offer_id", ""))):
            return False
        offer = db.trade_offers.find_one(
            {"_id": ObjectId(str(escrow["offer_id"]))}, {"from_user_id": 1, "to_user_id": 1}
        )
        return bool(offer) and user_id in (offer.get("from_user_id"), offer.get("to_user_id"))
    return False


def _require_member(thread_id: str, user: CurrentUser) -> None:
    if not _thread_member(thread_id, user.id):
        raise HTTPException(status_code=403, detail="No perteneces a esta conversación")


def _message_doc(thread_id: str, user: CurrentUser, payload: dict) -> dict:
    return {
        "thread_id": thread_id,
        "author_id": user.id,
        "author_name": user.username,
        # «partner» ya no lo puede declarar el cliente: o es suyo o es aviso de sistema.
        "sender": "system" if payload.get("sender") == "system" else "user",
        "text": payload.get("text"),
        "type": payload.get("type")
        if payload.get("type")
        in ("text", "image", "gif", "voice", "piece", "photo_verification", "escrow_alert")
        else "text",
        "photo_url": payload.get("photo_url"),
        "voice_url": payload.get("voice_url"),
        "voice_duration": payload.get("voice_duration"),
        "piece_id": payload.get("piece_id"),
        "piece_name": payload.get("piece_name"),
        "piece_image": payload.get("piece_image"),
        "piece_meta": payload.get("piece_meta"),
        "seal_code": payload.get("seal_code"),
        "condition": payload.get("condition"),
        "amount": payload.get("amount"),
        "reply_to_id": payload.get("reply_to_id"),
        "reply_to_text": payload.get("reply_to_text"),
        # Las reacciones NO las fija el cliente al crear (se podían fabricar).
        "reactions": [],
        "reactions_by": {},
        "client_id": payload.get("client_id"),
        "timestamp": _now().isoformat(),
    }


def _insert_message(doc: dict) -> dict:
    """Inserta; si ya existe uno con el mismo `client_id` del mismo autor en el
    hilo, devuelve ese (idempotente). El cliente reintenta por REST cuando el eco
    del WebSocket no llega y, sin esto, el mensaje se guardaba dos veces."""
    db = get_db()
    if doc.get("client_id"):
        prev = db.chat_messages.find_one(
            {"thread_id": doc["thread_id"], "author_id": doc["author_id"], "client_id": doc["client_id"]}
        )
        if prev:
            return prev
    res = db.chat_messages.insert_one(doc)
    doc["_id"] = res.inserted_id
    return doc


def _msg_out(doc: dict) -> ChatMessageOut:
    data = serialize(doc)
    assert data is not None
    by = data.get("reactions_by") or {}
    if by:
        data["reactions"] = list(by.values())
    return ChatMessageOut(**data)


@chat_router.get("/{thread_id}", response_model=list[ChatMessageOut])
def get_thread(thread_id: str, user: CurrentUser = Depends(get_current_user)) -> list[ChatMessageOut]:
    _require_member(thread_id, user)
    docs = list(
        get_db().chat_messages.find({"thread_id": thread_id}).sort("timestamp", 1)
    )
    return [_msg_out(d) for d in docs]


@chat_router.get("/{thread_id}/reads")
def get_reads(thread_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
    """Hasta cuándo leyó cada participante: {user_id: timestamp ISO}."""
    _require_member(thread_id, user)
    return {
        r["user_id"]: r.get("at")
        for r in get_db().chat_reads.find({"thread_id": thread_id}, {"user_id": 1, "at": 1})
    }


def _mark_read(thread_id: str, user_id: str, at: str) -> str:
    """Guarda la marca de lectura (solo avanza). Devuelve la vigente."""
    db = get_db()
    cur = db.chat_reads.find_one({"thread_id": thread_id, "user_id": user_id}) or {}
    if (cur.get("at") or "") >= at:
        return cur.get("at") or at
    db.chat_reads.update_one(
        {"thread_id": thread_id, "user_id": user_id}, {"$set": {"at": at}}, upsert=True
    )
    return at


@chat_router.post("/{thread_id}/read")
async def mark_read(thread_id: str, body: dict, user: CurrentUser = Depends(get_current_user)) -> dict:
    """Marca leído hasta `at` (timestamp del último mensaje visto) y avisa en vivo."""
    _require_member(thread_id, user)
    at = str(body.get("at") or _now().isoformat())[:40]
    at = _mark_read(thread_id, user.id, at)
    await chat_ws_manager.broadcast(thread_id, {"type": "read", "user_id": user.id, "at": at})
    return {"at": at}


@chat_router.post("/{thread_id}", response_model=ChatMessageOut, status_code=201)
async def add_message(
    thread_id: str,
    body: ChatMessageCreate,
    user: CurrentUser = Depends(get_current_user),
) -> ChatMessageOut:
    _require_member(thread_id, user)
    doc = _insert_message(_message_doc(thread_id, user, body.model_dump()))
    msg_out = _msg_out(doc)
    # Los mensajes por REST también llegan en vivo a quien tenga el hilo abierto.
    await chat_ws_manager.broadcast(thread_id, {"type": "message", "data": msg_out.model_dump()})
    return msg_out


@chat_router.post("/{thread_id}/{message_id}/reactions", response_model=ChatMessageOut)
async def add_reaction(
    thread_id: str,
    message_id: str,
    body: dict,
    user: CurrentUser = Depends(get_current_user),
) -> ChatMessageOut:
    """Una reacción POR USUARIO (como WhatsApp): la misma la quita, otra la cambia.

    Antes era una lista anónima: si dos ponían ❤️ y uno tocaba ❤️, se borraba
    la del otro."""
    _require_member(thread_id, user)
    emoji = str(body.get("emoji") or "").strip()
    if not emoji or len(emoji) > 8:
        raise HTTPException(status_code=400, detail="Emoji no válido")
    db = get_db()
    doc = db.chat_messages.find_one({"_id": oid(message_id), "thread_id": thread_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Mensaje no encontrado")
    by = dict(doc.get("reactions_by") or {})
    if by.get(user.id) == emoji:
        by.pop(user.id, None)
    else:
        by[user.id] = emoji
    reactions = list(by.values())
    db.chat_messages.update_one(
        {"_id": doc["_id"]}, {"$set": {"reactions_by": by, "reactions": reactions}}
    )
    doc["reactions_by"], doc["reactions"] = by, reactions
    msg_out = _msg_out(doc)
    await chat_ws_manager.broadcast(thread_id, {"type": "message", "data": msg_out.model_dump()})
    return msg_out


@chat_router.delete("/{thread_id}", status_code=204)
def clear_thread(thread_id: str, user: CurrentUser = Depends(get_current_user)) -> None:
    # Solo borra lo que escribió quien lo pide: no puede vaciar el historial del otro.
    _require_member(thread_id, user)
    get_db().chat_messages.delete_many({"thread_id": thread_id, "author_id": user.id})


# ── WebSocket: tiempo real ───────────────────────────────────────────────


class ChatConnectionManager:
    """Gestiona conexiones WebSocket activas por thread_id."""

    def __init__(self) -> None:
        self._active: Dict[str, Set[WebSocket]] = {}
        # ws → user_id: presencia real («en el chat») de cada participante.
        self._who: Dict[WebSocket, str] = {}

    async def connect(self, thread_id: str, ws: WebSocket, user_id: str = "") -> None:
        await ws.accept()
        self._active.setdefault(thread_id, set()).add(ws)
        self._who[ws] = user_id

    def disconnect(self, thread_id: str, ws: WebSocket) -> None:
        self._who.pop(ws, None)
        conns = self._active.get(thread_id)
        if conns:
            conns.discard(ws)
            if not conns:
                del self._active[thread_id]

    def present(self, thread_id: str) -> list[str]:
        return sorted({self._who.get(ws, "") for ws in self._active.get(thread_id, set())} - {""})

    async def announce_presence(self, thread_id: str) -> None:
        await self.broadcast(thread_id, {"type": "presence", "user_ids": self.present(thread_id)})

    async def broadcast(
        self,
        thread_id: str,
        message: dict,
        exclude_ws: WebSocket | None = None,
    ) -> None:
        conns = list(self._active.get(thread_id, set()))
        dead: list[WebSocket] = []
        for ws in conns:
            if exclude_ws is not None and ws is exclude_ws:
                continue
            try:
                await ws.send_text(json.dumps(message, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(thread_id, ws)


chat_ws_manager = ChatConnectionManager()


def _ws_user(token: str | None) -> CurrentUser | None:
    if not token:
        return None
    try:
        payload = decode_token(token, expected_type="access")
    except HTTPException:
        return None
    return CurrentUser(id=payload["sub"], username=payload.get("username", ""))


@chat_router.websocket("/ws/{thread_id}")
async def chat_ws(websocket: WebSocket, thread_id: str, token: str | None = None) -> None:
    """
    WebSocket para chat en tiempo real. Exige `?token=<access token>` y que el
    usuario pertenezca al hilo (4401 sin token válido, 4403 si no pertenece).
    Cliente envía: {"text": "...", "type": "text"} o {"action": "ping"}
    Servidor emite: {"type": "message", "data": {...}} o {"type": "pong"}
    """
    # Token por cabecera (preferido: no queda en los logs de acceso) o ?token= (compatibilidad).
    auth_h = websocket.headers.get("authorization", "")
    header_token = auth_h[7:] if auth_h.lower().startswith("bearer ") else None
    user = _ws_user(header_token or token)
    if user is None:
        await websocket.close(code=4401)
        return
    if not _thread_member(thread_id, user.id):
        await websocket.close(code=4403)
        return
    await chat_ws_manager.connect(thread_id, websocket, user.id)
    await chat_ws_manager.announce_presence(thread_id)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "detail": "JSON inválido"}))
                continue

            if payload.get("action") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            # «Escribiendo…» en vivo: se reemite al resto del hilo sin persistir.
            if payload.get("action") == "typing":
                await chat_ws_manager.broadcast(
                    thread_id,
                    {
                        "type": "typing",
                        "user_id": user.id,
                        "username": user.username,
                    },
                    exclude_ws=websocket,
                )
                continue

            # Leído hasta `at`: se guarda y se avisa al otro (doble check real).
            if payload.get("action") == "read":
                at = _mark_read(thread_id, user.id, str(payload.get("at") or _now().isoformat())[:40])
                await chat_ws_manager.broadcast(thread_id, {"type": "read", "user_id": user.id, "at": at})
                continue

            text = payload.get("text")
            has_media = bool(
                payload.get("photo_url")
                or payload.get("voice_url")
                or payload.get("type") in ("image", "gif", "voice", "piece")
            )
            if not has_media and (not text or not isinstance(text, str) or len(text) > 4000):
                await websocket.send_text(json.dumps({"type": "error", "detail": "text requerido"}))
                continue
            if text is not None and (not isinstance(text, str) or len(text) > 4000):
                await websocket.send_text(json.dumps({"type": "error", "detail": "text inválido"}))
                continue

            # Mismas reglas que por REST (antes el WS no validaba longitudes ni tipos).
            try:
                clean = ChatMessageCreate(
                    **{k: v for k, v in payload.items() if k in ChatMessageCreate.model_fields}
                ).model_dump()
            except Exception:
                await websocket.send_text(json.dumps({"type": "error", "detail": "mensaje inválido"}))
                continue
            doc = _insert_message(_message_doc(thread_id, user, clean))
            msg_out = _msg_out(doc)
            await chat_ws_manager.broadcast(
                thread_id,
                {"type": "message", "data": msg_out.model_dump()},
            )
    except WebSocketDisconnect:
        pass
    finally:
        chat_ws_manager.disconnect(thread_id, websocket)
        try:
            await chat_ws_manager.announce_presence(thread_id)
        except Exception:
            pass
