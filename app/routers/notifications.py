"""Push notifications — registro de tokens y envío via Expo Push API."""
from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.db import get_db
from app.security import CurrentUser, get_current_user

router = APIRouter(prefix="/notifications", tags=["notifications"])

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"


class RegisterToken(BaseModel):
    push_token: str


class SendNotification(BaseModel):
    to_user_id: str
    title: str
    body: str
    data: dict = {}


class NotificationOut(BaseModel):
    success: bool
    message: str


@router.post("/register", response_model=NotificationOut)
async def register_push_token(
    body: RegisterToken,
    user: CurrentUser = Depends(get_current_user),
) -> NotificationOut:
    """Registra el push token del usuario."""
    if not body.push_token or not body.push_token.startswith("ExponentPushToken["):
        raise HTTPException(status_code=400, detail="Push token inválido")

    db = get_db()
    # Actualizar o insertar token
    db.user_push_tokens.update_one(
        {"user_id": user.id},
        {
            "$set": {
                "user_id": user.id,
                "push_token": body.push_token,
                "updated_at": __import__("datetime").datetime.utcnow().isoformat(),
            }
        },
        upsert=True,
    )
    return NotificationOut(success=True, message="Token registrado")


def send_push(db, user_id: str, title: str, body: str, data: dict | None = None) -> None:
    """Push al usuario si registró token. En un hilo: nunca frena la petición.

    Antes existían /send y /send-bulk: cualquier usuario mandaba pushes con
    cualquier texto a cualquiera (spam/phishing). Ahora solo el servidor
    notifica, desde `alerts.notify()`.
    """
    try:
        doc = db.user_push_tokens.find_one({"user_id": user_id})
    except Exception:
        return
    if not doc or not doc.get("push_token"):
        return
    payload = {
        "to": doc["push_token"],
        "title": title,
        "body": body,
        "data": data or {},
        "sound": "default",
        "channelId": "default",
    }

    def _go() -> None:
        try:
            httpx.post(EXPO_PUSH_URL, json=payload, timeout=8)
        except Exception:
            pass

    import threading

    threading.Thread(target=_go, daemon=True).start()
