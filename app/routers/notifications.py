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


@router.post("/send", response_model=NotificationOut)
async def send_notification(
    body: SendNotification,
    user: CurrentUser = Depends(get_current_user),
) -> NotificationOut:
    """Envía una push notification a otro usuario via Expo."""
    db = get_db()

    # Buscar token del destinatario
    token_doc = db.user_push_tokens.find_one({"user_id": body.to_user_id})
    if not token_doc:
        return NotificationOut(success=False, message="Usuario sin token registrado")

    push_token = token_doc["push_token"]

    # Enviar via Expo Push API
    payload = {
        "to": push_token,
        "title": body.title,
        "body": body.body,
        "data": body.data,
        "sound": "default",
        "channelId": "default",
    }

    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                EXPO_PUSH_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            result = res.json()
            if res.status_code == 200:
                return NotificationOut(success=True, message="Notificación enviada")
            else:
                return NotificationOut(success=False, message=f"Error Expo: {result}")
    except Exception as exc:
        return NotificationOut(success=False, message=f"Error: {str(exc)}")


@router.post("/send-bulk", response_model=NotificationOut)
async def send_bulk(
    body: dict,
    user: CurrentUser = Depends(get_current_user),
) -> NotificationOut:
    """
    Envía notificaciones a múltiples usuarios.
    body: {"user_ids": [...], "title": "...", "body": "...", "data": {...}}
    """
    user_ids = body.get("user_ids", [])
    title = body.get("title", "")
    message = body.get("body", "")
    data = body.get("data", {})

    if not user_ids:
        return NotificationOut(success=False, message="No hay destinatarios")

    db = get_db()
    tokens = list(db.user_push_tokens.find({"user_id": {"$in": user_ids}}))

    if not tokens:
        return NotificationOut(success=False, message="Sin tokens registrados")

    # Enviar batch via Expo (hasta 100 por request)
    messages = [
        {
            "to": t["push_token"],
            "title": title,
            "body": message,
            "data": data,
            "sound": "default",
            "channelId": "default",
        }
        for t in tokens
    ]

    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                EXPO_PUSH_URL,
                json=messages,
                headers={"Content-Type": "application/json"},
            )
            if res.status_code == 200:
                return NotificationOut(success=True, message=f"Enviadas a {len(messages)} usuarios")
            else:
                return NotificationOut(success=False, message=f"Error Expo: {res.text}")
    except Exception as exc:
        return NotificationOut(success=False, message=f"Error: {str(exc)}")
