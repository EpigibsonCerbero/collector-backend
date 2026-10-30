"""Visión general sobre Claude — herramienta de ADMINISTRACIÓN.

Solo para cuentas en `ADMIN_USER_IDS`. No es un endpoint de IA para la app:
cada llamada cuenta contra el tope de lecturas pagadas y los errores del
proveedor no se devuelven al cliente (no se filtra el stack ni el saldo).

POST /ai/vision          → análisis general (modos general|objeto|texto|catalogo)
GET  /ai/vision/config   → defaults y valores admitidos (solo admin)
GET  /ai/vision/last     → tokens/latencia de la última llamada (solo admin)
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ratelimit import hit_or_429
from app.security import require_admin
from app.services import ai_catalog, claude_vision

router = APIRouter(prefix="/ai", tags=["ai"])

_MAX_IMAGE_BYTES = 12 * 1024 * 1024
_MAX_TIMEOUT_S = 30.0


def _decode_image_b64(raw: str) -> bytes:
    if "," in raw and raw.strip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw, validate=False)
    except Exception:
        raise HTTPException(status_code=400, detail="image_base64 inválido")
    if not data:
        raise HTTPException(status_code=400, detail="Falta la imagen")
    if len(data) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Imagen muy grande (máx 12 MB)")
    return data


class VisionIn(BaseModel):
    """Foto + cómo analizarla. Solo administradores."""

    image_base64: str
    mode: str = Field(default="general", description="general | objeto | texto | catalogo")
    model: Optional[str] = None
    speed: Optional[str] = Field(default=None, description="standard | fast")
    thinking: Optional[str] = Field(default=None, description="off | auto | extended")
    effort: Optional[str] = Field(default=None, description="low | medium | high | xhigh | max")
    timeout_s: Optional[float] = Field(default=None, ge=1, le=_MAX_TIMEOUT_S)


class VisionOut(BaseModel):
    ok: bool = False
    mode: str = "general"
    result: Optional[dict[str, Any]] = None
    usage: Optional[dict[str, Any]] = None
    call: Optional[dict[str, Any]] = None
    detail: Optional[str] = None


def _public_call(info: dict | None) -> dict | None:
    """Info de la llamada SIN el texto crudo del proveedor (puede nombrar
    saldo, claves o el stack). Solo lo operativo: modelo, tokens, latencia."""
    if not info:
        return None
    out = {
        k: info.get(k)
        for k in (
            "purpose", "model", "speed", "thinking", "effort",
            "http_status", "ms", "attempts", "usage", "stop_reason",
            "response_chars", "thinking_blocks", "degraded",
        )
    }
    if info.get("error"):
        out["error"] = "proveedor"
    return out


@router.get("/vision/config")
def vision_config(user=Depends(require_admin)) -> dict:
    """Defaults del entorno y valores admitidos (para un selector interno)."""
    cfg = claude_vision.CallOpts().resolved()
    return {
        "default": cfg,
        "models": [
            "claude-haiku-5-5",
            "claude-sonnet-5-5",
            "claude-opus-5-5",
            "claude-fable-5-1",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-5",
            "claude-opus-5",
        ],
        "speeds": ["standard", "fast"],
        "thinkings": ["off", "auto", "extended"],
        "efforts": [None, "low", "medium", "high", "xhigh", "max"],
        "modes": sorted(claude_vision._MODES.keys()),
        "verbose": claude_vision._VERBOSE,
    }


@router.get("/vision/last")
def vision_last(user=Depends(require_admin)) -> dict:
    """Uso de la última llamada propia (tokens, modelo, latencia). Sin errores
    del proveedor ni datos de otros usuarios."""
    return {"last": _public_call(claude_vision.last_call_info())}


@router.post("/vision", response_model=VisionOut)
async def vision_analyze(
    body: VisionIn,
    user=Depends(require_admin),
) -> VisionOut:
    """Identifica cualquier cosa en la foto con Claude. Solo administradores.

    Cada llamada cuenta contra el tope diario de lecturas pagadas. El proveedor
    nunca se nombra en la respuesta.
    """
    hit_or_429("ai_vision", user.id, limit=30, per_seconds=600)
    data = _decode_image_b64(body.image_base64)
    mode = (body.mode or "general").strip().lower()
    if mode not in claude_vision._MODES:
        raise HTTPException(status_code=400, detail="mode desconocido")
    db = None
    try:
        from app.db import get_db

        db = get_db()
        if not ai_catalog.reserve_ai_call(db, user.id):
            raise HTTPException(status_code=429, detail="Tope de lecturas alcanzado")
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        pass  # sin contadores no se bloquea la herramienta interna
    opts = claude_vision.CallOpts(
        model=body.model,
        speed=body.speed,
        thinking=body.thinking,
        effort=body.effort,
    )
    result = await asyncio.to_thread(
        claude_vision.analyze,
        data,
        mode,
        min(body.timeout_s or _MAX_TIMEOUT_S, _MAX_TIMEOUT_S),
        opts,
    )
    call = _public_call(claude_vision.last_call_info())
    if result is None:
        return VisionOut(
            ok=False,
            mode=mode,
            usage=(call or {}).get("usage"),
            call=call,
            detail="sin respuesta del modelo",
        )
    return VisionOut(
        ok=True,
        mode=mode,
        result=result,
        usage=(call or {}).get("usage"),
        call=call,
    )
