"""Upload de imágenes — guarda en disco y sirve URLs públicas."""
from __future__ import annotations

import base64
import hashlib
import os
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.security import CurrentUser, get_current_user

router = APIRouter(prefix="/uploads", tags=["uploads"])

# Directorio donde se guardan las imágenes.
# Prod (Oracle) usa /home/opc/...; en local cae a backend/data/uploads si no hay permiso.
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/home/opc/collector-manager/uploads"))
try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    UPLOAD_DIR = Path(__file__).resolve().parents[2] / "data" / "uploads"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Extensiones permitidas
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


class UploadResponse(BaseModel):
    url: str
    filename: str
    size: int


class Base64Upload(BaseModel):
    image_base64: str
    mime: str = "image/jpeg"
    prefix: str = "img"


_FORMAT_EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}


def _real_image_ext(data: bytes) -> str:
    """Valida que los bytes SEAN una imagen (no basta la extensión del nombre:
    se podía alojar cualquier archivo con nombre .jpg). Devuelve la extensión real."""
    import io

    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            fmt = (im.format or "").upper()
            im.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida")
    ext = _FORMAT_EXT.get(fmt)
    if not ext:
        raise HTTPException(status_code=400, detail="Formato no permitido (JPG, PNG, WEBP o GIF)")
    return ext


def _save_bytes(data: bytes, ext: str, prefix: str) -> tuple[str, int]:
    """Guarda bytes en disco y retorna (filename, size)."""
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Imagen muy grande (máx 10MB)")
    if len(data) < 100:
        raise HTTPException(status_code=400, detail="Imagen inválida o vacía")
    ext = _real_image_ext(data)
    prefix = "".join(ch for ch in prefix if ch.isalnum())[:16] or "img"

    # Nombre único basado en hash + timestamp
    content_hash = hashlib.md5(data).hexdigest()[:12]
    timestamp = int(time.time() * 1000)
    filename = f"{prefix}_{timestamp}_{content_hash}{ext}"
    filepath = UPLOAD_DIR / filename

    filepath.write_bytes(data)
    return filename, len(data)


@router.post("/image", response_model=UploadResponse, status_code=201)
async def upload_image(
    image: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
) -> UploadResponse:
    """Sube una imagen (multipart/form-data) y retorna su URL pública."""
    if not image.filename:
        raise HTTPException(status_code=400, detail="Archivo requerido")

    ext = Path(image.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Extensión no permitida: {ext}. Usa {', '.join(ALLOWED_EXTENSIONS)}",
        )

    data = await image.read()
    filename, size = _save_bytes(data, ext, "upload")

    return UploadResponse(
        url=f"/api/v1/uploads/{filename}",
        filename=filename,
        size=size,
    )


@router.post("/image-base64", response_model=UploadResponse, status_code=201)
async def upload_image_base64(
    body: Base64Upload,
    user: CurrentUser = Depends(get_current_user),
) -> UploadResponse:
    """Sube una imagen en base64 (para capturas de cámara)."""
    try:
        raw = body.image_base64
        if raw.startswith("data:") and "," in raw:
            raw = raw.split(",", 1)[1]
        data = base64.b64decode(raw, validate=False)
    except Exception:
        raise HTTPException(status_code=400, detail="base64 inválido")

    # Detectar extensión desde mime
    mime_ext = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }
    ext = mime_ext.get(body.mime, ".jpg")

    filename, size = _save_bytes(data, ext, body.prefix)

    # Ruta servible tal cual (antes `/uploads/…` daba 404: la ruta real lleva /api/v1).
    return UploadResponse(
        url=f"/api/v1/uploads/{filename}",
        filename=filename,
        size=size,
    )


@router.post("/voice-base64", response_model=UploadResponse, status_code=201)
async def upload_voice_base64(
    body: Base64Upload,
    user: CurrentUser = Depends(get_current_user),
) -> UploadResponse:
    """Sube una nota de voz (m4a/aac/mp3/wav) en base64."""
    try:
        raw = body.image_base64
        if raw.startswith("data:") and "," in raw:
            raw = raw.split(",", 1)[1]
        data = base64.b64decode(raw, validate=False)
    except Exception:
        raise HTTPException(status_code=400, detail="base64 inválido")

    mime_ext = {
        "audio/m4a": ".m4a",
        "audio/mp4": ".m4a",
        "audio/aac": ".aac",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
    }
    ext = mime_ext.get(body.mime, ".m4a")
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Audio muy grande (máx 10MB)")
    if len(data) < 50:
        raise HTTPException(status_code=400, detail="Audio inválido o vacío")

    prefix = "".join(ch for ch in (body.prefix or "voice") if ch.isalnum())[:16] or "voice"
    content_hash = hashlib.md5(data).hexdigest()[:12]
    timestamp = int(time.time() * 1000)
    filename = f"{prefix}_{timestamp}_{content_hash}{ext}"
    filepath = UPLOAD_DIR / filename
    filepath.write_bytes(data)
    return UploadResponse(
        url=f"/api/v1/uploads/{filename}",
        filename=filename,
        size=len(data),
    )


@router.get("/{filename}")
async def get_image(filename: str) -> FileResponse:
    """Sirve una imagen o audio subido."""
    filepath = UPLOAD_DIR / filename
    # Path traversal ANTES de tocar el disco.
    if not filepath.resolve().is_relative_to(UPLOAD_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Acceso denegado")
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    # Inmutables (nombre con hash): el teléfono puede cachearlas para siempre.
    return FileResponse(filepath, headers={"Cache-Control": "public, max-age=31536000, immutable"})
