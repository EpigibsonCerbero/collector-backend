"""
Identificación de piezas por foto — el cliente manda la imagen y recibe la pieza.

Toda la lógica vive en el servidor (barcode + embeddings + OCR + decisión);
la app solo sube la foto. Respuesta alineada con `PieceIdentificationResult`
del cliente para que el cambio sea transparente.
"""
from __future__ import annotations

import base64
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.db import as_id, get_db, oid, serialize, serialize_many
from app.services.vision import VisionMatch, identify_image

router = APIRouter(tags=["identify"])


# ── Schemas ─────────────────────────────────────────────────────────────────


class IdentifySuggestion(BaseModel):
    id: str
    name: Optional[str] = None
    toy_number: Optional[str] = None
    score: float = 0.0


class IdentifyOut(BaseModel):
    """Espejo de PieceIdentificationResult del cliente RN."""

    method: str = "m"  # 'b' barcode | 'o' ocr/texto | 'c' visual | 'm' manual
    confidence: float = 0.0
    needs_manual: bool = True
    upc: Optional[str] = None
    toy_number: Optional[str] = None
    item: Optional[dict[str, Any]] = None  # ItemVariantOut serializado
    suggestions: list[IdentifySuggestion] = Field(default_factory=list)
    timings_ms: dict[str, int] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)


class IdentifyBase64(BaseModel):
    """Alternativa JSON con la foto en base64 (sin multipart)."""

    image_base64: str
    mime: str = "image/jpeg"


# ── Resolución contra Mongo ─────────────────────────────────────────────────


def _variant_out(doc: dict, master: dict | None = None) -> dict[str, Any]:
    data = serialize(doc) or {}
    if master is not None:
        data["item_master"] = serialize(master)
    return data


def _find_variant_by_id(db, vid: str) -> tuple[dict | None, dict | None]:
    """Busca la variante por _id (uuid string u ObjectId)."""
    for cand in (vid, None):
        try:
            query = {"_id": as_id(vid)} if cand is not None else {"_id": vid}
            doc = db.variants.find_one(query)
        except Exception:
            doc = db.variants.find_one({"_id": vid})
        if doc:
            master = None
            mid = doc.get("item_master_id")
            if mid:
                try:
                    master = db.items.find_one({"_id": as_id(mid)})
                except Exception:
                    master = db.items.find_one({"_id": mid})
            return doc, master
    return None, None


def _find_variant_by_upc(db, upc: str) -> tuple[dict | None, dict | None]:
    doc = db.variants.find_one({"sku_upc": upc})
    if not doc:
        return None, None
    master = None
    mid = doc.get("item_master_id")
    if mid:
        try:
            master = db.items.find_one({"_id": as_id(mid)})
        except Exception:
            master = db.items.find_one({"_id": mid})
    return doc, master


def _find_variant_by_toy(db, toy: str) -> tuple[dict | None, dict | None]:
    norm = toy.upper().replace(" ", "")
    doc = db.variants.find_one({"toy_number": norm})
    if not doc:
        # toy_number con espacios / distinta capitalización
        doc = db.variants.find_one({"toy_number": {"$regex": f"^{norm}$", "$options": "i"}})
    if not doc:
        return None, None
    master = None
    mid = doc.get("item_master_id")
    if mid:
        try:
            master = db.items.find_one({"_id": as_id(mid)})
        except Exception:
            master = db.items.find_one({"_id": mid})
    return doc, master


def _to_suggestions(matches: list[VisionMatch]) -> list[IdentifySuggestion]:
    out: list[IdentifySuggestion] = []
    for m in matches[:5]:
        out.append(
            IdentifySuggestion(id=m.id, name=m.name, toy_number=m.toy, score=round(m.score, 4))
        )
    return out


# ── Endpoint ────────────────────────────────────────────────────────────────


def _log_decision(out: IdentifyOut, res) -> None:
    """Traza compacta de la decisión — para depurar en /tmp/uvicorn-collector.log."""
    name = (out.item or {}).get("name") or (out.item or {}).get("variant_name") or "-"
    top = res.matches[0] if res.matches else None
    second = res.matches[1] if len(res.matches) > 1 else None
    top_s = f"{top.score:.3f}" if top else "0"
    top_n = (top.name[:40] if top else "-")
    sec_s = f"{second.score:.3f}" if second else "0"
    ocr = (res.ocr_text or "")[:80]
    print(
        f"[identify] method={out.method} conf={out.confidence:.3f} manual={out.needs_manual} "
        f"item={name!r} toy={out.toy_number or '-'} upc={out.upc or '-'} | "
        f"top1={top_s} {top_n!r} top2={sec_s} | ocr={ocr!r} | ms={res.timings_ms}",
        flush=True,
    )


def _run_identify(image_bytes: bytes) -> IdentifyOut:
    if not image_bytes or len(image_bytes) < 100:
        raise HTTPException(status_code=400, detail="Imagen vacía o inválida")

    res = identify_image(image_bytes)
    db = get_db()

    out = IdentifyOut(
        method="m",
        confidence=res.confidence,
        needs_manual=res.needs_manual,
        upc=res.upc,
        toy_number=res.toy_number,
        suggestions=_to_suggestions(res.matches),
        timings_ms=res.timings_ms,
        debug={"ocr_preview": res.ocr_text[:300]},
    )

    # 1) UPC exacto
    upc_found_but_unmatched = False
    if res.upc:
        doc, master = _find_variant_by_upc(db, res.upc)
        if doc:
            out.method = "b"
            out.item = _variant_out(doc, master)
            out.needs_manual = False
            out.confidence = 0.99
            _log_decision(out, res)
            return out
        upc_found_but_unmatched = True

    # 2) toy_number exacto
    if res.toy_number:
        doc, master = _find_variant_by_toy(db, res.toy_number)
        if doc:
            out.method = "o"
            out.item = _variant_out(doc, master)
            out.needs_manual = False
            out.confidence = 0.95
            _log_decision(out, res)
            return out

    # Si hubo barcode pero no matcheó en DB, no confiar en el embedding:
    # una imagen de código de barras (mayormente blanca) matchea basura.
    if upc_found_but_unmatched:
        out.method = "m"
        out.needs_manual = True
        out.confidence = 0.5
        _log_decision(out, res)
        return out

    # 3) Embedding ganador (vision.py ya desambigua liveries con OCR)
    if not res.needs_manual and res.matches:
        top = res.matches[0]
        doc, master = _find_variant_by_id(db, top.id)
        if doc:
            out.method = "c"
            out.item = _variant_out(doc, master)
            out.needs_manual = False
            out.confidence = res.confidence
            _log_decision(out, res)
            return out

    # Sin match confiable → el cliente muestra sugerencias (sin inventar).
    out.method = "m"
    out.needs_manual = True
    _log_decision(out, res)
    return out


@router.post("/identify", response_model=IdentifyOut)
async def identify_upload(
    image: Optional[UploadFile] = File(default=None),
    image_base64: Optional[str] = Form(default=None),
) -> IdentifyOut:
    """Sube la foto (multipart `image` o form `image_base64`) y recibe la pieza."""
    data: bytes | None = None
    if image is not None:
        data = await image.read()
    elif image_base64:
        raw = image_base64
        if "," in raw and raw.strip().startswith("data:"):
            raw = raw.split(",", 1)[1]
        try:
            data = base64.b64decode(raw)
        except Exception:
            raise HTTPException(status_code=400, detail="image_base64 inválido")

    if not data:
        raise HTTPException(status_code=400, detail="Falta la imagen (image o image_base64)")

    return _run_identify(data)


@router.post("/identify/base64", response_model=IdentifyOut)
async def identify_b64(body: IdentifyBase64) -> IdentifyOut:
    """Alternativa JSON: { image_base64: '...' }."""
    raw = body.image_base64
    if "," in raw and raw.strip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="image_base64 inválido")
    return _run_identify(data)
