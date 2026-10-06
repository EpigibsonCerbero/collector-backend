"""
OCR con TrOCR (microsoft/trocr-base-printed) — reemplaza a Tesseract en el
hot path del identify. Todo en backend.

Por qué: Tesseract tardaba 5–18s y leía "Briiant White" / "RJ5". TrOCR
printed es ~3-5× más rápido en CPU y mucho más preciso en texto de empaque.

Uso:
    from app.services.trocr_ocr import read_text
    text = read_text(pil_image)   # o ndarray RGB

Fallback: si el modelo no carga, se devuelve "" y `vision.read_ocr` cae a
Tesseract (no romper el identify).
"""
from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_processor = None
_model = None
_device = "cpu"
_failed = False

# techo de píxeles por lado — TrOCR va a 384px de altura internamente,
# un input gigante solo hace lento el resize.
_MAX_EDGE = 1600


def _ensure_model():
    """Carga lazy de TrOCR. CPU en Oracle (Ampere)."""
    global _processor, _model, _device, _failed
    if _model is not None or _failed:
        return _model is not None
    with _lock:
        if _model is not None or _failed:
            return _model is not None
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            print("[trocr] cargando microsoft/trocr-base-printed…", flush=True)
            t0 = time.time()
            _processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            _model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
            _model.eval()
            try:
                import torch

                _device = "cuda" if torch.cuda.is_available() else "cpu"
                _model.to(_device)
            except Exception:  # noqa: BLE001
                _device = "cpu"
            print(
                f"[trocr] listo en {time.time()-t0:.1f}s device={_device}",
                flush=True,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[trocr] no se pudo cargar: {type(exc).__name__}: {exc}", flush=True)
            _failed = True
            return False


def _to_pil(im):
    from PIL import Image

    if isinstance(im, Image.Image):
        return im.convert("RGB")
    # ndarray
    return Image.fromarray(im).convert("RGB")


def read_text(im) -> str:
    """TrOCR sobre una imagen (PIL o ndarray). Devuelve '' si falla."""
    if not _ensure_model():
        return ""
    try:
        import torch

        pil = _to_pil(im)
        w, h = pil.size
        if max(w, h) > _MAX_EDGE:
            s = _MAX_EDGE / max(w, h)
            pil = pil.resize((int(w * s), int(h * s)), 2)  # BILINEAR
        t0 = time.time()
        inputs = _processor(images=pil, return_tensors="pt").to(_device)
        with torch.no_grad():
            out = _model.generate(**inputs, max_new_tokens=64)
        text = _processor.batch_decode(out, skip_special_tokens=True)[0]
        dt = (time.time() - t0) * 1000
        print(f"[trocr] {dt:.0f}ms → {text[:60]!r}", flush=True)
        return text.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[trocr] infer falló: {type(exc).__name__}: {exc}", flush=True)
        return ""


def read_lines(im, crops: list | None = None) -> str:
    """TrOCR sobre varios crops (p.ej. mitad inferior del blister) y concatena.

    El texto de identidad de los empaques Mini GT vive abajo; un solo pase
    a veces se queda solo con el branding.
    """
    parts: list[str] = []
    if crops:
        for c in crops:
            t = read_text(c)
            if t:
                parts.append(t)
    else:
        t = read_text(im)
        if t:
            parts.append(t)
    return "\n".join(parts)
