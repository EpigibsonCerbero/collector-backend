"""
OCR de scene-text con RapidOCR (ONNX) — empaques, blisters, etiquetas.

Por qué no TrOCR: `microsoft/trocr-base-printed` está entrenado en documentos
escaneados (una línea limpia). En fotos de blister devolvía 'RM' / '4'.
RapidOCR (PP-OCRv4 via ONNX) hace detección + reconocimiento de texto en
escenas reales: rápido (~0.5-1.5s CPU) y aguanta reflejos/fuente estilizada.

Uso:
    from app.services.scene_ocr import read_text
    text = read_text(pil_or_ndarray)

Fallback: '' → vision.read_ocr cae a Tesseract.
"""
from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_engine = None
_failed = False


def _ensure():
    global _engine, _failed
    if _engine is not None or _failed:
        return _engine is not None
    with _lock:
        if _engine is not None or _failed:
            return _engine is not None
        try:
            from rapidocr_onnxruntime import RapidOCR

            print("[ocr] cargando RapidOCR (PP-OCRv4 ONNX)…", flush=True)
            t0 = time.time()
            _engine = RapidOCR()
            print(f"[ocr] RapidOCR listo en {time.time()-t0:.1f}s", flush=True)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[ocr] RapidOCR no cargó: {type(exc).__name__}: {exc}", flush=True)
            _failed = True
            return False


def read_text(im) -> str:
    """RapidOCR sobre PIL/ndarray. Concatena las líneas detectadas."""
    if not _ensure():
        return ""
    try:
        import numpy as np
        from PIL import Image

        if isinstance(im, Image.Image):
            # RapidOCR acepta ruta o ndarray BGR/RGB
            arr = np.asarray(im.convert("RGB"))
        else:
            arr = im
        t0 = time.time()
        result, _elapse = _engine(arr)
        # result = [(box, text, score), ...]
        lines = []
        if result:
            for _box, text, score in result:
                t = (text or "").strip()
                if t and score >= 0.5:
                    lines.append(t)
        text = "\n".join(lines)
        print(f"[ocr] RapidOCR {int((time.time()-t0)*1000)}ms → {text[:70]!r}", flush=True)
        return text
    except Exception as exc:  # noqa: BLE001
        print(f"[ocr] RapidOCR infer falló: {type(exc).__name__}: {exc}", flush=True)
        return ""
