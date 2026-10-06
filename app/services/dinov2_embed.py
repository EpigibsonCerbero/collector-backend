"""
Embeddings visuales con DINOv2 (facebook/dinov2-base) — fine-grained diecast.

Mejor que MobileNetV3 576-d para distinguir variantes de color/livery/edición
(p.ej. GT-R Nismo White vs Red vs Silver). 768-d, L2-normalized.

Contrato de índice (igual que mobile):
    [{id, toy, name, view, vec}]  vec = 768 floats L2-normalized

Archivos:
    backend/data/dinov2-embeddings-full.json
"""
from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_model = None
_processor = None
_failed = False
DIM = 768


def _ensure():
    global _model, _processor, _failed
    if _model is not None or _failed:
        return _model is not None
    with _lock:
        if _model is not None or _failed:
            return _model is not None
        try:
            from transformers import AutoImageProcessor, AutoModel

            # 2 hilos (no los 4 núcleos): el embed normal y el del coche aislado
            # corren EN PARALELO; con 4 hilos cada uno se sobresuscribe la CPU y
            # el driver de Mongo no llega a tiempo → la 1ª consulta tras la visión
            # tardaba 0.2-2.6 s (medido 2026-10-06; con 2 hilos: 2 ms).
            import os

            import torch

            torch.set_num_threads(int(os.environ.get("DINOV2_THREADS", "2")))
            print("[dinov2] cargando facebook/dinov2-base…", flush=True)
            t0 = time.time()
            _processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            _model = AutoModel.from_pretrained("facebook/dinov2-base")
            _model.eval()
            print(f"[dinov2] listo en {time.time()-t0:.1f}s", flush=True)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[dinov2] no cargó: {type(exc).__name__}: {exc}", flush=True)
            _failed = True
            return False


def embed_image(im) -> list[float] | None:
    """DINOv2 CLS embedding (768-d, L2-normalized). None si falla."""
    if not _ensure():
        return None
    try:
        import torch
        from PIL import Image

        if not isinstance(im, Image.Image):
            im = Image.fromarray(im)
        im = im.convert("RGB")
        inputs = _processor(images=im, return_tensors="pt")
        with torch.no_grad():
            out = _model(**inputs)
        # CLS token
        vec = out.last_hidden_state[:, 0, :][0]
        vec = vec / (vec.norm() + 1e-8)
        return [round(float(x), 6) for x in vec.tolist()]
    except Exception as exc:  # noqa: BLE001
        print(f"[dinov2] embed falló: {type(exc).__name__}: {exc}", flush=True)
        return None


def cosine(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))
