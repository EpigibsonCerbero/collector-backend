"""
Text matching semántico con MiniLM (paraphrase-multilingual-MiniLM-L12-v2).

Reemplaza a difflib para comparar nombres del catálogo vs OCR garbled:
  "Briiant White Pane" ↔ "Brilliant White Pearl"  → sim ~0.85 (difflib: 0.4)

Todo en backend. Fallback: scoring por tokens (como hasta ahora) si falla.
"""
from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_model = None
_failed = False
# cache de embeddings de nombres del catálogo (id/name → vector)
_NAME_CACHE: dict[str, list[float]] = {}


def _ensure():
    global _model, _failed
    if _model is not None or _failed:
        return _model is not None
    with _lock:
        if _model is not None or _failed:
            return _model is not None
        try:
            from sentence_transformers import SentenceTransformer

            print("[minilm] cargando paraphrase-multilingual-MiniLM-L12-v2…", flush=True)
            t0 = time.time()
            _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            print(f"[minilm] listo en {time.time()-t0:.1f}s", flush=True)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[minilm] no cargó: {type(exc).__name__}: {exc}", flush=True)
            _failed = True
            return False


def encode(texts: list[str]) -> list[list[float]]:
    """Embeddings L2-normalizados de una lista de textos."""
    if not _ensure() or not texts:
        return []
    try:
        vecs = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vecs]
    except Exception as exc:  # noqa: BLE001
        print(f"[minilm] encode falló: {exc}", flush=True)
        return []


def similarity(a: str, b: str) -> float:
    """Similitud coseno 0..1 entre dos frases (1 = idénticas semánticamente)."""
    if not a or not b:
        return 0.0
    vecs = encode([a, b])
    if len(vecs) < 2:
        return 0.0
    return float(sum(x * y for x, y in zip(vecs[0], vecs[1])))


def best_name_match(query: str, names: list[str], top_k: int = 5) -> list[tuple[str, float]]:
    """Top-k nombres más parecidos al query (OCR). Devuelve [(name, score), …]."""
    if not query or not names:
        return []
    qv = encode([query])
    if not qv:
        return []
    nvs = encode(names)
    if not nvs:
        return []
    q = qv[0]
    scored = [(n, float(sum(x * y for x, y in zip(q, v)))) for n, v in zip(names, nvs)]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]
