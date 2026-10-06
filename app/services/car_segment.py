"""Segmentación del coche para identify de piezas SUELTAS.

Problema medido (2026-10-01): el embed DINOv2 del query se calculaba sobre el
encuadre COMPLETO. En una foto de celular de un coche suelto, el CLS token
promedia mesa + mano + sombras + desorden y la pieza ocupa una fracción del
vector. El índice, en cambio, son packshots sobre blanco. Resultado: los
sueltos reales casi nunca pasaban de 0.70.

Solución: U²-Net-p (salient object detection, 4.5 MB ONNX, ~0.1-0.2 s CPU a
320 px) → máscara del objeto principal → recorte al bbox de la mayor
componente → pegado sobre blanco de estudio, cuadrado. Misma "firma" que las
fotos del catálogo. El embed del recorte se fusiona con el del encuadre
completo (max por id), así no se degrada lo que ya acierta.

Fail-open: si el modelo no está, no carga o la máscara no es creíble,
devuelve None y el pipeline sigue como antes.

Modelo: backend/data/models/u2netp.onnx
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get("CAR_SEG_MODEL", str(ROOT / "data" / "models" / "u2netp.onnx")))
SIZE = 320

_lock = threading.Lock()
_sess = None
_failed = False


@dataclass
class SegResult:
    image: "object"          # PIL.Image — coche sobre blanco, cuadrado
    area_frac: float         # fracción de la foto que ocupa la máscara
    bbox: tuple[int, int, int, int]
    ms: int


def _get_session():
    global _sess, _failed
    if _sess is not None or _failed:
        return _sess
    with _lock:
        if _sess is not None or _failed:
            return _sess
        try:
            import onnxruntime as ort

            if not MODEL_PATH.exists():
                print(f"[seg] modelo no encontrado: {MODEL_PATH}", flush=True)
                _failed = True
                return None
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            _sess = ort.InferenceSession(str(MODEL_PATH), opts, providers=["CPUExecutionProvider"])
            print(f"[seg] U2Net-p cargado ({MODEL_PATH.name})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[seg] no cargó: {type(exc).__name__}: {exc}", flush=True)
            _failed = True
    return _sess


def saliency_mask(im) -> np.ndarray | None:
    """Máscara 0..1 (float32) del objeto saliente, al tamaño de `im`."""
    from PIL import Image

    sess = _get_session()
    if sess is None:
        return None
    w, h = im.size
    x = np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    x = x / max(float(x.max()), 1e-6)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    mn, mx = float(out.min()), float(out.max())
    out = (out - mn) / (mx - mn + 1e-8)
    m = Image.fromarray((out * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(m, dtype=np.float32) / 255.0


def _largest_component_bbox(binary: np.ndarray) -> tuple[int, int, int, int, int] | None:
    """bbox (x0,y0,x1,y1,area) de la mayor componente conexa (8-vecindad).

    Corre sobre una máscara reducida (≤128 px) → milisegundos sin scipy.
    """
    h, w = binary.shape
    lab = np.zeros((h, w), dtype=np.int32)
    best = None
    cur = 0
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or lab[y, x]:
                continue
            cur += 1
            stack = [(y, x)]
            lab[y, x] = cur
            x0 = x1 = x
            y0 = y1 = y
            n = 0
            while stack:
                cy, cx = stack.pop()
                n += 1
                x0, x1, y0, y1 = min(x0, cx), max(x1, cx), min(y0, cy), max(y1, cy)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not lab[ny, nx]:
                            lab[ny, nx] = cur
                            stack.append((ny, nx))
            if best is None or n > best[4]:
                best = (x0, y0, x1 + 1, y1 + 1, n)
    return best


def isolate_car(im, min_area: float = 0.04, max_area: float = 0.80) -> SegResult | None:
    """Foto → coche recortado sobre blanco de estudio, cuadrado. None si no es creíble.

    - `min_area`: por debajo, la máscara es ruido (o el coche es diminuto).
    - `max_area`: por encima, el "objeto" es toda la foto (packshot que ya es
      estudio, o un blister que llena el encuadre) → el recorte no aporta.
    """
    from PIL import Image, ImageFilter

    t0 = time.time()
    try:
        im = im.convert("RGB")
        mask = saliency_mask(im)
        if mask is None:
            return None
        w, h = im.size
        # componente principal sobre máscara reducida
        s = 128 / max(w, h)
        sw, sh = max(1, int(w * s)), max(1, int(h * s))
        small = np.asarray(
            Image.fromarray((mask * 255).astype(np.uint8)).resize((sw, sh), Image.Resampling.BILINEAR)
        ) > 127
        comp = _largest_component_bbox(small)
        if comp is None:
            return None
        cx0, cy0, cx1, cy1, n = comp
        area_frac = n / float(sw * sh)
        if not (min_area <= area_frac <= max_area):
            return None
        x0, y0 = int(cx0 / s), int(cy0 / s)
        x1, y1 = min(w, int(cx1 / s) + 1), min(h, int(cy1 / s) + 1)
        bw, bh = x1 - x0, y1 - y0
        pad = int(max(bw, bh) * 0.06)
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)

        car = im.crop((x0, y0, x1, y1))
        alpha = Image.fromarray((mask * 255).astype(np.uint8)).crop((x0, y0, x1, y1))
        # bordes suaves: sin halo duro que DINOv2 lea como contorno extra
        alpha = alpha.filter(ImageFilter.GaussianBlur(1.2))
        side = int(max(car.size) * 1.12)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        off = ((side - car.width) // 2, (side - car.height) // 2)
        canvas.paste(car, off, alpha)
        return SegResult(
            image=canvas,
            area_frac=round(area_frac, 3),
            bbox=(x0, y0, x1, y1),
            ms=int((time.time() - t0) * 1000),
        )
    except Exception as exc:  # noqa: BLE001 — fail-open
        print(f"[seg] isolate_car falló: {type(exc).__name__}: {exc}", flush=True)
        return None


def warmup() -> None:
    from PIL import Image

    if _get_session() is not None:
        isolate_car(Image.new("RGB", (64, 64), (200, 200, 200)))
