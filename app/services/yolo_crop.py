"""YOLO object detection — recortar el auto del empaque antes del embed.

En fotos de blister, el auto ocupa ~20% de la imagen (el resto es caja,
cartón, fondo). El embedding del encuadre completo promedia todo y no matchea
ningún producto. YOLO detecta el auto y recorta → embeddings mucho más limpios.

Modelo: YOLOv8n (6MB) — detecta 'car', 'motorcycle', etc. del COCO dataset.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

_lock = threading.Lock()
_model = None

# Clases COCO de vehículos
_VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Pesos en ruta ABSOLUTA: con una ruta relativa ultralytics intenta DESCARGAR
# el modelo (dentro de una petición). Sin fichero no se carga ni se descarga.
_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "data" / "models" / "yolov8n.pt"

# Caché negativa: tras un fallo de import/carga no se reintenta en esta ventana.
_NEG_TTL_S = 600.0
_clock = time.monotonic  # inyectable en pruebas
_failed_until = 0.0


class YoloUnavailable(RuntimeError):
    """YOLO no está disponible (sin pesos, fallo de carga o caché negativa)."""


def _weights_path() -> Path:
    return Path(os.environ.get("YOLO_WEIGHTS") or _DEFAULT_WEIGHTS)


def loaded() -> bool:
    """¿El modelo ya está en memoria? (sin cargarlo)."""
    return _model is not None


def _neg_active() -> bool:
    return _clock() < _failed_until


def _fail(why: str) -> None:
    """Abre la caché negativa; registra UNA línea por ventana."""
    global _failed_until
    first = not _neg_active()
    _failed_until = _clock() + _NEG_TTL_S
    if first:
        print(f"[yolo] no disponible ({why}); sin reintento en {int(_NEG_TTL_S)} s", flush=True)


def _get_yolo():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        if _neg_active():
            raise YoloUnavailable("caché negativa")
        path = _weights_path()
        if not path.is_file():
            _fail("sin fichero de pesos")
            raise YoloUnavailable("sin pesos")
        try:
            from ultralytics import YOLO

            model = YOLO(str(path))
        except Exception as exc:  # noqa: BLE001
            _fail(type(exc).__name__)
            raise YoloUnavailable(type(exc).__name__) from exc
        _model = model
        print("[yolo] cargado", flush=True)
    return _model


# Recarga en segundo plano (tras un fallo de arranque): un solo hilo a la vez
# y como máximo un intento por ventana de `_NEG_TTL_S`.
_bg_lock = threading.Lock()
_bg_running = False
_bg_next_at = 0.0


def _spawn(target) -> threading.Thread:
    t = threading.Thread(target=target, name="yolo-reload", daemon=True)
    t.start()
    return t


def ensure_loading_in_background() -> threading.Thread | None:
    """Sin modelo en memoria y con la caché negativa vencida: lanza UN intento
    de carga en un hilo daemon y vuelve enseguida (nunca carga en el hilo del
    llamador). None si no se lanzó nada."""
    global _bg_running, _bg_next_at
    if _model is not None:
        return None
    with _bg_lock:
        now = _clock()
        if _bg_running or _neg_active() or now < _bg_next_at:
            return None
        _bg_running = True
        _bg_next_at = now + _NEG_TTL_S

    def run() -> None:
        global _bg_running
        try:
            _get_yolo()
        except Exception:  # noqa: BLE001
            pass  # `_fail` ya abrió la caché negativa y lo registró
        finally:
            with _bg_lock:
                _bg_running = False

    try:
        return _spawn(run)
    except Exception:  # noqa: BLE001
        with _bg_lock:
            _bg_running = False
        return None


def warmup() -> None:
    """Carga al arrancar (fuera de cualquier cupo del identify). Nunca lanza."""
    try:
        _get_yolo()
    except Exception:  # noqa: BLE001
        pass


def detect_and_crop_vehicle(image_bytes: bytes) -> list[bytes]:
    """Detecta vehículos en la imagen y recorta cada uno.

    Returns: lista de crops JPEG (uno por vehículo detectado).
    Si no detecta nada, retorna la imagen original como único crop.
    """
    from PIL import Image
    import io

    try:
        model = _get_yolo()
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        results = model.predict(im, conf=0.25, verbose=False)
        if not results or not results[0].boxes:
            return [image_bytes]

        crops = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id not in _VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            # Expandir el box un 15% para no cortar el coche
            w, h = x2 - x1, y2 - y1
            pad_x, pad_y = int(w * 0.15), int(h * 0.15)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(im.width, x2 + pad_x)
            y2 = min(im.height, y2 + pad_y)

            crop = im.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=95)
            crops.append(buf.getvalue())

        if crops:
            print(
                f"[yolo] {len(crops)} vehículo(s) detectado(s), "
                f"clases: {[_VEHICLE_CLASSES.get(int(b.cls[0]), '?') for b in results[0].boxes]}",
                flush=True,
            )
            return crops

        return [image_bytes]

    except YoloUnavailable:
        return [image_bytes]  # ya registrado una vez por ventana
    except Exception as exc:
        print(f"[yolo] error: {type(exc).__name__}: {exc}", flush=True)
        return [image_bytes]


def vehicle_boxes(im) -> list[tuple[float, float, float, float]] | None:
    """Cajas (x0, y0, x1, y1) de vehículos COCO en una imagen PIL.

    None si YOLO no está disponible o falla (el llamador decide sin él);
    lista vacía si no hay ningún vehículo."""
    try:
        model = _get_yolo()
        results = model.predict(im, conf=0.25, verbose=False)
        boxes = []
        for box in (results[0].boxes if results else None) or []:
            if int(box.cls[0]) in _VEHICLE_CLASSES:
                boxes.append(tuple(box.xyxy[0].tolist()))
        return boxes
    except YoloUnavailable:
        return None  # ya registrado una vez por ventana
    except Exception as exc:  # noqa: BLE001
        print(f"[yolo] vehicle_boxes no disponible: {type(exc).__name__}", flush=True)
        return None
