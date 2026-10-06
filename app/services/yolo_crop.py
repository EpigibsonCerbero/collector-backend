"""YOLO object detection — recortar el auto del empaque antes del embed.

En fotos de blister, el auto ocupa ~20% de la imagen (el resto es caja,
cartón, fondo). El embedding del encuadre completo promedia todo y no matchea
ningún producto. YOLO detecta el auto y recorta → embeddings mucho más limpios.

Modelo: YOLOv8n (6MB) — detecta 'car', 'motorcycle', etc. del COCO dataset.
"""

from __future__ import annotations

import threading

import numpy as np

_lock = threading.Lock()
_model = None

# Clases COCO de vehículos
_VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def _get_yolo():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from ultralytics import YOLO

                print("[yolo] cargando YOLOv8n...", flush=True)
                _model = YOLO("yolov8n.pt")
                print("[yolo] cargado", flush=True)
    return _model


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

    except Exception as exc:
        print(f"[yolo] error: {type(exc).__name__}: {exc}", flush=True)
        return [image_bytes]
