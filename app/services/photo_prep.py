"""
Preprocesado de query para identify de piezas sueltas.

Problema: el índice se construyó con STUDIO SHOTS (fondo blanco/liso) pero el
usuario manda fotos de celular (mesa, sombras, ángulo, calidad variable).
DINOv2 caía de 100% (studio) a ~60% (foto real).

Solución "gauntlet":
  1) Recorte al auto (bbox cromática / centro)
  2) Fondo blanco tipo estudio (el auto "flota" como en el catálogo)
  3) Autocontrast + square pad — misma firma que las fotos del índice

Todo en backend, ~50ms.
"""
from __future__ import annotations


def _keep_pixel(r: int, g: int, b: int) -> bool:
    """Auto (croma) o muy oscuro — ignora fondo claro de mesa/estudio."""
    mx, mn = max(r, g, b), min(r, g, b)
    if (mx - mn) >= 18:
        return True
    return ((r + g + b) / 3.0) < 80


def _car_bbox(im, grid: int = 64):
    w, h = im.size
    small = im.resize((grid, grid))
    px = small.load()
    xs, ys = [], []
    for y in range(grid):
        for x in range(grid):
            if _keep_pixel(*px[x, y]):
                xs.append(x)
                ys.append(y)
    if not xs:
        cw, ch = int(w * 0.6), int(h * 0.6)
        return (w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad_x = max(1, int((x1 - x0) * 0.1))
    pad_y = max(1, int((y1 - y0) * 0.1))
    sx, sy = w / grid, h / grid
    return (
        max(0, int((x0 - pad_x) * sx)),
        max(0, int((y0 - pad_y) * sy)),
        min(w, int((x1 + 1 + pad_x) * sx)),
        min(h, int((y1 + 1 + pad_y) * sy)),
    )


def studioify(im):
    """Foto de celular → crop al auto + fondo blanco de estudio + square.

    Igual que `gen_minigt_visual_features._auto_bbox` pero para el QUERY.
    """
    from PIL import Image, ImageOps

    im = im.convert("RGB")
    box = _car_bbox(im)
    car = im.crop(box)
    # fondo blanco estudio
    s = max(car.size)
    if s < 200:
        scale = 220 / s
        car = car.resize((int(car.width * scale), int(car.height * scale)), Image.Resampling.LANCZOS)
        s = max(car.size)
    canvas = Image.new("RGB", (s, s), (255, 255, 255))
    canvas.paste(car, ((s - car.width) // 2, (s - car.height) // 2))
    canvas = ImageOps.autocontrast(canvas, cutoff=2)
    return canvas
