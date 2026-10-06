"""Claude Vision — lectura de empaques diecast como respaldo del identify.

Mismo contrato que gemini_vision.read_packaging_with_gemini: devuelve un dict
(brand, model, variant, toy_number, series, color, year, driver, full_text,
confidence) o None. Fail-open: cualquier error/timeout → None y el embed decide.

Se activa con VISION_FALLBACK=claude y ANTHROPIC_API_KEY en el entorno del
servicio (systemd). La clave NUNCA va en el repo ni en logs.
"""

import base64
import hashlib
import io
import json
import os
import re
import time

import requests

_URL = "https://api.anthropic.com/v1/messages"
_MODEL = os.environ.get("CLAUDE_VISION_MODEL", "claude-haiku-4-5-20251001")
_TIMEOUT_S = float(os.environ.get("CLAUDE_VISION_TIMEOUT_S", "4.5"))
_MAX_EDGE = 1280  # px: suficiente para leer «▶ 726» y baja tokens/latencia

_PROMPT = """\
Eres un experto en coleccionables diecast 1:64 (Mini GT, Hot Wheels, Inno64, Tarmac Works, Kaido House, Tomica...). Analiza esta foto de un empaque o coche y responde SOLO con un JSON, sin markdown:

{"brand": "marca del coche real", "model": "modelo", "variant": "color/livery/numero de carrera",
 "toy_number": "numero del empaque si es visible (el de la flecha ▶N o MGT#####), solo digitos",
 "series": "serie si aparece", "color": "color principal", "year": "año si aparece",
 "driver": "piloto si aplica", "full_text": "todo el texto legible del empaque",
 "confidence": "alto/medio/bajo"}

Si un campo no es visible, pon null. No inventes: si no lo lees, null."""

_cache: dict[str, dict | None] = {}
_CACHE_MAX = 50


def _prepare(image_bytes: bytes) -> tuple[str, str]:
    """(base64, media_type) con el lado mayor limitado a _MAX_EDGE."""
    from PIL import Image

    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = im.size
    if max(w, h) > _MAX_EDGE:
        k = _MAX_EDGE / max(w, h)
        im = im.resize((int(w * k), int(h * k)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


def read_packaging_with_claude(image_bytes: bytes) -> dict | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[claude] sin ANTHROPIC_API_KEY — se omite", flush=True)
        return None

    key = hashlib.sha256(image_bytes).hexdigest()[:16]
    if key in _cache:
        print(f"[claude] caché hit ({key})", flush=True)
        return _cache[key]

    t0 = time.time()
    try:
        b64, media = _prepare(image_bytes)
        r = requests.post(
            _URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": _MODEL,
                "max_tokens": 500,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
                            {"type": "text", "text": _PROMPT},
                        ],
                    }
                ],
            },
            timeout=_TIMEOUT_S,
        )
        if r.status_code != 200:
            # Sin cachear fallos. No se imprime el cuerpo completo (puede eco de cabeceras).
            print(f"[claude] HTTP {r.status_code}: {r.text[:120]}", flush=True)
            return None

        blocks = r.json().get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        text = re.sub(r"```(?:json)?", "", text).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            print(f"[claude] sin JSON en respuesta: {text[:120]}", flush=True)
            return None

        result = json.loads(text[start : end + 1])
        print(
            f"[claude] {int((time.time()-t0)*1000)}ms brand={result.get('brand')} "
            f"model={result.get('model')} toy={result.get('toy_number')}",
            flush=True,
        )
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[key] = result
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"[claude] error: {type(exc).__name__}: {exc}", flush=True)
        return None
