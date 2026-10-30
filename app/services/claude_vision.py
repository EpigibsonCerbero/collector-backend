"""Claude Vision — lectura de empaques diecast como respaldo del identify.

read_packaging_with_claude es un ADAPTADOR sobre read_for_catalog (una sola
llamada de pago y una sola caché por foto). Mismo contrato que
gemini_vision.read_packaging_with_gemini: devuelve un dict
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
import threading
import time

import requests

_URL = "https://api.anthropic.com/v1/messages"
_MODEL = os.environ.get("CLAUDE_VISION_MODEL", "claude-haiku-4-5-20251001")
_TIMEOUT_S = float(os.environ.get("CLAUDE_VISION_TIMEOUT_S", "4.5"))
_CATALOG_TIMEOUT_S = float(os.environ.get("CLAUDE_CATALOG_TIMEOUT_S", "12"))
_MAX_EDGE = 1280  # px: suficiente para leer «▶ 726» y baja tokens/latencia

_CAT_PROMPT = """\
Eres un experto en coleccionables diecast a escala (Mini GT, Hot Wheels, Inno64, Tarmac Works, Kaido House, Tomica...). Lee esta foto para dar de alta la pieza en un catalogo. Responde SOLO con un JSON, sin markdown ni texto extra:

{"es_diecast": true/false, "brand": "marca del fabricante del juguete", "casting": "modelo del coche real",
 "modelo": "nombre comercial de la pieza", "color": "color principal", "dorsal": "numero de carrera si es visible",
 "livery": "descripcion breve del diseno/decoracion", "texto": "texto legible del empaque o coche",
 "toy_number": "numero del empaque si es visible (MGT##### o el de la flecha), solo si se lee",
 "serie": "serie si aparece impresa", "year": entero si el ano esta impreso en la foto, si no null,
 "confidence": "alto" | "medio" | "bajo"}

es_diecast es false para cualquier cosa que no sea un coche a escala o su empaque (documentos, manos, cuartos, personas, otros objetos).
Cada campo que no se lea en la foto va en null. null si no se lee, nunca inventes. El ano solo si aparece impreso; nunca lo deduzcas."""

_CACHE_MAX = 50
_cat_cache: dict[str, dict] = {}

# --- Corte por errores de saldo / autenticacion -------------------------
_BREAKER_S = float(os.environ.get("CLAUDE_BREAKER_S", "300"))
_breaker_lock = threading.Lock()
_breaker_until = 0.0
_breaker_opened = False


def _now() -> float:
    return time.monotonic()


def _breaker_reset() -> None:
    global _breaker_until, _breaker_opened
    with _breaker_lock:
        _breaker_until = 0.0
        _breaker_opened = False


def breaker_open() -> bool:
    global _breaker_opened
    with _breaker_lock:
        if not _breaker_opened:
            return False
        if _now() >= _breaker_until:
            _breaker_opened = False
            return False
        return True


def _trip_if_fatal(status: int, text: str) -> None:
    """Abre el corte ante saldo agotado (400 credit balance) o clave rechazada."""
    global _breaker_until, _breaker_opened
    fatal = status in (401, 403) or (status == 400 and "credit balance" in (text or "").lower())
    if not fatal:
        return
    with _breaker_lock:
        already = _breaker_opened and _now() < _breaker_until
        _breaker_until = _now() + _BREAKER_S
        _breaker_opened = True
    if not already:
        print(f"[claude] breaker abierto {int(_BREAKER_S)}s tras HTTP {status}", flush=True)


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


_CONF = ("alto", "medio", "bajo")
_logged: set[str] = set()


def _log_once(tag: str, msg: str) -> None:
    if tag in _logged:
        return
    _logged.add(tag)
    print(msg, flush=True)


_STR_FIELDS = ("brand", "casting", "modelo", "color", "dorsal", "livery", "texto", "toy_number", "serie")


def _norm_catalog(raw: dict) -> dict:
    out: dict = {"es_diecast": raw.get("es_diecast") is True}
    for f in _STR_FIELDS:
        v = raw.get(f)
        if isinstance(v, int) and not isinstance(v, bool) and f in ("dorsal", "toy_number"):
            v = str(v)
        out[f] = v.strip() if isinstance(v, str) and v.strip() else None
    y = raw.get("year")
    out["year"] = y if isinstance(y, int) and not isinstance(y, bool) else None
    c = raw.get("confidence")
    out["confidence"] = c if c in _CONF else "bajo"
    return out


def read_for_catalog(image_bytes: bytes, timeout_s: float | None = None) -> dict | None:
    """Lectura de grado catalogo. None sin clave, con el corte abierto, o ante error."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _log_once("nokey", "[claude] sin ANTHROPIC_API_KEY — se omite")
        return None
    if breaker_open():
        return None

    key = "cat:" + hashlib.sha256(image_bytes).hexdigest()[:16]
    if key in _cat_cache:
        return dict(_cat_cache[key])

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
                "max_tokens": 700,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
                            {"type": "text", "text": _CAT_PROMPT},
                        ],
                    }
                ],
            },
            timeout=_CATALOG_TIMEOUT_S if timeout_s is None else timeout_s,
        )
        if r.status_code != 200:
            _trip_if_fatal(r.status_code, r.text)
            print(f"[claude] catalogo HTTP {r.status_code}", flush=True)
            return None
        blocks = r.json().get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        text = re.sub(r"```(?:json)?", "", text).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            _log_once("nojson", "[claude] catalogo: respuesta sin JSON")
            return None
        raw = json.loads(text[start : end + 1])
        if not isinstance(raw, dict):
            _log_once("nojson", "[claude] catalogo: respuesta sin JSON")
            return None
        result = _norm_catalog(raw)
        if len(_cat_cache) >= _CACHE_MAX:
            _cat_cache.clear()
        _cat_cache[key] = result
        return dict(result)
    except Exception as exc:  # noqa: BLE001
        print(f"[claude] catalogo error: {type(exc).__name__}", flush=True)
        return None


def read_packaging_with_claude(image_bytes: bytes) -> dict | None:
    """Adaptador del respaldo del identify sobre la lectura de catalogo.

    Misma llamada y misma caché que read_for_catalog (una sola de pago por foto);
    solo cambia el timeout (presupuesto corto del identify) y el formato de salida.
    """
    read = read_for_catalog(image_bytes, timeout_s=_TIMEOUT_S)
    if not read or not read.get("es_diecast"):
        return None
    variant = read.get("livery") or read.get("color")
    if variant and read.get("dorsal"):
        variant = f"{variant} {read['dorsal']}"
    return {
        "brand": read.get("brand"),
        "model": read.get("modelo") or read.get("casting"),
        "variant": variant,
        "toy_number": read.get("toy_number"),
        "series": read.get("serie"),
        "color": read.get("color"),
        "year": read.get("year"),
        "driver": None,
        "full_text": read.get("texto"),
        "confidence": read.get("confidence"),
    }
