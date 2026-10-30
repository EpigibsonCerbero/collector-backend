"""Claude Vision — cliente general de visión sobre la Messages API.

Identifica CUALQUIER cosa en una foto. La app CURIO usa el modo `catalogo`
(diecast); el resto de modos sirven para cualquier objeto, documento, escena
o texto. Todo pasa por `_call`, que registra modelo, velocidad, pensamiento,
esfuerzo, latencia y tokens de cada llamada.

Se activa con ANTHROPIC_API_KEY en el entorno del servicio (systemd). La
clave NUNCA va en el repo ni en logs.

Selección por petición (`CallOpts`) o por entorno:
  CLAUDE_VISION_MODEL     modelo (claude-haiku-5-5, claude-sonnet-5-5, …)
  CLAUDE_VISION_SPEED     standard | fast
  CLAUDE_VISION_THINKING  off | auto | extended
  CLAUDE_VISION_EFFORT    (vacío) | low | medium | high | xhigh | max
  CLAUDE_VERBOSE          1 (por defecto) = log completo de cada llamada
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
import time
from dataclasses import dataclass

import requests

_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = os.environ.get("CLAUDE_ANTHROPIC_VERSION", "2023-06-01")
_MAX_EDGE = 1280  # px: suficiente para leer texto fino y baja tokens/latencia

# ── Selección de modelo / velocidad / pensamiento ───────────────────────────
_MODEL = os.environ.get("CLAUDE_VISION_MODEL", "claude-haiku-4-5-20251001")
_SPEED = os.environ.get("CLAUDE_VISION_SPEED", "standard").strip().lower()
_THINKING = os.environ.get("CLAUDE_VISION_THINKING", "off").strip().lower()
_EFFORT = os.environ.get("CLAUDE_VISION_EFFORT", "").strip().lower()
_BUDGET_TOKENS = int(os.environ.get("CLAUDE_VISION_THINKING_BUDGET", "2048"))
_VERBOSE = os.environ.get("CLAUDE_VERBOSE", "1").strip() not in ("0", "false", "no", "")

_TIMEOUT_S = float(os.environ.get("CLAUDE_VISION_TIMEOUT_S", "4.5"))
_CATALOG_TIMEOUT_S = float(os.environ.get("CLAUDE_CATALOG_TIMEOUT_S", "12"))
_GENERIC_TIMEOUT_S = float(os.environ.get("CLAUDE_GENERIC_TIMEOUT_S", "20"))

_SPEEDS = {"standard", "fast"}
_THINKINGS = {"off", "auto", "extended"}
_EFFORTS = {"", "low", "medium", "high", "xhigh", "max"}

# Última llamada (para debug del endpoint): tokens, modelo, latencia.
_last_lock = threading.Lock()
_last_call: dict | None = None


@dataclass
class CallOpts:
    """Overrides por petición. Cualquier campo None usa el default del entorno."""

    model: str | None = None
    speed: str | None = None
    thinking: str | None = None
    effort: str | None = None

    def resolved(self) -> dict:
        model = (self.model or _MODEL).strip()
        speed = (self.speed or _SPEED).strip().lower()
        thinking = (self.thinking or _THINKING).strip().lower()
        effort = (self.effort if self.effort is not None else _EFFORT).strip().lower()
        if speed not in _SPEEDS:
            speed = "standard"
        if thinking not in _THINKINGS:
            thinking = "off"
        if effort not in _EFFORTS:
            effort = ""
        return {"model": model, "speed": speed, "thinking": thinking, "effort": effort}


def last_call_info() -> dict | None:
    """Resumen de la última llamada (tokens, modelo, ms) para debug."""
    with _last_lock:
        return dict(_last_call) if _last_call else None


# ── Modos ──────────────────────────────────────────────────────────────────
# Cada modo = prompt + max_tokens + parser. CURIO usa `catalogo` (diecast);
# los demás identifican cualquier cosa.

_PROMPT_GENERAL = """\
Describe e identifica qué hay en esta foto con precisión y sin inventar.
Responde SOLO con un JSON, sin markdown ni texto extra:

{"tipo": "objeto|escena|documento|persona|animal|vehiculo|texto|otro",
 "titulo": "nombre corto y concreto de lo que se ve",
 "descripcion": "2-4 frases con lo esencial",
 "marca": "marca visible si la hay, si no null",
 "modelo": "modelo o nombre de producto si se lee, si no null",
 "texto": "texto legible en la imagen, si no null",
 "colores": ["color1", "color2"],
 "detalles": {"clave": "valor útil"},
 "confidence": "alto" | "medio" | "bajo"}

Cada campo que no se pueda leer va en null o []. Nunca inventes datos."""

_PROMPT_OBJETO = """\
Identifica este objeto concreto. Responde SOLO con un JSON:
{"categoria": "...", "nombre": "...", "marca": "...", "modelo": "...",
 "material": "...", "color": "...", "estado": "...", "texto": "...",
 "identificador": "SKU, número de parte, código… si se lee, si no null",
 "confidence": "alto"|"medio"|"bajo"}
null si no se lee; nunca inventes."""

_PROMPT_TEXTO = """\
Extrae TODO el texto legible de esta imagen, respetando el orden.
Responde SOLO con un JSON: {"texto": "…", "lineas": ["…"], "idioma": "…", "confidence": "alto"|"medio"|"bajo"}
No resumas; transcribe. Si no hay texto: {"texto": null, "lineas": [], "confidence": "alto"}"""

_PROMPT_CATALOGO = """\
Eres un experto en coleccionables diecast a escala (Mini GT, Hot Wheels, Inno64, Tarmac Works, Kaido House, Tomica...). Lee esta foto para dar de alta la pieza en un catalogo. Responde SOLO con un JSON, sin markdown ni texto extra:

{"es_diecast": true/false, "es_foto_real": true/false, "brand": "marca del fabricante del juguete", "casting": "modelo del coche real",
 "modelo": "nombre comercial de la pieza", "color": "color principal", "dorsal": "numero de carrera si es visible",
 "livery": "descripcion breve del diseno/decoracion", "texto": "texto legible del empaque o coche",
 "toy_number": "numero del empaque si es visible (MGT##### o el de la flecha), solo si se lee",
 "serie": "serie si aparece impresa", "year": entero si el ano esta impreso en la foto, si no null,
 "confidence": "alto" | "medio" | "bajo"}

es_diecast es false para cualquier cosa que no sea un coche a escala o su empaque (documentos, manos, cuartos, personas, otros objetos).
es_foto_real es false si la imagen es un render, una ilustracion, una foto generada por IA, un recorte de videojuego o un coche de verdad a escala real (no un juguete 1:64). Solo true si se ve un juguete fisico o su empaque.
Cada campo que no se lea en la foto va en null. null si no se lee, nunca inventes. El ano solo si aparece impreso; nunca lo deduzcas del modelo del coche. brand, casting y serie solo si se leen o se reconocen sin duda; si dudas, null."""

_MODES = {
    "general": {"prompt": _PROMPT_GENERAL, "max_tokens": 900, "timeout": _GENERIC_TIMEOUT_S},
    "objeto": {"prompt": _PROMPT_OBJETO, "max_tokens": 700, "timeout": _GENERIC_TIMEOUT_S},
    "texto": {"prompt": _PROMPT_TEXTO, "max_tokens": 1200, "timeout": _GENERIC_TIMEOUT_S},
    "catalogo": {"prompt": _PROMPT_CATALOGO, "max_tokens": 700, "timeout": _CATALOG_TIMEOUT_S},
}

DEFAULT_MODE = "general"

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


# Sobrecarga / límite de ritmo (429, 529): corte breve, no el largo de saldo.
_OVERLOAD_BREAKER_S = 60.0


def _trip_if_fatal(status: int, text: str) -> None:
    """Abre el corte ante saldo agotado (400 credit balance) o clave rechazada
    (`_BREAKER_S`), y brevemente ante 429/529 (`_OVERLOAD_BREAKER_S`)."""
    global _breaker_until, _breaker_opened
    fatal = status in (401, 403) or (status == 400 and "credit balance" in (text or "").lower())
    if fatal:
        window = _BREAKER_S
    elif status in (429, 529):
        window = _OVERLOAD_BREAKER_S
    else:
        return
    with _breaker_lock:
        already = _breaker_opened and _now() < _breaker_until
        # Nunca acorta un corte más largo ya abierto.
        _breaker_until = max(_breaker_until if already else 0.0, _now() + window)
        _breaker_opened = True
    if not already:
        print(f"[claude] breaker abierto {int(window)}s tras HTTP {status}", flush=True)


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


_logged: set[str] = set()


def _log_once(tag: str, msg: str) -> None:
    if tag in _logged:
        return
    _logged.add(tag)
    print(msg, flush=True)


def _build_request_body(opts: dict, media: str, b64: str, prompt: str, max_tokens: int) -> dict:
    """Cuerpo del POST /v1/messages con modelo, velocidad y pensamiento."""
    body: dict = {
        "model": opts["model"],
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    if opts["speed"] == "fast":
        body["speed"] = "fast"
    thinking = opts["thinking"]
    # Claude 5.x no acepta thinking.enabled+budget (400); solo adaptive o disabled.
    if thinking == "off":
        body["thinking"] = {"type": "disabled"}
    elif thinking == "auto":
        body["thinking"] = {"type": "adaptive"}
    elif thinking == "extended":
        body["thinking"] = {"type": "enabled", "budget_tokens": max(1024, _BUDGET_TOKENS)}
    if opts["effort"]:
        body["output_config"] = {"effort": opts["effort"]}
    return body


def _headers(api_key: str, speed: str) -> dict:
    h = {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    # Algunos modelos requieren el beta header para speed=fast.
    if speed == "fast":
        h["anthropic-beta"] = "fast-mode-2026-02-01"
    return h


def _log_verbose(msg: str) -> None:
    if _VERBOSE:
        print(msg, flush=True)


def _record_call(info: dict) -> None:
    global _last_call
    with _last_lock:
        _last_call = info


def _usage_from(resp_json: dict) -> dict:
    u = resp_json.get("usage") or {}
    out_details = u.get("output_tokens_details") or {}
    return {
        "input_tokens": u.get("input_tokens"),
        "output_tokens": u.get("output_tokens"),
        "thinking_tokens": out_details.get("thinking_tokens"),
        "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": u.get("cache_read_input_tokens"),
        "service_tier": u.get("service_tier"),
        "speed": u.get("speed"),
        "inference_geo": u.get("inference_geo"),
    }


def _strip_offending_style(body: dict, err_text: str) -> dict | None:
    """Quita el parámetro de estilo que la API rechazó y devuelve el cuerpo
    limpio, o None si el error no es de estilo (saldo, clave, formato)."""
    low = (err_text or "").lower()
    if "does not support the `speed`" in low or ("speed" in low and "not support" in low):
        body.pop("speed", None)
        return body
    # fast mode sin cuota en el plan: degrada a standard en vez de fallar.
    if "fast mode" in low and ("rate limit" in low or "429" in low):
        body.pop("speed", None)
        return body
    if "thinking" in low and ("not support" in low or "invalid" in low or "400" in low):
        body.pop("thinking", None)
        return body
    if "output_config" in low or ("effort" in low and "not support" in low):
        body.pop("output_config", None)
        return body
    return None


# Claves de estilo que un modelo rechazó con un 400: no se vuelven a enviar a
# ese modelo durante `_STYLE_OMIT_TTL_S` (ahorra un viaje por llamada). Solo se
# aprende de un 400 de estilo; nunca se cambia lo que la API acepta.
_STYLE_KEYS = ("speed", "thinking", "output_config")
_STYLE_OMIT_TTL_S = float(os.environ.get("CLAUDE_STYLE_OMIT_TTL_S", "3600"))
_style_lock = threading.Lock()
_style_omit: dict[str, tuple[frozenset[str], float]] = {}


def _learned_omit(model: str) -> frozenset[str]:
    with _style_lock:
        entry = _style_omit.get(model)
        if entry is None:
            return frozenset()
        keys, until = entry
        if _now() >= until:
            _style_omit.pop(model, None)
            return frozenset()
        return keys


def _learn_omit(model: str, keys: set[str]) -> None:
    if not keys:
        return
    with _style_lock:
        prev = _style_omit.get(model)
        old = prev[0] if prev and _now() < prev[1] else frozenset()
        _style_omit[model] = (old | frozenset(keys), _now() + _STYLE_OMIT_TTL_S)


# Intentos por llamada (el 1º + un reintento sin el estilo rechazado) y tiempo
# mínimo que debe quedar hasta el `deadline` para lanzar una petición.
_MAX_ATTEMPTS = int(os.environ.get("CLAUDE_MAX_ATTEMPTS", "2"))
_MIN_RETRY_S = float(os.environ.get("CLAUDE_MIN_RETRY_S", "1.0"))


def _call(
    image_bytes: bytes,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
    opts: CallOpts | None,
    purpose: str,
    deadline: float | None = None,
) -> tuple[dict | None, dict | None]:
    """POST a /v1/messages. Devuelve (json_respuesta, info_llamada). Fail-open: (None, info).

    Si la API rechaza un parámetro de estilo (speed/thinking/effort) por el
    modelo elegido, se reintenta sin ese parámetro (máx. `_MAX_ATTEMPTS`
    peticiones): el backend degrada en vez de fallar.

    `deadline` (instante de `_now()`, opcional): cada petición usa
    `min(timeout_s, restante)` y no se lanza ninguna con menos de
    `_MIN_RETRY_S` restantes.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _log_once("nokey", "[claude] sin ANTHROPIC_API_KEY — se omite")
        return None, None
    if breaker_open():
        _log_verbose(f"[claude] {purpose}: corte abierto → se omite")
        return None, None

    cfg = (opts or CallOpts()).resolved()
    t0 = time.perf_counter()
    info: dict = {
        "purpose": purpose,
        "model": cfg["model"],
        "speed": cfg["speed"],
        "thinking": cfg["thinking"],
        "effort": cfg["effort"] or None,
        "timeout_s": timeout_s,
    }
    try:
        b64, media = _prepare(image_bytes)
        body = _build_request_body(cfg, media, b64, prompt, max_tokens)
        omitted = sorted(k for k in _learned_omit(cfg["model"]) if k in body)
        for k in omitted:
            body.pop(k, None)
        if omitted:
            info["omitted"] = omitted
        info["image_bytes"] = len(image_bytes)
        info["image_b64_len"] = len(b64)
        info["prompt_chars"] = len(prompt)
        info["max_tokens"] = max_tokens
        _log_verbose(
            f"[claude] → {purpose} model={cfg['model']} speed={cfg['speed']} "
            f"thinking={cfg['thinking']} effort={cfg['effort'] or '-'} "
            f"max_tokens={max_tokens} img={len(image_bytes)}B prompt={len(prompt)}ch "
            f"timeout={timeout_s}s"
        )
        def _left() -> float | None:
            return None if deadline is None else deadline - _now()

        left = _left()
        if left is not None and left < _MIN_RETRY_S:
            info["ms"] = int((time.perf_counter() - t0) * 1000)
            info["error"] = "deadline"
            _log_verbose(f"[claude] {purpose}: sin tiempo ({left:.1f}s) → se omite")
            _record_call(info)
            return None, info
        attempts = 0
        while True:
            attempts += 1
            left = _left()
            req_timeout = timeout_s if left is None else min(timeout_s, left)
            r = requests.post(
                _URL,
                headers=_headers(api_key, cfg["speed"] if "speed" in body else "standard"),
                json=dict(body),
                timeout=req_timeout,
            )
            left = _left()
            can_retry = attempts < _MAX_ATTEMPTS and (left is None or left >= _MIN_RETRY_S)
            if r.status_code in (400, 429):
                before = {k for k in _STYLE_KEYS if k in body}
                stripped = _strip_offending_style(dict(body), r.text)
                if stripped is not None and r.status_code == 400:
                    # Se aprende aunque no quede tiempo para reintentar.
                    _learn_omit(cfg["model"], before - set(stripped.keys()))
                if stripped is not None and can_retry:
                    info["degraded"] = sorted(
                        {"speed", "thinking", "output_config"} - set(stripped.keys())
                    )
                    # Sin el cuerpo de la respuesta del proveedor en el log.
                    _log_verbose(
                        f"[claude] ↻ {purpose} reintento sin {info['degraded']} "
                        f"(HTTP {r.status_code}, body={len(r.text or '')}ch)"
                    )
                    body = stripped
                    continue
            break
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        info["http_status"] = r.status_code
        info["ms"] = elapsed_ms
        info["attempts"] = attempts
        if r.status_code != 200:
            _trip_if_fatal(r.status_code, r.text)
            info["error"] = r.text[:300]
            _log_verbose(f"[claude] ← {purpose} HTTP {r.status_code} {elapsed_ms}ms body={r.text[:200]}")
            _record_call(info)
            return None, info

        data = r.json()
        usage = _usage_from(data)
        info["usage"] = usage
        info["stop_reason"] = data.get("stop_reason")
        info["id"] = data.get("id")
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        info["response_chars"] = len(text)
        thinking_blocks = [b for b in blocks if b.get("type") == "thinking"]
        info["thinking_blocks"] = len(thinking_blocks)
        _log_verbose(
            f"[claude] ← {purpose} OK {elapsed_ms}ms stop={data.get('stop_reason')} "
            f"in={usage.get('input_tokens')} out={usage.get('output_tokens')} "
            f"think={usage.get('thinking_tokens')} cache_r={usage.get('cache_read_input_tokens')} "
            f"cache_w={usage.get('cache_creation_input_tokens')} "
            f"tier={usage.get('service_tier')} speed={usage.get('speed')} "
            f"text={len(text)}ch think_blocks={len(thinking_blocks)}"
        )
        _record_call(info)
        return data, info
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        info["ms"] = elapsed_ms
        info["error"] = f"{type(exc).__name__}: {exc}"
        _log_verbose(f"[claude] ← {purpose} ERROR {type(exc).__name__} {elapsed_ms}ms")
        _record_call(info)
        return None, info


def _extract_json(data: dict) -> dict | None:
    blocks = data.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        raw = json.loads(text[start : end + 1])
    except Exception:  # noqa: BLE001
        return None
    return raw if isinstance(raw, dict) else None


def analyze(
    image_bytes: bytes,
    mode: str = DEFAULT_MODE,
    timeout_s: float | None = None,
    opts: CallOpts | None = None,
    deadline: float | None = None,
) -> dict | None:
    """Identifica lo que hay en la foto con el modo dado. None si falla.

    Modos: general | objeto | texto | catalogo. Devuelve el JSON parseado
    del modelo (fail-open: None ante error o respuesta ilegible).
    """
    spec = _MODES.get(mode) or _MODES[DEFAULT_MODE]
    cfg = (opts or CallOpts()).resolved()
    cache_key = ":".join(
        (
            mode,
            hashlib.sha256(image_bytes).hexdigest()[:16],
            cfg["model"],
            cfg["speed"],
            cfg["thinking"],
            cfg["effort"],
        )
    )
    if cache_key in _cat_cache:
        _log_verbose(f"[claude] → {mode} CACHE HIT key={cache_key[-24:]}")
        return dict(_cat_cache[cache_key])

    data, _info = _call(
        image_bytes,
        spec["prompt"],
        max_tokens=spec["max_tokens"],
        timeout_s=spec["timeout"] if timeout_s is None else timeout_s,
        opts=opts,
        purpose=mode,
        deadline=deadline,
    )
    if data is None:
        return None
    raw = _extract_json(data)
    if raw is None:
        _log_once(f"nojson-{mode}", f"[claude] {mode}: respuesta sin JSON")
        return None
    if len(_cat_cache) >= _CACHE_MAX:
        _cat_cache.clear()
    _cat_cache[cache_key] = raw
    return dict(raw)


# ── Especializaciones usadas por CURIO ─────────────────────────────────────


def read_for_catalog(
    image_bytes: bytes,
    timeout_s: float | None = None,
    opts: CallOpts | None = None,
    deadline: float | None = None,
) -> dict | None:
    """Lectura de grado catálogo diecast (modo `catalogo`).

    None sin clave, con el corte abierto, o ante error.
    """
    raw = analyze(image_bytes, mode="catalogo", timeout_s=timeout_s, opts=opts, deadline=deadline)
    if raw is None:
        return None
    return _norm_catalog(raw)


_STR_FIELDS = ("brand", "casting", "modelo", "color", "dorsal", "livery", "texto", "toy_number", "serie")
_CONF = ("alto", "medio", "bajo")


def _norm_catalog(raw: dict) -> dict:
    out: dict = {
        "es_diecast": raw.get("es_diecast") is True,
        # Render / ilustración / foto IA / coche real a escala 1:1 → False.
        "es_foto_real": raw.get("es_foto_real") is True,
    }
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


_PROMPT_VERIFY = """\
Mira esta foto con escepticismo, como un perito que detecta trampas.
¿Se ve un objeto FISICO fotografiado (un coche de juguete diecast a escala tipo 1:64 o su empaque de cartón/blister)?
O es otra cosa: render 3D, ilustracion, imagen generada por IA, captura de videojuego, coche real a escala 1:1, documento, persona, escena.

Responde SOLO con un JSON: {"fisico": true/false, "texto_leido": "…", "marca_leido": "…", "casting_leido": "…", "year_leido": entero|null, "detalle": "breve"}

fisico: true SOLO si se ve un juguete fisico o su empaque real. Si parece una foto de un render, una ilustracion o una imagen generada por IA, es false.
texto_leido: transcribe TAL CUAL las letras fisicamente impresas sobre el objeto o el empaque que puedas VER en la foto. Si no hay letras impresas, pon "".
marca_leido: la marca del fabricante del juguete SOLO si se reconoce sin duda; si dudas, "".
casting_leido: el modelo del coche SOLO si se reconoce sin duda; si dudas, "".
year_leido: el año SOLO si aparece impreso en la foto; si no, null.
No escribas texto que se deduzca del modelo del coche o de la decoracion: solo letras realmente visibles."""


def verify_real_object(
    image_bytes: bytes,
    opts: CallOpts | None = None,
    timeout_s: float | None = None,
    deadline: float | None = None,
) -> dict | None:
    """Segunda opinión pericial: ¿objeto físico real y qué texto/identidad hay?

    Necesario porque el lector de catálogo alucina en renders. `texto_leido`,
    `marca_leido`, `casting_leido` y `year_leido` sirven para CONTRASTAR con lo
    que reclama la lectura: un render cambia de casting y de año entre llamadas;
    una pieza real se mantiene. Fail-open: None = sin verificación.
    """
    data, _info = _call(
        image_bytes,
        _PROMPT_VERIFY,
        max_tokens=400,
        timeout_s=_GENERIC_TIMEOUT_S if timeout_s is None else timeout_s,
        opts=opts,
        purpose="verify",
        deadline=deadline,
    )
    if data is None:
        return None
    raw = _extract_json(data)
    if raw is None:
        return None
    y = raw.get("year_leido")
    return {
        "fisico": raw.get("fisico") is True,
        "texto_leido": str(raw.get("texto_leido") or "").strip(),
        "marca_leido": str(raw.get("marca_leido") or "").strip(),
        "casting_leido": str(raw.get("casting_leido") or "").strip(),
        "year_leido": y if isinstance(y, int) and not isinstance(y, bool) else None,
        "detalle": (str(raw.get("detalle") or "")[:200] or None),
    }


def read_packaging_with_claude(image_bytes: bytes, opts: CallOpts | None = None) -> dict | None:
    """Adaptador del respaldo del identify sobre la lectura de catálogo.

    Misma llamada y misma caché que read_for_catalog (una sola de pago por foto);
    solo cambia el timeout (presupuesto corto del identify) y el formato de salida.
    """
    read = read_for_catalog(image_bytes, timeout_s=_TIMEOUT_S, opts=opts)
    if not read or not read.get("es_diecast"):
        return None
    # El dorsal se conserva aunque no haya livery ni color.
    variant = " ".join(filter(None, [read.get("livery") or read.get("color"), read.get("dorsal")])) or None
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
