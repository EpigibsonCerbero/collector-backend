"""Gemini Vision API — lectura inteligente de empaques diecast.

Cuando Tesseract falla (fotos reales de blister con reflejos, ángulos, texto
pequeño), Gemini Vision lee el empaque y extrae marca, modelo, toy number,
piloto, color, serie — todo lo que necesitamos para matchear en el catálogo.

Solo se llama como respaldo cuando OCR tradicional no encuentra identidad.
Con retry + caché para manejar rate limits y cold starts.
"""

import base64
import hashlib
import json
import os
import re
import time

import requests

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-3.8-flash:generateContent"
)

_PROMPT = """\
Eres un experto en coleccionables diecast Mini GT 1:64. Analiza esta foto de un empaque o coche diecast y extrae EXACTAMENTE esta información en JSON:

{
  "brand": "marca del coche real (ej: Lamborghini, Toyota, Nissan, AlphaTauri)",
  "model": "modelo del coche (ej: Countach LPI 800-4, GR86, GT-R)",
  "variant": "variante/color/livery (ej: Rosso Mars, Larry Chen HKS, #22 Yuki Tsunoda)",
  "toy_number": "toy number Mini GT si es visible (ej: 720, 820, 726) — solo el número",
  "series": "serie si aparece (ej: MiJo Exclusives, Full Collection, Gold Series)",
  "color": "color principal del coche",
  "year": "año si aparece",
  "driver": "piloto si es un coche de carreras (ej: Yuki Tsunoda)",
  "full_text": "todo el texto legible del empaque",
  "confidence": "alto/medio/bajo según qué tan claro esté todo"
}

Si un campo no está visible, pon null. Responde SOLO con el JSON, sin markdown."""

# Caché simple: hash de imagen → respuesta de Gemini (evita re-llamar a la API)
_cache: dict[str, dict | None] = {}
_CACHE_MAX = 50

# Rate limit global: no llamar a Gemini más de una vez cada N segundos
_last_call_time = 0.0
_MIN_INTERVAL = 5.0  # segundos entre llamadas a Gemini


def _image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:16]


def read_packaging_with_gemini(image_bytes: bytes) -> dict | None:
    """Envía la imagen a Gemini Vision con retry + caché."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("EXPO_PUBLIC_GEMINI_API_KEY")
    if not api_key:
        print("[gemini] sin API key — se omite", flush=True)
        return None

    # Caché: misma imagen = misma respuesta
    img_hash = _image_hash(image_bytes)
    if img_hash in _cache:
        print(f"[gemini] caché hit ({img_hash})", flush=True)
        return _cache[img_hash]

    # Rate limit global: esperar si la última llamada fue hace poco
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < _MIN_INTERVAL:
        wait = _MIN_INTERVAL - elapsed
        print(f"[gemini] rate limit global, esperando {wait:.1f}s", flush=True)
        time.sleep(wait)

    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": _PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1024,
            },
        }

        # Gemini es un FALLBACK de último recurso (solo cuando el OCR no leyó
        # nada). Con timeout=30 y 3 retries se cepillaba hasta 90 s del budget
        # de 8 s (medido en prod 2026-09-29: gemini=27.9 s en una foto). Si no
        # contesta en 3 s, seguimos sin él — el embed ya decide.
        max_retries = 1
        for attempt in range(max_retries):
            r = requests.post(
                _GEMINI_URL,
                params={"key": api_key},
                json=payload,
                timeout=3,
            )
            _last_call_time = time.time()

            if r.status_code == 200:
                break
            elif r.status_code in (429, 503, 500):
                wait = [5, 15, 30][min(attempt, 2)]  # 5s, 15s, 30s
                print(f"[gemini] HTTP {r.status_code}, retry {attempt + 1}/{max_retries} en {wait}s", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    print(f"[gemini] todos los retries fallaron", flush=True)
                    pass  # no cachear fallos, reintentar despues
                    return None
            else:
                print(f"[gemini] HTTP {r.status_code}: {r.text[:200]}", flush=True)
                pass  # no cachear fallos
                return None

        data = r.json()
        candidates = data.get("candidates") or []
        if not candidates:
            pass  # no cachear fallos
            return None

        text = (
            candidates[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )

        # Extraer JSON de la respuesta
        text = text.strip()
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            print(f"[gemini] sin JSON en respuesta: {text[:150]}", flush=True)
            pass  # no cachear fallos
            return None

        result = json.loads(text[start : end + 1])
        print(
            f"[gemini] brand={result.get('brand')} model={result.get('model')} "
            f"toy={result.get('toy_number')} variant={result.get('variant')}",
            flush=True,
        )

        # Guardar en caché
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[img_hash] = result
        return result

    except Exception as exc:
        print(f"[gemini] error: {type(exc).__name__}: {exc}", flush=True)
        return None


def gemini_to_ocr_text(gemini_data: dict) -> str:
    """Convierte la respuesta de Gemini a texto para el pipeline de matching."""
    if not gemini_data:
        return ""
    parts = []
    for key in ("brand", "model", "variant", "series", "driver", "full_text"):
        v = gemini_data.get(key)
        if v:
            parts.append(str(v))
    toy = gemini_data.get("toy_number")
    if toy:
        parts.append(f"MGT{str(toy).zfill(5)[-5:]}")
        parts.append(f"▶ {toy}")
    return "\n".join(parts)
