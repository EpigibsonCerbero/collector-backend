"""
Peritaje semántico con MiMo vision — SOLO para casos dudosos.

No entra en el camino caliente del identify: se invoca únicamente cuando la
decisión ya fue "manual" (clúster de liveries gemelos, sin identidad textual)
y hay que DESAMPATAR entre candidatos. El embedding ve formas; aquí se lee el
livery ("pink monster livery with #80 decals") y se empareja con los nombres
del catálogo.

Reglas duras:
  · Nunca se llama si el embed ya acertó con confianza (presupuesto 8 s).
  · Fail-open: cualquier error, timeout o respuesta basura → devolvemos None y
    el comportamiento queda EXACTAMENTE como hoy (manual + sugerencias).
  · Latencia medida: 14-17 s. Por eso solo en el desempate, donde el usuario
    ya está mirando sugerencias.
"""
from __future__ import annotations

import base64
import json
import re
import os
import time
from typing import Any

import httpx

_BASE = os.environ.get("MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1")
_KEY = os.environ.get("MIMO_API_KEY") or os.environ.get("EXPO_PUBLIC_MIMO_API_KEY")
_MODEL = os.environ.get("MIMO_MODEL", "mimo-v2.6-flash")
_TIMEOUT_S = float(os.environ.get("MIMO_TIMEOUT_S", "22"))

_PROMPT = """Eres perito en coleccionismo diecast 1:64. Se te da una FOTO de una pieza
y una lista de candidatos del catálogo. Tu tarea es elegir cuál es, mirando
sobre todo el LIVERY (diseño/pintura) y el número de carrera (dorsal) impreso.

Candidatos:
{candidates}

Devuelve SOLO JSON:
{{"pick": "<el nombre EXACTO de un candidato de la lista, o null>",
  "color": "<color de carrocería>",
  "livery": "<descripción breve del diseño>",
  "confidence": <0..1>,
  "why": "<una frase>"}}
Si ninguno corresponde a la foto, devuelve pick=null. No inventes números de pieza."""


def rank_candidates(
    image_bytes: bytes,
    candidate_names: list[str],
    livery_hints: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Elige entre candidatos usando el livery de la foto. None = sin opinión.

    `livery_hints` mapea nombre → descripción de livery ya curada (del
    catálogo). Se le pasa al modelo como contexto: con eso distingue
    "Roxy" de "Rexy" aunque el nombre no lleve color.
    """
    if not image_bytes or len(candidate_names) < 2 or not _KEY:
        return None
    hints = livery_hints or {}
    lines = []
    for i, n in enumerate(candidate_names[:8]):
        hint = (hints.get(n) or "").strip()
        lines.append(f"{i + 1}. {n}" + (f"\n    livery conocido: {hint}" if hint else ""))
    listed = "\n".join(lines)
    body = {
        "model": _MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT.format(candidates=listed)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(image_bytes).decode()
                        },
                    },
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
    }
    t0 = time.time()
    parsed = None
    # MiMo es modelo de razonamiento: a veces se gasta todo el presupuesto
    # pensando y devuelve `content` vacío. Se reintenta UNA vez antes de
    # rendirse (fail-open: el comportamiento queda como si no hubiera opinión).
    for attempt in range(2):
        try:
            with httpx.Client(timeout=_TIMEOUT_S) as client:
                r = client.post(
                    f"{_BASE.rstrip('/')}/chat/completions",
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {_KEY}",
                    },
                )
            if r.status_code != 200:
                print(f"[peritaje] HTTP {r.status_code}: {r.text[:160]}", flush=True)
                continue
            data = r.json()
            usage = data.get("usage") or {}
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            txt = msg.get("content") or ""
            reason = (msg.get("reasoning_content") or "")
            print(
                f"[peritaje] intento {attempt+1}: content={len(txt)}c "
                f"reasoning={len(reason)}c tokens={usage.get('completion_tokens')}",
                flush=True,
            )
            parsed = _parse_json(txt)
            if parsed is not None:
                print(f"[peritaje] OK pick={parsed.get('pick')!r} conf={parsed.get('confidence')}", flush=True)
                break
        except Exception as exc:  # noqa: BLE001
            print(f"[peritaje] red: {type(exc).__name__}: {exc}", flush=True)
    if parsed is None:
        print("[peritaje] sin respuesta utilizable tras 2 intentos", flush=True)
        return None
        pick = parsed.get("pick")
        # el pick tiene que ser EXACTAMENTE uno de los candidatos
        if isinstance(pick, str) and pick.strip():
            for n in candidate_names:
                if pick.strip().lower() == n.strip().lower() or pick.strip() in n or n in pick.strip():
                    parsed["pick"] = n
                    break
            else:
                print(f"[peritaje] pick fuera de la lista: {pick!r}", flush=True)
                parsed["pick"] = None
        else:
            parsed["pick"] = None
        parsed["_ms"] = int((time.time() - t0) * 1000)
        return parsed


def _parse_json(txt: str) -> dict[str, Any] | None:
    """Extrae el JSON aunque venga con fences markdown o texto alrededor."""
    if not txt:
        return None
    s = txt.strip()
    # quitar fences markdown. OJO: solo ELIMINAR los ticks — reemplazar
    # "```json" por "{" metía un '{' de más y corrompía un JSON válido.
    if "```" in s:
        s = re.sub(r"```(?:json)?", "", s, flags=re.I)
    s = s.strip()
    s = re.sub(r"^json\s*", "", s, flags=re.I).strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    blob = s[i : j + 1]
    try:
        d = json.loads(blob)
        return d if isinstance(d, dict) else None
    except Exception:  # noqa: BLE001
        # a veces el modelo trunca: probar el último objeto completo
        for end in range(len(blob) - 1, 0, -1):
            if blob[end] != "}":
                continue
            try:
                d = json.loads(blob[: end + 1])
                return d if isinstance(d, dict) else None
            except Exception:  # noqa: BLE001
                continue
        return None
