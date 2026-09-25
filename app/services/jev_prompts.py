"""Prompts y criterios del servicio de decisión JEV (System One).

Viven SOLO en servidor — jamás se envían al cliente móvil (política APK:
ocultar el stack y los criterios de decisión). Migrados al backend FastAPI
al sustituir la capa de edge functions previa.

Casos (misma letra que `decide-cases.ts` del cliente):
  d — mismo release (noul)
  r — rareza (choice)
  t — triage de candidatos (noul × N)
  p — score de empaque 1-10
  i — inspección (rareza + empaque + error de fábrica)
"""
from __future__ import annotations

from typing import Any

RARITY_CRITERIA: dict[str, str] = {
    "common": "Mainline / regular de producción masiva, sin indicadores de rareza",
    "treasure_hunt": (
        "Treasure Hunt (TH): logo TH en packaging o tampo, goma de goma "
        "(rubber tires) en mainline"
    ),
    "super_treasure_hunt": (
        "Super Treasure Hunt ($TH/Spectraflame): pintura Spectraflame, "
        "Real Riders, logo $TH flame"
    ),
    "chase": "Chase / 0/5 / ZAMAC sin pintar / ediciones limitadas tipo Kaido House Chase",
    "rlc": "RLC (Red Line Club) u edición premium de membresía sellada",
    "premium": "Premium / Car Culture / Premium Box con Real Riders pero sin ser Chase ni STH",
    "error": "Error de fábrica (mal tampo, rueda equivocada, caja incorrecta) o pieza defectuosa",
}

PACKAGING_LEVELS: list[str] = [
    "1-2: Caja/blister destruido, faltan piezas, sin valor de colección",
    "3-4: Daño severo (esquinas rotas, burbuja despegada, humedad)",
    "5-6: Desgaste visible, esquinas dobladas, tarjeta con dobleces",
    "7-8: Muy buen estado, marcas leves de estantería, burbuja intacta",
    "9: Casi mint, imperfecciones mínimas de fábrica o transporte",
    "10: Mint / gem mint sellado, impecable, apto para grading profesional",
]


def item_from_state(x: dict[str, Any] | None) -> dict[str, Any]:
    x = x or {}
    return {
        "nombre": x.get("n"),
        "marca": x.get("m"),
        "anio": x.get("y"),
        "variante": x.get("v"),
        "numero_juguete": x.get("t"),
        "color": x.get("c"),
        "llantas": x.get("w"),
        "material": x.get("p"),
        "serie": x.get("s"),
        "sku": x.get("u"),
        "descripcion": x.get("d"),
        "tipo_empaque": x.get("k"),
        "notas": x.get("o"),
    }


def build_case(u: str, s: dict[str, Any]) -> dict[str, Any]:
    """Construye `state` + `questions` para una llamada a JEV (System One)."""
    if u == "d":
        return {
            "state": {
                "registro_a": item_from_state(s.get("a")),
                "registro_b": item_from_state(s.get("b")),
            },
            "questions": {
                "q": {
                    "type": "noul",
                    "instructions": {
                        "pregunta": (
                            "¿`registro_a` y `registro_b` describen el mismo modelo y el "
                            "mismo release (misma variante), no solo un casting similar?"
                        ),
                        "nota": (
                            "Un Mainline y su variante de mayor nivel NO son el mismo "
                            "release aunque compartan casi todo el nombre. Exige "
                            "coincidencia de variante, acabado, llantas y número de pieza."
                        ),
                    },
                    "criteria": {
                        "true": (
                            "Mismo release exacto: mismo casting + misma variante/"
                            "acabado/llantas/número"
                        ),
                        "false": "Castings distintos, variantes distintas, o datos insuficientes",
                    },
                },
            },
        }

    if u == "r":
        return {
            "state": item_from_state(s),
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": (
                        "Clasifica la rareza de esta pieza en UNA de las categorías "
                        "cerradas. Usa los indicadores de packaging, llantas, "
                        "acabado y serie."
                    ),
                    "criteria": RARITY_CRITERIA,
                },
            },
        }

    if u == "t":
        user = s.get("u") or {}
        cands = s.get("c") or []
        state: dict[str, Any] = {
            "solicitud_usuario": {
                "texto": user.get("f"),
                "nombre": user.get("n"),
                "marca": user.get("m"),
                "anio": user.get("y"),
                "notas": user.get("o"),
            },
            "candidatos_catalogo": [
                {"indice": i, **item_from_state(c)} for i, c in enumerate(cands)
            ],
        }
        questions: dict[str, Any] = {}
        for i in range(len(cands)):
            questions[f"q{i}"] = {
                "type": "noul",
                "instructions": {
                    "pregunta": (
                        f"¿`solicitud_usuario` describe la misma pieza que "
                        f"`candidatos_catalogo` con indice `{i}`?"
                    ),
                    "nota": (
                        "Debe ser el mismo release (casting + variante), "
                        "no solo un modelo parecido."
                    ),
                },
                "criteria": {
                    "true": "La solicitud corresponde a ese candidato",
                    "false": "No coincide o el candidato es una variante distinta",
                },
            }
        return {"state": state, "questions": questions}

    if u == "p":
        return {
            "state": {
                "pieza": s.get("n"),
                "tipo_empaque": s.get("k"),
                "condicion_declarada": s.get("o"),
                "tiene_protector": s.get("h"),
                "notas_inspeccion": s.get("d"),
            },
            "questions": {
                "q": {
                    "type": "score",
                    "instructions": (
                        "Gradúa la condición del empaque (caja, blister, tarjeta) "
                        "según las notas de inspección. Si no hay evidencia suficiente, "
                        "calibra hacia el centro y baja la confianza."
                    ),
                    "criteria": list(PACKAGING_LEVELS),
                },
            },
        }

    # u == 'i'
    return {
        "state": item_from_state(s),
        "questions": {
            "r": {
                "type": "choice",
                "instructions": (
                    "Clasifica la rareza de esta pieza en UNA de las categorías cerradas."
                ),
                "criteria": RARITY_CRITERIA,
            },
            "p": {
                "type": "score",
                "instructions": (
                    "Gradúa la condición del empaque (caja/blister/tarjeta) "
                    "según la evidencia disponible."
                ),
                "criteria": list(PACKAGING_LEVELS),
            },
            "e": {
                "type": "noul",
                "instructions": "¿Es una pieza error de fábrica o con defecto de producción?",
                "criteria": {
                    "true": (
                        "Error de tampo, rueda equivocada, caja incorrecta "
                        "u otro defecto de fábrica"
                    ),
                    "false": "Pieza normal de producción",
                },
            },
        },
    }
