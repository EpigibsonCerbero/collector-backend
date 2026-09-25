"""Servicio de decisión y curación (server-side, prompts en jev_prompts.py).

Implementación determinista y sin coste: similitud de campos + palabras clave.
Respuestas con la misma forma que el contrato `decide-client` del cliente.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/decide", tags=["decide"])
curate_router = APIRouter(prefix="/curate", tags=["decide"])

RarityTier = Literal[
    "common",
    "treasure_hunt",
    "super_treasure_hunt",
    "chase",
    "rlc",
    "premium",
    "error",
]


def _norm(text: str | None) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", str(text))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def _tokens(text: str | None) -> set[str]:
    return {t for t in _norm(text).split() if len(t) > 1}


def _identity_fields(d: dict[str, Any]) -> dict[str, str]:
    return {
        "name": str(d.get("n") or ""),
        "brand": str(d.get("m") or ""),
        "variant": str(d.get("v") or ""),
        "toy": str(d.get("t") or "").upper(),
        "color": str(d.get("c") or ""),
        "series": str(d.get("s") or ""),
        "upc": str(d.get("u") or ""),
        "desc": str(d.get("d") or ""),
        "year": str(d.get("y") or ""),
    }


def _similarity(a: dict[str, str], b: dict[str, str]) -> float:
    """Probabilidad 0..1 de que dos identidades sean el mismo release."""
    # Evidencia exacta domina.
    if a["toy"] and a["toy"] == b["toy"]:
        return 0.99
    if a["upc"] and a["upc"] == b["upc"]:
        return 0.99

    score = 0.0
    name_a = _tokens(f"{a['name']} {a['variant']}")
    name_b = _tokens(f"{b['name']} {b['variant']}")
    if name_a and name_b:
        inter = len(name_a & name_b)
        union = len(name_a | name_b)
        jaccard = inter / union if union else 0.0
        score += 0.55 * jaccard

    brand_a, brand_b = _norm(a["brand"]), _norm(b["brand"])
    if brand_a and brand_b:
        if brand_a == brand_b:
            score += 0.12
        elif brand_a in brand_b or brand_b in brand_a:
            score += 0.06

    if a["year"] and b["year"] and a["year"] == b["year"]:
        score += 0.10

    color_a, color_b = _norm(a["color"]), _norm(b["color"])
    if color_a and color_b and color_a == color_b:
        score += 0.06

    series_a, series_b = _norm(a["series"]), _norm(b["series"])
    if series_a and series_b and series_a == series_b:
        score += 0.07

    return max(0.0, min(1.0, score))


_RARITY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("super_treasure_hunt", ("super treasure hunt", "super treasure", "sth ")),
    ("treasure_hunt", ("treasure hunt", "treasurehunt", "th ")),
    ("chase", ("chase", "chase piece", "chase version")),
    ("rlc", ("rlc", "red line club")),
    ("premium", ("premium", "premium edition", "car culture", "team transport")),
    ("error", ("error", "misprint", "wrong wheels", "no tampo", "flipped base")),
]


def _classify_rarity(fields: dict[str, str]) -> tuple[str, float, dict[str, float]]:
    blob = _norm(f"{fields['name']} {fields['variant']} {fields['series']} {fields['desc']}")
    for tier, keys in _RARITY_KEYWORDS:
        for k in keys:
            if k.strip() and k.strip() in blob:
                # "th " es ambiguo: confianza menor
                conf = 0.72 if k.strip() in ("th", "sth") else 0.9
                return tier, conf, {tier: conf, "common": round(1 - conf, 2)}

    if fields["series"]:
        return "common", 0.75, {"common": 0.75, "premium": 0.25}
    return "common", 0.6, {"common": 0.6, "premium": 0.4}


def _grade_packaging(data: dict[str, Any]) -> tuple[float, float, dict[str, str]]:
    notes = _norm(str(data.get("d") or data.get("o") or ""))
    packaging = _norm(str(data.get("k") or ""))
    has_protector = bool(data.get("h"))

    score = 5.0
    legend: dict[str, str] = {}

    if any(w in notes for w in ("mint", "nuevo", "sin abrir", "sealed", "perfecto")):
        score += 2.5
        legend["estado"] = "Mint / sin abrir"
    elif any(w in notes for w in ("excelente", "near mint", "casi nuevo")):
        score += 1.5
        legend["estado"] = "Casi nuevo"
    elif any(w in notes for w in ("desgast", "golpe", "roto", "dano", "daño", "peel", "despeg")):
        score -= 2.5
        legend["estado"] = "Con daños visibles"
    elif any(w in notes for w in ("usado", "juegado", "wear", "abollad")):
        score -= 1.5
        legend["estado"] = "Usado"

    if packaging:
        if any(w in packaging for w in ("box", "caja", "acrylic", "acrílic")):
            score += 1.0
            legend["empaque"] = packaging.title()
        elif any(w in packaging for w in ("blister", "carded", "long card", "short card")):
            score += 0.5
            legend["empaque"] = packaging.title()
        elif any(w in packaging for w in ("loose", "suelt")):
            score -= 1.5
            legend["empaque"] = "Sin empaque"
        else:
            legend["empaque"] = packaging.title()

    if has_protector:
        score += 1.0
        legend["protector"] = "Con protector"

    score = max(1.0, min(10.0, score))
    confidence = 0.9 if legend.get("estado") else 0.55
    return score, confidence, legend


class DecideBody(BaseModel):
    """`u` = caso ('d'|'r'|'t'|'p'|'i'), `s` = estado compacto del cliente."""

    u: str
    s: Any = Field(default_factory=dict)


@router.post("")
def decide(body: DecideBody) -> dict[str, Any]:
    case = (body.u or "").strip().lower()
    s = body.s if isinstance(body.s, dict) else {}

    if case == "d":
        a = _identity_fields(s.get("a") or {})
        b = _identity_fields(s.get("b") or {})
        return {"p": round(_similarity(a, b), 4)}

    if case == "r":
        fields = _identity_fields(s)
        tier, conf, pr = _classify_rarity(fields)
        return {"t": tier, "c": round(conf, 4), "pr": pr}

    if case == "t":
        user = s.get("u") or {}
        user_fields = {
            "name": str(user.get("n") or ""),
            "brand": str(user.get("m") or ""),
            "variant": str(user.get("f") or ""),
            "toy": "",
            "color": "",
            "series": "",
            "upc": "",
            "desc": str(user.get("o") or user.get("f") or ""),
            "year": str(user.get("y") or ""),
        }
        candidates = s.get("c") or []
        ps = [
            round(_similarity(user_fields, _identity_fields(c if isinstance(c, dict) else {})), 4)
            for c in candidates
        ]
        return {"ps": ps}

    if case == "p":
        score, conf, legend = _grade_packaging(s)
        return {"s": round(score, 2), "c": round(conf, 4), "lg": legend}

    if case == "i":
        fields = _identity_fields(s)
        tier, conf, pr = _classify_rarity(fields)
        score, pconf, legend = _grade_packaging(s)
        is_error = 1 if tier == "error" else 0
        return {
            "r": {"t": tier, "c": round(conf, 4), "pr": pr},
            "p": {"s": round(score, 2), "c": round(pconf, 4), "lg": legend},
            "e": is_error,
        }

    # Caso desconocido: respuesta vacía pero válida (el cliente tolera defaults).
    return {}


class CurateItem(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    manufacturer: Optional[str] = None
    release_year: Optional[int] = None
    variant_name: Optional[str] = None
    toy_number: Optional[str] = None
    rarity: Optional[str] = None
    body_color: Optional[str] = None
    wheel_type: Optional[str] = None
    base_material: Optional[str] = None
    sku_upc: Optional[str] = None
    series: Optional[str] = None
    description: Optional[str] = None


class CurateBatchBody(BaseModel):
    incomingItems: list[CurateItem] = Field(default_factory=list)
    masterCandidates: list[CurateItem] = Field(default_factory=list)
    topN: int = 5


class CuratePairItem(BaseModel):
    incoming: CurateItem
    candidate: CurateItem
    incomingId: Optional[str] = None
    candidateId: Optional[str] = None


class CuratePairsBody(BaseModel):
    pairs: list[CuratePairItem] = Field(default_factory=list)


def _curate_one(incoming: dict[str, Any], candidates: list[dict[str, Any]], top_n: int) -> dict[str, Any]:
    inc_fields = {
        "name": str(incoming.get("name") or ""),
        "brand": str(incoming.get("manufacturer") or ""),
        "variant": str(incoming.get("variant_name") or ""),
        "toy": str(incoming.get("toy_number") or "").upper(),
        "color": str(incoming.get("body_color") or ""),
        "series": str(incoming.get("series") or ""),
        "upc": str(incoming.get("sku_upc") or ""),
        "desc": str(incoming.get("description") or ""),
        "year": str(incoming.get("release_year") or ""),
    }

    scored: list[tuple[float, dict[str, Any]]] = []
    for cand in candidates:
        cand_fields = {
            "name": str(cand.get("name") or ""),
            "brand": str(cand.get("manufacturer") or ""),
            "variant": str(cand.get("variant_name") or ""),
            "toy": str(cand.get("toy_number") or "").upper(),
            "color": str(cand.get("body_color") or ""),
            "series": str(cand.get("series") or ""),
            "upc": str(cand.get("sku_upc") or ""),
            "desc": str(cand.get("description") or ""),
            "year": str(cand.get("release_year") or ""),
        }
        scored.append((_similarity(inc_fields, cand_fields), cand))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, top_n)]

    best_p = top[0][0] if top else 0.0
    best_id = (top[0][1].get("id") if top else None) or (top[0][1].get("incomingId") if top else None)

    if best_p >= 0.95:
        action = "merge"
    elif best_p < 0.35:
        action = "create"
    else:
        action = "review"

    rarity = None
    if action == "create":
        tier, conf, _ = _classify_rarity(inc_fields)
        rarity = {"rarity": tier, "confidence": conf, "needsReview": conf < 0.8}

    return {
        "incomingId": incoming.get("id") or "",
        "bestMatchId": best_id,
        "probability": round(best_p, 4),
        "action": action,
        "rarity": rarity,
    }


@curate_router.post("/catalog")
def curate_catalog(body: CurateBatchBody) -> dict[str, Any]:
    incoming = [i.model_dump() for i in body.incomingItems]
    candidates = [c.model_dump() for c in body.masterCandidates]
    decisions = [_curate_one(i, candidates, body.topN) for i in incoming]
    return {
        "decisions": decisions,
        "usage": {"input_tokens": 0, "output_tokens": 0, "calls": len(decisions)},
    }


@curate_router.post("/pairs")
def curate_pairs(body: CuratePairsBody) -> dict[str, Any]:
    decisions = []
    for pair in body.pairs:
        inc = pair.incoming.model_dump()
        cand = pair.candidate.model_dump()
        if pair.incomingId:
            inc["id"] = pair.incomingId
        if pair.candidateId:
            cand["id"] = pair.candidateId
        decisions.append(_curate_one(inc, [cand], 1))
    return {
        "decisions": decisions,
        "usage": {"input_tokens": 0, "output_tokens": 0, "calls": len(decisions)},
    }
