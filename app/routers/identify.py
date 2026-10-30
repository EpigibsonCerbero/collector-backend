"""
Identificación de piezas por foto — el cliente manda la imagen y recibe la pieza.

Toda la lógica vive en el servidor (barcode + embeddings + OCR + decisión);
la app solo sube la foto. Respuesta alineada con `PieceIdentificationResult`
del cliente para que el cambio sea transparente.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import re
import threading
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.ratelimit import client_ip, hit_or_429
from app.security import CurrentUser, get_current_user
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db import as_id, get_db, oid, serialize, serialize_many
from app.services import revenuecat, subscription as subs
from app.services.debug_bus import publish as _dbg
from app.services.vision import (
    VisionMatch,
    colors_match_strict,
    identify_image,
)


def _name_family_safe(name: str | None) -> tuple[str, ...]:
    return tuple((name or "").lower().split()[:2])


def _livery_pick(image_bytes: bytes, names: list[str], hints: dict[str, str] | None = None) -> dict[str, Any] | None:
    """Peritaje semántico del livery. Fail-open: None = sin opinión."""
    try:
        from app.services.mimo_vision import rank_candidates

        return rank_candidates(image_bytes, names, livery_hints=hints)
    except Exception as exc:  # noqa: BLE001
        _dbg("peritaje", f"servicio no disponible: {exc}")
        return None


def _livery_hints_for(db, suggestions: list) -> dict[str, str]:
    """livery_desc curado por cada sugerencia (para el contexto del peritaje)."""
    hints: dict[str, str] = {}
    for s in suggestions[:8]:
        if not s.name:
            continue
        doc = db.variants.find_one({"variant_name": s.name}, {"livery_desc": 1, "livery_color": 1, "livery_number": 1})
        if not doc and s.id:
            doc = db.variants.find_one({"_id": s.id}, {"livery_desc": 1, "livery_color": 1, "livery_number": 1})
        if not doc:
            continue
        bits = [b for b in (doc.get("livery_desc"), doc.get("livery_color"), f"dorsal {doc.get('livery_number')}" if doc.get("livery_number") else None) if b]
        if bits:
            hints[s.name] = " · ".join(str(b) for b in bits)[:220]
    return hints


def _livery_tokens(s: str) -> set[str]:
    return {t for t in re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split() if len(t) >= 3}


def _apply_livery_score(db, res, image_colors: list[str] | None) -> None:
    """Reordena `res.matches` usando el livery curado (semántica sin latencia).

    Señales, todas baratas (lookups a Mongo de los 8 primeros):
      · dorsal curado (`livery_number`) vs números del OCR   → señal fuerte
      · `livery_color` vs color medido en la foto             → señal fuerte
      · tokens del `livery_desc` vs texto del OCR             → señal media
    No inventa candidatos: solo reordena los que el embed ya trajo. Es lo que
    faltaba para que "Championship White" le ganara a "Ultimate Edition" con
    un OCR que solo dice "White".
    """
    top_n = res.matches[:8]
    if len(top_n) < 2:
        return
    text = (res.ocr_text or "").lower()
    # Dorsales SOLO con marca explícita (#N / ▶N). Un número suelto en un OCR
    # basura ('QE\n9\nHOYUI') no es un dorsal: hacía que un Huracan #9 le
    # ganara al Corvette #3 que era el top visual correcto (medido 2026-09-30).
    ocr_nums = {m.group(1) for m in re.finditer(r"[#>»▶]\s*(\d{1,3})\b", text)}
    ocr_toks = _livery_tokens(text)
    for m in top_n:
        doc = db.variants.find_one(
            {"_id": m.id},
            {"livery_desc": 1, "livery_color": 1, "livery_number": 1, "body_color": 1},
        ) if m.id else None
        if not doc and m.toy:
            doc = db.variants.find_one(
                {"toy_number": m.toy},
                {"livery_desc": 1, "livery_color": 1, "livery_number": 1, "body_color": 1},
            )
        if not doc:
            continue
        factor = 1.0
        why = []
        num = doc.get("livery_number")
        if num:
            n = str(num).lstrip("#")
            if ocr_nums and n in ocr_nums:
                factor *= 1.15
                why.append(f"dorsal {n}")
            elif ocr_nums:
                factor *= 0.88
        col = doc.get("livery_color") or doc.get("body_color")
        if col and image_colors:
            if any(colors_match_strict(col, c) for c in image_colors):
                factor *= 1.12
                why.append(f"color {col}")
        desc = doc.get("livery_desc") or ""
        if desc and ocr_toks:
            dt = _livery_tokens(desc)
            overlap = len(dt & ocr_toks)
            if overlap >= 2:
                factor *= 1.08
                why.append(f"{overlap} palabras de livery")
        if factor != 1.0:
            m.score = max(0.0, min(1.0, m.score * factor))
            _dbg("livery", f"{m.name[:42]!r} ×{factor:.2f} ({', '.join(why)})")
    res.matches.sort(key=lambda x: -x.score)

router = APIRouter(tags=["identify"])


# ── Schemas ─────────────────────────────────────────────────────────────────


class IdentifySuggestion(BaseModel):
    id: str
    name: Optional[str] = None
    toy_number: Optional[str] = None
    score: float = 0.0
    # Miniatura del packshot para el grid de sugerencias de la app.
    image: Optional[str] = None
    body_color: Optional[str] = None


class IdentifyOut(BaseModel):
    """Espejo de PieceIdentificationResult del cliente RN."""

    method: str = "m"  # 'b' barcode | 'o' ocr/texto | 'c' visual | 'm' manual
    confidence: float = 0.0
    needs_manual: bool = True
    upc: Optional[str] = None
    toy_number: Optional[str] = None
    item: Optional[dict[str, Any]] = None  # ItemVariantOut serializado
    suggestions: list[IdentifySuggestion] = Field(default_factory=list)
    timings_ms: dict[str, int] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)
    # Cuota diaria: {used, limit, premium}. Opcional: clientes viejos lo ignoran.
    quota: Optional[dict] = None


class IdentifyBase64(BaseModel):
    """Alternativa JSON con la foto en base64 (sin multipart)."""

    image_base64: str
    mime: str = "image/jpeg"
    brand: Optional[str] = None


class PeritajeIn(BaseModel):
    """Desempate de livery bajo demanda (endpoint aparte).

    El identify principal NO espera al peritaje: devuelve las sugerencias en
    ~2 s y la app luego llama aquí para pedir el ⭐. Así el usuario ve las
    opciones en vez de mirar una pantalla quieta 40 s.
    """

    image_base64: str
    candidates: list[str] = Field(default_factory=list)
    livery_hints: dict[str, str] = Field(default_factory=dict)


class PeritajeOut(BaseModel):
    pick: Optional[str] = None
    color: Optional[str] = None
    livery: Optional[str] = None
    confidence: Optional[float] = None
    why: Optional[str] = None
    ms: int = 0


# ── Resolución contra Mongo ─────────────────────────────────────────────────


def _variant_out(doc: dict, master: dict | None = None) -> dict[str, Any]:
    data = serialize(doc) or {}
    if master is not None:
        data["item_master"] = serialize(master)
    return data


def _find_variant_by_id(db, vid: str) -> tuple[dict | None, dict | None]:
    """Busca la variante por _id (uuid string u ObjectId).

    Los embeddings pueden venir con el source-id (`hw-flf56`, `hww-…`) mientras
    Mongo guarda el uuid5(NS, source_id) — sin este normalizador el lookup
    fallaba y TODO se iba a "manual" aunque el score fuera 1.0.
    """
    candidates = [vid]
    # uuid5 con el mismo namespace que api.ts / seed
    try:
        import uuid as _uuid

        ns = _uuid.UUID("a9f73cdf-180a-5bd7-ac02-42c291653666")
        if not re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", vid or "", re.I
        ):
            candidates.append(str(_uuid.uuid5(ns, vid)))
    except Exception:  # noqa: BLE001
        pass
    for cand in candidates:
        for key in (cand, None):
            try:
                query = {"_id": as_id(cand)} if key is not None else {"_id": cand}
                doc = db.variants.find_one(query)
            except Exception:  # noqa: BLE001
                doc = db.variants.find_one({"_id": cand})
            if doc:
                master = None
                mid = doc.get("item_master_id")
                if mid:
                    try:
                        master = db.items.find_one({"_id": as_id(mid)})
                    except Exception:  # noqa: BLE001
                        master = db.items.find_one({"_id": mid})
                return doc, master
    return None, None


def _find_variant_by_upc(db, upc: str) -> tuple[dict | None, dict | None]:
    doc = db.variants.find_one({"sku_upc": upc})
    if not doc:
        return None, None
    master = None
    mid = doc.get("item_master_id")
    if mid:
        try:
            master = db.items.find_one({"_id": as_id(mid)})
        except Exception:
            master = db.items.find_one({"_id": mid})
    return doc, master


def _find_variant_by_toy(db, toy: str) -> tuple[dict | None, dict | None]:
    norm = toy.upper().replace(" ", "")
    doc = db.variants.find_one({"toy_number": norm})
    if not doc:
        # toy_number con espacios / distinta capitalización
        doc = db.variants.find_one({"toy_number": {"$regex": f"^{norm}$", "$options": "i"}})
    if not doc:
        return None, None
    master = None
    mid = doc.get("item_master_id")
    if mid:
        try:
            master = db.items.find_one({"_id": as_id(mid)})
        except Exception:
            master = db.items.find_one({"_id": mid})
    return doc, master


def _to_suggestions(matches: list[VisionMatch], db=None) -> list[IdentifySuggestion]:
    """Sugerencias con miniatura — el grid de la app necesita ver la pieza."""
    import time as _time

    out: list[IdentifySuggestion] = []
    for m in matches[:8]:
        img = color = None
        if db is not None:
            _t = _time.time()
            doc = db.variants.find_one(
                {"_id": m.id}, {"main_image_url": 1, "body_color": 1, "images": 1, "variant_name": 1}
            )
            _ms = int((_time.time() - _t) * 1000)
            if _ms > 150:
                _dbg("router", f"find_one lento: {_ms} ms (id={str(m.id)[:8]})")
            if not doc and m.toy:
                doc = db.variants.find_one(
                    {"toy_number": m.toy}, {"main_image_url": 1, "body_color": 1, "images": 1}
                )
            if doc:
                img = doc.get("main_image_url") or (doc.get("images") or [None])[0]
                color = doc.get("body_color")
        out.append(
            IdentifySuggestion(
                id=m.id,
                name=m.name,
                toy_number=m.toy,
                score=round(m.score, 4),
                image=img,
                body_color=color,
            )
        )
    return out


_NAME_COLOR_WORDS = (
    "pink", "rosa", "red", "rojo", "blue", "azul", "green", "verde",
    "yellow", "amarillo", "orange", "naranja", "purple", "morado",
    "gold", "dorado", "silver", "plata", "black", "negro", "white",
    "blanco", "grey", "gray", "gris", "pearl", "turquoise", "teal", "brown",
)


def _doc_color(doc: dict | None) -> str | None:
    """Color de carrocería de la pieza del catálogo.

    1) `body_color` si existe (la fuente fiable).
    2) `livery_color` — color declarado por el peritaje sobre la IMAGEN del
       catálogo (curado 2026-09-30 con visión). Muy fiable: es lo que se ve.
    3) Si no, la primera palabra de color del nombre de la variante — un
       "…Magic Pearl" o "…Gem Red" lleva el color en el nombre, y dejarlo
       pasar por no tener `body_color` era un agujero (el rosa se colaba
       elegindo "Magic Pearl" que no traía body_color).
    4) Y si no, la `description` del fabricante (curada 2026-09-30): su línea 2
       es el color/livery en claro ("Championship White", "Daytona Violet",
       "Orange Flame") — cubre piezas cuyo nombre no lleva color.
    """
    if not doc:
        return None
    raw = (
        doc.get("body_color")
        or (doc.get("metadata") or {}).get("body_color")
        or doc.get("livery_color")
    )
    if raw:
        return str(raw).strip()
    for blob in (
        f"{doc.get('variant_name') or ''} {doc.get('name') or ''}",
        doc.get("description") or "",
    ):
        low = blob.lower()
        for w in _NAME_COLOR_WORDS:
            if re.search(rf"\b{w}\b", low):
                return "Grey" if w == "gray" else w.capitalize()
    return None


def _color_matches(doc: dict | None, image_colors: list[str] | None) -> bool:
    """¿El body_color del catálogo es compatible con el color de la foto?

    `image_colors` solo trae buckets CONFIABLES (ver image_body_colors): o es
    una lista con el color inequívoco de la carrocería, o es [] = sin señal.
    Con señal confiable el match es ESTRICTO — un coche plata no es un coche
    rosa, y lo neutro no sirve de coartada.
    """
    if not image_colors:
        return True
    bc = _doc_color(doc)
    if not bc:
        return True
    return all(colors_match_strict(bc, img_c) for img_c in image_colors)


def _find_color_consistent(
    db, matches: list[VisionMatch], image_colors: list[str] | None, limit: int = 8
) -> tuple[dict | None, dict | None]:
    """Primer candidato del pool cuyo body_color casa con el color de la foto.

    Solo se consulta si hay señal de color en la imagen y el ganador chocaba:
    así un livery gemelo del color correcto puede ganarle al de otro color.
    """
    if not image_colors:
        return None, None
    for m in matches[:limit]:
        doc, master = _find_variant_by_id(db, m.id)
        if not doc and m.toy:
            doc, master = _find_variant_by_toy(db, m.toy)
        if doc and _color_matches(doc, image_colors):
            _dbg("color", f"alternativa consistente: {(doc.get('variant_name') or '')!r} "
                 f"body={_doc_color(doc)!r} (foto={image_colors})")
            return doc, master
    return None, None


# ── Endpoint ────────────────────────────────────────────────────────────────


def _log_decision(out: IdentifyOut, res) -> None:
    """Traza compacta de la decisión — para depurar en /tmp/uvicorn-collector.log."""
    name = (out.item or {}).get("name") or (out.item or {}).get("variant_name") or "-"
    top = res.matches[0] if res.matches else None
    second = res.matches[1] if len(res.matches) > 1 else None
    top_s = f"{top.score:.3f}" if top else "0"
    top_n = (top.name[:40] if top else "-")
    sec_s = f"{second.score:.3f}" if second else "0"
    ocr = (res.ocr_text or "")[:80]
    print(
        f"[identify] method={out.method} conf={out.confidence:.3f} manual={out.needs_manual} "
        f"item={name!r} toy={out.toy_number or '-'} upc={out.upc or '-'} | "
        f"top1={top_s} {top_n!r} top2={sec_s} | ocr={ocr!r} | ms={res.timings_ms}",
        flush=True,
    )
    _dbg(
        "resultado",
        f"method={out.method} conf={out.confidence:.3f} manual={out.needs_manual} "
        f"pieza={name!r} toy={out.toy_number or '—'} upc={out.upc or '—'}",
        timings=dict(res.timings_ms or {}),
        top=[
            {"name": m.name, "toy": m.toy, "score": round(m.score, 4)}
            for m in (res.matches or [])[:5]
        ],
    )


# ── Evidencia del EMPAQUE: marca Mini GT + nombre impreso ───────────────────
# Medido 2026-10-04 con 63 fotos reales del usuario (blisters y cajas Mini GT):
# el texto del empaque traía la pieza ("Porsche 911 Dakar Ice Gray Rally 1978",
# "Jaguar C-X75") y aun así ganaba un embed de otra pieza u otra MARCA (Hot
# Wheels, Inno64, Jada 1:20). Estas guardas usan esa evidencia con prudencia:
# solo intervienen ante una CONTRADICCIÓN clara, nunca para inventar.
_MGT_PACK_RE = re.compile(r"\bmin[i!l]?\b|mini\s*g\s*t", re.I)  # el OCR a veces lee «MIN»
_MGT_TOY_RE = re.compile(r"^(MGT|KHMG)", re.I)  # Kaido House x Mini GT va en caja Mini GT
_NAME_STOP = {
    "the", "and", "with", "ver", "version", "edition", "mini", "new", "limited",
    "exclusive", "exclusives", "set", "car", "lhd", "rhd", "shown", "series",
}
_OCR_CONFUSIONS = {"7": "1", "1": "7", "0": "8", "8": "0", "5": "6", "6": "5", "3": "8"}
_NAME_IDX: list[tuple[str, str, str, list[str]]] | None = None


def _norm_txt(s: str) -> str:
    # Sin acentos (catálogo «Huracán» vs OCR «Huracan») y gray→grey.
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().replace("gray", "grey")


def _tok_in(t: str, compact: str) -> bool:
    """Token presente en el OCR, tolerando UNA letra perdida («Metalic», «Raly»)."""
    if t in compact:
        return True
    return len(t) >= 5 and any((t[:i] + t[i + 1:]) in compact for i in range(len(t)))


def _name_tokens(name: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", _norm_txt(name))
    return [
        t for t in toks
        if t not in _NAME_STOP and (len(t) >= 3 or (len(t) >= 2 and any(c.isdigit() for c in t)))
    ]


def _ocr_compact(text: str) -> str:
    # El OCR pega palabras ("Porsche911Dakar"): se compara por subcadena.
    return re.sub(r"[^a-z0-9]", "", _norm_txt(text))


def _name_hits(name: str, compact: str) -> tuple[int, float, int]:
    """(caracteres acertados, cobertura 0-1, nº tokens acertados) del nombre en el OCR."""
    toks = _name_tokens(name)
    tot = sum(len(t) for t in toks)
    hit = [t for t in toks if _tok_in(t, compact)]
    hc = sum(len(t) for t in hit)
    return hc, (hc / tot if tot else 0.0), len(hit)


def _name_index(db) -> list[tuple[str, str, str, list[str]]]:
    global _NAME_IDX
    if _NAME_IDX is None:
        rows = []
        for d in db.variants.find(
            {"toy_number": {"$regex": "^(MGT|KHMG)"}}, {"variant_name": 1, "toy_number": 1}
        ):
            name = d.get("variant_name") or ""
            if len(_name_tokens(name)) >= 2:
                rows.append((d["_id"], d.get("toy_number") or "", name, _name_tokens(name)))
        _NAME_IDX = rows
    return _NAME_IDX


def _best_name_match(db, text: str, mgt_pack: bool) -> tuple[str, str, str, float] | None:
    """Pieza Mini GT cuyo nombre aparece (casi) entero en el texto del empaque.

    Puntuación = caracteres acertados × cobertura: premia el nombre más
    específico («911 Dakar Ice Grey Metallic Rally 1978» sobre «911 Dakar Ice
    Grey Metallic») y el más completo («Z LB★NATION WORKS LB Gold» sobre
    «WORKS GT NISSAN 35GT-RR … Gold Series»). Margen <3 → ambiguo → None.

    Aceptación: ≥3 tokens y cobertura ≥0.8 con logo Mini GT leído (≥0.9 sin
    él); nombres cortos («F1 Red», «Z8 Silver»): 2 tokens al 100 % y con logo.
    El 1er token (el MODELO) tiene que estar.
    """
    compact = _ocr_compact(text)
    if len(compact) < 8:
        return None
    scored = []
    for vid, toy, name, toks in _name_index(db):
        if not _tok_in(toks[0], compact):
            continue
        hc, cov, n = _name_hits(name, compact)
        ok = (n >= 3 and cov >= (0.8 if mgt_pack else 0.9)) or (
            mgt_pack and n == 2 and cov >= 0.999 and hc >= 5
        )
        if ok:
            scored.append((hc * cov, cov, vid, toy, name))
    if not scored:
        return None
    scored.sort(key=lambda r: (r[0], r[1]), reverse=True)
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 3:
        return None
    _sc, cov, vid, toy, name = scored[0]
    return vid, toy, name, cov


def _same_toy(a: str | None, b: str | None) -> bool:
    """MGT00023 == MGT00023-L (mismo número, versión LHD/RHD)."""
    return bool(a) and bool(b) and a.upper()[:8] == b.upper()[:8]


def _toy_confusions(toy: str) -> list[str]:
    """MGT07070 → MGT01070…: una sola sustitución de las confusiones típicas del OCR."""
    m = re.match(r"^(MGT)(\d{5})$", toy or "")
    if not m:
        return []
    digits = m.group(2)
    out = []
    for i, ch in enumerate(digits):
        alt = _OCR_CONFUSIONS.get(ch)
        if alt:
            out.append(m.group(1) + digits[:i] + alt + digits[i + 1:])
    return out


def _run_identify(image_bytes: bytes, brand: str | None = None) -> IdentifyOut:
    if not image_bytes or len(image_bytes) < 100:
        raise HTTPException(status_code=400, detail="Imagen vacía o inválida")
    # Techo de tamaño: sin él entraban 22 MB y el decode+resize se comía la RAM.
    if len(image_bytes) > 12 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Imagen demasiado grande ({len(image_bytes) // (1024*1024)} MB, máx 12 MB)",
        )

    _dbg("petición", f"recibidas {len(image_bytes)} bytes — arranca pipeline de visión")
    try:
        res = identify_image(image_bytes, brand_hint=brand)
    except ValueError as exc:
        # Imagen corrupta, truncada o de 1×1 px: error del CLIENTE, no 500.
        _dbg("error", f"imagen rechazada: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    import time as _time

    _t_r = _time.time()
    db = get_db()

    out = IdentifyOut(
        method="m",
        confidence=res.confidence,
        needs_manual=res.needs_manual,
        upc=res.upc,
        toy_number=res.toy_number,
        suggestions=_to_suggestions(res.matches, db),
        timings_ms=res.timings_ms,
        debug={"ocr_preview": res.ocr_text[:300], "image_colors": res.image_colors},
    )

    # Livery curado → reordena los candidatos ANTES de elegir ganador. Es la
    # señal semántica que falta (dorsal/color/palabras del diseño) y no cuesta
    # latencia: son 8 lookups a Mongo de docs ya curados.
    _t_s = _time.time()
    _apply_livery_score(db, res, res.image_colors)
    _t_l = _time.time()
    out.suggestions = _to_suggestions(res.matches, db)
    _dbg(
        "router",
        f"sugerencias {int((_t_s - _t_r) * 1000)} ms · livery {int((_t_l - _t_s) * 1000)} ms · "
        f"sugerencias2 {int((_time.time() - _t_l) * 1000)} ms",
    )

    # Evidencia del empaque (ver _best_name_match): marca Mini GT y nombre impreso.
    ocr_txt = res.ocr_text or ""
    mgt_pack = bool(_MGT_PACK_RE.search(ocr_txt))
    ocr_compact = _ocr_compact(ocr_txt)
    name_match = _best_name_match(db, ocr_txt, mgt_pack)
    if name_match:
        _dbg("empaque", f"nombre impreso → {name_match[2][:50]!r} ({name_match[1]}, cobertura {name_match[3]:.2f})")

    def _contradicts_pack(doc_name: str | None, doc_toy: str | None) -> bool:
        """El ganador visual choca con el empaque: otra marca, o el empaque nombra
        otra pieza y el nombre del ganador casi no aparece en el texto."""
        if not mgt_pack:
            return False
        if not _MGT_TOY_RE.match(doc_toy or ""):
            return True
        if name_match and not _same_toy(doc_toy, name_match[1]):
            return _pack_beats(doc_name)
        return False

    def _pack_beats(doc_name: str | None) -> bool:
        """El nombre del empaque está ≥0.3 más completo en el texto que el del
        ganador visual. Gemelos con cobertura parecida → no se interviene."""
        return name_match is not None and (
            _name_hits(doc_name or "", ocr_compact)[1] <= name_match[3] - 0.3
        )

    # 1) UPC exacto
    upc_found_but_unmatched = False
    if res.upc:
        _dbg("mongo", f"lookup por UPC {res.upc}…")
        doc, master = _find_variant_by_upc(db, res.upc)
        if doc:
            out.method = "b"
            out.item = _variant_out(doc, master)
            out.needs_manual = False
            out.confidence = 0.99
            _dbg("mongo", f"HIT por UPC → {(doc.get('variant_name') or doc.get('name'))!r}")
            _log_decision(out, res)
            return out
        _dbg("mongo", f"UPC {res.upc} no está en el catálogo")
        upc_found_but_unmatched = True

    # 2) Embedding casi exacto (≥ 0.99) — copia literal del studio shot.
    #    Va ANTES del toy_number de OCR porque el OCR puede leer "126" por "726"
    #    y matchear la pieza equivocada; un embed de 1.000 es más confiable.
    #    PERO el color real de la foto manda: una copia literal de OTRO color
    #    no es esta pieza (foto rosa vs catálogo verde).
    if not res.needs_manual and res.matches:
        top = res.matches[0]
        # Score CRUDO del embed: el de `matches` puede venir inflado por los
        # boosts de marca/dorsal (0.92 → 1.0) y fingir "copia literal".
        top_raw = float(getattr(top, "_raw_score", top.score))
        if top_raw >= 0.99 and _contradicts_pack(top.name, top.toy):
            _dbg("empaque", f"embed ≥0.99 {top.name[:40]!r} choca con el empaque Mini GT → se ignora la copia literal")
            top_raw = 0.0
        if top_raw >= 0.99:
            _dbg("mongo", f"lookup por id del embed {top.id} (raw={top_raw:.3f})…")
            doc, master = _find_variant_by_id(db, top.id)
            if doc and not _color_matches(doc, res.image_colors):
                _dbg("color", f"embed ≥0.99 PERO color choca: catálogo={_doc_color(doc)!r} "
                     f"foto={res.image_colors} → no se auto-confirma")
                doc, master = _find_color_consistent(db, res.matches, res.image_colors)
            if doc:
                out.method = "c"
                out.item = _variant_out(doc, master)
                out.toy_number = top.toy or out.toy_number
                out.needs_manual = False
                out.confidence = float(max(0.0, min(1.0, top.score)))
                _dbg("mongo", f"HIT por embed ≥0.99 → {(doc.get('variant_name') or doc.get('name'))!r}")
                _log_decision(out, res)
                return out
            _dbg("mongo", f"id {top.id} no resolvió en Mongo (se intenta toy/name)")

    # 3) toy_number exacto
    toy_unresolved = False
    if res.toy_number:
        _dbg("mongo", f"lookup por toy_number {res.toy_number}…")
        doc, master = _find_variant_by_toy(db, res.toy_number)
        if doc:
            out.method = "o"
            out.item = _variant_out(doc, master)
            out.needs_manual = False
            out.confidence = 0.95
            _dbg("mongo", f"HIT por toy_number → {(doc.get('variant_name') or doc.get('name'))!r}")
            _log_decision(out, res)
            return out
        _dbg("mongo", f"toy_number {res.toy_number} no está en el catálogo")
        # El OCR confunde 1↔7, 0↔8, 5↔6…: probar UNA sustitución, y aceptarla
        # solo si el nombre de esa pieza también está impreso en el empaque.
        for alt in _toy_confusions(res.toy_number):
            doc, master = _find_variant_by_toy(db, alt)
            if doc and _name_hits(doc.get("variant_name") or "", ocr_compact)[2] >= 2:
                out.method = "o"
                out.item = _variant_out(doc, master)
                out.toy_number = alt
                out.needs_manual = False
                out.confidence = 0.9
                _dbg("mongo", f"toy {res.toy_number} corregido a {alt} (nombre confirmado en el empaque) → {doc.get('variant_name')!r}")
                _log_decision(out, res)
                return out
        toy_unresolved = True

    # Si hubo barcode pero no matcheó en DB, no confiar en el embedding:
    # una imagen de código de barras (mayormente blanca) matchea basura.
    if upc_found_but_unmatched:
        out.method = "m"
        out.needs_manual = True
        out.confidence = 0.5
        _dbg("resultado", "UPC leído pero no está en el catálogo → manual")
        _log_decision(out, res)
        return out

    # 3b) Nombre impreso en el empaque: manda si el visual también lo propone
    #     (top-8), o si contradice al ganador con ≥0.3 más de cobertura.
    top0 = res.matches[0] if res.matches else None
    # Si el visual también lo tiene entre sus candidatos, texto y visual
    # COINCIDEN: es la mejor evidencia posible, aunque sean liveries gemelos.
    # En TODA la lista: el reordenado por livery puede sacar a la pieza nombrada
    # del top-8 (r52: el gemelo «#555 V2 2024» subió ×1.24 y dejó fuera al «#555 2023»).
    nm_in_visual = bool(name_match) and any(_same_toy(m.toy, name_match[1]) for m in res.matches)
    if name_match and (
        nm_in_visual or top0 is None or _pack_beats(top0.name)
    ):
        doc, master = _find_variant_by_id(db, name_match[0])
        if not doc:
            doc, master = _find_variant_by_toy(db, name_match[1])
        if doc:
            out.method = "o"
            out.item = _variant_out(doc, master)
            out.toy_number = name_match[1]
            out.needs_manual = False
            out.confidence = 0.9
            _dbg("mongo", f"HIT por nombre impreso en el empaque → {name_match[2]!r}")
            _log_decision(out, res)
            return out

    # Guardas de honestidad antes de autoconfirmar el ganador visual:
    #  · el empaque Mini GT contradice al ganador (otra marca / otra pieza)
    #  · el empaque traía un número que no resolvió y el ganador no está nombrado
    if not res.needs_manual and res.matches:
        top = res.matches[0]
        if _contradicts_pack(top.name, top.toy):
            _dbg("empaque", f"ganador {top.name[:40]!r} choca con el empaque Mini GT → manual")
            res.needs_manual = True
            # Las sugerencias Mini GT primero (orden estable): el usuario elige
            # entre las piezas de la marca que dice el empaque.
            res.matches.sort(key=lambda m: 0 if _MGT_TOY_RE.match(m.toy or "") else 1)
            out.suggestions = _to_suggestions(res.matches, db)
        elif toy_unresolved and _name_hits(top.name or "", ocr_compact)[1] < 0.4:
            _dbg("empaque", f"el empaque trae el número {res.toy_number} (no resolvió) y el ganador no aparece nombrado → manual")
            res.needs_manual = True

    # 3) Embedding ganador (vision.py ya desambigua liveries con OCR)
    if not res.needs_manual and res.matches:
        top = res.matches[0]
        _dbg("mongo", f"lookup por id/toy/name del ganador {top.id} toy={top.toy or '—'}…")
        doc, master = _find_variant_by_id(db, top.id)
        if not doc and top.toy:
            # ids del índice pueden divergir de Mongo (re-seed / uuid5 distinto)
            # → el toy_number (MGT00065, FLF56…) es estable.
            doc, master = _find_variant_by_toy(db, top.toy)
        # Color de la foto manda: un ganador de otro color no se auto-confirma.
        if doc and not _color_matches(doc, res.image_colors):
            _dbg("color", f"ganador choca en color: catálogo={_doc_color(doc)!r} "
                 f"foto={res.image_colors} → buscando alternativa del color correcto")
            alt, alt_master = _find_color_consistent(db, res.matches, res.image_colors)
            if alt:
                doc, master = alt, alt_master
            else:
                doc, master = None, None
        if doc:
            out.method = "c"
            out.item = _variant_out(doc, master)
            out.toy_number = top.toy or out.toy_number
            out.needs_manual = False
            out.confidence = res.confidence
            _dbg("mongo", f"HIT por embed → {(doc.get('variant_name') or doc.get('name'))!r}")
            _log_decision(out, res)
            return out
        # Último intento: nombre exacto del candidato
        if not doc and top.name:
            _dbg("mongo", f"lookup por nombre {top.name[:50]!r}…")
            doc2 = db.variants.find_one({"variant_name": top.name})
            if not doc2:
                doc2 = db.variants.find_one({"variant_name": {"$regex": re.escape(top.name[:40]), "$options": "i"}})
            if doc2:
                mid = doc2.get("item_master_id")
                master = db.items.find_one({"_id": mid}) if mid else None
                out.method = "c"
                out.item = _variant_out(doc2, master)
                out.toy_number = top.toy or doc2.get("toy_number") or out.toy_number
                out.needs_manual = False
                out.confidence = res.confidence
                _dbg("mongo", f"HIT por nombre → {(doc2.get('variant_name') or doc2.get('name'))!r}")
                _log_decision(out, res)
                return out
        _dbg("mongo", "el ganador del embed NO resolvió en Mongo")

    # Sin match confiable → el cliente muestra sugerencias (sin inventar).
    # PERO si hay varios candidatos que son gemelos del mismo casting, un
    # peritaje semántico del livery puede desampatar. SOLO aquí — nunca en el
    # camino que ya acierta (MiMo tarda 14-17 s). Fail-open: si falla o tarda,
    # nos quedamos EXACTAMENTE como hoy.
    out.method = "m"
    out.needs_manual = True
    # El peritaje ya NO corre aquí: es un endpoint aparte
    # (`POST /identify/peritaje`) que la app llama después de mostrar las
    # sugerencias. Antes quemaba 12-40 s dentro del request y el usuario
    # miraba una pantalla quieta. Se marca en el debug si el caso era apto
    # para que la cliente sepa que puede pedirlo.
    names = [s.name for s in out.suggestions if s.name]
    top_score = res.matches[0].score if res.matches else 0.0
    twins = (
        len(names) >= 2
        and res.matches
        and len(res.matches) >= 2
        and top_score >= 0.75
        and _name_family_safe(res.matches[0].name) == _name_family_safe(res.matches[1].name)
    )
    out.debug["peritaje_available"] = bool(twins)
    if twins:
        out.debug["peritaje_candidates"] = names[:8]
        out.debug["peritaje_hints"] = _livery_hints_for(db, out.suggestions)
        _dbg("peritaje", f"caso apto ({len(names)} gemelos) → se resuelve en el endpoint aparte")
    _log_decision(out, res)
    return out


# Abuso / estabilidad: el pipeline es CPU-bound (~1.5 s). Sin tope, unas pocas
# peticiones en paralelo saturaban la VM y TODAS pasaban de 8 s. Se encolan de
# 2 en 2 (latencia estable) y cada IP tiene un máximo por minuto.
_IDENTIFY_SLOTS = asyncio.Semaphore(2)
_MAX_IMAGE_BYTES = 12 * 1024 * 1024


# Reconciliación con RevenueCat bajo demanda: máximo una por usuario cada 60 s
# (en memoria, un solo proceso).
_RECONCILE_EVERY_S = 60
# RevenueCat desde el identify: corto para no romper el presupuesto de 8 s.
_RECONCILE_TIMEOUT_S = 2.0
_log = logging.getLogger(__name__)
_last_reconcile: dict[str, float] = {}
_reconcile_lock = threading.Lock()


def _reconcile_allowed(user_id: str) -> bool:
    now = time.monotonic()
    with _reconcile_lock:
        for k in [k for k, t in _last_reconcile.items() if now - t > 600]:
            del _last_reconcile[k]
        last = _last_reconcile.get(user_id)
        if last is not None and now - last < _RECONCILE_EVERY_S:
            return False
        _last_reconcile[user_id] = now
        return True


async def _reconcile(db, user_id: str) -> None:
    """Fail-open: si RevenueCat falla se mantiene el estado local."""
    if not _reconcile_allowed(user_id):
        return
    try:
        await asyncio.to_thread(
            revenuecat.sync_user, db, user_id, timeout=_RECONCILE_TIMEOUT_S
        )
    except Exception as exc:  # noqa: BLE001
        _dbg("router", f"reconciliación RevenueCat falló: {exc}")


def _quota_denied() -> JSONResponse:
    limit = get_settings().free_daily_scans
    return JSONResponse(
        status_code=402,
        content={
            "detail": "Llegaste al límite diario de escaneos. Hazte Premium para escanear sin límite.",
            "code": "scan_limit_reached",
            "limit": limit,
            "resets_at": subs.next_reset().isoformat(),
        },
    )


async def _identify_guarded(
    request: Request, data: bytes, brand: Optional[str], user: CurrentUser
):
    hit_or_429("identify", client_ip(request), limit=30, per_seconds=60)
    if len(data) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Imagen muy grande (máx 12 MB)")

    uid = user.id
    st = get_settings()
    if not st.scan_limit_enabled:
        async with _IDENTIFY_SLOTS:
            return await asyncio.to_thread(_run_identify, data, brand)

    db = get_db()
    held = {"reserved": False}  # lo fija el hilo: sobrevive a una cancelación

    def _reserve() -> tuple[bool, int]:
        ok, n = subs.reserve_scan(db, uid, skip_premium_check=True)
        if ok:
            held["reserved"] = True
        return ok, n

    keep = False
    try:
        premium, expired = await asyncio.to_thread(subs.premium_state, db, uid)
        if premium and expired:
            await _reconcile(db, uid)
            premium, _ = await asyncio.to_thread(subs.premium_state, db, uid)
        used = 0
        if not premium:
            ok, used = await asyncio.to_thread(_reserve)
            if not ok:
                await _reconcile(db, uid)
                premium = await asyncio.to_thread(subs.is_premium, db, uid)
                if not premium:
                    ok, used = await asyncio.to_thread(_reserve)
                    if not ok:
                        # Antes del semáforo: un 402 no ocupa slot de CPU.
                        return _quota_denied()
                else:
                    used = 0
        if held["reserved"]:
            # Escaneos gratis por usuario y día: sirve para detectar abuso de cuentas.
            _log.info("scan_free user=%s used=%d/%d", uid, used, st.free_daily_scans)

        async with _IDENTIFY_SLOTS:
            out = await asyncio.to_thread(_run_identify, data, brand)
        if held["reserved"] and out.item is None and not out.suggestions:
            await asyncio.to_thread(subs.release_scan, db, uid)
            held["reserved"] = False
            used = max(0, used - 1)
        out.quota = {"used": used, "limit": st.free_daily_scans, "premium": premium}
        keep = True
        return out
    finally:
        if held["reserved"] and not keep:
            await asyncio.to_thread(subs.release_scan, db, uid)


@router.post("/identify", response_model=IdentifyOut)
async def identify_upload(
    request: Request,
    image: Optional[UploadFile] = File(default=None),
    image_base64: Optional[str] = Form(default=None),
    brand: Optional[str] = Form(default=None),
    user: CurrentUser = Depends(get_current_user),
) -> IdentifyOut:
    """Sube la foto (multipart `image` o form `image_base64`) y recibe la pieza.

    `brand` es opcional: la marca que el usuario declara en el selector de la
    app (Mini GT, Hot Wheels, Tomica…). Manda sobre la que deduce el OCR y
    endurece el prior de marca (+0.20 / ×0.45 en vez de +0.12 / ×0.72).
    """
    data: bytes | None = None
    if image is not None:
        data = await image.read()
    elif image_base64:
        raw = image_base64
        if "," in raw and raw.strip().startswith("data:"):
            raw = raw.split(",", 1)[1]
        try:
            data = base64.b64decode(raw)
        except Exception:
            raise HTTPException(status_code=400, detail="image_base64 inválido")

    if not data:
        raise HTTPException(status_code=400, detail="Falta la imagen (image o image_base64)")

    # En un thread: el event loop queda libre para el SSE de /lab/stream
    # mientras corre el pipeline (que es CPU-bound y largo).
    return await _identify_guarded(request, data, brand, user)


@router.post("/identify/base64", response_model=IdentifyOut)
async def identify_b64(
    body: IdentifyBase64,
    request: Request,
    user: CurrentUser = Depends(get_current_user),
) -> IdentifyOut:
    """Alternativa JSON: { image_base64: '...' }."""
    raw = body.image_base64
    if "," in raw and raw.strip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="image_base64 inválido")
    return await _identify_guarded(request, data, body.brand, user)


@router.post("/identify/peritaje", response_model=PeritajeOut)
async def identify_peritaje(
    body: PeritajeIn,
    user: CurrentUser = Depends(get_current_user),
) -> PeritajeOut:
    """Desempate de livery bajo demanda. Fail-open: devuelve pick=None.

    Llama a un modelo de pago: exige sesión y tiene tope por usuario.
    """
    hit_or_429("peritaje", user.id, limit=10, per_seconds=600)
    try:
        data = base64.b64decode(body.image_base64.split(",", 1)[-1])
    except Exception:
        raise HTTPException(status_code=400, detail="image_base64 inválido")
    verdict = await asyncio.to_thread(
        _livery_pick, data, [c for c in body.candidates if c], body.livery_hints or None
    )
    if not verdict:
        return PeritajeOut()
    return PeritajeOut(
        pick=verdict.get("pick"),
        color=verdict.get("color"),
        livery=verdict.get("livery"),
        confidence=verdict.get("confidence"),
        why=verdict.get("why"),
        ms=int(verdict.get("_ms") or 0),
    )
