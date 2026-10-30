"""Autocompletado de catálogo: alta de piezas ausentes a partir de una lectura
de grado catálogo (`claude_vision.read_for_catalog`).

El catálogo es COMPARTIDO: un alta errónea la ven todos. Por eso las barreras
(`gate`, `find_duplicate`) se inclinan a NO crear. Este módulo no importa de
`app.routers.identify` (el identify lo importa a él).

Uso previsto (Task 5): `gate` → `daily_cap_reached` → `create_from_read` →
`consume_daily` si `created` → `strip_internal` antes de responder a la app.
"""
from __future__ import annotations

import hashlib
import io
import logging
import re
import unicodedata
import uuid
from datetime import datetime, timezone

import numpy as np

from app.config import get_settings
from app.db import as_id

_log = logging.getLogger(__name__)

# Espacio de nombres uuid5 de TODO el catálogo. NO cambiar.
_NS = uuid.UUID("a9f73cdf-180a-5bd7-ac02-42c291653666")

# Slugs de marca iguales a los de los scrapers (scripts/gen_diecast_extra_catalog.py,
# scrape_minigt_catalog.py). Clave: nombre de marca sin espacios ni signos.
_BRAND_SLUGS = {
    "minigt": "minigt",
    "inno64": "inno64",
    "tarmacworks": "tarmac-works",
    "kaidohouse": "kaidohouse",
    "poprace": "pop-race",
    "tomica": "tomica",
    "hotwheels": "hw",
}

# Campos internos de moderación: nunca salen hacia la app.
_INTERNAL = ("origin", "verified", "source_confidence", "confirmed_by", "created_by", "hidden", "created")

_YEAR_MIN = 1950


# ── Normalización e identidad ──────────────────────────────────────────────


def _fold(text) -> str:
    """Minúsculas, sin acentos; todo lo que no sea letra/dígito → espacio."""
    s = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _toks(*parts) -> set[str]:
    return {t for p in parts for t in _fold(p).split() if t}


def normalize_key(brand, casting, color, dorsal, livery) -> str:
    """Clave estable de identidad: tokens únicos ordenados (el orden de lectura no cuenta)."""
    return " ".join(sorted(_toks(brand, casting, color, dorsal, livery)))


def _brand_slug(brand) -> str:
    compact = _fold(brand).replace(" ", "")
    return _BRAND_SLUGS.get(compact, "-".join(_fold(brand).split()) or "unknown")


def _norm_toy(read: dict) -> str | None:
    raw = re.sub(r"\s+", "", str(read.get("toy_number") or "")).upper()
    raw = raw.lstrip("▶>")
    if not raw:
        return None
    if _brand_slug(read.get("brand")) == "minigt":
        m = re.fullmatch(r"(?:MGT)?0*(\d{1,5})", raw)
        if m:
            return f"MGT{int(m.group(1)):05d}"
    return raw


def make_source_id(read: dict) -> str:
    slug = _brand_slug(read.get("brand"))
    toy = _norm_toy(read)
    if toy:
        return f"{slug}-{toy}"
    key = normalize_key(read.get("brand"), read.get("casting"), read.get("color"),
                        read.get("dorsal"), read.get("livery"))
    return f"ai-{slug}-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"


def piece_id(source_id: str) -> str:
    return str(uuid.uuid5(_NS, source_id))


# ── Barreras ───────────────────────────────────────────────────────────────


def gate(read: dict | None, receipt: dict) -> str | None:
    """Motivo de rechazo o None si se puede crear.

    Un `top_score` ausente/0.0 (el identify salió a manual antes de puntuar)
    NO es motivo para crear: siguen mandando diecast + confianza alta + identidad."""
    if not read or read.get("es_diecast") is not True:
        return "no_diecast"
    if read.get("confidence") != "alto":
        return "low_confidence"
    if not str(read.get("brand") or "").strip() or not str(read.get("casting") or "").strip():
        return "missing_identity"
    rc = receipt or {}
    try:
        top = float(rc.get("top_score") or 0.0)
    except (TypeError, ValueError):
        top = 0.0
    if rc.get("twins") or top >= get_settings().ai_catalog_max_top_score:
        return "visual_candidate"
    return None


def _load_master(db, doc: dict) -> dict | None:
    from app.routers.catalog import _load_master as _lm

    return _lm(db, doc)


def _piece_out(doc: dict, master: dict | None, created: bool) -> dict:
    """Variante con el contrato de la app (`ItemVariantOut` + `item_master`)
    más los campos internos del doc y `created`. Pasar por `strip_internal`."""
    from app.routers.catalog import _variant_out

    out = _variant_out(doc, master).model_dump()
    for k in _INTERNAL:
        if k in doc:
            out[k] = doc[k]
    out["created"] = created
    return out


def _resolve_variant(db, vid) -> dict | None:
    if not vid:
        return None
    for q in ({"_id": as_id(vid)}, {"_id": vid}):
        try:
            doc = db.variants.find_one(q)
        except Exception:  # noqa: BLE001
            doc = None
        if doc:
            return doc
    return None


def find_duplicate(db, read: dict, crop_vec: list[float] | None) -> dict | None:
    """Pieza ya presente en el catálogo (o None). Conservador: ante la duda, existe.

    Orden: id determinista → toy# legible → texto (marca+casting+livery/color)
    → embedding del recorte contra el índice completo (vistas aisladas incluidas).
    Los fallos del índice se propagan: sin poder comprobar, no se crea."""
    st = get_settings()
    doc = _resolve_variant(db, piece_id(make_source_id(read)))
    if doc is None:
        # Los ids del scrape no salen del toy# (Mini GT: `minigt-<id del sitio>`),
        # así que el toy# legible se busca también como campo.
        toy = _norm_toy(read)
        if toy:
            doc = db.variants.find_one({"toy_number": toy})
    if doc is None:
        doc = _dup_by_text(db, read, st.ai_catalog_dup_text_overlap)
    if doc is None and crop_vec is not None:
        from app.services import vision

        hits = vision.match_embedding(np.asarray(crop_vec, dtype=np.float32), top_k=3, include_mv=True)
        if hits and hits[0].score >= st.ai_catalog_dup_embed_score:
            doc = _resolve_variant(db, hits[0].id)
    if doc is None:
        return None
    return _piece_out(doc, _load_master(db, doc), created=False)


def _dup_by_text(db, read: dict, threshold: float) -> dict | None:
    """Candidatos de `vision.search_catalog_by_text`, aceptados solo si casi
    todo su nombre está en la lectura (cobertura ≥ umbral) y comparten un
    token del casting. Un livery gemelo («Midnight Purple» frente a «Bayside
    Blue») no llega al umbral."""
    from app.services import vision

    brand, casting = read.get("brand"), read.get("casting")
    livery = read.get("livery") or read.get("color")
    query = " ".join(str(p) for p in (brand, casting, read.get("modelo"), livery, read.get("dorsal")) if p)
    read_toks = _toks(brand, casting, read.get("modelo"), read.get("livery"), read.get("color"), read.get("dorsal"))
    cast_toks = _toks(casting, read.get("modelo")) - _toks(brand)
    for m in vision.search_catalog_by_text(query, top_k=5):
        name_toks = _toks(m.name)
        if len(name_toks) < 2 or not (name_toks & cast_toks):
            continue
        cover = len(name_toks & read_toks) / len(name_toks)
        if cover >= threshold:
            doc = _resolve_variant(db, m.id)
            if doc:
                return doc
    return None


# ── Tope diario ────────────────────────────────────────────────────────────


def daily_cap_reached(db, user_id: str) -> bool:
    from app.services.subscription import today_key

    doc = db.ai_catalog_usage.find_one({"user_id": user_id, "day": today_key()})
    return int((doc or {}).get("count") or 0) >= get_settings().ai_catalog_daily_cap


def consume_daily(db, user_id: str) -> None:
    from app.services.subscription import today_key

    db.ai_catalog_usage.update_one(
        {"user_id": user_id, "day": today_key()}, {"$inc": {"count": 1}}, upsert=True
    )


# ── Alta ───────────────────────────────────────────────────────────────────


def _valid_year(y) -> int | None:
    if isinstance(y, bool) or not isinstance(y, int):
        return None
    return y if _YEAR_MIN <= y <= datetime.now(timezone.utc).year + 1 else None


def _clean(v) -> str | None:
    s = str(v).strip() if v is not None else ""
    return s or None


def _save_crop(img) -> str | None:
    """Guarda el recorte (nunca la foto cruda) y devuelve su URL pública."""
    from app.routers import uploads

    try:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=90)
        filename, _ = uploads._save_bytes(buf.getvalue(), ".jpg", "aicat")
        return f"/api/v1/uploads/{filename}"
    except Exception:  # noqa: BLE001
        _log.warning("ai_catalog: no se pudo guardar el recorte", exc_info=True)
        return None


def _append_live(entry: dict, vec) -> None:
    from app.services import vision

    try:
        vision.live_index_append(entry, list(vec))
    except Exception:  # noqa: BLE001
        _log.warning("ai_catalog: no se pudo añadir al índice vivo (%s)", entry, exc_info=True)


def create_from_read(db, user_id: str, read: dict, image_bytes: bytes) -> dict:
    """Crea la pieza (o devuelve la existente). Salida: ver `_piece_out`.

    No comprueba `gate` ni el tope: eso es del llamador."""
    from PIL import Image

    from app.routers.catalog import _upsert_catalog_piece
    from app.services import car_segment, dinov2_embed

    im = Image.open(io.BytesIO(image_bytes))
    im.load()
    im = im.convert("RGB")

    seg = car_segment.isolate_car(im)
    crop = seg.image if seg is not None else None
    crop_vec = dinov2_embed.embed_image(crop) if crop is not None else None

    dup = find_duplicate(db, read, crop_vec)
    if dup is not None:
        return dup

    brand = _clean(read.get("brand"))
    casting = _clean(read.get("casting"))
    color = _clean(read.get("color"))
    dorsal = _clean(read.get("dorsal"))
    livery = _clean(read.get("livery"))
    year = _valid_year(read.get("year"))
    series = _clean(read.get("serie"))
    toy = _norm_toy(read)
    source_id = make_source_id(read)
    vid = piece_id(source_id)

    url = _save_crop(crop) if crop is not None else None
    metadata: dict = {"scale": "1/64", "source_id": source_id}
    if year is not None:
        metadata["year"] = year
    if series:
        metadata["series"] = series

    variant = {
        "id": vid,
        "ai_local_hash": "ai-" + hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:20],
        "variant_name": livery or color or casting,
        "toy_number": toy,
        "main_image_url": url,
        "images": [url] if url else [],
        "body_color": color,
        "livery_desc": livery,
        "livery_color": color,
        "livery_number": dorsal,
        "metadata": metadata,
        "rarity": "regular",
        "base_material": "Metal/Plastic",
        "origin": "ai",
        "verified": False,
        "source_confidence": read.get("confidence"),
        "confirmed_by": [],
    }
    # `items.release_year` es obligatorio en el esquema: 0 = año desconocido
    # (la app lo trata como falso y las facetas/orden por año lo ignoran).
    master = {
        "name": casting,
        "manufacturer": brand,
        "release_year": year if year is not None else 0,
        "casting_name": casting,
        "scale": "1:64",
    }
    doc, master_doc, created = _upsert_catalog_piece(db, master, variant, user_id, dedupe_id=True)

    if created:
        name = " ".join(p for p in (brand, casting, livery or color) if p)
        full_vec = dinov2_embed.embed_image(im)
        if full_vec is not None:
            _append_live({"id": vid, "toy": toy, "name": name, "view": 0}, full_vec)
        if crop_vec is not None:
            _append_live({"id": vid, "toy": toy, "name": name, "view": 20}, crop_vec)
    return _piece_out(doc, master_doc, created=created)


def strip_internal(doc: dict) -> dict:
    """Copia sin los campos internos de moderación (no muta la entrada)."""
    return {k: v for k, v in doc.items() if k not in _INTERNAL}
