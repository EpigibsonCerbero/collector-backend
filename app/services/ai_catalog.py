"""Autocompletado de catálogo: alta de piezas ausentes a partir de una lectura
de grado catálogo (`claude_vision.read_for_catalog`).

El catálogo es COMPARTIDO: un alta errónea la ven todos. Por eso las barreras
(`gate`, `find_duplicate`) se inclinan a NO crear. Este módulo no importa de
`app.routers.identify` (el identify lo importa a él).

Uso previsto (Task 5): `gate` → `daily_cap_reached` → `create_from_read` →
según `outcome`: "created" (`consume_daily` y responder), "duplicate"
(responder con la existente) o "uncertain" (no se creó nada: flujo manual).
`strip_internal` antes de responder a la app.
"""
from __future__ import annotations

import hashlib
import io
import logging
import re
import unicodedata
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from pymongo import ReturnDocument

from app.config import get_settings
from app.db import as_id

_log = logging.getLogger(__name__)

# Espacio de nombres uuid5 de TODO el catálogo. NO cambiar.
_NS = uuid.UUID("a9f73cdf-180a-5bd7-ac02-42c291653666")

# Slugs de marca iguales a los de los scrapers (scripts/gen_diecast_extra_catalog.py,
# scrape_minigt_catalog.py). Se busca la raíz dentro del nombre compacto, así
# «TSM Mini GT», «MiniGT» y «Mini GT» caen en el mismo slug.
_BRAND_ALIASES = (
    ("minigt", "minigt"),
    ("hotwheels", "hw"),
    ("inno64", "inno64"),
    ("tarmac", "tarmac-works"),
    ("kaido", "kaidohouse"),
    ("poprace", "pop-race"),
    ("tomica", "tomica"),
)

# Solo Mini GT tiene un número de pieza único y validable (MGT + 5 dígitos).
# En el resto de marcas el número leído es metadato, nunca identidad.
_TOY_ID_BRANDS = {"minigt"}

# Campos de moderación/origen de una pieza: nunca salen hacia la app. Lista
# ÚNICA: la usan `strip_internal`, `confirm` y el `_variant_out` del identify.
MODERATION_FIELDS = (
    "origin", "verified", "source_confidence", "confirmed_by", "created_by", "hidden",
    "hidden_at", "user_submitted", "source_key",
)
# Además, campos de resultado del alta (solo viven en la respuesta interna).
_INTERNAL = MODERATION_FIELDS + ("created", "outcome", "reason")

# Presupuesto de la búsqueda textual anti-duplicado. Corre fuera del identify
# (no cuenta contra sus 8 s); si aun así se agota, el resultado es «incierto».
DUP_TEXT_BUDGET_S = 3.0

# Colores base (es/en) → canónico, para comparar lecturas con el catálogo.
_COLORS = {
    "azul": "blue", "blue": "blue", "rojo": "red", "red": "red", "negro": "black",
    "black": "black", "blanco": "white", "white": "white", "plata": "silver",
    "plateado": "silver", "silver": "silver", "gris": "grey", "grey": "grey",
    "gray": "grey", "verde": "green", "green": "green", "amarillo": "yellow",
    "yellow": "yellow", "naranja": "orange", "orange": "orange", "morado": "purple",
    "purpura": "purple", "purple": "purple", "violeta": "purple", "rosa": "pink",
    "pink": "pink", "cafe": "brown", "marron": "brown", "brown": "brown",
    "dorado": "gold", "oro": "gold", "gold": "gold", "turquesa": "teal", "teal": "teal",
}

# Cobertura mínima entre las descripciones de livery para darlas por iguales.
_LIVERY_MATCH = 0.6

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
    for root, slug in _BRAND_ALIASES:
        if root in compact:
            return slug
    return "-".join(_fold(brand).split()) or "unknown"


def _raw_toy(read: dict) -> str | None:
    raw = re.sub(r"\s+", "", str(read.get("toy_number") or "")).upper().lstrip("▶>")
    return raw or None


def _norm_toy(read: dict) -> str | None:
    """Número de pieza con valor de IDENTIDAD, o None.

    Solo Mini GT con forma validable (MGT00692, ▶692, 692). Cualquier otra
    marca o forma → None (se guarda aparte como metadato)."""
    raw = _raw_toy(read)
    if not raw or _brand_slug(read.get("brand")) not in _TOY_ID_BRANDS:
        return None
    m = re.fullmatch(r"(?:MGT)?0*(\d{1,5})", raw)
    if not m or not 1 <= int(m.group(1)) <= get_settings().ai_catalog_mgt_max:
        return None
    return f"MGT{int(m.group(1)):05d}"


# ── Saneado del texto libre del modelo ─────────────────────────────────────

# Campos de la lectura que acaban en el catálogo (o en sus consultas) y su
# longitud máxima. `texto` (todo el empaque) no se guarda: no se revisa.
_TEXT_CAPS = {
    "brand": 80, "casting": 80, "modelo": 80, "color": 80, "dorsal": 80,
    "livery": 200, "serie": 80, "toy_number": 80,
}
VARIANT_NAME_MAX = 150
_URL_RE = re.compile(r"https?|www\.|://", re.I)
_MARKUP_RE = re.compile(r"<\s*/?\s*[a-z!?]|&(?:#\d+|#x[0-9a-f]+|[a-z]+);", re.I)
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _clean_text(v, cap: int) -> str | None:
    s = _CTRL_RE.sub(" ", str(v)).replace("<", " ").replace(">", " ")
    s = re.sub(r"\s+", " ", s).strip()[:cap].strip()
    return s or None


def sanitize_read(read: dict | None) -> dict | None:
    """Copia saneada de la lectura, o None si hay que rechazarla.

    Rechazo (= barrera, no se crea nada): algún campo guardable trae algo con
    forma de URL (`http`, `www.`, `://`) o de marcado HTML. Si no: sin
    caracteres de control ni `<>`, espacios colapsados y longitudes tope.
    Un número Mini GT fuera de 1..`ai_catalog_mgt_max` se ignora (None).
    El año solo sobrevive si aparece en el texto leído (el modelo lo inventa)."""
    if not read:
        return read
    out = dict(read)
    for k, cap in _TEXT_CAPS.items():
        v = out.get(k)
        if v is None:
            continue
        if _URL_RE.search(str(v)) or _MARKUP_RE.search(str(v)):
            return None
        out[k] = _clean_text(v, cap)
    if out.get("toy_number") and _brand_slug(out.get("brand")) in _TOY_ID_BRANDS and _norm_toy(out) is None:
        out["toy_number"] = None
    year = out.get("year")
    if isinstance(year, int) and not isinstance(year, bool):
        texto = str(out.get("texto") or "")
        alnum = sum(ch.isalnum() for ch in texto)
        has_anchor = bool(out.get("toy_number")) or bool(out.get("serie")) or alnum >= 4
        if not has_anchor:
            # Sin evidencia física del objeto, el año lo suele inventar el modelo.
            out["year"] = None
    return out


def make_source_id(read: dict) -> str:
    slug = _brand_slug(read.get("brand"))
    toy = _norm_toy(read)
    if toy:
        return f"{slug}-{toy}"
    # Con livery, el color no entra en la clave (la IA lo redacta a su manera).
    livery = read.get("livery")
    key = normalize_key(_brand_slug(read.get("brand")), read.get("casting"),
                        None if livery else read.get("color"), read.get("dorsal"), livery)
    # Prefijo neutro: la semilla no delata el origen (el id sale de aquí).
    return f"x-{slug}-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"


def piece_id(source_id: str) -> str:
    return str(uuid.uuid5(_NS, source_id))


# ── Barreras ───────────────────────────────────────────────────────────────


def gate_receipt(receipt: dict | None) -> str | None:
    """Parte de la barrera que solo necesita el recibo (sin IA): gemelos o un
    candidato visual creíble → "visual_candidate". Va ANTES de pagar la lectura."""
    rc = receipt or {}
    try:
        top = float(rc.get("top_score") or 0.0)
    except (TypeError, ValueError):
        top = 0.0
    if rc.get("twins") or top >= get_settings().ai_catalog_max_top_score:
        return "visual_candidate"
    return None


def gate_read(read: dict | None) -> str | None:
    """Parte de la barrera que necesita la lectura: diecast, confianza, identidad.

    El orden importa: identidad ANTES que confianza. Un `missing_identity` no
    se rescata con el verificador; un `low_confidence` sí (si las lecturas
    coinciden). Al revés, una lectura «medio» sin marca ni modelo se colaba.
    """
    if not read or read.get("es_diecast") is not True:
        return "no_diecast"
    if not str(read.get("brand") or "").strip() or not str(read.get("casting") or "").strip():
        return "missing_identity"
    if read.get("confidence") != "alto":
        return "low_confidence"
    return None


# Ancla mínima de identidad para CREAR: sin evidencia física del objeto, no se
# da de alta. Un render o una foto IA puede parecer un coche y mentir sobre la
# marca; el texto impreso o el número de empaque no se inventan tan fácil.
_MIN_ANCHOR_CHARS = 4


def gate_create(read: dict | None) -> str | None:
    """Barrera de ALTA (más estricta que `gate_read`). Motivo o None.

    Rechaza además:
      - `not_foto_real`: render, ilustración, foto generada por IA o coche real
        a escala 1:1 (no un juguete físico).
      - `no_anchor`: sin toy_number, serie ni texto legible (≥ `_MIN_ANCHOR_CHARS`
        alfanuméricos). Es la barrera que frena un render sin empaque.
    """
    reason = gate_read(read)
    if reason:
        return reason
    if read.get("es_foto_real") is not True:
        return "not_foto_real"
    toy = str(read.get("toy_number") or "").strip()
    serie = str(read.get("serie") or "").strip()
    texto = str(read.get("texto") or "").strip()
    alnum = sum(ch.isalnum() for ch in texto)
    if not toy and not serie and alnum < _MIN_ANCHOR_CHARS:
        return "no_anchor"
    return None


def gate(read: dict | None, receipt: dict) -> str | None:
    """Motivo de rechazo o None si se puede crear (`gate_read` + `gate_receipt`).

    Un `top_score` ausente/0.0 (el identify salió a manual antes de puntuar)
    NO es motivo para crear: siguen mandando diecast + confianza alta + identidad."""
    return gate_read(read) or gate_receipt(receipt)


def _load_master(db, doc: dict) -> dict | None:
    from app.routers.catalog import _load_master as _lm

    return _lm(db, doc)


def _piece_out(doc: dict, master: dict | None, created: bool) -> dict:
    """Variante con el contrato de la app (`ItemVariantOut` + `item_master`)
    más los campos internos del doc y `created`. Pasar por `strip_internal`.

    No deja que un master a medias (datos previos rotos) tumbe la respuesta
    cuando la pieza YA está insertada: degrada a un dict mínimo.
    """
    from app.routers.catalog import _variant_out

    try:
        out = _variant_out(doc, master).model_dump()
    except Exception:  # noqa: BLE001
        out = {
            "id": doc.get("id") or str(doc.get("_id")),
            "variant_name": doc.get("variant_name"),
            "toy_number": doc.get("toy_number"),
            "main_image_url": doc.get("main_image_url"),
            "images": doc.get("images") or [],
            "body_color": doc.get("body_color"),
        }
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


@dataclass
class DupCheck:
    """Resultado de la comprobación anti-duplicado.

    outcome: "duplicate" (piece = la existente), "uncertain" (no crear) o
    "new" (se puede crear). `reason` explica el porqué."""

    outcome: str
    reason: str
    piece: dict | None = None


def _casting_toks(read_or_casting, brand=None) -> set[str]:
    if isinstance(read_or_casting, dict):
        r = read_or_casting
        toks = _toks(r.get("casting"), r.get("modelo"))
        brand = r.get("brand")
    else:
        toks = _toks(read_or_casting)
    return toks - _toks(brand) - {"tsm"}


def _casting_match(a: set[str], b: set[str]) -> bool:
    """Mismo casting: uno contiene al otro, o casi todo coincide y los tokens
    con dígitos (R33/R34, 911/930) son idénticos."""
    if not a or not b:
        return False
    if a <= b or b <= a:
        return True
    da = {t for t in a if any(c.isdigit() for c in t)}
    db_ = {t for t in b if any(c.isdigit() for c in t)}
    return da == db_ and len(a & b) / min(len(a), len(b)) >= 0.8


def _colors(*parts) -> set[str]:
    return {_COLORS[t] for t in _toks(*parts) if t in _COLORS}


def _dorsal(v) -> str | None:
    d = re.sub(r"\D+", "", str(v or ""))
    return d.lstrip("0") or (d and "0") or None


def _livery_verdict(read: dict, doc: dict) -> str:
    """"match" | "conflict" | "ambiguous" entre la lectura y una variante del
    mismo casting. La duda es "ambiguous" (no se crea): un duplicado es peor
    que una pieza que falta.

    - livery fuerte (núcleos ≥ 0.6): "match", salvo dorsal distinto ("conflict").
      El color no cuenta: la IA nombra distinto el color dominante de un livery
      multicolor.
    - liveries disjuntos: "conflict" solo si además hay color en AMBOS lados y
      difiere, o el dorsal difiere; si no, "ambiguous".
    - ninguno trae livery: "match" si colores (ambos) y dorsal coinciden.
    - todo lo demás: "ambiguous".

    El livery se compara sin palabras de color (la IA dice «azul», el scrape
    «Blue»); el color va aparte, canonizado es/en."""
    rd, dd = _dorsal(read.get("dorsal")), _dorsal(doc.get("livery_number"))
    dorsal_conflict = bool(rd and dd and rd != dd)
    rl = _toks(read.get("livery"))
    if doc.get("livery_desc"):
        dl = _toks(doc.get("livery_desc"))
    elif doc.get("origin") != "ai":
        dl = _toks(doc.get("variant_name"))  # en el scrape, variant_name = livery
    else:
        dl = set()  # pieza IA sin livery: su variant_name es color o casting
    rc = _colors(read.get("color"), read.get("livery"))
    dc = _colors(doc.get("body_color"), doc.get("livery_color"), " ".join(dl))
    color_diff = bool(rc and dc and not (rc & dc))
    rcore = {t for t in rl if t not in _COLORS}
    dcore = {t for t in dl if t not in _COLORS}
    if rcore and dcore:
        inter = rcore & dcore
        if inter and len(inter) / min(len(rcore), len(dcore)) >= _LIVERY_MATCH:
            return "conflict" if dorsal_conflict else "match"
        if not inter and (color_diff or dorsal_conflict):
            return "conflict"
        return "ambiguous"
    if not rl and not dl and rc and dc and not color_diff and rd == dd:
        return "match"  # ninguna trae livery, los colores y el dorsal coinciden
    return "ambiguous"


# Tope de documentos que revisa la búsqueda por casting. Si se alcanza, la
# comprobación no es concluyente → «incierto».
SAME_CASTING_SCAN_LIMIT = 500


def _same_casting_variants(db, read: dict) -> tuple[list[dict], bool]:
    """(variantes de la MISMA marca cuyo casting coincide, tope alcanzado)."""
    limit = SAME_CASTING_SCAN_LIMIT
    slug = _brand_slug(read.get("brand"))
    ct = _casting_toks(read)
    if not ct:
        return [], False
    anchor = max(sorted(ct), key=len)  # determinista
    rx = {"$regex": re.escape(anchor), "$options": "i"}
    masters = list(db.items.find({"$or": [{"name": rx}, {"casting_name": rx}]}).limit(limit))
    capped = len(masters) >= limit
    mids = []
    for m in masters:
        if _brand_slug(m.get("manufacturer")) != slug:
            continue
        mt = _casting_toks(m.get("casting_name") or m.get("name"), m.get("manufacturer"))
        if _casting_match(ct, mt):
            mids.append(str(m["_id"]))
    if not mids:
        return [], capped
    variants = list(db.variants.find({"item_master_id": {"$in": mids}}).limit(limit))
    return variants, capped or len(variants) >= limit


def check_duplicate(db, read: dict, crop_vec: list[float] | None) -> DupCheck:
    """Tres salidas: duplicate / uncertain / new. Ante la duda, NO crear.

    Orden: id determinista → toy# Mini GT (con marca y casting de acuerdo) →
    variantes de la misma marca y casting (livery/dorsal/color) → texto →
    embedding del recorte contra el índice completo (vistas aisladas incluidas).
    Los fallos del índice se propagan: sin poder comprobar, no se crea."""
    st = get_settings()

    def dup(doc: dict, reason: str) -> DupCheck:
        if doc.get("hidden"):
            # Pieza oculta por el dueño: ni se devuelve como existente ni se
            # recrea (el mismo id volvería a pisarla).
            return DupCheck("uncertain", "hidden_exists")
        return DupCheck("duplicate", reason, _piece_out(doc, _load_master(db, doc), created=False))

    doc = _resolve_variant(db, piece_id(make_source_id(read)))
    if doc is not None:
        return dup(doc, "same_id")

    slug = _brand_slug(read.get("brand"))
    ct = _casting_toks(read)
    toy = _norm_toy(read)
    if toy:
        # Los ids del scrape no salen del toy# (`minigt-<id del sitio>`): se busca
        # como campo, pero solo vale si la marca y el casting impresos coinciden.
        hits = list(db.variants.find({"toy_number": toy}).limit(20))
        for h in hits:
            m = _load_master(db, h) or {}
            if _brand_slug(m.get("manufacturer")) == slug and _casting_match(
                ct, _casting_toks(m.get("casting_name") or m.get("name"), m.get("manufacturer"))
            ):
                return dup(h, "same_toy")
        if hits:
            return DupCheck("uncertain", "toy_name_mismatch")

    same, capped = _same_casting_variants(db, read)
    verdicts = [(_livery_verdict(read, v), v) for v in same]
    verdicts.sort(key=lambda t: bool(t[1].get("hidden")))  # una visible gana a una oculta
    for verdict, v in verdicts:
        if verdict == "match":
            return dup(v, "same_casting_livery")
    if capped:
        return DupCheck("uncertain", "same_casting_scan_capped")
    if any(verdict == "ambiguous" for verdict, _ in verdicts):
        return DupCheck("uncertain", "same_casting_ambiguous")

    doc, truncated = _dup_by_text(db, read, st.ai_catalog_dup_text_overlap)
    if doc is not None:
        return dup(doc, "text")
    if truncated:
        return DupCheck("uncertain", "text_search_truncated")

    if crop_vec is not None:
        from app.services import vision

        hits = vision.match_embedding(np.asarray(crop_vec, dtype=np.float32), top_k=3, include_mv=True)
        for h in hits:
            if h.score < st.ai_catalog_dup_embed_score:
                break
            doc = _resolve_variant(db, h.id)
            if doc is not None:
                return dup(doc, "embedding")
    return DupCheck("new", "no_match")


def find_duplicate(db, read: dict, crop_vec: list[float] | None) -> dict | None:
    """Pieza existente o None. OJO: None NO significa «se puede crear» (puede
    ser incierto). Para decidir un alta (Task 5) usar `check_duplicate` o
    `create_from_read`, que distinguen duplicate / uncertain / new."""
    return check_duplicate(db, read, crop_vec).piece


def _dup_by_text(db, read: dict, threshold: float) -> tuple[dict | None, bool]:
    """(variante, truncada). Candidatos de `vision.search_catalog_by_text`,
    aceptados solo si casi todo su nombre está en la lectura (cobertura ≥
    umbral) y comparten un token del casting. Un livery gemelo («Midnight
    Purple» frente a «Bayside Blue») no llega al umbral. `truncada` = la
    búsqueda agotó su presupuesto (resultado parcial, no concluyente)."""
    from app.services import vision

    brand, casting = read.get("brand"), read.get("casting")
    livery = read.get("livery") or read.get("color")
    query = " ".join(str(p) for p in (brand, casting, read.get("modelo"), livery, read.get("dorsal")) if p)
    read_toks = _toks(brand, casting, read.get("modelo"), read.get("livery"), read.get("color"), read.get("dorsal"))
    cast_toks = _casting_toks(read)
    stats: dict = {}
    t0 = _clock()
    cands = vision.search_catalog_by_text(query, top_k=5, budget_s=DUP_TEXT_BUDGET_S, stats=stats)
    truncated = bool(stats.get("truncated")) or (_clock() - t0) >= DUP_TEXT_BUDGET_S
    for m in cands:
        name_toks = _toks(m.name)
        if len(name_toks) < 2 or not (name_toks & cast_toks):
            continue
        cover = len(name_toks & read_toks) / len(name_toks)
        if cover >= threshold:
            doc = _resolve_variant(db, m.id)
            if doc:
                return doc, truncated
    return None, truncated


def _clock() -> float:
    return time.monotonic()


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


# Contador global de lecturas pagadas (misma colección, usuario ficticio).
GLOBAL_USAGE_ID = "__global__"


def _bump_calls(db, user_id: str, day: str) -> int:
    doc = db.ai_catalog_usage.find_one_and_update(
        {"user_id": user_id, "day": day},
        {"$inc": {"ai_calls": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return int((doc or {}).get("ai_calls") or 0)


def reserve_ai_call(db, user_id: str) -> bool:
    """Cuenta UNA lectura pagada antes de hacerla. False = tope alcanzado
    (del usuario o global): no se llama a la IA. El incremento es atómico; al
    pasarse, el contador queda por encima del tope (inofensivo)."""
    from app.services.subscription import today_key

    st = get_settings()
    day = today_key()
    if _bump_calls(db, str(user_id), day) > st.ai_catalog_daily_ai_calls:
        return False
    return _bump_calls(db, GLOBAL_USAGE_ID, day) <= st.ai_catalog_global_daily_calls


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
        filename, _ = uploads._save_bytes(buf.getvalue(), ".jpg", "img")
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


@dataclass
class Prepared:
    """Trabajo de CPU sobre la foto (decodificar, aislar el coche, embedding
    del recorte). Es lo único del alta que debe ir bajo `_IDENTIFY_SLOTS`."""

    im: object
    crop: object | None
    crop_vec: list[float] | None
    # El recorte solo se publica si YOLO ve un vehículo sobre él.
    publishable: bool = False


# Fracción mínima de la caja más pequeña (recorte o vehículo) que debe
# solaparse para dar el recorte por «es un coche».
_CAR_OVERLAP = 0.5


def _overlap(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (ix * iy) / area if area > 0 else 0.0


def _crop_is_car(im, seg) -> bool:
    """El recorte (saliencia) es un coche: YOLO detecta un vehículo que se
    solapa con su caja. Sin YOLO, con error o sin detección → False."""
    bbox = getattr(seg, "bbox", None)
    if not bbox:
        return False
    from app.services import yolo_crop

    try:
        boxes = yolo_crop.vehicle_boxes(im)
    except Exception:  # noqa: BLE001
        _log.warning("ai_catalog: YOLO falló al validar el recorte", exc_info=True)
        return False
    return any(_overlap(b, bbox) >= _CAR_OVERLAP for b in boxes or [])


def prepare_image(image_bytes: bytes) -> Prepared:
    from PIL import Image

    from app.services import car_segment, dinov2_embed

    im = Image.open(io.BytesIO(image_bytes))
    im.load()
    im = im.convert("RGB")
    seg = car_segment.isolate_car(im)
    crop = seg.image if seg is not None else None
    crop_vec = dinov2_embed.embed_image(crop) if crop is not None else None
    # Esto corre bajo `_IDENTIFY_SLOTS`: si YOLO aún no está en memoria (lo
    # precalienta `vision.warmup`), no se carga aquí; sin comprobación no hay
    # foto publicada.
    from app.services import yolo_crop

    publishable = crop is not None and yolo_crop.loaded() and _crop_is_car(im, seg)
    return Prepared(im=im, crop=crop, crop_vec=crop_vec, publishable=publishable)


def check_result(chk: DupCheck) -> dict:
    """Salida de `create_from_read` para un `check_duplicate` que no es "new"."""
    if chk.outcome == "duplicate":
        return {**chk.piece, "outcome": "duplicate", "reason": chk.reason}
    # No se escribe nada (ni imagen, ni Mongo, ni índice): el llamador
    # devuelve el flujo manual.
    return {"outcome": "uncertain", "reason": chk.reason, "created": False}


def create_from_read(db, user_id: str, read: dict, image_bytes: bytes) -> dict:
    """Crea la pieza (o devuelve la existente). Salida: ver `_piece_out`.

    No comprueba `gate` ni el tope: eso es del llamador. Es la composición de
    `prepare_image` (CPU) → `check_duplicate` (texto + índice, sin CPU
    pesada) → `create_new` (alta); el endpoint llama a las tres por separado
    para que la comprobación anti-duplicado no ocupe un slot del identify."""
    read = sanitize_read(read)
    if read is None:
        return {"outcome": "uncertain", "reason": "unsafe_text", "created": False}
    prep = prepare_image(image_bytes)
    chk = check_duplicate(db, read, prep.crop_vec)
    if chk.outcome != "new":
        return check_result(chk)
    return create_new(db, user_id, read, prep)


def create_new(db, user_id: str, read: dict, prep: Prepared) -> dict:
    """Alta de una lectura YA comprobada como nueva (`check_duplicate`)."""
    from app.routers.catalog import _upsert_catalog_piece
    from app.services import dinov2_embed

    im, crop, crop_vec = prep.im, prep.crop, prep.crop_vec
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

    # Barrera 5: sin manos ni fondo. Solo se publica un recorte que YOLO
    # confirma como vehículo; si no, la pieza nace sin foto.
    url = _save_crop(crop) if crop is not None and prep.publishable else None
    metadata: dict = {"scale": "1/64"}
    if year is not None:
        metadata["year"] = year
    if series:
        metadata["series"] = series
    raw_toy = _raw_toy(read)
    if raw_toy and not toy:
        metadata["toy_number"] = raw_toy  # número leído sin valor de identidad

    variant = {
        "id": vid,
        "ai_local_hash": "h" + hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:24],
        "variant_name": (livery or color or casting or "")[:VARIANT_NAME_MAX].strip() or casting,
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
        # Clave interna de la semilla del id (nunca sale hacia la app).
        "source_key": source_id,
    }
    # `items.release_year` es obligatorio en el esquema: 0 = año desconocido
    # (la app lo trata como falso y las facetas/orden por año lo ignoran).
    # Sin marca ni modelo reconocibles NO se rellena con «Desconocido»: la
    # pieza no entra al catálogo compartido (el llamador ya pasó `gate_create`,
    # esto es la red de seguridad).
    if not brand or not casting:
        return {"outcome": "uncertain", "reason": "missing_identity", "created": False}
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
    out = _piece_out(doc, master_doc, created=created)
    out["outcome"] = "created" if created else "duplicate"
    out["reason"] = "created" if created else "race"
    return out


def confirm(db, variant_id: str, user_id: str) -> bool:
    """Confirmación comunitaria de una pieza de IA. Nunca lanza.

    El creador cuenta como primer confirmador: la pieza pasa a `verified` con
    >= 2 usuarios DISTINTOS en {created_by} ∪ confirmed_by. Idempotente, nunca
    desverifica y se AUTOCURA: la cuenta se evalúa siempre sobre el documento
    fresco devuelto por el `$addToSet`, así que una llamada posterior corrige
    una escritura previa a medias o una lectura vieja. Solo `origin == "ai"`;
    el resto no se toca (sin escritura). True solo si ESTA llamada puso
    `verified=True`."""
    try:
        doc = _resolve_variant(db, variant_id)
        if not doc or doc.get("origin") != "ai" or not user_id:
            return False
        if doc.get("verified") is True:
            return False
        uid = str(user_id)
        creator = str(doc["created_by"]) if doc.get("created_by") else None
        if uid != creator and uid not in {str(u) for u in (doc.get("confirmed_by") or [])}:
            fresh = db.variants.find_one_and_update(
                {"_id": doc["_id"], "origin": "ai"},
                {"$addToSet": {"confirmed_by": uid}},
                return_document=ReturnDocument.AFTER,
            )
            if not fresh:
                return False
        else:
            fresh = doc
        if fresh.get("verified") is True:
            return False
        users = {str(u) for u in (fresh.get("confirmed_by") or [])}
        if creator:
            users.add(creator)
        if len(users) < 2:
            return False
        res = db.variants.update_one(
            {"_id": doc["_id"], "origin": "ai", "verified": {"$ne": True}},
            {"$set": {"verified": True}},
        )
        return res.modified_count == 1
    except Exception:  # noqa: BLE001
        _log.warning("ai_catalog: confirm falló variant=%s", variant_id, exc_info=True)
        return False


def strip_internal(doc: dict) -> dict:
    """Copia sin los campos internos de moderación (no muta la entrada)."""
    return {k: v for k, v in doc.items() if k not in _INTERNAL}
