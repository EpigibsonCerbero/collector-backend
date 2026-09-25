"""Catálogo: marcas, series, categorías, items_master + variantes."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pymongo import DESCENDING, ReturnDocument

from app.db import as_id, get_db, oid, serialize, serialize_many
from app.schemas import (
    BrandCreate,
    BrandOut,
    CatalogItemCreate,
    CatalogItemOut,
    CategoryCreate,
    CategoryOut,
    ItemVariantOut,
    SeriesCreate,
    SeriesOut,
    SeriesProgressOut,
    SpecContributionCreate,
    SpecContributionOut,
    SpecFieldStatus,
    UpcExtractOut,
)
from app.security import CurrentUser, get_current_user

router = APIRouter(tags=["catalog"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _master_out(doc: dict | None) -> CatalogItemOut | None:
    if not doc:
        return None
    data = serialize(doc)
    assert data is not None
    return CatalogItemOut(**data)


def _variant_out(doc: dict, master: dict | None = None) -> ItemVariantOut:
    data = serialize(doc)
    assert data is not None
    if master is not None:
        data["item_master"] = _master_out(master)
    return ItemVariantOut(**data)


def _load_master(db, doc: dict) -> dict | None:
    mid = doc.get("item_master_id")
    if not mid:
        return None
    try:
        return db.items.find_one({"_id": as_id(mid)})
    except Exception:
        return db.items.find_one({"_id": mid})


# ── Marcas ────────────────────────────────────────────────────────────────


@router.get("/brands", response_model=list[BrandOut])
def list_brands() -> list[BrandOut]:
    docs = list(get_db().brands.find().sort("name", 1))
    return [BrandOut(**serialize(d)) for d in docs]  # type: ignore[arg-type]


@router.post("/brands", response_model=BrandOut, status_code=201)
def create_brand(
    body: BrandCreate,
    user: CurrentUser = Depends(get_current_user),
) -> BrandOut:
    db = get_db()
    if db.brands.find_one({"slug": body.slug}):
        raise HTTPException(status_code=409, detail="La marca ya existe")
    doc = body.model_dump()
    doc["created_at"] = _now()
    res = db.brands.insert_one(doc)
    doc["_id"] = res.inserted_id
    return BrandOut(**serialize(doc))  # type: ignore[arg-type]


@router.get("/brands/{brand_id}", response_model=BrandOut)
def get_brand(brand_id: str) -> BrandOut:
    doc = get_db().brands.find_one({"_id": oid(brand_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Marca no encontrada")
    return BrandOut(**serialize(doc))  # type: ignore[arg-type]


# ── Series ────────────────────────────────────────────────────────────────


@router.get("/series", response_model=list[SeriesOut])
def list_series(brand_id: Optional[str] = None) -> list[SeriesOut]:
    query = {"brand_id": brand_id} if brand_id else {}
    docs = list(get_db().series.find(query).sort([("release_year", -1), ("name", 1)]))
    return [SeriesOut(**serialize(d)) for d in docs]  # type: ignore[arg-type]


@router.post("/series", response_model=SeriesOut, status_code=201)
def create_series(
    body: SeriesCreate,
    user: CurrentUser = Depends(get_current_user),
) -> SeriesOut:
    db = get_db()
    if not db.brands.find_one({"_id": oid(body.brand_id)}):
        raise HTTPException(status_code=404, detail="Marca no encontrada")
    doc = body.model_dump()
    now = _now()
    doc["brand_id"] = body.brand_id
    doc["created_at"] = now
    doc["updated_at"] = now
    res = db.series.insert_one(doc)
    doc["_id"] = res.inserted_id
    return SeriesOut(**serialize(doc))  # type: ignore[arg-type]


@router.get("/series/{series_id}/progress", response_model=list[SeriesProgressOut])
def series_progress(
    series_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> list[SeriesProgressOut]:
    """Progreso de la serie para el usuario autenticado."""
    db = get_db()
    s = db.series.find_one({"_id": oid(series_id)})
    if not s:
        raise HTTPException(status_code=404, detail="Serie no encontrada")
    brand = db.brands.find_one({"_id": oid(s["brand_id"])}) if s.get("brand_id") else None

    variants = list(db.variants.find({"item_master_id": {"$in": _master_ids_for_series(db, series_id)}}))
    variant_ids = [str(v["_id"]) for v in variants]
    owned = db.user_items.distinct("item_variant_id", {"user_id": user.id, "item_variant_id": {"$in": variant_ids}})
    collected = len(set(owned))
    total = s.get("total_items_count") or 0
    pct = round((collected / total) * 100, 1) if total else 0.0
    return [
        SeriesProgressOut(
            series_id=str(s["_id"]),
            brand_id=str(s.get("brand_id", "")),
            brand_name=brand["name"] if brand else None,
            series_name=s["name"],
            release_year=s.get("release_year", 0),
            segment_name=s.get("segment_name"),
            total_items_count=total,
            items_collected_count=collected,
            completion_percentage=pct,
        )
    ]


def _master_ids_for_series(db, series_id: str) -> list:
    docs = db.items.find({"series_id": series_id}, {"_id": 1})
    return [d["_id"] for d in docs]


# ── Categorías ────────────────────────────────────────────────────────────


@router.get("/categories", response_model=list[CategoryOut])
def list_categories() -> list[CategoryOut]:
    docs = list(get_db().categories.find().sort("name", 1))
    return [CategoryOut(**serialize(d)) for d in docs]  # type: ignore[arg-type]


@router.post("/categories", response_model=CategoryOut, status_code=201)
def create_category(
    body: CategoryCreate,
    user: CurrentUser = Depends(get_current_user),
) -> CategoryOut:
    db = get_db()
    if db.categories.find_one({"slug": body.slug}):
        raise HTTPException(status_code=409, detail="La categoría ya existe")
    doc = body.model_dump()
    doc["created_at"] = _now()
    res = db.categories.insert_one(doc)
    doc["_id"] = res.inserted_id
    return CategoryOut(**serialize(doc))  # type: ignore[arg-type]


# ── Variantes / items ─────────────────────────────────────────────────────


@router.get("/catalog/variants", response_model=list[ItemVariantOut])
def list_variants(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    brand: Optional[str] = None,
    rarity: Optional[str] = None,
    search: Optional[str] = None,
) -> list[ItemVariantOut]:
    db = get_db()
    query: dict = {}
    if rarity:
        query["rarity"] = rarity
    if search:
        query["$or"] = [
            {"variant_name": {"$regex": search, "$options": "i"}},
            {"toy_number": {"$regex": search, "$options": "i"}},
            {"name": {"$regex": search, "$options": "i"}},
        ]
    if brand:
        master_ids = [d["_id"] for d in db.items.find({"brand_id": brand}, {"_id": 1})]
        query["item_master_id"] = {"$in": [str(m) for m in master_ids]}

    docs = list(db.variants.find(query).sort("created_at", -1).skip(offset).limit(limit))
    out: list[ItemVariantOut] = []
    for d in docs:
        out.append(_variant_out(d, _load_master(db, d)))
    return out


@router.get("/catalog/variants/by-hash/{ai_hash}", response_model=ItemVariantOut)
def get_variant_by_hash(ai_hash: str) -> ItemVariantOut:
    db = get_db()
    doc = db.variants.find_one({"ai_local_hash": ai_hash})
    if not doc:
        raise HTTPException(status_code=404, detail="Variante no encontrada")
    return _variant_out(doc, _load_master(db, doc))


@router.get("/catalog/variants/by-toy/{toy_number}", response_model=list[ItemVariantOut])
def get_variants_by_toy(toy_number: str) -> list[ItemVariantOut]:
    db = get_db()
    docs = list(db.variants.find({"toy_number": toy_number}))
    return [_variant_out(d, _load_master(db, d)) for d in docs]


@router.get("/catalog/variants/{variant_id}", response_model=ItemVariantOut)
def get_variant(variant_id: str) -> ItemVariantOut:
    db = get_db()
    doc = db.variants.find_one({"_id": as_id(variant_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Variante no encontrada")
    return _variant_out(doc, _load_master(db, doc))


@router.post("/catalog/variants", response_model=ItemVariantOut, status_code=201)
def create_catalog_item(
    body: CatalogItemCreate,
    user: CurrentUser = Depends(get_current_user),
) -> ItemVariantOut:
    """Upsert de items_master + item_variants (equivale a createCatalogItem)."""
    db = get_db()
    now = _now()
    data = body.model_dump()

    ai_hash = data.pop("ai_local_hash", None)
    variant_id = data.pop("id", None)
    master_fields = {
        "name": data.pop("name"),
        "manufacturer": data.pop("manufacturer"),
        "release_year": data.pop("release_year"),
        "category_id": data.pop("category_id"),
        "brand_id": data.pop("brand_id"),
        "series_id": data.pop("series_id"),
        "casting_name": data.pop("casting_name"),
        "designer": data.pop("designer"),
        "scale": data.pop("scale"),
    }

    # Dedup blando: mismo toy_number + variant_name → reutiliza variante
    existing = None
    if data.get("toy_number"):
        existing = db.variants.find_one(
            {"toy_number": data["toy_number"], "variant_name": body.variant_name}
        )
    if existing is None and ai_hash:
        existing = db.variants.find_one({"ai_local_hash": ai_hash})
    if existing:
        return _variant_out(existing, _load_master(db, existing))

    # Maestro: reutiliza por (name, manufacturer, release_year)
    master = db.items.find_one(
        {
            "name": master_fields["name"],
            "manufacturer": master_fields["manufacturer"],
            "release_year": master_fields["release_year"],
        }
    )
    if master is None:
        master_doc = {k: v for k, v in master_fields.items() if v is not None}
        master_doc["created_at"] = now
        res_m = db.items.insert_one(master_doc)
        master_id = str(res_m.inserted_id)
    else:
        master_id = str(master["_id"])

    variant_doc = data
    variant_doc["item_master_id"] = master_id
    variant_doc["ai_local_hash"] = ai_hash or f"hash-{(data.get('toy_number') or variant_id or master_id).lower()}"
    variant_doc["created_at"] = now
    if variant_id:
        variant_doc["_id"] = oid(variant_id) if _is_oid(variant_id) else variant_id

    try:
        res = db.variants.insert_one(variant_doc)
    except Exception as exc:
        # unique ai_local_hash
        if "ai_local_hash" in str(exc) and ai_hash:
            dup = db.variants.find_one({"ai_local_hash": ai_hash})
            if dup:
                return _variant_out(dup, _load_master(db, dup))
        raise HTTPException(status_code=409, detail="No se pudo crear la variante")
    variant_doc["_id"] = res.inserted_id
    return _variant_out(variant_doc, _load_master(db, variant_doc))


def _is_oid(value: str) -> bool:
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        ObjectId(value)
        return True
    except (InvalidId, TypeError):
        return False


@router.patch("/catalog/variants/{variant_id}", response_model=ItemVariantOut)
def update_variant(
    variant_id: str,
    body: dict,
    user: CurrentUser = Depends(get_current_user),
) -> ItemVariantOut:
    db = get_db()
    allowed = {
        "variant_name",
        "sku_upc",
        "main_image_url",
        "metadata",
        "toy_number",
        "rarity",
        "body_color",
        "premium_finish",
        "wheel_type",
        "tampo_details",
        "base_material",
        "country_of_origin",
    }
    updates = {k: v for k, v in body.items() if k in allowed}
    if not updates:
        raise HTTPException(status_code=400, detail="Sin campos válidos")
    doc = db.variants.find_one_and_update(
        {"_id": as_id(variant_id)},
        {"$set": updates},
        return_document=ReturnDocument.AFTER,
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Variante no encontrada")
    return _variant_out(doc, _load_master(db, doc))


@router.post("/extract-upc", response_model=UpcExtractOut)
async def extract_upc(
    image: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
) -> UpcExtractOut:
    """Extrae UPC de una foto de empaque y lo asocia al toy# correspondiente.

    Flujo colaborativo: cuando el usuario sube la foto del slot 'back'
    (Reverso & Holograma) en el peritaje fotográfico, se extrae automáticamente
    el código de barras y se cruza con el toy# del OCR. Si coinciden, se
    actualiza el catálogo maestro.
    """
    import io as _io

    import numpy as np
    from PIL import Image

    from app.services.vision import read_barcode, read_ocr

    data = await image.read()
    try:
        im = Image.open(_io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen inválida")
    if max(im.size) > 1024:
        s = 1024 / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.Resampling.BILINEAR)
    np_rgb = np.asarray(im)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_bar = pool.submit(read_barcode, np_rgb)
        fut_ocr = pool.submit(read_ocr, np_rgb)
        upc = fut_bar.result()
        text, toy, toy_conf, ocr_upc = fut_ocr.result()

    if not upc:
        upc = ocr_upc
    if not upc:
        return UpcExtractOut(
            upc=None, toy_number=toy, found=False, saved=False,
            message="No se detectó código de barras en la imagen.",
        )

    db = get_db()
    norm_toy = (toy or "").upper().replace(" ", "")

    if norm_toy:
        doc = db.variants.find_one({"toy_number": norm_toy})
        if not doc:
            doc = db.variants.find_one(
                {"toy_number": {"$regex": f"^{norm_toy}$", "$options": "i"}}
            )
        if doc:
            existing = db.variants.find_one({"sku_upc": upc, "_id": {"$ne": doc["_id"]}})
            if existing:
                return UpcExtractOut(
                    upc=upc, toy_number=norm_toy, found=True, saved=False,
                    message=f"UPC {upc} ya está registrado en {existing.get('toy_number', '?')}.",
                )
            db.variants.update_one({"_id": doc["_id"]}, {"$set": {"sku_upc": upc}})
            return UpcExtractOut(
                upc=upc, toy_number=norm_toy, found=True, saved=True,
                message=f"UPC {upc} registrado para {norm_toy}.",
            )
        return UpcExtractOut(
            upc=upc, toy_number=norm_toy, found=True, saved=False,
            message=f"UPC {upc} detectado pero {norm_toy} no está en el catálogo.",
        )

    existing = db.variants.find_one({"sku_upc": upc})
    if existing:
        return UpcExtractOut(
            upc=upc, toy_number=existing.get("toy_number"), found=True, saved=False,
            message=f"UPC {upc} ya registrado para {existing.get('toy_number', '?')}.",
        )

    return UpcExtractOut(
        upc=upc, toy_number=None, found=True, saved=False,
        message=f"UPC {upc} detectado. Sin toy# en el texto para cruzar.",
    )


# ── Contribución colaborativa de specs técnicas ──────────────────────────────


@router.post("/specs/contribute", response_model=SpecContributionOut, status_code=201)
def contribute_specs(
    body: SpecContributionCreate,
    user: CurrentUser = Depends(get_current_user),
) -> SpecContributionOut:
    """Contribuir specs técnicas de una variante (crowdsourcing).

    Flujo: el usuario ve la ficha técnica → contribuye datos del empaque /
    inspección → si 3+ usuarios confirman el mismo valor, se acepta
    automáticamente (consenso comunitario).
    """
    from app.schemas import SPEC_FIELDS

    # Validar campos
    clean_fields = {k: v.strip() for k, v in body.fields.items() if k in SPEC_FIELDS and v.strip()}
    if not clean_fields:
        raise HTTPException(status_code=400, detail="Ningún campo válido")

    db = get_db()

    # Verificar que la variante existe
    try:
        variant = db.variants.find_one({"_id": as_id(body.variant_id)})
    except Exception:
        variant = db.variants.find_one({"_id": body.variant_id})
    if not variant:
        raise HTTPException(status_code=404, detail="Variante no encontrada")

    doc = {
        "variant_id": body.variant_id,
        "user_id": user.id,
        "fields": clean_fields,
        "source": body.source,
        "notes": body.notes,
        "status": "pending",
        "confirmations": 0,
        "created_at": _now(),
    }
    res = db.spec_contributions.insert_one(doc)
    doc["_id"] = res.inserted_id

    # Auto-aceptar si hay consenso (3+ contribuciones iguales al mismo campo)
    _check_consensus(db, body.variant_id)

    return SpecContributionOut(**serialize(doc))


@router.get("/specs/{variant_id}/status", response_model=list[SpecFieldStatus])
def get_spec_status(variant_id: str) -> list[SpecFieldStatus]:
    """Estado de los campos técnicos de una variante (valor actual + pendientes)."""
    from app.schemas import SPEC_FIELDS

    db = get_db()
    try:
        variant = db.variants.find_one({"_id": as_id(variant_id)})
    except Exception:
        variant = db.variants.find_one({"_id": variant_id})
    if not variant:
        raise HTTPException(status_code=404, detail="Variante no encontrada")

    # Contar contribuciones pendientes por campo
    pending: dict[str, int] = {}
    for c in db.spec_contributions.find({"variant_id": variant_id, "status": "pending"}):
        for field in (c.get("fields") or {}):
            pending[field] = pending.get(field, 0) + 1

    out = []
    for field in SPEC_FIELDS:
        val = variant.get(field)
        out.append(SpecFieldStatus(
            field=field,
            value=val,
            is_default=bool(val) and val in _DEFAULT_VALUES.get(field, set()),
            pending_contributions=pending.get(field, 0),
        ))
    return out


@router.get("/specs/{variant_id}/contributions", response_model=list[SpecContributionOut])
def list_spec_contributions(
    variant_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> list[SpecContributionOut]:
    """Ver contribuciones de specs para una variante."""
    db = get_db()
    docs = list(db.spec_contributions.find({"variant_id": variant_id}).sort("created_at", DESCENDING))
    return [SpecContributionOut(**serialize(d)) for d in docs]


@router.post("/specs/contributions/{contribution_id}/confirm")
def confirm_contribution(
    contribution_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Confirmar una contribución de otro usuario (voto positivo)."""
    db = get_db()
    try:
        contrib = db.spec_contributions.find_one({"_id": as_id(contribution_id)})
    except Exception:
        contrib = db.spec_contributions.find_one({"_id": contribution_id})
    if not contrib:
        raise HTTPException(status_code=404, detail="Contribución no encontrada")

    # No confirmar la propia contribución
    if contrib.get("user_id") == user.id:
        raise HTTPException(status_code=400, detail="No puedes confirmar tu propia contribución")

    # Evitar doble voto
    existing = db.spec_contributions.find_one({
        "_id": as_id(contribution_id) if len(contribution_id) == 24 else contribution_id,
        "confirmed_by": user.id,
    })
    if existing:
        raise HTTPException(status_code=400, detail="Ya confirmaste esta contribución")

    db.spec_contributions.update_one(
        {"_id": contrib["_id"]},
        {
            "$inc": {"confirmations": 1},
            "$addToSet": {"confirmed_by": user.id},
        },
    )

    # Auto-aceptar con 3 confirmaciones
    if (contrib.get("confirmations", 0) + 1) >= 3:
        db.spec_contributions.update_one(
            {"_id": contrib["_id"]}, {"$set": {"status": "confirmed"}}
        )
        _apply_contribution(db, contrib)

    _check_consensus(db, contrib.get("variant_id", ""))
    return {"ok": True, "confirmations": contrib.get("confirmations", 0) + 1}


def _check_consensus(db, variant_id: str) -> None:
    """Auto-aceptar cuando 3+ contribuciones distintas proponen el mismo valor."""
    from app.schemas import SPEC_FIELDS

    for field in SPEC_FIELDS:
        # Agrupar contribuciones pendientes por valor propuesto
        pending = list(db.spec_contributions.find({
            "variant_id": variant_id,
            "status": "pending",
            f"fields.{field}": {"$exists": True},
        }))
        value_counts: dict[str, int] = {}
        for c in pending:
            val = (c.get("fields") or {}).get(field)
            if val:
                value_counts[val] = value_counts.get(val, 0) + 1

        for val, count in value_counts.items():
            if count >= 3:
                # Consenso → aplicar y marcar como confirmadas
                try:
                    db.variants.update_one(
                        {"_id": as_id(variant_id)}, {"$set": {field: val}}
                    )
                except Exception:
                    db.variants.update_one(
                        {"_id": variant_id}, {"$set": {field: val}}
                    )
                db.spec_contributions.update_many(
                    {
                        "variant_id": variant_id,
                        "status": "pending",
                        f"fields.{field}": val,
                    },
                    {"$set": {"status": "confirmed"}},
                )


def _apply_contribution(db, contrib: dict) -> None:
    """Aplicar una contribución confirmada a la variante."""
    fields = contrib.get("fields") or {}
    if not fields:
        return
    variant_id = contrib.get("variant_id")
    try:
        db.variants.update_one({"_id": as_id(variant_id)}, {"$set": fields})
    except Exception:
        db.variants.update_one({"_id": variant_id}, {"$set": fields})


# Valores por defecto (para saber si un campo es default vs data real)
_DEFAULT_VALUES: dict[str, set] = {
    "base_material": {"Metal/Plastic", "Zamac (aleación zinc-aluminio)", "Diecast (aleación zinc)"},
    "wheel_type": {"TPR (caucho termoplástico) con rin de plástico", "Rueda de goma con rin de plástico", "Plástico ABS (estilo Real Riders solo premium)"},
    "tampo_details": {"Tampografía serigráfica multi-color", "Tampografía pad printing", "Tampografía / calcomanías"},
    "premium_finish": {"Pintura brillante UV con livery tampografiado", "Pintura brillante estándar", "Pintura esmalte", "Pintura automotriz con clear coat"},
    "country_of_origin": {"China (fabricación TSM Models)", "China", "Malasia / Tailandia / China"},
    "scale": {"1:64", "1/64"},
}
