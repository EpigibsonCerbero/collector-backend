"""Catálogo: marcas, series, categorías, items_master + variantes."""
from __future__ import annotations

import re

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
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
from app.security import CurrentUser, get_current_user, require_admin

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
    user: CurrentUser = Depends(require_admin),
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
    user: CurrentUser = Depends(require_admin),
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
    user: CurrentUser = Depends(require_admin),
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
    response: Response,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    brand: Optional[str] = None,
    rarity: Optional[str] = None,
    search: Optional[str] = None,
    manufacturer: Optional[str] = None,
    sub_brand: Optional[str] = None,
    scale: Optional[str] = None,
    color: Optional[str] = None,
    decade: Optional[str] = None,
    sort: str = Query(default="name", pattern="^(name|year|code|recent)$"),
    sort_dir: str = Query(default="asc", pattern="^(asc|desc)$"),
) -> list[ItemVariantOut]:
    """Listado paginado de variantes con filtros y orden.

    Usa aggregation + $lookup para traer el master en la misma query (antes
    era N+1: un find_one por variante). Devuelve el total en la cabecera
    `X-Total-Count` para que la UI sepa cuándo parar de paginar.
    """
    db = get_db()
    match: dict = {}
    if rarity:
        match["rarity"] = rarity
    and_clauses: list[dict] = []
    if search:
        # Antes: la frase entera como regex SIN escapar (ReDoS) y solo sobre
        # campos de la variante — el nombre del modelo vive en el maestro
        # (`items`), así que «Skyline R34» casi nunca encontraba nada.
        # Ahora: cada palabra debe aparecer en modelo/marca/livery/toy#/UPC.
        for tok in [t for t in re.split(r"\s+", search.strip()) if t][:6]:
            rx = {"$regex": re.escape(tok), "$options": "i"}
            mids = [
                str(d["_id"])
                for d in db.items.find(
                    {"$or": [{"name": rx}, {"manufacturer": rx}, {"casting_name": rx}]}, {"_id": 1}
                ).limit(5000)
            ]
            and_clauses.append(
                {
                    "$or": [
                        {"variant_name": rx},
                        {"toy_number": rx},
                        {"sku_upc": rx},
                        {"livery_desc": rx},
                        {"item_master_id": {"$in": mids}},
                    ]
                }
            )
    if scale:
        # el seed guarda "1/64" (con barra); aceptar ambos formatos.
        # En $and: antes se mezclaba en el mismo $or que la búsqueda (unión, no intersección).
        sx = scale.replace(":", "/")
        and_clauses.append({"$or": [{"metadata.scale": sx}, {"metadata.scale": scale}]})
    if and_clauses:
        match["$and"] = and_clauses
    if color:
        # Grey/Gray se unifican en la UI
        if color == "Grey":
            match["body_color"] = {"$in": ["Grey", "Gray"]}
        else:
            match["body_color"] = color

    # manufacturer / sub_brand / brand viven en el master → filtrar por ids
    master_cond: dict = {}
    if manufacturer:
        master_cond["manufacturer"] = manufacturer
    if sub_brand:
        master_cond["$or"] = [
            {"sub_brand": sub_brand},
            {"metadata.sub_brand": sub_brand},
        ]
    if brand:
        master_cond["brand_id"] = brand

    if master_cond:
        master_ids = [str(d["_id"]) for d in db.items.find(master_cond, {"_id": 1})]
        if not master_ids:
            response.headers["X-Total-Count"] = "0"
            return []
        match["item_master_id"] = {"$in": master_ids}

    # Antes: $lookup de TODAS las variantes con su maestro, dos veces (contar y
    # ordenar) → 1.4 s el listado sin filtros. Ahora: se traen solo los campos
    # mínimos, se filtra/ordena en memoria con el caché de maestros y solo se
    # cargan completas las piezas de la página.
    mcache = _master_cache(db)
    rows = list(
        db.variants.find(match, {"_id": 1, "item_master_id": 1, "variant_name": 1, "toy_number": 1, "created_at": 1})
    )
    if decade:
        try:
            dec = int(float(decade))  # acepta «2020.0» de versiones viejas de la app
            rows = [r for r in rows if dec <= (mcache.get(str(r.get("item_master_id")), ("", None))[1] or -1) < dec + 10]
        except ValueError:
            pass
    response.headers["X-Total-Count"] = str(len(rows))

    desc = sort_dir == "desc"

    def _mname(r: dict) -> str:
        return (mcache.get(str(r.get("item_master_id")), ("", None))[0] or r.get("variant_name") or "").lower()

    if sort == "year":
        # Sin año siempre al final, en ambos sentidos.
        with_year = [r for r in rows if mcache.get(str(r.get("item_master_id")), ("", None))[1]]
        without = [r for r in rows if not mcache.get(str(r.get("item_master_id")), ("", None))[1]]
        with_year.sort(key=lambda r: mcache[str(r.get("item_master_id"))][1], reverse=desc)
        rows = with_year + without
    elif sort == "code":
        rows.sort(key=lambda r: (r.get("toy_number") is None, r.get("toy_number") or ""), reverse=desc)
    elif sort == "recent":
        rows.sort(key=lambda r: str(r.get("created_at") or ""), reverse=desc)
    else:
        rows.sort(key=lambda r: (_mname(r), (r.get("variant_name") or "").lower()), reverse=desc)

    page = rows[offset : offset + limit]
    if not page:
        return []
    full = {str(d["_id"]): d for d in db.variants.find({"_id": {"$in": [r["_id"] for r in page]}})}
    mkeys = []
    for r in page:
        mid = r.get("item_master_id")
        if mid:
            try:
                mkeys.append(as_id(str(mid)))
            except Exception:  # noqa: BLE001
                mkeys.append(str(mid))
    masters = {str(m["_id"]): m for m in db.items.find({"_id": {"$in": mkeys}})} if mkeys else {}
    out: list[ItemVariantOut] = []
    for r in page:
        d = full.get(str(r["_id"]))
        if d:
            out.append(_variant_out(d, masters.get(str(d.get("item_master_id")))))
    return out


_MASTER_CACHE: dict = {"at": 0.0, "data": {}}


def _master_cache(db) -> dict[str, tuple[str, int | None]]:
    """{id maestro: (nombre, año)} en memoria, 5 min. ~9k entradas, pocos KB."""
    import time

    now = time.monotonic()
    if now - _MASTER_CACHE["at"] > 300 or not _MASTER_CACHE["data"]:
        _MASTER_CACHE["data"] = {
            str(m["_id"]): (m.get("name") or "", m.get("release_year"))
            for m in db.items.find({}, {"name": 1, "release_year": 1})
        }
        _MASTER_CACHE["at"] = now
    return _MASTER_CACHE["data"]


@router.get("/catalog/variants/count")
def count_variants() -> dict:
    """Total de variantes (para la UI sin tener que paginar)."""
    return {"total": get_db().variants.count_documents({})}


@router.get("/catalog/facets")
def catalog_facets() -> dict:
    """Facetas para los chips de filtro del catálogo.

    Cuenta en Mongo (aggregation) en vez de bajar las 9k variantes al cliente
    para armar los contadores en JS. Devuelve solo valores con >=1 pieza.
    """
    db = get_db()

    def _agg(pipeline: list[dict]) -> list[dict]:
        return list(db.variants.aggregate(pipeline))

    # manufacturer vive en items (master) → contar por master
    brands_raw = list(
        db.items.aggregate(
            [
                {"$match": {"manufacturer": {"$exists": True, "$ne": ""}}},
                {"$group": {"_id": "$manufacturer", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
            ]
        )
    )

    # sub_brand: en metadata.sub_brand de la variante o sub_brand del master
    sb_var = _agg(
        [
            {"$match": {"metadata.sub_brand": {"$exists": True, "$ne": ""}}},
            {"$group": {"_id": "$metadata.sub_brand", "count": {"$sum": 1}}},
        ]
    )
    sb_master = list(
        db.items.aggregate(
            [
                {"$match": {"sub_brand": {"$exists": True, "$ne": ""}}},
                {"$group": {"_id": "$sub_brand", "count": {"$sum": 1}}},
            ]
        )
    )
    sb_counts: dict[str, int] = {}
    for r in sb_var + sb_master:
        key = str(r["_id"])
        sb_counts[key] = sb_counts.get(key, 0) + r["count"]

    scales_raw = _agg(
        [
            {"$match": {"metadata.scale": {"$exists": True, "$ne": ""}}},
            {"$group": {"_id": "$metadata.scale", "count": {"$sum": 1}}},
        ]
    )
    colors_raw = _agg(
        [
            {"$match": {"body_color": {"$exists": True, "$ne": ""}}},
            {"$group": {"_id": "$body_color", "count": {"$sum": 1}}},
        ]
    )
    rarities_raw = _agg([{"$group": {"_id": "$rarity", "count": {"$sum": 1}}}])

    # décadas desde release_year del master
    decades_raw = list(
        db.items.aggregate(
            [
                {"$match": {"release_year": {"$gt": 0}}},
                {
                    "$group": {
                        "_id": {
                            "$subtract": [
                                "$release_year",
                                {"$mod": ["$release_year", 10]},
                            ]
                        },
                        "count": {"$sum": 1},
                    }
                },
                {"$sort": {"_id": -1}},
            ]
        )
    )

    def _pack(raw: list[dict]) -> list[dict]:
        return [
            {"value": str(r["_id"]), "count": r["count"]}
            for r in raw
            if r.get("_id") not in (None, "")
        ]

    # normaliza Grey/Gray en un solo chip
    color_merged: dict[str, int] = {}
    for r in colors_raw:
        key = "Grey" if str(r["_id"]) == "Gray" else str(r["_id"])
        color_merged[key] = color_merged.get(key, 0) + r["count"]
    colors_out = sorted(
        ({"value": k, "count": v} for k, v in color_merged.items() if k),
        key=lambda x: -x["count"],
    )

    return {
        "brands": _pack(brands_raw),
        "sub_brands": sorted(
            ({"value": k, "count": v} for k, v in sb_counts.items() if k),
            key=lambda x: -x["count"],
        ),
        "scales": sorted(_pack(scales_raw), key=lambda x: x["value"]),
        "colors": colors_out,
        # Entero: algunos release_year son float (2024.0) y salía «2020.0s».
        "decades": [
            {"value": str(int(r["_id"])), "count": r["count"]}
            for r in decades_raw
            if r.get("_id") is not None
        ],
        "rarities": _pack(rarities_raw),
    }


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
    # Alta hecha por un usuario (no por el scrape): se marca para poder moderarla.
    variant_doc["created_by"] = user.id
    variant_doc["user_submitted"] = True
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
    user: CurrentUser = Depends(require_admin),
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
