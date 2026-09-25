"""Trading: publicaciones del hub, ofertas de trueque y escrow."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pymongo import DESCENDING, ReturnDocument

from app.db import get_db, oid, serialize
from app.schemas_social import (
    EscrowStep,
    EscrowUpdate,
    OfferItem,
    TradeEscrowOut,
    TradeListingCreate,
    TradeListingOut,
    TradeOfferCreate,
    TradeOfferOut,
)
from app.security import CurrentUser, get_current_user

router = APIRouter(prefix="/trading", tags=["trading"])

DEFAULT_ESCROW_STEPS = [
    {"id": "step-offer", "label": "Oferta aceptada", "status": "completed"},
    {"id": "step-ship", "label": "Envío confirmado", "status": "active"},
    {"id": "step-receive", "label": "Pieza recibida", "status": "pending"},
    {"id": "step-done", "label": "Intercambio completado", "status": "pending"},
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Listados del hub ──────────────────────────────────────────────────────


@router.get("/listings", response_model=list[TradeListingOut])
def list_listings(
    user: CurrentUser = Depends(get_current_user),
    type: str | None = None,
) -> list[TradeListingOut]:
    query = {"type": type} if type else {}
    docs = list(get_db().trade_listings.find(query).sort("created_at", DESCENDING))
    out = []
    for d in docs:
        data = serialize(d)
        assert data is not None
        data["is_own"] = data.get("user_id") == user.id
        out.append(TradeListingOut(**data))
    return out


@router.post("/listings", response_model=TradeListingOut, status_code=201)
def publish_listing(
    body: TradeListingCreate,
    user: CurrentUser = Depends(get_current_user),
) -> TradeListingOut:
    doc = body.model_dump()
    doc.update(
        {
            "user_id": user.id,
            "username": user.username,
            "avatar_emoji": "🏎️",
            "created_at": _now(),
        }
    )
    res = get_db().trade_listings.insert_one(doc)
    doc["_id"] = res.inserted_id
    data = serialize(doc)
    assert data is not None
    data["is_own"] = True
    return TradeListingOut(**data)


@router.delete("/listings/{listing_id}", status_code=204)
def remove_listing(
    listing_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    res = get_db().trade_listings.delete_one({"_id": oid(listing_id), "user_id": user.id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Publicación no encontrada")


# ── Ofertas ───────────────────────────────────────────────────────────────


@router.get("/offers", response_model=list[TradeOfferOut])
def list_offers(user: CurrentUser = Depends(get_current_user)) -> list[TradeOfferOut]:
    db = get_db()
    docs = list(
        db.trade_offers.find(
            {"$or": [{"from_user_id": user.id}, {"to_user_id": user.id}]}
        ).sort("created_at", DESCENDING)
    )
    out = []
    for d in docs:
        data = serialize(d)
        assert data is not None
        data["type"] = "outgoing" if data.get("from_user_id") == user.id else "incoming"
        out.append(TradeOfferOut(**data))
    return out


@router.post("/offers", response_model=TradeOfferOut, status_code=201)
def create_offer(
    body: TradeOfferCreate,
    user: CurrentUser = Depends(get_current_user),
) -> TradeOfferOut:
    if body.to_user_id == user.id:
        raise HTTPException(status_code=400, detail="No puedes ofrecerte a ti mismo")
    now = _now()
    doc = {
        "from_user_id": user.id,
        "from_username": user.username,
        "from_avatar": body.from_avatar,
        "to_user_id": body.to_user_id,
        "to_username": body.to_username,
        "to_avatar": "",
        "offered_items": [i.model_dump() for i in body.offered_items],
        "requested_items": [i.model_dump() for i in body.requested_items],
        "message": body.message,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
    }
    res = get_db().trade_offers.insert_one(doc)
    doc["_id"] = res.inserted_id
    data = serialize(doc)
    assert data is not None
    data["type"] = "outgoing"
    return TradeOfferOut(**data)


@router.post("/offers/{offer_id}/accept", response_model=TradeOfferOut)
def accept_offer(
    offer_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> TradeOfferOut:
    db = get_db()
    offer = db.trade_offers.find_one({"_id": oid(offer_id)})
    if not offer or offer.get("to_user_id") != user.id:
        raise HTTPException(status_code=404, detail="Oferta no encontrada")
    if offer.get("status") != "pending":
        raise HTTPException(status_code=409, detail=f"Oferta en estado {offer.get('status')}")

    now = _now()
    doc = db.trade_offers.find_one_and_update(
        {"_id": oid(offer_id)},
        {"$set": {"status": "accepted", "updated_at": now}},
        return_document=ReturnDocument.AFTER,
    )
    # Crear escrow
    first_item = (offer.get("offered_items") or [{}])[0]
    db.trade_escrows.insert_one(
        {
            "offer_id": offer_id,
            "partner_name": offer.get("from_username", ""),
            "item_name": first_item.get("variant_name", ""),
            "steps": [dict(s, timestamp=now.isoformat() if s["status"] == "completed" else None) for s in DEFAULT_ESCROW_STEPS],
            "tracking_number": None,
            "status": "in_progress",
            "created_at": now,
        }
    )
    data = serialize(doc)
    assert data is not None
    data["type"] = "incoming"
    return TradeOfferOut(**data)


@router.post("/offers/{offer_id}/reject", response_model=TradeOfferOut)
def reject_offer(
    offer_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> TradeOfferOut:
    db = get_db()
    offer = db.trade_offers.find_one({"_id": oid(offer_id)})
    if not offer or offer.get("to_user_id") != user.id:
        raise HTTPException(status_code=404, detail="Oferta no encontrada")
    doc = db.trade_offers.find_one_and_update(
        {"_id": oid(offer_id)},
        {"$set": {"status": "rejected", "updated_at": _now()}},
        return_document=ReturnDocument.AFTER,
    )
    data = serialize(doc)
    assert data is not None
    data["type"] = "incoming"
    return TradeOfferOut(**data)


# ── Escrow ────────────────────────────────────────────────────────────────


@router.get("/escrows", response_model=list[TradeEscrowOut])
def list_escrows(user: CurrentUser = Depends(get_current_user)) -> list[TradeEscrowOut]:
    db = get_db()
    offer_ids = [
        str(o["_id"])
        for o in db.trade_offers.find(
            {"$or": [{"from_user_id": user.id}, {"to_user_id": user.id}]}, {"_id": 1}
        )
    ]
    docs = list(db.trade_escrows.find({"offer_id": {"$in": offer_ids}}).sort("created_at", DESCENDING))
    return [TradeEscrowOut(**serialize(d)) for d in docs]  # type: ignore[arg-type]


@router.patch("/escrows/{escrow_id}", response_model=TradeEscrowOut)
def update_escrow(
    escrow_id: str,
    body: EscrowUpdate,
    user: CurrentUser = Depends(get_current_user),
) -> TradeEscrowOut:
    db = get_db()
    esc = db.trade_escrows.find_one({"_id": oid(escrow_id)})
    if not esc:
        raise HTTPException(status_code=404, detail="Escrow no encontrado")
    # Solo el dueño del offer puede actualizar
    offer = db.trade_offers.find_one({"_id": oid(esc["offer_id"])})
    if not offer or user.id not in (offer.get("from_user_id"), offer.get("to_user_id")):
        raise HTTPException(status_code=403, detail="Sin permiso")

    updates: dict = {}
    if body.status is not None:
        updates["status"] = body.status
    if body.tracking_number is not None:
        updates["tracking_number"] = body.tracking_number
    if body.steps is not None:
        updates["steps"] = [s.model_dump() for s in body.steps]
    if not updates:
        raise HTTPException(status_code=400, detail="Sin cambios")
    doc = db.trade_escrows.find_one_and_update(
        {"_id": oid(escrow_id)},
        {"$set": updates},
        return_document=ReturnDocument.AFTER,
    )
    return TradeEscrowOut(**serialize(doc))  # type: ignore[arg-type]
