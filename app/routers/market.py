"""Valoración histórica, transacciones y propuestas de variante."""
from __future__ import annotations

from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pymongo import DESCENDING

from app.db import as_id, get_db, oid, serialize
from app.schemas import (
    TransactionCreate,
    TransactionOut,
    ValueHistoryCreate,
    ValueHistoryOut,
    VariantProposalCreate,
    VariantProposalOut,
)
from app.security import CurrentUser, get_current_user

values_router = APIRouter(prefix="/values", tags=["values"])
tx_router = APIRouter(prefix="/transactions", tags=["transactions"])
proposals_router = APIRouter(prefix="/proposals", tags=["proposals"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


@values_router.get("/variants/{variant_id}/history", response_model=list[ValueHistoryOut])
def get_history(variant_id: str) -> list[ValueHistoryOut]:
    docs = list(
        get_db()
        .item_value_history.find({"item_variant_id": variant_id})
        .sort("recorded_date", DESCENDING)
    )
    return [ValueHistoryOut(**serialize(d)) for d in docs]  # type: ignore[arg-type]


@values_router.post(
    "/variants/{variant_id}/history",
    response_model=ValueHistoryOut,
    status_code=201,
)
def add_history(
    variant_id: str,
    body: ValueHistoryCreate,
    user: CurrentUser = Depends(get_current_user),
) -> ValueHistoryOut:
    db = get_db()
    if not db.variants.find_one({"_id": as_id(variant_id)}):
        raise HTTPException(status_code=404, detail="Variante no encontrada")
    doc = {
        "item_variant_id": variant_id,
        "estimated_value": body.estimated_value,
        "recorded_date": body.recorded_date or date.today().isoformat(),
        "price_source": body.price_source,
        "created_at": _now(),
    }
    res = db.item_value_history.insert_one(doc)
    doc["_id"] = res.inserted_id
    return ValueHistoryOut(**serialize(doc))  # type: ignore[arg-type]


@values_router.get("/variants/{variant_id}/latest", response_model=ValueHistoryOut)
def latest_value(variant_id: str) -> ValueHistoryOut:
    doc = (
        get_db()
        .item_value_history.find_one({"item_variant_id": variant_id}, sort=[("recorded_date", DESCENDING)])
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Sin valoraciones")
    return ValueHistoryOut(**serialize(doc))  # type: ignore[arg-type]


# ── Transacciones ─────────────────────────────────────────────────────────


@tx_router.get("", response_model=list[TransactionOut])
def list_transactions(user: CurrentUser = Depends(get_current_user)) -> list[TransactionOut]:
    docs = list(
        get_db().transactions.find({"user_id": user.id}).sort("created_at", DESCENDING)
    )
    return [TransactionOut(**serialize(d)) for d in docs]  # type: ignore[arg-type]


@tx_router.post("", response_model=TransactionOut, status_code=201)
def add_transaction(
    body: TransactionCreate,
    user: CurrentUser = Depends(get_current_user),
) -> TransactionOut:
    doc = body.model_dump()
    doc.update({"user_id": user.id, "created_at": _now()})
    res = get_db().transactions.insert_one(doc)
    doc["_id"] = res.inserted_id
    return TransactionOut(**serialize(doc))  # type: ignore[arg-type]


@tx_router.delete("/{tx_id}", status_code=204)
def delete_transaction(
    tx_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    db = get_db()
    res = db.transactions.delete_one({"_id": oid(tx_id), "user_id": user.id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Transacción no encontrada")


# ── Propuestas de variante ────────────────────────────────────────────────


@proposals_router.get("", response_model=list[VariantProposalOut])
def list_proposals(user: CurrentUser = Depends(get_current_user)) -> list[VariantProposalOut]:
    docs = list(
        get_db().variant_proposals.find({"user_id": user.id}).sort("created_at", DESCENDING)
    )
    return [VariantProposalOut(**serialize(d)) for d in docs]  # type: ignore[arg-type]


@proposals_router.post("", response_model=VariantProposalOut, status_code=201)
def submit_proposal(
    body: VariantProposalCreate,
    user: CurrentUser = Depends(get_current_user),
) -> VariantProposalOut:
    doc = body.model_dump()
    doc.update({"user_id": user.id, "status": "pending", "created_at": _now()})
    res = get_db().variant_proposals.insert_one(doc)
    doc["_id"] = res.inserted_id
    return VariantProposalOut(**serialize(doc))  # type: ignore[arg-type]
