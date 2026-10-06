"""Envíos y seguimiento de paquetes."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.db import get_db, oid, serialize
from app.security import CurrentUser, get_current_user

router = APIRouter(prefix="/shipping", tags=["shipping"])


class ShippingOut(BaseModel):
    id: str
    tracking_number: str
    carrier: str = "DHL"
    status: str = "pending"
    from_user: str = ""
    to_user: str = ""
    item_name: str = ""
    created_at: str
    estimated_delivery: str | None = None
    updates: list[dict] = []


def _now() -> datetime:
    return datetime.now(timezone.utc)


@router.get("", response_model=list[ShippingOut])
def list_shippings(
    user: CurrentUser = Depends(get_current_user),
) -> list[ShippingOut]:
    db = get_db()
    docs = list(
        db.shippings.find(
            {"$or": [{"from_user": user.id}, {"to_user": user.id}]}
        ).sort("created_at", -1)
    )
    return [ShippingOut(**serialize(d)) for d in docs]


@router.post("", response_model=ShippingOut, status_code=201)
def create_shipping(
    tracking_number: str,
    item_name: str,
    to_user: str,
    user: CurrentUser = Depends(get_current_user),
) -> ShippingOut:
    db = get_db()
    doc = {
        "tracking_number": tracking_number,
        "carrier": "DHL",
        "status": "pending",
        "from_user": user.id,
        "to_user": to_user,
        "item_name": item_name,
        "created_at": _now().isoformat(),
        "updates": [],
    }
    res = db.shippings.insert_one(doc)
    doc["_id"] = res.inserted_id
    return ShippingOut(**serialize(doc))
