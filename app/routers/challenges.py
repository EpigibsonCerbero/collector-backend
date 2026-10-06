"""Retos y desafíos de la comunidad."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.db import get_db, oid, serialize
from app.security import CurrentUser, get_current_user

router = APIRouter(prefix="/challenges", tags=["challenges"])


class ChallengeOut(BaseModel):
    id: str
    title: str
    description: str
    icon: str = "🏁"
    reward_xp: int = 50
    progress: float = 0.0
    target: int = 1
    completed: bool = False
    ends_at: str | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


@router.get("", response_model=list[ChallengeOut])
def list_challenges(
    user: CurrentUser = Depends(get_current_user),
) -> list[ChallengeOut]:
    db = get_db()
    docs = list(db.challenges.find({"user_id": user.id}).sort("ends_at", 1))
    return [ChallengeOut(**serialize(d)) for d in docs]


@router.post("/{challenge_id}/progress")
def update_challenge_progress(
    challenge_id: str,
    increment: int = 1,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    db = get_db()
    db.challenges.update_one(
        {"_id": oid(challenge_id), "user_id": user.id},
        {"$inc": {"progress": increment}},
    )
    return {"ok": True}
