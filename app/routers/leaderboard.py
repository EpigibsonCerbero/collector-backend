"""Leaderboard y rankings de la comunidad."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.db import get_db
from app.security import CurrentUser, get_current_user

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


class LeaderboardEntry(BaseModel):
    rank: int
    user_id: str
    username: str
    avatar: str = "🏎️"
    score: float = 0
    pieces_count: int = 0
    is_current_user: bool = False


@router.get("", response_model=list[LeaderboardEntry])
def get_leaderboard(
    user: CurrentUser = Depends(get_current_user),
    limit: int = 50,
) -> list[LeaderboardEntry]:
    db = get_db()
    
    # Aggregate user scores (based on items count + trades + missions)
    pipeline = [
        {
            "$lookup": {
                "from": "user_items",
                "localField": "_id",
                "foreignField": "user_id",
                "as": "items",
            }
        },
        {
            "$addFields": {
                "pieces_count": {"$size": "$items"},
                "score": {"$multiply": [{"$size": "$items"}, 10]},
            }
        },
        {"$sort": {"score": -1}},
        {"$limit": limit},
    ]
    
    users = list(db.users.aggregate(pipeline))
    
    out = []
    for rank, u in enumerate(users, 1):
        uid = str(u["_id"])
        out.append(
            LeaderboardEntry(
                rank=rank,
                user_id=uid,
                username=u.get("username") or "Unknown",
                avatar=u.get("avatar_url") or "🏎️",
                score=float(u.get("score") or 0),
                pieces_count=int(u.get("pieces_count") or 0),
                is_current_user=uid == user.id,
            )
        )

    return out
