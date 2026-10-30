"""Leaderboard y rankings de la comunidad (datos reales de todos los usuarios)."""
from __future__ import annotations

from typing import Literal

from bson import ObjectId
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.db import get_db
from app.security import CurrentUser, get_current_user

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


class LeaderboardEntry(BaseModel):
    rank: int
    user_id: str
    username: str
    avatar: str = ""
    score: float = 0
    pieces_count: int = 0
    value: float = 0
    distinct_models: int = 0
    total_xp: int = 0
    is_current_user: bool = False


_SORT = {"items": "pieces_count", "value": "value", "diversity": "distinct_models", "achievements": "total_xp"}


@router.get("", response_model=list[LeaderboardEntry])
def get_leaderboard(
    user: CurrentUser = Depends(get_current_user),
    limit: int = 50,
    by: Literal["items", "value", "diversity", "achievements"] = "items",
) -> list[LeaderboardEntry]:
    """Antes: `$lookup` de users._id (ObjectId) contra user_items.user_id (texto):
    nunca casaba y todos salían con 0 piezas. Se agrupa por el campo real."""
    db = get_db()
    limit = max(1, min(limit, 100))
    stats = {
        str(a["_id"]): a
        for a in db.user_items.aggregate(
            [
                {
                    "$group": {
                        "_id": "$user_id",
                        "pieces": {"$sum": {"$ifNull": ["$quantity", 1]}},
                        "value": {
                            "$sum": {
                                "$multiply": [
                                    {"$ifNull": ["$purchase_price", 0]},
                                    {"$ifNull": ["$quantity", 1]},
                                ]
                            }
                        },
                        "models": {"$addToSet": "$item_variant_id"},
                    }
                }
            ]
        )
        if a.get("_id")
    }
    ids = [ObjectId(uid) for uid in stats if ObjectId.is_valid(uid)]
    rows = []
    for u in db.users.find(
        {"_id": {"$in": ids}, "is_active": {"$ne": False}},
        {"username": 1, "avatar_url": 1, "total_xp": 1},
    ):
        uid = str(u["_id"])
        st = stats[uid]
        rows.append(
            {
                "user_id": uid,
                "username": u.get("username") or "coleccionista",
                "avatar": u.get("avatar_url") or "",
                "pieces_count": int(st.get("pieces") or 0),
                "value": round(float(st.get("value") or 0), 2),
                "distinct_models": len(st.get("models") or []),
                "total_xp": int(u.get("total_xp") or 0),
            }
        )
    key = _SORT[by]
    rows.sort(key=lambda r: (r[key], r["pieces_count"]), reverse=True)
    out = []
    for rank, r in enumerate(rows[:limit], 1):
        out.append(
            LeaderboardEntry(
                rank=rank,
                score=float(r[key]),
                is_current_user=r["user_id"] == user.id,
                **r,
            )
        )
    # El usuario siempre ve su posición aunque quede fuera del top.
    if not any(e.is_current_user for e in out):
        for rank, r in enumerate(rows, 1):
            if r["user_id"] == user.id:
                out.append(LeaderboardEntry(rank=rank, score=float(r[key]), is_current_user=True, **r))
                break
    return out
