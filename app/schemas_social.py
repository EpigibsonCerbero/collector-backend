"""Pydantic models — social / trading / misiones / eventos / chat."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ── Trade hub (publicaciones de venta/trueque) ────────────────────────────


class TradeListingCreate(BaseModel):
    variant_id: str
    variant_name: str
    item_name: str
    image_url: str = ""
    type: Literal["sell", "trade", "both"] = "trade"
    asking_price: Optional[float] = Field(default=None, ge=0)
    looking_for: Optional[str] = None
    condition: str = "Mint"
    location: str = ""


class TradeListingOut(BaseModel):
    id: str
    user_id: str
    username: str
    avatar_emoji: str = "🏎️"
    variant_id: str
    variant_name: str
    item_name: str
    image_url: str = ""
    type: str = "trade"
    asking_price: Optional[float] = None
    looking_for: Optional[str] = None
    condition: str = "Mint"
    location: str = ""
    created_at: Optional[str] = None
    is_own: bool = False


# ── Ofertas de trueque ────────────────────────────────────────────────────


class OfferItem(BaseModel):
    variant_id: str
    variant_name: str
    image_url: str = ""
    condition: str = "Mint"


class TradeOfferCreate(BaseModel):
    to_user_id: str
    to_username: str = ""
    from_avatar: str = ""
    offered_items: list[OfferItem] = Field(default_factory=list)
    requested_items: list[OfferItem] = Field(default_factory=list)
    message: Optional[str] = None


class TradeOfferOut(BaseModel):
    id: str
    type: Literal["incoming", "outgoing"]
    status: str = "pending"
    from_user_id: str
    from_username: str
    from_avatar: str = ""
    to_user_id: str
    to_username: str
    to_avatar: str = ""
    offered_items: list[OfferItem] = Field(default_factory=list)
    requested_items: list[OfferItem] = Field(default_factory=list)
    message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class EscrowStep(BaseModel):
    id: str
    label: str
    status: Literal["completed", "active", "pending"] = "pending"
    timestamp: Optional[str] = None


class TradeEscrowOut(BaseModel):
    id: str
    offer_id: str
    partner_name: str
    item_name: str
    steps: list[EscrowStep] = Field(default_factory=list)
    tracking_number: Optional[str] = None
    status: str = "in_progress"
    created_at: Optional[str] = None


class EscrowUpdate(BaseModel):
    status: Optional[Literal["in_progress", "shipped", "delivered", "completed"]] = None
    tracking_number: Optional[str] = None
    steps: Optional[list[EscrowStep]] = None


# ── Misiones diarias / XP ─────────────────────────────────────────────────


class MissionOut(BaseModel):
    id: str
    user_id: str
    date: str
    title: str
    description: str
    icon: str = "🏁"
    xp_reward: int = 25
    type: str = "scan"
    completed: bool = False
    progress: float = 0.0
    current: int = 0
    target: int = 1


class MissionProgress(BaseModel):
    increment: int = 1


class MissionStateOut(BaseModel):
    """Estado global de misiones del usuario (nivel/XP/racha)."""

    missions: list[MissionOut] = Field(default_factory=list)
    streak: int = 0
    longest_streak: int = 0
    total_xp: int = 0
    level: int = 1
    xp_for_next_level: int = 100
    next_level_threshold: int = 100
    last_active_date: Optional[str] = None


# ── Social feed ───────────────────────────────────────────────────────────


class FeedEventCreate(BaseModel):
    type: Literal[
        "added_item",
        "completed_series",
        "leveled_up",
        "traded",
        "achievement",
        "milestone",
    ]
    title: str
    description: str
    item_name: Optional[str] = None
    image_url: Optional[str] = None


class FeedEventOut(BaseModel):
    id: str
    user_id: str
    username: str
    avatar: str = ""
    type: str
    title: str
    description: str
    item_name: Optional[str] = None
    image_url: Optional[str] = None
    timestamp: Optional[str] = None
    reactions: dict[str, int] = Field(default_factory=dict)
    user_reaction: Optional[str] = None


class FeedReaction(BaseModel):
    reaction: Literal["🔥", "❤️", "👑", "🤩"]


# ── Amigos ────────────────────────────────────────────────────────────────


class FriendOut(BaseModel):
    id: str
    user_id: str
    friend_id: str
    username: str
    avatar: str = ""
    location: str = ""
    pieces_count: int = 0
    rarities_count: int = 0
    reputation: str = "Novato"
    verified_tsm: bool = False
    recent_add: str = ""
    specialties: list[str] = Field(default_factory=list)
    is_online: bool = False
    added_at: Optional[str] = None


class FriendAdd(BaseModel):
    # Se resuelve contra los usuarios reales: `username` (sin @, sin mayúsculas
    # exactas) o, si no, `friend_id` como usuario o id.
    friend_id: str = ""
    username: str = ""
    avatar: str = ""
    location: str = ""
    specialties: list[str] = Field(default_factory=list)


class ShowcaseItemOut(BaseModel):
    id: str
    friend_id: str
    variant_id: str = ""
    variant_name: str
    livery: str = ""
    box_number: str = ""
    image_url: str = ""
    condition: str = "Mint"
    value: float = 0
    is_tradeable: bool = False


# ── Eventos ───────────────────────────────────────────────────────────────


class EventCreate(BaseModel):
    name: str
    type: Literal["convention", "store", "meetup", "swap", "online_drop"]
    category: str = "Mini GT"
    city: str = ""
    venue: str = ""
    date: str
    attendees: str = "0"
    badges: list[str] = Field(default_factory=list)
    icon: str = "🏁"
    color: str = "#E10600"
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class EventOut(BaseModel):
    id: str
    user_id: Optional[str] = None
    name: str
    type: str
    category: str = "Mini GT"
    city: str = ""
    venue: str = ""
    date: str
    attendees: str = "0"
    badges: list[str] = Field(default_factory=list)
    icon: str = "🏁"
    color: str = "#E10600"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    is_favorite: bool = False
    is_custom: bool = False


# ── Chat P2P ──────────────────────────────────────────────────────────────


class ChatMessageCreate(BaseModel):
    sender: Literal["partner", "user", "system"] = "user"
    author_name: Optional[str] = None
    text: Optional[str] = Field(default=None, max_length=4000)
    type: Literal["text", "image", "gif", "voice", "piece", "photo_verification", "escrow_alert"] = "text"
    photo_url: Optional[str] = None
    voice_url: Optional[str] = None
    voice_duration: Optional[float] = None
    piece_id: Optional[str] = None
    piece_name: Optional[str] = Field(default=None, max_length=200)
    piece_image: Optional[str] = None
    piece_meta: Optional[str] = Field(default=None, max_length=120)
    seal_code: Optional[str] = None
    condition: Optional[str] = None
    amount: Optional[str] = None
    reply_to_id: Optional[str] = None
    reply_to_text: Optional[str] = Field(default=None, max_length=280)
    # Id que genera el cliente para casar su burbuja «enviando…» con el eco del servidor.
    client_id: Optional[str] = Field(default=None, max_length=64)


class ChatMessageOut(BaseModel):
    id: str
    thread_id: str
    sender: str
    author_id: Optional[str] = None
    author_name: Optional[str] = None
    timestamp: str
    text: Optional[str] = None
    type: str = "text"
    photo_url: Optional[str] = None
    voice_url: Optional[str] = None
    voice_duration: Optional[float] = None
    piece_id: Optional[str] = None
    piece_name: Optional[str] = None
    piece_image: Optional[str] = None
    piece_meta: Optional[str] = None
    seal_code: Optional[str] = None
    condition: Optional[str] = None
    amount: Optional[str] = None
    reply_to_id: Optional[str] = None
    reply_to_text: Optional[str] = None
    # Una reacción por usuario ({user_id: emoji}); `reactions` = sus emojis, para pintar.
    reactions: list[str] = Field(default_factory=list)
    reactions_by: dict[str, str] = Field(default_factory=dict)
    client_id: Optional[str] = None


class Page(BaseModel):
    items: list[Any]
    total: int
    limit: int
    offset: int
