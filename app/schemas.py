"""Pydantic request/response models — core domains (auth, catálogo, garaje)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, EmailStr, Field

# ── Auth / perfil ─────────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: "UserProfileOut"


class UserProfileOut(BaseModel):
    id: str
    email: str
    username: str
    avatar_url: Optional[str] = None
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ProfileUpdate(BaseModel):
    username: Optional[str] = Field(default=None, min_length=3, max_length=50)
    avatar_url: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ForgotPasswordResponse(BaseModel):
    """Sin SMTP configurado el código se entrega en la respuesta (self-hosted)."""

    ok: bool = True
    # "email": enviado por correo · "unavailable": correo sin configurar.
    # El código JAMÁS viaja en la respuesta (cualquiera con el email tomaba la cuenta).
    delivery: Literal["email", "unavailable"] = "email"
    reset_code: Optional[str] = None
    expires_in_min: int = 30


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    code: str = Field(min_length=4, max_length=12)
    new_password: str = Field(min_length=8, max_length=128)


class MagicCodeRequest(BaseModel):
    email: EmailStr


class MagicCodeResponse(BaseModel):
    """Self-hosted: sin SMTP el código de acceso se entrega en la respuesta."""

    ok: bool = True
    delivery: Literal["email", "unavailable"] = "email"
    code: Optional[str] = None
    expires_in_min: int = 10


class MagicLoginRequest(BaseModel):
    email: EmailStr
    code: str = Field(min_length=4, max_length=12)
    totp_code: Optional[str] = None


# ── Catálogo: marcas / series / categorías ────────────────────────────────


class BrandOut(BaseModel):
    id: str
    name: str
    slug: str
    logo_url: Optional[str] = None
    country: Optional[str] = None
    website_url: Optional[str] = None
    default_scale: str = "1:64"
    created_at: Optional[str] = None


class BrandCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    slug: str = Field(min_length=1, max_length=100)
    logo_url: Optional[str] = None
    country: Optional[str] = None
    website_url: Optional[str] = None
    default_scale: str = "1:64"


class SeriesOut(BaseModel):
    id: str
    brand_id: str
    name: str
    release_year: int
    segment_name: Optional[str] = None
    total_items_count: int = 0
    banner_url: Optional[str] = None
    assortment_code: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SeriesCreate(BaseModel):
    brand_id: str
    name: str = Field(min_length=1, max_length=150)
    release_year: int
    segment_name: Optional[str] = None
    total_items_count: int = 0
    banner_url: Optional[str] = None
    assortment_code: Optional[str] = None


class CategoryOut(BaseModel):
    id: str
    name: str
    slug: str
    description: Optional[str] = None
    created_at: Optional[str] = None


class CategoryCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    slug: str = Field(min_length=1, max_length=100)
    description: Optional[str] = None


# ── Catálogo: items_master + item_variants ────────────────────────────────


class CatalogItemOut(BaseModel):
    """items_master denormalizado dentro de la variante."""

    id: str
    name: str
    manufacturer: str
    release_year: int
    # category_id puede ser int (seed MiniGT puso 1) o str — aceptar ambos
    category_id: Optional[str | int] = None
    brand_id: Optional[str | int] = None
    series_id: Optional[str | int] = None
    casting_name: Optional[str] = None
    designer: Optional[str] = None
    scale: Optional[str] = "1:64"


class ItemVariantOut(BaseModel):
    id: str
    item_master_id: str
    variant_name: str
    sku_upc: Optional[str] = None
    main_image_url: Optional[str] = None
    # Vistas adicionales del packshot del catálogo (el scrape trae hasta 6).
    # La ficha de la app las muestra como galería; `main_image_url` sigue
    # siendo la foto principal (la oficial del catálogo).
    images: list[str] = Field(default_factory=list)
    # Livery semántico (curado con visión): descripción del diseño + color y
    # dorsal declarados. Lo usa el identify para desampatar gemelos y la app
    # para mostrar "qué lo hace distinto".
    livery_desc: Optional[str] = None
    livery_color: Optional[str] = None
    livery_number: Optional[str] = None
    ai_local_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    toy_number: Optional[str] = None
    rarity: str = "regular"
    body_color: Optional[str] = None
    premium_finish: Optional[str] = None
    wheel_type: Optional[str] = None
    tampo_details: Optional[str] = None
    base_material: str = "Metal/Plastic"
    country_of_origin: Optional[str] = None
    created_at: Optional[str] = None
    item_master: Optional[CatalogItemOut] = None


class CatalogItemCreate(BaseModel):
    """Upsert master+variante (equivale a api.createCatalogItem)."""

    # Variante
    id: Optional[str] = None  # preserva uuid5 determinista del scrape
    variant_name: str = Field(min_length=1, max_length=150)
    sku_upc: Optional[str] = None
    main_image_url: Optional[str] = None
    ai_local_hash: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    toy_number: Optional[str] = None
    rarity: str = "regular"
    body_color: Optional[str] = None
    premium_finish: Optional[str] = None
    wheel_type: Optional[str] = None
    tampo_details: Optional[str] = None
    base_material: str = "Metal/Plastic"
    country_of_origin: Optional[str] = None
    # Maestro
    name: str = Field(min_length=1, max_length=255)
    manufacturer: str = Field(min_length=1, max_length=150)
    release_year: int
    category_id: Optional[str] = None
    brand_id: Optional[str] = None
    series_id: Optional[str] = None
    casting_name: Optional[str] = None
    designer: Optional[str] = None
    scale: str = "1:64"


# ── Garaje: collections + user_items ──────────────────────────────────────


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    description: Optional[str] = None
    is_private: bool = False


class CollectionUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=150)
    description: Optional[str] = None
    is_private: Optional[bool] = None


class CollectionOut(BaseModel):
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    is_private: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


PackagingType = Literal[
    "Blister/Carded",
    "Short Card",
    "Long Card",
    "Box",
    "Acrylic Case",
    "Loose",
    "Special Edition",
    "Other",
]


class UserItemCreate(BaseModel):
    collection_id: str
    item_variant_id: str
    condition: str = "Mint"
    packaging_type: PackagingType = "Blister/Carded"
    notes: Optional[str] = None
    acquired_date: Optional[str] = None  # ISO date
    purchase_price: Optional[float] = Field(default=None, ge=0)
    custom_images: list[str] = Field(default_factory=list)
    quantity: int = Field(default=1, ge=1)
    is_tradeable: bool = False
    is_for_sale: bool = False
    sale_price: Optional[float] = Field(default=None, ge=0)
    card_condition: str = "Mint"
    has_protector: bool = False
    storage_location: Optional[str] = None


class UserItemUpdate(BaseModel):
    condition: Optional[str] = None
    packaging_type: Optional[PackagingType] = None
    notes: Optional[str] = None
    acquired_date: Optional[str] = None
    purchase_price: Optional[float] = Field(default=None, ge=0)
    custom_images: Optional[list[str]] = None
    quantity: Optional[int] = Field(default=None, ge=1)
    is_tradeable: Optional[bool] = None
    is_for_sale: Optional[bool] = None
    sale_price: Optional[float] = Field(default=None, ge=0)
    card_condition: Optional[str] = None
    has_protector: Optional[bool] = None
    storage_location: Optional[str] = None


class UserItemOut(BaseModel):
    id: str
    collection_id: str
    user_id: str
    item_variant_id: str
    condition: str = "Mint"
    packaging_type: str = "Blister/Carded"
    notes: Optional[str] = None
    acquired_date: Optional[str] = None
    purchase_price: Optional[float] = None
    custom_images: list[str] = Field(default_factory=list)
    quantity: int = 1
    is_tradeable: bool = False
    is_for_sale: bool = False
    sale_price: Optional[float] = None
    card_condition: str = "Mint"
    has_protector: bool = False
    storage_location: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    item_variant: Optional[ItemVariantOut] = None


# ── Listas (wishlist / trade / sale / custom) ─────────────────────────────


class UserListCreate(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    description: Optional[str] = None
    list_type: Literal["wishlist", "trade", "sale", "custom"] = "custom"
    is_private: bool = False
    auto_remove_on_garage_add: bool = True


class UserListOut(BaseModel):
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    list_type: str = "custom"
    is_private: bool = False
    auto_remove_on_garage_add: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class UserListItemCreate(BaseModel):
    item_variant_id: str
    priority: int = Field(default=1, ge=1, le=3)
    target_price: Optional[float] = Field(default=None, ge=0)
    notes: Optional[str] = None


class UserListItemOut(BaseModel):
    id: str
    user_list_id: str
    item_variant_id: str
    priority: int = 1
    target_price: Optional[float] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None
    item_variant: Optional[ItemVariantOut] = None


# ── Valoración ────────────────────────────────────────────────────────────


class ValueHistoryCreate(BaseModel):
    estimated_value: float = Field(ge=0)
    recorded_date: Optional[str] = None  # ISO date; default today
    price_source: str = "Market Average"


class ValueHistoryOut(BaseModel):
    id: str
    item_variant_id: str
    estimated_value: float
    recorded_date: str
    price_source: str = "Market Average"
    created_at: Optional[str] = None


# ── Transacciones (compra/venta) ──────────────────────────────────────────


class TransactionCreate(BaseModel):
    type: Literal["purchase", "sale", "trade", "gift", "other"] = "purchase"
    amount: Optional[float] = Field(default=None, ge=0)
    counterpart: Optional[str] = None
    notes: Optional[str] = None
    item_variant_id: Optional[str] = None
    item_name: Optional[str] = Field(default=None, max_length=200)


class TransactionOut(BaseModel):
    id: str
    user_id: str
    type: str
    amount: Optional[float] = None
    counterpart: Optional[str] = None
    notes: Optional[str] = None
    item_variant_id: Optional[str] = None
    item_name: Optional[str] = None
    created_at: Optional[str] = None


# ── Propuestas de variante (crowdsourced catalog) ─────────────────────────


class VariantProposalCreate(BaseModel):
    variant_name: str = Field(min_length=1, max_length=150)
    toy_number: Optional[str] = None
    brand: Optional[str] = None
    series: Optional[str] = None
    body_color: Optional[str] = None
    image_url: Optional[str] = None


class VariantProposalOut(BaseModel):
    id: str
    user_id: str
    variant_name: str
    toy_number: Optional[str] = None
    brand: Optional[str] = None
    series: Optional[str] = None
    body_color: Optional[str] = None
    image_url: Optional[str] = None
    status: str = "pending"
    created_at: Optional[str] = None


# ── Contribución de specs técnicas (crowdsourcing) ──────────────────────────

# Campos contribuibles — los que alimentan X-Ray Chassis / Certificado TSM
SPEC_FIELDS = [
    "base_material",
    "wheel_type",
    "tampo_details",
    "premium_finish",
    "country_of_origin",
    "interior_material",
    "chassis_material",
    "window_material",
    "typical_weight_g",
    "scale",
]


class SpecContributionCreate(BaseModel):
    variant_id: str
    fields: dict[str, str]  # campo → valor propuesto
    source: Optional[str] = None  # "empaque", "web oficial", "inspección física", etc.
    notes: Optional[str] = None


class SpecContributionOut(BaseModel):
    id: str
    variant_id: str
    user_id: str
    fields: dict[str, str]
    source: Optional[str] = None
    notes: Optional[str] = None
    status: str = "pending"  # pending | confirmed | rejected
    confirmations: int = 0
    created_at: Optional[str] = None


class SpecFieldStatus(BaseModel):
    field: str
    value: Optional[str] = None
    is_default: bool = True  # True si viene de defaults, False si es data real
    pending_contributions: int = 0


# ── Series progress (vista) ───────────────────────────────────────────────


class SeriesProgressOut(BaseModel):
    series_id: str
    brand_id: str
    brand_name: Optional[str] = None
    series_name: str
    release_year: int
    segment_name: Optional[str] = None
    total_items_count: int = 0
    items_collected_count: int = 0
    completion_percentage: float = 0.0


class UpcExtractOut(BaseModel):
    upc: Optional[str] = None
    toy_number: Optional[str] = None
    found: bool = False
    saved: bool = False
    message: str = ""
