"""Auth: registro, login, refresh, perfil y restablecimiento de contraseña."""
from __future__ import annotations

import re

import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pymongo import ReturnDocument

from app.db import get_db, oid, serialize
from app.ratelimit import client_ip, fail_check, fail_clear, fail_record, hit_or_429
from app.services.mailer import mail_enabled, send_mail
from app.schemas import (
    ChangePasswordRequest,
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    LoginRequest,
    MagicCodeRequest,
    MagicCodeResponse,
    MagicLoginRequest,
    ProfileUpdate,
    RefreshRequest,
    RegisterRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserProfileOut,
)
from app.security import (
    CurrentUser,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])
me_router = APIRouter(prefix="/me", tags=["me"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _user_out(doc: dict) -> UserProfileOut:
    data = serialize(doc)
    assert data is not None
    return UserProfileOut(**data)


def _tokens(user_doc: dict) -> TokenResponse:
    uid = str(user_doc["_id"])
    username = user_doc["username"]
    tv = int(user_doc.get("token_version", 0))
    return TokenResponse(
        access_token=create_access_token(uid, username, tv),
        refresh_token=create_refresh_token(uid, username, tv),
        user=_user_out(user_doc),
    )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest) -> TokenResponse:
    db = get_db()
    email = body.email.lower()
    if db.users.find_one({"email": email}):
        raise HTTPException(status_code=409, detail="El email ya está registrado")
    if db.users.find_one({"username": body.username}):
        raise HTTPException(status_code=409, detail="El nombre de usuario ya existe")

    now = _now()
    doc = {
        "email": email,
        "username": body.username,
        "pwd_hash": hash_password(body.password),
        "avatar_url": None,
        "is_active": True,
        "created_at": now,
        "updated_at": now,
    }
    res = db.users.insert_one(doc)
    doc["_id"] = res.inserted_id

    # Colección "Garaje" por defecto
    db.collections.insert_one(
        {
            "user_id": str(res.inserted_id),
            "name": "Mi Garaje",
            "description": "Colección principal",
            "is_private": False,
            "created_at": now,
            "updated_at": now,
        }
    )
    return _tokens(doc)


@router.post("/login")
def login(body: dict, request: Request):
    db = get_db()
    identifier = str(body.get("email", "")).lower().strip()
    password = str(body.get("password", ""))
    totp_code = str(body.get("totp_code", "") or "")
    ip = client_ip(request)
    key = f"login:{identifier}"
    fail_check(db, key, ip)

    # Acepta email O username: si trae @, busca por email; si no, por username
    if "@" in identifier:
        user = db.users.find_one({"email": identifier})
    else:
        user = db.users.find_one({"username": {"$regex": f"^{re.escape(identifier)}$", "$options": "i"}})

    if not user or not verify_password(password, user["pwd_hash"]):
        fail_record(db, key, ip)
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Cuenta desactivada")

    # Si 2FA está activo, requerir código TOTP
    if user.get("totp_enabled"):
        if not totp_code:
            return {
                "requires_2fa": True,
                "message": "Se requiere código de verificación",
            }
        import pyotp
        secret = user.get("totp_secret")
        if not secret:
            raise HTTPException(status_code=500, detail="Error de configuración 2FA")
        totp = pyotp.TOTP(secret)
        if not totp.verify(totp_code, valid_window=1):
            fail_record(db, key, ip)
            raise HTTPException(status_code=401, detail="Código 2FA inválido")

    fail_clear(db, key)
    return _tokens(user)


@router.post("/refresh", response_model=TokenResponse)
def refresh(body: RefreshRequest) -> TokenResponse:
    payload = decode_token(body.refresh_token, expected_type="refresh")
    db = get_db()
    user = db.users.find_one({"_id": oid(payload["sub"])})
    if not user or not user.get("is_active", True):
        raise HTTPException(status_code=401, detail="Usuario no válido")
    if int(payload.get("tv", 0)) != int(user.get("token_version", 0)):
        raise HTTPException(status_code=401, detail="Sesión cerrada. Inicia sesión de nuevo.")
    return _tokens(user)


@me_router.get("", response_model=UserProfileOut)
def read_me(user: CurrentUser = Depends(get_current_user)) -> UserProfileOut:
    doc = get_db().users.find_one({"_id": oid(user.id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return _user_out(doc)


@me_router.patch("", response_model=UserProfileOut)
def update_me(
    body: ProfileUpdate,
    user: CurrentUser = Depends(get_current_user),
) -> UserProfileOut:
    db = get_db()
    updates: dict = {"updated_at": _now()}
    if body.username is not None:
        if db.users.find_one({"username": body.username, "_id": {"$ne": oid(user.id)}}):
            raise HTTPException(status_code=409, detail="El nombre de usuario ya existe")
        updates["username"] = body.username
    if body.avatar_url is not None:
        updates["avatar_url"] = body.avatar_url
    doc = db.users.find_one_and_update(
        {"_id": oid(user.id)},
        {"$set": updates},
        return_document=ReturnDocument.AFTER,
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return _user_out(doc)


RESET_TTL_MIN = 30


@router.post("/change-password", response_model=TokenResponse)
def change_password(
    body: ChangePasswordRequest,
    user: CurrentUser = Depends(get_current_user),
) -> TokenResponse:
    db = get_db()
    doc = db.users.find_one({"_id": oid(user.id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    if not verify_password(body.current_password, doc["pwd_hash"]):
        raise HTTPException(status_code=401, detail="La contraseña actual es incorrecta")
    # Sube la versión de token: el resto de sesiones se cierran; esta recibe tokens nuevos.
    doc = db.users.find_one_and_update(
        {"_id": oid(user.id)},
        {
            "$set": {"pwd_hash": hash_password(body.new_password), "updated_at": _now()},
            "$inc": {"token_version": 1},
        },
        return_document=ReturnDocument.AFTER,
    )
    return _tokens(doc)


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
def forgot_password(body: ForgotPasswordRequest, request: Request) -> ForgotPasswordResponse:
    """Envía por correo un código de restablecimiento.

    Nunca devuelve el código. Sin SMTP responde `unavailable`. Si la cuenta no
    existe responde igual (no filtra usuarios registrados).
    """
    hit_or_429("forgot", client_ip(request), limit=5, per_seconds=900)
    if not mail_enabled():
        return ForgotPasswordResponse(delivery="unavailable")
    db = get_db()
    user = db.users.find_one({"email": body.email.lower()})
    if not user or not user.get("is_active", True):
        return ForgotPasswordResponse()

    code = f"{secrets.randbelow(1_000_000):06d}"
    db.users.update_one(
        {"_id": user["_id"]},
        {
            "$set": {
                "pwd_reset": {
                    "code_hash": hash_password(code),
                    "expires_at": _now() + timedelta(minutes=RESET_TTL_MIN),
                    "attempts": 0,
                },
                "updated_at": _now(),
            }
        },
    )
    send_mail(
        user["email"],
        "Tu código para restablecer la contraseña de CURIO",
        f"Tu código es: {code}\n\nCaduca en {RESET_TTL_MIN} minutos. Si no lo pediste, ignora este correo.",
    )
    return ForgotPasswordResponse(expires_in_min=RESET_TTL_MIN)


@router.post("/reset-password", status_code=status.HTTP_204_NO_CONTENT)
def reset_password(body: ResetPasswordRequest, request: Request) -> None:
    db = get_db()
    key, ip = f"reset:{body.email.lower()}", client_ip(request)
    fail_check(db, key, ip)
    user = db.users.find_one({"email": body.email.lower()})
    if not user:
        raise HTTPException(status_code=400, detail="Código inválido o expirado")
    reset = user.get("pwd_reset") or {}
    expires_at = reset.get("expires_at")
    if isinstance(expires_at, datetime) and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if not expires_at or not isinstance(expires_at, datetime) or expires_at < _now():
        raise HTTPException(status_code=400, detail="Código inválido o expirado")
    if int(reset.get("attempts", 0)) >= 5:
        raise HTTPException(status_code=429, detail="Demasiados intentos. Solicita un código nuevo.")
    if not verify_password(body.code, reset.get("code_hash", "")):
        db.users.update_one({"_id": user["_id"]}, {"$inc": {"pwd_reset.attempts": 1}})
        fail_record(db, key, ip)
        raise HTTPException(status_code=400, detail="Código inválido o expirado")

    db.users.update_one(
        {"_id": user["_id"]},
        {
            "$set": {"pwd_hash": hash_password(body.new_password), "updated_at": _now()},
            "$unset": {"pwd_reset": ""},
            "$inc": {"token_version": 1},
        },
    )
    fail_clear(db, key)


MAGIC_TTL_MIN = 10


@router.post("/magic-code", response_model=MagicCodeResponse)
def magic_code(body: MagicCodeRequest, request: Request) -> MagicCodeResponse:
    """Envía por correo un código de acceso de un solo uso (equivale al magic link).

    Nunca devuelve el código. Sin SMTP responde `unavailable`. Si la cuenta no
    existe responde igual (no filtra usuarios registrados).
    """
    hit_or_429("magic", client_ip(request), limit=5, per_seconds=900)
    if not mail_enabled():
        return MagicCodeResponse(delivery="unavailable")
    db = get_db()
    user = db.users.find_one({"email": body.email.lower()})
    if not user or not user.get("is_active", True):
        return MagicCodeResponse()

    code = f"{secrets.randbelow(1_000_000):06d}"
    db.users.update_one(
        {"_id": user["_id"]},
        {
            "$set": {
                "magic_code": {
                    "code_hash": hash_password(code),
                    "expires_at": _now() + timedelta(minutes=MAGIC_TTL_MIN),
                    "attempts": 0,
                },
                "updated_at": _now(),
            }
        },
    )
    send_mail(
        user["email"],
        "Tu código de acceso a CURIO",
        f"Tu código es: {code}\n\nCaduca en {MAGIC_TTL_MIN} minutos. Si no lo pediste, ignora este correo.",
    )
    return MagicCodeResponse(expires_in_min=MAGIC_TTL_MIN)


@router.post("/magic-login", response_model=TokenResponse)
def magic_login(body: MagicLoginRequest, request: Request) -> TokenResponse:
    db = get_db()
    key, ip = f"magic:{body.email.lower()}", client_ip(request)
    fail_check(db, key, ip)
    user = db.users.find_one({"email": body.email.lower()})
    if not user or not user.get("is_active", True):
        raise HTTPException(status_code=400, detail="Código inválido o expirado")

    saved = user.get("magic_code") or {}
    expires_at = saved.get("expires_at")
    if isinstance(expires_at, datetime) and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if not expires_at or not isinstance(expires_at, datetime) or expires_at < _now():
        raise HTTPException(status_code=400, detail="Código inválido o expirado")
    if int(saved.get("attempts", 0)) >= 5:
        raise HTTPException(status_code=429, detail="Demasiados intentos. Solicita un código nuevo.")
    if not verify_password(body.code, saved.get("code_hash", "")):
        db.users.update_one({"_id": user["_id"]}, {"$inc": {"magic_code.attempts": 1}})
        fail_record(db, key, ip)
        raise HTTPException(status_code=400, detail="Código inválido o expirado")

    # El código por correo NO se salta el 2FA.
    if user.get("totp_enabled"):
        import pyotp

        if not body.totp_code or not user.get("totp_secret") or not pyotp.TOTP(user["totp_secret"]).verify(
            body.totp_code, valid_window=1
        ):
            fail_record(db, key, ip)
            raise HTTPException(status_code=401, detail="Se requiere el código de verificación 2FA")

    db.users.update_one({"_id": user["_id"]}, {"$unset": {"magic_code": ""}})
    fail_clear(db, key)
    user.pop("magic_code", None)
    return _tokens(user)


# ── 2FA (Two-Factor Authentication) ──────────────────────────────────────


class TwoFactorSetupResponse(dict):
    pass


@router.post("/2fa/setup")
async def setup_2fa(user: CurrentUser = Depends(get_current_user)) -> dict:
    """Genera secreto TOTP y retorna URL para QR code."""
    import pyotp

    db = get_db()
    db_user = db.users.find_one({"_id": oid(user.id)})
    email = db_user.get("email", user.username) if db_user else user.username

    secret = pyotp.random_base32()
    # Guardar secreto (no activo hasta verificar)
    db.users.update_one(
        {"_id": oid(user.id)},
        {"$set": {"totp_secret": secret, "totp_enabled": False}},
    )

    # Generar URL para apps autenticadoras
    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(
        name=email,
        issuer_name="Mini GT Collector",
    )

    return {
        "secret": secret,
        "qr_url": provisioning_uri,
        "enabled": False,
    }


@router.post("/2fa/enable")
async def enable_2fa(
    body: dict,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Verifica código TOTP y activa 2FA."""
    import pyotp

    code = body.get("code", "")
    if not code or len(code) != 6:
        raise HTTPException(status_code=400, detail="Código de 6 dígitos requerido")

    db = get_db()
    db_user = db.users.find_one({"_id": oid(user.id)})
    secret = db_user.get("totp_secret") if db_user else None
    if not secret:
        raise HTTPException(status_code=400, detail="Primero ejecuta /2fa/setup")

    totp = pyotp.TOTP(secret)
    if not totp.verify(code, valid_window=1):
        raise HTTPException(status_code=400, detail="Código inválido")

    # Activar 2FA
    db.users.update_one(
        {"_id": oid(user.id)},
        {"$set": {"totp_enabled": True}},
    )
    return {"enabled": True, "message": "2FA activado correctamente"}


@router.post("/2fa/disable")
async def disable_2fa(
    body: dict,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Desactiva 2FA (requiere código TOTP válido)."""
    import pyotp

    code = body.get("code", "")
    if not code or len(code) != 6:
        raise HTTPException(status_code=400, detail="Código de 6 dígitos requerido")

    db = get_db()
    db_user = db.users.find_one({"_id": oid(user.id)})
    secret = db_user.get("totp_secret") if db_user else None
    if not secret or not db_user.get("totp_enabled"):
        raise HTTPException(status_code=400, detail="2FA no está activo")

    totp = pyotp.TOTP(secret)
    if not totp.verify(code, valid_window=1):
        raise HTTPException(status_code=400, detail="Código inválido")

    db.users.update_one(
        {"_id": oid(user.id)},
        {"$set": {"totp_enabled": False}, "$unset": {"totp_secret": ""}},
    )
    return {"enabled": False, "message": "2FA desactivado"}


@router.post("/2fa/verify")
async def verify_2fa(
    body: dict,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Verifica un código TOTP sin cambiar estado (para login)."""
    import pyotp

    code = body.get("code", "")
    if not code or len(code) != 6:
        return {"valid": False}

    db = get_db()
    db_user = db.users.find_one({"_id": oid(user.id)})
    secret = db_user.get("totp_secret") if db_user else None
    if not secret or not db_user.get("totp_enabled"):
        return {"valid": False, "error": "2FA no activo"}

    totp = pyotp.TOTP(secret)
    valid = totp.verify(code, valid_window=1)
    return {"valid": valid}
