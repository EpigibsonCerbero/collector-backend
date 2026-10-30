"""Passwords (argon2id), JWT access/refresh tokens, auth dependencies."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings

_hasher = PasswordHasher(time_cost=2, memory_cost=64 * 1024, parallelism=2)
_bearer = HTTPBearer(auto_error=False)


def hash_password(plain: str) -> str:
    return _hasher.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _hasher.verify(hashed, plain)
    except VerifyMismatchError:
        return False
    except Exception:
        return False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_token(sub: str, token_type: str, ttl: timedelta, **extra: object) -> str:
    settings = get_settings()
    payload = {
        "sub": sub,
        "type": token_type,
        "iat": int(_now().timestamp()),
        "exp": int((_now() + ttl).timestamp()),
        **extra,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_alg)


def create_access_token(sub: str, username: str, tv: int = 0) -> str:
    return create_token(
        sub,
        "access",
        timedelta(minutes=get_settings().access_ttl_min),
        username=username,
        tv=tv,
    )


def create_refresh_token(sub: str, username: str, tv: int = 0) -> str:
    # `tv` (token_version): al cambiar/restablecer la contraseña sube y todos
    # los refresh anteriores dejan de valer (cierra sesiones robadas).
    return create_token(
        sub,
        "refresh",
        timedelta(days=get_settings().refresh_ttl_days),
        username=username,
        tv=tv,
    )


def decode_token(token: str, expected_type: str) -> dict:
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_alg])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expirado")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido")
    if payload.get("type") != expected_type:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Tipo de token inválido")
    return payload


class CurrentUser(dict):
    """Authenticated principal: {"id": ..., "username": ...}."""

    @property
    def id(self) -> str:
        return self["id"]

    @property
    def username(self) -> str:
        return self["username"]


def require_admin(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> "CurrentUser":
    """Solo ids listados en ADMIN_USER_IDS (editar el catálogo global, etc.)."""
    user = get_current_user(creds)
    admins = {x.strip() for x in get_settings().admin_user_ids.split(",") if x.strip()}
    if user.id not in admins:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Solo administradores")
    return user


def get_current_user(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> CurrentUser:
    if creds is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No autenticado")
    payload = decode_token(creds.credentials, expected_type="access")
    return CurrentUser(id=payload["sub"], username=payload.get("username", ""))
