"""Límites de abuso.

- `fail_*`: fallos de autenticación persistidos en Mongo (`auth_failures`, con TTL),
  sobreviven a reinicios. Frenan la fuerza bruta de contraseñas y códigos.
- `hit_or_429`: ventana deslizante en memoria por IP (un solo proceso uvicorn)
  para endpoints caros (identify, peritaje, envío de códigos).
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request

AUTH_WINDOW = timedelta(minutes=15)
MAX_FAILS_PER_KEY = 8
MAX_FAILS_PER_IP = 40


def client_ip(request: Request | None) -> str:
    if request is None:
        return "?"
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "?"


def fail_check(db, key: str, ip: str) -> None:
    since = datetime.now(timezone.utc) - AUTH_WINDOW
    if db.auth_failures.count_documents({"key": key, "at": {"$gte": since}}) >= MAX_FAILS_PER_KEY or (
        db.auth_failures.count_documents({"ip": ip, "at": {"$gte": since}}) >= MAX_FAILS_PER_IP
    ):
        raise HTTPException(
            status_code=429,
            detail="Demasiados intentos. Espera unos minutos e inténtalo de nuevo.",
        )


def fail_record(db, key: str, ip: str) -> None:
    db.auth_failures.insert_one({"key": key, "ip": ip, "at": datetime.now(timezone.utc)})


def fail_clear(db, key: str) -> None:
    db.auth_failures.delete_many({"key": key})


_hits: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


def hit_or_429(bucket: str, ip: str, limit: int, per_seconds: int) -> None:
    now = time.monotonic()
    k = f"{bucket}:{ip}"
    with _lock:
        q = _hits[k]
        while q and now - q[0] > per_seconds:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(
                status_code=429,
                detail="Vas muy rápido. Espera un momento e inténtalo de nuevo.",
            )
        q.append(now)
