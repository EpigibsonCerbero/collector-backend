"""FastAPI application entrypoint — CollectorManager API."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.db import ensure_indexes, get_db, ping
from app.routers import auth, catalog, community, decide, garage, identify, lists, market, missions, social, trading
from app.services.seed import run_seed
from app.services.vision import warmup as warmup_vision


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        ensure_indexes()
        run_seed(get_db())
    except Exception as exc:  # DB down at boot: serve anyway, /ready reports state
        print(f"[startup] MongoDB no disponible: {exc}")
    try:
        # Calienta el modelo de visión para que el 1º identify sea rápido.
        warmup_vision()
    except Exception as exc:
        print(f"[startup] warmup visión falló: {exc}")
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

    api = "/api/v1"
    app.include_router(auth.router, prefix=api)
    app.include_router(auth.me_router, prefix=api)
    app.include_router(catalog.router, prefix=api)
    app.include_router(identify.router, prefix=api)
    app.include_router(decide.router, prefix=api)
    app.include_router(decide.curate_router, prefix=api)
    app.include_router(garage.router, prefix=api)
    app.include_router(lists.router, prefix=api)
    app.include_router(market.values_router, prefix=api)
    app.include_router(market.tx_router, prefix=api)
    app.include_router(market.proposals_router, prefix=api)
    app.include_router(trading.router, prefix=api)
    app.include_router(missions.router, prefix=api)
    app.include_router(community.feed_router, prefix=api)
    app.include_router(community.friends_router, prefix=api)
    app.include_router(social.events_router, prefix=api)
    app.include_router(social.chat_router, prefix=api)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        mongo_ok = ping()
        return {"mongo": mongo_ok, "status": "ready" if mongo_ok else "degraded"}

    return app


app = create_app()
