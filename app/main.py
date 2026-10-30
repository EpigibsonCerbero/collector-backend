"""FastAPI application entrypoint — CollectorManager API."""
from __future__ import annotations

import os

import asyncio
import ipaddress
import json
import queue
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.config import get_settings
from app.db import ensure_indexes, get_db, ping
from app.routers import ai, alerts, auth, catalog, challenges, community, decide, garage, identify, leaderboard, lists, market, missions, notifications, shipping, social, stats, subscription, trading, uploads
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


_LAB_STREAMS = {"n": 0}


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

    # /lab* expone el debug de TODOS los identify (texto leído de fotos de
    # usuarios). Con LAB_PASSWORD definido exige usuario/contraseña (HTTP Basic:
    # el navegador lo pide una vez y lo reenvía al stream y al proxy).
    @app.middleware("http")
    async def _lab_guard(request, call_next):
        if request.url.path.startswith("/lab") and os.environ.get("LAB_PASSWORD"):
            import base64 as _b64
            import secrets as _secrets

            ok = False
            auth_h = request.headers.get("authorization", "")
            if auth_h.lower().startswith("basic "):
                try:
                    user, _, pwd = _b64.b64decode(auth_h[6:]).decode().partition(":")
                    ok = _secrets.compare_digest(pwd, os.environ["LAB_PASSWORD"]) and _secrets.compare_digest(
                        user, os.environ.get("LAB_USER", "lab")
                    )
                except Exception:
                    ok = False
            if not ok:
                return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Identify Lab"'})
        return await call_next(request)

    api = "/api/v1"
    app.include_router(auth.router, prefix=api)
    app.include_router(auth.me_router, prefix=api)
    app.include_router(catalog.router, prefix=api)
    app.include_router(identify.router, prefix=api)
    app.include_router(ai.router, prefix=api)
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
    app.include_router(alerts.router, prefix=api)
    app.include_router(challenges.router, prefix=api)
    app.include_router(stats.router, prefix=api)
    app.include_router(leaderboard.router, prefix=api)
    app.include_router(shipping.router, prefix=api)
    app.include_router(uploads.router, prefix=api)
    app.include_router(notifications.router, prefix=api)
    app.include_router(subscription.router, prefix=api)
    app.include_router(subscription.webhook_router, prefix=api)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        mongo_ok = ping()
        return {"mongo": mongo_ok, "status": "ready" if mongo_ok else "degraded"}

    # ── Identify Lab: banco de pruebas de visión (HTML estático) ──────────
    # Sirve la página en el MISMO origen que la API → sin CORS, y prueba el
    # pipeline real de producción. Ver backend/static/identify-lab.html.
    _LAB_HTML = Path(__file__).resolve().parents[1] / "static" / "identify-lab.html"

    @app.get("/lab", include_in_schema=False)
    def identify_lab() -> FileResponse:
        return FileResponse(_LAB_HTML, media_type="text/html; charset=utf-8")

    # Fotos de ejemplo de "cómo sacar la foto" (how-to del identify).
    _LAB_TIPS = Path(__file__).resolve().parents[1] / "static" / "tips"
    if _LAB_TIPS.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/lab/tips", StaticFiles(directory=str(_LAB_TIPS)), name="lab-tips")

    @app.get("/lab/img", include_in_schema=False)
    async def lab_img(url: str) -> Response:
        """Proxy de imágenes del catálogo para la captura (evita taint/CORS).

        Solo HTTPS hacia hosts públicos y content-type de imagen — no es un
        proxy abierto. Máx 8 MB.
        """
        def _check_public(u: str) -> None:
            parsed = urlparse(u)
            if parsed.scheme != "https" or not parsed.hostname:
                raise HTTPException(status_code=400, detail="Solo se admite https:// con host")
            try:
                infos = socket.getaddrinfo(parsed.hostname, 443, proto=socket.IPPROTO_TCP)
            except socket.gaierror as exc:
                raise HTTPException(status_code=400, detail=f"Host no resoluble: {exc}") from exc
            for info in infos:
                ip = ipaddress.ip_address(info[4][0])
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                    raise HTTPException(status_code=400, detail="Host no público")

        # Redirecciones a mano, revalidando CADA salto: con follow_redirects=True
        # una URL pública que redirigiera a 169.254.169.254 (metadatos de la
        # nube) se saltaba la validación (SSRF). 2026-10-06: hay escáneres
        # externos probando justo eso contra la API.
        _check_public(url)
        async with httpx.AsyncClient(follow_redirects=False, timeout=15.0) as client:
            try:
                r = await client.get(url)
                for _ in range(3):
                    if not r.is_redirect:
                        break
                    url = str(r.next_request.url) if r.next_request else ""
                    _check_public(url)
                    r = await client.get(url)
                if r.is_redirect:
                    raise HTTPException(status_code=502, detail="Demasiadas redirecciones")
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=f"No se pudo descargar: {exc}") from exc
        ctype = r.headers.get("content-type", "")
        if not ctype.startswith("image/"):
            raise HTTPException(status_code=415, detail=f"No es una imagen: {ctype}")
        if len(r.content) > 8 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Imagen demasiado grande")
        return Response(
            content=r.content,
            media_type=ctype,
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/lab/stream", include_in_schema=False)
    async def lab_stream() -> StreamingResponse:
        """SSE con el debug en vivo del pipeline de identify.

        El navegador abre EventSource a esta URL ANTES de subir la foto y ve
        cada etapa (barcode → embed → OCR → match → texto → decisión) llegar
        en tiempo real.
        """
        from app.services import debug_bus

        # Cada conexión es un bucle infinito: sin tope, abrir muchas tumbaba el servidor.
        if _LAB_STREAMS["n"] >= 5:
            raise HTTPException(status_code=429, detail="Demasiadas consolas abiertas")

        async def gen():
            # Alta y baja DENTRO del generador: si el cliente se va antes de
            # empezar, no queda ni suscripción ni contador colgados.
            _LAB_STREAMS["n"] += 1
            q = debug_bus.subscribe(warm=80)
            try:
                while True:
                    try:
                        evt = q.get_nowait()
                        yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
                    except queue.Empty:
                        # keep-alive: mantiene viva la conexión NAT/proxy
                        yield ": ping\n\n"
                        await asyncio.sleep(0.35)
            finally:
                debug_bus.unsubscribe(q)
                _LAB_STREAMS["n"] -= 1

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/lab/debug", include_in_schema=False)
    def lab_debug(limit: int = 200) -> list:
        """Snapshot del buffer de debug (para copiar/diagnóstico sin SSE)."""
        from app.services import debug_bus

        return debug_bus.snapshot(limit=max(1, min(limit, 800)))

    return app


app = create_app()
