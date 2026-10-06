"""Bus de eventos de debug — alimenta la consola en vivo de /lab.

Cada etapa del pipeline de identify publica un evento aquí. El endpoint
`GET /lab/stream` (SSE) los empuja al navegador mientras la petición corre,
para que se pueda ver exactamente dónde se va el tiempo.

Los prints a stdout siguen existiendo (van al log de systemd); esto es un
canal paralelo, estructurado y sin bloquear.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any

_LOCK = threading.Lock()
_BUFFER: deque[dict[str, Any]] = deque(maxlen=800)
_SUBS: list[queue.Queue] = []


def publish(stage: str, msg: str, **extra: Any) -> None:
    """Publica un evento de debug. Nunca lanza."""
    evt: dict[str, Any] = {"t": time.time(), "stage": stage, "msg": msg}
    if extra:
        evt["extra"] = extra
    try:
        with _LOCK:
            _BUFFER.append(evt)
            subs = list(_SUBS)
        for q in subs:
            try:
                q.put_nowait(evt)
            except queue.Full:
                pass
    except Exception:  # noqa: BLE001 — el debug jamás debe tumbar el identify
        pass


def subscribe(warm: int = 60) -> queue.Queue:
    """Suscribe una cola y le manda los últimos `warm` eventos."""
    q: queue.Queue = queue.Queue(maxsize=2000)
    with _LOCK:
        _SUBS.append(q)
        recent = list(_BUFFER)[-warm:]
    for evt in recent:
        try:
            q.put_nowait(evt)
        except queue.Full:
            break
    return q


def unsubscribe(q: queue.Queue) -> None:
    with _LOCK:
        try:
            _SUBS.remove(q)
        except ValueError:
            pass


def snapshot(limit: int = 200) -> list[dict[str, Any]]:
    with _LOCK:
        return list(_BUFFER)[-limit:]


def clear() -> None:
    with _LOCK:
        _BUFFER.clear()
