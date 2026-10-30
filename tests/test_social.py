"""Chat/amigos/feed/trading con Mongo simulado (mongomock): no toca producción.

Requiere: pip install mongomock pytest httpx
"""
import sys
import mongomock
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import app.db as dbmod

_client = mongomock.MongoClient()
_db = _client["t"]
dbmod.get_db = lambda: _db

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from app.routers import alerts, auth, community, market, missions, social, trading

api = FastAPI()
for r in (auth.router, auth.me_router, community.feed_router, community.friends_router,
          social.chat_router, trading.router, alerts.router, missions.router, market.tx_router):
    api.include_router(r, prefix="/api/v1")
c = TestClient(api)
P = "/api/v1"


def reg(name):
    r = c.post(f"{P}/auth/register", json={"email": f"{name}@x.com", "username": name, "password": "Secreta123!"})
    assert r.status_code == 201, r.text
    tok = r.json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}
    me = c.get(f"{P}/me", headers=h).json()
    return {"h": h, "id": me["id"], "tok": tok, "name": name}


@pytest.fixture(scope="module")
def users():
    return reg("Kenji_JP"), reg("papu"), reg("intruso")


def recv_msg(ws):
    """Siguiente evento que no sea de presencia."""
    while True:
        ev = ws.receive_json()
        if ev.get("type") != "presence":
            return ev


def dm(a, b):
    x, y = sorted([a["id"], b["id"]])
    return f"dm-{x}-{y}"


def test_login_regex_injection(users):
    r = c.post(f"{P}/auth/login", json={"identifier": ".*", "password": "Secreta123!"})
    assert r.status_code == 401


def test_friends_real_users(users):
    a, b, x = users
    assert c.post(f"{P}/friends", json={"username": "noexiste"}, headers=b["h"]).status_code == 404
    assert c.post(f"{P}/friends", json={"username": "papu"}, headers=b["h"]).status_code == 400
    r = c.post(f"{P}/friends", json={"username": "@kenji_jp"}, headers=b["h"])
    assert r.status_code == 201, r.text
    assert r.json()["friend_id"] == a["id"] and r.json()["username"] == "Kenji_JP"
    # mutua
    _db.friends.insert_one({"user_id": a["id"], "friend_id": "texto_libre", "username": "texto_libre"})
    fa = c.get(f"{P}/friends", headers=a["h"]).json()
    assert [f["friend_id"] for f in fa] == [b["id"]]
    assert c.post(f"{P}/friends", json={"username": "kenji_jp"}, headers=b["h"]).status_code == 409


def test_showcase_reads_real_garage(users):
    a, b, x = users
    _db.variants.insert_one({"_id": "v1", "variant_name": "GT-R R34", "toy_number": "MGT00001", "main_image_url": "http://img"})
    pub = _db.collections.insert_one({"user_id": a["id"], "is_private": False}).inserted_id
    priv = _db.collections.insert_one({"user_id": a["id"], "is_private": True}).inserted_id
    _db.user_items.insert_one({"user_id": a["id"], "collection_id": str(pub), "item_variant_id": "v1", "is_tradeable": True})
    _db.user_items.insert_one({"user_id": a["id"], "collection_id": str(priv), "item_variant_id": "v1"})
    rel = c.get(f"{P}/friends", headers=b["h"]).json()[0]
    assert rel["pieces_count"] == 1
    items = c.get(f"{P}/friends/{rel['id']}/showcase", headers=b["h"]).json()
    assert len(items) == 1 and items[0]["variant_id"] == "v1" and items[0]["box_number"] == "MGT00001"
    # por id de usuario (ranking) también
    assert len(c.get(f"{P}/friends/{a['id']}/showcase", headers=x["h"]).json()) == 1


def test_chat_membership(users):
    a, b, x = users
    t = dm(a, b)
    r = c.post(f"{P}/chat/{t}", json={"text": "hola", "sender": "partner", "author_name": "falso"}, headers=a["h"])
    assert r.status_code == 201, r.text
    m = r.json()
    assert m["author_id"] == a["id"] and m["author_name"] == "Kenji_JP" and m["sender"] == "user"
    msgs = c.get(f"{P}/chat/{t}", headers=b["h"]).json()
    assert len(msgs) == 1 and msgs[0]["author_id"] == a["id"]
    assert c.get(f"{P}/chat/{t}", headers=x["h"]).status_code == 403
    assert c.post(f"{P}/chat/{t}", json={"text": "x"}, headers=x["h"]).status_code == 403
    assert c.get(f"{P}/chat/chat-{a['id']}", headers=a["h"]).status_code == 403
    assert c.get(f"{P}/chat/dm-{a['id']}-zzz", headers=a["h"]).status_code == 403
    # id desordenado → 403 (un solo hilo por pareja)
    y, z = sorted([a["id"], b["id"]])
    assert c.get(f"{P}/chat/dm-{z}-{y}", headers=a["h"]).status_code == 403
    # B no puede borrar lo de A
    assert c.delete(f"{P}/chat/{t}", headers=b["h"]).status_code == 204
    assert len(c.get(f"{P}/chat/{t}", headers=a["h"]).json()) == 1
    assert c.delete(f"{P}/chat/{t}", headers=x["h"]).status_code == 403


def test_ws_auth_and_broadcast(users):
    a, b, x = users
    t = dm(a, b)
    with pytest.raises(WebSocketDisconnect) as e:
        with c.websocket_connect(f"{P}/chat/ws/{t}") as ws:
            ws.receive_text()
    assert e.value.code == 4401
    with pytest.raises(WebSocketDisconnect) as e:
        with c.websocket_connect(f"{P}/chat/ws/{t}?token={x['tok']}") as ws:
            ws.receive_text()
    assert e.value.code == 4403
    with c.websocket_connect(f"{P}/chat/ws/{t}", headers={"Authorization": f"Bearer {a['tok']}"}) as wa, \
         c.websocket_connect(f"{P}/chat/ws/{t}?token={b['tok']}") as wb:
        wa.send_json({"text": "por ws", "sender": "partner"})
        got = recv_msg(wb)
        assert got["type"] == "message" and got["data"]["author_id"] == a["id"]
        assert recv_msg(wa)["data"]["text"] == "por ws"


def test_feed_reaction_per_user(users):
    a, b, x = users
    ev = c.post(f"{P}/community/feed", json={"type": "added_item", "title": "t", "description": "d"}, headers=a["h"]).json()
    r = c.post(f"{P}/community/feed/{ev['id']}/react", json={"reaction": "🔥"}, headers=a["h"]).json()
    assert r["user_reaction"] == "🔥" and r["reactions"]["🔥"] == 1
    fb = [e for e in c.get(f"{P}/community/feed", headers=b["h"]).json() if e["id"] == ev["id"]][0]
    assert fb["user_reaction"] is None and fb["reactions"]["🔥"] == 1
    r = c.post(f"{P}/community/feed/{ev['id']}/react", json={"reaction": "🔥"}, headers=b["h"]).json()
    assert r["reactions"]["🔥"] == 2
    r = c.post(f"{P}/community/feed/{ev['id']}/react", json={"reaction": "🔥"}, headers=a["h"]).json()
    assert r["reactions"]["🔥"] == 1 and r["user_reaction"] is None


def test_trade_offer_escrow_thread(users):
    a, b, x = users
    item = {"variant_id": "v1", "variant_name": "GT-R", "image_url": "", "condition": "Mint"}
    r = c.post(f"{P}/trading/offers", json={"to_user_id": b["id"], "to_username": "papu",
               "offered_items": [item], "requested_items": [item]}, headers=a["h"])
    assert r.status_code == 201, r.text
    inc = c.get(f"{P}/trading/offers", headers=b["h"]).json()
    assert inc[0]["type"] == "incoming"
    assert c.post(f"{P}/trading/offers/{inc[0]['id']}/accept", headers=b["h"]).status_code == 200
    esc = c.get(f"{P}/trading/escrows", headers=a["h"]).json()[0]
    t = f"trade-{esc['id']}"
    assert c.post(f"{P}/chat/{t}", json={"text": "envío mañana"}, headers=b["h"]).status_code == 201
    assert c.get(f"{P}/chat/{t}", headers=a["h"]).status_code == 200
    assert c.get(f"{P}/chat/{t}", headers=x["h"]).status_code == 403


def test_trade_rules_and_alerts(users):
    a, b, x = users
    item = {"variant_id": "v1", "variant_name": "GT-R", "image_url": "", "condition": "Mint"}
    body = {"to_user_id": "000000000000000000000000", "offered_items": [item], "requested_items": [item]}
    assert c.post(f"{P}/trading/offers", json=body, headers=a["h"]).status_code == 404
    body["to_user_id"] = b["id"]
    body["to_username"] = "inventado"
    o = c.post(f"{P}/trading/offers", json=body, headers=a["h"]).json()
    assert o["to_username"] == "papu"
    assert any(al["title"] == "Nueva oferta de canje" for al in c.get(f"{P}/alerts", headers=b["h"]).json())
    # solo quien envía la retira; luego ya no se puede rechazar
    assert c.post(f"{P}/trading/offers/{o['id']}/cancel", headers=b["h"]).status_code == 404
    assert c.post(f"{P}/trading/offers/{o['id']}/cancel", headers=a["h"]).json()["status"] == "cancelled"
    assert c.post(f"{P}/trading/offers/{o['id']}/reject", headers=b["h"]).status_code == 409
    # contraparte correcta para cada lado
    esc_a = c.get(f"{P}/trading/escrows", headers=a["h"]).json()[0]
    esc_b = c.get(f"{P}/trading/escrows", headers=b["h"]).json()[0]
    assert esc_a["partner_name"] == "papu" and esc_b["partner_name"] == "Kenji_JP"
    r = c.patch(f"{P}/trading/escrows/{esc_a['id']}", json={"tracking_number": "DHL123", "status": "shipped"}, headers=a["h"])
    assert r.status_code == 200 and r.json()["partner_name"] == "papu"
    assert any(al["title"] == "Pieza enviada" for al in c.get(f"{P}/alerts", headers=b["h"]).json())
    assert c.patch(f"{P}/trading/escrows/{esc_a['id']}", json={"status": "completed"}, headers=x["h"]).status_code == 403


def test_friend_alert(users):
    a, b, x = users
    assert any(al["type"] == "friend" for al in c.get(f"{P}/alerts", headers=a["h"]).json())


def test_missions_streak_progress_claim(users):
    a, b, x = users
    st = c.get(f"{P}/missions/state", headers=a["h"]).json()
    assert st["streak"] == 1 and len(st["missions"]) == 4
    assert c.get(f"{P}/missions/state", headers=a["h"]).json()["streak"] == 1  # mismo día
    scan = [m for m in st["missions"] if m["type"] == "scan"][0]
    assert c.post(f"{P}/missions/{scan['id']}/claim", headers=a["h"]).status_code == 409
    r = c.post(f"{P}/missions/{scan['id']}/progress", json={"increment": 1}, headers=a["h"]).json()
    assert r["current"] == 1
    assert c.post(f"{P}/missions/{scan['id']}/claim", headers=a["h"]).status_code == 200
    assert c.get(f"{P}/missions/state", headers=a["h"]).json()["total_xp"] == scan["xp_reward"]
    # misiones ajenas
    assert c.post(f"{P}/missions/{scan['id']}/progress", json={"increment": 1}, headers=b["h"]).status_code == 404


def test_transactions_item_name(users):
    a, b, x = users
    r = c.post(f"{P}/transactions", json={"type": "sale", "amount": 25, "item_name": "Skyline R34"}, headers=a["h"])
    assert r.status_code == 201, r.text
    lst = c.get(f"{P}/transactions", headers=a["h"]).json()
    assert lst[0]["item_name"] == "Skyline R34"
    assert c.get(f"{P}/transactions", headers=b["h"]).json() == []
    assert c.delete(f"{P}/transactions/{lst[0]['id']}", headers=b["h"]).status_code == 404


def test_all_missions_bonus_once(users):
    a, b, x = users
    st = c.get(f"{P}/missions/state", headers=b["h"]).json()
    total = 0
    for m in st["missions"]:
        c.post(f"{P}/missions/{m['id']}/progress", json={"increment": m["target"]}, headers=b["h"])
        assert c.post(f"{P}/missions/{m['id']}/claim", headers=b["h"]).status_code == 200
        total += m["xp_reward"]
    assert c.get(f"{P}/missions/state", headers=b["h"]).json()["total_xp"] == total + 50
    # reclamar otra vez no vuelve a dar el bono
    assert c.post(f"{P}/missions/{st['missions'][0]['id']}/claim", headers=b["h"]).status_code == 409
    assert c.get(f"{P}/missions/state", headers=b["h"]).json()["total_xp"] == total + 50


def test_rest_message_reaches_open_websocket(users):
    """Un mensaje enviado por REST llega en vivo a quien tiene el chat abierto."""
    a, b, x = users
    t = dm(a, b)
    with c.websocket_connect(f"{P}/chat/ws/{t}", headers={"Authorization": f"Bearer {b['tok']}"}) as wb:
        assert c.post(f"{P}/chat/{t}", json={"text": "por REST"}, headers=a["h"]).status_code == 201
        got = recv_msg(wb)
        assert got["type"] == "message" and got["data"]["text"] == "por REST"


def test_chat_reactions_are_per_user(users):
    a, b, x = users
    t = dm(a, b)
    m = c.post(f"{P}/chat/{t}", json={"text": "reacciona"}, headers=a["h"]).json()
    r = c.post(f"{P}/chat/{t}/{m['id']}/reactions", json={"emoji": "❤️"}, headers=a["h"]).json()
    r = c.post(f"{P}/chat/{t}/{m['id']}/reactions", json={"emoji": "❤️"}, headers=b["h"]).json()
    assert sorted(r["reactions"]) == ["❤️", "❤️"]
    # B quita la suya: la de A sigue (antes se borraba la del otro)
    r = c.post(f"{P}/chat/{t}/{m['id']}/reactions", json={"emoji": "❤️"}, headers=b["h"]).json()
    assert r["reactions"] == ["❤️"] and r["reactions_by"] == {a["id"]: "❤️"}
    # cambiar de emoji reemplaza, no suma
    r = c.post(f"{P}/chat/{t}/{m['id']}/reactions", json={"emoji": "😂"}, headers=a["h"]).json()
    assert r["reactions"] == ["😂"]
    # reacciones fabricadas al crear: ignoradas; ajenos: 403
    m2 = c.post(f"{P}/chat/{t}", json={"text": "x", "reactions": ["🔥"] * 50}, headers=a["h"]).json()
    assert m2["reactions"] == []
    assert c.post(f"{P}/chat/{t}/{m['id']}/reactions", json={"emoji": "❤️"}, headers=x["h"]).status_code == 403


def test_chat_client_id_echo_and_ws_validation(users):
    a, b, x = users
    t = dm(a, b)
    with c.websocket_connect(f"{P}/chat/ws/{t}", headers={"Authorization": f"Bearer {a['tok']}"}) as wa:
        assert wa.receive_json()["type"] == "presence"
        wa.send_json({"text": "hola", "client_id": "loc-123"})
        got = wa.receive_json()
        assert got["type"] == "message" and got["data"]["client_id"] == "loc-123"
        wa.send_json({"text": "x" * 5000})
        assert wa.receive_json()["type"] == "error"
        wa.send_json({"type": "piece", "piece_name": "y" * 500})
        assert wa.receive_json()["type"] == "error"


def test_chat_presence_and_read_receipts(users):
    a, b, x = users
    t = dm(a, b)
    with c.websocket_connect(f"{P}/chat/ws/{t}", headers={"Authorization": f"Bearer {a['tok']}"}) as wa:
        assert wa.receive_json() == {"type": "presence", "user_ids": [a["id"]]}
        with c.websocket_connect(f"{P}/chat/ws/{t}", headers={"Authorization": f"Bearer {b['tok']}"}) as wb:
            assert sorted(wa.receive_json()["user_ids"]) == sorted([a["id"], b["id"]])
            wb.receive_json()  # su propia presencia
            wb.send_json({"action": "read", "at": "2099-01-01T00:00:00"})
            ev = wa.receive_json()
            assert ev == {"type": "read", "user_id": b["id"], "at": "2099-01-01T00:00:00"}
        assert wa.receive_json() == {"type": "presence", "user_ids": [a["id"]]}  # B salió
    reads = c.get(f"{P}/chat/{t}/reads", headers=a["h"]).json()
    assert reads[b["id"]] == "2099-01-01T00:00:00"
    # la marca solo avanza
    c.post(f"{P}/chat/{t}/read", json={"at": "2000-01-01T00:00:00"}, headers=b["h"])
    assert c.get(f"{P}/chat/{t}/reads", headers=a["h"]).json()[b["id"]] == "2099-01-01T00:00:00"
    assert c.get(f"{P}/chat/{t}/reads", headers=x["h"]).status_code == 403


def test_chat_send_is_idempotent_by_client_id(users):
    """Reintento REST tras un envío WS que sí llegó: un solo mensaje."""
    a, b, x = users
    t = dm(a, b)
    with c.websocket_connect(f"{P}/chat/ws/{t}", headers={"Authorization": f"Bearer {a['tok']}"}) as wa:
        wa.send_json({"text": "una vez", "client_id": "dup-1"})
        first = recv_msg(wa)["data"]
    again = c.post(f"{P}/chat/{t}", json={"text": "una vez", "client_id": "dup-1"}, headers=a["h"]).json()
    assert again["id"] == first["id"]
    msgs = c.get(f"{P}/chat/{t}", headers=a["h"]).json()
    assert sum(1 for m in msgs if m.get("client_id") == "dup-1") == 1
    # otro autor con el mismo client_id NO colisiona
    other = c.post(f"{P}/chat/{t}", json={"text": "mío", "client_id": "dup-1"}, headers=b["h"]).json()
    assert other["id"] != first["id"]
