"""Revisión de piezas creadas por IA (origin == "ai").

  python scripts/review_ai_pieces.py                  # tabla de pendientes
  python scripts/review_ai_pieces.py --demote <id>    # DRY-RUN: no escribe
  python scripts/review_ai_pieces.py --demote <id> --apply   # oculta (hidden: true)
  python scripts/review_ai_pieces.py --reconcile      # DRY-RUN: IA con toy# ya oficial
  python scripts/review_ai_pieces.py --reconcile --apply     # oculta esas piezas de IA

Nunca borra nada: ocultar es reversible (quitar `hidden`). Las consultas del
catálogo y del identify ignoran las piezas con `hidden: true`.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone


def list_pending(db, limit: int = 100) -> list[dict]:
    """Piezas de IA sin verificar y no ocultas, más recientes primero."""
    cur = (
        db.variants.find({"origin": "ai", "verified": False, "hidden": {"$ne": True}})
        .sort("created_at", -1)
        .limit(int(limit))
    )
    rows = []
    for d in cur:
        rows.append(
            {
                "id": str(d["_id"]),
                "name": d.get("variant_name") or d.get("name"),
                "brand": d.get("manufacturer") or d.get("brand"),
                "created_by": d.get("created_by"),
                "created_at": d.get("created_at"),
                "source_confidence": d.get("source_confidence"),
                "confirmed_by": len(d.get("confirmed_by") or []),
            }
        )
    return rows


def demote(db, piece_id: str, apply: bool = False) -> dict:
    """Oculta una pieza de IA. Sin `apply` no escribe. Lanza ValueError si no procede."""
    doc = db.variants.find_one({"_id": piece_id})
    if doc is None:
        raise ValueError(f"No existe la pieza {piece_id}")
    if doc.get("origin") != "ai":
        raise ValueError(f"La pieza {piece_id} no es de origen IA: no se oculta")
    result = {"id": piece_id, "already_hidden": bool(doc.get("hidden")), "applied": False}
    if apply and not result["already_hidden"]:
        db.variants.update_one(
            {"_id": piece_id, "origin": "ai"},
            {"$set": {"hidden": True, "hidden_at": datetime.now(timezone.utc)}},
        )
        result["applied"] = True
    return result


def find_reconcilable(db) -> list[dict]:
    """Piezas de IA visibles cuyo `toy_number` ya tiene una pieza que no es de
    IA (el scrape oficial la trajo después). Esas filas de IA sobran."""
    rows = []
    for d in db.variants.find(
        {"origin": "ai", "hidden": {"$ne": True}, "toy_number": {"$nin": [None, ""]}}
    ).sort("created_at", 1):
        official = db.variants.find_one(
            {"toy_number": d["toy_number"], "origin": {"$ne": "ai"}, "_id": {"$ne": d["_id"]}}, {"_id": 1}
        )
        if official:
            rows.append({"id": str(d["_id"]), "toy_number": d["toy_number"], "official_id": str(official["_id"])})
    return rows


def reconcile(db, apply: bool = False) -> list[dict]:
    """Oculta (`hidden: true`) las piezas de `find_reconcilable`. Sin `apply`
    no escribe. Nunca borra: los garajes siguen funcionando por id."""
    rows = find_reconcilable(db)
    if apply:
        now = datetime.now(timezone.utc)
        for r in rows:
            db.variants.update_one(
                {"_id": r["id"], "origin": "ai"}, {"$set": {"hidden": True, "hidden_at": now}}
            )
    return rows


def _fmt(v) -> str:
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d %H:%M")
    return "" if v is None else str(v)


def _print_table(rows: list[dict]) -> None:
    cols = ["id", "name", "brand", "created_by", "created_at", "source_confidence", "confirmed_by"]
    if not rows:
        print("Sin piezas pendientes.")
        return
    table = [[_fmt(r[c]) for c in cols] for r in rows]
    widths = [max(len(c), *(len(t[i]) for t in table)) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for t in table:
        print("  ".join(x.ljust(w) for x, w in zip(t, widths)))


def main(argv: list[str] | None = None, db=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demote", metavar="ID", help="oculta la pieza de IA (dry-run salvo --apply)")
    ap.add_argument("--reconcile", action="store_true",
                    help="IA con toy# que ya tiene pieza oficial (dry-run salvo --apply)")
    ap.add_argument("--apply", action="store_true", help="escribe de verdad (con --demote o --reconcile)")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args(argv)
    if args.apply and not (args.demote or args.reconcile):
        print("--apply solo se usa con --demote <id> o --reconcile", file=sys.stderr)
        return 2
    if db is None:
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from app.db import get_db  # misma URL que la app: MONGO_URL/.env, sin imprimirla

        db = get_db()
    if args.reconcile:
        print("Recordatorio: toma un respaldo de Mongo ANTES de escribir en producción.")
        rows = reconcile(db, apply=args.apply)
        for r in rows:
            print(f"{r['id']}  toy={r['toy_number']}  oficial={r['official_id']}")
        if not rows:
            print("Nada que reconciliar.")
        elif args.apply:
            print(f"{len(rows)} pieza(s) de IA ocultas (hidden: true). No se borró nada.")
        else:
            print(f"DRY-RUN: ocultaría {len(rows)} pieza(s) de IA. No se escribió nada.")
            print("Para aplicar: añade --apply (respaldo de Mongo primero).")
        return 0
    if not args.demote:
        _print_table(list_pending(db, args.limit))
        return 0
    if args.apply:
        print("Recordatorio: toma un respaldo de Mongo ANTES de escribir en producción.")
    try:
        res = demote(db, args.demote, apply=args.apply)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if res["already_hidden"]:
        print(f"La pieza {args.demote} ya estaba oculta. Nada que hacer.")
    elif res["applied"]:
        print(f"Pieza {args.demote} oculta (hidden: true). No se borró nada.")
    else:
        print(f"DRY-RUN: pondría hidden: true en la pieza {args.demote}. No se escribió nada.")
        print("Para aplicar: añade --apply (respaldo de Mongo primero).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
