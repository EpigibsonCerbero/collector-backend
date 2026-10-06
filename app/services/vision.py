"""
Motor de visión on-server para identificar piezas diecast.

Cascada (rápido → lento), todo local y sin costo:
  1. Barcode/UPC  (OpenCV)          — exacto, ~50ms
  2. Embeddings   (MobileNetV3)     — visual, ~300ms
  3. OCR toy#/UPC (Tesseract)       — texto del empaque/base, ~400ms

El preprocess del embedding DEBE coincidir con el precompute
(`scripts/gen_dnn_embeddings.py`):
  · center-crop cuadrado a resolución completa
  · resize 224×224
  · píxeles RAW [0,255]  (MobileNetV3 de Keras escala internamente)

Meter [-1,1] satura la red y todos los embeddings salen idénticos (cos ~0.999)
— ya validado. Self-test con la foto oficial del catálogo da 1.0000.
"""
from __future__ import annotations

import io
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.services.debug_bus import publish as _dbg


# ── Rutas ────────────────────────────────────────────────────────────────────
# backend/app/services/vision.py → parents[2] = backend/ (local) o /app/ (Docker)
ROOT = Path(__file__).resolve().parents[2]
EMB_PATH = Path(
    os.environ.get(
        "EMB_INDEX_PATH", str(ROOT / "data" / "minigt-embeddings-full.json")
    )
)
# Índices adicionales de marcas (Hot Wheels / Mattel / diecast extra). Se mergean en get_index().
EMB_EXTRA_PATHS = [
    Path(os.environ.get("EMB_HW_PATH", str(ROOT / "data" / "hotwheels-embeddings-full.json"))),
    Path(os.environ.get("EMB_EXTRA_PATH", str(ROOT / "data" / "diecast-extra-embeddings-full.json"))),
]
# DINOv2 768-d — si existe, es el índice primario (fine-grained).
# ATENCIÓN: mixturar 576-d y 768-d rompe el producto matricial — solo se
# cargan índices de la MISMA dimensión.
DINOV2_INDEX = Path(os.environ.get("EMB_DINOV2_PATH", str(ROOT / "data" / "dinov2-embeddings-full.json")))

IMG_SIZE = 224
EMB_DIM = 576

_lock = threading.Lock()
_model = None
_index: list[dict] | None = None
_index_sig: tuple | None = None  # firma (path, mtime, size) para invalidar el caché
_easyocr_reader = None


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        with _lock:
            if _easyocr_reader is None:
                import easyocr

                print("[vision] cargando EasyOCR...", flush=True)
                _easyocr_reader = easyocr.Reader(["en", "es"], gpu=False, verbose=False)
                print("[vision] EasyOCR cargado", flush=True)
    return _easyocr_reader
_matrix: np.ndarray | None = None  # (N, 576) L2-normalized


# ── Modelos / índice (lazy) ─────────────────────────────────────────────────

def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import tensorflow as tf

                base = tf.keras.applications.MobileNetV3Small(
                    include_top=False,
                    pooling="avg",
                    weights="imagenet",
                    input_shape=(IMG_SIZE, IMG_SIZE, 3),
                )
                base.trainable = False
                inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name="input")
                out = base(inp)
                _model = tf.keras.Model(inp, out, name="minigt_embedder")
    return _model


def _index_paths() -> list[Path]:
    # DINOv2 768-d manda si existe (fine-grained). NO mezclar con los 576-d:
    # la dimensión debe ser homogénea para matrix @ query.
    if DINOV2_INDEX.exists():
        return [DINOV2_INDEX]
    return [EMB_PATH, *EMB_EXTRA_PATHS]


def _index_signature(paths: list[Path]) -> tuple:
    """Firma de los archivos de índice: si cambia, el índice se recarga.

    El job de sync de catálogo (`scripts/sync_catalog.py`) escribe nuevos
    vectores a estos JSON; sin esto, las piezas nuevas no se identificarían
    hasta reiniciar uvicorn.
    """
    sig = []
    for p in paths:
        try:
            st = p.stat()
            sig.append((str(p), st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append((str(p), None, None))
    return tuple(sig)


def get_index() -> tuple[list[dict], np.ndarray]:
    global _index, _matrix, _index_sig
    paths = _index_paths()
    sig = _index_signature(paths)
    if _index is not None and sig == _index_sig:
        return _index, _matrix
    with _lock:
        sig = _index_signature(paths)
        if _index is not None and sig == _index_sig:
            return _index, _matrix
        entries: list[dict] = []
        t0 = time.time()
        for path in paths:
            if not path.exists():
                print(f"[vision] índice no encontrado: {path}", flush=True)
                continue
            # Cache .npy de la matriz: el JSON de 200MB tarda 3-8s en parsear
            # en cada reload. El .npy carga en ~0.5s.
            npy_path = path.with_suffix(".npy")
            if path == DINOV2_INDEX or path.suffix == ".json":
                try:
                    if npy_path.exists() and npy_path.stat().st_mtime >= path.stat().st_mtime:
                        arr = np.load(npy_path)
                        meta_path = path.with_suffix(".meta.json")
                        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []
                        print(
                            f"[vision] índice .npy {npy_path.name}: {arr.shape} en "
                            f"{int((time.time()-t0)*1000)}ms",
                            flush=True,
                        )
                        entries.extend(meta if meta else [{"id": str(i)} for i in range(len(arr))])
                        vecs = arr
                        break
                except Exception as exc:  # noqa: BLE001
                    print(f"[vision] .npy fallback: {exc}", flush=True)
            print(f"[vision] cargando índice desde {path}", flush=True)
            entries.extend(json.loads(path.read_text(encoding="utf-8")))
        else:
            vecs = np.array([e["vec"] for e in entries], dtype=np.float32) if entries else None
        if entries and vecs is None:
            vecs = np.array([e["vec"] for e in entries], dtype=np.float32)
        if not entries:
            raise FileNotFoundError(f"ningún índice de embeddings en {paths}")
        # L2-normalize por si el precompute no lo hizo
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _matrix = vecs / norms
        _index = entries
        _index_sig = sig
        print(
            f"[vision] índice cargado: {len(entries)} vectores dim={_matrix.shape[1]} "
            f"en {int((time.time()-t0)*1000)}ms",
            flush=True,
        )
        # Guardar .npy para el próximo reload
        try:
            if paths and paths[0].suffix == ".json":
                npy_path = paths[0].with_suffix(".npy")
                if not npy_path.exists() or npy_path.stat().st_mtime < paths[0].stat().st_mtime:
                    np.save(npy_path, vecs.astype(np.float32))
                    meta_path = paths[0].with_suffix(".meta.json")
                    slim = [{k: e.get(k) for k in ("id", "toy", "name", "view")} for e in entries]
                    meta_path.write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")
                    print(f"[vision] cache .npy escrito {npy_path.name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[vision] no se pudo cachear .npy: {exc}", flush=True)
    return _index, _matrix


# ── Preprocess (idéntico a scripts/gen_dnn_embeddings.py) ───────────────────

def preprocess(im) -> np.ndarray:
    """PIL Image → (1,224,224,3) float32 en [0,255]."""
    from PIL import Image

    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    im = im.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.float32)
    return arr[None, ...]


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# ── Resultados ──────────────────────────────────────────────────────────────

@dataclass
class VisionMatch:
    id: str
    toy: str | None
    name: str
    score: float
    source: str  # 'barcode' | 'embed' | 'ocr'


@dataclass
class VisionResult:
    matches: list[VisionMatch] = field(default_factory=list)
    upc: str | None = None
    toy_number: str | None = None
    ocr_text: str = ""
    method: str = "manual"
    confidence: float = 0.0
    needs_manual: bool = True
    timings_ms: dict[str, int] = field(default_factory=dict)
    # Buckets de color detectados en la FOTO subida (no en el catálogo).
    # Vacío = señal ambigua (fondo/packshot dominan). Lo usa el router para
    # no auto-confirmar una variante de otro color (rosa vs verde).
    image_colors: list[str] = field(default_factory=list)


# ── Desambiguación de liveries (bug AlphaTauri #21 vs #22) ─────────────────

def _tokens(text: str) -> set[str]:
    return {t for t in re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split() if len(t) >= 3}


def _token_weight(tok: str) -> float:
    """Los tokens cortos/números ('22', 'gp', 'f1') son ambiguos entre coches
    del mismo piloto; los largos ('alphatauri', 'tsunoda', 'australian') son
    los que identifican la pieza."""
    return 1.0 if len(tok) >= 5 else 0.55


_COLOR_WORDS = (
    "white", "blanco", "black", "negro", "red", "rojo", "blue", "azul",
    "silver", "plata", "grey", "gray", "gris", "yellow", "amarillo",
    "green", "verde", "orange", "naranja", "purple", "morado", "pink",
    "rosa", "gold", "dorado", "pearl",
)
# Caché de derivados del OCR (una clave: el texto del request en curso) y de
# ratios difflib — ver docstring de score_name_against_ocr_text.
_TXT_KEY: str | None = None
_TXT_TOKS: set[str] = set()
_TXT_COLORS: set[str] = set()
_RATIO_CACHE: dict[tuple[str, str], float] = {}
# Tokens de nombres del catálogo, cacheados: se consultan 27k veces por request
# en el prefilter de search_catalog_by_text.
_NAME_TOK_CACHE: dict[str, set[str]] = {}


def _cached_name_tokens(name: str) -> set[str]:
    toks = _NAME_TOK_CACHE.get(name)
    if toks is None:
        toks = _tokens(name)
        if len(_NAME_TOK_CACHE) < 80000:
            _NAME_TOK_CACHE[name] = toks
    return toks


def visual_cluster_is_ambiguous(matches: list[VisionMatch], window: float = 0.02) -> bool:
    """¿top1/top2 casi empatados (liveries gemelos)? El embedding no decide."""
    if len(matches) < 2:
        return False
    s0 = matches[0].score
    return any(s0 - m.score < window for m in matches[1:])


# ── Color REAL de la foto subida ─────────────────────────────────────────────
# El penalizador de color anterior solo miraba el TEXTO del OCR ("White Pearl").
# Si la foto no trae palabras de color, un livery gemelo de otro color ganaba
# (medido en prod 2026-09-29: foto ROSA #80 → devolvió el VERDE #77 al 94.8 %).
# Aquí se mide el color de la propia imagen.

# RGB de referencia (misma tabla que scripts/enrich_hotwheels_filter_fields.py)
_BODY_RGB = {
    "Red": (200, 40, 40),
    "Blue": (40, 70, 180),
    "Green": (40, 140, 60),
    "Yellow": (230, 200, 40),
    "Orange": (230, 130, 30),
    "Purple": (130, 50, 160),
    "Pink": (230, 120, 170),
    "Brown": (120, 80, 40),
    "Gold": (200, 170, 60),
    "Silver": (170, 170, 175),
    "Grey": (110, 110, 110),
    "Black": (25, 25, 25),
    "White": (235, 235, 235),
    "Turquoise": (40, 180, 180),
    "Teal": (30, 130, 130),
}

# Familias compatibles: "Pink" es un tono de rojo, "Gold" de amarillo,
# "Turquoise"/"Teal" son azules-cian (pintura "Light Blue"). Solo se castiga
# cuando dos colores CROMÁTICOS de familias distintas chocan.
_COLOR_FAMILY = {
    "Red": "red", "Pink": "red",
    "Orange": "orange",
    "Yellow": "yellow", "Gold": "yellow",
    "Green": "green",
    "Blue": "blue", "Teal": "blue", "Turquoise": "blue",
    "Purple": "purple",
    "White": "neutral", "Silver": "neutral", "Grey": "neutral",
    "Black": "neutral", "Brown": "neutral",
}
_NEUTRAL = "neutral"


def _nearest_body(rgb: tuple[int, int, int]) -> str | None:
    """Bucket de carrocería más cercano; None si es fondo packshot."""
    r, g, b = rgb
    mx, mn = max(rgb), min(rgb)
    if mx > 245 and mx - mn < 12:  # casi-blanco puro → fondo
        return None
    if mx < 18:  # casi-negro puro → sombra
        return None
    best, best_d = None, 10**9
    for name, ref in _BODY_RGB.items():
        d = (r - ref[0]) ** 2 + (g - ref[1]) ** 2 + (b - ref[2]) ** 2
        if name == "White" and mx < 210:
            continue
        if name == "Black" and mx > 70:
            continue
        if d < best_d:
            best_d, best = d, name
    return best


def image_body_colors(im, max_side: int = 192, min_share: float = 0.12) -> list[str]:
    """Bucket de carrocería dominante de la FOTO, SOLO si la señal es inequívoca.

    Devuelve [] (señal ambigua → no se castiga) en los casos donde el empaque
    o el fondo dominan. Medido sobre los fixtures (2026-09-29):

      · studio shot coche BLANCO/GRIS → valid 1-5 %  → []  (el empaque rojo
        del blister es lo único saturado; no es la carrocería)
      · box amarillo/cian + coche azul → Yellow 37 % / Turquoise 27 % → []
        (sin dominante claro)
      · foto real de coche ROSA llenando el encuadre → valid alto + Pink con
        mayoría clara → ['Pink']  ← el caso que queremos castigar

    Puerta de confianza (TODAS deben cumplirse):
      1. ≥15 % de píxeles saturados (si no, es un coche neutro o hay poco
         coche en el encuadre → no hay señal de color fiable)
      2. el bucket dominante es CROMÁTICO (no blanco/plata/negro/marrón)
      3. dominante ≥40 % del peso total
      4. dominante ≥1.4× el segundo (si no, hay dos colores compitiendo
         → packaging vs carrocería)
    """
    try:
        from PIL import Image as PILImage
        import numpy as np

        w, h = im.size
        side = int(min(w, h) * 0.72)
        cx, cy = w // 2, h // 2
        box = (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)
        crop = im.crop(box).resize((max_side, max_side), PILImage.Resampling.BILINEAR)
        arr = np.asarray(crop, dtype=np.float32)

        mx = arr.max(axis=2)
        mn = arr.min(axis=2)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)

        # Fondo packshot / sombras / grises apagados → fuera
        valid = (mx > 28) & (mx < 248) & (sat > 0.16)
        valid_ratio = float(valid.mean())
        if valid_ratio < 0.15:
            return []  # coche neutro o casi todo fondo → sin señal

        # Cuenta por bucket ponderando saturación (la pintura manda)
        counts: dict[str, float] = {}
        rs = arr[..., 0][valid]
        gs = arr[..., 1][valid]
        bs = arr[..., 2][valid]
        ss = sat[valid]
        for i in range(0, len(rs), 7):  # muestreo: rápido y suficiente
            bucket = _nearest_body((int(rs[i]), int(gs[i]), int(bs[i])))
            if not bucket:
                continue
            counts[bucket] = counts.get(bucket, 0.0) + float(ss[i])
        if not counts:
            return []
        total = sum(counts.values()) or 1.0
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        top, top_w = ranked[0]
        top_share = top_w / total

        # 2) el dominante debe ser cromático
        if _COLOR_FAMILY.get(top, _NEUTRAL) == _NEUTRAL:
            return []
        # 3) mayoría clara
        if top_share < 0.40:
            return []
        # 4) margen sobre el segundo
        second_w = ranked[1][1] if len(ranked) > 1 else 0.0
        if second_w > 0 and top_w < second_w * 1.4:
            return []
        return [top]
    except Exception:  # noqa: BLE001 — el color nunca debe tumbar el identify
        return []


def colors_compatible(a: str | None, b: str | None) -> bool:
    """¿Dos buckets de color pueden ser la misma carrocería?

    Usado con señales DÉBILES (texto del nombre, colores ambiguos de foto):
    ahí lo neutro (blanco/plata/negro/cast) no debe castigar.
    """
    if not a or not b:
        return True
    fa, fb = _COLOR_FAMILY.get(a, _NEUTRAL), _COLOR_FAMILY.get(b, _NEUTRAL)
    if fa == _NEUTRAL or fb == _NEUTRAL:
        return True  # señal débil → no castigar
    return fa == fb


def colors_match_strict(a: str | None, b: str | None) -> bool:
    """Match exigente: para cuando la foto ya dio un color CONFIABLE.

    Un coche plata NO es un coche rosa: si la foto dice 'Pink' inequívocamente,
    solo una variante de la familia 'red' (Red/Pink) es aceptable. Lo neutro
    deja de servir de coartada.
    """
    if not a or not b:
        return True
    return _COLOR_FAMILY.get(a, _NEUTRAL) == _COLOR_FAMILY.get(b, _NEUTRAL)


# ── Número de competición: el desempate de liveries gemelos ─────────────────
# "Rexy" (#77, verde) y "Roxy" (#80, rosa) son el MISMO casting con el MISMO
# livery de dinosaurio: el embed no los distingue. Pero el dorsal SÍ está
# impreso en el coche y el OCR lo lee. Medido 2026-09-29: foto rosa #80 devolvía
# el verde #77 al 94.8 % aunque el OCR había leído '88\\n80'.
_RACE_NUM_RE = re.compile(r"#\s*(\d{1,3})\b")


def _name_race_num(name: str | None) -> str | None:
    """Dorsal declarado en el nombre del catálogo ('#80' → '80')."""
    if not name:
        return None
    m = _RACE_NUM_RE.search(name)
    return m.group(1) if m else None


def _ocr_race_nums(text: str) -> set[str]:
    """Dorsales que el OCR leyó en la foto (tolerante a '#', '##', espacios)."""
    if not text:
        return set()
    found = set(_RACE_NUM_RE.findall(text))
    # el OCR pierde el '#' a menudo: '88\\n80' → también aceptamos sueltos,
    # pero solo números de 2 dígitos (un '1' de "▶1" o '12' de "12 Hrs" son ruido)
    for m in re.finditer(r"\b(\d{2})\b", text):
        found.add(m.group(1))
    return found


def race_num_bonus(name: str | None, text: str, family_ok: bool = True) -> float:
    """Multiplicador por dorsal del coche vs dorsal leído en la foto.

    SOLO desempata dentro de la MISMA familia de casting (`family_ok`): un
    dorsal leído por el OCR (o ruido '88' de una pegatina) no puede catapultar
    a un coche de otra marca/escala por encima del ganador visual. Medido
    2026-09-30: con multiplicador global, la foto rosa devolvía un **Audi 1:18
    #88** porque el OCR lee '88\\n80'.

    · familia + '#NN' coincide con el OCR   → ×1.10 (es ESTE livery)
    · familia + '#NN' distinto al leído     → ×0.85 (es el gemelo)
    · sin dorsal, OCR sin número, o otra familia → ×1.0
    """
    if not family_ok:
        return 1.0
    nn = _name_race_num(name)
    if not nn:
        return 1.0
    ocr_nums = _ocr_race_nums(text)
    if not ocr_nums:
        return 1.0
    return 1.10 if nn in ocr_nums else 0.85


def _name_family(name: str | None) -> tuple[str, ...]:
    """Primeras 2 palabras del nombre — la familia de casting."""
    return tuple((name or "").lower().split()[:2])


def inject_race_num_candidates(
    pool: list[VisionMatch], text: str, top_n: int = 6
) -> list[VisionMatch]:
    """Añade al pool los candidatos del catálogo cuyo dorsal coincide con el OCR.

    Necesario porque el embed puede dejar FUERA a la pieza correcta: "Rexy"
    (#77 verde) y "Roxy" (#80 rosa) comparten livery de dinosaurio y el rosa
    ni siquiera entraba al top-8 aunque el OCR hubiera leído '80'. Aquí se
    recuperan por el dorsal impreso en el coche.

    Entrar al pool NO gana: se inyectan con score alto (0.92) para poder
    competir, y el multiplicador + color deciden después. Si no hay pool
    todavía, no se inyecta nada (sin familia de referencia).
    """
    nums = _ocr_race_nums(text)
    if not nums or not pool:
        return pool
    try:
        index, _ = get_index()
    except Exception:  # noqa: BLE001
        return pool
    have = {m.id for m in pool}
    # familia de referencia = la del actual top (para no colar otras marcas)
    ref_family = _name_family(pool[0].name if pool else None)
    injected: list[VisionMatch] = []
    for e in index:
        name = e.get("name") or ""
        nn = _name_race_num(name)
        if not nn or nn not in nums or e.get("id") in have:
            continue
        if _name_family(name) != ref_family:
            continue  # otra familia de casting → no inyectar
        injected.append(
            VisionMatch(
                id=e["id"], toy=e.get("toy"), name=name, score=0.92, source="ocr"
            )
        )
        # Marcado: estos NO son evidencia visual, entran al pool para que el
        # usuario los vea en sugerencias. Un inyectado solo no debe poder
        # autoconfirmar la pieza.
        injected[-1]._injected = True  # type: ignore[attr-defined]
        have.add(e["id"])
        if len(injected) >= top_n:
            break
    if injected:
        pool = pool + injected
        pool.sort(key=lambda x: -x.score)
        _dbg(
            "dorsal",
            f"{len(injected)} inyectados por dorsal {sorted(nums)} "
            f"(familia {ref_family}): " + ", ".join(f"{m.name[:34]}#{_name_race_num(m.name)}" for m in injected[:3]),
        )
    else:
        _dbg("dorsal", f"sin inyección (nums={sorted(nums)}, familia={ref_family}, pool={len(pool)})")
    return pool


# ── Señal de marca desde el OCR/empaque ─────────────────────────────────────
# Un Hot Wheels no debe ganarle a un Mini GT solo por forma similar: si el
# texto del empaque dice "HOT WHEELS", priorizamos esa marca.
BRAND_TEXT_HINTS: dict[str, tuple[str, ...]] = {
    "Mini GT": ("mini gt", "minigt", "mgt", "tsm models"),
    "Mattel": (
        "hot wheels",
        "hotwheels",
        "matchbox",
        "mattel",
        "monster trucks",
        "treasure hunt",
    ),
    "Inno64": ("inno64", "inno 64"),
    "Tarmac Works": ("tarmac", "global64", "hobby64"),
    "Kaido House": ("kaido house", "kaidohouse", "kaido"),
    "Tomica": ("tomica", "トミカ", "takara tomy", "takaratomy"),
    "Pop Race": ("pop race", "tiny hong kong"),
    "Cars de Disney y Pixar": ("cars", "disney", "pixar", "lightning mcqueen"),
}

# Alias → marca canónica. Lo que el usuario elige en el selector de la app
# (o escribe a mano) debe caer en las mismas claves que BRAND_TEXT_HINTS.
BRAND_ALIASES: dict[str, str] = {
    "mini gt": "Mini GT", "minigt": "Mini GT", "mgt": "Mini GT", "tsm": "Mini GT",
    "tsm models": "Mini GT",
    "hot wheels": "Mattel", "hotwheels": "Mattel", "hw": "Mattel", "mattel": "Mattel",
    "matchbox": "Mattel", "monster trucks": "Mattel",
    "inno64": "Inno64", "inno 64": "Inno64",
    "tarmac": "Tarmac Works", "tarmac works": "Tarmac Works", "global64": "Tarmac Works",
    "hobby64": "Tarmac Works",
    "kaido": "Kaido House", "kaido house": "Kaido House", "kaidohouse": "Kaido House",
    "tomica": "Tomica", "takara tomy": "Tomica",
    "pop race": "Pop Race", "poprace": "Pop Race",
    "cars disney": "Cars de Disney y Pixar", "disney": "Cars de Disney y Pixar",
    "pixar": "Cars de Disney y Pixar",
}


def normalize_brand(raw: str | None) -> str | None:
    """Marca declarada por el usuario → clave canónica. None si no se reconoce."""
    if not raw:
        return None
    key = re.sub(r"\s+", " ", str(raw).strip().lower())
    return BRAND_ALIASES.get(key) or BRAND_ALIASES.get(key.replace("-", " "))


def detect_brand_hints(text: str) -> set[str]:
    """Marcas mencionadas en el OCR del empaque / foto."""
    t = (text or "").lower()
    flat = re.sub(r"\s+", " ", t)
    hints: set[str] = set()
    for brand, keys in BRAND_TEXT_HINTS.items():
        if any(k in t or k in flat for k in keys):
            hints.add(brand)
    # "MINI GT" a veces el OCR lo parte en líneas: MINI\nG\nTM
    if re.search(r"mini\s*g\s*t", t) or re.search(r"\bmgt\s?\d", t):
        hints.add("Mini GT")
    # "Hot Wheels" implica Mattel
    if "Mattel" in hints:
        hints.add("Mattel")
    return hints


def _match_brand(match: VisionMatch) -> str | None:
    """Marca del candidato (del nombre — el índice no trae manufacturer)."""
    n = (match.name or "").lower()
    for brand, keys in BRAND_TEXT_HINTS.items():
        if any(k in n for k in keys):
            return brand
    toy = (match.toy or "").upper()
    if toy.startswith("MGT"):
        return "Mini GT"
    if toy.startswith("IN64"):
        return "Inno64"
    if toy.startswith("T64") or toy.startswith("T18"):
        return "Tarmac Works"
    return None


def apply_brand_prior(
    matches: list[VisionMatch],
    brand_hints: set[str],
    image_colors: list[str] | None = None,
    brand_explicit: bool = False,
) -> list[VisionMatch]:
    """Reordena/ajusta scores: la marca del empaque manda sobre la forma visual.

    `image_colors` son los buckets de color medidos en la FOTO subida (no del
    catálogo). Un candidato cuyo nombre declara un color cromático de otra
    familia que la foto se castiga — esto es lo que evita que un livery gemelo
    de otro color gane cuando el OCR no trae palabras de color.

    `brand_explicit=True` significa que la marca la Dijo el usuario (selector
    de la app), no que la dedujo el OCR: ahí el prior es más contundente,
    porque la voz del usuario es más fiable que un OCR ruidoso. Cada marca
    además trae su propio vocabulario de identidad (MGT##### en Mini GT,
    UPC/SKU en Hot Wheels, JAN en Tomica) — el selector alimenta ese camino.
    """
    if not matches:
        return matches
    # Color del OCR vs nombre del candidato — aplica a TODOS (visual y texto).
    # Un "…Orange Chrome" no le gana a "…White Pearl" si el empaque dice White.
    _COLOR_WORDS = (
        "white", "blanco", "black", "negro", "red", "rojo", "blue", "azul",
        "silver", "plata", "grey", "gray", "gris", "yellow", "amarillo",
        "green", "verde", "orange", "naranja", "purple", "morado", "pink",
        "rosa", "gold", "dorado", "pearl",
    )
    text_l = (" ".join(sorted(brand_hints)) if brand_hints else "").lower()
    # brand_hints no trae el OCR completo — el caller debe pasar color_text
    # de todos modos: lo sacamos de un atributo opcional en matches[0] si existe.
    ocr_text_l = ""
    for m in matches:
        meta = getattr(m, "_ocr_text", None)
        if isinstance(meta, str) and len(meta) > len(ocr_text_l):
            ocr_text_l = meta.lower()
    text_colors = {c for c in _COLOR_WORDS if re.search(rf"\b{c}\b", ocr_text_l)}
    if "white" in ocr_text_l or "blanco" in ocr_text_l or "pearl" in ocr_text_l:
        text_colors.add("white")

    # Bucket → palabra de color para poder comparar contra el nombre.
    _BUCKET_WORD = {
        "Red": "red", "Pink": "pink", "Orange": "orange", "Yellow": "yellow",
        "Gold": "gold", "Green": "green", "Teal": "green", "Turquoise": "green",
        "Blue": "blue", "Purple": "purple", "White": "white", "Silver": "silver",
        "Grey": "grey", "Black": "black", "Brown": "brown",
    }
    img_buckets = list(image_colors or [])

    # Familia del actual top: el dorsal solo desempata DENTRO de esa familia.
    ref_family = _name_family(matches[0].name if matches else None)
    # Con marca declarada por el usuario el prior se endurece: +0.20 / ×0.45
    # en vez de +0.12 / ×0.72. Es la diferencia entre "creo que dice MINI GT"
    # (OCR) y "esto ES un Mini GT" (el usuario lo eligió).
    boost = 0.20 if brand_explicit else 0.12
    penalty = 0.45 if brand_explicit else 0.72

    adjusted: list[VisionMatch] = []
    for m in matches:
        mb = _match_brand(m)
        score = m.score
        if brand_hints:
            if mb and mb in brand_hints:
                score = min(0.99, score + boost)
            elif mb and mb not in brand_hints:
                score = score * penalty
        name_l = (m.name or "").lower()
        name_colors = {c for c in _COLOR_WORDS if re.search(rf"\b{c}\b", name_l)}
        if name_colors and text_colors and not (name_colors & text_colors):
            score *= 0.55
        # Color de la FOTO vs color declarado en el nombre del candidato:
        # solo CASTIGA si no casa. Un boost positivo (×1.12) se probó el
        # 2026-09-30 y reventó dos casos del harness sin ganar nada — en los
        # coches sueltos neutros el detector de color devuelve [] y el boost
        # ni se usaba, solo inflaba falsos positivos.
        if name_colors and img_buckets:
            ok = any(
                colors_compatible(_BUCKET_WORD.get(b), w)
                for b in img_buckets
                for w in name_colors
            )
            if not ok:
                score *= 0.55
        # Dorsal de competición: lo impreso en el coche desambigua liveries
        # gemelos ("Rexy" #77 verde vs "Roxy" #80 rosa — mismo casting, mismo
        # livery de dinosaurio). Solo DENTRO de la familia del top: un '88' de
        # ruido no puede catapultar a un Audi 1:18 por encima del Porsche.
        score *= race_num_bonus(m.name, ocr_text_l, _name_family(m.name) == ref_family)
        # Nunca por encima de 1.0: el boost de marca (+0.20) y el de dorsal
        # (×1.10) podían apilarse y publicar scores de 1.012 en las sugerencias.
        score = max(0.0, min(1.0, score))
        adj = VisionMatch(
            id=m.id, toy=m.toy, name=m.name, score=score, source=m.source
        )
        # Guardamos el score CRUDO del embed: la rama de "copia literal del
        # studio shot" (≥0.99) debe mirar este, no el inflado por boosts. Un
        # candidato INYECTADO por dorsal no trae score visual propio → 0.0, y
        # así nunca fingirá copia literal ni autoconfirmará solo.
        try:
            adj._raw_score = (
                0.0
                if getattr(m, "_injected", False)
                else float(getattr(m, "_raw_score", m.score))
            )
        except Exception:  # noqa: BLE001
            pass
        adjusted.append(adj)
    adjusted.sort(key=lambda x: -x.score)
    return adjusted


def score_name_against_ocr_text(name: str | None, ocr_text: str, sem_cache: dict | None = None) -> float:
    """0..1 — cuánto confirma el texto del empaque a un candidato del catálogo.

    Tokens ponderados: "AlphaTauri AT04 #22 Yuki Tsunoda" debe ganar a
    "RB VCARB 01 #22 Yuki Tsunoda" aunque compartan el piloto y el número.

    Matching difuso: OCR de imágenes IA lee "AT04" como "ATB", "2023" como
    "2029". Un token exacto fallaría; con similitud >= 0.65 contamos medio hit.

    OJO con el coste: `search_catalog_by_text` llama esta función sobre TODO el
    índice (27k nombres) con el MISMO `ocr_text`. Recalcular tokens/colores del
    OCR en cada llamada y correr difflib a ciegas eran ~4-9 s por request (medido
    en prod 2026-09-29). Por eso se cachean los derivados del OCR (una sola
    clave: el texto del request) y los ratios difflib.
    """
    import difflib

    text = (ocr_text or "").lower()
    if not text or not name:
        return 0.0

    # Derivados del OCR cacheados por texto (1 entrada: el del request actual).
    global _TXT_KEY, _TXT_TOKS, _TXT_COLORS
    if text != _TXT_KEY:
        _TXT_KEY = text
        _TXT_TOKS = _tokens(text)
        # Tope de tokens del OCR: en imágenes patológicas el OCR devuelve
        # párrafos enteros (20-40 tokens) y el bucle difusivo interno se iba a
        # millones de comparaciones (medido: `texto` = 26-90 s). Nos quedamos
        # con los más largos, que son los informativos (nombres de modelo,
        # marcas), y descartamos el ruido corto.
        if len(_TXT_TOKS) > 14:
            _TXT_TOKS = set(sorted(_TXT_TOKS, key=len, reverse=True)[:14])
        _TXT_COLORS = {c for c in _COLOR_WORDS if re.search(rf"\b{c}\b", text)}
        if "white" in text or "blanco" in text or "pearl" in text:
            _TXT_COLORS.add("white")
        _RATIO_CACHE.clear()

    name_tokens = _tokens(name)
    if not name_tokens:
        return 0.0
    text_tokens = _TXT_TOKS
    total = sum(_token_weight(t) for t in name_tokens)
    hit = 0.0
    for t in name_tokens:
        w = _token_weight(t)
        if t in text_tokens:
            hit += w
        else:
            # fuzzy: ¿algún token del OCR se parece a este?
            for tt in text_tokens:
                key = (t, tt) if t <= tt else (tt, t)
                ratio = _RATIO_CACHE.get(key)
                if ratio is None:
                    ratio = difflib.SequenceMatcher(None, t, tt).ratio()
                    if len(_RATIO_CACHE) < 20000:
                        _RATIO_CACHE[key] = ratio
                if ratio >= 0.65:
                    hit += w * 0.6  # hit parcial por similitud
                    break
    if not total:
        return 0.0
    base = hit / total
    # Especificidad: un nombre corto ("'17 Nissan GT-R") que matchea sus 3
    # palabras no debe ganarle a "Nissan GT-R Nismo 2024 Brilliant White Pearl"
    # que cubre mucho más del OCR. Bonus por nº de tokens únicos del nombre.
    uniq = len(name_tokens)
    specificity = min(1.0, uniq / 7.0)
    score = base * (0.72 + 0.28 * specificity)
    # Consistencia de COLOR: OCR "White Pearl" vs candidato "…NISMO Red" → penal.
    name_colors = {c for c in _COLOR_WORDS if re.search(rf"\b{c}\b", (name or "").lower())}
    if name_colors and _TXT_COLORS and not (name_colors & _TXT_COLORS):
        score *= 0.55
    # MiniLM semántico: corrige OCR garbled ("Briiant White Pane" ↔
    # "Brilliant White Pearl"). 70% tokens + 30% semántico.
    # SOLO si `sem_cache` es dict (fase-2 del ranking): correr MiniLM 16k veces
    # por request colgaba el identify. La fase-1 (tokens) es la criba barata.
    if isinstance(sem_cache, dict):
        try:
            from app.services.text_match import similarity as _sem_sim

            key = (name or "")[:120]
            sem = sem_cache.get(key)
            if sem is None:
                sem = _sem_sim(ocr_text or "", name or "")
                sem_cache[key] = sem
            if sem > 0:
                score = 0.7 * score + 0.3 * min(1.0, sem * 1.25)
        except Exception:  # noqa: BLE001
            pass
    return score


def pick_ocr_confirmed_match(
    matches: list[VisionMatch],
    ocr_text: str,
    min_score: float = 0.35,
    min_margin: float = 0.10,
) -> VisionMatch | None:
    """Candidato confirmado inequívocamente por el texto (mejor score + margen).

    Umbrales tolerantes a OCR imperfecto (imágenes IA / empaques borrosos):
    "AlphaTauri ATB #22 Yuki Tsunoda" debe seguir llevando a MGT00726 aunque
    "AT04" salga garbled como "ATB".
    """
    if not matches:
        return None
    scored = sorted(
        ((m, score_name_against_ocr_text(m.name, ocr_text)) for m in matches),
        key=lambda x: -x[1],
    )
    (first, s1), rest = scored[0], scored[1:]
    if s1 < min_score:
        return None
    # Rechazar branding genérico: "MINI GT logo" matchea por "MINI GT" en el
    # watermark/OCR pero NO identifica una pieza concreta.
    if re.search(r"\blogo\b", first.name, re.I) and not _TOY_RE.search(first.name):
        return None
    # Con scores altos casi empatados, preferir el nombre MÁS específico
    # (más tokens) — "'17 Nissan GT-R" no debe ganar a "…2024 Brilliant White Pearl".
    if rest:
        gap = s1 - rest[0][1]
        if gap < min_margin:
            # Empate real entre candidatos distintos (ej: dos coches del mismo
            # piloto) → NO elegir. Solo desempatar si uno es claramente más
            # específico Y sus tokens extra aparecen en el OCR (no solo "más largo").
            tied = [(m, s) for m, s in scored if s >= s1 - 0.05]
            text_tokens = {t for t in re.sub(r"[^a-z0-9]+", " ", (ocr_text or "").lower()).split() if len(t) >= 3}
            def _extra_hits(m: VisionMatch) -> int:
                toks = [t for t in re.sub(r"[^a-z0-9]+", " ", (m.name or "").lower()).split() if len(t) >= 3]
                return sum(1 for t in toks if t in text_tokens)
            best_extra = max(_extra_hits(m) for m, _ in tied)
            if best_extra < 4:
                return None  # empate ambiguo — no inventar
            specific = [m for m, _ in tied if _extra_hits(m) == best_extra]
            specific.sort(key=lambda x: -len((x.name or "").split()))
            return specific[0]
        return first
    return first


# ── 1) Barcode ──────────────────────────────────────────────────────────────

def read_barcode(np_rgb: np.ndarray) -> str | None:
    try:
        import cv2

        gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
        det = cv2.barcode_BarcodeDetector()
        ok, decoded, _, _ = det.detectAndDecode(gray)
        if ok and decoded:
            for code in decoded:
                c = (code or "").strip()
                if len(c) >= 8:
                    return c
        # upscale como 2º intento (códigos pequeños)
        big = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        ok, decoded, _, _ = det.detectAndDecode(big)
        if ok and decoded:
            for code in decoded:
                c = (code or "").strip()
                if len(c) >= 8:
                    return c
    except Exception:
        return None
    return None


# ── 2) Embeddings ───────────────────────────────────────────────────────────

def embed_image(im) -> np.ndarray | None:
    """Embedding visual del query. DINOv2 (768-d) si está, MobileNet (576-d) fallback.

    DEBE coincidir con la dim del índice cargado por get_index().
    """
    # 1) DINOv2 — fine-grained (variantes de color/livery)
    try:
        from app.services.dinov2_embed import embed_image as dino_embed

        vec = dino_embed(im)
        if vec is not None:
            arr = np.asarray(vec, dtype=np.float32)
            n = float(np.linalg.norm(arr))
            return arr / n if n > 0 else arr
    except Exception as exc:  # noqa: BLE001
        print(f"[vision] dinov2 no disponible: {exc}", flush=True)
    # 2) MobileNetV3 (legacy)
    try:
        print(f"[vision] embed: cargando modelo...", flush=True)
        model = get_model()
        print(f"[vision] embed: modelo cargado, preprocesando...", flush=True)
        x = preprocess(im)
        print(f"[vision] embed: preprocesado {x.shape}, prediciendo...", flush=True)
        vec = model.predict(x, verbose=0)[0]
        print(f"[vision] embed: predicción OK, dim={vec.shape}", flush=True)
        return l2_normalize(vec.astype(np.float32))
    except Exception as exc:  # noqa: BLE001
        print(f"[vision] embed falló: {type(exc).__name__}: {exc}", flush=True)
        return None


_MV_MASK: tuple[int, np.ndarray] | None = None


def _mv_rows(index: list[dict]) -> np.ndarray:
    """True en las vistas MULTIVISTA (view ≥ 20: coche aislado de la galería)."""
    global _MV_MASK
    if _MV_MASK is None or _MV_MASK[0] != id(index):
        _MV_MASK = (id(index), np.array([int(e.get("view") or 0) >= 20 for e in index]))
    return _MV_MASK[1]


_MGT_GAL: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
_MGT_GAL_FAILED = False


def _mgt_gallery() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Galería Mini GT con el coche AISLADO: CLS DINOv2 de las fotos 0-5 de cada
    pieza (`scripts/mgt_feats.py` → data/mgt_feats.npz). (matriz, toy, id)."""
    global _MGT_GAL, _MGT_GAL_FAILED
    if _MGT_GAL is not None or _MGT_GAL_FAILED:
        return _MGT_GAL
    try:
        p = Path(__file__).resolve().parents[2] / "data" / "mgt_feats.npz"
        z = np.load(p, allow_pickle=True)
        m = z["cls"].astype(np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-8
        _MGT_GAL = (m, np.asarray(z["toy"]), np.asarray(z["vid"]))
        print(f"[vision] galería Mini GT aislada: {len(m)} fotos", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[vision] galería Mini GT no disponible: {exc}", flush=True)
        _MGT_GAL_FAILED = True
    return _MGT_GAL


def match_mgt_gallery(emb: np.ndarray, top_k: int = 8) -> list[VisionMatch]:
    """Coche aislado vs galería Mini GT (mismo tratamiento a ambos lados).

    Medido 2026-10-05 con 7 sueltos reales: la pieza correcta 1ª en 4/7 y en el
    top-4 en 6/7; contra el índice mezclado (todas las marcas + foto principal
    con caja) el R32 quedaba 28º y el RX-7 25º. Un cabezal entrenado sobre la
    galería subía la prueba de catálogo (top-1 20 %→43 %) pero EMPEORABA en
    fotos reales (sobreajuste al estilo estudio) → no se usa."""
    gal = _mgt_gallery()
    if gal is None:
        return []
    m, toys, vids = gal
    sims = m @ emb
    names = _index_names()
    out: list[VisionMatch] = []
    seen: set[str] = set()
    for i in np.argsort(-sims):
        t = str(toys[i])[:8]
        if t in seen:
            continue
        seen.add(t)
        vid = str(vids[i])
        out.append(VisionMatch(id=vid, toy=t, name=names.get(vid, t), score=float(sims[i]), source="embed:seg"))
        if len(out) >= top_k:
            break
    return out


_NAMES: tuple[int, dict[str, str]] | None = None


def _index_names() -> dict[str, str]:
    global _NAMES
    index, _ = get_index()
    if _NAMES is None or _NAMES[0] != id(index):
        _NAMES = (id(index), {str(e["id"]): (e.get("name") or "") for e in index})
    return _NAMES[1]


def match_embedding(emb: np.ndarray, top_k: int = 8, include_mv: bool = False) -> list[VisionMatch]:
    """Top-k por id. Las vistas multivista (coche aislado sobre blanco) solo se
    consultan con `include_mv=True` — es decir, desde el embed del coche
    AISLADO. Medido 2026-10-05: si el encuadre completo las ve, una foto con
    fondo desordenado casa con coches aislados cualesquiera (Urus amarillo por
    un Civic azul) y el bench baja 103→100."""
    index, matrix = get_index()
    sims = matrix @ emb
    if not include_mv:
        mask = _mv_rows(index)
        if mask.any():
            sims = np.where(mask, -9.0, sims)
    order = np.argsort(-sims)
    # Multi-view: el mismo id puede tener varias vistas; nos quedamos con la
    # mejor puntuación por id para que un producto no acapare el top-k.
    seen: dict[str, float] = {}
    results: list[VisionMatch] = []
    for i in order:
        iid = index[i]["id"]
        if iid in seen:
            continue
        seen[iid] = float(sims[i])
        results.append(
            VisionMatch(
                id=iid,
                toy=index[i].get("toy"),
                name=index[i].get("name") or "",
                score=float(sims[i]),
                source="embed",
            )
        )
        if len(results) >= top_k:
            break
    return results


def search_catalog_by_text(
    ocr_text: str, top_k: int = 8, budget_s: float = 2.0
) -> list[VisionMatch]:
    """Búsqueda textual en TODO el catálogo por tokens del OCR.

    Necesaria cuando el embedding falla (foto IA, blister, ángulo raro): el
    top-K visual puede no contener la pieza real, pero el texto del empaque
    ("ALPHATAURI", "AT04 #22", "Tsunoda") sí la identifica en el catálogo.

    Usa score_name_against_ocr_text (fuzzy + tokens ponderados) porque la
    intersección exacta falla con OCR garbled: "AT04" → "ATB&", "2023" → "2028".

    `budget_s` es un TECHO duro de tiempo: sobre 27k nombres el scorer difusivo
    se disparaba a 26-90 s con OCR largo (imágenes IA). Con presupuesto agotado
    devolvemos lo que llevamos acumulado — peor es colgar el request.
    """
    t_start = time.time()
    text = (ocr_text or "").lower()
    if len(text.strip()) < 4:
        return []
    text_toks = _tokens(text)
    if not text_toks:
        return []
    if len(text_toks) > 14:
        text_toks = set(sorted(text_toks, key=len, reverse=True)[:14])

    index, _ = get_index()
    # Fase 1: prefilter barato + scorer por tokens sobre el catálogo.
    # Fase 2: MiniLM solo en los ~20 mejores.
    cheap: list[tuple[VisionMatch, float]] = []
    for e in index:
        if time.time() - t_start > budget_s:
            break
        name = e.get("name") or ""
        if not name:
            continue
        # Prefilter: sin afinidad mínima con el OCR no gastamos el scorer
        # difusivo. Se acepta coincidencia exacta de token O prefijo de 2
        # (cubre el OCR garbled "AT04"↔"ATB").
        nt = _cached_name_tokens(name)
        if nt:
            aff = False
            for n in nt:
                if n in text_toks:
                    aff = True
                    break
                for t in text_toks:
                    if n[:2] == t[:2]:
                        aff = True
                        break
                if aff:
                    break
            if not aff:
                continue
        s = score_name_against_ocr_text(name, text)  # sem_cache=None → solo tokens
        if s < 0.20:
            continue
        cheap.append(
            (
                VisionMatch(
                    id=e["id"],
                    toy=e.get("toy"),
                    name=name,
                    score=float(s),
                    source="ocr",
                ),
                s,
            )
        )
    cheap.sort(key=lambda x: -x[1])
    shortlist = [m for m, _s in cheap[:20]]  # solo VisionMatch, no tuplas
    # MiniLM BATCH: un encode del query + uno del lote de nombres (2 calls),
    # no 50× similarity secuenciales (100 encodes ≈ 13s → colgaba el request).
    try:
        from app.services.text_match import best_name_match

        names = [m.name for m in shortlist]
        ranked = best_name_match(text, names, top_k=top_k)
        if ranked:
            by_name = {m.name: m for m in shortlist}
            scored = []
            for n, s in ranked:
                m = by_name.get(n)
                if m is not None:
                    m.score = float(s)
                    scored.append(m)
            return scored[:top_k]
    except Exception as exc:  # noqa: BLE001
        print(f"[vision] MiniLM rerank falló: {exc}", flush=True)
    # fallback: solo tokens (sem_cache=None)
    for m in shortlist:
        m.score = score_name_against_ocr_text(m.name, text)
    shortlist.sort(key=lambda x: -x.score)
    return shortlist[:top_k]


# ── 3) OCR ──────────────────────────────────────────────────────────────────

# OCR lee "MGT01160" como "MGTO1160" (letra O por cero) — toleramos ambos.
_TOY_RE = re.compile(r"\b(MGT\s?[O0]?\s?\d{3,5})\b", re.I)
_UPC_RE = re.compile(r"\b(\d{12,14})\b")


def _ocr_once(im, timeout: int = 15) -> str:
    """Un pase de Tesseract (eng+spa) sobre una imagen PIL.

    timeout=15: en la Ampere A1 (Oracle) un pase sobre crop×3 puede tardar
    >5s y pytesseract lanza RuntimeError — con timeout=5 el catch devolvía
    "" y el identify se quedaba SIN texto (falla real en producción).
    """
    import pytesseract

    return pytesseract.image_to_string(im, lang="eng+spa", timeout=timeout) or ""


def _easyocr_read(im) -> str:
    """EasyOCR respaldo — deshabilitado por latencia en Lambda.
    Gemini Vision es más rápido y más inteligente como respaldo de Tesseract."""
    return ""


# Techo de píxeles para las variantes OCR — sin esto, una foto de 4000px
# escalada 2.5x genera un lienzo de 10000px y Tesseract tarda 15-30s.
_OCR_MAX_EDGE = 3200


def _ocr_variants(np_rgb: np.ndarray) -> list:
    """Variantes de la imagen donde el texto del empaque suele sobrevivir.

    Empaques Mini GT: tarjeta negra con texto blanco + etiqueta pequeña abajo
    con "AlphaTauri AT04 #22 / Yuki Tsunoda". Un solo pase pierde texto.
    Para fotos de blister reales: grayscale + contraste + binarización ayudan
    a Tesseract a leer texto pequeño sobre fondos complejos.
    """
    from PIL import Image, ImageOps, ImageFilter

    base = Image.fromarray(np_rgb).convert("RGB")
    w, h = base.size
    variants = []

    def scaled(img, factor):
        tw, th = int(img.width * factor), int(img.height * factor)
        m = max(tw, th)
        if m > _OCR_MAX_EDGE:
            s = _OCR_MAX_EDGE / m
            tw, th = int(tw * s), int(th * s)
        return img.resize((tw, th), Image.Resampling.LANCZOS)

    def to_gray_enhanced(img):
        """Grayscale + autocontrast + sharpen — el combo más efectivo para Tesseract."""
        gray = img.convert("L")
        gray = ImageOps.autocontrast(gray, cutoff=2)
        gray = gray.filter(ImageFilter.SHARPEN)
        return gray.convert("RGB")

    # Orden: primero las variantes más efectivas para blister real.
    # El primer lote (4 workers) debe capturar la mayoría de los casos.

    # 1) Bottom 35% enhanced — donde vive "▶ NNN" / nombre del modelo
    bot_label = base.crop((0, int(h * 0.65), w, h))
    variants.append(scaled(to_gray_enhanced(bot_label), 3.0))

    # 2) Bottom half enhanced
    bot = base.crop((0, h // 2, w, h))
    variants.append(scaled(to_gray_enhanced(bot), 2.5))

    # 3) Completa enhanced
    variants.append(scaled(to_gray_enhanced(base), 2.0))

    # 4) Bottom half invertida (blanco sobre negro)
    variants.append(ImageOps.invert(scaled(bot, 2.5)))

    # --- Segundo lote (solo si falta identidad) ---

    # 5) Completa original escalada
    variants.append(scaled(base, 2.0))

    # 6) Bottom label sin enhanced (fallback)
    variants.append(scaled(bot_label, 3.0))

    return variants


# Pistas de que el texto del OCR habla de coches de colección/empaque (para no
# confundir un documento o captura con una pieza). Solo se usa en _doc_like.
_DIECAST_HINT_RE = re.compile(
    r"mini\s*g\s*t|hot\s*wheels|matchbox|inno\s*64|tomica|tarmac|kaido|pop\s*race|"
    r"majorette|greenlight|auto\s*world|jada|maisto|bburago|1\s*[:/]\s*(64|43|18)|"
    r"die\s*-?\s*cast|collector|[LR]H[DO]\s*version|mgt\s*\d|\bupc\b|ages?\s*\d|mattel",
    re.I,
)
# Rastro de «MINI» leído al revés/espejo: IMIM, WINI, MIN! (solo mayúsculas).
_MIRROR_HINT_RE = re.compile(r"[IMWN1l!|]{3,}")
_PACK_MARK_RE = re.compile(r"[LR]H[DO]\s*version|mini\s*g\s*t", re.I)


def _toy_from_text(text: str) -> tuple[str | None, float]:
    """toy# de un texto de OCR de empaque Mini GT → (MGT#####, confianza).

    Orden de fiabilidad: MGT##### literal (.95) → ▶N (.80) → N junto a
    «LHD/RHD version shown» (.75) → único número 1-999 con marca de empaque
    (.72). El OCR pierde la flecha, parte «MINI GT» en líneas y confunde D/O,
    así que el número se busca también en la línea contigua a «version shown».
    """
    m = _TOY_RE.search(text)
    if m:
        raw = re.sub(r"\s+", "", m.group(1)).upper()
        digits = re.sub(r"[^0-9]", "", raw)  # MGTO1160 → 1160
        if digits:
            return "MGT" + digits.zfill(5)[-5:], 0.95
    m = re.search(r"[>»▶]+\s*(\d{1,5})\b", text)
    if m:
        return "MGT" + m.group(1).zfill(5)[-5:], 0.80
    m = re.search(r"\b(\d{1,5})\s*[LR]H[DO]\s*version", text, re.I | re.S)
    if m:
        return "MGT" + m.group(1).zfill(5)[-5:], 0.75
    # Tarjeta BLISTER Mini GT: abajo a la izquierda «▶NNN» y a su lado la edad
    # «14+». El OCR pierde la flecha y el «14+» rompía la regla de «número
    # único». Medido 2026-10-04 en 14 blisters reales: el OCR leía el número
    # en 14/14 (985, 990, 1051…) pero no se extraía ninguno. El número es el que
    # está a ±2 líneas de la edad; 1-4 dígitos (hay toy# ≥1000).
    if re.search(r"\bmin[i!l]?\b|mini\s*g\s*t", text, re.I):  # el OCR a veces lee «MIN»
        lines = [ln.strip() for ln in text.splitlines()]
        near_age: set[int] = set()
        for i, ln in enumerate(lines):
            if re.fullmatch(r"1[45]\s*\+?", ln):
                for j in (i - 1, i + 1, i - 2, i + 2):
                    if 0 <= j < len(lines) and re.fullmatch(r"\d{1,4}", lines[j]):
                        n = int(lines[j])
                        if 1 <= n <= 9999 and n not in (14, 15):
                            near_age.add(n)
                            break
        if len(near_age) == 1:
            return "MGT" + str(next(iter(near_age))).zfill(5), 0.78
    if _PACK_MARK_RE.search(text):
        lines = [ln.strip() for ln in text.splitlines()]
        near: set[int] = set()
        for i, ln in enumerate(lines):
            if re.search(r"[LR]H[DO]\s*version", ln, re.I):
                for j in (i - 1, i + 1):
                    if 0 <= j < len(lines) and re.fullmatch(r"\d{1,3}", lines[j]) and 1 <= int(lines[j]) <= 999:
                        near.add(int(lines[j]))
        if len(near) == 1:
            return "MGT" + str(next(iter(near))).zfill(5), 0.75
        # SOLO si hay UN ÚNICO número en todo el OCR: con dos ('88\n80' =
        # dorsales de competición) es ambiguo y no opinamos.
        cands = {int(c) for c in re.findall(r"\b(\d{1,3})\b", text) if 1 <= int(c) <= 999}
        if len(cands) == 1:
            return "MGT" + str(next(iter(cands))).zfill(5), 0.72
    return None, 0.0


def read_ocr(np_rgb: np.ndarray) -> tuple[str, str | None, float, str | None]:
    """Devuelve (texto_crudo, toy_number, toy_confidence, upc).

    Multi-recorte adaptativo. El texto de IDENTIDAD de los empaques Mini GT
    vive abajo ("AlphaTauri AT04 #22 / Yuki Tsunoda", "▶ 726") y el branding
    grande ("SCUDERIA / ALPHATAURI") es solo marca — confundir uno con otro
    dejaba el matching por texto sin pieza posible. Por eso el recorte
    inferior SIEMPRE corre; los extras solo si aún falta señal.

    Se extrae también el indicador "▶ 726" de los empaques Mini GT: el
    sufijo numérico de MGT00726. Es la señal más fiable de identidad.

    Orden: **TrOCR primero** (rápido y preciso en texto impreso de empaque),
    Tesseract como fallback si TrOCR falla o devuelve poco.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"[vision] OCR: iniciando, shape={np_rgb.shape}", flush=True)
    texts: list[str] = []

    # ── 0) RapidOCR (primario) — scene-text de empaques, ~0.3-0.7s ───────
    t_fast = time.time()
    try:
        from app.services.scene_ocr import read_text as scene_read

        from PIL import Image as PILImage

        pil = PILImage.fromarray(np_rgb).convert("RGB")
        w, h = pil.size
        scene_crops = [
            pil.crop((0, int(h * 0.55), w, h)),  # etiqueta inferior
            pil,  # completa
        ]
        scene_texts = []
        for c in scene_crops:
            t = scene_read(c)
            if t and len(t.strip()) >= 4:
                scene_texts.append(t)
        if scene_texts:
            def _q(t: str) -> int:
                return len(re.findall(r"[A-Za-z0-9]{2,}", t or ""))
            best = max(scene_texts, key=_q)
            extras = [t for t in scene_texts if t is not best and _q(t) >= 3]
            texts.append(best)
            texts.extend(extras)
            print(
                f"[vision] RapidOCR OK {int((time.time()-t_fast)*1000)}ms → {best[:55]!r}",
                flush=True,
            )
    except Exception as exc:
        print(f"[vision] RapidOCR falló: {type(exc).__name__}: {exc}", flush=True)

    # ── 0b) TrOCR (secundario) SOLO si RapidOCR no vio nada ────────────────
    # Con texto ya leído — aunque sea corto/garbage ("88\n80") — TrOCR corre
    # en CPU ~15 s y devuelve lo mismo. Medido en prod 2026-09-29: ocr=15469 ms
    # por este salto, sobre el budget de 8 s.
    if not "\n".join(texts).strip():
        t_trocr = time.time()
        try:
            from app.services.trocr_ocr import read_text as trocr_read

            from PIL import Image as PILImage

            pil = PILImage.fromarray(np_rgb).convert("RGB")
            w, h = pil.size
            trocr_texts = []
            for c in (pil.crop((0, int(h * 0.55), w, h)), pil):
                t = trocr_read(c)
                if t:
                    trocr_texts.append(t)
            if trocr_texts:
                def _q(t: str) -> int:
                    return len(re.findall(r"[A-Za-z0-9]{2,}", t or ""))
                texts.append(max(trocr_texts, key=_q))
                print(
                    f"[vision] TrOCR extra {int((time.time()-t_trocr)*1000)}ms",
                    flush=True,
                )
        except Exception as exc:
            print(f"[vision] TrOCR falló: {type(exc).__name__}: {exc}", flush=True)

    joined_fast = "\n".join(texts)
    has_id_early = bool(_TOY_RE.search(joined_fast)) or bool(
        re.search(r"[>»▶]+\s*\d{3,5}\b", joined_fast)
    )
    # Si RapidOCR ya dio texto útil (marca+modelo), NO correr Tesseract:
    # aportaba poco y costaba 5-15s (era el resto del budget de 8s).
    fast_useful = len(re.findall(r"[A-Za-z0-9]{2,}", joined_fast)) >= 4
    # RapidOCR ya produjo ALGO (aunque sea basura): Tesseract va a leer la
    # misma basura 20× más lento. Medido en prod (foto real de coche en mano,
    # 2026-09-29): RapidOCR → 'YIN 00100 slamn' (3 tokens) disparaba el spray
    # de variantes Tesseract = 6.4 s inútiles sobre el budget de 8 s.
    rapid_said_something = bool(joined_fast.strip())

    if not fast_useful:
        # Solo el spray completo si RapidOCR NO vio nada: ahí Tesseract es la
        # única esperanza. Si RapidOCR ya dijo algo, un único pase barato en el
        # crop inferior (donde vive el "▶ 826") por si cazó mal el toy number.
        if not rapid_said_something:
            try:
                variants = _ocr_variants(np_rgb)
                t_ocr = time.time()
                BUDGET = 12.0  # solo fallback

                def run_variant(im):
                    try:
                        return _ocr_once(im)
                    except Exception as exc:
                        print(f"[vision] OCR variant falló: {type(exc).__name__}: {exc}", flush=True)
                        return ""

                with ThreadPoolExecutor(max_workers=4) as pool:
                    futs = [pool.submit(run_variant, im) for im in variants[:4]]
                    try:
                        for fut in as_completed(futs, timeout=BUDGET):
                            t = fut.result()
                            if t and t.strip():
                                texts.append(t)
                    except TimeoutError:
                        for fut in futs:
                            if fut.done() and not fut.cancelled():
                                try:
                                    t = fut.result(timeout=0)
                                    if t and t.strip():
                                        texts.append(t)
                                except Exception:
                                    pass

                joined = "\n".join(texts)
                has_identity = bool(_TOY_RE.search(joined)) or bool(
                    re.search(r"[>»▶]+\s*\d{3,5}\b", joined)
                )
                remaining = BUDGET - (time.time() - t_ocr)
                if not has_identity and not has_id_early and remaining > 1.0:
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        futs = [pool.submit(run_variant, im) for im in variants[4:]]
                        try:
                            for fut in as_completed(futs, timeout=remaining):
                                t = fut.result()
                                if t and t.strip():
                                    texts.append(t)
                        except TimeoutError:
                            for fut in futs:
                                if fut.done() and not fut.cancelled():
                                    try:
                                        t = fut.result(timeout=0)
                                        if t and t.strip():
                                            texts.append(t)
                                    except Exception:
                                        pass
            except Exception as exc:
                print(f"[vision] OCR fallback falló: {exc}", flush=True)
        elif not has_id_early:
            # RapidOCR vio texto pero SIN toy#/▶NNN: un solo pase rápido en la
            # etiqueta inferior para recuperar el número, budget corto fijo.
            try:
                t_one = time.time()
                from PIL import Image as PILImage

                pil0 = PILImage.fromarray(np_rgb).convert("RGB")
                w0, h0 = pil0.size
                bottom = pil0.crop((0, int(h0 * 0.55), w0, h0))
                extra = _ocr_once(bottom, timeout=3)
                if extra and extra.strip():
                    texts.append(extra)
                print(
                    f"[vision] OCR bottom-recovery {int((time.time()-t_one)*1000)}ms",
                    flush=True,
                )
            except Exception as exc:
                print(f"[vision] OCR bottom-recovery falló: {exc}", flush=True)

    text = "\n".join(texts)

    # Elegir el MEJOR texto (no solo concatenar): si un pase devolvió basura
    # corta ("AE") y otro "Nissan GT-R Nismo 2024 Brilliant White Pearl",
    # el mejor manda para el matching; el resto se agrega como contexto.
    def _text_quality(t: str) -> int:
        return len(re.findall(r"[A-Za-z0-9]{2,}", t or ""))

    if texts:
        best = max(texts, key=_text_quality)
        extras = [t for t in texts if t is not best and _text_quality(t) >= 3]
        text = best + ("\n" + "\n".join(extras) if extras else "")

    # EasyOCR respaldo: SOLO si RapidOCR no dio nada (mismo criterio que
    # Tesseract). Con texto ya leído — aunque sea corto/garbage — EasyOCR
    # cuesta 2-4s y repite lo mismo.
    if not rapid_said_something and len(text.strip()) < 20:
        try:
            from PIL import Image as PILImage
            full_im = PILImage.fromarray(np_rgb).convert("RGB")
            easy_text = _easyocr_read(full_im)
            if easy_text and len(easy_text.strip()) > len(text.strip()):
                text = easy_text
                print(f"[vision] EasyOCR encontró {len(easy_text)} chars (Tesseract falló)", flush=True)
        except Exception as exc:
            print(f"[vision] EasyOCR falló: {exc}", flush=True)

    upc = None
    toy, toy_confidence = _toy_from_text(text)

    # Foto en ESPEJO (cámara frontal / flip): el texto sale al revés y RapidOCR
    # devuelve basura ("IMIM", "WINI"). Sin identidad, un pase barato sobre el
    # recorte inferior volteado. Solo si no hay toy#/▶N Y RapidOCR vio algo (texto
    # basura): sin texto alguno (imágenes IA, escenas) no aporta y cuesta ~1 s.
    if toy is None and rapid_said_something and _MIRROR_HINT_RE.search(text) and not _TOY_RE.search(text):
        try:
            from PIL import Image as PILImage, ImageOps
            from app.services.scene_ocr import read_text as scene_read

            t_flip = time.time()
            pil_m = ImageOps.mirror(PILImage.fromarray(np_rgb).convert("RGB"))
            wm, hm = pil_m.size
            mirrored = scene_read(pil_m.crop((0, int(hm * 0.55), wm, hm)))
            if mirrored and len(mirrored.strip()) >= 4:
                toy_m, conf_m = _toy_from_text(mirrored)
                if toy_m is not None:
                    toy, toy_confidence = toy_m, conf_m
                    text = text + "\n" + mirrored
            print(f"[vision] OCR espejo {int((time.time()-t_flip)*1000)}ms → {toy}", flush=True)
        except Exception as exc:
            print(f"[vision] OCR espejo falló: {type(exc).__name__}: {exc}", flush=True)
    m = _UPC_RE.search(text)
    if m:
        upc = m.group(1)
    return text[:4000], toy, toy_confidence, upc


# ── Multi-crop para varias piezas en el encuadre ────────────────────────────

def _multicrop_embed(im) -> list[VisionMatch]:
    """Embed de centro + 4 cuadrantes → candidatos individuales.

    Cuando hay varias piezas en una foto, el embed del encuadre completo es
    un promedio que no matchea ninguna. Cada crop puede aislar una pieza.
    """
    from PIL import Image

    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    w, h = im.size
    crops = [
        # centro (60% del área)
        im.crop((int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8))),
        # 4 cuadrantes
        im.crop((0, 0, w // 2, h // 2)),
        im.crop((w // 2, 0, w, h // 2)),
        im.crop((0, h // 2, w // 2, h)),
        im.crop((w // 2, h // 2, w, h)),
    ]
    results: dict[str, VisionMatch] = {}
    for crop in crops:
        emb = embed_image(crop)
        if emb is None:
            continue
        for m in match_embedding(emb, top_k=3):
            prev = results.get(m.id)
            if prev is None or m.score > prev.score:
                results[m.id] = m
    return sorted(results.values(), key=lambda x: -x.score)[:10]

def _seg_embed(im) -> tuple[np.ndarray | None, float | None, int]:
    """Embed del coche AISLADO (U²-Net-p → recorte → fondo blanco de estudio).

    El embed del encuadre completo promedia mesa/mano/sombras con la pieza; el
    índice son packshots sobre blanco. Este embed pone el query en la MISMA
    firma que el catálogo. Devuelve (vec, área de la máscara, ms). Fail-open:
    (None, None, ms) si el modelo no está o la máscara no es creíble.
    """
    import time as _t

    t = _t.time()
    try:
        from app.services.car_segment import isolate_car

        seg = isolate_car(im)
        if seg is None:
            return None, None, int((_t.time() - t) * 1000)
        vec = embed_image(seg.image)
        return vec, seg.area_frac, int((_t.time() - t) * 1000)
    except Exception as exc:  # noqa: BLE001
        print(f"[vision] seg_embed falló: {type(exc).__name__}: {exc}", flush=True)
        return None, None, int((_t.time() - t) * 1000)


def merge_seg_matches(base: list[VisionMatch], seg: list[VisionMatch]) -> tuple[list[VisionMatch], int]:
    """Fusiona candidatos del embed aislado con los del encuadre completo.

    Max por id: si el recorte ve la pieza mejor, sube su score; si no, el
    candidato del encuadre completo queda igual → no degrada lo que ya
    acierta. Devuelve (pool ordenado, nº de candidatos que mejoraron/entraron).
    """
    by_id = {m.id: m for m in base}
    improved = 0
    for m in seg:
        prev = by_id.get(m.id)
        if prev is None:
            m.source = "embed:seg"
            by_id[m.id] = m
            improved += 1
        elif m.score > prev.score:
            prev.score = m.score
            prev.source = "embed:seg"
            improved += 1
    return sorted(by_id.values(), key=lambda x: -x.score), improved


# ── Cascada principal ───────────────────────────────────────────────────────

def _yolo_embed_crops(image_bytes: bytes) -> list[VisionMatch]:
    """YOLO deshabilitado — PyTorch + TensorFlow juntos superan 3GB en Lambda.
    Gemini Vision identifica el auto del empaque sin necesidad de crop."""
    return []


_TXT_STOP = {"the", "and", "with", "ver", "version", "edition", "mini", "new", "limited",
             "exclusive", "exclusives", "set", "car", "lhd", "rhd", "shown", "series"}


def _txt_compact(text: str) -> str:
    import unicodedata

    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower().replace("gray", "grey")
    return re.sub(r"[^a-z0-9]", "", t)


def _name_text_hits(name: str, compact: str) -> tuple[float, int]:
    """(caracteres×cobertura, nº tokens) del nombre de una pieza en el texto del OCR.
    Tolera una letra perdida en palabras ≥5 («Metalic»)."""
    toks = [
        t for t in re.findall(r"[a-z0-9]+", _txt_compact_keep_spaces(name))
        if t not in _TXT_STOP and (len(t) >= 3 or (len(t) >= 2 and any(c.isdigit() for c in t)))
    ]
    if not toks or not compact:
        return 0.0, 0

    def _in(t: str) -> bool:
        return t in compact or (len(t) >= 5 and any((t[:i] + t[i + 1:]) in compact for i in range(len(t))))

    hit = [t for t in toks if _in(t)]
    tot = sum(len(t) for t in toks)
    hc = sum(len(t) for t in hit)
    return (hc * (hc / tot) if tot else 0.0), len(hit)


def _txt_compact_keep_spaces(text: str) -> str:
    import unicodedata

    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower().replace("gray", "grey")
    return re.sub(r"[^a-z0-9]+", " ", t)


def identify_image(image_bytes: bytes, brand_hint: str | None = None) -> VisionResult:
    """Corre la cascada completa sobre los bytes de la foto.

    Barcode / embeddings / OCR son independientes → se corren en paralelo.
    El cuello de botella era el embed (carga del modelo la 1ª vez); con
    `warmup()` al arrancar el server, cada identify queda en ~0.5-1s.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    t0 = time.time()
    res = VisionResult()

    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 — imagen corrupta/truncada/no-imagen
        raise ValueError(f"Imagen ilegible o corrupta: {type(exc).__name__}") from exc
    if im.width < 2 or im.height < 2:
        raise ValueError(f"Imagen demasiado pequeña: {im.width}×{im.height}")
    # acotar para que todo corra rápido (barcode/embeds no necesitan 4K)
    orig_size = im.size
    max_edge = 1024
    im_small = im
    if max(im.size) > max_edge:
        scale = max_edge / max(im.size)
        # max(1, …): un lado de 1 px con un largo de 5000 escalaba a 0 y
        # `resize((0, 2048))` tiraba el request entero (harness 2026-09-30).
        im_small = im.resize(
            (max(1, int(im.width * scale)), max(1, int(im.height * scale))),
            Image.Resampling.BILINEAR,
        )
    # OCR a resolución alta (no original): el texto pequeño del empaque
    # ("AlphaTauri AT04 #22 / Yuki Tsunoda", "▶ 849") desaparece a 1024px,
    # pero a 4000px Tesseract tarda 15-60s. 2048px es el punto dulce.
    ocr_edge = 2048
    if max(im.size) > ocr_edge:
        s = ocr_edge / max(im.size)
        im_ocr = im.resize(
            (max(1, int(im.width * s)), max(1, int(im.height * s))),
            Image.Resampling.BILINEAR,
        )
    else:
        im_ocr = im
    np_ocr = np.asarray(im_ocr)
    np_rgb = np.asarray(im_small)
    print(
        f"[vision] img bytes={len(image_bytes)} orig={orig_size} work={im_small.size}",
        flush=True,
    )
    _dbg(
        "imagen",
        f"bytes={len(image_bytes)} original={orig_size} trabajo={im_small.size} "
        f"ocr={im_ocr.size}",
    )

    # Color real de la carrocería en la foto — para no confirmar un livery
    # gemelo de otro color (foto rosa → catálogo verde). ~15 ms.
    t_col = time.time()
    res.image_colors = image_body_colors(im_small)
    _dbg(
        "color",
        f"{res.image_colors or 'señal ambigua (fondo/packshot)'} "
        f"({int((time.time() - t_col) * 1000)} ms)",
    )

    with ThreadPoolExecutor(max_workers=5) as pool:
        _dbg("pipeline", "lanzando en paralelo: barcode · embed DINOv2 · embed aislado · OCR · YOLO")
        fut_barcode = pool.submit(read_barcode, np_rgb)
        fut_embed = pool.submit(embed_image, im_small)
        # Coche suelto: el mismo embed pero del coche RECORTADO sobre blanco.
        fut_seg = pool.submit(_seg_embed, im_small)
        fut_ocr = pool.submit(read_ocr, np_ocr)
        # YOLO: detectar el auto dentro del blister → crops limpios para embed
        fut_yolo = pool.submit(_yolo_embed_crops, image_bytes)

        t = time.time()
        upc = fut_barcode.result()
        res.timings_ms["barcode"] = int((time.time() - t) * 1000)
        _dbg("barcode", f"{'UPC ' + upc if upc else 'sin código de barras'} ({res.timings_ms['barcode']} ms)")

        t = time.time()
        emb = fut_embed.result()
        res.timings_ms["embed"] = int((time.time() - t) * 1000)
        _dbg(
            "embed",
            f"{'vector OK dim=' + str(len(emb)) if emb is not None else 'embed FALLÓ'} "
            f"({res.timings_ms['embed']} ms)",
        )

        t = time.time()
        text, toy, toy_conf, ocr_upc = fut_ocr.result()
        res.timings_ms["ocr"] = int((time.time() - t) * 1000)
        _dbg(
            "ocr",
            f"toy={toy or '—'} upc={ocr_upc or '—'} conf={toy_conf} "
            f"({res.timings_ms['ocr']} ms) → {(text or '')[:160]!r}",
        )

        t = time.time()
        yolo_matches = fut_yolo.result()
        res.timings_ms["yolo"] = int((time.time() - t) * 1000)
        _dbg("yolo", f"{len(yolo_matches)} crops ({res.timings_ms['yolo']} ms)")

        seg_emb, seg_area, seg_ms = fut_seg.result()
        res.timings_ms["seg"] = seg_ms
        _dbg(
            "aislado",
            (f"coche aislado área={seg_area} → vector OK" if seg_emb is not None
             else "sin recorte creíble (packshot/blister lleno o modelo ausente)")
            + f" ({seg_ms} ms)",
        )

    t = time.time()
    dnn_matches = match_embedding(emb) if emb is not None else []
    res.timings_ms["match"] = int((time.time() - t) * 1000)
    _dbg(
        "match",
        f"{len(dnn_matches)} candidatos visuales en {res.timings_ms['match']} ms"
        + (f" · top={dnn_matches[0].name[:48]!r} {dnn_matches[0].score:.3f}" if dnn_matches else ""),
    )

    # Merge YOLO crop matches (embed de crops limpios del auto)
    for m in yolo_matches:
        if not any(existing.id == m.id for existing in dnn_matches):
            dnn_matches.append(m)
    dnn_matches.sort(key=lambda x: -x.score)

    # Embed aislado → solo cuando NO hay identidad exacta (toy#/UPC): con
    # empaque legible manda el texto y el recorte del blister no aporta.
    # NO se fusiona (max-por-id inflaba: aislado-vs-aislado puntúa alto con
    # cualquier coche sobre blanco). Se guarda para VERIFICAR al ganador de un
    # coche suelto sin texto (ver «verificación por coche aislado» abajo).
    seg_matches: list[VisionMatch] = []
    if seg_emb is not None and not (upc or toy or ocr_upc):
        # Galería Mini GT aislada (el foco de la app); si no está, el índice completo.
        seg_matches = match_mgt_gallery(seg_emb) or match_embedding(seg_emb, include_mv=True)
        for m in seg_matches:
            m.source = "embed:seg"
        _dbg(
            "aislado",
            "top aislado: " + " · ".join(f"{m.name[:28]!r} {m.score:.3f}" for m in seg_matches[:3]),
        )

    if upc:
        res.upc = upc
    res.ocr_text = text[:2000]
    if toy:
        res.toy_number = toy
    if ocr_upc and not res.upc:
        res.upc = ocr_upc

    # Mezclar candidatos TEXTUALES del catálogo completo con los visuales.
    # Sin esto, una foto IA/empaque cuyo top-K visual no contiene la pieza
    # real queda sin camino de victoria aunque el OCR lea el modelo.
    # ── Gemini Vision: si OCR no encontró identidad, leer el empaque ────
    has_identity = bool(_TOY_RE.search(text)) or bool(
        re.search(r"[>»▶]+\s*\d{3,5}\b", text) or re.search(r"\b\d{3,5}\s+[LR]HD\s+version", text, re.I)
    )
    # Gemini solo si el OCR no dio NADA útil (marca + modelo). Un "Nissan GT-R
    # Nismo 2024 White" ya identifica la pieza — llamar a Gemini costaba 21s
    # y reventaba el budget de 8s (era el 65% de la latencia total).
    useful_text = bool(re.search(r"[A-Za-z]{3,}\s+[A-Za-z0-9]{2,}", text or ""))
    # Gemini solo si el OCR no dio NADA útil Y el visual tampoco está claro.
    # Con un embed ya en 0.90+ llamar a la nube es tiempo tirado (medido en
    # prod 2026-09-30: +3.0 s en dos de cuatro fotos reales que ya acertaba).
    visual_ok = bool(dnn_matches) and dnn_matches[0].score >= 0.90
    # Gemini DESACTIVADO por defecto en el identify (2026-10-01): en prod devolvía
    # HTTP 429/503 en ~95 % de las llamadas (306 + 80 vs 21 lecturas OK) y, entre
    # la espera global de 5 s y el timeout de 3 s, costaba hasta 8 s por foto.
    # Reactivar con GEMINI_FALLBACK=1 cuando la clave tenga cuota de pago.
    # VISION_FALLBACK=claude|gemini|off (por defecto off; GEMINI_FALLBACK=1 se
    # mantiene por compatibilidad). Claude Haiku 4.5 es el recomendado.
    _fb = os.environ.get("VISION_FALLBACK") or ("gemini" if os.environ.get("GEMINI_FALLBACK") == "1" else "off")
    _gemini_on = _fb in ("claude", "gemini")
    _mc_future = None
    if _gemini_on and not has_identity and toy is None and not useful_text and not visual_ok:
        # El multicrop (CPU, ~2 s) y Gemini (red, ~2-3 s) son independientes:
        # se solapan. Solo si el multicrop va a hacer falta (visual flojo).
        if emb is not None and not visual_ok and (not dnn_matches or dnn_matches[0].score < 0.78):
            from concurrent.futures import ThreadPoolExecutor as _TPE

            _mc_pool = _TPE(max_workers=1)
            _mc_future = _mc_pool.submit(_multicrop_embed, im_small)
            _mc_pool.shutdown(wait=False)
        try:
            t_gem = time.time()
            from app.services.gemini_vision import read_packaging_with_gemini, gemini_to_ocr_text

            if _fb == "claude":
                from app.services.claude_vision import read_packaging_with_claude as _read_fb
            else:
                _read_fb = read_packaging_with_gemini
            gem_data = _read_fb(image_bytes)
            res.timings_ms["gemini"] = int((time.time() - t_gem) * 1000)
            if gem_data:
                gem_text = gemini_to_ocr_text(gem_data)
                if gem_text:
                    text = (text + "\n" + gem_text).strip()
                    res.ocr_text = text[:2000]
                    # Re-extraer toy del texto de Gemini
                    m = _TOY_RE.search(gem_text)
                    if m and not toy:
                        digits = re.sub(r"[^0-9]", "", m.group(1))
                        if digits:
                            # NO forzar prefijo MGT: solo si el texto sugiere Mini GT.
                            # Un Hot Wheels con "5785" no es MGT05785.
                            gem_brand = (gem_data or {}).get("brand") or ""
                            blob = (gem_text + " " + gem_brand).lower()
                            if any(k in blob for k in ("mini gt", "minigt", "tsm")):
                                toy = "MGT" + digits.zfill(5)[-5:]
                                toy_confidence = 0.75
                            # NO setear res.toy_number — el router lo usaría como
                            # match directo (method=o). Que el embed decida con
                            # el texto de Gemini en search_catalog_by_text.
                    print(f"[gemini] identidad: {gem_data.get('brand')} {gem_data.get('model')} toy={gem_data.get('toy_number')}", flush=True)
        except Exception as exc:
            print(f"[vision] gemini falló: {exc}", flush=True)

    # Texto de IDENTIDAD (marca/modelo/toy#) — NO usar len(text): el OCR de un
    # escritorio lee ruido ("SUPR", "P") y desactivaría bare_car. Se calcula
    # AQUÍ (tras el bloque Gemini, con `text` final) porque lo necesitan el
    # corte de MiniLM y la condición de multicrop más abajo.
    _identity_txt = bool(
        _TOY_RE.search(text)
        or re.search(r"[>»▶]+\s*\d{3,5}\b", text or "")
        or re.search(
            r"mini\s*g\s*t|hot\s*wheels|matchbox|inno64|tomica|gt-r|civic|skyline|"
            r"supra|porsche|nissan|bmw|corolla|camaro|mustang|beetle|countach",
            text or "",
            re.I,
        )
    )

    # Documento/captura (diapositiva, tabla, hoja): MUCHO texto corrido, ninguna
    # pista de diecast, sin toy#/UPC y visual flojo. Medido 2026-10-01: las
    # «imágenes IA» de 04 son capturas de análisis FODA; RapidOCR las lee (2.5 s),
    # MiniLM busca ese texto en el catálogo (2.1 s) y el multicrop corre 2 s más:
    # 8-9.5 s para llegar a «manual». Con este corte no se pierde ninguna pieza:
    # un empaque real siempre trae marca/escala/«version shown»/toy#.
    _n_words = len(re.findall(r"[A-Za-zÁ-ú]{3,}", text or ""))
    _top_vis = dnn_matches[0].score if dnn_matches else 0.0
    _doc_like = (
        _n_words >= 20
        and not _identity_txt
        and not _DIECAST_HINT_RE.search(text or "")
        and not (res.upc or res.toy_number)
        and _top_vis < 0.75
    )
    if _doc_like:
        _dbg("documento", f"texto corrido ({_n_words} palabras) sin pistas de diecast, visual {_top_vis:.2f} → se omiten MiniLM y multicrop")

    # Texto: solo si el visual NO está claro (≥0.92) o hay clúster de liveries,
    # Y además el OCR trajo texto con el que realmente se pueda comparar.
    # Medido en prod (2026-09-29): `search_catalog_by_text` cuesta 8.6-13.4 s
    # por llamada (filtro token sobre 27k nombres + MiniLM). En fotos con el
    # embed ya en 0.99 no cambia el ganador (su top salía a 0.31-0.55), y con
    # OCR basura ('YIN 00100 slamn') tampoco: no hay contra qué comparar y
    # aun así quemaba 4.4 s. El OCR sigue alimentando brand/color/livery.
    text_matches: list[VisionMatch] = []
    visual_clear = bool(dnn_matches) and dnn_matches[0].score >= 0.92
    _cluster = visual_cluster_is_ambiguous(dnn_matches)
    _txt_usable = len(re.findall(r"[A-Za-z0-9]{2,}", text or "")) >= 4
    _need_text = ((not visual_clear) or _cluster) and _txt_usable and not _doc_like
    _dbg(
        "texto",
        f"visual_clear={visual_clear} cluster={_cluster} identidad={_identity_txt} "
        f"texto_usable={_txt_usable} → "
        + (
            "MiniLM SALTADO ("
            + ("visual decisivo" if visual_clear and not _cluster else "OCR sin texto usable")
            + ", no cambia el ganador)"
            if not _need_text
            else "MiniLM se EJECUTA (visual dudoso o clúster de liveries)"
        ),
    )
    if _need_text:
        t_txt = time.time()
        text_matches = search_catalog_by_text(text, top_k=15)
        res.timings_ms["texto"] = int((time.time() - t_txt) * 1000)
        _dbg(
            "texto",
            f"{len(text_matches)} candidatos textuales en {res.timings_ms['texto']} ms"
            + (f" · top={text_matches[0].name[:48]!r} {text_matches[0].score:.3f}" if text_matches else ""),
        )
    # Top de la búsqueda textual ANTES de mezclar: al fusionar con el pool
    # visual el candidato hereda source='embed' y pierde su rastro.
    text_top: VisionMatch | None = text_matches[0] if text_matches else None
    merged: dict[str, VisionMatch] = {}
    for m in dnn_matches:
        merged[m.id] = m
    # Texto con identidad fuerte (marca + modelo) le gana al embed:
    # la foto de blister arrastra arte de packaging y el embed puntúa
    # liveries parecidos; el nombre exacto del OCR es la verdad.
    has_strong_text = bool(
        re.search(
            r"mini\s*g\s*t|hot\s*wheels|matchbox|inno64|tomica|gt-r|gt r|civic|skyline|supra",
            (text or ""),
            re.I,
        )
    ) and len(re.findall(r"[A-Za-z0-9]{2,}", text or "")) >= 4
    for m in text_matches:
        prev = merged.get(m.id)
        if prev is None:
            if has_strong_text and m.score >= 0.55:
                m.score = min(0.97, m.score * 1.35)  # empuja el nombre exacto
            merged[m.id] = m
        else:
            # acuerdo texto+visual: sube
            if has_strong_text and m.score >= 0.55:
                prev.score = min(0.99, max(prev.score, m.score * 1.2))
            else:
                prev.score = min(0.99, max(prev.score, m.score * 0.9))
    # Prior de marca: si el empaque dice HOT WHEELS, un Mini GT no gana.
    # Marca: la que diga el usuario (selector de la app) MANDA sobre la que
    # deduzca el OCR. Cada marca trae su vocabulario de identidad: MGT##### y
    # dorsal #NN en Mini GT, UPC/SKU en Hot Wheels, JAN en Tomica.
    # Si el usuario la declara es un FILTRO: se usa SOLO esa, sin diluirla con
    # lo que el OCR crea ver (si no, un "MINI GT" impreso le seguía ganando a
    # un usuario que pidió filtrar por Hot Wheels).
    ocr_hints = detect_brand_hints(text)
    declared = normalize_brand(brand_hint)
    brand_explicit = declared is not None
    brand_hints = {declared} if declared else ocr_hints
    if declared:
        _dbg(
            "marca",
            f"marca DECLARADA por el usuario: {declared!r} → filtro activo "
            f"(el OCR creía ver: {sorted(ocr_hints) or 'ninguna'})",
        )
    # El texto OCR se adjunta para el penalty de color (White vs Orange…)
    pool = list(merged.values())
    # Dorsal impreso en el coche: si el OCR lo leyó, recuperamos del catálogo a
    # los gemelos de livery que el embed dejó fuera (Roxy #80 vs Rexy #77).
    pool = inject_race_num_candidates(pool, text)
    _dbg("marca", f"hints={brand_hints or 'ninguna'} · pool={len(pool)} candidatos")
    for m in pool:
        try:
            m._ocr_text = text  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    res.matches = apply_brand_prior(
        pool, brand_hints, res.image_colors, brand_explicit=brand_explicit
    )
    res.timings_ms["total"] = int((time.time() - t0) * 1000)

    # ── Multi-crop: SOLO si el encuadre completo falló y no hay identidad ──
    # Antes corría con top<0.85 aunque el OCR ya hubiera identificado la pieza:
    # 5 embeds extra = +7s sobre el budget de 8s.
    top_score_initial = res.matches[0].score if res.matches else 0.0
    _mc_skip = _doc_like or not (
        top_score_initial < 0.78
        and emb is not None
        and not _identity_txt
        and not res.upc
        and not res.toy_number
    )
    _dbg(
        "multicrop",
        "SALTADO ("
        + (
            f"top={top_score_initial:.3f} ≥ 0.78" if top_score_initial >= 0.78
            else "hay identidad/toy/upc" if (_identity_txt or res.upc or res.toy_number)
            else "sin embed"
            if emb is None
            else "condiciones no cumplidas"
        )
        + ")"
        if _mc_skip
        else "EJECUTANDO multi-crop…",
    )
    if not _mc_skip:
        try:
            t_crop = time.time()
            crop_matches = _mc_future.result() if _mc_future is not None else _multicrop_embed(im_small)
            existing_ids = {m.id for m in res.matches}
            for m in crop_matches:
                if m.id not in existing_ids:
                    res.matches.append(m)
                    existing_ids.add(m.id)
            res.matches.sort(key=lambda x: -x.score)
            res.timings_ms["multicrop"] = int((time.time() - t_crop) * 1000)
            _dbg("multicrop", f"+{len(crop_matches)} crops en {res.timings_ms['multicrop']} ms")
        except Exception as exc:
            print(f"[vision] multicrop falló: {exc}", flush=True)
            _dbg("multicrop", f"FALLÓ: {exc}")
    res.timings_ms["total"] = int((time.time() - t0) * 1000)

    # ── Decisión ────────────────────────────────────────────────────────
    # UPC exacto → WIN (lo resuelve el router contra Mongo)
    if res.upc:
        res.method = "barcode"
        res.confidence = 0.99
        res.needs_manual = False
        return res

    # toy_number exacto → WIN (señal fuerte "MGT#####").
    # Señal débil ("▶NNN") → solo gana si el embedding o el texto dan respaldo,
    # porque el regex puede cazar números en ruido OCR.
    if res.toy_number:
        if toy_conf >= 0.90:
            res.method = "ocr"
            res.confidence = 0.95
            res.needs_manual = False
            return res
        # Señal débil: solo aceptamos si el embed CONFIRMA el toy.
        # Si el embed tiene un match fuerte con OTRO toy (ej: embed dice
        # MGT00726 con score 1.0 pero OCR garbled extrajo MGT00126),
        # el embed tiene prioridad — no forzar un match por señal débil.
        top_score = res.matches[0].score if res.matches else 0.0
        toy_in_matches = any(
            (m.toy or "").upper().replace(" ", "") == res.toy_number
            or (m.toy or "").upper().replace(" ", "") in res.toy_number
            for m in res.matches[:5]
        )
        if toy_in_matches:
            res.method = "ocr"
            res.confidence = 0.85
            res.needs_manual = False
            return res
        # Embed fuerte (≥ 0.85) pero toy NO coincide → dejar que el embed decida
        if top_score < 0.85:
            res.method = "ocr"
            res.confidence = 0.85
            res.needs_manual = False
            return res

    # Embedding → WIN solo con ganador ÚNICO.
    # Bug real: una foto del blister AlphaTauri AT04 #22 Yuki Tsunoda devolvió
    # otra pieza. Los liveries #21/#22 puntúan ~0.986 entre sí, así que un
    # score alto sin más apuesta al top1 (que a veces es el piloto equivocado).
    #  · ganador despegado del pack → 0.98+, o 0.86+ con margen 0.03
    #  · clúster de liveries (top1/top2 a <0.02) → solo gana el que el texto
    #    del empaque confirme (piloto / número / modelo).
    # Usar el pool MEZCLADO (embed + búsqueda textual) para la decisión:
    # si solo miramos dnn_matches, una pieza que el embedding no trajo pero
    # que el texto del empaque identifica (MGT00726 en la foto Gemini) no
    # puede ganar nunca.
    top = res.matches[0] if res.matches else None
    second = res.matches[1] if len(res.matches) > 1 else None
    ambiguous = visual_cluster_is_ambiguous(res.matches)
    ocr_pick = pick_ocr_confirmed_match(res.matches, text)

    # Coche SIN empaque (sin barcode, sin identidad textual): el umbral 0.86
    # era demasiado alto — fotos reales de pieza suelta puntúan 0.70–0.82.
    # `_identity_txt` ya se calculó arriba (con `text` final).
    bare_car = (not res.upc) and (not res.toy_number) and (not _identity_txt)

    # Clúster de scores casi empatados (top5 a <0.05) → nunca auto-aceptar en
    # coche suelto: son liveries gemelos del mismo casting y el embed no sabe
    # cuál es. Se calcula AQUÍ para que valga también en el FALLBACK (antes
    # solo se miraba en una rama y el fallback lo colaba igual).
    tight_cluster = False
    if bare_car and len(res.matches) >= 3:
        fifth = res.matches[min(4, len(res.matches) - 1)].score
        tight_cluster = (top.score - fifth) < 0.05

    # Contradicción TEXTO ↔ VISUAL: si el OCR nombra un modelo y el ganador
    # visual es de otra familia, alguien miente. Medido 2026-09-30 (harness):
    # una imagen IA con "ALPHATAURI #22" impreso devolvía un "Red Bull RB16B
    # #33 Verstappen" autoconfirmado al 0.81. En fotos reales el OCR suele ser
    # ruido (sin rival) o coincide con el ganador → no cambia esos casos.
    rival_text: VisionMatch | None = None
    if top is not None and text_top is not None and text_top.score >= 0.45:
        # El rival tiene que ser una PIEZA alternativa creíble, no un junk del
        # catálogo. Un "MINI GT logo" (marca pura) o un nombre de 2 palabras no
        # compite con un modelo real. Medido 2026-09-30: en un GT-R suelto el
        # OCR cazó el logo "MINI GT", el text top fue 'MINI GT logo' y la
        # guarda marcó contradicción → mandaba a manual una pieza correcta.
        rival_name = text_top.name or ""
        rival_is_piece = len(_tokens(rival_name)) >= 3
        if rival_is_piece and _name_family(rival_name) != _name_family(top.name):
            rival_text = text_top

    winner: VisionMatch | None = None
    if ambiguous:
        # Copia casi literal del studio shot del catálogo (top1 ≥ 0.99) → es
        # LA misma foto de producto; el gemelo de livery queda a 0.986 y no
        # puede ser el mismo píxel. Aceptamos.
        # ¡OJO! Tiene que ser el score CRUDO del embed: los boosts de marca y
        # dorsal inflaban un 0.92 a 1.0 y entraba por aquí fingiendo copia
        # literal (medido 2026-09-30: dos #80 rosa empatados a 1.012 y se
        # elegía uno al azar por el orden de inserción).
        top_raw = float(getattr(top, "_raw_score", top.score)) if top else 0.0
        # Empate solo cuenta entre GEMELOS del mismo casting (mismo modelo con
        # distinto livery). Un Corvette 0.910 vs un Huracán 0.907 no es un
        # empate de liveries: son coches distintos y el top puede ganar.
        tied_top = bool(
            top is not None
            and second is not None
            and abs(top.score - second.score) < 0.005
            and _name_family(top.name) == _name_family(second.name)
        )
        if top and top_raw >= 0.99:
            winner = top
        elif tied_top:
            # Empate real entre candidatos y sin copia literal que lo zanje:
            # no elegimos a suertes. El usuario decide entre las sugerencias.
            _dbg(
                "decisión",
                f"EMPATE entre candidatos ({top.score:.3f} vs {second.score:.3f}) "
                f"sin copia literal (raw={top_raw:.3f}) → manual",
            )
        # El texto del empaque desambigua… pero SOLO si no le gana un
        # candidato visualmente mucho mejor (max score manda).
        elif ocr_pick is not None and (
            not top or ocr_pick.score >= top.score * 0.7
        ):
            if top is None or ocr_pick.score >= top.score:
                winner = ocr_pick
            else:
                winner = top
        elif top and top.score >= 0.85:
            winner = top
        elif bare_car and top and top.score >= 0.68:
            # Cluster de variantes del mismo coche (0.747/0.737/0.728): el
            # emparejado top es correcto aunque esté apretado — aceptar y
            # dejar las sugerencias para que el usuario cambie de variante.
            winner = top
        elif top and top.score >= 0.65 and _identity_txt:
            # Cluster + OCR con identidad ("MINI Honda Civic Type"): el texto
            # confirma el coche aunque el score visual sea medio.
            if score_name_against_ocr_text(top.name, text) >= 0.45:
                winner = top
        # bare car + cluster apretado: NO auto-aceptar (falsos positivos)
    elif top:
        margin = (top.score - second.score) if second else 1.0
        # Si top1 y top2 son el MISMO MODELO (comparten prefijo), el embed está
        # seguro del coche pero no de la variante → aceptar con umbral menor.
        # Ej: "Toyota GR86 Larry Chen's HKS" (MGT00820) vs "...2024 LADC" (MGT00821)
        same_model = False
        if second:
            w1 = top.name.lower().split()[:3]
            w2 = second.name.lower().split()[:3]
            same_model = w1 == w2 and len(w1) >= 2
        # cluster de scores casi empatados (top5 a <0.05) → nunca auto-aceptar
        # en coche suelto: son formas genéricas que matchean cualquier 1:64.
        # (`tight_cluster` ya se calculó arriba, para que valga en el fallback.)
        # Mismo modelo entre top3 (p.ej. 3 Civic Type R): el embed acierta el
        # coche pero no la variante de color/SKU — aceptar el top es razonable
        # y MUCHO mejor que "no identificado".
        same_family = False
        if bare_car and len(res.matches) >= 2:
            def _fw(n: str) -> tuple:
                return tuple((n or "").lower().split()[:2])
            # top1 vs top2 comparten familia ("Honda Civic" vs "Honda Civic")
            same_family = _fw(top.name) == _fw(second.name) if second else False
        # Sin identidad (toy#/UPC/texto de marca) y sin copia literal del
        # studio shot (≥0.98) exigimos margen AMPLIO: el embed puede estar
        # seguro del casting y equivocarse de variante, o inventar sobre algo
        # que ni es un diecast. Medido 2026-09-30 (harness): una imagen IA
        # autoconfirmó MGT00799 (Bull RB16B) y dos ángulos confundieron
        # cross-variante con needs_manual=false — peor que un manual honesto.
        #
        # PERO si el top despega (≥0.88) y los rivales son GEMELOS del mismo
        # casting, el margen apretado es lo esperado (son liveries) y el
        # usuario ya puede cambiar de variante con las sugerencias. Medido
        # 2026-09-30: un GT-R "Light Blue" 0.899 con un "Satin Gold" 0.858 de
        # segundo se iba a manual pese a que la correcta era la nº 1.
        margin_needed = 0.03 if not bare_car else 0.08
        if (
            top.score >= 0.98
            or (top.score >= 0.86 and margin >= margin_needed)
            or (same_model and top.score >= 0.82 and margin >= max(0.01, margin_needed * 0.5))
        ):
            # EXCEPCIÓN: bare car + clúster apretado = liveries gemelos del
            # mismo casting. Medido 2026-09-29: foto ROSA de un Porsche 911 GT3 R
            # devolvía el VERDE #77 al 94.8 % porque top5 iban 0.948/0.914/0.910/
            # 0.907/0.906 — el margen top2 (0.034) engañaba al umbral pero el
            # coche no está claro. Sin texto de identidad, no auto-aceptamos:
            # mejor manual + sugerencias que confirmar el color equivocado.
            if tight_cluster and top.score < 0.98 and not _identity_txt:
                _dbg(
                    "decisión",
                    f"CLÚSTER APRETADO (top5 en {top.score - res.matches[min(4, len(res.matches)-1)].score:.3f}) "
                    f"sin identidad → manual, el color/livery no está claro",
                )
            else:
                winner = top
        elif bare_car and top.score >= 0.68 and not tight_cluster:
            # Pieza suelta sin texto: las fotos reales de escritorio puntúan
            # 0.68–0.80. Pedir 0.88+ dejaba TODO en "no identificado".
            # Se acepta con la confianza real (0.70 ≠ 0.99) y el top-8 de
            # sugerencias permite corregir con un tap.
            winner = top
        elif bare_car and same_family and top.score >= 0.65:
            winner = top
        elif top.score >= 0.65 and _identity_txt:
            # El OCR vio identidad ("MINI Honda Civic Type") y el top visual
            # es consistente con ese texto → aceptar aunque el score visual
            # sea medio (foto de escritorio 0.70-0.80, no studio shot).
            conf_txt = score_name_against_ocr_text(top.name, text)
            if conf_txt >= 0.50:
                winner = top
        elif ocr_pick is not None and ocr_pick.score >= top.score * 0.7:
            # Ganador = el MAYOR score. El texto confirmado solo desempata
            # cuando está a la par; no le gana a un candidato claramente mejor
            # (antes un RC genérico a 0.88 le ganaba a White Nismo a 0.97).
            if ocr_pick.score >= top.score:
                winner = ocr_pick
            else:
                winner = top
        elif top.score >= 0.85:
            winner = top

    if winner is not None and rival_text is not None and winner.score < 0.95:
        _dbg(
            "decisión",
            f"CONTRADICCIÓN texto↔visual: OCR dice {rival_text.name[:42]!r} "
            f"pero el visual es {winner.name[:42]!r} → manual",
        )
        winner = None

    # ── Verificación por COCHE AISLADO (solo coche suelto sin texto) ───────
    # Medido 2026-10-05 con 77 sueltos de verdad conocida: cuando el encuadre
    # completo y el coche aislado (índice multivista) coinciden, aciertan 24/25.
    # Cuando difieren, el encuadre completo acierta en fotos sintéticas pero
    # 0/7 en sueltos REALES del usuario (casaba GT3 genéricos a 0.85-0.89).
    # Política: coinciden → se confirma; el aislado es copia literal de una
    # foto de galería (≥0.99) → gana el aislado; difieren → manual, con ambos
    # candidatos al frente de las sugerencias.
    # ── Desempate por TEXTO entre ambas búsquedas (sin toy#/UPC) ───────────
    # El OCR leyó algo («BMW Z8 Sver», «NISSAN MOTUL AUTECH Nism…») pero no
    # un número: entre los candidatos del encuadre completo y del coche
    # aislado, gana el que el texto nombra con claridad. Medido 2026-10-05: el
    # aislado traía «BMW Z8 Silver» 1º y ganaba un «Dodge Concept Car» que el
    # texto no menciona; el GT-R MOTUL estaba 5º en el aislado.
    text_pick = False
    compact = _txt_compact(text)
    if not (res.upc or res.toy_number) and seg_matches and len(compact) >= 4:
        pool: list[VisionMatch] = []
        for m in res.matches[:8] + seg_matches[:8]:
            if not any(m.id == p.id for p in pool):
                pool.append(m)
        scored = sorted(
            ((_name_text_hits(m.name, compact), m) for m in pool),
            key=lambda x: (x[0][0], x[1].score),
            reverse=True,
        )
        if scored:
            (bs, bn), best = scored[0]
            second_s = scored[1][0][0] if len(scored) > 1 else 0.0
            cur = winner or top
            cur_s = _name_text_hits(cur.name, compact)[0] if cur else 0.0
            if bn >= 2 and bs - second_s >= 3 and bs - cur_s >= 3:
                winner = best
                text_pick = True
                res.matches = [best] + [m for m in pool if m.id != best.id]
                _dbg("texto", f"desempate por texto entre visual y aislado → {best.name[:44]!r}")

    seg_veto = False
    # Verificar: coche suelto sin texto, o hay texto pero NO respalda al ganador.
    _cand = winner or top
    _unbacked = (
        not (res.upc or res.toy_number)
        and _cand is not None
        and _name_text_hits(_cand.name, compact)[1] == 0
    )
    if (bare_car or _unbacked) and seg_matches and not text_pick:
        st = seg_matches[0]
        # piezas que comparten la MISMA foto de galería (-L/-R, gemelos) empatan
        near = [m for m in seg_matches if m.score >= st.score - 0.002]
        cand = winner or top

        def _same_piece(a: VisionMatch, b: VisionMatch) -> bool:
            if a.id == b.id:
                return True
            ta, tb = (a.toy or "").upper(), (b.toy or "").upper()
            return bool(ta) and bool(tb) and ta[:8] == tb[:8]

        agree = cand is not None and any(_same_piece(cand, m) for m in near)
        if agree:
            _dbg("aislado", f"coincide con el visual → {cand.name[:40]!r}")
        elif st.score >= 0.99 and len(near) == 1:
            winner = st
            res.matches = [st] + [m for m in res.matches if not _same_piece(m, st)]
            _dbg("aislado", f"copia literal de galería ({st.score:.3f}) → {st.name[:40]!r}")
        else:
            seg_veto = True
            winner = None
            # Las puntuaciones del aislado (aislado-vs-aislado) salen infladas y
            # el router re-ordena por score: se desplazan para que el top del
            # aislado quede JUSTO DEBAJO del top visual y las sugerencias se
            # intercalen (si no, el ganador visual desaparecía del top-8).
            # Orden de sugerencias: top visual, luego las 3 primeras del aislado
            # (en sueltos reales la verdad estaba en el top-4 del aislado 6/7).
            # Las puntuaciones se escalonan justo por debajo del top visual
            # porque el router re-ordena por score.
            head = [m for m in (top, *seg_matches[:3]) if m is not None]
            if top is not None:
                # Las 3 primeras del aislado van JUSTO detrás del top visual (aunque
                # su similitud cruda sea menor: son dos espacios distintos y en
                # sueltos reales el aislado acertaba más). El resto, por debajo.
                for k, m in enumerate(seg_matches, 1):
                    m.score = top.score - 0.001 * k if k <= 3 else min(m.score, top.score - 0.001 * k)
                floor = top.score - 0.004
                for m in res.matches:
                    if m is not top and not any(m.id == x.id for x in seg_matches[:3]):
                        m.score = min(m.score, floor - 0.0001)
            rest = [m for m in res.matches + seg_matches[1:] if not any(_same_piece(m, h) for h in head)]
            dedup: list[VisionMatch] = []
            for m in head + rest:
                if not any(_same_piece(m, d) for d in dedup):
                    dedup.append(m)
            res.matches = dedup
            _dbg(
                "aislado",
                f"NO coincide: visual {top.name[:30] if top else '—'!r} vs aislado {st.name[:30]!r} "
                f"({st.score:.3f}) → manual con ambos al frente",
            )

    if winner is not None:
        # El router toma matches[0] — ponemos al ganador (OCR si hubo clúster).
        res.matches = [winner] + [m for m in res.matches if m.id != winner.id]
        res.method = "embed"
        res.confidence = float(min(0.98, max(0.7, winner.score)))
        # Guarda multi-marca SOLO si el empaque nombra explícitamente otra
        # marca (p.ej. "HOT WHEELS" en un MiniGT) y el score es justo.
        # Con score alto (≥0.90) confiamos en el visual aunque el OCR sea ruidoso.
        winner_brand = _match_brand(winner)
        if brand_hints and winner_brand and winner_brand not in brand_hints:
            if winner.score < 0.90:
                res.method = "manual"
                res.confidence = float(max(0.0, min(1.0, winner.score)))
                res.needs_manual = True
                _dbg("decisión", f"RECHAZO multi-marca: {winner.name!r} no coincide con hints={brand_hints}")
                return res
        res.needs_manual = False
        _dbg(
            "decisión",
            f"GANADOR method=embed conf={res.confidence:.3f} · {winner.name!r} ({winner.toy or '—'})",
        )
        return res

    # Fallback: ganador decente sin camino formal — mejor el top con confianza
    # real que "no identificado". El top-8 de sugerencias permite corregir.
    # PERO con clúster apretado en coche suelto el top NO está claro: aquí se
    # colaba el bug (la guarda solo vivía en la rama de arriba) y un livery
    # gemelo de otro color terminaba auto-confirmado.
    if top and top.score >= 0.72 and not seg_veto:
        if tight_cluster and top.score < 0.98 and not _identity_txt:
            _dbg(
                "decisión",
                f"FALLBACK BLOQUEADO por clúster apretado (top={top.score:.3f}) → manual",
            )
        elif rival_text is not None and top.score < 0.95:
            _dbg(
                "decisión",
                f"FALLBACK BLOQUEADO por contradicción texto↔visual ({rival_text.name[:34]!r}) → manual",
            )
        else:
            res.matches = [top] + [m for m in res.matches if m.id != top.id]
            res.method = "embed"
            res.confidence = float(min(0.97, max(0.7, top.score)))
            # Sin identidad y sin score ALTO DEL EMBED no aseguramos la pieza:
            # la devolvemos como sugerencia. Hay que mirar el score crudo, no
            # el inflado por boosts de marca/dorsal (0.92 → 1.0 fingía alto).
            top_raw_fb = float(getattr(top, "_raw_score", top.score))
            res.needs_manual = bool(bare_car and top_raw_fb < 0.90)
            _dbg(
                "decisión",
                f"FALLBACK top conf={res.confidence:.3f} raw={top_raw_fb:.3f} · "
                f"{top.name!r} · manual={res.needs_manual}",
            )
            return res

    # Sin match confiable → sugerencias, nunca inventamos.
    res.method = "manual"
    res.confidence = float(max(0.0, min(1.0, top.score))) if top else 0.0
    res.needs_manual = True
    _dbg(
        "decisión",
        f"SIN MATCH confiable → manual (top={top.score:.3f} {top.name!r})" if top else "SIN candidatos → manual",
    )
    return res


def warmup() -> None:
    """Carga modelo + índice al arrancar. EasyOCR/YOLO cargan lazy
    (son muy pesados para el init phase de Lambda que solo da 10s)."""
    try:
        get_model()
        get_index()
        # un forward de calentamiento (XNNPACK/oneDNN inicializan lazy)
        import numpy as np
        from PIL import Image

        dummy = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        embed_image(dummy)
    except Exception as exc:  # no tumbar el arranque por un warmup fallido
        print(f"[vision] warmup falló: {exc}")
    try:
        from app.services import car_segment

        car_segment.warmup()
    except Exception as exc:  # noqa: BLE001
        print(f"[vision] warmup segmentación falló: {exc}", flush=True)
    try:
        # MiniLM cargaba en FRÍO dentro del 1º request que lo necesitaba:
        # +27 s de golpe (medido en prod 2026-09-29). Mejor pagarlos al arrancar.
        from app.services import text_match

        text_match._ensure()
        print("[vision] warmup MiniLM OK", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[vision] warmup MiniLM falló: {exc}", flush=True)
    try:
        # TrOCR (OCR de rescate cuando RapidOCR no ve texto) cargaba en FRÍO en
        # el 1er request sin texto: ocr=14 s (medido 2026-10-01). Se paga aquí.
        from app.services import trocr_ocr

        trocr_ocr._ensure_model()
        print("[vision] warmup TrOCR OK", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[vision] warmup TrOCR falló: {exc}", flush=True)
    try:
        _mgt_gallery()
        _index_names()
    except Exception as exc:  # noqa: BLE001
        print(f"[vision] warmup galería Mini GT falló: {exc}", flush=True)
