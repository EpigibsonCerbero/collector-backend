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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── Rutas ────────────────────────────────────────────────────────────────────
# backend/app/services/vision.py → parents[3] = raíz del repo (collector-manager)
ROOT = Path(__file__).resolve().parents[3]
EMB_PATH = Path(
    os.environ.get(
        "EMB_INDEX_PATH", str(ROOT / "backend" / "data" / "minigt-embeddings-full.json")
    )
)

IMG_SIZE = 224
EMB_DIM = 576

_lock = threading.Lock()
_model = None
_index: list[dict] | None = None
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


def get_index() -> tuple[list[dict], np.ndarray]:
    global _index, _matrix
    if _index is None:
        with _lock:
            if _index is None:
                entries = json.loads(EMB_PATH.read_text(encoding="utf-8"))
                vecs = np.array([e["vec"] for e in entries], dtype=np.float32)
                # L2-normalize por si el precompute no lo hizo
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                _matrix = vecs / norms
                _index = entries
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


# ── Desambiguación de liveries (bug AlphaTauri #21 vs #22) ─────────────────

def _tokens(text: str) -> set[str]:
    return {t for t in re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split() if len(t) >= 3}


def _token_weight(tok: str) -> float:
    """Los tokens cortos/números ('22', 'gp', 'f1') son ambiguos entre coches
    del mismo piloto; los largos ('alphatauri', 'tsunoda', 'australian') son
    los que identifican la pieza."""
    return 1.0 if len(tok) >= 5 else 0.55


def visual_cluster_is_ambiguous(matches: list[VisionMatch], window: float = 0.02) -> bool:
    """¿top1/top2 casi empatados (liveries gemelos)? El embedding no decide."""
    if len(matches) < 2:
        return False
    s0 = matches[0].score
    return any(s0 - m.score < window for m in matches[1:])


def score_name_against_ocr_text(name: str | None, ocr_text: str) -> float:
    """0..1 — cuánto confirma el texto del empaque a un candidato del catálogo.

    Tokens ponderados: "AlphaTauri AT04 #22 Yuki Tsunoda" debe ganar a
    "RB VCARB 01 #22 Yuki Tsunoda" aunque compartan el piloto y el número.

    Matching difuso: OCR de imágenes IA lee "AT04" como "ATB", "2023" como
    "2029". Un token exacto fallaría; con similitud >= 0.65 contamos medio hit.
    """
    import difflib

    text = (ocr_text or "").lower()
    if not text or not name:
        return 0.0
    name_tokens = _tokens(name)
    if not name_tokens:
        return 0.0
    text_tokens = _tokens(text)
    total = sum(_token_weight(t) for t in name_tokens)
    hit = 0.0
    for t in name_tokens:
        w = _token_weight(t)
        if t in text_tokens:
            hit += w
        else:
            # fuzzy: ¿algún token del OCR se parece a este?
            for tt in text_tokens:
                ratio = difflib.SequenceMatcher(None, t, tt).ratio()
                if ratio >= 0.65:
                    hit += w * 0.6  # hit parcial por similitud
                    break
    return hit / total if total else 0.0


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
    if rest and s1 - rest[0][1] < min_margin:
        return None
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
    try:
        model = get_model()
        x = preprocess(im)
        vec = model.predict(x, verbose=0)[0]
        return l2_normalize(vec.astype(np.float32))
    except Exception:
        return None


def match_embedding(emb: np.ndarray, top_k: int = 8) -> list[VisionMatch]:
    index, matrix = get_index()
    sims = matrix @ emb
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


def search_catalog_by_text(ocr_text: str, top_k: int = 5) -> list[VisionMatch]:
    """Búsqueda textual en TODO el catálogo por tokens del OCR.

    Necesaria cuando el embedding falla (foto IA, blister, ángulo raro): el
    top-K visual puede no contener la pieza real, pero el texto del empaque
    ("ALPHATAURI", "AT04 #22", "Tsunoda") sí la identifica en el catálogo.
    """
    text = (ocr_text or "").lower()
    if len(text.strip()) < 4:
        return []
    q = _tokens(text)
    if not q:
        return []
    index, _ = get_index()
    scored: list[VisionMatch] = []
    for e in index:
        name = e.get("name") or ""
        hay = _tokens(f"{name} {e.get('toy') or ''}")
        if not hay:
            continue
        inter = len(q & hay)
        if inter == 0:
            continue
        # cubre mejor = mayor score; penaliza nombres que matchean poco
        score = inter / max(len(q), 1) * 0.7 + inter / max(len(hay), 1) * 0.3
        scored.append(
            VisionMatch(
                id=e["id"],
                toy=e.get("toy"),
                name=name,
                score=float(score),
                source="ocr",
            )
        )
    scored.sort(key=lambda m: -m.score)
    return scored[:top_k]


# ── 3) OCR ──────────────────────────────────────────────────────────────────

# OCR lee "MGT01160" como "MGTO1160" (letra O por cero) — toleramos ambos.
_TOY_RE = re.compile(r"\b(MGT\s?[O0]?\s?\d{3,5})\b", re.I)
_UPC_RE = re.compile(r"\b(\d{12,14})\b")


def _ocr_once(im) -> str:
    """Un pase de Tesseract (eng+spa) sobre una imagen PIL."""
    import pytesseract

    return pytesseract.image_to_string(im, lang="eng+spa") or ""


# Techo de píxeles para las variantes OCR — sin esto, una foto de 4000px
# escalada 2.5x genera un lienzo de 10000px y Tesseract tarda 15-30s.
_OCR_MAX_EDGE = 3200


def _ocr_variants(np_rgb: np.ndarray) -> list:
    """Variantes de la imagen donde el texto del empaque suele sobrevivir.

    Empaques Mini GT: tarjeta negra con texto blanco + etiqueta pequeña abajo
    con "AlphaTauri AT04 #22 / Yuki Tsunoda". Un solo pase pierde texto.
    """
    from PIL import Image, ImageOps

    base = Image.fromarray(np_rgb).convert("RGB")
    w, h = base.size
    variants = []

    def scaled(img, factor):
        tw, th = int(img.width * factor), int(img.height * factor)
        # No escalar más allá del techo: Tesseract no gana nada con >3200px
        m = max(tw, th)
        if m > _OCR_MAX_EDGE:
            s = _OCR_MAX_EDGE / m
            tw, th = int(tw * s), int(th * s)
        return img.resize((tw, th), Image.Resampling.LANCZOS)

    # 1) completa + invertida (blanco sobre negro)
    variants.append(scaled(base, 2.0))
    variants.append(ImageOps.invert(scaled(base, 2.0)))

    # 2) mitades superior/inferior — la etiqueta del modelo suele estar abajo
    top = base.crop((0, 0, w, h // 2))
    bot = base.crop((0, h // 2, w, h))
    variants.append(scaled(bot, 2.5))
    variants.append(ImageOps.invert(scaled(bot, 2.5)))
    variants.append(scaled(top, 2.0))

    # 3) tercio central + invertido (donde va la tarjeta de arte)
    mid = base.crop((0, h // 3, w, 2 * h // 3))
    variants.append(scaled(mid, 2.5))

    # 4) baja resolución forzada (a veces Tesseract prefiere menos blur de JPEG)
    variants.append(base.resize((max(200, w // 2), max(200, h // 2)), Image.Resampling.BILINEAR))
    return variants


def read_ocr(np_rgb: np.ndarray) -> tuple[str, str | None, float, str | None]:
    """Devuelve (texto_crudo, toy_number, toy_confidence, upc).

    Multi-recorte adaptativo. El texto de IDENTIDAD de los empaques Mini GT
    vive abajo ("AlphaTauri AT04 #22 / Yuki Tsunoda", "▶ 726") y el branding
    grande ("SCUDERIA / ALPHATAURI") es solo marca — confundir uno con otro
    dejaba el matching por texto sin pieza posible. Por eso el recorte
    inferior SIEMPRE corre; los extras solo si aún falta señal.

    Se extrae también el indicador "▶ 726" de los empaques Mini GT: el
    sufijo numérico de MGT00726. Es la señal más fiable de identidad.

    Variantes en paralelo con time budget de 5s: si Tesseract se lenta,
    nos quedamos con lo leído hasta ahora.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    texts: list[str] = []
    try:
        variants = _ocr_variants(np_rgb)
        t_ocr = time.time()
        BUDGET = 5.0  # segundos máximos para todo el OCR

        def run_variant(im):
            try:
                return _ocr_once(im)
            except Exception:
                return ""

        # Lote 1: completa + invertida + MITAD INFERIOR (+invertida).
        # La etiqueta de identidad va abajo; si solo miramos la tarjeta entera
        # se queda con el branding y no con el modelo/piloto/toy#.
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(run_variant, im) for im in variants[:4]]
            for fut in as_completed(futs, timeout=BUDGET):
                t = fut.result()
                if t and t.strip():
                    texts.append(t)

        joined = "\n".join(texts)
        # ¿ya hay señal de IDENTIDAD (toy# / indicador ▶NNN / número de coche)?
        # El branding ("ALPHATAURI") NO cuenta: está en mil piezas distintas.
        has_identity = bool(_TOY_RE.search(joined)) or bool(
            re.search(r"[>»▶]+\s*\d{3,5}\b", joined)
        )
        remaining = BUDGET - (time.time() - t_ocr)
        if not has_identity and remaining > 1.0:
            with ThreadPoolExecutor(max_workers=3) as pool:
                futs = [pool.submit(run_variant, im) for im in variants[4:]]
                try:
                    for fut in as_completed(futs, timeout=remaining):
                        t = fut.result()
                        if t and t.strip():
                            texts.append(t)
                except TimeoutError:
                    pass  # nos quedamos con lo que haya
    except Exception as exc:
        print(f"[vision] OCR falló: {exc}", flush=True)
        return "", None, None

    text = "\n".join(texts)
    toy = None
    toy_confidence = 0.0
    upc = None
    m = _TOY_RE.search(text)
    if m:
        raw = re.sub(r"\s+", "", m.group(1)).upper()
        # OCR lee "MGTO1160" (letra O por cero) → extraer solo dígitos y normalizar
        digits = re.sub(r"[^0-9]", "", raw)  # MGTO1160 → 1160
        if digits:
            toy = "MGT" + digits.zfill(5)[-5:]
            toy_confidence = 0.95
    else:
        # Empaque Mini GT: "▶ 726" / "> 726" es el sufijo de MGT00726.
        # Señal más débil: puede salir de ruido OCR, así que confianza menor.
        m = re.search(r"[>»▶]+\s*(\d{3,5})\b", text)
        if m:
            toy = "MGT" + m.group(1).zfill(5)[-5:]
            toy_confidence = 0.80
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

# ── Cascada principal ───────────────────────────────────────────────────────

def identify_image(image_bytes: bytes) -> VisionResult:
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

    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # acotar para que todo corra rápido (barcode/embeds no necesitan 4K)
    orig_size = im.size
    max_edge = 1024
    im_small = im
    if max(im.size) > max_edge:
        scale = max_edge / max(im.size)
        im_small = im.resize(
            (int(im.width * scale), int(im.height * scale)), Image.Resampling.BILINEAR
        )
    # OCR a resolución alta (no original): el texto pequeño del empaque
    # ("AlphaTauri AT04 #22 / Yuki Tsunoda", "▶ 849") desaparece a 1024px,
    # pero a 4000px Tesseract tarda 15-60s. 2048px es el punto dulce.
    ocr_edge = 2048
    if max(im.size) > ocr_edge:
        s = ocr_edge / max(im.size)
        im_ocr = im.resize(
            (int(im.width * s), int(im.height * s)), Image.Resampling.BILINEAR
        )
    else:
        im_ocr = im
    np_ocr = np.asarray(im_ocr)
    np_rgb = np.asarray(im_small)
    print(
        f"[vision] img bytes={len(image_bytes)} orig={orig_size} work={im_small.size}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_barcode = pool.submit(read_barcode, np_rgb)
        fut_embed = pool.submit(embed_image, im_small)
        fut_ocr = pool.submit(read_ocr, np_ocr)

        t = time.time()
        upc = fut_barcode.result()
        res.timings_ms["barcode"] = int((time.time() - t) * 1000)

        t = time.time()
        emb = fut_embed.result()
        res.timings_ms["embed"] = int((time.time() - t) * 1000)

        t = time.time()
        text, toy, toy_conf, ocr_upc = fut_ocr.result()
        res.timings_ms["ocr"] = int((time.time() - t) * 1000)

    dnn_matches = match_embedding(emb) if emb is not None else []

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
    text_matches = search_catalog_by_text(text)
    merged: dict[str, VisionMatch] = {}
    for m in dnn_matches:
        merged[m.id] = m
    for m in text_matches:
        prev = merged.get(m.id)
        if prev is None:
            merged[m.id] = m
        else:
            # conservar el score visual si existe (más fiable), pero registrar
            # que hubo acuerdo texto+visual elevando un poco
            prev.score = min(0.99, max(prev.score, m.score * 0.9))
    res.matches = sorted(merged.values(), key=lambda x: -x.score)
    res.timings_ms["total"] = int((time.time() - t0) * 1000)

    # ── Multi-crop: varias piezas en el encuadre ─────────────────────────
    # Si el embed del encuadre completo da score mediocre (< 0.85), puede
    # haber varias piezas y el promedio no matchea ninguna. Probamos crops
    # estratégicos (centro + 4 cuadrantes) y agregamos lo que encuentre.
    top_score_initial = res.matches[0].score if res.matches else 0.0
    if top_score_initial < 0.85 and emb is not None:
        try:
            t_crop = time.time()
            crop_matches = _multicrop_embed(im_small)
            # Agregar candidatos de crops que no estén ya en el pool
            existing_ids = {m.id for m in res.matches}
            for m in crop_matches:
                if m.id not in existing_ids:
                    res.matches.append(m)
                    existing_ids.add(m.id)
            res.matches.sort(key=lambda x: -x.score)
            res.timings_ms["multicrop"] = int((time.time() - t_crop) * 1000)
        except Exception as exc:
            print(f"[vision] multicrop falló: {exc}", flush=True)
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
        # Señal débil: ¿el embedding al menos apunta a un candidato cercano?
        top_score = res.matches[0].score if res.matches else 0.0
        toy_in_matches = any(
            (m.toy or "").upper().replace(" ", "") == res.toy_number
            or (m.toy or "").upper().replace(" ", "") in res.toy_number
            for m in res.matches[:5]
        )
        if toy_in_matches or top_score >= 0.75:
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

    winner: VisionMatch | None = None
    if ambiguous:
        # Copia casi literal del studio shot del catálogo (top1 ≥ 0.99) → es
        # LA misma foto de producto; el gemelo de livery queda a 0.986 y no
        # puede ser el mismo píxel. Aceptamos.
        if top and top.score >= 0.99:
            winner = top
        # El texto del empaque ya desambiguó (pick_ocr_confirmed_match tiene
        # sus propios umbrales de score + margen) — confiamos en él aunque el
        # score visual sea mediocre (foto IA / blister / ángulo raro).
        elif ocr_pick is not None:
            winner = ocr_pick
    elif top:
        margin = (top.score - second.score) if second else 1.0
        if top.score >= 0.98 or (top.score >= 0.86 and margin >= 0.03):
            winner = top
        elif ocr_pick is not None:
            winner = ocr_pick

    if winner is not None:
        # El router toma matches[0] — ponemos al ganador (OCR si hubo clúster).
        res.matches = [winner] + [m for m in dnn_matches if m.id != winner.id]
        res.method = "embed"
        res.confidence = float(min(0.98, max(0.7, winner.score)))
        # Si el embed score es muy bajo, verificar que el texto tenga señal de
        # IDENTIDAD real (marca+modelo, #NN, toy#) y no solo branding genérico.
        # "MINI GT logo" matcheado por "MINI GT" watermark → falso positivo.
        # "AlphaTauri AT04 #22" en imagen IA → identidad real, aceptar.
        if winner.score < 0.5 and winner is ocr_pick:
            has_identity_text = bool(
                re.search(r"#\s*\d{1,3}\b", text)  # "#22" — número de coche
                or _TOY_RE.search(text)  # "MGT00726"
                or re.search(r"[>»▶]+\s*\d{3,5}\b", text)  # "▶ 726"
            )
            if not has_identity_text:
                res.method = "manual"
                res.confidence = float(winner.score)
                res.needs_manual = True
                return res
        res.needs_manual = False
        return res

    # Sin match confiable → sugerencias, nunca inventamos.
    res.method = "manual"
    res.confidence = float(top.score) if top else 0.0
    res.needs_manual = True
    return res


def warmup() -> None:
    """Carga modelo + índice al arrancar, para que el 1º identify no pague la carga."""
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
