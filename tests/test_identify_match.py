"""Regresión del bug AlphaTauri AT04 #22 — desambiguación de liveries.

Foto del blister "#22 Yuki Tsunoda 2023 Australian GP" devolvía otra pieza:
los embeddings de #21/#22 empatan a ~0.986 y el top1 por score salía mal.
El texto del empaque es lo único que discrimina piloto/número.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.vision import (  # noqa: E402
    VisionMatch,
    pick_ocr_confirmed_match,
    score_name_against_ocr_text,
    visual_cluster_is_ambiguous,
)

NAME_22 = "AlphaTauri AT04 #22 Yuki Tsunoda  2023 F1  2023 Australian GP"
NAME_21 = "AlphaTauri AT04 #21 Nyck de Vries  2023 F1  2023 Australian GP"

OCR_PACKAGING = (
    "MINI GT SCUDERIA ALPHATAURI 22 AlphaTauri AT04 #22 "
    "2023 F1 2023 Australian GP Yuki Tsunoda"
)


def _m(mid: str, name: str, score: float) -> VisionMatch:
    return VisionMatch(id=mid, toy=None, name=name, score=score, source="embed")


class TestCluster:
    def test_livery_twins_are_ambiguous(self):
        ranking = [_m("mgt00727", NAME_21, 0.991), _m("mgt00726", NAME_22, 0.986)]
        assert visual_cluster_is_ambiguous(ranking) is True

    def test_clear_winner_is_not_ambiguous(self):
        ranking = [_m("a", "Ford GT", 0.99), _m("b", "Porsche 911", 0.91)]
        assert visual_cluster_is_ambiguous(ranking) is False

    def test_single_candidate_is_not_ambiguous(self):
        assert visual_cluster_is_ambiguous([_m("a", "x", 0.99)]) is False


class TestOcrScore:
    def test_packaging_text_confirms_22(self):
        assert score_name_against_ocr_text(NAME_22, OCR_PACKAGING) > 0.7

    def test_22_beats_21_on_the_same_text(self):
        s22 = score_name_against_ocr_text(NAME_22, OCR_PACKAGING)
        s21 = score_name_against_ocr_text(NAME_21, OCR_PACKAGING)
        assert s22 > s21
        assert s22 - s21 > 0.15

    def test_empty_text_scores_zero(self):
        assert score_name_against_ocr_text(NAME_22, "") == 0.0


class TestPickConfirmed:
    # El embedding pone #21 arriba por 0.005 — el texto debe corregirlo.
    RANKING = [
        _m("mgt00727", NAME_21, 0.991),
        _m("mgt00726", NAME_22, 0.986),
        _m("mgt01007", "RB VCARB 01 #22 Yuki Tsunoda  2024 F1 2024 Bahrain GP", 0.91),
    ]

    def test_packaging_text_picks_22_over_embed_top1(self):
        picked = pick_ocr_confirmed_match(self.RANKING, OCR_PACKAGING)
        assert picked is not None
        assert picked.id == "mgt00726"

    def test_21_packaging_text_picks_21(self):
        picked = pick_ocr_confirmed_match(
            self.RANKING,
            "MINI GT ALPHATAURI AT04 #21 Nyck de Vries 2023 Australian GP",
        )
        assert picked is not None
        assert picked.id == "mgt00727"

    def test_no_text_returns_none(self):
        assert pick_ocr_confirmed_match(self.RANKING, "") is None

    def test_irrelevant_text_returns_none(self):
        assert pick_ocr_confirmed_match(self.RANKING, "foto de una pared blanca") is None


class TestGarbledAiText:
    """Imagen IA (ej. Gemini): OCR imperfecto — 'ATB' en vez de 'AT04'."""

    RANKING = [
        _m("mgt01021", "Mazda RX-7 RE-Amemiya Chrome Blue One Night in Malaysia 2025", 0.844),
        _m("mgt00577", "LB WORKS Nissan GT-R R35 Type 1 Rear Wing ver 1 RHD Gold Series", 0.835),
        _m("mgt00726", NAME_22, 0.798),
        _m("mgt01007", "RB VCARB 01 #22 Yuki Tsunoda  2024 F1 2024 Bahrain GP", 0.333),
    ]

    def test_garbled_packaging_text_still_picks_22(self):
        picked = pick_ocr_confirmed_match(
            self.RANKING,
            "ALPHATAURI 22 AlphaTauri ATB #22 Yuki Tsunoda Cenaalan Grand Prix",
        )
        assert picked is not None
        assert picked.id == "mgt00726"

    def test_vcardb_does_not_beat_alphatauri_on_pilot_tokens(self):
        # '22' y 'Yuki Tsunoda' están en varias piezas; AlphaTauri manda.
        s22 = score_name_against_ocr_text(
            NAME_22, "ALPHATAURI ATB #22 Yuki Tsunoda"
        )
        s_vc = score_name_against_ocr_text(
            "RB VCARB 01 #22 Yuki Tsunoda  2024 F1 2024 Bahrain GP",
            "ALPHATAURI ATB #22 Yuki Tsunoda",
        )
        assert s22 > s_vc

    def test_only_pilot_name_is_ambiguous(self):
        # Sin marca/modelo, Tsunoda solo → no hay pieza "la".
        picked = pick_ocr_confirmed_match(
            self.RANKING, "2026 F1 Cenaalan Grand Prix Yuki Tsunoda"
        )
        assert picked is None
