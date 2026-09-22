"""Tests unitarios para el harness de evaluación RAG, esquemas y ground truth."""

import pytest
from app.schemas.evaluation import EvaluationJudgment, EvaluationResult, GoldenSetItem
from scripts.evaluate_rag import _evaluate_ground_truth, _load_golden_set


def test_load_golden_set_valid_schema():
    """Valida que data/golden_set.json contenga las preguntas del catálogo ampliado y valide contra GoldenSetItem."""
    items = _load_golden_set()
    assert len(items) == 34
    for item in items:
        assert isinstance(item.question, str)
        assert len(item.question) > 10


def test_evaluate_ground_truth_matching():
    """Valida que la función de ground truth detecte fuentes y artículos correctamente."""
    item = GoldenSetItem(
        question="¿Cómo se calcula la indemnización por despido?",
        expected_source=["LEY-20744", "DEC-390-1976"],
        expected_article="245",
        expected_answer_contains=["un mes de sueldo", "antigüedad"],
    )

    answer = "Conforme al artículo 245 de la LCT (Decreto 390/1976), la indemnización equivale a un mes de sueldo por año de antigüedad."
    sources = ["DEC-390-1976 Art. 245", "LEY-20744 Art. 245"]

    src_rec, art_rec, contains_exp = _evaluate_ground_truth(item, answer, sources)
    assert src_rec is True
    assert art_rec is True
    assert contains_exp is True


def test_evaluate_ground_truth_out_of_domain():
    """Valida que preguntas fuera de dominio (sin fuentes esperadas) aprueben el ground truth."""
    item = GoldenSetItem(
        question="¿Cuánto sale viajar a Madrid?",
        expected_source=None,
        expected_article=None,
        expected_answer_contains=["asistente legal", "derecho argentino"],
    )

    answer = "Como asistente legal en derecho argentino, no dispongo de información sobre tarifas turísticas."
    sources = []

    src_rec, art_rec, contains_exp = _evaluate_ground_truth(item, answer, sources)
    assert src_rec is True
    assert art_rec is True
    assert contains_exp is True


def test_evaluate_ground_truth_missing_article():
    """Valida que si no se cita el artículo requerido, falle el article_recall."""
    item = GoldenSetItem(
        question="¿Cuál es el plazo de garantía?",
        expected_source="LEY-24240",
        expected_article="11",
        expected_answer_contains="meses",
    )

    answer = "Según la Ley de Defensa del Consumidor (Ley 24.240), el plazo general es de seis meses."
    sources = ["LEY-24240 Art. 10"]

    src_rec, art_rec, contains_exp = _evaluate_ground_truth(item, answer, sources)
    assert src_rec is True
    assert art_rec is False
    assert contains_exp is True


def test_evaluation_result_schema_serialization():
    """Valida la serialización de EvaluationResult con las nuevas métricas."""
    res = EvaluationResult(
        question="Test Q",
        answer="Test Answer",
        sources_used=["LEY-26994 Art. 1198"],
        expected_source="LEY-26994",
        expected_article="1198",
        source_recalled=True,
        article_recalled=True,
        contains_expected=True,
        faithfulness=0.95,
        relevance=0.90,
        passed=True,
    )
    dumped = res.model_dump()
    assert dumped["passed"] is True
    assert dumped["source_recalled"] is True
    assert dumped["faithfulness"] == 0.95


def test_normalize_law_id_variants():
    """Valida la generación de variantes de normalización para matching flexible."""
    from scripts.evaluate_rag import _normalize_law_id

    v1 = _normalize_law_id("LEY-26994")
    assert "LEY-26994" in v1
    assert "26.994" in v1
    assert "26994" in v1

    v2 = _normalize_law_id("DNU-70-2023")
    assert "DNU-70-2023" in v2
    assert "DNU 70/2023" in v2
    assert "70/2023" in v2

    v3 = _normalize_law_id("DEC-390-1976")
    assert "DEC-390-1976" in v3
    assert "DEC 390/1976" in v3
    assert "390/1976" in v3

