"""Pruebas unitarias para el Analizador Relacional de Consultas Jurídicas (LegalQueryExpander)
y el filtrado de citaciones utilizadas en LegalAgent.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.legal.legal_agent import _filter_used_citations
from app.schemas.legal.citation import LegalCitation
from app.services.legal.query_expander import LegalRelationAnalysis, expand_legal_query


def test_filter_used_citations_matching_articles_referenced():
    """Valida que solo se conserven las citas declaradas en articles_referenced."""
    cit1 = LegalCitation(
        law_identifier="LEY-11179",
        law_title="Código Penal",
        article_number="172",
        exact_quote="Será reprimido con prisión de un mes a seis años...",
        confidence_score=0.9,
    )
    cit2 = LegalCitation(
        law_identifier="LEY-11179",
        law_title="Código Penal",
        article_number="62",
        exact_quote="La acción penal se prescribirá...",
        confidence_score=0.85,
    )
    cit_noise = LegalCitation(
        law_identifier="LEY-24240",
        law_title="Defensa del Consumidor",
        article_number="40",
        exact_quote="Si el daño resulta del vicio o defecto...",
        confidence_score=0.4,
    )

    all_citations = [cit1, cit2, cit_noise]
    articles_referenced = ["LEY-11179:172", "LEY-11179:62"]
    answer = "El fraude tiene pena de 6 años (art. 172) y prescribe a los 6 años (art. 62)."

    used = _filter_used_citations(all_citations, articles_referenced, answer)

    assert len(used) == 2
    assert any(c.article_number == "172" for c in used)
    assert any(c.article_number == "62" for c in used)
    assert not any(c.article_number == "40" for c in used)


def test_filter_used_citations_by_body_mention():
    """Valida que si articles_referenced viene vacío pero el texto menciona el artículo, se reconozca."""
    cit1 = LegalCitation(
        law_identifier="LEY-11179",
        law_title="Código Penal",
        article_number="62",
        exact_quote="La acción penal prescribe...",
        confidence_score=0.8,
    )
    cit_noise = LegalCitation(
        law_identifier="LEY-20744",
        law_title="LCT",
        article_number="245",
        exact_quote="Despido...",
        confidence_score=0.5,
    )

    all_citations = [cit1, cit_noise]
    articles_referenced = []
    answer = "Conforme surge del artículo 62 del Código Penal, la acción prescribe al término de la pena máxima."

    used = _filter_used_citations(all_citations, articles_referenced, answer)

    assert len(used) == 1
    assert used[0].article_number == "62"


def test_filter_used_citations_fallback_on_no_match():
    """Valida que si no hay coincidencias se preserven solo las top 2 citas para evitar bloat de 10."""
    citations = [
        LegalCitation(law_identifier="LEY-1", law_title="T1", article_number=str(i), exact_quote="...")
        for i in range(10)
    ]
    used = _filter_used_citations(citations, [], "Respuesta general sin mención explícita de artículos.")

    assert len(used) == 2


@pytest.mark.asyncio
async def test_expand_legal_query_fallback_without_llm():
    """Verifica el fallback seguro ante ausencia de LLM."""
    query = "¿Cuánto tarda en prescribir el fraude?"
    res = await expand_legal_query(llm=None, query=query)

    assert isinstance(res, LegalRelationAnalysis)
    assert res.is_relational is False
    assert res.sub_queries == [query]
    assert res.canonical_articles == []


@pytest.mark.asyncio
async def test_expand_legal_query_with_mock_llm():
    """Verifica que el analizador estructure sub-queries y artículos canónicos para prescripción penal."""
    mock_llm = MagicMock()
    mock_response = LegalRelationAnalysis(
        question_type="prescription",
        is_relational=True,
        sub_queries=[
            "delito de estafa defraudacion pena de prision articulo 172 codigo penal",
            "prescripcion de la accion penal pena maxima fijada para el delito articulo 62 codigo penal",
            "suspension e interrupcion de la prescripcion penal articulo 67 codigo penal",
        ],
        canonical_articles=["LEY-11179:172", "LEY-11179:62", "LEY-11179:67"],
        reasoning="La prescripción del fraude exige articular la escala penal del Art. 172 con las reglas de prescripción del Art. 62.",
    )

    # In invoke_structured_with_retry, it either calls with_structured_output or ainvoke
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value=mock_response)
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

    query = "¿Cuánto tarda en prescribir el fraude?"
    res = await expand_legal_query(llm=mock_llm, query=query)

    assert res.is_relational is True
    assert res.question_type == "prescription"
    assert len(res.sub_queries) == 3
    assert "LEY-11179:62" in res.canonical_articles
    assert "LEY-11179:172" in res.canonical_articles
