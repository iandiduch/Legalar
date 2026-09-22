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


def test_filter_used_citations_empty_on_no_match():
    """Valida que si no hay coincidencias ni citas en la respuesta se devuelva lista vacía (cero fuentes inventadas)."""
    citations = [
        LegalCitation(law_identifier="LEY-1", law_title="T1", article_number=str(i), exact_quote="...")
        for i in range(10)
    ]
    used = _filter_used_citations(citations, [], "Respuesta general sin mención explícita de artículos.")

    assert len(used) == 0



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


@pytest.mark.asyncio
async def test_expand_legal_query_multi_aspect_leases_dnu70():
    """Verifica que consultas compuestas multi-aspecto (precio y depósito) se desglosen con sus dos artículos canónicos."""
    mock_llm = MagicMock()
    mock_response = LegalRelationAnalysis(
        question_type="general",
        is_relational=True,
        sub_queries=[
            "ajustes de precio indice actualizacion locacion habitacional DNU 70 2023 articulo 1199 codigo civil y comercial",
            "depositos en garantia fianza devolucion locacion habitacional DNU 70 2023 articulo 1196 codigo civil y comercial",
        ],
        canonical_articles=["LEY-26994:1196", "DNU-70-2023:255", "LEY-26994:1199", "DNU-70-2023:257", "DNU-70-2023:249"],
        reasoning="La consulta combina dos institutos independientes: depósitos en garantía (Art. 1196 / 255 DNU) y ajustes de precio (Art. 1199 / 257 DNU).",
    )

    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value=mock_response)
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

    query = "¿Cómo quedaron regulados los ajustes de precio y los depósitos en garantía en contratos de alquiler habitacional tras el DNU 70/2023?"
    res = await expand_legal_query(llm=mock_llm, query=query)

    assert res.is_relational is True
    assert len(res.sub_queries) == 2
    assert "LEY-26994:1196" in res.canonical_articles
    assert "LEY-26994:1199" in res.canonical_articles
    assert "DNU-70-2023:255" in res.canonical_articles
    assert "DNU-70-2023:257" in res.canonical_articles



@pytest.mark.asyncio
async def test_general_inquiry_node_returns_empty_citations():
    """Valida que general_inquiry_node responda cordialmente con cero citas normativas."""
    from app.agents.legal.general_inquiry import general_inquiry_node
    from app.agents.legal.state import LegalAgentState

    state = {
        "query": "hola",
        "messages": [],
    }
    config = {"configurable": {"llm_client": None}}

    res = await general_inquiry_node(state, config)

    assert res["citations"] == []
    assert "Legalar" in res["final_answer"]
    assert len(res["messages"]) == 1


def test_route_intent_edge_routes_general_inquiry():
    """Valida que route_intent_edge derive GENERAL_INQUIRY al nodo general_inquiry."""
    from app.agents.legal.graph import route_intent_edge
    from app.domain.models import QueryIntent

    state = {"intent": QueryIntent.GENERAL_INQUIRY}
    edge = route_intent_edge(state)
    assert edge == "general_inquiry"


@pytest.mark.asyncio
async def test_validator_appends_warning_block_on_invalid():
    """Valida que si el validador determina is_valid=False incorpore un bloque de advertencias visible."""
    from unittest.mock import AsyncMock, MagicMock
    from app.agents.legal.validator import ValidatorCheck, legal_validator_node
    from app.schemas.legal.citation import LegalCitation

    mock_llm = MagicMock()
    mock_check = ValidatorCheck(
        is_valid=False,
        all_norms_in_force=False,
        unsupported_claims=["La indemnización reclamada no cuenta con sustento normativo."],
        warning_notes=["La Ley 27.551 fue derogada por el DNU 70/2023."],
        synthesized_final_answer="Respuesta preliminar con advertencias sobre normas derogadas.",
    )
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value=mock_check)
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

    citations = [
        LegalCitation(
            law_identifier="LEY-27551",
            law_title="Ley de Alquileres",
            article_number="14",
            exact_quote="Texto de la ley...",
            status="derogada",
        )
    ]
    state = {
        "thread_id": "test-invalid-thread",
        "draft_answer": "Borrador con norma derogada.",
        "citations": citations,
    }
    config = {
        "configurable": {
            "llm_client": mock_llm,
        }
    }

    res = await legal_validator_node(state, config)

    assert res["validation_result"].is_valid is False
    assert "[!WARNING]" in res["final_answer"]
    assert "Ley 27.551 fue derogada" in res["final_answer"]
    assert "Observaciones de Auditoría Jurídica" in res["final_answer"]


def test_prompts_aligned_with_guidelines():
    """Valida que los prompts contengan las directivas de completitud, no cálculo numérico y preservación de preguntas."""
    from app.services.prompt_manager import get_default_prompt_manager

    pm = get_default_prompt_manager()
    doc_prompt = pm._load_from_disk("document_analyzer").content
    legal_prompt = pm._load_from_disk("legal_agent").content
    rewrite_prompt = pm._load_from_disk("query_rewrite").content
    validator_prompt = pm._load_from_disk("validator").content

    # 1. Prohibición de cálculos aritméticos de liquidaciones
    assert "PROHIBICIÓN DE CÁLCULOS ARITMÉTICOS" in legal_prompt
    assert "Vizzoti" in legal_prompt

    # 2. Query rewrite multiturrno para respuestas fácticas
    assert "RESPUESTA FÁCTICA" in rewrite_prompt

    # 3. Preservación de preguntas de aclaración en el validador
    assert "PRESERVACIÓN DE PREGUNTAS DE ACLARACIÓN" in validator_prompt

    # 4. Completitud documental en document_analyzer
    assert "DOCUMENTO O FRAGMENTO INCOMPLETO" in doc_prompt

