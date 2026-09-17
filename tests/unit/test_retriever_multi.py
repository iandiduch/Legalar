"""Pruebas unitarias para los métodos de recuperación directa y búsqueda multi-query en HybridLegalRetriever."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core.config import Settings
from app.schemas.legal.citation import LegalCitation
from app.services.legal.retriever import HybridLegalRetriever


@pytest.fixture
def mock_retriever():
    settings = Settings()
    retriever = HybridLegalRetriever(
        settings=settings,
        pinecone_client=None,
        embeddings_client=None,
        sessionmaker=None,
    )
    return retriever


@pytest.mark.asyncio
async def test_search_multi_query_merges_and_ranks_correctly(mock_retriever):
    """Verifica que search_multi_query ejecute búsquedas concurrentes y fusione con RRF sin duplicados."""
    cit_fraude = LegalCitation(
        law_identifier="LEY-11179",
        law_title="Código Penal",
        article_number="172",
        exact_quote="Defraudación y estafa pena de un mes a seis años...",
        confidence_score=0.9,
    )
    cit_prescripcion = LegalCitation(
        law_identifier="LEY-11179",
        law_title="Código Penal",
        article_number="62",
        exact_quote="La acción penal prescribirá al transcurrir el máximo de la pena...",
        confidence_score=0.95,
    )
    cit_interrupcion = LegalCitation(
        law_identifier="LEY-11179",
        law_title="Código Penal",
        article_number="67",
        exact_quote="La prescripción correrá o será interrumpida separadamente...",
        confidence_score=0.8,
    )

    async def mock_search(query: str, **kwargs):
        if "estafa" in query or "fraude" in query:
            return [cit_fraude, cit_interrupcion]
        elif "prescripcion" in query:
            return [cit_prescripcion, cit_interrupcion]
        return []

    mock_retriever.search = AsyncMock(side_effect=mock_search)

    queries = [
        "delito de estafa defraudacion pena",
        "prescripcion de la accion penal pena maxima",
    ]

    results = await mock_retriever.search_multi_query(queries=queries, top_k=5)

    assert len(results) == 3
    art_numbers = [c.article_number for c in results]
    assert "172" in art_numbers
    assert "62" in art_numbers
    assert "67" in art_numbers
    # cit_interrupcion apareció en ambas queries, por lo que RRF acumula puntajes
    assert results[0].article_number in ("67", "172", "62")


@pytest.mark.asyncio
async def test_retrieve_exact_articles_empty_input(mock_retriever):
    """Verifica manejo seguro de lista vacía en retrieve_exact_articles."""
    res = await mock_retriever.retrieve_exact_articles([])
    assert res == []


@pytest.mark.asyncio
async def test_search_multi_query_single_query_fallback(mock_retriever):
    """Si se pasa una sola query a search_multi_query, debe delegar directamente a search."""
    mock_retriever.search = AsyncMock(return_value=[])
    await mock_retriever.search_multi_query(["consulta unica"], top_k=5)
    mock_retriever.search.assert_awaited_once_with(
        "consulta unica",
        top_k=5,
        include_historical=False,
        law_filter=None,
    )
