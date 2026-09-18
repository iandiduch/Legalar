"""Tests unitarios para la Fase 2: Desacoplamiento de /analyze a segundo plano con colas Redis y SSE."""

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.core.config import Settings
from app.db.models_orm import AnalysisJob
from app.domain.models import ConfidenceLevel, QueryIntent
from app.schemas.legal.analysis import AnalysisJobResponse, AnalysisJobStatusResponse
from app.schemas.legal.chat import LegalChatResponse, LegalValidationSummary
from app.services.legal.analysis_worker import _process_analysis_job, recover_orphaned_analysis_jobs


def test_analysis_schemas():
    """Valida los schemas de respuesta 202 y estado de auditoría."""
    job_id = uuid.uuid4()
    resp_202 = AnalysisJobResponse(
        analysis_id=str(job_id),
        status="PENDING",
        document_type="contrato",
        filename="contrato_alquiler.pdf",
        events_url=f"/api/v1/legal/analyze/{job_id}/events",
    )
    assert resp_202.analysis_id == str(job_id)
    assert resp_202.status == "PENDING"
    assert "/events" in resp_202.events_url

    status_resp = AnalysisJobStatusResponse(
        analysis_id=str(job_id),
        status="COMPLETED",
        document_type="contrato",
        filename="contrato_alquiler.pdf",
        result={"answer": "Dictamen de prueba", "intent": QueryIntent.DOCUMENT_ANALYSIS.value},
        processing_time_seconds=3.45,
    )
    assert status_resp.status == "COMPLETED"
    assert status_resp.result["answer"] == "Dictamen de prueba"


@pytest.mark.asyncio
async def test_process_analysis_job_success():
    """Valida que el worker procese el job, llame al grafo, actualice la DB y publique eventos en Redis."""
    job_id = uuid.uuid4()
    mock_job = AnalysisJob(
        analysis_id=job_id,
        document_type="contrato",
        filename="locacion.txt",
        document_text="Cláusula 1: El locatario pagará un depósito en moneda extranjera no reintegrable.",
        status="PENDING",
    )

    mock_session = AsyncMock()
    mock_session.get.return_value = mock_job
    mock_redis = AsyncMock()

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "final_answer": "### Dictamen Legal\nLa cláusula de depósito no reintegrable vulnera el art. 1196 del CCCN.",
        "citations": [],
        "confidence": ConfidenceLevel.HIGH,
        "validation_result": LegalValidationSummary(is_valid=True),
    }

    settings = Settings(
        OPENAI_API_KEY="dummy",
        PINECONE_API_KEY="dummy",
        GRAPH_RECURSION_LIMIT=25,
    )

    await _process_analysis_job(
        job_id=job_id,
        session=mock_session,
        redis=mock_redis,
        app_graph=mock_graph,
        llm=MagicMock(),
        retriever=MagicMock(),
        diff_client=MagicMock(),
        settings=settings,
    )

    # Verifica que el job se marque como COMPLETED con su resultado
    assert mock_job.status == "COMPLETED"
    assert mock_job.result is not None
    assert "vulnera el art. 1196" in mock_job.result["answer"]

    # Verifica que se hayan publicado eventos SSE a Redis Pub/Sub
    pubsub_channel = f"legal:analysis:events:{job_id}"
    published_calls = [call for call in mock_redis.publish.call_args_list if call[0][0] == pubsub_channel]
    assert len(published_calls) >= 3  # status (processing) -> status (analyzing) -> done

    last_call_payload = json.loads(published_calls[-1][0][1])
    assert last_call_payload["event"] == "done"
    assert "result" in last_call_payload


@pytest.mark.asyncio
async def test_process_analysis_job_handles_failure():
    """Valida que si el grafo falla, el job se marque como FAILED y se emita el evento error por SSE."""
    job_id = uuid.uuid4()
    mock_job = AnalysisJob(
        analysis_id=job_id,
        document_type="contrato",
        filename="locacion.txt",
        document_text="Texto válido de prueba.",
        status="PENDING",
    )

    mock_session = AsyncMock()
    mock_session.get.return_value = mock_job
    mock_redis = AsyncMock()

    mock_graph = AsyncMock()
    mock_graph.ainvoke.side_effect = RuntimeError("Fallo simulado de conexión LLM")

    settings = Settings(
        OPENAI_API_KEY="dummy",
        PINECONE_API_KEY="dummy",
        GRAPH_RECURSION_LIMIT=25,
    )

    await _process_analysis_job(
        job_id=job_id,
        session=mock_session,
        redis=mock_redis,
        app_graph=mock_graph,
        llm=MagicMock(),
        retriever=MagicMock(),
        diff_client=MagicMock(),
        settings=settings,
    )

    assert mock_job.status == "FAILED"
    assert "Fallo simulado de conexión LLM" in mock_job.error_message

    pubsub_channel = f"legal:analysis:events:{job_id}"
    published_calls = [call for call in mock_redis.publish.call_args_list if call[0][0] == pubsub_channel]
    last_call_payload = json.loads(published_calls[-1][0][1])
    assert last_call_payload["event"] == "error"
    assert "Fallo simulado" in last_call_payload["message"]
