"""Worker de auditoría documental: consume la cola Redis legal:analysis:jobs en un proceso separado del API.

Ejecuta el grafo de LangGraph (Document Analyzer + Validator) de forma asíncrona,
actualiza el estado en PostgreSQL (AnalysisJob) y emite notificaciones en tiempo
real mediante Redis Pub/Sub para el streaming SSE del frontend.
"""

import asyncio
import json
import logging
import time
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from uuid import UUID

from langchain_core.messages import HumanMessage
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.legal.graph import build_legal_graph
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.metrics import ANALYSIS_JOBS_TOTAL
from app.core.redis import build_redis_client
from app.core.telemetry import setup_telemetry
from app.db.models_orm import AnalysisJob
from app.db.session import build_engine, build_sessionmaker
from app.domain.models import ConfidenceLevel, QueryIntent
from app.schemas.legal.chat import LegalChatResponse, LegalValidationSummary
from app.services.legal.legalize_api_client import LegalizeApiClient
from app.services.legal.local_diff_engine import LocalGitDiffEngine
from app.services.legal.retriever import HybridLegalRetriever
from app.services.llm_factory import build_chat_model
from app.services.rag_service import build_embeddings_client, build_pinecone_client

logger = logging.getLogger(__name__)

_HEARTBEAT_KEY = "worker:analysis:heartbeat"
_HEARTBEAT_TTL_SECONDS = 60
_CLEANUP_INTERVAL_SECONDS = 300  # Cada 5 minutos


async def run_worker(settings: Settings) -> None:
    """Bucle principal de ejecución del worker de análisis."""
    setup_telemetry(settings)
    engine = build_engine(settings)
    sessionmaker = build_sessionmaker(engine)
    redis = build_redis_client(settings)
    pinecone_client = build_pinecone_client(settings)
    embeddings_client = build_embeddings_client(settings)

    retriever = HybridLegalRetriever(
        settings=settings,
        pinecone_client=pinecone_client,
        embeddings_client=embeddings_client,
        sessionmaker=sessionmaker,
    )
    local_diff_engine = LocalGitDiffEngine(repo_path=settings.LEGALIZE_REPO_PATH)
    diff_client = LegalizeApiClient(
        settings=settings,
        local_diff_engine=local_diff_engine,
    )
    llm = build_chat_model(settings)
    app_graph = build_legal_graph(checkpointer=None)

    logger.info("analysis_worker.started", extra={"queue": settings.REDIS_ANALYSIS_QUEUE_KEY})

    # Recuperación inicial de jobs huérfanos al arrancar
    await recover_orphaned_analysis_jobs(sessionmaker, redis, settings)
    cleanup_task = asyncio.create_task(_orphan_cleanup_loop(sessionmaker, redis, settings))

    try:
        while True:
            await _beat(redis)
            try:
                item = await redis.blpop([settings.REDIS_ANALYSIS_QUEUE_KEY], timeout=2)
            except (TimeoutError, RedisTimeoutError):
                continue
            except RedisConnectionError as exc:
                logger.warning("analysis_worker.redis_connection_retry", extra={"error": str(exc)})
                await asyncio.sleep(1)
                continue

            if item is None:
                continue

            _, raw_job_id = item
            try:
                job_id = UUID(raw_job_id)
            except ValueError:
                logger.error("analysis_worker.invalid_job_id", extra={"raw": raw_job_id})
                continue

            async with sessionmaker() as session:
                await _process_analysis_job(
                    job_id=job_id,
                    session=session,
                    redis=redis,
                    app_graph=app_graph,
                    llm=llm,
                    retriever=retriever,
                    diff_client=diff_client,
                    settings=settings,
                )
            await _beat(redis)
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        await redis.aclose()
        if pinecone_client:
            await pinecone_client.close()
        await engine.dispose()


async def _beat(redis: Redis) -> None:
    """Envía un pulso de vida a Redis."""
    await redis.set(_HEARTBEAT_KEY, datetime.now(UTC).isoformat(), ex=_HEARTBEAT_TTL_SECONDS)


async def _orphan_cleanup_loop(
    sessionmaker: async_sessionmaker[AsyncSession],
    redis: Redis,
    settings: Settings,
) -> None:
    """Loop en segundo plano que recupera jobs de análisis huérfanos."""
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
        try:
            await recover_orphaned_analysis_jobs(sessionmaker, redis, settings)
        except Exception as exc:  # noqa: BLE001
            logger.error("analysis_worker.cleanup_loop_error", extra={"error": str(exc)})


async def recover_orphaned_analysis_jobs(
    sessionmaker: async_sessionmaker[AsyncSession],
    redis: Redis,
    settings: Settings,
) -> int:
    """Recupera jobs de análisis estancados en PROCESSING o PENDING tras timeout."""
    cutoff = datetime.now(UTC) - timedelta(minutes=settings.ANALYSIS_JOB_TIMEOUT_MINUTES)
    recovered_count = 0

    async with sessionmaker() as session:
        stmt = select(AnalysisJob).where(
            (
                (AnalysisJob.status == "PROCESSING")
                & (AnalysisJob.started_at < cutoff)
            )
            | (
                (AnalysisJob.status == "PENDING")
                & (AnalysisJob.created_at < cutoff)
            )
        )
        orphaned_jobs = list((await session.execute(stmt)).scalars().all())

        for job in orphaned_jobs:
            if job.retry_count < 2:
                job.status = "PENDING"
                job.retry_count += 1
                job.started_at = None
                job.error_message = f"Reintentando job huérfano tras timeout (intento {job.retry_count}/2)"
                await redis.rpush(settings.REDIS_ANALYSIS_QUEUE_KEY, str(job.analysis_id))
                recovered_count += 1
                logger.warning("analysis_worker.requeued_orphaned_job", extra={"analysis_id": str(job.analysis_id)})
            else:
                job.status = "FAILED"
                job.completed_at = datetime.now(UTC)
                job.error_message = (
                    f"Se superó el tiempo límite ({settings.ANALYSIS_JOB_TIMEOUT_MINUTES} min) y reintentos máximos."
                )
                ANALYSIS_JOBS_TOTAL.labels(status="FAILED", document_type=job.document_type).inc()
                logger.error("analysis_worker.orphaned_job_failed", extra={"analysis_id": str(job.analysis_id)})

        if orphaned_jobs:
            await session.commit()

    return recovered_count


async def _process_analysis_job(
    job_id: UUID,
    session: AsyncSession,
    redis: Redis,
    app_graph,
    llm,
    retriever,
    diff_client,
    settings: Settings,
) -> None:
    """Ejecuta la auditoría del documento y publica eventos SSE a través de Redis."""
    pubsub_channel = f"legal:analysis:events:{job_id}"

    job = await session.get(AnalysisJob, job_id)
    if job is None:
        for attempt in range(3):
            await asyncio.sleep(0.5 * (attempt + 1))
            session.expire_all()
            job = await session.get(AnalysisJob, job_id)
            if job is not None:
                break
        if job is None:
            logger.error("analysis_worker.job_not_found", extra={"job_id": str(job_id)})
            return

    job.status = "PROCESSING"
    job.started_at = datetime.now(UTC)
    await session.commit()

    # Notificar inicio de procesamiento vía Redis Pub/Sub
    await redis.publish(
        pubsub_channel,
        json.dumps({
            "event": "status",
            "stage": "processing",
            "message": f"Iniciando auditoría jurídica del {job.document_type} y cotejo normativo...",
        }),
    )

    start_time = time.time()
    req_query = f"Auditar el siguiente {job.document_type} a la luz del derecho argentino:"

    initial_state = {
        "messages": [HumanMessage(content=req_query)],
        "thread_id": str(job.analysis_id),
        "query": req_query,
        "intent": QueryIntent.DOCUMENT_ANALYSIS,
        "citations": [],
        "draft_answer": None,
        "validation_result": None,
        "document_text": job.document_text,
        "document_type": job.document_type,
        "diff_request_params": None,
        "diff_result": None,
        "final_answer": None,
        "confidence": ConfidenceLevel.HIGH,
        "iteration": 0,
    }

    config = {
        "configurable": {
            "thread_id": str(job.analysis_id),
            "llm_client": llm,
            "legal_retriever": retriever,
            "legalize_api_client": diff_client,
            "settings": settings,
        },
        "recursion_limit": settings.GRAPH_RECURSION_LIMIT,
    }

    try:
        # Notificar etapa de razonamiento del modelo
        await redis.publish(
            pubsub_channel,
            json.dumps({
                "event": "status",
                "stage": "analyzing",
                "message": "Detectando cláusulas abusivas, contingencias y validando vigencia de leyes...",
            }),
        )

        final_state = await asyncio.wait_for(
            app_graph.ainvoke(initial_state, config=config),
            timeout=120.0,
        )

        duration = time.time() - start_time
        val_res = final_state.get("validation_result") or LegalValidationSummary(is_valid=True)

        chat_response = LegalChatResponse(
            thread_id=str(job.analysis_id),
            query=req_query,
            intent=QueryIntent.DOCUMENT_ANALYSIS,
            answer=final_state.get("final_answer") or final_state.get("draft_answer") or "Auditoría completada.",
            citations=final_state.get("citations", []),
            confidence=final_state.get("confidence", ConfidenceLevel.HIGH),
            validation=val_res,
            processing_time_seconds=round(duration, 3),
        )

        job.status = "COMPLETED"
        job.result = chat_response.model_dump()
        job.completed_at = datetime.now(UTC)
        await session.commit()

        ANALYSIS_JOBS_TOTAL.labels(status="COMPLETED", document_type=job.document_type).inc()
        logger.info("analysis_worker.job_completed", extra={"analysis_id": str(job.analysis_id), "duration": duration})

        # Notificar evento 'done' con el dictamen estructurado completo
        await redis.publish(
            pubsub_channel,
            json.dumps({
                "event": "done",
                "result": job.result,
            }),
        )

    except Exception as exc:  # noqa: BLE001 - Resguardo para capturar fallo y notificar por SSE
        duration = time.time() - start_time
        logger.exception("analysis_worker.job_failed", extra={"analysis_id": str(job.analysis_id), "error": str(exc)})

        job.status = "FAILED"
        job.error_message = str(exc)
        job.completed_at = datetime.now(UTC)
        await session.commit()

        ANALYSIS_JOBS_TOTAL.labels(status="FAILED", document_type=job.document_type).inc()

        # Notificar evento 'error'
        await redis.publish(
            pubsub_channel,
            json.dumps({
                "event": "error",
                "message": f"Error durante la auditoría: {str(exc)}",
            }),
        )


if __name__ == "__main__":
    _settings = get_settings()
    configure_logging(_settings)
    asyncio.run(run_worker(_settings))
