"""Worker de ingesta: consume la cola Redis en un proceso separado del API.

Incluye:
- Manejo y recuperación automática de jobs huérfanos (jobs en PROCESSING que hayan
  superado el timeout de procesamiento tras una caída del worker).
- Publicación de latidos en Redis para healthcheck en Docker.
- Soporte para escalado concurrente multi-instancia (múltiples réplicas de worker).
- Registro de métricas Prometheus para observabilidad de pipeline.

"""

import asyncio
import logging
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlalchemy import and_, delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    DocumentParsingError,
    EmbeddingGenerationError,
    PineconeUpsertError,
    SuspiciousContentDetectedError,
    UnsupportedFileTypeError,
    VectorStoreError,
)
from app.core.logging import configure_logging
from app.core.metrics import INGESTION_JOBS_TOTAL
from app.core.prompt_injection_scanner import scan_for_injection_patterns
from app.core.redis import build_redis_client
from app.db.models_orm import DocumentChunk, IngestionJob
from app.db.session import build_engine, build_sessionmaker
from app.domain.models import FileType, TaskStatus
from app.schemas.ingest import ChunkWithMetadata
from app.services.ingestion_service import build_chunks, parse_document
from app.services.rag_service import (
    build_embeddings_client,
    build_pinecone_client,
    embed_texts,
    get_index,
    upsert_chunks,
)

logger = logging.getLogger(__name__)

_HEARTBEAT_KEY = "worker:ingestion:heartbeat"
_HEARTBEAT_TTL_SECONDS = 60
_CLEANUP_INTERVAL_SECONDS = 300  # cada 5 minutos

_JOB_ERRORS = (
    UnsupportedFileTypeError,
    DocumentParsingError,
    SuspiciousContentDetectedError,
    EmbeddingGenerationError,
    PineconeUpsertError,
    VectorStoreError,
)


async def run_worker(settings: Settings) -> None:
    engine = build_engine(settings)
    sessionmaker = build_sessionmaker(engine)
    redis = build_redis_client(settings)
    pinecone_client = build_pinecone_client(settings)
    index = await get_index(pinecone_client, settings)
    embeddings_client = build_embeddings_client(settings)

    logger.info("ingestion_worker.started", extra={"queue": settings.REDIS_INGEST_QUEUE_KEY})

    # Recuperación inicial de jobs huérfanos al arrancar el worker
    await recover_orphaned_jobs(sessionmaker, redis, settings)
    cleanup_task = asyncio.create_task(_orphan_cleanup_loop(sessionmaker, redis, settings))

    try:
        while True:
            await _beat(redis)
            try:
                item = await redis.blpop([settings.REDIS_INGEST_QUEUE_KEY], timeout=2)
            except (TimeoutError, RedisTimeoutError):
                continue
            except RedisConnectionError as exc:
                logger.warning("ingestion_worker.redis_connection_retry", extra={"error": str(exc)})
                await asyncio.sleep(1)
                continue
            if item is None:
                continue
            _, raw_job_id = item
            async with sessionmaker() as session:
                await _process_job(UUID(raw_job_id), session, index, embeddings_client, settings)
            await _beat(redis)
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        await redis.aclose()
        await index.close()
        await pinecone_client.close()
        await engine.dispose()


async def _beat(redis: Redis) -> None:
    await redis.set(_HEARTBEAT_KEY, datetime.now(UTC).isoformat(), ex=_HEARTBEAT_TTL_SECONDS)


async def _orphan_cleanup_loop(
    sessionmaker: async_sessionmaker[AsyncSession],
    redis: Redis,
    settings: Settings,
) -> None:
    """Loop en segundo plano que revisa periódicamente si hay jobs huérfanos."""
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
        try:
            await recover_orphaned_jobs(sessionmaker, redis, settings)
        except Exception as exc:  # noqa: BLE001 - Error en tarea de mantenimiento periódica
            logger.error("ingestion_worker.cleanup_loop_error", extra={"error": str(exc)})


async def recover_orphaned_jobs(
    sessionmaker: async_sessionmaker[AsyncSession],
    redis: Redis,
    settings: Settings,
) -> int:
    """Detecta jobs estancados en PROCESSING cuyo tiempo de ejecución superó el timeout.

    Si no superaron el máximo de reintentos, los re-encola en Redis en estado PENDING.
    Si agotaron los reintentos, los marca como FAILED.
    """
    cutoff = datetime.now(UTC) - timedelta(minutes=settings.INGESTION_JOB_TIMEOUT_MINUTES)
    recovered_count = 0

    async with sessionmaker() as session:
        stmt = select(IngestionJob).where(
            or_(
                and_(
                    IngestionJob.status == TaskStatus.PROCESSING.value,
                    IngestionJob.started_at < cutoff,
                ),
                and_(
                    IngestionJob.status == TaskStatus.PENDING.value,
                    IngestionJob.created_at < cutoff,
                ),
            )
        )
        result = await session.execute(stmt)
        orphaned_jobs = list(result.scalars().all())

        for job in orphaned_jobs:
            if job.retry_count < settings.INGESTION_MAX_RETRIES:
                job.status = TaskStatus.PENDING.value
                job.retry_count += 1
                job.error_message = f"Reintentando job huérfano tras timeout (intento {job.retry_count}/{settings.INGESTION_MAX_RETRIES})"
                await session.flush()
                await redis.rpush(settings.REDIS_INGEST_QUEUE_KEY, str(job.job_id))
                logger.warning(
                    "ingestion_worker.requeued_orphaned_job",
                    extra={"job_id": str(job.job_id), "retry_count": job.retry_count},
                )
            else:
                job.status = TaskStatus.FAILED.value
                job.error_message = (
                    f"Se superó el tiempo límite de procesamiento ({settings.INGESTION_JOB_TIMEOUT_MINUTES} min) "
                    f"y el máximo de reintentos ({settings.INGESTION_MAX_RETRIES})"
                )
                job.completed_at = datetime.now(UTC)
                INGESTION_JOBS_TOTAL.labels(status=TaskStatus.FAILED.value, file_type=job.file_type).inc()
                logger.error(
                    "ingestion_worker.orphaned_job_failed_max_retries",
                    extra={"job_id": str(job.job_id), "retries": job.retry_count},
                )
            recovered_count += 1

        if orphaned_jobs:
            await session.commit()

    return recovered_count


async def _process_job(
    job_id: UUID,
    session: AsyncSession,
    index,
    embeddings_client,
    settings: Settings,
) -> None:
    job = await session.get(IngestionJob, job_id)
    if job is None:
        # Reintento breve con backoff para absorber cualquier latencia de commit entre procesos
        for attempt in range(3):
            await asyncio.sleep(0.5 * (attempt + 1))
            session.expire_all()
            job = await session.get(IngestionJob, job_id)
            if job is not None:
                break
        if job is None:
            logger.error("ingestion_worker.job_not_found", extra={"job_id": str(job_id)})
            return

    job.status = TaskStatus.PROCESSING.value
    job.started_at = datetime.now(UTC)
    await session.commit()

    try:
        file_type = FileType(job.file_type)
        pages = await parse_document(Path(job.storage_path), file_type)

        if settings.INGESTION_CONTENT_SCAN_ENABLED:
            matches = scan_for_injection_patterns("\n".join(page.text for page in pages))
            if matches:
                logger.warning(
                    "ingestion_worker.suspicious_content_detected",
                    extra={"job_id": str(job_id), "document_filename": job.filename, "patterns": matches},
                )
                raise SuspiciousContentDetectedError(
                    f"El documento {job.filename} fue rechazado: contiene patrones sospechosos de "
                    f"prompt injection ({', '.join(matches)}). Revisalo antes de volver a subirlo."
                )

        chunks = build_chunks(pages, job_id, job.filename, file_type, settings)
        if not chunks:
            raise DocumentParsingError(f"El documento {job.filename} no produjo contenido indexable")

        # Purga chunks y vectores anteriores si el documento ya se habia indexado previamente
        stmt_prev = select(DocumentChunk.chunk_id).where(DocumentChunk.filename == job.filename)
        prev_chunk_ids = list((await session.execute(stmt_prev)).scalars().all())
        if prev_chunk_ids:
            try:
                for i in range(0, len(prev_chunk_ids), 1000):
                    await index.delete(ids=prev_chunk_ids[i : i + 1000])
                logger.info(
                    "ingestion_worker.purged_previous_vectors",
                    extra={"document_filename": job.filename, "purged_chunks": len(prev_chunk_ids)},
                )
            except Exception as exc:  # noqa: BLE001 - Resguardo de resiliencia ante fallos de borrado en Pinecone
                logger.warning(
                    "ingestion_worker.delete_previous_pinecone_vectors_failed",
                    extra={"document_filename": job.filename, "error": str(exc)},
                )
            await session.execute(delete(DocumentChunk).where(DocumentChunk.filename == job.filename))
            await session.flush()

        embeddings = await embed_texts([c.text for c in chunks], embeddings_client)
        chunks_indexed = await upsert_chunks(index, chunks, embeddings, settings)
        session.add_all(_to_document_chunks(chunks))

        job.status = TaskStatus.COMPLETED.value
        job.chunks_indexed = chunks_indexed
        job.completed_at = datetime.now(UTC)
        await session.commit()

        INGESTION_JOBS_TOTAL.labels(status=TaskStatus.COMPLETED.value, file_type=job.file_type).inc()
        logger.info("ingestion_worker.job_completed", extra={"job_id": str(job_id), "chunks": chunks_indexed})

    except _JOB_ERRORS as exc:
        job.status = TaskStatus.FAILED.value
        job.error_message = str(exc)
        job.completed_at = datetime.now(UTC)
        await session.commit()

        INGESTION_JOBS_TOTAL.labels(status=TaskStatus.FAILED.value, file_type=job.file_type).inc()
        logger.error("ingestion_worker.job_failed", extra={"job_id": str(job_id), "error": str(exc)})

    except Exception as exc:  # noqa: BLE001 - Resguardo ante fallos imprevistos para no derribar el proceso worker
        job.status = TaskStatus.FAILED.value
        job.error_message = f"Error inesperado durante la ingesta: {exc}"
        job.completed_at = datetime.now(UTC)
        await session.commit()

        INGESTION_JOBS_TOTAL.labels(status=TaskStatus.FAILED.value, file_type=job.file_type).inc()
        logger.exception("ingestion_worker.unexpected_job_failure", extra={"job_id": str(job_id), "error": str(exc)})


def _to_document_chunks(chunks: list[ChunkWithMetadata]) -> list[DocumentChunk]:
    """Copia el texto y metadatos de cada chunk en PostgreSQL para el Full-Text Search."""
    return [
        DocumentChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.metadata.document_id,
            filename=chunk.metadata.filename,
            file_type=chunk.metadata.file_type.value,
            page=chunk.metadata.page,
            section=chunk.metadata.section,
            source=chunk.metadata.source,
            text=chunk.text,
        )
        for chunk in chunks
    ]


if __name__ == "__main__":
    _settings = get_settings()
    configure_logging(_settings)
    asyncio.run(run_worker(_settings))
