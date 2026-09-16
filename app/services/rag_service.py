"""Pinecone como vector store (vía AsyncPinecone nativo) + Búsqueda Léxica en PostgreSQL (FTS).

Embeddings OpenAI vía langchain_openai con retries y tipado estricto.
El retrieval híbrido combina búsqueda semántica densa en Pinecone con búsqueda léxica
distribuida en PostgreSQL (Full-Text Search con tsvector/plainto_tsquery y ts_rank_cd),
permitiendo escalabilidad horizontal sin requerir memoria en cada instancia de API.
"""

import asyncio
import logging
from typing import Any

from langchain_openai import OpenAIEmbeddings
from openai import OpenAIError
from pinecone import AsyncIndex, AsyncPinecone, PineconeError, ServerlessSpec
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.exceptions import (
    EmbeddingGenerationError,
    PineconeUpsertError,
    VectorStoreError,
)
from app.db.models_orm import DocumentChunk
from app.schemas.ingest import ChunkMetadata, ChunkWithMetadata, RetrievedChunk

logger = logging.getLogger(__name__)


def build_pinecone_client(settings: Settings) -> AsyncPinecone:
    return AsyncPinecone(api_key=settings.PINECONE_API_KEY.get_secret_value())


def build_embeddings_client(settings: Settings) -> OpenAIEmbeddings:
    kwargs: dict[str, Any] = {
        "model": settings.effective_embedding_model,
        "api_key": settings.OPENAI_API_KEY.get_secret_value(),
    }
    base_url = settings.effective_openai_base_url
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIEmbeddings(**kwargs)




async def ensure_index_exists(client: AsyncPinecone, settings: Settings) -> None:
    try:
        if await client.indexes.exists(settings.PINECONE_INDEX_NAME):
            return
        await client.indexes.create(
            name=settings.PINECONE_INDEX_NAME,
            dimension=settings.PINECONE_DIMENSION,
            metric=settings.PINECONE_METRIC,
            spec=ServerlessSpec(cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION),
        )
    except PineconeError as exc:
        raise VectorStoreError(f"No se pudo asegurar el indice de Pinecone: {exc}") from exc


async def get_index(client: AsyncPinecone, settings: Settings) -> AsyncIndex:
    return await client.index(name=settings.PINECONE_INDEX_NAME)


async def embed_texts(texts: list[str], embeddings_client: OpenAIEmbeddings) -> list[list[float]]:
    if not texts:
        return []
    try:
        return await embeddings_client.aembed_documents(texts)
    except OpenAIError as exc:
        raise EmbeddingGenerationError(f"Fallo del proveedor OpenAI al generar embeddings: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - Error de transporte o cliente no previsto en SDK
        raise EmbeddingGenerationError(f"Fallo la generacion de embeddings: {exc}") from exc


async def upsert_chunks(
    index: AsyncIndex,
    chunks: list[ChunkWithMetadata],
    embeddings: list[list[float]],
    settings: Settings,
) -> int:
    if len(chunks) != len(embeddings):
        raise PineconeUpsertError("La cantidad de chunks y de embeddings no coincide")

    vectors = [
        {
            "id": chunk.chunk_id,
            "values": vector,
            "metadata": {
                **{k: v for k, v in chunk.metadata.model_dump(mode="json").items() if v is not None},
                "text": chunk.text,
            },
        }
        for chunk, vector in zip(chunks, embeddings, strict=True)
    ]

    try:
        response = await index.upsert(
            vectors=vectors, batch_size=settings.PINECONE_UPSERT_BATCH_SIZE, show_progress=False
        )
    except PineconeError as exc:
        raise PineconeUpsertError(f"Fallo el upsert a Pinecone: {exc}") from exc

    return response.upserted_count


async def semantic_search(
    query: str,
    top_k: int,
    index: AsyncIndex,
    embeddings_client: OpenAIEmbeddings,
    metadata_filter: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    [query_vector] = await embed_texts([query], embeddings_client)

    try:
        result = await index.query(vector=query_vector, top_k=top_k, include_metadata=True, filter=metadata_filter)
    except PineconeError as exc:
        raise VectorStoreError(f"Fallo la busqueda semantica: {exc}") from exc

    chunks: list[RetrievedChunk] = []
    for match in result.matches:
        metadata = dict(match.metadata or {})
        text = metadata.pop("text", "")
        chunks.append(
            RetrievedChunk(chunk_id=match.id, score=match.score, text=text, metadata=ChunkMetadata(**metadata))
        )
    return chunks


async def lexical_search_postgres(
    query: str,
    top_k: int,
    sessionmaker: async_sessionmaker[AsyncSession] | None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[tuple[DocumentChunk, float]]:
    """Búsqueda léxica distribuida sobre PostgreSQL Full-Text Search (FTS).

    Utiliza to_tsvector('spanish', text), plainto_tsquery('spanish', query) y ts_rank_cd
    sin consumir memoria en el proceso del API ni requerir replicación de índices en RAM.
    """
    if sessionmaker is None or not query.strip():
        return []

    try:
        async with sessionmaker() as session:
            ts_query = func.websearch_to_tsquery("spanish", query)
            ts_vector = func.to_tsvector("spanish", DocumentChunk.text)
            rank = func.ts_rank_cd(ts_vector, ts_query).label("rank")

            stmt = select(DocumentChunk, rank).where(ts_vector.op("@@")(ts_query)).order_by(desc(rank)).limit(top_k)

            if metadata_filter:
                for key, value in metadata_filter.items():
                    if hasattr(DocumentChunk, key):
                        stmt = stmt.where(getattr(DocumentChunk, key) == value)

            result = await session.execute(stmt)
            rows = result.all()

            if not rows:
                ts_query_plain = func.plainto_tsquery("spanish", query)
                rank_plain = func.ts_rank_cd(ts_vector, ts_query_plain).label("rank")
                stmt_plain = (
                    select(DocumentChunk, rank_plain)
                    .where(ts_vector.op("@@")(ts_query_plain))
                    .order_by(desc(rank_plain))
                    .limit(top_k)
                )
                if metadata_filter:
                    for key, value in metadata_filter.items():
                        if hasattr(DocumentChunk, key):
                            stmt_plain = stmt_plain.where(getattr(DocumentChunk, key) == value)
                result_plain = await session.execute(stmt_plain)
                rows = result_plain.all()

            return [(row[0], float(row[1])) for row in rows]
    except Exception as exc:  # noqa: BLE001 - FTS resiliente; si Postgres FTS falla, hybrid_search cae a vectorial puro
        logger.warning("rag_service.postgres_fts_error", extra={"error": str(exc), "query": query})
        return []


async def hybrid_search(
    query: str,
    top_k: int,
    index: AsyncIndex,
    embeddings_client: OpenAIEmbeddings,
    sessionmaker: async_sessionmaker[AsyncSession] | None,
    settings: Settings,
    metadata_filter: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    """Combina vectorial (Pinecone) + léxico (Postgres FTS) por score ponderado.

    Utiliza LEXICAL_WEIGHT y VECTOR_WEIGHT configurables. No requiere un clasificador
    previo, evitando latencia adicional por consulta.
    """
    if not settings.HYBRID_RETRIEVAL_ENABLED or sessionmaker is None:
        return await semantic_search(query, top_k, index, embeddings_client, metadata_filter)

    candidate_k = top_k * settings.HYBRID_CANDIDATE_MULTIPLIER
    vector_results, fts_raw = await asyncio.gather(
        semantic_search(query, candidate_k, index, embeddings_client, metadata_filter),
        lexical_search_postgres(query, candidate_k, sessionmaker, metadata_filter),
    )

    max_fts = max((score for _, score in fts_raw), default=0.0)

    combined: dict[str, RetrievedChunk] = {
        chunk.chunk_id: RetrievedChunk(
            chunk_id=chunk.chunk_id,
            score=chunk.score * settings.VECTOR_WEIGHT,
            text=chunk.text,
            metadata=chunk.metadata,
        )
        for chunk in vector_results
    }

    for doc_chunk, raw_score in fts_raw:
        weighted = (raw_score / max_fts if max_fts > 0 else 0.0) * settings.LEXICAL_WEIGHT
        existing = combined.get(doc_chunk.chunk_id)
        if existing is not None:
            existing.score += weighted
        else:
            combined[doc_chunk.chunk_id] = RetrievedChunk(
                chunk_id=doc_chunk.chunk_id,
                score=weighted,
                text=doc_chunk.text,
                metadata=ChunkMetadata(
                    document_id=doc_chunk.document_id,
                    filename=doc_chunk.filename,
                    file_type=doc_chunk.file_type,
                    page=doc_chunk.page,
                    section=doc_chunk.section,
                    source=doc_chunk.source,
                ),
            )

    seen_texts: set[str] = set()
    deduped: list[RetrievedChunk] = []
    for chunk in sorted(combined.values(), key=lambda c: c.score, reverse=True):
        # Normaliza espacios y signos para detectar fragmentos redundantes si un documento se subió más de una vez
        norm_key = " ".join(chunk.text.split()[:40]).lower()
        if norm_key in seen_texts:
            continue
        seen_texts.add(norm_key)
        deduped.append(chunk)
        if len(deduped) >= top_k:
            break

    return deduped
