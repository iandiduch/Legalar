"""Retriever Híbrido especializado para el ordenamiento jurídico argentino.

Combina Búsqueda Léxica en PostgreSQL (Full-Text Search con tsvector/ts_rank_cd sobre texto jerárquico
y match exacto de normas y artículos) con Búsqueda Densa Semántica en Pinecone.
Aplica Reciprocal Rank Fusion (RRF) para ordenar con máxima precisión los artículos recuperados.
"""

import logging
from typing import Any

from langchain_openai import OpenAIEmbeddings
from pinecone import AsyncIndex, AsyncPinecone, PineconeError
from sqlalchemy import desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.exceptions import VectorStoreError
from app.db.models_orm import LegalArticle, LegalLaw
from app.schemas.legal.citation import LegalCitation
from app.services.rag_service import embed_texts

logger = logging.getLogger(__name__)


class HybridLegalRetriever:
    """Motor de búsqueda híbrida (Léxica + Densa) con ranking RRF para artículos legales."""

    def __init__(
        self,
        settings: Settings,
        pinecone_client: AsyncPinecone | None,
        embeddings_client: OpenAIEmbeddings | None,
        sessionmaker: async_sessionmaker[AsyncSession] | None,
    ) -> None:
        self.settings = settings
        self.pinecone = pinecone_client
        self.embeddings = embeddings_client
        self.sessionmaker = sessionmaker
        self.namespace = settings.PINECONE_LEGAL_NAMESPACE

    async def search(
        self,
        query: str,
        top_k: int = 8,
        include_historical: bool = False,
        law_filter: list[str] | None = None,
    ) -> list[LegalCitation]:
        """Ejecuta búsqueda híbrida paralela en PostgreSQL y Pinecone, fusionando con RRF."""
        lexical_results = await self._search_lexical_postgres(
            query=query,
            top_k=top_k * 2,
            include_historical=include_historical,
            law_filter=law_filter,
        )

        dense_results = await self._search_dense_pinecone(
            query=query,
            top_k=top_k * 2,
            include_historical=include_historical,
            law_filter=law_filter,
        )

        fused = self._reciprocal_rank_fusion(lexical_results, dense_results, top_k=top_k)
        return fused

    async def _search_lexical_postgres(
        self,
        query: str,
        top_k: int,
        include_historical: bool,
        law_filter: list[str] | None,
    ) -> list[tuple[LegalArticle, LegalLaw, float]]:
        """Búsqueda de texto completo y de precisión exacta en PostgreSQL."""
        if not self.sessionmaker or not query.strip():
            return []

        try:
            async with self.sessionmaker() as session:
                ts_query = func.websearch_to_tsquery("spanish", query)
                ts_vector = func.to_tsvector("spanish", LegalArticle.text_searchable)
                fts_rank = func.ts_rank_cd(ts_vector, ts_query).label("fts_rank")

                stmt = (
                    select(LegalArticle, LegalLaw, fts_rank)
                    .join(LegalLaw, LegalArticle.law_id == LegalLaw.id)
                    .where(ts_vector.op("@@")(ts_query))
                )

                if not include_historical:
                    stmt = stmt.where(LegalArticle.status == "in_force")

                if law_filter:
                    stmt = stmt.where(LegalArticle.law_identifier.in_(law_filter))

                stmt = stmt.order_by(desc(fts_rank)).limit(top_k)
                result = await session.execute(stmt)
                rows = result.all()

                return [(row[0], row[1], float(row[2])) for row in rows]
        except Exception as exc:
            logger.warning("Fallo en búsqueda léxica PostgreSQL: %s", exc)
            return []

    async def _search_dense_pinecone(
        self,
        query: str,
        top_k: int,
        include_historical: bool,
        law_filter: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Búsqueda semántica densa sobre el namespace de legislación en Pinecone."""
        if not self.pinecone or not self.embeddings:
            return []

        try:
            index: AsyncIndex = await self.pinecone.index(name=self.settings.PINECONE_INDEX_NAME)
            [query_vector] = await embed_texts([query], self.embeddings)

            filter_dict: dict[str, Any] = {}
            if not include_historical:
                filter_dict["status"] = "in_force"
            if law_filter:
                filter_dict["law_identifier"] = {"$in": law_filter}

            response = await index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace,
                filter=filter_dict or None,
            )

            results = []
            for match in response.matches:
                meta = match.metadata or {}
                results.append(
                    {
                        "id": match.id,
                        "score": match.score,
                        "law_identifier": meta.get("law_identifier"),
                        "law_title": meta.get("law_title"),
                        "article_number": meta.get("article_number"),
                        "epigraph": meta.get("epigraph"),
                        "content": meta.get("text"),
                        "status": meta.get("status", "in_force"),
                        "infoleg_url": meta.get("infoleg_url"),
                        "commit_sha": meta.get("commit_sha"),
                    }
                )
            return results
        except Exception as exc:
            logger.warning("Fallo en búsqueda semántica Pinecone: %s", exc)
            return []

    def _reciprocal_rank_fusion(
        self,
        lexical: list[tuple[LegalArticle, LegalLaw, float]],
        dense: list[dict[str, Any]],
        top_k: int,
        rrf_constant: int = 60,
    ) -> list[LegalCitation]:
        """Fusiona rankings léxicos y vectoriales usando Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        item_data: dict[str, LegalCitation] = {}

        # Procesar léxico
        for rank_idx, (art, law, _) in enumerate(lexical):
            key = f"{art.law_identifier}:{art.article_number}"
            score = 1.0 / (rrf_constant + rank_idx + 1)
            scores[key] = scores.get(key, 0.0) + score

            if key not in item_data:
                item_data[key] = LegalCitation(
                    law_identifier=art.law_identifier,
                    law_title=law.title if law else art.law_identifier,
                    article_number=art.article_number,
                    epigraph=art.epigraph,
                    exact_quote=art.content[:300] + ("..." if len(art.content) > 300 else ""),
                    status=art.status,
                    infoleg_url=law.source_url if law else None,
                    commit_sha=art.commit_sha,
                    confidence_score=score,
                )

        # Procesar denso
        for rank_idx, match in enumerate(dense):
            law_id = match.get("law_identifier", "")
            art_num = match.get("article_number", "")
            if not law_id or not art_num:
                continue
            key = f"{law_id}:{art_num}"
            score = 1.0 / (rrf_constant + rank_idx + 1)
            scores[key] = scores.get(key, 0.0) + score

            if key not in item_data:
                item_data[key] = LegalCitation(
                    law_identifier=law_id,
                    law_title=match.get("law_title") or law_id,
                    article_number=art_num,
                    epigraph=match.get("epigraph"),
                    exact_quote=(match.get("content") or "")[:300] + "...",
                    status=match.get("status", "in_force"),
                    infoleg_url=match.get("infoleg_url"),
                    commit_sha=match.get("commit_sha"),
                    confidence_score=score,
                )

        # Ordenar por score RRF acumulado
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]
        result: list[LegalCitation] = []
        for k in sorted_keys:
            cit = item_data[k]
            cit.confidence_score = scores[k]
            result.append(cit)

        return result
