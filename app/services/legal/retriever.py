"""Retriever Híbrido especializado para el ordenamiento jurídico argentino.

Combina Búsqueda Léxica en PostgreSQL (Full-Text Search con tsvector/ts_rank_cd sobre texto jerárquico
y match exacto de normas y artículos) con Búsqueda Densa Semántica en Pinecone.
Aplica Reciprocal Rank Fusion (RRF) para ordenar con máxima precisión los artículos recuperados.
"""

import asyncio
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
        self._index: AsyncIndex | None = None

    async def _get_index(self) -> AsyncIndex | None:
        """Retorna la instancia de AsyncIndex cacheada para evitar llamadas repetidas a control plane."""
        if not self.pinecone:
            return None
        if self._index is None:
            self._index = await self.pinecone.index(name=self.settings.PINECONE_INDEX_NAME)
        return self._index

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
        await self._enrich_full_text(fused)
        return fused

    async def retrieve_citations(
        self,
        query: str,
        top_k: int = 8,
        include_historical: bool = False,
        law_filter: list[str] | None = None,
    ) -> list[LegalCitation]:
        """Alias para search compatible con la tool modular de búsqueda jurídica (rag_tools)."""
        return await self.search(
            query=query,
            top_k=top_k,
            include_historical=include_historical,
            law_filter=law_filter,
        )

    async def retrieve_exact_articles(
        self,
        references: list[str | dict[str, str]],
        include_historical: bool = False,
    ) -> list[LegalCitation]:
        """Recupera artículos específicos por identificador de norma y número de artículo.

        Acepta strings en formato 'LEY-XXXX:NUM' (ej: 'LEY-11179:62') o diccionarios
        {'law_identifier': 'LEY-11179', 'article_number': '62'}.
        """
        if not references:
            return []

        parsed: list[tuple[str, str]] = []
        for ref in references:
            if isinstance(ref, str) and ":" in ref:
                parts = ref.strip().split(":", 1)
                parsed.append((parts[0].strip().upper(), parts[1].strip()))
            elif isinstance(ref, dict):
                law_id = ref.get("law_identifier", "").strip().upper()
                art_num = str(ref.get("article_number", "")).strip()
                if law_id and art_num:
                    parsed.append((law_id, art_num))

        if not parsed:
            return []

        citations: list[LegalCitation] = []

        # 1. Búsqueda exacta en PostgreSQL si está disponible
        if self.sessionmaker:
            try:
                conditions = [
                    (LegalArticle.law_identifier == law_id) & (LegalArticle.article_number == art_num)
                    for law_id, art_num in parsed
                ]
                async with self.sessionmaker() as session:
                    stmt = (
                        select(LegalArticle, LegalLaw)
                        .outerjoin(LegalLaw, LegalArticle.law_id == LegalLaw.id)
                        .where(or_(*conditions))
                    )
                    if not include_historical:
                        stmt = stmt.where(LegalArticle.status == "in_force")

                    res = await session.execute(stmt)
                    rows = res.all()

                    for art, law in rows:
                        raw_content = (art.content or "").strip()
                        quote = (
                            raw_content if len(raw_content) <= 4000
                            else raw_content[:4000].strip() + "\n... [artículo extenso truncado]"
                        )
                        citations.append(
                            LegalCitation(
                                law_identifier=art.law_identifier,
                                law_title=law.title if law else art.law_identifier,
                                article_number=art.article_number,
                                epigraph=art.epigraph,
                                exact_quote=quote,
                                status=art.status,
                                infoleg_url=law.source_url if law else None,
                                commit_sha=art.commit_sha,
                                confidence_score=1.0,
                            )
                        )
            except Exception as exc:
                logger.warning("Fallo en retrieve_exact_articles vía PostgreSQL: %s", exc)

        # 2. Si no hubo citas de BD o no hay BD, intentar vía Pinecone con metadata filter
        if not citations and self.pinecone and self.embeddings:
            try:
                index = await self._get_index()
                if index is not None:
                    for law_id, art_num in parsed:
                        [vec] = await embed_texts([f"{law_id} Artículo {art_num}"], self.embeddings)
                        filter_dict: dict[str, Any] = {
                            "law_identifier": law_id,
                            "article_number": art_num,
                        }
                        if not include_historical:
                            filter_dict["status"] = "in_force"

                        response = await index.query(
                            vector=vec,
                            top_k=1,
                            include_metadata=True,
                            namespace=self.namespace,
                            filter=filter_dict,
                        )
                        for match in response.matches:
                            meta = match.metadata or {}
                            content = (meta.get("text") or "").strip()
                            citations.append(
                                LegalCitation(
                                    law_identifier=meta.get("law_identifier", law_id),
                                    law_title=meta.get("law_title", law_id),
                                    article_number=meta.get("article_number", art_num),
                                    epigraph=meta.get("epigraph"),
                                    exact_quote=content[:4000],
                                    status=meta.get("status", "in_force"),
                                    infoleg_url=meta.get("infoleg_url"),
                                    commit_sha=meta.get("commit_sha"),
                                    confidence_score=1.0,
                                )
                            )
            except Exception as exc:
                logger.warning("Fallo en retrieve_exact_articles vía Pinecone: %s", exc)

        await self._enrich_full_text(citations)
        return citations

    async def search_multi_query(
        self,
        queries: list[str],
        top_k: int = 8,
        include_historical: bool = False,
        law_filter: list[str] | None = None,
    ) -> list[LegalCitation]:
        """Ejecuta múltiples búsquedas híbridas en paralelo para consultas relacionales.

        Fusiona los resultados de todas las sub-queries utilizando Reciprocal Rank Fusion (RRF),
        asegurando que tanto las figuras especiales como las reglas generales alcancen el ranking superior.
        """
        valid_queries = [q.strip() for q in queries if q and q.strip()]
        if not valid_queries:
            return []
        if len(valid_queries) == 1:
            return await self.search(valid_queries[0], top_k=top_k, include_historical=include_historical, law_filter=law_filter)

        tasks = [
            self.search(query=q, top_k=top_k, include_historical=include_historical, law_filter=law_filter)
            for q in valid_queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Fusión RRF entre los resultados de todas las sub-queries
        rrf_constant = 60
        scores: dict[str, float] = {}
        item_data: dict[str, LegalCitation] = {}

        for res in results:
            if isinstance(res, Exception) or not res:
                continue
            for rank_idx, cit in enumerate(res):
                key = f"{cit.law_identifier}:{cit.article_number}"
                score = 1.0 / (rrf_constant + rank_idx + 1)
                scores[key] = scores.get(key, 0.0) + score
                if key not in item_data:
                    item_data[key] = cit
                else:
                    existing = item_data[key]
                    if len(cit.exact_quote or "") > len(existing.exact_quote or ""):
                        item_data[key] = cit

        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]
        fused: list[LegalCitation] = []
        for k in sorted_keys:
            c = item_data[k]
            c.confidence_score = scores[k]
            fused.append(c)

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
            index = await self._get_index()
            if index is None:
                return []
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
        """Fusiona rankings léxicos y vectoriales usando Reciprocal Rank Fusion preservando texto completo."""
        scores: dict[str, float] = {}
        item_data: dict[str, LegalCitation] = {}

        # Procesar léxico
        for rank_idx, (art, law, _) in enumerate(lexical):
            key = f"{art.law_identifier}:{art.article_number}"
            score = 1.0 / (rrf_constant + rank_idx + 1)
            scores[key] = scores.get(key, 0.0) + score

            raw_content = (art.content or "").strip()
            content_quote = (
                raw_content if len(raw_content) <= 4000
                else raw_content[:4000].strip() + "\n... [artículo extenso truncado a 4000 caracteres]"
            )

            if key not in item_data:
                item_data[key] = LegalCitation(
                    law_identifier=art.law_identifier,
                    law_title=law.title if law else art.law_identifier,
                    article_number=art.article_number,
                    epigraph=art.epigraph,
                    exact_quote=content_quote,
                    status=art.status,
                    infoleg_url=law.source_url if law else None,
                    commit_sha=art.commit_sha,
                    confidence_score=score,
                )
            else:
                existing = item_data[key]
                if len(content_quote) > len(existing.exact_quote or ""):
                    existing.exact_quote = content_quote

        # Procesar denso
        for rank_idx, match in enumerate(dense):
            law_id = match.get("law_identifier", "")
            art_num = match.get("article_number", "")
            if not law_id or not art_num:
                continue
            key = f"{law_id}:{art_num}"
            score = 1.0 / (rrf_constant + rank_idx + 1)
            scores[key] = scores.get(key, 0.0) + score

            raw_dense = (match.get("content") or "").strip()
            dense_quote = (
                raw_dense if len(raw_dense) <= 4000
                else raw_dense[:4000].strip() + "\n... [artículo extenso truncado a 4000 caracteres]"
            )

            if key not in item_data:
                item_data[key] = LegalCitation(
                    law_identifier=law_id,
                    law_title=match.get("law_title") or law_id,
                    article_number=art_num,
                    epigraph=match.get("epigraph"),
                    exact_quote=dense_quote,
                    status=match.get("status", "in_force"),
                    infoleg_url=match.get("infoleg_url"),
                    commit_sha=match.get("commit_sha"),
                    confidence_score=score,
                )
            else:
                existing = item_data[key]
                if len(dense_quote) > len(existing.exact_quote or ""):
                    existing.exact_quote = dense_quote

        # Ordenar por score RRF acumulado
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]
        result: list[LegalCitation] = []
        for k in sorted_keys:
            cit = item_data[k]
            cit.confidence_score = scores[k]
            result.append(cit)

        return result

    async def _enrich_full_text(self, citations: list[LegalCitation]) -> None:
        """Enriquece las citas seleccionadas con el texto íntegro desde PostgreSQL."""
        if not self.sessionmaker or not citations:
            return

        to_enrich = [c for c in citations if not c.exact_quote or len(c.exact_quote) < 1200 or c.exact_quote.endswith("...")]
        if not to_enrich:
            return

        try:
            conditions = [
                (LegalArticle.law_identifier == c.law_identifier) & (LegalArticle.article_number == c.article_number)
                for c in to_enrich
            ]
            if not conditions:
                return

            async with self.sessionmaker() as session:
                stmt = select(
                    LegalArticle.law_identifier,
                    LegalArticle.article_number,
                    LegalArticle.content,
                ).where(or_(*conditions))
                res = await session.execute(stmt)
                db_rows = res.all()
                content_map = {f"{row[0]}:{row[1]}": row[2] for row in db_rows if row[2]}

                for cit in to_enrich:
                    key = f"{cit.law_identifier}:{cit.article_number}"
                    db_content = content_map.get(key)
                    if db_content and len(db_content.strip()) > len(cit.exact_quote or ""):
                        content_str = db_content.strip()
                        cit.exact_quote = (
                            content_str if len(content_str) <= 4000
                            else content_str[:4000].strip() + "\n... [artículo extenso truncado a 4000 caracteres]"
                        )
        except Exception as exc:
            logger.debug("No se pudo enriquecer texto completo desde BD: %s", exc)
