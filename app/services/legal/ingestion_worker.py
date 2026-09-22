"""Worker de ingesta legal selectiva con detección de cambios y hashing de artículos.

Compara content_hash (sha256 del contenido dispositivo) contra PostgreSQL para omitir
la re-generación de embeddings en artículos no alterados tras un `git pull`, ahorrando
cuota y tiempo de procesamiento.
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any

from langchain_openai import OpenAIEmbeddings
from pinecone import AsyncIndex, AsyncPinecone
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models_orm import LegalArticle, LegalLaw, LegalRevision, LegalSyncRun
from app.schemas.ingest import ChunkMetadata, ChunkWithMetadata
from app.services.legal.git_sync_service import GitSyncService
from app.services.legal.markdown_parser import ParsedLaw, parse_markdown_law
from app.services.rag_service import embed_texts

logger = logging.getLogger(__name__)


class LegalIngestionService:
    """Servicio de ingesta y sincronización de normas nacionales en PostgreSQL y Pinecone."""

    def __init__(
        self,
        settings: Settings,
        sessionmaker: async_sessionmaker[AsyncSession],
        embeddings_client: OpenAIEmbeddings,
        pinecone_client: AsyncPinecone,
    ) -> None:
        self.settings = settings
        self.sessionmaker = sessionmaker
        self.embeddings = embeddings_client
        self.pinecone = pinecone_client
        self.git_sync = GitSyncService(
            repo_path=settings.LEGALIZE_REPO_PATH,
            country_code=settings.LEGALIZE_COUNTRY_CODE,
        )
        self.namespace = settings.PINECONE_LEGAL_NAMESPACE

    async def ingest_law_file(
        self,
        relative_path: str,
        commit_sha: str | None = None,
    ) -> dict[str, int]:
        """Procesa una única norma desde el repositorio Markdown, aplicando hashing selectivo."""
        full_path = os.path.join(self.settings.LEGALIZE_REPO_PATH, relative_path)
        if not os.path.exists(full_path):
            logger.warning("El archivo %s no existe en el disco.", full_path)
            return {"indexed": 0, "skipped": 0}

        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            raw_content = f.read()

        law_data: ParsedLaw = parse_markdown_law(raw_content, os.path.basename(relative_path).replace(".md", ""))

        async with self.sessionmaker() as session:
            # 1. Upsert o actualización de LegalLaw
            stmt_law = select(LegalLaw).where(LegalLaw.identifier == law_data.identifier)
            res_law = await session.execute(stmt_law)
            existing_law = res_law.scalar_one_or_none()

            if not existing_law:
                law_record = LegalLaw(
                    identifier=law_data.identifier,
                    title=law_data.title,
                    country=law_data.country,
                    rank=law_data.rank,
                    status=law_data.status,
                    publication_date=law_data.publication_date,
                    last_updated=law_data.last_updated,
                    enactment_date=law_data.enactment_date,
                    department=law_data.department,
                    source_url=law_data.source,
                    infoleg_id=law_data.infoleg_id,
                    reform_quality=law_data.reform_quality,
                    commit_sha=commit_sha,
                    law_metadata=law_data.extra_metadata,
                )
                session.add(law_record)
                await session.flush()
                law_id = law_record.id
                existing_hashes: dict[str, str] = {}
            else:
                existing_law.title = law_data.title
                existing_law.status = law_data.status
                existing_law.last_updated = law_data.last_updated
                existing_law.commit_sha = commit_sha
                existing_law.law_metadata = law_data.extra_metadata
                law_id = existing_law.id

                # Obtener hashes existentes para esta ley
                stmt_arts = select(LegalArticle.article_number, LegalArticle.content_hash).where(
                    LegalArticle.law_id == law_id
                )
                res_arts = await session.execute(stmt_arts)
                existing_hashes = {row[0]: row[1] for row in res_arts.all()}

            articles_to_embed = []
            articles_to_update_orm = []
            indexed_count = 0
            skipped_count = 0

            # 2. Filtrar artículos modificados vs no modificados
            for art in law_data.articles:
                prev_hash = existing_hashes.get(art.article_number)
                if prev_hash == art.content_hash:
                    # El contenido no cambió -> omitir embedding
                    skipped_count += 1
                    continue

                pinecone_id = f"{law_data.identifier}:{art.article_number}"
                articles_to_embed.append((art, pinecone_id))
                indexed_count += 1

            # 3. Generar embeddings por lotes solo para los artículos cambiados
            if articles_to_embed:
                texts = [a[0].contextualized_text for a in articles_to_embed]
                embeddings_vectors = await embed_texts(texts, self.embeddings)

                index: AsyncIndex = await self.pinecone.index(name=self.settings.PINECONE_INDEX_NAME)

                vectors = []
                for (art, p_id), vector in zip(articles_to_embed, embeddings_vectors, strict=True):
                    vectors.append(
                        {
                            "id": p_id,
                            "values": vector,
                            "metadata": {
                                "law_identifier": law_data.identifier,
                                "law_title": law_data.title,
                                "article_number": art.article_number,
                                "epigraph": art.epigraph or "",
                                "status": art.status,
                                "rank": law_data.rank,
                                "country": law_data.country,
                                "infoleg_url": law_data.source or "",
                                "commit_sha": commit_sha or "",
                                "text": art.content_raw[:3500],  # Texto normativo amplio para evidencia
                            },
                        }
                    )

                # Upsert en Pinecone
                for b_start in range(0, len(vectors), self.settings.PINECONE_UPSERT_BATCH_SIZE):
                    batch = vectors[b_start : b_start + self.settings.PINECONE_UPSERT_BATCH_SIZE]
                    await index.upsert(vectors=batch, namespace=self.namespace)

                # 4. Actualizar o insertar artículos en PostgreSQL
                for art, p_id in articles_to_embed:
                    # Eliminar versión anterior si existía
                    await session.execute(
                        delete(LegalArticle).where(
                            LegalArticle.law_id == law_id,
                            LegalArticle.article_number == art.article_number,
                        )
                    )

                    art_orm = LegalArticle(
                        law_id=law_id,
                        law_identifier=law_data.identifier,
                        article_number=art.article_number,
                        article_order=art.article_order,
                        epigraph=art.epigraph,
                        content=art.content_raw,
                        text_searchable=art.contextualized_text,
                        book=art.hierarchy.book,
                        title_section=art.hierarchy.title_section,
                        chapter=art.hierarchy.chapter,
                        section=art.hierarchy.section,
                        status=art.status,
                        content_hash=art.content_hash,
                        pinecone_id=p_id,
                        commit_sha=commit_sha,
                        article_metadata={
                            "incisos": art.incisos,
                            "editorial_notes": art.editorial_notes,
                        },
                    )
                    session.add(art_orm)

            await session.commit()
            return {"indexed": indexed_count, "skipped": skipped_count}

    async def sync_incremental(self, target_commit: str = "HEAD") -> LegalSyncRun:
        """Sincroniza los cambios entre el último commit indexado y target_commit."""
        start_time = time.time()

        async with self.sessionmaker() as session:
            # Obtener último commit indexado
            stmt_last_sync = select(LegalSyncRun).order_by(LegalSyncRun.created_at.desc()).limit(1)
            res = await session.execute(stmt_last_sync)
            last_run = res.scalar_one_or_none()
            last_commit = last_run.target_commit if last_run else None

        # Fetch upstream si es posible
        self.git_sync.fetch_upstream()
        current_head = self.git_sync.get_current_head()

        changes = self.git_sync.detect_changes(last_commit, target_commit)
        files_to_process = [*changes["added"], *changes["modified"]]

        total_indexed = 0
        total_skipped = 0

        for file_path in files_to_process:
            try:
                counts = await self.ingest_law_file(file_path, commit_sha=current_head)
                total_indexed += counts["indexed"]
                total_skipped += counts["skipped"]
            except Exception as exc:
                logger.error("Fallo al ingestar %s: %s", file_path, exc)

        duration = time.time() - start_time

        sync_record = LegalSyncRun(
            sync_source="git_local",
            start_commit=last_commit,
            target_commit=current_head,
            files_added=len(changes["added"]),
            files_modified=len(changes["modified"]),
            articles_indexed=total_indexed,
            articles_skipped_unchanged=total_skipped,
            duration_seconds=duration,
            status="COMPLETED",
        )

        async with self.sessionmaker() as session:
            session.add(sync_record)
            await session.commit()

        return sync_record
