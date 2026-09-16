"""ORM de las tablas propias del Sistema Legal Argentino.

Incluye modelos para legislación nacional versionada (LegalLaw, LegalArticle, LegalRevision, LegalSyncRun),
jobs de ingesta y procesamiento de documentos de usuarios (IngestionJob, DocumentChunk),
prompts de agentes (AgentPrompt) y llaves de acceso (ApiKey).
"""

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class LegalLaw(Base):
    """Metadatos de una norma consolidada proveniente de legalize-ar / InfoLEG."""

    __tablename__ = "legal_laws"
    __table_args__ = (
        CheckConstraint(
            "status IN ('in_force', 'repealed', 'partially_repealed', 'annulled', 'expired')",
            name="ck_legal_laws_status",
        ),
        Index("ix_legal_laws_status", "status"),
        Index("ix_legal_laws_rank", "rank"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    identifier: Mapped[str] = mapped_column(String(100), unique=True, index=True)  # ej: "LEY-26994", "DNU-70-2023"
    title: Mapped[str] = mapped_column(Text)
    country: Mapped[str] = mapped_column(String(10), default="ar")
    rank: Mapped[str] = mapped_column(String(50))  # ley, decreto, dnu, decreto_ley, constitucion, resolucion
    status: Mapped[str] = mapped_column(String(30), default="in_force")
    publication_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    last_updated: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    enactment_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    department: Mapped[str | None] = mapped_column(String(255), default=None)
    source_url: Mapped[str | None] = mapped_column(Text, default=None)
    infoleg_id: Mapped[str | None] = mapped_column(String(50), default=None)
    reform_quality: Mapped[str | None] = mapped_column(String(30), default=None)  # clean, partial, bootstrap-only
    commit_sha: Mapped[str | None] = mapped_column(String(64), default=None)
    law_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class LegalArticle(Base):
    """Artículos individuales para Búsqueda Léxica FTS en PostgreSQL y enlace semántico con Pinecone."""

    __tablename__ = "legal_articles"
    __table_args__ = (
        Index("ix_legal_articles_fts", text("to_tsvector('spanish', text_searchable)"), postgresql_using="gin"),
        Index("ix_legal_articles_law_number", "law_identifier", "article_number", unique=True),
        Index("ix_legal_articles_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    law_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), index=True)
    law_identifier: Mapped[str] = mapped_column(String(100), index=True)  # "LEY-26994"
    article_number: Mapped[str] = mapped_column(String(50), index=True)   # "1198", "11 bis"
    article_order: Mapped[int] = mapped_column(Integer, default=0)
    epigraph: Mapped[str | None] = mapped_column(Text, default=None)
    content: Mapped[str] = mapped_column(Text)
    text_searchable: Mapped[str] = mapped_column(Text)

    # Jerarquía normativa (breadcrumbs)
    book: Mapped[str | None] = mapped_column(String(255), default=None)
    title_section: Mapped[str | None] = mapped_column(String(255), default=None)
    chapter: Mapped[str | None] = mapped_column(String(255), default=None)
    section: Mapped[str | None] = mapped_column(String(255), default=None)

    status: Mapped[str] = mapped_column(String(30), default="in_force")
    content_hash: Mapped[str] = mapped_column(String(64), index=True)  # sha256 del contenido
    pinecone_id: Mapped[str] = mapped_column(String(128), index=True)
    commit_sha: Mapped[str | None] = mapped_column(String(64), default=None)
    article_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class LegalRevision(Base):
    """Trazabilidad de reformas históricas extraídas de git log o Legalize API."""

    __tablename__ = "legal_revisions"
    __table_args__ = (
        Index("ix_legal_revisions_law_commit", "law_identifier", "commit_sha", unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    law_identifier: Mapped[str] = mapped_column(String(100), index=True)
    commit_sha: Mapped[str] = mapped_column(String(64), index=True)
    commit_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    change_type: Mapped[str] = mapped_column(String(50))  # bootstrap, reform, metadata_update
    source_id: Mapped[str | None] = mapped_column(String(50), default=None)
    source_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    affected_articles: Mapped[list] = mapped_column(JSONB, default=list)
    commit_message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LegalSyncRun(Base):
    """Auditoría de ejecuciones de sincronización del repositorio."""

    __tablename__ = "legal_sync_runs"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sync_source: Mapped[str] = mapped_column(String(50))  # git_local, git_pull, legalize_api
    start_commit: Mapped[str | None] = mapped_column(String(64), default=None)
    target_commit: Mapped[str | None] = mapped_column(String(64), default=None)
    files_added: Mapped[int] = mapped_column(Integer, default=0)
    files_modified: Mapped[int] = mapped_column(Integer, default=0)
    articles_indexed: Mapped[int] = mapped_column(Integer, default=0)
    articles_skipped_unchanged: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float] = mapped_column(default=0.0)
    status: Mapped[str] = mapped_column(String(30), default="COMPLETED")
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class IngestionJob(Base):
    """Trabajo de ingesta de archivos de usuarios para análisis jurídico."""

    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        CheckConstraint("file_type IN ('pdf','docx','txt','md')", name="ck_ingestion_jobs_file_type"),
        CheckConstraint("status IN ('PENDING','PROCESSING','COMPLETED','FAILED')", name="ck_ingestion_jobs_status"),
        Index("ix_ingestion_jobs_status", "status"),
        Index("ix_ingestion_jobs_created_at", "created_at"),
    )

    job_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str]
    file_type: Mapped[str]
    storage_path: Mapped[str]
    status: Mapped[str] = mapped_column(default="PENDING")
    retry_count: Mapped[int] = mapped_column(default=0)
    chunks_indexed: Mapped[int | None] = mapped_column(default=None)
    error_message: Mapped[str | None] = mapped_column(default=None)
    job_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class DocumentChunk(Base):
    """Copia del texto de cada chunk y metadatos de documentos cargados por usuarios para FTS."""

    __tablename__ = "document_chunks"
    __table_args__ = (
        CheckConstraint("file_type IN ('pdf','docx','txt','md')", name="ck_document_chunks_file_type"),
        Index("ix_document_chunks_fts", text("to_tsvector('spanish', text)"), postgresql_using="gin"),
        Index("ix_document_chunks_document_id", "document_id"),
    )

    chunk_id: Mapped[str] = mapped_column(primary_key=True)
    document_id: Mapped[str]
    filename: Mapped[str]
    file_type: Mapped[str]
    page: Mapped[int | None] = mapped_column(default=None)
    section: Mapped[str | None] = mapped_column(default=None)
    source: Mapped[str]
    text: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentPrompt(Base):
    __tablename__ = "agent_prompts"

    agent_id: Mapped[str] = mapped_column(primary_key=True)
    content: Mapped[str]
    version: Mapped[int] = mapped_column(default=1)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    updated_by: Mapped[str | None] = mapped_column(default=None)


class ApiKey(Base):
    __tablename__ = "api_keys"
    __table_args__ = (
        CheckConstraint("scope IN ('client','admin')", name="ck_api_keys_scope"),
        Index("ix_api_keys_is_active", "is_active"),
    )

    key_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_hash: Mapped[str] = mapped_column(unique=True)
    name: Mapped[str]
    scope: Mapped[str]
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    created_by: Mapped[str | None] = mapped_column(default=None)
