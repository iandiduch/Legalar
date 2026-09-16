from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.domain.models import FileType, TaskStatus


class IngestUploadResponse(BaseModel):
    job_id: UUID
    status: TaskStatus
    filename: str


class IngestJobStatusResponse(BaseModel):
    job_id: UUID
    status: TaskStatus
    filename: str
    file_type: FileType
    chunks_indexed: int | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ChunkMetadata(BaseModel):
    """Metadata minima por chunk."""

    document_id: str
    filename: str
    file_type: FileType
    page: int | None = None
    section: str | None = None
    source: str


class ChunkWithMetadata(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    chunk_id: str
    score: float
    text: str
    metadata: ChunkMetadata


class PageContent(BaseModel):
    page: int | None = None
    section: str | None = None
    text: str = Field(..., min_length=1)
