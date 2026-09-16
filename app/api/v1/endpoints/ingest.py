"""POST /ingest (multipart), GET /ingest/status/{job_id} y GET /ingest (listado).
El endpoint solo se ocupa de HTTP/multipart (leer el archivo, escribirlo a disco,
validar tamano/extension) y de mapear filas ORM a schema de respuesta; crear el
registro del job, encolarlo y listar/filtrar jobs es logica de servicio (ver
app/services/ingestion_service.py). El procesamiento pesado (parseo/chunking/
embeddings) corre en el worker separado (app/services/ingestion_worker.py),
nunca dentro de esta request.

Todo el router exige scope admin (ver app/api/v1/router.py) -- incluye el
listado, no solo la carga: quien puede auditar que se subio o se rechazo es
el mismo conjunto de confianza que puede escribir en la base documental."""

from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Query, UploadFile

from app.api.dependencies import DbSessionDep, RedisDep, SettingsDep
from app.core.exceptions import (
    BusinessValidationError,
    JobNotFoundError,
    UnsupportedFileTypeError,
)
from app.db.models_orm import IngestionJob
from app.domain.models import FileType, TaskStatus
from app.schemas.ingest import IngestJobStatusResponse, IngestUploadResponse
from app.services.ingestion_service import create_ingestion_job, list_ingestion_jobs

router = APIRouter()

_EXTENSION_TO_FILE_TYPE = {"pdf": FileType.PDF, "docx": FileType.DOCX, "txt": FileType.TXT, "md": FileType.MD}
_READ_CHUNK_BYTES = 1024 * 1024


@router.post(
    "/ingest",
    response_model=IngestUploadResponse,
    status_code=202,
    summary="Cargar documento para ingesta RAG",
    description="Sube un archivo institucional (.pdf, .docx, .txt, .md), valida su tamaño y extensión, y encola un job asíncrono en Redis para parseo, chunking y generación de embeddings.",
    response_description="Confirmación del job de ingesta creado con su UUID para seguimiento.",
)
async def ingest_document(
    file: UploadFile, db: DbSessionDep, redis: RedisDep, settings: SettingsDep
) -> IngestUploadResponse:
    raw_filename = file.filename or ""
    display_filename = Path(raw_filename).name or str(uuid4())
    extension = display_filename.rsplit(".", 1)[-1].lower() if "." in display_filename else ""
    file_type = _EXTENSION_TO_FILE_TYPE.get(extension)
    if file_type is None:
        raise UnsupportedFileTypeError(f"Extension no soportada: .{extension or '?'}")

    job_id = uuid4()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    upload_dir = Path(settings.UPLOAD_DIR).resolve() / str(job_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    storage_path = upload_dir / f"document.{extension}"

    size = 0
    with storage_path.open("wb") as buffer:
        while chunk := await file.read(_READ_CHUNK_BYTES):
            size += len(chunk)
            if size > max_bytes:
                buffer.close()
                storage_path.unlink(missing_ok=True)
                raise BusinessValidationError(f"El archivo supera el limite de {settings.MAX_UPLOAD_SIZE_MB}MB")
            buffer.write(chunk)

    job = await create_ingestion_job(
        db, redis, settings, job_id=job_id, filename=display_filename, file_type=file_type, storage_path=storage_path
    )
    return IngestUploadResponse(job_id=job.job_id, status=TaskStatus(job.status), filename=job.filename)


@router.get(
    "/ingest/status/{job_id}",
    response_model=IngestJobStatusResponse,
    summary="Consultar estado de un job de ingesta",
    description="Devuelve el estado actual de procesamiento (PENDING, PROCESSING, COMPLETED, FAILED), cantidad de chunks indexados y errores si los hubiera.",
    response_description="Detalles y métricas del job de ingesta solicitado.",
)
async def get_ingest_status(job_id: UUID, db: DbSessionDep) -> IngestJobStatusResponse:
    job = await db.get(IngestionJob, job_id)
    if job is None:
        raise JobNotFoundError(f"Job {job_id} no encontrado")
    return _to_status_response(job)


@router.get(
    "/ingest",
    response_model=list[IngestJobStatusResponse],
    summary="Listar jobs de ingesta",
    description=(
        "Lista los jobs de ingesta documental, mas recientes primero, con filtros opcionales por estado y "
        "tipo de archivo. Pensado para que un admin audite que se subio (y que se rechazo, por ejemplo por "
        "el content-scanner de prompt injection) sin tener que guardar cada job_id devuelto por POST /ingest."
    ),
    response_description="Listado paginado de jobs de ingesta.",
)
async def list_ingest_jobs(
    db: DbSessionDep,
    status: Annotated[TaskStatus | None, Query()] = None,
    file_type: Annotated[FileType | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[IngestJobStatusResponse]:
    jobs = await list_ingestion_jobs(db, status=status, file_type=file_type, limit=limit, offset=offset)
    return [_to_status_response(job) for job in jobs]


def _to_status_response(job: IngestionJob) -> IngestJobStatusResponse:
    return IngestJobStatusResponse(
        job_id=job.job_id,
        status=TaskStatus(job.status),
        filename=job.filename,
        file_type=FileType(job.file_type),
        chunks_indexed=job.chunks_indexed,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )
