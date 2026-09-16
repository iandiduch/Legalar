"""Parsing/chunking de documentos (sin infra externa) + upsert/busqueda
contra un Pinecone mockeado (sin pegarle a la nube real ni gastar API keys).

test_list_ingestion_jobs_filters_by_status_and_file_type es la excepcion:
necesita Postgres real (usa el fixture db_session, que se salta solo via
pytest.skip si no hay Postgres disponible -- ver tests/conftest.py)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.exceptions import UnsupportedFileTypeError
from app.db.models_orm import IngestionJob
from app.domain.models import FileType, TaskStatus
from app.schemas.ingest import ChunkMetadata, ChunkWithMetadata
from app.services.ingestion_service import build_chunks, list_ingestion_jobs, parse_document
from app.services.rag_service import semantic_search, upsert_chunks


async def test_parse_txt_and_chunk(tmp_path, test_settings):
    path = tmp_path / "ley_ejemplo.txt"
    path.write_text("Articulo 1. " + ("El presente régimen legal regula esto. " * 100), encoding="utf-8")

    pages = await parse_document(path, FileType.TXT)
    assert len(pages) == 1

    chunks = build_chunks(pages, uuid4(), "ley_ejemplo.txt", FileType.TXT, test_settings)
    assert len(chunks) > 1
    assert all(c.metadata.filename == "ley_ejemplo.txt" for c in chunks)
    assert all(c.metadata.source == "ley_ejemplo.txt" for c in chunks)


async def test_parse_md_splits_by_heading(tmp_path):
    path = tmp_path / "plan.md"
    path.write_text("# Correlatividades\ncontenido uno\n\n# Regimen de examenes\ncontenido dos\n", encoding="utf-8")

    pages = await parse_document(path, FileType.MD)
    assert [p.section for p in pages] == ["Correlatividades", "Regimen de examenes"]


async def test_parse_document_unsupported_extension(tmp_path):
    path = tmp_path / "algo.xyz"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError):
        await parse_document(path, "xyz")  # type: ignore[arg-type]


async def test_upsert_chunks_batches_and_strips_none_metadata(test_settings):
    chunk = ChunkWithMetadata(
        chunk_id="doc_0",
        text="contenido",
        metadata=ChunkMetadata(
            document_id="doc", filename="a.txt", file_type=FileType.TXT, page=None, section=None, source="a.txt"
        ),
    )
    fake_index = AsyncMock()
    fake_index.upsert = AsyncMock(return_value=MagicMock(upserted_count=1))

    count = await upsert_chunks(fake_index, [chunk], [[0.1, 0.2]], test_settings)

    assert count == 1
    fake_index.upsert.assert_called_once()
    vectors = fake_index.upsert.call_args.kwargs["vectors"]
    assert "page" not in vectors[0]["metadata"]
    assert "section" not in vectors[0]["metadata"]
    assert vectors[0]["metadata"]["text"] == "contenido"


async def test_semantic_search_maps_matches_to_retrieved_chunks():
    fake_match = MagicMock(
        id="doc_0",
        score=0.87,
        metadata={"text": "hola", "document_id": "doc", "filename": "a.txt", "file_type": "txt", "source": "a.txt"},
    )
    fake_index = AsyncMock()
    fake_index.query = AsyncMock(return_value=MagicMock(matches=[fake_match]))

    fake_embeddings = AsyncMock()
    fake_embeddings.aembed_documents = AsyncMock(return_value=[[0.1, 0.2]])

    results = await semantic_search("hola", 5, fake_index, fake_embeddings)

    assert len(results) == 1
    assert results[0].text == "hola"
    assert results[0].score == 0.87


async def test_list_ingestion_jobs_filters_by_status_and_file_type(db_session):
    now = datetime.now(UTC)
    db_session.add_all(
        [
            IngestionJob(
                job_id=uuid4(),
                filename="reglamento.pdf",
                file_type=FileType.PDF.value,
                storage_path="data/uploads/a/document.pdf",
                status=TaskStatus.COMPLETED.value,
                created_at=now - timedelta(minutes=2),
            ),
            IngestionJob(
                job_id=uuid4(),
                filename="plan_sospechoso.docx",
                file_type=FileType.DOCX.value,
                storage_path="data/uploads/b/document.docx",
                status=TaskStatus.FAILED.value,
                error_message="rechazado: patrones sospechosos de prompt injection (ignore_instructions_en)",
                created_at=now - timedelta(minutes=1),
            ),
        ]
    )
    await db_session.commit()

    failed = await list_ingestion_jobs(db_session, status=TaskStatus.FAILED)
    assert [job.filename for job in failed] == ["plan_sospechoso.docx"]
    assert failed[0].error_message is not None and "prompt injection" in failed[0].error_message

    pdfs = await list_ingestion_jobs(db_session, file_type=FileType.PDF)
    assert [job.filename for job in pdfs] == ["reglamento.pdf"]

    all_jobs = await list_ingestion_jobs(db_session)
    assert [job.filename for job in all_jobs] == ["plan_sospechoso.docx", "reglamento.pdf"]  # mas reciente primero

    paginated = await list_ingestion_jobs(db_session, limit=1, offset=1)
    assert [job.filename for job in paginated] == ["reglamento.pdf"]
