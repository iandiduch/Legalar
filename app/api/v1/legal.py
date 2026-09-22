"""Endpoints de la API para el Agente Legal Argentino."""

import asyncio
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.legal.graph import build_legal_graph
from app.agents.legal.state import LegalAgentState
from app.core.config import Settings
from app.agents.legal.sanitizer import strip_leaked_metadata
from app.core.idempotency import (
    IDEMPOTENCY_HEADER,
    check_or_acquire_idempotency,
    generate_idempotency_fingerprint,
    release_idempotency_lock,
    save_idempotency_result,
)
from app.core.security import _client_ip, check_legal_rate_limit, require_scope
from app.db.session import get_db, get_sessionmaker
from app.db.models_orm import AnalysisJob
from app.domain.models import ApiKeyScope, ConfidenceLevel, FileType, QueryIntent
from app.schemas.legal.analysis import AnalysisJobResponse, AnalysisJobStatusResponse
from app.schemas.legal.chat import LegalChatRequest, LegalChatResponse, LegalValidationSummary
from app.schemas.legal.diff import DiffResponse
from app.services.ingestion_service import parse_document
from app.services.legal.ingestion_worker import LegalIngestionService
from app.services.legal.legalize_api_client import LegalizeApiClient
from app.services.legal.local_diff_engine import LocalGitDiffEngine
from app.services.legal.retriever import HybridLegalRetriever
from app.services.legal.secure_file_parser import (
    audit_prompt_injection,
    extract_document_text,
    validate_file_magic_bytes,
)
from app.services.llm_factory import build_chat_model
from app.services.rag_service import build_embeddings_client, build_pinecone_client


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/legal", tags=["Legal Agent"])


from app.api.dependencies import (
    get_legal_retriever,
    get_legalize_api_client,
    get_local_diff_engine,
    get_settings_dep as get_settings,
)


@router.post(
    "/chat",
    response_model=LegalChatResponse,
    summary="Consulta Jurídica Interactiva",
    description="Responde preguntas jurídicas fundamentadas en la legislación argentina vigente con citas de artículos.",
)
async def legal_chat_endpoint(
    request: LegalChatRequest,
    http_request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    retriever: Annotated[HybridLegalRetriever, Depends(get_legal_retriever)],
    diff_client: Annotated[LegalizeApiClient, Depends(get_legalize_api_client)],
    _rate_limit: Annotated[None, Depends(check_legal_rate_limit)] = None,
) -> LegalChatResponse:
    start_time = time.time()
    thread_id = request.thread_id or str(uuid.uuid4())

    # 1. Idempotencia y deduplicación con Redis (anti doble cobro de tokens)
    redis_client = getattr(http_request.app.state, "redis_client", None)
    idempotency_header = http_request.headers.get(IDEMPOTENCY_HEADER)
    idempotency_key = idempotency_header or generate_idempotency_fingerprint(
        client_ip=_client_ip(http_request),
        path=http_request.url.path,
        body_str=f"{request.query}|{thread_id}",
    )

    if settings.IDEMPOTENCY_ENABLED and redis_client:
        status_idem, cached_payload = await check_or_acquire_idempotency(
            redis=redis_client,
            idempotency_key=idempotency_key,
            ttl_seconds=settings.IDEMPOTENCY_TTL_SECONDS,
        )
        if status_idem == "COMPLETED" and cached_payload:
            return LegalChatResponse(**cached_payload)
        elif status_idem == "PROCESSING":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Una consulta idéntica ya se encuentra en procesamiento. Por favor aguarde.",
            )

    # 2. Reutilizar singleton de LLM de app.state en lugar de reconstruir en cada llamada
    llm = getattr(http_request.app.state, "llm_client", None) or build_chat_model(settings)

    app_graph = getattr(http_request.app.state, "compiled_graph", None) or build_legal_graph(checkpointer=None)

    initial_messages = []
    if request.history:
        for item in request.history[-6:]:
            role = item.get("role", "")
            content = item.get("content", "")
            if not content:
                continue
            if role == "user":
                initial_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                initial_messages.append(AIMessage(content=content))
    initial_messages.append(HumanMessage(content=request.query))

    initial_state: LegalAgentState = {
        "messages": initial_messages,
        "thread_id": thread_id,
        "query": request.query,
        "intent": QueryIntent.LEGAL_CONSULTATION,
        "citations": [],
        "draft_answer": None,
        "validation_result": None,
        "document_text": None,
        "document_type": None,
        "diff_request_params": None,
        "diff_result": None,
        "final_answer": None,
        "confidence": ConfidenceLevel.HIGH,
        "iteration": 0,
    }

    prompt_mgr = getattr(http_request.app.state, "prompt_manager", None)
    config = {
        "configurable": {
            "thread_id": thread_id,
            "llm_client": llm,
            "legal_retriever": retriever,
            "legalize_api_client": diff_client,
            "settings": settings,
            "prompt_manager": prompt_mgr,
        },
        "recursion_limit": settings.GRAPH_RECURSION_LIMIT,
    }

    try:
        final_state = await asyncio.wait_for(
            app_graph.ainvoke(initial_state, config=config),
            timeout=85.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="La consulta superó el tiempo máximo de procesamiento (85s). Por favor formule una pregunta más específica.",
        )
    except Exception as exc:
        if settings.IDEMPOTENCY_ENABLED and redis_client:
            await release_idempotency_lock(redis_client, idempotency_key)
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fallo en la ejecución del Agente Legal: {exc}",
        ) from exc

    duration = time.time() - start_time
    val_res = final_state.get("validation_result") or LegalValidationSummary(is_valid=True)

    chat_response = LegalChatResponse(
        thread_id=thread_id,
        query=request.query,
        intent=final_state.get("intent", QueryIntent.LEGAL_CONSULTATION),
        answer=strip_leaked_metadata(
            final_state.get("final_answer") or final_state.get("draft_answer") or "No se pudo generar respuesta."
        ),
        citations=final_state.get("citations", []),
        confidence=final_state.get("confidence", ConfidenceLevel.MEDIUM),
        validation=val_res,
        diff_data=final_state.get("diff_data"),
        processing_time_seconds=round(duration, 3),
    )

    if settings.IDEMPOTENCY_ENABLED and redis_client:
        await save_idempotency_result(
            redis=redis_client,
            idempotency_key=idempotency_key,
            payload=chat_response.model_dump(),
            ttl_seconds=settings.IDEMPOTENCY_TTL_SECONDS,
        )

    return chat_response


@router.post(
    "/chat/stream",
    summary="Consulta Jurídica con Streaming de Respuesta Final y Fases en Vivo",
    description="Ejecuta la orquestación multi-agente emitiendo fases de progreso en tiempo real y transmite la respuesta final validada mediante Server-Sent Events (SSE).",
)
async def legal_chat_stream_endpoint(
    request: LegalChatRequest,
    http_request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    retriever: Annotated[HybridLegalRetriever, Depends(get_legal_retriever)],
    diff_client: Annotated[LegalizeApiClient, Depends(get_legalize_api_client)],
    _rate_limit: Annotated[None, Depends(check_legal_rate_limit)] = None,
):
    start_time = time.time()
    thread_id = request.thread_id or str(uuid.uuid4())

    async def event_generator():
        # Comprobación de desconexión temprana antes de ejecutar operaciones
        if await http_request.is_disconnected():
            logger.info("Cliente desconectado antes de iniciar streaming (%s)", thread_id)
            return

        # 1. Enviar comentario SSE ': ping' inmediato para que proxies (Cloudflare/Dokploy/Nginx) abran el canal
        yield ": ping\n\n"

        # 2. Notificar fase inicial de enrutamiento al usuario
        yield f"event: status\ndata: {json.dumps({'stage': 'routing', 'message': 'Analizando consulta e identificando materia jurídica...'})}\n\n"

        llm = getattr(http_request.app.state, "llm_client", None) or build_chat_model(settings)
        app_graph = getattr(http_request.app.state, "compiled_graph", None) or build_legal_graph(checkpointer=None)

        initial_messages = []
        if request.history:
            for item in request.history[-6:]:
                role = item.get("role", "")
                content = item.get("content", "")
                if not content:
                    continue
                if role == "user":
                    initial_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    initial_messages.append(AIMessage(content=content))
        initial_messages.append(HumanMessage(content=request.query))

        initial_state: LegalAgentState = {
            "messages": initial_messages,
            "thread_id": thread_id,
            "query": request.query,
            "intent": QueryIntent.LEGAL_CONSULTATION,
            "citations": [],
            "draft_answer": None,
            "validation_result": None,
            "document_text": None,
            "document_type": None,
            "diff_request_params": None,
            "diff_result": None,
            "final_answer": None,
            "confidence": ConfidenceLevel.HIGH,
            "iteration": 0,
        }

        prompt_mgr = getattr(http_request.app.state, "prompt_manager", None)
        config = {
            "configurable": {
                "thread_id": thread_id,
                "llm_client": llm,
                "legal_retriever": retriever,
                "legalize_api_client": diff_client,
                "settings": settings,
                "prompt_manager": prompt_mgr,
            },
            "recursion_limit": settings.GRAPH_RECURSION_LIMIT,
        }

        event_queue: asyncio.Queue = asyncio.Queue()
        is_done = False
        final_state: dict = dict(initial_state)

        async def run_graph_task():
            try:
                async for chunk in app_graph.astream(initial_state, config=config, stream_mode="updates"):
                    for node_name, node_update in chunk.items():
                        final_state.update(node_update)

                        if node_name == "router":
                            intent = final_state.get("intent")
                            if intent == QueryIntent.VERSION_DIFF:
                                await event_queue.put(("status", {"stage": "diff", "message": "Cotejando versiones normativas en el repositorio legal..."}))
                            elif intent == QueryIntent.DOCUMENT_ANALYSIS:
                                await event_queue.put(("status", {"stage": "analyzing", "message": "Examinando validez y cláusulas contractuales..."}))
                            elif intent == QueryIntent.URL_FACT_CHECK:
                                await event_queue.put(("status", {"stage": "fact_check", "message": "Extrayendo contenido del enlace y cotejando con normativa..."}))
                            elif intent == QueryIntent.GENERAL_INQUIRY:
                                await event_queue.put(("status", {"stage": "greeting", "message": "Iniciando respuesta..."}))
                            else:
                                await event_queue.put(("status", {"stage": "retrieval", "message": "Consultando legislación y corpus normativo oficial..."}))

                        elif node_name == "legal_agent":
                            await event_queue.put(("status", {"stage": "validating", "message": "Auditando vigencia actual de leyes y citas oficiales..."}))

                        elif node_name == "document_analyzer":
                            await event_queue.put(("status", {"stage": "validating", "message": "Auditando vigencia y consistencia legal de las cláusulas..."}))

                await event_queue.put(("DONE", None))
            except Exception as exc:
                logger.error("Error en ejecución de astream del agente legal: %s", exc)
                await event_queue.put(("ERROR", exc))

        async def heartbeat_task():
            while not is_done:
                await asyncio.sleep(4.5)
                if not is_done:
                    await event_queue.put(("ping", None))

        graph_runner = asyncio.create_task(run_graph_task())
        pinger = asyncio.create_task(heartbeat_task())

        start_wait = time.time()
        max_duration = 85.0  # Ventana holgada de 85s para análisis exhaustivo y validación completa

        try:
            while True:
                elapsed = time.time() - start_wait
                remaining = max_duration - elapsed
                if remaining <= 0:
                    raise asyncio.TimeoutError("Timeout global superado")

                item_type, item_data = await asyncio.wait_for(event_queue.get(), timeout=min(remaining, 5.0))

                if await http_request.is_disconnected():
                    logger.info("Cliente desconectado durante ejecución del grafo (%s)", thread_id)
                    return

                if item_type == "ping":
                    yield ": ping\n\n"
                elif item_type == "status":
                    yield f"event: status\ndata: {json.dumps(item_data)}\n\n"
                elif item_type == "DONE":
                    break
                elif item_type == "ERROR":
                    raise item_data

        except asyncio.TimeoutError:
            logger.warning("Timeout de 85s superado en grafo legal para thread %s", thread_id)
            yield f"event: error\ndata: {json.dumps({'error': 'La consulta demoró más de lo esperado en resolverse exhaustivamente. Por favor reintenta con una formulación más específica.'})}\n\n"
            return
        except Exception as exc:
            logger.error("Error en flujo SSE del agente legal: %s", exc)
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            return
        finally:
            is_done = True
            pinger.cancel()
            if not graph_runner.done():
                graph_runner.cancel()

        if await http_request.is_disconnected():
            logger.info("Cliente desconectado tras finalización del grafo (%s)", thread_id)
            return

        final_answer = strip_leaked_metadata(
            final_state.get("final_answer")
            or final_state.get("draft_answer")
            or "No se pudo generar respuesta."
        )
        raw_citations = final_state.get("citations", [])
        citations_data = [
            c.model_dump() if hasattr(c, "model_dump") else dict(c)
            for c in raw_citations
        ]
        val_res = final_state.get("validation_result")
        val_dict = (
            val_res.model_dump()
            if hasattr(val_res, "model_dump")
            else ({"is_valid": True, "all_norms_in_force": True, "unsupported_claims": [], "warning_notes": []})
        )

        # 3. Notificar inicio de la respuesta final validada
        yield f"event: start\ndata: {json.dumps({'thread_id': thread_id})}\n\n"

        # 4. Transmitir tokens de la respuesta final
        tokens = re.findall(r"\S+|\s+", final_answer)
        for token in tokens:
            if await http_request.is_disconnected():
                logger.info("Cliente desconectado durante streaming de tokens (%s)", thread_id)
                return
            yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
            await asyncio.sleep(0.008)

        # 5. Notificar finalización con metadatos completos y citas oficiales
        duration = time.time() - start_time
        done_payload = {
            "thread_id": thread_id,
            "query": request.query,
            "intent": final_state.get("intent", QueryIntent.LEGAL_CONSULTATION),
            "answer": final_answer,
            "citations": citations_data,
            "confidence": final_state.get("confidence", ConfidenceLevel.MEDIUM),
            "validation": val_dict,
            "diff_data": final_state.get("diff_data"),
            "processing_time_seconds": round(duration, 3),
        }
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/analyze",
    response_model=AnalysisJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Auditoría de Documento Legal (Asíncrona)",
    description=(
        "Encola la auditoría de contratos, convenios o cartas documento (PDF/DOCX/TXT) para procesamiento en segundo plano. "
        "Responde inmediatamente con HTTP 202 Accepted y el analysis_id para seguimiento mediante SSE (/events) o polling."
    ),
)
async def analyze_document_endpoint(
    http_request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile | None = File(default=None),
    raw_text: str | None = Form(default=None),
    document_type: str = Form(default="contrato"),
    _rate_limit: Annotated[None, Depends(check_legal_rate_limit)] = None,
) -> AnalysisJobResponse:
    doc_content = ""
    display_filename = "texto_directo.txt"

    if file and file.filename:
        display_filename = Path(file.filename).name or "documento"
        file_bytes = await file.read()
        mime_type = validate_file_magic_bytes(
            file_bytes=file_bytes,
            filename=file.filename,
            max_size=settings.MAX_FILE_UPLOAD_SIZE,
        )
        doc_content = await extract_document_text(
            file_bytes=file_bytes,
            mime_type=mime_type,
            settings=settings,
        )
    elif raw_text:
        doc_content = raw_text.strip()
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Debe proporcionar un archivo (file) o el texto plano del documento (raw_text).",
        )

    if len(doc_content.strip()) < 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El documento es demasiado corto para realizar un análisis jurídico consistente.",
        )

    # Auditoría activa de seguridad contra Prompt Injection antes de encolar
    audit_prompt_injection(doc_content)

    analysis_id = uuid.uuid4()
    job = AnalysisJob(
        analysis_id=analysis_id,
        document_type=document_type,
        filename=display_filename,
        document_text=doc_content,
        status="PENDING",
    )
    db.add(job)
    await db.commit()

    redis_client = getattr(http_request.app.state, "redis_client", None)
    if redis_client:
        try:
            await redis_client.rpush(settings.REDIS_ANALYSIS_QUEUE_KEY, str(analysis_id))
        except Exception as exc:  # noqa: BLE001
            job.status = "FAILED"
            job.error_message = f"Fallo al encolar en Redis: {exc}"
            await db.commit()
            logger.error("analyze_endpoint.enqueue_failed", extra={"analysis_id": str(analysis_id), "error": str(exc)})
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="El servicio de colas de auditoría no se encuentra disponible temporalmente.",
            )

    return AnalysisJobResponse(
        analysis_id=str(analysis_id),
        status="PENDING",
        document_type=document_type,
        filename=display_filename,
        message="Auditoría encolada exitosamente para procesamiento en segundo plano.",
        events_url=f"/api/v1/legal/analyze/{analysis_id}/events",
    )


@router.get(
    "/analyze/{analysis_id}",
    response_model=AnalysisJobStatusResponse,
    summary="Consultar Estado de Auditoría Documental",
    description="Devuelve el estado actual de la auditoría (PENDING, PROCESSING, COMPLETED, FAILED) y el dictamen final una vez concluido.",
)
async def get_analysis_status(
    analysis_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    _rate_limit: Annotated[None, Depends(check_legal_rate_limit)] = None,
) -> AnalysisJobStatusResponse:
    job = await db.get(AnalysisJob, analysis_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Auditoría documental con ID {analysis_id} no encontrada.",
        )

    proc_time = None
    if job.completed_at and job.started_at:
        proc_time = round((job.completed_at - job.started_at).total_seconds(), 3)

    return AnalysisJobStatusResponse(
        analysis_id=str(job.analysis_id),
        status=job.status,
        document_type=job.document_type,
        filename=job.filename,
        error_message=job.error_message,
        result=job.result,
        processing_time_seconds=proc_time,
    )


@router.get(
    "/analyze/{analysis_id}/events",
    summary="Streaming de Eventos SSE de Auditoría",
    description="Transmite eventos en tiempo real (Server-Sent Events) sobre el avance y resultado final del análisis del documento.",
)
async def stream_analysis_events(
    analysis_id: uuid.UUID,
    http_request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    _rate_limit: Annotated[None, Depends(check_legal_rate_limit)] = None,
) -> StreamingResponse:
    job = await db.get(AnalysisJob, analysis_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Auditoría con ID {analysis_id} no encontrada.",
        )

    async def event_generator():
        # Si el job ya estaba finalizado al momento de la conexión
        if job.status == "COMPLETED" and job.result:
            yield f"event: done\ndata: {json.dumps(job.result)}\n\n"
            return
        elif job.status == "FAILED":
            yield f"event: error\ndata: {json.dumps({'detail': job.error_message or 'Fallo en la auditoría'})}\n\n"
            return

        redis_client = getattr(http_request.app.state, "redis_client", None)
        if not redis_client:
            yield f"event: error\ndata: {json.dumps({'detail': 'Broker Redis no configurado'})}\n\n"
            return

        pubsub_channel = f"legal:analysis:events:{analysis_id}"
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(pubsub_channel)

        yield f"event: status\ndata: {json.dumps({'stage': 'queued', 'message': 'Auditoría en cola de espera...'})}\n\n"

        max_wait_seconds = 180
        started_waiting = time.time()

        try:
            while True:
                if await http_request.is_disconnected():
                    break

                if time.time() - started_waiting > max_wait_seconds:
                    yield f"event: error\ndata: {json.dumps({'detail': 'Timeout esperando resultado del worker (180s)'})}\n\n"
                    break

                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=2.0)
                if message and message.get("type") == "message":
                    try:
                        raw_data = message.get("data")
                        if isinstance(raw_data, bytes):
                            raw_data = raw_data.decode("utf-8")
                        payload = json.loads(raw_data)
                        ev_type = payload.get("event", "status")

                        if ev_type == "status":
                            yield f"event: status\ndata: {json.dumps(payload)}\n\n"
                        elif ev_type == "done":
                            yield f"event: done\ndata: {json.dumps(payload.get('result', {}))}\n\n"
                            break
                        elif ev_type == "error":
                            yield f"event: error\ndata: {json.dumps({'detail': payload.get('message', 'Error en auditoría')})}\n\n"
                            break
                    except Exception as parse_err:
                        logger.warning("sse_parse_error: %s", parse_err)
                else:
                    # Ping keep-alive para mantener el socket SSE activo a través de proxies/Nginx
                    yield ": ping\n\n"
        finally:
            with suppress(Exception):
                await pubsub.unsubscribe(pubsub_channel)
                await pubsub.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/diff",
    response_model=DiffResponse,
    summary="Comparador Histórico de Reformas",
    description="Compara redacciones de una ley o artículo específico entre dos fechas o commits con fallback local Git transparente.",
)
async def legal_diff_endpoint(
    law_id: str = Query(..., description="Identificador de la norma (ej: 'LEY-26994', 'LEY-19550')"),
    article: str | None = Query(default=None, description="Artículo específico opcional (ej: '1198')"),
    date_a: str | None = Query(default=None, description="Fecha inicial (YYYY-MM-DD)"),
    date_b: str | None = Query(default=None, description="Fecha posterior (YYYY-MM-DD)"),
    api_client: LegalizeApiClient = Depends(get_legalize_api_client),
    _rate_limit: Annotated[None, Depends(check_legal_rate_limit)] = None,
) -> DiffResponse:
    res = await api_client.get_diff(
        law_identifier=law_id,
        date_a=date_a,
        date_b=date_b,
        article=article,
    )
    return res


@router.post(
    "/sync",
    summary="Sincronización Incremental de Leyes",
    description="Ejecuta la sincronización Git incremental sobre legalize-ar e indexa normas con hashing de artículos.",
    dependencies=[Depends(require_scope(ApiKeyScope.ADMIN))],
)
async def trigger_legal_sync(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    sessionmaker = getattr(request.app.state, "db_sessionmaker", None) or get_sessionmaker(settings)
    embeddings = getattr(request.app.state, "embeddings_client", None) or build_embeddings_client(settings)
    pinecone_c = getattr(request.app.state, "pinecone_client", None) or build_pinecone_client(settings)

    worker = LegalIngestionService(
        settings=settings,
        sessionmaker=sessionmaker,
        embeddings_client=embeddings,
        pinecone_client=pinecone_c,
    )

    sync_run = await worker.sync_incremental()
    return {
        "sync_id": str(sync_run.id),
        "status": sync_run.status,
        "files_added": sync_run.files_added,
        "files_modified": sync_run.files_modified,
        "articles_indexed": sync_run.articles_indexed,
        "articles_skipped_unchanged": sync_run.articles_skipped_unchanged,
        "duration_seconds": sync_run.duration_seconds,
    }
