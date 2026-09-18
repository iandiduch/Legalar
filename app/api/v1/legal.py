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
from app.core.idempotency import (
    IDEMPOTENCY_HEADER,
    check_or_acquire_idempotency,
    generate_idempotency_fingerprint,
    release_idempotency_lock,
    save_idempotency_result,
)
from app.core.security import _client_ip, check_legal_rate_limit, require_scope
from app.db.session import get_db, get_sessionmaker
from app.domain.models import ApiKeyScope, ConfidenceLevel, FileType, QueryIntent
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

    config = {
        "configurable": {
            "thread_id": thread_id,
            "llm_client": llm,
            "legal_retriever": retriever,
            "legalize_api_client": diff_client,
            "settings": settings,
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
        answer=final_state.get("final_answer") or final_state.get("draft_answer") or "No se pudo generar respuesta.",
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

        config = {
            "configurable": {
                "thread_id": thread_id,
                "llm_client": llm,
                "legal_retriever": retriever,
                "legalize_api_client": diff_client,
                "settings": settings,
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

        final_answer = (
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
    response_model=LegalChatResponse,
    summary="Auditoría de Documento Legal",
    description="Analiza contratos, convenios o cartas documento (PDF/DOCX/TXT) y detecta cláusulas abusivas o nulas según la ley argentina.",
)
async def analyze_document_endpoint(
    http_request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    retriever: Annotated[HybridLegalRetriever, Depends(get_legal_retriever)],
    diff_client: Annotated[LegalizeApiClient, Depends(get_legalize_api_client)],
    file: UploadFile | None = File(default=None),
    raw_text: str | None = Form(default=None),
    document_type: str = Form(default="contrato"),
    _rate_limit: Annotated[None, Depends(check_legal_rate_limit)] = None,
) -> LegalChatResponse:
    doc_content = ""
    if file and file.filename:
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

    # Auditoría activa de seguridad contra Prompt Injection antes de llegar al LLM
    audit_prompt_injection(doc_content)

    # Control de idempotencia para evitar doble análisis de contratos pesados
    redis_client = getattr(http_request.app.state, "redis_client", None)
    idempotency_header = http_request.headers.get(IDEMPOTENCY_HEADER)
    idempotency_key = idempotency_header or generate_idempotency_fingerprint(
        client_ip=_client_ip(http_request),
        path=http_request.url.path,
        body_str=f"{document_type}|{doc_content[:500]}",
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
                detail="Una auditoría para este documento ya se encuentra en procesamiento. Por favor aguarde.",
            )

    req = LegalChatRequest(
        query=f"Auditar el siguiente {document_type} a la luz del derecho argentino:",
        thread_id=str(uuid.uuid4()),
    )

    start_time = time.time()
    llm = getattr(http_request.app.state, "llm_client", None) or build_chat_model(settings)

    app_graph = getattr(http_request.app.state, "compiled_graph", None) or build_legal_graph(checkpointer=None)

    initial_state: LegalAgentState = {
        "messages": [HumanMessage(content=req.query)],
        "thread_id": req.thread_id or str(uuid.uuid4()),
        "query": req.query,
        "intent": QueryIntent.DOCUMENT_ANALYSIS,
        "citations": [],
        "draft_answer": None,
        "validation_result": None,
        "document_text": doc_content,
        "document_type": document_type,
        "diff_request_params": None,
        "diff_result": None,
        "final_answer": None,
        "confidence": ConfidenceLevel.HIGH,
        "iteration": 0,
    }

    config = {
        "configurable": {
            "thread_id": req.thread_id,
            "llm_client": llm,
            "legal_retriever": retriever,
            "legalize_api_client": diff_client,
            "settings": settings,
        },
        "recursion_limit": settings.GRAPH_RECURSION_LIMIT,
    }

    try:
        final_state = await asyncio.wait_for(
            app_graph.ainvoke(initial_state, config=config),
            timeout=85.0,
        )
    except asyncio.TimeoutError:
        logger.warning("Timeout de 85s superado en auditoría de documento (%s)", req.thread_id)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="La auditoría del documento excedió el tiempo límite máximo de 85 segundos. Por favor reintente con una sección más acotada.",
        )
    except Exception as exc:
        if settings.IDEMPOTENCY_ENABLED and redis_client:
            await release_idempotency_lock(redis_client, idempotency_key)
        logger.error("Error durante auditoría de documento: %s", exc)
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el análisis del documento: {str(exc)}",
        )

    duration = time.time() - start_time
    val_res = final_state.get("validation_result") or LegalValidationSummary(is_valid=True)

    analysis_response = LegalChatResponse(
        thread_id=req.thread_id or str(uuid.uuid4()),
        query=req.query,
        intent=QueryIntent.DOCUMENT_ANALYSIS,
        answer=final_state.get("final_answer") or final_state.get("draft_answer") or "Análisis completado.",
        citations=final_state.get("citations", []),
        confidence=final_state.get("confidence", ConfidenceLevel.HIGH),
        validation=val_res,
        processing_time_seconds=round(duration, 3),
    )

    if settings.IDEMPOTENCY_ENABLED and redis_client:
        await save_idempotency_result(
            redis=redis_client,
            idempotency_key=idempotency_key,
            payload=analysis_response.model_dump(),
            ttl_seconds=settings.IDEMPOTENCY_TTL_SECONDS,
        )

    return analysis_response


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
