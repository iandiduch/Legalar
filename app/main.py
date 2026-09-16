"""Entrypoint FastAPI: lifespan, middleware, exception handlers, routers y métricas.

Todo lo compartido y costoso de inicializar se construye UNA vez en el lifespan y vive en
`app.state` (ver app/api/dependencies.py). El worker de ingesta corre como proceso separado
(app/services/ingestion_worker.py).
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from app.agents.graph import build_graph
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.services.llm_factory import build_chat_model

from app.core.exceptions import (
    AppException,
    AuthenticationError,
    InfrastructureError,
    RateLimitExceededError,
)
from app.core.logging import configure_logging
from app.core.metrics import PrometheusMetricsMiddleware, metrics_router
from app.core.rate_limit import RateLimiter
from app.core.redis import build_redis_client
from app.core.security import require_scope
from app.core.security_headers import SecurityHeadersMiddleware
from app.core.telemetry import setup_telemetry, shutdown_telemetry
from app.db.checkpointer import build_checkpointer, build_checkpointer_pool
from app.db.session import build_engine, build_sessionmaker
from app.domain.models import ApiKeyScope
from app.services.legal.legalize_api_client import LegalizeApiClient
from app.services.legal.local_diff_engine import LocalGitDiffEngine
from app.services.legal.retriever import HybridLegalRetriever
from app.services.prompt_manager import PromptManager
from app.services.rag_service import (
    build_embeddings_client,
    build_pinecone_client,
    get_index,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings)
    app.state.settings = settings

    if not settings.API_KEY_PEPPER.get_secret_value():
        logger.warning("lifespan.api_key_pepper_empty", extra={"hint": "generar con: openssl rand -hex 32"})

    setup_telemetry(settings)

    engine = build_engine(settings)
    app.state.db_engine = engine
    app.state.db_sessionmaker = build_sessionmaker(engine)

    checkpointer_pool = await build_checkpointer_pool(settings)
    app.state.checkpointer_pool = checkpointer_pool
    checkpointer = build_checkpointer(checkpointer_pool)

    app.state.llm_client = build_chat_model(settings)

    app.state.pinecone_client = build_pinecone_client(settings)

    app.state.rag_index = await get_index(app.state.pinecone_client, settings)
    app.state.embeddings_client = build_embeddings_client(settings)

    # Retrieval híbrido legal conectado a PostgreSQL FTS + Pinecone
    app.state.legal_retriever = HybridLegalRetriever(
        settings=settings,
        pinecone_client=app.state.pinecone_client,
        embeddings_client=app.state.embeddings_client,
        sessionmaker=app.state.db_sessionmaker,
    )

    # Motor de diff local sobre Git y cliente Legalize API con fallback
    app.state.local_diff_engine = LocalGitDiffEngine(repo_path=settings.LEGALIZE_REPO_PATH)
    app.state.legalize_api_client = LegalizeApiClient(
        settings=settings,
        local_diff_engine=app.state.local_diff_engine,
    )

    app.state.prompt_manager = PromptManager(app.state.db_sessionmaker)
    app.state.compiled_graph = build_graph(checkpointer)

    redis_client = build_redis_client(settings)
    try:
        await redis_client.ping()
    except Exception as exc:  # noqa: BLE001 - Aviso no bloqueante al arranque
        logger.warning("lifespan.redis_unreachable", extra={"error": str(exc)})
    app.state.redis_client = redis_client

    rate_limiter = RateLimiter(redis_client)
    try:
        await rate_limiter.preload()
    except Exception as exc:  # noqa: BLE001 - Se reintentará bajo demanda con eval si Redis no está listo
        logger.warning("lifespan.rate_limiter_preload_skipped", extra={"error": str(exc)})
    app.state.rate_limiter = rate_limiter

    logger.info("lifespan.startup_complete")
    yield

    await app.state.redis_client.aclose()
    await app.state.rag_index.close()
    await app.state.pinecone_client.close()
    await app.state.checkpointer_pool.close()
    await app.state.db_engine.dispose()
    shutdown_telemetry()
    logger.info("lifespan.shutdown_complete")


_settings = get_settings()
_docs_enabled = (
    _settings.DOCS_ENABLED if _settings.DOCS_ENABLED is not None else (_settings.ENVIRONMENT != "production")
)

app = FastAPI(
    title="Sistema Legal Argentino - Agente Jurídico Inteligente",
    description="""## Plataforma de Inteligencia Artificial para el Ámbito Jurídico y Legislativo Argentino

Sistema de grado de producción que integra:
* **Orquestación Legal con LangGraph**: Router de Intenciones Jurídicas, Agente Legal Especialista, Auditor Documental, Comparador de Reformas y Validador de Citas.
* **Fuente Primaria de Legislación**: Sincronización continua con el repositorio Git `legalize-ar` (InfoLEG / SAIJ), versionado a nivel de artículo.
* **Fallback Autónomo de Cuota**: Motor local de diffs y time-travel por Git ante límites mensuales de la Legalize API.
* **RAG Híbrido Distribuido**: Búsqueda semántica densa en Pinecone combinada con búsqueda léxica en PostgreSQL (Full-Text Search).
* **Auditoría Documental de Contratos**: Detección de cláusulas nulas o abusivas según el Código Civil y Comercial y leyes especiales.
* **Seguridad y Observabilidad**: Autenticación por API Key, rate limiting por Redis, OpenTelemetry y métricas Prometheus.
""",
    version="2.0.0",
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/api/v1/openapi.json" if _docs_enabled else None,
    openapi_tags=[
        {
            "name": "Legal Agent",
            "description": "Endpoints de consulta jurídica, auditoría de contratos, comparador de reformas y sincronización normativa.",
        },
        {
            "name": "ingest",
            "description": "Carga y procesamiento de documentos legales de usuarios.",
        },
        {
            "name": "prompts",
            "description": "Inspección y actualización en caliente de directivas y system prompts del agente legal.",
        },
        {
            "name": "admin",
            "description": "Administración de credenciales y claves de acceso a la API.",
        },
        {
            "name": "health",
            "description": "Diagnóstico de conectividad y estado operativo de servicios dependientes.",
        },
    ],
    lifespan=lifespan,
)

app.add_middleware(PrometheusMetricsMiddleware)
app.include_router(
    metrics_router,
    dependencies=[Depends(require_scope(ApiKeyScope.ADMIN))],
)

app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    if exc.http_status >= 500:
        logger.error(
            "app_exception",
            extra={"path": request.url.path, "error_code": exc.error_code, "detail": exc.message},
        )

    client_message = (
        "El sistema no pudo completar la operacion porque uno de los servicios de los que depende "
        "no esta disponible en este momento. Volve a intentarlo en unos minutos."
        if isinstance(exc, InfrastructureError)
        else exc.message
    )
    response = JSONResponse(
        status_code=exc.http_status,
        content={"error_code": exc.error_code, "message": client_message, "details": exc.details},
    )
    if isinstance(exc, RateLimitExceededError):
        response.headers["Retry-After"] = str(exc.retry_after_seconds)
    if isinstance(exc, AuthenticationError):
        response.headers["WWW-Authenticate"] = "ApiKey"
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error_code": "validation_error",
            "message": "Los datos enviados no son validos",
            "details": {"errors": jsonable_encoder(exc.errors())},
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Único lugar del sistema que atrapa Exception genérica: borde HTTP final (noqa: BLE001)
    logger.exception("unhandled_exception", extra={"path": request.url.path, "error": str(exc)})  # noqa: BLE001
    return JSONResponse(
        status_code=500,
        content={"error_code": "internal_error", "message": "Ocurrio un error interno inesperado"},
    )


app.include_router(api_router, prefix="/api/v1")

# asgi_app envuelve con ProxyHeadersMiddleware para resolver la IP real del cliente de forma segura
asgi_app = ProxyHeadersMiddleware(app, trusted_hosts=get_settings().TRUSTED_PROXY_IPS.split(","))
