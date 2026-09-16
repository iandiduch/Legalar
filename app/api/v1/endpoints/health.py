"""GET /health: diagnóstico integral y chequeo activo de dependencias externas
(PostgreSQL, Redis, Pinecone, OpenAI, Arize Phoenix).
"""

import asyncio

import httpx
from fastapi import APIRouter, Request, Response
from pinecone import PineconeError
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import Settings
from app.schemas.health import HealthCheckResponse, ServiceStatus

router = APIRouter()

# Postgres y Redis son dependencias críticas para la operación base.
# Si caen, el endpoint retorna HTTP 503 para que los balanceadores y Docker detecten el fallo.
_CRITICAL_SERVICES = {"postgres", "redis"}


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Diagnóstico de salud del sistema",
    description="Verifica de forma concurrente la conectividad y disponibilidad de PostgreSQL, Redis, Pinecone, OpenAI y Arize Phoenix. Retorna 503 si las dependencias críticas fallan.",
    response_description="Estado general (ok/degraded) y detalle por cada servicio.",
)
async def health(request: Request, response: Response) -> HealthCheckResponse:
    settings: Settings = request.app.state.settings
    results = await asyncio.gather(
        _check_postgres(request),
        _check_redis(request),
        _check_pinecone(request),
        _check_openai(settings),
        _check_phoenix(settings),
    )
    overall = "ok" if all(r.reachable for r in results) else "degraded"
    if any(not r.reachable for r in results if r.name in _CRITICAL_SERVICES):
        response.status_code = 503
    return HealthCheckResponse(status=overall, services=list(results))


async def _check_postgres(request: Request) -> ServiceStatus:
    try:
        async with request.app.state.db_sessionmaker() as session:
            await session.execute(text("SELECT 1"))
        return ServiceStatus(name="postgres", reachable=True)
    except SQLAlchemyError as exc:
        return ServiceStatus(name="postgres", reachable=False, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - Red de seguridad de diagnóstico en healthcheck
        return ServiceStatus(name="postgres", reachable=False, detail=str(exc))


async def _check_redis(request: Request) -> ServiceStatus:
    try:
        await request.app.state.redis_client.ping()
        return ServiceStatus(name="redis", reachable=True)
    except RedisError as exc:
        return ServiceStatus(name="redis", reachable=False, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - Red de seguridad de diagnóstico en healthcheck
        return ServiceStatus(name="redis", reachable=False, detail=str(exc))


async def _check_pinecone(request: Request) -> ServiceStatus:
    try:
        await request.app.state.pinecone_client.indexes.list()
        return ServiceStatus(name="pinecone", reachable=True)
    except PineconeError as exc:
        return ServiceStatus(name="pinecone", reachable=False, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - Red de seguridad de diagnóstico en healthcheck
        return ServiceStatus(name="pinecone", reachable=False, detail=str(exc))


async def _check_openai(settings: Settings) -> ServiceStatus:
    configured = bool(settings.OPENAI_API_KEY.get_secret_value())
    return ServiceStatus(
        name="openai", reachable=configured, detail=None if configured else "OPENAI_API_KEY no configurada"
    )


async def _check_phoenix(settings: Settings) -> ServiceStatus:
    if not settings.PHOENIX_ENABLED:
        return ServiceStatus(name="phoenix", reachable=False, detail="deshabilitado por configuración")
    base_url = settings.PHOENIX_COLLECTOR_ENDPOINT.split("/v1/")[0]
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(base_url)
        return ServiceStatus(name="phoenix", reachable=response.status_code < 500)
    except httpx.HTTPError as exc:
        return ServiceStatus(name="phoenix", reachable=False, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - Red de seguridad de diagnóstico en healthcheck
        return ServiceStatus(name="phoenix", reachable=False, detail=str(exc))
