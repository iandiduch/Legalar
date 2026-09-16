"""Agrupador central de rutas v1 del Sistema Legal Argentino."""

from fastapi import APIRouter, Depends

from app.api.v1 import legal
from app.api.v1.endpoints import (
    api_keys,
    health,
    ingest,
    prompts,
)
from app.core.security import rate_limited, require_scope
from app.domain.models import ApiKeyScope

api_router = APIRouter()

_client = Depends(require_scope(ApiKeyScope.CLIENT))
_client_rate_limited = Depends(rate_limited())
_admin = Depends(require_scope(ApiKeyScope.ADMIN))
_admin_rate_limited = Depends(rate_limited())

# Rutas del Agente Legal
api_router.include_router(
    legal.router,
    dependencies=[_client, _client_rate_limited],
)

api_router.include_router(ingest.router, tags=["ingest"], dependencies=[_admin, _admin_rate_limited])
api_router.include_router(prompts.router, tags=["prompts"], dependencies=[_admin, _admin_rate_limited])
api_router.include_router(api_keys.router, tags=["admin"], dependencies=[_admin, _admin_rate_limited])
api_router.include_router(health.router, tags=["health"])  # sin protección: health checks
