"""Autenticacion por API key + autorizacion por scope + rate limiting. Tres
dependencias componibles: get_current_api_key (identidad + throttle por IP),
require_scope (autorizacion), rate_limited (throttle por key ya autenticada)
-- ver README para el detalle de cada decision (hashing, pepper, etc)."""

import hashlib
import logging
import os
import secrets
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    RateLimitExceededError,
)
from app.core.ip_filter import is_ip_in_allowlist
from app.core.rate_limit import RateLimiter
from app.db.models_orm import ApiKey
from app.db.session import get_db_session
from app.domain.models import ApiKeyScope

logger = logging.getLogger(__name__)

_SCOPE_HIERARCHY: dict[ApiKeyScope, set[ApiKeyScope]] = {
    ApiKeyScope.ADMIN: {ApiKeyScope.ADMIN, ApiKeyScope.CLIENT},
    ApiKeyScope.CLIENT: {ApiKeyScope.CLIENT},
}

# os.getenv en vez de get_settings(): Settings() exige OPENAI_API_KEY/PINECONE_API_KEY
# (sin default), y APIKeyHeader se construye al importar el modulo -- ese require
# no tiene nada que ver con auth y rompia poder importar/testear este archivo aislado
# sin credenciales de LLM/vector store presentes en el entorno.
_api_key_header = APIKeyHeader(name=os.getenv("API_KEY_HEADER_NAME", "X-API-Key"), auto_error=False)


def generate_api_key() -> str:
    return f"isk_{secrets.token_urlsafe(32)}"


def hash_api_key(plaintext: str, settings: Settings) -> str:
    pepper = settings.API_KEY_PEPPER.get_secret_value()
    return hashlib.sha256(f"{pepper}{plaintext}".encode()).hexdigest()


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


async def get_current_api_key(
    request: Request,
    raw_key: Annotated[str | None, Security(_api_key_header)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ApiKey:
    client_ip = _client_ip(request)

    if raw_key is None:
        if settings.RATE_LIMIT_ENABLED:
            rate_limiter: RateLimiter = request.app.state.rate_limiter
            result = await rate_limiter.check(
                f"ip:{client_ip}", settings.RATE_LIMIT_ANONYMOUS_PER_MINUTE, settings.RATE_LIMIT_WINDOW_SECONDS
            )
            if not result.allowed:
                logger.warning("rate_limit.ip_exceeded", extra={"ip": client_ip, "path": request.url.path})
                raise RateLimitExceededError(result.retry_after_seconds)
        logger.warning("auth.missing_key", extra={"ip": client_ip, "path": request.url.path})
        raise AuthenticationError("Falta la API key")

    key_hash = hash_api_key(raw_key, settings)
    row = await db.execute(select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True)))
    record = row.scalar_one_or_none()
    if record is None:
        if settings.RATE_LIMIT_ENABLED:
            rate_limiter: RateLimiter = request.app.state.rate_limiter
            result = await rate_limiter.check(
                f"ip:{client_ip}", settings.RATE_LIMIT_ANONYMOUS_PER_MINUTE, settings.RATE_LIMIT_WINDOW_SECONDS
            )
            if not result.allowed:
                logger.warning("rate_limit.ip_exceeded", extra={"ip": client_ip, "path": request.url.path})
                raise RateLimitExceededError(result.retry_after_seconds)
        logger.warning("auth.invalid_key", extra={"ip": client_ip, "path": request.url.path})
        raise AuthenticationError("API key invalida")

    record.last_used_at = datetime.now(UTC)
    return record


def require_scope(required: ApiKeyScope) -> Callable:
    async def _dependency(
        request: Request,
        key_record: Annotated[ApiKey, Depends(get_current_api_key)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> ApiKey:
        key_scope = ApiKeyScope(key_record.scope)
        if required not in _SCOPE_HIERARCHY[key_scope]:
            raise AuthorizationError(f"Esta operacion requiere una API key con scope '{required.value}'")

        if required == ApiKeyScope.ADMIN and not is_ip_in_allowlist(_client_ip(request), settings.ADMIN_IP_ALLOWLIST):
            logger.warning("auth.admin_ip_denied", extra={"ip": _client_ip(request), "path": request.url.path})
            raise AuthorizationError("Tu IP no esta autorizada para operaciones administrativas")

        return key_record

    return _dependency


def rate_limited(*, strict_chat_limit: bool = False) -> Callable:
    async def _dependency(
        request: Request,
        key_record: Annotated[ApiKey, Depends(get_current_api_key)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> None:
        if not settings.RATE_LIMIT_ENABLED:
            return

        key_scope = ApiKeyScope(key_record.scope)
        if key_scope == ApiKeyScope.ADMIN:
            limit = settings.RATE_LIMIT_ADMIN_PER_MINUTE
        elif strict_chat_limit:
            limit = settings.RATE_LIMIT_CLIENT_CHAT_PER_MINUTE
        else:
            limit = settings.RATE_LIMIT_CLIENT_PER_MINUTE

        rate_limiter: RateLimiter = request.app.state.rate_limiter
        result = await rate_limiter.check(f"key:{key_record.key_id}", limit, settings.RATE_LIMIT_WINDOW_SECONDS)
        if not result.allowed:
            logger.warning(
                "rate_limit.key_exceeded", extra={"key_id": str(key_record.key_id), "path": request.url.path}
            )
            raise RateLimitExceededError(result.retry_after_seconds)

    return _dependency
