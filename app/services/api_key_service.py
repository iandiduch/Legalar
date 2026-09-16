"""CRUD de API keys sobre la tabla api_keys. Mismo patron que EscalationService:
los endpoints nunca tocan SQLAlchemy directo."""

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.exceptions import NotFoundError
from app.core.security import generate_api_key, hash_api_key
from app.db.models_orm import ApiKey
from app.domain.models import ApiKeyScope
from app.schemas.security import ApiKeyCreateResponse, ApiKeyDTO


class ApiKeyNotFoundError(NotFoundError):
    error_code = "api_key_not_found"


class ApiKeyService:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self._session = session
        self._settings = settings

    async def create_key(self, *, name: str, scope: ApiKeyScope, created_by: str | None) -> ApiKeyCreateResponse:
        plaintext = generate_api_key()
        record = ApiKey(
            key_hash=hash_api_key(plaintext, self._settings),
            name=name,
            scope=scope.value,
            created_by=created_by,
        )
        self._session.add(record)
        await self._session.flush()
        await self._session.refresh(record)
        return ApiKeyCreateResponse(**ApiKeyDTO.model_validate(record).model_dump(), plaintext_key=plaintext)

    async def list_keys(self) -> list[ApiKeyDTO]:
        result = await self._session.execute(select(ApiKey).order_by(ApiKey.created_at.desc()))
        return [ApiKeyDTO.model_validate(row) for row in result.scalars().all()]

    async def revoke_key(self, key_id: UUID) -> ApiKeyDTO:
        record = await self._session.get(ApiKey, key_id)
        if record is None:
            raise ApiKeyNotFoundError(f"API key {key_id} no encontrada")
        record.is_active = False
        record.revoked_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(record)
        return ApiKeyDTO.model_validate(record)


async def seed_bootstrap_admin_key(sessionmaker: async_sessionmaker[AsyncSession], settings: Settings) -> None:
    """Siembra la primera key admin desde BOOTSTRAP_ADMIN_API_KEY, unica vez
    (solo si api_keys esta vacia) -- mismo patron idempotente que
    PromptManager.seed_defaults_if_empty. Se corre desde scripts/init_db.py,
    nunca desde el lifespan de main.py."""
    if settings.BOOTSTRAP_ADMIN_API_KEY is None:
        return

    async with sessionmaker() as session:
        existing = await session.execute(select(ApiKey.key_id).limit(1))
        if existing.first() is not None:
            return

        plaintext = settings.BOOTSTRAP_ADMIN_API_KEY.get_secret_value()
        session.add(
            ApiKey(
                key_hash=hash_api_key(plaintext, settings),
                name="bootstrap-admin",
                scope=ApiKeyScope.ADMIN.value,
                created_by="scripts.init_db",
            )
        )
        await session.commit()
