from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.models import ApiKeyScope


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="Etiqueta humana, ej. 'Frontend institucional'")
    scope: ApiKeyScope
    created_by: str | None = Field(default=None, max_length=128)


class ApiKeyDTO(BaseModel):
    """Nunca lleva key_hash ni el plaintext -- es lo que se devuelve en listados."""

    model_config = ConfigDict(from_attributes=True)

    key_id: UUID
    name: str
    scope: ApiKeyScope
    is_active: bool
    created_at: datetime
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None
    created_by: str | None = None


class ApiKeyCreateResponse(ApiKeyDTO):
    """El plaintext solo existe en esta respuesta -- se muestra una unica vez
    al crear la key y nunca vuelve a poder leerse despues (ni por API ni
    directo de la base, que solo guarda el hash)."""

    plaintext_key: str
