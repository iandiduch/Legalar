"""Gestión de credenciales y API keys administrativas y de clientes.

Permite generar, listar y revocar claves con hash SHA-256 + pepper. Requiere scope 'admin'.
"""

from uuid import UUID

from fastapi import APIRouter

from app.api.dependencies import ApiKeyServiceDep
from app.schemas.security import ApiKeyCreateRequest, ApiKeyCreateResponse, ApiKeyDTO

router = APIRouter()


@router.post(
    "/admin/api-keys",
    response_model=ApiKeyCreateResponse,
    status_code=201,
    summary="Crear nueva API Key",
    description="Genera una clave criptográfica aleatoria para consumo del API con scope 'client' o 'admin'. La clave en texto plano solo se devuelve una única vez en la creación.",
    response_description="Clave generada en texto plano y metadatos del registro.",
)
async def create_api_key(payload: ApiKeyCreateRequest, service: ApiKeyServiceDep) -> ApiKeyCreateResponse:
    return await service.create_key(name=payload.name, scope=payload.scope, created_by=payload.created_by)


@router.get(
    "/admin/api-keys",
    response_model=list[ApiKeyDTO],
    summary="Listar API Keys registradas",
    description="Devuelve el listado de claves registradas (sin exponer los hashes ni el texto plano), indicando su estado activo/revocado y fecha de último uso.",
    response_description="Lista de claves y sus metadatos.",
)
async def list_api_keys(service: ApiKeyServiceDep) -> list[ApiKeyDTO]:
    return await service.list_keys()


@router.delete(
    "/admin/api-keys/{key_id}",
    response_model=ApiKeyDTO,
    summary="Revocar API Key",
    description="Inactiva inmediatamente una clave de acceso, impidiendo que continúe autenticando solicitudes.",
    response_description="Registro de la clave con estado revocado.",
)
async def revoke_api_key(key_id: UUID, service: ApiKeyServiceDep) -> ApiKeyDTO:
    return await service.revoke_key(key_id)
