"""Gestión dinámica de directivas y system prompts de agentes.

Permite consultar y actualizar en caliente los system prompts de cada agente
(legal_agent, document_analyzer, validator, supervisor)
sin reiniciar los servidores.

"""

from fastapi import APIRouter

from app.api.dependencies import PromptManagerDep
from app.schemas.prompts import AgentPromptDTO, PromptListResponse, PromptUpdateRequest

router = APIRouter()


@router.get(
    "/prompts",
    response_model=PromptListResponse,
    summary="Listar directivas y system prompts activos",
    description="Devuelve la configuración y contenido actual de los system prompts para todos los agentes del sistema.",
    response_description="Lista de directivas por agente con sus versiones y fechas de modificación.",
)
async def list_prompts(prompt_manager: PromptManagerDep) -> PromptListResponse:
    return PromptListResponse(prompts=await prompt_manager.list_prompts())


@router.put(
    "/prompts/{agent_id}",
    response_model=AgentPromptDTO,
    summary="Actualizar en caliente el system prompt de un agente",
    description="Modifica las instrucciones y directivas del agente indicado (`legal_agent`, `document_analyzer`, `validator`, `supervisor`), incrementando su número de versión.",

    response_description="Directiva actualizada con su nuevo número de versión.",
)
async def update_prompt(
    agent_id: str, payload: PromptUpdateRequest, prompt_manager: PromptManagerDep
) -> AgentPromptDTO:
    return await prompt_manager.update_prompt(agent_id, payload.content, payload.updated_by)
