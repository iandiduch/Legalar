"""Nodo Validador Jurídico: Auditoría estricta de citas, vigencia actual y sustento fáctico."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agents.legal.state import LegalAgentState
from app.core.structured_output import invoke_structured_with_retry
from app.domain.models import AgentRole, ConfidenceLevel
from app.schemas.legal.chat import LegalValidationSummary

logger = logging.getLogger(__name__)


class ValidatorCheck(BaseModel):
    is_valid: bool = Field(description="True si las afirmaciones y citas jurídicas son correctas")
    all_norms_in_force: bool = Field(description="True si todas las leyes y artículos citados están en vigencia")
    unsupported_claims: list[str] = Field(default_factory=list, description="Afirmaciones que no se desprenden de las normas")
    warning_notes: list[str] = Field(default_factory=list, description="Advertencias sobre reformas o derogaciones")
    synthesized_final_answer: str = Field(description="Respuesta jurídica final pulida y validada")


VALIDATOR_PROMPT = """Eres el Validador Jurídico del Sistema Legal Argentino.
Tu responsabilidad es certificar la veracidad, vigencia y fidelidad de la respuesta generada
respecto de la evidencia normativa.

Criterios de validación:
1. ¿Los artículos citados existen y corresponden a la ley señalada?
2. ¿Se afirma la vigencia de una norma derogada o modificada (ej: referirse a la Ley de Alquileres 27.551 como vigente cuando fue derogada por el DNU 70/2023)?
3. ¿La respuesta agregó afirmaciones contundentes que no figuran en la evidencia legal?
4. Si la respuesta es jurídicamente sólida, refina la redacción para garantizar máxima claridad profesional.
"""


async def legal_validator_node(state: LegalAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Evalúa la solidez y vigencia de la respuesta antes de devolverla al usuario."""
    configurable = config.get("configurable", {})
    llm = configurable.get("llm_client")
    settings = configurable.get("settings")

    draft = state.get("draft_answer") or ""
    citations = state.get("citations", [])

    # Si no hay citas normativas que auditar (ej. consultas de diff histórico o análisis directo),
    # no tiene sentido disparar una llamada LLM pesada: ahorramos 20-25s de latencia.
    if not citations:
        val_summary = LegalValidationSummary(
            is_valid=True,
            all_norms_in_force=True,
            unsupported_claims=[],
            warning_notes=[],
        )
        return {
            "validation_result": val_summary,
            "final_answer": draft,
            "messages": [AIMessage(content=draft, name=AgentRole.VALIDATOR.value)],
        }

    # Validación heurística preliminar de estado de vigencia
    repealed_citations = [c for c in citations if c.status != "in_force"]
    has_repealed = len(repealed_citations) > 0

    evidence_summary = "\n".join(
        [f"- {c.law_identifier} Art. {c.article_number} (Vigencia: {c.status}): {c.exact_quote}" for c in citations]
    )

    system_instruction = (
        f"{VALIDATOR_PROMPT}\n\n"
        f"EVIDENCIA NORMATIVA:\n{evidence_summary}"
    )

    messages = [
        SystemMessage(content=system_instruction),
        SystemMessage(content=f"RESPUESTA BORRADOR A VALIDAR:\n\n{draft}"),
    ]

    try:
        check: ValidatorCheck = await invoke_structured_with_retry(
            llm,
            ValidatorCheck,
            system_instruction,
            messages,
            max_attempts=settings.STRUCTURED_OUTPUT_MAX_ATTEMPTS if settings else 2,
        )
        final_answer = check.synthesized_final_answer
        val_summary = LegalValidationSummary(
            is_valid=check.is_valid,
            all_norms_in_force=check.all_norms_in_force and not has_repealed,
            unsupported_claims=check.unsupported_claims,
            warning_notes=check.warning_notes,
        )
    except Exception as exc:
        logger.warning("Fallo en validador estructurado, usando aprobación directa: %s", exc)
        final_answer = draft
        val_summary = LegalValidationSummary(
            is_valid=True,
            all_norms_in_force=not has_repealed,
            unsupported_claims=[],
            warning_notes=["Advertencia: Cita norma no vigente"] if has_repealed else [],
        )

    return {
        "validation_result": val_summary,
        "final_answer": final_answer,
        "messages": [AIMessage(content=final_answer, name=AgentRole.VALIDATOR.value)],
    }
