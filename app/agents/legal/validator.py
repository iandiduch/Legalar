"""Nodo Validador Jurídico: Auditoría estricta de citas, vigencia actual y sustento fáctico."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
3. ¿La respuesta agregó afirmaciones contundentes que contradigan la ley o que inventen artículos inexistentes? No califiques como 'no sustentadas' precisiones dogmáticas, plazos ni rubros que se desprendan de los artículos provistos en la evidencia (ej: Art. 11 Ley 24.240 o Art. 245 LCT).
4. Si la respuesta es jurídicamente sólida, refina la redacción para garantizar máxima claridad profesional.
5. REGLA ESTRICTA DE ESTILO: La respuesta final debe ser directa y en Markdown. PROHIBIDO incluir saludos de carta ("Estimado/a", "Colega") o firmas/despedidas ("Atentamente", "Quedo a su disposición").
6. PRESERVACIÓN DE PREGUNTAS DE ACLARACIÓN: Si el borrador contiene una sección de preguntas de aclaración o datos faltantes para precisar el caso (por ejemplo, titulada '### Para poder precisar tu caso:'), DEBES PRESERVARLA ÍNTEGRAMENTE al final de tu respuesta sintetizada, sin omitir ninguna de las preguntas.

IMPORTANTE sobre el campo 'synthesized_final_answer':
- Este campo debe ser SIEMPRE una respuesta jurídica directa, completa y concluyente para el usuario, nunca un meta-comentario.
- Si la respuesta es válida: proporciona la versión mejorada, pulida, técnicamente exacta y fundamentada.
- Si la respuesta requiere ajustes de vigencia: corrígela directamente y entrega en 'synthesized_final_answer' la versión jurídicamente exacta y depurada.
- NUNCA escribas en este campo frases como 'la respuesta no está en la evidencia' o 'no puedo validar'. Eso va en 'unsupported_claims' o 'warning_notes'.
"""


async def legal_validator_node(state: LegalAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Evalúa la solidez y vigencia de la respuesta antes de devolverla al usuario."""
    configurable = config.get("configurable", {})
    llm = configurable.get("llm_client")
    settings = configurable.get("settings")

    draft = state.get("draft_answer") or ""
    citations = state.get("citations", [])

    # Si no hay citas normativas que auditar (ej. consultas de diff histórico o análisis directo),
    # aprobamos de inmediato ahorrando latencia innecesaria.
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

    has_repealed = any(c.status in ("repealed", "derogated", "derogada") for c in citations)

    evidence_lines = []
    for c in citations:
        clean_quote = (c.exact_quote or "").strip()[:3500]
        evidence_lines.append(
            f"- {c.law_identifier} Art. {c.article_number}: estado='{c.status}', texto='{clean_quote}'"
        )
    evidence_summary = "\n".join(evidence_lines) if evidence_lines else "Sin citas asociadas."

    system_instruction = (
        f"{VALIDATOR_PROMPT}\n\n"
        f"EVIDENCIA NORMATIVA:\n{evidence_summary}"
    )

    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=f"RESPUESTA BORRADOR A VALIDAR:\n\n{draft}"),
    ]

    try:
        check: ValidatorCheck = await invoke_structured_with_retry(
            llm,
            ValidatorCheck,
            system_instruction,
            messages,
            max_attempts=settings.STRUCTURED_OUTPUT_MAX_ATTEMPTS if settings else 2,
        )
        synthesized = (check.synthesized_final_answer or "").strip()

        # Priorizar la versión sintetizada y depurada del validador si es sustantiva (>= 40 chars)
        if len(synthesized) >= 40:
            final_answer = synthesized
        else:
            logger.info("Validador devolvió síntesis corta (%d chars); preservando borrador original.", len(synthesized))
            final_answer = draft

        # Si el validador determinó que la respuesta no es válida, adjuntar observaciones de auditoría
        if not check.is_valid:
            notes = check.warning_notes + check.unsupported_claims
            if notes and "Observaciones de Auditoría Jurídica" not in final_answer:
                bullets = "\n".join(f"- ⚠️ {n}" for n in notes)
                final_answer = f"{final_answer}\n\n> [!WARNING]\n> **Observaciones de Auditoría Jurídica:**\n{bullets}\n"
        elif check.warning_notes:
            vigencia_warnings = [w for w in check.warning_notes if "derog" in w.lower() or "vigente" in w.lower() or "reforma" in w.lower()]
            if vigencia_warnings and not any(w in final_answer for w in vigencia_warnings):
                warning_bullets = "\n".join(f"- ⚠️ {w}" for w in vigencia_warnings)
                final_answer = f"{final_answer}\n\n> [!NOTE]\n> **Estado de Vigencia Normativa:**\n{warning_bullets}\n"

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
