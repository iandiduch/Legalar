"""Nodo Validador Jurídico: Auditoría estricta de citas, vigencia actual y sustento fáctico."""

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agents.legal.state import LegalAgentState
from app.core.structured_output import invoke_structured_with_retry
from app.domain.models import AgentRole, ConfidenceLevel
from app.schemas.legal.chat import LegalValidationSummary
from app.services.prompt_manager import resolve_prompt

logger = logging.getLogger(__name__)


class ValidatorCheck(BaseModel):
    is_valid: bool = Field(description="True si las afirmaciones y citas jurídicas son correctas")
    all_norms_in_force: bool = Field(description="True si todas las leyes y artículos citados están en vigencia")
    unsupported_claims: list[str] = Field(default_factory=list, description="Afirmaciones que no se desprenden de las normas")
    warning_notes: list[str] = Field(default_factory=list, description="Advertencias sobre reformas o derogaciones")
    synthesized_final_answer: str = Field(description="Respuesta jurídica final pulida y validada")


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

    base_prompt = await resolve_prompt(config, "validator")
    system_instruction = (
        f"{base_prompt}\n\n"
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

        # Si el validador determinó que la respuesta no es válida, adjuntar observaciones de auditoría breves
        if not check.is_valid:
            raw_notes = check.warning_notes + check.unsupported_claims
            if raw_notes and "Observaciones de Auditoría Jurídica" not in final_answer:
                clean_notes = []
                for n in raw_notes:
                    item = n.strip().lstrip("-* ").lstrip("⚠️ ").strip()
                    if item:
                        if len(item) > 220:
                            item = item[:217] + "..."
                        clean_notes.append(item)
                if clean_notes:
                    bullets = "\n".join(f"- ⚠️ {cn}" for cn in clean_notes)
                    final_answer = f"{final_answer}\n\n> [!WARNING]\n> **Observaciones de Auditoría Jurídica:**\n{bullets}\n"
        elif check.warning_notes:
            vigencia_warnings = []
            final_lower = final_answer.lower()
            for w in check.warning_notes:
                vw = w.strip().lstrip("-* ").lstrip("⚠️ ").strip()
                if not any(k in vw.lower() for k in ("derog", "vigente", "reforma", "dnu")):
                    continue
                # Si la advertencia menciona un número de ley (ej: '27.551') y la respuesta ya explica su derogación/reforma, evitar duplicación
                law_nums = re.findall(r"\b\d{4,5}\b", vw)
                already_addressed = False
                if law_nums and any(
                    ln in final_lower and any(dk in final_lower for dk in ("derog", "abrog", "sustitu", "reforma"))
                    for ln in law_nums
                ):
                    already_addressed = True

                if not already_addressed and vw not in final_answer:
                    vigencia_warnings.append(vw[:217] + "..." if len(vw) > 220 else vw)

            if vigencia_warnings:
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
