"""Analizador y Expansor de Consultas Jurídicas para RAG Legal Relacional (Multi-Norma).

Detecta cuando una consulta requiere articular una figura jurídica específica (delito, contrato, despido)
con una regla o instituto general (cómputo de prescripción, indemnización, nulidad, inicio de plazos,
interrupción/suspensión), generando sub-consultas dirigidas y referencias a normas canónicas.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.core.structured_output import invoke_structured_with_retry
from app.services.prompt_manager import resolve_prompt

logger = logging.getLogger(__name__)


class LegalRelationAnalysis(BaseModel):
    question_type: str = Field(
        default="general",
        description="Tipo de relación jurídica: prescription, compensation, nullity, eviction, employment_termination, procedural_rights, general",
    )
    is_relational: bool = Field(
        default=False,
        description="True si la consulta requiere combinar dos o más normas o capítulos distintos (ej: tipo penal específico + regla general de prescripción)",
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Lista de 2 o 3 sub-queries complementarias optimizadas para búsqueda vectorial y FTS",
    )
    canonical_articles: list[str] = Field(
        default_factory=list,
        description="Identificadores de normas y artículos canónicos clave si son conocidos con formato LEY-XXXX:NUM (ej: ['LEY-11179:62', 'LEY-11179:172', 'DEC-390-1976:245'])",
    )
    reasoning: str = Field(
        default="",
        description="Breve justificación jurídica de por qué se requieren esas normas complementarias",
    )


async def expand_legal_query(
    llm: Any,
    query: str,
    conversation_context: list | None = None,
    max_attempts: int = 2,
    prompt_manager: Any | None = None,
) -> LegalRelationAnalysis:
    """Analiza y descompone la consulta legal en sub-queries relacionales y artículos canónicos.

    Si el LLM no está configurado o la invocación falla, devuelve un fallback seguro
    con la consulta original sin alterar el flujo del agente.
    """
    if not llm or not query.strip():
        return LegalRelationAnalysis(
            question_type="general",
            is_relational=False,
            sub_queries=[query.strip()] if query.strip() else [],
            canonical_articles=[],
            reasoning="Fallback determinista por ausencia de LLM o query vacía",
        )

    context_messages = []
    if conversation_context:
        # Últimos 3 mensajes para contextualizar referencias elípticas
        context_messages = conversation_context[-3:]

    expander_prompt = await resolve_prompt(
        {"configurable": {"prompt_manager": prompt_manager}} if prompt_manager else None,
        "query_expander",
    )

    messages = [
        SystemMessage(content=expander_prompt),
        *context_messages,
        HumanMessage(content=f"CONSULTA JURÍDICA A ANALIZAR:\n{query}"),
    ]

    try:
        analysis: LegalRelationAnalysis = await invoke_structured_with_retry(
            llm,
            LegalRelationAnalysis,
            expander_prompt,
            messages,
            max_attempts=max_attempts,
        )
        if not analysis.sub_queries:
            analysis.sub_queries = [query]

        logger.info(
            "expand_legal_query: type=%s, relational=%s, sub_queries=%d, canonical=%s",
            analysis.question_type,
            analysis.is_relational,
            len(analysis.sub_queries),
            analysis.canonical_articles,
        )
        return analysis
    except Exception as exc:
        logger.warning("Fallo en expand_legal_query, usando fallback con query original: %s", exc)
        return LegalRelationAnalysis(
            question_type="general",
            is_relational=False,
            sub_queries=[query],
            canonical_articles=[],
            reasoning="Fallback ante fallo en LLM",
        )
