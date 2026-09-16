"""Nodo del Agente Legal: Recuperación y generación de dictámenes jurídicos fundamentados."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agents.legal.state import LegalAgentState
from app.core.structured_output import invoke_structured_with_retry
from app.domain.models import AgentRole, ConfidenceLevel
from app.schemas.legal.citation import LegalCitation
from app.services.legal.retriever import HybridLegalRetriever

logger = logging.getLogger(__name__)


class LegalAnswerPayload(BaseModel):
    answer: str = Field(description="Respuesta jurídica integral, clara y con citas normativas específicas")
    articles_referenced: list[str] = Field(
        default_factory=list,
        description="Identificadores y números de artículos citados (ej: ['LEY-26994:1198', 'DNU-70-2023:256'])",
    )
    confidence: ConfidenceLevel = Field(description="Nivel de certidumbre según el respaldo normativo disponible")
    unsupported_notes: list[str] = Field(
        default_factory=list,
        description="Aspectos de la consulta que la legislación no cubre de forma expresa",
    )


LEGAL_AGENT_PROMPT = """Eres el Agente Legal Especialista en Derecho Argentino.
Tu función es responder con estricto rigor jurídico a la consulta del usuario, basándote PRIMARIAMENTE
en los artículos y normas provistos en la evidencia.

Directivas obligatorias:
1. Responde de forma directa, analítica y profesional en formato Markdown estructurado.
2. PROHIBIDO incluir saludos de carta o correos ("Estimado", "Colega", "Hola") o despedidas/firmas ("Atentamente", "Quedo a su disposición"). Ve directamente al dictamen jurídico.
3. Cita siempre la norma y el artículo específico que fundamenta tu afirmación (ej: "Conforme al art. 1198 del Código Civil y Comercial de la Nación (Ley 26.994)...").
4. Si una reforma reciente (como el DNU 70/2023) modificó el artículo, indícalo expresamente.
5. Distingue entre normas de orden público (irrenunciables) y normas supletorias (disponibles por las partes).
6. No inventes artículos ni leyes. Si no hay suficiente información en la evidencia, acláralo con honestidad profesional.
7. Utiliza lenguaje jurídico claro y accesible, manteniendo la máxima precisión técnica.
"""


async def legal_agent_node(state: LegalAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Recupera normas relevantes mediante el retriever híbrido y sintetiza el dictamen legal."""
    configurable = config.get("configurable", {})
    llm = configurable.get("llm_client")
    retriever: HybridLegalRetriever | None = configurable.get("legal_retriever")
    settings = configurable.get("settings")

    query = state["query"]
    citations: list[LegalCitation] = []

    # 1. Recuperación híbrida (PostgreSQL FTS + Pinecone)
    if retriever:
        try:
            citations = await retriever.search(query=query, top_k=6)
        except Exception as exc:
            logger.error("Error en retriever híbrido: %s", exc)

    # 2. Formatear evidencia para el LLM
    evidence_lines = []
    for c in citations:
        ep_str = f" ({c.epigraph})" if c.epigraph else ""
        evidence_lines.append(
            f"--- NORMA: {c.law_identifier} ({c.law_title}) | ARTÍCULO: {c.article_number}{ep_str} ---\n"
            f"ESTADO: {c.status} | URL: {c.infoleg_url or 'N/A'}\n"
            f"TEXTO: {c.exact_quote}\n"
        )

    evidence_block = "\n".join(evidence_lines) if evidence_lines else "(No se hallaron artículos coincidentes)"

    system_instruction = (
        f"{LEGAL_AGENT_PROMPT}\n\n"
        f"EVIDENCIA NORMATIVA DISPONIBLE:\n{evidence_block}"
    )

    messages = [
        SystemMessage(content=system_instruction),
        *state["messages"],
    ]

    try:
        payload: LegalAnswerPayload = await invoke_structured_with_retry(
            llm,
            LegalAnswerPayload,
            system_instruction,
            messages,
            max_attempts=settings.STRUCTURED_OUTPUT_MAX_ATTEMPTS if settings else 2,
        )
        answer = payload.answer
        confidence = payload.confidence
    except Exception as exc:
        logger.warning("Fallo en estructurado de legal_agent, usando generación directa: %s", exc)
        res = await llm.ainvoke(messages)
        answer = str(res.content)
        confidence = ConfidenceLevel.MEDIUM if citations else ConfidenceLevel.LOW

    return {
        "citations": citations,
        "draft_answer": answer,
        "confidence": confidence,
        "messages": [AIMessage(content=answer, name=AgentRole.LEGAL_AGENT.value)],
    }
