"""Nodo de Análisis Documental Jurídico: Auditoría de contratos y documentos contra el derecho argentino."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agents.legal.state import LegalAgentState
from app.core.structured_output import invoke_structured_with_retry
from app.domain.models import AgentRole, ConfidenceLevel
from app.schemas.legal.analysis import DocumentAnalysisResponse, DocumentRiskItem
from app.services.legal.retriever import HybridLegalRetriever

logger = logging.getLogger(__name__)


DOCUMENT_ANALYZER_PROMPT = """Eres el Especialista en Auditoría y Análisis Documental del Agente Legal Argentino.
Tu tarea es examinar minuciosamente el contrato o documento legal suministrado por el usuario
a la luz de la legislación argentina vigente (Código Civil y Comercial de la Nación, Ley de Contrato de Trabajo,
Ley de Defensa del Consumidor, DNU 70/2023, etc.).

Tu análisis debe:
1. Resumen ejecutivo de la naturaleza y validez del acuerdo.
2. Identificar cláusulas abusivas, nulas, de nulidad relativa o que violen el orden público.
3. Contrastar con las normas específicas del Código y leyes aplicables.
4. Proponer redacciones alternativas recomendadas para proteger los derechos de la parte.
"""


async def document_analyzer_node(state: LegalAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Audita cláusulas del texto legal del usuario contra el ordenamiento positivo argentino."""
    configurable = config.get("configurable", {})
    llm = configurable.get("llm_client")
    retriever: HybridLegalRetriever | None = configurable.get("legal_retriever")
    settings = configurable.get("settings")

    doc_text = state.get("document_text") or state["query"]
    doc_type = state.get("document_type") or "contrato"

    # 1. Recuperar legislación aplicable al tipo de documento
    citations = []
    if retriever:
        search_query = f"{doc_type} orden público cláusulas nulas {doc_text[:200]}"
        try:
            citations = await retriever.search(query=search_query, top_k=5)
        except Exception as exc:
            logger.warning("Error en retriever para análisis documental: %s", exc)

    evidence_summary = "\n".join(
        [f"- {c.law_identifier} Art. {c.article_number} ({c.epigraph}): {c.exact_quote[:200]}" for c in citations]
    )

    system_instruction = (
        f"{DOCUMENT_ANALYZER_PROMPT}\n\n"
        f"NORMAS DE REFERENCIA RECUPERADAS:\n{evidence_summary}"
    )

    messages = [
        SystemMessage(content=system_instruction),
        SystemMessage(content=f"DOCUMENTO A AUDITAR ({doc_type}):\n\n{doc_text}"),
    ]

    try:
        analysis_res: DocumentAnalysisResponse = await invoke_structured_with_retry(
            llm,
            DocumentAnalysisResponse,
            system_instruction,
            messages,
            max_attempts=settings.STRUCTURED_OUTPUT_MAX_ATTEMPTS if settings else 2,
        )
        final_text = (
            f"### Dictamen Jurídico sobre el {doc_type.capitalize()}\n\n"
            f"**Resumen:** {analysis_res.executive_summary}\n\n"
            f"**Conformidad Legal Global:** {'Aprobado' if analysis_res.overall_compliance else 'Requiere Modificaciones / Riesgos Detectados'}\n\n"
            f"#### Observaciones y Riesgos Detectados ({len(analysis_res.risks_detected)}):\n"
        )
        for r in analysis_res.risks_detected:
            final_text += (
                f"- **{r.clause_reference}** [Severidad: {r.severity.value.upper()}]: {r.legal_issue}\n"
                f"  *Texto:* _{r.clause_text_excerpt}_\n"
            )
            if r.suggested_revision:
                final_text += f"  *Propuesta de corrección:* {r.suggested_revision}\n"

    except Exception as exc:
        logger.warning("Fallo en estructurado de auditoría, generando análisis directo: %s", exc)
        raw_res = await llm.ainvoke(messages)
        final_text = str(raw_res.content)

    return {
        "citations": citations,
        "draft_answer": final_text,
        "confidence": ConfidenceLevel.HIGH if citations else ConfidenceLevel.MEDIUM,
        "messages": [AIMessage(content=final_text, name=AgentRole.DOCUMENT_ANALYZER.value)],
    }
