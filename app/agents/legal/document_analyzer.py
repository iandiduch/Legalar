"""Nodo de Análisis Documental Jurídico: Auditoría de contratos y documentos contra el derecho argentino."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agents.legal.state import LegalAgentState
from app.core.structured_output import invoke_structured_with_retry
from app.domain.models import AgentRole, ConfidenceLevel
from app.schemas.legal.analysis import DocumentAnalysisResponse, DocumentRiskItem
from app.services.legal.retriever import HybridLegalRetriever
from app.services.prompt_manager import resolve_prompt

logger = logging.getLogger(__name__)


async def document_analyzer_node(
    state: LegalAgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Audita contratos y convenios contra el ordenamiento jurídico argentino."""
    llm = config["configurable"]["llm_client"]
    retriever: HybridLegalRetriever = config["configurable"].get("legal_retriever")
    settings = config["configurable"].get("settings")

    doc_text = state.get("document_text") or state.get("query") or ""
    doc_type = state.get("document_type") or "contrato"

    if len(doc_text.strip()) < 30:
        clarification = (
            "No se ha provisto el texto de las cláusulas del contrato o documento para auditar. "
            "Por favor, transcribe o adjunta el texto completo del contrato o convenio que deseas revisar."
        )
        return {
            "citations": [],
            "draft_answer": clarification,
            "final_answer": clarification,
            "confidence": ConfidenceLevel.LOW,
            "messages": [AIMessage(content=clarification, name=AgentRole.DOCUMENT_ANALYZER.value)],
        }

    citations = []
    if retriever and doc_text:
        query_snippet = doc_text[:400]
        try:
            citations = await retriever.retrieve_citations(
                query=f"{doc_type} validez cláusulas {query_snippet}",
                top_k=4,
            )
        except Exception as exc:
            logger.warning("No se pudieron recuperar citas para auditoría documental: %s", exc)

    evidence_summary = (
        "\n".join(f"- {c.law_identifier} Art. {c.article_number}: {(c.exact_quote or '').strip()[:3500]}" for c in citations)
        if citations
        else "Sin citas normativas directas precargadas."
    )

    base_prompt = await resolve_prompt(config, "document_analyzer")
    system_instruction = (
        f"{base_prompt}\n\n"
        f"NORMAS DE REFERENCIA RECUPERADAS:\n{evidence_summary}"
    )

    safe_doc_text = doc_text.replace("</contract_to_audit>", "[TAG_ESCAPADO]")

    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(
            content=(
                f"Por favor audita minuciosamente el siguiente documento legal ({doc_type}) según las directivas:\n\n"
                f"<contract_to_audit>\n{safe_doc_text}\n</contract_to_audit>"
            )
        ),
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
