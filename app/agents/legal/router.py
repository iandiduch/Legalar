"""Router inteligente para clasificar consultas jurídicas hacia el nodo de ejecución adecuado."""

import logging
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agents.legal.state import LegalAgentState
from app.core.structured_output import invoke_structured_with_retry
from app.domain.models import QueryIntent
from app.services.legal.web_reader import fetch_web_page_content, find_urls_in_text
from app.services.prompt_manager import resolve_prompt

logger = logging.getLogger(__name__)


class RouterDecision(BaseModel):
    intent: QueryIntent = Field(
        description=(
            "Intención clasificada. "
            "Debe ser 'legal_consultation' para la gran mayoría de consultas jurídicas sustantivas: asesoramiento ante casos reales, conflictos, oposiciones de marcas ante el INPI, mediaciones, cartas documento, telegramas, juicios, plazos legales, indemnizaciones, cálculos de rubros o reformas de leyes. "
            "Debe ser 'document_analysis' ÚNICAMENTE si el usuario pide explícitamente auditar o revisar las cláusulas de un contrato, convenio o acuerdo para verificar cláusulas abusivas, nulas o riesgos de validez. "
            "Debe ser 'version_diff' ÚNICAMENTE si el usuario solicita explícitamente ver el diff técnico o cotejo textual de redacciones de una ley. "
            "Debe ser 'general_inquiry' para saludos, agradecimientos o preguntas sobre la identidad del asistente."
        )
    )
    law_identifier_hint: str | None = Field(default=None, description="Norma mencionada explícitamente (ej: 'LEY-26994', 'LEY-22362')")
    article_hint: str | None = Field(
        default=None,
        description="Número de artículo específico si fue mencionado explícitamente (ej: '1198', '245', '14 bis', '3'). Null si la consulta es conceptual o no refiere a un artículo puntual.",
    )
    wants_explanation: bool = Field(
        default=False,
        description="Aplica ÚNICAMENTE si intent='version_diff': True si el usuario solicita explicación ciudadana de la reforma. False si no pide pedagogía o si intent no es version_diff.",
    )
    reasoning: str = Field(description="Explicación concisa de la decisión de ruteo considerando el historial conversacional")


async def router_node(state: LegalAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Clasifica la consulta, detecta enlaces web para fact-checking e indicios de leyes."""
    configurable = config.get("configurable", {})
    llm = configurable.get("llm_client")
    settings = configurable.get("settings")

    # 1. Si ya se especificó document_text en el estado (ej: endpoint /analyze), es DOCUMENT_ANALYSIS
    if state.get("document_text"):
        return {"intent": QueryIntent.DOCUMENT_ANALYSIS}

    # 2. Detección automática de enlaces web (URL Fact-Checking)
    urls = find_urls_in_text(state["query"])
    if urls:
        target_url = urls[0]
        logger.info("Detectado enlace web en la consulta: %s", target_url)
        web_res = await fetch_web_page_content(target_url, timeout_seconds=4.0)

        if web_res.success:
            doc_text = (
                f"URL FUENTE: {web_res.url} (Dominio: {web_res.domain})\n"
                f"TÍTULO DE LA PUBLICACIÓN: {web_res.title or 'Sin título'}\n\n"
                f"CONTENIDO EXTRAÍDO:\n{web_res.clean_text}"
            )
            return {
                "intent": QueryIntent.URL_FACT_CHECK,
                "document_text": doc_text,
                "document_type": f"enlace_web ({web_res.domain})",
            }
        else:
            fallback_text = (
                f"AVISO SOBRE EL ENLACE ({target_url}):\n{web_res.error}\n\n"
                "Instrucción: No fue posible acceder al contenido del enlace debido a la restricción indicada. "
                "Responde la consulta jurídica planteada por el usuario en su texto, aclarando de forma concisa "
                "que no se pudo leer el contenido del link provisto."
            )
            return {
                "intent": QueryIntent.LEGAL_CONSULTATION,
                "document_text": fallback_text,
                "document_type": "enlace_no_accesible",
            }

    if not llm:
        # Fallback determinista cuando no hay cliente LLM configurado
        q = state["query"].lower().strip()
        if q in ("hola", "buenas", "buen dia", "buen día", "buenas tardes", "buenas noches", "gracias", "muchas gracias", "chau", "hola!"):
            return {"intent": QueryIntent.GENERAL_INQUIRY}
        if "diff" in q or "cotejo textual" in q:
            return {"intent": QueryIntent.VERSION_DIFF}
        return {"intent": QueryIntent.LEGAL_CONSULTATION}

    try:
        router_prompt = await resolve_prompt(config, "router")
        recent_messages = state["messages"][-6:] if len(state["messages"]) > 6 else state["messages"]
        messages = [
            SystemMessage(content=router_prompt),
            *recent_messages,
        ]
        decision: RouterDecision = await invoke_structured_with_retry(
            llm, RouterDecision, router_prompt, messages, max_attempts=settings.STRUCTURED_OUTPUT_MAX_ATTEMPTS if settings else 2
        )

        # Saneamiento estructural: si el modelo extrajo una frase descriptiva en vez de un artículo puntual, limpiarla
        if decision.article_hint:
            cleaned_hint = decision.article_hint.strip()
            if len(cleaned_hint.split()) > 3 or not any(char.isdigit() for char in cleaned_hint):
                logger.info("router_node: descartando article_hint no numérico/descriptivo: %r", cleaned_hint)
                decision.article_hint = None

        diff_params = dict(state.get("diff_request_params") or {})
        if decision.law_identifier_hint:
            diff_params["law_identifier"] = decision.law_identifier_hint
        if decision.article_hint:
            diff_params["article_number"] = decision.article_hint
        diff_params["wants_explanation"] = decision.wants_explanation

        return {
            "intent": decision.intent,
            "diff_request_params": diff_params or None,
        }
    except Exception as exc:
        logger.warning("Fallo en router estructurado, usando fallback: %s", exc)
        q = state["query"].lower().strip()
        if q in ("hola", "buenas", "buen dia", "buen día", "buenas tardes", "buenas noches", "gracias", "muchas gracias", "chau", "hola!"):
            return {"intent": QueryIntent.GENERAL_INQUIRY}
        return {"intent": QueryIntent.LEGAL_CONSULTATION}

