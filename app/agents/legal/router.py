"""Router inteligente para clasificar consultas jurídicas hacia el nodo de ejecución adecuado."""

import logging
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agents.legal.state import LegalAgentState
from app.core.structured_output import invoke_structured_with_retry
from app.domain.models import QueryIntent

logger = logging.getLogger(__name__)


class RouterDecision(BaseModel):
    intent: QueryIntent = Field(description="Intención clasificada de la consulta del usuario")
    law_identifier_hint: str | None = Field(default=None, description="Norma mencionada explícitamente (ej: 'LEY-26994')")
    article_hint: str | None = Field(default=None, description="Artículo mencionado (ej: '1198')")
    reasoning: str = Field(description="Explicación concisa de la decisión de ruteo")


ROUTER_PROMPT = """Eres el clasificador de intenciones del Agente Legal Argentino.
Tu objetivo es determinar qué tipo de tarea jurídica solicita el usuario:
- 'legal_consultation': Pregunta doctrinaria, interpretativa o consulta sobre leyes argentinas vigentes.
- 'document_analysis': El usuario proporciona un texto de contrato, convenio, carta documento o pide analizar cláusulas.
- 'version_diff': El usuario pregunta qué cambió en una ley, cómo era la redacción anterior, qué modificó una reforma (ej DNU 70/2023 o Ley 27.551) respecto de artículos previos.
- 'general_inquiry': Preguntas generales, saludos o consultas no normativas.
"""


async def router_node(state: LegalAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Clasifica la consulta y detecta indicios de leyes y artículos."""
    configurable = config.get("configurable", {})
    llm = configurable.get("llm_client")
    settings = configurable.get("settings")

    # Si ya se especificó document_text en el estado, es DOCUMENT_ANALYSIS
    if state.get("document_text"):
        return {"intent": QueryIntent.DOCUMENT_ANALYSIS}

    if not llm:
        # Fallback determinista por palabras clave
        q = state["query"].lower()
        if "qué cambió" in q or "que cambio" in q or "reforma" in q or "redacción anterior" in q:
            return {"intent": QueryIntent.VERSION_DIFF}
        return {"intent": QueryIntent.LEGAL_CONSULTATION}

    try:
        messages = [
            SystemMessage(content=ROUTER_PROMPT),
            *state["messages"],
        ]
        decision: RouterDecision = await invoke_structured_with_retry(
            llm, RouterDecision, ROUTER_PROMPT, messages, max_attempts=settings.STRUCTURED_OUTPUT_MAX_ATTEMPTS if settings else 2
        )
        diff_params = dict(state.get("diff_request_params") or {})
        if decision.law_identifier_hint:
            diff_params["law_identifier"] = decision.law_identifier_hint
        if decision.article_hint:
            diff_params["article_number"] = decision.article_hint

        return {
            "intent": decision.intent,
            "diff_request_params": diff_params or None,
        }
    except Exception as exc:
        logger.warning("Fallo en router estructurado, usando fallback: %s", exc)
        return {"intent": QueryIntent.LEGAL_CONSULTATION}

