"""Nodo de Consultas Generales y Saludos (General Inquiry Node).

Maneja saludos, agradecimientos, despedidas y preguntas sobre las capacidades de Legalar
de forma ultra-rápida, cordial y sin ejecutar recuperación RAG ni citar normas innecesarias.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agents.legal.state import LegalAgentState
from app.domain.models import AgentRole, ConfidenceLevel
from app.services.prompt_manager import resolve_prompt

logger = logging.getLogger(__name__)


async def general_inquiry_node(state: LegalAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Responde a saludos y consultas no jurídicas de forma ágil sin RAG ni citas."""
    configurable = config.get("configurable", {})
    llm = configurable.get("llm_client")
    query = state.get("query", "").strip()
    messages_history = state.get("messages", [])

    fallback_text = (
        "¡Hola! Soy Legalar, tu asistente especializado en legislación y derecho argentino. "
        "Puedo ayudarte a resolver dudas normativas, comparar reformas de leyes, auditar contratos o "
        "verificar la veracidad legal de noticias. ¿En qué consulta jurídica puedo ayudarte hoy?"
    )

    if not llm:
        return {
            "citations": [],
            "draft_answer": fallback_text,
            "final_answer": fallback_text,
            "confidence": ConfidenceLevel.HIGH,
            "messages": [AIMessage(content=fallback_text, name=AgentRole.LEGAL_AGENT.value)],
        }

    inquiry_prompt = await resolve_prompt(config, "general_inquiry")
    recent_messages = messages_history[-4:] if len(messages_history) > 4 else messages_history
    messages = [
        SystemMessage(content=inquiry_prompt),
        *recent_messages,
        HumanMessage(content=query),
    ]

    try:
        # Invocación directa y ligera con max_tokens acotado para respuesta instantánea (< 500ms)
        res = await llm.ainvoke(messages, config={"max_tokens": 200})
        answer = str(res.content).strip()
        if not answer:
            answer = fallback_text
    except Exception as exc:
        logger.warning("Fallo en general_inquiry_node, usando fallback: %s", exc)
        answer = fallback_text

    return {
        "citations": [],
        "draft_answer": answer,
        "final_answer": answer,
        "confidence": ConfidenceLevel.HIGH,
        "messages": [AIMessage(content=answer, name=AgentRole.LEGAL_AGENT.value)],
    }
