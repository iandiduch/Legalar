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

logger = logging.getLogger(__name__)

GENERAL_INQUIRY_PROMPT = """Eres Legalar, un asistente inteligente de grado de producción especializado en Derecho Positivo Argentino.

El usuario se ha comunicado con un saludo, agradecimiento, despedida o pregunta general sobre qué puedes hacer.
Tu objetivo es responder de manera amable, cercana, profesional y concisa (1 o 2 párrafos breves).

Directivas obligatorias:
1. Responde cordialmente al saludo o comentario del usuario.
2. Explica brevemente en qué puedes asistirlo:
   - Consultas doctrinales y normativas sobre leyes argentinas (Código Civil y Comercial, Código Penal, Ley de Contrato de Trabajo, etc.).
   - Impacto de reformas recientes (DNU 70/2023, Ley de Bases).
   - Auditoría y análisis de contratos, convenios o cartas documento.
   - Verificación jurídica de noticias o enlaces web sobre temas legales.
3. Invita al usuario a formular su consulta legal específica o plantear su caso.
4. PROHIBIDO citar artículos, leyes o números normativos si el usuario solo está saludando o haciendo una consulta general.
5. NO uses saludos formales epistolares acartonados de carta ("Estimado/a", "De mi mayor consideración"); mantén un tono moderno, profesional y conversacional.
"""


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

    recent_messages = messages_history[-4:] if len(messages_history) > 4 else messages_history
    messages = [
        SystemMessage(content=GENERAL_INQUIRY_PROMPT),
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
