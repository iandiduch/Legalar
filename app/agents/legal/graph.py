"""Grafo de orquestación LangGraph para el Agente Legal Argentino."""

import logging
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from app.agents.legal.diff_node import diff_node
from app.agents.legal.document_analyzer import document_analyzer_node
from app.agents.legal.legal_agent import legal_agent_node
from app.agents.legal.router import router_node
from app.agents.legal.state import LegalAgentState
from app.agents.legal.validator import legal_validator_node
from app.domain.models import QueryIntent

logger = logging.getLogger(__name__)


def route_intent_edge(state: LegalAgentState) -> str:
    """Rutea según la intención clasificada por el router."""
    intent = state.get("intent")
    if intent == QueryIntent.DOCUMENT_ANALYSIS:
        return "document_analyzer"
    if intent == QueryIntent.VERSION_DIFF:
        return "diff_node"
    return "legal_agent"


def build_legal_graph(checkpointer: BaseCheckpointSaver | None = None) -> Any:
    """Construye y compila el flujo del Agente Legal en LangGraph."""
    workflow = StateGraph(LegalAgentState)

    # 1. Definir nodos
    workflow.add_node("router", router_node)
    workflow.add_node("legal_agent", legal_agent_node)
    workflow.add_node("document_analyzer", document_analyzer_node)
    workflow.add_node("diff_node", diff_node)
    workflow.add_node("validator", legal_validator_node)

    # 2. Conectar aristas
    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_intent_edge,
        {
            "legal_agent": "legal_agent",
            "document_analyzer": "document_analyzer",
            "diff_node": "diff_node",
        },
    )

    # Los agentes de consulta jurídica y auditoría pasan por el validador
    workflow.add_edge("legal_agent", "validator")
    workflow.add_edge("document_analyzer", "validator")
    # El diff normativo es fáctico y determinista (Git nativo), finaliza directamente sin latencia
    workflow.add_edge("diff_node", END)

    # El validador finaliza el ciclo para consultas y análisis
    workflow.add_edge("validator", END)

    return workflow.compile(checkpointer=checkpointer)
