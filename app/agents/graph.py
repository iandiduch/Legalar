"""Topología del grafo del Agente Legal Argentino.

Flujo de producción del dominio legal:
Router -> [Legal Agent | Document Analyzer | Diff Node] -> Validator -> END.
"""

from typing import Any
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from app.agents.legal.graph import build_legal_graph


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Construye y compila el grafo del Agente Legal Argentino."""
    return build_legal_graph(checkpointer=checkpointer)
