"""Estado del Grafo Multi-Paso del Agente Legal Argentino."""

from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.domain.models import ConfidenceLevel, QueryIntent
from app.schemas.legal.chat import LegalValidationSummary
from app.schemas.legal.citation import LegalCitation
from app.schemas.legal.diff import DiffResponse


class LegalAgentState(TypedDict):
    """Estado compartido a través de los nodos del flujo LangGraph."""

    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str
    query: str
    intent: QueryIntent
    citations: list[LegalCitation]
    draft_answer: str | None
    validation_result: LegalValidationSummary | None
    document_text: str | None
    document_type: str | None
    diff_request_params: dict[str, Any] | None
    diff_result: DiffResponse | None
    final_answer: str | None
    confidence: ConfidenceLevel
    iteration: int
