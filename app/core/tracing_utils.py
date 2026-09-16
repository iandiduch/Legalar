"""Spans manuales para eventos de negocio que la auto-instrumentación de
OpenInference no puede conocer (suficiencia de retrieval, decisiones de supervisor,
escalamientos HOTL y anomalías de cuota/rate-limiting de proveedores LLM).

Complementa las trazas automáticas generadas por LangChainInstrumentor y OpenAIInstrumentor.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.domain.models import AgentRole

_tracer = trace.get_tracer("intelligence-system.agents")


def _record_span_exception(span: trace.Span, exc: Exception) -> None:
    """Registra la excepción en el span con atributos estandarizados para alertas."""
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))
    span.set_attribute("error.type", exc.__class__.__name__)
    span.set_attribute("error.message", str(exc))

    error_text = str(exc).lower()
    if "rate_limit" in error_text or "429" in error_text or "quota" in error_text or "insufficient_quota" in error_text:
        span.set_attribute("llm.provider.rate_limited", True)
        span.set_attribute("error.quota_exceeded", True)


@contextmanager
def agent_span(name: str, *, thread_id: str, agent: AgentRole, **attributes: Any) -> Iterator[trace.Span]:
    with _tracer.start_as_current_span(name) as span:
        span.set_attribute("session.id", thread_id)
        span.set_attribute("agent.name", agent.value)
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            _record_span_exception(span, exc)
            raise


@contextmanager
def hotl_escalation_span(
    *,
    thread_id: str,
    ticket_id: str,
    reason: str,
    priority: str,
    contact_channel: str,
    origin_agent: str = AgentRole.ESCALATION.value,
) -> Iterator[trace.Span]:
    """los eventos HOTL deben quedar identificados con
    atributos propios, permitiendo filtros directos en Arize Phoenix."""
    with _tracer.start_as_current_span("hotl.escalation") as span:
        span.set_attribute("session.id", thread_id)
        span.set_attribute("hotl.escalation", True)
        span.set_attribute("hotl.ticket_id", ticket_id)
        span.set_attribute("hotl.thread_id", thread_id)
        span.set_attribute("hotl.reason", reason)
        span.set_attribute("hotl.priority", priority)
        span.set_attribute("hotl.contact_channel", contact_channel)
        span.set_attribute("hotl.origin_agent", origin_agent)
        span.add_event("hotl.ticket_created")
        try:
            yield span
        except Exception as exc:
            _record_span_exception(span, exc)
            raise
