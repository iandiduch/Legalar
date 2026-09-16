"""Setup de observabilidad: Arize Phoenix via OpenTelemetry/OpenInference.

Instrumentacion automatica (LangChain cubre los nodos de LangGraph porque son
Runnables; OpenAI cubre llamadas que no pasan por LangChain, como el worker de
ingesta) mas spans manuales para eventos de negocio -- ver app/core/tracing_utils.py.

Si Phoenix no esta disponible, register() igual arma un TracerProvider con un
BatchSpanProcessor: los exports fallan en segundo plano sin bloquear la app, asi
que no hace falta un try/except defensivo aca.
"""

from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from phoenix.otel import register

from app.core.config import Settings

_tracer_provider: TracerProvider | None = None


def setup_telemetry(settings: Settings) -> TracerProvider | None:
    global _tracer_provider

    if not settings.PHOENIX_ENABLED:
        return None

    tracer_provider = register(
        project_name=settings.PHOENIX_PROJECT_NAME,
        endpoint=settings.PHOENIX_COLLECTOR_ENDPOINT,
        auto_instrument=False,
        batch=True,
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    _tracer_provider = tracer_provider
    return tracer_provider


def shutdown_telemetry() -> None:
    global _tracer_provider
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        _tracer_provider = None
