"""Patron de auto-correccion para salidas estructuradas: si el LLM devuelve algo
que no valida contra el schema Pydantic, se reintenta reinyectando el error como
instruccion de correccion en el prompt del intento siguiente.

`Runnable.with_structured_output(...).with_retry()` no alcanza para esto: reintenta
la misma invocacion sin poder mutar el prompt entre intentos. Por eso el loop es
manual con `AsyncRetrying`.

Cuidado con donde va el except: tiene que estar SIEMPRE dentro del bloque que
controla `AsyncRetrying` y relanzar la excepcion, nunca devolver un valor default
en su lugar -- si no, tenacity nunca ve el error y el retry queda declarado pero
inerte.
"""

import logging
from typing import TypeVar

import httpx
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

try:
    import openai

    _OPENAI_ERRORS = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )
except ImportError:
    _OPENAI_ERRORS = ()

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_BASE_RETRYABLE = (
    ValidationError,
    OutputParserException,
    httpx.ConnectError,
    httpx.TimeoutException,
    *_OPENAI_ERRORS,
)


def _is_retryable_exception(exc: BaseException) -> bool:
    """True para errores de validación de esquema, límites de tasa (429) o fallos de red transitorios."""
    if isinstance(exc, _BASE_RETRYABLE):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (429, 502, 503, 504):
        return True
    # Soporte para wrappers de LangChain que anidan excepciones en __cause__
    cause = getattr(exc, "__cause__", None)
    if cause and _is_retryable_exception(cause):
        return True
    return False


async def invoke_structured_with_retry(
    llm: BaseChatModel,
    schema: type[SchemaT],
    system_prompt: str,
    messages: list[BaseMessage],
    max_attempts: int = 3,
) -> SchemaT:
    """Invoca el LLM con salida estructurada aplicando auto-corrección de esquema y backoff exponencial."""
    structured_llm = llm.with_structured_output(schema, method="function_calling")
    feedback: str | None = None

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        retry=retry_if_exception(_is_retryable_exception),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    ):
        with attempt:
            prompt = system_prompt if feedback is None else f"{system_prompt}\n\nCORRECCION REQUERIDA: {feedback}"
            try:
                result = await structured_llm.ainvoke([SystemMessage(content=prompt), *messages])
            except Exception as exc:
                if isinstance(exc, (ValidationError, OutputParserException)):
                    feedback = str(exc)
                logger.warning(
                    "structured_output: reintento %d/%d tras error (%s): %s",
                    attempt.retry_state.attempt_number,
                    max_attempts,
                    type(exc).__name__,
                    exc,
                )
                raise
            return result  # type: ignore[return-value]

    raise AssertionError("unreachable: AsyncRetrying siempre retorna o relanza")

