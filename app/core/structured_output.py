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

from typing import TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_RETRYABLE_ERRORS = (ValidationError, OutputParserException)


async def invoke_structured_with_retry(
    llm: BaseChatModel,
    schema: type[SchemaT],
    system_prompt: str,
    messages: list[BaseMessage],
    max_attempts: int,
) -> SchemaT:
    structured_llm = llm.with_structured_output(schema, method="function_calling")
    feedback: str | None = None

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        reraise=True,
    ):
        with attempt:
            prompt = system_prompt if feedback is None else f"{system_prompt}\n\nCORRECCION REQUERIDA: {feedback}"
            try:
                result = await structured_llm.ainvoke([SystemMessage(content=prompt), *messages])
            except _RETRYABLE_ERRORS as exc:
                feedback = str(exc)
                raise
            return result  # type: ignore[return-value]

    raise AssertionError("unreachable: AsyncRetrying siempre retorna o relanza")
