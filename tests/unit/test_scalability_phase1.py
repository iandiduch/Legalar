"""Tests unitarios para la Fase 1 de Escalabilidad:
1. Configuración de Pool de Postgres y Concurrencia de Workers.
2. Mecanismo de Idempotencia y Deduplicación con Redis (Anti-Doble Cobro).
3. Backoff Exponencial en llamadas LLM ante HTTP 429 / Rate Limits.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.idempotency import (
    IdempotencyStatus,
    check_or_acquire_idempotency,
    generate_idempotency_fingerprint,
    release_idempotency_lock,
    save_idempotency_result,
)
from app.core.structured_output import invoke_structured_with_retry


class DummySchema(BaseModel):
    mensaje: str = Field(description="Mensaje de prueba")


def test_scalability_pool_and_worker_settings():
    """Valida que Settings cargue los parámetros ampliados de la Fase 1 para 100 usuarios."""
    settings = Settings(
        OPENAI_API_KEY="dummy",
        PINECONE_API_KEY="dummy",
    )
    assert settings.DB_POOL_SIZE >= 20
    assert settings.DB_MAX_OVERFLOW >= 30
    assert settings.CHECKPOINTER_POOL_MAX_SIZE >= 20
    assert settings.UVICORN_WORKERS >= 4
    assert settings.IDEMPOTENCY_ENABLED is True
    assert settings.IDEMPOTENCY_TTL_SECONDS >= 30


def test_generate_idempotency_fingerprint():
    """Verifica que huellas idénticas generen el mismo hash SHA-256."""
    h1 = generate_idempotency_fingerprint("192.168.1.1", "/api/v1/legal/chat", "consulta|thread-123")
    h2 = generate_idempotency_fingerprint("192.168.1.1", "/api/v1/legal/chat", "consulta|thread-123")
    h3 = generate_idempotency_fingerprint("192.168.1.2", "/api/v1/legal/chat", "consulta|thread-123")

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64


@pytest.mark.asyncio
async def test_idempotency_workflow_with_mock_redis():
    """Valida el ciclo de vida completo de idempotencia: Lock -> Processing -> Completed -> Cache Hit."""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True

    key = "test-key-123"

    # 1. Primera petición adquiere el lock
    status_1, payload_1 = await check_or_acquire_idempotency(mock_redis, key, ttl_seconds=60)
    assert status_1 == "ACQUIRED"
    assert payload_1 is None

    # 2. Petición concurrente duplicada en vuelo (devuelve PROCESSING)
    mock_redis.get.return_value = json.dumps({"status": IdempotencyStatus.PROCESSING})
    status_2, payload_2 = await check_or_acquire_idempotency(mock_redis, key, ttl_seconds=60)
    assert status_2 == "PROCESSING"
    assert payload_2 is None

    # 3. Guardar resultado final exitoso
    sample_result = {"answer": "Dictamen legal fundamentado", "citations": []}
    await save_idempotency_result(mock_redis, key, sample_result, ttl_seconds=60)
    mock_redis.set.assert_called()

    # 4. Petición posterior idéntica devuelve la respuesta cacheada (Zero tokens)
    mock_redis.get.return_value = json.dumps({"status": IdempotencyStatus.COMPLETED, "payload": sample_result})
    status_3, payload_3 = await check_or_acquire_idempotency(mock_redis, key, ttl_seconds=60)
    assert status_3 == "COMPLETED"
    assert payload_3 == sample_result

    # 5. Liberación de lock ante error
    await release_idempotency_lock(mock_redis, key)
    mock_redis.delete.assert_called_with(f"idempotency:{key}")


@pytest.mark.asyncio
async def test_invoke_structured_with_retry_backs_off_on_429():
    """Verifica que invoke_structured_with_retry soporte rate limits HTTP 429 aplicando backoff."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()

    # Primera llamada arroja HTTP 429 Too Many Requests, segunda llamada responde con éxito
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp_429 = httpx.Response(429, request=req)
    err_429 = httpx.HTTPStatusError("Rate limit exceeded", request=req, response=resp_429)

    success_result = DummySchema(mensaje="Respuesta procesada tras reintento")
    mock_structured.ainvoke = AsyncMock(side_effect=[err_429, success_result])
    mock_llm.with_structured_output.return_value = mock_structured

    # Parchear wait_exponential para no demorar la prueba unitaria
    with patch("app.core.structured_output.wait_exponential", return_value=lambda retry_state: 0.01):
        result = await invoke_structured_with_retry(
            llm=mock_llm,
            schema=DummySchema,
            system_prompt="Test system prompt",
            messages=[],
            max_attempts=3,
        )

    assert result.mensaje == "Respuesta procesada tras reintento"
    assert mock_structured.ainvoke.call_count == 2
