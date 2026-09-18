"""Módulo de Idempotencia y Deduplicación con Redis para operaciones del Agente Legal.

Evita el antipatrón de doble ejecución ("el bug del doble cobro de tokens")
ante clics repetidos, reconexiones o reintentos de clientes HTTP.
"""

import hashlib
import json
import logging
from typing import Any

from fastapi import HTTPException, Request, status
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

IDEMPOTENCY_HEADER = "Idempotency-Key"


class IdempotencyStatus:
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"


def generate_idempotency_fingerprint(client_ip: str, path: str, body_str: str) -> str:
    """Genera una huella SHA-256 única a partir de la IP, la ruta y el contenido."""
    raw = f"{client_ip}|{path}|{body_str}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def check_or_acquire_idempotency(
    redis: Redis | None,
    idempotency_key: str,
    ttl_seconds: int = 60,
) -> tuple[str, dict[str, Any] | None]:
    """Evalúa el estado de idempotencia para la clave dada.

    Retorna:
        ("ACQUIRED", None) -> El proceso puede continuar (se adquirió el lock 'PROCESSING').
        ("PROCESSING", None) -> Ya hay una petición idéntica en ejecución (debe bloquearse).
        ("COMPLETED", cached_data) -> La petición ya finalizó previamente; se devuelve la respuesta en caché.
    """
    if redis is None:
        return "ACQUIRED", None

    redis_key = f"idempotency:{idempotency_key}"

    try:
        # Intentar obtener estado actual si ya existe
        existing = await redis.get(redis_key)
        if existing:
            try:
                data = json.loads(existing)
                if data.get("status") == IdempotencyStatus.COMPLETED:
                    logger.info("Idempotencia: respuesta recuperada de caché para clave '%s'", idempotency_key[:16])
                    return "COMPLETED", data.get("payload")
                elif data.get("status") == IdempotencyStatus.PROCESSING:
                    logger.warning("Idempotencia: solicitud duplicada en vuelo bloqueada ('%s')", idempotency_key[:16])
                    return "PROCESSING", None
            except Exception:
                pass

        # Adquirir lock atómico con SET NX
        payload_init = json.dumps({"status": IdempotencyStatus.PROCESSING})
        acquired = await redis.set(redis_key, payload_init, nx=True, ex=ttl_seconds)

        if not acquired:
            # Otra coroutine o worker adquirió el lock una fracción de segundo antes
            return "PROCESSING", None

        return "ACQUIRED", None
    except Exception as exc:
        logger.warning("Fallo en verificación de idempotencia con Redis: %s", exc)
        return "ACQUIRED", None


async def save_idempotency_result(
    redis: Redis | None,
    idempotency_key: str,
    payload: dict[str, Any],
    ttl_seconds: int = 60,
) -> None:
    """Guarda el resultado exitoso en Redis para responder de forma instantánea a duplicados."""
    if redis is None:
        return

    redis_key = f"idempotency:{idempotency_key}"
    try:
        data = json.dumps({"status": IdempotencyStatus.COMPLETED, "payload": payload})
        await redis.set(redis_key, data, ex=ttl_seconds)
    except Exception as exc:
        logger.warning("No se pudo persistir el resultado de idempotencia en Redis: %s", exc)


async def release_idempotency_lock(
    redis: Redis | None,
    idempotency_key: str,
) -> None:
    """Libera el lock ante un fallo en el procesamiento para que el usuario pueda reintentar."""
    if redis is None:
        return

    redis_key = f"idempotency:{idempotency_key}"
    try:
        await redis.delete(redis_key)
    except Exception as exc:
        logger.warning("No se pudo liberar el lock de idempotencia en Redis: %s", exc)
