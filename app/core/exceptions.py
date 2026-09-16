"""Jerarquia centralizada de excepciones de dominio.

Regla del proyecto: nunca `except Exception: pass`. Cada capa (servicios, agentes,
worker) debe lanzar una subclase especifica de aca abajo. El unico lugar donde se
atrapa `Exception` de forma generica es el handler de borde en app/main.py, como
red de seguridad final, y siempre logueado.
"""

from typing import Any


class AppException(Exception):
    http_status: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


# --- Errores de dominio (400) ---


class DomainError(AppException):
    http_status = 400
    error_code = "domain_error"


class BusinessValidationError(DomainError):
    """Validacion semantica/de negocio, distinta de la validacion de schema de Pydantic."""

    http_status = 422
    error_code = "validation_error"


class UnsupportedFileTypeError(DomainError):
    error_code = "unsupported_file_type"


class DocumentParsingError(DomainError):
    error_code = "document_parsing_error"


class SuspiciousContentDetectedError(DomainError):
    """El texto extraido de un documento matcheo un patron heuristico de
    prompt injection (ver app/core/prompt_injection_scanner.py). Se trata
    como error de ingesta, no de infraestructura -- es un rechazo
    deliberado, no una falla tecnica."""

    error_code = "suspicious_content_detected"


class NotFoundError(DomainError):
    http_status = 404
    error_code = "not_found"


class TicketNotFoundError(NotFoundError):
    error_code = "ticket_not_found"


class JobNotFoundError(NotFoundError):
    error_code = "job_not_found"


class PromptNotFoundError(NotFoundError):
    error_code = "prompt_not_found"


class ConflictError(DomainError):
    http_status = 409
    error_code = "conflict"


class InvalidStateTransitionError(ConflictError):
    error_code = "invalid_state_transition"


# --- Errores de infraestructura (502/503) ---


class InfrastructureError(AppException):
    http_status = 503
    error_code = "infrastructure_error"


class DatabaseError(InfrastructureError):
    error_code = "database_error"


class RedisUnavailableError(InfrastructureError):
    error_code = "redis_unavailable"


class VectorStoreError(InfrastructureError):
    error_code = "vector_store_error"


class EmbeddingGenerationError(InfrastructureError):
    error_code = "embedding_generation_error"


class PineconeUpsertError(InfrastructureError):
    error_code = "pinecone_upsert_error"


class LLMProviderError(InfrastructureError):
    http_status = 502
    error_code = "llm_provider_error"


class LegalAPIError(InfrastructureError):
    http_status = 502
    error_code = "legal_api_error"



# --- Errores de ejecucion de agentes (500) ---


class AgentExecutionError(AppException):
    http_status = 500
    error_code = "agent_execution_error"


class ConversationLoopLimitError(AgentExecutionError):
    error_code = "conversation_loop_limit"


# --- Errores de seguridad (401/403/429) ---


class AuthenticationError(AppException):
    http_status = 401
    error_code = "authentication_required"


class AuthorizationError(AppException):
    http_status = 403
    error_code = "forbidden"


class RateLimitExceededError(AppException):
    http_status = 429
    error_code = "rate_limit_exceeded"

    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("Se supero el limite de requests. Intenta de nuevo mas tarde.")
        self.retry_after_seconds = retry_after_seconds
