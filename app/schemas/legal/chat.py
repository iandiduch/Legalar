"""Schemas de solicitud y respuesta para la consulta interactiva jurídica."""

from typing import Any
from pydantic import BaseModel, Field
from app.domain.models import ConfidenceLevel, QueryIntent
from app.schemas.legal.citation import LegalCitation


class LegalValidationSummary(BaseModel):
    """Resultado de la verificación de consistencia jurídica."""
    is_valid: bool = Field(description="Si todas las citas y afirmaciones están respaldadas")
    all_norms_in_force: bool = Field(default=True, description="Si todas las normas citadas están vigentes")
    unsupported_claims: list[str] = Field(default_factory=list, description="Afirmaciones sin sustento normativo")
    warning_notes: list[str] = Field(default_factory=list, description="Advertencias sobre reformas recientes o derogaciones")


class LegalChatRequest(BaseModel):
    """Consulta del usuario al Agente Legal Argentino."""
    query: str = Field(min_length=3, description="Pregunta jurídica sobre el ordenamiento argentino")
    thread_id: str | None = Field(default=None, description="Identificador de sesión para conversación continua")
    include_historical: bool = Field(default=False, description="Si se deben incluir normas derogadas/históricas")
    law_filter: list[str] | None = Field(default=None, description="Filtro opcional por identificadores (ej: ['LEY-26994'])")


class LegalChatResponse(BaseModel):
    """Respuesta generada por el Agente Legal con trazabilidad de artículos."""
    thread_id: str
    query: str
    intent: QueryIntent
    answer: str = Field(description="Fundamentación jurídica redactada en lenguaje claro y técnico")
    citations: list[LegalCitation] = Field(default_factory=list, description="Artículos y normas de sustento")
    confidence: ConfidenceLevel
    validation: LegalValidationSummary
    processing_time_seconds: float = 0.0
