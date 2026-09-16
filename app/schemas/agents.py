"""Schemas de salida estructurada de los nodos del Agente Legal Argentino."""

from pydantic import BaseModel, Field

from app.domain.models import ConfidenceLevel, QueryIntent
from app.schemas.legal.chat import LegalValidationSummary
from app.schemas.legal.citation import LegalCitation



class RouterDecision(BaseModel):
    intent: QueryIntent = Field(description="Intención clasificada de la consulta del usuario")
    law_identifier_hint: str | None = Field(default=None, description="Norma mencionada (ej: 'LEY-26994')")
    article_hint: str | None = Field(default=None, description="Artículo mencionado (ej: '1198')")
    reasoning: str = Field(description="Explicación concisa de la decisión de ruteo")


class LegalAnswerPayload(BaseModel):
    answer: str = Field(description="Respuesta jurídica integral con citas normativas específicas")
    articles_referenced: list[str] = Field(
        default_factory=list,
        description="Identificadores de artículos citados (ej: ['LEY-26994:1198'])",
    )
    confidence: ConfidenceLevel = Field(description="Nivel de certidumbre según el respaldo normativo disponible")
    unsupported_notes: list[str] = Field(
        default_factory=list,
        description="Aspectos de la consulta que la legislación no cubre de forma expresa",
    )
