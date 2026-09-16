"""Schemas para análisis de documentos legales subidos por el usuario (contratos, demandas, convenios)."""

from enum import Enum
from pydantic import BaseModel, Field
from app.schemas.legal.citation import LegalCitation


class RiskSeverity(str, Enum):
    HIGH = "high"         # Cláusula nula de nulidad absoluta, ilícita o prohibida
    MEDIUM = "medium"     # Cláusula potencialmente abusiva o sujeta a contingencia
    LOW = "low"           # Recomendación de redacción o ambigüedad
    INFO = "info"         # Información normativa relevante


class DocumentRiskItem(BaseModel):
    """Observación jurídica o riesgo detectado en una cláusula de un documento analizado."""
    clause_reference: str = Field(description="Ej: 'Cláusula Cuarta: Plazo de Locación'")
    clause_text_excerpt: str = Field(description="Texto del documento observado")
    severity: RiskSeverity
    legal_issue: str = Field(description="Descripción del conflicto o riesgo normativo")
    applicable_laws: list[LegalCitation] = Field(default_factory=list, description="Normas argentinas involucradas")
    suggested_revision: str | None = Field(default=None, description="Redacción alternativa propuesta")


class DocumentAnalysisRequest(BaseModel):
    """Petición de análisis sobre un texto o documento cargado."""
    document_text: str = Field(min_length=20, description="Texto completo del contrato o documento a examinar")
    document_type: str = Field(default="contrato", description="Tipo de documento (contrato, estatuto, convenio, etc.)")
    specific_questions: list[str] | None = Field(default=None, description="Consultas específicas sobre el documento")


class DocumentAnalysisResponse(BaseModel):
    """Dictamen del análisis jurídico automatizado sobre el documento del usuario."""
    document_type: str
    executive_summary: str = Field(description="Resumen ejecutivo del estado legal del documento")
    overall_compliance: bool = Field(description="Si el documento en general se ajusta al ordenamiento argentino")
    risks_detected: list[DocumentRiskItem] = Field(default_factory=list)
    relevant_statutes_applied: list[LegalCitation] = Field(default_factory=list)
    conclusions_and_next_steps: list[str] = Field(default_factory=list)
