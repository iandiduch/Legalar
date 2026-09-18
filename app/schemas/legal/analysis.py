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


class AnalysisJobResponse(BaseModel):
    """Respuesta inmediata HTTP 202 Accepted tras encolar el análisis del documento."""
    analysis_id: str = Field(description="Identificador único UUID de la auditoría para tracking y eventos SSE")
    status: str = Field(default="PENDING", description="Estado inicial del job (PENDING)")
    document_type: str = Field(description="Tipo de documento analizado (ej: contrato, carta_documento)")
    filename: str = Field(description="Nombre del archivo o identificador")
    message: str = Field(
        default="Auditoría encolada exitosamente para procesamiento en segundo plano.",
        description="Mensaje informativo para el cliente o frontend",
    )
    events_url: str = Field(
        description="URL para suscribirse a Server-Sent Events (SSE) y recibir progreso y resultado en vivo"
    )


class AnalysisJobStatusResponse(BaseModel):
    """Consulta de estado y resultado de un trabajo de análisis de documento."""
    analysis_id: str
    status: str = Field(description="PENDING, PROCESSING, COMPLETED o FAILED")
    document_type: str
    filename: str
    error_message: str | None = None
    result: dict | None = Field(default=None, description="Resultado final del análisis (LegalChatResponse)")
    processing_time_seconds: float | None = None

