"""Schemas para citas normativas con trazabilidad legal y auditoría."""

from pydantic import BaseModel, Field


class LegalCitation(BaseModel):
    """Cita jurídica formal y verificable."""
    law_identifier: str = Field(description="Ej: 'LEY-26994', 'DNU-70-2023'")
    law_title: str = Field(description="Título oficial de la norma")
    article_number: str = Field(description="Ej: '1198', '14 bis'")
    epigraph: str | None = Field(default=None, description="Epígrafe del artículo")
    exact_quote: str | None = Field(default=None, description="Fragmento textual citado")
    status: str = Field(default="in_force", description="Estado de vigencia ('in_force', 'repealed')")
    infoleg_url: str | None = Field(default=None, description="Enlace a la fuente oficial InfoLEG / SAIJ")
    commit_sha: str | None = Field(default=None, description="Commit SHA en legalize-ar que respalda la versión")
    confidence_score: float = Field(default=1.0, description="Score de coincidencia en retrieval")
