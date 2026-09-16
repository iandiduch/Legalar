"""Schemas para comparación de versiones normativas y reformas en leyes argentinas."""

from typing import Literal
from pydantic import BaseModel, Field


class ArticleDiff(BaseModel):
    """Diferencia textual y contextual de un artículo específico entre dos versiones."""
    law_identifier: str
    article_number: str
    date_a: str | None = None
    date_b: str | None = None
    sha_a: str | None = None
    sha_b: str | None = None
    text_a: str = Field(description="Redacción original o anterior")
    text_b: str = Field(description="Redacción reformada o posterior")
    unified_diff: str = Field(description="Diff unificado en formato standard")
    has_changes: bool
    summary_of_changes: str | None = Field(default=None, description="Resumen analítico de la modificación")


class DiffRequest(BaseModel):
    """Petición de comparación de una norma o artículo entre dos momentos históricos."""
    law_identifier: str = Field(description="Identificador de la norma (ej: 'LEY-26994', 'LEY-19550')")
    date_a: str | None = Field(default=None, description="Fecha previa en formato YYYY-MM-DD")
    date_b: str | None = Field(default=None, description="Fecha posterior en formato YYYY-MM-DD")
    sha_a: str | None = Field(default=None, description="Commit SHA previo opcional")
    sha_b: str | None = Field(default=None, description="Commit SHA posterior opcional")
    article_number: str | None = Field(default=None, description="Número de artículo específico a comparar")


class DiffResponse(BaseModel):
    """Respuesta con las modificaciones detectadas y trazabilidad del origen."""
    law_identifier: str
    law_title: str
    article_number: str | None = None
    diff_source: Literal["api", "git_local"] = Field(description="Indica si se resolvió por API o por Fallback Local Git")
    diff_text: str = Field(description="Diff unificado")
    analysis: str | None = Field(default=None, description="Explicación jurídica de la reforma")
    article_diffs: list[ArticleDiff] = Field(default_factory=list)
