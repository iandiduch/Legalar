"""Schemas para el resultado del parsing estructurado de leyes y decretos argentinos."""

from datetime import date
from typing import Any
from pydantic import BaseModel, Field


class ParsedHierarchy(BaseModel):
    """Contexto jerárquico del ordenamiento normativo."""
    book: str | None = Field(default=None, description="Libro normativo (ej: 'Libro Tercero')")
    title_section: str | None = Field(default=None, description="Título normativo (ej: 'Título IV — Contratos')")
    chapter: str | None = Field(default=None, description="Capítulo normativo (ej: 'Capítulo 3 — Locación')")
    section: str | None = Field(default=None, description="Sección normativa (ej: 'Sección 1ª')")
    annex: str | None = Field(default=None, description="Anexo normativo (ej: 'Anexo I')")


class ParsedArticle(BaseModel):
    """Estructura detallada de un artículo jurídico extraído de Markdown."""
    law_identifier: str = Field(description="Identificador de la norma (ej: 'LEY-26994')")
    article_number: str = Field(description="Número de artículo normalizado (ej: '1198', '11 bis')")
    article_order: int = Field(description="Orden de aparición en el texto")
    epigraph: str | None = Field(default=None, description="Epígrafe o título del artículo")
    content_raw: str = Field(description="Cuerpo dispositivo completo del artículo")
    incisos: list[dict[str, str]] = Field(default_factory=list, description="Lista de incisos detectados")
    editorial_notes: list[str] = Field(default_factory=list, description="Notas de sustitución / derogación de InfoLEG")
    hierarchy: ParsedHierarchy = Field(default_factory=ParsedHierarchy)
    content_hash: str = Field(description="Hash SHA-256 del contenido dispositivo para evitar re-embedding")
    contextualized_text: str = Field(description="Texto con breadcrumbs jerárquicos listo para Embedding")


class ParsedLaw(BaseModel):
    """Metadatos y artículos de una norma extraída de Markdown + YAML Frontmatter."""
    identifier: str
    title: str
    country: str = "ar"
    rank: str
    status: str = "in_force"
    publication_date: date | None = None
    last_updated: date | None = None
    enactment_date: date | None = None
    department: str | None = None
    source: str | None = None
    infoleg_id: str | None = None
    reform_quality: str | None = None
    summary: str | None = None
    times_modified: int = 0
    modifies_count: int = 0
    extra_metadata: dict[str, Any] = Field(default_factory=dict)
    articles: list[ParsedArticle] = Field(default_factory=list)
