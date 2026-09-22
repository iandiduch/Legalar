"""Tests unitarios para el saneador de respuestas strip_leaked_metadata."""

import pytest
from app.agents.legal.sanitizer import strip_leaked_metadata


def test_strip_leaked_metadata_exact_user_leak():
    """Prueba el caso exacto reportado por el usuario con Confianza y articles_referenced al final."""
    raw = (
        "Es importante destacar que la Ley de Defensa del Consumidor es de orden público y sus disposiciones son irrenunciables.\n\n"
        "Confianza: high\n"
        "articles_referenced:['LEY-24240:23', 'LEY-24240:17', 'LEY-24240:11', 'LEY-24240:16', 'LEY-24240:40', 'LEY-24240:13']"
    )

    cleaned = strip_leaked_metadata(raw)
    assert "Confianza:" not in cleaned
    assert "articles_referenced:" not in cleaned
    assert "LEY-24240:23" not in cleaned
    assert cleaned.endswith("sus disposiciones son irrenunciables.")


def test_strip_leaked_metadata_multiple_internal_fields():
    """Valida la limpieza de múltiples campos técnicos (confidence, unsupported_notes, warning_notes)."""
    raw = (
        "Dictamen legal sobre locaciones conforme al DNU 70/2023.\n\n"
        "Confidence: medium\n"
        "unsupported_notes: ['aspecto no contemplado']\n"
        "warning_notes: ['nota interna']\n"
        "is_valid: true\n"
    )

    cleaned = strip_leaked_metadata(raw)
    assert cleaned == "Dictamen legal sobre locaciones conforme al DNU 70/2023."


def test_strip_leaked_metadata_preserves_clean_legal_text():
    """Valida que textos legales normales con palabras como 'confianza' en contexto no se mutilen."""
    clean_legal = "En el contrato de depósito o fiducia, la confianza entre las partes es un elemento esencial."
    assert strip_leaked_metadata(clean_legal) == clean_legal


def test_strip_leaked_metadata_empty_or_none():
    assert strip_leaked_metadata("") == ""
    assert strip_leaked_metadata(None) == ""
