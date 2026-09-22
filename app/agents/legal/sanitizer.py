"""Utilidades de saneamiento y depuración de respuestas legales para evitar filtraciones de metadatos internos."""

import re


def strip_leaked_metadata(text: str) -> str:
    """Elimina metadatos estructurados o campos de debug que el LLM haya filtrado en el cuerpo del texto.

    Ejemplos de fuga a remover:
    - Confianza: high / medium / low
    - Confidence: high
    - articles_referenced: ['LEY-24240:23', ...]
    - unsupported_notes: [...]
    - unsupported_claims: [...]
    - warning_notes: [...]
    """
    if not text:
        return ""

    cleaned = text

    # 1. Eliminar listas estructuradas con formato python/json: articles_referenced: [...], etc.
    cleaned = re.sub(
        r"(?im)^\s*(?:articles_referenced|unsupported_notes|unsupported_claims|warning_notes)\s*:\s*\[.*?\]\s*$",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\n+\s*(?:articles_referenced|unsupported_notes|unsupported_claims|warning_notes)\s*:\s*\[.*?\]",
        "",
        cleaned,
    )

    # 2. Eliminar líneas con Confianza / Confidence (español o inglés)
    cleaned = re.sub(
        r"(?im)^\s*(?:confianza|confidence)\s*:\s*(?:high|medium|low|alta|media|baja)\b.*$",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\n+\s*(?:confianza|confidence)\s*:\s*(?:high|medium|low|alta|media|baja)\b.*",
        "",
        cleaned,
    )

    # 3. Eliminar posibles flags booleanos de auditoría
    cleaned = re.sub(
        r"(?im)^\s*(?:is_valid|all_norms_in_force)\s*:\s*(?:true|false)\b.*$",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\n+\s*(?:is_valid|all_norms_in_force)\s*:\s*(?:true|false)\b.*",
        "",
        cleaned,
    )

    return cleaned.strip()
