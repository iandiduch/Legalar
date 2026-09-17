"""Parser y transformador de unified diffs para el visualizador visual de reformas (MessageDiffBlock).

Convierte diffs de Git o de la API de Legalize en estructuras JSON enriquecidas con
tipado de línea ('header', 'delete', 'add', 'context'), numeración de líneas anterior/posterior,
y conteo de modificaciones/supresiones para renderizado estilo GitHub en tiempo real (< 10ms).
"""

import re
from typing import Any

from app.schemas.legal.diff import DiffResponse


def parse_unified_diff_to_lines(unified_diff: str) -> tuple[list[dict[str, Any]], int, int]:
    """Parsea un texto en formato unified diff en una lista de objetos de línea estructurados."""
    if not unified_diff or not unified_diff.strip():
        return [], 0, 0

    lines = unified_diff.splitlines()
    diff_lines: list[dict[str, Any]] = []
    modifications_count = 0
    deletions_count = 0

    old_line_no = 0
    new_line_no = 0

    for line in lines:
        if line.startswith("---") or line.startswith("+++"):
            continue

        if line.startswith("@@"):
            m = re.search(r"@@\s*-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s*@@", line)
            if m:
                old_line_no = int(m.group(1))
                new_line_no = int(m.group(2))
            diff_lines.append({
                "type": "header",
                "text": line,
                "lineOld": None,
                "lineNew": None,
            })
        elif line.startswith("-"):
            deletions_count += 1
            diff_lines.append({
                "type": "delete",
                "text": line,
                "lineOld": old_line_no if old_line_no > 0 else None,
                "lineNew": None,
            })
            if old_line_no > 0:
                old_line_no += 1
        elif line.startswith("+"):
            modifications_count += 1
            diff_lines.append({
                "type": "add",
                "text": line,
                "lineOld": None,
                "lineNew": new_line_no if new_line_no > 0 else None,
            })
            if new_line_no > 0:
                new_line_no += 1
        else:
            clean_text = line[1:] if line.startswith(" ") else line
            diff_lines.append({
                "type": "context",
                "text": clean_text,
                "lineOld": old_line_no if old_line_no > 0 else None,
                "lineNew": new_line_no if new_line_no > 0 else None,
            })
            if old_line_no > 0:
                old_line_no += 1
            if new_line_no > 0:
                new_line_no += 1

    return diff_lines, modifications_count, deletions_count


def build_diff_block_data(
    diff_response: DiffResponse,
    citizen_explanation: str | None = None,
    article_epigraph: str | None = None,
    reform_name: str | None = None,
    reform_date: str | None = None,
) -> dict[str, Any]:
    """Construye el payload exacto requerido por MessageDiffBlock.jsx."""
    law_id = diff_response.law_identifier
    art_num = diff_response.article_number or "General"
    law_title = diff_response.law_title or law_id

    has_changes = False
    unified_diff_text = diff_response.diff_text or ""
    other_articles: list[str] = []

    if diff_response.article_diffs:
        art_diff = diff_response.article_diffs[0]
        has_changes = art_diff.has_changes
        unified_diff_text = art_diff.unified_diff
        other_articles = [
            d.article_number
            for d in diff_response.article_diffs
            if d.article_number and d.article_number != art_num
        ]
    elif unified_diff_text and "sin modificaciones" not in unified_diff_text.lower() and "sin diferencias" not in unified_diff_text.lower():
        has_changes = True

    diff_lines, adds, dels = parse_unified_diff_to_lines(unified_diff_text)

    if not has_changes or not diff_lines or (adds == 0 and dels == 0):
        has_changes = False
        diff_lines = []
        adds = 0
        dels = 0
        citizen_explanation = None

    friendly_source = "Registro Oficial de Reformas Normativas" if diff_response.diff_source == "git_local" else "Legalize API Oficial"

    return {
        "id": f"diff-{law_id}-{art_num}",
        "lawIdentifier": law_id,
        "lawTitle": law_title,
        "articleNumber": art_num,
        "articleEpigraph": article_epigraph or f"Artículo {art_num}",
        "reformName": reform_name or friendly_source,
        "reformDate": reform_date or "Texto consolidado oficial",
        "hasChanges": has_changes,
        "summary": {
            "modificationsCount": adds,
            "deletionsCount": dels,
            "substitutionsCount": min(adds, dels),
            "citizenExplanation": citizen_explanation if has_changes else None,
        },
        "diffLines": diff_lines,
        "otherModifiedArticles": other_articles,
    }
