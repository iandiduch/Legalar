"""Parser especializado para legislación argentina en Markdown versionado (legalize-ar / InfoLEG).

Maneja sanitización de caracteres de control C1 (Windows-1252 to UTF-8 como U+0097),
extracción estructurada de jerarquías (Libro, Título, Capítulo, Sección),
reconocimiento de artículos multiformato (#, **, plain), epígrafes, incisos,
anotaciones editoriales de InfoLEG y generación de hashes sha256 para embeddings selectivos.
"""

import hashlib
import logging
import re
from datetime import date, datetime
from typing import Any

import yaml

from app.schemas.legal.parser import ParsedArticle, ParsedHierarchy, ParsedLaw

logger = logging.getLogger(__name__)

# Mapa de sustitución para caracteres de control C1 frecuentes de InfoLEG
_C1_REPLACEMENTS = {
    "\x97": "—",  # em-dash
    "\x96": "–",  # en-dash
    "\x91": "'",  # left single quote
    "\x92": "'",  # right single quote
    "\x93": '"',  # left double quote
    "\x94": '"',  # right double quote
    "\x85": "\n", # newline
    "\xa0": " ",  # non-breaking space
}

# Expresiones regulares para artículos
_RE_ART_HEADING = re.compile(
    r"^(?P<hashes>#{1,6})\s*(?:art[íi]culo|art\.?)\s*(?P<num>\d+(?:\s*(?:bis|ter|quater|quinquies|[a-z]))?[°º\.]*?)(?:[—–\-\.:\s]|\Z)(?P<rest>.*)$",
    re.IGNORECASE,
)
_RE_ART_BOLD = re.compile(
    r"^(?P<stars>\*{2,3})\s*(?:art[íi]culo|art\.?)\s*(?P<num>\d+(?:\s*(?:bis|ter|quater|quinquies|[a-z]))?[°º\.]*)[^\w\n]*?\*+(?:[—–\-\.:\s]+|\s*|\Z)(?P<rest>.*)$",
    re.IGNORECASE,
)
_RE_ART_PLAIN = re.compile(
    r"^(?:art[íi]culo|art\.?)\s*(?P<num>\d+(?:\s*(?:bis|ter|quater|quinquies|[a-z]))?[°º\.]*?)(?:[—–\-\.:])(?P<rest>.*)$",
    re.IGNORECASE,
)

# Expresiones regulares para jerarquías
_RE_LIBRO = re.compile(r"^#{1,6}\s*(?:libro|lib\.)\s+([^\n]+)", re.IGNORECASE)
_RE_PARTE = re.compile(r"^#{1,6}\s*(?:parte)\s+([^\n]+)", re.IGNORECASE)
_RE_TITULO = re.compile(r"^#{1,6}\s*(?:t[íi]tulo|t[íi]t\.)\s+([^\n]+)", re.IGNORECASE)
_RE_CAPITULO = re.compile(r"^#{1,6}\s*(?:cap[íi]tulo|cap\.)\s+([^\n]+)", re.IGNORECASE)
_RE_SECCION = re.compile(r"^#{1,6}\s*(?:secci[óo]n|sec\.)\s+([^\n]+)", re.IGNORECASE)
_RE_ANEXO = re.compile(r"^#{1,6}\s*(?:anexo)\s+([^\n]+)", re.IGNORECASE)

# Incisos
_RE_INCISO = re.compile(r"^\s*([a-z]|\d+)\)\s*(.+)$", re.IGNORECASE)

# Notas editoriales de InfoLEG
_RE_EDITORIAL_NOTE = re.compile(r"\*+\((?:art[íi]culo|nota|texto|denominaci[óo]n)[^)]*\)\*+", re.IGNORECASE)


def sanitize_text(text: str) -> str:
    """Reemplaza caracteres de control C1 y artefactos de codificación Windows-1252."""
    for char, replacement in _C1_REPLACEMENTS.items():
        if char in text:
            text = text.replace(char, replacement)
    # Eliminar cualquier otro código de control C1 (0x80 a 0x9F) que no sea espacio válido
    text = re.sub(r"[\x80-\x9f]", " ", text)
    return text


def clean_article_number(raw_num: str) -> str | None:
    """Normaliza el número de artículo (elimina ordinales, puntos finales, espacios redundantes)."""
    cleaned = raw_num.replace("º", "").replace("°", "").replace(".", "").strip()
    if not cleaned or not cleaned[0].isdigit():
        return None
    # No admitir frases largas como números de artículo
    if len(cleaned.split()) > 3:
        return None
    return cleaned.lower()


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Extrae y sanitiza el YAML frontmatter de la norma."""
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    raw_yaml = parts[1]
    body = parts[2]

    sanitized_yaml = sanitize_text(raw_yaml)
    try:
        data = yaml.safe_load(sanitized_yaml)
        return (data if isinstance(data, dict) else {}), body
    except Exception as exc:
        logger.warning("Fallo al parsear YAML frontmatter: %s", exc)
        return {}, body


def parse_date(date_val: Any) -> date | None:
    """Convierte strings o fechas a date estándar, ignorando fechas nulas o placeholders."""
    if not date_val:
        return None
    if isinstance(date_val, (date, datetime)):
        return date_val if isinstance(date_val, date) else date_val.date()
    str_val = str(date_val).strip()
    if str_val in ("1900-01-01", "", "None", "null"):
        return None
    try:
        return datetime.strptime(str_val, "%Y-%m-%d").date()
    except ValueError:
        return None


def extract_epigraph_and_body(first_line_rest: str, subsequent_lines: list[str]) -> tuple[str | None, str]:
    """Separa el epígrafe (título del artículo) del texto dispositivo."""
    first_line_rest = first_line_rest.strip()
    epigraph = None
    body_parts = []

    # Caso frecuente CCC: "ARTICULO 2°.- Interpretación. La ley debe ser interpretada..."
    dot_split = first_line_rest.split(".", 1)
    if len(dot_split) == 2 and len(dot_split[0].strip()) < 80 and not dot_split[0].strip().startswith("—"):
        candidate = dot_split[0].strip(" -—–:")
        if candidate and not candidate.isdigit() and len(candidate.split()) < 10:
            epigraph = candidate
            remaining_text = dot_split[1].strip(" -—–:")
            if remaining_text:
                body_parts.append(remaining_text)
        else:
            if first_line_rest:
                body_parts.append(first_line_rest.strip(" -—–:"))
    else:
        if first_line_rest:
            body_parts.append(first_line_rest.strip(" -—–:"))

    body_parts.extend(subsequent_lines)
    full_body = "\n".join(body_parts).strip()
    return epigraph, full_body


def build_contextualized_text(
    law_identifier: str,
    law_title: str,
    hierarchy: ParsedHierarchy,
    article_number: str,
    epigraph: str | None,
    content: str,
) -> str:
    """Genera el texto enriquecido para embeddings óptimos en Pinecone."""
    crumbs = [f"[{law_identifier}: {law_title}]"]
    if hierarchy.book:
        crumbs.append(f"[{hierarchy.book}]")
    if hierarchy.title_section:
        crumbs.append(f"[{hierarchy.title_section}]")
    if hierarchy.chapter:
        crumbs.append(f"[{hierarchy.chapter}]")
    if hierarchy.section:
        crumbs.append(f"[{hierarchy.section}]")
    if hierarchy.annex:
        crumbs.append(f"[{hierarchy.annex}]")

    epigraph_str = f" ({epigraph})" if epigraph else ""
    header = " ".join(crumbs)
    return f"{header}\nArtículo {article_number}{epigraph_str}:\n{content}"


def parse_markdown_law(content: str, fallback_identifier: str = "NORMA") -> ParsedLaw:
    """Parsea completamente un archivo Markdown de legislación argentina."""
    metadata, body = parse_frontmatter(content)
    identifier = str(metadata.get("identifier") or fallback_identifier).strip()
    title = str(metadata.get("title") or identifier).strip()
    rank = str(metadata.get("rank") or "ley").strip().lower()
    status = str(metadata.get("status") or "in_force").strip().lower()

    lines = body.splitlines()

    current_hierarchy = ParsedHierarchy()
    articles: list[ParsedArticle] = []

    current_art_num: str | None = None
    current_first_rest: str = ""
    current_art_lines: list[str] = []
    current_epigraph_candidate: str | None = None

    def flush_current_article() -> None:
        nonlocal current_art_num, current_first_rest, current_art_lines, current_epigraph_candidate
        if not current_art_num:
            return

        extracted_epigraph, full_content = extract_epigraph_and_body(current_first_rest, current_art_lines)
        epigraph = extracted_epigraph or current_epigraph_candidate

        # Extraer notas editoriales
        notes = _RE_EDITORIAL_NOTE.findall(full_content)
        cleaned_content = _RE_EDITORIAL_NOTE.sub("", full_content).strip()

        # Extraer incisos
        incisos = []
        for line in cleaned_content.splitlines():
            inc_match = _RE_INCISO.match(line)
            if inc_match:
                incisos.append({"id": inc_match.group(1).lower(), "text": inc_match.group(2).strip()})

        content_hash = hashlib.sha256(cleaned_content.encode("utf-8")).hexdigest()

        contextualized = build_contextualized_text(
            law_identifier=identifier,
            law_title=title,
            hierarchy=current_hierarchy,
            article_number=current_art_num,
            epigraph=epigraph,
            content=cleaned_content,
        )

        articles.append(
            ParsedArticle(
                law_identifier=identifier,
                article_number=current_art_num,
                article_order=len(articles) + 1,
                epigraph=epigraph,
                content_raw=cleaned_content,
                incisos=incisos,
                editorial_notes=[n.strip("*() ") for n in notes],
                hierarchy=current_hierarchy.model_copy(),
                content_hash=content_hash,
                contextualized_text=contextualized,
            )
        )

        current_art_num = None
        current_first_rest = ""
        current_art_lines = []
        current_epigraph_candidate = None

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            if current_art_num:
                current_art_lines.append("")
            continue

        # Detección de jerarquías
        m_lib = _RE_LIBRO.match(line_stripped)
        if m_lib:
            flush_current_article()
            current_hierarchy.book = m_lib.group(1).strip()
            current_hierarchy.chapter = None
            current_hierarchy.section = None
            continue

        m_tit = _RE_TITULO.match(line_stripped)
        if m_tit:
            flush_current_article()
            current_hierarchy.title_section = m_tit.group(1).strip()
            current_hierarchy.chapter = None
            current_hierarchy.section = None
            continue

        m_cap = _RE_CAPITULO.match(line_stripped)
        if m_cap:
            flush_current_article()
            current_hierarchy.chapter = m_cap.group(1).strip()
            current_hierarchy.section = None
            continue

        m_sec = _RE_SECCION.match(line_stripped)
        if m_sec:
            flush_current_article()
            current_hierarchy.section = m_sec.group(1).strip()
            continue

        m_anx = _RE_ANEXO.match(line_stripped)
        if m_anx:
            flush_current_article()
            current_hierarchy.annex = m_anx.group(1).strip()
            continue

        # Chequear si la línea es un epígrafe solitario previo al artículo (ej: "Fuentes de regulación\n###### Art. 1º")
        if not current_art_num and not line_stripped.startswith("#") and len(line_stripped) < 70:
            if i + 1 < len(lines):
                next_l = lines[i + 1].strip()
                if _RE_ART_HEADING.match(next_l) or _RE_ART_BOLD.match(next_l):
                    current_epigraph_candidate = line_stripped
                    continue

        # Detección de inicio de artículo
        m_art_h = _RE_ART_HEADING.match(line_stripped)
        m_art_b = _RE_ART_BOLD.match(line_stripped)
        m_art_p = _RE_ART_PLAIN.match(line_stripped)

        match = m_art_h or m_art_b or m_art_p
        if match:
            raw_num = match.group("num")
            rest_str = match.group("rest").strip()
            # Si el resto dice "de la Ley X" o "del Código Y", es una cita doctrinal/jurisprudencial en los considerandos
            if rest_str.lower().startswith(("de la ", "del ", "de los ", "de las ", "de ", "que ")):
                if current_art_num:
                    current_art_lines.append(line_stripped)
                continue

            cleaned_num = clean_article_number(raw_num)
            if cleaned_num:
                flush_current_article()
                current_art_num = cleaned_num
                current_first_rest = rest_str
                current_art_lines = []
                continue

        # Si estamos dentro de un artículo, acumular líneas
        if current_art_num:
            current_art_lines.append(line_stripped)

    # Flush final
    flush_current_article()

    return ParsedLaw(
        identifier=identifier,
        title=title,
        country=str(metadata.get("country") or "ar").strip().lower(),
        rank=rank,
        status=status,
        publication_date=parse_date(metadata.get("publication_date")),
        last_updated=parse_date(metadata.get("last_updated")),
        enactment_date=parse_date(metadata.get("enactment_date")),
        department=metadata.get("department"),
        source=metadata.get("source"),
        infoleg_id=str(metadata.get("infoleg_id")) if metadata.get("infoleg_id") else None,
        reform_quality=metadata.get("reform_quality"),
        summary=metadata.get("summary") or metadata.get("texto_resumido"),
        times_modified=int(metadata.get("times_modified") or 0),
        modifies_count=int(metadata.get("modifies_count") or 0),
        extra_metadata=metadata,
        articles=articles,
    )
