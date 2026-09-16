"""Pruebas unitarias de los componentes del Agente Legal Argentino.

Valida:
1. Sanitización de caracteres C1 (U+0097) y parsing de leyes reales.
2. Motor de diff local sobre Git con comparación granular por artículo.
3. Cliente resiliente de Legalize API con fallback ante cuota agotada.
4. Generación determinista de hashes de artículos para ahorro de embeddings.
"""

import os
import pytest

from app.core.config import Settings
from app.schemas.legal.diff import DiffResponse
from app.services.legal.legalize_api_client import LegalizeApiClient
from app.services.legal.local_diff_engine import LocalGitDiffEngine
from app.services.legal.markdown_parser import parse_markdown_law, sanitize_text


@pytest.fixture
def repo_path() -> str:
    path = os.path.abspath("repo_legalize_ar")
    if not os.path.exists(path):
        pytest.skip("Repositorio legalize-ar no encontrado en el entorno local")
    return path


def test_sanitize_c1_characters():
    """Verifica que caracteres de control C1 como \x97 se reemplacen y no rompan PyYAML."""
    raw_with_c1 = "title: 'NORMA CON GUION \x97 LARGO'\nidentifier: 'TEST-001'\n"
    cleaned = sanitize_text(raw_with_c1)
    assert "\x97" not in cleaned
    assert "—" in cleaned


def test_parse_real_law_ccc(repo_path: str):
    """Verifica el parsing completo del Código Civil y Comercial (LEY-26994)."""
    file_path = os.path.join(repo_path, "ar", "LEY-26994.md")
    if not os.path.exists(file_path):
        pytest.skip("LEY-26994.md no disponible")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    law = parse_markdown_law(content, "LEY-26994")
    assert law.identifier == "LEY-26994"
    assert "CODIGO CIVIL Y COMERCIAL" in law.title.upper()
    assert len(law.articles) > 2000

    # Buscar el artículo 1198 (Plazo de la locación)
    art1198 = next((a for a in law.articles if a.article_number == "1198"), None)
    assert art1198 is not None
    assert "Plazo de la locación" in (art1198.epigraph or "")
    assert len(art1198.content_hash) == 64
    assert "[LEY-26994" in art1198.contextualized_text


def test_local_diff_engine_article_comparison(repo_path: str):
    """Verifica la comparación semántica y diff unificado de artículos con Git nativo."""
    engine = LocalGitDiffEngine(repo_path=repo_path)
    res: DiffResponse = engine.compute_diff(
        law_identifier="LEY-26994",
        article_number="1198",
    )
    assert res.law_identifier == "LEY-26994"
    assert res.article_number == "1198"
    assert res.diff_source == "git_local"
    assert len(res.article_diffs) == 1
    art_diff = res.article_diffs[0]
    assert art_diff.article_number == "1198"
    assert len(art_diff.text_b) > 20


@pytest.mark.asyncio
async def test_legalize_api_client_fallback(repo_path: str):
    """Verifica que si no hay API key o se agota la cuota, conmuta automáticamente a Git local."""
    settings = Settings(
        LEGALIZE_REPO_PATH=repo_path,
        LEGALIZE_API_KEY="",  # Sin API key para forzar fallback local
        LEGALIZE_API_MONTHLY_LIMIT=2000,
    )
    client = LegalizeApiClient(settings=settings)
    res = await client.get_diff(law_identifier="LEY-26994", article="1198")
    assert res.diff_source == "git_local"
    assert res.article_number == "1198"
