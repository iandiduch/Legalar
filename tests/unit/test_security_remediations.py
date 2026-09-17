"""Tests unitarios para las remediaciones de seguridad (SSRF, Path Traversal, IP Spoofing)."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from app.core.security import _client_ip
from app.services.legal.local_diff_engine import LocalGitDiffEngine
from app.services.legal.web_reader import is_safe_public_url


def test_client_ip_does_not_trust_unverified_xff_header():
    """Verifica que _client_ip no lea directamente X-Forwarded-For arbitrario enviado por el atacante."""
    scope = {
        "type": "http",
        "client": ("192.168.1.50", 12345),
        "headers": [(b"x-forwarded-for", b"203.0.113.199, 10.0.0.1")],
    }
    request = Request(scope)
    assert _client_ip(request) == "192.168.1.50"


def test_client_ip_handles_missing_client():
    """Verifica manejo seguro cuando request.client es None."""
    scope = {
        "type": "http",
        "client": None,
        "headers": [],
    }
    request = Request(scope)
    assert _client_ip(request) == "unknown"


def test_local_diff_engine_clean_id_path_traversal():
    """Verifica que _clean_id neutralice secuencias de escape o caracteres peligrosos."""
    assert LocalGitDiffEngine._clean_id("../../etc/passwd") == "etc-passwd"
    assert LocalGitDiffEngine._clean_id("..\\..\\windows\\system32") == "windows-system32"
    assert LocalGitDiffEngine._clean_id("LEY-26994") == "LEY-26994"
    assert LocalGitDiffEngine._clean_id("DNU/70/2023") == "DNU-70-2023"

    with pytest.raises(ValueError, match="Identificador de norma inválido"):
        LocalGitDiffEngine._clean_id("///")


def test_local_diff_engine_resolve_paths_bounds_checking():
    """Verifica que _resolve_paths valide que la ruta se mantenga dentro del repositorio."""
    engine = LocalGitDiffEngine(repo_path="repo_legalize_ar")
    rel_path, disk_path = engine._resolve_paths("LEY-26994")
    assert rel_path == "ar/LEY-26994.md"
    assert "repo_legalize_ar" in disk_path
    assert os.path.isabs(disk_path)


def test_is_safe_public_url_blocks_private_and_loopback():
    """Verifica que URLs que apuntan a localhost, redes privadas o metadatos sean rechazadas."""
    is_safe, err = is_safe_public_url("http://127.0.0.1:8000/api")
    assert is_safe is False
    assert "local" in err.lower() or "privada" in err.lower()

    is_safe, err = is_safe_public_url("http://localhost:5432")
    assert is_safe is False

    is_safe, err = is_safe_public_url("http://169.254.169.254/latest/meta-data/")
    assert is_safe is False
    assert "privada" in err.lower() or "restringida" in err.lower()

    is_safe, err = is_safe_public_url("ftp://example.com/file")
    assert is_safe is False
    assert "esquema" in err.lower()
