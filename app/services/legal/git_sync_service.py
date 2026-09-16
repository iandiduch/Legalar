"""Servicio de detección de cambios y sincronización incremental vía Git.

Extrae normas añadidas, modificadas o derogadas mediante `git diff` y analiza los trailers
estructurados de Legalize (`Source-Id`, `Source-Date`, `Norm-Id`) para auditar reformas.
"""

import logging
import os
import re
import subprocess
from datetime import datetime
from typing import Any

from app.db.models_orm import LegalRevision

logger = logging.getLogger(__name__)

# Expresión regular para trailers estructurados de Legalize
_RE_TRAILER_SOURCE_ID = re.compile(r"^Source-Id:\s*(\S+)", re.MULTILINE | re.IGNORECASE)
_RE_TRAILER_SOURCE_DATE = re.compile(r"^Source-Date:\s*(\S+)", re.MULTILINE | re.IGNORECASE)
_RE_TRAILER_NORM_ID = re.compile(r"^Norm-Id:\s*(\S+)", re.MULTILINE | re.IGNORECASE)
_RE_COMMIT_TYPE = re.compile(r"^\[(reform|bootstrap|metadata|correction)\]", re.IGNORECASE)


class GitSyncService:
    """Gestor de sincronización y detección de reformas en el repositorio legalize-ar."""

    def __init__(self, repo_path: str = "repo_legalize_ar") -> None:
        self.repo_path = os.path.abspath(repo_path)

    def _run_git(self, args: list[str]) -> str:
        res = subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return res.stdout.strip()

    def get_current_head(self) -> str:
        """Obtiene el SHA del commit actual de HEAD."""
        return self._run_git(["rev-parse", "HEAD"])

    def fetch_upstream(self) -> bool:
        """Ejecuta git fetch origin main para actualizar referencias sin alterar el árbol local."""
        try:
            self._run_git(["fetch", "origin", "main"])
            return True
        except Exception as exc:
            logger.warning("Fallo al ejecutar git fetch upstream (puede estar offline): %s", exc)
            return False

    def detect_changes(self, last_indexed_commit: str | None = None, target_ref: str = "HEAD") -> dict[str, list[str]]:
        """Compara last_indexed_commit contra target_ref para obtener archivos agregados, modificados y eliminados."""
        if not last_indexed_commit:
            # Si no hay commit previo, todos los archivos en ar/ se consideran agregados
            ar_dir = os.path.join(self.repo_path, "ar")
            if not os.path.exists(ar_dir):
                return {"added": [], "modified": [], "deleted": []}
            all_files = [f"ar/{f}" for f in os.listdir(ar_dir) if f.endswith(".md")]
            return {"added": all_files, "modified": [], "deleted": []}

        diff_output = self._run_git(
            ["diff", "--name-status", last_indexed_commit, target_ref, "--", "ar/"]
        )

        added = []
        modified = []
        deleted = []

        for line in diff_output.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            status, path = parts[0], parts[1]
            if not path.endswith(".md"):
                continue

            if status.startswith("A"):
                added.append(path)
            elif status.startswith("M"):
                modified.append(path)
            elif status.startswith("D"):
                deleted.append(path)

        return {"added": added, "modified": modified, "deleted": deleted}

    def extract_revisions(self, since_commit: str | None = None, until_ref: str = "HEAD") -> list[dict[str, Any]]:
        """Extrae el historial de reformas con trailers estructurados de Legalize."""
        rev_range = f"{since_commit}..{until_ref}" if since_commit else until_ref
        try:
            # Delimitador \x1f para separar commits y \x09 para campos
            log_output = self._run_git(
                ["log", rev_range, "--format=%H%x09%cI%x09%B%x1f", "--", "ar/"]
            )
        except Exception as exc:
            logger.warning("Fallo al obtener git log de reformas: %s", exc)
            return []

        revisions: list[dict[str, Any]] = []
        raw_commits = log_output.split("\x1f")

        for entry in raw_commits:
            entry = entry.strip()
            if not entry:
                continue
            fields = entry.split("\x09", 2)
            if len(fields) < 3:
                continue

            sha, date_str, msg = fields[0].strip(), fields[1].strip(), fields[2].strip()

            source_id_m = _RE_TRAILER_SOURCE_ID.search(msg)
            source_date_m = _RE_TRAILER_SOURCE_DATE.search(msg)
            norm_id_m = _RE_TRAILER_NORM_ID.search(msg)
            type_m = _RE_COMMIT_TYPE.search(msg)

            change_type = type_m.group(1).lower() if type_m else "reform"
            source_id = source_id_m.group(1) if source_id_m else None
            source_date_str = source_date_m.group(1) if source_date_m else None
            norm_id = norm_id_m.group(1) if norm_id_m else None

            # Si no vino Norm-Id en el trailer, intentar deducirlo del título del commit
            if not norm_id:
                norm_guess = re.search(r"\b(LEY|DEC|DNU|DL)-\d+[\w-]*", msg)
                if norm_guess:
                    norm_id = norm_guess.group(0)

            try:
                commit_dt = datetime.fromisoformat(date_str)
            except ValueError:
                commit_dt = datetime.utcnow()

            source_dt = None
            if source_date_str:
                try:
                    source_dt = datetime.strptime(source_date_str, "%Y-%m-%d")
                except ValueError:
                    pass

            if norm_id:
                revisions.append(
                    {
                        "law_identifier": norm_id,
                        "commit_sha": sha,
                        "commit_date": commit_dt,
                        "change_type": change_type,
                        "source_id": source_id,
                        "source_date": source_dt,
                        "affected_articles": [],
                        "commit_message": msg.splitlines()[0] if msg else "",
                    }
                )

        return revisions
