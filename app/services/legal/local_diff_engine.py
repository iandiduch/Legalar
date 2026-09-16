"""Motor local de diffs normativos basado en Git y AST de artículos.

Proporciona fallback 100% autónomo y sin límite de llamadas a la Legalize API.
Permite comparar redacciones de cualquier ley o artículo específico entre dos fechas
o commits históricos directamente sobre el repositorio local 'legalize-ar'.
"""

import difflib
import logging
import os
import subprocess
from datetime import datetime
from typing import Any

from app.schemas.legal.diff import ArticleDiff, DiffResponse
from app.services.legal.markdown_parser import parse_markdown_law

logger = logging.getLogger(__name__)


class LocalGitDiffEngine:
    """Motor de diffs locales ejecutando Git nativo sobre el repositorio de leyes."""

    def __init__(self, repo_path: str = "repo_legalize_ar") -> None:
        self.repo_path = os.path.abspath(repo_path)
        if not os.path.exists(os.path.join(self.repo_path, ".git")):
            logger.warning("El directorio %s no contiene un repositorio Git válido.", self.repo_path)

    def _run_git(self, args: list[str]) -> str:
        """Ejecuta un comando git en el repositorio local y retorna stdout."""
        try:
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
        except subprocess.CalledProcessError as exc:
            logger.error("Error ejecutando git %s: %s", " ".join(args), exc.stderr)
            raise RuntimeError(f"Fallo de comando Git: {exc.stderr.strip()}") from exc

    @staticmethod
    def _clean_id(law_identifier: str) -> str:
        return law_identifier.replace("/", "-").strip()

    def resolve_commit_for_date(self, law_identifier: str, target_date: str) -> str | None:
        """Encuentra el commit más reciente de una ley en o antes de target_date (YYYY-MM-DD)."""
        clean_id = self._clean_id(law_identifier)
        file_path = f"ar/{clean_id}.md"
        # --until acepta formato de fecha ISO
        output = self._run_git(
            ["log", "-n", "1", f"--until={target_date} 23:59:59", "--format=%H", "--", file_path]
        )
        return output if output else None

    def get_file_content_at_commit(self, law_identifier: str, commit_sha: str | None = None) -> str:
        """Obtiene el texto íntegro del Markdown de una norma en un commit dado (o HEAD si None)."""
        clean_id = self._clean_id(law_identifier)
        file_path = f"ar/{clean_id}.md"
        if not commit_sha or commit_sha.upper() == "HEAD":
            disk_path = os.path.join(self.repo_path, file_path)
            if not os.path.exists(disk_path):
                raise FileNotFoundError(f"La norma {clean_id} no existe en el repositorio.")
            with open(disk_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

        return self._run_git(["show", f"{commit_sha}:{file_path}"])


    def compute_diff(
        self,
        law_identifier: str,
        date_a: str | None = None,
        date_b: str | None = None,
        sha_a: str | None = None,
        sha_b: str | None = None,
        article_number: str | None = None,
    ) -> DiffResponse:
        """Calcula las diferencias históricas a nivel de norma o artículo específico."""
        # 1. Resolver SHAs si se pasaron fechas
        commit_a = sha_a
        if not commit_a and date_a:
            commit_a = self.resolve_commit_for_date(law_identifier, date_a)
            if not commit_a:
                # Si no hay commit previo a date_a, tomar el primer commit del archivo
                first_commit = self._run_git(["log", "--reverse", "-n", "1", "--format=%H", "--", f"ar/{law_identifier}.md"])
                commit_a = first_commit

        commit_b = sha_b
        if not commit_b and date_b:
            commit_b = self.resolve_commit_for_date(law_identifier, date_b)

        # Si no se pasó commit_a ni date_a, comparar contra la versión inmediatamente previa (HEAD~1)
        if not commit_a:
            try:
                clean_id = self._clean_id(law_identifier)
                log_shas = self._run_git(["log", "-n", "2", "--format=%H", "--", f"ar/{clean_id}.md"]).splitlines()
                if len(log_shas) >= 2:
                    commit_a = log_shas[1]  # Versión anterior inmediata
                elif len(log_shas) == 1:
                    commit_a = log_shas[0]
            except Exception as e:
                logger.warning("No se pudo resolver commit previo automático para %s: %s", law_identifier, e)

        # 2. Obtener contenidos en ambas versiones
        try:
            content_a = self.get_file_content_at_commit(law_identifier, commit_a) if commit_a else ""
        except Exception as exc:
            logger.warning("No se pudo obtener versión A (%s) de %s: %s", commit_a, law_identifier, exc)
            content_a = ""

        try:
            content_b = self.get_file_content_at_commit(law_identifier, commit_b)
        except Exception as exc:
            logger.warning("No se pudo obtener versión B (%s) de %s: %s", commit_b, law_identifier, exc)
            content_b = ""

        law_a = parse_markdown_law(content_a, law_identifier) if content_a else None
        law_b = parse_markdown_law(content_b, law_identifier) if content_b else None

        law_title = (law_b.title if law_b else (law_a.title if law_a else law_identifier))

        # 3. Si se pidió un artículo específico
        if article_number:
            clean_art = article_number.replace("º", "").replace("°", "").replace(".", "").strip().lower()
            art_a = next((a for a in law_a.articles if a.article_number == clean_art), None) if law_a else None
            art_b = next((a for a in law_b.articles if a.article_number == clean_art), None) if law_b else None

            text_a = art_a.content_raw if art_a else "(Artículo no existía en esta versión)"
            text_b = art_b.content_raw if art_b else "(Artículo derogado o no presente en esta versión)"

            lines_a = text_a.splitlines(keepends=True)
            lines_b = text_b.splitlines(keepends=True)

            diff_lines = list(
                difflib.unified_diff(
                    lines_a,
                    lines_b,
                    fromfile=f"{law_identifier} Art. {clean_art} ({date_a or commit_a or 'Inicial'})",
                    tofile=f"{law_identifier} Art. {clean_art} ({date_b or commit_b or 'Vigente'})",
                )
            )
            unified_str = "".join(diff_lines)
            has_changes = bool(unified_str.strip())

            art_diff = ArticleDiff(
                law_identifier=law_identifier,
                article_number=clean_art,
                date_a=date_a,
                date_b=date_b,
                sha_a=commit_a,
                sha_b=commit_b,
                text_a=text_a,
                text_b=text_b,
                unified_diff=unified_str if has_changes else "Sin modificaciones en la redacción del artículo.",
                has_changes=has_changes,
                summary_of_changes="Modificación sustancial de redacción detectada." if has_changes else "Texto idéntico.",
            )

            return DiffResponse(
                law_identifier=law_identifier,
                law_title=law_title,
                article_number=clean_art,
                diff_source="git_local",
                diff_text=unified_str or "Sin modificaciones.",
                analysis=f"Comparativa del artículo {clean_art} de la norma {law_identifier}.",
                article_diffs=[art_diff],
            )

        # 4. Comparativa global de toda la norma
        lines_a = content_a.splitlines(keepends=True)
        lines_b = content_b.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                lines_a,
                lines_b,
                fromfile=f"{law_identifier} ({date_a or commit_a or 'Inicial'})",
                tofile=f"{law_identifier} ({date_b or commit_b or 'Vigente'})",
            )
        )
        unified_str = "".join(diff_lines)

        # Identificar artículos con cambios comparando hashes
        articles_changed: list[ArticleDiff] = []
        if law_a and law_b:
            articles_map_a = {a.article_number: a for a in law_a.articles}
            articles_map_b = {b.article_number: b for b in law_b.articles}
            all_numbers = sorted(set(articles_map_a.keys()) | set(articles_map_b.keys()))

            for num in all_numbers:
                a_art = articles_map_a.get(num)
                b_art = articles_map_b.get(num)
                if not a_art or not b_art or a_art.content_hash != b_art.content_hash:
                    t_a = a_art.content_raw if a_art else "(No existía)"
                    t_b = b_art.content_raw if b_art else "(Derogado)"
                    u_diff = "".join(
                        difflib.unified_diff(
                            t_a.splitlines(keepends=True),
                            t_b.splitlines(keepends=True),
                            fromfile=f"Art {num} (v1)",
                            tofile=f"Art {num} (v2)",
                        )
                    )
                    articles_changed.append(
                        ArticleDiff(
                            law_identifier=law_identifier,
                            article_number=num,
                            date_a=date_a,
                            date_b=date_b,
                            sha_a=commit_a,
                            sha_b=commit_b,
                            text_a=t_a,
                            text_b=t_b,
                            unified_diff=u_diff,
                            has_changes=True,
                            summary_of_changes="Artículo modificado entre versiones.",
                        )
                    )

        return DiffResponse(
            law_identifier=law_identifier,
            law_title=law_title,
            article_number=None,
            diff_source="git_local",
            diff_text=unified_str if unified_str else "Sin diferencias globales detectadas.",
            analysis=f"Se identificaron {len(articles_changed)} artículos con reformas entre ambas versiones.",
            article_diffs=articles_changed[:50],  # Límite de seguridad
        )
