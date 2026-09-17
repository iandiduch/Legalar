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

    def _run_git(self, args: list[str], timeout: float = 4.0) -> str:
        """Ejecuta un comando git en el repositorio local y retorna stdout con timeout estricto."""
        try:
            res = subprocess.run(
                ["git", *args],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
                timeout=timeout,
            )
            return res.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.warning("Timeout al ejecutar git %s en %s (%ss)", " ".join(args[:3]), self.repo_path, timeout)
            return ""
        except subprocess.CalledProcessError as exc:
            logger.error("Error ejecutando git %s: %s", " ".join(args[:3]), exc.stderr)
            return ""
        except Exception as exc:
            logger.error("Excepción al ejecutar git %s: %s", " ".join(args[:3]), exc)
            return ""

    @staticmethod
    def _clean_id(law_identifier: str) -> str:
        return law_identifier.replace("/", "-").strip()

    def resolve_commit_for_date(self, law_identifier: str, target_date: str) -> str | None:
        """Encuentra el commit más reciente de una ley en o antes de target_date (YYYY-MM-DD)."""
        clean_id = self._clean_id(law_identifier)
        file_path = f"ar/{clean_id}.md"
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
        """Calcula las diferencias históricas a nivel de norma o artículo específico con máxima velocidad."""
        clean_id = self._clean_id(law_identifier)
        file_path = f"ar/{clean_id}.md"

        # 1. Resolver SHAs
        commit_a = sha_a
        if not commit_a and date_a:
            commit_a = self.resolve_commit_for_date(law_identifier, date_a)

        commit_b = sha_b
        if not commit_b and date_b:
            commit_b = self.resolve_commit_for_date(law_identifier, date_b)

        # Si no se pasó commit_a o commit_b, consultar el historial con rev-list (ultra-rápido en C)
        log_shas: list[str] = []
        if not commit_a or not commit_b:
            try:
                raw_revs = self._run_git(["rev-list", "-n", "2", "HEAD", "--", file_path])
                log_shas = raw_revs.splitlines() if raw_revs else []
            except Exception as e:
                logger.warning("No se pudo resolver rev-list para %s: %s", law_identifier, e)

        if not commit_b:
            commit_b = log_shas[0] if log_shas else "HEAD"

        if not commit_a:
            if len(log_shas) >= 2:
                commit_a = log_shas[1]  # Versión anterior inmediata
            elif len(log_shas) == 1:
                commit_a = log_shas[0]
            else:
                commit_a = commit_b

        eff_a = commit_a
        eff_b = commit_b

        # ATAJO INSTANTÁNEO 1: Si ambos commits resueltos son idénticos (ej. norma con 1 sola versión como DNU-70-2023 o LEY-24013)
        if eff_a and eff_b and eff_a == eff_b:
            clean_art = article_number.replace("º", "").replace("°", "").replace(".", "").strip().lower() if article_number else None
            if clean_art:
                content = self.get_file_content_at_commit(law_identifier, eff_b)
                law = parse_markdown_law(content, law_identifier) if content else None
                law_title = law.title if law else law_identifier
                art = next((a for a in law.articles if a.article_number == clean_art), None) if law else None
                art_text = art.content_raw if art else "(Artículo no presente en esta norma)"
                art_diff = ArticleDiff(
                    law_identifier=law_identifier,
                    article_number=clean_art,
                    date_a=date_a,
                    date_b=date_b,
                    sha_a=eff_a,
                    sha_b=eff_b,
                    text_a=art_text,
                    text_b=art_text,
                    unified_diff="Sin modificaciones en la redacción del artículo.",
                    has_changes=False,
                    summary_of_changes="Texto idéntico entre versiones.",
                )
                return DiffResponse(
                    law_identifier=law_identifier,
                    law_title=law_title,
                    article_number=clean_art,
                    diff_source="git_local",
                    diff_text="Sin modificaciones en la redacción del artículo.",
                    analysis=f"El artículo {clean_art} de la norma {law_identifier} se mantiene idéntico en su versión registrada.",
                    article_diffs=[art_diff],
                )

            return DiffResponse(
                law_identifier=law_identifier,
                law_title=law_identifier,
                article_number=None,
                diff_source="git_local",
                diff_text="Sin diferencias textuales detectadas entre las versiones analizadas (texto idéntico).",
                analysis=f"La norma {law_identifier} cuenta con una única versión en el repositorio de control de versiones o se contrastó contra el mismo commit.",
                article_diffs=[],
            )

        # 2. Si se pidió un artículo específico y los commits difieren
        if article_number:
            clean_art = article_number.replace("º", "").replace("°", "").replace(".", "").strip().lower()
            try:
                content_a = self.get_file_content_at_commit(law_identifier, eff_a)
            except Exception:
                content_a = ""
            try:
                content_b = self.get_file_content_at_commit(law_identifier, eff_b)
            except Exception:
                content_b = ""

            # ATAJO INSTANTÁNEO 2: Si el contenido del archivo no cambió en nada entre ambos commits
            if content_a and content_b and content_a == content_b:
                return DiffResponse(
                    law_identifier=law_identifier,
                    law_title=law_identifier,
                    article_number=clean_art,
                    diff_source="git_local",
                    diff_text="Sin modificaciones en la redacción del artículo.",
                    analysis=f"El artículo {clean_art} de la norma {law_identifier} es idéntico entre las fechas consultadas.",
                    article_diffs=[],
                )

            law_a = parse_markdown_law(content_a, law_identifier) if content_a else None
            law_b = parse_markdown_law(content_b, law_identifier) if content_b else None
            law_title = (law_b.title if law_b else (law_a.title if law_a else law_identifier))

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
                    fromfile=f"{law_identifier} Art. {clean_art} ({date_a or eff_a[:8] or 'Inicial'})",
                    tofile=f"{law_identifier} Art. {clean_art} ({date_b or eff_b[:8] or 'Vigente'})",
                )
            )
            unified_str = "".join(diff_lines)
            has_changes = bool(unified_str.strip())

            art_diff = ArticleDiff(
                law_identifier=law_identifier,
                article_number=clean_art,
                date_a=date_a,
                date_b=date_b,
                sha_a=eff_a,
                sha_b=eff_b,
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

        # 3. Comparativa global de toda la norma mediante Git nativo (C-level ultra-rápido en ~5ms)
        try:
            unified_str = self._run_git(["diff", "-u", eff_a, eff_b, "--", file_path])
        except Exception as exc:
            logger.warning("Fallo al ejecutar git diff nativo para %s: %s", law_identifier, exc)
            unified_str = ""

        if not unified_str.strip():
            return DiffResponse(
                law_identifier=law_identifier,
                law_title=law_identifier,
                article_number=None,
                diff_source="git_local",
                diff_text="Sin diferencias globales detectadas entre ambas versiones.",
                analysis="No se registraron cambios textuales entre los commits comparados.",
                article_diffs=[],
            )

        # Detectar qué artículos cambiaron directamente desde el diff nativo o parseo acotado
        articles_changed: list[ArticleDiff] = []
        try:
            content_a = self.get_file_content_at_commit(law_identifier, eff_a)
            content_b = self.get_file_content_at_commit(law_identifier, eff_b)
            law_a = parse_markdown_law(content_a, law_identifier) if content_a else None
            law_b = parse_markdown_law(content_b, law_identifier) if content_b else None
            law_title = (law_b.title if law_b else (law_a.title if law_a else law_identifier))

            if law_a and law_b:
                map_a = {a.article_number: a for a in law_a.articles}
                map_b = {b.article_number: b for b in law_b.articles}
                all_nums = sorted(set(map_a.keys()) | set(map_b.keys()))
                for num in all_nums:
                    a_art = map_a.get(num)
                    b_art = map_b.get(num)
                    if not a_art or not b_art or a_art.content_hash != b_art.content_hash:
                        t_a = a_art.content_raw if a_art else "(No existía)"
                        t_b = b_art.content_raw if b_art else "(Derogado)"
                        articles_changed.append(
                            ArticleDiff(
                                law_identifier=law_identifier,
                                article_number=num,
                                date_a=date_a,
                                date_b=date_b,
                                sha_a=eff_a,
                                sha_b=eff_b,
                                text_a=t_a,
                                text_b=t_b,
                                unified_diff="Artículo modificado.",
                                has_changes=True,
                                summary_of_changes="Artículo modificado entre versiones.",
                            )
                        )
        except Exception as exc:
            logger.warning("Fallo al extraer artículos modificados de %s: %s", law_identifier, exc)
            law_title = law_identifier

        return DiffResponse(
            law_identifier=law_identifier,
            law_title=law_title,
            article_number=None,
            diff_source="git_local",
            diff_text=unified_str,
            analysis=f"Se identificaron {len(articles_changed)} artículos con reformas entre ambas versiones.",
            article_diffs=articles_changed[:50],  # Límite de seguridad
        )
