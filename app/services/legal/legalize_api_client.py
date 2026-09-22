"""Cliente resiliente para la API de Legalize con SDK oficial, conmutación por cuota y fallback local.

Gestiona la integración con legalize.dev mediante la librería oficial `legalize` (AsyncLegalize),
garantizando la resolución estricta de parámetros obligatorios (date_a, date_b) para evitar errores HTTP 422,
normalización de anclas de artículos con reintento ante sugerencias `closest`, control de cuota mensual
y fallback transparente a `LocalGitDiffEngine`.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any

from legalize import APIError, AsyncLegalize, NotFoundError, RateLimitError

from app.core.config import Settings
from app.schemas.legal.diff import ArticleDiff, DiffResponse
from app.services.legal.local_diff_engine import LocalGitDiffEngine

logger = logging.getLogger(__name__)


def normalize_article_anchor(article: str) -> str:
    """Normaliza una referencia o número de artículo al formato de ancla requerido por Legalize API.

    Ejemplos:
        '92' -> 'articulo-92'
        '92 ter' -> 'articulo-92-ter'
        'Art. 1198' -> 'articulo-1198'
        'articulo-92-ter' -> 'articulo-92-ter'
    """
    cleaned = article.strip().lower()
    cleaned = cleaned.replace("º", "").replace("°", "").replace(".", "").strip()
    cleaned = re.sub(r"^(artículo|articulo|art)[\s\-_]*", "", cleaned).strip()
    cleaned = re.sub(r"[\s_]+", "-", cleaned)
    return f"articulo-{cleaned}"


def extract_closest_anchor(err: NotFoundError) -> str | None:
    """Extrae la primera ancla sugerida del cuerpo de error si Legalize API devolvió 'closest'."""
    if not err.body:
        return None
    try:
        raw_body = err.body.decode("utf-8") if isinstance(err.body, bytes) else str(err.body)
        body_data = json.loads(raw_body)
        closest = body_data.get("closest") or body_data.get("detail", {}).get("closest")
        if closest and isinstance(closest, list) and len(closest) > 0:
            return str(closest[0])
    except Exception:
        pass
    return None


class LegalizeApiClient:
    """Cliente oficial para Legalize API con control de cuota, prevención de 422 y fallback a Git."""

    def __init__(self, settings: Settings, local_diff_engine: LocalGitDiffEngine | None = None) -> None:
        self.settings = settings
        self.base_url = settings.LEGALIZE_API_BASE_URL.rstrip("/")
        self.api_key = settings.LEGALIZE_API_KEY.get_secret_value() if settings.LEGALIZE_API_KEY else ""
        self.monthly_limit = settings.LEGALIZE_API_MONTHLY_LIMIT
        self._local_engine = local_diff_engine or LocalGitDiffEngine(
            repo_path=settings.LEGALIZE_REPO_PATH,
            country_code=settings.LEGALIZE_COUNTRY_CODE,
        )
        self._calls_this_month = 0
        self._sdk_client: AsyncLegalize | None = None

    @property
    def is_quota_available(self) -> bool:
        """Verifica si aún queda saldo en la cuota gratuita mensual."""
        return self._calls_this_month < self.monthly_limit

    @property
    def has_valid_api_key(self) -> bool:
        """Verifica que la clave de API tenga el formato esperado por el SDK de Legalize ('leg_...')."""
        return bool(self.api_key and self.api_key.startswith("leg_"))

    def _get_sdk_client(self) -> AsyncLegalize | None:
        """Obtiene o reutiliza la instancia de AsyncLegalize con conexión asíncrona."""
        if not self.has_valid_api_key:
            return None
        if self._sdk_client is None:
            self._sdk_client = AsyncLegalize(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=3.0,
                max_retries=0,
            )
        return self._sdk_client

    async def aclose(self) -> None:
        """Cierra ordenadamente los recursos de red del cliente del SDK."""
        if self._sdk_client is not None:
            try:
                await self._sdk_client.aclose()
            except Exception as exc:
                logger.debug("Excepción silenciada al cerrar AsyncLegalize: %s", exc)
            self._sdk_client = None

    async def _resolve_dates(
        self,
        client: AsyncLegalize,
        law_identifier: str,
        country: str,
        date_a: str | None,
        date_b: str | None,
    ) -> tuple[str | None, str | None, str | None]:
        """Resuelve dinámicamente dos fechas válidas (YYYY-MM-DD) para el diff.

        Devuelve: (date_a, date_b, error_reason)
        """
        # Si ambas fechas ya fueron provistas explícitamente
        if date_a and date_b:
            da = date_a.strip()[:10]
            db = date_b.strip()[:10]
            if da > db:
                da, db = db, da
            if da == db:
                return None, None, "equal_dates"
            return da, db, None

        # Si faltan fechas, consultar el historial oficial de reformas de la norma
        try:
            reforms_res = await client.reforms.list(country=country, law_id=law_identifier)
            reforms = reforms_res.reforms if reforms_res else []
        except NotFoundError:
            logger.info("Norma %s no registrada en Legalize API (404). Usando fallback local.", law_identifier)
            return None, None, "law_not_found"
        except RateLimitError:
            logger.warning("Cuota mensual de Legalize API alcanzada al listar reformas (429).")
            self._calls_this_month = self.monthly_limit
            return None, None, "quota_exceeded"
        except Exception as exc:
            logger.warning("Fallo al consultar reformas de %s: %s. Conmutando a fallback local.", law_identifier, exc)
            return None, None, "reforms_lookup_failed"

        # Si no posee reformas registradas (ej: DNU-70-2023 u ordenanzas en versión original)
        if len(reforms) == 0:
            logger.info("Norma %s tiene 0 reformas registradas en Legalize API (versión original única).", law_identifier)
            return None, None, "zero_reforms"

        # Si posee 2 o más reformas, comparar la última versión reformada contra la versión anterior
        if len(reforms) >= 2:
            resolved_b = date_b.strip()[:10] if date_b else reforms[0].date
            resolved_a = date_a.strip()[:10] if date_a else reforms[1].date
            if resolved_a > resolved_b:
                resolved_a, resolved_b = resolved_b, resolved_a
            if resolved_a == resolved_b:
                return None, None, "equal_dates"
            return resolved_a, resolved_b, None

        # Si posee menos de 2 reformas en la API (ej: LEY-24013 con 1 sola reforma registrada),
        # Legalize API no dispone de dos versiones históricas en su repositorio git.
        # NUNCA usar publication_date (ej 1991-12-17) porque la API rechaza con HTTP 400 before_history
        # fechas anteriores a 2006. Conmutar directamente al motor local.
        logger.info(
            "Norma %s tiene solo %d reforma registrada en Legalize API. Conmutando a LocalGitDiffEngine.",
            law_identifier,
            len(reforms),
        )
        return None, None, "insufficient_api_reforms"

    async def get_diff(
        self,
        law_identifier: str,
        date_a: str | None = None,
        date_b: str | None = None,
        article: str | None = None,
        country: str = "ar",
    ) -> DiffResponse:
        """Consulta el diff a Legalize API con resolución estricta anti-422 y fallback a Git local.

        Garantías del flujo:
        1. Si no hay cuota disponible o la API key no es válida, conmuta de inmediato al motor local.
        2. Si faltan date_a o date_b, consulta el historial oficial de reformas para deducirlas.
        3. Si la norma es versión única (0 reformas) o no existe en la API remota (404), conmuta a Git
           local sin ejecutar ninguna llamada a /diff que resulte en error 422 o timeout.
        4. Si se solicita un artículo y la API sugiere alternativas ('closest'), reintenta automáticamente.
        5. Procesa normas extensas con 'diff_omitted' presentando los artículos modificados.
        """
        # 1. Preferencia local: si la norma está en el repositorio Git clonado, resolver localmente (< 0.3s)
        clean_id = law_identifier.replace("/", "-").strip()
        disk_path = os.path.join(self._local_engine.repo_path, f"ar/{clean_id}.md")
        if os.path.exists(disk_path):
            logger.info("Resolviendo diff para %s vía LocalGitDiffEngine local (< 0.3s)", law_identifier)
            return await asyncio.to_thread(
                self._local_engine.compute_diff,
                law_identifier=law_identifier,
                date_a=date_a,
                date_b=date_b,
                article_number=article,
            )

        # 2. Fallback si no hay cuota o API key adecuada
        if not self.is_quota_available or not self.has_valid_api_key:
            logger.info("Usando LocalGitDiffEngine para %s (Cuota agotada o API key ausente)", law_identifier)
            return await asyncio.to_thread(
                self._local_engine.compute_diff,
                law_identifier=law_identifier,
                date_a=date_a,
                date_b=date_b,
                article_number=article,
            )

        client = self._get_sdk_client()
        if client is None:
            return await asyncio.to_thread(
                self._local_engine.compute_diff,
                law_identifier=law_identifier,
                date_a=date_a,
                date_b=date_b,
                article_number=article,
            )

        # 2. Resolver fechas requeridas (date_a y date_b) para evitar HTTP 422 y HTTP 400
        resolved_a, resolved_b, reason = await self._resolve_dates(
            client=client,
            law_identifier=law_identifier,
            country=country,
            date_a=date_a,
            date_b=date_b,
        )

        # Si no hay dos fechas válidas, NUNCA llamar a /diff (evita el 422/400 en la API remota)
        if not resolved_a or not resolved_b:
            logger.info(
                "Conmutando a LocalGitDiffEngine para %s (Razón de fechas: %s)",
                law_identifier,
                reason,
            )
            return await asyncio.to_thread(
                self._local_engine.compute_diff,
                law_identifier=law_identifier,
                date_a=date_a,
                date_b=date_b,
                article_number=article,
            )

        # 3. Preparar parámetros con fechas verificadas
        params: dict[str, Any] = {
            "date_a": resolved_a,
            "date_b": resolved_b,
        }

        target_article = None
        if article:
            target_article = normalize_article_anchor(article)
            params["article"] = target_article

        endpoint_url = f"/api/v1/{country}/laws/{law_identifier}/diff"

        # 4. Invocar la API mediante el SDK oficial
        try:
            data = await client.request("GET", endpoint_url, params=params)
            self._calls_this_month += 1
        except NotFoundError as err:
            # Si el artículo no fue hallado pero la API sugiere alternativas en 'closest'
            closest_anchor = extract_closest_anchor(err)
            if closest_anchor and closest_anchor != target_article:
                logger.info("Reintentando diff de %s con ancla sugerida por la API: %s", law_identifier, closest_anchor)
                params["article"] = closest_anchor
                try:
                    data = await client.request("GET", endpoint_url, params=params)
                    self._calls_this_month += 1
                except Exception as retry_err:
                    logger.warning("Fallo en reintento con ancla sugerida %s: %s", closest_anchor, retry_err)
                    return await asyncio.to_thread(
                        self._local_engine.compute_diff,
                        law_identifier=law_identifier,
                        date_a=resolved_a,
                        date_b=resolved_b,
                        article_number=article,
                    )
            else:
                logger.info("Artículo o norma no encontrado en Legalize API (%s). Conmutando a local.", err)
                return await asyncio.to_thread(
                    self._local_engine.compute_diff,
                    law_identifier=law_identifier,
                    date_a=resolved_a,
                    date_b=resolved_b,
                    article_number=article,
                )
        except RateLimitError:
            logger.warning("Cuota mensual de Legalize API alcanzada (429). Conmutando a LocalGitDiffEngine.")
            self._calls_this_month = self.monthly_limit
            return await asyncio.to_thread(
                self._local_engine.compute_diff,
                law_identifier=law_identifier,
                date_a=resolved_a,
                date_b=resolved_b,
                article_number=article,
            )
        except (APIError, Exception) as exc:
            logger.warning("Error al consultar Legalize API (%s). Conmutando a LocalGitDiffEngine.", exc)
            return await asyncio.to_thread(
                self._local_engine.compute_diff,
                law_identifier=law_identifier,
                date_a=resolved_a,
                date_b=resolved_b,
                article_number=article,
            )

        # 5. Formatear la respuesta estructurada desde Legalize API
        law_title = data.get("citation") or data.get("title") or law_identifier
        diff_omitted = bool(data.get("diff_omitted", False))
        diff_text = data.get("diff")

        if not diff_text:
            if diff_omitted:
                changed = data.get("changed_headings") or []
                items_str = "\n".join(
                    f"- {h.get('title', 'Artículo')} (ancla: {h.get('anchor', 'n/a')})"
                    for h in changed
                )
                diff_text = (
                    f"La norma supera el límite de longitud para diff completo sin segmentar.\n"
                    f"Se detectaron reformas entre {resolved_a} y {resolved_b} en los siguientes artículos:\n"
                    f"{items_str or 'Sin artículos especificados.'}\n\n"
                    f"Nota oficial: {data.get('note', 'Consulte un artículo específico para ver su redacción comparada.')}"
                )
            else:
                diff_text = "No se registraron diferencias textuales entre las versiones analizadas."

        analysis = (
            data.get("note")
            or data.get("analysis")
            or f"Comparación histórica entre {resolved_a} y {resolved_b} provista por Legalize API."
        )

        article_diffs: list[ArticleDiff] = []
        if article and data.get("diff"):
            article_diffs.append(
                ArticleDiff(
                    law_identifier=law_identifier,
                    article_number=article,
                    date_a=resolved_a,
                    date_b=resolved_b,
                    sha_a=data.get("sha_a"),
                    sha_b=data.get("sha_b"),
                    text_a="(Versión previa en Legalize API)",
                    text_b="(Versión reformada en Legalize API)",
                    unified_diff=data["diff"],
                    has_changes=bool(data.get("changed", True)),
                    summary_of_changes=data.get("citation"),
                )
            )

        return DiffResponse(
            law_identifier=law_identifier,
            law_title=law_title,
            article_number=article,
            diff_source="api",
            diff_text=diff_text,
            analysis=analysis,
            article_diffs=article_diffs,
        )

    async def get_changes(
        self,
        since: str | None = None,
        cursor: int | None = None,
        country: str = "ar",
    ) -> dict[str, Any] | None:
        """Consulta el endpoint /changes para saber qué normas sufrieron reformas recientemente."""
        if not self.is_quota_available or not self.has_valid_api_key:
            return None

        client = self._get_sdk_client()
        if client is None:
            return None

        params: dict[str, Any] = {}
        if since:
            params["since"] = since
        if cursor is not None:
            params["cursor"] = cursor

        try:
            data = await client.request("GET", f"/api/v1/{country}/changes", params=params)
            self._calls_this_month += 1
            return data
        except RateLimitError:
            self._calls_this_month = self.monthly_limit
            return None
        except Exception as exc:
            logger.warning("Fallo al consultar changes en Legalize API: %s", exc)
            return None
