"""Cliente resiliente para la API de Legalize con conmutación por cuota y fallback local.

Gestiona el límite gratuito de ~2.000 llamadas al mes mediante control local de cuota.
Si se excede el límite mensual o la API responde 429 Too Many Requests, conmuta
transparentemente al LocalGitDiffEngine sin interrumpir la operación del usuario.
"""

import logging
from typing import Any

import httpx

from app.core.config import Settings
from app.schemas.legal.diff import DiffResponse
from app.services.legal.local_diff_engine import LocalGitDiffEngine

logger = logging.getLogger(__name__)


class LegalizeApiClient:
    """Cliente HTTP para Legalize API con control de cuota y fallback transparente a Git."""

    def __init__(self, settings: Settings, local_diff_engine: LocalGitDiffEngine | None = None) -> None:
        self.settings = settings
        self.base_url = settings.LEGALIZE_API_BASE_URL.rstrip("/")
        self.api_key = settings.LEGALIZE_API_KEY.get_secret_value() if settings.LEGALIZE_API_KEY else ""
        self.monthly_limit = settings.LEGALIZE_API_MONTHLY_LIMIT
        self._local_engine = local_diff_engine or LocalGitDiffEngine(settings.LEGALIZE_REPO_PATH)
        self._calls_this_month = 0

    @property
    def is_quota_available(self) -> bool:
        """Verifica si aún queda saldo en la cuota gratuita mensual."""
        return self._calls_this_month < self.monthly_limit

    def _get_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def get_diff(
        self,
        law_identifier: str,
        date_a: str | None = None,
        date_b: str | None = None,
        article: str | None = None,
        country: str = "ar",
    ) -> DiffResponse:
        """Consulta el diff a la API o conmuta automáticamente al motor local si no hay cuota o falla."""
        # 1. Si no hay cuota disponible o no hay API key, ir directo a local
        if not self.is_quota_available or not self.api_key:
            logger.info("Usando LocalGitDiffEngine para %s (Cuota agotada o sin API key)", law_identifier)
            return self._local_engine.compute_diff(
                law_identifier=law_identifier,
                date_a=date_a,
                date_b=date_b,
                article_number=article,
            )

        # 2. Intentar llamar a la API
        params: dict[str, Any] = {}
        if date_a:
            params["date_a"] = date_a
        if date_b:
            params["date_b"] = date_b
        if article:
            params["article"] = article

        url = f"{self.base_url}/api/v1/{country}/laws/{law_identifier}/diff"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params, headers=self._get_headers())

                if resp.status_code == 200:
                    self._calls_this_month += 1
                    data = resp.json()
                    return DiffResponse(
                        law_identifier=law_identifier,
                        law_title=data.get("title") or law_identifier,
                        article_number=article,
                        diff_source="api",
                        diff_text=data.get("diff") or "Sin cambios registrados.",
                        analysis=data.get("analysis") or "Diff provisto por Legalize API.",
                        article_diffs=[],
                    )

                if resp.status_code == 429:
                    logger.warning("Cuota mensual de Legalize API alcanzada (429). Conmutando a LocalGitDiffEngine.")
                    self._calls_this_month = self.monthly_limit  # Marcar cuota agotada
                    return self._local_engine.compute_diff(
                        law_identifier=law_identifier,
                        date_a=date_a,
                        date_b=date_b,
                        article_number=article,
                    )

                logger.warning(
                    "Respuesta no exitosa de Legalize API (status %d). Conmutando a fallback local.",
                    resp.status_code,
                )
        except Exception as exc:
            logger.warning("Fallo al conectar con Legalize API (%s). Conmutando a fallback local.", exc)

        # 3. Fallback en caso de cualquier error
        return self._local_engine.compute_diff(
            law_identifier=law_identifier,
            date_a=date_a,
            date_b=date_b,
            article_number=article,
        )

    async def get_changes(self, since: str | None = None, cursor: int | None = None, country: str = "ar") -> dict[str, Any] | None:
        """Consulta el endpoint /changes para saber qué normas sufrieron reformas recientemente."""
        if not self.is_quota_available or not self.api_key:
            return None

        url = f"{self.base_url}/api/v1/{country}/changes"
        params: dict[str, Any] = {}
        if since:
            params["since"] = since
        if cursor is not None:
            params["cursor"] = cursor

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(url, params=params, headers=self._get_headers())
                if resp.status_code == 200:
                    self._calls_this_month += 1
                    return resp.json()
                if resp.status_code == 429:
                    self._calls_this_month = self.monthly_limit
                    return None
        except Exception as exc:
            logger.warning("Fallo al consultar changes en Legalize API: %s", exc)

        return None
