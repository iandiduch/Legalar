"""Métricas de Prometheus y observabilidad de series temporales.

Expone métricas clave del sistema (tráfico HTTP, latencia, ejecuciones de agentes,
tickets HOTL creados y jobs de ingesta) bajo el endpoint estándar `/metrics`.
"""

import time
from typing import ClassVar

from fastapi import APIRouter, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# --- Definición de Métricas Prometheus ---

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total de peticiones HTTP recibidas",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Duración de peticiones HTTP en segundos",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

HOTL_TICKETS_TOTAL = Counter(
    "hotl_tickets_total",
    "Total de tickets de escalamiento HOTL creados",
    ["priority", "channel", "escalation_type"],
)

INGESTION_JOBS_TOTAL = Counter(
    "ingestion_jobs_total",
    "Total de jobs de ingesta procesados",
    ["status", "file_type"],
)

AGENT_EXECUTIONS_TOTAL = Counter(
    "agent_executions_total",
    "Total de ejecuciones por agente en el grafo",
    ["agent_name"],
)


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware para registrar métricas de tráfico y latencia HTTP."""

    _EXCLUDED_PATHS: ClassVar[set[str]] = {"/metrics", "/api/v1/health", "/docs", "/redoc", "/api/v1/openapi.json"}

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if path in self._EXCLUDED_PATHS:
            return await call_next(request)

        method = request.method
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = str(response.status_code)
        except Exception:
            status_code = "500"
            raise
        finally:
            duration = time.perf_counter() - start_time
            # Normalizar endpoint para evitar cardinalidad infinita en URLs con IDs
            endpoint = self._normalize_path(path)
            HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(method=method, endpoint=endpoint).observe(duration)

        return response

    @staticmethod
    def _normalize_path(path: str) -> str:
        parts = path.split("/")
        normalized = []
        for part in parts:
            if len(part) == 36 and "-" in part:  # UUID detectado
                normalized.append("{id}")
            else:
                normalized.append(part)
        return "/".join(normalized)


metrics_router = APIRouter()


@metrics_router.get("/metrics", summary="Métricas de Prometheus", tags=["metrics"], include_in_schema=False)
async def get_metrics() -> Response:
    """Exporta las métricas recopiladas en formato de texto Prometheus."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
