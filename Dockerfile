ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


FROM python:${PYTHON_VERSION}-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && git config --system --add safe.directory "*" \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


WORKDIR /app
COPY app/ app/
COPY scripts/ scripts/
COPY data/ data/
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN mkdir -p data/uploads repo_legalize_ar \
    && chmod +x /app/docker-entrypoint.sh \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request as u, sys; sys.exit(0 if u.urlopen('http://localhost:8000/api/v1/health', timeout=3).status == 200 else 1)"

ENV WEB_CONCURRENCY=4

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:asgi_app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
