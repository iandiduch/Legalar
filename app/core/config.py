from functools import lru_cache
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priorizar el archivo .env del proyecto por sobre variables globales del sistema operativo
        return (
            init_settings,
            dotenv_settings,
            env_settings,
            file_secret_settings,
        )

    # --- App ---
    ENVIRONMENT: Literal["local", "docker", "production"] = "local"
    LOG_LEVEL: str = "INFO"
    API_PORT: int = 8000
    UVICORN_WORKERS: int = 4
    CORS_ORIGINS: list[str] = ["*"]
    DOCS_ENABLED: bool | None = None

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v_clean = v.strip()
            if not v_clean:
                return ["*"]
            if v_clean.startswith("[") and v_clean.endswith("]"):
                try:
                    import json
                    parsed = json.loads(v_clean)
                    if isinstance(parsed, list):
                        return [str(x).strip() for x in parsed]
                except Exception:
                    pass
            return [x.strip() for x in v_clean.split(",") if x.strip()]
        return ["*"]


    @field_validator("DOCS_ENABLED", mode="before")
    @classmethod
    def _parse_docs_enabled(cls, v: Any) -> bool | None:
        if v == "" or v is None:
            return None
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in ("true", "1", "yes", "on"):
                return True
            if clean in ("false", "0", "no", "off"):
                return False
            if clean == "":
                return None
        return v

    # --- LLM / OpenAI / OpenRouter ---
    OPENAI_API_KEY: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("OPENAI_API_KEY", "OPENAI_APIKEY", "OPENROUTER_API_KEY"),
    )
    OPENAI_BASE_URL: str | None = None
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_VISION_MODEL: str = "google/gemini-2.0-flash-exp:free"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 2500
    LLM_REQUEST_TIMEOUT: float = 40.0
    MAX_FILE_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB

    @property
    def is_openrouter(self) -> bool:
        key = self.OPENAI_API_KEY.get_secret_value()
        base = self.OPENAI_BASE_URL or ""
        return key.startswith("sk-or-") or "openrouter.ai" in base

    @property
    def effective_openai_base_url(self) -> str | None:
        if self.OPENAI_BASE_URL:
            return self.OPENAI_BASE_URL.rstrip("/")
        if self.is_openrouter:
            return "https://openrouter.ai/api/v1"
        return None

    @property
    def effective_chat_model(self) -> str:
        model = self.OPENAI_CHAT_MODEL
        if self.is_openrouter and "/" not in model:
            return f"openai/{model}"
        return model

    @property
    def effective_vision_model(self) -> str:
        model = self.OPENAI_VISION_MODEL
        if self.is_openrouter and "/" not in model:
            return f"openai/{model}"
        return model

    @property
    def effective_embedding_model(self) -> str:
        model = self.OPENAI_EMBEDDING_MODEL
        if self.is_openrouter and "/" not in model:
            return f"openai/{model}"
        return model


    # --- Pinecone ---
    PINECONE_API_KEY: SecretStr = Field(
        default=SecretStr(""), validation_alias=AliasChoices("PINECONE_API_KEY", "PINECONE_APIKEY")
    )
    PINECONE_INDEX_NAME: str = "intelligence-system"
    PINECONE_DIMENSION: int = 1536
    PINECONE_METRIC: str = "cosine"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"
    PINECONE_UPSERT_BATCH_SIZE: int = 100

    # --- Postgres (tablas propias: SQLAlchemy async + asyncpg) ---
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "intelligence_system"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: SecretStr = SecretStr("postgres")
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 30

    # --- Checkpointer LangGraph (psycopg, pool separado) ---
    CHECKPOINTER_POOL_MIN_SIZE: int = 2
    CHECKPOINTER_POOL_MAX_SIZE: int = 20

    # --- Idempotencia y Concurrencia ---
    IDEMPOTENCY_ENABLED: bool = True
    IDEMPOTENCY_TTL_SECONDS: int = 60

    # --- Redis ---
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_INGEST_QUEUE_KEY: str = "ingestion:jobs"

    # --- Ingesta ---
    MAX_UPLOAD_SIZE_MB: int = 20
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 300
    UPLOAD_DIR: str = "data/uploads"
    INGESTION_JOB_TIMEOUT_MINUTES: int = 15
    INGESTION_MAX_RETRIES: int = 3
    INGESTION_CONTENT_SCAN_ENABLED: bool = True

    # --- Agentes / grafo ---
    MAX_SUPERVISOR_ITERATIONS: int = 6
    GRAPH_RECURSION_LIMIT: int = 25
    STRUCTURED_OUTPUT_MAX_ATTEMPTS: int = 3

    # --- Legalize / Legislación Argentina ---
    LEGALIZE_REPO_PATH: str = "repo_legalize_ar"
    LEGALIZE_API_BASE_URL: str = "https://legalize.dev"
    LEGALIZE_API_KEY: SecretStr = Field(default=SecretStr(""))
    LEGALIZE_API_MONTHLY_LIMIT: int = 2000
    PINECONE_LEGAL_NAMESPACE: str = "ar-legislation"

    # --- Observabilidad & Métricas ---
    PHOENIX_COLLECTOR_ENDPOINT: str = "http://localhost:6006/v1/traces"
    PHOENIX_PROJECT_NAME: str = "intelligence-system"
    PHOENIX_ENABLED: bool = True
    PROMETHEUS_METRICS_ENABLED: bool = True

    # --- Seguridad / auth ---
    API_KEY_HEADER_NAME: str = "X-API-Key"
    API_KEY_PEPPER: SecretStr = SecretStr("")
    BOOTSTRAP_ADMIN_API_KEY: SecretStr | None = None

    # --- Rate limiting ---
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    RATE_LIMIT_ANONYMOUS_PER_MINUTE: int = 20
    RATE_LIMIT_CLIENT_PER_MINUTE: int = 60
    RATE_LIMIT_CLIENT_CHAT_PER_MINUTE: int = 15
    RATE_LIMIT_ADMIN_PER_MINUTE: int = 120

    # --- Proxy / IP ---
    TRUSTED_PROXY_IPS: str = "127.0.0.1"
    ADMIN_IP_ALLOWLIST: str = ""

    # --- Retrieval hibrido (FTS Postgres + vectorial Pinecone) ---
    HYBRID_RETRIEVAL_ENABLED: bool = True
    LEXICAL_WEIGHT: float = 0.4
    VECTOR_WEIGHT: float = 0.6
    HYBRID_CANDIDATE_MULTIPLIER: int = 4
    RAG_TOP_K: int = 8

    # --- Evaluacion RAG (LLM-as-judge) ---
    EVALUATION_PASS_THRESHOLD: float = 0.7

    @property
    def database_url(self) -> str:
        """DSN async para SQLAlchemy/asyncpg. No es un computed_field a proposito:
        una property comun nunca se serializa en model_dump(), asi la password
        no puede terminar filtrada en un log o una respuesta accidental."""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD.get_secret_value()}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def postgres_dsn(self) -> str:
        """DSN libpq plano (sin '+asyncpg') para el pool psycopg del checkpointer."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD.get_secret_value()}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
