"""Fixtures compartidas de pytest.

Factories de sesión SQLAlchemy, base de datos de prueba y mocks de LLM.
"""

import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.db.models_orm import Base


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    return Settings(
        OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY", "sk-test-placeholder"),
        PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY", "pcsk-test-placeholder"),
        POSTGRES_HOST=os.environ.get("POSTGRES_HOST", "localhost"),
        POSTGRES_DB=os.environ.get("POSTGRES_DB", "intelligence_system"),
        POSTGRES_USER=os.environ.get("POSTGRES_USER", "postgres"),
        POSTGRES_PASSWORD=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        REDIS_HOST=os.environ.get("REDIS_HOST", "localhost"),
        PHOENIX_ENABLED=False,
        PROMETHEUS_METRICS_ENABLED=True,
    )


@pytest_asyncio.fixture
async def db_sessionmaker(test_settings: Settings) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(test_settings.database_url)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:  # noqa: BLE001 - Salta tests si PostgreSQL no está levantado
        await engine.dispose()
        pytest.skip(f"PostgreSQL no disponible ({exc}) -- levantar 'docker compose up -d postgres' para este test")

    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield sessionmaker
    finally:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
        except Exception:  # noqa: BLE001 - Teardown silencioso de base de datos de test
            pass
        await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_sessionmaker: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    async with db_sessionmaker() as session:
        yield session


class _FakeStructuredRunnable:
    def __init__(self, responses: list[BaseModel]) -> None:
        self._responses = responses

    async def ainvoke(self, messages: object) -> BaseModel:
        if not self._responses:
            raise AssertionError("FakeLLM se quedo sin respuestas programadas para este schema")
        return self._responses.pop(0)


class FakeLLM:
    """Cubre la superficie mínima de BaseChatModel que usa este proyecto:
    with_structured_output() (para invoke_structured_with_retry) y
    bind_tools()+ainvoke() (para tool-calling de agentes legales)."""

    def __init__(self) -> None:
        self._structured: dict[type, list[BaseModel]] = {}
        self._tool_calls: list[AIMessage] = []

    def program_structured(self, schema: type, *responses: BaseModel) -> None:
        self._structured.setdefault(schema, []).extend(responses)

    def program_tool_call(self, message: AIMessage) -> None:
        self._tool_calls.append(message)

    def with_structured_output(self, schema: type, *args: object, **kwargs: object) -> _FakeStructuredRunnable:
        return _FakeStructuredRunnable(self._structured.setdefault(schema, []))

    def bind_tools(self, tools: object) -> "FakeLLM":
        return self

    async def ainvoke(self, messages: object) -> AIMessage:
        if self._tool_calls:
            return self._tool_calls.pop(0)
        return AIMessage(content="")


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def new_thread_id() -> str:
    return str(uuid4())
