"""Providers de dependencias para FastAPI.

Lo compartido y de alto costo (motores, clientes vectoriales, modelos, retriever híbrido)
vive en request.app.state.*, inicializado una única vez durante el lifespan.
"""

from typing import Annotated, Any

from fastapi import Depends, Request
from langgraph.graph.state import CompiledStateGraph
from opentelemetry import trace
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.security import get_current_api_key
from app.db.models_orm import ApiKey
from app.db.session import get_db_session
from app.services.api_key_service import ApiKeyService
from app.services.legal.legalize_api_client import LegalizeApiClient
from app.services.legal.local_diff_engine import LocalGitDiffEngine
from app.services.legal.retriever import HybridLegalRetriever
from app.services.prompt_manager import PromptManager


def get_settings_dep() -> Settings:
    return get_settings()


def get_redis_client(request: Request) -> Redis:
    return request.app.state.redis_client


def get_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.compiled_graph


def get_tracer() -> trace.Tracer:
    return trace.get_tracer("legal-agent.api")


def get_llm_client(request: Request) -> Any:
    return request.app.state.llm_client


def get_legal_retriever(request: Request) -> HybridLegalRetriever:
    return request.app.state.legal_retriever


def get_local_diff_engine(request: Request) -> LocalGitDiffEngine:
    return request.app.state.local_diff_engine


def get_legalize_api_client(request: Request) -> LegalizeApiClient:
    return request.app.state.legalize_api_client


def get_prompt_manager(request: Request) -> PromptManager:
    return request.app.state.prompt_manager


def get_api_key_service(
    session: Annotated[AsyncSession, Depends(get_db_session)], settings: Annotated[Settings, Depends(get_settings_dep)]
) -> ApiKeyService:
    return ApiKeyService(session, settings)


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]
RedisDep = Annotated[Redis, Depends(get_redis_client)]
GraphDep = Annotated[CompiledStateGraph, Depends(get_graph)]
LLMClientDep = Annotated[Any, Depends(get_llm_client)]
ApiKeyServiceDep = Annotated[ApiKeyService, Depends(get_api_key_service)]
CurrentApiKeyDep = Annotated[ApiKey, Depends(get_current_api_key)]
LegalRetrieverDep = Annotated[HybridLegalRetriever, Depends(get_legal_retriever)]
LocalDiffEngineDep = Annotated[LocalGitDiffEngine, Depends(get_local_diff_engine)]
LegalizeApiClientDep = Annotated[LegalizeApiClient, Depends(get_legalize_api_client)]
PromptManagerDep = Annotated[PromptManager, Depends(get_prompt_manager)]
