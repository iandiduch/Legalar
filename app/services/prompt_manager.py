"""Cache en memoria + lock async sobre agent_prompts. Los nodos del grafo nunca
leen JSON ni Postgres directo -- siempre pasan por PromptManager.

El seed inicial (defaults versionados en el repo) corre una unica vez desde
scripts/init_db.py, nunca desde el lifespan de main.py."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import PromptNotFoundError
from app.db.models_orm import AgentPrompt
from app.schemas.prompts import AgentPromptDTO

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "agents" / "prompts" / "defaults"


class PromptManager:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession] | None = None,
        defaults_dir: Path | None = None,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._defaults_dir = defaults_dir if defaults_dir is not None else _DEFAULT_DIR
        self._cache: dict[str, AgentPromptDTO] = {}
        self._lock = asyncio.Lock()

    async def get_prompt(self, agent_id: str) -> str:
        cached = self._cache.get(agent_id)
        if cached is not None:
            return cached.content

        async with self._lock:
            cached = self._cache.get(agent_id)  # otro coroutine pudo poblarlo mientras esperabamos el lock
            if cached is not None:
                return cached.content
            dto = await self._load_from_db(agent_id)
            if dto is None:
                # Fallback resiliente al archivo .md correspondiente en disco
                dto = self._load_from_disk(agent_id)
            if dto is None:
                raise PromptNotFoundError(f"No hay prompt cargado para el agente '{agent_id}'")
            self._cache[agent_id] = dto
            return dto.content

    def _load_from_disk(self, agent_id: str) -> AgentPromptDTO | None:
        if self._defaults_dir and self._defaults_dir.exists():
            file_path = self._defaults_dir / f"{agent_id}.md"
            if file_path.is_file():
                content = file_path.read_text(encoding="utf-8")
                return AgentPromptDTO(
                    agent_id=agent_id,
                    content=content,
                    version=1,
                    updated_at=datetime.now(timezone.utc),
                    updated_by="disk_default",
                )
        return None

    async def update_prompt(self, agent_id: str, content: str, updated_by: str | None) -> AgentPromptDTO:
        if self._sessionmaker is None:
            raise RuntimeError("No se puede persistir la actualización del prompt sin conexión a base de datos")
        async with self._lock:
            async with self._sessionmaker() as session:
                row = await session.get(AgentPrompt, agent_id)
                if row is None:
                    raise PromptNotFoundError(f"No hay prompt cargado para el agente '{agent_id}'")
                row.content = content
                row.version += 1
                row.updated_by = updated_by
                await session.commit()
                await session.refresh(row)
                dto = AgentPromptDTO.model_validate(row, from_attributes=True)
            self._cache[agent_id] = dto
            return dto

    async def list_prompts(self) -> list[AgentPromptDTO]:
        if self._sessionmaker is None:
            return []
        async with self._sessionmaker() as session:
            result = await session.execute(select(AgentPrompt))
            return [AgentPromptDTO.model_validate(row, from_attributes=True) for row in result.scalars().all()]

    async def seed_defaults_if_empty(self, defaults_dir: Path) -> None:
        if self._sessionmaker is None:
            return
        async with self._sessionmaker() as session:
            for path in sorted(defaults_dir.glob("*.md")):
                content = path.read_text(encoding="utf-8")
                existing = await session.get(AgentPrompt, path.stem)
                if existing is None:
                    session.add(AgentPrompt(agent_id=path.stem, content=content, version=1, updated_by="system_seed"))
                elif existing.updated_by == "system_seed":
                    # Actualiza con el default más reciente del repo si no fue editado por un admin
                    existing.content = content
            await session.commit()

    async def _load_from_db(self, agent_id: str) -> AgentPromptDTO | None:
        if self._sessionmaker is None:
            return None
        async with self._sessionmaker() as session:
            row = await session.get(AgentPrompt, agent_id)
            return AgentPromptDTO.model_validate(row, from_attributes=True) if row is not None else None


_DEFAULT_MANAGER: PromptManager | None = None


def get_default_prompt_manager() -> PromptManager:
    """Retorna una instancia singleton de PromptManager con carga de prompts por defecto en disco."""
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = PromptManager(sessionmaker=None)
    return _DEFAULT_MANAGER


async def resolve_prompt(config: Any | None, agent_id: str, fallback: str | None = None) -> str:
    """Resuelve dinámicamente un system prompt para un nodo del grafo.

    Prioridad de resolución:
      1. Instancia `prompt_manager` inyectada en `config["configurable"]` (BD Postgres con caché en RAM).
      2. Instancia singleton default en memoria (`defaults/<agent_id>.md`).
      3. Fallback estático en código si el archivo no existe.
    """
    if config is not None:
        try:
            configurable = config.get("configurable", {}) if isinstance(config, dict) else getattr(config, "configurable", {})
            pm = configurable.get("prompt_manager")
            if pm is not None and hasattr(pm, "get_prompt"):
                return await pm.get_prompt(agent_id)
        except Exception:
            pass

    try:
        return await get_default_prompt_manager().get_prompt(agent_id)
    except Exception:
        return fallback or ""
