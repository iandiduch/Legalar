"""Cache en memoria + lock async sobre agent_prompts. Los nodos del grafo nunca
leen JSON ni Postgres directo -- siempre pasan por PromptManager.

El seed inicial (defaults versionados en el repo) corre una unica vez desde
scripts/init_db.py, nunca desde el lifespan de main.py."""

import asyncio
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import PromptNotFoundError
from app.db.models_orm import AgentPrompt
from app.schemas.prompts import AgentPromptDTO


class PromptManager:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker
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
                raise PromptNotFoundError(f"No hay prompt cargado para el agente '{agent_id}'")
            self._cache[agent_id] = dto
            return dto.content

    async def update_prompt(self, agent_id: str, content: str, updated_by: str | None) -> AgentPromptDTO:
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
        async with self._sessionmaker() as session:
            result = await session.execute(select(AgentPrompt))
            return [AgentPromptDTO.model_validate(row, from_attributes=True) for row in result.scalars().all()]

    async def seed_defaults_if_empty(self, defaults_dir: Path) -> None:
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
        async with self._sessionmaker() as session:
            row = await session.get(AgentPrompt, agent_id)
            return AgentPromptDTO.model_validate(row, from_attributes=True) if row is not None else None
