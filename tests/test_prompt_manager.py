"""Cache + lock + persistencia de PromptManager sobre agent_prompts real."""

import pytest

from app.core.exceptions import PromptNotFoundError
from app.services.prompt_manager import PromptManager


async def test_seed_defaults_populates_empty_table(db_sessionmaker, tmp_path):
    defaults_dir = tmp_path / "defaults"
    defaults_dir.mkdir()
    (defaults_dir / "supervisor.md").write_text("Sos el supervisor de prueba.", encoding="utf-8")

    manager = PromptManager(db_sessionmaker)
    await manager.seed_defaults_if_empty(defaults_dir)

    prompts = await manager.list_prompts()
    assert len(prompts) == 1
    assert prompts[0].agent_id == "supervisor"
    assert prompts[0].version == 1


async def test_seed_defaults_is_noop_when_not_empty(db_sessionmaker, tmp_path):
    defaults_dir = tmp_path / "defaults"
    defaults_dir.mkdir()
    (defaults_dir / "supervisor.md").write_text("v1", encoding="utf-8")

    manager = PromptManager(db_sessionmaker)
    await manager.seed_defaults_if_empty(defaults_dir)
    await manager.update_prompt("supervisor", "editado a mano", "ian")

    (defaults_dir / "supervisor.md").write_text("v2 -- no deberia pisar nada", encoding="utf-8")
    await manager.seed_defaults_if_empty(defaults_dir)

    assert await manager.get_prompt("supervisor") == "editado a mano"


async def test_get_prompt_missing_raises(db_sessionmaker):
    manager = PromptManager(db_sessionmaker)
    with pytest.raises(PromptNotFoundError):
        await manager.get_prompt("agente_inexistente")


async def test_update_prompt_bumps_version_and_invalidates_cache(db_sessionmaker, tmp_path):
    defaults_dir = tmp_path / "defaults"
    defaults_dir.mkdir()
    (defaults_dir / "validator.md").write_text("v1", encoding="utf-8")

    manager = PromptManager(db_sessionmaker)
    await manager.seed_defaults_if_empty(defaults_dir)
    await manager.get_prompt("validator")  # puebla la cache

    updated = await manager.update_prompt("validator", "v2", "ian")

    assert updated.version == 2
    assert await manager.get_prompt("validator") == "v2"


async def test_update_prompt_missing_agent_raises(db_sessionmaker):
    manager = PromptManager(db_sessionmaker)
    with pytest.raises(PromptNotFoundError):
        await manager.update_prompt("agente_inexistente", "contenido", "ian")
