"""Tests unitarios para la carga y fallback de prompts por defecto de PromptManager."""

import pytest
from app.services.prompt_manager import PromptManager


@pytest.mark.asyncio
async def test_prompt_manager_loads_all_defaults_from_disk():
    # Instancia sin DB (modo offline/fallback)
    manager = PromptManager(sessionmaker=None)

    expected_agents = [
        "router",
        "legal_agent",
        "query_rewrite",
        "query_expander",
        "document_analyzer",
        "validator",
        "general_inquiry",
        "diff_explanation",
        "supervisor",
    ]

    for agent_id in expected_agents:
        content = await manager.get_prompt(agent_id)
        assert content is not None
        assert len(content.strip()) > 30, f"Prompt {agent_id} está vacío o es demasiado corto"

    # Verificar que las consultas repetidas usan la caché en memoria
    cached_content = await manager.get_prompt("router")
    assert cached_content == await manager.get_prompt("router")
