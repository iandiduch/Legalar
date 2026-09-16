"""Verifica que el estado de una conversación legal sobrevive un reinicio:
se crea un checkpointer/pool NUEVO contra PostgreSQL y se confirma que el thread_id
sigue resolviendo el historial acumulado en el grafo legal.
"""

import pytest
from langchain_core.messages import HumanMessage

from app.agents.graph import build_graph
from app.agents.legal.state import LegalAgentState
from app.db.checkpointer import build_checkpointer, build_checkpointer_pool
from app.domain.models import ConfidenceLevel, QueryIntent
from app.schemas.agents import LegalAnswerPayload, RouterDecision


async def test_thread_state_survives_fresh_checkpointer(test_settings, fake_llm, new_thread_id):
    fake_llm.program_structured(
        RouterDecision,
        RouterDecision(intent=QueryIntent.LEGAL_CONSULTATION, reasoning="consulta general"),
    )
    fake_llm.program_structured(
        LegalAnswerPayload,
        LegalAnswerPayload(
            answer="Respuesta legal persistida.",
            confidence=ConfidenceLevel.HIGH,
            articles_referenced=["LEY-26994:1198"],
        ),
    )

    try:
        pool_a = await build_checkpointer_pool(test_settings)
        checkpointer_a = build_checkpointer(pool_a)
        await checkpointer_a.setup()
    except Exception as exc:  # noqa: BLE001 - Salta test si PostgreSQL no está disponible
        pytest.skip(f"Postgres/Checkpointer no disponible ({exc}) -- levantar 'docker compose up -d postgres'")

    graph_a = build_graph(checkpointer_a)
    config = {
        "configurable": {
            "thread_id": new_thread_id,
            "llm_client": fake_llm,
            "legal_retriever": None,
            "legalize_api_client": None,
            "settings": test_settings,
        },
        "recursion_limit": test_settings.GRAPH_RECURSION_LIMIT,
    }

    initial_state: LegalAgentState = {
        "messages": [HumanMessage(content="¿Cuál es el plazo de locación?")],
        "thread_id": new_thread_id,
        "query": "¿Cuál es el plazo de locación?",
        "intent": QueryIntent.LEGAL_CONSULTATION,
        "citations": [],
        "draft_answer": None,
        "validation_result": None,
        "document_text": None,
        "document_type": None,
        "diff_request_params": None,
        "diff_result": None,
        "final_answer": None,
        "confidence": ConfidenceLevel.HIGH,
        "iteration": 0,
    }

    await graph_a.ainvoke(initial_state, config=config)
    await pool_a.close()

    # "Reinicio": pool y checkpointer nuevos, sin ningún objeto compartido
    pool_b = await build_checkpointer_pool(test_settings)
    try:
        checkpointer_b = build_checkpointer(pool_b)
        graph_b = build_graph(checkpointer_b)

        snapshot = await graph_b.aget_state({"configurable": {"thread_id": new_thread_id}})
    finally:
        await pool_b.close()

    assert snapshot.values["query"] == "¿Cuál es el plazo de locación?"
    assert snapshot.values["thread_id"] == new_thread_id
    assert len(snapshot.values["messages"]) >= 1
