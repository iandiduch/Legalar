"""Harness LLM-as-judge (RAG Triad: faithfulness + answer relevance + ground-truth recall)
contra data/golden_set.json para el Agente Legal Argentino.

Se corre para verificar la calidad del pipeline RAG y la fidelidad jurídica:
    python -m scripts.evaluate_rag

Evalúa el comportamiento real del Agente Legal con el HybridLegalRetriever
(PostgreSQL FTS + Pinecone) y el grafo completo de decisión y validación.
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from app.agents.legal.graph import build_legal_graph
from app.agents.legal.state import LegalAgentState
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.structured_output import invoke_structured_with_retry
from app.db.session import build_engine, build_sessionmaker
from app.domain.models import ConfidenceLevel, QueryIntent
from app.schemas.evaluation import EvaluationJudgment, EvaluationResult, GoldenSetItem
from app.services.legal.legalize_api_client import LegalizeApiClient
from app.services.legal.local_diff_engine import LocalGitDiffEngine
from app.services.legal.retriever import HybridLegalRetriever
from app.services.llm_factory import build_chat_model
from app.services.rag_service import build_embeddings_client, build_pinecone_client

logger = logging.getLogger(__name__)

_GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_set.json"
_RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "evaluation_results.json"

_JUDGE_SYSTEM_PROMPT = """Sos un evaluador jurídico imparcial de un sistema de Inteligencia Artificial para Derecho Argentino.

Te dan una consulta jurídica, la evidencia normativa recuperada de las leyes y códigos vigentes, y la respuesta final generada por el agente.

Tu tarea es calificar objetivamente dos métricas entre 0.0 y 1.0:
  1. faithfulness: ¿La respuesta afirma cuestiones respaldadas por la evidencia jurídica o por la normativa positiva argentina, sin inventar artículos ni atribuir vigencia a leyes derogadas?
     - Si la consulta es ajena al ámbito legal argentino (ej: pasajes de avión, recetas, cotizaciones comerciales) y el asistente declina respetuosamente responder por exceder su especialidad jurídica sin inventar leyes, califica faithfulness=1.0.
     - Si el asistente fundamenta en normas vigentes acordes a la ley argentina aplicable y no inventa hechos ni artículos, califica faithfulness=1.0.
  2. relevance: ¿La respuesta aborda de forma directa, certera y útil la consulta jurídica del usuario?
     - Si la consulta es ajena al ámbito jurídico y el asistente aclara su rol y límites de cobertura, califica relevance=1.0.
     - Si responde con el régimen legal aplicable a lo consultado de manera precisa, califica relevance=1.0.

Sé riguroso y objetivo. Justifica brevemente cada puntaje."""


def _load_golden_set() -> list[GoldenSetItem]:
    if not _GOLDEN_SET_PATH.exists():
        return []
    raw = json.loads(_GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    return [GoldenSetItem(**item) for item in raw]


def _normalize_law_id(raw: str) -> list[str]:
    """Genera variantes comunes de un identificador de norma (ej: 'LEY-26994' -> 'LEY-26994', 'LEY 26.994', '26994')."""
    clean = raw.upper().strip()
    variants = [clean]
    if "-" in clean:
        parts = clean.split("-")
        variants.append(" ".join(parts))
        if len(parts) == 3:
            variants.append(f"{parts[0]} {parts[1]}/{parts[2]}")
            variants.append(f"{parts[1]}/{parts[2]}")
        elif len(parts) == 2 and parts[1].isdigit():
            num = parts[1]
            variants.append(num)
            if len(num) > 3:
                variants.append(f"{num[:-3]}.{num[-3:]}")
    return variants


def _evaluate_ground_truth(
    item: GoldenSetItem,
    answer: str,
    sources_used: list[str],
) -> tuple[bool, bool, bool]:
    """Evalúa source_recalled, article_recalled y contains_expected contra el ground truth."""
    # 1. Source recall con soporte para variantes de citación (ej: 'Ley 26.994' o 'LEY-26994')
    if not item.expected_source:
        source_recalled = True
    else:
        exp_sources = [item.expected_source] if isinstance(item.expected_source, str) else item.expected_source
        all_variants: list[str] = []
        for s in exp_sources:
            all_variants.extend(_normalize_law_id(s))

        sources_str = " ".join(sources_used).upper()
        answer_upper = answer.upper()
        source_recalled = any(
            v in sources_str or v in answer_upper
            for v in all_variants
        )

    # 2. Article recall
    if not item.expected_article:
        article_recalled = True
    else:
        exp_articles = [item.expected_article] if isinstance(item.expected_article, str) else item.expected_article
        exp_articles_clean = [str(a).lower().strip() for a in exp_articles]
        answer_lower = answer.lower()
        sources_lower = " ".join(sources_used).lower()
        article_recalled = any(
            f"art. {art}" in answer_lower
            or f"artículo {art}" in answer_lower
            or f"articulo {art}" in answer_lower
            or f"art {art}" in answer_lower
            or f"art. {art}" in sources_lower
            or f":{art}" in sources_lower
            or f" {art} " in answer_lower
            for art in exp_articles_clean
        )

    # 3. Contains expected (keywords / concepts)
    if not item.expected_answer_contains:
        contains_expected = True
    else:
        exp_phrases = (
            [item.expected_answer_contains]
            if isinstance(item.expected_answer_contains, str)
            else item.expected_answer_contains
        )
        answer_lower = answer.lower()
        contains_expected = any(
            phrase.lower().strip() in answer_lower
            for phrase in exp_phrases
        )

    return source_recalled, article_recalled, contains_expected


async def _judge(
    question: str, context_block: str, answer: str, llm: Any, settings: Settings
) -> EvaluationJudgment:
    judge_input = f"Consulta jurídica: {question}\n\nNormativa recuperada:\n{context_block}\n\nRespuesta generada:\n{answer}"
    return await invoke_structured_with_retry(
        llm,
        EvaluationJudgment,
        _JUDGE_SYSTEM_PROMPT,
        [HumanMessage(content=judge_input)],
        settings.STRUCTURED_OUTPUT_MAX_ATTEMPTS,
    )


async def main() -> None:
    settings = get_settings()
    configure_logging(settings)

    golden_set = _load_golden_set()
    if not golden_set:
        logger.warning("evaluate_rag.empty_golden_set")
        print(f"No hay preguntas en {_GOLDEN_SET_PATH}.")
        return

    engine = build_engine(settings)
    sessionmaker = build_sessionmaker(engine)
    pinecone_client = build_pinecone_client(settings)
    embeddings_client = build_embeddings_client(settings)

    retriever = HybridLegalRetriever(
        settings=settings,
        pinecone_client=pinecone_client,
        embeddings_client=embeddings_client,
        sessionmaker=sessionmaker,
    )
    diff_engine = LocalGitDiffEngine(repo_path=settings.LEGALIZE_REPO_PATH)
    diff_client = LegalizeApiClient(settings=settings, local_diff_engine=diff_engine)
    llm = build_chat_model(settings)

    app_graph = build_legal_graph(checkpointer=None)

    print("=================================================================")
    print("      EVALUACIÓN RAG TRIAD · AGENTE LEGAL ARGENTINO              ")
    print("=================================================================")
    print(f"Preguntas en Golden Set: {len(golden_set)}")
    print(f"Umbral de aprobación:   {settings.EVALUATION_PASS_THRESHOLD}")
    print(f"Top-K de Retrieval:     {getattr(settings, 'RAG_TOP_K', 8)}")
    print("-----------------------------------------------------------------")

    try:
        results: list[EvaluationResult] = []
        for idx, item in enumerate(golden_set, start=1):
            initial_state: LegalAgentState = {
                "messages": [HumanMessage(content=item.question)],
                "thread_id": str(uuid.uuid4()),
                "query": item.question,
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

            config = {
                "configurable": {
                    "llm_client": llm,
                    "legal_retriever": retriever,
                    "legalize_api_client": diff_client,
                    "settings": settings,
                }
            }

            final_state = await app_graph.ainvoke(initial_state, config=config)
            answer = final_state.get("final_answer") or final_state.get("draft_answer") or "Sin respuesta"
            citations = final_state.get("citations", [])

            sources = [f"{c.law_identifier} Art. {c.article_number}" for c in citations]
            context_block = "\n\n".join(
                [f"[{c.law_identifier} Art. {c.article_number}]: {c.exact_quote}" for c in citations]
            ) or "(Sin evidencia recuperada)"

            judgment = await _judge(item.question, context_block, answer, llm, settings)

            src_rec, art_rec, contains_exp = _evaluate_ground_truth(item, answer, sources)

            triad_pass = (
                judgment.faithfulness >= settings.EVALUATION_PASS_THRESHOLD
                and judgment.relevance >= settings.EVALUATION_PASS_THRESHOLD
            )
            passed = triad_pass and src_rec and art_rec and contains_exp

            results.append(
                EvaluationResult(
                    question=item.question,
                    answer=answer,
                    sources_used=sources,
                    expected_source=item.expected_source,
                    expected_article=item.expected_article,
                    source_recalled=src_rec,
                    article_recalled=art_rec,
                    contains_expected=contains_exp,
                    faithfulness=judgment.faithfulness,
                    relevance=judgment.relevance,
                    passed=passed,
                )
            )

            status_str = "PASS" if passed else "FAIL"
            print(
                f"[{idx:02d}/{len(golden_set):02d}] {status_str} | "
                f"Faithfulness: {judgment.faithfulness:.2f} | "
                f"Relevance: {judgment.relevance:.2f} | "
                f"SrcRec: {src_rec} | "
                f"ArtRec: {art_rec} | "
                f"{item.question[:45]}..."
            )

    finally:
        await pinecone_client.close()
        await engine.dispose()

    avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
    avg_relevance = sum(r.relevance for r in results) / len(results)
    source_recall_pct = (sum(1 for r in results if r.source_recalled) / len(results)) * 100
    article_recall_pct = (sum(1 for r in results if r.article_recalled) / len(results)) * 100
    contains_pct = (sum(1 for r in results if r.contains_expected) / len(results)) * 100
    passed_count = sum(1 for r in results if r.passed)

    print("=================================================================")
    print("                    RESUMEN DE EVALUACIÓN                        ")
    print("=================================================================")
    print(f"Preguntas aprobadas:    {passed_count}/{len(results)} ({(passed_count/len(results))*100:.1f}%)")
    print(f"Faithfulness promedio:  {avg_faithfulness:.2f}")
    print(f"Relevance promedio:     {avg_relevance:.2f}")
    print(f"Source Recall:          {source_recall_pct:.1f}%")
    print(f"Article Recall:         {article_recall_pct:.1f}%")
    print(f"Concept Match:          {contains_pct:.1f}%")
    print("=================================================================")

    _RESULTS_PATH.write_text(
        json.dumps([r.model_dump() for r in results], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Resultados detallados guardados en {_RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

