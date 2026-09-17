import asyncio
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agents.legal.state import LegalAgentState
from app.domain.models import ConfidenceLevel
from app.schemas.legal.diff import DiffResponse
from app.services.legal.diff_parser import build_diff_block_data
from app.services.legal.legalize_api_client import LegalizeApiClient

logger = logging.getLogger(__name__)

DIFF_EXPLANATION_PROMPT = """Eres un jurista y docente de derecho argentino.
Tu función es explicar de manera breve, clara y en lenguaje ciudadano qué cambió en la práctica jurídica a partir de las siguientes líneas modificadas de un artículo normativo.

REGLAS ESTRICTAS:
1. Máximo 2 párrafos concisos (menos de 150 palabras).
2. Explica qué decía antes (líneas '-') y qué rige ahora (líneas '+').
3. NO uses saludos, ni despedidas, ni cartas. Ve directo a la explicación fáctica.
4. Basa tu explicación ÚNICAMENTE en las líneas provistas.
"""


def _extract_atomic_diff_chunk(unified_diff: str, max_lines: int = 40) -> str:
    """Aísla exclusivamente las líneas modificadas (+ y -) con su contexto inmediato.

    Evita enviar volcados masivos o normas completas al LLM (menos de 300 tokens).
    """

    lines = unified_diff.splitlines()
    filtered = []
    chunk_lines = 0
    for line in lines:
        if line.startswith("---") or line.startswith("+++"):
            continue
        filtered.append(line)
        chunk_lines += 1
        if chunk_lines >= max_lines:
            filtered.append("... [diff truncado a líneas principales de la modificación]")
            break
    return "\n".join(filtered)


async def diff_node(state: LegalAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Obtiene el diff normativo (vía API o Git local) y construye el bloque visual GitHub."""
    configurable = config.get("configurable", {})
    llm = configurable.get("llm_client")
    api_client: LegalizeApiClient | None = configurable.get("legalize_api_client")

    params = state.get("diff_request_params") or {}
    law_id = params.get("law_identifier", "LEY-26994")
    art_num = params.get("article_number")
    date_a = params.get("date_a")
    date_b = params.get("date_b")

    diff_response: DiffResponse | None = None
    if api_client:
        try:
            diff_response = await api_client.get_diff(
                law_identifier=law_id,
                date_a=date_a,
                date_b=date_b,
                article=art_num,
            )
        except Exception as exc:
            logger.error("Error al obtener diff para %s Art. %s: %s", law_id, art_num, exc)

    if not diff_response:
        diff_response = DiffResponse(
            law_identifier=law_id,
            law_title=law_id,
            article_number=art_num,
            diff_source="git_local",
            diff_text="No se pudo recuperar el diff de la norma en el repositorio.",
            analysis="No fue posible recuperar las versiones normativas.",
            article_diffs=[],
        )

    # Evaluar si hubo modificaciones textuales reales
    has_modifications = False
    unified_diff_text = diff_response.diff_text or ""
    if diff_response.article_diffs:
        has_modifications = any(d.has_changes for d in diff_response.article_diffs)
        if diff_response.article_diffs:
            unified_diff_text = diff_response.article_diffs[0].unified_diff
    elif unified_diff_text and "sin diferencias" not in unified_diff_text.lower() and "sin modificaciones" not in unified_diff_text.lower():
        has_modifications = True

    # REGLA 1: Si no hubo modificaciones, retorno instantáneo (< 0.2s) sin LLM
    if not has_modifications:
        diff_data = build_diff_block_data(
            diff_response=diff_response,
            citizen_explanation=None,
        )
        explanation = (
            f"### Cotejo Normativo Oficial: {law_id}" + (f" Art. {art_num}" if art_num else "") + "\n\n"
            f"- **Resultado del análisis**: No se encontraron modificaciones textuales en el artículo analizado.\n"
            f"- **Estado de redacción**: El texto oficial se mantiene idéntico en el registro normativo consolidado."
        )
        return {
            "diff_result": diff_response,
            "diff_data": diff_data,
            "draft_answer": explanation,
            "final_answer": explanation,
            "confidence": ConfidenceLevel.HIGH,
            "messages": [AIMessage(content=explanation, name="diff_node")],
        }

    # REGLA 2: Si el Router determinó que el usuario solo pide el diff técnico,
    # emitir el bloque visual de inmediato sin invocar al LLM (< 0.4s)
    query_text = state.get("query", "")
    wants_explanation = bool(params.get("wants_explanation", False))

    citizen_explanation = None
    if wants_explanation and llm:
        # REGLA 3: Si el usuario pidió explicación explícita, enviar ÚNICAMENTE las líneas
        # atómicas modificadas (~200 tokens) con timeout acotado (8s).
        atomic_diff = _extract_atomic_diff_chunk(unified_diff_text)
        prompt_content = (
            f"CONSULTA DEL USUARIO: {query_text}\n\n"
            f"NORMA: {law_id} (Artículo {art_num or 'general'})\n"
            f"LÍNEAS MODIFICADAS:\n```diff\n{atomic_diff}\n```"
        )
        try:
            res = await asyncio.wait_for(
                llm.ainvoke([
                    SystemMessage(content=DIFF_EXPLANATION_PROMPT),
                    SystemMessage(content=prompt_content),
                ]),
                timeout=8.0,
            )
            citizen_explanation = str(res.content).strip()
        except Exception as exc:
            logger.warning("No se pudo obtener explicación LLM para diff: %s", exc)
            citizen_explanation = "Se detectaron reformas en la redacción del artículo. Revise la comparativa visual línea por línea abajo."

    diff_data = build_diff_block_data(
        diff_response=diff_response,
        citizen_explanation=citizen_explanation,
    )

    summary_counts = diff_data.get("summary", {})
    adds = summary_counts.get("modificationsCount", 0)
    dels = summary_counts.get("deletionsCount", 0)

    final_text = citizen_explanation or (
        f"### Comparativa Normativa: {law_id} Art. {art_num or ''}\n\n"
        f"- **Líneas incorporadas (+)**: {adds}\n"
        f"- **Líneas suprimidas (-)**: {dels}\n"
        f"- **Fuente**: Registro oficial de legislación consolidada"
    )

    return {
        "diff_result": diff_response,
        "diff_data": diff_data,
        "draft_answer": final_text,
        "final_answer": final_text,
        "confidence": ConfidenceLevel.HIGH,
        "messages": [AIMessage(content=final_text, name="diff_node")],
    }

