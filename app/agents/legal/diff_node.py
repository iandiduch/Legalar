import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agents.legal.state import LegalAgentState
from app.domain.models import AgentRole, ConfidenceLevel
from app.services.legal.legalize_api_client import LegalizeApiClient

logger = logging.getLogger(__name__)

DIFF_ANALYSIS_PROMPT = """Eres el Especialista en Historia y Reformas Legislativas del Derecho Argentino.
Tu función es analizar el diff textual entre dos versiones de una ley o artículo y explicar con claridad:
1. Qué texto fue suprimido y qué texto fue incorporado.
2. Cuál es el impacto práctico y jurídico de la reforma para los ciudadanos y abogados.
3. Qué norma introdujo la modificación (ej: DNU 70/2023, Ley 27.551, etc.).

REGLAS ESTRICTAS DE FORMATO Y ESTILO:
- Responde de forma directa, analítica y técnica en formato Markdown estructurado con subtítulos o viñetas.
- PROHIBIDO usar saludos de carta o correo (NO uses "Estimado/a colega", "Hola", "Buenos días", etc.). Ve directamente a la información.
- PROHIBIDO usar firmas, despedidas o cierres epistolares (NO uses "Atentamente", "Quedo a su disposición", "Especialista en...", etc.). No finjas ser una persona física firmando una carta.
- Sé conciso y profesional (máximo 400 palabras).
- NUNCA reproduzcas el diff completo ni repitas notas al pie o encabezados de forma cíclica. Sintetiza los cambios normativos y su impacto real.
"""


async def diff_node(state: LegalAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Obtiene el diff normativo (vía API o Git local) y genera la explicación jurídica."""
    configurable = config.get("configurable", {})
    llm = configurable.get("llm_client")
    api_client: LegalizeApiClient | None = configurable.get("legalize_api_client")

    params = state.get("diff_request_params") or {}
    law_id = params.get("law_identifier", "LEY-26994")
    art_num = params.get("article_number")
    date_a = params.get("date_a")
    date_b = params.get("date_b")

    diff_response = None
    if api_client:
        try:
            diff_response = await api_client.get_diff(
                law_identifier=law_id,
                date_a=date_a,
                date_b=date_b,
                article=art_num,
            )
        except Exception as exc:
            logger.error("Error al obtener diff: %s", exc)

    source_info = f"Fuente del diff: {diff_response.diff_source if diff_response else 'Desconocida'}"
    diff_content = diff_response.diff_text if diff_response else "No se pudo recuperar el diff de la norma."

    # ATAJO ULTRA-RÁPIDO: Si el motor de diff detectó que no hay cambios textuales, responder al instante sin llamar al LLM
    has_modifications = False
    if diff_response:
        if diff_response.article_diffs:
            has_modifications = any(d.has_changes for d in diff_response.article_diffs)
        elif diff_response.diff_text and "Sin diferencias" not in diff_response.diff_text and "idéntico" not in diff_response.diff_text.lower():
            has_modifications = True

    if not has_modifications:
        target_ref = f"el artículo {art_num} de la norma {law_id}" if art_num else f"la norma {law_id}"
        explanation = (
            f"### Análisis de Modificaciones: {law_id}" + (f" (Art. {art_num})" if art_num else "") + "\n\n"
            f"- **Resultado del cotejo**: No se registran modificaciones textuales en {target_ref} entre las versiones consultadas en el repositorio.\n"
            f"- **Estado de redacción**: El texto normativo oficial se mantiene idéntico en su redacción registrada.\n"
            f"- **Fuente de verificación**: {source_info}."
        )
        return {
            "diff_result": diff_response,
            "draft_answer": explanation,
            "confidence": ConfidenceLevel.HIGH,
            "messages": [AIMessage(content=explanation, name="diff_node")],
        }

    # Si hubo modificaciones, solicitar análisis sintetizado al LLM con timeout de 20s
    if len(diff_content) > 12000:
        diff_content = diff_content[:12000] + "\n\n... [diff truncado por extensión para análisis conciso]"

    messages = [
        SystemMessage(content=DIFF_ANALYSIS_PROMPT),
        SystemMessage(content=f"CONSULTA DEL USUARIO: {state['query']}\n\nDIFF ({source_info}):\n{diff_content}"),
    ]

    try:
        res = await asyncio.wait_for(llm.ainvoke(messages), timeout=20.0)
        explanation = str(res.content)
    except Exception as exc:
        logger.warning("Fallo al generar explicación de diff con LLM: %s", exc)
        explanation = f"### Comparativa Normativa ({source_info})\n\n```diff\n{diff_content}\n```"

    return {
        "diff_result": diff_response,
        "draft_answer": explanation,
        "confidence": ConfidenceLevel.HIGH,
        "messages": [AIMessage(content=explanation, name="diff_node")],
    }
