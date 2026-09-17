"""Router inteligente para clasificar consultas jurídicas hacia el nodo de ejecución adecuado."""

import logging
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agents.legal.state import LegalAgentState
from app.core.structured_output import invoke_structured_with_retry
from app.domain.models import QueryIntent
from app.services.legal.web_reader import fetch_web_page_content, find_urls_in_text

logger = logging.getLogger(__name__)


class RouterDecision(BaseModel):
    intent: QueryIntent = Field(description="Intención clasificada de la consulta del usuario")
    law_identifier_hint: str | None = Field(default=None, description="Norma mencionada explícitamente (ej: 'LEY-26994')")
    article_hint: str | None = Field(default=None, description="Artículo mencionado (ej: '1198')")
    reasoning: str = Field(description="Explicación concisa de la decisión de ruteo")


ROUTER_PROMPT = """Eres el clasificador de intenciones del Agente Legal Argentino.
Tu objetivo es determinar qué tipo de tarea jurídica solicita el usuario:
- 'legal_consultation': Pregunta sustantiva, doctrinaria, interpretativa o consulta sobre leyes argentinas y su régimen actual (ej: "¿Cómo quedaron regulados los alquileres tras el DNU 70/2023?", "¿Qué depósitos o plazos rigen?", "¿Es válido este plazo de preaviso?"). Toda pregunta que indague cómo se regula una situación o materia jurídica, INCLUSO si menciona un DNU o reforma histórica ("tras el DNU 70/2023", "a partir de la Ley de Bases"), es 'legal_consultation' porque requiere aplicar el derecho positivo de fondo (ej CCyC, LCT).
- 'version_diff': ÚNICAMENTE cuando el usuario pide de forma EXPLÍCITA una comparativa textual de redacciones, un diff de reformas o contrastar la redacción previa vs la actual de un artículo o ley (ej: "¿Qué modificó la reforma en LEY-24013 artículo 153?", "mostrame el diff del artículo 92 ter de la LCT", "redacción anterior vs vigente"). NO uses version_diff para preguntas generales sobre cómo quedó regulado un contrato o régimen.
- 'document_analysis': El usuario proporciona un texto de contrato, convenio, carta documento o pide analizar cláusulas.
- 'url_fact_check': El usuario incluye un enlace o link web (noticia, publicación) para contrastar su veracidad con la ley.
- 'general_inquiry': Preguntas generales, saludos o consultas no normativas.

Identificadores de normas canónicas en el repositorio:
- Constitución Nacional de la República Argentina -> LEY-24430
- Código Civil y Comercial de la Nación (CCyC) -> LEY-26994
- Ley de Contrato de Trabajo (LCT / T.O. 1976 vigente con Art. 245) -> DEC-390-1976 (o LEY-20744)
- Ley General de Sociedades -> LEY-19550
- Ley de Defensa del Consumidor -> LEY-24240
- Código Penal de la Nación -> LEY-11179
- DNU 70/2023 (Bases para la Reconstrucción) -> DNU-70-2023
- Ley de Bases -> LEY-27742
- Ley de Riesgos del Trabajo (ART) -> LEY-24557
- Ley de Empleo -> LEY-24013
- Protección de Datos Personales -> LEY-25326
"""


async def router_node(state: LegalAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Clasifica la consulta, detecta enlaces web para fact-checking e indicios de leyes."""
    configurable = config.get("configurable", {})
    llm = configurable.get("llm_client")
    settings = configurable.get("settings")

    # 1. Si ya se especificó document_text en el estado (ej: endpoint /analyze), es DOCUMENT_ANALYSIS
    if state.get("document_text"):
        return {"intent": QueryIntent.DOCUMENT_ANALYSIS}

    # 2. Detección automática de enlaces web (URL Fact-Checking)
    urls = find_urls_in_text(state["query"])
    if urls:
        target_url = urls[0]
        logger.info("Detectado enlace web en la consulta: %s", target_url)
        web_res = await fetch_web_page_content(target_url, timeout_seconds=4.0)

        if web_res.success:
            doc_text = (
                f"URL FUENTE: {web_res.url} (Dominio: {web_res.domain})\n"
                f"TÍTULO DE LA PUBLICACIÓN: {web_res.title or 'Sin título'}\n\n"
                f"CONTENIDO EXTRAÍDO:\n{web_res.clean_text}"
            )
            return {
                "intent": QueryIntent.URL_FACT_CHECK,
                "document_text": doc_text,
                "document_type": f"enlace_web ({web_res.domain})",
            }
        else:
            fallback_text = (
                f"AVISO SOBRE EL ENLACE ({target_url}):\n{web_res.error}\n\n"
                "Instrucción: No fue posible acceder al contenido del enlace debido a la restricción indicada. "
                "Responde la consulta jurídica planteada por el usuario en su texto, aclarando de forma concisa "
                "que no se pudo leer el contenido del link provisto."
            )
            return {
                "intent": QueryIntent.LEGAL_CONSULTATION,
                "document_text": fallback_text,
                "document_type": "enlace_no_accesible",
            }

    if not llm:
        # Fallback determinista por palabras clave
        q = state["query"].lower()
        if "qué cambió" in q or "que cambio" in q or "reforma" in q or "redacción anterior" in q:
            return {"intent": QueryIntent.VERSION_DIFF}
        return {"intent": QueryIntent.LEGAL_CONSULTATION}

    try:
        messages = [
            SystemMessage(content=ROUTER_PROMPT),
            *state["messages"],
        ]
        decision: RouterDecision = await invoke_structured_with_retry(
            llm, RouterDecision, ROUTER_PROMPT, messages, max_attempts=settings.STRUCTURED_OUTPUT_MAX_ATTEMPTS if settings else 2
        )
        diff_params = dict(state.get("diff_request_params") or {})
        if decision.law_identifier_hint:
            diff_params["law_identifier"] = decision.law_identifier_hint
        if decision.article_hint:
            diff_params["article_number"] = decision.article_hint

        return {
            "intent": decision.intent,
            "diff_request_params": diff_params or None,
        }
    except Exception as exc:
        logger.warning("Fallo en router estructurado, usando fallback: %s", exc)
        return {"intent": QueryIntent.LEGAL_CONSULTATION}

