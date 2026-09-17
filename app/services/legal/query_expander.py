"""Analizador y Expansor de Consultas Jurídicas para RAG Legal Relacional (Multi-Norma).

Detecta cuando una consulta requiere articular una figura jurídica específica (delito, contrato, despido)
con una regla o instituto general (cómputo de prescripción, indemnización, nulidad, inicio de plazos,
interrupción/suspensión), generando sub-consultas dirigidas y referencias a normas canónicas.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.core.structured_output import invoke_structured_with_retry

logger = logging.getLogger(__name__)


class LegalRelationAnalysis(BaseModel):
    question_type: str = Field(
        default="general",
        description="Tipo de relación jurídica: prescription, compensation, nullity, eviction, employment_termination, procedural_rights, general",
    )
    is_relational: bool = Field(
        default=False,
        description="True si la consulta requiere combinar dos o más normas o capítulos distintos (ej: tipo penal específico + regla general de prescripción)",
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Lista de 2 o 3 sub-queries complementarias optimizadas para búsqueda vectorial y FTS",
    )
    canonical_articles: list[str] = Field(
        default_factory=list,
        description="Identificadores de normas y artículos canónicos clave si son conocidos con formato LEY-XXXX:NUM (ej: ['LEY-11179:62', 'LEY-11179:172', 'DEC-390-1976:245'])",
    )
    reasoning: str = Field(
        default="",
        description="Breve justificación jurídica de por qué se requieren esas normas complementarias",
    )


LEGAL_EXPANDER_PROMPT = """Eres el Analizador y Expansor Jurídico del Sistema Legal Argentino.
Tu tarea es analizar la consulta legal y determinar si, para responder con estricto rigor dogmático,
se requiere la ARTICULACIÓN DE MÚLTIPLES NORMAS que no se encuentran en un mismo artículo ni capítulo
(por ejemplo, la combinación de una Parte Especial con una Parte General de un código).

Casos paradigmáticos en el derecho argentino:
1. PRESCRIPCIÓN PENAL (ej: "¿cuándo prescribe el fraude/estafa?", "prescripción de homicidio"):
   - Requiere la figura penal específica para determinar la escala penal (ej: Art. 172 CP para estafa, Art. 79 CP para homicidio).
   - Y las reglas dogmáticas generales: Art. 62 CP (cómputo según el máximo de la pena fijada), Art. 63 CP (inicio del cómputo), Art. 67 CP (suspensión e interrupción).
   - Canónicos: ['LEY-11179:62', 'LEY-11179:63', 'LEY-11179:67', 'LEY-11179:172'] (según el delito).
2. DESPIDO E INDEMNIZACIONES LABORALES (LCT):
   - Requiere el cálculo de antigüedad (Art. 245 LCT), preaviso (Arts. 231/232 LCT), integración (Art. 233 LCT) y prescripción bienal (Art. 256 LCT).
   - Canónicos: ['DEC-390-1976:245', 'DEC-390-1976:232', 'DEC-390-1976:256'].
3. CONTRATOS Y LOCACIÓN (Código Civil y Comercial - Ley 26.994):
   - Art. 1198 (plazo mínimo), Art. 1219 (resolución imputable), Art. 1221 (resolución anticipada), Art. 1222 (intimación previa al desalojo).
   - Canónicos: ['LEY-26994:1198', 'LEY-26994:1221', 'LEY-26994:1222'].
4. PRESCRIPCIÓN CIVIL Y COMERCIAL:
   - Art. 2560 CCyC (plazo genérico de 5 años) o plazos especiales (Arts. 2561 a 2564 CCyC) + causales de suspensión/interrupción (Arts. 2539 a 2549 CCyC).

Instrucciones de clasificación:
- Si la consulta es puntual, unívoca o no requiere articular reglas generales con figuras especiales:
  * is_relational = False
  * sub_queries = [consulta_original]
  * canonical_articles = []
- Si la consulta requiere articular normas dogmáticas separadas:
  * is_relational = True
  * Genera 2 o 3 sub_queries enfocadas:
    - Sub-query 1: Dirigida al hecho/delito/contrato específico y su pena o régimen.
    - Sub-query 2: Dirigida a la regla general de cómputo, plazo o instituto dogmático.
    - Sub-query 3 (si aplica): Dirigida a causales de inicio, suspensión o interrupción.
  * Incluye en canonical_articles los identificadores canónicos reconocidos (ej: 'LEY-11179:62').
"""


async def expand_legal_query(
    llm: Any,
    query: str,
    conversation_context: list | None = None,
    max_attempts: int = 2,
) -> LegalRelationAnalysis:
    """Analiza y descompone la consulta legal en sub-queries relacionales y artículos canónicos.

    Si el LLM no está configurado o la invocación falla, devuelve un fallback seguro
    con la consulta original sin alterar el flujo del agente.
    """
    if not llm or not query.strip():
        return LegalRelationAnalysis(
            question_type="general",
            is_relational=False,
            sub_queries=[query.strip()] if query.strip() else [],
            canonical_articles=[],
            reasoning="Fallback determinista por ausencia de LLM o query vacía",
        )

    context_messages = []
    if conversation_context:
        # Últimos 3 mensajes para contextualizar referencias elípticas
        context_messages = conversation_context[-3:]

    messages = [
        SystemMessage(content=LEGAL_EXPANDER_PROMPT),
        *context_messages,
        HumanMessage(content=f"CONSULTA JURÍDICA A ANALIZAR:\n{query}"),
    ]

    try:
        analysis: LegalRelationAnalysis = await invoke_structured_with_retry(
            llm,
            LegalRelationAnalysis,
            LEGAL_EXPANDER_PROMPT,
            messages,
            max_attempts=max_attempts,
        )
        if not analysis.sub_queries:
            analysis.sub_queries = [query]

        logger.info(
            "expand_legal_query: type=%s, relational=%s, sub_queries=%d, canonical=%s",
            analysis.question_type,
            analysis.is_relational,
            len(analysis.sub_queries),
            analysis.canonical_articles,
        )
        return analysis
    except Exception as exc:
        logger.warning("Fallo en expand_legal_query, usando fallback con query original: %s", exc)
        return LegalRelationAnalysis(
            question_type="general",
            is_relational=False,
            sub_queries=[query],
            canonical_articles=[],
            reasoning="Fallback ante fallo en LLM",
        )
