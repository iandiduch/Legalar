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
2. DESPIDO E INDEMNIZACIONES LABORALES (Ley de Contrato de Trabajo - Ley 20.744 / T.O. Dec. 390/1976):
   - Requiere el régimen indemnizatorio del despido incausado (Art. 245 LCT), preaviso (Arts. 231/232 LCT), integración (Art. 233 LCT) y prescripción bienal (Art. 256 LCT).
   - Canónicos: ['DEC-390-1976:245', 'LEY-20744:245', 'DEC-390-1976:232', 'DEC-390-1976:256'].
3. CONTRATOS, LOCACIONES Y REFORMA DNU 70/2023 (Código Civil y Comercial / Ley 27.551 / DNU 70/2023):
   - Art. 1198 CCyC (plazo de locación acordado por las partes o 2 años para destino habitacional), Art. 256 DNU 70/2023 (sustitución de Art. 1198 CCyC), Art. 249 DNU 70/2023 (derogación expresa de la Ley 27.551).
   - Canónicos: ['LEY-26994:1198', 'DNU-70-2023:256', 'DNU-70-2023:249', 'LEY-27551:3'].
4. PRESCRIPCIÓN CIVIL Y COMERCIAL:
   - Art. 2560 CCyC (plazo genérico de 5 años) o plazos especiales (Arts. 2561 a 2564 CCyC) + causales de suspensión/interrupción (Arts. 2539 a 2549 CCyC).
   - Canónicos: ['LEY-26994:2560', 'LEY-26994:2554'].
5. GARANTÍA LEGAL DE COSAS MUEBLES NO CONSUMIBLES (Ley de Defensa del Consumidor - Ley 24.240):
   - Art. 11 Ley 24.240 (garantía legal de 3 meses para cosas usadas y 6 meses para cosas nuevas no consumibles).
   - Canónicos: ['LEY-24240:11'].
6. TUTELA Y PAGO DE REMUNERACIÓN LABORAL (LCT):
   - Art. 105 LCT (formas de remuneración), Art. 107 LCT (tope máximo del 20% para pago en especie), Art. 124 LCT (pago en dinero y prohibición de mercaderías).
   - Canónicos: ['DEC-390-1976:105', 'DEC-390-1976:107', 'DEC-390-1976:124', 'LEY-20744:105'].
7. DERECHO DE MARCAS Y OPOSICIONES ANTE EL INPI (Ley de Marcas y Designaciones - Ley 22.362):
   - Confundibilidad y prohibición de registro: Art. 3 inc. a y b (marcas idénticas o similares para los mismos productos o servicios, o susceptibles de producir confusión en el público consumidor).
   - Procedimiento de oposiciones: Arts. 12 a 17 (notificación de la oposición, plazo legal de 3 meses para negociación o levantamiento, instancia de resolución administrativa de oposiciones en el INPI).
   - Canónicos: ['LEY-22362:3', 'LEY-22362:14', 'LEY-22362:15', 'LEY-22362:16'].

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
