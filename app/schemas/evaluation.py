from pydantic import BaseModel, Field


class GoldenSetItem(BaseModel):
    question: str
    expected_source: str | list[str] | None = Field(
        default=None, description="Identificador de la norma esperada (ej: 'LEY-26994')"
    )
    expected_article: str | None = Field(
        default=None, description="Artículo específico esperado (ej: '1198')"
    )
    expected_answer_contains: str | None = Field(
        default=None, description="Fragmento que la respuesta debería mencionar, si aplica"
    )



class EvaluationJudgment(BaseModel):
    """Salida estructurada del LLM actuando de juez (RAG Triad: faithfulness + answer relevance)."""

    faithfulness: float = Field(
        ge=0, le=1, description="Que tan fundamentada esta la respuesta en el contexto recuperado"
    )
    faithfulness_reasoning: str
    relevance: float = Field(ge=0, le=1, description="Que tan bien la respuesta contesta la pregunta original")
    relevance_reasoning: str


class EvaluationResult(BaseModel):
    question: str
    answer: str
    sources_used: list[str]
    expected_source: str | list[str] | None = None
    faithfulness: float
    relevance: float
    passed: bool
