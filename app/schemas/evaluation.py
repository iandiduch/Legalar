from pydantic import BaseModel, Field


class GoldenSetItem(BaseModel):
    question: str
    expected_source: str | list[str] | None = Field(
        default=None, description="Identificador(es) de la norma esperada (ej: 'LEY-26994' o ['LEY-20744', 'DEC-390-1976'])"
    )
    expected_article: str | list[str] | None = Field(
        default=None, description="Artículo(s) específico(s) esperado(s) (ej: '1198' o ['1198', '256'])"
    )
    expected_answer_contains: str | list[str] | None = Field(
        default=None, description="Fragmento(s) o conceptos clave que la respuesta debería mencionar"
    )



class EvaluationJudgment(BaseModel):
    """Salida estructurada del LLM actuando de juez (RAG Triad: faithfulness + answer relevance)."""

    faithfulness: float = Field(
        ge=0, le=1, description="Que tan fundamentada esta la respuesta en el contexto recuperado o normativa positiva"
    )
    faithfulness_reasoning: str
    relevance: float = Field(ge=0, le=1, description="Que tan bien la respuesta contesta la pregunta original")
    relevance_reasoning: str


class EvaluationResult(BaseModel):
    question: str
    answer: str
    sources_used: list[str]
    expected_source: str | list[str] | None = None
    expected_article: str | list[str] | None = None
    source_recalled: bool = True
    article_recalled: bool = True
    contains_expected: bool = True
    faithfulness: float
    relevance: float
    passed: bool

