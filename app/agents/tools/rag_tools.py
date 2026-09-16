"""Tool de búsqueda jurídica para agentes o flujos de tool calling.

Se construye vía factory para inyectar el retriever híbrido legal.
"""

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.services.legal.retriever import HybridLegalRetriever


class BuscarLegislacionInput(BaseModel):
    query: str = Field(..., description="Pregunta o concepto jurídico a buscar en la legislación argentina")
    top_k: int = Field(default=5, ge=1, le=20)


def build_rag_tools(
    retriever: HybridLegalRetriever,
) -> list[BaseTool]:
    @tool("buscar_legislacion_argentina", args_schema=BuscarLegislacionInput)
    async def buscar_legislacion_argentina(query: str, top_k: int = 5) -> list[dict]:
        """Busca en el corpus de leyes argentinas (CCyC, LCT, Ley de Sociedades, etc.)
        y devuelve los artículos vigentes más relevantes junto con su cita jurídica."""
        citations = await retriever.retrieve_citations(query, top_k=top_k)
        return [c.model_dump(mode="json") for c in citations]

    return [buscar_legislacion_argentina]
