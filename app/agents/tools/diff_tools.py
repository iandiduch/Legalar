"""Tool modular de comparación de reformas y control de versiones normativas.

Sigue la arquitectura del sistema implementando BaseTool de LangChain, esquemas
Pydantic para validación de argumentos y delegación en el cliente oficial LegalizeApiClient
con conmutación transparente a LocalGitDiffEngine.
"""

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.services.legal.legalize_api_client import LegalizeApiClient


class CompararNormasInput(BaseModel):
    """Esquema de entrada validado para la herramienta de comparación de reformas."""

    law_identifier: str = Field(
        ...,
        description="Identificador oficial de la ley o decreto (ej: 'LEY-20744', 'LEY-26994', 'DNU-70-2023', 'LEY-19550')",
    )
    article_number: str | None = Field(
        default=None,
        description="Número o referencia del artículo específico a contrastar (ej: '92 ter', '1198', '14')",
    )
    date_a: str | None = Field(
        default=None,
        description="Fecha histórica inicial en formato YYYY-MM-DD. Si no se indica, se deduce de la reforma previa.",
    )
    date_b: str | None = Field(
        default=None,
        description="Fecha histórica posterior en formato YYYY-MM-DD. Si no se indica, se deduce de la reforma vigente.",
    )


def build_diff_tools(api_client: LegalizeApiClient) -> list[BaseTool]:
    """Factory para instanciar las herramientas de diff normativo del agente legal."""

    @tool("comparar_reformas_normativas", args_schema=CompararNormasInput)
    async def comparar_reformas_normativas(
        law_identifier: str,
        article_number: str | None = None,
        date_a: str | None = None,
        date_b: str | None = None,
    ) -> dict:
        """Compara la evolución textual de una ley o artículo entre dos momentos históricos.

        Consulta el historial oficial de reformas vía Legalize API o conmuta automáticamente
        al motor Git local si la cuota mensual se agotó o la norma se encuentra versionada localmente.
        Devuelve el diff unificado, la fuente de procedencia (api o git_local) y el análisis de cambios.
        """
        diff_resp = await api_client.get_diff(
            law_identifier=law_identifier,
            date_a=date_a,
            date_b=date_b,
            article=article_number,
        )
        return diff_resp.model_dump(mode="json")

    return [comparar_reformas_normativas]
