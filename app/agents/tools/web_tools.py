"""Tool modular de extracción y análisis de enlaces web para verificación jurídica.

Sigue la arquitectura del sistema implementando BaseTool de LangChain, esquemas
Pydantic para validación de argumentos y delegación en el servicio de seguridad
web_reader (Anti-SSRF y Anti-Prompt Injection).
"""

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.services.legal.web_reader import fetch_web_page_content


class WebReaderInput(BaseModel):
    """Esquema de entrada validado para la herramienta de lectura web."""

    url: str = Field(
        ...,
        description="URL o enlace web público a inspeccionar (noticia, fallo, publicación, artículo periodístico)",
    )
    timeout_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=10.0,
        description="Tiempo máximo de espera de conexión en segundos para no demorar la respuesta",
    )


@tool("analizar_enlace_web", args_schema=WebReaderInput)
async def analizar_enlace_web(url: str, timeout_seconds: float = 5.0) -> dict:
    """Lee, extrae y limpia el texto de un enlace web público de forma segura.

    Aplica defensas activas Anti-SSRF (bloquea IPs locales/privadas y nombres de red),
    audita contra Indirect Prompt Injection y devuelve el texto limpio para su
    posterior contraste frente a las leyes vigentes.
    """
    res = await fetch_web_page_content(url=url, timeout_seconds=timeout_seconds)
    return res.model_dump()


def build_web_tools() -> list[BaseTool]:
    """Factory para instanciar las herramientas de análisis web del agente legal."""
    return [analizar_enlace_web]
