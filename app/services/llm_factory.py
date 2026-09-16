"""Factoría unificada para modelos de lenguaje (OpenAI / OpenRouter / compatibles)."""

from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import Settings


def build_chat_model(settings: Settings, **kwargs: Any) -> ChatOpenAI:
    """Crea una instancia de ChatOpenAI configurada según Settings.
    Soporta OpenAI directo y OpenRouter automáticamente según la clave o base_url provistos.
    """
    model_kwargs: dict[str, Any] = {
        "model": settings.effective_chat_model,
        "api_key": settings.OPENAI_API_KEY.get_secret_value() or "dummy",
        "temperature": kwargs.get("temperature", settings.LLM_TEMPERATURE),
    }

    base_url = settings.effective_openai_base_url
    if base_url:
        model_kwargs["base_url"] = base_url

    for k, v in kwargs.items():
        if k != "temperature":
            model_kwargs[k] = v

    return ChatOpenAI(**model_kwargs)


def build_vision_model(settings: Settings, **kwargs: Any) -> ChatOpenAI:
    """Crea una instancia de ChatOpenAI configurada con el modelo multimodal de visión.
    Permite transcribir imágenes de contratos, cartas documento o fojas escaneadas.
    """
    model_kwargs: dict[str, Any] = {
        "model": settings.effective_vision_model,
        "api_key": settings.OPENAI_API_KEY.get_secret_value() or "dummy",
        "temperature": kwargs.get("temperature", 0.0),
    }

    base_url = settings.effective_openai_base_url
    if base_url:
        model_kwargs["base_url"] = base_url

    for k, v in kwargs.items():
        if k != "temperature":
            model_kwargs[k] = v

    return ChatOpenAI(**model_kwargs)

