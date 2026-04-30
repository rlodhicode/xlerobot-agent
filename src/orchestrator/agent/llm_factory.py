from typing import Protocol, Any, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_google_vertexai import ChatVertexAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from ..config import Settings, configure_langsmith, get_settings, resolve_model
from ..capabilities.registry import TOOLS

class LLMLike(Protocol):
    def invoke(self, input_data: Any) -> Any: ...
    async def ainvoke(self, input_data: Any) -> Any: ...

def get_vertex_llm(settings: Settings, model_name: str, callbacks: Optional[list[BaseCallbackHandler]] = None) -> ChatVertexAI:
    kwargs = dict(
        model_name=model_name or settings.vertex_model,
        project=settings.vertex_project_id,
        location=settings.vertex_location,
        temperature=0,
        include_thoughts=True,
    )
    if callbacks:
        kwargs["callbacks"] = callbacks
    return ChatVertexAI(**kwargs)

def get_ollama_llm(settings: Settings, model_name: str, callbacks: Optional[list[BaseCallbackHandler]] = None) -> ChatOllama:
    kwargs = dict(
        model=model_name or settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        reasoning=True,
    )
    if callbacks:
        kwargs["callbacks"] = callbacks
    return ChatOllama(**kwargs)

def get_openai_llm(settings: Settings, model_name: str, callbacks: Optional[list[BaseCallbackHandler]] = None) -> ChatOpenAI:
    kwargs = dict(
        model=model_name or settings.openai_model,
        api_key=settings.openai_api_key or None,
        temperature=0,
    )
    if callbacks:
        kwargs["callbacks"] = callbacks
    return ChatOpenAI(**kwargs)

def get_anthropic_llm(settings: Settings, model_name: str, callbacks: Optional[list[BaseCallbackHandler]] = None) -> ChatAnthropic:
    kwargs: dict = dict(
        model=model_name or settings.anthropic_model,
        api_key=settings.anthropic_api_key or None,
    )
    if settings.anthropic_thinking_budget > 0:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": settings.anthropic_thinking_budget}
        kwargs["temperature"] = 1
    else:
        kwargs["temperature"] = 0
    if callbacks:
        kwargs["callbacks"] = callbacks
    return ChatAnthropic(**kwargs)

_PROVIDER_FACTORIES = {
    "vertex": get_vertex_llm,
    "ollama": get_ollama_llm,
    "openai": get_openai_llm,
    "anthropic": get_anthropic_llm,
}

def get_llm(callbacks: Optional[list[BaseCallbackHandler]] = None) -> tuple[LLMLike, LLMLike]:
    """Return (tool_bound_llm, raw_llm).

    raw_llm is used for compaction summaries — it does not have robot tools bound,
    so it cannot accidentally call capabilities during summarisation.
    """
    settings = get_settings()
    configure_langsmith(settings)

    provider = settings.llm_provider.strip().lower()
    model_name = resolve_model(settings)

    factory = _PROVIDER_FACTORIES.get(provider)
    if factory is None:
        supported = ", ".join(sorted(_PROVIDER_FACTORIES))
        raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'. Choose one of: {supported}.")

    raw = factory(settings, model_name, callbacks=callbacks)
    return raw.bind_tools(TOOLS), raw

