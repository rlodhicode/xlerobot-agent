"""Configuration and environment loading."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default models per provider — used when LLM_MODEL is not set.
PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "vertex": "gemini-2.5-flash",
    "ollama": "qwen3.5:4b",
    "openai": "gpt-4o",
    "anthropic": "claude-haiku-4-5",
}


class Settings(BaseSettings):
    """Runtime settings for local/dev and Docker execution."""

    llm_provider: str = "vertex"
    llm_model: str = ""  # overrides provider default when set

    # Vertex / Gemini
    vertex_project_id: str = ""
    vertex_location: str = "us-central1"
    vertex_model: str = "gemini-2.5-flash"
    google_application_credentials: str = ""

    # Ollama (local or remote)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:latest"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5"
    anthropic_thinking_budget: int = 8000  # 0 disables extended thinking

    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "xlerobot-orchestrator"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    max_iterations: int = 20


def get_settings() -> Settings:
    """Load environment variables and return validated settings."""
    load_dotenv()
    settings = Settings()
    if settings.google_application_credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
    return settings


def resolve_model(settings: Settings) -> str:
    """Return the effective model name: LLM_MODEL if set, else provider default."""
    if settings.llm_model.strip():
        return settings.llm_model.strip()
    provider = settings.llm_provider.strip().lower()
    return PROVIDER_DEFAULT_MODELS.get(provider, "")


def configure_langsmith(settings: Settings) -> None:
    """Set LangSmith tracing env vars if enabled."""
    if not settings.langsmith_tracing:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
