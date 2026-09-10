from pathlib import Path

from ai_agent.config import Settings
from ai_agent.providers.base import ChatProvider
from ai_agent.providers.codex import CodexProvider
from ai_agent.providers.openai import OpenAIProvider


def create_provider(settings: Settings) -> ChatProvider:
    """Build the configured provider behind the shared provider interface."""
    if settings.provider == "codex":
        return CodexProvider(
            command=settings.codex_path,
            model=settings.model,
            cwd=Path.cwd(),
        )

    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider")
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.model or "gpt-5.5",
        )

    raise ValueError(
        f"Unsupported provider {settings.provider!r}. "
        "Supported providers: codex, openai"
    )
