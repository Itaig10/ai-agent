from pathlib import Path

from ai_agent.config import Settings
from ai_agent.providers.base import ChatProvider
from ai_agent.providers.codex import CodexProvider
from ai_agent.providers.litellm import LiteLLMProvider
from ai_agent.providers.openai import OpenAIProvider
from ai_agent.tools import ToolRegistry


def create_provider(
    settings: Settings, tool_registry: ToolRegistry | None = None
) -> ChatProvider:
    """Build the configured provider behind the shared provider interface."""
    if settings.provider == "codex":
        return CodexProvider(
            command=settings.codex_path,
            model=settings.model,
            effort=settings.effort,
            cwd=Path.cwd(),
            tool_registry=tool_registry,
        )

    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider")
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.model or "gpt-5.5",
            effort=settings.effort,
            tool_registry=tool_registry,
        )

    if settings.provider == "litellm":
        if not settings.model:
            raise ValueError("AI_AGENT_MODEL is required for the LiteLLM provider")
        return LiteLLMProvider(
            model=settings.model,
            effort=settings.effort,
            api_key=settings.litellm_api_key,
            api_base=settings.litellm_api_base,
            timeout_seconds=settings.litellm_timeout_seconds,
            context_window=settings.context_window,
        )

    raise ValueError(
        f"Unsupported provider {settings.provider!r}. "
        "Supported providers: codex, openai, litellm"
    )
