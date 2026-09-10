from ai_agent.config import Settings
from ai_agent.providers.base import ChatProvider
from ai_agent.providers.openai import OpenAIProvider


def create_provider(settings: Settings) -> ChatProvider:
    """Build the configured provider behind the shared provider interface."""
    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider")
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.model,
        )

    raise ValueError(
        f"Unsupported provider {settings.provider!r}. Supported providers: openai"
    )
