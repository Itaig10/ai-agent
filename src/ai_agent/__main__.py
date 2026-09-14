from dataclasses import replace

from ai_agent.app import AgentApp
from ai_agent.config import Settings
from ai_agent.providers.factory import create_provider
from ai_agent.providers.litellm import fetch_litellm_models
from ai_agent.session import ChatSession
from ai_agent.tools import ToolRegistry


def main() -> None:
    """Run the terminal application."""
    try:
        settings = Settings.from_env()
        tool_registry = ToolRegistry()
        provider = create_provider(settings, tool_registry)
    except ValueError as error:
        raise SystemExit(f"Configuration error: {error}") from error

    def provider_factory(provider_name: str, model: str | None):
        selected = replace(settings, provider=provider_name, model=model)
        return create_provider(selected, tool_registry)

    async def model_fetcher(provider_name: str) -> tuple[str, ...]:
        if provider_name != "litellm":
            raise ValueError("Model discovery is currently available for LiteLLM")
        return await fetch_litellm_models(
            api_base=settings.litellm_api_base,
            api_key=settings.litellm_api_key,
            timeout_seconds=settings.litellm_timeout_seconds,
        )

    AgentApp(
        ChatSession(provider),
        provider_factory=provider_factory,
        model_fetcher=model_fetcher,
        tool_registry=tool_registry,
    ).run()


if __name__ == "__main__":
    main()
