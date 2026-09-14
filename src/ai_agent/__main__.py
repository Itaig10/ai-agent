from dataclasses import replace

from ai_agent.app import AgentApp
from ai_agent.config import Settings
from ai_agent.providers.factory import create_provider
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

    AgentApp(
        ChatSession(provider),
        provider_factory=provider_factory,
        tool_registry=tool_registry,
    ).run()


if __name__ == "__main__":
    main()
