from ai_agent.app import AgentApp
from ai_agent.config import Settings
from ai_agent.providers.factory import create_provider
from ai_agent.session import ChatSession


def main() -> None:
    """Run the terminal application."""
    try:
        settings = Settings.from_env()
        provider = create_provider(settings)
    except ValueError as error:
        raise SystemExit(f"Configuration error: {error}") from error

    AgentApp(ChatSession(provider)).run()


if __name__ == "__main__":
    main()
