from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from ai_agent.config import Settings
from ai_agent.providers.factory import create_provider
from ai_agent.tools import ToolRegistry


def make_settings(**changes: object) -> Settings:
    settings = Settings(
        provider="codex",
        model=None,
        effort="medium",
        openai_api_key=None,
        litellm_api_key=None,
        litellm_api_base=None,
        context_window=None,
        codex_path="codex-custom",
    )
    return replace(settings, **changes)


class ProviderFactoryTests(unittest.TestCase):
    @patch("ai_agent.providers.factory.CodexProvider")
    def test_builds_codex_with_shared_tool_registry(self, provider: object) -> None:
        registry = ToolRegistry()
        expected = object()
        provider.return_value = expected

        result = create_provider(make_settings(), registry)

        self.assertIs(result, expected)
        provider.assert_called_once_with(
            command="codex-custom",
            model=None,
            effort="medium",
            cwd=Path.cwd(),
            tool_registry=registry,
        )

    @patch("ai_agent.providers.factory.OpenAIProvider")
    def test_builds_openai_with_configured_model(self, provider: object) -> None:
        registry = ToolRegistry()
        settings = make_settings(
            provider="openai",
            model="gpt-test",
            openai_api_key="secret",
        )

        create_provider(settings, registry)

        provider.assert_called_once_with(
            api_key="secret",
            model="gpt-test",
            effort="medium",
            tool_registry=registry,
        )

    @patch("ai_agent.providers.factory.LiteLLMProvider")
    def test_builds_litellm_with_proxy_settings(self, provider: object) -> None:
        settings = make_settings(
            provider="litellm",
            model="anthropic/example",
            litellm_api_key="proxy-key",
            litellm_api_base="https://proxy.example.test",
            context_window=200_000,
        )

        create_provider(settings)

        provider.assert_called_once_with(
            model="anthropic/example",
            effort="medium",
            api_key="proxy-key",
            api_base="https://proxy.example.test",
            context_window=200_000,
        )

    def test_missing_provider_credentials_are_actionable(self) -> None:
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            create_provider(make_settings(provider="openai"))
        with self.assertRaisesRegex(ValueError, "AI_AGENT_MODEL"):
            create_provider(make_settings(provider="litellm"))

    def test_unknown_provider_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Supported providers"):
            create_provider(make_settings(provider="unknown"))


if __name__ == "__main__":
    unittest.main()
