import os
import unittest
from unittest.mock import patch

from ai_agent.config import Settings


class SettingsTests(unittest.TestCase):
    @patch("ai_agent.config.load_dotenv")
    def test_defaults_select_codex_account_settings(self, _load: object) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.provider, "codex")
        self.assertIsNone(settings.model)
        self.assertIsNone(settings.effort)
        self.assertIsNone(settings.context_window)
        self.assertEqual(settings.litellm_timeout_seconds, 2.0)
        self.assertTrue(settings.litellm_verify_tls)
        self.assertEqual(settings.codex_path, "codex")

    @patch("ai_agent.config.load_dotenv")
    def test_environment_values_are_normalized(self, _load: object) -> None:
        environment = {
            "AI_AGENT_PROVIDER": " LiteLLM ",
            "AI_AGENT_MODEL": " openai/example ",
            "AI_AGENT_EFFORT": " HIGH ",
            "AI_AGENT_CONTEXT_WINDOW": "128000",
            "AI_AGENT_LITELLM_API_KEY": "proxy-key",
            "AI_AGENT_LITELLM_API_BASE": "https://proxy.example.test",
            "AI_AGENT_LITELLM_TIMEOUT_SECONDS": "2.5",
            "AI_AGENT_LITELLM_VERIFY_TLS": "false",
            "OPENAI_API_KEY": "openai-key",
            "CODEX_PATH": "/opt/codex",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.provider, "litellm")
        self.assertEqual(settings.model, "openai/example")
        self.assertEqual(settings.effort, "high")
        self.assertEqual(settings.context_window, 128_000)
        self.assertEqual(settings.litellm_api_key, "proxy-key")
        self.assertEqual(settings.litellm_timeout_seconds, 2.5)
        self.assertFalse(settings.litellm_verify_tls)
        self.assertEqual(settings.openai_api_key, "openai-key")
        self.assertEqual(settings.codex_path, "/opt/codex")

    @patch("ai_agent.config.load_dotenv")
    def test_invalid_effort_is_rejected(self, _load: object) -> None:
        with patch.dict(os.environ, {"AI_AGENT_EFFORT": "extreme"}, clear=True):
            with self.assertRaisesRegex(ValueError, "AI_AGENT_EFFORT"):
                Settings.from_env()

    @patch("ai_agent.config.load_dotenv")
    def test_invalid_context_windows_are_rejected(self, _load: object) -> None:
        for value in ("many", "0", "-1"):
            with self.subTest(value=value):
                with patch.dict(
                    os.environ,
                    {"AI_AGENT_CONTEXT_WINDOW": value},
                    clear=True,
                ):
                    with self.assertRaisesRegex(ValueError, "CONTEXT_WINDOW"):
                        Settings.from_env()

    @patch("ai_agent.config.load_dotenv")
    def test_invalid_litellm_timeouts_are_rejected(self, _load: object) -> None:
        for value in ("many", "0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                with patch.dict(
                    os.environ,
                    {"AI_AGENT_LITELLM_TIMEOUT_SECONDS": value},
                    clear=True,
                ):
                    with self.assertRaisesRegex(ValueError, "LITELLM_TIMEOUT"):
                        Settings.from_env()

    @patch("ai_agent.config.load_dotenv")
    def test_invalid_litellm_tls_setting_is_rejected(self, _load: object) -> None:
        with patch.dict(
            os.environ,
            {"AI_AGENT_LITELLM_VERIFY_TLS": "sometimes"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "VERIFY_TLS"):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
