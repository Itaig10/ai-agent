import unittest
from unittest.mock import patch

from ai_agent.providers.codex import CodexProvider


class CodexProviderTests(unittest.TestCase):
    @patch("ai_agent.providers.codex.shutil.which", return_value=None)
    def test_missing_executable_has_actionable_error(self, _which: object) -> None:
        with self.assertRaisesRegex(ValueError, "CODEX_PATH"):
            CodexProvider()

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_uses_account_default_model(self, _which: object) -> None:
        provider = CodexProvider()

        self.assertEqual(provider.model, "account default")
        self.assertIsNone(provider.requested_model)

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_explicit_model_is_preserved(self, _which: object) -> None:
        provider = CodexProvider(model="example-model")

        self.assertEqual(provider.model, "example-model")
        self.assertEqual(provider.requested_model, "example-model")

    def test_formats_protocol_errors(self) -> None:
        self.assertEqual(
            CodexProvider._format_error({"message": "request failed"}),
            "request failed",
        )
