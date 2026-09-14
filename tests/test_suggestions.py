import unittest

from ai_agent.suggestions import SlashCommandSuggester


class SlashCommandSuggesterTests(unittest.IsolatedAsyncioTestCase):
    async def test_suggests_commands_case_insensitively(self) -> None:
        suggester = SlashCommandSuggester(
            lambda: ["/skills", "/skill run testing ", "/theme nord"]
        )

        self.assertEqual(await suggester.get_suggestion("/SKI"), "/skills")
        self.assertEqual(
            await suggester.get_suggestion("/skill r"),
            "/skill run testing ",
        )
        self.assertIsNone(await suggester.get_suggestion("normal message"))

    async def test_dynamic_candidates_are_refreshed(self) -> None:
        candidates = ["/theme nord"]
        suggester = SlashCommandSuggester(lambda: candidates)

        self.assertEqual(await suggester.get_suggestion("/theme n"), "/theme nord")
        candidates[:] = ["/theme monokai"]
        self.assertEqual(
            await suggester.get_suggestion("/theme m"),
            "/theme monokai",
        )

    async def test_exact_value_has_no_redundant_suggestion(self) -> None:
        suggester = SlashCommandSuggester(lambda: ["/help"])

        self.assertIsNone(await suggester.get_suggestion("/help"))


if __name__ == "__main__":
    unittest.main()
