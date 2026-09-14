import unittest

from ai_agent.subagents import SUBAGENT_ROLES, build_subagent_prompt


class SubagentTests(unittest.TestCase):
    def test_each_role_builds_a_scoped_prompt(self) -> None:
        for name in SUBAGENT_ROLES:
            role, prompt = build_subagent_prompt(name, "inspect the parser")
            self.assertEqual(role.name, name)
            self.assertIn("inspect the parser", prompt)
            self.assertIn("Do not modify files", prompt)

    def test_unknown_role_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown subagent role"):
            build_subagent_prompt("unknown", "task")

    def test_role_names_are_case_insensitive_and_tasks_are_trimmed(self) -> None:
        role, prompt = build_subagent_prompt("REVIEWER", "  inspect auth  ")

        self.assertEqual(role.name, "reviewer")
        self.assertIn("Delegated task: inspect auth", prompt)

    def test_empty_task_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            build_subagent_prompt("tester", "   ")


if __name__ == "__main__":
    unittest.main()
