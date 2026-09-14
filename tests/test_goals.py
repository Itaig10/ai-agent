import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_agent.goals import (
    GoalStore,
    build_goal_prompt,
    clean_goal_reply,
    goal_reply_complete,
    parse_timeout,
)


class GoalStoreTests(unittest.TestCase):
    def test_goal_progress_persists(self) -> None:
        with TemporaryDirectory() as directory:
            store = GoalStore(Path(directory))
            goal = store.create("finish the parser", 300)
            goal.attempts = 2
            goal.last_result = "Tests are running"
            store.save(goal)

            restored = store.active()
            self.assertIsNotNone(restored)
            self.assertEqual(restored.objective, "finish the parser")
            self.assertEqual(restored.attempts, 2)
            self.assertEqual(restored.last_result, "Tests are running")

    def test_timeout_parser(self) -> None:
        self.assertEqual(parse_timeout("30s"), 30)
        self.assertEqual(parse_timeout("1.5m"), 90)
        self.assertEqual(parse_timeout("2h"), 7200)
        with self.assertRaisesRegex(ValueError, "for example"):
            parse_timeout("soon")

    def test_timeout_parser_accepts_days_and_rejects_zero(self) -> None:
        self.assertEqual(parse_timeout("2D"), 172_800)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            parse_timeout("0s")

    def test_goal_status_marker(self) -> None:
        reply = "Everything passes.\n<goal-status>complete</goal-status>"
        self.assertTrue(goal_reply_complete(reply))
        self.assertEqual(clean_goal_reply(reply), "Everything passes.")
        unfinished = "More work remains.\n<goal-status>continue</goal-status>"
        self.assertFalse(
            goal_reply_complete(unfinished)
        )

    def test_goal_status_must_be_the_final_marker(self) -> None:
        quoted = (
            "Do not emit <goal-status>complete</goal-status> until tests pass. "
            "More work remains."
        )
        inline_quote = "For example: <goal-status>complete</goal-status>"
        superseded = (
            "<goal-status>complete</goal-status>\n"
            "Actually, more work remains.\n"
            "<goal-status>continue</goal-status>"
        )

        self.assertFalse(goal_reply_complete(quoted))
        self.assertFalse(goal_reply_complete(inline_quote))
        self.assertFalse(goal_reply_complete(superseded))
        self.assertIn("complete", clean_goal_reply(quoted))
        self.assertNotIn("continue", clean_goal_reply(superseded))

    def test_unrepresentable_timeout_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            store = GoalStore(Path(directory))
            with self.assertRaisesRegex(ValueError, "too large"):
                store.create(
                    "impossible deadline",
                    parse_timeout("999999999999999999999d"),
                )

    def test_goal_creation_validates_objective_and_timeout(self) -> None:
        with TemporaryDirectory() as directory:
            store = GoalStore(Path(directory))
            with self.assertRaisesRegex(ValueError, "objective cannot be empty"):
                store.create("   ", 30)
            with self.assertRaisesRegex(ValueError, "greater than zero"):
                store.create("valid objective", 0)

    def test_invalid_goal_file_is_rejected_and_skipped_from_listing(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = GoalStore(root)
            (root / "broken.json").write_text("{", encoding="utf-8")
            (root / "missing-fields.json").write_text(
                json.dumps({"objective": "incomplete"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "is invalid"):
                store.load("broken")
            self.assertEqual(store.list(), [])

    def test_goal_prompt_includes_progress_and_attempt_number(self) -> None:
        with TemporaryDirectory() as directory:
            goal = GoalStore(Path(directory)).create("finish tests", 300)
            goal.attempts = 2
            goal.last_result = "Added regression tests"

            prompt = build_goal_prompt(goal)

            self.assertIn("Goal: finish tests", prompt)
            self.assertIn("Attempt: 3", prompt)
            self.assertIn("Added regression tests", prompt)
            self.assertIn("on its own final line", prompt)

    def test_goal_id_cannot_escape_store(self) -> None:
        with TemporaryDirectory() as directory:
            store = GoalStore(Path(directory))

            with self.assertRaisesRegex(ValueError, "Invalid goal ID"):
                store.load("../outside")


if __name__ == "__main__":
    unittest.main()
