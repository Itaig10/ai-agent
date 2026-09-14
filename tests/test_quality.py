import asyncio
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_agent.quality import (
    AutoFixWorkflow,
    TestRun,
    build_review_prompt,
    collect_workspace_diff,
    detect_test_command,
    parse_autofix_args,
)
from ai_agent.execution import ExecutionLimits


class AutoFixWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_passes_without_requesting_a_fix(self) -> None:
        async def executor(command: object, timeout: float) -> TestRun:
            return TestRun(("test",), 0, "all passed", 10)

        fix_calls = 0

        async def fixer(prompt: str) -> str:
            nonlocal fix_calls
            fix_calls += 1
            return "unused"

        report = await AutoFixWorkflow(executor=executor).run(("test",), fixer)

        self.assertTrue(report.passed)
        self.assertEqual(fix_calls, 0)
        self.assertEqual(len(report.runs), 1)

    async def test_requests_fix_and_reruns_until_success(self) -> None:
        calls = 0
        prompts: list[str] = []
        progress: list[str] = []

        async def executor(command: object, timeout: float) -> TestRun:
            nonlocal calls
            calls += 1
            return TestRun(
                ("python", "-m", "unittest"),
                1 if calls == 1 else 0,
                "AssertionError: expected 2" if calls == 1 else "OK",
                10,
            )

        async def fixer(prompt: str) -> str:
            prompts.append(prompt)
            return "fixed implementation"

        async def record_progress(message: str) -> None:
            progress.append(message)

        report = await AutoFixWorkflow(executor=executor).run(
            ("python", "-m", "unittest"),
            fixer,
            retries=2,
            progress=record_progress,
        )

        self.assertTrue(report.passed)
        self.assertEqual(len(report.runs), 2)
        self.assertEqual(report.fix_responses, ("fixed implementation",))
        self.assertIn("Do not hide, weaken, or delete tests", prompts[0])
        self.assertIn("AssertionError", prompts[0])
        self.assertEqual(len(progress), 3)

    async def test_stops_at_retry_limit(self) -> None:
        async def executor(command: object, timeout: float) -> TestRun:
            return TestRun(("test",), 1, "still broken", 10)

        async def fixer(prompt: str) -> str:
            return "attempted fix"

        report = await AutoFixWorkflow(executor=executor).run(
            ("test",), fixer, retries=2
        )

        self.assertFalse(report.passed)
        self.assertEqual(len(report.runs), 3)
        self.assertEqual(len(report.fix_responses), 2)
        self.assertIn("still failing", report.render())

    async def test_retry_and_command_validation(self) -> None:
        workflow = AutoFixWorkflow()

        async def fixer(prompt: str) -> str:
            return prompt

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            await workflow.run((), fixer)
        with self.assertRaisesRegex(ValueError, "between 0 and 5"):
            await workflow.run(("test",), fixer, retries=6)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            await workflow.run(("test",), fixer, timeout_seconds=0)

    async def test_real_test_command_captures_success_failure_and_timeout(self) -> None:
        workflow = AutoFixWorkflow()
        passed = await workflow.run_test_command(
            (sys.executable, "-c", "print('OK')"), 2
        )
        failed = await workflow.run_test_command(
            (sys.executable, "-c", "import sys; print('bad'); sys.exit(3)"), 2
        )
        timed_out = await workflow.run_test_command(
            (
                sys.executable,
                "-c",
                "import time; print('diagnostic', flush=True); time.sleep(5)",
            ),
            0.5,
        )

        self.assertTrue(passed.passed)
        self.assertIn("OK", passed.output)
        self.assertEqual(failed.returncode, 3)
        self.assertIn("bad", failed.output)
        self.assertTrue(timed_out.timed_out)
        self.assertIn("Timed out", timed_out.output)
        self.assertIn("diagnostic", timed_out.output)

    async def test_real_test_command_keeps_only_a_bounded_output_tail(self) -> None:
        limits = ExecutionLimits(output_bytes=1024)
        workflow = AutoFixWorkflow(limits=limits)

        result = await workflow.run_test_command(
            (
                sys.executable,
                "-c",
                "import sys; sys.stdout.write('x' * 100000 + 'TAIL')",
            ),
            2,
        )

        self.assertLessEqual(len(result.output.encode()), 1024)
        self.assertTrue(result.output.endswith("TAIL"))


class QualityHelpersTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_autofix_options_and_explicit_command(self) -> None:
        retries, timeout, command = parse_autofix_args(
            ["--retries", "3", "--timeout", "2m", "--", "npm", "test"]
        )

        self.assertEqual(retries, 3)
        self.assertEqual(timeout, 120)
        self.assertEqual(command, ("npm", "test"))

    def test_parse_autofix_rejects_invalid_options(self) -> None:
        for arguments, pattern in (
            (["--retries"], "requires a number"),
            (["--retries", "many"], "whole number"),
            (["--retries", "9", "--", "test"], "between 0 and 5"),
            (["--timeout"], "requires a duration"),
            (["--unknown", "test"], "Unknown autofix option"),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, pattern):
                    parse_autofix_args(arguments)

    def test_detects_common_test_commands(self) -> None:
        cases = {
            "pyproject.toml": (sys.executable, "-m", "unittest"),
            "package.json": ("npm", "test"),
            "Cargo.toml": ("cargo", "test"),
            "go.mod": ("go", "test"),
        }
        for marker, prefix in cases.items():
            with self.subTest(marker=marker), TemporaryDirectory() as directory:
                workspace = Path(directory)
                (workspace / marker).write_text("", encoding="utf-8")
                self.assertEqual(detect_test_command(workspace)[: len(prefix)], prefix)

    def test_review_prompt_is_structured_and_bounded(self) -> None:
        prompt = build_review_prompt("+changed\n", "security only")

        self.assertIn("independent, read-only code review", prompt)
        self.assertIn("correctness, security, missing tests", prompt)
        self.assertIn("security only", prompt)
        with self.assertRaisesRegex(ValueError, "no workspace changes"):
            build_review_prompt("   ")

    async def test_collects_tracked_and_untracked_changes(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._git(workspace, "init")
            self._git(workspace, "config", "user.name", "Test User")
            self._git(workspace, "config", "user.email", "test@example.com")
            tracked = workspace / "tracked.py"
            tracked.write_text("value = 1\n", encoding="utf-8")
            self._git(workspace, "add", "tracked.py")
            self._git(workspace, "commit", "-m", "initial")
            tracked.write_text("value = 2\n", encoding="utf-8")
            (workspace / "new.py").write_text("created = True\n", encoding="utf-8")
            hidden = workspace / ".ai-agent"
            hidden.mkdir()
            (hidden / "state.txt").write_text("private state", encoding="utf-8")

            diff = await collect_workspace_diff(workspace)

            self.assertIn("tracked.py", diff)
            self.assertIn("new.py", diff)
            self.assertNotIn("private state", diff)

    @staticmethod
    def _git(workspace: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
