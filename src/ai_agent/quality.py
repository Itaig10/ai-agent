import asyncio
import os
import signal
import sys
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from ai_agent.execution import ExecutionLimits
from ai_agent.goals import parse_timeout


MAX_REVIEW_DIFF = 200_000


@dataclass(frozen=True, slots=True)
class TestRun:
    """Result of one bounded test command."""

    command: tuple[str, ...]
    returncode: int
    output: str
    duration_ms: int
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@dataclass(frozen=True, slots=True)
class AutoFixReport:
    """Outcome of a test-and-fix workflow."""

    runs: tuple[TestRun, ...]
    fix_responses: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return bool(self.runs) and self.runs[-1].passed

    def render(self) -> str:
        final = self.runs[-1]
        state = "passed" if self.passed else "still failing"
        command = " ".join(final.command)
        return (
            f"Autofix {state} after {len(self.runs)} test run(s) and "
            f"{len(self.fix_responses)} fix attempt(s).\n"
            f"Command: {command}\n\nFinal test output:\n{final.output[-4000:]}"
        )


Fixer = Callable[[str], Awaitable[str]]
ProgressHandler = Callable[[str], Awaitable[None]]
TestExecutor = Callable[[Sequence[str], float], Awaitable[TestRun]]


class AutoFixWorkflow:
    """Run tests and ask a model for bounded repair attempts until they pass."""

    def __init__(
        self,
        workspace: Path | None = None,
        *,
        executor: TestExecutor | None = None,
        limits: ExecutionLimits | None = None,
    ) -> None:
        self.workspace = (workspace or Path.cwd()).resolve()
        self.limits = limits or ExecutionLimits()
        self._executor = executor or self.run_test_command

    async def run(
        self,
        command: Sequence[str],
        fixer: Fixer,
        *,
        retries: int = 2,
        timeout_seconds: float = 300,
        progress: ProgressHandler | None = None,
    ) -> AutoFixReport:
        if not command:
            raise ValueError("Test command cannot be empty")
        if retries < 0 or retries > 5:
            raise ValueError("Autofix retries must be between 0 and 5")
        if timeout_seconds <= 0:
            raise ValueError("Test timeout must be greater than zero")

        runs: list[TestRun] = []
        responses: list[str] = []
        for attempt in range(retries + 1):
            if progress is not None:
                await progress(f"Running tests ({attempt + 1}/{retries + 1})")
            result = await self._executor(command, timeout_seconds)
            runs.append(result)
            if result.passed or attempt == retries:
                break
            if progress is not None:
                await progress(f"Requesting fix ({attempt + 1}/{retries})")
            response = await fixer(self._fix_prompt(result, attempt + 1, retries))
            responses.append(response.strip())
        return AutoFixReport(tuple(runs), tuple(responses))

    async def run_test_command(
        self,
        command: Sequence[str],
        timeout_seconds: float,
    ) -> TestRun:
        started = monotonic()
        effective_timeout = self.limits.effective_timeout(timeout_seconds)
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **self.limits.subprocess_options(),
        )
        output = bytearray()
        output_limit = self.limits.output_bytes

        async def collect_output() -> None:
            if process.stdout is None:
                raise RuntimeError("Test command output is unavailable")
            while chunk := await process.stdout.read(8192):
                output.extend(chunk)
                overflow = len(output) - output_limit
                if overflow > 0:
                    del output[:overflow]
            await process.wait()

        try:
            await asyncio.wait_for(
                collect_output(),
                timeout=effective_timeout,
            )
            timed_out = False
        except asyncio.TimeoutError:
            await self._terminate(process)
            await collect_output()
            timed_out = True
        except asyncio.CancelledError:
            await self._terminate(process)
            raise
        decoded = output.decode(errors="replace").strip()
        if timed_out:
            decoded = f"Timed out after {effective_timeout:g}s.\n{decoded}".strip()
        elif not decoded:
            decoded = "Tests completed with no output."
        return TestRun(
            tuple(command),
            process.returncode if process.returncode is not None else 124,
            decoded,
            round((monotonic() - started) * 1000),
            timed_out,
        )

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        else:
            with suppress(ProcessLookupError):
                process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
            return
        except asyncio.TimeoutError:
            pass
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            with suppress(ProcessLookupError):
                process.kill()
        await process.wait()

    @staticmethod
    def _fix_prompt(result: TestRun, attempt: int, retries: int) -> str:
        command = " ".join(result.command)
        return (
            "Act as an implementation agent. The test command below failed. Inspect "
            "the workspace, diagnose the root cause, and make the smallest safe code "
            "change that fixes it. Do not hide, weaken, or delete tests. Do not merely "
            "describe a fix: edit the files. The controller will rerun the tests.\n\n"
            f"Fix attempt: {attempt} of {retries}\n"
            f"Command: {command}\n"
            f"Exit status: {result.returncode}\n\n"
            f"Test output:\n{result.output[-20_000:]}"
        )


def parse_autofix_args(
    arguments: Sequence[str],
    workspace: Path | None = None,
) -> tuple[int, float, tuple[str, ...]]:
    """Parse `/autofix` arguments without invoking a shell."""

    retries = 2
    timeout_seconds = 300.0
    command: list[str] = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == "--":
            command = list(arguments[index + 1 :])
            break
        if value == "--retries":
            index += 1
            if index >= len(arguments):
                raise ValueError("--retries requires a number")
            try:
                retries = int(arguments[index])
            except ValueError as error:
                raise ValueError("--retries must be a whole number") from error
        elif value == "--timeout":
            index += 1
            if index >= len(arguments):
                raise ValueError("--timeout requires a duration")
            timeout_seconds = parse_timeout(arguments[index])
        elif value.startswith("--"):
            raise ValueError(f"Unknown autofix option {value!r}")
        else:
            command = list(arguments[index:])
            break
        index += 1
    if retries < 0 or retries > 5:
        raise ValueError("Autofix retries must be between 0 and 5")
    if not command:
        command = list(detect_test_command(workspace or Path.cwd()))
    return retries, timeout_seconds, tuple(command)


def detect_test_command(workspace: Path) -> tuple[str, ...]:
    """Select a conservative test command from common project markers."""

    if (workspace / "pyproject.toml").is_file() or (workspace / "setup.py").is_file():
        return (sys.executable, "-m", "unittest", "discover", "-s", "tests")
    if (workspace / "package.json").is_file():
        return ("npm", "test")
    if (workspace / "Cargo.toml").is_file():
        return ("cargo", "test")
    if (workspace / "go.mod").is_file():
        return ("go", "test", "./...")
    raise ValueError("Could not detect tests; pass a command after --")


def build_review_prompt(diff: str, scope: str = "all changes") -> str:
    """Build a read-only, structured independent-review request."""

    diff = diff.strip()
    if not diff:
        raise ValueError("There are no workspace changes to review")
    return (
        "Perform an independent, read-only code review. Do not modify files or run "
        "destructive commands. Review only the supplied changes and prioritize real, "
        "actionable problems. Check correctness, security, missing tests, regressions, "
        "and unnecessary complexity. For every finding include severity, file, the "
        "relevant line or symbol, impact, and a concrete fix. If there are no "
        "findings, say so explicitly. End with a short risk summary.\n\n"
        f"Requested scope: {scope.strip() or 'all changes'}\n\n"
        f"Workspace changes:\n{diff[:MAX_REVIEW_DIFF]}"
    )


async def collect_workspace_diff(
    workspace: Path | None = None,
    *,
    max_bytes: int = MAX_REVIEW_DIFF,
) -> str:
    """Collect tracked and bounded untracked text changes for review."""

    workspace = (workspace or Path.cwd()).resolve()
    tracked = await _git_output(
        workspace,
        "diff",
        "--no-ext-diff",
        "--no-color",
        "HEAD",
    )
    untracked_output = await _git_output(
        workspace,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    sections = [tracked] if tracked.strip() else []
    used = len(tracked.encode("utf-8"))
    for name in untracked_output.split("\0"):
        if not name or name.startswith(".ai-agent/"):
            continue
        path = (workspace / name).resolve()
        if not path.is_relative_to(workspace) or not path.is_file():
            continue
        remaining = max_bytes - used
        if remaining <= 0:
            break
        data = path.read_bytes()[:remaining]
        if b"\0" in data:
            continue
        content = data.decode("utf-8", errors="replace")
        section = (
            f"diff --git a/{name} b/{name}\n"
            "new untracked file\n"
            f"--- /dev/null\n+++ b/{name}\n"
            + "".join(f"+{line}" for line in content.splitlines(keepends=True))
        )
        sections.append(section)
        used += len(section.encode("utf-8"))
    combined = "\n".join(sections)
    encoded = combined.encode("utf-8")
    if len(encoded) > max_bytes:
        combined = encoded[:max_bytes].decode("utf-8", errors="ignore")
        combined += "\n[diff truncated at safety limit]"
    return combined


async def _git_output(workspace: Path, *arguments: str) -> str:
    process = await asyncio.create_subprocess_exec(
        "git",
        *arguments,
        cwd=workspace,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        message = (stderr or stdout).decode(errors="replace").strip()
        raise ValueError(message or f"git exited with status {process.returncode}")
    return stdout.decode("utf-8", errors="replace")
