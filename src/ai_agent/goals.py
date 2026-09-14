from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
import re
from uuid import uuid4


TERMINAL_GOAL_STATES = {"completed", "failed", "stopped", "timed_out"}
_TIMEOUT_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)(s|m|h|d)$", re.IGNORECASE)
_GOAL_STATUS_PATTERN = re.compile(
    r"^[ \t]*<goal-status>\s*(complete|continue)\s*</goal-status>[ \t]*\Z",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(slots=True)
class Goal:
    """A persistent objective pursued across multiple model turns."""

    id: str
    objective: str
    status: str
    created_at: str
    deadline: str
    attempts: int = 0
    last_result: str = ""

    @property
    def deadline_at(self) -> datetime:
        return datetime.fromisoformat(self.deadline)

    @property
    def remaining_seconds(self) -> float:
        return (self.deadline_at - datetime.now(UTC)).total_seconds()


class GoalStore:
    """Store goal state as workspace-local JSON for restart recovery."""

    _VALID_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,79}$")

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or Path.cwd() / ".ai-agent" / "goals"

    def create(self, objective: str, timeout_seconds: float) -> Goal:
        objective = objective.strip()
        if not objective:
            raise ValueError("Goal objective cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("Goal timeout must be greater than zero")
        if not math.isfinite(timeout_seconds):
            raise ValueError("Goal timeout is too large")
        now = datetime.now(UTC)
        try:
            deadline = now + timedelta(seconds=timeout_seconds)
        except OverflowError as error:
            raise ValueError("Goal timeout is too large") from error
        goal = Goal(
            id=f"{now.strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}",
            objective=objective,
            status="active",
            created_at=now.isoformat(),
            deadline=deadline.isoformat(),
        )
        self.save(goal)
        return goal

    def save(self, goal: Goal) -> Path:
        path = self._path(goal.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(asdict(goal), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def load(self, goal_id: str) -> Goal:
        path = self._path(goal_id)
        if not path.is_file():
            raise ValueError(f"Goal {goal_id!r} does not exist")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return Goal(**payload)
        except (TypeError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Goal {goal_id!r} is invalid") from error

    def list(self) -> list[Goal]:
        if not self.directory.exists():
            return []
        goals = []
        for path in self.directory.glob("*.json"):
            try:
                goals.append(self.load(path.stem))
            except ValueError:
                continue
        return sorted(goals, key=lambda goal: goal.created_at, reverse=True)

    def active(self) -> Goal | None:
        return next((goal for goal in self.list() if goal.status == "active"), None)

    def _path(self, goal_id: str) -> Path:
        if not self._VALID_ID.fullmatch(goal_id):
            raise ValueError("Invalid goal ID")
        return self.directory / f"{goal_id}.json"


def parse_timeout(value: str) -> float:
    """Parse a compact duration such as 30s, 10m, 2h, or 1d."""

    match = _TIMEOUT_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError("Timeout must use s, m, h, or d; for example 30m")
    amount = float(match.group(1))
    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2).lower()]
    seconds = amount * multiplier
    if seconds <= 0:
        raise ValueError("Goal timeout must be greater than zero")
    if not math.isfinite(seconds):
        raise ValueError("Goal timeout is too large")
    return seconds


def goal_reply_complete(reply: str) -> bool:
    match = _GOAL_STATUS_PATTERN.search(reply)
    return match is not None and match.group(1).lower() == "complete"


def clean_goal_reply(reply: str) -> str:
    return _GOAL_STATUS_PATTERN.sub("", reply).strip()


def build_goal_prompt(goal: Goal) -> str:
    previous = (
        f"Previous attempt result:\n{goal.last_result[-4000:]}\n\n"
        if goal.last_result
        else ""
    )
    return (
        "You are working autonomously toward a persistent goal. Complete one or "
        "more concrete next steps now, using available tools when useful. Verify "
        "your work before declaring completion.\n\n"
        f"Goal: {goal.objective}\n"
        f"Attempt: {goal.attempts + 1}\n"
        f"Time remaining: {max(0, round(goal.remaining_seconds))} seconds\n\n"
        f"{previous}"
        "End your response with a status marker on its own final line. Use exactly "
        "<goal-status>complete</goal-status> only when the entire goal is achieved "
        "and verified. Otherwise state the next unfinished step and end with exactly "
        "<goal-status>continue</goal-status> on its own final line."
    )
