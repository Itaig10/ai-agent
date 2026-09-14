from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolActivity:
    """A user-visible update from a provider tool."""

    kind: str
    status: str
    title: str
    detail: str = ""
    output: str = ""
    duration_ms: int | None = None


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """A provider action that must pause for a user decision."""

    kind: str
    title: str
    detail: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One step in the provider's current task plan."""

    text: str
    status: str


@dataclass(frozen=True, slots=True)
class PlanUpdate:
    """A complete replacement for the visible task plan."""

    steps: tuple[PlanStep, ...]
    explanation: str = ""


@dataclass(frozen=True, slots=True)
class DiffUpdate:
    """The latest unified diff produced during the active turn."""

    diff: str


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A streamed fragment of the active assistant response."""

    text: str


@dataclass(frozen=True, slots=True)
class NotificationEvent:
    """A message shown in the in-app notification center."""

    title: str
    message: str
    severity: str = "information"
