from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_agent.goals import parse_timeout


POLICY_ACTIONS = {"allow", "ask", "deny"}


@dataclass(frozen=True, slots=True)
class CommandRule:
    """A persistent action for one argument-safe command prefix."""

    id: int
    action: str
    command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """The resolved action and optional matching rule."""

    action: str
    rule: CommandRule | None = None


class CommandPolicyStore:
    """Persist and evaluate longest-prefix command permission rules."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path.cwd() / ".ai-agent" / "command-policy.json"
        self.default_action = "ask"
        self._rules: list[CommandRule] = []
        self._next_id = 1
        self.load_error: str | None = None
        self._load()

    def list(self) -> tuple[CommandRule, ...]:
        return tuple(sorted(self._rules, key=lambda rule: rule.id))

    def evaluate(self, command: tuple[str, ...] | list[str]) -> PolicyDecision:
        normalized = tuple(command)
        matches = [
            rule
            for rule in self._rules
            if len(normalized) >= len(rule.command)
            and normalized[: len(rule.command)] == rule.command
        ]
        if not matches:
            return PolicyDecision(self.default_action)
        rule = max(
            matches,
            key=lambda item: (
                len(item.command),
                {"allow": 0, "ask": 1, "deny": 2}[item.action],
                item.id,
            ),
        )
        return PolicyDecision(rule.action, rule)

    def add(self, action: str, command: list[str] | tuple[str, ...]) -> CommandRule:
        action = action.lower()
        normalized = tuple(part for part in command if part)
        if action not in POLICY_ACTIONS:
            raise ValueError("Permission action must be allow, ask, or deny")
        if not normalized:
            raise ValueError("Command rule cannot be empty")
        rule = CommandRule(self._next_id, action, normalized)
        self._next_id += 1
        self._rules.append(rule)
        self._save()
        return rule

    def remove(self, rule_id: int) -> CommandRule:
        for rule in self._rules:
            if rule.id == rule_id:
                self._rules.remove(rule)
                self._save()
                return rule
        raise ValueError(f"Permission rule {rule_id} does not exist")

    def set_default(self, action: str) -> None:
        action = action.lower()
        if action not in POLICY_ACTIONS:
            raise ValueError("Permission action must be allow, ask, or deny")
        self.default_action = action
        self._save()

    def reset(self) -> None:
        self.default_action = "ask"
        self._rules.clear()
        self._next_id = 1
        self._save()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("policy must be an object")
            default = payload.get("default", "ask")
            raw_rules = payload.get("rules", [])
            if not isinstance(raw_rules, list) or any(
                not isinstance(item, dict)
                or not isinstance(item.get("command"), list)
                or any(
                    not isinstance(argument, str) or not argument
                    for argument in item["command"]
                )
                for item in raw_rules
            ):
                raise ValueError("rules must contain command arrays")
            rules = [
                CommandRule(
                    id=int(item["id"]),
                    action=str(item["action"]),
                    command=tuple(map(str, item["command"])),
                )
                for item in raw_rules
            ]
            if default not in POLICY_ACTIONS:
                raise ValueError("invalid default action")
            if any(
                rule.id <= 0
                or rule.action not in POLICY_ACTIONS
                or not rule.command
                for rule in rules
            ) or len({rule.id for rule in rules}) != len(rules):
                raise ValueError("invalid command rule")
        except (
            KeyError,
            OSError,
            TypeError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            self.load_error = f"Command policy file {self.path} is invalid: {error}"
            return
        self.default_action = default
        self._rules = rules
        self._next_id = max((rule.id for rule in rules), default=0) + 1

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "default": self.default_action,
            "rules": [
                {"id": rule.id, "action": rule.action, "command": rule.command}
                for rule in self.list()
            ],
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


@dataclass(slots=True)
class ExecutionLimits:
    """Resource limits applied to locally spawned project commands."""

    wall_time_seconds: float = 300
    output_bytes: int = 100_000
    cpu_seconds: int | None = None
    memory_mb: int | None = None
    file_size_mb: int | None = None
    processes: int | None = None

    def effective_timeout(self, requested: float | None = None) -> float:
        if requested is None:
            return self.wall_time_seconds
        return min(requested, self.wall_time_seconds)

    def subprocess_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if os.name == "posix":
            options["start_new_session"] = True
        if os.name != "posix" or not any(
            value is not None
            for value in (
                self.cpu_seconds,
                self.memory_mb,
                self.file_size_mb,
                self.processes,
            )
        ):
            return options

        cpu_seconds = self.cpu_seconds
        memory_mb = self.memory_mb
        file_size_mb = self.file_size_mb
        processes = self.processes

        def apply_limits() -> None:
            import resource

            if cpu_seconds is not None:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            if memory_mb is not None:
                memory_bytes = memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            if file_size_mb is not None:
                file_bytes = file_size_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
            if processes is not None:
                resource.setrlimit(resource.RLIMIT_NPROC, (processes, processes))

        options["preexec_fn"] = apply_limits
        return options


class ExecutionLimitStore:
    """Persist configurable local command limits."""

    DEFAULTS = ExecutionLimits()
    NAMES = {
        "timeout": "wall_time_seconds",
        "output": "output_bytes",
        "cpu": "cpu_seconds",
        "memory": "memory_mb",
        "file-size": "file_size_mb",
        "processes": "processes",
    }

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path.cwd() / ".ai-agent" / "execution-limits.json"
        self.limits = ExecutionLimits()
        self.load_error: str | None = None
        self._load()

    def set(self, name: str, value: str) -> None:
        field = self.NAMES.get(name)
        if field is None:
            raise ValueError(f"Unknown limit {name!r}")
        parsed = self._parse(name, value)
        setattr(self.limits, field, parsed)
        self._save()

    def reset(self) -> None:
        defaults = asdict(self.DEFAULTS)
        for field, value in defaults.items():
            setattr(self.limits, field, value)
        self._save()

    def report(self) -> str:
        values = {
            "timeout": f"{self.limits.wall_time_seconds:g}s",
            "output": _format_bytes(self.limits.output_bytes),
            "cpu": _optional(self.limits.cpu_seconds, "s"),
            "memory": _optional(self.limits.memory_mb, "MB"),
            "file-size": _optional(self.limits.file_size_mb, "MB"),
            "processes": _optional(self.limits.processes, ""),
        }
        return "Execution limits:\n" + "\n".join(
            f"{name}: {value}" for name, value in values.items()
        )

    def _parse(self, name: str, value: str) -> int | float | None:
        normalized = value.strip().lower()
        if normalized in {"off", "none"}:
            if name in {"timeout", "output"}:
                raise ValueError(f"{name} cannot be disabled")
            return None
        if name == "timeout":
            return parse_timeout(normalized)
        if name == "cpu":
            return math.ceil(parse_timeout(normalized))
        if name in {"memory", "file-size"}:
            size = parse_size(normalized)
            megabytes = math.ceil(size / (1024 * 1024))
            if megabytes < 1:
                raise ValueError(f"{name} must be at least 1MB")
            return megabytes
        if name == "output":
            size = parse_size(normalized)
            if size < 1024 or size > 100_000_000:
                raise ValueError("output must be between 1KB and 100MB")
            return size
        try:
            processes = int(normalized)
        except ValueError as error:
            raise ValueError("processes must be a whole number") from error
        if processes < 1 or processes > 4096:
            raise ValueError("processes must be between 1 and 4096")
        return processes

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("limits must be an object")
            loaded = ExecutionLimits()
            for name, field in self.NAMES.items():
                if name in payload:
                    setattr(loaded, field, self._parse(name, str(payload[name])))
        except (
            OSError,
            TypeError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            self.load_error = f"Execution limit file {self.path} is invalid: {error}"
            return
        self.limits = loaded

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timeout": f"{self.limits.wall_time_seconds}s",
            "output": f"{self.limits.output_bytes}b",
            "cpu": (
                f"{self.limits.cpu_seconds}s"
                if self.limits.cpu_seconds is not None
                else "off"
            ),
            "memory": (
                f"{self.limits.memory_mb}mb"
                if self.limits.memory_mb is not None
                else "off"
            ),
            "file-size": (
                f"{self.limits.file_size_mb}mb"
                if self.limits.file_size_mb is not None
                else "off"
            ),
            "processes": self.limits.processes or "off",
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


def parse_size(value: str) -> int:
    """Parse byte sizes such as 100kb, 32mb, or 1gb."""

    normalized = value.strip().lower()
    units = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3}
    for suffix in ("gb", "mb", "kb", "b"):
        if normalized.endswith(suffix):
            number = normalized[: -len(suffix)]
            try:
                size = float(number) * units[suffix]
            except ValueError as error:
                raise ValueError("Size must look like 100KB, 32MB, or 1GB") from error
            if not math.isfinite(size) or size <= 0:
                raise ValueError("Size must be greater than zero")
            return math.ceil(size)
    raise ValueError("Size must use B, KB, MB, or GB")


def native_command_is_simple(source: str) -> bool:
    """Return whether shell text is safe for argument-prefix auto-approval.

    Native Codex approvals contain shell source rather than a structured argv.
    Compound operators and active shell expansion must therefore fall back to a
    user prompt even if the leading words match an allow rule.
    """

    quote: str | None = None
    escaped = False
    for character in source:
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote != "'":
            escaped = True
            continue
        if character == "'" and quote != '"':
            quote = None if quote == "'" else "'"
            continue
        if character == '"' and quote != "'":
            quote = None if quote == '"' else '"'
            continue
        if quote != "'" and character in "$`":
            return False
        if quote is None and character in ";&|<>()\n\r":
            return False
    return True


def _optional(value: int | None, suffix: str) -> str:
    return "off" if value is None else f"{value}{suffix}"


def _format_bytes(value: int) -> str:
    if value % (1024 * 1024) == 0:
        return f"{value // (1024 * 1024)}MB"
    if value % 1024 == 0:
        return f"{value // 1024}KB"
    return f"{value}B"
