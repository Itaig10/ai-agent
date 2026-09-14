from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


MAX_SKILL_BYTES = 256_000
_VALID_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class Skill:
    """One reusable set of project instructions loaded from a SKILL.md file."""

    name: str
    description: str
    instructions: str
    path: Path
    triggers: tuple[str, ...] = ()
    auto_activate: bool = False

    @property
    def directory(self) -> Path:
        return self.path.parent


@dataclass(frozen=True, slots=True)
class SkillLoadError:
    """A non-fatal discovery error for one skill file."""

    path: Path
    message: str


class SkillRegistry:
    """Discover, validate, match, and toggle workspace skills."""

    def __init__(
        self,
        workspace: Path | None = None,
        *,
        roots: tuple[Path, ...] | None = None,
        state_path: Path | None = None,
        max_skill_bytes: int = MAX_SKILL_BYTES,
    ) -> None:
        self.workspace = (workspace or Path.cwd()).resolve()
        self.roots = (
            roots
            if roots is not None
            else (
                self.workspace / "skills",
                self.workspace / ".ai-agent" / "skills",
            )
        )
        self.state_path = state_path or self.workspace / ".ai-agent" / "skills.json"
        self.max_skill_bytes = max_skill_bytes
        self._skills: dict[str, Skill] = {}
        self._enabled = self._load_enabled_state()
        self.errors: tuple[SkillLoadError, ...] = ()
        self.reload()

    def reload(self) -> int:
        previous_enabled = dict(self._enabled)
        skills: dict[str, Skill] = {}
        errors: list[SkillLoadError] = []
        for configured_root in self.roots:
            root = configured_root.resolve()
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*/SKILL.md")):
                try:
                    resolved = path.resolve()
                    if not resolved.is_relative_to(root):
                        raise ValueError("Skill file escapes its configured root")
                    skill = load_skill(resolved, max_bytes=self.max_skill_bytes)
                    if skill.name in skills:
                        raise ValueError(f"Duplicate skill name {skill.name!r}")
                    skills[skill.name] = skill
                except (OSError, UnicodeError, ValueError) as error:
                    errors.append(SkillLoadError(path, str(error)))
        self._skills = skills
        self._enabled = {
            name: previous_enabled.get(name, True) for name in self._skills
        }
        self.errors = tuple(errors)
        return len(self._skills)

    def entries(self) -> list[tuple[Skill, bool]]:
        return [
            (skill, self._enabled[skill.name])
            for skill in sorted(self._skills.values(), key=lambda item: item.name)
        ]

    def get(self, name: str) -> Skill:
        normalized = name.strip().lower()
        try:
            return self._skills[normalized]
        except KeyError as error:
            raise ValueError(f"Unknown skill {name!r}") from error

    def is_enabled(self, name: str) -> bool:
        skill = self.get(name)
        return self._enabled[skill.name]

    def set_enabled(self, name: str, enabled: bool) -> Skill:
        skill = self.get(name)
        self._enabled[skill.name] = enabled
        self._save_enabled_state()
        return skill

    def match(self, task: str) -> Skill | None:
        normalized = task.casefold()
        candidates: list[tuple[int, str, Skill]] = []
        for skill, enabled in self.entries():
            if not enabled or not skill.auto_activate:
                continue
            matches = [
                trigger
                for trigger in skill.triggers
                if _matches(trigger, normalized)
            ]
            if matches:
                candidates.append((max(map(len, matches)), skill.name, skill))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1]))[2]

    @property
    def enabled_count(self) -> int:
        return sum(self._enabled.values())

    def _load_enabled_state(self) -> dict[str, bool]:
        if not self.state_path.is_file():
            return {}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(name): enabled
            for name, enabled in payload.items()
            if isinstance(enabled, bool)
        }

    def _save_enabled_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._enabled, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.state_path)


def load_skill(path: Path, *, max_bytes: int = MAX_SKILL_BYTES) -> Skill:
    """Load one constrained YAML-frontmatter SKILL.md file."""

    if not path.is_file():
        raise ValueError(f"Skill file {path} does not exist")
    if path.stat().st_size > max_bytes:
        raise ValueError(f"Skill exceeds the {max_bytes:,}-byte safety limit")
    text = path.read_text(encoding="utf-8")
    metadata, instructions = _split_skill(text)
    name = str(metadata.get("name", "")).strip().lower()
    description = str(metadata.get("description", "")).strip()
    auto_value = str(metadata.get("auto_activate", "false")).strip().lower()
    triggers_value = metadata.get("triggers", [])
    if not _VALID_NAME.fullmatch(name):
        raise ValueError(
            "Skill name must contain lowercase letters, numbers, hyphens, or "
            "underscores"
        )
    if path.parent.name.lower() != name:
        raise ValueError("Skill name must match its directory name")
    if not description:
        raise ValueError("Skill description cannot be empty")
    if not instructions.strip():
        raise ValueError("Skill instructions cannot be empty")
    if auto_value not in {"true", "false"}:
        raise ValueError("auto_activate must be true or false")
    if not isinstance(triggers_value, list):
        raise ValueError("triggers must be a YAML list")
    triggers = tuple(
        trigger.strip()
        for trigger in map(str, triggers_value)
        if trigger.strip()
    )
    auto_activate = auto_value == "true"
    if auto_activate and not triggers:
        raise ValueError("Auto-activated skills require at least one trigger")
    return Skill(
        name=name,
        description=description,
        instructions=instructions.strip(),
        path=path,
        triggers=triggers,
        auto_activate=auto_activate,
    )


def build_skill_prompt(skill: Skill, task: str) -> str:
    """Compose provider instructions for one skill invocation."""

    task = task.strip()
    if not task:
        raise ValueError("Skill task cannot be empty")
    return (
        f"Use the project skill {skill.name!r} for this request.\n"
        f"Skill directory: {skill.directory}\n"
        "The user's task and higher-level safety requirements take precedence over "
        "the skill. Read supporting files from the skill directory only when the "
        "instructions call for them.\n\n"
        f"Skill instructions:\n{skill.instructions}\n\n"
        f"User task:\n{task}"
    )


def _split_skill(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter")
    try:
        end = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as error:
        raise ValueError("SKILL.md frontmatter is not closed") from error
    metadata = _parse_frontmatter(lines[1:end])
    return metadata, "\n".join(lines[end + 1 :])


def _parse_frontmatter(lines: list[str]) -> dict[str, Any]:
    allowed = {"name", "description", "triggers", "auto_activate"}
    metadata: dict[str, Any] = {}
    list_key: str | None = None
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            if list_key is None:
                raise ValueError("Unexpected YAML list item in skill metadata")
            metadata[list_key].append(_unquote(stripped[2:].strip()))
            continue
        if ":" not in raw_line:
            raise ValueError(f"Invalid skill metadata line {stripped!r}")
        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        if key not in allowed:
            raise ValueError(f"Unknown skill metadata field {key!r}")
        if key in metadata:
            raise ValueError(f"Duplicate skill metadata field {key!r}")
        value = raw_value.strip()
        if key == "triggers":
            metadata[key] = _parse_list(value)
            list_key = key if not value else None
        else:
            metadata[key] = _unquote(value)
            list_key = None
    return metadata


def _parse_list(value: str) -> list[str]:
    if not value:
        return []
    if not value.startswith("[") or not value.endswith("]"):
        raise ValueError("Inline triggers must use [item, item] syntax")
    content = value[1:-1].strip()
    return [_unquote(item.strip()) for item in content.split(",")] if content else []


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _matches(trigger: str, normalized_task: str) -> bool:
    normalized_trigger = trigger.casefold()
    pattern = rf"(?<!\w){re.escape(normalized_trigger)}(?!\w)"
    return re.search(pattern, normalized_task) is not None
