import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class AppearanceSettings:
    """Persisted TUI theme and accessibility preferences."""

    theme: str = "textual-dark"
    high_contrast: bool = False
    reduced_motion: bool = False
    density: str = "comfortable"


class AppearanceStore:
    """Load and atomically persist display preferences."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path.cwd() / ".ai-agent" / "appearance.json"
        self.settings = AppearanceSettings()
        self.load_error: str | None = None
        self._load()

    def update(self, name: str, value: str) -> AppearanceSettings:
        normalized = value.strip().lower()
        if name == "theme":
            if not normalized:
                raise ValueError("Theme cannot be empty")
            self.settings.theme = normalized
        elif name == "contrast":
            self.settings.high_contrast = _parse_toggle(normalized, "contrast")
        elif name == "motion":
            if normalized not in {"full", "reduced"}:
                raise ValueError("Motion must be full or reduced")
            self.settings.reduced_motion = normalized == "reduced"
        elif name == "density":
            if normalized not in {"compact", "comfortable"}:
                raise ValueError("Density must be compact or comfortable")
            self.settings.density = normalized
        else:
            raise ValueError(f"Unknown appearance setting {name!r}")
        self._save()
        return self.settings

    def report(self) -> str:
        settings = self.settings
        return (
            f"Theme: {settings.theme}\n"
            f"High contrast: {'on' if settings.high_contrast else 'off'}\n"
            f"Motion: {'reduced' if settings.reduced_motion else 'full'}\n"
            f"Density: {settings.density}"
        )

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            settings = AppearanceSettings(**payload)
            if not isinstance(settings.theme, str) or not settings.theme:
                raise ValueError("invalid theme")
            if not isinstance(settings.high_contrast, bool):
                raise ValueError("invalid contrast")
            if not isinstance(settings.reduced_motion, bool):
                raise ValueError("invalid motion")
            if settings.density not in {"compact", "comfortable"}:
                raise ValueError("invalid density")
        except (
            OSError,
            TypeError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            self.load_error = f"Appearance file {self.path} is invalid: {error}"
            return
        self.settings = settings

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(asdict(self.settings), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


def _parse_toggle(value: str, name: str) -> bool:
    if value not in {"on", "off"}:
        raise ValueError(f"{name.title()} must be on or off")
    return value == "on"
