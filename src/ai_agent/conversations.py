from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any

from ai_agent.messages import ChatMessage


class ConversationStore:
    """Save and load named conversations as workspace-local JSON."""

    _VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or Path.cwd() / ".ai-agent" / "conversations"

    def save(
        self,
        name: str,
        messages: list[ChatMessage],
        *,
        provider: str,
        model: str,
    ) -> Path:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "name": name,
            "saved_at": datetime.now(UTC).isoformat(),
            "provider": provider,
            "model": model,
            "messages": [asdict(message) for message in messages],
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path

    def load(self, name: str) -> tuple[list[ChatMessage], dict[str, Any]]:
        path = self._path(name)
        if not path.exists():
            raise ValueError(f"Conversation {name!r} does not exist")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            messages = []
            for item in payload["messages"]:
                role = item["role"]
                content = item["content"]
                if role not in {"user", "assistant"} or not isinstance(content, str):
                    raise TypeError("invalid message")
                messages.append(ChatMessage(role=role, content=content))
        except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Conversation {name!r} is invalid") from error
        return messages, payload

    def list(self) -> list[str]:
        if not self.directory.exists():
            return []
        return sorted(path.stem for path in self.directory.glob("*.json"))

    def _path(self, name: str) -> Path:
        name = name.strip()
        if not self._VALID_NAME.fullmatch(name):
            raise ValueError(
                "Conversation names may contain letters, numbers, dots, dashes, "
                "and underscores"
            )
        return self.directory / f"{name}.json"
