from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings sourced from environment variables."""

    provider: str
    model: str | None
    openai_api_key: str | None
    codex_path: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        provider = os.getenv("AI_AGENT_PROVIDER", "codex").strip().lower()
        model = os.getenv("AI_AGENT_MODEL", "").strip() or None

        if not provider:
            raise ValueError("AI_AGENT_PROVIDER cannot be empty")
        return cls(
            provider=provider,
            model=model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            codex_path=os.getenv("CODEX_PATH", "codex").strip() or "codex",
        )
