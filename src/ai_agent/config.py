from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings sourced from environment variables."""

    provider: str
    model: str
    openai_api_key: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        provider = os.getenv("AI_AGENT_PROVIDER", "openai").strip().lower()
        model = os.getenv("AI_AGENT_MODEL", "gpt-5.5").strip()

        if not provider:
            raise ValueError("AI_AGENT_PROVIDER cannot be empty")
        if not model:
            raise ValueError("AI_AGENT_MODEL cannot be empty")

        return cls(
            provider=provider,
            model=model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
