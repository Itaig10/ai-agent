from dataclasses import dataclass
import math
import os

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings sourced from environment variables."""

    provider: str
    model: str | None
    effort: str | None
    openai_api_key: str | None
    litellm_api_key: str | None
    litellm_api_base: str | None
    litellm_timeout_seconds: float
    litellm_verify_tls: bool
    context_window: int | None
    codex_path: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        provider = os.getenv("AI_AGENT_PROVIDER", "codex").strip().lower()
        model = os.getenv("AI_AGENT_MODEL", "").strip() or None
        effort = os.getenv("AI_AGENT_EFFORT", "").strip().lower() or None
        context_window_text = os.getenv("AI_AGENT_CONTEXT_WINDOW", "").strip()
        litellm_timeout_text = os.getenv(
            "AI_AGENT_LITELLM_TIMEOUT_SECONDS", "2"
        ).strip()
        litellm_verify_tls_text = os.getenv(
            "AI_AGENT_LITELLM_VERIFY_TLS", "true"
        ).strip().lower()

        if not provider:
            raise ValueError("AI_AGENT_PROVIDER cannot be empty")
        if effort not in {None, "low", "medium", "high", "xhigh"}:
            raise ValueError(
                "AI_AGENT_EFFORT must be low, medium, high, or xhigh"
            )
        try:
            context_window = int(context_window_text) if context_window_text else None
        except ValueError as error:
            raise ValueError("AI_AGENT_CONTEXT_WINDOW must be an integer") from error
        if context_window is not None and context_window <= 0:
            raise ValueError("AI_AGENT_CONTEXT_WINDOW must be greater than zero")
        try:
            litellm_timeout_seconds = float(litellm_timeout_text)
        except ValueError as error:
            raise ValueError(
                "AI_AGENT_LITELLM_TIMEOUT_SECONDS must be a number"
            ) from error
        if not math.isfinite(litellm_timeout_seconds) or litellm_timeout_seconds <= 0:
            raise ValueError(
                "AI_AGENT_LITELLM_TIMEOUT_SECONDS must be greater than zero"
            )
        if litellm_verify_tls_text not in {"true", "false"}:
            raise ValueError(
                "AI_AGENT_LITELLM_VERIFY_TLS must be true or false"
            )
        return cls(
            provider=provider,
            model=model,
            effort=effort,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            litellm_api_key=os.getenv("AI_AGENT_LITELLM_API_KEY"),
            litellm_api_base=os.getenv("AI_AGENT_LITELLM_API_BASE"),
            litellm_timeout_seconds=litellm_timeout_seconds,
            litellm_verify_tls=litellm_verify_tls_text == "true",
            context_window=context_window,
            codex_path=os.getenv("CODEX_PATH", "codex").strip() or "codex",
        )
