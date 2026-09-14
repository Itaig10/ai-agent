from collections.abc import Awaitable, Callable, Sequence
import importlib
from time import monotonic
from typing import Any

from ai_agent.events import TextDelta
from ai_agent.messages import ChatMessage
from ai_agent.metrics import UsageMetrics


CompletionFunction = Callable[..., Awaitable[Any]]
TextDeltaHandler = Callable[[TextDelta], Awaitable[None]]


class LiteLLMProvider:
    """Provider backed by LiteLLM's unified async completion API."""

    name = "litellm"

    def __init__(
        self,
        *,
        model: str,
        effort: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        context_window: int | None = None,
        completion_function: CompletionFunction | None = None,
        text_delta_handler: TextDeltaHandler | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("AI_AGENT_MODEL is required for the LiteLLM provider")
        if context_window is not None and context_window <= 0:
            raise ValueError("AI_AGENT_CONTEXT_WINDOW must be greater than zero")

        self.model = model.strip()
        self.effort = self._validate_effort(effort)
        self.api_key = api_key
        self.api_base = api_base
        self.context_window = context_window
        self.context_tokens = 0
        self.context_window_supported = context_window is not None
        self._completion = completion_function or self._load_completion()
        self._text_delta_handler = text_delta_handler
        self.last_metrics = UsageMetrics()
        self.session_metrics = UsageMetrics()

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        started = monotonic()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
        }
        if self.effort:
            request["reasoning_effort"] = self.effort
        if self.api_key:
            request["api_key"] = self.api_key
        if self.api_base:
            request["api_base"] = self.api_base

        if self._text_delta_handler is not None:
            content, response = await self._complete_streaming(request)
        else:
            response = await self._completion(**request)
            choices = self._get(response, "choices", [])
            if not choices:
                raise RuntimeError("LiteLLM returned no choices")
            message = self._get(choices[0], "message")
            content = self._get(message, "content")
        self._update_usage(self._get(response, "usage"))
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LiteLLM returned an empty response")
        self._record_metrics(response, round((monotonic() - started) * 1000))
        return content

    def set_event_handlers(
        self,
        *,
        text_delta: TextDeltaHandler | None = None,
        **_: Any,
    ) -> None:
        self._text_delta_handler = text_delta

    async def _complete_streaming(self, request: dict[str, Any]) -> tuple[str, Any]:
        stream = await self._completion(
            **request,
            stream=True,
            stream_options={"include_usage": True},
        )
        parts: list[str] = []
        final_chunk: Any = None
        async for chunk in stream:
            final_chunk = chunk
            choices = self._get(chunk, "choices", [])
            if choices:
                delta = self._get(self._get(choices[0], "delta"), "content", "") or ""
                parts.append(delta)
                if delta and self._text_delta_handler is not None:
                    await self._text_delta_handler(TextDelta(delta))
        if final_chunk is None:
            raise RuntimeError("LiteLLM stream ended without a response")
        return "".join(parts), final_chunk

    async def set_effort(self, effort: str | None) -> None:
        self.effort = self._validate_effort(effort)

    @property
    def context_percent(self) -> float | None:
        if not self.context_window:
            return None
        return min(100.0, self.context_tokens / self.context_window * 100)

    def _update_usage(self, usage: Any) -> None:
        if usage is None:
            return
        total = self._get(usage, "total_tokens", 0)
        self.context_tokens = int(total or 0)

    def _record_metrics(self, response: Any, latency_ms: int) -> None:
        usage = self._get(response, "usage")
        hidden = self._get(response, "_hidden_params", {})
        cost = self._get(hidden, "response_cost")
        self.last_metrics = UsageMetrics(
            input_tokens=int(self._get(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(self._get(usage, "completion_tokens", 0) or 0),
            total_tokens=int(self._get(usage, "total_tokens", 0) or 0),
            cost_usd=float(cost) if cost is not None else None,
            latency_ms=latency_ms,
            requests=1,
        )
        self.session_metrics.add(self.last_metrics)

    @staticmethod
    def _get(value: Any, key: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

    @staticmethod
    def _validate_effort(effort: str | None) -> str | None:
        if effort not in {None, "low", "medium", "high", "xhigh"}:
            raise ValueError("effort must be low, medium, high, or xhigh")
        return effort

    @staticmethod
    def _load_completion() -> CompletionFunction:
        try:
            module = importlib.import_module("litellm")
        except ImportError as error:
            raise ValueError(
                "LiteLLM is not installed. Run: python3 -m pip install -e ."
            ) from error
        return module.acompletion
