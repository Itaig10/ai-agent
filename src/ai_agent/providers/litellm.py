import asyncio
from collections.abc import Awaitable, Callable, Sequence
import importlib
import json
import math
import os
import ssl
from time import monotonic
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from ai_agent.events import TextDelta
from ai_agent.messages import ChatMessage
from ai_agent.metrics import UsageMetrics


CompletionFunction = Callable[..., Awaitable[Any]]
TextDeltaHandler = Callable[[TextDelta], Awaitable[None]]
MAX_MODEL_LIST_BYTES = 2 * 1024 * 1024


async def fetch_litellm_models(
    *,
    api_base: str | None,
    api_key: str | None = None,
    timeout_seconds: float = 2.0,
    verify_tls: bool = True,
) -> tuple[str, ...]:
    """Fetch model IDs exposed by a configured LiteLLM proxy."""
    if not api_base or not api_base.strip():
        raise ValueError(
            "AI_AGENT_LITELLM_API_BASE is required to discover LiteLLM models"
        )
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("LiteLLM timeout must be greater than zero")

    base = api_base.strip().rstrip("/")
    if urlsplit(base).scheme not in {"http", "https"}:
        raise ValueError("AI_AGENT_LITELLM_API_BASE must use http or https")
    return await asyncio.to_thread(
        _fetch_litellm_models_sync,
        f"{base}/models",
        api_key,
        timeout_seconds,
        verify_tls,
    )


def _fetch_litellm_models_sync(
    url: str,
    api_key: str | None,
    timeout_seconds: float,
    verify_tls: bool = True,
) -> tuple[str, ...]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, headers=headers, method="GET")
    context = None
    if not verify_tls:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    with urlopen(
        request,
        timeout=timeout_seconds,
        context=context,
    ) as response:
        raw = response.read(MAX_MODEL_LIST_BYTES + 1)
    if len(raw) > MAX_MODEL_LIST_BYTES:
        raise RuntimeError("LiteLLM model list exceeded the 2 MiB limit")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("LiteLLM returned an invalid model list") from error
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise RuntimeError("LiteLLM model list is missing the data array")
    model_ids = {
        model_id.strip()
        for item in data
        if isinstance(item, dict)
        and isinstance((model_id := item.get("id")), str)
        and model_id.strip()
    }
    return tuple(sorted(model_ids, key=str.casefold))


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
        timeout_seconds: float = 2.0,
        verify_tls: bool = True,
        context_window: int | None = None,
        completion_function: CompletionFunction | None = None,
        text_delta_handler: TextDeltaHandler | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("AI_AGENT_MODEL is required for the LiteLLM provider")
        if context_window is not None and context_window <= 0:
            raise ValueError("AI_AGENT_CONTEXT_WINDOW must be greater than zero")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("LiteLLM timeout must be greater than zero")

        self.model = model.strip()
        self.effort = self._validate_effort(effort)
        self.api_key = api_key
        self.api_base = api_base
        self.timeout_seconds = timeout_seconds
        self.verify_tls = verify_tls
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
            "timeout": self.timeout_seconds,
            "ssl_verify": self.verify_tls,
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
        # LiteLLM otherwise downloads its model-cost map during import. Use the
        # bundled map so a slow or unavailable URL cannot delay app startup.
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        try:
            module = importlib.import_module("litellm")
        except ImportError as error:
            raise ValueError(
                "LiteLLM is not installed. Run: python3 -m pip install -e ."
            ) from error
        return module.acompletion
