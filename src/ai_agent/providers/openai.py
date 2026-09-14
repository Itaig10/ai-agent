from collections.abc import Awaitable, Callable, Sequence
from time import monotonic
from typing import Any

from openai import AsyncOpenAI

from ai_agent.events import TextDelta, ToolActivity
from ai_agent.messages import ChatMessage
from ai_agent.metrics import UsageMetrics
from ai_agent.tools import ToolRegistry


ActivityHandler = Callable[[ToolActivity], Awaitable[None]]
TextDeltaHandler = Callable[[TextDelta], Awaitable[None]]


class OpenAIProvider:
    """OpenAI implementation backed by the Responses API."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        effort: str | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        activity_handler: ActivityHandler | None = None,
        text_delta_handler: TextDeltaHandler | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.system_prompt = system_prompt
        self._activity_handler = activity_handler
        self._text_delta_handler = text_delta_handler
        self.tool_registry = tool_registry or ToolRegistry()
        self.last_metrics = UsageMetrics()
        self.session_metrics = UsageMetrics()
        self._client = AsyncOpenAI(api_key=api_key)

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        started = monotonic()
        request = {
            "model": self.model,
            "instructions": self.system_prompt,
            "input": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "store": False,
        }
        if self.tool_registry.is_enabled("web_search", self.name):
            request["tools"] = [{"type": "web_search"}]
        if self.effort:
            request["reasoning"] = {"effort": self.effort}
        if self._text_delta_handler is not None:
            text, response = await self._complete_streaming(request)
        else:
            response = await self._client.responses.create(**request)
            text = response.output_text
        await self._emit_web_search_activity(response)
        self._record_metrics(response, round((monotonic() - started) * 1000))
        return text

    def set_event_handlers(
        self,
        *,
        activity: ActivityHandler,
        text_delta: TextDeltaHandler | None = None,
        **_: Any,
    ) -> None:
        self._activity_handler = activity
        self._text_delta_handler = text_delta

    async def _complete_streaming(self, request: dict[str, Any]) -> tuple[str, Any]:
        stream = await self._client.responses.create(**request, stream=True)
        parts: list[str] = []
        final_response: Any = None
        async for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                parts.append(delta)
                if delta and self._text_delta_handler is not None:
                    await self._text_delta_handler(TextDelta(delta))
            elif event_type == "response.completed":
                final_response = getattr(event, "response", None)
        if final_response is None:
            raise RuntimeError("OpenAI stream ended without a completed response")
        return "".join(parts), final_response

    def _record_metrics(self, response: Any, latency_ms: int) -> None:
        usage = getattr(response, "usage", None)
        self.last_metrics = UsageMetrics(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            latency_ms=latency_ms,
            requests=1,
        )
        self.session_metrics.add(self.last_metrics)

    async def _emit_web_search_activity(self, response: Any) -> None:
        if self._activity_handler is None:
            return
        for item in getattr(response, "output", []):
            item_type = (
                item.get("type")
                if isinstance(item, dict)
                else getattr(item, "type", "")
            )
            if item_type == "web_search_call":
                await self._activity_handler(
                    ToolActivity(
                        kind="webSearch",
                        status="completed",
                        title="Web search",
                        detail="OpenAI native web search",
                    )
                )

    async def set_effort(self, effort: str | None) -> None:
        if effort not in {None, "low", "medium", "high", "xhigh"}:
            raise ValueError("effort must be low, medium, high, or xhigh")
        self.effort = effort

    async def close(self) -> None:
        """Release pooled HTTP connections owned by the async SDK client."""

        await self._client.close()
