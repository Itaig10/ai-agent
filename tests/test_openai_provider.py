import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from ai_agent.events import ToolActivity
from ai_agent.messages import ChatMessage
from ai_agent.providers.openai import OpenAIProvider
from ai_agent.tools import ToolRegistry


class OpenAIProviderTests(unittest.TestCase):
    def test_native_web_search_is_enabled_and_reported(self) -> None:
        activities: list[ToolActivity] = []

        async def record(activity: ToolActivity) -> None:
            activities.append(activity)

        provider = OpenAIProvider(
            api_key="test-key",
            model="test-model",
            activity_handler=record,
        )
        response = SimpleNamespace(
            output_text="answer",
            output=[SimpleNamespace(type="web_search_call")],
        )
        provider._client.responses.create = AsyncMock(return_value=response)

        result = asyncio.run(
            provider.complete([ChatMessage(role="user", content="latest news")])
        )

        self.assertEqual(result, "answer")
        request = provider._client.responses.create.await_args.kwargs
        self.assertEqual(request["tools"], [{"type": "web_search"}])
        self.assertEqual(activities[0].kind, "webSearch")
        self.assertEqual(activities[0].status, "completed")

    def test_response_is_streamed_and_metrics_are_recorded(self) -> None:
        deltas: list[str] = []

        async def record(delta: object) -> None:
            deltas.append(delta.text)

        async def events():
            yield SimpleNamespace(type="response.output_text.delta", delta="Hel")
            yield SimpleNamespace(type="response.output_text.delta", delta="lo")
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    output=[],
                    usage=SimpleNamespace(
                        input_tokens=10,
                        output_tokens=2,
                        total_tokens=12,
                    ),
                ),
            )

        provider = OpenAIProvider(
            api_key="test-key",
            model="test-model",
            text_delta_handler=record,
        )
        provider._client.responses.create = AsyncMock(return_value=events())

        result = asyncio.run(
            provider.complete([ChatMessage(role="user", content="Hello")])
        )

        self.assertEqual(result, "Hello")
        self.assertEqual(deltas, ["Hel", "lo"])
        self.assertEqual(provider.last_metrics.total_tokens, 12)
        self.assertEqual(provider.session_metrics.requests, 1)

    def test_close_releases_sdk_client(self) -> None:
        provider = OpenAIProvider(api_key="test-key", model="test-model")
        provider._client.close = AsyncMock()

        asyncio.run(provider.close())

        provider._client.close.assert_awaited_once_with()

    def test_effort_is_sent_and_disabled_web_search_is_omitted(self) -> None:
        registry = ToolRegistry()
        registry.set_enabled("web_search", False)
        provider = OpenAIProvider(
            api_key="test-key",
            model="test-model",
            effort="high",
            tool_registry=registry,
        )
        response = SimpleNamespace(
            output_text="answer",
            output=[],
            usage=SimpleNamespace(
                input_tokens=3,
                output_tokens=2,
                total_tokens=5,
            ),
        )
        provider._client.responses.create = AsyncMock(return_value=response)

        result = asyncio.run(
            provider.complete([ChatMessage(role="user", content="Hello")])
        )

        request = provider._client.responses.create.await_args.kwargs
        self.assertEqual(result, "answer")
        self.assertNotIn("tools", request)
        self.assertEqual(request["reasoning"], {"effort": "high"})
        self.assertEqual(provider.last_metrics.total_tokens, 5)

    def test_stream_without_completed_event_is_rejected(self) -> None:
        async def events():
            yield SimpleNamespace(type="response.output_text.delta", delta="partial")

        provider = OpenAIProvider(
            api_key="test-key",
            model="test-model",
            text_delta_handler=AsyncMock(),
        )
        provider._client.responses.create = AsyncMock(return_value=events())

        with self.assertRaisesRegex(RuntimeError, "without a completed response"):
            asyncio.run(
                provider.complete([ChatMessage(role="user", content="Hello")])
            )

    def test_invalid_effort_change_is_rejected(self) -> None:
        provider = OpenAIProvider(api_key="test-key", model="test-model")

        with self.assertRaisesRegex(ValueError, "effort must be"):
            asyncio.run(provider.set_effort("extreme"))


if __name__ == "__main__":
    unittest.main()
