import unittest
from types import SimpleNamespace

from ai_agent.messages import ChatMessage
from ai_agent.providers.litellm import LiteLLMProvider


class LiteLLMProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_completion_uses_unified_chat_format(self) -> None:
        received = {}

        async def complete(**request: object) -> object:
            received.update(request)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Hello from LiteLLM")
                    )
                ],
                usage=SimpleNamespace(total_tokens=250),
            )

        provider = LiteLLMProvider(
            model="anthropic/example-model",
            effort="high",
            api_key="test-key",
            api_base="https://example.test",
            context_window=1000,
            completion_function=complete,
        )

        reply = await provider.complete(
            [ChatMessage(role="user", content="Hello")]
        )

        self.assertEqual(reply, "Hello from LiteLLM")
        self.assertEqual(received["model"], "anthropic/example-model")
        self.assertEqual(received["reasoning_effort"], "high")
        self.assertEqual(received["api_key"], "test-key")
        self.assertEqual(received["api_base"], "https://example.test")
        self.assertEqual(
            received["messages"],
            [{"role": "user", "content": "Hello"}],
        )
        self.assertEqual(provider.context_percent, 25.0)

    async def test_dictionary_response_is_supported(self) -> None:
        async def complete(**_request: object) -> object:
            return {
                "choices": [{"message": {"content": "Dictionary response"}}],
                "usage": {"total_tokens": 10},
            }

        provider = LiteLLMProvider(
            model="ollama/example",
            completion_function=complete,
        )

        reply = await provider.complete(
            [ChatMessage(role="user", content="Hello")]
        )

        self.assertEqual(reply, "Dictionary response")
        self.assertFalse(provider.context_window_supported)
        self.assertIsNone(provider.context_percent)

    async def test_effort_can_change(self) -> None:
        async def complete(**_request: object) -> object:
            return {}

        provider = LiteLLMProvider(
            model="openai/example",
            completion_function=complete,
        )

        await provider.set_effort("medium")

        self.assertEqual(provider.effort, "medium")

    async def test_empty_response_is_rejected(self) -> None:
        async def complete(**_request: object) -> object:
            return {"choices": []}

        provider = LiteLLMProvider(
            model="openai/example",
            completion_function=complete,
        )

        with self.assertRaisesRegex(RuntimeError, "no choices"):
            await provider.complete([ChatMessage(role="user", content="Hello")])

    async def test_blank_message_content_is_rejected(self) -> None:
        async def complete(**_request: object) -> object:
            return {"choices": [{"message": {"content": "   "}}]}

        provider = LiteLLMProvider(
            model="openai/example",
            completion_function=complete,
        )

        with self.assertRaisesRegex(RuntimeError, "empty response"):
            await provider.complete([ChatMessage(role="user", content="Hello")])

    def test_model_and_context_window_are_validated(self) -> None:
        async def complete(**_request: object) -> object:
            return {}

        with self.assertRaisesRegex(ValueError, "AI_AGENT_MODEL"):
            LiteLLMProvider(model="", completion_function=complete)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            LiteLLMProvider(
                model="openai/example",
                context_window=0,
                completion_function=complete,
            )
        with self.assertRaisesRegex(ValueError, "effort must be"):
            LiteLLMProvider(
                model="openai/example",
                effort="extreme",
                completion_function=complete,
            )

    async def test_streaming_and_cost_metrics(self) -> None:
        deltas: list[str] = []

        async def record(delta: object) -> None:
            deltas.append(delta.text)

        async def chunks():
            yield {
                "choices": [{"delta": {"content": "Hi"}}],
                "usage": None,
            }
            yield {
                "choices": [],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 1,
                    "total_tokens": 5,
                },
                "_hidden_params": {"response_cost": 0.001},
            }

        async def complete(**_request: object) -> object:
            return chunks()

        provider = LiteLLMProvider(
            model="openai/example",
            completion_function=complete,
            text_delta_handler=record,
        )

        result = await provider.complete([ChatMessage(role="user", content="Hello")])

        self.assertEqual(result, "Hi")
        self.assertEqual(deltas, ["Hi"])
        self.assertEqual(provider.last_metrics.total_tokens, 5)
        self.assertEqual(provider.last_metrics.cost_usd, 0.001)

    async def test_empty_stream_is_rejected(self) -> None:
        async def chunks():
            if False:
                yield None

        async def complete(**_request: object) -> object:
            return chunks()

        provider = LiteLLMProvider(
            model="openai/example",
            completion_function=complete,
            text_delta_handler=lambda _delta: None,
        )

        with self.assertRaisesRegex(RuntimeError, "without a response"):
            await provider.complete([ChatMessage(role="user", content="Hello")])

    async def test_context_percentage_is_capped_and_invalid_effort_rejected(
        self,
    ) -> None:
        async def complete(**_request: object) -> object:
            return {
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"total_tokens": 150},
            }

        provider = LiteLLMProvider(
            model="openai/example",
            context_window=100,
            completion_function=complete,
        )
        await provider.complete([ChatMessage(role="user", content="Hello")])

        self.assertEqual(provider.context_percent, 100.0)
        with self.assertRaisesRegex(ValueError, "effort must be"):
            await provider.set_effort("extreme")


if __name__ == "__main__":
    unittest.main()
