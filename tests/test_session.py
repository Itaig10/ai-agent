import unittest

from ai_agent.messages import ChatMessage
from ai_agent.session import ChatSession


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, reply: str = "Hello from the provider") -> None:
        self.reply = reply
        self.received: list[ChatMessage] = []

    async def complete(self, messages: list[ChatMessage]) -> str:
        self.received = list(messages)
        return self.reply


class ChatSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_stores_both_sides_of_conversation(self) -> None:
        provider = FakeProvider()
        session = ChatSession(provider)

        reply = await session.send("Hello")

        self.assertEqual(reply, "Hello from the provider")
        self.assertEqual(
            session.messages,
            [
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hello from the provider"),
            ],
        )

    async def test_send_passes_existing_history_to_provider(self) -> None:
        provider = FakeProvider(reply="Second answer")
        session = ChatSession(provider)
        session.messages = [
            ChatMessage(role="user", content="First question"),
            ChatMessage(role="assistant", content="First answer"),
        ]

        await session.send("Second question")

        self.assertEqual(
            provider.received,
            [
                ChatMessage(role="user", content="First question"),
                ChatMessage(role="assistant", content="First answer"),
                ChatMessage(role="user", content="Second question"),
            ],
        )

    async def test_failed_request_does_not_change_history(self) -> None:
        class FailingProvider(FakeProvider):
            async def complete(self, messages: list[ChatMessage]) -> str:
                raise RuntimeError("provider failed")

        session = ChatSession(FailingProvider())

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            await session.send("Hello")

        self.assertEqual(session.messages, [])

    async def test_empty_message_is_rejected(self) -> None:
        session = ChatSession(FakeProvider())

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            await session.send("   ")

    def test_clear_removes_history(self) -> None:
        session = ChatSession(FakeProvider())
        session.messages.append(ChatMessage(role="user", content="Hello"))

        session.clear()

        self.assertEqual(session.messages, [])


if __name__ == "__main__":
    unittest.main()
