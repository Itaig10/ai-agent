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

    async def test_provider_prompt_is_used_but_not_saved_to_history(self) -> None:
        provider = FakeProvider()
        session = ChatSession(provider)

        await session.send(
            "Review this change",
            provider_prompt="Hidden skill instructions\n\nReview this change",
        )

        self.assertEqual(
            provider.received[-1],
            ChatMessage(
                role="user",
                content="Hidden skill instructions\n\nReview this change",
            ),
        )
        self.assertEqual(
            session.messages[0],
            ChatMessage(role="user", content="Review this change"),
        )

    async def test_failed_request_does_not_change_history(self) -> None:
        class FailingProvider(FakeProvider):
            async def complete(self, messages: list[ChatMessage]) -> str:
                raise RuntimeError("provider failed")

        session = ChatSession(FailingProvider())

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            await session.send("Hello")

        self.assertEqual(session.messages, [])

    async def test_transient_completion_does_not_change_history(self) -> None:
        provider = FakeProvider(reply="Goal progress")
        session = ChatSession(provider)
        original = [
            ChatMessage(role="user", content="Normal question"),
            ChatMessage(role="assistant", content="Normal answer"),
        ]
        session.messages = list(original)

        reply = await session.complete_transient("internal goal instruction")

        self.assertEqual(reply, "Goal progress")
        self.assertEqual(session.messages, original)
        self.assertEqual(
            provider.received[-1],
            ChatMessage(role="user", content="internal goal instruction"),
        )

    async def test_empty_message_is_rejected(self) -> None:
        session = ChatSession(FakeProvider())

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            await session.send("   ")

    async def test_empty_provider_response_does_not_change_history(self) -> None:
        session = ChatSession(FakeProvider(reply="   "))

        with self.assertRaisesRegex(RuntimeError, "empty response"):
            await session.send("Hello")

        self.assertEqual(session.messages, [])

    def test_clear_removes_history(self) -> None:
        session = ChatSession(FakeProvider())
        session.messages.append(ChatMessage(role="user", content="Hello"))

        session.clear()

        self.assertEqual(session.messages, [])

    def test_restore_copies_messages_and_provider_can_be_replaced(self) -> None:
        first = FakeProvider()
        second = FakeProvider()
        session = ChatSession(first)
        restored = [ChatMessage(role="user", content="Hello")]

        session.restore(restored)
        session.replace_provider(second)
        restored.clear()

        self.assertEqual(
            session.messages,
            [ChatMessage(role="user", content="Hello")],
        )
        self.assertIs(session.provider, second)


if __name__ == "__main__":
    unittest.main()
