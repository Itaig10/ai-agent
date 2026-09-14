import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ai_agent.conversations import ConversationStore
from ai_agent.messages import ChatMessage


class ConversationStoreTests(unittest.TestCase):
    def test_round_trip_and_listing(self) -> None:
        with TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory))
            messages = [
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi"),
            ]

            path = store.save(
                "demo-session",
                messages,
                provider="codex",
                model="account default",
            )
            loaded, metadata = store.load("demo-session")

            self.assertTrue(path.exists())
            self.assertEqual(loaded, messages)
            self.assertEqual(metadata["provider"], "codex")
            self.assertEqual(store.list(), ["demo-session"])

    def test_invalid_name_cannot_escape_store(self) -> None:
        with TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory))

            with self.assertRaisesRegex(ValueError, "Conversation names"):
                store.save(
                    "../outside",
                    [],
                    provider="codex",
                    model="default",
                )

    def test_invalid_saved_messages_are_rejected(self) -> None:
        invalid_messages = [
            [{"role": "system", "content": "not allowed"}],
            [{"role": "user", "content": 123}],
        ]
        with TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory))
            path = Path(directory) / "invalid.json"
            for messages in invalid_messages:
                with self.subTest(messages=messages):
                    path.write_text(
                        json.dumps({"messages": messages}),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, "is invalid"):
                        store.load("invalid")

    def test_malformed_json_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory))
            (Path(directory) / "broken.json").write_text("{", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "is invalid"):
                store.load("broken")

    def test_listing_ignores_atomic_temporary_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "saved.json").write_text("{}", encoding="utf-8")
            (root / "pending.json.tmp").write_text("{}", encoding="utf-8")

            self.assertEqual(ConversationStore(root).list(), ["saved"])


if __name__ == "__main__":
    unittest.main()
