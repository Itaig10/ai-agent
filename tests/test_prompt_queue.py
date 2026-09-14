import unittest

from ai_agent.prompt_queue import PromptQueue


class PromptQueueTests(unittest.TestCase):
    def test_fifo_order_and_stable_ids(self) -> None:
        queue = PromptQueue()
        first = queue.enqueue("first")
        second = queue.enqueue("second")

        self.assertEqual((first.id, second.id), (1, 2))
        self.assertEqual(queue.pop(), first)
        self.assertEqual(queue.pop(), second)
        self.assertIsNone(queue.pop())

    def test_remove_and_clear(self) -> None:
        queue = PromptQueue()
        first = queue.enqueue("first")
        queue.enqueue("second")

        self.assertEqual(queue.remove(first.id), first)
        self.assertEqual(queue.clear(), 1)
        self.assertEqual(len(queue), 0)

    def test_empty_prompt_and_unknown_id_are_rejected(self) -> None:
        queue = PromptQueue()

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            queue.enqueue("   ")
        with self.assertRaisesRegex(ValueError, "does not exist"):
            queue.remove(999)

    def test_list_returns_a_snapshot(self) -> None:
        queue = PromptQueue()
        expected = queue.enqueue("first")

        snapshot = queue.list()
        snapshot.clear()

        self.assertEqual(queue.list(), [expected])


if __name__ == "__main__":
    unittest.main()
