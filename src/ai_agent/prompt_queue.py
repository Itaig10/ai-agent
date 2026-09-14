from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class QueuedPrompt:
    """One user prompt waiting for the active turn to finish."""

    id: int
    text: str
    enqueued_at: datetime


class PromptQueue:
    """A small in-memory FIFO queue with stable item IDs."""

    def __init__(self) -> None:
        self._items: list[QueuedPrompt] = []
        self._next_id = 1

    def enqueue(self, text: str) -> QueuedPrompt:
        text = text.strip()
        if not text:
            raise ValueError("Queued prompt cannot be empty")
        item = QueuedPrompt(self._next_id, text, datetime.now(UTC))
        self._next_id += 1
        self._items.append(item)
        return item

    def pop(self) -> QueuedPrompt | None:
        return self._items.pop(0) if self._items else None

    def list(self) -> list[QueuedPrompt]:
        return list(self._items)

    def remove(self, item_id: int) -> QueuedPrompt:
        for index, item in enumerate(self._items):
            if item.id == item_id:
                return self._items.pop(index)
        raise ValueError(f"Queued prompt {item_id} does not exist")

    def clear(self) -> int:
        count = len(self._items)
        self._items.clear()
        return count

    def __len__(self) -> int:
        return len(self._items)
