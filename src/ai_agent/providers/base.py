from typing import Protocol, Sequence

from ai_agent.messages import ChatMessage


class ChatProvider(Protocol):
    """Minimal interface implemented by every model provider."""

    name: str
    model: str

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Return the assistant's response to a conversation."""
        ...
