from ai_agent.messages import ChatMessage
from ai_agent.providers.base import ChatProvider


class ChatSession:
    """Owns provider-independent conversation state."""

    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider
        self.messages: list[ChatMessage] = []

    async def send(self, prompt: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Message cannot be empty")

        user_message = ChatMessage(role="user", content=prompt)
        pending_messages = [*self.messages, user_message]
        reply = await self.provider.complete(pending_messages)
        reply = reply.strip()
        if not reply:
            raise RuntimeError("The provider returned an empty response")

        self.messages.extend(
            [user_message, ChatMessage(role="assistant", content=reply)]
        )
        return reply

    def clear(self) -> None:
        self.messages.clear()
