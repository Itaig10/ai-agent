from ai_agent.messages import ChatMessage
from ai_agent.providers.base import ChatProvider


class ChatSession:
    """Owns provider-independent conversation state."""

    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider
        self.messages: list[ChatMessage] = []

    async def send(self, prompt: str, *, provider_prompt: str | None = None) -> str:
        user_message, reply = await self._complete(
            prompt,
            provider_prompt=provider_prompt,
        )

        self.messages.extend(
            [user_message, ChatMessage(role="assistant", content=reply)]
        )
        return reply

    async def complete_transient(self, prompt: str) -> str:
        """Complete a prompt without adding orchestration text to chat history."""

        _, reply = await self._complete(prompt)
        return reply

    async def _complete(
        self,
        prompt: str,
        *,
        provider_prompt: str | None = None,
    ) -> tuple[ChatMessage, str]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Message cannot be empty")

        user_message = ChatMessage(role="user", content=prompt)
        provider_content = (
            provider_prompt.strip() if provider_prompt is not None else prompt
        )
        if not provider_content:
            raise ValueError("Provider message cannot be empty")
        provider_message = ChatMessage(role="user", content=provider_content)
        reply = await self.provider.complete([*self.messages, provider_message])
        reply = reply.strip()
        if not reply:
            raise RuntimeError("The provider returned an empty response")
        return user_message, reply

    def clear(self) -> None:
        self.messages.clear()

    def restore(self, messages: list[ChatMessage]) -> None:
        self.messages = list(messages)

    def replace_provider(self, provider: ChatProvider) -> None:
        self.provider = provider
