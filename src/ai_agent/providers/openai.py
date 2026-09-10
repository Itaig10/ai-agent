from collections.abc import Sequence

from openai import AsyncOpenAI

from ai_agent.messages import ChatMessage


class OpenAIProvider:
    """OpenAI implementation backed by the Responses API."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self._client = AsyncOpenAI(api_key=api_key)

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        response = await self._client.responses.create(
            model=self.model,
            instructions=self.system_prompt,
            input=[
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            store=False,
        )
        return response.output_text
