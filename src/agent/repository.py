from abc import ABC, abstractmethod

from langchain_core.messages import AIMessage


class LLMRepository(ABC):
    @abstractmethod
    async def invoke(self, messages: list[dict[str, str]]) -> AIMessage: ...
