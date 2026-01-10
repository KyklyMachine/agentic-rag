import os
from typing import override

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.agent.repository import LLMRepository


class LLMConfig(BaseModel):
    model_name: str

class OpenrouterLLM(LLMRepository):
    _model: ChatOpenAI

    @override
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self._model = ChatOpenAI(
            model=config.model_name,
            api_key=os.environ.get("OPENROUTER_TOKEN"), # type: ignore
            base_url="https://openrouter.ai/api/v1"
        )

    @override
    async def invoke(self, messages: list[dict[str, str]]) -> AIMessage: 
        response = await self._model.ainvoke(
            input=messages,
        )
        return response
