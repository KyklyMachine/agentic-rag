from typing import Annotated

from fastapi import Depends, Request

from .repository import LLMRepository


def get_llm(request: Request) -> LLMRepository:
    return request.app.state.llm

LLMDependency = Annotated[LLMRepository, Depends(get_llm)]
