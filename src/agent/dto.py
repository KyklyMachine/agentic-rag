from typing import Optional

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    messages: list[dict[str, str]] = Field(..., description="List of messages to be processed by the agent.")
    index_name: Optional[str] = Field(default=None, description="Name of the index to use for processing the messages.")

class AgentResponse(BaseModel):
    content: str
    documents: Optional[list[str]] = Field(default=None, description="List of documents added to the agent.")
