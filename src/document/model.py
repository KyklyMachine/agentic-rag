from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Payload(BaseModel):
    content: str

class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    embedding: Optional[list[float]] = Field(default=None)
    payload: Payload = Field(...)
