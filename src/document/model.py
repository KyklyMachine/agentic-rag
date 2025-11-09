from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str
    embedding: Optional[list[float]] = Field(default=None)
    metadata: Optional[dict[str, Any]] = Field(default=None)
