from typing import Any, Optional
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field


class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str
    embedding: Optional[list[np.float32]]
    metadata: Optional[dict[str, Any]]
