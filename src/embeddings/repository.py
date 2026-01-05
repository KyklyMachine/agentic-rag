from abc import ABC, abstractmethod
from typing import Optional

from src.document.model import Document


class Embedder(ABC):
    @abstractmethod
    async def invoke(self, documents: list[Document], model_name: Optional[str]=None) -> list[Document]: ...
