from abc import ABC, abstractmethod
from uuid import UUID

from .model import Document, DocumentsSearchResult, SearchParams


class VectorDBRepository(ABC):
    @abstractmethod
    async def search_topics(self, index_name: str, query: list[float], search_params: SearchParams) -> DocumentsSearchResult: ...

    @abstractmethod
    async def get_topics(self, index_name: str, search_params: SearchParams) -> list[Document]: ...

    @abstractmethod
    async def add_topic(self, index_name: str, document: Document) -> None: ...

    @abstractmethod
    async def delete_topics(self, index_name: str, topics_ids: list[UUID]) -> None: ...

    @abstractmethod
    async def get_indexes(self) -> list[str]: ...

    @abstractmethod
    async def add_index(self, index_name: str) -> None: ...

    @abstractmethod
    async def delete_index(self, index_name: str) -> None: ...
