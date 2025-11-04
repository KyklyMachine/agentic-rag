from abc import ABC, abstractmethod
from uuid import UUID

from src.document.model import Document

from .model import DocumentsSearchResult, SearchParams


class VectorDBRepository(ABC):
    @abstractmethod
    async def search_documents(self, index: str, query: list[float], search_params: SearchParams) -> DocumentsSearchResult: ...

    @abstractmethod
    async def get_documents(self, index: str, search_params: SearchParams) -> list[Document]: ...

    @abstractmethod
    async def add_document(self, index: str, document: Document) -> None: ...

    @abstractmethod
    async def delete_document(self, index: str, document_id: UUID) -> None: ...

    @abstractmethod
    async def get_indexes(self) -> list[str]: ...

    @abstractmethod
    async def add_index(self, name: str) -> None: ...

    @abstractmethod
    async def delete_index(self, index: str) -> None: ...
