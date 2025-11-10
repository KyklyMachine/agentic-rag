from abc import ABC, abstractmethod
from uuid import UUID

from ..embeddings.dependency import EmbedderDep
from .model import Document, DocumentsSearchResult, IndexOperationResult, SearchParams
    

class VectorDBRepository(ABC):
    @abstractmethod
    async def search_documents(self, index_name: str, query: str, search_params: SearchParams, embedder: EmbedderDep) -> DocumentsSearchResult: ...

    @abstractmethod
    async def get_documents(self, index_name: str, search_params: SearchParams) -> list[Document]: ...

    @abstractmethod
    async def add_documents(self, index_name: str, document: Document, embedder: EmbedderDep) -> IndexOperationResult: ...

    @abstractmethod
    async def delete_documents(self, index_name: str, documents_ids: list[UUID]) -> IndexOperationResult: ...

    @abstractmethod
    async def get_indexes(self) -> list[str]: ...

    @abstractmethod
    async def add_index(self, index_name: str) -> IndexOperationResult: ...

    @abstractmethod
    async def delete_index(self, index_name: str) -> IndexOperationResult: ...
