from abc import ABC, abstractmethod
from typing import Annotated
from uuid import UUID

from fastapi import Depends

from ..embeddings.dependency import EmbedderDep

# from .dependency import SearchParamDep, VectorSearchParamDep
from .model import (
    Document,
    DocumentsSearchResult,
    IndexInfo,
    IndexOperationResult,
    SearchParams,
    VectorSearchParam,
)

SearchParamDep = Annotated[SearchParams, Depends(SearchParams)]
VectorSearchParamDep = Annotated[VectorSearchParam, Depends(VectorSearchParam)]


class VectorDBRepository(ABC):
    @abstractmethod
    async def search_documents(self, index_name: str, query: str, search_params: VectorSearchParamDep, embedder: EmbedderDep) -> DocumentsSearchResult: ...

    @abstractmethod
    async def get_documents(self, index_name: str, search_params: SearchParamDep) -> list[Document]: ...

    @abstractmethod
    async def add_documents(self, index_name: str, document: Document, embedder: EmbedderDep) -> IndexOperationResult: ...

    @abstractmethod
    async def delete_documents(self, index_name: str, documents_ids: list[UUID]) -> IndexOperationResult: ...

    @abstractmethod
    async def get_indexes(self) -> list[IndexInfo]: ...

    @abstractmethod
    async def create_index(self, index_name: str) -> IndexOperationResult: ...

    @abstractmethod
    async def delete_index(self, index_name: str) -> IndexOperationResult: ...
