from uuid import UUID

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models.models import ScoredPoint

from .model import Document, DocumentsSearchResult, SearchItem, SearchParams
from .repository import VectorDBRepository


class QdrantVectorDBConfig(BaseModel):
    url: str


class QdrantVectorDB(VectorDBRepository):
    _client: AsyncQdrantClient

    def __init__(self, config: QdrantVectorDBConfig) -> None:
        self._client = AsyncQdrantClient(
            url=config.url
        )

    async def search_documents(self, index: str, query: list[float], search_params: SearchParams) -> DocumentsSearchResult: 
        scored_points: list[ScoredPoint] = await self._client.search(
            collection_name=index,
            query_vector=query,
            limit=search_params.limit,
            offset=search_params.offset
        )
        search_items: list[SearchItem] = []
        for item in scored_points:
            if not item.payload:
                raise Exception("Error Search Qdrant: item.payload is None!")
            if not isinstance(item.vector, list):
                raise Exception("Error Search Qdrant: item.vector is not list[float]!")
            search_item = SearchItem(
                score=item.score, 
                document=Document(
                    id=UUID(int=int(item.id)), 
                    content=item.payload.get("document", ""), 
                    embedding=item.vector, 
                    metadata={}
                    )
                )
            search_items.append(search_item)
        return DocumentsSearchResult(index_name=index, search_params=search_params, items=search_items)

    async def get_documents(self, index: str, search_params: SearchParams) -> list[Document]: ...

    async def add_document(self, index: str, document: Document) -> None: ...

    async def delete_document(self, index: str, document_id: UUID) -> None: ...

    async def get_indexes(self) -> list[str]: ...

    async def add_index(self, name: str) -> None: ...

    async def delete_index(self, index: str) -> None: ...