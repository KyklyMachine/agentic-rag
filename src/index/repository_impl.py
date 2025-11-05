from uuid import UUID

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models.models import ScoredPoint

from .exceptions import IndexIsNotExist
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

    async def search_topics(self, index_name: str, query: list[float], search_params: SearchParams) -> DocumentsSearchResult: 
        scored_points: list[ScoredPoint] = await self._client.search(
            collection_name=index_name,
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
        return DocumentsSearchResult(index_name=index_name, search_params=search_params, items=search_items)

    async def get_topics(self, index_name: str, search_params: SearchParams) -> list[Document]: 
        raw_documents, ids = await self._client.scroll(collection_name=index_name, limit=search_params.limit, offset=search_params.offset)
        documents = []
        for raw_doc in raw_documents:
            if not raw_doc.payload:
                raise Exception("Error Search Qdrant: item.payload is None!")
            if not isinstance(raw_doc.vector, list):
                raise Exception("Error Search Qdrant: item.vector is not list[float]!")
            documents.append(
                Document(
                    id=UUID(int=int(raw_doc.id)),
                    content=raw_doc.payload.get("document", ""), 
                    embedding=raw_doc.vector, 
                    metadata={}
                )
            )
        return documents

    async def add_topic(self, index_name: str, document: Document) -> None: 
        try:
            await self._client.upsert(
                collection_name=index_name,
                points=[
                    models.PointStruct(
                        id=str(document.id), # type: ignore
                        vector=document.embedding, # type: ignore
                        payload={"text": document.content, "metadata": document.metadata},
                    )
                ],
            )
        except UnexpectedResponse as unexp_resp:
            raise IndexIsNotExist(unexp_resp)

    async def delete_topics(self, index_name: str, topics_ids: list[UUID]) -> None: 
        await self._client.delete(collection_name=index_name, points_selector=[str(id) for id in documents_id])

    async def get_indexes(self) -> list[str]:
        indexes = await self._client.get_collections()
        indexes = [index.name for index in indexes.collections]
        return indexes

    async def add_index(self, index_name: str) -> None: 
        await self._client.create_collection(collection_name=index_name)

    async def delete_index(self, index_name: str) -> None:
        await self._client.delete_collection(collection_name=index_name)
