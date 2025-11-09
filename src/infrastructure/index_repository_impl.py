from uuid import UUID

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models.models import ScoredPoint
from qdrant_client.models import Distance, VectorParams

from src.embeddings.dependency import EmbedderDep
from src.index.exceptions import IndexIsNotExist
from src.index.model import Document, DocumentsSearchResult, SearchItem, SearchParams
from src.index.repository import VectorDBRepository


class QdrantVectorDBConfig(BaseModel):
    url: str


class QdrantVectorDB(VectorDBRepository):
    _client: AsyncQdrantClient

    def __init__(self, config: QdrantVectorDBConfig) -> None:
        self._client = AsyncQdrantClient(
            url=config.url
        )

    async def search_documents(self, index_name: str, query: str, search_params: SearchParams, embedder: EmbedderDep) -> DocumentsSearchResult: 
        embedded_document: Document = (await embedder.invoke([Document(content=query)]))[0]
        if not embedded_document.embedding: 
            raise Exception()
        query_embedding: list[float] = embedded_document.embedding
        scored_points: list[ScoredPoint] = await self._client.search(
            collection_name=index_name,
            query_vector=query_embedding,
            limit=search_params.limit,
            offset=search_params.offset
        )
        search_items: list[SearchItem] = []
        for item in scored_points:
            if not item.payload:
                raise Exception("Error Search Qdrant: item.payload is None!")
            # if not isinstance(item.vector, list):
            #     raise Exception("Error Search Qdrant: item.vector is not list[float]!")
            search_item = SearchItem(
                score=item.score, 
                document=Document(
                    id=UUID(item.id), 
                    content=item.payload.get("text", ""), 
                    embedding=item.vector,  # type: ignore
                    metadata={}
                    )
                )
            search_items.append(search_item)
        return DocumentsSearchResult(index_name=index_name, search_params=search_params, items=search_items)

    async def get_documents(self, index_name: str, search_params: SearchParams) -> list[Document]: 
        raw_documents, _ = await self._client.scroll(
            collection_name=index_name, 
            limit=search_params.limit, 
            offset=search_params.offset, 
            with_payload=True, 
            with_vectors=True
        )
        documents = []
        for raw_doc in raw_documents:
            if not raw_doc.payload:
                raise Exception("Error Search Qdrant: item.payload is None!")
            documents.append(
                Document(
                    id=UUID(str(raw_doc.id)),
                    content=raw_doc.payload.get("text", ""), 
                    embedding=raw_doc.vector,  # type: ignore
                    metadata={}
                )
            )
        return documents

    async def add_documents(self, index_name: str, document: Document, embedder: EmbedderDep) -> None: 
        if not document.embedding:
            raise Exception()
        try:
            await self._client.upsert(
                collection_name=index_name,
                wait=True,
                points=[
                    models.PointStruct(
                        id=str(document.id),
                        vector=document.embedding, 
                        payload={"text": document.content, "metadata": document.metadata},
                    )
                ],
            )
        except UnexpectedResponse as unexp_resp:
            raise IndexIsNotExist(unexp_resp)

    async def delete_documents(self, index_name: str, documents_ids: list[UUID]) -> None: 
        await self._client.delete(collection_name=index_name, points_selector=[str(id) for id in documents_ids])

    async def get_indexes(self) -> list[str]:
        indexes = await self._client.get_collections()
        indexes = [index.name for index in indexes.collections]
        return indexes

    async def add_index(self, index_name: str) -> None: 
        await self._client.create_collection(
            collection_name=index_name,
            vectors_config=VectorParams(size=4096, distance=Distance.COSINE)
        )

    async def delete_index(self, index_name: str) -> None:
        await self._client.delete_collection(collection_name=index_name)
