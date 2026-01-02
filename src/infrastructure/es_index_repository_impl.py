from typing import cast
from uuid import UUID

from elasticsearch import AsyncElasticsearch
from pydantic import BaseModel

from src.embeddings.dependency import EmbedderDep
from src.index.model import (
    Document,
    DocumentsSearchResult,
    IndexInfo,
    IndexOperationResult,
    SearchItem,
)
from src.index.repository import (
    SearchParamDep,
    VectorDBRepository,
    VectorSearchParamDep,
)


class ESVectorDBConfig(BaseModel):
    host: str


class ESVectorDB(VectorDBRepository):
    _client: AsyncElasticsearch

    def __init__(self, config: ESVectorDBConfig) -> None:
        self._client = AsyncElasticsearch(
            hosts=[config.host]
        )

    async def search_documents(self, index_name: str, query: str, search_params: VectorSearchParamDep, embedder: EmbedderDep) -> DocumentsSearchResult: 
        embedded_document: Document = (await embedder.invoke([Document(content=query)]))[0]
        if not embedded_document.embedding: 
            raise Exception()
        query_embedding: list[float] = embedded_document.embedding

        knn_query = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": search_params.limit,
            "num_candidates": 100
        }
        scored_points = await self._client.search(
            index=index_name,
            body={
                "knn": knn_query
            }
        )
        search_items: list[SearchItem] = []
        for hit in scored_points["hits"]["hits"]:
            item = hit["_source"]
            search_item = SearchItem(
                score=hit["_score"],
                document=Document(
                    id=UUID(str(hit["_id"])),
                    content=item["content"],
                    embedding=item["embedding"],
                    )
                )
            search_items.append(search_item)
        return DocumentsSearchResult(index_name=index_name, search_params=search_params, items=search_items)

    async def get_documents(self, index_name: str, search_params: SearchParamDep) -> list[Document]: 
        raw_documents = await self._client.search(
            index=index_name,
            body={
                "query": {
                    "match_all": {}
                },
                "size": search_params.limit
            }
        )
        documents = []
        for hit in raw_documents["hits"]["hits"]:
            source = hit["_source"]
            embedding = cast(list[float], source["embedding"])
            documents.append(
                Document(
                    id=UUID(str(hit["_id"])),
                    content=source["content"],
                    embedding=embedding,
                )
            )
        return documents

    async def add_documents(self, index_name: str, document: Document, embedder: EmbedderDep) -> IndexOperationResult: 
        if not document.embedding:
            raise Exception()

        await self._client.create(
            index=index_name,
            id=str(document.id),
            body={
                "title": document.title,
                "context": document.context,
                "content": document.content,
                "embedding_text": document.embedding_text,
                "embedding": document.embedding,
                "metadata": {
                    "source": document.metadata.source,
                    "tags": document.metadata.tags
                }
            }
        )
        return IndexOperationResult(operation="add_documents", ids=[str(document.id)])

    async def delete_documents(self, index_name: str, documents_ids: list[UUID]) -> IndexOperationResult: 
        for doc_id in documents_ids:
            await self._client.delete(index=index_name, id=str(doc_id))
        return IndexOperationResult(operation="delete_documents", ids=list(map(lambda x: str(x), documents_ids)))

    async def get_indexes(self) -> list[IndexInfo]:
        response = await self._client.cat.indices(format='json', h=['index', 'docs.count'])
        indices: list[IndexInfo] = [
            IndexInfo(name=item['index'], documents_count=int(item['docs.count'])) # type: ignore
            for item in response
        ]
        return indices

    async def create_index(self, index_name: str) -> IndexOperationResult: 
        await self._client.indices.create(
            index=index_name,
            body={
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "russian_custom": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": [
                                    "lowercase",
                                    "russian_stop",
                                    "russian_stemmer"
                                ]
                            }
                        },
                        "filter": {
                            "russian_stop": {
                                "type": "stop",
                                "stopwords": "_russian_"
                            },
                            "russian_stemmer": {
                                "type": "stemmer",
                                "language": "russian"
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "title": {
                            "type": "text",
                            "analyzer": "russian_custom"
                        },
                        "context": {
                            "type": "text",
                            "analyzer": "russian_custom"
                        },
                        "content": {
                            "type": "text",
                            "analyzer": "russian_custom"
                        },
                        "embedding_text": {
                            "type": "text",
                            "analyzer": "russian_custom",
                            "index": False
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "index": True,
                            "similarity": "cosine"
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "keyword"
                                },
                                "tags": {
                                    "type": "keyword"
                                }
                            }
                        }
                    }
                }
            }
        )
        return IndexOperationResult(operation="add_index", ids=[index_name])

    async def delete_index(self, index_name: str) -> IndexOperationResult:
        await self._client.indices.delete(index=index_name)
        return IndexOperationResult(operation="delete_index", ids=[index_name])
