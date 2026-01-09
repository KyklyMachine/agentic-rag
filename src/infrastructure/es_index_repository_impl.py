from typing import Any, override
from uuid import UUID

import elastic_transport
from elasticsearch import AsyncElasticsearch
from pydantic import BaseModel

from src.document.model import Metadata
from src.embeddings.dependency import EmbedderDep
from src.index.exceptions import InconsistentIndex, ServiceUnavaliable
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

    @override
    def __init__(self, config: ESVectorDBConfig) -> None:
        super().__init__()
        self._client = AsyncElasticsearch(
            hosts=[config.host]
        )

    @override
    async def search_documents(self, index_name: str, query: str, search_params: VectorSearchParamDep, embedder: EmbedderDep) -> DocumentsSearchResult: 
        embedded_document: Document = (await embedder.invoke([Document(content=query)]))[0]
        if not embedded_document.embedding: 
            raise Exception()
        query_embedding: list[float] = embedded_document.embedding

        knn_query: dict[str, Any] = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": search_params.limit,
            "num_candidates": 100
        }

        body: dict[str, dict[str, Any]] = {
            "knn": knn_query
        }

        if not search_params.return_embeddings:
            body["_source"] = {
                "excludes": ["embedding"]
            }
        try:
            scored_points = await self._client.search(
                index=index_name,
                body=body
            )
        except elastic_transport.ConnectionError:
            raise ServiceUnavaliable("Elasticsearch service is unavailable")

        search_items: list[SearchItem] = []
        for hit in scored_points["hits"]["hits"]:
            source = hit["_source"]
            if "_id" not in hit:
                raise InconsistentIndex("_id not in hit")
            if "content" not in source:
                raise InconsistentIndex("content not in source")
            if "metadata" not in source:
                raise InconsistentIndex("metadata not in source")

            search_item = SearchItem(
                score=hit["_score"],
                document=Document(
                    id=UUID(str(hit.get("_id", None))),
                    title=source.get("title", None),
                    context=source.get("context", None),
                    content=source.get("content", None),
                    embedding_text=source.get("embedding_text", None),
                    embedding=source.get("embedding", None),
                    metadata=source.get("metadata", Metadata())
                    )
                )
            search_items.append(search_item)
        return DocumentsSearchResult(index_name=index_name, search_params=search_params, items=search_items)

    @override
    async def get_documents(self, index_name: str, search_params: SearchParamDep) -> list[Document]: 
        
        body: dict[str, Any] = {
                "query": {
                    "match_all": {}
                },
                "size": search_params.limit
            }
        
        if not search_params.return_embeddings:
            body["_source"] = {
                "excludes": ["embedding"]
            }
        try:
            raw_documents = await self._client.search(
                index=index_name,
                body=body,
            )
        except elastic_transport.ConnectionError:
            raise ServiceUnavaliable("Elasticsearch service is unavailable")

        documents: list[Document] = []
        for hit in raw_documents["hits"]["hits"]:
            source = hit["_source"]
            if "_id" not in hit:
                raise InconsistentIndex("_id not in hit")
            if "content" not in source:
                raise InconsistentIndex("content not in source")
            if "metadata" not in source:
                raise InconsistentIndex("metadata not in source")
            documents.append(
                Document(
                    id=UUID(str(hit.get("_id", None))),
                    title=source.get("title", None),
                    context=source.get("context", None),
                    content=source.get("content", None),
                    embedding_text=source.get("embedding_text", None),
                    embedding=source.get("embedding", None),
                    metadata=source.get("metadata", Metadata())
                )
            )
        return documents

    @override
    async def add_documents(self, index_name: str, document: Document, embedder: EmbedderDep) -> IndexOperationResult: 
        if not document.embedding:
            raise Exception()

        try:
            _ = await self._client.create(
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
        except elastic_transport.ConnectionError:
            raise ServiceUnavaliable("Elasticsearch service is unavailable")

        return IndexOperationResult(operation="add_documents")

    async def delete_documents(self, index_name: str, documents_ids: list[UUID]) -> IndexOperationResult: 
        errs: list[dict[Any, Any]] = []
        for doc_id in documents_ids:
            try:
                _ = await self._client.delete(index=index_name, id=str(doc_id))
            except elastic_transport.ConnectionError:
                errs.append(
                        {
                            doc_id: ServiceUnavaliable("Elasticsearch service is unavailable")
                        }
                    )
        return IndexOperationResult(operation="delete_documents", errors={"error_docs": errs})

    @override
    async def get_indexes(self) -> list[IndexInfo]:
        try:
            response = await self._client.cat.indices(format='json', h=['index', 'docs.count'])
        except elastic_transport.ConnectionError:
            raise ServiceUnavaliable("Elasticsearch service is unavailable")
        indices: list[IndexInfo] = [
            IndexInfo(name=item['index'], documents_count=int(item['docs.count'])) # type: ignore
            for item in response
        ]
        return indices

    @override
    async def create_index(self, index_name: str) -> IndexOperationResult: 
        body: dict[str, Any] = {
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
        try:
            _ = await self._client.indices.create(
                index=index_name,
                body=body
            )
        except elastic_transport.ConnectionError:
            raise ServiceUnavaliable("Elasticsearch service is unavailable")

        return IndexOperationResult(operation="add_index")

    @override
    async def delete_index(self, index_name: str) -> IndexOperationResult:
        try:
            _ = await self._client.indices.delete(index=index_name)
        except elastic_transport.ConnectionError:
            raise ServiceUnavaliable("Elasticsearch service is unavailable")

        return IndexOperationResult(operation="delete_index")
