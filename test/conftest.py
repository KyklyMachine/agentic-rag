from typing import Optional
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.document.model import Document, Metadata
from src.embeddings.repository import Embedder
from src.index.exceptions import IndexIsNotExist, ServiceUnavaliable
from src.index.model import (
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
from src.router import router


class MockVectorDB(VectorDBRepository):
    """Mock implementation of VectorDBRepository for testing"""

    def __init__(self):
        self.indexes: dict[str, list[Document]] = {}
        self.should_raise_service_unavailable = False
        self.should_raise_index_not_exist = False

    async def search_documents(
        self,
        index_name: str,
        query: str,
        search_params: VectorSearchParamDep,
        embedder,
    ) -> DocumentsSearchResult:
        if self.should_raise_service_unavailable:
            raise ServiceUnavaliable("Service unavailable")

        if index_name not in self.indexes:
            return DocumentsSearchResult(
                index_name=index_name, search_params=search_params, items=[]
            )

        documents = self.indexes[index_name][: search_params.limit]
        items = [
            SearchItem(document=doc, score=0.95 - i * 0.1)
            for i, doc in enumerate(documents)
        ]
        return DocumentsSearchResult(
            index_name=index_name, search_params=search_params, items=items
        )

    async def get_documents(
        self, index_name: str, search_params: SearchParamDep
    ) -> list[Document]:
        if self.should_raise_service_unavailable:
            raise ServiceUnavaliable("Service unavailable")

        if index_name not in self.indexes:
            return []

        start = search_params.offset
        end = start + search_params.limit
        documents = self.indexes[index_name][start:end]

        if not search_params.return_embeddings:
            for doc in documents:
                doc.embedding = None

        return documents

    async def add_documents(
        self, index_name: str, document: Document, embedder
    ) -> IndexOperationResult:
        if self.should_raise_service_unavailable:
            raise ServiceUnavaliable("Service unavailable")

        if self.should_raise_index_not_exist:
            raise IndexIsNotExist(f"Index {index_name} does not exist")

        if index_name not in self.indexes:
            raise IndexIsNotExist(f"Index {index_name} does not exist")

        self.indexes[index_name].append(document)
        return IndexOperationResult(operation="add_documents")

    async def delete_documents(
        self, index_name: str, documents_ids: list[UUID]
    ) -> IndexOperationResult:
        if self.should_raise_service_unavailable:
            raise ServiceUnavaliable("Service unavailable")

        if index_name in self.indexes:
            self.indexes[index_name] = [
                doc for doc in self.indexes[index_name] if doc.id not in documents_ids
            ]

        return IndexOperationResult(operation="delete_documents")

    async def get_indexes(self) -> list[IndexInfo]:
        if self.should_raise_service_unavailable:
            raise ServiceUnavaliable("Service unavailable")

        return [
            IndexInfo(name=name, documents_count=len(docs))
            for name, docs in self.indexes.items()
        ]

    async def create_index(self, index_name: str) -> IndexOperationResult:
        if self.should_raise_service_unavailable:
            raise ServiceUnavaliable("Service unavailable")

        self.indexes[index_name] = []
        return IndexOperationResult(operation="add_index")

    async def delete_index(self, index_name: str) -> IndexOperationResult:
        if self.should_raise_service_unavailable:
            raise ServiceUnavaliable("Service unavailable")

        if index_name in self.indexes:
            del self.indexes[index_name]

        return IndexOperationResult(operation="delete_index")


class MockEmbedder(Embedder):
    """Mock implementation of Embedder for testing"""

    def __init__(self):
        self.should_raise_error = False

    async def invoke(
        self, documents: list[Document], model_name: Optional[str] = None
    ) -> list[Document]:
        if self.should_raise_error:
            raise Exception("Embedding service error")

        for doc in documents:
            if doc.embedding is None:
                doc.embedding = [0.1] * 768

        return documents


@pytest.fixture
def mock_vector_db():
    """Fixture providing a mock vector database"""
    return MockVectorDB()


@pytest.fixture
def mock_embedder():
    """Fixture providing a mock embedder"""
    return MockEmbedder()


@pytest.fixture
async def test_app(mock_vector_db, mock_embedder):
    """Fixture providing a test FastAPI application with mock dependencies"""
    app = FastAPI()
    app.state.vector_db = mock_vector_db
    app.state.embedder = mock_embedder
    app.include_router(router)
    return app


@pytest.fixture
async def client(test_app):
    """Fixture providing an async HTTP client for testing"""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_document():
    """Fixture providing a sample document for testing"""
    return Document(
        title="Test Document",
        context="Test context",
        content="Test content",
        embedding=[0.1] * 768,
        metadata=Metadata(source="test", tags=["tag1", "tag2"]),
    )


@pytest.fixture
def sample_documents():
    """Fixture providing multiple sample documents for testing"""
    return [
        Document(
            title=f"Document {i}",
            context=f"Context {i}",
            content=f"Content {i}",
            embedding=[0.1 * i] * 768,
            metadata=Metadata(source="test", tags=[f"tag{i}"]),
        )
        for i in range(1, 4)
    ]
