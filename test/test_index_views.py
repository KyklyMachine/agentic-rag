from uuid import uuid4

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestIndexManagement:
    """Tests for index management endpoints"""

    async def test_get_indexes_empty(self, client: AsyncClient):
        """Test getting indexes when no indexes exist"""
        response = await client.get("/indexes")
        assert response.status_code == 200
        assert response.json() == []

    async def test_get_indexes_with_data(
        self, client: AsyncClient, mock_vector_db, sample_documents
    ):
        """Test getting indexes with existing data"""
        mock_vector_db.indexes["test_index"] = sample_documents
        mock_vector_db.indexes["another_index"] = []

        response = await client.get("/indexes")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert any(idx["name"] == "test_index" and idx["documents_count"] == 3 for idx in data)
        assert any(idx["name"] == "another_index" and idx["documents_count"] == 0 for idx in data)

    async def test_get_indexes_service_unavailable(self, client: AsyncClient, mock_vector_db):
        """Test getting indexes when service is unavailable"""
        mock_vector_db.should_raise_service_unavailable = True

        response = await client.get("/indexes")
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["error"] == "infra_unavailable"

    async def test_add_index_success(self, client: AsyncClient):
        """Test successfully adding a new index"""
        response = await client.post("/indexes", params={"index_name": "new_index"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "acknowledged"
        assert data["operation"] == "add_index"

    async def test_add_index_service_unavailable(self, client: AsyncClient, mock_vector_db):
        """Test adding index when service is unavailable"""
        mock_vector_db.should_raise_service_unavailable = True

        response = await client.post("/indexes", params={"index_name": "new_index"})
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["error"] == "infra_unavailable"

    async def test_delete_index_success(self, client: AsyncClient, mock_vector_db, sample_documents):
        """Test successfully deleting an existing index"""
        mock_vector_db.indexes["test_index"] = sample_documents

        response = await client.delete("/indexes/test_index")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "acknowledged"
        assert data["operation"] == "delete_index"
        assert "test_index" not in mock_vector_db.indexes

    async def test_delete_index_not_existing(self, client: AsyncClient):
        """Test deleting a non-existing index"""
        response = await client.delete("/indexes/nonexistent")
        assert response.status_code == 200

    async def test_delete_index_service_unavailable(self, client: AsyncClient, mock_vector_db):
        """Test deleting index when service is unavailable"""
        mock_vector_db.should_raise_service_unavailable = True

        response = await client.delete("/indexes/test_index")
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["error"] == "infra_unavailable"


@pytest.mark.asyncio
class TestDocumentRetrieval:
    """Tests for document retrieval endpoints"""

    async def test_get_documents_success(
        self, client: AsyncClient, mock_vector_db, sample_documents
    ):
        """Test successfully getting documents from an index"""
        mock_vector_db.indexes["test_index"] = sample_documents

        response = await client.get("/indexes/test_index/documents")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]["title"] == "Document 1"
        assert data[0]["content"] == "Content 1"

    async def test_get_documents_with_pagination(
        self, client: AsyncClient, mock_vector_db, sample_documents
    ):
        """Test getting documents with pagination parameters"""
        mock_vector_db.indexes["test_index"] = sample_documents

        response = await client.get("/indexes/test_index/documents?limit=2&offset=1")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["title"] == "Document 2"

    async def test_get_documents_without_embeddings(
        self, client: AsyncClient, mock_vector_db, sample_documents
    ):
        """Test getting documents without embeddings"""
        mock_vector_db.indexes["test_index"] = sample_documents

        response = await client.get("/indexes/test_index/documents?return_embeddings=false")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]["embedding"] is None

    async def test_get_documents_with_embeddings(
        self, client: AsyncClient, mock_vector_db, sample_documents
    ):
        """Test getting documents with embeddings"""
        mock_vector_db.indexes["test_index"] = sample_documents

        response = await client.get("/indexes/test_index/documents?return_embeddings=true")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]["embedding"] is not None
        assert len(data[0]["embedding"]) == 768

    async def test_get_documents_empty_index(self, client: AsyncClient, mock_vector_db):
        """Test getting documents from an empty index"""
        mock_vector_db.indexes["empty_index"] = []

        response = await client.get("/indexes/empty_index/documents")
        assert response.status_code == 200
        assert response.json() == []

    async def test_get_documents_nonexistent_index(self, client: AsyncClient):
        """Test getting documents from a non-existent index"""
        response = await client.get("/indexes/nonexistent/documents")
        assert response.status_code == 200
        assert response.json() == []

    async def test_get_documents_service_unavailable(
        self, client: AsyncClient, mock_vector_db
    ):
        """Test getting documents when service is unavailable"""
        mock_vector_db.should_raise_service_unavailable = True

        response = await client.get("/indexes/test_index/documents")
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["error"] == "infra_unavailable"


@pytest.mark.asyncio
class TestDocumentSearch:
    """Tests for document search endpoints"""

    async def test_search_documents_success(
        self, client: AsyncClient, mock_vector_db, sample_documents
    ):
        """Test successfully searching documents"""
        mock_vector_db.indexes["test_index"] = sample_documents

        response = await client.post(
            "/indexes/test_index/documents/search",
            params={"query": "test query"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["index_name"] == "test_index"
        assert len(data["items"]) > 0
        assert "score" in data["items"][0]
        assert "document" in data["items"][0]

    async def test_search_documents_with_limit(
        self, client: AsyncClient, mock_vector_db, sample_documents
    ):
        """Test searching documents with limit parameter"""
        mock_vector_db.indexes["test_index"] = sample_documents

        response = await client.post(
            "/indexes/test_index/documents/search",
            params={"query": "test query", "limit": 2},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) <= 2

    async def test_search_documents_without_embeddings(
        self, client: AsyncClient, mock_vector_db, sample_documents
    ):
        """Test searching documents without returning embeddings"""
        mock_vector_db.indexes["test_index"] = sample_documents

        response = await client.post(
            "/indexes/test_index/documents/search",
            params={"query": "test query", "return_embeddings": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["search_params"]["return_embeddings"] is False

    async def test_search_documents_empty_index(
        self, client: AsyncClient, mock_vector_db
    ):
        """Test searching in an empty index"""
        mock_vector_db.indexes["empty_index"] = []

        response = await client.post(
            "/indexes/empty_index/documents/search",
            params={"query": "test query"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []

    async def test_search_documents_nonexistent_index(self, client: AsyncClient):
        """Test searching in a non-existent index"""
        response = await client.post(
            "/indexes/nonexistent/documents/search",
            params={"query": "test query"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []

    async def test_search_documents_service_unavailable(
        self, client: AsyncClient, mock_vector_db
    ):
        """Test searching documents when service is unavailable"""
        mock_vector_db.should_raise_service_unavailable = True

        response = await client.post(
            "/indexes/test_index/documents/search",
            params={"query": "test query"},
        )
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["error"] == "infra_unavailable"


@pytest.mark.asyncio
class TestDocumentManagement:
    """Tests for document management endpoints"""

    async def test_add_documents_success(self, client: AsyncClient, mock_vector_db):
        """Test successfully adding documents to an index"""
        mock_vector_db.indexes["test_index"] = []

        request_data = {
            "documents": [
                {
                    "title": "Test Doc",
                    "context": "Test context",
                    "content": "Test content",
                    "metadata": {"source": "test", "tags": ["tag1"]},
                }
            ]
        }

        response = await client.post("/indexes/test_index/documents", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "acknowledged"
        assert data["operation"] == "add_documents"
        assert data["errors"] is None
        assert len(mock_vector_db.indexes["test_index"]) == 1

    async def test_add_multiple_documents(self, client: AsyncClient, mock_vector_db):
        """Test adding multiple documents at once"""
        mock_vector_db.indexes["test_index"] = []

        request_data = {
            "documents": [
                {"content": f"Content {i}", "metadata": {"tags": [f"tag{i}"]}}
                for i in range(3)
            ]
        }

        response = await client.post("/indexes/test_index/documents", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "acknowledged"
        assert len(mock_vector_db.indexes["test_index"]) == 3

    async def test_add_documents_with_embeddings(
        self, client: AsyncClient, mock_vector_db
    ):
        """Test that embeddings are generated for documents without them"""
        mock_vector_db.indexes["test_index"] = []

        request_data = {
            "documents": [{"content": "Test content"}]
        }

        response = await client.post("/indexes/test_index/documents", json=request_data)
        assert response.status_code == 200

        added_doc = mock_vector_db.indexes["test_index"][0]
        assert added_doc.embedding is not None
        assert len(added_doc.embedding) == 768

    async def test_add_documents_nonexistent_index(self, client: AsyncClient):
        """Test adding documents to a non-existent index"""
        request_data = {
            "documents": [{"content": "Test content"}]
        }

        response = await client.post("/indexes/nonexistent/documents", json=request_data)
        # The service returns 200 with errors in the response body for per-document failures
        assert response.status_code == 200
        data = response.json()
        assert data["errors"] is not None
        assert len(data["errors"]["err_documents"]) == 1

    async def test_add_documents_service_unavailable(
        self, client: AsyncClient, mock_vector_db
    ):
        """Test adding documents when service is unavailable"""
        mock_vector_db.indexes["test_index"] = []
        mock_vector_db.should_raise_service_unavailable = True

        request_data = {
            "documents": [{"content": "Test content"}]
        }

        response = await client.post("/indexes/test_index/documents", json=request_data)
        # The service handles errors per document, so service unavailable in add_documents
        # will return 200 with errors
        assert response.status_code == 200
        data = response.json()
        assert data["errors"] is not None

    async def test_add_documents_with_partial_errors(
        self, client: AsyncClient, mock_vector_db
    ):
        """Test adding documents when some fail"""
        mock_vector_db.indexes["test_index"] = []

        request_data = {
            "documents": [
                {"content": "Content 1"},
                {"content": "Content 2"},
            ]
        }

        response = await client.post("/indexes/test_index/documents", json=request_data)
        assert response.status_code == 200

    async def test_delete_documents_success(
        self, client: AsyncClient, mock_vector_db, sample_documents
    ):
        """Test successfully deleting documents"""
        mock_vector_db.indexes["test_index"] = sample_documents.copy()
        doc_id = str(sample_documents[0].id)

        response = await client.request(
            "DELETE",
            "/indexes/test_index/documents",
            json=[doc_id],
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "acknowledged"
        assert data["operation"] == "delete_documents"
        assert len(mock_vector_db.indexes["test_index"]) == 2

    async def test_delete_multiple_documents(
        self, client: AsyncClient, mock_vector_db, sample_documents
    ):
        """Test deleting multiple documents at once"""
        mock_vector_db.indexes["test_index"] = sample_documents.copy()
        doc_ids = [str(doc.id) for doc in sample_documents[:2]]

        response = await client.request(
            "DELETE",
            "/indexes/test_index/documents",
            json=doc_ids,
        )
        assert response.status_code == 200
        assert len(mock_vector_db.indexes["test_index"]) == 1

    async def test_delete_documents_nonexistent_ids(
        self, client: AsyncClient, mock_vector_db, sample_documents
    ):
        """Test deleting documents with non-existent IDs"""
        mock_vector_db.indexes["test_index"] = sample_documents.copy()
        fake_id = str(uuid4())

        response = await client.request(
            "DELETE",
            "/indexes/test_index/documents",
            json=[fake_id],
        )
        assert response.status_code == 200
        assert len(mock_vector_db.indexes["test_index"]) == 3

    async def test_delete_documents_service_unavailable(
        self, client: AsyncClient, mock_vector_db
    ):
        """Test deleting documents when service is unavailable"""
        mock_vector_db.should_raise_service_unavailable = True
        doc_id = str(uuid4())

        response = await client.request(
            "DELETE",
            "/indexes/test_index/documents",
            json=[doc_id],
        )
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["error"] == "infra_unavailable"


@pytest.mark.asyncio
class TestEdgeCases:
    """Tests for edge cases and validation"""

    async def test_get_documents_invalid_limit(self, client: AsyncClient, mock_vector_db):
        """Test getting documents with invalid limit parameter"""
        mock_vector_db.indexes["test_index"] = []
        response = await client.get("/indexes/test_index/documents?limit=101")
        assert response.status_code == 422

    async def test_get_documents_invalid_offset(self, client: AsyncClient, mock_vector_db):
        """Test getting documents with invalid offset parameter"""
        mock_vector_db.indexes["test_index"] = []
        response = await client.get("/indexes/test_index/documents?offset=-1")
        assert response.status_code == 422

    async def test_search_documents_invalid_limit(self, client: AsyncClient):
        """Test searching documents with invalid limit parameter"""
        response = await client.post(
            "/indexes/test_index/documents/search",
            params={"query": "test", "limit": 101},
        )
        assert response.status_code == 422

    async def test_add_documents_empty_content(self, client: AsyncClient, mock_vector_db):
        """Test adding document with empty content - should succeed but with empty string"""
        mock_vector_db.indexes["test_index"] = []

        request_data = {
            "documents": [{"content": ""}]
        }

        response = await client.post("/indexes/test_index/documents", json=request_data)
        # Empty string is technically valid, API doesn't enforce min_length
        assert response.status_code == 200

    async def test_add_documents_missing_content(self, client: AsyncClient, mock_vector_db):
        """Test adding document without content field"""
        mock_vector_db.indexes["test_index"] = []

        request_data = {
            "documents": [{"title": "Test"}]
        }

        response = await client.post("/indexes/test_index/documents", json=request_data)
        assert response.status_code == 422
