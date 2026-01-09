from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, HTTPException, Path, Query, status

from src.embeddings.dependency import EmbedderDep
from src.index.model import DocumentsSearchResult

from .dependency import VectorDBDep
from .dto import AddDocumentsRequest
from .exceptions import (
    IndexIsNotExist,
    IndexNotFoundException,
    ServiceUnavaliable,
    service_unavaliable_http_exception,
)
from .model import Document, IndexInfo, IndexOperationResult
from .repository import SearchParamDep, VectorSearchParamDep
from .service import IndexService

router = APIRouter(prefix="/indexes", tags=["indexes"])


@router.get(
    "",
    status_code=status.HTTP_200_OK,
    summary="Get all indexes",
    description="Retrieve a list of all available indexes with their document counts",
    responses={
        200: {"description": "List of indexes retrieved successfully"},
        503: {"description": "Service unavailable - database connection error"},
    },
)
async def get_indexes(vector_db: VectorDBDep) -> list[IndexInfo]:
    """Get all available indexes with their document counts"""
    try:
        return await IndexService().get_indexes(vector_db=vector_db)
    except ServiceUnavaliable:
        raise service_unavaliable_http_exception


@router.post(
    "",
    status_code=status.HTTP_200_OK,
    summary="Create new index",
    description="Create a new index with the specified name",
    responses={
        200: {"description": "Index created successfully"},
        503: {"description": "Service unavailable - database connection error"},
    },
)
async def add_index(
    vector_db: VectorDBDep,
    index_name: Annotated[str, Query(description="Name of the index to create")],
) -> IndexOperationResult:
    """Create a new index with Russian text analyzer and vector search support"""
    try:
        return await IndexService().add_index(vector_db, index_name)
    except ServiceUnavaliable:
        raise service_unavaliable_http_exception


@router.delete(
    "/{index_name}",
    status_code=status.HTTP_200_OK,
    summary="Delete index",
    description="Delete an existing index and all its documents",
    responses={
        200: {"description": "Index deleted successfully"},
        503: {"description": "Service unavailable - database connection error"},
    },
)
async def delete_index(
    vector_db: VectorDBDep,
    index_name: Annotated[str, Path(description="Name of the index to delete")],
) -> IndexOperationResult:
    """Delete an index and all its documents"""
    try:
        return await IndexService().delete_index(vector_db, index_name)
    except ServiceUnavaliable:
        raise service_unavaliable_http_exception
    except IndexNotFoundException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"error": "index_not_found", "message": str(e)})

@router.get(
    "/{index_name}/documents",
    status_code=status.HTTP_200_OK,
    summary="Get documents from index",
    description="Retrieve documents from an index with pagination support",
    responses={
        200: {"description": "Documents retrieved successfully"},
        503: {"description": "Service unavailable - database connection error"},
    },
)
async def get_documents(
    vector_db: VectorDBDep,
    search_params: SearchParamDep,
    index_name: Annotated[str, Path(description="Name of the index to retrieve documents from")],
) -> list[Document]:
    """
    Get documents from an index with pagination.

    Query parameters:
    - limit: Maximum number of documents to return (1-100, default: 5)
    - offset: Number of documents to skip (default: 0)
    - return_embeddings: Whether to include embeddings in response (default: true)
    """
    try:
        return await IndexService().get_documents(
            vector_db=vector_db, search_params=search_params, index_name=index_name
        )
    except ServiceUnavaliable:
        raise service_unavaliable_http_exception
    except IndexNotFoundException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"error": "index_not_found", "message": str(e)})


@router.post(
    "/{index_name}/documents/search",
    status_code=status.HTTP_200_OK,
    summary="Search documents by vector similarity",
    description="Perform vector similarity search on documents using embeddings",
    responses={
        200: {"description": "Search completed successfully"},
        503: {"description": "Service unavailable - database connection error"},
    },
)
async def search_documents(
    vector_db: VectorDBDep,
    search_params: VectorSearchParamDep,
    embedder: EmbedderDep,
    index_name: Annotated[str, Path(description="Name of the index to search in")],
    query: Annotated[str, Query(description="Search query text")],
) -> DocumentsSearchResult:
    """
    Search documents using vector similarity (KNN search).

    Query parameters:
    - query: Text to search for (will be embedded automatically)
    - limit: Maximum number of results to return (0-100, default: 4)
    - return_embeddings: Whether to include embeddings in response (default: true)
    """
    try:
        return await IndexService().search_documents(
            vector_db=vector_db,
            search_params=search_params,
            index_name=index_name,
            query=query,
            embedder=embedder,
        )
    except ServiceUnavaliable:
        raise service_unavaliable_http_exception
    except IndexNotFoundException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"error": "index_not_found", "message": str(e)})

@router.post(
    "/{index_name}/documents",
    status_code=status.HTTP_200_OK,
    summary="Add documents to index",
    description="Add one or more documents to an index with automatic embedding generation",
    responses={
        200: {
            "description": "Documents processed. Check 'errors' field for per-document failures",
            "content": {
                "application/json": {
                    "example": {
                        "status": "acknowledged",
                        "operation": "add_documents",
                        "errors": None,
                    }
                }
            },
        },
        404: {"description": "Index does not exist"},
        503: {"description": "Service unavailable - database connection error"},
    },
)
async def add_documents(
    vector_db: VectorDBDep,
    embedder: EmbedderDep,
    index_name: Annotated[str, Path(description="Name of the index to add documents to")],
    request: Annotated[AddDocumentsRequest, Body(description="Documents to add")],
) -> IndexOperationResult:
    """
    Add documents to an index.

    Documents without embeddings will have embeddings generated automatically.
    Returns 200 even if some documents fail - check the 'errors' field in response.

    Request body should contain a list of documents with:
    - content: Required text content
    - title: Optional document title
    - context: Optional context for better understanding
    - metadata: Optional metadata (source, tags)
    """
    try:
        return await IndexService().add_documents(
            vector_db=vector_db,
            index_name=index_name,
            documents=request.to_document_model(),
            embedder=embedder,
        )
    except IndexIsNotExist as unexp_resp:
        raise HTTPException(status_code=404, detail=str(unexp_resp))
    except ServiceUnavaliable:
        raise service_unavaliable_http_exception


@router.delete(
    "/{index_name}/documents",
    status_code=status.HTTP_200_OK,
    summary="Delete documents from index",
    description="Delete one or more documents from an index by their IDs",
    responses={
        200: {"description": "Documents deleted successfully"},
        503: {"description": "Service unavailable - database connection error"},
    },
)
async def delete_documents(
    vector_db: VectorDBDep,
    index_name: Annotated[str, Path(description="Name of the index to delete documents from")],
    documents_ids: Annotated[
        list[UUID],
        Body(
            description="List of document UUIDs to delete",
            examples=[["3fa85f64-5717-4562-b3fc-2c963f66afa6"]],
        ),
    ],
) -> IndexOperationResult:
    """
    Delete documents from an index by their UUIDs.

    Request body should be a JSON array of UUIDs:
    ["uuid1", "uuid2", ...]
    """
    try:
        return await IndexService().delete_documents(
            vector_db, index_name=index_name, documents_ids=documents_ids
        )
    except ServiceUnavaliable:
        raise service_unavaliable_http_exception
    except IndexNotFoundException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"error": "index_not_found", "message": str(e)})
