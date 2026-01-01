from uuid import UUID

from fastapi import APIRouter, HTTPException

from src.embeddings.dependency import EmbedderDep
from src.index.model import DocumentsSearchResult

from .dependency import VectorDBDep
from .exceptions import (
    IndexIsNotExist,
    ServiceUnavaliable,
    service_unavaliable_http_exception,
)
from .model import Document, IndexOperationResult
from .repository import SearchParamDep, VectorSearchParamDep
from .service import IndexService

router = APIRouter(prefix="/index", tags=["index"])


@router.get("/")
async def get_indexes(vector_db: VectorDBDep) -> list[str]:
    try:
        return await IndexService().get_indexes(vector_db=vector_db)
    except ServiceUnavaliable:
        raise service_unavaliable_http_exception
    
@router.post("/")
async def add_index(vector_db: VectorDBDep, index_name: str) -> IndexOperationResult:
    try:
        return await IndexService().add_index(vector_db, index_name)
    except ServiceUnavaliable:
        raise service_unavaliable_http_exception

@router.delete("/{index_name}")
async def delete_index(vector_db: VectorDBDep, index_name: str) -> IndexOperationResult:
    try:
        return await IndexService().delete_index(vector_db, index_name)
    except ServiceUnavaliable:
        raise service_unavaliable_http_exception

@router.get("/{index_name}")
async def get_documents(vector_db: VectorDBDep, search_params: SearchParamDep, index_name: str) -> list[Document]:
    try:
        return await IndexService().get_documents(vector_db=vector_db, search_params=search_params, index_name=index_name)
    except ServiceUnavaliable:
        raise service_unavaliable_http_exception
    
@router.post("/{index_name}/search")
async def search_documents(vector_db: VectorDBDep, search_params: VectorSearchParamDep, index_name: str, query: str, embedder: EmbedderDep) -> DocumentsSearchResult:
    try:
        return await IndexService().search_documents(vector_db=vector_db, search_params=search_params, index_name=index_name, query=query, embedder=embedder)
    except ServiceUnavaliable:
        raise service_unavaliable_http_exception

@router.post("/{index_name}/document")
async def add_documents(vector_db: VectorDBDep, index_name: str, document: Document, embedder: EmbedderDep) -> IndexOperationResult:
    try:
        return await IndexService().add_documents(vector_db=vector_db, index_name=index_name, document=document, embedder=embedder)
    except IndexIsNotExist as unexp_resp:
        raise HTTPException(status_code=404, detail=unexp_resp)
    except ServiceUnavaliable:
        raise service_unavaliable_http_exception

@router.delete("/{index_name}/document")
async def delete_documents(vector_db: VectorDBDep, index_name: str, documents_ids: list[UUID]) -> IndexOperationResult:
    try:
        return await IndexService().delete_documents(vector_db, index_name=index_name, documents_ids=documents_ids)
    except ServiceUnavaliable:
        raise service_unavaliable_http_exception
