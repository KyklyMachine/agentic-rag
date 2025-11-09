from uuid import UUID

from fastapi import APIRouter, HTTPException

from src.embeddings.dependency import EmbedderDep

from .dependency import SearchParamDep, VectorDBDep
from .exceptions import IndexIsNotExist
from .model import Document
from .service import IndexService

router = APIRouter(prefix="/index", tags=["index"])


@router.get("/")
async def get_indexes(vector_db: VectorDBDep) -> list[str]:
    return await IndexService().get_indexes(vector_db=vector_db)
    
@router.post("/")
async def add_index(vector_db: VectorDBDep, index_name: str):
    return await IndexService().add_index(vector_db, index_name)

@router.delete("/{index_name}")
async def delete_index(vector_db: VectorDBDep, index_name: str):
    return await IndexService().delete_index(vector_db, index_name)



@router.get("/{index_name}")
async def get_topics(vector_db: VectorDBDep, search_params: SearchParamDep, index_name: str) -> list[Document]:
    return await IndexService().get_topics(vector_db=vector_db, search_params=search_params, index_name=index_name)
    
@router.post("/{index_name}/search")
async def search_topics(vector_db: VectorDBDep, search_params: SearchParamDep, index_name: str, query: str, embedder: EmbedderDep):
    return await IndexService().search_topics(vector_db=vector_db, search_params=search_params, index_name=index_name, query=query, embedder=embedder)

@router.post("/{index_name}/topic")
async def add_topic(vector_db: VectorDBDep, index_name: str, document: Document, embedder: EmbedderDep) -> None:
    try:
        return await IndexService().add_topic(vector_db=vector_db, index_name=index_name, document=document, embedder=embedder)
    except IndexIsNotExist as unexp_resp:
        raise HTTPException(status_code=404, detail=unexp_resp)

@router.delete("/{index_name}/topic")
async def delete_topic(vector_db: VectorDBDep, index_name: str, documents_ids: list[UUID]):
    return await IndexService().delete_topic(vector_db, index_name=index_name, documents_ids=documents_ids)
