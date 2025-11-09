from uuid import UUID

from ..embeddings.dependency import EmbedderDep
from ..index.model import DocumentsSearchResult
from .dependency import SearchParamDep, VectorDBDep
from .model import Document

# TODO: 2) нужно сделать отдельный dep для параметров поиска в индексе
# TODO: 4) нужно сделать обработку ошибок если нет инфры (например, qdrant)
# TODO: 5) добавить ответы при удаче запросов
# TODO: 6) привязать к индексу эмбеддер

class IndexService:
    async def get_indexes(self, vector_db: VectorDBDep) -> list[str]:
        return await vector_db.get_indexes()
    
    async def add_index(self, vector_db: VectorDBDep, index_name: str) -> None:
        return await vector_db.add_index(index_name=index_name)
    
    async def delete_index(self, vector_db: VectorDBDep, index_name: str) -> None:
        return await vector_db.delete_index(index_name=index_name)

    async def get_documents(self, vector_db: VectorDBDep, search_params: SearchParamDep, index_name: str) -> list[Document]:
        return await vector_db.get_documents(index_name=index_name, search_params=search_params)
    
    async def search_documents(self, vector_db: VectorDBDep, search_params: SearchParamDep, index_name: str, query: str, embedder: EmbedderDep) -> DocumentsSearchResult:
        return await vector_db.search_documents(index_name=index_name, search_params=search_params, query=query, embedder=embedder)
    
    async def add_documents(self, vector_db: VectorDBDep, index_name: str, document: Document, embedder: EmbedderDep) -> None:
        if not document.embedding:
            document = (await embedder.invoke([document]))[0]
        await vector_db.add_documents(index_name=index_name, document=document, embedder=embedder)
    
    async def delete_documents(self, vector_db: VectorDBDep, index_name: str, documents_ids: list[UUID]) -> None:
        await vector_db.delete_documents(index_name=index_name, documents_ids=documents_ids)
