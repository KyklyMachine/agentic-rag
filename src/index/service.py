from uuid import UUID

from .dependency import SearchParamDep, VectorDBDep
from .model import Document


class IndexService:
    async def get_indexes(self, vector_db: VectorDBDep) -> list[str]:
        return await vector_db.get_indexes()
    async def add_index(self, vector_db: VectorDBDep, index_name: str) -> None:
        return await vector_db.add_index(index_name=index_name)
    async def delete_index(self, vector_db: VectorDBDep, index_name: str) -> None:
        return await vector_db.delete_index(index_name=index_name)

    async def get_topics(self, vector_db: VectorDBDep, search_params: SearchParamDep, index_name: str) -> list[Document]:
        return await vector_db.get_topics(index_name=index_name, search_params=search_params)
    async def search_topics(self, vector_db: VectorDBDep, search_params: SearchParamDep, index_name: str, query: list[float]) -> None:
        await vector_db.search_topics(index_name=index_name, search_params=search_params, query=query)
    async def add_topic(self, vector_db: VectorDBDep, index_name: str, topic: Document) -> None:
        await vector_db.add_topic(index_name=index_name, document=topic)
    async def delete_topic(self, vector_db: VectorDBDep, index_name: str, topics_ids: list[UUID]) -> None:
        await vector_db.delete_topics(index_name=index_name, topics_ids=topics_ids)
