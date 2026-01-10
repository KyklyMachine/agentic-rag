from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessage

from index.model import DocumentsSearchResult
from src.agent.dependency import LLMDependency
from src.embeddings.dependency import EmbedderDep
from src.index.dependency import VectorDBDep
from src.index.model import VectorSearchParamDep
from src.index.service import IndexService


class AgentService:
    async def generage_response(self, index_name: Optional[str], messages: list[dict[str, str]], llm: LLMDependency, vector_db: VectorDBDep, embedder: EmbedderDep, search_params: VectorSearchParamDep) -> tuple[str, list[str]]:
        if not messages:
            raise ValueError("Messages list cannot be empty")
        if isinstance(messages[-1], HumanMessage):
            raise ValueError("Last message in the list must be a HumanMessage")
        if index_name:
            documents: DocumentsSearchResult = await IndexService().search_documents(
                vector_db=vector_db,
                search_params=search_params,
                index_name=index_name,
                query=messages[-1]["content"],
                embedder=embedder
            )
            docs_to_process: list[dict[str, str]] = [{"role": "user", "content": doc.document.content} for doc in documents.items]
        else:
            docs_to_process = []
        
        total_messages = messages + docs_to_process
        response: AIMessage = await llm.invoke(total_messages)
        return response.model_dump().get("content", ""), [d["content"] for d in docs_to_process]
