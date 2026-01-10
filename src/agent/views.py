from fastapi import APIRouter, HTTPException, status

from src.agent.dependency import LLMDependency
from src.agent.dto import AgentRequest, AgentResponse
from src.embeddings.dependency import EmbedderDep
from src.index.dependency import VectorDBDep
from src.index.exceptions import (
    IndexNotFoundException,
    ServiceUnavaliable,
    service_unavaliable_http_exception,
)
from src.index.repository import VectorSearchParamDep

from .service import AgentService

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post(
    "/compleations",
    status_code=status.HTTP_200_OK,
    summary="Get Agent response "
    )
async def compleations(request: AgentRequest, llm: LLMDependency, vector_db: VectorDBDep, embedder: EmbedderDep, search_params: VectorSearchParamDep) -> AgentResponse:
    try:
        answer, docs = await AgentService().generage_response(
            index_name=request.index_name, 
            messages=request.messages, 
            llm=llm, 
            vector_db=vector_db, 
            embedder=embedder, 
            search_params=search_params)
        return AgentResponse(content=answer, documents=docs)
    except ServiceUnavaliable:
        raise service_unavaliable_http_exception
    except IndexNotFoundException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"error": "index_not_found", "message": str(e)})

