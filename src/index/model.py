from typing import Annotated, Any, Literal, Optional

from fastapi import Depends
from pydantic import BaseModel, Field

from src.document.model import Document


class Index(BaseModel):
    name: str
    documents: Optional[list[Document]]

class SearchParams(BaseModel):
    offset: int = Field(ge=0, default=0)
    limit: int = Field(ge=1, le=100, default=5)
    return_embeddings: bool = Field(default=True)

class VectorSearchParam(BaseModel):
    limit: int = Field(ge=0, le=100, default=4)
    return_embeddings: bool = Field(default=True)

class SearchItem(BaseModel):
    document: Document
    score: float

class DocumentsSearchResult(BaseModel):
    index_name: str
    search_params: VectorSearchParam
    items: list[SearchItem]

class IndexOperationResult(BaseModel):
    status: Literal["acknowledged"] = Field(default="acknowledged")
    operation: Literal["delete_index", "add_index", "delete_documents", "add_documents"]
    result_description: Optional[dict[Any, Any]] = Field(default=None, description="Dictionary with result details")
    errors: Optional[dict[Any, Any]] = Field(default=None, description="Dictionary of errors, where key is document id and value is error message")

class IndexInfo(BaseModel):
    name: str
    documents_count: int

SearchParamDep = Annotated[SearchParams, Depends(SearchParams)]
VectorSearchParamDep = Annotated[VectorSearchParam, Depends(VectorSearchParam)]
