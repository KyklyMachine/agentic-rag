from typing import Literal, Optional

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
    ids: list[str]

class IndexInfo(BaseModel):
    name: str
    documents_count: int
