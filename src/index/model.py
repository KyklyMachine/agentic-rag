from typing import Optional

from pydantic import BaseModel, Field

from src.document.model import Document


class Index(BaseModel):
    name: str
    documents: Optional[list[Document]]

class SearchParams(BaseModel):
    offset: int = Field(ge=0, default=0)
    limit: int = Field(ge=1, le=100, default=5)

class SearchItem(BaseModel):
    document: Document
    score: float

class DocumentsSearchResult(BaseModel):
    index_name: str
    search_params: SearchParams
    items: list[SearchItem]
