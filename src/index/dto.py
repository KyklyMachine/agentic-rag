from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.document.model import Document, Metadata


class DocumentMetadataDTO(BaseModel):
    source: Optional[str] = Field(None, description="The source of the document")
    tags: list[str] = Field(default_factory=list, description="Tags associated with the document")

class DocumentDTO(BaseModel):
    title: Optional[str] = Field(None, description="Title of the document")
    context: Optional[str] = Field(None, description="Context or summary of the document")
    content: str = Field(..., description="Content of the document")
    metadata: Optional[DocumentMetadataDTO] = Field(None, description="Additional metadata about the document")


class AddDocumentsRequest(BaseModel):
    documents: list[DocumentDTO] = Field(..., description="List of documents to add")

    def to_document_model(self) -> list[Document]:
        return [
            Document(
                id=uuid4(),
                title=doc.title,
                context=doc.context,
                content=doc.content,
                embedding=None,
                metadata=Metadata() if not doc.metadata else Metadata(**doc.metadata.model_dump())
            ) for doc in self.documents]
