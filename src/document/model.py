from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class Metadata(BaseModel):
    source: str = Field(default="", description="Источник документа")
    tags: list[str] = Field(default_factory=list, description="Теги для категоризации")

class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    
    title: str = Field(default="", description="Заголовок документа")
    context: str = Field(default="", description="Контекст для понимания документа")
    content: str = Field(..., description="Основной текст документа")
    
    embedding_text: Optional[str] = Field(default=None, description="Текст использованный для построения эмбеддинга (title + context + text)")
    embedding: Optional[list[float]] = Field(default=None, description="Векторное представление")
    
    metadata: Metadata = Field(default=Metadata())

    @model_validator(mode='after')
    def generate_embedding_text(self):
        """Автоматически формирует embedding_text из title, context и content"""
        if self.embedding_text is None:
            parts = []
            if self.title:
                parts.append(self.title)
            if self.context:
                parts.append(self.context)
            if self.content:
                parts.append(self.content)
            
            self.embedding_text = "\n".join(parts)
        
        return self
