import os
from typing import List, Optional

from openai import OpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse

from ..document.model import Document
from ..embeddings.exceptions import ModelNotFoundException
from ..embeddings.repository import Embedder


class QwenEmbedder(Embedder):
    _client: OpenAI
    def __init__(self) -> None:
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_TOKEN"),
            )

    async def invoke(self, documents: list[Document], model_name: Optional[str]=None) -> list[Document]: 
        contents: list[str] = [document.payload.content for document in documents]
        model = ""
        if "EMBEDDER_MODEL" in os.environ and not model_name:
            model = os.environ.get("EMBEDDER_MODEL", "")
        elif model_name:
            model = model_name
        else:
            raise ModelNotFoundException("Embedding Model not found in ENV. Please, set EMBEDDER_MODEL")
        embeddings_response: CreateEmbeddingResponse = self._client.embeddings.create(
            model=model,
            input=contents,
            encoding_format="float"
        )
        embeddings: list[List[float]] = [embed.embedding for embed in embeddings_response.data]
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        return documents
