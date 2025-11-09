import os

from dotenv import load_dotenv
from pydantic import BaseModel

from src.infrastructure.index_repository_impl import QdrantVectorDBConfig

load_dotenv()


class Config(BaseModel):
    vectordb: QdrantVectorDBConfig = QdrantVectorDBConfig(url=os.environ.get("VECTOR_DB_URL", ""))


