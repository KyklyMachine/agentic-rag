import os

from dotenv import load_dotenv
from infrastructure.es_index_repository_impl import ESVectorDBConfig
from pydantic import BaseModel

load_dotenv()


class Config(BaseModel):
    vectordb: ESVectorDBConfig = ESVectorDBConfig(host=os.environ.get("VECTOR_DB_HOST", ""))
