import os

from dotenv import load_dotenv
from pydantic import BaseModel

from src.infrastructure.es_index_repository_impl import ESVectorDBConfig
from src.infrastructure.llm_repository_impl import LLMConfig

_ = load_dotenv()


class Config(BaseModel):
    vectordb: ESVectorDBConfig = ESVectorDBConfig(host=os.environ.get("VECTOR_DB_HOST", ""))
    llm_config: LLMConfig = LLMConfig(model_name=os.environ.get("LLM_NAME", ""))
