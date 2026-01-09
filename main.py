import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from infrastructure.es_index_repository_impl import ESVectorDB
from src.config import Config
from src.infrastructure.embeddings_repository_impl import QwenEmbedder
from src.router import router

_ = load_dotenv()

config = Config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vector_db = ESVectorDB(config=config.vectordb)
    app.state.embedder = QwenEmbedder()
    try:
        yield
    finally:
        return


if __name__ == "__main__":
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    uvicorn.run(app=app)
