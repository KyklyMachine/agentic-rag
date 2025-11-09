import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from src.config import Config
from src.infrastructure.embeddings_repository_impl import QwenEmbedder
from src.infrastructure.index_repository_impl import QdrantVectorDB
from src.router import router

load_dotenv()

config = Config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vector_db = QdrantVectorDB(config=config.vectordb)
    app.state.embedder = QwenEmbedder()
    try:
        yield
    finally:
        return


if __name__ == "__main__":
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    uvicorn.run(app=app)
