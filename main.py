import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from src.config import Config
from src.index.repository_impl import QdrantVectorDB

load_dotenv()

config = Config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vector_db = QdrantVectorDB(config=config.vectordb)
    try:
        yield
    finally:
        return


if __name__ == "__main__":
    app = FastAPI(lifespan=lifespan)
    uvicorn.run(app=app)
