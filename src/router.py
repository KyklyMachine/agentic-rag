from fastapi import APIRouter

from src.index.views import router as index_router

router = APIRouter()

router.include_router(index_router)
