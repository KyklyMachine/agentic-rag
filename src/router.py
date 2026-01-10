from fastapi import APIRouter

from src.agent.views import router as agent_router
from src.index.views import router as index_router

router = APIRouter()

router.include_router(index_router)
router.include_router(agent_router)
