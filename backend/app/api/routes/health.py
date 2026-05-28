from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings


router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    environment: str
    qdrant_url: str


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        environment=settings.app_env,
        qdrant_url=settings.qdrant_url,
    )
