from fastapi import APIRouter

from app.api.routes import health, papers, rag, search


api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(papers.router, tags=["papers"])
api_router.include_router(search.router, tags=["search"])
api_router.include_router(rag.router, tags=["rag"])
