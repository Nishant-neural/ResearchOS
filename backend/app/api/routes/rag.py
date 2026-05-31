from fastapi import APIRouter

from app.retrieval.models import SearchRequest
from app.rag.rag_pipeline import answer_query


router = APIRouter()


@router.post("/ask")
async def ask_question(
    request: SearchRequest,
):
    return answer_query(
        query=request.query,
        limit=request.limit,
    )