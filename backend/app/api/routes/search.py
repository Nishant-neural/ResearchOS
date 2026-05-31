from fastapi import APIRouter

from app.retrieval.models import SearchRequest
from app.retrieval.retriever import search


router = APIRouter()


@router.post("/search")
async def semantic_search(
    request: SearchRequest,
):
    results = search(
        query=request.query,
        limit=request.limit,
    )

    formatted = []

    for result in results:
        formatted.append(
            {
                "chunk_id": str(result.id),
                "score": result.score,
                "text": result.payload["text"],
                "metadata": result.payload,
            }
        )

    return {
        "query": request.query,
        "results": formatted,
    }
