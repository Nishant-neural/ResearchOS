from fastapi import APIRouter

from app.retrieval.models import SearchRequest
from app.retrieval.retriever import search_hybrid_with_reranking


router = APIRouter()


@router.post("/search")
async def semantic_search(
    request: SearchRequest,
):
    results = search_hybrid_with_reranking(
        query=request.query,
        limit=request.limit,
        source_filename=request.source_filename,
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
        "source_filename": request.source_filename,
        "results": formatted,
    }
