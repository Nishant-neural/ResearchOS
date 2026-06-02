from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from app.retrieval.embeddings import embed_text
from app.retrieval.noise_filter import is_retrieval_noise
from app.retrieval.qdrant_store import (
    COLLECTION_NAME,
    client,
    generate_chunk_id,
)
from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.reranker import rerank


def search(
    query: str,
    limit: int = 5,
    source_filename: str | None = None,
):
    """Semantic search only (legacy, kept for backward compatibility)."""
    query_vector = embed_text(query)
    candidate_limit = max(limit * 6, limit + 10)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=_source_filter(source_filename),
        limit=candidate_limit,
    ).points

    filtered_results = [
        result
        for result in results
        if not is_retrieval_noise(result.payload)
    ]

    return (filtered_results or results)[:limit]


def search_hybrid_with_reranking(
    query: str,
    limit: int = 5,
    source_filename: str | None = None,
):
    """
    Hybrid retrieval (semantic + keyword) with cross-encoder reranking.
    Returns results in the same format as semantic search for compatibility.
    """
    candidate_limit = max(limit * 6, limit + 10)
    
    # Step 1: Hybrid retrieval (semantic + keyword)
    hybrid_results = hybrid_search(
        query=query,
        limit=candidate_limit,
        source_filename=source_filename,
    )
    
    # Step 2: Combine results and apply source filtering
    candidate_entries = []
    seen_ids = set()
    
    # Add semantic results
    for result in hybrid_results.get("semantic", []):
        if not is_retrieval_noise(result.payload):
            if source_filename and result.payload.get("source_filename") != source_filename:
                continue
            seen_ids.add(result.id)
            candidate_entries.append(
                {
                    "id": result.id,
                    "text": result.payload.get("text"),
                    "result": result,
                    "is_semantic": True,
                }
            )
    
    # Add keyword results (convert TextChunk to compatible format)
    for chunk, score in hybrid_results.get("keyword", []):
        chunk_source = chunk.metadata.get("source_filename", "unknown")
        if source_filename and chunk_source != source_filename:
            continue
        
        chunk_id = generate_chunk_id(
            filename=chunk_source,
            chunk_index=chunk.chunk_index,
        )
        
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        candidate_entries.append(
            {
                "id": chunk_id,
                "text": chunk.text,
                "is_semantic": False,
                "keyword_score": score,
                "chunk": chunk,
                "chunk_id": chunk_id,
            }
        )
    
    texts_to_rerank = [entry["text"] for entry in candidate_entries]
    
    if not texts_to_rerank:
        return []
    
    reranked = rerank(query, texts_to_rerank[:candidate_limit])
    
    retrieved_results = []
    for ranked_text, rerank_score, original_index in reranked[:limit]:
        original_data = candidate_entries[original_index]
        
        if original_data["is_semantic"]:
            result = original_data["result"]
            result.payload["rerank_score"] = float(rerank_score)
            retrieved_results.append(result)
        else:
            chunk = original_data["chunk"]
            chunk_source = chunk.metadata.get("source_filename", "unknown")
            point = PointStruct(
                id=original_data["chunk_id"],
                vector=[],
                payload={
                    "text": chunk.text,
                    "source_filename": chunk_source,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "character_count": chunk.character_count,
                    "start_page": chunk.start_page,
                    "end_page": chunk.end_page,
                    "rerank_score": float(rerank_score),
                    "keyword_score": original_data["keyword_score"],
                    **chunk.metadata,
                },
            )
            point.score = float(rerank_score)
            retrieved_results.append(point)
    
    return retrieved_results


def _source_filter(source_filename: str | None) -> Filter | None:
    if not source_filename:
        return None

    return Filter(
        must=[
            FieldCondition(
                key="source_filename",
                match=MatchValue(value=source_filename),
            )
        ]
    )
