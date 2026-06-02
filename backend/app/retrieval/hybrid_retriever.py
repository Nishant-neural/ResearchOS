from app.retrieval.keyword_search import BM25Retriever


bm25 = BM25Retriever()


def hybrid_search(
    query: str,
    limit: int = 5,
    source_filename: str | None = None,
):
    from app.retrieval.retriever import search as semantic_search
    
    semantic_results = semantic_search(
        query=query,
        limit=limit,
        source_filename=source_filename,
    )

    keyword_results = bm25.search(
        query=query,
        limit=limit,
    )

    return {
        "semantic": semantic_results,
        "keyword": keyword_results,
    }