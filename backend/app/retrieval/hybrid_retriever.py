from app.retrieval.retriever import search as semantic_search
from app.retrieval.keyword_search import BM25Retriever


bm25 = BM25Retriever()


def hybrid_search(query: str):
    semantic_results = semantic_search(query)

    keyword_results = bm25.search(query)

    return {
        "semantic": semantic_results,
        "keyword": keyword_results,
    }