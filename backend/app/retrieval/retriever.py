from app.retrieval.embeddings import embed_text
from app.retrieval.qdrant_store import (
    COLLECTION_NAME,
    client,
)


def search(
    query: str,
    limit: int = 5,
):
    query_vector = embed_text(query)

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit,
    )

    return results