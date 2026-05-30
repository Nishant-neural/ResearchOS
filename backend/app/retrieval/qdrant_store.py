import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.ingestion.chunker import TextChunk
from app.retrieval.embeddings import embed_texts


COLLECTION_NAME = "research_memory"


client = QdrantClient(
    host="localhost",
    port=6333,
)


def initialize_collection() -> None:
    collections = client.get_collections().collections
    existing = [collection.name for collection in collections]

    if COLLECTION_NAME in existing:
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )


def generate_chunk_id(
    filename: str,
    chunk_index: int,
) -> str:
    raw = f"{filename}:{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def store_chunks(chunks: list[TextChunk]) -> None:
    if not chunks:
        return

    texts = [chunk.text for chunk in chunks]

    embeddings = embed_texts(texts)

    points = []

    for chunk, embedding in zip(chunks, embeddings):
        filename = str(
            chunk.metadata.get(
                "source_filename",
                "unknown",
            )
        )

        point = PointStruct(
            id=generate_chunk_id(
                filename=filename,
                chunk_index=chunk.chunk_index,
            ),
            vector=embedding,
            payload={
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "character_count": chunk.character_count,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                **chunk.metadata,
            },
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )