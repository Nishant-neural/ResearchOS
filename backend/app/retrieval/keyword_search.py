from rank_bm25 import BM25Okapi

from app.ingestion.chunker import TextChunk


class BM25Retriever:
    def __init__(self):
        self.chunks: list[TextChunk] = []
        self.bm25 = None

    def add_chunks(
        self,
        chunks: list[TextChunk],
    ) -> None:
        self.chunks.extend(chunks)

        tokenized = [
            chunk.text.split()
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized)

    def search(
        self,
        query: str,
        limit: int = 5,
    ):
        if not self.bm25:
            return []

        scores = self.bm25.get_scores(
            query.split()
        )

        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:limit]