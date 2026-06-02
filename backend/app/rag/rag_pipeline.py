from app.rag.prompt_builder import build_rag_prompt
from app.rag.transformer import generate_text
from app.retrieval.retriever import search_hybrid_with_reranking


def answer_query(
    query: str,
    limit: int = 5,
    source_filename: str | None = None,
):
    retrieved = search_hybrid_with_reranking(
        query=query,
        limit=limit,
        source_filename=source_filename,
    )

    contexts = [
        result.payload["text"]
        for result in retrieved
    ]

    prompt = build_rag_prompt(
        query=query,
        contexts=contexts,
    )

    answer = generate_text(prompt)

    return {
        "query": query,
        "source_filename": source_filename,
        "answer": answer,
        "contexts": contexts,
    }
