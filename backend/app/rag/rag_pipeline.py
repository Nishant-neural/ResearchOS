from app.rag.prompt_builder import build_rag_prompt
from app.rag.transformer import generate_text
from app.retrieval.retriever import search


def answer_query(
    query: str,
    limit: int = 5,
):
    retrieved = search(
        query=query,
        limit=limit,
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
        "answer": answer,
        "contexts": contexts,
    }
