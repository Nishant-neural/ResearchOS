def build_rag_prompt(
    query: str,
    contexts: list[str],
) -> str:
    joined_context = "\n\n".join(contexts)

    return f"""
You are a research assistant.

Answer the question using ONLY the provided context.

If the answer is not in the context, say:
"I could not find the answer in the provided documents."

Context:
{joined_context}

Question:
{query}

Answer:
""".strip()