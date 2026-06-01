import torch

from app.retrieval.retriever import search

from app.rag.transformer import (
    extract_hidden_states,
)


def mean_pool_hidden_state(
    hidden_state,
):
    return hidden_state.mean(dim=1)


def hidden_state_experiment(
    query: str,
    limit: int = 5,
):
    retrieved = search(
        query=query,
        limit=limit,
    )

    results = []

    for item in retrieved:
        text = item.payload["text"]

        hidden_outputs = extract_hidden_states(
            text
        )

        pooled = mean_pool_hidden_state(
            hidden_outputs["last_hidden_state"]
        )

        results.append(
            {
                "text": text,
                "hidden_state_shape": list(
                    hidden_outputs[
                        "last_hidden_state"
                    ].shape
                ),
                "pooled_shape": list(
                    pooled.shape
                ),
                "pooled_vector": pooled.tolist(),
            }
        )

    return {
        "query": query,
        "results": results,
    }