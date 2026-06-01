import torch

from app.rag.hidden_state_store import deserialize_hidden_state
from app.retrieval.qdrant_store import COLLECTION_NAME, client


def retrieve_with_hidden_states(
    query_embedding: list[float],
    limit: int = 5,
):
    """
    Retrieve chunks by similarity and return pre-computed full token states.

    The document encoder is not run here. Retrieved hidden states are shaped
    [seq_len, hidden_dim] so they can be assembled into decoder memory.
    """
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=limit,
    ).points

    retrieved_data = []

    for result in results:
        payload = result.payload
        has_hidden_states = payload.get("has_hidden_states")
        hidden_state_json = payload.get("hidden_state_full")
        attention_mask_json = payload.get("hidden_state_attention_mask")

        if has_hidden_states and hidden_state_json and attention_mask_json:
            hidden_state = deserialize_hidden_state(hidden_state_json)
            attention_mask = deserialize_hidden_state(attention_mask_json).long()

            retrieved_data.append(
                {
                    "chunk_id": result.id,
                    "text": payload.get("text"),
                    "hidden_state": hidden_state,
                    "attention_mask": attention_mask,
                    "similarity_score": result.score,
                    "metadata": _metadata_from_payload(payload),
                }
            )
            continue

        retrieved_data.append(
            {
                "chunk_id": result.id,
                "text": payload.get("text"),
                "hidden_state": None,
                "attention_mask": None,
                "similarity_score": result.score,
                "metadata": _metadata_from_payload(payload),
            }
        )

    return retrieved_data


def aggregate_hidden_states(
    hidden_states: list[torch.Tensor],
    attention_masks: list[torch.Tensor],
    method: str = "sequence_concatenate",
) -> dict[str, torch.Tensor] | None:
    """
    Assemble FLAN-T5-compatible encoder memory from retrieved chunk states.

    The only generation-safe aggregation here is sequence concatenation:
    [seq1, hidden_dim] + [seq2, hidden_dim] -> [1, seq1 + seq2, hidden_dim].
    Legacy aliases are kept so older requests do not break immediately.
    """
    if not hidden_states:
        return None

    if method not in {"sequence_concatenate", "concatenate", "stacking"}:
        raise ValueError(
            f"Aggregation method {method!r} is not compatible with full FLAN-T5 "
            "encoder memory. Use 'sequence_concatenate'."
        )

    if len(hidden_states) != len(attention_masks):
        raise ValueError("hidden_states and attention_masks must have the same length")

    hidden_dims = {state.shape[-1] for state in hidden_states}
    if len(hidden_dims) != 1:
        raise ValueError("All hidden states must have the same hidden dimension")

    encoder_hidden_states = torch.cat(hidden_states, dim=0).unsqueeze(0)
    attention_mask = torch.cat(attention_masks, dim=0).unsqueeze(0)

    return {
        "encoder_hidden_states": encoder_hidden_states,
        "attention_mask": attention_mask,
    }


def _metadata_from_payload(payload: dict) -> dict:
    return {
        "source": payload.get("source_filename"),
        "chunk_index": payload.get("chunk_index"),
        "page": payload.get("start_page"),
    }
