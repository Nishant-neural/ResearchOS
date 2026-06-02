import torch

from app.rag.hidden_state_store import deserialize_hidden_state
from app.retrieval.noise_filter import is_retrieval_noise
from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.reranker import rerank
from app.retrieval.qdrant_store import (
    generate_chunk_id,
    client,
    COLLECTION_NAME,
)


def retrieve_with_hidden_states(
    query: str,
    limit: int = 5,
    source_filename: str | None = None,
):
    """
    Retrieve chunks using hybrid retrieval (semantic + keyword) with reranking,
    then return pre-computed full token states.

    The document encoder is not run here. Retrieved hidden states are shaped
    [seq_len, hidden_dim] so they can be assembled into decoder memory.
    """
    candidate_limit = max(limit * 6, limit + 10)
    
    # Step 1: Hybrid retrieval (semantic + keyword)
    hybrid_results = hybrid_search(
        query=query,
        limit=candidate_limit,
        source_filename=source_filename,
    )
    
    # Step 2: Combine semantic and keyword results
    candidate_entries = []
    seen_ids = set()
    
    # Add semantic results
    for result in hybrid_results.get("semantic", []):
        if not is_retrieval_noise(result.payload):
            if source_filename and result.payload.get("source_filename") != source_filename:
                continue
            seen_ids.add(result.id)
            candidate_entries.append(
    {
        "id": result.id,
        "text": result.payload.get("text"),
        "result": result,
        "retrieval_type": "semantic",
    }
)
    
    # Add keyword results (TextChunk objects)
   
    for chunk, score in hybrid_results.get("keyword", []):
        chunk_source = chunk.metadata.get("source_filename", "unknown")

        if source_filename and chunk_source != source_filename:
            continue

        chunk_id = generate_chunk_id(
        filename=chunk_source,
        chunk_index=chunk.chunk_index,
    )

        if chunk_id in seen_ids:
          continue

        seen_ids.add(chunk_id)

    # IMPORTANT:
    # Fetch the FULL Qdrant point so keyword-retrieved chunks
    # can also access stored hidden states.
        retrieved_points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[chunk_id],
    )

        qdrant_result = retrieved_points[0] if retrieved_points else None

        candidate_entries.append(
        {
            "id": chunk_id,
            "text": chunk.text,
            "chunk": chunk,
            "result": qdrant_result,
            "retrieval_type": "keyword",  # may contain hidden states
        }
    )

    texts_to_rerank = [entry["text"] for entry in candidate_entries]
    
    if not texts_to_rerank:
        return []
    
    reranked = rerank(query, texts_to_rerank[:candidate_limit])
    
    retrieved_data = []
    for ranked_text, rerank_score, original_index in reranked[:limit]:
        original_data = candidate_entries[original_index]
        print("\n[DEBUG] RERANK VALIDATION")

        print("RERANKED TEXT:")
        print(ranked_text[:400])

        print("\nMATCHED ORIGINAL:")
        print(original_data["text"][:400])

        print("=" * 80)
        if original_data["result"] is not None:
            result = original_data["result"]
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
                        "rerank_score": float(rerank_score),
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
                    "rerank_score": float(rerank_score),
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

    print("\n[DEBUG] HIDDEN STATE AGGREGATION")

    for i, state in enumerate(hidden_states):
        print(f"Chunk {i} hidden shape: {state.shape}")

    print(f"Final encoder hidden shape: {encoder_hidden_states.shape}")
    print(f"Final attention mask shape: {attention_mask.shape}")

    print("\n=====================\n")
    
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
