import json
import torch

from qdrant_client import QdrantClient

from app.retrieval.qdrant_store import COLLECTION_NAME, generate_chunk_id
from app.rag.transformer import extract_hidden_states


client = QdrantClient(
    host="localhost",
    port=6333,
)


def serialize_hidden_state(hidden_state: torch.Tensor) -> str:
    """Convert tensor to JSON-serializable string."""
    return json.dumps(hidden_state.cpu().numpy().tolist())


def deserialize_hidden_state(hidden_state_json: str) -> torch.Tensor:
    """Convert JSON string back to tensor."""
    return torch.tensor(json.loads(hidden_state_json), dtype=torch.float32)


def store_hidden_states_for_texts(
    texts: list[str],
    chunk_metadata: list[dict],
) -> dict:
    """
    Preprocess texts offline: extract hidden states and store them.
    
    Args:
        texts: List of text chunks
        chunk_metadata: List of dicts with 'source_filename' and 'chunk_index'
    
    Returns:
        Status dict with count of stored hidden states
    """
    if not texts:
        return {"stored": 0, "failed": 0}

    stored_count = 0
    failed_count = 0

    for text, metadata in zip(texts, chunk_metadata):
        try:
            # Store full per-token encoder memory for decoder cross-attention.
            hidden_outputs = extract_hidden_states(text)
            last_hidden_state = hidden_outputs["last_hidden_state"]
            attention_mask = hidden_outputs["attention_mask"]
            pooled_state = last_hidden_state.mean(dim=1)[0]

            filename = metadata.get("source_filename", "unknown")
            chunk_index = metadata.get("chunk_index", 0)

            chunk_id = generate_chunk_id(
                filename=filename,
                chunk_index=chunk_index,
            )

            # Retrieve existing point
            try:
                existing = client.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=[chunk_id],
                )
                if existing:
                    client.set_payload(
                        collection_name=COLLECTION_NAME,
                        points=[chunk_id],
                        payload={
                            "hidden_state_pooled": serialize_hidden_state(
                                pooled_state
                            ),
                            "hidden_state_full": serialize_hidden_state(
                                last_hidden_state[0]
                            ),
                            "hidden_state_attention_mask": serialize_hidden_state(
                                attention_mask[0]
                            ),
                            "hidden_state_shape": list(last_hidden_state[0].shape),
                            "encoder_model": "google/flan-t5-base",
                            "has_hidden_states": True,
                        },
                    )
                    stored_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"Error updating chunk {chunk_id}: {e}")
                failed_count += 1

        except Exception as e:
            print(f"Error processing text: {e}")
            failed_count += 1

    return {
        "stored": stored_count,
        "failed": failed_count,
    }


def has_hidden_states(chunk_id: str) -> bool:
    """Check if a chunk has pre-computed hidden states."""
    try:
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[chunk_id],
        )
        if points:
            return points[0].payload.get("has_hidden_states", False)
        return False
    except:
        return False
