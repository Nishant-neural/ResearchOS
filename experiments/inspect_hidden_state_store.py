"""
Inspect whether Qdrant chunks have cached FLAN-T5 hidden states.
"""

from __future__ import annotations

from qdrant_client import QdrantClient


COLLECTION_NAME = "research_memory"


def main() -> None:
    client = QdrantClient(host="localhost", port=6333)
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )

    total = len(points)
    hidden_ready = 0

    print(f"Inspected points: {total}")
    for point in points:
        payload = point.payload or {}
        has_hidden = bool(payload.get("has_hidden_states"))
        has_full = bool(payload.get("hidden_state_full"))
        has_mask = bool(payload.get("hidden_state_attention_mask"))
        if has_hidden and has_full and has_mask:
            hidden_ready += 1

        print(
            "id={id} chunk_index={chunk} hidden={hidden} full={full} "
            "mask={mask} shape={shape}".format(
                id=point.id,
                chunk=payload.get("chunk_index"),
                hidden=has_hidden,
                full=has_full,
                mask=has_mask,
                shape=payload.get("hidden_state_shape"),
            )
        )

    print(f"\nHidden-state-ready chunks: {hidden_ready}/{total}")


if __name__ == "__main__":
    main()
