from fastapi import APIRouter
from pydantic import BaseModel

from app.retrieval.models import SearchRequest
from app.retrieval.embeddings import embed_text

from app.rag.rag_pipeline import (
    answer_query,
)

from app.rag.hidden_state_pipeline import (
    hidden_state_experiment,
)

from app.rag.hidden_state_store import (
    store_hidden_states_for_texts,
)

from app.rag.hidden_state_retrieval import (
    retrieve_with_hidden_states,
    aggregate_hidden_states,
)

from app.rag.generation import (
    generate_from_hidden_states,
)

import time


router = APIRouter()


class PreprocessRequest(BaseModel):
    texts: list[str]
    chunk_metadata: list[dict]  # Each dict: {"source_filename": str, "chunk_index": int}


class HiddenStateQueryRequest(BaseModel):
    query: str
    limit: int = 5
    aggregation_method: str = "sequence_concatenate"
    conditioning_strategy: str = "framed_memory"


@router.post("/ask")
async def ask_question(
    request: SearchRequest,
):
    return answer_query(
        query=request.query,
        limit=request.limit,
    )


@router.post("/hidden-states")
async def hidden_state_test(
    request: SearchRequest,
):
    return hidden_state_experiment(
        query=request.query,
        limit=request.limit,
    )


@router.post("/preprocess-hidden-states")
async def preprocess_chunks(request: PreprocessRequest):
    """
    Offline preprocessing: extract and store hidden states for chunks.
    
    This should be called after chunks are stored in Qdrant.
    It will:
    1. Extract encoder hidden states for each chunk
    2. Store them in Qdrant payload
    3. Mark chunks as having hidden states available
    """
    result = store_hidden_states_for_texts(
        texts=request.texts,
        chunk_metadata=request.chunk_metadata,
    )

    return {
        "status": "preprocessing_complete",
        "stored": result["stored"],
        "failed": result["failed"],
    }


@router.post("/ask-with-hidden-states")
async def ask_with_hidden_states(request: HiddenStateQueryRequest):
    """
    Query using pre-computed hidden states.
    
    Pipeline:
    1. Embed query using same encoder as retrieval
    2. Retrieve pre-computed hidden states from Qdrant
    3. Concatenate full token hidden states along the sequence dimension
    4. Frame cached memory with encoded separator/instruction states and generate
    
    Aggregation methods:
    - sequence_concatenate: Join retrieved chunk memories along sequence length
    - concatenate: Legacy alias for sequence_concatenate
    - stacking: Legacy alias for sequence_concatenate
    
    Returns: answer + timing + comparison metrics
    """
    query = request.query
    limit = request.limit
    method = request.aggregation_method
    conditioning_strategy = request.conditioning_strategy

    start_time = time.time()

    # Step 1: Embed query for retrieval
    query_embedding = embed_text(query)
    embedding_time = time.time() - start_time

    # Step 2: Retrieve pre-computed hidden states
    start_retrieval = time.time()
    retrieved = retrieve_with_hidden_states(
        query_embedding=query_embedding,
        limit=limit,
    )
    retrieval_time = time.time() - start_retrieval

    # Separate available hidden states from fallback text-only results
    available_hidden_states = [
        r["hidden_state"]
        for r in retrieved
        if r["hidden_state"] is not None
    ]
    available_attention_masks = [
        r["attention_mask"]
        for r in retrieved
        if r["hidden_state"] is not None
    ]
    contexts = [r["text"] for r in retrieved]

    # Step 3: Aggregate hidden states
    if available_hidden_states:
        start_aggregation = time.time()
        
        aggregated_memory = aggregate_hidden_states(
            available_hidden_states,
            available_attention_masks,
            method=method,
        )
        
        aggregation_time = time.time() - start_aggregation

        # Step 4: Generate using hidden states
        start_generation = time.time()
        answer = generate_from_hidden_states(
            query=query,
            encoder_hidden_states=aggregated_memory["encoder_hidden_states"],
            attention_mask=aggregated_memory["attention_mask"],
            conditioning_strategy=conditioning_strategy,
        )
        generation_time = time.time() - start_generation

        used_hidden_states = True
    else:
        # Fallback: no hidden states available yet
        answer = "No pre-computed hidden states available. Please run /api/preprocess-hidden-states first."
        aggregation_time = 0
        generation_time = 0
        used_hidden_states = False

    total_time = time.time() - start_time

    return {
        "query": query,
        "answer": answer,
        "contexts": contexts,
        "aggregation_method": method,
        "conditioning_strategy": conditioning_strategy,
        "used_hidden_states": used_hidden_states,
        "num_chunks_with_hidden_states": len(available_hidden_states),
        "timing": {
            "embedding_ms": round(embedding_time * 1000, 2),
            "retrieval_ms": round(retrieval_time * 1000, 2),
            "aggregation_ms": round(aggregation_time * 1000, 2),
            "generation_ms": round(generation_time * 1000, 2),
            "total_ms": round(total_time * 1000, 2),
        },
    }
