from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from app.ingestion.pdf_parser import ParsedPaper, parse_pdf_bytes

from app.rag.hidden_state_store import store_hidden_states_for_texts
from app.retrieval.qdrant_store import (
    initialize_collection,
    store_chunks,
)
from app.retrieval.keyword_search import bm25


router = APIRouter()


@router.post("/upload-paper", response_model=ParsedPaper)
async def upload_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> ParsedPaper:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    if file.content_type not in {
        "application/pdf",
        "application/octet-stream",
    }:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a PDF.",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        parsed_paper = parse_pdf_bytes(
            file_bytes=file_bytes,
            filename=file.filename,
        )

        initialize_collection()

        # Add source_filename to chunk metadata before storing
        for chunk in parsed_paper.chunks:
            chunk.metadata["source_filename"] = file.filename

        store_chunks(parsed_paper.chunks)

        # Also add chunks to BM25 retriever for keyword search
        bm25.add_chunks(parsed_paper.chunks)

        if parsed_paper.chunks:
            texts = [chunk.text for chunk in parsed_paper.chunks]
            chunk_metadata = [
                {
                    "source_filename": file.filename,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in parsed_paper.chunks
            ]
            background_tasks.add_task(
                store_hidden_states_for_texts,
                texts=texts,
                chunk_metadata=chunk_metadata,
            )

        return parsed_paper

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not parse PDF file.",
        ) from exc
