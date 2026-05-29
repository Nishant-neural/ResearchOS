from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.ingestion.pdf_parser import ParsedPaper, parse_pdf_bytes


router = APIRouter()


@router.post("/upload-paper", response_model=ParsedPaper)
async def upload_paper(file: UploadFile = File(...)) -> ParsedPaper:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    if file.content_type not in {"application/pdf", "application/octet-stream"}:
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
        return parse_pdf_bytes(file_bytes=file_bytes, filename=file.filename)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not parse PDF file.",
        ) from exc
