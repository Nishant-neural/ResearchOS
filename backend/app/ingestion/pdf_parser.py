import re
from typing import Any

import fitz
from pydantic import BaseModel, Field

from app.ingestion.chunker import TextChunk, chunk_pages


class ParsedPage(BaseModel):
    page_number: int
    text: str
    character_count: int


class ParsedPaper(BaseModel):
    filename: str
    title: str | None = None
    author: str | None = None
    page_count: int
    text: str
    character_count: int
    pages: list[ParsedPage] = Field(default_factory=list)
    chunks: list[TextChunk] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def parse_pdf_bytes(file_bytes: bytes, filename: str) -> ParsedPaper:
    document = fitz.open(stream=file_bytes, filetype="pdf")

    try:
        raw_metadata = document.metadata or {}
        pages: list[ParsedPage] = []
        for page in document:
            page_text = clean_extracted_text(page.get_text("text"))
            pages.append(
                ParsedPage(
                    page_number=page.number + 1,
                    text=page_text,
                    character_count=len(page_text),
                )
            )
    finally:
        document.close()

    full_text = "\n\n".join(page.text for page in pages if page.text)
    title = _clean_metadata_value(raw_metadata.get("title"))
    author = _clean_metadata_value(raw_metadata.get("author"))

    return ParsedPaper(
        filename=filename,
        title=title,
        author=author,
        page_count=len(pages),
        text=full_text,
        character_count=len(full_text),
        pages=pages,
        chunks=chunk_pages(
            pages,
            metadata={
                "source_filename": filename,
                "paper_title": title,
            },
        ),
        metadata={key: value for key, value in raw_metadata.items() if value},
    )


def clean_extracted_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_metadata_value(value: str | None) -> str | None:
    if not value:
        return None

    cleaned = value.strip()
    return cleaned or None
