import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    chunk_index: int
    text: str
    token_count: int
    character_count: int
    start_page: int | None = None
    end_page: int | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class ChunkingConfig(BaseModel):
    max_tokens: int = 512
    overlap_tokens: int = 64


class TokenCounter(Protocol):
    def count(self, text: str) -> int:
        """Return the number of model-sized tokens in the given text."""


class SimpleTokenCounter:
    """Approximate token counter that can be swapped for tiktoken later."""

    def count(self, text: str) -> int:
        return len(_TOKEN_PATTERN.findall(text))


class PageLike(Protocol):
    page_number: int
    text: str


@dataclass(frozen=True)
class TextUnit:
    text: str
    page_number: int | None = None
    metadata: dict[str, str | int | float | bool | None] | None = None


@dataclass(frozen=True)
class ChunkDraft:
    text: str
    start_page: int | None = None
    end_page: int | None = None
    metadata: dict[str, str | int | float | bool | None] | None = None


class ChunkingEngine:
    def __init__(
        self,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._token_counter = token_counter or SimpleTokenCounter()

    def chunk_text(
        self,
        text: str,
        config: ChunkingConfig | None = None,
        metadata: dict[str, str | int | float | bool | None] | None = None,
    ) -> list[TextChunk]:
        return self.chunk_units([TextUnit(text=text, metadata=metadata)], config=config)

    def chunk_pages(
        self,
        pages: Sequence[PageLike],
        config: ChunkingConfig | None = None,
        metadata: dict[str, str | int | float | bool | None] | None = None,
    ) -> list[TextChunk]:
        units = [
            TextUnit(text=page.text, page_number=page.page_number, metadata=metadata)
            for page in pages
            if page.text.strip()
        ]
        return self.chunk_units(units, config=config)

    def chunk_units(
        self,
        units: Sequence[TextUnit],
        config: ChunkingConfig | None = None,
    ) -> list[TextChunk]:
        active_config = config or ChunkingConfig()
        _validate_config(active_config)

        normalized_units = [
            TextUnit(
                text=_normalize_chunk_text(unit.text),
                page_number=unit.page_number,
                metadata=unit.metadata,
            )
            for unit in units
            if unit.text.strip()
        ]
        drafts = _split_section_aware_chunks(
            normalized_units,
            active_config,
            self._token_counter,
        )

        return [
            TextChunk(
                chunk_index=index,
                text=draft.text,
                token_count=self._token_counter.count(draft.text),
                character_count=len(draft.text),
                start_page=draft.start_page,
                end_page=draft.end_page,
                metadata={
                    "method": "section_aware",
                    "max_tokens": active_config.max_tokens,
                    "overlap_tokens": active_config.overlap_tokens,
                    **(draft.metadata or {}),
                    **_chunk_identity_metadata(index, draft.metadata),
                },
            )
            for index, draft in enumerate(drafts)
            if draft.text.strip()
        ]


_TOKEN_PATTERN = re.compile(r"\S+")
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
_NUMBERED_HEADING_PATTERN = re.compile(r"^\d+(\.\d+)*\.?\s+.+$")
_KNOWN_SECTION_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "literature review",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiment",
    "experiments",
    "evaluation",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "references",
    "bibliography",
    "appendix",
    "acknowledgements",
    "acknowledgments",
}


def chunk_text(
    text: str,
    config: ChunkingConfig | None = None,
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> list[TextChunk]:
    return ChunkingEngine().chunk_text(text, config=config, metadata=metadata)


def chunk_pages(
    pages: Sequence[PageLike],
    config: ChunkingConfig | None = None,
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> list[TextChunk]:
    return ChunkingEngine().chunk_pages(pages, config=config, metadata=metadata)


def _validate_config(config: ChunkingConfig) -> None:
    if config.max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    if config.overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")

    if config.overlap_tokens >= config.max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens.")


def _split_section_aware_chunks(
    units: Sequence[TextUnit],
    config: ChunkingConfig,
    token_counter: TokenCounter,
) -> list[ChunkDraft]:
    section_units: list[TextUnit] = []
    drafts: list[ChunkDraft] = []
    current_section: str | None = None

    for unit in units:
        for paragraph in _paragraphs(unit.text):
            if _looks_like_section_heading(paragraph, token_counter):
                current_section = _normalize_section_heading(paragraph)
                continue

            metadata = {**(unit.metadata or {})}
            if current_section:
                metadata["section"] = current_section

            if token_counter.count(paragraph) <= config.max_tokens:
                section_units.append(
                    TextUnit(
                        text=paragraph,
                        page_number=unit.page_number,
                        metadata=metadata,
                    )
                )
                continue

            section_units.extend(
                _split_large_paragraph(
                    paragraph,
                    unit.page_number,
                    config,
                    token_counter,
                    metadata=metadata,
                )
            )

    current_section = None
    current_units: list[TextUnit] = []

    for unit in section_units:
        section = _metadata_value(unit, "section")
        if current_units and section != current_section:
            drafts.extend(_pack_units(current_units, config, token_counter))
            current_units = []

        current_section = section
        current_units.append(unit)

    if current_units:
        drafts.extend(_pack_units(current_units, config, token_counter))

    return drafts


def _split_large_paragraph(
    paragraph: str,
    page_number: int | None,
    config: ChunkingConfig,
    token_counter: TokenCounter,
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> list[TextUnit]:
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY_PATTERN.split(paragraph)
        if sentence.strip()
    ]
    units: list[TextUnit] = []

    for sentence in sentences:
        if token_counter.count(sentence) <= config.max_tokens:
            units.append(
                TextUnit(text=sentence, page_number=page_number, metadata=metadata)
            )
            continue

        units.extend(
            TextUnit(text=draft.text, page_number=page_number, metadata=metadata)
            for draft in _split_token_window(
                [TextUnit(text=sentence, page_number=page_number, metadata=metadata)],
                max_tokens=config.max_tokens,
            )
        )

    return units


def _split_token_window(
    units: Sequence[TextUnit],
    max_tokens: int,
) -> list[ChunkDraft]:
    metadata = _shared_metadata(units)
    words_with_pages = _words_with_pages(units)
    if not words_with_pages:
        return []

    drafts: list[ChunkDraft] = []

    for start in range(0, len(words_with_pages), max_tokens):
        window = words_with_pages[start : start + max_tokens]
        text = " ".join(word for word, _ in window)
        page_numbers = [page for _, page in window if page is not None]
        drafts.append(
            ChunkDraft(
                text=text,
                start_page=min(page_numbers) if page_numbers else None,
                end_page=max(page_numbers) if page_numbers else None,
                metadata=metadata,
            )
        )

    return drafts


def _pack_units(
    units: Sequence[TextUnit],
    config: ChunkingConfig,
    token_counter: TokenCounter,
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    current: list[TextUnit] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = token_counter.count(unit.text)

        if current and current_tokens + unit_tokens > config.max_tokens:
            draft = _draft_from_units(current)
            drafts.append(draft)
            current = _overlap_units(draft, config.overlap_tokens)
            current_tokens = sum(token_counter.count(item.text) for item in current)

        if current and current_tokens + unit_tokens > config.max_tokens:
            current = _trim_units_to_last_tokens(
                current,
                token_limit=max(0, config.max_tokens - unit_tokens),
            )
            current_tokens = sum(token_counter.count(item.text) for item in current)

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        drafts.append(_draft_from_units(current))

    return drafts


def _draft_from_units(units: Sequence[TextUnit]) -> ChunkDraft:
    text = _normalize_chunk_text("\n\n".join(unit.text for unit in units))
    page_numbers = [unit.page_number for unit in units if unit.page_number is not None]
    metadata = _shared_metadata(units)

    return ChunkDraft(
        text=text,
        start_page=min(page_numbers) if page_numbers else None,
        end_page=max(page_numbers) if page_numbers else None,
        metadata=metadata,
    )


def _overlap_units(draft: ChunkDraft, overlap_tokens: int) -> list[TextUnit]:
    if overlap_tokens == 0:
        return []

    words = draft.text.split()
    if not words:
        return []

    return [
        TextUnit(
            text=" ".join(words[-overlap_tokens:]),
            page_number=draft.end_page,
            metadata=draft.metadata,
        )
    ]


def _trim_units_to_last_tokens(
    units: Sequence[TextUnit],
    token_limit: int,
) -> list[TextUnit]:
    if token_limit <= 0:
        return []

    words_with_pages = _words_with_pages(units)
    trimmed = words_with_pages[-token_limit:]
    page_numbers = [page for _, page in trimmed if page is not None]

    return [
        TextUnit(
            text=" ".join(word for word, _ in trimmed),
            page_number=max(page_numbers) if page_numbers else None,
            metadata=_shared_metadata(units),
        )
    ]


def _words_with_pages(units: Sequence[TextUnit]) -> list[tuple[str, int | None]]:
    words_with_pages: list[tuple[str, int | None]] = []
    for unit in units:
        words_with_pages.extend((word, unit.page_number) for word in unit.text.split())
    return words_with_pages


def _normalize_chunk_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]


def _looks_like_section_heading(paragraph: str, token_counter: TokenCounter) -> bool:
    heading = paragraph.strip()
    normalized = _normalize_section_heading(heading)

    if normalized.lower() in _KNOWN_SECTION_HEADINGS:
        return True

    if token_counter.count(heading) > 12:
        return False

    if heading.endswith((".", ",", ";", ":")):
        return False

    if _NUMBERED_HEADING_PATTERN.match(heading):
        return True

    words = heading.split()
    title_case_words = sum(1 for word in words if word[:1].isupper())
    return bool(words) and title_case_words / len(words) >= 0.7


def _normalize_section_heading(heading: str) -> str:
    heading = re.sub(r"^\d+(\.\d+)*\.?\s+", "", heading.strip())
    return re.sub(r"\s+", " ", heading).strip()


def _metadata_value(
    unit: TextUnit,
    key: str,
) -> str | int | float | bool | None:
    if not unit.metadata:
        return None
    return unit.metadata.get(key)


def _shared_metadata(
    units: Sequence[TextUnit],
) -> dict[str, str | int | float | bool | None]:
    shared: dict[str, str | int | float | bool | None] = {}
    metadata_items = [unit.metadata or {} for unit in units]
    keys = {key for metadata in metadata_items for key in metadata}

    for key in keys:
        values = {metadata.get(key) for metadata in metadata_items}
        if len(values) == 1:
            shared[key] = values.pop()

    return shared


def _chunk_identity_metadata(
    chunk_index: int,
    metadata: dict[str, str | int | float | bool | None] | None,
) -> dict[str, str]:
    if not metadata:
        return {}

    source_filename = metadata.get("source_filename")
    if not isinstance(source_filename, str) or not source_filename:
        return {}

    return {"chunk_id": f"{source_filename}:chunk:{chunk_index}"}
