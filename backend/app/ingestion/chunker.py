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
    method: str = "semantic"
    max_tokens: int = 500
    overlap_tokens: int = 75


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


@dataclass(frozen=True)
class ChunkDraft:
    text: str
    start_page: int | None = None
    end_page: int | None = None


class ChunkingStrategy(Protocol):
    name: str

    def split(
        self,
        units: Sequence[TextUnit],
        config: ChunkingConfig,
        token_counter: TokenCounter,
    ) -> list[ChunkDraft]:
        """Split text units into chunk drafts."""


class TokenWindowChunker:
    name = "token_window"

    def split(
        self,
        units: Sequence[TextUnit],
        config: ChunkingConfig,
        token_counter: TokenCounter,
    ) -> list[ChunkDraft]:
        del token_counter
        words_with_pages = _words_with_pages(units)
        if not words_with_pages:
            return []

        drafts: list[ChunkDraft] = []
        start = 0
        step = config.max_tokens - config.overlap_tokens

        while start < len(words_with_pages):
            window = words_with_pages[start : start + config.max_tokens]
            text = " ".join(word for word, _ in window)
            page_numbers = [page for _, page in window if page is not None]
            drafts.append(
                ChunkDraft(
                    text=text,
                    start_page=min(page_numbers) if page_numbers else None,
                    end_page=max(page_numbers) if page_numbers else None,
                )
            )
            start += step

        return drafts


class SemanticChunker:
    name = "semantic"

    def split(
        self,
        units: Sequence[TextUnit],
        config: ChunkingConfig,
        token_counter: TokenCounter,
    ) -> list[ChunkDraft]:
        semantic_units = _split_semantic_units(units, config, token_counter)
        drafts: list[ChunkDraft] = []
        current: list[TextUnit] = []
        current_tokens = 0

        for unit in semantic_units:
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


class ChunkingEngine:
    def __init__(
        self,
        strategies: Sequence[ChunkingStrategy] | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._strategies: dict[str, ChunkingStrategy] = {}
        for strategy in (SemanticChunker(), TokenWindowChunker()):
            self.register_strategy(strategy)
        for strategy in strategies or ():
            self.register_strategy(strategy)
        self._token_counter = token_counter or SimpleTokenCounter()

    @property
    def available_methods(self) -> tuple[str, ...]:
        return tuple(sorted(self._strategies))

    def register_strategy(self, strategy: ChunkingStrategy) -> None:
        self._strategies[strategy.name] = strategy

    def chunk_text(self, text: str, config: ChunkingConfig | None = None) -> list[TextChunk]:
        return self.chunk_units([TextUnit(text=text)], config=config)

    def chunk_pages(
        self,
        pages: Sequence[PageLike],
        config: ChunkingConfig | None = None,
    ) -> list[TextChunk]:
        units = [
            TextUnit(text=page.text, page_number=page.page_number)
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

        strategy = self._strategies.get(active_config.method)
        if strategy is None:
            available = ", ".join(sorted(self._strategies))
            raise ValueError(
                f"Unknown chunking method '{active_config.method}'. "
                f"Available methods: {available}."
            )

        normalized_units = [
            TextUnit(text=_normalize_chunk_text(unit.text), page_number=unit.page_number)
            for unit in units
            if unit.text.strip()
        ]
        drafts = strategy.split(normalized_units, active_config, self._token_counter)

        return [
            TextChunk(
                chunk_index=index,
                text=draft.text,
                token_count=self._token_counter.count(draft.text),
                character_count=len(draft.text),
                start_page=draft.start_page,
                end_page=draft.end_page,
                metadata={
                    "method": active_config.method,
                    "max_tokens": active_config.max_tokens,
                    "overlap_tokens": active_config.overlap_tokens,
                },
            )
            for index, draft in enumerate(drafts)
            if draft.text.strip()
        ]


_TOKEN_PATTERN = re.compile(r"\S+")
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, config: ChunkingConfig | None = None) -> list[TextChunk]:
    return ChunkingEngine().chunk_text(text, config=config)


def chunk_pages(
    pages: Sequence[PageLike],
    config: ChunkingConfig | None = None,
) -> list[TextChunk]:
    return ChunkingEngine().chunk_pages(pages, config=config)


def _validate_config(config: ChunkingConfig) -> None:
    if config.max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    if config.overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")

    if config.overlap_tokens >= config.max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens.")


def _split_semantic_units(
    units: Sequence[TextUnit],
    config: ChunkingConfig,
    token_counter: TokenCounter,
) -> list[TextUnit]:
    semantic_units: list[TextUnit] = []

    for unit in units:
        paragraphs = [part.strip() for part in re.split(r"\n{2,}", unit.text) if part.strip()]
        for paragraph in paragraphs:
            if token_counter.count(paragraph) <= config.max_tokens:
                semantic_units.append(TextUnit(text=paragraph, page_number=unit.page_number))
                continue

            semantic_units.extend(
                _split_large_paragraph(paragraph, unit.page_number, config, token_counter)
            )

    return semantic_units


def _split_large_paragraph(
    paragraph: str,
    page_number: int | None,
    config: ChunkingConfig,
    token_counter: TokenCounter,
) -> list[TextUnit]:
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY_PATTERN.split(paragraph)
        if sentence.strip()
    ]
    units: list[TextUnit] = []

    for sentence in sentences:
        if token_counter.count(sentence) <= config.max_tokens:
            units.append(TextUnit(text=sentence, page_number=page_number))
            continue

        units.extend(
            TextUnit(text=draft.text, page_number=page_number)
            for draft in TokenWindowChunker().split(
                [TextUnit(text=sentence, page_number=page_number)],
                ChunkingConfig(
                    method=config.method,
                    max_tokens=config.max_tokens,
                    overlap_tokens=0,
                ),
                token_counter,
            )
        )

    return units


def _draft_from_units(units: Sequence[TextUnit]) -> ChunkDraft:
    text = _normalize_chunk_text("\n\n".join(unit.text for unit in units))
    page_numbers = [unit.page_number for unit in units if unit.page_number is not None]

    return ChunkDraft(
        text=text,
        start_page=min(page_numbers) if page_numbers else None,
        end_page=max(page_numbers) if page_numbers else None,
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
