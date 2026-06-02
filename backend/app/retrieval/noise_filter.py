import re
from typing import Any


_REFERENCE_SECTION_NAMES = {
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
}
_BRACKETED_CITATION_PATTERN = re.compile(r"\[\d+\]")
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
_REFERENCE_TERMS = (
    "arxiv",
    "preprint",
    "proceedings",
    "conference",
    "journal",
    "transactions",
    "association for computational linguistics",
    "international conference",
)


def is_retrieval_noise(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False

    section = str(payload.get("section") or "").strip().lower()
    if section in _REFERENCE_SECTION_NAMES:
        return True

    text = str(payload.get("text") or "")
    normalized = " ".join(text.lower().split())
    if not normalized:
        return False

    if normalized.startswith(("references ", "bibliography ")):
        return True

    citation_count = len(_BRACKETED_CITATION_PATTERN.findall(text))
    year_count = len(_YEAR_PATTERN.findall(text))
    reference_term_count = sum(1 for term in _REFERENCE_TERMS if term in normalized)

    # Bibliography chunks are dense with numbered citations, years, venues, and names.
    if citation_count >= 4 and (year_count >= 3 or reference_term_count >= 2):
        return True

    if citation_count >= 8:
        return True

    return False
