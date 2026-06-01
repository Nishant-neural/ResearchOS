"""
Evaluate baseline RAG against persistent encoder-memory generation.

Expected workflow:
1. Start Qdrant.
2. Start the FastAPI backend.
3. Upload your papers through /api/upload-paper.
4. Wait for hidden-state preprocessing to finish.
5. Fill experiments/evaluation_queries.json from the example file.
6. Run this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://localhost:8000/api"
DEFAULT_QUERY_FILE = Path("experiments/evaluation_queries.json")
DEFAULT_OUT_DIR = Path("experiments/results")
DEFAULT_AGGREGATION_METHOD = "sequence_concatenate"
DEFAULT_CONDITIONING_STRATEGY = "framed_memory"


@dataclass
class EvaluationRow:
    query_id: str
    query: str
    expected_answer: str
    expected_keywords: list[str]
    retrieved_chunk_ids: list[str]
    baseline_answer: str
    hidden_state_answer: str
    baseline_latency_ms: float
    hidden_state_latency_ms: float
    latency_delta_ms: float
    baseline_correct: bool | None
    hidden_state_correct: bool | None
    hidden_states_used: bool
    hidden_chunks_used: int
    conditioning_strategy: str
    baseline_error: str | None = None
    hidden_state_error: str | None = None


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    query_file = Path(args.queries)
    out_dir = Path(args.out_dir)

    check_backend(base_url)
    queries = load_queries(query_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"hidden_state_eval_{timestamp}.json"
    csv_path = out_dir / f"hidden_state_eval_{timestamp}.csv"

    rows: list[EvaluationRow] = []
    for index, item in enumerate(queries, start=1):
        query = item["query"]
        query_id = str(item.get("id") or f"q{index}")
        print(f"\n[{index}/{len(queries)}] {query}")

        retrieved_chunk_ids = search_chunk_ids(base_url, query, args.limit)
        baseline_response, baseline_latency_ms, baseline_error = post_json_timed(
            f"{base_url}/ask",
            {"query": query, "limit": args.limit},
            timeout=args.timeout,
        )
        hidden_response, hidden_latency_ms, hidden_error = post_json_timed(
            f"{base_url}/ask-with-hidden-states",
            {
                "query": query,
                "limit": args.limit,
                "aggregation_method": args.aggregation_method,
                "conditioning_strategy": args.conditioning_strategy,
            },
            timeout=args.timeout,
        )

        expected_answer = str(item.get("expected_answer") or "")
        expected_keywords = [str(word) for word in item.get("expected_keywords", [])]
        baseline_answer = response_answer(baseline_response)
        hidden_state_answer = response_answer(hidden_response)

        baseline_correct = score_answer(
            baseline_answer,
            expected_keywords,
            expected_answer,
        )
        hidden_state_correct = score_answer(
            hidden_state_answer,
            expected_keywords,
            expected_answer,
        )

        if args.interactive:
            baseline_correct = ask_manual_score("Baseline", baseline_answer)
            hidden_state_correct = ask_manual_score("Hidden-state", hidden_state_answer)

        row = EvaluationRow(
            query_id=query_id,
            query=query,
            expected_answer=expected_answer,
            expected_keywords=expected_keywords,
            retrieved_chunk_ids=retrieved_chunk_ids,
            baseline_answer=baseline_answer,
            hidden_state_answer=hidden_state_answer,
            baseline_latency_ms=round(baseline_latency_ms, 2),
            hidden_state_latency_ms=round(hidden_latency_ms, 2),
            latency_delta_ms=round(hidden_latency_ms - baseline_latency_ms, 2),
            baseline_correct=baseline_correct,
            hidden_state_correct=hidden_state_correct,
            hidden_states_used=bool(
                (hidden_response or {}).get("used_hidden_states", False)
            ),
            hidden_chunks_used=int(
                (hidden_response or {}).get("num_chunks_with_hidden_states", 0)
            ),
            conditioning_strategy=str(
                (hidden_response or {}).get(
                    "conditioning_strategy",
                    args.conditioning_strategy,
                )
            ),
            baseline_error=baseline_error,
            hidden_state_error=hidden_error,
        )
        rows.append(row)
        print_row(row)

    write_json(json_path, rows)
    write_csv(csv_path, rows)
    print_summary(rows)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline RAG vs hidden-state memory generation."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--queries", default=str(DEFAULT_QUERY_FILE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--aggregation-method", default=DEFAULT_AGGREGATION_METHOD)
    parser.add_argument("--conditioning-strategy", default=DEFAULT_CONDITIONING_STRATEGY)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Manually mark baseline and hidden-state answers as correct/incorrect.",
    )
    return parser.parse_args()


def check_backend(base_url: str) -> None:
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            f"Backend is not reachable at {base_url}. Start FastAPI first. Error: {exc}"
        ) from exc


def load_queries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"Query file not found: {path}\n"
            "Copy experiments/evaluation_queries.example.json to "
            "experiments/evaluation_queries.json and fill in your expected answers."
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or not data:
        raise SystemExit("Query file must contain a non-empty JSON list.")

    for item in data:
        if not isinstance(item, dict) or not item.get("query"):
            raise SystemExit("Each query item must be an object with a 'query' field.")

    return data


def search_chunk_ids(base_url: str, query: str, limit: int) -> list[str]:
    try:
        response = requests.post(
            f"{base_url}/search",
            json={"query": query, "limit": limit},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return []

    return [
        str(result.get("chunk_id") or result.get("metadata", {}).get("chunk_id"))
        for result in payload.get("results", [])
        if result.get("chunk_id") or result.get("metadata", {}).get("chunk_id")
    ]


def post_json_timed(
    url: str,
    payload: dict[str, Any],
    timeout: int,
) -> tuple[dict[str, Any] | None, float, str | None]:
    start = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        return response.json(), elapsed_ms, None
    except requests.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        response_text = getattr(exc.response, "text", "")
        return None, elapsed_ms, f"{exc}; response={response_text}"
    except requests.RequestException as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return None, elapsed_ms, str(exc)


def response_answer(response: dict[str, Any] | None) -> str:
    if not response:
        return ""
    return str(response.get("answer") or "")


def score_answer(
    answer: str,
    expected_keywords: list[str],
    expected_answer: str,
) -> bool | None:
    normalized_answer = normalize(answer)

    if expected_keywords:
        return all(normalize(keyword) in normalized_answer for keyword in expected_keywords)

    if expected_answer:
        return normalize(expected_answer) in normalized_answer

    return None


def ask_manual_score(label: str, answer: str) -> bool:
    print(f"\n{label} answer:\n{answer}\n")
    while True:
        response = input(f"Mark {label} answer correct? [y/n]: ").strip().lower()
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def print_row(row: EvaluationRow) -> None:
    print(
        "baseline_correct={baseline} hidden_state_correct={hidden} "
        "latency_delta_ms={delta} chunks={chunks}".format(
            baseline=format_score(row.baseline_correct),
            hidden=format_score(row.hidden_state_correct),
            delta=row.latency_delta_ms,
            chunks=",".join(row.retrieved_chunk_ids) or "none",
        )
    )


def print_summary(rows: list[EvaluationRow]) -> None:
    print("\nEvaluation Table")
    print("-" * 100)
    print(
        f"{'query':<42} {'baseline':<10} {'hidden':<10} "
        f"{'delta_ms':>10} {'chunks':<20}"
    )
    print("-" * 100)
    for row in rows:
        query = shorten(row.query, 40)
        chunks = shorten(",".join(row.retrieved_chunk_ids), 18)
        print(
            f"{query:<42} {format_score(row.baseline_correct):<10} "
            f"{format_score(row.hidden_state_correct):<10} "
            f"{row.latency_delta_ms:>10.2f} {chunks:<20}"
        )

    baseline_scores = [row.baseline_correct for row in rows if row.baseline_correct is not None]
    hidden_scores = [
        row.hidden_state_correct
        for row in rows
        if row.hidden_state_correct is not None
    ]
    if baseline_scores:
        print(f"\nBaseline accuracy:     {sum(baseline_scores)}/{len(baseline_scores)}")
    if hidden_scores:
        print(f"Hidden-state accuracy: {sum(hidden_scores)}/{len(hidden_scores)}")


def format_score(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def shorten(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def write_json(path: Path, rows: list[EvaluationRow]) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "rows": [asdict(row) for row in rows],
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def write_csv(path: Path, rows: list[EvaluationRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            data["expected_keywords"] = "|".join(row.expected_keywords)
            data["retrieved_chunk_ids"] = "|".join(row.retrieved_chunk_ids)
            writer.writerow(data)


if __name__ == "__main__":
    main()
