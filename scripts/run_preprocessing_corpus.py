from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from miasm.expression.expression import Expr  # noqa: E402
from miasm.expression.simplifications import expr_simp  # noqa: E402

from msynth.parsing import parse_infix_expr  # noqa: E402
from msynth.simplification.preprocessing import (  # noqa: E402
    AstNormalizationPass,
    Preprocessor,
)
from msynth.simplification.simba import SimbaPass  # noqa: E402
from msynth.utils.expr_utils import get_subexpressions  # noqa: E402

DEFAULT_CORPUS = REPO_ROOT / "datasets" / "corpora" / "cobra.jsonl.gz"
DEFAULT_JOBS = os.cpu_count() or 1

_PREPROCESSOR: Preprocessor | None = None


@dataclass(frozen=True)
class CorpusRecord:
    id: str
    source: str
    suite: str
    expr_text: str
    expected_text: str
    size: int

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CorpusRecord":
        return cls(
            id=str(data["id"]),
            source=str(data["source"]),
            suite=str(data["suite"]),
            expr_text=str(data["expr_text"]),
            expected_text=str(data["expected_text"]),
            size=int(data["size"]),
        )


@dataclass(frozen=True)
class CheckResult:
    id: str
    source: str
    status: str
    detail: str
    expr_text: str
    expected_text: str
    preprocessed_text: str | None
    preprocessed_repr: str | None
    original_nodes: int | None
    preprocessed_nodes: int | None
    elapsed_seconds: float

    @property
    def passed(self) -> bool:
        return self.status in {"ground_truth", "shorter"}


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def load_corpus(
    path: Path,
    *,
    limit: int,
    suites: set[str],
    source_prefixes: tuple[str, ...],
    sources: set[str],
) -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    seen_ids: set[str] = set()
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                record = CorpusRecord.from_json(json.loads(stripped))
            except Exception as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid corpus row: {exc}"
                ) from exc
            if record.id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {record.id!r}")
            seen_ids.add(record.id)
            if suites and record.suite not in suites:
                continue
            if sources and record.source not in sources:
                continue
            if source_prefixes and not record.source.startswith(source_prefixes):
                continue
            records.append(record)
            if len(records) >= limit:
                break
    return records


def node_count(expr: Expr) -> int:
    try:
        return len(expr.graph().nodes())
    except Exception:
        return len(get_subexpressions(expr))


def init_worker() -> None:
    global _PREPROCESSOR
    _PREPROCESSOR = Preprocessor([AstNormalizationPass(), SimbaPass()])


def check_record(record: CorpusRecord) -> CheckResult:
    if _PREPROCESSOR is None:
        raise RuntimeError("worker preprocessor was not initialized")

    start = time.time()
    preprocessed_text = None
    preprocessed_repr = None
    original_nodes = None
    preprocessed_nodes = None
    try:
        expression = parse_infix_expr(record.expr_text, size=record.size)
        expected = parse_infix_expr(record.expected_text, size=record.size)
        original_nodes = node_count(expression)
        try:
            preprocessed = _PREPROCESSOR.run(expression)
        except Exception as exc:
            status = "error"
            preprocessed_nodes = original_nodes
            detail = f"preprocessing failed: {exc}"
        else:
            preprocessed_text = str(preprocessed)
            preprocessed_repr = repr(preprocessed)
            preprocessed_nodes = node_count(preprocessed)
            if expr_simp(preprocessed) == expr_simp(expected):
                status = "ground_truth"
                detail = (
                    f"ground truth reached: nodes "
                    f"{original_nodes} -> {preprocessed_nodes}"
                )
            elif preprocessed_nodes < original_nodes:
                status = "shorter"
                detail = (
                    f"shorter than input: nodes "
                    f"{original_nodes} -> {preprocessed_nodes}"
                )
            else:
                status = "not_shorter"
                detail = f"nodes {original_nodes} -> {preprocessed_nodes}"
    except Exception as exc:
        status = "error"
        detail = str(exc)

    return CheckResult(
        id=record.id,
        source=record.source,
        status=status,
        detail=detail,
        expr_text=record.expr_text,
        expected_text=record.expected_text,
        preprocessed_text=preprocessed_text,
        preprocessed_repr=preprocessed_repr,
        original_nodes=original_nodes,
        preprocessed_nodes=preprocessed_nodes,
        elapsed_seconds=round(time.time() - start, 6),
    )


def run_checks(
    records: list[CorpusRecord],
    *,
    jobs: int,
    fail_fast: bool,
) -> list[CheckResult]:
    if jobs == 1:
        init_worker()
        results = []
        for record in records:
            result = check_record(record)
            results.append(result)
            if fail_fast and not result.passed:
                break
        return results

    results: list[CheckResult] = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=init_worker) as executor:
        for result in executor.map(check_record, records):
            results.append(result)
            if fail_fast and not result.passed:
                break
    return results


def truncate(value: str | None, *, max_length: int = 500) -> str | None:
    if value is not None and len(value) > max_length:
        return f"{value[: max_length - 3]}..."
    return value


def format_failure(result: CheckResult) -> str:
    preprocessed = truncate(result.preprocessed_text)
    preprocessed_ir = truncate(result.preprocessed_repr)
    return "\n".join(
        [
            f"{result.id} ({result.source}) [{result.status}] {result.detail}",
            f"  expr:     {result.expr_text}",
            f"  expected: {result.expected_text}",
            f"  nodes:    {result.original_nodes} -> {result.preprocessed_nodes}",
            f"  preprocessed: {preprocessed}",
            f"  preprocessed_ir: {preprocessed_ir}",
        ]
    )


def summarize(results: list[CheckResult]) -> dict[str, int]:
    summary = {
        "checked": len(results),
        "passed": 0,
        "failed": 0,
        "ground_truth": 0,
        "shorter": 0,
        "not_shorter": 0,
        "error": 0,
    }
    for result in results:
        if result.passed:
            summary["passed"] += 1
        else:
            summary["failed"] += 1
        summary[result.status] = summary.get(result.status, 0) + 1
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the first N rows of a compact MBA corpus with the "
            "AST + SiMBA preprocessing pipeline."
        )
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help=f"Input corpus JSONL or JSONL.GZ. Defaults to {DEFAULT_CORPUS}.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=100,
        help=(
            "Maximum number of matching corpus rows to check from the start. "
            "Defaults to 100."
        ),
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help=(
            "Only check records from this suite. Can be passed multiple times. "
            "By default, all suites are included."
        ),
    )
    parser.add_argument(
        "--source-prefix",
        action="append",
        default=[],
        help=(
            "Only check records whose source starts with this prefix. Can be "
            "passed multiple times. By default, all sources are included."
        ),
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help=(
            "Only check records from this exact source. Can be passed multiple "
            "times. By default, all sources are included."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=positive_int,
        default=DEFAULT_JOBS,
        help=f"Parallel worker count. Defaults to all available cores ({DEFAULT_JOBS}).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first mismatch or error.",
    )
    parser.add_argument(
        "--max-failures",
        type=non_negative_int,
        default=10,
        help="Maximum number of failures printed to stderr. Defaults to 10.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.corpus.is_file():
        raise SystemExit(f"corpus does not exist: {args.corpus}")

    records = load_corpus(
        args.corpus,
        limit=args.limit,
        suites=set(args.suite),
        source_prefixes=tuple(args.source_prefix),
        sources=set(args.source),
    )
    start = time.time()
    results = run_checks(records, jobs=args.jobs, fail_fast=args.fail_fast)
    elapsed = time.time() - start

    summary = summarize(results)
    print(
        "checked={checked} passed={passed} failed={failed} "
        "ground_truth={ground_truth} shorter={shorter} "
        "not_shorter={not_shorter} "
        "error={error} jobs={jobs} "
        "passes=ast,simba seconds={seconds:.3f}".format(
            **summary,
            jobs=args.jobs,
            seconds=elapsed,
        )
    )

    failures = [result for result in results if not result.passed]
    for result in failures[: args.max_failures]:
        print(format_failure(result), file=sys.stderr)

    return 0 if not failures and len(results) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
