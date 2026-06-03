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
from msynth.simplification.pipeline import PipelineMode  # noqa: E402
from msynth.simplification.simplifier import Simplifier  # noqa: E402
from msynth.utils.expr_utils import get_subexpressions  # noqa: E402

DEFAULT_CORPUS = REPO_ROOT / "datasets" / "corpora" / "cobra.jsonl.gz"
DEFAULT_ORACLE = REPO_ROOT / "oracle.pickle"
DEFAULT_JOBS = os.cpu_count() or 1

_SIMPLIFIER: Simplifier | None = None


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
    simplified_text: str | None
    simplified_repr: str | None
    original_nodes: int | None
    simplified_nodes: int | None
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
    path: Path, *, limit: int, suites: set[str] | None = None
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
            # Suite filter: when ``suites`` is non-empty, keep only rows whose
            # ``suite`` field is in the allow-set. Used to isolate the GAMBA
            # vs SimBA halves of the corpus from the long-tail suites.
            if suites and record.suite not in suites:
                continue
            records.append(record)
            # ``limit=0`` means "no cap" — load the entire corpus. Any positive
            # value caps the result set after that many rows.
            if limit > 0 and len(records) >= limit:
                break
    return records


def node_count(expr: Expr) -> int:
    try:
        return len(expr.graph().nodes())
    except Exception:
        return len(get_subexpressions(expr))


def init_worker(
    oracle_path: str | None,
    solver_timeout: int,
    enforce_equivalence: bool,
    pipeline_mode_name: str,
) -> None:
    global _SIMPLIFIER
    # ``oracle_path=None`` (passed when ``--empty-oracle`` is set) constructs
    # the simplifier with the in-memory empty oracle — every oracle lookup
    # misses, so the corpus measures only the pipeline + subtree-SimBA work
    # (and CEGIS, when enabled). This is the configuration used to compare
    # GAMBA-pipeline contributions in isolation.
    _SIMPLIFIER = Simplifier(
        Path(oracle_path) if oracle_path else None,
        pipeline_mode=PipelineMode[pipeline_mode_name],
        solver_timeout=solver_timeout,
        enforce_equivalence=enforce_equivalence,
    )


def check_record(record: CorpusRecord) -> CheckResult:
    if _SIMPLIFIER is None:
        raise RuntimeError("worker simplifier was not initialized")

    start = time.time()
    simplified_text = None
    simplified_repr = None
    original_nodes = None
    simplified_nodes = None
    try:
        expression = parse_infix_expr(record.expr_text, size=record.size)
        expected = parse_infix_expr(record.expected_text, size=record.size)
        original_nodes = node_count(expression)
        try:
            simplified = _SIMPLIFIER.simplify(expression)
        except Exception as exc:
            status = "error"
            simplified_nodes = original_nodes
            detail = f"simplification failed: {exc}"
        else:
            simplified_text = str(simplified)
            simplified_repr = repr(simplified)
            simplified_nodes = node_count(simplified)
            if expr_simp(simplified) == expr_simp(expected):
                status = "ground_truth"
                detail = f"ground truth reached: nodes {original_nodes} -> {simplified_nodes}"
            elif simplified_nodes < original_nodes:
                status = "shorter"
                detail = (
                    f"shorter than input: nodes {original_nodes} -> {simplified_nodes}"
                )
            else:
                status = "not_shorter"
                detail = f"nodes {original_nodes} -> {simplified_nodes}"
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
        simplified_text=simplified_text,
        simplified_repr=simplified_repr,
        original_nodes=original_nodes,
        simplified_nodes=simplified_nodes,
        elapsed_seconds=round(time.time() - start, 6),
    )


def run_checks(
    records: list[CorpusRecord],
    *,
    oracle_path: Path | None,
    jobs: int,
    solver_timeout: int,
    enforce_equivalence: bool,
    fail_fast: bool,
    pipeline_mode_name: str,
) -> list[CheckResult]:
    oracle_arg: str | None = str(oracle_path) if oracle_path is not None else None

    if jobs == 1:
        init_worker(
            oracle_arg,
            solver_timeout,
            enforce_equivalence,
            pipeline_mode_name,
        )
        results = []
        for record in records:
            result = check_record(record)
            results.append(result)
            if fail_fast and not result.passed:
                break
        return results

    results: list[CheckResult] = []
    with ProcessPoolExecutor(
        max_workers=jobs,
        initializer=init_worker,
        initargs=(
            oracle_arg,
            solver_timeout,
            enforce_equivalence,
            pipeline_mode_name,
        ),
    ) as executor:
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
    simplified = truncate(result.simplified_text)
    simplified_ir = truncate(result.simplified_repr)
    return "\n".join(
        [
            f"{result.id} ({result.source}) [{result.status}] {result.detail}",
            f"  expr:     {result.expr_text}",
            f"  expected: {result.expected_text}",
            f"  nodes:    {result.original_nodes} -> {result.simplified_nodes}",
            f"  simplified: {simplified}",
            f"  simplified_ir: {simplified_ir}",
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
            "Check the first N rows of a compact MBA corpus with msynth "
            "oracle-backed simplification."
        )
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help=f"Input corpus JSONL or JSONL.GZ. Defaults to {DEFAULT_CORPUS}.",
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        default=DEFAULT_ORACLE,
        help=(
            f"Simplification oracle path. Defaults to {DEFAULT_ORACLE}. "
            "Ignored when --empty-oracle is set."
        ),
    )
    parser.add_argument(
        "--empty-oracle",
        action="store_true",
        help=(
            "Bypass the pickle and run with an empty in-memory oracle so the "
            "corpus measures only the configured pipeline (and optional CEGIS). "
            "Used to isolate the GAMBA pipeline's contribution from oracle hits."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=[m.name for m in PipelineMode],
        default=PipelineMode.SIMBA.name,
        help=(
            "PipelineMode for the Simplifier. Defaults to SIMBA for back-compat; "
            "use GAMBA for the gamba_pipeline() configuration."
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help=(
            "If set, emit one JSON record per row (id, suite, status, "
            "elapsed_seconds, original_nodes, simplified_nodes, …) to this "
            "path. Enables compare_corpus_runs.py to diff two runs."
        ),
    )
    parser.add_argument(
        "--limit",
        type=non_negative_int,
        default=100,
        help=(
            "Maximum number of corpus rows to check from the start. "
            "0 means no cap (full corpus). Defaults to 100."
        ),
    )
    parser.add_argument(
        "--suites",
        type=str,
        default=None,
        help=(
            "Comma-separated list of corpus suites to include (e.g. "
            "'simba,gamba'). When set, rows whose ``suite`` field is not "
            "in the list are skipped. Defaults to no filtering."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=positive_int,
        default=DEFAULT_JOBS,
        help=f"Parallel worker count. Defaults to all available cores ({DEFAULT_JOBS}).",
    )
    parser.add_argument(
        "--solver-timeout",
        type=positive_int,
        default=1,
        help="Simplifier Z3 timeout in seconds. Defaults to 1.",
    )
    parser.add_argument(
        "--enforce-equivalence",
        action="store_true",
        help="Require the simplifier's internal Z3 check to prove equivalence.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first mismatch, timeout, or error.",
    )
    parser.add_argument(
        "--max-failures",
        type=non_negative_int,
        default=10,
        help="Maximum number of failures printed to stderr. Defaults to 10.",
    )
    return parser.parse_args()


def write_json_output(path: Path, results: list[CheckResult]) -> None:
    """Emit one JSON record per row for offline analysis / diffing."""
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "id": result.id,
                        "source": result.source,
                        "status": result.status,
                        "elapsed_seconds": result.elapsed_seconds,
                        "original_nodes": result.original_nodes,
                        "simplified_nodes": result.simplified_nodes,
                    }
                )
                + "\n"
            )


def main() -> int:
    args = parse_args()
    if not args.corpus.is_file():
        raise SystemExit(f"corpus does not exist: {args.corpus}")
    # When --empty-oracle is set, --oracle is ignored; otherwise the pickle
    # path must exist (default points at the repo-bundled oracle.pickle).
    if not args.empty_oracle and not args.oracle.is_file():
        raise SystemExit(f"oracle does not exist: {args.oracle}")

    suites_filter: set[str] | None = None
    if args.suites:
        suites_filter = {s.strip() for s in args.suites.split(",") if s.strip()}
    records = load_corpus(args.corpus, limit=args.limit, suites=suites_filter)
    start = time.time()
    results = run_checks(
        records,
        oracle_path=None if args.empty_oracle else args.oracle,
        jobs=args.jobs,
        solver_timeout=args.solver_timeout,
        enforce_equivalence=args.enforce_equivalence,
        fail_fast=args.fail_fast,
        pipeline_mode_name=args.mode,
    )
    elapsed = time.time() - start

    summary = summarize(results)
    print(
        "checked={checked} passed={passed} failed={failed} "
        "ground_truth={ground_truth} shorter={shorter} "
        "not_shorter={not_shorter} error={error} jobs={jobs} "
        "mode={mode} oracle={oracle_kind} "
        "seconds={seconds:.3f}".format(
            **summary,
            jobs=args.jobs,
            mode=args.mode,
            oracle_kind="empty" if args.empty_oracle else "pickle",
            seconds=elapsed,
        )
    )

    if args.json_output is not None:
        write_json_output(args.json_output, results)

    failures = [result for result in results if not result.passed]
    for result in failures[: args.max_failures]:
        print(format_failure(result), file=sys.stderr)

    return 0 if not failures and len(results) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
