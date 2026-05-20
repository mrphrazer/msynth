from __future__ import annotations

import argparse
import os
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import z3  # noqa: E402
from miasm.expression.expression import Expr  # noqa: E402
from miasm.ir.translators.z3_ir import TranslatorZ3  # noqa: E402

from msynth import Synthesizer  # noqa: E402
from msynth.utils.expr_utils import get_subexpressions, parse_expr  # noqa: E402

DEFAULT_CASE_FILE = REPO_ROOT / "data" / "synthesis_corpus" / "cases.jsonl"
DEFAULT_REPORT_DIR = REPO_ROOT / "data" / "synthesis_corpus" / "reports"


@dataclass(frozen=True)
class CorpusCase:
    id: str
    expr_text: str
    tags: tuple[str, ...]
    features: tuple[str, ...]
    samples: int
    timeout: float
    seeds: tuple[int, ...]

    @classmethod
    def from_json(
        cls,
        data: dict[str, Any],
        *,
        default_samples: int,
        default_timeout: float,
        default_seeds: tuple[int, ...],
    ) -> "CorpusCase":
        return cls(
            id=str(data["id"]),
            expr_text=str(data["expr"]),
            tags=tuple(str(tag) for tag in data.get("tags", ())),
            features=tuple(str(feature) for feature in data.get("features", ())),
            samples=int(data.get("samples", default_samples)),
            timeout=float(data.get("timeout", default_timeout)),
            seeds=tuple(int(seed) for seed in data.get("seeds", default_seeds)),
        )


def load_cases(
    path: Path,
    *,
    default_samples: int,
    default_timeout: float,
    default_seeds: tuple[int, ...],
) -> list[CorpusCase]:
    cases: list[CorpusCase] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            data = json.loads(stripped)
            case = CorpusCase.from_json(
                data,
                default_samples=default_samples,
                default_timeout=default_timeout,
                default_seeds=default_seeds,
            )
            if case.id in seen_ids:
                raise ValueError(f"Duplicate case id {case.id!r} at line {line_number}")
            seen_ids.add(case.id)
            cases.append(case)
    return cases


def filter_cases(
    cases: list[CorpusCase],
    *,
    case_ids: set[str],
    tags: set[str],
    features: set[str],
) -> list[CorpusCase]:
    filtered = []
    for case in cases:
        if case_ids and case.id not in case_ids:
            continue
        if tags and not tags.issubset(case.tags):
            continue
        if features and not features.issubset(case.features):
            continue
        filtered.append(case)
    return filtered


def node_count(expr: Expr) -> int:
    try:
        return len(expr.graph().nodes())
    except Exception:
        return len(get_subexpressions(expr))


def json_score(score: float) -> float | str:
    if math.isfinite(score):
        return score
    if score > 0:
        return "inf"
    return "-inf"


def verify_equivalence(
    original: Expr, synthesized: Expr, timeout: float
) -> dict[str, Any]:
    if original.size != synthesized.size:
        return {
            "status": "different",
            "detail": f"size mismatch: {original.size} != {synthesized.size}",
        }

    solver = z3.Solver()
    solver.set("timeout", int(timeout * 1000))
    translator = TranslatorZ3()
    try:
        solver.add(translator.from_expr(original) != translator.from_expr(synthesized))
        result = solver.check()
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}

    if result == z3.unsat:
        return {"status": "equivalent", "detail": "unsat"}
    if result == z3.sat:
        return {"status": "different", "detail": "sat"}
    return {"status": "unknown", "detail": solver.reason_unknown()}


def run_attempt(case: CorpusCase, expr: Expr, seed: int) -> dict[str, Any]:
    random.seed(seed)
    start = time.time()
    try:
        synthesized, score = Synthesizer().synthesize_from_expression(
            expr, num_samples=case.samples, timeout=case.timeout
        )
        elapsed = time.time() - start
    except Exception as exc:
        return {
            "seed": seed,
            "status": "error",
            "score": "inf",
            "time_seconds": round(time.time() - start, 6),
            "error": str(exc),
        }

    verification = None
    if score == 0.0:
        verification = verify_equivalence(expr, synthesized, timeout=case.timeout)
        if verification["status"] == "equivalent":
            status = "solved"
        elif verification["status"] == "different":
            status = "mismatch"
        else:
            status = "unverified"
    else:
        status = "miss"

    return {
        "seed": seed,
        "status": status,
        "score": json_score(score),
        "time_seconds": round(elapsed, 6),
        "expr": repr(synthesized),
        "nodes": node_count(synthesized),
        "verification": verification,
    }


def run_attempt_job(case: CorpusCase, seed: int) -> tuple[str, dict[str, Any]]:
    try:
        expr = parse_expr(case.expr_text)
    except Exception as exc:
        return (
            case.id,
            {
                "seed": seed,
                "status": "error",
                "score": "inf",
                "time_seconds": 0.0,
                "error": f"parse error: {exc}",
            },
        )
    return case.id, run_attempt(case, expr, seed)


def best_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [attempt for attempt in attempts if isinstance(attempt["score"], float)]
    if not scored:
        return None
    return min(scored, key=lambda attempt: attempt["score"])


def summarize_case(
    case: CorpusCase, expr: Expr, attempts: list[dict[str, Any]]
) -> dict[str, Any]:
    solved_seeds = [
        attempt["seed"] for attempt in attempts if attempt["status"] == "solved"
    ]
    zero_score_seeds = [
        attempt["seed"]
        for attempt in attempts
        if isinstance(attempt["score"], float) and attempt["score"] == 0.0
    ]
    error_seeds = [
        attempt["seed"] for attempt in attempts if attempt["status"] == "error"
    ]

    if solved_seeds:
        status = "solved"
    elif len(error_seeds) == len(attempts):
        status = "error"
    elif any(attempt["status"] == "mismatch" for attempt in attempts):
        status = "mismatch"
    elif any(attempt["status"] == "unverified" for attempt in attempts):
        status = "unverified"
    else:
        status = "unsolved"

    best = best_attempt(attempts)
    return {
        "id": case.id,
        "status": status,
        "expr": case.expr_text,
        "tags": list(case.tags),
        "features": list(case.features),
        "samples": case.samples,
        "timeout": case.timeout,
        "seeds": list(case.seeds),
        "input_nodes": node_count(expr),
        "best_score": best["score"] if best else "inf",
        "best_expr": best.get("expr") if best else None,
        "best_nodes": best.get("nodes") if best else None,
        "solved_seeds": solved_seeds,
        "zero_score_seeds": zero_score_seeds,
        "error_seeds": error_seeds,
        "attempts": attempts,
    }


def run_cases(cases: list[CorpusCase], jobs: int) -> list[dict[str, Any]]:
    case_by_id = {case.id: case for case in cases}
    expr_by_id: dict[str, Expr] = {}
    attempts_by_id: dict[str, list[dict[str, Any]]] = {case.id: [] for case in cases}
    result_by_id: dict[str, dict[str, Any]] = {}

    for case in cases:
        try:
            expr_by_id[case.id] = parse_expr(case.expr_text)
        except Exception as exc:
            result_by_id[case.id] = {
                "id": case.id,
                "status": "error",
                "expr": case.expr_text,
                "tags": list(case.tags),
                "features": list(case.features),
                "samples": case.samples,
                "timeout": case.timeout,
                "seeds": list(case.seeds),
                "input_nodes": None,
                "best_score": "inf",
                "best_expr": None,
                "best_nodes": None,
                "solved_seeds": [],
                "zero_score_seeds": [],
                "error_seeds": list(case.seeds),
                "attempts": [
                    {
                        "seed": seed,
                        "status": "error",
                        "score": "inf",
                        "time_seconds": 0.0,
                        "error": f"parse error: {exc}",
                    }
                    for seed in case.seeds
                ],
            }

    jobs = max(1, jobs)
    attempt_jobs = [
        (case, seed)
        for case in cases
        if case.id not in result_by_id
        for seed in case.seeds
    ]

    if jobs == 1:
        for case, seed in attempt_jobs:
            case_id, attempt = run_attempt_job(case, seed)
            attempts_by_id[case_id].append(attempt)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(run_attempt_job, case, seed)
                for case, seed in attempt_jobs
            ]
            for future in as_completed(futures):
                case_id, attempt = future.result()
                attempts_by_id[case_id].append(attempt)

    for case in cases:
        if case.id in result_by_id:
            continue
        attempts = attempts_by_id[case.id]
        seed_order = {seed: index for index, seed in enumerate(case.seeds)}
        attempts.sort(key=lambda attempt: seed_order[attempt["seed"]])
        result_by_id[case.id] = summarize_case(
            case_by_id[case.id], expr_by_id[case.id], attempts
        )

    return [result_by_id[case.id] for case in cases]


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    total_time = 0.0
    for result in results:
        statuses[result["status"]] = statuses.get(result["status"], 0) + 1
        total_time += sum(attempt["time_seconds"] for attempt in result["attempts"])

    return {
        "total": len(results),
        "statuses": statuses,
        "time_seconds": round(total_time, 6),
    }


def load_result_map(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    return {result["id"]: result for result in report.get("results", [])}


def compare_results(
    reference_path: Path, results: list[dict[str, Any]]
) -> dict[str, list[str]]:
    reference = load_result_map(reference_path)
    current = {result["id"]: result for result in results}
    comparison = {
        "improved": [],
        "regressed": [],
        "changed": [],
        "unchanged": [],
        "new": [],
        "removed": [],
    }

    for case_id, result in current.items():
        old = reference.get(case_id)
        if old is None:
            comparison["new"].append(case_id)
            continue

        old_status = old["status"]
        new_status = result["status"]
        if old_status == new_status:
            comparison["unchanged"].append(case_id)
        elif old_status != "solved" and new_status == "solved":
            comparison["improved"].append(case_id)
        elif old_status == "solved" and new_status != "solved":
            comparison["regressed"].append(case_id)
        else:
            comparison["changed"].append(case_id)

    for case_id in reference:
        if case_id not in current:
            comparison["removed"].append(case_id)

    return comparison


def default_report_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return DEFAULT_REPORT_DIR / f"report-{timestamp}.json"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    statuses = summary["statuses"]
    print(
        f"cases={summary['total']} "
        f"solved={statuses.get('solved', 0)} "
        f"unsolved={statuses.get('unsolved', 0)} "
        f"mismatch={statuses.get('mismatch', 0)} "
        f"unverified={statuses.get('unverified', 0)} "
        f"error={statuses.get('error', 0)} "
        f"attempt_time={summary['time_seconds']}s "
        f"wall={summary['wall_time_seconds']}s"
    )
    print()
    print(f"{'status':<10} {'best':>12} {'seeds':<11} case")
    print(f"{'-' * 10} {'-' * 12} {'-' * 11} {'-' * 32}")
    for result in report["results"]:
        seeds = f"{len(result['solved_seeds'])}/{len(result['seeds'])}"
        print(
            f"{result['status']:<10} "
            f"{str(result['best_score']):>12} "
            f"{seeds:<11} "
            f"{result['id']}"
        )

    comparison = report.get("comparison")
    if comparison:
        print()
        print(
            "comparison "
            f"improved={len(comparison['improved'])} "
            f"regressed={len(comparison['regressed'])} "
            f"changed={len(comparison['changed'])} "
            f"new={len(comparison['new'])} "
            f"removed={len(comparison['removed'])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the standalone stochastic synthesis corpus."
    )
    parser.add_argument(
        "case_file",
        nargs="?",
        type=Path,
        default=DEFAULT_CASE_FILE,
        help="JSONL corpus file. Defaults to data/synthesis_corpus/cases.jsonl.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel worker processes for case/seed attempts. Defaults to CPU count.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only the selected case id. Can be repeated.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Run only cases containing this tag. Can be repeated.",
    )
    parser.add_argument(
        "--feature",
        action="append",
        default=[],
        help="Run only cases containing this feature. Can be repeated.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help="Default sample count for cases that do not specify samples.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Default per-seed timeout for cases that do not specify timeout.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=[],
        help="Default seed list for cases that do not specify seeds. Can be repeated.",
    )
    parser.add_argument(
        "--override-samples",
        type=int,
        help="Override the sample count for every selected case.",
    )
    parser.add_argument(
        "--override-timeout",
        type=float,
        help="Override the per-seed timeout for every selected case.",
    )
    parser.add_argument(
        "--override-seed",
        action="append",
        type=int,
        default=[],
        help="Override the seed list for every selected case. Can be repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Report path. Defaults to data/synthesis_corpus/reports/report-<timestamp>.json.",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Compare current results against an existing report.",
    )
    return parser.parse_args()


def apply_overrides(
    cases: list[CorpusCase],
    *,
    samples: int | None,
    timeout: float | None,
    seeds: tuple[int, ...] | None,
) -> list[CorpusCase]:
    return [
        CorpusCase(
            id=case.id,
            expr_text=case.expr_text,
            tags=case.tags,
            features=case.features,
            samples=samples if samples is not None else case.samples,
            timeout=timeout if timeout is not None else case.timeout,
            seeds=seeds if seeds is not None else case.seeds,
        )
        for case in cases
    ]


def main() -> int:
    args = parse_args()
    default_seeds = tuple(args.seed) if args.seed else (0,)

    cases = load_cases(
        args.case_file,
        default_samples=args.samples,
        default_timeout=args.timeout,
        default_seeds=default_seeds,
    )
    cases = filter_cases(
        cases,
        case_ids=set(args.case_id),
        tags=set(args.tag),
        features=set(args.feature),
    )
    if not cases:
        raise SystemExit("No corpus cases selected.")

    override_seeds = tuple(args.override_seed) if args.override_seed else None
    cases = apply_overrides(
        cases,
        samples=args.override_samples,
        timeout=args.override_timeout,
        seeds=override_seeds,
    )

    start_time = time.time()
    results = run_cases(cases, jobs=args.jobs)
    summary = build_summary(results)
    summary["wall_time_seconds"] = round(time.time() - start_time, 6)
    report: dict[str, Any] = {
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_file": display_path(args.case_file),
        "jobs": max(1, args.jobs),
        "results": results,
        "summary": summary,
    }

    if args.compare:
        report["comparison"] = compare_results(args.compare, results)

    output = args.output or default_report_path()
    write_json(output, report)

    print_summary(report)
    print()
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
