from __future__ import annotations

"""Bucket the *uncovered* rows of a corpus run to drive the optimization loop.

Reads the per-row JSON emitted by ``run_simplification_corpus.py --json-output``,
joins it back with the corpus to recover the infix ``expr_text`` / ``expected_text``,
and groups the rows that are not yet "covered" (equivalent-and-no-larger-than-the
reference). For each source it reports the covered rate and a shape breakdown:

  - ``WRONG``       — output not equivalent to the input (a real soundness bug),
  - ``polynomial``  — input is a non-linear MBA (contains a genuine product),
  - ``linear_big``  — linear MBA whose output is larger than the reference form.

With ``--examples N`` it re-simplifies a few uncovered rows per source (with the
given pipeline mode, empty oracle, no CEGIS) and prints input / expected / our
output with node counts so the responsible code path is obvious.
"""

import argparse
import collections
import gzip
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msynth.parsing import parse_corpus_expression  # noqa: E402
from msynth.parsing.datasets import detect_corpus_encoding, corpus_expr_field  # noqa: E402
from msynth.simplification.gamba import classify_linear_nonlinear  # noqa: E402
from msynth.simplification.pipeline import PipelineMode  # noqa: E402
from msynth.simplification.simplifier import Simplifier  # noqa: E402
from msynth.utils.expr_utils import get_subexpressions  # noqa: E402


def node_count(expr) -> int:
    try:
        return len(expr.graph().nodes())
    except Exception:
        return len(get_subexpressions(expr))


def load_corpus_index(path: Path, suite: str | None) -> dict:
    index = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            if suite and row.get("suite") != suite:
                continue
            index[str(row["id"])] = row
    return index


def shape_of(row: dict, equivalent) -> str:
    if equivalent is False:
        return "WRONG"
    try:
        encoding = detect_corpus_encoding(row)
        expr = parse_corpus_expression(
            corpus_expr_field(row), encoding=encoding, size=int(row.get("size", 64))
        )
        return (
            "polynomial"
            if classify_linear_nonlinear(expr) == "nonlinear"
            else "linear_big"
        )
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, required=True, help="per-row JSON from a run")
    ap.add_argument(
        "--corpus",
        type=Path,
        default=REPO_ROOT / "datasets" / "corpora" / "cobra.jsonl.gz",
    )
    ap.add_argument(
        "--suite", type=str, default=None, help="restrict the corpus join to one suite"
    )
    ap.add_argument("--mode", choices=[m.name for m in PipelineMode], default="SIMBA")
    ap.add_argument(
        "--examples",
        type=int,
        default=0,
        help="re-simplify N uncovered rows per source",
    )
    ap.add_argument(
        "--shape", type=str, default=None, help="only show examples of this shape"
    )
    ap.add_argument(
        "--sources",
        type=str,
        default=None,
        help="comma-separated source filter for examples",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results = [json.loads(l) for l in args.json.open() if l.strip()]
    corpus = load_corpus_index(args.corpus, args.suite)

    per_source = collections.defaultdict(lambda: collections.Counter())
    uncovered_rows = collections.defaultdict(list)
    for r in results:
        src = r["source"]
        c = per_source[src]
        c["tot"] += 1
        if r.get("covered"):
            c["covered"] += 1
            continue
        row = corpus.get(r["id"])
        shape = shape_of(row, r.get("equivalent")) if row else "unknown"
        c[shape] += 1
        uncovered_rows[src].append((r, row, shape))

    print(f"=== covered breakdown ({args.json.name}) ===")
    grand_tot = grand_cov = 0
    for src in sorted(per_source):
        c = per_source[src]
        grand_tot += c["tot"]
        grand_cov += c["covered"]
        extra = " ".join(
            f"{k}={c[k]}"
            for k in ("WRONG", "polynomial", "linear_big", "unknown")
            if c[k]
        )
        print(
            f"  {src:28s} tot={c['tot']:6d} covered={c['covered']:6d} "
            f"({100 * c['covered'] / c['tot']:5.1f}%)  {extra}"
        )
    print(
        f"  TOTAL covered {grand_cov}/{grand_tot} = {100 * grand_cov / max(1, grand_tot):.2f}%"
    )

    if args.examples > 0:
        rng = random.Random(args.seed)
        src_filter = (
            {s.strip() for s in args.sources.split(",")} if args.sources else None
        )
        sim = Simplifier(
            oracle_path=None, pipeline_mode=PipelineMode[args.mode], enable_cegis=False
        )
        print(f"\n=== examples (mode={args.mode}) ===")
        for src in sorted(uncovered_rows):
            if src_filter and src not in src_filter:
                continue
            pool = [
                t
                for t in uncovered_rows[src]
                if args.shape is None or t[2] == args.shape
            ]
            if not pool:
                continue
            print(
                f"\n--- {src} ({len(pool)} uncovered{'' if args.shape is None else ' ' + args.shape}) ---"
            )
            for r, row, shape in rng.sample(pool, min(args.examples, len(pool))):
                enc = detect_corpus_encoding(row)
                size = int(row.get("size", 64))
                e = parse_corpus_expression(
                    corpus_expr_field(row), encoding=enc, size=size
                )
                ex = parse_corpus_expression(
                    corpus_expr_field(row, expected=True), encoding=enc, size=size
                )
                out = sim.simplify(e)
                print(
                    f"  [{shape}] in(n={node_count(e)}) exp(n={node_count(ex)}) our(n={node_count(out)})"
                    f" eq={r.get('equivalent')}"
                )
                print(f"     in : {row.get('expr_text', corpus_expr_field(row))[:96]}")
                print(f"     exp: {row.get('expected_text', '')[:96]}")
                print(f"     our: {str(out)[:96]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
