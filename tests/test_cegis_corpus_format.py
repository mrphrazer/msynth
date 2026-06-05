"""Schema, soundness, and CEGIS-coverage checks for the dedicated CEGIS corpus.

``datasets/corpora/cegis.jsonl.gz`` is evaluated by
``scripts/run_simplification_corpus.py --cegis``. These tests pin that it loads in
the runner's IR schema, that every row's ground truth is a true equivalent of the
(larger) obfuscated input, and that the set genuinely exercises the CEGIS constant
solver (some rows are uncovered without it, all covered with it).
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from miasm.expression.expression import Expr  # noqa: E402

from msynth import Simplifier  # noqa: E402
from msynth.utils.expr_utils import (  # noqa: E402
    compile_expr_to_python,
    get_unique_variables,
    parse_expr,
)
from msynth.utils.sampling import (  # noqa: E402
    _rename_variables_for_compilation,
    gen_adversarial_inputs,
)
from scripts.run_simplification_corpus import (  # noqa: E402
    expressions_equivalent,
    node_count,
)

CORPUS = REPO_ROOT / "datasets" / "corpora" / "cegis.jsonl.gz"


def _equivalent(a: Expr, b: Expr, *, trials: int = 256, seed: int = 1) -> bool:
    """Independent compile-based equivalence probe (random + edge inputs)."""
    variables = sorted(
        set(get_unique_variables(a)) | set(get_unique_variables(b)), key=str
    )
    fa = compile_expr_to_python(_rename_variables_for_compilation(a, variables))
    fb = compile_expr_to_python(_rename_variables_for_compilation(b, variables))
    for inputs in gen_adversarial_inputs(variables):
        if fa(inputs) != fb(inputs):
            return False
    rng = random.Random(seed)
    width = len(variables)
    for _ in range(trials):
        inputs = [rng.getrandbits(64) for _ in range(width)]
        if fa(inputs) != fb(inputs):
            return False
    return True


def _rows():
    with gzip.open(CORPUS, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_cegis_corpus_schema_and_sound_ground_truth() -> None:
    required = {"id", "source", "suite", "size", "expr_miasm", "expected_miasm"}
    rows = _rows()
    assert rows, "cegis corpus is empty"
    seen_ids = set()
    for row in rows:
        assert required <= set(row)
        assert row["suite"] == "cegis"
        assert row["id"] not in seen_ids
        seen_ids.add(row["id"])
        inp = parse_expr(row["expr_miasm"])
        exp = parse_expr(row["expected_miasm"])
        # The recorded ground truth must be a genuine equivalent of the input...
        assert _equivalent(inp, exp), f"unsound ground truth for {row['id']}"
        # ...and the input must be obfuscated (strictly larger), so the "covered"
        # metric (output <= expected nodes) actually requires simplification.
        assert node_count(inp) > node_count(exp), f"{row['id']} is not obfuscated"


def _covered(simplifier: Simplifier, row: dict) -> bool:
    inp = parse_expr(row["expr_miasm"])
    exp = parse_expr(row["expected_miasm"])
    out = simplifier.simplify(inp)
    equivalent = expressions_equivalent(inp, out)
    return equivalent is not False and node_count(out) <= node_count(exp)


def test_cegis_corpus_is_covered_with_cegis() -> None:
    """One representative row per category must be covered with CEGIS enabled."""
    rows = _rows()
    by_category: dict[str, dict] = {}
    for row in rows:
        by_category.setdefault(row.get("category", row["id"]), row)
    simplifier = Simplifier(None, enable_cegis=True)  # empty oracle, CEGIS on
    for category, row in by_category.items():
        assert _covered(simplifier, row), f"CEGIS failed to cover {category}"


def test_cegis_corpus_exercises_cegis() -> None:
    """The non-linear categories must be *uncovered* without CEGIS — otherwise
    the corpus would not be measuring the constant solver at all."""
    rows = _rows()
    baseline = Simplifier(None, enable_cegis=False)  # SiMBA/GAMBA only
    nonlinear = [
        r
        for r in rows
        if any(k in r.get("category", "") for k in ("mulxor", "product", "quadratic"))
    ]
    assert nonlinear, "expected non-linear categories in the corpus"
    # At least one genuinely needs CEGIS (the linear simplifiers cannot reach it).
    assert any(not _covered(baseline, r) for r in nonlinear[:6])
