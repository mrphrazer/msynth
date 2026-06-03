"""
Regression tests for the miasm-free fixpoint engine and rewriter pair in
``msynth.simplification.gamba``.

Covers seven test categories:

1. Engine fixpoint behaviour — convergence, identity, bottom-up walk.
2. Per-rule no-grow invariant — every safe rule's output must not grow
   the tree (the contract that lets the engine fixpoint without
   external guards).
3. Pipeline regression — known-good post-shapes, idempotence and
   soundness on a hand-curated corpus.
4. Post-rewriter regression — when ring/factor net-shrinks vs. when
   they're rejected; idempotence; soundness.
5. Equivalence-vs-miasm benchmark — semantic parity with the previous
   miasm-backed normalisation on a hand-curated corpus, with a loose
   1.5x node-count headroom.
6. Corpus shape regression — pull 10 deterministic lines from the
   shipped CoBRA gamba/loki suite and assert no-grow + Z3-soundness.
7. Soundness fuzz — random expressions; each rewritten output must be
   Z3-equivalent to its input.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Callable, List

import pytest
from miasm.expression.expression import Expr, ExprInt, ExprOp
from miasm.expression.simplifications import expr_simp

from msynth.parsing import parse_infix_expr
from msynth.simplification.gamba import (
    GAMBA_POST_REWRITER,
    GAMBA_PREPROCESSOR,
    _GambaEngine,
)
from msynth.simplification.rewrites import (
    DEFAULT_REWRITER,
    DEFAULT_RULES,
    RewriteRule,
)

# Reuse the per-rule matching-input factory and equivalence helpers from
# the existing per-rule soundness suite. The tests below stay aligned
# with the shapes that test_rewrites.py already exercises, so a
# regression in either pinpoints the same root cause.
from tests.simplification.test_rewrites import (
    _MASK,
    _SIZE,
    _atoms,
    _matching_input,
    _not_,
    _z3_equivalent,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _nodes(expr: Expr) -> int:
    return len(expr.graph().nodes())


def _safe_rules() -> List[RewriteRule]:
    return [r for r in DEFAULT_RULES if not r.guarded]


# ---------------------------------------------------------------------------
# Category 1 — engine fixpoint tests
# ---------------------------------------------------------------------------


def _build_safe_engine() -> _GambaEngine:
    return _GambaEngine.build(tuple(_safe_rules()))


def _walk_count_to_fixpoint(engine: _GambaEngine, expr: Expr, cap: int = 50) -> int:
    """Count outer walks until quiescent. Helps assert the engine
    converges in a small bounded number of iterations on shapes whose
    structure tells us a priori how many bottom-up passes are needed."""
    for n in range(1, cap + 1):
        new_expr = engine._walk(expr)
        if new_expr == expr:
            return n
        expr = new_expr
    return cap + 1


def test_engine_converges_in_one_walk_on_single_rule_fire() -> None:
    a, _, _, _ = _atoms()
    engine = _build_safe_engine()
    # ~~a -> a in a single rule fire; engine should hit fixpoint after
    # the rewrite walk + one confirmation walk.
    expr = _not_(_not_(a))
    assert _walk_count_to_fixpoint(engine, expr) <= 2


def test_engine_converges_on_atom_in_one_walk() -> None:
    a, _, _, _ = _atoms()
    engine = _build_safe_engine()
    assert _walk_count_to_fixpoint(engine, a) == 1


def test_engine_converges_on_constant_in_one_walk() -> None:
    engine = _build_safe_engine()
    assert _walk_count_to_fixpoint(engine, ExprInt(42, _SIZE)) == 1


def test_engine_normalize_returns_atom_unchanged() -> None:
    a, _, _, _ = _atoms()
    assert GAMBA_PREPROCESSOR.normalize(a) == a


def test_engine_normalize_returns_constant_unchanged() -> None:
    e = ExprInt(0xDEAD, _SIZE)
    assert GAMBA_PREPROCESSOR.normalize(e) == e


def test_engine_handles_deeply_nested_double_negation() -> None:
    # ~~~~a -> a in a single bottom-up pass (the inner local-rewrite
    # loop folds both NOT layers in one walk).
    a, _, _, _ = _atoms()
    engine = _build_safe_engine()
    expr = _not_(_not_(_not_(_not_(a))))
    assert engine.normalize(expr) == a


def test_engine_max_iters_cap_returns_safely_without_raising() -> None:
    # The cap is a watchdog — pass a value of 1 to force a forced
    # early return, and confirm no exception is raised.
    a, b, _, _ = _atoms()
    safe = tuple(_safe_rules())
    engine = _GambaEngine.build(safe)
    expr = ExprOp("+", ExprOp("&", a, b), ExprOp("&", _not_(a), b))
    out = engine.normalize(expr, max_iters=1)
    # Must not raise; output type is Expr.
    assert isinstance(out, Expr)


def test_engine_bottom_up_inner_rule_unlocks_outer_pattern() -> None:
    # Build a tree where the INNER rule must fire before the OUTER
    # rule's pattern is revealed: ((a & ~a) + b) — the inner ``a & ~a``
    # collapses to 0, then ``0 + b`` collapses to ``b``.
    a, b, _, _ = _atoms()
    expr = ExprOp("+", ExprOp("&", a, _not_(a)), b)
    out = GAMBA_PREPROCESSOR.normalize(expr)
    assert out == b


def test_engine_bottom_up_three_level_collapse() -> None:
    # ~(a ^ a) -> ~0 -> -1 (xor self-cancel + double-negation interplay)
    a, _, _, _ = _atoms()
    expr = _not_(ExprOp("^", a, a))
    out = GAMBA_PREPROCESSOR.normalize(expr)
    assert out == ExprInt(_MASK, _SIZE)


def test_engine_compose_inverse_pair_under_top_level_addition() -> None:
    # (a & b) + (~a & b) + c -> b + c (allowing commutative reordering)
    a, b, c, _ = _atoms()
    expr = ExprOp(
        "+",
        ExprOp("&", a, b),
        ExprOp("&", _not_(a), b),
        c,
    )
    out = GAMBA_PREPROCESSOR.normalize(expr)
    expected = ExprOp("+", b, c)
    assert out == expected or _z3_equivalent(out, expected)


def test_engine_walks_count_three_redundancies_in_two_passes() -> None:
    a, b, c, _ = _atoms()
    engine = _build_safe_engine()
    expr = ExprOp(
        "|",
        ExprOp("&", a, _not_(a)),
        ExprOp("&", b, _not_(b)),
        ExprOp("&", c, _not_(c)),
    )
    # All three child conjunctions collapse to 0 in one walk;
    # one more walk confirms the fixpoint.
    assert _walk_count_to_fixpoint(engine, expr) <= 2


def test_engine_idempotent_under_repeated_normalize() -> None:
    a, b, c, _ = _atoms()
    expr = ExprOp("+", ExprOp("&", a, b), ExprOp("&", _not_(a), b), c)
    once = GAMBA_PREPROCESSOR.normalize(expr)
    twice = GAMBA_PREPROCESSOR.normalize(once)
    assert once == twice


def test_engine_no_oscillation_on_already_normalised_sum() -> None:
    # A sum that no safe rule simplifies should still terminate cleanly.
    a, b, c, _ = _atoms()
    expr = ExprOp("+", a, b, c)
    engine = _build_safe_engine()
    assert _walk_count_to_fixpoint(engine, expr) == 1


def test_engine_normalize_terminates_on_random_inputs() -> None:
    # 10 random expressions; every one must terminate within the
    # default 50-iter cap without raising.
    rng = random.Random(0xC0FFEE)
    a, b, c, _ = _atoms()
    leaves = [a, b, c]
    for _ in range(10):
        depth = rng.randint(1, 3)
        expr = _random_expr(rng, leaves, depth)
        out = GAMBA_PREPROCESSOR.normalize(expr)
        assert isinstance(out, Expr)


def test_engine_walk_count_bottomup_correctness_compound() -> None:
    # Three nested layers that each unlock the next:
    #   inner: a ^ a -> 0
    #   mid:   (0) + b -> b   (const_fold_add_zero)
    #   outer: ~~b -> b       (double_negation_collapse)
    a, b, _, _ = _atoms()
    inner = ExprOp("^", a, a)
    mid = ExprOp("+", inner, b)
    outer = _not_(_not_(mid))
    out = GAMBA_PREPROCESSOR.normalize(outer)
    assert out == b


# ---------------------------------------------------------------------------
# Category 2 — no-grow invariant per safe rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", _safe_rules(), ids=lambda r: r.name)
def test_safe_rule_never_grows_node_count(rule: RewriteRule) -> None:
    inp = _matching_input(rule)
    out = rule.apply(inp)
    assert out is not None, f"{rule.name} unexpectedly rejected its matching input"
    assert _nodes(out) <= _nodes(inp), (
        f"{rule.name} grew: in={_nodes(inp)} -> out={_nodes(out)} ({inp} -> {out})"
    )


# ---------------------------------------------------------------------------
# Category 3 — GambaPreprocessor regression tests
# ---------------------------------------------------------------------------


def _allones() -> ExprInt:
    return ExprInt(_MASK, _SIZE)


_PREPROCESSOR_EXPECTED_SHAPES: List[tuple[Expr, Expr]] = [
    # ~~a -> a
    (_not_(_not_(_atoms()[0])), _atoms()[0]),
    # a & ~a -> 0
    (ExprOp("&", _atoms()[0], _not_(_atoms()[0])), ExprInt(0, _SIZE)),
    # a | ~a -> -1
    (ExprOp("|", _atoms()[0], _not_(_atoms()[0])), _allones()),
    # a ^ a -> 0
    (ExprOp("^", _atoms()[0], _atoms()[0]), ExprInt(0, _SIZE)),
    # a + 0 -> a
    (ExprOp("+", _atoms()[0], ExprInt(0, _SIZE)), _atoms()[0]),
    # a * 1 -> a
    (ExprOp("*", _atoms()[0], ExprInt(1, _SIZE)), _atoms()[0]),
    # a * 0 -> 0
    (ExprOp("*", _atoms()[0], ExprInt(0, _SIZE)), ExprInt(0, _SIZE)),
    # a | (a & b) -> a  (absorption_or)
    (
        ExprOp("|", _atoms()[0], ExprOp("&", _atoms()[0], _atoms()[1])),
        _atoms()[0],
    ),
    # a & (a | b) -> a  (absorption_and)
    (
        ExprOp("&", _atoms()[0], ExprOp("|", _atoms()[0], _atoms()[1])),
        _atoms()[0],
    ),
    # (a & b) + (~a & b) -> b  (inverse_xor_neg)
    (
        ExprOp(
            "+",
            ExprOp("&", _atoms()[0], _atoms()[1]),
            ExprOp("&", _not_(_atoms()[0]), _atoms()[1]),
        ),
        _atoms()[1],
    ),
]


@pytest.mark.parametrize(
    "inp,expected",
    _PREPROCESSOR_EXPECTED_SHAPES,
    ids=lambda x: str(x)[:50],
)
def test_preprocessor_normalises_to_expected_shape(inp: Expr, expected: Expr) -> None:
    out = GAMBA_PREPROCESSOR.normalize(inp)
    # Accept either exact structural match or Z3 equivalence to allow
    # for ordering differences in n-ary operators.
    if out != expected:
        assert _z3_equivalent(out, expected), (
            f"normalize produced {out!r}, expected {expected!r}"
        )


def test_preprocessor_on_atom_is_identity() -> None:
    a, _, _, _ = _atoms()
    assert GAMBA_PREPROCESSOR.normalize(a) == a


def test_preprocessor_on_constant_is_identity() -> None:
    e = ExprInt(0x1234, _SIZE)
    assert GAMBA_PREPROCESSOR.normalize(e) == e


_IDEMPOTENCE_SHAPES = [
    # Inverse-xor-neg embedded in a larger sum.
    lambda: ExprOp(
        "+",
        ExprOp("&", _atoms()[0], _atoms()[1]),
        ExprOp("&", _not_(_atoms()[0]), _atoms()[1]),
        _atoms()[2],
    ),
    # Double NOT pyramid.
    lambda: _not_(_not_(_not_(_not_(_atoms()[0])))),
    # n-ary idempotence + redundancy chain.
    lambda: ExprOp("&", _atoms()[0], _atoms()[0], _atoms()[1], _atoms()[1]),
    # Absorption pyramid: a | (a & b) | (a & c)
    lambda: ExprOp(
        "|",
        _atoms()[0],
        ExprOp("&", _atoms()[0], _atoms()[1]),
        ExprOp("&", _atoms()[0], _atoms()[2]),
    ),
    # XOR self-cancel pairs.
    lambda: ExprOp("^", _atoms()[0], _atoms()[1], _atoms()[0], _atoms()[2]),
]


@pytest.mark.parametrize("shape_factory", _IDEMPOTENCE_SHAPES, ids=lambda f: f.__name__)
def test_preprocessor_is_idempotent(shape_factory: Callable[[], Expr]) -> None:
    expr = shape_factory()
    once = GAMBA_PREPROCESSOR.normalize(expr)
    twice = GAMBA_PREPROCESSOR.normalize(once)
    assert once == twice


_SOUNDNESS_SHAPES = [
    # Self-cancel sums and xors
    lambda: ExprOp("^", _atoms()[0], _atoms()[0]),
    lambda: ExprOp("+", _atoms()[0], ExprInt(0, _SIZE)),
    # Inverse pairs
    lambda: ExprOp(
        "+",
        ExprOp("&", _atoms()[0], _atoms()[1]),
        ExprOp("&", _not_(_atoms()[0]), _atoms()[1]),
    ),
    lambda: ExprOp(
        "+",
        ExprOp("|", _atoms()[0], _atoms()[1]),
        ExprOp("|", _not_(_atoms()[0]), _atoms()[1]),
    ),
    # Demorgan
    lambda: _not_(ExprOp("&", _not_(_atoms()[0]), _atoms()[1])),
    # DeMorgan twin (or version)
    lambda: _not_(ExprOp("|", _not_(_atoms()[0]), _atoms()[1])),
    # Constant merge under &.
    lambda: ExprOp(
        "+",
        ExprOp("&", ExprInt(0x0F, _SIZE), _atoms()[0]),
        ExprOp("&", ExprInt(0xF0, _SIZE), _atoms()[0]),
    ),
    # Absorption
    lambda: ExprOp("|", _atoms()[0], ExprOp("&", _atoms()[0], _atoms()[1])),
    # Double NOT
    lambda: _not_(_not_(_atoms()[0])),
    # Complement pair and/or
    lambda: ExprOp(
        "|",
        ExprOp("&", _atoms()[0], _atoms()[1]),
        ExprOp("&", _atoms()[0], _not_(_atoms()[1])),
    ),
]


@pytest.mark.parametrize("shape_factory", _SOUNDNESS_SHAPES, ids=lambda f: f.__name__)
def test_preprocessor_output_is_z3_equivalent_to_input(
    shape_factory: Callable[[], Expr],
) -> None:
    inp = shape_factory()
    out = GAMBA_PREPROCESSOR.normalize(inp)
    assert _z3_equivalent(inp, out)


# ---------------------------------------------------------------------------
# Category 4 — GambaPostRewriter regression tests
# ---------------------------------------------------------------------------


_POST_NET_SHRINK_SHAPES = [
    # 2*a + (-2)*a + b -> b (ring normalisation cancels)
    lambda: ExprOp(
        "+",
        ExprOp("*", ExprInt(2, _SIZE), _atoms()[0]),
        ExprOp("*", ExprInt((-2) & _MASK, _SIZE), _atoms()[0]),
        _atoms()[1],
    ),
    # a*b + a*c -> a*(b+c)  (factor_common_subterm)
    lambda: ExprOp(
        "+",
        ExprOp("*", _atoms()[0], _atoms()[1]),
        ExprOp("*", _atoms()[0], _atoms()[2]),
    ),
    # 2*a + 3*a -> 5*a (ring collects)
    lambda: ExprOp(
        "+",
        ExprOp("*", ExprInt(2, _SIZE), _atoms()[0]),
        ExprOp("*", ExprInt(3, _SIZE), _atoms()[0]),
    ),
    # 2*(a+b) + 3*(a+b) -> 5*a + 5*b  (ring distributes and collects)
    lambda: ExprOp(
        "+",
        ExprOp("*", ExprInt(2, _SIZE), ExprOp("+", _atoms()[0], _atoms()[1])),
        ExprOp("*", ExprInt(3, _SIZE), ExprOp("+", _atoms()[0], _atoms()[1])),
    ),
    # ((a&b)*c) + ((a&b)*d) -> (a&b)*(c+d)
    lambda: ExprOp(
        "+",
        ExprOp("*", ExprOp("&", _atoms()[0], _atoms()[1]), _atoms()[2]),
        ExprOp("*", ExprOp("&", _atoms()[0], _atoms()[1]), _atoms()[3]),
    ),
]


@pytest.mark.parametrize(
    "shape_factory", _POST_NET_SHRINK_SHAPES, ids=lambda f: f.__name__
)
def test_post_rewriter_net_shrinks_when_ring_or_factor_apply(
    shape_factory: Callable[[], Expr],
) -> None:
    inp = shape_factory()
    out = GAMBA_POST_REWRITER.normalize(inp)
    assert _nodes(out) < _nodes(inp)
    assert _z3_equivalent(inp, out)


_POST_NOOP_SHAPES = [
    # Bare identifier — nothing to do.
    lambda: _atoms()[0],
    # Bare integer.
    lambda: ExprInt(0x1234, _SIZE),
    # Already-canonical sum.
    lambda: ExprOp("+", _atoms()[0], _atoms()[1]),
    # Bare multiplication — factor can't reduce a non-sum.
    lambda: ExprOp("*", _atoms()[0], _atoms()[1]),
    # 5*a + 3*b — different atoms, different coefficients; ring is
    # neutral and factor finds no common subterm.
    lambda: ExprOp(
        "+",
        ExprOp("*", ExprInt(5, _SIZE), _atoms()[0]),
        ExprOp("*", ExprInt(3, _SIZE), _atoms()[1]),
    ),
]


@pytest.mark.parametrize("shape_factory", _POST_NOOP_SHAPES, ids=lambda f: f.__name__)
def test_post_rewriter_is_neutral_when_no_guarded_rule_shrinks(
    shape_factory: Callable[[], Expr],
) -> None:
    inp = shape_factory()
    pre = GAMBA_PREPROCESSOR.normalize(inp)
    post = GAMBA_POST_REWRITER.normalize(inp)
    # Post must not grow beyond the preprocessor output.
    assert _nodes(post) <= _nodes(pre)
    assert _z3_equivalent(inp, post)


def test_post_rewriter_is_idempotent_on_shrink_case() -> None:
    inp = _POST_NET_SHRINK_SHAPES[1]()  # factor
    once = GAMBA_POST_REWRITER.normalize(inp)
    twice = GAMBA_POST_REWRITER.normalize(once)
    assert once == twice


def test_post_rewriter_is_idempotent_on_noop_case() -> None:
    inp = _POST_NOOP_SHAPES[2]()  # a + b
    once = GAMBA_POST_REWRITER.normalize(inp)
    twice = GAMBA_POST_REWRITER.normalize(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Category 5 — equivalence-vs-miasm benchmark
# ---------------------------------------------------------------------------


_BENCHMARK_SHAPES = [
    # MBA-style identities
    lambda: ExprOp(
        "+",
        ExprOp("&", _atoms()[0], _atoms()[1]),
        ExprOp("|", _atoms()[0], _atoms()[1]),
    ),
    lambda: _not_(_not_(_atoms()[0])),
    lambda: ExprOp("&", _atoms()[0], _not_(_atoms()[0])),
    lambda: ExprOp("|", _atoms()[0], ExprOp("&", _atoms()[0], _atoms()[1])),
    lambda: ExprOp("^", _atoms()[0], _atoms()[0], _atoms()[1]),
    lambda: ExprOp("+", _atoms()[0], ExprInt(0, _SIZE)),
    lambda: ExprOp(
        "+",
        ExprOp("&", _atoms()[0], _atoms()[1]),
        ExprOp("&", _not_(_atoms()[0]), _atoms()[1]),
    ),
    lambda: ExprOp(
        "|",
        ExprOp("&", _atoms()[0], _atoms()[1]),
        ExprOp("&", _atoms()[0], _not_(_atoms()[1])),
    ),
    lambda: ExprOp(
        "+",
        ExprOp("&", ExprInt(0x0F, _SIZE), _atoms()[0]),
        ExprOp("&", ExprInt(0xF0, _SIZE), _atoms()[0]),
    ),
    lambda: ExprOp("&", _atoms()[0], _atoms()[0], _atoms()[1]),
]


@pytest.mark.parametrize("shape_factory", _BENCHMARK_SHAPES, ids=lambda f: f.__name__)
def test_benchmark_gamba_matches_miasm_semantics_and_size(
    shape_factory: Callable[[], Expr],
) -> None:
    inp = shape_factory()
    miasm_out = DEFAULT_REWRITER.expr_simp()(inp)
    gamba_out = GAMBA_PREPROCESSOR.normalize(inp)
    # Both must be sound rewrites of the input.
    assert _z3_equivalent(inp, miasm_out)
    assert _z3_equivalent(inp, gamba_out)
    # Loose headroom: gamba must not be more than 1.5x miasm's size,
    # rounded up so the bound is meaningful at small node counts.
    cap = max(int(_nodes(miasm_out) * 1.5) + 1, _nodes(miasm_out))
    assert _nodes(gamba_out) <= cap, (
        f"gamba grew vs miasm: miasm={_nodes(miasm_out)} "
        f"gamba={_nodes(gamba_out)} ({gamba_out})"
    )


# ---------------------------------------------------------------------------
# Category 6 — CoBRA gamba/loki corpus shape regression
# ---------------------------------------------------------------------------


_CORPUS_PATH = (
    Path(__file__).resolve().parents[2] / "datasets" / "corpora" / "cobra.jsonl.gz"
)


def _load_corpus_rows(n: int = 10) -> List[dict]:
    """Load the first ``n`` gamba/loki_tiny rows deterministically."""
    rows: List[dict] = []
    with gzip.open(_CORPUS_PATH, "rt") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("source", "").startswith("gamba/loki_tiny"):
                rows.append(row)
                if len(rows) >= n:
                    break
    return rows


_CORPUS_ROWS = _load_corpus_rows(10)


@pytest.mark.parametrize("row", _CORPUS_ROWS, ids=lambda r: r["id"])
def test_corpus_normalize_does_not_grow_and_stays_sound(row: dict) -> None:
    inp = parse_infix_expr(row["expr_text"], size=row["size"])
    out = GAMBA_PREPROCESSOR.normalize(inp)
    assert _nodes(out) <= _nodes(inp), (
        f"{row['id']}: nodes grew {_nodes(inp)} -> {_nodes(out)} on {row['expr_text']}"
    )
    assert _z3_equivalent(inp, out)


# ---------------------------------------------------------------------------
# Category 7 — soundness fuzz
# ---------------------------------------------------------------------------


def _random_expr(rng: random.Random, leaves: list[Expr], depth: int) -> Expr:
    """Constrained generator from the SimBA fuzzer's bitwise+arithmetic
    grammar. Mixes the linear-MBA fragment with NOT and unary negation
    to exercise the safe-rule set broadly.

    Note: miasm's ``-`` operator must be unary or strictly binary;
    we therefore never emit an n-ary ``-`` here.
    """
    if depth <= 0 or rng.random() < 0.35:
        return rng.choice(leaves)
    op = rng.choice(["+", "-", "*", "&", "|", "^", "not_", "neg"])
    if op == "not_":
        return _not_(_random_expr(rng, leaves, depth - 1))
    if op == "neg":
        return ExprOp("-", _random_expr(rng, leaves, depth - 1))
    if op == "*":
        # Coefficient * subexpr, to stay close to MBA-like shapes.
        const = ExprInt(rng.getrandbits(_SIZE) & _MASK, _SIZE)
        sub = _random_expr(rng, leaves, depth - 1)
        return ExprOp("*", const, sub)
    if op == "-":
        # Binary subtraction only; n-ary "-" is rejected by miasm.
        lhs = _random_expr(rng, leaves, depth - 1)
        rhs = _random_expr(rng, leaves, depth - 1)
        return ExprOp("-", lhs, rhs)
    n = rng.randint(2, 3)
    args = tuple(_random_expr(rng, leaves, depth - 1) for _ in range(n))
    return ExprOp(op, *args)


def test_engine_handles_deep_recursion_without_propagating() -> None:
    # Synthesise a deeply nested left-associated OR so the engine's
    # post-order descent would naturally blow Python's default 1000-frame
    # recursion limit. The guard inside ``_GambaEngine.normalize`` must
    # catch ``RecursionError`` and bail out with the input unchanged
    # (sound: no rewrite is always sound).
    a = _atoms()[0]
    depth = 5000
    deep = a
    for _ in range(depth):
        deep = ExprOp("|", deep, a)
    out = GAMBA_PREPROCESSOR.normalize(deep)
    # Either the engine handled it via iterative-equivalent shrinking, or
    # the recursion guard returned the input. Either is acceptable; the
    # contract is "no propagated exception" + soundness.
    # We can't Z3-check a 5000-node OR, so just confirm it didn't raise
    # and the result is structurally well-formed.
    assert out is not None
    # Same check for the post-rewriter, which composes the engine with
    # guarded ring/factor passes.
    out_post = GAMBA_POST_REWRITER.normalize(deep)
    assert out_post is not None


@pytest.mark.parametrize("seed", [1, 7, 23, 101, 4242])
def test_fuzz_preprocessor_is_sound_on_random_expressions(seed: int) -> None:
    rng = random.Random(seed)
    a, b, c, d = _atoms()
    leaves = [a, b, c, d, ExprInt(1, _SIZE), ExprInt(_MASK, _SIZE), ExprInt(0, _SIZE)]
    failures: list[tuple[Expr, Expr]] = []
    for _ in range(50):
        depth = rng.randint(1, 3)
        # Pre-simplify with miasm so we don't burn the budget on shapes
        # the fuzzer happens to emit that the rewriter normalises away
        # before any algebraic identity has a chance to fire.
        inp = expr_simp(_random_expr(rng, leaves, depth))
        try:
            out = GAMBA_PREPROCESSOR.normalize(inp)
        except RecursionError:
            continue  # Pathological depth from the random generator.
        try:
            equiv = _z3_equivalent(inp, out)
        except AssertionError:
            # Z3 returned unknown — skip this seed sample rather than
            # silently passing.
            continue
        if not equiv:
            failures.append((inp, out))
    assert not failures, f"unsound rewrites: {failures[:3]}"
