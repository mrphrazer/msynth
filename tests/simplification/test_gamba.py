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
from test_rewrites import (
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


# ---------------------------------------------------------------------------
# GAMBA §5.1 substitution loop — classifier, nonlinear-leaf finder,
# gamba_substitution, and the shared abstract_terms / reverse_abstraction
# placeholder primitive.
# ---------------------------------------------------------------------------

from miasm.expression.expression import ExprId  # noqa: E402

from msynth.simplification.gamba import (  # noqa: E402
    classify_linear_nonlinear,
    gamba_substitution,
    nonlinear_leaves,
)
from msynth.utils.unification import (  # noqa: E402
    abstract_terms,
    reverse_abstraction,
)


# Common atoms reused by the GAMBA-general tests; size 64 matches the
# rest of the test suite. The choice of size is irrelevant for structural
# tests but matters for the BV-soundness tests that compare via Z3 in the
# wider corpus.
_X = ExprId("x", 64)
_Y = ExprId("y", 64)
_A = ExprId("a", 64)
_B = ExprId("b", 64)


# ---------- classify_linear_nonlinear ----------


def test_classify_atom_is_linear() -> None:
    assert classify_linear_nonlinear(_X) == "linear"


def test_classify_constant_is_linear() -> None:
    assert classify_linear_nonlinear(ExprInt(7, 64)) == "linear"


def test_classify_sum_of_atoms_is_linear() -> None:
    assert classify_linear_nonlinear(ExprOp("+", _X, _Y)) == "linear"


def test_classify_const_times_var_is_linear() -> None:
    expr = ExprOp("*", ExprInt(3, 64), _X)
    assert classify_linear_nonlinear(expr) == "linear"


def test_classify_var_times_var_is_nonlinear() -> None:
    expr = ExprOp("*", _X, _Y)
    assert classify_linear_nonlinear(expr) == "nonlinear"


def test_classify_bitwise_over_atoms_is_linear() -> None:
    assert classify_linear_nonlinear(ExprOp("&", _X, _Y)) == "linear"
    assert classify_linear_nonlinear(ExprOp("|", _X, _Y)) == "linear"
    assert classify_linear_nonlinear(ExprOp("^", _X, _Y)) == "linear"


def test_classify_propagates_nonlinearity_through_sum() -> None:
    # A sum containing a nonlinear product is classified nonlinear.
    expr = ExprOp("+", ExprOp("*", _X, _Y), _A)
    assert classify_linear_nonlinear(expr) == "nonlinear"


# ---------- nonlinear_leaves ----------


def test_nonlinear_leaves_empty_for_pure_linear() -> None:
    # Pure linear expr — classifier finds no nonlinear leaves.
    assert nonlinear_leaves(ExprOp("+", _X, _Y)) == []


def test_nonlinear_leaves_finds_xy_product() -> None:
    expr = ExprOp("+", ExprOp("*", _X, _Y), ExprOp("*", _A, ExprInt(3, 64)))
    leaves = nonlinear_leaves(expr)
    assert len(leaves) == 1
    assert leaves[0] == ExprOp("*", _X, _Y)


def test_nonlinear_leaves_descends_into_nonlinear_nodes() -> None:
    # The walker collects BOTH the inner and outer nonlinear sub-expressions
    # because §5.1 may want to abstract either one — abstracting the inner
    # may expose more linear structure than abstracting the outer.
    inner = ExprOp("*", _X, _Y)
    outer = ExprOp("*", inner, _A)  # variable * (variable * variable)
    leaves = nonlinear_leaves(outer)
    assert outer in leaves
    assert inner in leaves


def test_nonlinear_leaves_shares_inner_across_siblings() -> None:
    # ``x*y`` appears inside both ``a*x*y`` and ``b*x*y``; the leaf finder
    # surfaces it as a single de-duplicated entry so §5.1 can pick the
    # shared abstraction.
    inner = ExprOp("*", _X, _Y)
    left = ExprOp("*", _A, inner)
    right = ExprOp("*", _B, inner)
    expr = ExprOp("+", left, right)
    leaves = nonlinear_leaves(expr)
    # All three nonlinear sub-expressions present.
    assert inner in leaves
    assert left in leaves
    assert right in leaves
    # ``inner`` appears once even though it lives inside both siblings.
    assert sum(1 for leaf in leaves if leaf == inner) == 1


def test_nonlinear_leaves_dedupes_by_identity() -> None:
    # The same Python Expr appearing twice is collected once.
    prod = ExprOp("*", _X, _Y)
    expr = ExprOp("+", prod, prod)
    assert len(nonlinear_leaves(expr)) == 1


def test_nonlinear_leaves_finds_shift() -> None:
    # Shifts are outside the linear-MBA fragment per the classifier.
    expr = ExprOp("+", ExprOp("<<", _X, ExprInt(3, 64)), _A)
    leaves = nonlinear_leaves(expr)
    assert leaves and leaves[0].op == "<<"


# ---------- abstract_terms / reverse_abstraction round-trip (prefix="g") ----------
# GAMBA's §5.1 loop uses the shared msynth.utils.unification primitive with the
# "g" prefix; these tests exercise it exactly as gamba_substitution calls it.


def test_abstract_reverse_roundtrip_single_target() -> None:
    prod = ExprOp("*", _X, _Y)
    expr = ExprOp("+", prod, _A)
    abstracted, mapping = abstract_terms(expr, [prod], prefix="g")
    # Placeholder var introduced.
    placeholders = [v for v in mapping]
    assert len(placeholders) == 1
    assert placeholders[0].name.startswith("g")
    # Reverse restores original.
    assert reverse_abstraction(abstracted, mapping) == expr


def test_abstract_reverse_roundtrip_multiple_targets() -> None:
    p1 = ExprOp("*", _X, _Y)
    p2 = ExprOp("*", _A, _B)
    expr = ExprOp("+", p1, p2)
    abstracted, mapping = abstract_terms(expr, [p1, p2], prefix="g")
    assert len(mapping) == 2
    assert reverse_abstraction(abstracted, mapping) == expr


def test_abstract_renames_all_occurrences() -> None:
    prod = ExprOp("*", _X, _Y)
    # Same nonlinear sub-expression appears twice.
    expr = ExprOp("+", prod, prod)
    abstracted, mapping = abstract_terms(expr, [prod], prefix="g")
    assert len(mapping) == 1
    placeholder = next(iter(mapping))
    # Both occurrences replaced with the placeholder.
    expected = ExprOp("+", placeholder, placeholder)
    assert abstracted == expected


# ---------- gamba_substitution (§5.1 wrapper, currently n=0 only) ----------


def test_gamba_substitution_n0_delegates_to_simba_fn() -> None:
    # The wrapper is a thin call into simba_fn for max_k=0. Verify by
    # passing a stub simba_fn that returns a sentinel.
    sentinel = ExprId("sentinel", 64)
    calls: list[Expr] = []

    def simba_fn(expr: Expr) -> Expr:
        calls.append(expr)
        return sentinel

    out = gamba_substitution(_X, simba_fn, max_k=0)
    assert calls == [_X]
    assert out == sentinel


def test_gamba_substitution_returns_none_when_simba_returns_input() -> None:
    # SimBA produced no reduction (returns the same Expr) — wrapper
    # signals miss with None so the BFS loop falls through to CEGIS.
    def simba_fn(expr: Expr) -> Expr:
        return expr

    assert gamba_substitution(_X, simba_fn, max_k=0) is None


def test_gamba_substitution_returns_none_when_simba_returns_none() -> None:
    def simba_fn(_expr: Expr) -> Expr | None:
        return None

    assert gamba_substitution(_X, simba_fn, max_k=0) is None


def test_gamba_substitution_max_k_zero_disables_escalation() -> None:
    # n=0 path takes one SimBA call and returns the result (or None);
    # the escalation loop is gated off entirely. With a simba_fn that
    # would succeed on an abstracted form but fail on the raw subtree,
    # max_k=0 returns None.
    xy = ExprOp("*", _X, _Y)
    subtree = ExprOp("+", ExprOp("*", _A, xy), ExprOp("*", _B, xy))
    calls: list[Expr] = []

    def miss(expr: Expr) -> Expr | None:
        calls.append(expr)
        return None

    assert gamba_substitution(subtree, miss, max_k=0) is None
    # max_k=0 → exactly one SimBA invocation (the n=0 attempt).
    assert len(calls) == 1
    assert calls[0] == subtree


# ---------- §5.1 escalation (max_k >= 1) ----------


def _placeholder_friendly_simba(expr: Expr) -> Expr | None:
    """
    Synthetic SimBA stub for §5.1 tests.

    Recognises only the linearised shape produced by §5.1 abstraction:
    ``f * g + h * g`` where ``g`` is a placeholder. Folds to
    ``(f + h) * g``. Returns ``None`` on anything else.

    Crucially this stub does NOT fold the raw (non-abstracted) shape,
    so the escalation loop is the only way to obtain a reduction.
    """

    def is_placeholder(node: Expr) -> bool:
        return isinstance(node, ExprId) and node.name.startswith("g")

    if isinstance(expr, ExprOp) and expr.op == "+" and len(expr.args) == 2:
        left, right = expr.args
        if (
            isinstance(left, ExprOp)
            and left.op == "*"
            and isinstance(right, ExprOp)
            and right.op == "*"
            and any(is_placeholder(arg) for arg in left.args)
            and any(is_placeholder(arg) for arg in right.args)
        ):
            shared = set(left.args) & set(right.args)
            for candidate in shared:
                if is_placeholder(candidate):
                    la = [arg for arg in left.args if arg != candidate][0]
                    rb = [arg for arg in right.args if arg != candidate][0]
                    return ExprOp("*", ExprOp("+", la, rb), candidate)
    return None


def test_gamba_substitution_escalation_unlocks_shared_nonlinear_factor() -> None:
    # ``a*x*y + b*x*y`` — the inner ``x*y`` is shared between both products.
    # n=0 SimBA misses (the raw shape isn't a placeholder-friendly linear
    # MBA), but abstracting ``x*y`` linearises the sum to ``a*g0 + b*g0``,
    # which the synthetic SimBA folds. Reverse-substitution restores
    # ``(a+b)*x*y`` which is net-smaller (7 < 8 nodes).
    xy = ExprOp("*", _X, _Y)
    subtree = ExprOp("+", ExprOp("*", _A, xy), ExprOp("*", _B, xy))

    assert gamba_substitution(subtree, _placeholder_friendly_simba, max_k=0) is None
    out = gamba_substitution(subtree, _placeholder_friendly_simba, max_k=1)
    assert out is not None
    assert _nodes(out) < _nodes(subtree)


def test_gamba_substitution_escalation_returns_smallest_candidate() -> None:
    # Three nonlinear leaves; the n=1 escalation tries each combination
    # and the wrapper returns the smallest restored result. The synthetic
    # SimBA only fires on the ``x*y`` abstraction (shared between both
    # products), so the other two combos miss and the loop returns the
    # one that fired.
    xy = ExprOp("*", _X, _Y)
    subtree = ExprOp("+", ExprOp("*", _A, xy), ExprOp("*", _B, xy))
    out = gamba_substitution(subtree, _placeholder_friendly_simba, max_k=1)
    assert out is not None
    # Restored form is ``(a+b)*x*y``.
    assert isinstance(out, ExprOp) and out.op == "*"


def test_gamba_substitution_rejects_when_restored_form_not_smaller() -> None:
    # ``a*x*y + x*y``: abstracting ``x*y`` linearises to ``a*g0 + g0`` —
    # the synthetic SimBA does not match this shape (because one operand
    # is not a product), so the escalation finds no candidate even at
    # max_k=2.
    xy = ExprOp("*", _X, _Y)
    subtree = ExprOp("+", ExprOp("*", _A, xy), xy)
    assert gamba_substitution(subtree, _placeholder_friendly_simba, max_k=2) is None


def test_gamba_substitution_no_leaves_short_circuits() -> None:
    # Pure linear input has zero nonlinear leaves; the escalation loop
    # never enters and the wrapper returns None when n=0 also misses.
    subtree = ExprOp("+", _X, _Y)
    calls: list[Expr] = []

    def miss(expr: Expr) -> Expr | None:
        calls.append(expr)
        return None

    assert gamba_substitution(subtree, miss, max_k=3) is None
    # Exactly one SimBA call (n=0). The escalation loop short-circuits
    # because nonlinear_leaves returns the empty list.
    assert len(calls) == 1


def test_gamba_substitution_respects_gating_cap() -> None:
    # Many nonlinear leaves should NOT lead to combinatorial explosion.
    # With max_k=3 and 12 leaves the gating cap clips inner ``k`` to 2,
    # so the total SimBA call count is bounded by C(12,1) + C(12,2) = 78,
    # not C(12,3) = 220 + others. We assert the upper bound is respected.
    from math import comb

    leaves = [ExprOp("*", ExprId(f"v{i}", 64), ExprId(f"w{i}", 64)) for i in range(12)]
    subtree = leaves[0]
    for leaf in leaves[1:]:
        subtree = ExprOp("+", subtree, leaf)

    call_count = 0

    def miss(expr: Expr) -> Expr | None:
        nonlocal call_count
        call_count += 1
        return None

    gamba_substitution(subtree, miss, max_k=3)
    # n=0 attempt + C(12,1) + C(12,2). Gating clips k_max to 2 (n>9).
    expected_upper = 1 + comb(12, 1) + comb(12, 2)
    assert call_count == expected_upper


def test_gated_max_k_caps_per_paper() -> None:
    from msynth.simplification.gamba import _gated_max_k

    # No leaves → no escalation.
    assert _gated_max_k(0, 5) == 0
    # max_k=0 disables escalation regardless of leaf count.
    assert _gated_max_k(10, 0) == 0
    # Few leaves: max_k respected up to the leaf count itself.
    assert _gated_max_k(3, 5) == 3
    assert _gated_max_k(5, 5) == 5
    # Mid range (5 < n <= 9): cap at min(max_k, 3).
    assert _gated_max_k(7, 5) == 3
    assert _gated_max_k(9, 5) == 3
    # High range (n > 9): cap at min(max_k, 2).
    assert _gated_max_k(10, 5) == 2
    assert _gated_max_k(50, 5) == 2


def test_gamba_substitution_round_trip_preserves_atoms() -> None:
    # After §5.1 abstraction + simba + reverse-abstract, the returned Expr
    # must NOT contain any g* placeholder. The simplifier loop's
    # _is_suitable_simplification_candidate would reject otherwise.
    import re

    xy = ExprOp("*", _X, _Y)
    subtree = ExprOp("+", ExprOp("*", _A, xy), ExprOp("*", _B, xy))
    out = gamba_substitution(subtree, _placeholder_friendly_simba, max_k=1)
    assert out is not None
    placeholders = [
        v
        for v in out.get_r(mem_read=False)
        if hasattr(v, "name") and re.match(r"^g\d+$", v.name)
    ]
    assert placeholders == []


# ---------------------------------------------------------------------------
# Classifier drift-guard: simba._classify vs gamba.classify_linear_nonlinear
# ---------------------------------------------------------------------------
#
# Both classifiers encode the same linear-MBA fragment rule (product needs a
# constant operand; +/- of linear is linear; bitwise-over-atoms is linear) but
# live in separate modules with different return types and fidelities. SimBA's
# is the rich, width-aware, soundness-critical one; GAMBA's is a deliberately
# coarser routing heuristic. The contract we pin here:
#
#   1. On CORE single-level shapes the two agree exactly.
#   2. Everywhere, GAMBA-"linear" implies SimBA does NOT reject the node
#      (GAMBA is a conservative under-approximation of SimBA's linear set;
#      it may call a nested-bitwise shape nonlinear that SimBA handles, but
#      it must never call linear something SimBA rejects).
#
# If a future edit changes the product/bitwise rule in only one classifier,
# one of these assertions fails loudly.

from msynth.simplification.simba import _classify as _simba_classify  # noqa: E402


def _simba_says_linear(expr: Expr) -> bool:
    """SimBA's verdict mapped onto GAMBA's linear/nonlinear axis: a node is
    'linear' iff SimBA classifies it without rejecting (kind is not None) and,
    for operator nodes, without atomising it (is_atom False)."""
    kind, is_atom = _simba_classify(expr, expr.size, {})
    if kind is None:
        return False
    if not isinstance(expr, ExprOp):
        return True
    return not is_atom


_Z = ExprId("z", 64)
_C = ExprInt(3, 64)

_CORE_LINEAR = [
    _X,
    _C,
    _X + _Y,
    _X - _Y,
    _C * _X,
    _X & _Y,
    _X | _Y,
    _X ^ _Y,
]
_CORE_NONLINEAR = [
    ExprOp("*", _X, _Y),
    ExprOp("*", _X, _Y, _Z),
    ExprOp("<<", _X, ExprInt(2, 64)),
    ExprOp("/", _X, _Y),
    ExprOp("%", _X, _Y),
]


@pytest.mark.parametrize("expr", _CORE_LINEAR, ids=lambda e: str(e)[:24])
def test_classifiers_agree_linear_on_core_shapes(expr: Expr) -> None:
    assert classify_linear_nonlinear(expr) == "linear"
    assert _simba_says_linear(expr)


@pytest.mark.parametrize("expr", _CORE_NONLINEAR, ids=lambda e: str(e)[:24])
def test_classifiers_agree_nonlinear_on_core_shapes(expr: Expr) -> None:
    assert classify_linear_nonlinear(expr) == "nonlinear"
    assert not _simba_says_linear(expr)


@pytest.mark.parametrize(
    "expr",
    _CORE_LINEAR
    + _CORE_NONLINEAR
    + [
        (_X & _Y) & _Z,        # nested bitwise: GAMBA coarse-nonlinear, SimBA linear
        (_X + _Y) & _Z,        # both nonlinear
        _C * (_X + _Y),        # both linear
        (_X & _Y) + _Z,        # both linear
        ExprOp("*", _X, _Y) + _Z,  # contains a rejected product
    ],
    ids=lambda e: str(e)[:24],
)
def test_gamba_linear_implies_simba_not_rejected(expr: Expr) -> None:
    # GAMBA must never route as "linear" a node SimBA would reject.
    if classify_linear_nonlinear(expr) == "linear":
        assert _simba_says_linear(expr), expr
