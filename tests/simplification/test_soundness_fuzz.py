"""
Soundness regression harness for the rewrite / GAMBA / CEGIS machinery.

Motivation
----------
The previously-shipped rule ``disj_conj_negation_absorb`` was *unsound*: its
matcher accepted a broader structural class than the proved identity covered
(``-x | (~(-x) & 2*x)`` was rewritten to ``-x``). It slipped through the
per-rule tests because those only exercised the single canonical matching
input (base ``a``), never the ``-a``-based variants the matcher also accepted.

These tests guard against that whole class of bug with checks that do NOT
depend on hand-authored inputs:

1. Whatever the live rewriter (``DEFAULT_REWRITER``) and the GAMBA engines
   produce on *random* expressions must be semantically equal to the input.
2. For every registered rule, every *base-substituted* variant its matcher
   still fires on must be equivalent to the rewrite it returns.
3. ``CegisSolver.try_synthesize`` is *sample-validated*, not self-proving:
   it no longer runs its own Z3 equivalence proof (equivalence is enforced by
   the simplifier's shared suitability gate, like the oracle/SimBA tiers). So
   here we assert (a) on clean affine targets the synthesised candidate is in
   fact equivalent, and (b) the *gated* path ``Simplifier(enable_cegis=True,
   enforce_equivalence=True).simplify`` is sound, i.e. the gate is the real
   soundness boundary.

Equivalence is decided exactly with Z3 over the expression's own bit-width; a
Z3 ``unknown`` (hard multiply / timeout) is skipped rather than treated as a
pass, so an unproven case never masquerades as sound.
"""

from __future__ import annotations

import itertools
import random

import z3
from miasm.expression.expression import Expr, ExprId, ExprInt, ExprOp
from miasm.ir.translators.z3_ir import TranslatorZ3

from msynth.simplification.cegis import CegisSolver, TemplateOracle
from msynth.simplification.simplifier import Simplifier
from msynth.simplification.gamba import GAMBA_POST_REWRITER, GAMBA_PREPROCESSOR
from msynth.simplification.rewrites import DEFAULT_REWRITER, DEFAULT_RULES
from msynth.utils.unification import gen_unification_dict

# Reuse the per-rule matching-input factory.
from test_rewrites import _matching_input

_TRANSLATOR = TranslatorZ3()


def _z3_status(a: Expr, b: Expr, *, timeout_ms: int = 4000) -> str:
    """Return 'equivalent', 'counterexample', or 'unknown' for ``a`` vs ``b``."""
    if a.size != b.size:
        return "counterexample"
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.add(_TRANSLATOR.from_expr(a) != _TRANSLATOR.from_expr(b))
    result = solver.check()
    if result == z3.unsat:
        return "equivalent"
    if result == z3.sat:
        return "counterexample"
    return "unknown"


def _assert_not_counterexample(original: Expr, rewritten: Expr, ctx: str) -> bool:
    """Fail on a proven non-equivalence; return True iff Z3 proved equivalence."""
    status = _z3_status(original, rewritten)
    assert status != "counterexample", (
        f"UNSOUND rewrite ({ctx}):\n  in:  {original}\n  out: {rewritten}"
    )
    return status == "equivalent"


# ---------------------------------------------------------------------------
# Random expression generator
# ---------------------------------------------------------------------------

_FUZZ_SIZE = 8
_FUZZ_VARS = [ExprId(f"p{i}", _FUZZ_SIZE) for i in range(2)]
_FUZZ_CONSTS = [0, 1, 2, 3, 4, 8, (1 << _FUZZ_SIZE) - 1, (-1) & ((1 << _FUZZ_SIZE) - 1)]
_BINOPS = ["+", "-", "*", "&", "|", "^"]


def _rand_leaf(rng: random.Random) -> Expr:
    if rng.random() < 0.6:
        return rng.choice(_FUZZ_VARS)
    return ExprInt(rng.choice(_FUZZ_CONSTS), _FUZZ_SIZE)


def _rand_expr(rng: random.Random, depth: int) -> Expr:
    if depth <= 0 or rng.random() < 0.3:
        return _rand_leaf(rng)
    roll = rng.random()
    if roll < 0.15:  # unary negation
        return ExprOp("-", _rand_expr(rng, depth - 1))
    if roll < 0.30:  # bitwise NOT (x ^ all_ones)
        return ExprOp(
            "^", _rand_expr(rng, depth - 1), ExprInt((1 << _FUZZ_SIZE) - 1, _FUZZ_SIZE)
        )
    if roll < 0.45:  # small-coefficient multiple (drives factor/double rules)
        coeff = rng.choice([2, 3, 4, 5])
        return ExprOp("*", ExprInt(coeff, _FUZZ_SIZE), _rand_expr(rng, depth - 1))
    op = rng.choice(_BINOPS)
    return ExprOp(op, _rand_expr(rng, depth - 1), _rand_expr(rng, depth - 1))


def _fuzz_corpus(seed: int, count: int, depth: int = 4) -> list[Expr]:
    rng = random.Random(seed)
    return [_rand_expr(rng, depth) for _ in range(count)]


# ---------------------------------------------------------------------------
# 1. Live-rewriter soundness on random expressions
# ---------------------------------------------------------------------------


def test_default_rewriter_sound_on_random_exprs() -> None:
    proved = 0
    for expr in _fuzz_corpus(seed=0xA5A5, count=300):
        out = DEFAULT_REWRITER.normalize(expr)
        if out == expr:
            continue
        proved += _assert_not_counterexample(expr, out, "DEFAULT_REWRITER.normalize")
    # The corpus must actually exercise the rewriter (guard against a
    # degenerate generator that never triggers a rewrite).
    assert proved > 0


def test_gamba_engines_sound_on_random_exprs() -> None:
    corpus = _fuzz_corpus(seed=0x1234, count=200)
    proved = 0
    for expr in corpus:
        for engine, name in (
            (GAMBA_PREPROCESSOR, "GAMBA_PREPROCESSOR"),
            (GAMBA_POST_REWRITER, "GAMBA_POST_REWRITER"),
        ):
            out = engine.normalize(expr)
            if out == expr:
                continue
            proved += _assert_not_counterexample(expr, out, f"{name}.normalize")
    assert proved > 0


# ---------------------------------------------------------------------------
# 2. Per-rule soundness on structural variants (the RW-1 failure mode)
# ---------------------------------------------------------------------------
#
# RW-1 was unsound only when DIFFERENT occurrences of the same base took
# DIFFERENT values (``~(-x) & 2*x`` mixes base ``-x`` and ``x``). A uniform
# atom substitution can never expose that -- it keeps every occurrence equal.
# So instead we fill each LEAF POSITION of a rule's canonical matching input
# *independently* from a small pool, preserving the operator skeleton (and the
# integer leaves: masks like ``0xFF`` and coefficients like ``2`` that the
# matcher keys on). When the leaf count is small we enumerate every
# combination, which deterministically reaches the mixed-base cases.


def _op_children(expr: Expr):
    return getattr(expr, "args", ())


def _count_leaf_positions(expr: Expr) -> int:
    if isinstance(expr, ExprId):
        return 1
    if isinstance(expr, ExprInt):
        return 0
    return sum(_count_leaf_positions(a) for a in _op_children(expr))


def _rebuild_with(expr: Expr, values) -> Expr:
    """Rebuild ``expr`` consuming one pool value per id-leaf (in tree order)."""
    if isinstance(expr, ExprId):
        return next(values)
    if isinstance(expr, ExprInt):
        return expr
    if isinstance(expr, ExprOp):
        return ExprOp(expr.op, *[_rebuild_with(a, values) for a in expr.args])
    # Other node kinds (Slice/Compose/Cond/Mem) are left intact; the rules
    # whose matching inputs use them are out of this probe's scope.
    return expr


def _leaf_pool(size: int) -> list[Expr]:
    a = ExprId("a", size)
    b = ExprId("b", size)
    return [a, b, ExprOp("-", a), ExprOp("-", b)]


def test_each_rule_sound_on_structural_variants() -> None:
    """
    For every registered rule, fill each leaf position of its canonical
    matching input independently from a small pool (``a``, ``b``, ``-a``,
    ``-b``) and, whenever the matcher still fires, prove the rewrite sound.

    Enumerating leaf fillings reaches the mixed-base shapes that uniform
    substitution cannot -- this is the probe that catches the historical
    ``disj_conj_negation_absorb`` unsoundness (it fired on ``~(-x) & 2*x``,
    which is not absorbed by ``-x``).
    """
    rng = random.Random(0xF0F0)
    fired = 0
    for rule in DEFAULT_RULES:
        base = _matching_input(rule)
        n_positions = _count_leaf_positions(base)
        if n_positions == 0:
            continue
        pool = _leaf_pool(base.size)

        total = len(pool) ** n_positions
        if total <= 4096:
            combos = itertools.product(pool, repeat=n_positions)
        else:
            combos = (
                [rng.choice(pool) for _ in range(n_positions)] for _ in range(400)
            )

        for combo in combos:
            variant = _rebuild_with(base, iter(list(combo)))
            applied = rule.apply(variant)
            if applied is None or applied == variant:
                continue
            fired += 1
            _assert_not_counterexample(
                variant, applied, f"rule {rule.name} on structural variant"
            )
    # Sanity: the probe must have actually triggered rules.
    assert fired > 0


# ---------------------------------------------------------------------------
# 3. CEGIS synthesis soundness
# ---------------------------------------------------------------------------


def test_cegis_synthesis_sound_on_random_affine() -> None:
    """CEGIS affine synthesis is correct, and the gated path is sound.

    ``try_synthesize`` is sample-validated (no self-proof), so we check both:
    (a) on clean affine targets the synthesised candidate really is equivalent,
    and (b) the gated simplifier (which is the actual soundness boundary now)
    never emits a non-equivalent result under ``enforce_equivalence=True``.
    """
    size = 8
    v0 = ExprId("v0", size)
    oracle = TemplateOracle.gen_runtime_oracle(num_variables=1)
    solver = CegisSolver(oracle, max_variables=1)
    gated = Simplifier(enable_cegis=True, enforce_equivalence=True)

    rng = random.Random(0xCE)
    proved = 0
    for _ in range(20):
        c0 = rng.randrange(1, 1 << size)
        c1 = rng.randrange(0, 1 << size)
        expr = v0 * ExprInt(c0, size) + ExprInt(c1, size)
        udict = gen_unification_dict(expr)
        unified = expr.replace_expr(udict)
        candidate = solver.try_synthesize(expr, unified, udict)
        if candidate is not None:
            proved += _assert_not_counterexample(
                expr, candidate, f"CEGIS try_synthesize({c0}*v0 + {c1})"
            )
        # The real soundness boundary: the gated simplifier must never emit a
        # non-equivalent result, no matter what try_synthesize proposed.
        _assert_not_counterexample(
            expr, gated.simplify(expr), f"gated simplify({c0}*v0 + {c1})"
        )
    # The runtime oracle covers the affine template, so at least some of the
    # random affine targets must be synthesised AND proven equivalent.
    assert proved > 0
