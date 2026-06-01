"""
Tests for the ring-normalisation post-pass.

Every accepted rewrite must be both *semantically equivalent* to the
input and *strictly smaller* by Miasm node count. The net-smaller guard
inside ``ring_normalize`` makes the latter automatic — these tests pin
the cases where it must actually fire as well as the cases where it
must not.
"""

from __future__ import annotations

import random
from typing import Sequence

from miasm.expression.expression import Expr, ExprId, ExprInt, ExprOp
from miasm.expression.simplifications import expr_simp

from msynth.simplification.ring import ring_normalize
from msynth.utils.expr_utils import normalize


def _nodes(expr: Expr) -> int:
    return len(expr.graph().nodes())


def _evaluate(expr: Expr, env: dict) -> int:
    """Concrete evaluation for semantic-equivalence checking."""
    replacements = {var: ExprInt(value, var.size) for var, value in env.items()}
    return int(expr_simp(expr.replace_expr(replacements)))


def _assert_equivalent(
    left: Expr, right: Expr, variables: Sequence[Expr], *, seed: int
) -> None:
    """Random-input equivalence check; 32 sample vectors."""
    random.seed(seed)
    size = variables[0].size
    mask = (1 << size) - 1
    for _ in range(32):
        env = {v: random.getrandbits(size) & mask for v in variables}
        assert _evaluate(left, env) == _evaluate(right, env)


def test_ring_no_op_on_already_flat_sum() -> None:
    size = 8
    v0 = ExprId("v0", size)
    v1 = ExprId("v1", size)
    v2 = ExprId("v2", size)
    expr = expr_simp(ExprOp("+", v0, v1, ExprOp("*", ExprInt(2, size), v2)))
    assert ring_normalize(expr) == expr


def test_ring_distributes_and_collects_across_compound_sum() -> None:
    # The shape that motivated ring-normalize: subtree-SiMBA / CEGIS often
    # produce expressions of the form
    #   v0 * c1 + v2 * c2 + (v0+v1) * c3 + (v0+v1-v2) * c4 + (v1+v0*c5) * c6
    # which Miasm's `expr_simp` leaves at 17 nodes for the canonical demo.
    # Ring-normalize must distribute, collect like terms, drop zeros, and
    # produce strictly fewer nodes.
    size = 64
    v0 = ExprId("v0", size)
    v1 = ExprId("v1", size)
    v2 = ExprId("v2", size)
    neg2 = ExprInt((-2) & ((1 << size) - 1), size)
    expr = expr_simp(
        ExprOp(
            "+",
            ExprOp("*", neg2, v0),
            ExprOp("*", ExprInt(5, size), v2),
            ExprOp("*", ExprInt(2, size), ExprOp("+", v0, v1)),
            ExprOp(
                "*",
                ExprInt(2, size),
                ExprOp("+", v0, v1, ExprOp("-", v2)),
            ),
            ExprOp(
                "*",
                ExprInt(2, size),
                ExprOp("+", v1, ExprOp("*", ExprInt(2, size), v0)),
            ),
        )
    )
    result = ring_normalize(expr)
    _assert_equivalent(result, expr, [v0, v1, v2], seed=1)
    assert _nodes(result) < _nodes(expr)


def test_ring_no_op_when_dag_sharing_blocks_distribution() -> None:
    # When Miasm's DAG sharing keeps the factored form at the same node
    # count as the distributed form, ring-normalize must NOT rewrite —
    # the net-smaller guard preserves the input. Example:
    # `v0 + (v0+v1)*2` has 6 nodes either way under DAG accounting.
    size = 8
    v0 = ExprId("v0", size)
    v1 = ExprId("v1", size)
    expr = expr_simp(
        ExprOp("+", ExprOp("*", ExprInt(2, size), ExprOp("+", v0, v1)), v0)
    )
    result = ring_normalize(expr)
    assert result == expr


def test_ring_collects_like_terms() -> None:
    # v0 * 2 + v1 + v0 * 3  ->  v0 * 5 + v1
    size = 8
    v0 = ExprId("v0", size)
    v1 = ExprId("v1", size)
    expr = expr_simp(
        ExprOp(
            "+",
            ExprOp("*", ExprInt(2, size), v0),
            v1,
            ExprOp("*", ExprInt(3, size), v0),
        )
    )
    result = ring_normalize(expr)
    _assert_equivalent(result, expr, [v0, v1], seed=2)
    # `expr_simp` may already canonicalise this; require result is at
    # most as large as input.
    assert _nodes(result) <= _nodes(expr)
    # And the v0 coefficient is 5.
    assert _evaluate(result, {v0: 1, v1: 0}) == 5


def test_ring_preserves_when_distribution_would_inflate() -> None:
    # (v0 + v1 + v2 + v3) * 2 in isolation — distributing inflates from
    # 6 to 9 nodes. Net-smaller guard must reject.
    size = 8
    v0 = ExprId("v0", size)
    v1 = ExprId("v1", size)
    v2 = ExprId("v2", size)
    v3 = ExprId("v3", size)
    inner_sum = ExprOp("+", v0, v1, v2, v3)
    expr = expr_simp(ExprOp("*", ExprInt(2, size), inner_sum))
    result = ring_normalize(expr)
    # Either unchanged or, if Miasm canonicalises differently, at least
    # not larger.
    assert _nodes(result) <= _nodes(expr)
    _assert_equivalent(result, expr, [v0, v1, v2, v3], seed=3)


def test_ring_handles_zero_coefficient_atoms() -> None:
    # 3*v0 + v1 - 3*v0  ->  v1
    size = 8
    v0 = ExprId("v0", size)
    v1 = ExprId("v1", size)
    three_v0 = ExprOp("*", ExprInt(3, size), v0)
    expr = ExprOp("+", three_v0, v1, ExprOp("-", three_v0))
    # Build manually to keep the `-` in place (expr_simp may already
    # collapse).
    result = ring_normalize(expr)
    _assert_equivalent(result, expr, [v0, v1], seed=4)
    # Must reduce to (effectively) just v1.
    assert _evaluate(result, {v0: 42, v1: 7}) == 7


def test_ring_preserves_semantics_under_random_inputs() -> None:
    # A larger composite expression: 2*(v0 + v1) + 3*(v0 - v2) + v1
    # Ring should fold to: v0*5 + v1*3 + v2*-3 (or equivalent).
    size = 8
    v0 = ExprId("v0", size)
    v1 = ExprId("v1", size)
    v2 = ExprId("v2", size)
    expr = ExprOp(
        "+",
        ExprOp("*", ExprInt(2, size), ExprOp("+", v0, v1)),
        ExprOp(
            "*",
            ExprInt(3, size),
            ExprOp("+", v0, ExprOp("-", v2)),
        ),
        v1,
    )
    result = ring_normalize(expr)
    _assert_equivalent(result, expr, [v0, v1, v2], seed=5)


def test_normalize_pipelines_expr_simp_then_ring() -> None:
    # ``normalize`` is the single-entry pass that callers should use. On
    # the regression shape it must produce the same result as the explicit
    # composition (expr_simp -> ring_normalize), and must do real work
    # (strictly fewer nodes than the raw input).
    size = 64
    v0 = ExprId("v0", size)
    v1 = ExprId("v1", size)
    v2 = ExprId("v2", size)
    neg2 = ExprInt((-2) & ((1 << size) - 1), size)
    raw = ExprOp(
        "+",
        ExprOp("*", neg2, v0),
        ExprOp("*", ExprInt(5, size), v2),
        ExprOp("*", ExprInt(2, size), ExprOp("+", v0, v1)),
        ExprOp("*", ExprInt(2, size), ExprOp("+", v0, v1, ExprOp("-", v2))),
        ExprOp(
            "*",
            ExprInt(2, size),
            ExprOp("+", v1, ExprOp("*", ExprInt(2, size), v0)),
        ),
    )
    direct = normalize(raw)
    composed = ring_normalize(expr_simp(raw))
    assert direct == composed
    assert _nodes(direct) < _nodes(raw)
    _assert_equivalent(direct, raw, [v0, v1, v2], seed=6)


def test_ring_size_safe_returns_unchanged_on_non_sum() -> None:
    # Non-sum-rooted expressions short-circuit.
    size = 8
    v0 = ExprId("v0", size)
    expr = ExprOp("*", ExprInt(3, size), v0)
    assert ring_normalize(expr) is expr
