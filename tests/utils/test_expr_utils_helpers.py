from __future__ import annotations

from miasm.expression.expression import ExprCompose, ExprCond, ExprId, ExprInt, ExprLoc, ExprMem, ExprOp, LocKey

from msynth.utils.expr_utils import get_subexpressions, get_unification_candidates, get_unique_variables, parse_expr


def test_parse_expr_roundtrip_complex() -> None:
    r0 = ExprId("r0", 32)
    mem = ExprMem(r0, 8)
    cond = ExprCond(ExprId("c", 1), ExprInt(1, 8), ExprInt(2, 8))
    expr = ExprCompose(mem, cond)

    parsed = parse_expr(repr(expr))

    assert parsed == expr


def test_get_unique_variables_sorted() -> None:
    a = ExprId("b", 8)
    b = ExprId("a", 8)
    expr = ExprOp("+", a, ExprOp("+", b, a))

    vars_sorted = get_unique_variables(expr)

    assert vars_sorted == [b, a]


def test_get_unification_candidates_excludes_ints() -> None:
    r0 = ExprId("r0", 8)
    r1 = ExprId("r1", 32)
    loc = ExprLoc(LocKey(3), 8)
    mem = ExprMem(r1, 8)
    expr = ExprOp("+", ExprInt(1, 8), ExprOp("^", r0, ExprOp("+", mem, loc)))

    candidates = get_unification_candidates(expr)

    assert r0 in candidates
    assert mem in candidates
    assert loc in candidates
    assert ExprInt(1, 8) not in candidates


def test_get_subexpressions_includes_root() -> None:
    a = ExprId("a", 8)
    b = ExprId("b", 8)
    expr = ExprOp("+", a, b)

    subs = get_subexpressions(expr)

    assert expr in subs
    assert a in subs
    assert b in subs
