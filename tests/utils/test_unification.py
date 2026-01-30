from __future__ import annotations

from miasm.expression.expression import ExprId, ExprInt, ExprLoc, ExprMem, ExprOp, LocKey

from msynth.utils.unification import gen_unification_dict, invert_dict, reverse_unification


def test_invert_dict() -> None:
    x = ExprId("x", 8)
    y = ExprId("y", 8)
    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 8)

    d = {x: p0, y: p1}
    inverted = invert_dict(d)

    assert inverted[p0] == x
    assert inverted[p1] == y


def test_gen_unification_dict_candidates_and_sizes() -> None:
    r0 = ExprId("r0", 8)
    r1 = ExprId("r1", 32)
    loc = ExprLoc(LocKey(2), 8)
    mem = ExprMem(r1, 8)
    expr = ExprOp("+", r0, ExprOp("^", mem, loc))

    unification = gen_unification_dict(expr)

    # should include ids, mem, and loc candidates; integers are excluded.
    keys = sorted(list(unification.keys()), key=lambda e: str(e))
    values = [unification[k] for k in keys]

    assert len(keys) == len(values)
    assert all(v.name.startswith("p") for v in values)
    assert all(k.size == v.size for k, v in zip(keys, values))


def test_reverse_unification_restores_originals() -> None:
    x = ExprId("x", 8)
    y = ExprId("y", 8)
    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 8)

    unified_expr = ExprOp("+", p0, p1)
    unification = {x: p0, y: p1}

    restored = reverse_unification(unified_expr, unification)

    assert restored == ExprOp("+", x, y)
