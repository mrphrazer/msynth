from __future__ import annotations

from miasm.expression.expression import ExprCompose, ExprCond, ExprId, ExprInt, ExprOp

from msynth.simplification.ast import AbstractSyntaxTreeTranslator


def test_ast_translator_fixed_arity_for_ops() -> None:
    t = AbstractSyntaxTreeTranslator()
    a = ExprId("a", 8)
    b = ExprId("b", 8)
    c = ExprId("c", 8)

    expr = ExprOp("+", a, b, c)
    translated = t.from_expr(expr)

    assert translated == ExprOp("+", ExprOp("+", a, b), c)


def test_ast_translator_fixed_arity_for_compose() -> None:
    t = AbstractSyntaxTreeTranslator()
    a = ExprId("a", 8)
    b = ExprId("b", 8)
    c = ExprId("c", 8)

    expr = ExprCompose(a, b, c)
    translated = t.from_expr(expr)

    assert translated == ExprCompose(ExprCompose(a, b), c)


def test_ast_translator_preserves_conditional() -> None:
    t = AbstractSyntaxTreeTranslator()
    cond = ExprId("c", 1)
    src1 = ExprInt(1, 8)
    src2 = ExprInt(2, 8)

    expr = ExprCond(cond, src1, src2)
    translated = t.from_expr(expr)

    assert translated == ExprCond(cond, src1, src2)
