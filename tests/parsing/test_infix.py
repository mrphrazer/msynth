from __future__ import annotations

import pytest
from miasm.expression.expression import ExprId, ExprInt, ExprOp

from msynth.parsing import InfixParseError, parse_infix_expr


def test_parse_variables_and_binary_precedence() -> None:
    x = ExprId("x", 64)
    y = ExprId("y", 64)
    z = ExprId("z", 64)

    assert parse_infix_expr("x + y * z") == ExprOp("+", x, ExprOp("*", y, z))


def test_parse_unary_not_as_bitvector_complement() -> None:
    x = ExprId("x", 8)

    assert parse_infix_expr("~x", size=8) == ExprOp("^", x, ExprInt(0xFF, 8))
    assert parse_infix_expr("~0", size=8) == ExprInt(0xFF, 8)


def test_parse_negative_constants_and_hex() -> None:
    x = ExprId("x", 8)

    assert parse_infix_expr("-6 * x", size=8) == ExprOp("*", ExprInt(0xFA, 8), x)
    assert parse_infix_expr("x ^ 0xff", size=8) == ExprOp("^", x, ExprInt(0xFF, 8))


def test_parse_representative_cobra_expression() -> None:
    x = ExprId("x", 64)
    y = ExprId("y", 64)
    z = ExprId("z", 64)
    t = ExprId("t", 64)

    parsed = parse_infix_expr("-6*((z^(x|(~y|z)))&~t)")
    expected = ExprOp(
        "*",
        ExprInt((-6) & ((1 << 64) - 1), 64),
        ExprOp(
            "&",
            ExprOp(
                "^",
                z,
                ExprOp("|", x, ExprOp("|", ExprOp("^", y, ExprInt(-1, 64)), z)),
            ),
            ExprOp("^", t, ExprInt(-1, 64)),
        ),
    )

    assert parsed == expected


def test_parse_exponentiation() -> None:
    x = ExprId("x", 64)

    assert parse_infix_expr("(x**2)") == ExprOp("**", x, ExprInt(2, 64))


def test_parse_floor_division_as_miasm_division() -> None:
    x = ExprId("x", 8)

    assert parse_infix_expr("x // 3", size=8) == ExprOp("/", x, ExprInt(3, 8))


def test_parse_subscript_variable_name() -> None:
    assert parse_infix_expr("X[0] ^ X[1]", size=8) == ExprOp(
        "^", ExprId("X[0]", 8), ExprId("X[1]", 8)
    )


def test_parse_deep_left_associative_chain() -> None:
    expr = parse_infix_expr("+".join(["x", *["1"] * 1500]), size=8)

    assert expr.size == 8


def test_reject_unsupported_syntax() -> None:
    with pytest.raises(InfixParseError):
        parse_infix_expr("f(x)")
