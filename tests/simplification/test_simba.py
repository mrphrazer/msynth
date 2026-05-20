from __future__ import annotations

from miasm.expression.expression import (
    Expr,
    ExprCompose,
    ExprCond,
    ExprId,
    ExprInt,
    ExprMem,
    ExprOp,
    ExprSlice,
)
from miasm.expression.simplifications import expr_simp

from msynth.parsing import parse_infix_expr
from msynth.simplification.simba import SimbaPass
from msynth.utils.expr_utils import get_unique_variables


def simplify(text: str, *, size: int = 8) -> Expr:
    return SimbaPass().run(parse_infix_expr(text, size=size))


def assert_simplifies_to(source: str, expected: str, *, size: int = 8) -> None:
    source_expr = parse_infix_expr(source, size=size)
    expected_expr = parse_infix_expr(expected, size=size)
    simplified = SimbaPass().run(source_expr)

    assert expr_simp(simplified) == expr_simp(expected_expr)
    assert_equivalent(source_expr, simplified)


def assert_equivalent(left: Expr, right: Expr) -> None:
    variables = sorted(
        set(get_unique_variables(left)) | set(get_unique_variables(right)),
        key=lambda expr: str(expr),
    )
    assert left.size == right.size
    assert len(variables) <= 5

    for assignment in range(1 << len(variables)):
        env = {
            variable: (assignment >> index) & 1
            for index, variable in enumerate(variables)
        }
        assert evaluate(left, env) == evaluate(right, env)


def evaluate(expr: Expr, env: dict[Expr, int]) -> int:
    mask = (1 << expr.size) - 1
    if isinstance(expr, ExprInt):
        return int(expr) & mask
    if isinstance(expr, ExprId):
        return env.get(expr, 0) & mask
    if isinstance(expr, ExprOp):
        args = [evaluate(arg, env) for arg in expr.args]
        if expr.op == "-" and len(args) == 1:
            return (-args[0]) & mask
        if expr.op == "+":
            return sum(args) & mask
        if expr.op == "-" and len(args) >= 2:
            result = args[0]
            for arg in args[1:]:
                result -= arg
            return result & mask
        if expr.op == "*":
            result = 1
            for arg in args:
                result *= arg
            return result & mask
        if expr.op == "&":
            result = mask
            for arg in args:
                result &= arg
            return result & mask
        if expr.op == "|":
            result = 0
            for arg in args:
                result |= arg
            return result & mask
        if expr.op == "^":
            result = 0
            for arg in args:
                result ^= arg
            return result & mask
    raise AssertionError(f"unsupported test expression: {expr!r}")


def node_count(expr: Expr) -> int:
    return len(expr.graph().nodes())


def test_simba_simplifies_and_or_sum_identity() -> None:
    assert_simplifies_to("(x & y) + (x | y)", "x + y")


def test_simba_simplifies_masked_or_subtraction() -> None:
    assert_simplifies_to("(x | y) - (~x & y) - (x & ~y)", "x & y")


def test_simba_simplifies_xor_refinement() -> None:
    assert_simplifies_to("-(a | ~b) + ~b + (a & ~b) + b", "a ^ b")


def test_simba_simplifies_representative_paper_expression() -> None:
    assert_simplifies_to(
        "2*(s&~t)+2*(s^t)-(s|t)+2*~(s^t)-~t-~(s&t)",
        "s",
    )


def test_simba_simplifies_constant_expression() -> None:
    assert_simplifies_to("(x ^ x) + 5", "5")


def test_simba_simplifies_bitwise_not() -> None:
    assert_simplifies_to("~x", "~x")


def test_simba_simplifies_arithmetic_negation_to_modular_coefficient() -> None:
    x = ExprId("x", 8)

    assert simplify("-x") == ExprOp("*", ExprInt(0xFF, 8), x)


def test_simba_simplifies_affine_output_encoding() -> None:
    assert_simplifies_to("7 * ((x & y) + (x | y)) + 3", "7*x + 7*y + 3")


def test_simba_simplifies_more_than_three_variable_generic_case() -> None:
    source = parse_infix_expr("((x & y) + (x | y)) + z + w", size=8)
    simplified = SimbaPass().run(source)

    assert expr_simp(simplified) == expr_simp(parse_infix_expr("x + y + z + w", size=8))
    assert_equivalent(source, simplified)


def test_simba_refines_after_expression_loses_extra_variables() -> None:
    assert_simplifies_to("(x ^ y) + (z - z) + (w - w)", "x ^ y")


def test_simba_reduces_representative_node_count() -> None:
    source = parse_infix_expr(
        "2*(s&~t)+2*(s^t)-(s|t)+2*~(s^t)-~t-~(s&t)",
        size=8,
    )
    simplified = SimbaPass().run(source)

    assert node_count(simplified) < node_count(source)


def test_simba_returns_non_linear_multiplication_unchanged() -> None:
    expr = parse_infix_expr("x * y", size=8)

    assert SimbaPass().run(expr) is expr


def test_simba_returns_mixed_bitwise_arithmetic_unchanged() -> None:
    expr = parse_infix_expr("(x + y) & z", size=8)

    assert SimbaPass().run(expr) is expr


def test_simba_returns_constants_inside_bitwise_operands_unchanged() -> None:
    expr = parse_infix_expr("x & 1", size=8)

    assert SimbaPass().run(expr) is expr


def test_simba_returns_shift_unchanged() -> None:
    expr = parse_infix_expr("x << 1", size=8)

    assert SimbaPass().run(expr) is expr


def test_simba_returns_mixed_width_slice_unchanged() -> None:
    x = ExprId("x", 8)
    expr = ExprSlice(x, 0, 4)

    assert SimbaPass().run(expr) is expr


def test_simba_returns_compose_unchanged() -> None:
    expr = ExprCompose(ExprId("x", 4), ExprId("y", 4))

    assert SimbaPass().run(expr) is expr


def test_simba_returns_memory_unchanged() -> None:
    expr = ExprMem(ExprId("ptr", 8), 8)

    assert SimbaPass().run(expr) is expr


def test_simba_returns_condition_unchanged() -> None:
    expr = ExprCond(ExprId("c", 1), ExprId("x", 8), ExprId("y", 8))

    assert SimbaPass().run(expr) is expr


def test_simba_unsupported_child_does_not_raise() -> None:
    expr = ExprOp("+", ExprMem(ExprId("ptr", 8), 8), ExprInt(1, 8))

    assert SimbaPass().run(expr) is expr
