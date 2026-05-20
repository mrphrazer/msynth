from __future__ import annotations

from miasm.expression.expression import Expr, ExprId, ExprInt, ExprOp
from miasm.expression.simplifications import expr_simp

from msynth.synthesis.oracle import SynthesisOracle
from msynth.synthesis.smir import (
    AddConstantRule,
    AffineRule,
    AndOrMaskRule,
    MulConstantRule,
    PolynomialRule,
    ShiftConstantRule,
    SmirEngine,
    XorConstantRule,
)
from msynth.synthesis.scoring import score_expression
from msynth.synthesis.synthesizer import Synthesizer


def build_oracle(expr: Expr, variable: Expr, values: list[int]) -> SynthesisOracle:
    synthesis_map = {}
    for value in values:
        concrete = ExprInt(value, variable.size)
        output = expr_simp(expr.replace_expr({variable: concrete}))
        synthesis_map[(concrete,)] = output
    return SynthesisOracle(synthesis_map)


def run_rule(rule, target: Expr, base: Expr, variable: Expr) -> Expr:
    oracle = build_oracle(target, variable, [0, 1, 2, 3, 0x55, 0xAA, 0xFF])
    result = SmirEngine([rule]).evaluate(base, oracle, [variable])
    assert result.best_score == 0.0
    return result.best_expr


def test_smir_add_constant_rule() -> None:
    p0 = ExprId("p0", 8)

    inferred = run_rule(
        AddConstantRule(),
        p0 + ExprInt(0xA7, 8),
        p0,
        p0,
    )

    assert inferred == expr_simp(p0 + ExprInt(0xA7, 8))


def test_smir_xor_constant_rule() -> None:
    p0 = ExprId("p0", 8)

    inferred = run_rule(
        XorConstantRule(),
        p0 ^ ExprInt(0xA7, 8),
        p0,
        p0,
    )

    assert inferred == expr_simp(p0 ^ ExprInt(0xA7, 8))


def test_smir_mul_constant_rule() -> None:
    p0 = ExprId("p0", 8)

    inferred = run_rule(
        MulConstantRule(),
        p0 * ExprInt(5, 8),
        p0,
        p0,
    )

    assert inferred == expr_simp(p0 * ExprInt(5, 8))


def test_smir_and_or_mask_rule() -> None:
    p0 = ExprId("p0", 8)

    run_rule(
        AndOrMaskRule(),
        (p0 & ExprInt(0xF0, 8)) | ExprInt(0x0A, 8),
        p0,
        p0,
    )


def test_smir_shift_and_rotate_rules() -> None:
    p0 = ExprId("p0", 8)

    run_rule(ShiftConstantRule("<<"), p0 << ExprInt(3, 8), p0, p0)
    run_rule(ShiftConstantRule(">>"), p0 >> ExprInt(3, 8), p0, p0)
    run_rule(ShiftConstantRule("<<<"), ExprOp("<<<", p0, ExprInt(3, 8)), p0, p0)
    run_rule(ShiftConstantRule(">>>"), ExprOp(">>>", p0, ExprInt(3, 8)), p0, p0)


def test_smir_affine_rule() -> None:
    p0 = ExprId("p0", 8)

    inferred = run_rule(
        AffineRule(),
        (ExprInt(5, 8) * p0) + ExprInt(7, 8),
        p0,
        p0,
    )

    assert inferred == expr_simp((ExprInt(5, 8) * p0) + ExprInt(7, 8))


def test_smir_polynomial_rule() -> None:
    p0 = ExprId("p0", 8)
    target = (p0 * p0) + (ExprInt(3, 8) * p0) + ExprInt(5, 8)

    inferred = run_rule(PolynomialRule(degree=2), target, p0, p0)

    oracle = build_oracle(target, p0, [4, 5, 6, 7])
    assert score_expression(inferred, oracle, [p0]) == 0.0


def test_synthesizer_uses_smir_by_default_for_large_constant() -> None:
    p0 = ExprId("p0", 8)
    expr = p0 + ExprInt(0xA7, 8)

    synthesized, score = Synthesizer().synthesize_from_expression(
        expr, num_samples=8, timeout=0.1
    )

    assert score == 0.0
    assert synthesized == expr_simp(expr)


def test_synthesizer_can_disable_smir() -> None:
    synth = Synthesizer(use_smir=False)

    assert synth.use_smir is False
