from __future__ import annotations


import pytest
from miasm.expression.expression import ExprId, ExprInt, ExprOp
from miasm.expression.simplifications import expr_simp

import msynth.synthesis.oracle as synth_oracle


def test_synthesis_oracle_from_expression(monkeypatch: pytest.MonkeyPatch) -> None:
    # deterministic inputs for sampling
    seq = iter([1, 2, 3, 4, 5, 6])

    def fake_get_rand_input() -> int:
        return next(seq)

    monkeypatch.setattr(synth_oracle, "get_rand_input", fake_get_rand_input)

    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 8)
    expr = ExprOp("+", p0, p1)
    oracle = synth_oracle.SynthesisOracle.gen_from_expression(expr, [p0, p1], num_samples=3)

    assert len(oracle.synthesis_map) == 3
    for inputs, output in oracle.synthesis_map.items():
        replacements = {p0: inputs[0], p1: inputs[1]}
        expected = expr_simp(expr.replace_expr(replacements))
        assert int(output) == int(expected)


def test_synthesis_oracle_rejects_empty_map() -> None:
    with pytest.raises(ValueError, match="empty"):
        synth_oracle.SynthesisOracle({})


def test_synthesis_oracle_requires_exprint_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    # force expr_simp to return a non-int expression to trigger the type check.
    def fake_expr_simp(_expr):
        return ExprId("x", 8)

    monkeypatch.setattr(synth_oracle, "expr_simp", fake_expr_simp)

    p0 = ExprId("p0", 8)
    expr = ExprOp("+", p0, ExprInt(1, 8))

    with pytest.raises(TypeError, match="ExprInt"):
        synth_oracle.SynthesisOracle.gen_from_expression(expr, [p0], num_samples=1)


def test_synthesis_oracle_overwrites_duplicate_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_rand_input() -> int:
        return 1

    monkeypatch.setattr(synth_oracle, "get_rand_input", fake_get_rand_input)

    p0 = ExprId("p0", 8)
    expr = ExprOp("+", p0, ExprInt(1, 8))
    oracle = synth_oracle.SynthesisOracle.gen_from_expression(expr, [p0], num_samples=3)

    assert len(oracle.synthesis_map) == 1
