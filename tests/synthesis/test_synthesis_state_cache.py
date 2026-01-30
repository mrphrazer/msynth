from __future__ import annotations

from miasm.expression.expression import ExprId, ExprInt, ExprOp

from msynth.synthesis.oracle import SynthesisOracle
import msynth.synthesis.state as synthesis_state
from msynth.synthesis.state import SynthesisState


def test_state_compiled_cache_reused(monkeypatch) -> None:
    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 8)
    expr = ExprOp("+", p0, p1)

    synthesis_map = {
        (ExprInt(1, 8), ExprInt(2, 8)): ExprInt(3, 8),
        (ExprInt(5, 8), ExprInt(7, 8)): ExprInt(12, 8),
    }
    oracle = SynthesisOracle(synthesis_map)
    state = SynthesisState(expr)

    calls = {"count": 0}

    real_compile = synthesis_state.compile_expr_to_python

    def counted_compile(e):
        calls["count"] += 1
        return real_compile(e)

    monkeypatch.setattr(synthesis_state, "compile_expr_to_python", counted_compile)

    assert state.get_score(oracle, [p0, p1]) == 0.0
    assert state.get_score(oracle, [p0, p1]) == 0.0

    assert calls["count"] == 1


def test_state_falls_back_when_compilation_fails(monkeypatch) -> None:
    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 8)
    expr = ExprOp("+", p0, p1)

    synthesis_map = {
        (ExprInt(1, 8), ExprInt(2, 8)): ExprInt(3, 8),
        (ExprInt(5, 8), ExprInt(7, 8)): ExprInt(12, 8),
    }
    oracle = SynthesisOracle(synthesis_map)
    state = SynthesisState(expr)

    def fail_compile(_e):
        raise ValueError("no compile")

    monkeypatch.setattr(synthesis_state, "compile_expr_to_python", fail_compile)

    assert state.get_score(oracle, [p0, p1]) == 0.0
