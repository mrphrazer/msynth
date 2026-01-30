from __future__ import annotations

from miasm.expression.expression import ExprId, ExprInt, ExprOp

from msynth.synthesis.oracle import SynthesisOracle
from msynth.synthesis.state import SynthesisState
from msynth.utils.expr_utils import get_unique_variables


def test_synthesis_state_score_zero_for_matching() -> None:
    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 8)
    expr = ExprOp("+", p0, p1)

    synthesis_map = {
        (ExprInt(1, 8), ExprInt(2, 8)): ExprInt(3, 8),
        (ExprInt(5, 8), ExprInt(7, 8)): ExprInt(12, 8),
    }
    oracle = SynthesisOracle(synthesis_map)
    state = SynthesisState(expr)

    score = state.get_score(oracle, [p0, p1])

    assert score == 0.0


def test_synthesis_state_score_positive_for_mismatch() -> None:
    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 8)
    expr = ExprOp("+", p0, p1)

    synthesis_map = {
        (ExprInt(1, 8), ExprInt(2, 8)): ExprInt(1 ^ 2, 8),
        (ExprInt(5, 8), ExprInt(7, 8)): ExprInt(5 ^ 7, 8),
    }
    oracle = SynthesisOracle(synthesis_map)
    state = SynthesisState(expr)

    score = state.get_score(oracle, [p0, p1])

    assert score > 0.0


def test_synthesis_state_cleanup_removes_dead_replacements() -> None:
    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 8)
    t1 = ExprId("t1", 8)
    t2 = ExprId("t2", 8)
    t3 = ExprId("t3", 8)

    expr_ast = ExprOp("+", t1, t2)
    replacements = {t1: p0, t2: p1, t3: p0}
    state = SynthesisState(expr_ast, replacements)

    state.cleanup()

    used_vars = set(get_unique_variables(state.expr_ast))
    assert set(state.replacements.keys()) == used_vars


def test_synthesis_state_score_sparse_variables() -> None:
    p0 = ExprId("p0", 8)
    p2 = ExprId("p2", 8)
    expr = ExprOp("+", p0, p2)

    synthesis_map = {
        (ExprInt(3, 8), ExprInt(4, 8)): ExprInt(7, 8),
        (ExprInt(10, 8), ExprInt(5, 8)): ExprInt(15, 8),
    }
    oracle = SynthesisOracle(synthesis_map)
    state = SynthesisState(expr)

    score = state.get_score(oracle, [p0, p2])

    assert score == 0.0
