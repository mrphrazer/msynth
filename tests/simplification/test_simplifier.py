from __future__ import annotations

import pickle
from pathlib import Path

import z3
from miasm.expression.expression import ExprId, ExprInt, ExprLoc, ExprOp, LocKey

from msynth.simplification.oracle import SimplificationOracle
from msynth.simplification.simplifier import Simplifier


def _write_min_oracle(tmp_path: Path) -> Path:
    # create a minimal oracle instance without invoking multiprocessing-heavy init.
    oracle = SimplificationOracle.__new__(SimplificationOracle)
    oracle.num_variables = 1
    oracle.num_samples = 3
    oracle.inputs = [[0], [1], [2]]
    oracle.oracle_map = {}

    path = tmp_path / "oracle.pkl"
    with open(path, "wb") as f:
        pickle.dump(oracle, f)
    return path


def test_skip_subtree_terminals(tmp_path: Path) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path))

    assert simplifier._skip_subtree(ExprId("p0", 8))
    assert simplifier._skip_subtree(ExprInt(1, 8))
    assert simplifier._skip_subtree(ExprLoc(LocKey(1), 8))
    assert not simplifier._skip_subtree(ExprOp("+", ExprId("p0", 8), ExprInt(1, 8)))


def test_determine_equivalence_class_adds_constant(tmp_path: Path) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path))

    p0 = ExprId("p0", 8)
    expr = ExprOp("-", p0, p0)  # always zero
    equiv_class = simplifier.determine_equivalence_class(expr)

    assert equiv_class in simplifier.oracle.oracle_map
    assert simplifier.oracle.oracle_map[equiv_class] == [ExprInt(0, 8)]


def test_reverse_global_unification_iterative(tmp_path: Path) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path))

    g0 = simplifier._gen_global_variable_replacement(0, 8)
    g1 = simplifier._gen_global_variable_replacement(1, 8)
    x = ExprId("x", 8)
    y = ExprId("y", 8)

    expr = ExprOp("+", g0, g1)
    unification = {g0: ExprOp("+", x, g1), g1: y}

    rewritten = simplifier._reverse_global_unification(expr, unification)

    assert rewritten == ExprOp("+", ExprOp("+", x, y), y)


def test_is_suitable_simplification_candidate_rejects_placeholder(tmp_path: Path) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path))

    expr = ExprOp("+", ExprId("p0", 8), ExprInt(1, 8))
    simplified = ExprId("p0", 8)

    assert not simplifier._is_suitable_simplification_candidate(expr, simplified)


def test_is_suitable_simplification_candidate_rejects_expr_simp_equivalence(tmp_path: Path) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path))

    p0 = ExprId("p0", 8)
    expr = ExprOp("+", p0, ExprInt(0, 8))

    assert not simplifier._is_suitable_simplification_candidate(expr, p0)


def test_is_suitable_simplification_candidate_enforce_equivalence(tmp_path: Path, monkeypatch) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path), enforce_equivalence=True)

    expr = ExprOp("+", ExprId("p0", 8), ExprInt(1, 8))
    simplified = ExprOp("^", ExprId("p0", 8), ExprInt(1, 8))

    monkeypatch.setattr(simplifier, "check_semantical_equivalence", lambda _a, _b: z3.unknown)

    assert not simplifier._is_suitable_simplification_candidate(expr, simplified)


def test_is_suitable_simplification_candidate_accepts_unknown_without_enforce(tmp_path: Path, monkeypatch) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path), enforce_equivalence=False)

    expr = ExprOp("+", ExprId("x", 8), ExprInt(1, 8))
    simplified = ExprOp("^", ExprId("x", 8), ExprInt(1, 8))

    monkeypatch.setattr(simplifier, "check_semantical_equivalence", lambda _a, _b: z3.unknown)

    assert simplifier._is_suitable_simplification_candidate(expr, simplified)


def test_is_suitable_simplification_candidate_rejects_sat(tmp_path: Path, monkeypatch) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path))

    expr = ExprOp("+", ExprId("p0", 8), ExprInt(1, 8))
    simplified = ExprOp("^", ExprId("p0", 8), ExprInt(1, 8))

    monkeypatch.setattr(simplifier, "check_semantical_equivalence", lambda _a, _b: z3.sat)

    assert not simplifier._is_suitable_simplification_candidate(expr, simplified)
