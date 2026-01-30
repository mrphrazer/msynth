from __future__ import annotations

from pathlib import Path

import msynth.simplification.oracle as simpl_oracle
from miasm.expression.expression import ExprId, ExprInt, ExprOp

from msynth.simplification.oracle import SimplificationOracle


def test_get_outputs_falls_back_to_tree_eval(monkeypatch) -> None:
    oracle = SimplificationOracle.__new__(SimplificationOracle)
    oracle.num_variables = 1
    oracle.num_samples = 2
    oracle.inputs = [[1], [2]]
    oracle.oracle_map = {}

    def fail_compile(_expr):
        raise ValueError("unsupported")

    monkeypatch.setattr(simpl_oracle, "compile_expr_to_python", fail_compile)

    p0 = ExprId("p0", 8)
    expr = ExprOp("+", p0, ExprInt(1, 8))

    assert oracle.get_outputs(expr) == [2, 3]


def test_runtime_cache_set_and_contains(tmp_path: Path) -> None:
    oracle = SimplificationOracle.__new__(SimplificationOracle)
    oracle.num_variables = 1
    oracle.num_samples = 1
    oracle.inputs = [[0]]
    oracle.oracle_map = {}
    oracle._runtime_cache = {}

    equiv_class = "00" * 20
    oracle.set_equiv_class(equiv_class, [ExprInt(0, 8)])

    assert equiv_class in oracle._runtime_cache
    assert oracle.contains_equiv_class(equiv_class)
