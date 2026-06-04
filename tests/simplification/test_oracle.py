from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from miasm.expression.expression import ExprId, ExprInt, ExprOp

import msynth.simplification.oracle as simpl_oracle
from msynth.simplification.oracle import SimplificationOracle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_library(tmp_path: Path) -> Path:
    # Keep expressions simple and valid for eval() in oracle module scope.
    lines = [
        "ExprOp('+', ExprId('p0', 8), ExprId('p1', 8))",
        "ExprOp('^', ExprId('p0', 8), ExprId('p1', 8))",
        "ExprInt(1, 8)",
    ]
    path = tmp_path / "lib.txt"
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Oracle construction / serialization
# ---------------------------------------------------------------------------


def test_oracle_map_builds_and_skips_ints(tmp_path: Path) -> None:
    library = write_library(tmp_path)
    oracle = SimplificationOracle(num_variables=2, num_samples=5, library_path=library)
    # only non-constant expressions should be included.
    assert sum(len(v) for v in oracle.oracle_map.values()) == 2


def test_oracle_equiv_class_determinism(tmp_path: Path) -> None:
    library = write_library(tmp_path)
    oracle = SimplificationOracle(num_variables=2, num_samples=5, library_path=library)
    expr = ExprOp("+", ExprId("p0", 8), ExprId("p1", 8))
    outputs = oracle.get_outputs(expr)
    equiv = oracle.determine_equiv_class(expr, outputs)
    assert oracle.contains_equiv_class(equiv)


def test_oracle_roundtrip(tmp_path: Path) -> None:
    library = write_library(tmp_path)
    oracle = SimplificationOracle(num_variables=2, num_samples=5, library_path=library)
    out_path = tmp_path / "oracle.pkl"
    oracle.dump_to_file(out_path)
    loaded = SimplificationOracle.load_from_file(out_path)
    assert loaded.num_variables == oracle.num_variables
    assert loaded.num_samples == oracle.num_samples
    assert loaded.oracle_map.keys() == oracle.oracle_map.keys()


def test_oracle_sqlite_roundtrip(tmp_path: Path) -> None:
    library = write_library(tmp_path)
    oracle = SimplificationOracle(num_variables=2, num_samples=5, library_path=library)
    out_path = tmp_path / "oracle.db"
    oracle.dump_to_file(out_path, use_sqlite=True)

    with SimplificationOracle.load_from_file(out_path) as loaded:
        assert loaded.num_variables == oracle.num_variables
        assert loaded.num_samples == oracle.num_samples
        for key in oracle.oracle_map.keys():
            assert loaded.contains_equiv_class(key)
            original_members = [e for e in oracle.oracle_map[key]]
            loaded_members = [e for e in loaded.oracle_map[key]]
            assert loaded_members == original_members


def test_oracle_load_rejects_wrong_type(tmp_path: Path) -> None:
    out_path = tmp_path / "oracle.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"not": "an oracle"}, f)

    with pytest.raises(TypeError, match="SimplificationOracle"):
        SimplificationOracle.load_from_file(out_path)


# ---------------------------------------------------------------------------
# Evaluation fallback + runtime cache (previously in test_oracle_more.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Empty-oracle factory (used by Simplifier when no oracle_path is given)
# ---------------------------------------------------------------------------


def test_empty_oracle_construction() -> None:
    oracle = SimplificationOracle.empty()
    assert isinstance(oracle, SimplificationOracle)
    assert oracle.num_variables == 3
    assert oracle.num_samples == 8
    assert oracle.oracle_map == {}
    # Inputs matrix is correctly sized: num_samples rows of num_variables columns.
    assert len(oracle.inputs) == 8
    assert all(len(row) == 3 for row in oracle.inputs)


def test_empty_oracle_construction_accepts_custom_sizing() -> None:
    oracle = SimplificationOracle.empty(num_variables=5, num_samples=12)
    assert oracle.num_variables == 5
    assert oracle.num_samples == 12
    assert oracle.oracle_map == {}
    assert len(oracle.inputs) == 12
    assert all(len(row) == 5 for row in oracle.inputs)


def test_empty_oracle_inputs_are_deterministic() -> None:
    # Two empty oracles with the same sizing produce identical inputs.
    o1 = SimplificationOracle.empty(num_variables=2, num_samples=3)
    o2 = SimplificationOracle.empty(num_variables=2, num_samples=3)
    assert o1.inputs == o2.inputs


def test_empty_oracle_contains_no_equiv_classes() -> None:
    oracle = SimplificationOracle.empty()
    p0 = ExprId("p0", 8)
    outputs = oracle.get_outputs(p0)
    equiv = oracle.determine_equiv_class(p0, outputs)
    assert not oracle.contains_equiv_class(equiv)
