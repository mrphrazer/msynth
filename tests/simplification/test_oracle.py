from __future__ import annotations

import pickle
import pytest
from pathlib import Path

from miasm.expression.expression import ExprId, ExprInt, ExprOp

from msynth.simplification.oracle import SimplificationOracle


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


def test_oracle_map_builds_and_skips_ints(tmp_path: Path) -> None:
    library = write_library(tmp_path)
    oracle = SimplificationOracle(num_variables=2, num_samples=5, library_path=library)
    # Only non-constant expressions should be included.
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
