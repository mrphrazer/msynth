from __future__ import annotations

from pathlib import Path

import pytest
from miasm.expression.expression import ExprId

from msynth.utils.sqlite import NotSqliteOracleError, SqliteOracleMap, dump_sqlite_oracle_data, load_sqlite_oracle_data


def test_sqlite_roundtrip_and_map_access(tmp_path: Path) -> None:
    file_path = tmp_path / "oracle.db"
    oracle_map = {"00" * 20: [ExprId("p0", 8)]}
    inputs = [[1], [2]]

    dump_sqlite_oracle_data(
        file_path=file_path,
        num_variables=1,
        num_samples=2,
        inputs=inputs,
        oracle_map=oracle_map,
    )

    meta, conn, oracle_map_loaded = load_sqlite_oracle_data(file_path)

    assert meta["num_variables"] == 1
    assert meta["num_samples"] == 2
    assert meta["inputs"] == inputs

    assert isinstance(oracle_map_loaded, SqliteOracleMap)
    assert "00" * 20 in oracle_map_loaded
    assert list(oracle_map_loaded.keys()) == ["00" * 20]
    assert list(oracle_map_loaded.items()) == [("00" * 20, [ExprId("p0", 8)])]
    assert len(oracle_map_loaded) == 1
    assert oracle_map_loaded["00" * 20] == [ExprId("p0", 8)]

    conn.close()


def test_sqlite_missing_key_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "oracle.db"
    dump_sqlite_oracle_data(
        file_path=file_path,
        num_variables=1,
        num_samples=1,
        inputs=[[0]],
        oracle_map={"00" * 20: [ExprId("p0", 8)]},
    )

    _, conn, oracle_map_loaded = load_sqlite_oracle_data(file_path)

    with pytest.raises(KeyError):
        _ = oracle_map_loaded["11" * 20]

    conn.close()


def test_load_sqlite_oracle_data_rejects_invalid_file(tmp_path: Path) -> None:
    file_path = tmp_path / "not_sqlite.db"
    file_path.write_text("not a sqlite db")

    with pytest.raises(NotSqliteOracleError):
        load_sqlite_oracle_data(file_path)
