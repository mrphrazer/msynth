import pickle
import sqlite3
import zlib
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from miasm.expression.expression import Expr
from msynth.utils.expr_utils import parse_expr


class NotSqliteOracleError(Exception):
    """Raised when a file is not a valid sqlite oracle database."""

    pass


class SqliteOracleMap:
    """Dict-like lazy-loading proxy for sqlite oracle storage.

    Key hashes are stored as 20 byte binary BLOBs.
    Members are stored as zlib-compressed newline-separated repr strings.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    @staticmethod
    def _to_bin(hex_key: str) -> bytes:
        """Convert hex string key to binary for storage."""
        return bytes.fromhex(hex_key)

    @staticmethod
    def _to_hex(bin_key: bytes) -> str:
        """Convert binary key back to hex string."""
        return bin_key.hex()

    @staticmethod
    def _exprs_to_bytes(exprs: List[Expr]) -> bytes:
        """Serialize list of expressions as compressed repr strings."""
        text = "\n".join(repr(e) for e in exprs)
        return zlib.compress(text.encode(), level=9)

    @staticmethod
    def _bytes_to_exprs(data: bytes) -> List[Expr]:
        """Deserialize compressed repr strings back to expressions."""
        text = zlib.decompress(data).decode()
        return [parse_expr(line) for line in text.split("\n") if line]

    def __contains__(self, key: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM equiv_classes WHERE equiv_class = ?", (self._to_bin(key),)
        )
        return cur.fetchone() is not None

    def __getitem__(self, key: str) -> List[Expr]:
        cur = self._conn.execute(
            "SELECT members FROM equiv_classes WHERE equiv_class = ?",
            (self._to_bin(key),),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(key)
        return self._bytes_to_exprs(row[0])

    def keys(self) -> Iterator[str]:
        cur = self._conn.execute("SELECT equiv_class FROM equiv_classes")
        return (self._to_hex(row[0]) for row in cur)

    def items(self) -> Iterator[Tuple[str, List[Expr]]]:
        cur = self._conn.execute("SELECT equiv_class, members FROM equiv_classes")
        for row in cur:
            yield self._to_hex(row[0]), self._bytes_to_exprs(row[1])

    def __len__(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM equiv_classes")
        return cur.fetchone()[0]


def load_sqlite_oracle_data(
    file_path: Path,
) -> Tuple[Dict[str, Any], sqlite3.Connection, SqliteOracleMap]:
    """Load oracle data from sqlite database.

    Args:
        file_path: Path to sqlite oracle file.

    Returns:
        Tuple of (metadata dict, connection, oracle_map).
        Metadata contains 'num_variables', 'num_samples', 'inputs'.

    Raises:
        NotSqliteOracleError: If the file is not a valid sqlite oracle database.
    """
    try:
        conn = sqlite3.connect(file_path)

        cur = conn.execute("SELECT key, value FROM metadata")

        raw_meta = dict(cur.fetchall())

        meta = {
            "num_variables": raw_meta["num_variables"],
            "num_samples": raw_meta["num_samples"],
            "inputs": pickle.loads(raw_meta["inputs"]),
        }

        return meta, conn, SqliteOracleMap(conn)
    except sqlite3.DatabaseError as e:
        raise NotSqliteOracleError(f"Not a valid sqlite oracle: {e}") from e


def dump_sqlite_oracle_data(
    file_path: Path,
    num_variables: int,
    num_samples: int,
    inputs: List[List[int]],
    oracle_map: Dict[str, List[Expr]],
) -> None:
    """Save oracle data to sqlite database.

    Args:
        file_path: Path to write sqlite file.
        num_variables: Number of variables.
        num_samples: Number of samples.
        inputs: Input samples.
        oracle_map: Equivalence class map.
    """
    if file_path.exists():
        # Remove existing to rebuild from scratch
        file_path.unlink()

    conn = sqlite3.connect(file_path)
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value BLOB)")
    conn.execute(
        "CREATE TABLE equiv_classes (equiv_class BLOB PRIMARY KEY, members BLOB)"
    )

    conn.execute("INSERT INTO metadata VALUES (?, ?)", ("num_variables", num_variables))
    conn.execute("INSERT INTO metadata VALUES (?, ?)", ("num_samples", num_samples))
    conn.execute("INSERT INTO metadata VALUES (?, ?)", ("inputs", pickle.dumps(inputs)))

    for k, v in oracle_map.items():
        conn.execute(
            "INSERT INTO equiv_classes VALUES (?, ?)",
            (bytes.fromhex(k), SqliteOracleMap._exprs_to_bytes(v)),
        )
    conn.commit()
    conn.close()
