from __future__ import annotations

import gzip
import json
import pickle
import subprocess
import sys
from pathlib import Path

from msynth.simplification.oracle import SimplificationOracle

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_simplification_corpus.py"
)


def write_min_oracle(tmp_path: Path) -> Path:
    oracle = SimplificationOracle.__new__(SimplificationOracle)
    oracle.num_variables = 1
    oracle.num_samples = 3
    oracle.inputs = [[0], [1], [2]]
    oracle.oracle_map = {}

    path = tmp_path / "oracle.pickle"
    with path.open("wb") as handle:
        pickle.dump(oracle, handle)
    return path


def write_corpus(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "corpus.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")
    return path


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        text=True,
        capture_output=True,
    )


def base_record(**updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "case_000000",
        "source": "simba/example.txt",
        "suite": "simba",
        "expr_text": "x + 0",
        "expected_text": "x",
        "size": 8,
    }
    record.update(updates)
    return record


def test_run_simplification_corpus_script_accepts_matching_case(tmp_path: Path) -> None:
    oracle = write_min_oracle(tmp_path)
    corpus = write_corpus(tmp_path, [base_record()])

    result = run_script(
        "--corpus",
        str(corpus),
        "--oracle",
        str(oracle),
        "--limit",
        "1",
        "--jobs",
        "1",
    )

    assert result.returncode == 0
    assert "checked=1" in result.stdout
    assert "passed=1" in result.stdout
    assert "ground_truth=1" in result.stdout
    assert result.stderr == ""


def test_run_simplification_corpus_script_reports_mismatch(tmp_path: Path) -> None:
    oracle = write_min_oracle(tmp_path)
    corpus = write_corpus(
        tmp_path,
        [base_record(expr_text="x ^ 1", expected_text="x")],
    )

    result = run_script(
        "--corpus",
        str(corpus),
        "--oracle",
        str(oracle),
        "--limit",
        "1",
        "--jobs",
        "1",
    )

    assert result.returncode == 1
    assert "checked=1" in result.stdout
    assert "failed=1" in result.stdout
    assert "case_000000" in result.stderr
    assert "not_shorter" in result.stderr
    assert "nodes:    3 -> 3" in result.stderr
    assert "simplified: x ^ 0x1" in result.stderr
    assert (
        "simplified_ir: ExprOp('^', ExprId('x', 8), ExprInt(0x1, 8))" in result.stderr
    )


def test_run_simplification_corpus_script_accepts_shorter_case(tmp_path: Path) -> None:
    oracle = write_min_oracle(tmp_path)
    corpus = write_corpus(
        tmp_path,
        [base_record(expr_text="x + 0", expected_text="x ^ 1")],
    )

    result = run_script(
        "--corpus",
        str(corpus),
        "--oracle",
        str(oracle),
        "--limit",
        "1",
        "--jobs",
        "1",
    )

    assert result.returncode == 0
    assert "checked=1" in result.stdout
    assert "passed=1" in result.stdout
    assert "ground_truth=0" in result.stdout
    assert "shorter=1" in result.stdout
    assert result.stderr == ""


def test_run_simplification_corpus_script_reports_simplifier_error(
    tmp_path: Path,
) -> None:
    oracle = write_min_oracle(tmp_path)
    corpus = write_corpus(
        tmp_path,
        [base_record(expr_text="x + y", expected_text="x + y")],
    )

    result = run_script(
        "--corpus",
        str(corpus),
        "--oracle",
        str(oracle),
        "--limit",
        "1",
        "--jobs",
        "1",
    )

    assert result.returncode == 1
    assert "checked=1" in result.stdout
    assert "failed=1" in result.stdout
    assert "error=1" in result.stdout
    assert "simplification failed" in result.stderr
    assert "simplified: None" in result.stderr
    assert "simplified_ir: None" in result.stderr


def test_run_simplification_corpus_script_runs_parallel(tmp_path: Path) -> None:
    oracle = write_min_oracle(tmp_path)
    corpus = write_corpus(
        tmp_path,
        [
            base_record(id="case_000000", expr_text="x + 0", expected_text="x"),
            base_record(id="case_000001", expr_text="y + 0", expected_text="y"),
        ],
    )

    result = run_script(
        "--corpus",
        str(corpus),
        "--oracle",
        str(oracle),
        "--limit",
        "2",
        "--jobs",
        "2",
    )

    assert result.returncode == 0
    assert "checked=2" in result.stdout
    assert "passed=2" in result.stdout
    assert "jobs=2" in result.stdout


def test_run_simplification_corpus_script_limit_uses_initial_rows(
    tmp_path: Path,
) -> None:
    oracle = write_min_oracle(tmp_path)
    corpus = write_corpus(
        tmp_path,
        [
            base_record(id="case_000000", expr_text="x + 0", expected_text="x"),
            base_record(id="case_000001", expr_text="x + y", expected_text="x + y"),
        ],
    )

    result = run_script(
        "--corpus",
        str(corpus),
        "--oracle",
        str(oracle),
        "--limit",
        "1",
        "--jobs",
        "1",
    )

    assert result.returncode == 0
    assert "checked=1" in result.stdout
    assert "ground_truth=1" in result.stdout
    assert "case_000001" not in result.stderr
