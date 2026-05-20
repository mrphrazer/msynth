from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_preprocessing_corpus.py"


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
        "expr_text": "(x & y) + (x | y)",
        "expected_text": "x + y",
        "size": 8,
    }
    record.update(updates)
    return record


def test_run_preprocessing_corpus_script_accepts_matching_case(
    tmp_path: Path,
) -> None:
    corpus = write_corpus(tmp_path, [base_record()])

    result = run_script("--corpus", str(corpus), "--limit", "1", "--jobs", "1")

    assert result.returncode == 0
    assert "checked=1" in result.stdout
    assert "passed=1" in result.stdout
    assert "ground_truth=1" in result.stdout
    assert "passes=ast,simba" in result.stdout
    assert result.stderr == ""


def test_run_preprocessing_corpus_script_reports_mismatch(tmp_path: Path) -> None:
    corpus = write_corpus(
        tmp_path,
        [base_record(expr_text="x + y", expected_text="x")],
    )

    result = run_script("--corpus", str(corpus), "--limit", "1", "--jobs", "1")

    assert result.returncode == 1
    assert "checked=1" in result.stdout
    assert "failed=1" in result.stdout
    assert "case_000000" in result.stderr
    assert "not_shorter" in result.stderr
    assert "nodes:    3 -> 3" in result.stderr
    assert "preprocessed: x + y" in result.stderr
    assert (
        "preprocessed_ir: ExprOp('+', ExprId('x', 8), ExprId('y', 8))" in result.stderr
    )


def test_run_preprocessing_corpus_script_accepts_shorter_case(
    tmp_path: Path,
) -> None:
    corpus = write_corpus(
        tmp_path,
        [base_record(expected_text="x ^ y")],
    )

    result = run_script("--corpus", str(corpus), "--limit", "1", "--jobs", "1")

    assert result.returncode == 0
    assert "checked=1" in result.stdout
    assert "passed=1" in result.stdout
    assert "ground_truth=0" in result.stdout
    assert "shorter=1" in result.stdout
    assert result.stderr == ""


def test_run_preprocessing_corpus_script_reports_parse_error(tmp_path: Path) -> None:
    corpus = write_corpus(
        tmp_path,
        [base_record(expr_text="f(x)", expected_text="x")],
    )

    result = run_script("--corpus", str(corpus), "--limit", "1", "--jobs", "1")

    assert result.returncode == 1
    assert "checked=1" in result.stdout
    assert "failed=1" in result.stdout
    assert "error=1" in result.stdout
    assert "unsupported syntax Call" in result.stderr
    assert "preprocessed: None" in result.stderr
    assert "preprocessed_ir: None" in result.stderr


def test_run_preprocessing_corpus_script_runs_parallel(tmp_path: Path) -> None:
    corpus = write_corpus(
        tmp_path,
        [
            base_record(id="case_000000", expr_text="x + 0", expected_text="x"),
            base_record(id="case_000001", expr_text="y + 0", expected_text="y"),
        ],
    )

    result = run_script("--corpus", str(corpus), "--limit", "2", "--jobs", "2")

    assert result.returncode == 0
    assert "checked=2" in result.stdout
    assert "passed=2" in result.stdout
    assert "jobs=2" in result.stdout


def test_run_preprocessing_corpus_script_limit_uses_initial_rows(
    tmp_path: Path,
) -> None:
    corpus = write_corpus(
        tmp_path,
        [
            base_record(id="case_000000", expr_text="x + 0", expected_text="x"),
            base_record(id="case_000001", expr_text="x ^ 1", expected_text="x"),
        ],
    )

    result = run_script("--corpus", str(corpus), "--limit", "1", "--jobs", "1")

    assert result.returncode == 0
    assert "checked=1" in result.stdout
    assert "ground_truth=1" in result.stdout
    assert "case_000001" not in result.stderr


def test_run_preprocessing_corpus_script_filters_by_suite(tmp_path: Path) -> None:
    corpus = write_corpus(
        tmp_path,
        [
            base_record(id="case_000000", suite="other"),
            base_record(id="case_000001", suite="simba"),
        ],
    )

    result = run_script(
        "--corpus", str(corpus), "--limit", "10", "--jobs", "1", "--suite", "simba"
    )

    assert result.returncode == 0
    assert "checked=1" in result.stdout
    assert "case_000000" not in result.stderr


def test_run_preprocessing_corpus_script_filters_by_source_prefix(
    tmp_path: Path,
) -> None:
    corpus = write_corpus(
        tmp_path,
        [
            base_record(id="case_000000", source="other/example.txt"),
            base_record(id="case_000001", source="simba/example.txt"),
        ],
    )

    result = run_script(
        "--corpus",
        str(corpus),
        "--limit",
        "10",
        "--jobs",
        "1",
        "--source-prefix",
        "simba/",
    )

    assert result.returncode == 0
    assert "checked=1" in result.stdout
    assert "case_000000" not in result.stderr


def test_run_preprocessing_corpus_script_filters_by_exact_source(
    tmp_path: Path,
) -> None:
    corpus = write_corpus(
        tmp_path,
        [
            base_record(id="case_000000", source="simba/other.txt"),
            base_record(id="case_000001", source="simba/example.txt"),
        ],
    )

    result = run_script(
        "--corpus",
        str(corpus),
        "--limit",
        "10",
        "--jobs",
        "1",
        "--source",
        "simba/example.txt",
    )

    assert result.returncode == 0
    assert "checked=1" in result.stdout
    assert "case_000000" not in result.stderr
