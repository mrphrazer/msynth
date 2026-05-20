from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_dataset.py"


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        text=True,
        capture_output=True,
    )


def test_preprocess_dataset_script_outputs_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.txt"
    dataset.write_text("# header\nx + 0, x\n~x, x\n", encoding="utf-8")

    result = run_script(str(dataset), "--size", "8")

    assert result.returncode == 0
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(records) == 2
    assert "source_path" not in records[0]
    assert "line_number" not in records[0]
    assert "samples" not in records[0]
    assert "timeout" not in records[0]
    assert "seeds" not in records[0]
    assert "expr" not in records[0]
    assert "expected_expr" not in records[0]
    assert "features" not in records[0]
    assert "tags" not in records[0]
    assert records[0]["id"] == "case_000000"
    assert records[0]["source"] == "dataset.txt"
    assert records[0]["suite"] == "dataset"
    assert records[0]["size"] == 8
    assert records[0]["expr_text"] == "x + 0"
    assert records[0]["expected_text"] == "x"
    assert records[1]["expr_text"] == "~x"
    assert records[1]["expected_text"] == "x"


def test_preprocess_dataset_script_walks_directory(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second_dir = tmp_path / "nested"
    second_dir.mkdir()
    second = second_dir / "b.txt"
    first.write_text("x + 0, x\n", encoding="utf-8")
    second.write_text("y + 0, y\n", encoding="utf-8")

    result = run_script(str(tmp_path), "--size", "8", "--format", "text")

    assert result.returncode == 0
    assert "x + 0\tx" in result.stdout
    assert "y + 0\ty" in result.stdout
    assert "a.txt" not in result.stdout
    assert "b.txt" not in result.stdout


def test_preprocess_dataset_script_can_skip_preprocessing(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.txt"
    dataset.write_text("x + 0, x\n", encoding="utf-8")

    result = run_script(str(dataset), "--size", "8", "--passes", "none")

    assert result.returncode == 0
    record = json.loads(result.stdout)
    assert record["expr_text"] == "x + 0"
    assert record["expected_text"] == "x"


def test_preprocess_dataset_script_writes_gzip_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.txt"
    output = tmp_path / "cases.jsonl.gz"
    dataset.write_text("x + 0, x\n", encoding="utf-8")

    result = run_script(str(dataset), "--size", "8", "--output", str(output))

    assert result.returncode == 0
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        record = json.loads(handle.readline())
    assert record["expr_text"] == "x + 0"
    assert record["expected_text"] == "x"


def test_preprocess_dataset_script_reports_parse_errors(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.txt"
    dataset.write_text("x + 0, x\nf(x), x\n", encoding="utf-8")

    result = run_script(str(dataset), "--size", "8")

    assert result.returncode == 1
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(records) == 1
    assert records[0]["expr_text"] == "x + 0"
    assert "line" not in result.stderr
    assert "dataset.txt" not in result.stderr


def test_preprocess_dataset_script_rejects_missing_expected_marker(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.txt"
    dataset.write_text("x + 0\t-\n", encoding="utf-8")

    result = run_script(str(dataset), "--size", "8")

    assert result.returncode == 1
    assert result.stdout == ""
    assert "missing expected expression" in result.stderr


def test_preprocess_dataset_script_can_use_expression_for_missing_expected(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.txt"
    dataset.write_text("x + 0\t-\n", encoding="utf-8")

    result = run_script(
        str(dataset),
        "--size",
        "8",
        "--passes",
        "none",
        "--missing-expected",
        "self",
    )

    assert result.returncode == 0
    record = json.loads(result.stdout)
    assert record["expr_text"] == "x + 0"
    assert record["expected_text"] == "x + 0"


def test_preprocess_dataset_script_records_relative_source_and_suite(
    tmp_path: Path,
) -> None:
    suite_dir = tmp_path / "simba"
    suite_dir.mkdir()
    dataset = suite_dir / "e1.txt"
    dataset.write_text("x + 0, x\n", encoding="utf-8")

    result = run_script(str(tmp_path), "--size", "8")

    assert result.returncode == 0
    record = json.loads(result.stdout)
    assert record["source"] == "simba/e1.txt"
    assert record["suite"] == "simba"
