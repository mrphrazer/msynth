from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_synthesis_corpus.py"
CORPUS = (
    Path(__file__).resolve().parents[1] / "datasets" / "corpora" / "synthesis.jsonl.gz"
)


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        text=True,
        capture_output=True,
    )


def test_run_synthesis_corpus_prints_concise_success_summary(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"

    result = run_script(
        str(CORPUS),
        "--case-id",
        "var_x_u8",
        "--samples",
        "4",
        "--timeout",
        "0.2",
        "--seed",
        "0",
        "--jobs",
        "1",
        "--output",
        str(output),
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.startswith("checked=1 passed=1 failed=0 ")
    assert "solved=1" in result.stdout
    assert "wrote" not in result.stdout

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["passed"] == 1
    assert report["summary"]["failed"] == 0


def test_run_synthesis_corpus_reports_failures_to_stderr(tmp_path: Path) -> None:
    corpus = tmp_path / "bad.jsonl"
    output = tmp_path / "report.json"
    corpus.write_text(
        json.dumps({"id": "bad_case", "expr": "not valid python"}) + "\n",
        encoding="utf-8",
    )

    result = run_script(
        str(corpus),
        "--jobs",
        "1",
        "--output",
        str(output),
    )

    assert result.returncode == 1
    assert "checked=1 passed=0 failed=1" in result.stdout
    assert "error=1" in result.stdout
    assert "bad_case [error]" in result.stderr
    assert "parse error" in result.stderr

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["passed"] == 0
    assert report["summary"]["failed"] == 1
