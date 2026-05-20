from __future__ import annotations

import gzip
import json
from pathlib import Path


CORPUS = Path(__file__).resolve().parents[1] / "datasets" / "corpora" / "cobra.jsonl.gz"


def test_cobra_corpus_rows_use_compact_source_schema() -> None:
    required = {"id", "source", "suite", "expr_text", "expected_text", "size"}
    omitted = {
        "expr",
        "expected_expr",
        "tags",
        "features",
        "samples",
        "timeout",
        "seeds",
    }
    count = 0

    with gzip.open(CORPUS, "rt", encoding="utf-8") as handle:
        for count, line in enumerate(handle, start=1):
            record = json.loads(line)
            assert required <= set(record)
            assert omitted.isdisjoint(record)

    assert count == 76559
