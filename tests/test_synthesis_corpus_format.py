from __future__ import annotations

import json
from pathlib import Path


CORPUS = (
    Path(__file__).resolve().parents[1] / "datasets" / "corpora" / "synthesis.jsonl.gz"
)


def test_synthesis_corpus_rows_use_shared_dataset_schema() -> None:
    required = {"id", "expr", "expected_expr", "expr_text", "expected_text"}
    execution_fields = {"samples", "timeout", "seeds"}

    import gzip

    with gzip.open(CORPUS, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            assert required <= set(record)
            assert execution_fields.isdisjoint(record)
