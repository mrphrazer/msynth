from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msynth.parsing import DatasetParseError, ParsedDatasetRow, parse_dataset_line  # noqa: E402
from msynth.simplification.preprocessing import (  # noqa: E402
    AstNormalizationPass,
    Preprocessor,
)


PASS_FACTORIES = {
    "ast": AstNormalizationPass,
}
DEFAULT_PASSES = ("ast",)


def parse_pass_list(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise argparse.ArgumentTypeError("pass list must not be empty")
    if "none" in names:
        if names != ("none",):
            raise argparse.ArgumentTypeError(
                "pass 'none' cannot be combined with other preprocessing passes"
            )
        return names

    unknown = sorted(set(names) - set(PASS_FACTORIES))
    if unknown:
        available = ", ".join(sorted(PASS_FACTORIES))
        raise argparse.ArgumentTypeError(
            f"unknown preprocessing pass {unknown[0]!r}; available: {available}"
        )
    return names


def build_preprocessor(pass_names: tuple[str, ...]) -> Preprocessor:
    if pass_names == ("none",):
        return Preprocessor([])
    return Preprocessor([PASS_FACTORIES[name]() for name in pass_names])


def iter_dataset_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.txt") if path.is_file())
    raise FileNotFoundError(f"input path does not exist: {input_path}")


def iter_dataset_rows(
    path: Path, *, size: int, fail_fast: bool, allow_missing_expected: bool
) -> tuple[list[ParsedDatasetRow], list[str]]:
    rows: list[ParsedDatasetRow] = []
    errors: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = parse_dataset_line(
                    line,
                    size=size,
                    allow_missing_expected=allow_missing_expected,
                )
            except DatasetParseError as exc:
                if fail_fast:
                    raise
                errors.append(sanitize_error(exc))
                continue
            if row is not None:
                rows.append(row)

    return rows, errors


def serialize_row(
    row: ParsedDatasetRow,
    preprocessor: Preprocessor,
    *,
    record_id: str,
    source: str,
    suite: str,
    size: int,
    missing_expected: str,
) -> dict[str, Any]:
    if row.expected is None and missing_expected == "error":
        raise ValueError("missing expected expression")

    preprocessor.run(row.expression)
    expected = row.expression if row.expected is None else row.expected
    preprocessor.run(expected)

    return {
        "id": record_id,
        "source": source,
        "suite": suite,
        "expr_text": row.expression_text,
        "expected_text": row.expression_text
        if row.expected is None
        else row.expected_text,
        "size": size,
    }


def sanitize_error(error: Exception) -> str:
    return re.sub(r" at line \d+, column \d+", "", str(error))


def write_jsonl(records: list[dict[str, Any]], output: TextIO) -> None:
    for record in records:
        output.write(json.dumps(record, separators=(",", ":")))
        output.write("\n")


def write_text(records: list[dict[str, Any]], output: TextIO) -> None:
    for record in records:
        output.write(f"{record['expr_text']}\t{record['expected_text']}\n")


def open_output(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def source_for_path(path: Path, input_path: Path) -> str:
    if input_path.is_dir():
        return path.relative_to(input_path).as_posix()
    return path.name


def suite_for_source(source: str) -> str:
    path = Path(source)
    if len(path.parts) > 1:
        return path.parts[0]
    return path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse CoBRA-style MBA dataset text, validate optional preprocessing, "
            "and serialize compact source-text corpus records."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Dataset .txt file or directory containing dataset .txt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file. Defaults to stdout.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        help="Bit-vector width for parsed variables and constants. Defaults to 64.",
    )
    parser.add_argument(
        "--passes",
        type=parse_pass_list,
        default=DEFAULT_PASSES,
        help=(
            "Comma-separated preprocessing passes. Available: "
            f"{', '.join((*sorted(PASS_FACTORIES), 'none'))}. "
            "Defaults to ast."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "text"),
        default="jsonl",
        help="Output format. Defaults to jsonl.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first malformed dataset row.",
    )
    parser.add_argument(
        "--missing-expected",
        choices=("error", "self"),
        default="error",
        help=(
            "How to handle rows whose expected expression is '-'. "
            "Use 'self' to serialize the source expression as its own expected "
            "equivalent. Defaults to error."
        ),
    )
    parser.add_argument(
        "--id-prefix",
        default="case",
        help="Prefix for generated record ids. Defaults to case.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preprocessor = build_preprocessor(args.passes)

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in iter_dataset_paths(args.input_path):
        source = source_for_path(path, args.input_path)
        suite = suite_for_source(source)
        rows, parse_errors = iter_dataset_rows(
            path,
            size=args.size,
            fail_fast=args.fail_fast,
            allow_missing_expected=args.missing_expected == "self",
        )
        errors.extend(parse_errors)
        for row in rows:
            try:
                record_id = f"{args.id_prefix}_{len(records):06d}"
                records.append(
                    serialize_row(
                        row,
                        preprocessor,
                        record_id=record_id,
                        source=source,
                        suite=suite,
                        size=args.size,
                        missing_expected=args.missing_expected,
                    )
                )
            except Exception as exc:
                if args.fail_fast:
                    raise
                errors.append(f"preprocessing error: {sanitize_error(exc)}")

    if args.output is None:
        output = sys.stdout
        close_output = False
    else:
        output = open_output(args.output)
        close_output = True

    try:
        if args.format == "jsonl":
            write_jsonl(records, output)
        else:
            write_text(records, output)
    finally:
        if close_output:
            output.close()

    for error in errors:
        print(f"error={error}", file=sys.stderr)

    total_ok = len(records)
    total_errors = len(errors)
    print(
        f"processed={total_ok} errors={total_errors} output={args.output or 'stdout'}",
        file=sys.stderr,
    )
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
