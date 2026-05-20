from __future__ import annotations

from pathlib import Path

import pytest
from miasm.expression.expression import ExprId, ExprInt, ExprOp

from msynth.parsing import DatasetParseError, iter_dataset_file, parse_dataset_line


def test_parse_dataset_line_skips_comments_and_blank_lines() -> None:
    assert parse_dataset_line("# x + y") is None
    assert parse_dataset_line("   ") is None


def test_parse_comma_dataset_line() -> None:
    row = parse_dataset_line("x + y, x ^ y", size=8)

    assert row is not None
    assert row.expression_text == "x + y"
    assert row.expected_text == "x ^ y"
    assert row.expression == ExprOp("+", ExprId("x", 8), ExprId("y", 8))
    assert row.expected == ExprOp("^", ExprId("x", 8), ExprId("y", 8))


def test_parse_tab_dataset_line() -> None:
    row = parse_dataset_line("x+y\t(x+y)", size=8)

    assert row is not None
    assert row.expression == ExprOp("+", ExprId("x", 8), ExprId("y", 8))
    assert row.expected == ExprOp("+", ExprId("x", 8), ExprId("y", 8))


def test_parse_dataset_line_ignores_extra_metadata_columns() -> None:
    row = parse_dataset_line("x + y, x, y", size=8)

    assert row is not None
    assert row.expression_text == "x + y"
    assert row.expected_text == "x"


def test_parse_dataset_line_strips_known_trailing_notes() -> None:
    row = parse_dataset_line(
        "x + y (with resets = 10)\t-", size=8, allow_missing_expected=True
    )

    assert row is not None
    assert row.expression_text == "x + y"
    assert row.expression == ExprOp("+", ExprId("x", 8), ExprId("y", 8))


def test_parse_dataset_line_keeps_trailing_parenthesized_expression() -> None:
    row = parse_dataset_line("x + (y), x", size=8)

    assert row is not None
    assert row.expression_text == "x + (y)"
    assert row.expression == ExprOp("+", ExprId("x", 8), ExprId("y", 8))


def test_parse_dataset_line_normalizes_constant_ground_truth_marker() -> None:
    row = parse_dataset_line("x + 0\t(constant 123)", size=8)

    assert row is not None
    assert row.expected_text == "123"
    assert row.expected == ExprInt(123, 8)


def test_parse_dataset_line_strips_original_assignment_label() -> None:
    row = parse_dataset_line("original = x + y\t-", size=8, allow_missing_expected=True)

    assert row is not None
    assert row.expression_text == "x + y"
    assert row.expression == ExprOp("+", ExprId("x", 8), ExprId("y", 8))


def test_iter_dataset_file_preserves_source_metadata(tmp_path: Path) -> None:
    path = tmp_path / "dataset.txt"
    path.write_text("# header\nx+y, x+y\n~x, x\n", encoding="utf-8")

    rows = list(iter_dataset_file(path, size=8))

    assert [row.line_number for row in rows] == [2, 3]
    assert [row.source_path for row in rows] == [path, path]


def test_dataset_parse_error_for_missing_separator() -> None:
    with pytest.raises(DatasetParseError):
        parse_dataset_line("x + y")


def test_parse_dataset_line_allows_missing_expected_marker() -> None:
    row = parse_dataset_line("x + 0\t-", size=8, allow_missing_expected=True)

    assert row is not None
    assert row.expected is None


def test_dataset_parse_error_includes_location(tmp_path: Path) -> None:
    path = tmp_path / "dataset.txt"

    with pytest.raises(DatasetParseError, match=r"dataset\.txt:7"):
        parse_dataset_line("f(x), x", source_path=path, line_number=7)
