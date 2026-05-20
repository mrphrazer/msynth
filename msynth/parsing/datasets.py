from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator

from miasm.expression.expression import Expr

from msynth.parsing.infix import InfixParseError, parse_infix_expr


class DatasetParseError(ValueError):
    """Raised when a dataset row cannot be split or parsed."""


@dataclass(frozen=True)
class ParsedDatasetRow:
    source_path: Path | None
    line_number: int | None
    expression_text: str
    expected_text: str
    expression: Expr
    expected: Expr | None


def parse_dataset_line(
    line: str,
    *,
    size: int = 64,
    source_path: Path | None = None,
    line_number: int | None = None,
    allow_missing_expected: bool = False,
) -> ParsedDatasetRow | None:
    """
    Parse one CoBRA-style dataset line.

    Blank lines and comment lines return ``None``. Data rows must contain either
    ``expression, expected`` or ``expression<TAB>expected``. Extra columns after
    the expected expression are treated as dataset metadata and ignored.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    expression_text, expected_text = _split_dataset_line(stripped)

    try:
        expression = parse_infix_expr(expression_text, size=size)
        if expected_text == "-":
            if allow_missing_expected:
                expected = None
            else:
                prefix = _format_location(source_path, line_number)
                raise DatasetParseError(f"{prefix}missing expected expression")
        else:
            expected = parse_infix_expr(expected_text, size=size)
    except InfixParseError as exc:
        prefix = _format_location(source_path, line_number)
        raise DatasetParseError(f"{prefix}{exc}") from exc

    return ParsedDatasetRow(
        source_path=source_path,
        line_number=line_number,
        expression_text=expression_text,
        expected_text=expected_text,
        expression=expression,
        expected=expected,
    )


def iter_dataset_file(path: Path, *, size: int = 64) -> Iterator[ParsedDatasetRow]:
    """Yield parsed rows from one CoBRA-style dataset text file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = parse_dataset_line(
                line, size=size, source_path=path, line_number=line_number
            )
            if row is not None:
                yield row


def _split_dataset_line(line: str) -> tuple[str, str]:
    if "\t" in line:
        fields = _split_top_level(line, "\t")
    elif "," in line:
        fields = _split_top_level(line, ",")
    else:
        raise DatasetParseError(f"dataset row has no comma or tab separator: {line!r}")
    if len(fields) < 2:
        raise DatasetParseError(f"dataset row has no expected expression: {line!r}")

    expression_text = _clean_dataset_field(fields[0])
    expected_text = _clean_dataset_field(fields[1])
    if not expression_text or not expected_text:
        raise DatasetParseError(f"dataset row has an empty field: {line!r}")
    return expression_text, expected_text


def _clean_dataset_field(field: str) -> str:
    stripped = field.strip()
    constant_match = re.fullmatch(
        r"\(constant\s+([+-]?(?:0x[0-9a-fA-F]+|\d+))\)", stripped
    )
    if constant_match:
        return constant_match.group(1)
    assignment_match = re.fullmatch(r"original\s*=\s*(.+)", stripped)
    if assignment_match:
        stripped = assignment_match.group(1).strip()
    return re.sub(r"\s+\((?:needs Z3|with resets = \d+)\)\s*$", "", stripped)


def _split_top_level(line: str, separator: str) -> list[str]:
    fields: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(line):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == separator and depth == 0:
            fields.append(line[start:index])
            start = index + 1
    fields.append(line[start:])
    return fields


def _format_location(source_path: Path | None, line_number: int | None) -> str:
    if source_path is None and line_number is None:
        return ""
    if source_path is None:
        return f"line {line_number}: "
    if line_number is None:
        return f"{source_path}: "
    return f"{source_path}:{line_number}: "
