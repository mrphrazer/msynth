from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable

from miasm.expression.expression import Expr, ExprId, ExprInt, ExprOp


class InfixParseError(ValueError):
    """Raised when an infix MBA expression cannot be parsed."""


@dataclass(frozen=True)
class _ParseContext:
    text: str
    size: int

    @property
    def mask(self) -> int:
        return (1 << self.size) - 1

    def constant(self, value: int) -> ExprInt:
        return ExprInt(value & self.mask, self.size)

    def error(self, node: ast.AST, message: str) -> InfixParseError:
        location = ""
        lineno = getattr(node, "lineno", None)
        col_offset = getattr(node, "col_offset", None)
        if lineno is not None and col_offset is not None:
            location = f" at line {lineno}, column {col_offset + 1}"
        return InfixParseError(f"{message}{location}: {self.text!r}")


_BIN_OPS: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.Pow: "**",
    ast.FloorDiv: "/",
}


def parse_infix_expr(text: str, *, size: int = 64) -> Expr:
    """
    Parse trusted infix MBA text into a Miasm expression.

    The supported syntax intentionally matches CoBRA/SiMBA-style dataset rows:
    variables, integer constants, parentheses, unary ``~``/``-``/``+``, and the
    arithmetic/bitwise binary operators commonly used in MBA datasets.
    """
    if size <= 0:
        raise ValueError("size must be positive")

    stripped = text.strip()
    if not stripped:
        raise InfixParseError("empty expression")

    try:
        root = ast.parse(stripped, mode="eval")
    except RecursionError as exc:
        raise InfixParseError(f"expression too deeply nested: {stripped!r}") from exc
    except SyntaxError as exc:
        location = ""
        if exc.lineno is not None and exc.offset is not None:
            location = f" at line {exc.lineno}, column {exc.offset}"
        raise InfixParseError(f"syntax error{location}: {stripped!r}") from exc

    context = _ParseContext(stripped, size)
    try:
        return _convert_node(root.body, context)
    except RecursionError as exc:
        raise InfixParseError(f"expression too deeply nested: {stripped!r}") from exc


def _convert_node(node: ast.AST, context: _ParseContext) -> Expr:
    if isinstance(node, ast.Name):
        return ExprId(node.id, context.size)

    if isinstance(node, ast.Subscript):
        return _convert_subscript_id(node, context)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise context.error(node, "only integer constants are supported")
        return context.constant(node.value)

    if isinstance(node, ast.UnaryOp):
        return _convert_unary(node, context)

    if isinstance(node, ast.BinOp):
        return _convert_binary(node, context)

    raise context.error(node, f"unsupported syntax {type(node).__name__}")


def _convert_subscript_id(node: ast.Subscript, context: _ParseContext) -> ExprId:
    if not isinstance(node.value, ast.Name):
        raise context.error(node, "only named subscript variables are supported")
    if not isinstance(node.slice, ast.Constant) or not isinstance(
        node.slice.value, int
    ):
        raise context.error(node, "only integer subscript variables are supported")
    return ExprId(f"{node.value.id}[{node.slice.value}]", context.size)


def _fold_constant(
    expr: Expr, fn: Callable[[int], int], context: _ParseContext
) -> Expr:
    if isinstance(expr, ExprInt):
        return context.constant(fn(int(expr)))
    return (
        ExprOp("-", expr) if fn is _negate else ExprOp("^", expr, context.constant(-1))
    )


def _negate(value: int) -> int:
    return -value


def _invert(value: int) -> int:
    return ~value


def _convert_unary(node: ast.UnaryOp, context: _ParseContext) -> Expr:
    operand = _convert_node(node.operand, context)

    if isinstance(node.op, ast.UAdd):
        return operand
    if isinstance(node.op, ast.USub):
        return _fold_constant(operand, _negate, context)
    if isinstance(node.op, ast.Invert):
        return _fold_constant(operand, _invert, context)

    raise context.error(node, f"unsupported unary operator {type(node.op).__name__}")


def _convert_binary(node: ast.BinOp, context: _ParseContext) -> Expr:
    stack: list[tuple[ast.BinOp, str, ast.AST]] = []
    current: ast.AST = node

    while isinstance(current, ast.BinOp):
        op = _BIN_OPS.get(type(current.op))
        if op is None:
            raise context.error(
                current, f"unsupported binary operator {type(current.op).__name__}"
            )
        stack.append((current, op, current.right))
        current = current.left

    result = _convert_node(current, context)
    for op_node, op, right_node in reversed(stack):
        try:
            right = _convert_node(right_node, context)
        except RecursionError as exc:
            raise context.error(op_node, "expression too deeply nested") from exc
        result = ExprOp(op, result, right)
    return result
