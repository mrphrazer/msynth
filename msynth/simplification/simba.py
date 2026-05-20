"""
SiMBA preprocessing pass for linear mixed Boolean-arithmetic expressions.

This is a Miasm-native reimplementation of the algorithm described in:
Benjamin Reichenwallner and Peter Meerwald-Stadler,
"Efficient Deobfuscation of Linear Mixed Boolean-Arithmetic Expressions",
CheckMATE 2022, DOI 10.1145/3560831.3564256, arXiv:2209.06335.

Reference implementation: https://github.com/DenuvoSoftwareSolutions/SiMBA
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations

from miasm.expression.expression import Expr, ExprId, ExprInt, ExprOp

from msynth.utils.expr_utils import get_unique_variables


class _ExpressionKind(Enum):
    ARITHMETIC = "arithmetic"
    BITWISE = "bitwise"
    MIXED = "mixed"


@dataclass(frozen=True)
class SimbaPass:
    """Simplify supported linear MBAs before oracle-backed simplification."""

    name: str = "simba"

    def run(self, expr: Expr) -> Expr:
        simplifier = _SimbaSimplifier(expr)
        return simplifier.simplify()


class _SimbaSimplifier:
    def __init__(self, expr: Expr):
        self.expr = expr
        self.size = expr.size
        self.modulus = 1 << self.size
        self.mask = self.modulus - 1
        self.variables = get_unique_variables(expr)

    def simplify(self) -> Expr:
        if self.size <= 0:
            return self.expr
        if self._classify(self.expr) is None:
            return self.expr

        try:
            signature = self._signature(self.expr, self.variables)
            simplified = self._simplify_signature(signature, self.variables)
        except (KeyError, ValueError, OverflowError):
            return self.expr

        if self._effective_variable_count(simplified) <= 3 and len(self.variables) > 3:
            simplified = self._simplify_fewer_variables(simplified)

        return simplified

    def _classify(self, expr: Expr) -> _ExpressionKind | None:
        if expr.size != self.size:
            return None
        if isinstance(expr, ExprInt):
            return _ExpressionKind.ARITHMETIC
        if isinstance(expr, ExprId):
            return _ExpressionKind.BITWISE
        if not isinstance(expr, ExprOp):
            return None

        kinds = [self._classify(arg) for arg in expr.args]
        if any(kind is None for kind in kinds):
            return None

        if expr.op == "-" and len(expr.args) == 1:
            return (
                _ExpressionKind.ARITHMETIC
                if kinds[0] is _ExpressionKind.ARITHMETIC
                else _ExpressionKind.MIXED
            )

        if expr.op in {"+", "-"} and len(expr.args) >= 2:
            return (
                _ExpressionKind.ARITHMETIC
                if all(kind is _ExpressionKind.ARITHMETIC for kind in kinds)
                else _ExpressionKind.MIXED
            )

        if expr.op == "*" and len(expr.args) >= 2:
            non_arithmetic = sum(
                kind is not _ExpressionKind.ARITHMETIC for kind in kinds
            )
            if non_arithmetic > 1:
                return None
            return (
                _ExpressionKind.ARITHMETIC
                if non_arithmetic == 0
                else _ExpressionKind.MIXED
            )

        if expr.op in {"&", "|"} and len(expr.args) >= 2:
            if all(kind is _ExpressionKind.ARITHMETIC for kind in kinds):
                return _ExpressionKind.ARITHMETIC
            if all(kind is _ExpressionKind.BITWISE for kind in kinds):
                return _ExpressionKind.BITWISE
            return None

        if expr.op == "^" and len(expr.args) >= 2:
            bitwise_args = []
            for arg, kind in zip(expr.args, kinds):
                if kind is _ExpressionKind.BITWISE:
                    bitwise_args.append(arg)
                elif not self._is_all_ones(arg):
                    if any(k is _ExpressionKind.BITWISE for k in kinds):
                        return None
            if bitwise_args:
                return _ExpressionKind.BITWISE
            return _ExpressionKind.ARITHMETIC

        return None

    def _is_all_ones(self, expr: Expr) -> bool:
        return isinstance(expr, ExprInt) and int(expr) == self.mask

    def _signature(self, expr: Expr, variables: list[Expr]) -> list[int]:
        values = []
        for assignment in range(1 << len(variables)):
            env = {
                variable: (assignment >> index) & 1
                for index, variable in enumerate(variables)
            }
            values.append(self._evaluate(expr, env))
        return values

    def _evaluate(self, expr: Expr, env: dict[Expr, int]) -> int:
        if isinstance(expr, ExprInt):
            return int(expr) & self.mask
        if isinstance(expr, ExprId):
            return env[expr] & self.mask
        if not isinstance(expr, ExprOp):
            raise ValueError(f"unsupported expression {type(expr).__name__}")

        args = [self._evaluate(arg, env) for arg in expr.args]
        if expr.op == "-" and len(args) == 1:
            return (-args[0]) & self.mask
        if expr.op == "+":
            return sum(args) & self.mask
        if expr.op == "-" and len(args) >= 2:
            result = args[0]
            for arg in args[1:]:
                result -= arg
            return result & self.mask
        if expr.op == "*":
            result = 1
            for arg in args:
                result *= arg
            return result & self.mask
        if expr.op == "&":
            result = self.mask
            for arg in args:
                result &= arg
            return result & self.mask
        if expr.op == "|":
            result = 0
            for arg in args:
                result |= arg
            return result & self.mask
        if expr.op == "^":
            result = 0
            for arg in args:
                result ^= arg
            return result & self.mask

        raise ValueError(f"unsupported operation {expr.op!r}")

    def _simplify_signature(self, signature: list[int], variables: list[Expr]) -> Expr:
        if len(set(signature)) == 1:
            return self._const(signature[0])

        generic = self._generic_linear_combination(signature, variables)
        if len(variables) <= 3:
            return self._refine(signature, variables, generic)
        return generic

    def _generic_linear_combination(
        self, signature: list[int], variables: list[Expr]
    ) -> Expr:
        residual = [value & self.mask for value in signature]
        terms: list[Expr] = []

        constant = residual[0] & self.mask
        if constant:
            terms.append(self._const(constant))
        for index in range(1, len(residual)):
            residual[index] = (residual[index] - constant) & self.mask

        for degree in range(1, len(variables) + 1):
            for variable_indexes in combinations(range(len(variables)), degree):
                row_index = sum(1 << index for index in variable_indexes)
                coefficient = residual[row_index] & self.mask
                if coefficient == 0:
                    continue

                conjunction = self._conjunction(
                    [variables[index] for index in variable_indexes]
                )
                terms.append(self._multiply(coefficient, conjunction))

                for assignment in range(len(residual)):
                    if assignment == row_index:
                        continue
                    if all((assignment >> index) & 1 for index in variable_indexes):
                        residual[assignment] = (
                            residual[assignment] - coefficient
                        ) & self.mask

        return self._sum(terms)

    def _refine(
        self, signature: list[int], variables: list[Expr], generic: Expr
    ) -> Expr:
        term_count = self._term_count(generic)
        if term_count <= 1:
            return generic

        result_values = set(signature)
        if len(result_values) == 2:
            if signature[0] == 0:
                return self._expression_for_each_unique_value(signature, variables)

            negated = self._try_find_negated_single_expression(signature, variables)
            if negated is not None:
                return negated

        if term_count <= 2:
            return generic

        constant = signature[0] & self.mask
        shifted = [((value - constant) & self.mask) for value in signature]
        shifted_values = set(shifted)

        if len(shifted_values) == 2:
            return self._sum(
                [
                    self._const(constant),
                    self._expression_for_each_unique_value(shifted, variables),
                ]
            )

        if len(shifted_values) == 3 and constant == 0:
            return self._expression_for_each_unique_value(shifted, variables)

        unique_nonzero = sorted(value for value in shifted_values if value != 0)
        if len(shifted_values) == 4 and constant == 0:
            eliminated = self._try_eliminate_unique_value(
                unique_nonzero, shifted, variables
            )
            if eliminated is not None:
                return eliminated

        if term_count == 3:
            return generic

        if constant == 0:
            return self._expression_for_each_unique_value(shifted, variables)

        eliminated = self._try_eliminate_unique_value(
            unique_nonzero, shifted, variables
        )
        if eliminated is not None:
            return self._sum([self._const(constant), eliminated])

        return generic

    def _try_find_negated_single_expression(
        self, signature: list[int], variables: list[Expr]
    ) -> Expr | None:
        values = list(set(signature))
        if len(values) != 2:
            return None

        first, second = values
        if self._is_double_modulo(first, second):
            low, high = second, first
        elif self._is_double_modulo(second, first):
            low, high = first, second
        else:
            return None

        if signature[0] == high:
            return None

        predicate = [int(value == high) for value in signature]
        bitwise = self._lookup_bitwise_expression(predicate, variables)
        if bitwise is None:
            return None
        return self._multiply((-low) & self.mask, self._invert(bitwise))

    def _try_eliminate_unique_value(
        self, values: list[int], signature: list[int], variables: list[Expr]
    ) -> Expr | None:
        if len(values) > 4:
            return None

        for i, first in enumerate(values[:-1]):
            for j, second in enumerate(values[i + 1 :], start=i + 1):
                for k, combined in enumerate(values):
                    if k in {i, j}:
                        continue
                    if not self._is_sum_modulo(first, second, combined):
                        continue

                    terms = [
                        self._term_for_value(signature, variables, first, combined),
                        self._term_for_value(signature, variables, second, combined),
                    ]
                    for value in values:
                        if value not in {first, second, combined}:
                            terms.append(
                                self._term_for_value(signature, variables, value)
                            )
                    return self._sum(terms)

        if len(values) < 4:
            return None

        total = sum(values) & self.mask
        for index, value in enumerate(values):
            if (2 * value) & self.mask != total:
                continue
            terms = [
                self._term_for_value(signature, variables, other)
                for other_index, other in enumerate(values)
                if other_index != index
            ]
            return self._sum(terms)

        return None

    def _expression_for_each_unique_value(
        self, signature: list[int], variables: list[Expr]
    ) -> Expr:
        terms = [
            self._term_for_value(signature, variables, value)
            for value in sorted(set(signature))
            if value != 0
        ]
        return self._sum(terms)

    def _term_for_value(
        self,
        signature: list[int],
        variables: list[Expr],
        value: int,
        alternate_value: int | None = None,
    ) -> Expr:
        predicate = [
            int(
                current == value
                or (alternate_value is not None and current == alternate_value)
            )
            for current in signature
        ]
        bitwise = self._lookup_bitwise_expression(predicate, variables)
        if bitwise is None:
            raise ValueError("could not build bitwise expression")
        return self._multiply(value, bitwise)

    def _lookup_bitwise_expression(
        self, predicate: list[int], variables: list[Expr]
    ) -> Expr | None:
        table = self._table_to_int(predicate)
        variable_tables = [
            self._table_to_int(
                [(assignment >> index) & 1 for assignment in range(1 << len(variables))]
            )
            for index in range(len(variables))
        ]

        if table == 0:
            return self._const(0)

        for index, variable_table in enumerate(variable_tables):
            if table == variable_table:
                return variables[index]

        for degree in range(2, len(variables) + 1):
            for indexes in combinations(range(len(variables)), degree):
                xor_table = 0
                and_table = (1 << (1 << len(variables))) - 1
                or_table = 0
                for index in indexes:
                    xor_table ^= variable_tables[index]
                    and_table &= variable_tables[index]
                    or_table |= variable_tables[index]
                selected = [variables[index] for index in indexes]
                if table == xor_table:
                    return self._xor(selected)
                if table == and_table:
                    return self._conjunction(selected)
                if table == or_table:
                    return self._or(selected)

        return self._dnf_expression(predicate, variables)

    def _dnf_expression(
        self, predicate: list[int], variables: list[Expr]
    ) -> Expr | None:
        terms = []
        for assignment, enabled in enumerate(predicate):
            if not enabled:
                continue
            if assignment == 0:
                return None
            literals = [
                variables[index]
                if (assignment >> index) & 1
                else self._invert(variables[index])
                for index in range(len(variables))
            ]
            terms.append(self._conjunction(literals))
        return self._or(terms) if terms else self._const(0)

    def _simplify_fewer_variables(self, expr: Expr) -> Expr:
        occurring = get_unique_variables(expr)
        if len(occurring) > 3:
            return expr
        inner = _SimbaSimplifier(expr)
        return inner.simplify()

    def _effective_variable_count(self, expr: Expr) -> int:
        return len(get_unique_variables(expr))

    def _table_to_int(self, values: list[int]) -> int:
        result = 0
        for index, value in enumerate(values):
            if value:
                result |= 1 << index
        return result

    def _is_sum_modulo(self, first: int, second: int, result: int) -> bool:
        return (first + second - result) % self.modulus == 0

    def _is_double_modulo(self, result: int, value: int) -> bool:
        return (2 * value - result) % self.modulus == 0

    def _term_count(self, expr: Expr) -> int:
        if isinstance(expr, ExprOp) and expr.op == "+":
            return sum(self._term_count(arg) for arg in expr.args)
        return 1

    def _const(self, value: int) -> ExprInt:
        return ExprInt(value & self.mask, self.size)

    def _multiply(self, coefficient: int, expr: Expr) -> Expr:
        coefficient &= self.mask
        if coefficient == 0:
            return self._const(0)
        if coefficient == 1:
            return expr
        if isinstance(expr, ExprInt):
            return self._const(coefficient * int(expr))
        return ExprOp("*", self._const(coefficient), expr)

    def _sum(self, terms: list[Expr]) -> Expr:
        filtered = [
            term for term in terms if not (isinstance(term, ExprInt) and int(term) == 0)
        ]
        if not filtered:
            return self._const(0)
        if len(filtered) == 1:
            return filtered[0]
        return ExprOp("+", *filtered)

    def _conjunction(self, terms: list[Expr]) -> Expr:
        if not terms:
            return self._const(self.mask)
        if len(terms) == 1:
            return terms[0]
        return ExprOp("&", *terms)

    def _or(self, terms: list[Expr]) -> Expr:
        if not terms:
            return self._const(0)
        if len(terms) == 1:
            return terms[0]
        return ExprOp("|", *terms)

    def _xor(self, terms: list[Expr]) -> Expr:
        if not terms:
            return self._const(0)
        if len(terms) == 1:
            return terms[0]
        return ExprOp("^", *terms)

    def _invert(self, expr: Expr) -> Expr:
        if isinstance(expr, ExprInt):
            return self._const(~int(expr))
        return ExprOp("^", expr, self._const(self.mask))
