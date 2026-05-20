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
    """Coarse expression classes used by the linear-MBA validator."""

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

        # SiMBA is only sound for linear MBAs. This pass is deliberately
        # conservative: if the root expression is not in the supported linear
        # fragment, preprocessing is a no-op and later simplification stages can
        # still attempt their normal oracle-based handling.
        if self._classify(self.expr) is None:
            return self.expr

        try:
            # The core theorem in the paper says a linear MBA is determined by
            # its values on all Boolean assignments. We evaluate the whole Miasm
            # expression directly on those assignments instead of decomposing it
            # into coefficient/bitwise-expression terms first.
            signature = self._signature(self.expr, self.variables)
            simplified = self._simplify_signature(signature, self.variables)
        except (KeyError, ValueError, OverflowError):
            # Any unsupported Miasm form, missing variable binding, or arithmetic
            # edge case should make the pass transparent. Preprocessing must not
            # turn a valid expression into an exception for the simplifier.
            return self.expr

        if self._effective_variable_count(simplified) <= 3 and len(self.variables) > 3:
            # Generic reconstruction can eliminate variables. If that leaves a
            # small-variable expression, rerun SiMBA so the lookup/refinement path
            # for one, two, or three variables can produce a more compact result.
            simplified = self._simplify_fewer_variables(simplified)

        return simplified

    def _classify(self, expr: Expr) -> _ExpressionKind | None:
        """
        Return the linear-MBA kind of ``expr`` or None if unsupported.

        The checker mirrors upstream SiMBA's parser-level linearity rules, but
        applies them to Miasm nodes. Arithmetic operations may combine mixed
        terms linearly. Multiplication is allowed only when at most one operand
        is non-arithmetic, which models "constant times bitwise expression".
        Bitwise operators are allowed only over purely bitwise operands, except
        for XOR with an all-ones constant, which is how msynth's infix parser
        represents bitwise NOT.
        """
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
            # Linear MBA terms can be multiplied by arithmetic constants, but a
            # product of two variable-dependent bitwise/mixed expressions would
            # be polynomial/nonlinear and is outside this pass.
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
            # ``~x`` is represented by the parser as ``x ^ all_ones``. Treat that
            # as a bitwise unary operation while still rejecting arbitrary mixed
            # arithmetic XORs such as ``(x + y) ^ z``.
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
        """
        Evaluate ``expr`` on the Boolean cube for the sorted variable list.

        Assignment index bits encode variable values, matching the ordering used
        in the SiMBA paper and upstream code:
        0 -> (0, 0, ...), 1 -> (1, 0, ...), 2 -> (0, 1, ...), etc.
        """
        values = []
        for assignment in range(1 << len(variables)):
            env = {
                variable: (assignment >> index) & 1
                for index, variable in enumerate(variables)
            }
            values.append(self._evaluate(expr, env))
        return values

    def _evaluate(self, expr: Expr, env: dict[Expr, int]) -> int:
        """Evaluate the supported linear-MBA fragment under one Boolean assignment."""
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

        # First build the always-valid conjunction-basis representation. For up
        # to three variables, try the paper's lookup/refinement rules afterward
        # because those often recover compact bitwise forms such as x ^ y.
        generic = self._generic_linear_combination(signature, variables)
        if len(variables) <= 3:
            return self._refine(signature, variables, generic)
        return generic

    def _generic_linear_combination(
        self, signature: list[int], variables: list[Expr]
    ) -> Expr:
        """
        Reconstruct a linear MBA in the conjunction basis.

        The basis is ``1``, every single variable, every pairwise conjunction,
        and so on. Because the Boolean assignment order makes each conjunction's
        first nonzero row unique, coefficients can be read from the residual
        vector without general-purpose linear algebra.
        """
        residual = [value & self.mask for value in signature]
        terms: list[Expr] = []

        # Row zero is the all-zero assignment, so it is the constant term. Remove
        # it from all other rows before solving the remaining conjunction terms.
        constant = residual[0] & self.mask
        if constant:
            terms.append(self._const(constant))
        for index in range(1, len(residual)):
            residual[index] = (residual[index] - constant) & self.mask

        for degree in range(1, len(variables) + 1):
            for variable_indexes in combinations(range(len(variables)), degree):
                # The first row where x_i1 & ... & x_ik is true has exactly
                # those assignment bits set. After smaller-degree terms have
                # been subtracted, that row contains this conjunction's
                # coefficient.
                row_index = sum(1 << index for index in variable_indexes)
                coefficient = residual[row_index] & self.mask
                if coefficient == 0:
                    continue

                conjunction = self._conjunction(
                    [variables[index] for index in variable_indexes]
                )
                terms.append(self._multiply(coefficient, conjunction))

                # Subtract the found coefficient from every row where this
                # conjunction is true, so later higher-degree conjunctions see
                # only their still-unexplained residual.
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
        """
        Try small-variable refinements from the SiMBA paper.

        The conjunction-basis expression is always valid but not always compact.
        For at most three variables, truth tables are small enough to map simple
        Boolean predicates back to bitwise expressions and combine them with the
        observed output coefficients.
        """
        term_count = self._term_count(generic)
        if term_count <= 1:
            return generic

        result_values = set(signature)
        if len(result_values) == 2:
            if signature[0] == 0:
                # One nonzero region: turn that region into a bitwise predicate
                # and multiply it by the nonzero output value.
                return self._expression_for_each_unique_value(signature, variables)

            # Two nonzero values can sometimes be represented as a single
            # negated predicate using ~p == -p - 1 in modular arithmetic.
            negated = self._try_find_negated_single_expression(signature, variables)
            if negated is not None:
                return negated

        if term_count <= 2:
            return generic

        constant = signature[0] & self.mask
        # Many refinement cases are easier after peeling off the constant row.
        # This transforms "constant plus one predicate" into the same shape as
        # the zero-constant cases above.
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
            # If one observed value is the modular sum of two others, we can
            # merge predicate regions and use fewer bitwise terms.
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
        """
        Detect the two-value pattern that can be represented as coeff * ~p.

        If the values are ``a`` and ``2a`` modulo 2^n and the all-zero row is
        ``a``, then rows with ``2a`` form predicate ``p`` and the expression can
        be written as ``(-a) * ~p``.
        """
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
        """
        Reduce term count by merging regions whose coefficients add modulo 2^n.

        This implements the paper's small truth-table refinement: if value c is
        the modular sum of values a and b, regions carrying c can be included in
        both the a-predicate and b-predicate instead of requiring a third term.
        """
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
        """Build one coefficient * predicate term for each nonzero signature value."""
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
        # A predicate row is true wherever the signature equals ``value``. When
        # ``alternate_value`` is supplied, those rows are included too; this is
        # how value-elimination shares one region between two coefficient terms.
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
        """
        Convert a Boolean truth table to a compact bitwise expression.

        Upstream SiMBA ships lookup tables for up to three variables. To avoid a
        bundled table, this implementation recognizes the common compact forms
        directly and falls back to a DNF expression for remaining predicates.
        """
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
                # Check simple n-ary XOR/AND/OR over every variable subset before
                # falling back to DNF. These are the forms that make examples
                # like x ^ y and x | y stay compact.
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
        """
        Build a disjunctive-normal-form predicate.

        DNF is only a fallback for small truth tables. If the predicate includes
        the all-zero row, building that minterm would require a constant true
        expression; returning None lets the caller abandon that refinement.
        """
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
        """Rerun SiMBA after generic reconstruction removes unused variables."""
        occurring = get_unique_variables(expr)
        if len(occurring) > 3:
            return expr
        inner = _SimbaSimplifier(expr)
        return inner.simplify()

    def _effective_variable_count(self, expr: Expr) -> int:
        return len(get_unique_variables(expr))

    def _table_to_int(self, values: list[int]) -> int:
        """Pack a truth table into an integer so tables can be compared cheaply."""
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
        """Build coefficient * expr while preserving SiMBA's modular constants."""
        coefficient &= self.mask
        if coefficient == 0:
            return self._const(0)
        if coefficient == 1:
            return expr
        if isinstance(expr, ExprInt):
            return self._const(coefficient * int(expr))
        return ExprOp("*", self._const(coefficient), expr)

    def _sum(self, terms: list[Expr]) -> Expr:
        """Build a variadic sum, dropping explicit zero terms."""
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
