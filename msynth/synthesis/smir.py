"""
Search Modulo Inference Rules (Smir) support for msynth synthesis.

Smir is the generic technique introduced in:

    Vidal Attias, Nicolas Bellec, Grégoire Menguy, Sébastien Bardin,
    and Jean-Yves Marion. "Augmenting Search-based Program Synthesis
    with Local Inference Rules to Improve Black-box Deobfuscation."
    CCS 2025.
    https://binsec.github.io/assets/publications/papers/2025-ccs.pdf

The paper's prototype is named XSmir because it implements Smir on top of
Xyntia. This module uses the generic technique name because the implementation
is integrated into msynth's own synthesis engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import List, Sequence

import z3
from miasm.expression.expression import Expr, ExprInt, ExprOp
from miasm.expression.simplifications import expr_simp

from msynth.synthesis.oracle import SynthesisOracle
from msynth.synthesis.scoring import (
    CompiledCache,
    evaluate_expression,
    score_expression,
)


@dataclass(frozen=True)
class SmirRuleResult:
    rule_name: str
    expr: Expr
    score: float


@dataclass(frozen=True)
class SmirEvaluation:
    best_expr: Expr
    best_score: float
    best_rule: str
    aggregate_score: float
    rule_results: Sequence[SmirRuleResult]


class SmirContext:
    """
    Per-evaluation context shared by Smir rules.

    Rules repeatedly score expressions produced from the same base candidate.
    Keeping scoring and evaluation in one context gives all rules the same view
    of the oracle samples and lets them share the compiled-expression cache.
    """

    def __init__(
        self,
        oracle: SynthesisOracle,
        variables: List[Expr],
        compiled_cache: CompiledCache,
    ):
        self.oracle = oracle
        self.variables = variables
        self.compiled_cache = compiled_cache
        self.outputs = [output for _inputs, output in oracle._samples_int]

    def evaluate(self, expr: Expr) -> List[int]:
        return evaluate_expression(
            expr, self.oracle, self.variables, self.compiled_cache
        )

    def score(self, expr: Expr) -> float:
        return score_expression(expr, self.oracle, self.variables, self.compiled_cache)


class SmirRule:
    name = "smir-rule"

    def infer(self, expr: Expr, context: SmirContext) -> Expr:
        raise NotImplementedError


class IdentityRule(SmirRule):
    name = "identity"

    def infer(self, expr: Expr, _context: SmirContext) -> Expr:
        return expr


class AddConstantRule(SmirRule):
    name = "add-constant"

    def infer(self, expr: Expr, context: SmirContext) -> Expr:
        # If the target is e + c, every sample yields the same c = output - e(input).
        # If samples disagree, the best candidate among those sample-derived constants
        # is still useful as a search guide, exactly as Smir's argmin rule prescribes.
        mask = _mask(expr.size)
        values = context.evaluate(expr)
        candidates = {
            (oracle_output - expr_output) & mask
            for expr_output, oracle_output in zip(values, context.outputs)
        }
        return _best_constant_candidate(expr, "+", candidates, context)


class XorConstantRule(SmirRule):
    name = "xor-constant"

    def infer(self, expr: Expr, context: SmirContext) -> Expr:
        # XOR is its own inverse, so output ^ e(input) directly gives the mask
        # whenever the target is e ^ c.
        values = context.evaluate(expr)
        candidates = {
            oracle_output ^ expr_output
            for expr_output, oracle_output in zip(values, context.outputs)
        }
        return _best_constant_candidate(expr, "^", candidates, context)


class MulConstantRule(SmirRule):
    name = "mul-constant"

    def infer(self, expr: Expr, context: SmirContext) -> Expr:
        # Multiplication by an arbitrary bit-vector is not always invertible in
        # BV[width]. Odd values are invertible modulo 2**width, so only samples
        # where e(input) is odd can produce candidate constants. If no such
        # sample exists, the paper's rule falls back to the current expression.
        mod = 1 << expr.size
        values = context.evaluate(expr)
        candidates = set()
        for expr_output, oracle_output in zip(values, context.outputs):
            if expr_output & 1:
                candidates.add((oracle_output * pow(expr_output, -1, mod)) % mod)
        if not candidates:
            return expr
        return _best_constant_candidate(expr, "*", candidates, context)


class AndOrMaskRule(SmirRule):
    name = "and-or-mask"

    def infer(self, expr: Expr, context: SmirContext) -> Expr:
        # This is the paper's (e & c_and) | c_or rule. The masks are inferred
        # bitwise from samples: bits that must stay zero in target outputs are
        # cleared through c_and, while bits that must stay one are introduced
        # through c_or. The rule is intentionally local; a later score check
        # decides whether this mask relation is actually a good candidate.
        mask = _mask(expr.size)
        values = context.evaluate(expr)
        target_or = _bitwise_or(context.outputs)
        target_and = _bitwise_and(context.outputs, mask)
        expr_or = _bitwise_or(values)
        expr_and = _bitwise_and(values, mask)

        c_and = (target_or | (~expr_or & mask)) & mask
        c_or = (target_and & (~expr_and & mask)) & mask
        return _simplify((expr & ExprInt(c_and, expr.size)) | ExprInt(c_or, expr.size))


class ShiftConstantRule(SmirRule):
    def __init__(self, op: str):
        self.op = op
        self.name = f"{op}-constant"

    def infer(self, expr: Expr, context: SmirContext) -> Expr:
        # Constant shifts/rotates have only width possible shift amounts. Smir
        # can therefore search all of them and keep the one minimizing the
        # objective, instead of relying on stochastic mutations to discover it.
        candidates = (
            _simplify(ExprOp(self.op, expr, ExprInt(amount, expr.size)))
            for amount in range(expr.size)
        )
        return min(candidates, key=context.score)


class AffineRule(SmirRule):
    name = "affine"

    def infer(self, expr: Expr, context: SmirContext) -> Expr:
        # Affine inference targets c1 * e + c2. Solving for c1 requires an
        # invertible difference between two sampled e(input) values, which in
        # BV[width] means an odd denominator. If all differences are even, this
        # rule cannot infer a reliable coefficient and falls back to e.
        candidate = _infer_affine(expr, context)
        return expr if candidate is None else candidate


class PolynomialRule(SmirRule):
    name = "polynomial"

    def __init__(self, degree: int = 2, solver_timeout_ms: int = 100):
        self.degree = degree
        self.solver_timeout_ms = solver_timeout_ms

    def infer(self, expr: Expr, context: SmirContext) -> Expr:
        # The paper describes bit-vector Lagrange interpolation with a fallback
        # to the affine rule. For msynth we use Z3 to solve for coefficients over
        # all sampled I/O pairs. This is intentionally bounded by degree and
        # timeout: polynomial inference is powerful, but an unbounded solver call
        # would dominate the stochastic search loop.
        if self.degree < 2:
            candidate = _infer_affine(expr, context)
            return expr if candidate is None else candidate

        size = expr.size
        mod = 1 << size
        values = context.evaluate(expr)
        coefficients = [z3.BitVec(f"smir_c{i}", size) for i in range(self.degree + 1)]

        solver = z3.Solver()
        solver.set("timeout", self.solver_timeout_ms)
        for expr_output, oracle_output in zip(values, context.outputs):
            terms = [
                coefficients[power] * z3.BitVecVal(pow(expr_output, power, mod), size)
                for power in range(self.degree + 1)
            ]
            solver.add(
                sum(terms, z3.BitVecVal(0, size)) == z3.BitVecVal(oracle_output, size)
            )

        if solver.check() != z3.sat:
            candidate = _infer_affine(expr, context)
            return expr if candidate is None else candidate

        model = solver.model()
        coeff_values = [
            model.eval(coefficient, model_completion=True).as_long()
            for coefficient in coefficients
        ]
        return _build_polynomial(expr, coeff_values)


class SmirEngine:
    """
    Applies ordered Smir rules and computes Algorithm 3-style scores.

    The raw local-search candidate e is not scored directly. Instead, every rule
    produces R_i(e, S), and the search accepts mutations based on the product of
    those inferred candidates' scores. This reshapes the local-search surface:
    candidates that are one inferred constant, mask, shift, affine transform, or
    polynomial away from the target become attractive before the exact final
    expression appears in the grammar.
    """

    def __init__(self, rules: Sequence[SmirRule]):
        if not rules:
            raise ValueError("SmirEngine requires at least one inference rule")
        self.rules = list(rules)
        self.compiled_cache: CompiledCache = {}

    def evaluate(
        self, expr: Expr, oracle: SynthesisOracle, variables: List[Expr]
    ) -> SmirEvaluation:
        context = SmirContext(oracle, variables, self.compiled_cache)
        results = []

        for rule in self.rules:
            inferred_expr = rule.infer(expr, context)
            results.append(
                SmirRuleResult(rule.name, inferred_expr, context.score(inferred_expr))
            )

        zero_result = next((result for result in results if result.score == 0.0), None)
        best = (
            zero_result
            if zero_result is not None
            else min(results, key=lambda r: r.score)
        )
        return SmirEvaluation(
            best_expr=best.expr,
            best_score=best.score,
            best_rule=best.rule_name,
            aggregate_score=_aggregate_scores(result.score for result in results),
            rule_results=tuple(results),
        )


def default_smir_rules(polynomial_degree: int = 2) -> List[SmirRule]:
    return [
        IdentityRule(),
        AddConstantRule(),
        XorConstantRule(),
        MulConstantRule(),
        AndOrMaskRule(),
        ShiftConstantRule("<<"),
        ShiftConstantRule(">>"),
        ShiftConstantRule("<<<"),
        ShiftConstantRule(">>>"),
        AffineRule(),
        PolynomialRule(degree=polynomial_degree),
    ]


def _infer_affine(expr: Expr, context: SmirContext) -> Expr | None:
    values = context.evaluate(expr)
    mask = _mask(expr.size)
    mod = 1 << expr.size
    candidates = set()

    # The paper presents the two-point formula directly. We try all invertible
    # pairs from the sample set because different noisy/non-affine pairs can
    # suggest different coefficients; scoring selects the best local guide.
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            denominator = (values[left] - values[right]) & mask
            if denominator & 1 == 0:
                continue

            numerator = (context.outputs[left] - context.outputs[right]) & mask
            c1 = (numerator * pow(denominator, -1, mod)) % mod
            c2 = (context.outputs[right] - c1 * values[right]) & mask
            candidates.add((c1, c2))

    if not candidates:
        return None

    return min(
        (_build_affine(expr, c1, c2) for c1, c2 in candidates), key=context.score
    )


def _build_affine(expr: Expr, c1: int, c2: int) -> Expr:
    size = expr.size
    return _simplify((ExprInt(c1, size) * expr) + ExprInt(c2, size))


def _build_polynomial(expr: Expr, coefficients: Sequence[int]) -> Expr:
    size = expr.size
    terms = []
    power_expr: Expr = ExprInt(1, size)

    for power, coefficient in enumerate(coefficients):
        if power == 1:
            power_expr = expr
        elif power > 1:
            power_expr = _simplify(power_expr * expr)

        coefficient &= _mask(size)
        if coefficient == 0:
            continue
        if power == 0:
            terms.append(ExprInt(coefficient, size))
        elif coefficient == 1:
            terms.append(power_expr)
        else:
            terms.append(ExprInt(coefficient, size) * power_expr)

    if not terms:
        return ExprInt(0, size)

    result = terms[0]
    for term in terms[1:]:
        result = result + term
    return _simplify(result)


def _best_constant_candidate(
    expr: Expr, op: str, constants: set[int], context: SmirContext
) -> Expr:
    candidates = (
        _simplify(ExprOp(op, expr, ExprInt(value & _mask(expr.size), expr.size)))
        for value in constants
    )
    return min(candidates, key=context.score)


def _aggregate_scores(scores) -> float:
    scores = list(scores)
    if any(score == 0.0 for score in scores):
        return 0.0
    # Algorithm 3 multiplies rule scores so that a candidate close under several
    # independent rules is preferred. Scores are non-negative, so normal float
    # overflow simply behaves as a very poor aggregate candidate.
    return prod(scores)


def _bitwise_or(values: Sequence[int]) -> int:
    result = 0
    for value in values:
        result |= value
    return result


def _bitwise_and(values: Sequence[int], mask: int) -> int:
    result = mask
    for value in values:
        result &= value
    return result


def _mask(size: int) -> int:
    return (1 << size) - 1


def _simplify(expr: Expr) -> Expr:
    return expr_simp.expr_simp(expr)
