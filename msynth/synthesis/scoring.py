from __future__ import annotations

from math import log2
from typing import Callable, Dict, List, Tuple

from miasm.expression.expression import Expr, ExprInt
from miasm.expression.simplifications import expr_simp

from msynth.synthesis.oracle import SynthesisOracle
from msynth.utils.expr_utils import compile_expr_to_python


CompiledCache = Dict[str, Callable[[List[int]], int] | None]


def score_expression(
    expr: Expr,
    oracle: SynthesisOracle,
    variables: List[Expr],
    compiled_cache: CompiledCache | None = None,
) -> float:
    """
    Score an expression against a synthesis oracle.

    This is the synthesis objective used by both the legacy local-search states
    and Smir-inferred candidates:

        sum(log2(1 + abs(candidate_output - oracle_output)))

    A score of 0.0 means that the expression matches every sampled I/O pair.
    The compiled fast path mirrors the previous SynthesisState implementation;
    unsupported expression forms fall back to Miasm expression rewriting and
    simplification.
    """
    compiled = _get_compiled_evaluator(expr, compiled_cache)
    samples_int = _get_int_samples(oracle)

    if compiled is not None:
        evaluator = _build_compiled_input_adapter(compiled, variables)
        return sum(
            log2(1 + abs(evaluator(inputs) - oracle_output))
            for inputs, oracle_output in samples_int
        )

    return sum(
        log2(1 + abs(int(_evaluate_tree(expr, inputs, variables)) - int(oracle_output)))
        for inputs, oracle_output in oracle.synthesis_map.items()
    )


def evaluate_expression(
    expr: Expr,
    oracle: SynthesisOracle,
    variables: List[Expr],
    compiled_cache: CompiledCache | None = None,
) -> List[int]:
    """
    Evaluate an expression over every sampled input vector in an oracle.

    Smir rules need the current candidate's sampled outputs to infer constants,
    masks, and affine/polynomial coefficients. This helper keeps that evaluation
    behavior aligned with scoring.
    """
    compiled = _get_compiled_evaluator(expr, compiled_cache)
    samples_int = _get_int_samples(oracle)

    if compiled is not None:
        evaluator = _build_compiled_input_adapter(compiled, variables)
        return [evaluator(inputs) for inputs, _oracle_output in samples_int]

    return [
        int(_evaluate_tree(expr, inputs, variables))
        for inputs, _oracle_output in oracle.synthesis_map.items()
    ]


def _get_compiled_evaluator(
    expr: Expr, compiled_cache: CompiledCache | None
) -> Callable[[List[int]], int] | None:
    key = repr(expr)

    if compiled_cache is not None and key in compiled_cache:
        return compiled_cache[key]

    try:
        compiled = compile_expr_to_python(expr)
    except ValueError:
        compiled = None

    if compiled_cache is not None:
        compiled_cache[key] = compiled

    return compiled


def _build_compiled_input_adapter(
    compiled: Callable[[List[int]], int], variables: List[Expr]
) -> Callable[[Tuple[int, ...]], int]:
    var_indices: List[int] = [int(v.name[1:]) for v in variables]
    max_idx: int = max(var_indices, default=-1)
    dense_indices: bool = var_indices == list(range(len(var_indices)))

    def eval_compiled(inputs: Tuple[int, ...]) -> int:
        if dense_indices:
            return compiled(inputs)

        input_list = [0] * (max_idx + 1)
        for idx, expr_val in zip(var_indices, inputs):
            input_list[idx] = expr_val
        return compiled(input_list)

    return eval_compiled


def _get_int_samples(oracle: SynthesisOracle) -> List[Tuple[Tuple[int, ...], int]]:
    samples_int: List[Tuple[Tuple[int, ...], int]] | None = getattr(
        oracle, "_samples_int", None
    )
    if samples_int is None:
        samples_int = [
            (tuple(int(v) for v in inputs), int(output))
            for inputs, output in oracle.synthesis_map.items()
        ]
        oracle._samples_int = samples_int
    return samples_int


def _evaluate_tree(expr: Expr, inputs: Tuple[Expr, ...], variables: List[Expr]) -> Expr:
    replacements = {variables[i]: inputs[i] for i in range(len(inputs))}
    result = expr_simp.expr_simp(expr.replace_expr(replacements))
    if not isinstance(result, ExprInt):
        raise TypeError(
            f"Expected ExprInt result from expression evaluation, got {type(result)}"
        )
    return result
