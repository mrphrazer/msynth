from __future__ import annotations

from typing import Dict, Tuple, List

from miasm.expression.expression import Expr, ExprInt
from miasm.expression.simplifications import expr_simp
from msynth.utils.sampling import get_rand_input


class SynthesisOracle:
    """
    Synthesis Oracle used for I/O sampling.

    Intuitively, it models the semantic I/O behavior of a
    single mathematical function f(x0, ..., xi) for i parameters
    and a fixed number of input-output samples.

    The function is approximated by storing a dictionary (the synthesis map)
    of input arrays. Each input array maps represents a concrete execution
    of the function and maps to a single output. Inputs and outputs are stored
    in Miasm IL expressions.

    Attributes:
        synthesis_map (Dict[Tuple[Expr, ...], Expr]): Maps concrete inputs to outputs.
    """

    def __init__(self, synthesis_map: Dict[Tuple[Expr, ...], Expr]):
        """
        Initializes an SynthesisOracle instance.

        Args:
            synthesis_map (Dict[Tuple[Expr, ...], Expr]): Dictionary of input-output behavior.
        """
        # ensure that synthesis_map contains at least one I/O pair
        if len(synthesis_map) == 0:
            raise ValueError(
                "SynthesisOracle is empty but should contain at least one I/O pair."
            )
        self.synthesis_map: Dict[Tuple[Expr, ...], Expr] = synthesis_map
        # cache integer samples for faster scoring
        self._samples_int: List[Tuple[Tuple[int, ...], int]] = [
            (tuple(int(v) for v in inputs), int(output))
            for inputs, output in synthesis_map.items()
        ]

    def gen_from_expression(
        expr: Expr, variables: List[Expr], num_samples: int
    ) -> SynthesisOracle:
        """
        Builds a SynthesisOracle instance from a given expression.

        For a given expression, `num_samples` independent I/O pairs are
        evaluated as follows:

        1. We generate a list of random values, one for each variable. Random values
           are represented in Miasm IL.
        2. We evaluate the expression by replacing all variables in the expression
           by their corresponding value and do a constant propagation.
        3. We map the list of inputs to the obtained integer value (in Miasm IL).

        Args:
            expr (Expr): Expression representing a function f(x0, ..., xi).
            variables (List[Expr]): List of variables contained in `expr`.
            num_samples (int): Number of I/O samples to evaluate.

        Returns:
            SynthesisOracle: Generated SynthesisOracle instance.
        """
        synthesis_map = {}

        for raw_inputs in _gen_sample_inputs(variables, num_samples):
            inputs = []
            replacements = {}
            for variable, value in zip(variables, raw_inputs):
                concrete = ExprInt(value, variable.size)
                replacements[variable] = concrete
                inputs.append(concrete)

            # evaluate expression to obtain output
            result = expr_simp(expr.replace_expr(replacements))
            # output should be an ExprInt
            if not result.is_int():
                raise TypeError(
                    f"Expected ExprInt result from expression evaluation, got {type(result)}"
                )
            # map list of inputs to output
            synthesis_map[tuple(inputs)] = result

        return SynthesisOracle(synthesis_map)


def _gen_sample_inputs(variables: List[Expr], num_samples: int) -> List[List[int]]:
    """
    Generate paper-style fixed samples first, then random samples.

    Smir/Xyntia use fixed vectors for common boundary values before random I/O
    examples. For mixed-size expressions, each variable receives the boundary
    value masked to its own width.
    """
    samples = []

    for fixed_index in range(min(num_samples, 5)):
        samples.append(
            [
                _fixed_value_for_index(variable.size, fixed_index)
                for variable in variables
            ]
        )

    for _ in range(max(0, num_samples - len(samples))):
        samples.append([get_rand_input() for _variable in variables])

    return samples


def _fixed_value_for_index(size: int, index: int) -> int:
    if index == 0:
        return 0
    if index == 1:
        return 1
    if index == 2:
        return (1 << size) - 1
    if index == 3:
        return (1 << (size - 1)) - 1
    if index == 4:
        return 1 << (size - 1)
    raise ValueError(f"Unsupported fixed sample index: {index}")
