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
        # init map
        synthesis_map = {}

        # walk over number of samples
        for _ in range(num_samples):
            # list of inputs
            inputs = []
            # dictionary of expression replacements
            replacements = {}
            # walk over all variables
            for v in variables:
                # generate a random value
                value = get_rand_input()
                # replace variable with random value
                replacements[v] = ExprInt(value, v.size)
                # add random value to list of inputs
                inputs.append(ExprInt(value, v.size))

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
