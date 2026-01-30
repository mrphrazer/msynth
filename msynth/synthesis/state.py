from __future__ import annotations

from math import log2
from typing import Callable, Dict, List, Tuple

from miasm.expression.expression import Expr
from miasm.expression.simplifications import expr_simp
from msynth.synthesis.oracle import SynthesisOracle
from msynth.utils.expr_utils import compile_expr_to_python, get_unique_variables


class SynthesisState:
    """
    Container that wraps an expression as a synthesis state.

    It provides functionality to manage the expression and to calculate it's score 
    (used as feedback for the synthesizer).

    Understanding an expression as an abstract syntax tree (AST), variables in the expression
    represent leaves in the AST; an AST can have several leaves that represent the same variable.
    To allow the modification of individual leaves without changing all leaves 
    that represent the same variable, the expression management is realized as follows:

    The SynthesisState manages an expression `expr_ast` in which each variable is unique. Additionally,
    it keeps a dictionary of replacements called `replacements` that maps each variable/leaf 
    to a variable in the synthesis domain. To optimize score calculation, it also keeps the original
    expression in the internal data variable `_expr`.

    For example, x and y are leaves in the AST (x + y) * x. We store the expression 
    as (t1 + t2) * t3 and create a replacement dictionary {t1: x, t2: y, t3: x}. The internal
    variable `_expr` stores (x + y) * x.

    Attributes:
        expr_ast (Expr): Expression in Miasm IR with unique variables/leaves.
        replacements (Dict[Expr, Expr]): Dictionary of variable replacements.
    
    Private Attributes:
        _expr (Expr): Internal expression, stored for optimization.
    """

    def __init__(self, expr: Expr, replacements: Dict[Expr, Expr] = {}):
        """
        Initializes a SynthesisState instance.

        Attributes:
            expr (Expr): Expression in Miasm IR with unique variables/leaves.
            replacements (Dict[Expr, Expr], optional): Dictionary of variable replacements.
        """
        self.expr_ast: Expr = expr
        self._expr: Expr = expr.replace_expr(replacements)
        self.replacements: Dict[Expr, Expr] = replacements
        # cache compiled evaluation for current _expr; invalidate by key when _expr changes
        self._compiled_key: str | None = None
        self._compiled_func: Callable[[List[int]], int] | None = None
        self._compiled_fail_key: str | None = None

    def clone(self) -> SynthesisState:
        """
        Returns a copy of itself.

        Returns:
            SynthesisState: Cloned state.
        """
        return SynthesisState(self.expr_ast.copy(), self.replacements.copy())

    def get_expr(self) -> Expr:
        """
        Returns the expression with applied replacements.

        If the stored AST is (t1 + t2) * t3 and the mapping
        {t1: x, t2: y, t3: x}, it returns (x + y) * z.

        Returns:
            Expr: Expression with applied replacements.
        """
        return self.expr_ast.replace_expr(self.replacements)

    def get_score(self, oracle: SynthesisOracle, variables: List[Expr]) -> float:
        """
        Calculates the state's score used as feedback for the synthesizer.

        To calculate the score, the function compares the state's output behavior with
        the oracle's output for each input array. The score is calculated by adding up
        the arithmetic distances between the two outputs in the logarithmic domain:

        sum(log2(1 + abs(state_output - oracle_output))) for all input arrays in the synthesis oracle.

        The smaller the score, the closer are the state's outputs to those of the oracle. A score
        of 0 means that state and oracle have the same I/O behavior.

        Args:
            oracle (SynthesisOracle): I/O oracle.
            variables (List[Expr]): List of variables in Miasm IR.

        Returns:
            float: State's score.
        """
        # apply current replacement dictionary to speed up expression evaluation
        self._apply_replacements()

        # use compiled evaluation when possible; fallback preserves correctness for unsupported ops.
        compiled: Callable[[List[int]], int] | None = None
        compiled_key: str = repr(self._expr)
        if compiled_key == self._compiled_key:
            # hit cache for identical expression
            compiled = self._compiled_func
        elif compiled_key != self._compiled_fail_key:
            # try to compile once per unique expression
            try:
                compiled = compile_expr_to_python(self._expr)
                self._compiled_key = compiled_key
                self._compiled_func = compiled
            except ValueError:
                # remember unsupported expressions to skip repeat exceptions
                self._compiled_fail_key = compiled_key

        if compiled is not None:
            # map variables to their p# indices for input alignment
            var_indices: List[int] = [int(v.name[1:]) for v in variables]
            max_idx: int = max(var_indices, default=-1)
            dense_indices: bool = var_indices == list(range(len(var_indices)))

            def eval_compiled(inputs: Tuple[int, ...]) -> int:
                if dense_indices:
                    # fast path: inputs already ordered p0..pN
                    return compiled(inputs)
                # sparse p# indices: expand into a dense list
                input_list = [0] * (max_idx + 1)
                for idx, expr_val in zip(var_indices, inputs):
                    input_list[idx] = expr_val
                return compiled(input_list)

            # use cached int samples from the oracle when available
            samples_int: List[Tuple[Tuple[int, ...], int]] | None = getattr(oracle, "_samples_int", None)
            if samples_int is None:
                # build the cache lazily for legacy oracles
                samples_int = [
                    (tuple(int(v) for v in inputs), int(output))
                    for inputs, output in oracle.synthesis_map.items()
                ]
                oracle._samples_int = samples_int

            return sum(
                log2(1 + abs(eval_compiled(inputs) - oracle_output))
                for inputs, oracle_output in samples_int
            )

        # fall back to tree-walking evaluation for unsupported expression types
        return sum(
            log2(1 + abs(int(self._evaluate(inputs, variables)) - int(oracle_output)))
            for inputs, oracle_output in oracle.synthesis_map.items()
        )

    def _evaluate(self, inputs: Tuple[Expr, ...], variables: List[Expr]) -> Expr:
        """
        Evaluates the state to an integer (in Miasm IL) for a provided input array.

        The evaluation is performed as follows:

        1. A replacement dictionary is built that maps variables to integers.
        2. All variables in the expression are replaced with integers.
        3. The expression is simplified via constant folding.

        For example, for the expression (x + y) * z and the input array [5, 10, 3] we
        1. build the dictionary {x: 5, y: 10, z: 3},
        2. apply the replacements to the expression and get (5 + 10) * 3 and
        3. evaluate the expression to 45.

        Args:
            inputs (Tuple[Expr, ...]): List of integer inputs in Miasm IL.
            variables (List[Expr]): List of variables in Miasm IL.

        Returns:
            Expr: Integer in Miasm IL.
        """
        replacements = {variables[i]: inputs[i] for i in range(len(inputs))}
        return expr_simp.expr_simp(self._expr.replace_expr(replacements))

    def _apply_replacements(self) -> None:
        """
        Applies the current replacement dictionary to the stored AST.
        """
        self._expr = self.expr_ast.replace_expr(self.replacements)

    def get_expr_simplified(self) -> Expr:
        """
        Returns the expression with applied replacements 
        and Miasm's simplification rules.

        Returns:
            Expr: Simplified expression.
        """
        return expr_simp.expr_simp(self.get_expr())

    def cleanup(self) -> None:
        """
        Cleanup the replacement dictionary.

        For optimization purposes, the function scans
        all variables that are used in the current AST, 
        copies their replacements and replaces the original
        dictionary afterward. This way, dead variables in the
        replacement dictionary are removed.
        """
        # build new replacement dictionary
        replacements = {}

        # walk over unique variables and fill dictionary
        for v in get_unique_variables(self.expr_ast):
            replacements[v] = self.replacements[v]

        # replace replacement dictionary
        self.replacements = replacements
