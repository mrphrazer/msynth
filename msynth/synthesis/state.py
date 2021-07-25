from __future__ import annotations

from math import log2
from typing import Dict, List, Tuple

from miasm.expression.expression import Expr
from miasm.expression.simplifications import expr_simp
from msynth.synthesis.oracle import SynthesisOracle
from msynth.utils.expr_utils import get_unique_variables


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
        # sum the scores
        return sum([
            # calculate score for input array
            log2(1 + abs(int(self._evaluate(inputs, variables)) - int(oracle_output)))
            # walk over all input arrays in the oracle
            for inputs, oracle_output in oracle.synthesis_map.items()
        ])

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
