import logging
import re
from pathlib import Path
from typing import Dict, Tuple

import z3
from miasm.expression.expression import Expr, ExprId, ExprInt
from miasm.expression.simplifications import expr_simp
from miasm.ir.translators.z3_ir import TranslatorZ3

from msynth.simplification.ast import AbstractSyntaxTreeTranslator
from msynth.simplification.oracle import SimplificationOracle
from msynth.utils.expr_utils import get_subexpressions, get_unique_variables
from msynth.utils.unification import gen_unification_dict, reverse_unification


logger = logging.getLogger("msynth.simplifier")


class Simplifier:
    """
    Expression simplification based on a pre-computed simplification oracle.

    The Simplifier has access to a pre-computed simplification oracle, stores
    inputs, evaluates expressions, determines the equivalence class of
    an expression (based on its input-output behavior) and holds a map
    of equivalence classes that map a list of expressions with the same
    I/O behavior.

    Based on this oracle, the Simplifier walks over an expression
    represented as an abstract syntax tree (AST) from the root downwards
    and tries to simplify subtrees based on oracle-lookups.

    The approach is inspired by:
    "QSynth: A Program Synthesis based Approach for Binary Code Deobfuscation" by
    Robin David, Luigi Coniglio and Mariano Ceccato (NDSS, BAR 2020).
    Link: https://archive.bar/pdfs/bar2020-preprint9.pdf

    Similar to QSynth, the Simplifier replaces already simplified subtrees
    in the original expression with placeholder variables to reduce
    the number of variables in too complex expressions. For this, the
    `_global_variable_prefix` attribute is used.

    The Simplifier applies an SMT-based equivalence check before replacing
    subexpressions for verification. By default, it uses a pre-configured
    timeout and applies the replacement if the equivalence has been proven
    or the timeout is triggered. In case a counter-example has been found,
    the replacement is withdrawn. For higher confidence, the user can limit
    replacements to successful equivalence checks (ignoring timeouts).
    For this, the variable `enforce_equivalence` has to be set and,
    optionally, the `solver_timeout` to be increased.


    Attributes:
        oracle (SimplificationOracle): Pre-computed simplification oracle.
        enforce_equivalence (bool): Flag to enforce semantic equivalence checks before replacements.
        solver_timeout (int): SMT solver timeout in seconds.

    Private Attributes:
        _translator_ast (AbstractSyntaxTreeTranslator): Translator to translate Miasm IR expressions into ASTs.
        _translator_z3 (TranslatorZ3): Translator to translate Miasm IR expressions into Z3 expressions.
        _solver (Z3Solver): SMT Solver instance.
        _global_variable_prefix (str): Variable prefix for placeholder variables.


    """

    def __init__(
        self,
        oracle_path: Path,
        enforce_equivalence: bool = False,
        solver_timeout: int = 1,
    ):
        """
        Intializes an instance of Simplifier.

        Args:
            oracle_path: File path to pre-computed simplification oracle.
            enforce_equivalence: Flag to enforce semantic equivalence checks before replacements.
            solver_timeout: SMT solver timeout in seconds.
        """
        # public attributes
        self.oracle = SimplificationOracle.load_from_file(oracle_path)
        self.enforce_equivalence = enforce_equivalence
        self.solver_timeout = solver_timeout

        # internal attributes
        self._translator_ast = AbstractSyntaxTreeTranslator()
        self._translator_z3 = TranslatorZ3()
        self._solver = z3.Solver()
        self._global_variable_prefix = "global_reg"

    def check_semantical_equivalence(self, f1: Expr, f2: Expr) -> z3.CheckSatResult:
        """
        Checks with an SMT solver if two expressions are semantically equivalent.

        Two expressions are semantically equivalent if
        SMT(f1 != f2) returns UNSAT. In case of SAT,
        the SMT solver found a concrete counterexample.
        For UNKNOWN, the defined timeout was triggered.

        Args:
            f1: Expression used in semantic equivalence check.
            f2: Expression used in semantic equivalence check.

        Returns:
            SAT, UNSAT or UNKNOWN
        """
        # reset solver
        self._solver.reset()
        # set solver timeout (Z3 expects timeout in ms)
        self._solver.set("timeout", self.solver_timeout * 1000)
        # add contraints
        self._solver.add(
            self._translator_z3.from_expr(f1) != self._translator_z3.from_expr(f2)
        )

        return self._solver.check()

    @staticmethod
    def _skip_subtree(expr: Expr) -> bool:
        """
        Skips the subtree if an expression is a terminal expression.

        A terminal expression is a leaf in the abstract syntax tree,
        such as an ExprInt (register/variable), ExprMem (memory)
        or ExprLoc (location label) or ExprInt (integer).

        Args:
            expr: Expression to test.

        Returns:
            True if expr is terminal expression.
        """
        return expr.is_id() or expr.is_int() or expr.is_loc()  # type: ignore

    def determine_equivalence_class(self, expr: Expr) -> str:
        """
        Determines the equivalence class of an expression.

        To determine the equivalence class, we compute the
        expression's output behavior and query the simplification
        oracle.

        In case an expression always has the same constant
        output (e.g., [10, 10, 10, ..., 10]), we add the constant
        as new equivalence class to the oracle. This way, we can
        simplify constants that are not part of the pre-computed
        oracle.

        Args:
            expr: Expression to determine the equivalence class for.

        Returns:
            Expression's equivalence class as string.
        """
        # get output behavior
        outputs = self.oracle.get_outputs(expr)
        # get equivalence class
        equiv_class: str = self.oracle.determine_equiv_class(expr, outputs)

        # if all evaluate to same constant, add/replace equiv class with constant
        if len(set(outputs)) == 1:
            self.oracle.set_equiv_class(equiv_class, [ExprInt(outputs[0], expr.size)])

        return equiv_class

    def _reverse_global_unification(
        self, expr: Expr, unification_dict: Dict[Expr, Expr]
    ) -> Expr:
        """
        Iteratively reverses the global unifications of an expression.

        For the given unification dictionary, unification variables can
        be part of other unification rules. To reverse all unifications
        in a given expression, the reverse unification process is applied
        iteratively.

        Example: Given: {r0: x + r1, r1: y} and expression r0 + r1.
                 We first transform it into (x + r1) + y and then to
                 (x + y) + y.


        Args:
            expr: Expression to reverse unification for.
            unification_dict: Dictionary of expressions containing unifications.

        Returns:
            Expression with reversed unification.
        """
        # while there is any unification variable remaining in the expression
        while any(
            [
                v.name.startswith(self._global_variable_prefix)
                for v in get_unique_variables(expr)
            ]
        ):
            # replace in expression
            expr = expr.replace_expr(unification_dict)

        return expr

    def _gen_global_variable_replacement(self, index: int, size: int) -> Expr:
        """
        Helper function to generate a global placeholder variable.

        Global placeholder variables are used in the simplifier to
        reduce the number of variables in too complex expressions.

        Args:
            index: Index of the placeholder variable.
            size: Size of the placeholder variable.

        Returns:
            Placeholder variable as expression.
        """
        return ExprId(f"{self._global_variable_prefix}{index}", size)

    def _is_suitable_simplification_candidate(
        self, expr: Expr, simplified: Expr
    ) -> bool:
        """
        Checks if a simplification candidate is not suitable.

        This check ensures the semantical correctness of the simplification.

        We skip the simplification candiate

        1. If the simplification candidate contains any unification variable.
           In this case, not every variable of the simplification candidate
           can be matched to a terminal expression in the original one.


        2. If Miasm's expression simplification results in the same expression for
           the original and the simplified one. In this case, the lookup in the
           simplification oracle is not required.

        3. If the original expression is semantically equivalent to the simplified one.
           Since this query is computationally expensive, we, by default, set a small
           timeout and check only if the SMT solver is not able to find a proof for
           inequivalence in the provided time. If the solver was not able to proof
           the equivalence within the provided time, we still accept it.

           The user has the possibility to enforce the SMT-based equivalence check
           to be successful by setting the `enforce_equivalence` flag and
           (optionally) increasing the `solver_timeout`.

        Args:
            expr: Original expression.
            simplified: Simplified expression candidate.

        Returns:
            True if simplification should be skipped, False otherwise.
        """
        # contains placeholder variables
        if any(
            [re.search("^p[0-9]*", v.name) for v in get_unique_variables(simplified)]
        ):
            return False
        # same normalized expression
        if not simplified.is_int() and expr_simp(expr) == expr_simp(simplified):
            return False
        # SMT solver proves non-equivalence or timeouts
        if (
            self.enforce_equivalence
            and self.check_semantical_equivalence(expr, simplified) != z3.unsat
        ):
            return False
        # SMT solver finds a counter example
        if self.check_semantical_equivalence(expr, simplified) == z3.sat:
            return False
        return True

    def _find_suitable_simplification(
        self, equiv_class: str, expr: Expr, unification_dict: Dict[Expr, Expr]
    ) -> Tuple[bool, Expr]:
        """
        Finds a suitable simplified expression from the equivalence class.

        We query the oracle for all simplification candidates for a given equivalence
        class and iteratively check if we find a suitable candidate. For each candidate,
        we have to inverse the unification (replacing p0, p1 etc. with terminal symbols
        of the original expression) and check if the simplification is suitable. In other
        words, we check if the candidate is actually shorter and whether we could replace
        our expression with the simplified one. We return the first suitable canddiate
        found.

        Args:
            equiv_class: The expression's equivalence class as string.
            expr: Expression to find a suitable simplification for.
            unification_dict: Dictionary of unification variables.

        Returns:
            Tuple of True and simplified candidate if successful, False and original expression otherewise.
        """
        # walk over all simplification candidates
        for candidate in self.oracle.get_equiv_class_members(equiv_class):
            # reverse unification of simplification candidate
            simplified = reverse_unification(candidate, unification_dict)

            # skip simplification if necessary
            if not self._is_suitable_simplification_candidate(expr, simplified):
                continue

            return True, simplified

        return False, expr

    def simplify(self, expr: Expr) -> Expr:
        """
        High-level algorithm to simplify an expression.

        Given an expression, we generate an abstract syntax tree (AST)
        and simplify the AST as follows in a fixpoint iteration:

        1. We do a BFS over the AST (top to bottom) and try to simplify
           the largest possible subtree.

        2. For each subtree, we check if its input-output behavior
           can be represented as an equivalence class that is already
           contained in the pre-computed oracle. For this, we have to
           unify the subtree (by replacing terminal nodes with place
           holder variables), re-apply the unifications to simplification
           candidates and check if it is suitable.

        3. If a suitable simplification candidate is found, we store it in an
           dictionary and replace the subtree with a placeholder variable in the
           AST.

        4. If no more simplifications can be applied, we recursively replace all
           place holder variables with the simplified subtrees in the AST.

        Args:
            expr: Expression to simplify

        Returns:
            Simplified expression
        """
        # transform expr to abstract syntax tree
        ast = self._translator_ast.from_expr(expr)
        # dictionary to map to placeholder variables to simplified subtrees
        global_unification_dict: Dict[Expr, Expr] = {}
        # placeholder variable counter
        global_ctr = 0

        # fixpoint iteration
        while True:
            before = ast

            # walk over all subtrees
            for subtree in get_subexpressions(ast):
                # skip subtree if possible
                if self._skip_subtree(subtree):
                    continue

                # build unification dictionary
                unification_dict = gen_unification_dict(subtree)

                # determine subtree's equivalence class
                equiv_class = self.determine_equivalence_class(
                    subtree.replace_expr(unification_dict)
                )

                # if the equivalence class is in the pre-computed oracle:
                if self.oracle.contains_equiv_class(equiv_class):
                    # check if there is a simpler subtree in the equivalence class
                    success, simplified = self._find_suitable_simplification(
                        equiv_class, subtree, unification_dict
                    )

                    # skip if no candidate found
                    if not success:
                        continue

                    # generate global placeholder variable
                    global_variable = self._gen_global_variable_replacement(
                        global_ctr, subtree.size
                    )
                    global_ctr += 1

                    # map global placeholder variable to simplified subtree
                    global_unification_dict[global_variable] = simplified

                    # replace original subtree with global placeholder variable
                    ast = ast.replace_expr({subtree: global_variable})
                    break

            # check if fixpoint is reached
            if before == ast:
                break

        # replace global placeholder variables with simplified subtrees in ast
        ast = self._reverse_global_unification(ast, global_unification_dict)

        return expr_simp(ast)
