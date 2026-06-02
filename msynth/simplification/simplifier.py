import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import z3
from miasm.expression.expression import Expr, ExprId, ExprInt
from miasm.ir.translators.z3_ir import TranslatorZ3

from msynth.simplification.oracle import SimplificationOracle
from msynth.simplification.preprocessing import Preprocessor, default_preprocessor
from msynth.simplification.cegis import CegisSolver, TemplateOracle
from msynth.simplification.simba import SimbaPass
from msynth.utils.expr_utils import (
    get_subexpressions,
    get_unique_variables,
    is_strictly_smaller_tree,
)
from msynth.simplification.rewrites import DEFAULT_REWRITER
from msynth.utils.sampling import has_adversarial_counterexample
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
        _translator_z3 (TranslatorZ3): Translator to translate Miasm IR expressions into Z3 expressions.
        _solver (Z3Solver): SMT Solver instance.
        _global_variable_prefix (str): Variable prefix for placeholder variables.


    """

    def __init__(
        self,
        oracle_path: Path,
        enforce_equivalence: bool = False,
        solver_timeout: int = 1,
        preprocessor: Preprocessor | None = None,
        enable_subtree_simba: bool = True,
        subtree_simba_max_vars: int = 5,
        subtree_simba_max_nodes: int = 30,
        enable_cegis: bool = False,
        cegis_max_templates: int = 50,
        cegis_timeout: int = 2,
        cegis_max_variables: int = 3,
        cegis_runtime_templates: int = 80,
        cegis_refinement_iters: int = 3,
        cegis_validation_samples: int = 16,
        cegis_expand_templates: bool = True,
        cegis_expansion_budget: int = 40,
    ):
        """
        Intializes an instance of Simplifier.

        Args:
            oracle_path: File path to pre-computed simplification oracle.
            enforce_equivalence: Flag to enforce semantic equivalence checks before replacements.
            solver_timeout: SMT solver timeout in seconds.
            preprocessor: Optional preprocessing pipeline applied before oracle simplification.
            enable_subtree_simba: Enable SimbaPass as a fallback on oracle misses
                during the simplification loop. The global SimbaPass in the
                preprocessing pipeline runs once over the whole expression; this
                fallback applies it to inner subtrees that the oracle did not
                match.
            subtree_simba_max_vars: Skip subtree SiMBA when the unification dict
                has more than this many terminals.
            subtree_simba_max_nodes: Skip subtree SiMBA when the Miasm graph of
                the subtree has more than this many nodes.
            enable_cegis: Enable CEGIS constant synthesis as a last-resort
                fallback on oracle + subtree-SiMBA miss. **Off by default** —
                the CEGIS path runs Z3 against up to ``cegis_max_templates``
                templates with a per-template timeout of ``cegis_timeout``
                seconds, which is non-trivial. Turn on for workloads whose
                subtrees contain arbitrary constants the precomputed oracle
                cannot cover (e.g. ``v0 * 0xDEADBEEF + 0x1337``). See
                :mod:`msynth.simplification.cegis` for the algorithm.
            cegis_max_templates: Max templates attempted per subtree.
            cegis_timeout: Per-template Z3 timeout in seconds.
            cegis_max_variables: Skip CEGIS on subtrees with more than this
                many unified terminals.
            cegis_runtime_templates: Size of the hand-crafted runtime
                template oracle generated at construction.
            cegis_refinement_iters: Max counter-example refinement
                iterations per template attempt.
            cegis_validation_samples: Validation samples per refinement step.
            cegis_expand_templates: When True, wrap base templates with
                light constant decorations (``+c``, ``^c``, ``(&c)|c'``) to
                broaden coverage without manual enumeration.
            cegis_expansion_budget: Cap on total expanded templates.
        """
        # public attributes
        self.oracle = SimplificationOracle.load_from_file(oracle_path)
        self.enforce_equivalence = enforce_equivalence
        self.solver_timeout = solver_timeout
        extra_passes = None if preprocessor is None else preprocessor.passes
        self.preprocessor = default_preprocessor(extra_passes)

        # internal attributes
        self._translator_z3 = TranslatorZ3()
        self._solver = z3.Solver()
        self._global_variable_prefix = "global_reg"
        self._subtree_simba_pass: Optional[SimbaPass] = (
            SimbaPass() if enable_subtree_simba else None
        )
        self._subtree_simba_max_vars = subtree_simba_max_vars
        self._subtree_simba_max_nodes = subtree_simba_max_nodes
        # CEGIS solver — built lazily, only when enable_cegis=True, so the
        # off-path costs nothing besides one None check per fallback hop.
        self._cegis_solver: Optional[CegisSolver] = None
        if enable_cegis:
            # The template oracle's input matrix needs at least one column per
            # placeholder variable the solver may instantiate. Sizing it to
            # cegis_max_variables keeps that invariant under caller-tunable
            # variable budgets — without this, raising cegis_max_variables
            # above the gen_runtime_oracle default crashed the evaluator with
            # IndexError on the first >3-variable subtree.
            template_oracle = TemplateOracle.gen_runtime_oracle(
                num_variables=cegis_max_variables,
                template_budget=cegis_runtime_templates,
            )
            self._cegis_solver = CegisSolver(
                template_oracle,
                max_templates=cegis_max_templates,
                solver_timeout=cegis_timeout,
                max_variables=cegis_max_variables,
                refinement_iters=cegis_refinement_iters,
                validation_samples=cegis_validation_samples,
                expand_templates=cegis_expand_templates,
                expansion_budget=cegis_expansion_budget,
            )

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

    @staticmethod
    def _is_simba_op_candidate(expr: Expr) -> bool:
        """
        Restrict subtree-level SiMBA to linear-friendly operators.

        Multiplication is allowed only when at least one operand is a constant,
        and shifts only with a constant shift amount. These guards keep subtree
        SiMBA on the linear-MBA fragment SimbaPass actually supports and avoid
        spending work on subtrees its internal classifier would reject anyway.
        """
        if not expr.is_op():
            return False
        op = expr.op
        if op in {"+", "-", "^", "&", "|"}:
            return True
        if op == "*":
            return any(arg.is_int() for arg in expr.args)
        if op in {"<<", ">>"}:
            return len(expr.args) == 2 and expr.args[1].is_int()
        return False

    def _try_subtree_simba(
        self, subtree: Expr, unification_dict: Dict[Expr, Expr]
    ) -> Optional[Expr]:
        """
        Run SimbaPass on a single subtree, guarded by conservative limits.

        Triggered as a fallback when the oracle lookup fails. Each guard is a
        cheap pre-filter that avoids running SimbaPass on subtrees where the
        linear-MBA reconstruction would either be unsound or expensive:
        operator must be in the linear-friendly whitelist, terminal-variable
        count under ``subtree_simba_max_vars``, AST node count under
        ``subtree_simba_max_nodes``.

        ``SimbaPass.run`` returns the input unchanged when SiMBA does not
        apply; that case is normalized to ``None`` so callers can treat
        "no improvement" uniformly.
        """
        if self._subtree_simba_pass is None:
            return None

        # Skip subtrees whose unification dict contains a ``global_reg*``
        # placeholder from an earlier oracle hit. The cube reconstruction
        # itself is mathematically sound here — the linear-MBA theorem
        # holds whether the atoms are registers or already-substituted
        # subtrees — but the rewrite locks the simplifier into a strictly
        # worse fixed point. SiMBA emits the *canonical* linear-MBA
        # reconstruction in the conjunction basis, which produces forms
        # like ``0xFF * g_k`` (in place of ``-g_k``) and ``g_k << 1`` (in
        # place of ``g_k + g_k``). Once such a form is cemented into a
        # new placeholder body, neither ``ring_normalize`` nor a
        # subsequent oracle hit can fold it back together: like-term
        # collection compares structurally, and the oracle's
        # ``is_strictly_smaller_tree`` guard measures candidates against
        # the already-canonicalised input and rejects the shorter
        # candidate the library actually contains. The end result is the
        # same coefficient combination but a larger structural form.
        if any(
            terminal.is_id() and terminal.name.startswith(self._global_variable_prefix)
            for terminal in unification_dict
        ):
            return None

        if len(unification_dict) > self._subtree_simba_max_vars:
            return None
        if len(subtree.graph().nodes()) > self._subtree_simba_max_nodes:
            return None
        if not self._is_simba_op_candidate(subtree):
            return None

        simplified = self._subtree_simba_pass.run(subtree)
        if simplified == subtree:
            return None
        return simplified

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


        2. If the simplification candidate is not structurally smaller than the
           original expression after reverse unification. This rejects no-op,
           equal-size, and larger replacements while still allowing candidates
           that Miasm can normalize to the same expression as the original.

        3. If the original expression is not semantically equivalent to the simplified one.
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
        # Reject concrete size regressions before doing SMT work. The helper is
        # intentionally bounded and iterative because Miasm graph construction can
        # be expensive or recursive on very large reverse-unified candidates.
        if not is_strictly_smaller_tree(simplified, expr):
            return False
        equivalence_result = self.check_semantical_equivalence(expr, simplified)
        # SMT solver proves non-equivalence or timeouts
        if self.enforce_equivalence and equivalence_result != z3.unsat:
            return False
        # SMT solver finds a counter example
        if equivalence_result == z3.sat:
            return False
        # In permissive mode, UNKNOWN is normally accepted to avoid slow SMT proofs.
        # Before accepting it, run a tiny deterministic edge-value probe to catch
        # sampled-oracle collisions such as variable-shift candidates that agree on
        # random oracle inputs but fail around modular wraparound values.
        if equivalence_result == z3.unknown and has_adversarial_counterexample(
            expr, simplified
        ):
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
        # transform expr to the preprocessed abstract syntax tree
        ast = self.preprocessor.run(expr)
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

                simplified: Optional[Expr] = None

                # The oracle's I/O inputs matrix has self.oracle.num_variables
                # columns. Subtrees with more unified terminals overflow the
                # compiled evaluator's index lookup (i[idx] raises IndexError).
                # Skip the oracle path for those and fall through to subtree
                # SiMBA / CEGIS, which do their own scaling checks.
                if len(unification_dict) <= self.oracle.num_variables:
                    # determine subtree's equivalence class
                    equiv_class = self.determine_equivalence_class(
                        subtree.replace_expr(unification_dict)
                    )

                    # pre-computed oracle lookup
                    if self.oracle.contains_equiv_class(equiv_class):
                        success, candidate = self._find_suitable_simplification(
                            equiv_class, subtree, unification_dict
                        )
                        if success:
                            simplified = candidate

                # subtree-level SiMBA fallback on oracle miss
                if simplified is None:
                    candidate = self._try_subtree_simba(subtree, unification_dict)
                    if (
                        candidate is not None
                        and self._is_suitable_simplification_candidate(
                            subtree, candidate
                        )
                    ):
                        simplified = candidate

                # CEGIS constant synthesis on oracle + subtree-SiMBA miss.
                # Opt-in via enable_cegis; recovers expressions whose shape
                # matches a template but whose constants are arbitrary and
                # therefore absent from the precomputed oracle.
                if simplified is None and self._cegis_solver is not None:
                    unified_subtree = subtree.replace_expr(unification_dict)
                    candidate = self._cegis_solver.try_synthesize(
                        subtree, unified_subtree, unification_dict
                    )
                    if (
                        candidate is not None
                        and self._is_suitable_simplification_candidate(
                            subtree, candidate
                        )
                    ):
                        simplified = candidate

                # skip if no candidate found
                if simplified is None:
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

        return DEFAULT_REWRITER.normalize(ast)
