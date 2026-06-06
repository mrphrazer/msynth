import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional, Tuple

import z3
from miasm.expression.expression import Expr, ExprId, ExprInt
from miasm.ir.translators.z3_ir import TranslatorZ3

from msynth.simplification.ast import AbstractSyntaxTreeTranslator
from msynth.simplification.oracle import SimplificationOracle
from msynth.simplification.pipeline import (
    Pipeline,
    PipelineMode,
    default_pipeline,
    gamba_pipeline,
    simba_pipeline,
)
from msynth.simplification.cegis import CegisSolver, TemplateOracle
from msynth.simplification.dead_variable_elimination import (
    DeadVariableEliminationConfig,
    DeadVariableEliminationPass,
)
import random as _random

from msynth.simplification.simba import SimbaPass
from msynth.utils.expr_utils import (
    compile_expr_to_python,
    get_subexpressions,
    get_unique_variables,
    is_strictly_smaller_tree,
)
from msynth.simplification.gamba import (
    GAMBA_POST_REWRITER,
    GAMBA_PREPROCESSOR,
    gamba_substitution,
)
from msynth.simplification.rewrites import DEFAULT_REWRITER
from msynth.utils.sampling import (
    _rename_variables_for_compilation,
    gen_adversarial_inputs,
    has_adversarial_counterexample,
)
from msynth.utils.unification import gen_unification_dict, reverse_unification

# Number of random I/O probes used by the permissive equivalence check, in
# addition to the deterministic edge-case inputs. Kept above the corpus runner's
# probe count so a candidate that passes this gate also passes the runner's
# independent equivalence check.
_PERMISSIVE_RANDOM_PROBES = 128


def binarized_node_count(expr: Expr) -> int:
    """Graph-node count of ``expr`` after canonicalizing to a strict binary tree.

    Representation-independent size: miasm's ``expr_simp`` emits variadic
    ``ExprOp`` nodes (``a+b+c`` is one graph node over three leaves) while a
    binarised form has the extra inner nodes. A raw ``len(expr.graph().nodes())``
    would therefore rank a variadic-but-not-actually-smaller stage above a
    genuinely smaller binary one. Used to select the smallest among the
    pipeline / AST / closing-rewriter stages, matching the corpus ``node_count``.
    """
    try:
        binary = AbstractSyntaxTreeTranslator().from_expr(expr)
    except Exception:
        binary = expr
    try:
        return len(binary.graph().nodes())
    except Exception:
        return len(get_subexpressions(binary))


logger = logging.getLogger("msynth.simplifier")


class Simplifier:
    """
    Expression simplification driven by an oracle, a pipeline, and
    (optionally) CEGIS-based constant synthesis.

    The Simplifier walks an expression top-down and attempts to rewrite
    each subtree using one of several mechanisms, in this order:

    1. **Pipeline** (top-level, runs once at the start of
       :meth:`simplify`). Selected via :class:`PipelineMode`; see
       :mod:`msynth.simplification.pipeline` for the three presets
       (``AST`` / ``SIMBA`` / ``GAMBA``). Optionally prefixed by a
       :class:`~msynth.simplification.dead_variable_elimination.DeadVariableEliminationPass`
       (``enable_dead_variable_elimination=True``, off by default) that prunes
       semantically dead/opaque variables before SimBA/GAMBA.
    2. **Oracle lookup** (per subtree). Looks the subtree up in
       :class:`SimplificationOracle` by its I/O equivalence class.
    3. **Subtree-SimBA** (per subtree, on oracle miss). Re-runs SimBA
       on the subtree alone — useful when the global pipeline's SimBA
       call rejected the whole expression but an inner subtree is a
       valid linear MBA. Enabled iff ``pipeline_mode`` runs SimBA at
       all (SIMBA / GAMBA), and wraps with GAMBA pre/post iff under
       GAMBA mode.
    4. **CEGIS** (per subtree, on oracle + SimBA miss). Off by default.
       Synthesises constants for a fixed family of templates using Z3 —
       useful for subtrees containing arbitrary constants the oracle
       cannot cover (e.g. ``v0 * 0xDEADBEEF + 0x1337``).
    5. **Closing rewriter** (top-level, runs once at the end).
       :data:`msynth.simplification.rewrites.DEFAULT_REWRITER` applies
       miasm's ``expr_simp`` plus the two guarded rules (ring
       normalisation, common-subterm factoring) on the post-substitution
       AST, catching shapes that became simplifiable only after
       reverse-unification produced them.

    Three configuration axes — pipeline, oracle, CEGIS — are orthogonal
    and combine freely. Common configurations:

    +-------------------------------+-----------------+---------+-------+
    | What you want                 | pipeline_mode   | oracle  | cegis |
    +===============================+=================+=========+=======+
    | Just normalise + apply oracle | ``AST``         | file    | off   |
    +-------------------------------+-----------------+---------+-------+
    | Pure SimBA, no oracle         | ``SIMBA``       | empty   | off   |
    +-------------------------------+-----------------+---------+-------+
    | GAMBA sandwich on tough MBAs  | ``GAMBA``       | empty   | off   |
    +-------------------------------+-----------------+---------+-------+
    | Arbitrary constants per case  | ``SIMBA``/etc.  | any     | on    |
    +-------------------------------+-----------------+---------+-------+

    **Default (``Simplifier()``):** ``pipeline_mode=PipelineMode.SIMBA``,
    no oracle path (empty in-memory oracle is built automatically), no
    CEGIS. End-to-end behaviour:

    1. The pipeline binarises the input (the simplifier loop's
       ``get_subexpressions`` walk requires the binary form to expose
       intermediate sub-pair nodes) and runs SimBA reconstruction over
       the whole expression.
    2. The subtree walk runs: the oracle is empty (every lookup misses),
       the per-subtree SimBA fallback is on (mode=SIMBA), CEGIS is off.
    3. The closing :data:`DEFAULT_REWRITER` applies miasm's
       ``expr_simp`` plus ring/factor.

    Pick a non-default ``pipeline_mode`` for a different trade-off:

    - ``AST`` — oracle lookups only: no SimBA reconstruction at the
      front and no per-subtree SimBA fallback, so the closing rewriter is
      the only stage doing active work (use with a populated oracle).
    - ``GAMBA`` — same as the SIMBA default plus wraps both the global
      SimBA call and the subtree-SimBA fallback with GAMBA's §5.2
      algebraic pre/post rewriters.

    **Placeholder substitution.** Inspired by QSynth (David, Coniglio,
    Ceccato; BAR 2020 — https://archive.bar/pdfs/bar2020-preprint9.pdf),
    the Simplifier replaces already-simplified subtrees with
    ``global_reg{N}`` placeholders to reduce variable count in complex
    expressions; placeholders are reverse-substituted at the end.

    **SMT verification.** Before committing a replacement, the
    Simplifier checks semantic equivalence with Z3. By default the
    replacement is committed when Z3 returns UNSAT or hits the
    ``solver_timeout`` (1s). Setting ``enforce_equivalence=True``
    requires a proven UNSAT — a timeout is treated as failure.

    Attributes:
        oracle (SimplificationOracle): Pre-computed oracle (or
            :meth:`SimplificationOracle.empty` if ``oracle_path=None``).
        pipeline (Pipeline): Active simplification pipeline. Determined
            by ``pipeline_mode`` or by an explicit ``pipeline=``
            override.
        enforce_equivalence (bool): Reject candidates on Z3 timeout
            instead of accepting them.
        solver_timeout (int): SMT solver timeout in seconds.

    Private Attributes:
        _pipeline_mode (PipelineMode): Mode that selected the pipeline
            and controls subtree-SimBA wrapping.
        _subtree_simba_pass (SimbaPass | None): Subtree-level SimBA
            fallback; ``None`` under AST mode.
        _cegis_solver (CegisSolver | None): CEGIS solver, lazily built
            when ``enable_cegis=True``.
        _translator_z3 (TranslatorZ3): Miasm IR → Z3 translator.
        _solver (Z3Solver): Z3 solver instance.
        _global_variable_prefix (str): Prefix for placeholder variables.
    """

    def __init__(
        self,
        oracle_path: Path | None = None,
        pipeline_mode: PipelineMode = PipelineMode.SIMBA,
        pipeline: Pipeline | None = None,
        enforce_equivalence: bool = False,
        solver_timeout: int = 1,
        subtree_simba_max_vars: int = 5,
        subtree_simba_max_nodes: int = 30,
        gamba_substitution_max_k: int = 3,
        enable_cegis: bool = False,
        cegis_max_templates: int = 256,
        cegis_timeout: int = 2,
        cegis_max_variables: int = 3,
        cegis_runtime_templates: int = 256,
        cegis_refinement_iters: int = 3,
        cegis_validation_samples: int = 16,
        cegis_expand_templates: bool = True,
        cegis_expansion_budget: int = 40,
        cegis_min_nodes: int = 5,
        enable_dead_variable_elimination: bool = False,
        dead_variable_elimination_config: DeadVariableEliminationConfig | None = None,
    ):
        """
        Initializes an instance of Simplifier.

        Args:
            oracle_path: Optional file path to a pre-computed simplification
                oracle. If ``None``, an in-memory empty oracle is used
                (see :meth:`SimplificationOracle.empty`). The oracle lookup
                still runs every iteration but always misses — the
                pipeline + subtree-SiMBA + (optionally) CEGIS path is
                what produces the simplification in that case.
            pipeline_mode: Pre-defined pipeline configuration; see
                :class:`PipelineMode`. ``AST`` (default) does only
                binarisation; ``SIMBA`` adds SimBA reconstruction and
                enables subtree-SimBA; ``GAMBA`` wraps SimBA in GAMBA
                pre/post and enables subtree-SimBA with the same wrap.
                Ignored when an explicit ``pipeline`` is given.
            pipeline: Optional explicit pipeline that overrides the
                preset selected by ``pipeline_mode``. Used by callers
                that need a custom composition (e.g. inserting a
                domain-specific pass between SimBA and AstNorm).
                Subtree-SimBA still tracks ``pipeline_mode``, since the
                explicit pipeline alone can't tell the simplifier
                whether subtree-SimBA should fire.
            enforce_equivalence: Flag to enforce semantic equivalence
                checks before replacements.
            solver_timeout: SMT solver timeout in seconds.
            subtree_simba_max_vars: Skip subtree SiMBA when the unification
                dict has more than this many terminals.
            subtree_simba_max_nodes: Skip subtree SiMBA when the Miasm
                graph of the subtree has more than this many nodes.
            gamba_substitution_max_k: Maximum ``k`` for the §5.1
                substitution escalation inside :func:`gamba_substitution`.
                ``0`` disables escalation (equivalent to plain subtree-
                SimBA). ``>= 1`` enables abstraction of up to ``k``
                nonlinear leaves per attempt; combinatorial gating in
                :func:`_gated_max_k` clips the effective bound based on
                the leaf count of each subtree. Default ``3`` matches
                upstream GAMBA's ``simplify_general`` cap.
            enable_cegis: Enable CEGIS constant synthesis as a last-resort
                fallback on oracle + subtree-SiMBA miss. **Off by default**
                — the CEGIS path runs Z3 against up to
                ``cegis_max_templates`` templates with a per-template
                timeout of ``cegis_timeout`` seconds, which is non-trivial.
                Turn on for workloads whose subtrees contain arbitrary
                constants the precomputed oracle cannot cover (e.g.
                ``v0 * 0xDEADBEEF + 0x1337``). See
                :mod:`msynth.simplification.cegis` for the algorithm.
            cegis_max_templates: Max templates attempted per subtree.
            cegis_timeout: Per-template Z3 timeout in seconds.
            cegis_max_variables: Skip CEGIS on subtrees with more than
                this many unified terminals.
            cegis_runtime_templates: Size of the hand-crafted runtime
                template oracle generated at construction.
            cegis_refinement_iters: Max counter-example refinement
                iterations per template attempt.
            cegis_validation_samples: Validation samples per refinement step.
            cegis_expand_templates: When True, wrap base templates with
                light constant decorations (``+c``, ``^c``, ``(&c)|c'``)
                to broaden coverage without manual enumeration.
            cegis_expansion_budget: Cap on total expanded templates.
            enable_dead_variable_elimination: Prepend a
                :class:`~msynth.simplification.dead_variable_elimination.DeadVariableEliminationPass`
                to the pipeline so it runs **before** SimBA/GAMBA. **Off by
                default.** It detects semantically dead/opaque variables by
                sampling and replaces them with ``0`` (validated by a sampling
                gate plus a short-timeout Z3 check), shrinking the variable
                count so the heavier simplifiers can fire. Sound by
                construction: it returns the original on any validation failure,
                and the final equivalence gate re-checks against the input.
            dead_variable_elimination_config: Optional
                :class:`DeadVariableEliminationConfig` overriding the pass's
                tunables. Ignored unless ``enable_dead_variable_elimination`` is
                set (which forces ``enabled=True``).
        """
        # public attributes
        self.oracle = (
            SimplificationOracle.load_from_file(oracle_path)
            if oracle_path is not None
            else SimplificationOracle.empty()
        )
        self.enforce_equivalence = enforce_equivalence
        self.solver_timeout = solver_timeout

        # Pipeline: explicit override > mode default.
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = {
                PipelineMode.AST: default_pipeline,
                PipelineMode.SIMBA: simba_pipeline,
                PipelineMode.GAMBA: gamba_pipeline,
            }[pipeline_mode]()

        # Optional dead/opaque-variable elimination, prepended so it runs BEFORE
        # SimBA/GAMBA: pruning fake variables shrinks the variable count, which
        # is what gates the heavier simplifiers. The pass self-validates and
        # returns the original on any failure; the final equivalence gate in
        # ``simplify`` (against the original input) is an additional backstop.
        self._dead_var_pass: Optional[DeadVariableEliminationPass] = None
        if enable_dead_variable_elimination:
            config = dead_variable_elimination_config or DeadVariableEliminationConfig(
                enabled=True
            )
            # The Simplifier flag is the enablement switch; honour it regardless
            # of the config object's own ``enabled`` default.
            if not config.enabled:
                config = replace(config, enabled=True)
            self._dead_var_pass = DeadVariableEliminationPass(config)
            self.pipeline = Pipeline([self._dead_var_pass, *self.pipeline.passes])

        # internal attributes
        self._translator_z3 = TranslatorZ3()
        self._solver = z3.Solver()
        self._global_variable_prefix = "global_reg"
        # Subtree-SimBA tracks the mode: enabled iff the mode runs SimBA
        # at all. Stored as a flag so _try_subtree_simba can also tell
        # whether to wrap its SimBA call with GAMBA pre/post.
        self._pipeline_mode = pipeline_mode
        self._subtree_simba_pass: Optional[SimbaPass] = (
            SimbaPass()
            if pipeline_mode in (PipelineMode.SIMBA, PipelineMode.GAMBA)
            else None
        )
        self._subtree_simba_max_vars = subtree_simba_max_vars
        self._subtree_simba_max_nodes = subtree_simba_max_nodes
        self._gamba_substitution_max_k = gamba_substitution_max_k
        self._cegis_min_nodes = cegis_min_nodes
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

        # Subtree-SimBA wrapping mirrors the pipeline mode. Under GAMBA
        # mode the global pipeline wraps SimBA with GAMBA pre/post, so the
        # subtree-level call wraps too — algebraic structure collapsed on
        # both sides exposes more linear-MBA shapes to SimBA's classifier
        # and re-collapses its conjunction-basis output. Under SIMBA mode
        # the global pipeline doesn't wrap, so the subtree-level call
        # doesn't either; subtree-SimBA gets the raw subtree.
        #
        # The actual SimBA invocation is routed through
        # :func:`gamba_substitution` (§5.1 wrapper). At ``max_k=0`` the
        # wrapper degenerates to "plain SimBA on subtree", subsuming the
        # previous direct call exactly. ``max_k`` will become a tunable in
        # follow-up work, escalating to the full §5.1 abstraction loop.
        if self._pipeline_mode == PipelineMode.GAMBA:

            def _simba_fn(arg: Expr) -> Optional[Expr]:
                preprocessed = GAMBA_PREPROCESSOR.normalize(arg)
                rewritten = self._subtree_simba_pass.run(preprocessed)
                if rewritten == preprocessed:
                    return None
                return GAMBA_POST_REWRITER.normalize(rewritten)
        else:

            def _simba_fn(arg: Expr) -> Optional[Expr]:
                rewritten = self._subtree_simba_pass.run(arg)
                if rewritten == arg:
                    return None
                return rewritten

        # §5.1 escalation budget per subtree. Default ``3`` matches upstream
        # GAMBA; callers can lower to ``0`` to recover the pre-escalation
        # subtree-SimBA-only behaviour for measurement purposes.
        simplified = gamba_substitution(
            subtree, _simba_fn, max_k=self._gamba_substitution_max_k
        )
        if simplified is None:
            return None
        # SimBA's reconstruction helpers (_sum, _or, _xor, _conjunction
        # in simba.py) emit variadic ExprOps. Re-binarise before
        # returning so the candidate respects the main loop's binary-
        # tree invariant: ``get_subexpressions`` only exposes
        # intermediate sub-pair nodes when they physically exist in
        # the AST, and ``is_strictly_smaller_tree`` compares structural
        # node counts that differ between variadic and binary shapes.
        # Without this, a variadic candidate looks artificially smaller
        # than its binary-shaped equivalent and may bypass the suitability
        # check on its raw-arity node count.
        return AbstractSyntaxTreeTranslator().from_expr(simplified)

    def _is_suitable_simplification_candidate(
        self, expr: Expr, simplified: Expr
    ) -> bool:
        """
        Checks whether a simplification candidate is suitable to accept.

        This check ensures the semantical correctness of the simplification.
        The candidate is rejected (this method returns ``False``) in any of the
        following cases:

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
            True if the candidate is suitable and should be accepted, False if
            it should be skipped.
        """
        # contains placeholder variables (p0, p1, ...): anchored full match so
        # real variables that merely start with 'p' (e.g. 'ptr') are not
        # mistaken for unification placeholders.
        if any(
            re.fullmatch(r"p[0-9]+", v.name) for v in get_unique_variables(simplified)
        ):
            return False
        # Reject concrete size regressions before doing SMT work. The helper is
        # intentionally bounded and iterative because Miasm graph construction can
        # be expensive or recursive on very large reverse-unified candidates.
        if not is_strictly_smaller_tree(simplified, expr):
            return False
        # Rigorous mode: require the SMT solver to prove equivalence.
        if self.enforce_equivalence:
            return self.check_semantical_equivalence(expr, simplified) == z3.unsat
        # Permissive mode (default): verify equivalence by evaluating both
        # expressions on deterministic edge-case inputs plus many random I/O
        # pairs. This is the fast soundness gate — far cheaper than an SMT call
        # per candidate (which dominates runtime on bitwise-heavy subtrees) while
        # still catching non-equivalent rewrites with overwhelming probability.
        return self._permissive_equivalent(expr, simplified)

    def _permissive_equivalent(self, expr: Expr, simplified: Expr) -> bool:
        """Fast random/edge-case equivalence check (no SMT).

        Returns True when the two expressions agree on every probe. If the
        expressions cannot be compiled to fast evaluators (an unusual IR shape),
        falls back to the deterministic adversarial probe so behaviour degrades
        gracefully rather than rejecting outright.
        """
        variables = sorted(
            set(get_unique_variables(expr)) | set(get_unique_variables(simplified)),
            key=lambda variable: str(variable),
        )
        try:
            evaluate_expr = compile_expr_to_python(
                _rename_variables_for_compilation(expr, variables)
            )
            evaluate_simplified = compile_expr_to_python(
                _rename_variables_for_compilation(simplified, variables)
            )
        except Exception:
            return not has_adversarial_counterexample(expr, simplified)
        for inputs in gen_adversarial_inputs(variables):
            if evaluate_expr(inputs) != evaluate_simplified(inputs):
                return False
        rng = _random.Random(0x5117BA)
        width = len(variables)
        for _ in range(_PERMISSIVE_RANDOM_PROBES):
            inputs = [rng.getrandbits(64) for _ in range(width)]
            if evaluate_expr(inputs) != evaluate_simplified(inputs):
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
        End-to-end simplification: pipeline → oracle/SimBA/CEGIS loop →
        closing rewriter.

        Five phases, in order:

        1. **Pipeline** (runs once at the top, before any subtree walk).
           ``self.pipeline.run(expr)`` produces the AST the loop walks
           over. Under ``PipelineMode.AST`` the only pass is
           :class:`AstNormalizationPass` (binarisation — required by
           :func:`get_subexpressions`). Under ``SIMBA`` / ``GAMBA`` the
           pipeline additionally applies SimBA (and, in GAMBA, the
           algebraic pre/post rewriters around it). The output is the
           "preprocessed AST".

        2. **Fixpoint loop.** A BFS walks every subtree of the
           preprocessed AST top-down. For each subtree we attempt
           candidates in three tiers (any tier hitting wins):

           a) **Oracle lookup** — gated by
              ``len(unification_dict) <= self.oracle.num_variables`` so
              we don't overflow the compiled evaluator's input matrix.
              The subtree's unified form is keyed by I/O equivalence
              class and looked up in ``self.oracle.oracle_map``. With
              the default empty oracle this branch always misses; with
              a precomputed oracle it carries most of the simplification
              load.
           b) **Subtree-SimBA** — re-runs SimBA on the subtree alone.
              Useful when the global pipeline's SimBA call rejected the
              whole expression but an inner subtree is a valid linear
              MBA. Enabled iff the pipeline mode runs SimBA at all
              (SIMBA / GAMBA); wraps with GAMBA pre/post iff the mode
              is GAMBA. See :meth:`_try_subtree_simba`.
           c) **CEGIS** — synthesises constants for a fixed family of
              templates via Z3. Opt-in (``enable_cegis=True``);
              recovers shapes whose constants the oracle cannot cover
              (e.g. ``v0 * 0xDEADBEEF + 0x1337``).

           If any tier produces a candidate that
           :meth:`_is_suitable_simplification_candidate` accepts
           (strictly smaller + Z3-equivalent under
           ``solver_timeout``), the subtree is replaced with a fresh
           ``global_reg{N}`` placeholder and the body is recorded in
           ``global_unification_dict`` for later reverse-substitution.
           This is the QSynth-inspired placeholder trick: it reduces
           the variable count of containing subtrees so that later
           oracle lookups remain within ``oracle.num_variables``.

        3. **Convergence.** The outer ``while`` loop iterates until a
           full BFS pass produces no replacement (``before == ast``).
           The placeholder substitution is monotone (subtrees only ever
           shrink or get replaced), so the fixpoint always terminates.

        4. **Reverse unification.** Each ``global_reg{N}`` is replaced
           by its recorded body. This can re-introduce shapes the
           pipeline never saw — for example, an oracle returned
           ``2 * p1 + 2 * p2`` and ``p1 / p2`` map to bitwise-equivalent
           subtrees that now sit under a common ``+``.

        5. **Closing rewriter.**
           :data:`~msynth.simplification.rewrites.DEFAULT_REWRITER`
           applies miasm's ``expr_simp`` (constant folding, commutative
           canonicalisation, slice/compose normalisation) plus the two
           guarded msynth rules (``ring_normalize``,
           ``factor_common_subterm``). This pass catches shapes that
           became simplifiable only after step 4 produced them, and is
           also the only stage that does real simplification work under
           ``PipelineMode.AST`` with an empty oracle and CEGIS off.

        Args:
            expr: Expression to simplify.

        Returns:
            Simplified expression, semantically equivalent to ``expr``
            modulo the configured equivalence-check policy
            (``enforce_equivalence`` / ``solver_timeout``).
        """
        # Phase 1: pipeline. The result is a binarised AST (always, since
        # every pipeline ends with AstNormalizationPass) plus any
        # pipeline-mode-specific simplification (SimBA, GAMBA pre/post).
        ast = self.pipeline.run(expr)
        # The pipeline output is already a sound, often-minimal reconstruction.
        # Keep it as a floor: the later BFS loop / reverse-substitution / closing
        # rewriter are all equivalence-preserving but can occasionally rebuild a
        # compact SimBA form into a larger one (e.g. expr_simp expanding
        # ``-(((y^z)&x)^(y^z))``), so the final result never exceeds this size.
        pipeline_output = ast
        # global_reg{N} -> simplified body. Reverse-substituted in
        # phase 4 after the fixpoint loop terminates.
        global_unification_dict: Dict[Expr, Expr] = {}
        global_ctr = 0

        # Phase 2 + 3: oracle/SimBA/CEGIS fixpoint loop.
        while True:
            before = ast
            # Subtrees made stale by an in-pass replacement (the replaced subtree
            # and its descendants). Tracking them lets the pass *continue*
            # instead of restarting the whole walk after every hit: the old code
            # was O(simplifications x subtrees); this is O(passes x subtrees).
            # The outer fixpoint loop still reprocesses ancestors that gained a
            # placeholder child, so the result is unchanged.
            removed: set = set()

            for subtree in get_subexpressions(ast):
                if subtree in removed:
                    continue
                # Terminals (ExprId / ExprInt / ExprLoc) have no
                # substructure to fold; skip the round-trip.
                if self._skip_subtree(subtree):
                    continue

                # Unify terminals to p0, p1, … so equivalence-class
                # lookups and SimBA/CEGIS templates see a canonical
                # variable naming.
                unification_dict = gen_unification_dict(subtree)

                simplified: Optional[Expr] = None

                # Tier (a): oracle lookup.
                # Gated by self.oracle.num_variables — the oracle's I/O
                # inputs matrix has that many columns, and subtrees with
                # more unified terminals would overflow the compiled
                # evaluator's index lookup (``i[idx]`` IndexError). The
                # default empty oracle (num_variables=3) still admits
                # this branch; the contains_equiv_class check just
                # always misses on an empty oracle_map.
                if len(unification_dict) <= self.oracle.num_variables:
                    equiv_class = self.determine_equivalence_class(
                        subtree.replace_expr(unification_dict)
                    )
                    if self.oracle.contains_equiv_class(equiv_class):
                        success, candidate = self._find_suitable_simplification(
                            equiv_class, subtree, unification_dict
                        )
                        if success:
                            simplified = candidate

                # Tier (b): subtree-level SimBA fallback. Gated by mode
                # via ``self._subtree_simba_pass is not None`` inside
                # _try_subtree_simba; under GAMBA mode the call is
                # additionally wrapped with GAMBA pre/post.
                if simplified is None:
                    candidate = self._try_subtree_simba(subtree, unification_dict)
                    if (
                        candidate is not None
                        and self._is_suitable_simplification_candidate(
                            subtree, candidate
                        )
                    ):
                        simplified = candidate

                # Tier (c): CEGIS constant synthesis on oracle +
                # subtree-SimBA miss. Opt-in via enable_cegis; recovers
                # expressions whose shape matches a template but whose
                # constants are arbitrary and therefore absent from the
                # precomputed oracle.
                if (
                    simplified is None
                    and self._cegis_solver is not None
                    # CEGIS's smallest reducible shape is the 7-node
                    # ``(x|k)+(x&k)``; a subtree below that bound has no smaller
                    # constant-bearing equivalent to recover, so skip the
                    # (per-subtree, on every real-world subtree) template walk.
                    and len(subtree.graph().nodes()) >= self._cegis_min_nodes
                ):
                    unified_subtree = subtree.replace_expr(unification_dict)
                    candidate = self._cegis_solver.try_synthesize(
                        subtree, unified_subtree, unification_dict
                    )
                    if candidate is not None:
                        # Re-binarise for the same reason as subtree-
                        # SimBA: CEGIS templates resize and instantiate
                        # variadic ExprOps. Binary form makes
                        # ``is_strictly_smaller_tree`` a fair
                        # structural comparison and keeps the
                        # placeholder body binary for the final
                        # post-pass.
                        candidate = AbstractSyntaxTreeTranslator().from_expr(candidate)
                    if (
                        candidate is not None
                        and self._is_suitable_simplification_candidate(
                            subtree, candidate
                        )
                    ):
                        simplified = candidate

                if simplified is None:
                    continue

                # Commit the simplification: substitute the subtree
                # with a fresh placeholder so later oracle lookups in
                # containing subtrees stay within
                # ``oracle.num_variables``.
                global_variable = self._gen_global_variable_replacement(
                    global_ctr, subtree.size
                )
                new_ast = ast.replace_expr({subtree: global_variable})
                if new_ast == ast:
                    # Subtree no longer present in the (incrementally rewritten)
                    # AST -- it was folded into an earlier in-pass replacement.
                    # Skip without consuming a placeholder index.
                    continue
                ast = new_ast
                global_ctr += 1
                global_unification_dict[global_variable] = simplified
                # Mark this subtree and its descendants stale for the remainder
                # of the pass; do NOT restart the walk.
                removed.update(get_subexpressions(subtree))

            if before == ast:
                break

        # Phase 4: reverse-substitute placeholders with their bodies.
        ast = self._reverse_global_unification(ast, global_unification_dict)

        # Phase 5: closing rewriter (miasm expr_simp + guarded ring /
        # factor rules). Catches shapes that only became simplifiable
        # after reverse-substitution exposed common structure across
        # previously-separate subtrees.
        rewritten = DEFAULT_REWRITER.normalize(ast)

        # Return the smallest form across the pipeline output, the post-
        # substitution AST, and the closing-rewriter output that is verified
        # equivalent to the original input. The stages are individually
        # equivalence-preserving by construction, but a faulty algebraic rule
        # anywhere in the pipeline (GAMBA pre/post rewriters, the closing
        # rewriter, …) could in principle emit a non-equivalent form; gating the
        # whole pipeline's output on a fast random/edge-case equivalence check
        # guarantees the simplifier only ever returns a correct expression, and
        # the original ``expr`` is always a sound fallback.
        equivalent = [
            candidate
            for candidate in (pipeline_output, ast, rewritten)
            if self._permissive_equivalent(expr, candidate)
        ]
        if equivalent:
            return min(equivalent, key=binarized_node_count)
        return expr
