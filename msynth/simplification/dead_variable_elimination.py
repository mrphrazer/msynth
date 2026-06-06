"""
Probabilistic opaque / dead-variable elimination.

Obfuscated MBAs frequently carry **dead / opaque / fake variables** -- inputs
that never affect the output (``x + (fake ^ fake) * y``, opaque-predicate junk,
unused VM context slots). They inflate the variable count, which is exactly what
gates the heavier simplifiers (SimBA, GAMBA, the oracle) out: those tiers only
fire on small variable counts. Pruning dead variables *before* the heavy pipeline
shrinks the problem so the existing simplifiers become effective again.

A variable ``vi`` is *semantically dead* iff changing it while all other
variables stay fixed never changes the output::

    forall rest, forall a, b:  f(rest, vi=a) == f(rest, vi=b)

If that holds, ``vi`` can be replaced by any constant (we use ``0``). This pass
finds likely-dead variables by randomized + adversarial + AST-constant sampling
(the *sensitivity scan*), replaces all candidates in bulk, post-simplifies, and
then **validates** the transformed expression against the original before
accepting it. The validation is a two-stage gate:

1. a strong *sampling* check (adversarial edge cases + many seeded randoms), and
2. a short-timeout *Z3* check on ``original != candidate`` with the policy
   **SAT (counterexample) => reject, UNSAT => accept, UNKNOWN (timeout) =>
   accept** -- identical to the CEGIS soundness gate
   (:meth:`msynth.simplification.cegis.CegisSolver._z3_counterexample`).

If the bulk replacement is rejected (at least one candidate was a false
positive), a block-splitting / delta-debugging *repair* recovers the largest
sound subset. The pass returns a modified expression **only** when the final
gate passes against the *original*; otherwise it returns the original unchanged,
so it is always safe to compose in front of an aggressive pipeline.

The pass is a duck-typed pipeline pass (``run(expr) -> expr``); see
:mod:`msynth.simplification.pipeline`. msynth simplifies a single
:class:`~miasm.expression.expression.Expr` (no multi-output / state model), so
outputs are compared as scalars.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import z3
from miasm.expression.expression import Expr, ExprInt
from miasm.ir.translators.z3_ir import TranslatorZ3

from msynth.simplification.rewrites import DEFAULT_REWRITER
from msynth.utils.expr_utils import (
    compile_expr_to_python,
    get_subexpressions,
    get_unique_variables,
)
from msynth.utils.sampling import (
    _rename_variables_for_compilation,
    gen_adversarial_inputs,
    gen_adversarial_values,
)


class _BudgetExceeded(Exception):
    """Raised internally when the per-expression wall-time budget is hit."""


class VariableStatus(Enum):
    """Classification of a variable produced by the sensitivity scan."""

    LIVE = "live"
    PROBABLY_DEAD = "probably_dead"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DeadVariableEliminationConfig:
    """
    Tunables for :class:`DeadVariableEliminationPass`.

    Attributes:
        enabled: Master switch for the standalone pass. When the pass is wired
            into the Simplifier the flag there controls enablement; this field
            keeps the pass usable as a self-contained object.
        outer_rounds: Number of independent random contexts (assignments to the
            *other* variables) used when testing one variable.
        mutations_per_variable: Number of distinct values the tested variable is
            mutated through inside each context (on top of all harvested AST
            constants, which are always included).
        validation_randoms: Number of seeded random rows in the final sampling
            gate (in addition to the deterministic adversarial rows).
        use_adversarial_values: Include bit-vector edge cases (0/1/sign-bit/
            mask/wraparound tail) in the value pools.
        use_ast_constants: Harvest integer constants from the AST and probe
            ``c, c+-1, ~c, -c, c^mask`` -- catches variables that only become
            live near magic constants (``0xDEADBEEF``, ``0x80000000``, ...).
        replacement_value: Constant substituted for dead variables (default 0).
        repair: Recover a sound subset via block-splitting when the bulk
            replacement fails validation.
        z3_timeout_ms: Per-query Z3 timeout for the acceptance gate. Kept small;
            on timeout the gate accepts (see module docstring).
        allow_sampling_fallback_on_unknown: When Z3 returns UNKNOWN, accept iff
            the sampling gate ran clean. With sampling unavailable *and* Z3
            unknown the transform is rejected (never accepted unvalidated).
        max_variables: Skip the pass above this many variables (None = no cap).
        min_variables: Skip the pass below this many variables.
        max_mutation_values: Hard cap on the per-variable mutation set. Bounds
            the scan when an expression carries a large number of distinct AST
            constants (each expands to several probe values); without it the
            scan can blow up on constant-heavy compose/slice MBAs.
        time_budget_s: Per-expression wall-clock budget (None = unbounded). When
            the scan / validation / repair exceeds it the pass aborts and
            returns the original unchanged. This guarantees the pass never
            stalls on pathological expressions (e.g. ones whose compiled
            evaluator is slow); aborting only ever *keeps* a variable, so it is
            sound. Checked cooperatively between evaluations.
        min_variables: Skip the pass below this many variables.
        random_seed: Seed for all sampling (reproducible runs).
    """

    enabled: bool = False
    outer_rounds: int = 12
    mutations_per_variable: int = 16
    validation_randoms: int = 2048
    use_adversarial_values: bool = True
    use_ast_constants: bool = True
    replacement_value: int = 0
    repair: bool = True
    z3_timeout_ms: int = 50
    allow_sampling_fallback_on_unknown: bool = True
    max_variables: Optional[int] = None
    max_mutation_values: int = 64
    time_budget_s: Optional[float] = 1.0
    min_variables: int = 2
    random_seed: int = 0


@dataclass
class VariableTestResult:
    """Per-variable result of the sensitivity scan (debugging / tuning)."""

    variable: Expr
    status: VariableStatus
    tested_contexts: int
    tested_mutations: int
    observed_changes: int
    first_witness: Optional[List[int]] = None


@dataclass
class DeadVariableEliminationResult:
    """Summary of one :meth:`DeadVariableEliminationPass.run` invocation."""

    original_variable_count: int
    candidate_count: int
    accepted_count: int
    candidates: List[Expr]
    accepted_variables: List[Expr]
    rejected_variables: List[Expr]
    final_equivalent: bool
    variable_results: List[VariableTestResult] = field(default_factory=list)


class DeadVariableEliminationPass:
    """
    Pipeline pass that eliminates semantically dead variables.

    Duck-typed pass: exposes ``name`` and ``run(expr) -> Expr``. ``run`` returns
    a pruned-and-validated expression, or the original unchanged when nothing is
    dead or the validation gate fails -- so it is sound to prepend to any
    :class:`~msynth.simplification.pipeline.Pipeline`.

    The last invocation's metadata is available on :attr:`last_result`.
    """

    name = "dead_variable_elimination"

    def __init__(
        self,
        config: Optional[DeadVariableEliminationConfig] = None,
        *,
        post_simplify: Optional[Callable[[Expr], Expr]] = None,
    ) -> None:
        self.config = config or DeadVariableEliminationConfig(enabled=True)
        # Default post-simplifier: the closing rewriter (miasm expr_simp + ring/
        # factor). Cheap; collapses ``(0 ^ 0) * 0 -> 0`` so the pruning's benefit
        # is visible to the gate and to downstream passes.
        self._post_simplify = post_simplify or DEFAULT_REWRITER.normalize
        self._translator = TranslatorZ3()
        self.last_result: Optional[DeadVariableEliminationResult] = None
        # Wall-clock deadline for the current ``run`` (None = unbounded).
        self._deadline: Optional[float] = None

    def _budget_check(self) -> None:
        if self._deadline is not None and time.monotonic() > self._deadline:
            raise _BudgetExceeded()

    # ------------------------------------------------------------------ #
    # Pipeline-pass entry point
    # ------------------------------------------------------------------ #
    def run(self, expr: Expr) -> Expr:
        cfg = self.config
        if not cfg.enabled:
            return expr

        variables = get_unique_variables(expr)
        count = len(variables)
        if count < cfg.min_variables:
            return expr
        if cfg.max_variables is not None and count > cfg.max_variables:
            return expr

        try:
            original_eval = compile_expr_to_python(
                _rename_variables_for_compilation(expr, variables)
            )
        except Exception:
            # Unsupported IR shape (e.g. memory / opaque op) -> cannot evaluate
            # safely; leave the expression untouched.
            return expr

        self._deadline = (
            time.monotonic() + cfg.time_budget_s
            if cfg.time_budget_s is not None
            else None
        )
        try:
            return self._run(expr, variables, original_eval)
        except _BudgetExceeded:
            # Ran out of time analysing this expression. Aborting only ever
            # keeps variables, so returning the original is sound.
            self.last_result = DeadVariableEliminationResult(
                original_variable_count=count,
                candidate_count=0,
                accepted_count=0,
                candidates=[],
                accepted_variables=[],
                rejected_variables=[],
                final_equivalent=True,
            )
            return expr

    def _run(
        self,
        expr: Expr,
        variables: Sequence[Expr],
        original_eval: Callable[[List[int]], int],
    ) -> Expr:
        cfg = self.config
        count = len(variables)
        rng = random.Random(cfg.random_seed)
        ast_constants = (
            self._collect_ast_constants(expr) if cfg.use_ast_constants else {}
        )
        adversarial_cache = self._build_adversarial_cache(variables)

        # Sensitivity scan.
        variable_results: List[VariableTestResult] = []
        candidates: List[Expr] = []
        for index, variable in enumerate(variables):
            result = self._test_variable(
                original_eval,
                variables,
                index,
                ast_constants,
                adversarial_cache,
                rng,
            )
            variable_results.append(result)
            if result.status == VariableStatus.PROBABLY_DEAD:
                candidates.append(variable)

        if not candidates:
            self.last_result = DeadVariableEliminationResult(
                original_variable_count=count,
                candidate_count=0,
                accepted_count=0,
                candidates=[],
                accepted_variables=[],
                rejected_variables=[],
                final_equivalent=True,
                variable_results=variable_results,
            )
            return expr

        # Bulk replacement + post-simplify, then validate against the ORIGINAL.
        candidate_expr = self._post_simplify(self._replace(expr, candidates))
        ok, counterexample = self._check(
            expr, candidate_expr, variables, eval_original=original_eval
        )
        if ok:
            self.last_result = DeadVariableEliminationResult(
                original_variable_count=count,
                candidate_count=len(candidates),
                accepted_count=len(candidates),
                candidates=list(candidates),
                accepted_variables=list(candidates),
                rejected_variables=[],
                final_equivalent=True,
                variable_results=variable_results,
            )
            return candidate_expr

        if not cfg.repair:
            self.last_result = DeadVariableEliminationResult(
                original_variable_count=count,
                candidate_count=len(candidates),
                accepted_count=0,
                candidates=list(candidates),
                accepted_variables=[],
                rejected_variables=list(candidates),
                final_equivalent=False,
                variable_results=variable_results,
            )
            return expr

        # Repair: recover the largest sound subset.
        accepted = self._repair(
            expr, candidates, variables, counterexample, original_eval
        )
        if accepted:
            repaired = self._post_simplify(self._replace(expr, accepted))
            ok_repaired, _ = self._check(
                expr, repaired, variables, eval_original=original_eval
            )
            if ok_repaired:
                rejected = [v for v in candidates if v not in accepted]
                self.last_result = DeadVariableEliminationResult(
                    original_variable_count=count,
                    candidate_count=len(candidates),
                    accepted_count=len(accepted),
                    candidates=list(candidates),
                    accepted_variables=list(accepted),
                    rejected_variables=rejected,
                    final_equivalent=True,
                    variable_results=variable_results,
                )
                return repaired

        self.last_result = DeadVariableEliminationResult(
            original_variable_count=count,
            candidate_count=len(candidates),
            accepted_count=0,
            candidates=list(candidates),
            accepted_variables=[],
            rejected_variables=list(candidates),
            final_equivalent=False,
            variable_results=variable_results,
        )
        return expr

    # ------------------------------------------------------------------ #
    # Sensitivity scan
    # ------------------------------------------------------------------ #
    def _test_variable(
        self,
        eval_f: Callable[[List[int]], int],
        variables: Sequence[Expr],
        index: int,
        ast_constants: Dict[int, List[int]],
        adversarial_cache: Dict[int, List[int]],
        rng: random.Random,
    ) -> VariableTestResult:
        """Vary only ``variables[index]``; LIVE on first output change."""
        cfg = self.config
        variable = variables[index]
        width = variable.size
        mutation_values = self._mutation_values(
            width, ast_constants.get(width, ()), adversarial_cache.get(width, ()), rng
        )

        contexts = 0
        mutations = 0
        for _ in range(cfg.outer_rounds):
            self._budget_check()
            row = self._gen_context(variables, ast_constants, adversarial_cache, rng)
            # Anchor the tested variable, observe the baseline output.
            row[index] = 0
            base_output = eval_f(row)
            for value in mutation_values:
                if value == 0:
                    continue
                trial = list(row)
                trial[index] = value
                mutations += 1
                if mutations % 64 == 0:
                    self._budget_check()
                if eval_f(trial) != base_output:
                    return VariableTestResult(
                        variable=variable,
                        status=VariableStatus.LIVE,
                        tested_contexts=contexts + 1,
                        tested_mutations=mutations,
                        observed_changes=1,
                        first_witness=trial,
                    )
            contexts += 1

        return VariableTestResult(
            variable=variable,
            status=VariableStatus.PROBABLY_DEAD,
            tested_contexts=contexts,
            tested_mutations=mutations,
            observed_changes=0,
            first_witness=None,
        )

    def _gen_context(
        self,
        variables: Sequence[Expr],
        ast_constants: Dict[int, List[int]],
        adversarial_cache: Dict[int, List[int]],
        rng: random.Random,
    ) -> List[int]:
        """A full assignment for all variables, mixing structured + random."""
        row: List[int] = []
        for variable in variables:
            width = variable.size
            roll = rng.random()
            ast_pool = ast_constants.get(width)
            adv_pool = adversarial_cache.get(width)
            if ast_pool and roll < 0.3:
                row.append(rng.choice(ast_pool))
            elif adv_pool and roll < 0.6:
                row.append(rng.choice(adv_pool))
            else:
                row.append(rng.getrandbits(width))
        return row

    def _mutation_values(
        self,
        width: int,
        ast_pool: Sequence[int],
        adv_pool: Sequence[int],
        rng: random.Random,
    ) -> List[int]:
        """
        Values to mutate the tested variable through.

        AST-derived constants are prioritised (they encode magic-constant gates
        random/adversarial values would miss); on top of those, up to
        ``mutations_per_variable`` adversarial + random values. The whole set is
        capped at ``max_mutation_values`` so a constant-heavy expression (many
        distinct AST constants) cannot blow up the scan -- if the gating
        constant is dropped by the cap, the Z3 acceptance gate is the backstop.
        """
        cap = self.config.max_mutation_values
        # Deterministic AST sample under the cap (keep a budget for non-AST too).
        ast_budget = max(0, cap - self.config.mutations_per_variable)
        values: List[int] = list(dict.fromkeys(ast_pool))[:ast_budget]
        seen = set(values)
        budget = min(cap, len(values) + self.config.mutations_per_variable)

        for value in adv_pool:
            if len(values) >= budget:
                break
            if value not in seen:
                values.append(value)
                seen.add(value)
        # Fill the remainder with fresh randoms. Bound the attempts: for a
        # small-width variable the value space (e.g. {0, 1} for a 1-bit flag)
        # can be smaller than ``budget``, in which case no new random will ever
        # appear and an unbounded loop would spin forever.
        attempts = 0
        max_attempts = budget * 4 + 8
        while len(values) < budget and attempts < max_attempts:
            attempts += 1
            candidate = rng.getrandbits(width)
            if candidate not in seen:
                values.append(candidate)
                seen.add(candidate)
        return values

    # ------------------------------------------------------------------ #
    # Value harvesting
    # ------------------------------------------------------------------ #
    @staticmethod
    def _collect_ast_constants(expr: Expr) -> Dict[int, List[int]]:
        """Harvest integer constants -> {width: [c, c+-1, ~c, -c, c^mask]}."""
        by_width: Dict[int, set] = {}
        for sub in get_subexpressions(expr):
            if sub.is_int():
                width = sub.size
                mask = (1 << width) - 1
                value = int(sub) & mask
                pool = by_width.setdefault(width, set())
                pool.update(
                    {
                        value,
                        (value - 1) & mask,
                        (value + 1) & mask,
                        (~value) & mask,
                        (-value) & mask,
                        value ^ mask,
                    }
                )
        return {width: sorted(values) for width, values in by_width.items()}

    def _build_adversarial_cache(
        self, variables: Sequence[Expr]
    ) -> Dict[int, List[int]]:
        if not self.config.use_adversarial_values:
            return {}
        return {
            variable.size: gen_adversarial_values(variable.size)
            for variable in variables
        }

    # ------------------------------------------------------------------ #
    # Replacement
    # ------------------------------------------------------------------ #
    def _replace(self, expr: Expr, dead: Sequence[Expr]) -> Expr:
        return expr.replace_expr(
            {
                variable: ExprInt(self.config.replacement_value, variable.size)
                for variable in dead
            }
        )

    # ------------------------------------------------------------------ #
    # Validation gate: sampling, then short-timeout Z3
    # ------------------------------------------------------------------ #
    def _check(
        self,
        original: Expr,
        candidate: Expr,
        variables: Sequence[Expr],
        *,
        eval_original: Optional[Callable[[List[int]], int]] = None,
    ) -> Tuple[bool, Optional[List[int]]]:
        """
        Decide whether ``candidate`` may replace ``original``.

        Returns ``(accept, counterexample)``. A returned counterexample (from
        sampling or Z3) is a full input row over ``variables`` on which the two
        disagree -- used to guide repair. ``eval_original`` lets the caller pass
        the already-compiled original evaluator (it is constant across a run, so
        repair avoids recompiling it every iteration).
        """
        sampling_ran = False
        try:
            if eval_original is None:
                eval_original = compile_expr_to_python(
                    _rename_variables_for_compilation(original, variables)
                )
            eval_candidate = compile_expr_to_python(
                _rename_variables_for_compilation(candidate, variables)
            )
            sampling_ran = True
        except Exception:
            pass

        if sampling_ran:
            counterexample = self._sampling_counterexample(
                eval_original, eval_candidate, variables
            )
            if counterexample is not None:
                return False, counterexample

        verdict, counterexample = self._z3_verdict(original, candidate, variables)
        if verdict == "counter":
            return False, counterexample
        if verdict == "equiv":
            return True, None
        # verdict == "unknown": accept only if sampling vouched for it.
        if sampling_ran and self.config.allow_sampling_fallback_on_unknown:
            return True, None
        return False, None

    def _sampling_counterexample(
        self,
        eval_original: Callable[[List[int]], int],
        eval_candidate: Callable[[List[int]], int],
        variables: Sequence[Expr],
    ) -> Optional[List[int]]:
        """Deterministic edge cases + seeded randoms; first disagreement wins."""
        for row in gen_adversarial_inputs(list(variables)):
            if eval_original(row) != eval_candidate(row):
                return row
        rng = random.Random(self.config.random_seed ^ 0x5EED)
        for iteration in range(self.config.validation_randoms):
            if iteration % 64 == 0:
                self._budget_check()
            row = [rng.getrandbits(variable.size) for variable in variables]
            if eval_original(row) != eval_candidate(row):
                return row
        return None

    def _z3_verdict(
        self, original: Expr, candidate: Expr, variables: Sequence[Expr]
    ) -> Tuple[str, Optional[List[int]]]:
        """
        Short-timeout Z3 query on ``original != candidate``.

        Returns ``("equiv", None)`` on UNSAT, ``("counter", row)`` on SAT (the
        disagreeing assignment), ``("unknown", None)`` on timeout or any error.
        """
        try:
            renamed_original = _rename_variables_for_compilation(
                original, list(variables)
            )
            renamed_candidate = _rename_variables_for_compilation(
                candidate, list(variables)
            )
            z3_original = self._translator.from_expr(renamed_original)
            z3_candidate = self._translator.from_expr(renamed_candidate)
            solver = z3.Solver()
            solver.set("timeout", self.config.z3_timeout_ms)
            solver.add(z3_original != z3_candidate)
            result = solver.check()
        except Exception:
            return "unknown", None

        if result == z3.unsat:
            return "equiv", None
        if result == z3.sat:
            model = solver.model()
            row: List[int] = []
            for index, variable in enumerate(variables):
                bitvec = z3.BitVec(f"p{index}", variable.size)
                row.append(model.eval(bitvec, model_completion=True).as_long())
            return "counter", row
        return "unknown", None

    # ------------------------------------------------------------------ #
    # Repair: block-splitting / delta-debugging
    # ------------------------------------------------------------------ #
    def _repair(
        self,
        original: Expr,
        candidates: Sequence[Expr],
        variables: Sequence[Expr],
        counterexample: Optional[List[int]],
        eval_original: Optional[Callable[[List[int]], int]] = None,
    ) -> List[Expr]:
        """
        Find the largest subset of ``candidates`` that can be replaced soundly.

        Accept whole blocks when the subset still validates; otherwise split the
        failing block in half. 1-element failing blocks are dropped (cannot be
        eliminated). The counterexample (if any) orders the search so variables
        that were *active* on it -- the likely false positives -- are isolated
        last.
        """
        ordered = self._order_by_counterexample(candidates, variables, counterexample)
        accepted: List[Expr] = []
        worklist: List[List[Expr]] = [ordered]
        while worklist:
            self._budget_check()
            block = worklist.pop()
            trial = accepted + block
            trial_expr = self._post_simplify(self._replace(original, trial))
            ok, _ = self._check(
                original, trial_expr, variables, eval_original=eval_original
            )
            if ok:
                accepted.extend(block)
                continue
            if len(block) == 1:
                continue
            mid = len(block) // 2
            # Push the second half first so the first half is processed next
            # (depth-first, deterministic).
            worklist.append(block[mid:])
            worklist.append(block[:mid])
        return accepted

    @staticmethod
    def _order_by_counterexample(
        candidates: Sequence[Expr],
        variables: Sequence[Expr],
        counterexample: Optional[List[int]],
    ) -> List[Expr]:
        if counterexample is None:
            return list(candidates)
        index_of = {variable: i for i, variable in enumerate(variables)}

        def active_on_counterexample(variable: Expr) -> int:
            position = index_of.get(variable)
            if position is None or position >= len(counterexample):
                return 0
            # Suspicious (non-zero at the counterexample) -> sort last.
            return 1 if counterexample[position] != 0 else 0

        return sorted(candidates, key=active_on_counterexample)
