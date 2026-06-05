"""
CEGIS helpers for constant synthesis.

This module implements a runtime-only template oracle and a CEGIS solver.
It is designed to augment oracle-based simplification with constant recovery:

- Templates contain placeholders c0, c1, ... for unknown constants.
- Subtrees are unified to p0, p1, ... before solving.
- Z3 is used to solve the constants from a handful of I/O samples; the
  candidate is then validated by counterexample-guided I/O sampling. CEGIS does
  NOT prove equivalence itself — equivalence is enforced by the caller's
  suitability gate (``Simplifier._is_suitable_simplification_candidate``:
  strictly-smaller + Z3 equivalence with an adversarial edge-probe, permissive
  on Z3 timeout), the single acceptance check shared with the oracle and
  subtree-SimBA tiers. A self-proof here would time out on obfuscated MBA
  subtrees and make CEGIS give up on otherwise-correct candidates.

The solver is conservative by design and meant to be used as a fallback when
the main oracle lookup fails.

When to use CEGIS
-----------------
CEGIS targets the case where a subtree's *shape* is one of a handful of
common MBA patterns but its *constants* are arbitrary and therefore absent
from any precomputed equivalence-class oracle. Typical example:
``v0 * 0xDEADBEEF + 0x1337`` matches the template ``p0 * c0 + c1``; Z3
solves ``c0`` and ``c1`` from the subtree's observed I/O behaviour.

This is orthogonal to the two other simplification paths used alongside
CEGIS in :class:`msynth.simplification.simplifier.Simplifier`:

* The **precomputed oracle** covers fixed-constant expressions in the
  library it was built from. Limited to constants seen at build time.
* **SiMBA** (global and subtree-level) handles *linear MBAs* whose
  Boolean structure encodes pure boolean/arithmetic identities, but does
  not synthesise unknown constants.
* **CEGIS** handles "known shape + unknown constants" by template-driven
  Z3 solving with counter-example refinement.

The defaults in :func:`TemplateOracle.gen_runtime_oracle` are tuned for
three-variable templates with an 8-bit truncated lookup key. Raising
``template_bits`` increases key precision (fewer false bucket matches) at
the cost of bucket density; raising ``num_variables`` enlarges the runtime
template set.
"""

import logging
import random
import re
import time
from typing import Dict, Iterable, Iterator, List, Optional

import z3
from miasm.expression.expression import (
    Expr,
    ExprAssign,
    ExprCompose,
    ExprCond,
    ExprId,
    ExprInt,
    ExprMem,
    ExprOp,
    ExprSlice,
)
from miasm.ir.translators.z3_ir import TranslatorZ3

from msynth.simplification.oracle import SimplificationOracle, calc_hash
from msynth.utils.expr_utils import compile_expr_to_python, get_unique_variables
from msynth.utils.sampling import gen_inputs
from msynth.utils.unification import reverse_unification


def _make_evaluator(expr: Expr):
    """Return a fast callable ``inputs -> int`` for a unified expression.

    Uses :func:`compile_expr_to_python` (native arithmetic over a ``p0,p1,…``
    input vector) so an expression is compiled once and evaluated many times,
    instead of the per-call ``expr_simp`` round-trip in
    ``SimplificationOracle.evaluate_expression``. Falls back to the slow
    Miasm evaluator for the rare expression shape the compiler rejects.
    """
    try:
        return compile_expr_to_python(expr)
    except Exception:
        return lambda inputs, _e=expr: SimplificationOracle.evaluate_expression(
            _e, inputs
        )


logger = logging.getLogger("msynth.cegis")


class TemplateOracle:
    """
    Template oracle for constant synthesis.

    Each entry maps a truncated I/O behavior (key) to a list of expression
    templates where constants were replaced by symbolic placeholders (c0, c1, ...).

    This runtime-only oracle is meant to be small (dozens of rules) and fast
    to query. It supplies candidate templates for the CEGIS solver, which then
    tries to instantiate the placeholders to match observed I/O behavior.

    Key ideas:
    - The oracle key is computed from *truncated* outputs (template_bits),
      which reduces the search space while still preserving enough structure
      to find plausible templates.
    - Templates use unified variables (p0, p1, ...) and placeholder constants
      (c0, c1, ...). The solver binds constants while the caller later reverses
      unification to map pN back to original terminals.

    Attributes:
        template_bits: Bit-width used to compute the truncated oracle key.
        num_variables: Number of variables expected in templates (p0..pN-1).
        num_samples: Number of input samples used to define behavior.
        inputs: Concrete input samples for evaluating expressions.
        oracle_map: Mapping from oracle key -> list of templates.
    """

    def __init__(
        self,
        template_bits: int,
        num_variables: int,
        num_samples: int,
        inputs: List[List[int]],
        oracle_map: Dict[str, List[Expr]],
    ) -> None:
        """
        Initializes a template oracle.

        Args:
            template_bits: Bit-width for truncated output keys.
            num_variables: Number of unified variables expected (p0..pN-1).
            num_samples: Number of stored I/O samples.
            inputs: Concrete input samples used to evaluate expressions.
            oracle_map: Mapping from key -> list of templates.
        """
        self.template_bits = template_bits
        self.num_variables = num_variables
        self.num_samples = num_samples
        self.inputs = inputs
        self.oracle_map = oracle_map
        # Skeleton index: hash of the template's operator tree (with constants
        # and constant-placeholders treated as wildcards) -> list of templates.
        # Lets the solver look up "templates whose shape matches this target's
        # shape" in O(1), independent of the constants involved. The
        # hand-crafted runtime templates are indexed here at construction;
        # callers can extend the oracle via :meth:`add_template`.
        self._skeleton_buckets: Dict[str, List[Expr]] = {}
        for templates in oracle_map.values():
            for template in templates:
                self._index_skeleton(template)

    @staticmethod
    def skeleton_key(expr: Expr) -> str:
        """
        Structural hash of ``expr``'s operator tree.

        Both ``ExprInt`` nodes and constant-placeholder ``ExprId`` nodes
        (named ``c0, c1, ...``) collapse to the wildcard ``<HOLE>``;
        variable-placeholder ``ExprId`` nodes (``p0, p1, ...``) keep their
        positional index so a template only matches a target with the same
        variable-arity at the same positions; everything else collapses to
        its type name.

        Two expressions share a skeleton iff they could be unified by
        replacing constants with each other while preserving the operator
        structure and variable positions.
        """
        return calc_hash(TemplateOracle._skeleton_signature(expr))

    # Operators that commute in bit-vector arithmetic: arguments can appear
    # in any order at the AST level, so the skeleton signature sorts their
    # sub-signatures to make matching order-independent.
    _COMMUTATIVE_OPS = frozenset({"+", "*", "^", "&", "|"})

    @staticmethod
    def _skeleton_signature(expr: Expr) -> str:
        """Recursive signature builder; the hashable input to ``skeleton_key``."""
        if expr.is_int():
            return "<HOLE>"
        if expr.is_id():
            name = expr.name
            if re.match(r"^c\d+$", name):
                return "<HOLE>"
            if re.match(r"^p\d+$", name):
                # Keep position; two templates with the same shape but
                # different variable positions must not collide.
                return f"<p{name[1:]}>"
            return f"<id:{name}>"
        if isinstance(expr, ExprOp):
            arg_sigs = [TemplateOracle._skeleton_signature(arg) for arg in expr.args]
            if expr.op in TemplateOracle._COMMUTATIVE_OPS:
                arg_sigs.sort()
            args = ",".join(arg_sigs)
            return f"({expr.op} {args})"
        if isinstance(expr, ExprSlice):
            return f"(slice {expr.start}:{expr.stop} {TemplateOracle._skeleton_signature(expr.arg)})"
        if isinstance(expr, ExprCond):
            return (
                f"(cond {TemplateOracle._skeleton_signature(expr.cond)} "
                f"{TemplateOracle._skeleton_signature(expr.src1)} "
                f"{TemplateOracle._skeleton_signature(expr.src2)})"
            )
        if isinstance(expr, ExprCompose):
            args = ",".join(
                TemplateOracle._skeleton_signature(arg) for arg in expr.args
            )
            return f"(compose {args})"
        if isinstance(expr, ExprMem):
            return f"(mem {expr.size} {TemplateOracle._skeleton_signature(expr.ptr)})"
        return f"<{type(expr).__name__}>"

    def _index_skeleton(self, template: Expr) -> None:
        """Insert ``template`` into the skeleton bucket for its shape."""
        key = self.skeleton_key(template)
        self._skeleton_buckets.setdefault(key, []).append(template)

    def get_skeleton_templates(self, expr: Expr) -> Iterator[Expr]:
        """
        Iterate templates whose skeleton matches ``expr``'s skeleton.

        Empty iterator when no template's shape matches; the caller is
        expected to fall through to the I/O-keyed lookup in that case.
        """
        key = self.skeleton_key(expr)
        for template in self._skeleton_buckets.get(key, []):
            yield template

    def determine_equiv_key(self, outputs: Iterable[int]) -> str:
        """
        Computes the template oracle key for a list of outputs.

        The outputs are truncated to template_bits and hashed together with the
        bit-width to minimize collisions across different sizes.

        Args:
            outputs: Output values produced by the candidate expression.

        Returns:
            Hash key used to retrieve matching templates.
        """
        # Truncate outputs to the template bit-width before hashing.
        mask = (1 << self.template_bits) - 1 if self.template_bits > 0 else 0
        truncated = [int(output) & mask for output in outputs]
        identifier = str(self.template_bits)
        return calc_hash(identifier + str(truncated))

    def get_templates(self, equiv_key: str) -> Iterator[Expr]:
        """
        Returns templates for a given equivalence key.

        Args:
            equiv_key: Hash key computed from truncated outputs.

        Returns:
            Iterator over candidate templates.
        """
        for template in self.oracle_map.get(equiv_key, []):
            yield template

    def all_templates(self) -> Iterator[Expr]:
        """
        Iterates over all templates contained in the oracle.

        This is used as a fallback when the key is not present (e.g., when the
        truncated behavior is too coarse). It is bounded by the caller.
        """
        for templates in self.oracle_map.values():
            for template in templates:
                yield template

    def add_template(self, template: Expr) -> None:
        """
        Adds a template to the runtime oracle.

        Writes the template to both the synthetic ``"*"`` bucket (so it is
        reachable via :meth:`all_templates`) and to its skeleton bucket
        (so :meth:`get_skeleton_templates` finds it for later targets of
        the same shape).

        Args:
            template: Unified template with placeholders (c0, c1, ...).
        """
        self.oracle_map.setdefault("*", []).append(template)
        self._index_skeleton(template)

    def get_outputs(self, expr: Expr) -> List[int]:
        """
        Evaluates an expression on the template oracle inputs.

        Args:
            expr: Expression to evaluate (unified form expected).

        Returns:
            List of integer outputs, aligned with self.inputs.
        """
        # Evaluate on the fixed sample inputs for this oracle.
        return [
            SimplificationOracle.evaluate_expression(expr, inputs)
            for inputs in self.inputs
        ]

    @staticmethod
    def _gen_runtime_templates(
        template_bits: int,
        num_variables: int,
        max_placeholders: int,
        template_budget: int,
    ) -> List[Expr]:
        """
        Generates a small set of hand-crafted template rules at runtime.

        Templates are intentionally simple (binary ops, affine forms, masks,
        shifts) and resemble the patterns commonly found in MBA expressions.
        The goal is to cover a wide range of constant placement shapes without
        becoming too large for per-subtree solving.

        Args:
            template_bits: Template bit-width (placeholder + variable sizes).
            num_variables: Number of variables (p0, p1, p2).
            max_placeholders: Maximum number of constant placeholders.
            template_budget: Hard cap on number of templates returned.

        Returns:
            List of template expressions.
        """
        # Runtime templates use unified variables (p0, p1, ...) and placeholders.
        vars_list = [ExprId(f"p{i}", template_bits) for i in range(num_variables)]
        c0 = ExprId("c0", template_bits)
        c1 = ExprId("c1", template_bits)
        c2 = ExprId("c2", template_bits)

        templates: List[Expr] = []
        if num_variables == 0:
            return templates

        # Track the highest constant index each template needs so the
        # ``max_placeholders`` cap can drop the ones that need too many.
        def add(expr: Expr, max_const: int = 1) -> None:
            templates.append((expr, max_const))  # type: ignore[arg-type]

        p0 = vars_list[0]
        # One-variable affine / mask / shift style templates.
        add(p0 + c0, 0)
        add(p0 - c0, 0)
        add(p0 ^ c0, 0)
        add(p0 * c0, 0)
        add(p0 & c0, 0)
        add(p0 | c0, 0)
        add((p0 & c0) | c1, 1)
        add((p0 & c0) ^ c1, 1)
        add((p0 & c0) + c1, 1)
        add((p0 | c0) + c1, 1)
        add((c0 * p0) + c1, 1)
        add((p0 ^ c0) + c1, 1)
        add((p0 + c0) ^ c1, 1)
        add((p0 + c0) + c1, 1)
        add(p0 << c0, 0)
        add(p0 >> c0, 0)
        add((c0 * p0) ^ c1, 1)
        # ``c0*p0*p0`` (quadratic in one variable) — affine in c0.
        add((c0 * (p0 * p0)) + c1, 1)
        # Full single-variable quadratic ``c0*x^2 + c1*x + c2`` (affine in the
        # three coefficients) — recovers an expanded/obfuscated polynomial.
        add((c0 * (p0 * p0)) + (c1 * p0) + c2, 2)
        add((p0 * p0) + c0, 0)
        add((p0 & c0) + (p0 & c1), 1)

        if num_variables >= 2:
            p1 = vars_list[1]
            # Two-variable linear/bitwise templates plus constant mixing.
            add(p0 + p1, 0)
            add(p0 ^ p1, 0)
            add(p0 | p1, 0)
            add(p0 & p1, 0)
            add((p0 + p1) + c0, 0)
            add((p0 ^ p1) ^ c0, 0)
            add((p0 ^ p1) + c0, 0)
            add((p0 & p1) + c0, 0)
            add((p0 | p1) + c0, 0)
            add((p0 + p1) ^ c0, 0)
            add((p0 & c0) + p1, 0)
            add((p0 ^ c0) + p1, 0)
            add(p0 + (p1 & c0), 0)
            add(p0 + (p1 ^ c0), 0)
            add((p0 & c0) | (p1 & c1), 1)
            add((p0 & c0) ^ (p1 & c1), 1)
            add((p0 | c0) + (p1 | c1), 1)
            add((p0 ^ c0) + (p1 ^ c1), 1)
            # Multi-coefficient affine and products (solved fast by _solve_affine).
            add((c0 * p0) + p1 + c1, 1)
            add((c0 * p0) + (c1 * p1), 1)
            add((c0 * p0) + (c1 * p1) + c2, 2)
            add((p0 * p1) + c0, 0)
            add((c0 * (p0 * p1)) + c1, 1)
            # Two-variable multiply-xor ``(c0*x) ^ (c1*y)`` (non-affine: the
            # bit-serial lifter solves it) — recovers an xor of scaled variables
            # hidden behind an ``(a|b)-(a&b)`` identity.
            add((c0 * p0) ^ (c1 * p1), 1)
            add((p0 * p1) + p0 + c0, 0)
            add((c0 * (p0 ^ p1)) + c1, 1)
            add((p0 * p0) + p1, 0)
            add((p0 & c0) | p1, 0)
            add((p0 * p1) + p1 + c0, 0)
            # Cross-term product (non-affine: c0*c1, c0*p1, c1*p0) — Z3 fallback.
            add((p0 + c0) * (p1 + c1), 1)

        if num_variables >= 3:
            p2 = vars_list[2]
            # Three-variable blends common in MBA patterns.
            add(p0 + p1 + p2, 0)
            add(p0 + p1 + p2 + c0, 0)
            add(p0 ^ p1 ^ p2, 0)
            add((p0 ^ p1 ^ p2) ^ c0, 0)
            add(p0 + (p1 & p2), 0)
            add(p0 ^ (p1 | p2), 0)
            add((p0 & p1) + p2, 0)
            add((p0 | p1) + p2, 0)
            add((p0 ^ p1) + p2, 0)
            add((p0 + p1) ^ p2, 0)
            add((p0 & p1) + p2 + c0, 0)
            add((c0 * p0) + p1 + p2, 0)
            add((p0 * p1) + p2 + c0, 0)
            add((c0 * p0) + (c1 * p1) + (c2 * p2), 2)
            add((c0 * (p0 + p1 + p2)) + c1, 1)
            add((p0 * p1 * p2) + c0, 0)
            add((p0 * p1) + (p0 * p2) + c0, 0)

        # Enforce the placeholder budget: drop templates needing more constants
        # than allowed (``max_placeholders`` counts c0..c{max-1}).
        kept: List[Expr] = []
        for expr, max_const in templates:  # type: ignore[misc]
            if max_const < max_placeholders:
                kept.append(expr)
        return kept[:template_budget]

    @staticmethod
    def gen_runtime_oracle(
        template_bits: int = 8,
        num_variables: int = 3,
        num_samples: int = 32,
        max_placeholders: int = 3,
        template_budget: int = 256,
    ) -> "TemplateOracle":
        """
        Generates a small template oracle at runtime without a precomputed library.

        The returned oracle uses "*" as a synthetic key containing all templates,
        while determine_equiv_key is still available for key-based lookups.

        Args:
            template_bits: Bit-width used for the truncated oracle key.
            num_variables: Number of variables for runtime templates.
            num_samples: Number of I/O samples to generate.
            max_placeholders: Maximum placeholders (c0, c1) in templates.
            template_budget: Maximum number of runtime templates to keep.

        Returns:
            TemplateOracle instance with runtime templates.
        """
        # Fixed sample inputs define the truncated oracle behavior.
        inputs = gen_inputs(num_variables, num_samples)
        templates = TemplateOracle._gen_runtime_templates(
            template_bits, num_variables, max_placeholders, template_budget
        )
        return TemplateOracle(
            template_bits,
            num_variables,
            num_samples,
            inputs,
            {"*": templates},
        )


class CegisSolver:
    """
    CEGIS-based constant synthesis using a template oracle.

    The solver takes a unified subtree (p0, p1, ...) and attempts to match its
    I/O behavior against a small set of templates that contain placeholder
    constants (c0, c1, ...). For each template, it uses Z3 to solve for the
    placeholders and then runs a counterexample-guided refinement loop to
    validate the candidate on additional samples.

    CEGIS additions:
    - Counterexample refinement: validation samples are added when a candidate
      fails, and the solver re-runs with the expanded sample set.
    - Adaptive template expansion: small wrappers are generated on-demand to
      broaden coverage without a large static rule set.

    The returned expression is still unified; the caller reverses unification to
    map pN back to original terminals.
    """

    def __init__(
        self,
        template_oracle: TemplateOracle,
        max_templates: int = 200,
        solver_timeout: int = 2,
        max_variables: int = 3,
        refinement_iters: int = 3,
        validation_samples: int = 16,
        expand_templates: bool = True,
        expansion_budget: int = 40,
        seed: int = 0,
    ) -> None:
        """
        Initializes the CEGIS solver.

        Args:
            template_oracle: Runtime template oracle to query.
            max_templates: Max templates to try per subtree.
            solver_timeout: Z3 timeout per template (seconds).
            max_variables: Max unified variables in target subtree.
            refinement_iters: Max counterexample refinement iterations.
            validation_samples: Samples per validation iteration.
            expand_templates: Enable adaptive template expansion.
            expansion_budget: Max expanded templates to consider.
            seed: Seed for the validation-sampling RNG. Sampling drives
                counterexample-guided refinement and candidate acceptance here;
                final equivalence is enforced by the caller's suitability gate.
                A fixed seed keeps the refinement path reproducible across runs.
        """
        self.template_oracle = template_oracle
        self.max_templates = max_templates
        self.solver_timeout = solver_timeout
        self.max_variables = max_variables
        self.refinement_iters = refinement_iters
        self.validation_samples = validation_samples
        self.expand_templates = expand_templates
        self.expansion_budget = expansion_budget
        # Structural synthesis tier (abstract the target's own constants and
        # solve) is what recovers clean, never-templated shapes. It is decoupled
        # from ``expand_templates`` (which only governs the wrapper bloat added
        # to the static walk) so the wrappers can be dropped for speed without
        # losing the synthesis tier.
        self.structural_synthesis = expand_templates
        self._translator_z3 = TranslatorZ3()
        self._rng = random.Random(seed)
        # Separate RNG for the affine solver's affinity probes, so its draws do
        # not perturb the deterministic validation-input sequence (``self._rng``).
        self._affine_rng = random.Random(seed ^ 0x9E3779B97F4A7C15)
        # Cache of (evaluator, placeholders, base, resized) per (template, size).
        # Templates are fixed, so each is resized + compiled once and reused
        # across every target subtree instead of recompiling per call.
        self._template_cache: Dict[tuple, tuple] = {}
        # Per-call ceiling (ms) on a single Z3 constant-solve, and a per-target
        # cumulative budget (ms) across all Z3 attempts in one ``try_synthesize``.
        # The affine and closed-form solvers handle the common templates; Z3 is
        # only a fallback for mixed arith+bitwise constant positions, where a
        # matching template (tried first via the skeleton index) solves within
        # the per-call cap, while a miss is bounded by the cumulative budget.
        self._z3_timeout_ms = 150
        self._z3_target_budget_ms = 300
        # Budget for the exact exists-forall constant-synthesis escalation. It
        # only fires after a Z3 disproof flags a masked/underdetermined constant
        # (a handful of times over a whole corpus), so a generous budget here
        # buys reliable hard-tail coverage without touching the common-case path.
        self._z3_forall_timeout_ms = 250
        # Side-channel: which solver produced the most recent _solve_template
        # result ('affine'/'bitwise'/'bit_serial'/'z3'/None). Lets _solve_validated
        # skip the Z3 equivalence gate for exact (affine) solves.
        self._last_solve_kind = None
        # Side-channels: whether the most recent bit-serial / bitwise solve was
        # forced at every bit (a uniquely-determined constant).
        self._bit_serial_unique = False
        self._bitwise_unique = False
        # Cache of (target_expr, its Z3 translation) for the equivalence gate.
        self._z3_target_cache: tuple = (None, None)
        # Cache of structural class ('shift'/'bitwise'/'mixed'/'arith') per
        # template, keyed like _template_cache by (id(template), size).
        self._class_cache: Dict[tuple, str] = {}
        # Truncated-I/O prefilter for the tier-3 template walk. Low-bits-closed
        # ops (+ - * & | ^ ~ and constant <<) make the low ``_io_k`` bits of a
        # result depend only on the low ``_io_k`` bits of the inputs/constants,
        # so a template can match a target only if their ``_io_k``-bit behaviour
        # over a fixed probe set agrees for *some* constant assignment. The
        # achievable behaviour set per template is enumerated once per bit-width
        # (cached in ``_walk_cache``); a target's behaviour then selects only the
        # compatible templates instead of solving all ~100.
        # Number of oracle base inputs to seed the per-template solve with (the
        # rest of the discriminating signal comes from the structured probes).
        self._solve_base_inputs = 12
        self._io_k = 4
        self._io_kmask = (1 << self._io_k) - 1
        self._io_filter_inputs = [
            (1, 2, 3), (3, 5, 7), (7, 1, 4), (5, 6, 2), (2, 7, 5),
            (4, 3, 6), (6, 4, 1), (0, 1, 2), (15, 8, 11), (9, 14, 5),
        ]
        self._walk_cache: Dict[int, tuple] = {}
        # Max placeholder constants the synthesis tier will abstract from a
        # target's own skeleton (bounds the bit-serial 2^ncols per-bit search).
        self._synthesis_max_constants = 6

    @staticmethod
    def _resize_expr(expr: Expr, size: int) -> Expr:
        """
        Rebuilds an expression with all leaves resized to a target bit-width.

        This keeps the operator structure intact while ensuring all ExprInt
        and ExprId nodes have the desired size. It is used to lift runtime
        templates (typically 8-bit) to the subtree's bit-width.

        Args:
            expr: Expression to resize.
            size: Target bit-width.

        Returns:
            Resized expression.
        """
        if expr.is_id():
            return ExprId(expr.name, size)
        if expr.is_int():
            return ExprInt(int(expr), size)
        if isinstance(expr, ExprOp):
            return ExprOp(
                expr.op, *[CegisSolver._resize_expr(arg, size) for arg in expr.args]
            )
        if isinstance(expr, ExprSlice):
            return ExprSlice(
                CegisSolver._resize_expr(expr.arg, size), expr.start, expr.stop
            )
        if isinstance(expr, ExprCond):
            return ExprCond(
                CegisSolver._resize_expr(expr.cond, size),
                CegisSolver._resize_expr(expr.src1, size),
                CegisSolver._resize_expr(expr.src2, size),
            )
        if isinstance(expr, ExprCompose):
            args = [CegisSolver._resize_expr(arg, size) for arg in expr.args]
            result = args[0]
            for arg in args[1:]:
                result = ExprCompose(result, arg)
            return result
        if isinstance(expr, ExprAssign):
            return ExprAssign(
                CegisSolver._resize_expr(expr.dst, size),
                CegisSolver._resize_expr(expr.src, size),
            )
        if isinstance(expr, ExprMem):
            # The "value" of a memory access is the loaded word; resizing
            # means changing the access width to ``size``. The pointer is an
            # address expression and must keep its own width untouched.
            return ExprMem(expr.ptr, size)
        return expr

    @staticmethod
    def _placeholder_vars(expr: Expr, prefix: str) -> List[ExprId]:
        """
        Returns placeholder variables matching a prefix (p or c).

        Args:
            expr: Expression to scan.
            prefix: Variable prefix (e.g., "p" or "c").

        Returns:
            Sorted list of placeholder variables.
        """
        variables = []
        for v in get_unique_variables(expr):
            if v.is_id() and re.match(rf"^{prefix}[0-9]+$", v.name):
                variables.append(v)  # type: ignore[arg-type]
        return sorted(variables, key=lambda v: int(v.name[len(prefix) :]))

    @staticmethod
    def _evaluate_unified(expr: Expr, inputs: List[int]) -> int:
        """
        Evaluates a unified expression (p0, p1, ...) for a concrete input vector.

        Args:
            expr: Unified expression to evaluate.
            inputs: Concrete input vector.

        Returns:
            Integer result of evaluation.
        """
        # Use SimplificationOracle evaluation to respect Miasm semantics.
        return SimplificationOracle.evaluate_expression(expr, inputs)

    def _gen_validation_inputs(
        self, num_vars: int, size: int, count: int
    ) -> List[List[int]]:
        """
        Generates a deterministic prefix of inputs, then fills with random values.

        The random tail is drawn from the solver's seeded RNG (``self._rng``),
        so a given solver instance produces the same validation inputs across
        runs.

        Args:
            num_vars: Number of unified variables.
            size: Bit-width to sample.
            count: Number of input vectors to produce.

        Returns:
            List of input vectors.
        """
        if count <= 0 or num_vars <= 0:
            return []
        mask = (1 << size) - 1 if size > 0 else 0
        base = [
            [0 for _ in range(num_vars)],
            [1 & mask for _ in range(num_vars)],
            [2 & mask for _ in range(num_vars)],
        ]
        inputs = base[: min(len(base), count)]
        while len(inputs) < count:
            inputs.append([self._rng.getrandbits(size) & mask for _ in range(num_vars)])
        return inputs

    def _structured_probe_inputs(self, num_vars: int, size: int) -> List[List[int]]:
        """Bit-discriminating input vectors that pin down constant placeholders.

        A constant's effect can be *masked* on random inputs (e.g. in
        ``(x & c) | y`` the bit of ``c`` is invisible wherever ``y``'s bit is
        1), leaving it underdetermined and the solved candidate wrong on inputs
        the random sample happened to miss. These vectors isolate each variable
        — driving one variable across full / alternating bit patterns while the
        others are held at 0 (or all-ones) — so every bit position of every
        constant is exercised against a discriminating input. For ``(x & c) | y``
        the row ``x=all-ones, y=0`` reveals ``c`` in full. Used both to seed the
        solver (so it recovers constants correctly the first time) and to harden
        validation.
        """
        if num_vars <= 0 or size <= 0:
            return []
        mask = (1 << size) - 1
        alt = 0
        for bit in range(0, size, 2):
            alt |= 1 << bit
        # Full-ones plus the two alternating masks catch bit-position-dependent
        # constants (e.g. a constant only visible on odd bit positions).
        patterns = [mask, alt & mask, (~alt) & mask]
        rows: List[List[int]] = [[0] * num_vars, [mask] * num_vars]
        for i in range(num_vars):
            for pat in patterns:
                isolate_high = [0] * num_vars
                isolate_high[i] = pat
                rows.append(isolate_high)
                others_high = [mask] * num_vars
                others_high[i] = (~pat) & mask
                rows.append(others_high)
        # A few independent random vectors (own RNG, so the validation sampler's
        # stream is left untouched) to break ties the structured rows can't.
        for _ in range(4):
            rows.append(
                [self._affine_rng.getrandbits(size) & mask for _ in range(num_vars)]
            )
        seen = set()
        deduped: List[List[int]] = []
        for row in rows:
            key = tuple(row)
            if key not in seen:
                seen.add(key)
                deduped.append(row)
        return deduped

    def _validate_candidate(
        self,
        target_eval,
        candidate: Expr,
        num_vars: int,
        size: int,
    ) -> Optional[List[int]]:
        """
        Validates a candidate and returns a counterexample input if found.

        Args:
            target_eval: Compiled evaluator for the ground-truth unified subtree.
            candidate: Candidate unified expression.
            num_vars: Number of variables.
            size: Bit-width.

        Returns:
            Counterexample input vector if validation fails, otherwise None.
        """
        candidate_eval = _make_evaluator(candidate)
        # Cheap, high-yield checks: structured probes (catch masked constants on
        # independent vars) then the random/edge sample. This rejects most wrong
        # candidates without a Z3 call. The authoritative soundness check is the
        # Z3 equivalence gate applied by the caller when sampling passes; it is
        # exact (modulo timeout) and strictly stronger than any finite sample,
        # so power-of-2 bit-walking is reserved for the *solve* set (escalation
        # in :meth:`_solve_validated`) rather than spent here on every candidate.
        for inputs in self._structured_probe_inputs(num_vars, size):
            if target_eval(inputs) != candidate_eval(inputs):
                return inputs
        for inputs in self._gen_validation_inputs(
            num_vars, size, self.validation_samples
        ):
            if target_eval(inputs) != candidate_eval(inputs):
                return inputs
        return None

    def _synthesize_constants_forall(
        self, resized: Expr, unified_subtree: Expr, size: int
    ) -> Optional[Expr]:
        """Exact constant synthesis via an exists-forall SMT query.

        Solves ``Exists c. ForAll p. template(p, c) == target(p)`` directly: the
        placeholder constants are free BitVecs, the input variables are
        universally quantified, and the body equates the template to the target.
        A model gives constants that make the template **provably** equivalent to
        the target for *all* inputs — not merely consistent with a finite sample
        — so it resolves the coupled, masked two-constant shapes (e.g.
        ``(x + c0) & (x & c1)``) that sample-based lifting cannot pin down.

        This is the article's "SMT lifts the constants" step in its complete
        form. It is reserved for the escalation path (the cheap closed-form and
        bit-serial solvers carry the common cases, including the 64-bit ``bvmul``
        shapes this quantified query would choke on) and bounded by the Z3
        timeout, so an undecidable case simply declines. Returns the
        constant-instantiated candidate, or ``None``.
        """
        placeholders = self._placeholder_vars(resized, "c")
        if not placeholders:
            return None
        try:
            solver = z3.Solver()
            solver.set("timeout", self._z3_forall_timeout_ms)
            p_vars = self._placeholder_vars(unified_subtree, "p")
            p_bvs = [z3.BitVec(p.name, size) for p in p_vars]
            template_z3 = self._translator_z3.from_expr(resized)
            target_z3 = self._translator_z3.from_expr(unified_subtree)
            if p_bvs:
                solver.add(z3.ForAll(p_bvs, template_z3 == target_z3))
            else:
                solver.add(template_z3 == target_z3)
            if solver.check() != z3.sat:
                return None
            model = solver.model()
            mask = (1 << size) - 1 if size > 0 else 0
            repl = {}
            for placeholder in placeholders:
                bv = z3.BitVec(placeholder.name, size)
                value = model.eval(bv, model_completion=True).as_long() & mask
                repl[placeholder] = ExprInt(value, size)
            return resized.replace_expr(repl)
        except Exception:
            return None

    def _z3_counterexample(
        self, target: Expr, candidate: Expr, num_vars: int, size: int
    ) -> Optional[List[int]]:
        """Final soundness gate: ask Z3 for an input where ``target`` and
        ``candidate`` disagree.

        Both expressions are fully concrete (constants already solved), so this
        is a plain bitvector equivalence query — fast for the bitwise / linear /
        constant-multiply shapes where masked constant bits make sampling
        unreliable (e.g. ``(x - c) | (y | k)``: ``c``'s bits are hidden wherever
        ``y|k`` is 1, so a wrong ``c`` agrees with the target on almost every
        random input yet is not equivalent). A returned vector is the *ideal*
        counterexample-guided-refinement witness — it pins exactly the bit the
        sample missed, so the next solve converges on the true constant.

        Returns the disagreeing input vector, or ``None`` when Z3 proves
        equivalence **or** cannot decide within the timeout. ``None`` on timeout
        is deliberately permissive — it mirrors the caller's authoritative
        suitability gate, which also treats a Z3 timeout as non-blocking, so a
        genuinely-equivalent but hard-to-prove MBA candidate is not discarded.
        """
        try:
            # The target is identical across every gate call within one
            # ``try_synthesize`` (only the candidate changes per template /
            # refinement), so cache its Z3 translation by object identity.
            if self._z3_target_cache[0] is target:
                t = self._z3_target_cache[1]
            else:
                t = self._translator_z3.from_expr(target)
                self._z3_target_cache = (target, t)
            c = self._translator_z3.from_expr(candidate)
            solver = z3.Solver()
            solver.set("timeout", self._z3_timeout_ms)
            solver.add(t != c)
            result = solver.check()
            if result != z3.sat:
                # unsat == proven equivalent; unknown == timed out -> permissive.
                return None
            model = solver.model()
            counterexample = []
            for i in range(num_vars):
                bv = z3.BitVec(f"p{i}", size)
                counterexample.append(
                    model.eval(bv, model_completion=True).as_long()
                )
            return counterexample
        except Exception:
            # Any translation/solver failure: defer to sampling (already passed).
            return None

    def _bitwalk_probe_inputs(self, num_vars: int, size: int) -> List[List[int]]:
        """Power-of-2 bit-walking probes: one variable set to ``2**b`` (all bit
        positions), the others held at 0 and at all-ones.

        These expose constant bits that masking makes invisible to random
        sampling. ``x & -x`` is the lowest set bit of ``x``, so for ``(x & c) &
        (-x)`` only ``x == 2**b`` reveals ``c``'s bit ``b``; the diagonal
        ``every var == 2**b`` row covers same-variable couplings across operands.
        """
        if num_vars <= 0 or size <= 0:
            return []
        mask = (1 << size) - 1
        rows: List[List[int]] = []
        for b in range(size):
            bit = 1 << b
            diag_low = [bit] * num_vars
            rows.append(diag_low)
            for i in range(num_vars):
                row = [0] * num_vars
                row[i] = bit
                rows.append(row)
                row_hi = [mask] * num_vars
                row_hi[i] = bit
                rows.append(row_hi)
        return rows

    def _expand_templates(self, templates: List[Expr], size: int) -> List[Expr]:
        """
        Expands templates with light wrappers to add coverage.

        Layout: every base template's bare form lands in the output first,
        then wrappers are appended round-by-round (one round = "apply
        wrapper W to every base"). This guarantees that even high-index
        base templates such as ``(c0 * p0) + c1`` reach the iteration
        pool before the budget runs out — the previous design exhausted
        the budget on five wrappers of the first eight bases, leaving
        every subsequent base unreachable.

        Wrapper placeholders use the names ``c{N}`` for N starting at the
        smallest unused index across the base template set. Base
        templates use ``c0`` and ``c1``, so wrappers use ``c2``/``c3``;
        Z3 then solves wrapper constants independently of the inner
        template's constants instead of forcing them to share a value.
        The previous design reused the names ``c0``/``c1`` which made
        every expanded variant a *strict subset* of what the base
        template already matched (the wrapper had to satisfy the same
        Z3 variable as one of the inner placeholders).

        ``expansion_budget`` is now the *additional* template count
        beyond the base set. Setting it to 0 is equivalent to
        ``expand_templates=False``.

        Args:
            templates: Base templates to expand.
            size: Bit-width for the expansion.

        Returns:
            Bases (verbatim) followed by up to ``expansion_budget``
            wrapper variants.
        """
        if not templates or self.expansion_budget <= 0:
            return [self._resize_expr(t, size) for t in templates]

        resized = [self._resize_expr(t, size) for t in templates]
        # Fresh wrapper placeholders. ``c2``/``c3`` keep the ``c``
        # prefix so :meth:`_placeholder_vars` discovers them via the
        # existing ``^c[0-9]+$`` regex and Z3 solves them.
        c2 = ExprId("c2", size)
        c3 = ExprId("c3", size)
        wrappers = [
            lambda r: r + c2,
            lambda r: r ^ c2,
            lambda r: (r & c2) | c3,
            lambda r: (r + c2) ^ c3,
        ]

        out: List[Expr] = list(resized)
        added = 0
        for wrap in wrappers:
            for r in resized:
                if added >= self.expansion_budget:
                    return out
                out.append(wrap(r))
                added += 1
        return out

    @staticmethod
    def _solve_linear_mod(
        rows: List[List[int]], rhs: List[int], ncols: int, size: int
    ):
        """Solve ``rows · x == rhs`` over Z/2^size by odd-pivot elimination.

        Returns ``("solved", x)`` with a consistent solution; ``("unsat", None)``
        when odd pivots exist for every column but the over-determined system is
        inconsistent (provably no constants satisfy this affine template — the
        caller can skip Z3); or ``("underdetermined", None)`` when some column has
        no odd pivot (this method can't decide; the caller may still try Z3).
        """
        modulus = 1 << size
        mask = modulus - 1
        # Augmented matrix (mutable copy): each row is [coeff_0..coeff_{n-1}, rhs].
        aug = [list(r) + [b & mask] for r, b in zip(rows, rhs)]
        pivot_rows: List[int] = []
        used = [False] * len(aug)
        for col in range(ncols):
            pivot = -1
            for i in range(len(aug)):
                if not used[i] and (aug[i][col] & 1):
                    pivot = i
                    break
            if pivot == -1:
                return ("underdetermined", None)
            used[pivot] = True
            pivot_rows.append(pivot)
            inv = pow(aug[pivot][col], -1, modulus)
            aug[pivot] = [(value * inv) & mask for value in aug[pivot]]
            for i in range(len(aug)):
                if i == pivot:
                    continue
                factor = aug[i][col]
                if factor:
                    pr = aug[pivot]
                    aug[i] = [
                        (aug[i][j] - factor * pr[j]) & mask for j in range(ncols + 1)
                    ]
        solution = [0] * ncols
        for col, pivot in enumerate(pivot_rows):
            solution[col] = aug[pivot][ncols] & mask
        # Consistency: every original row must satisfy the solution.
        for r, b in zip(rows, rhs):
            acc = 0
            for j in range(ncols):
                acc = (acc + r[j] * solution[j]) & mask
            if acc != (b & mask):
                return ("unsat", None)
        return ("solved", solution)

    def _cached_template(self, template: Expr, size: int) -> tuple:
        """Resize + compile a template's affine evaluator once, then reuse it.

        Returns ``(evaluator, placeholders, base, resized)`` where ``evaluator``
        takes a combined ``[p0..p_{base-1}, c0..c_{ncols-1}]`` input vector
        (constants renamed to fresh ``p`` indices so it compiles natively).
        """
        key = (id(template), size)
        cached = self._template_cache.get(key)
        if cached is not None:
            return cached
        resized = self._resize_expr(template, size)
        placeholders = self._placeholder_vars(resized, "c")
        base = self.template_oracle.num_variables
        rename = {
            c: ExprId(f"p{base + j}", size) for j, c in enumerate(placeholders)
        }
        evaluator = _make_evaluator(resized.replace_expr(rename))
        result = (evaluator, placeholders, base, resized)
        self._template_cache[key] = result
        return result

    def _template_class(self, resized: Expr, key: tuple) -> str:
        """Cached structural class of a template: one of ``shift`` (a constant
        shift amount), ``bitwise`` (pure ``&``/``|``/``^``), ``mixed`` (bitwise
        combined with arithmetic) or ``arith`` (pure arithmetic).

        The three structural walks behind this dispatch are otherwise repeated on
        every solve of the same (stable) template; caching them by the same
        ``(id, size)`` key as the compiled evaluator removes that per-call cost.
        """
        klass = self._class_cache.get(key)
        if klass is None:
            if self._has_constant_shift(resized):
                klass = "shift"
            elif self._is_pure_bitwise(resized):
                klass = "bitwise"
            elif self._contains_bitwise_op(resized):
                klass = "mixed"
            else:
                klass = "arith"
            self._class_cache[key] = klass
        return klass

    def _solve_affine(
        self,
        evaluator,
        placeholders: List[ExprId],
        base: int,
        outputs: List[int],
        inputs: List[List[int]],
        size: int,
    ) -> Optional[Dict[ExprId, int]]:
        """Fast constant solve when the template is affine in its constants.

        Many templates (``p+c``, ``c*p+c1``, ``c*x*y``, ``x*y+c``, ``c0*x+c1*y``,
        sums of variables plus constants) are affine in the placeholder vector
        ``c`` over Z/2^size: ``T(p,c) = A(p) + Σ_j c_j·M_j(p)``. The columns
        ``M_j`` and offset ``A`` are read off by evaluating the template at
        ``c=0`` and ``c=e_j``; affinity is verified on random ``c`` vectors
        (catching cross-terms like ``c0·c1``); the constants then fall out of a
        small linear system solved over Z/2^size. No SMT involved.
        """
        mask = (1 << size) - 1
        ncols = len(placeholders)
        if ncols == 0 or not inputs:
            return None

        def pad(row: List[int]) -> List[int]:
            if len(row) >= base:
                return list(row[:base])
            return list(row) + [0] * (base - len(row))

        padded = [pad(row) for row in inputs]

        def eval_all(c_values: List[int]) -> Optional[List[int]]:
            # Defensive against any arithmetic fault (e.g. a division/shift edge
            # in a future template) — a faulting evaluation just abandons the
            # affine attempt rather than crashing the solver.
            try:
                return [evaluator(row + c_values) for row in padded]
            except (ZeroDivisionError, ValueError, OverflowError):
                return None

        zeros = [0] * ncols
        offset = eval_all(zeros)
        if offset is None:
            return "not_affine"
        columns: List[List[int]] = []
        for j in range(ncols):
            unit = [0] * ncols
            unit[j] = 1
            evaluated = eval_all(unit)
            if evaluated is None:
                return "not_affine"
            columns.append([(evaluated[i] - offset[i]) & mask for i in range(len(inputs))])
        # Verify affinity on random constant vectors. A failure means the
        # template is not affine in its constants (xor/mask/shift positions);
        # signal that with the ``"not_affine"`` sentinel so the caller tries Z3.
        for _ in range(3):
            sample_c = [self._affine_rng.getrandbits(size) & mask for _ in range(ncols)]
            actual = eval_all(sample_c)
            if actual is None:
                return "not_affine"
            for i in range(len(inputs)):
                predicted = offset[i]
                for j in range(ncols):
                    predicted = (predicted + sample_c[j] * columns[j][i]) & mask
                if actual[i] != predicted:
                    return "not_affine"
        matrix = [[columns[j][i] for j in range(ncols)] for i in range(len(inputs))]
        residual = [(outputs[i] - offset[i]) & mask for i in range(len(inputs))]
        status, solution = self._solve_linear_mod(matrix, residual, ncols, size)
        if status == "solved":
            return {placeholders[j]: solution[j] & mask for j in range(ncols)}
        if status == "unsat":
            # Affine template, but provably no constants reproduce the target's
            # behaviour — Z3 would only confirm UNSAT slowly. Skip it.
            return "unsat"
        return "not_affine"  # underdetermined here; let Z3 attempt it

    @staticmethod
    def _is_pure_bitwise(expr: Expr) -> bool:
        """True iff ``expr`` uses only bit-parallel operators (``&``/``|``/``^``).

        ``~`` is represented as ``x ^ all_ones`` so it counts as ``^``. Such an
        expression is bit-independent: output bit ``b`` depends only on input and
        constant bit ``b``, which lets every constant bit be solved in isolation.
        """
        if isinstance(expr, ExprOp):
            if expr.op not in ("&", "|", "^"):
                return False
            return all(CegisSolver._is_pure_bitwise(arg) for arg in expr.args)
        return True

    @staticmethod
    def _contains_bitwise_op(expr: Expr) -> bool:
        """True iff any ``&``/``|``/``^`` op appears in the expression.

        A placeholder constant inside a bitwise op never enters affinely, so a
        template containing one cannot be solved by the affine solver — used to
        route it straight to the bit-serial lifter without an eval-based probe.
        """
        stack = [expr]
        while stack:
            node = stack.pop()
            if isinstance(node, ExprOp):
                if node.op in ("&", "|", "^"):
                    return True
                stack.extend(node.args)
            elif isinstance(node, ExprSlice):
                stack.append(node.arg)
            elif isinstance(node, ExprCompose):
                stack.extend(node.args)
            elif isinstance(node, ExprCond):
                stack.extend((node.cond, node.src1, node.src2))
        return False

    def _solve_bitwise(
        self,
        evaluator,
        placeholders: List[ExprId],
        base: int,
        outputs: List[int],
        inputs: List[List[int]],
        size: int,
    ) -> Optional[Dict[ExprId, int]]:
        """Closed-form constant solve for bit-independent (pure-bitwise) templates.

        Because the template is bit-parallel, evaluating it with each constant
        set to all-zeros or all-ones (one evaluation per pattern of the ncols
        constants, ``2^ncols`` total) yields, for every output bit position, the
        bit-function value under that pattern. Each output bit's constant bits are
        then read off independently; an unmatched bit means no constants satisfy
        the samples, so this fails fast on a non-matching template (no SMT).
        """
        ncols = len(placeholders)
        mask = (1 << size) - 1
        if ncols == 0 or not inputs:
            return None

        def pad(row: List[int]) -> List[int]:
            if len(row) >= base:
                return list(row[:base])
            return list(row) + [0] * (base - len(row))

        padded = [pad(row) for row in inputs]
        patterns = 1 << ncols
        # table[pattern][i] = template output for sample i with c_j = mask if
        # bit j of pattern set else 0.
        table = []
        try:
            for pattern in range(patterns):
                c_values = [mask if (pattern >> j) & 1 else 0 for j in range(ncols)]
                table.append([evaluator(row + c_values) for row in padded])
        except (ZeroDivisionError, ValueError, OverflowError):
            return None

        solved = [0] * ncols
        unique = True  # every bit pinned by exactly one constant pattern
        for bit in range(size):
            target = [(outputs[i] >> bit) & 1 for i in range(len(inputs))]
            matched = -1
            n_consistent = 0
            for pattern in range(patterns):
                column = table[pattern]
                if all(((column[i] >> bit) & 1) == target[i] for i in range(len(inputs))):
                    n_consistent += 1
                    if matched == -1:
                        matched = pattern
            if matched == -1:
                self._bitwise_unique = False
                return None
            if n_consistent != 1:
                unique = False
            for j in range(ncols):
                if (matched >> j) & 1:
                    solved[j] |= 1 << bit
        self._bitwise_unique = unique
        return {placeholders[j]: solved[j] & mask for j in range(ncols)}

    @staticmethod
    def _has_constant_shift(expr: Expr) -> bool:
        """True iff a constant placeholder is used as a shift *amount*.

        For such templates the bit-serial solver is invalid (the amount's bits
        affect every output bit, not just the bits at and above their position),
        so those fall back to Z3.
        """
        if isinstance(expr, ExprOp):
            if expr.op in ("<<", ">>", "a>>") and len(expr.args) == 2:
                amount = expr.args[1]
                if any(
                    v.is_id() and re.fullmatch(r"c[0-9]+", v.name)
                    for v in get_unique_variables(amount)
                ):
                    return True
            return any(CegisSolver._has_constant_shift(a) for a in expr.args)
        return False

    def _solve_bit_serial(
        self,
        evaluator,
        placeholders: List[ExprId],
        base: int,
        outputs: List[int],
        inputs: List[List[int]],
        size: int,
    ) -> Optional[Dict[ExprId, int]]:
        """General constant solve by Hensel-style bit lifting (LSB -> MSB).

        For any template built from ``+ - * & | ^ ~`` (no shift-amount
        constants), output bit ``b`` depends only on constant bits ``<= b``
        (carries from lower bits are already fixed once those bits are solved).
        So the constant bits at each position are found by trying the ``2^ncols``
        patterns for that position and keeping those consistent with every
        sample's bit ``b``; ambiguities branch (bounded). This solves shapes Z3
        cannot at 64-bit — multiply-then-xor, degree-2 cross-terms, etc.
        """
        ncols = len(placeholders)
        mask = (1 << size) - 1
        if ncols == 0 or not inputs:
            return None

        def pad(row: List[int]) -> List[int]:
            if len(row) >= base:
                return list(row[:base])
            return list(row) + [0] * (base - len(row))

        padded = [pad(row) for row in inputs]
        target_bits = [
            [(outputs[i] >> bit) & 1 for i in range(len(inputs))] for bit in range(size)
        ]
        patterns = 1 << ncols
        node_budget = [4096]  # bound on total search nodes (ambiguity guard)
        # Whether the accepted path was forced at every bit (exactly one
        # consistent pattern). If so the samples admit a *single* assignment, so
        # in the synthesis tier — where the target IS this template at the true
        # constants — that assignment must be the true one, and the Z3
        # equivalence gate can be skipped. Any branch flips this to False.
        deterministic = [True]

        def recurse(bit: int, solved: List[int]) -> Optional[List[int]]:
            if bit == size:
                return list(solved)
            if node_budget[0] <= 0:
                return None
            consistent = []
            for pattern in range(patterns):
                node_budget[0] -= 1
                trial = [
                    solved[j] | (((pattern >> j) & 1) << bit) for j in range(ncols)
                ]
                want = target_bits[bit]
                try:
                    ok = all(
                        ((evaluator(row + trial) >> bit) & 1) == want[i]
                        for i, row in enumerate(padded)
                    )
                except (ZeroDivisionError, ValueError, OverflowError):
                    return None
                if ok:
                    consistent.append(pattern)
            if len(consistent) != 1:
                deterministic[0] = False
            for pattern in consistent:
                trial = [
                    solved[j] | (((pattern >> j) & 1) << bit) for j in range(ncols)
                ]
                result = recurse(bit + 1, trial)
                if result is not None:
                    return result
            return None

        solution = recurse(0, [0] * ncols)
        if solution is None:
            self._bit_serial_unique = False
            return None
        self._bit_serial_unique = deterministic[0]
        return {placeholders[j]: solution[j] & mask for j in range(ncols)}

    def _solve_template(
        self,
        template: Expr,
        outputs: List[int],
        inputs: List[List[int]],
        size: int,
    ) -> Optional[Dict[ExprId, int]]:
        """
        Solves constant placeholders for a template.

        Tries the fast affine solver first (closed-form linear algebra over
        Z/2^size, no SMT) and only falls back to Z3 for templates whose
        constants enter non-affinely (masks, shifts, xor positions, nested
        cross-terms). Z3 path: each placeholder is a BitVec of ``size`` bits;
        for every I/O sample the variables p0,p1,… are substituted with the
        concrete inputs and the result constrained to the expected output.

        Args:
            template: Unified expression template with placeholders.
            outputs: Expected outputs for each input sample.
            inputs: Input sample list aligned with outputs.
            size: Bit-width to solve for.

        Returns:
            Mapping from placeholder ExprId to solved integer value, or None.
        """
        evaluator, placeholders, base, resized = self._cached_template(template, size)
        if not placeholders:
            self._last_solve_kind = None
            return None

        # Structural dispatch (cached, no evaluation) picks the one applicable
        # solver instead of trying the eval-based affine probe on every template:
        #   * a constant shift amount -> Z3 (bit-serial is invalid there);
        #   * pure &/|/^ -> the bit-independent closed-form bitwise solver;
        #   * any bitwise op mixed with arithmetic -> bit-serial directly (a
        #     constant inside &/|/^ is never affine, so the affine probe would
        #     only burn ~ncols+4 full evaluations to reach the same fallback);
        #   * otherwise pure arithmetic -> the affine solver, with bit-serial as
        #     the fallback for non-affine arithmetic (cross-terms like c0*c1).
        klass = self._template_class(resized, (id(template), size))

        if klass == "shift":
            z3_solved = self._solve_constants_z3(
                resized, placeholders, outputs, inputs, size
            )
            self._last_solve_kind = "z3" if z3_solved else None
            return z3_solved

        if klass == "bitwise":
            self._bitwise_unique = False
            bitwise = self._solve_bitwise(
                evaluator, placeholders, base, outputs, inputs, size
            )
            if not bitwise:
                self._last_solve_kind = None
            elif self._bitwise_unique:
                self._last_solve_kind = "bitwise_unique"
            else:
                self._last_solve_kind = "bitwise"
            return bitwise

        if klass == "arith":
            # Pure arithmetic: try the closed-form affine solver first.
            fast = self._solve_affine(
                evaluator, placeholders, base, outputs, inputs, size
            )
            if isinstance(fast, dict):
                # Affine solves are exact: every coefficient is independently
                # observable through the linear probes, so the closed-form
                # solution is globally equivalent and the Z3 gate can be skipped.
                self._last_solve_kind = "affine"
                return fast
            if fast == "unsat":
                # Affine template proven not to match — Z3 only confirms UNSAT.
                self._last_solve_kind = None
                return None
            # ``fast in (None, "not_affine")``: non-affine arithmetic (cross
            # terms). Fall through to bit-serial.

        # Mixed arith+bitwise or non-affine arithmetic: the Hensel bit-serial
        # lifter handles it in closed form — and unlike Z3 it does not choke on
        # 64-bit ``bvmul``. It is complete on the samples for shift-free
        # templates, so None means no constants match.
        self._bit_serial_unique = False
        bit_serial = self._solve_bit_serial(
            evaluator, placeholders, base, outputs, inputs, size
        )
        if not bit_serial:
            self._last_solve_kind = None
        elif self._bit_serial_unique:
            self._last_solve_kind = "bit_serial_unique"
        else:
            self._last_solve_kind = "bit_serial"
        return bit_serial

    def _solve_constants_z3(
        self,
        resized: Expr,
        placeholders: List[ExprId],
        outputs: List[int],
        inputs: List[List[int]],
        size: int,
    ) -> Optional[Dict[ExprId, int]]:
        """Solve every placeholder constant with Z3 from the I/O constraints.

        Each placeholder is a ``size``-bit BitVec; for every sample the p-vars
        are bound to the concrete inputs and the result constrained to the
        output. Used for shift-amount constants (which the Hensel solver cannot
        lift) and as the escalation fallback for masked add/and/or shapes where
        bit-serial lifting does not converge but the constants enter linearly /
        bitwise (no 64-bit ``bvmul``), so Z3 dispatches them quickly.
        """
        solver = z3.Solver()
        # Cap the per-template Z3 budget tightly. A *matching* template solves
        # in a few ms; a long run means the template does not fit, so a high
        # timeout only inflates the cost of walking templates on a miss (the old
        # 2s cap let a single miss spend seconds in Z3). The closed-form /
        # affine solvers already carry the common cases.
        solver.set("timeout", min(self.solver_timeout * 1000, self._z3_timeout_ms))
        mask = (1 << size) - 1 if size > 0 else 0

        p_vars = self._placeholder_vars(resized, "p")

        for inputs_row, expected in zip(inputs, outputs):
            replacements = {
                p_var: ExprInt(inputs_row[int(p_var.name[1:])], size)
                for p_var in p_vars
                if int(p_var.name[1:]) < len(inputs_row)
            }
            # Substitute unified inputs and constrain to expected output.
            expr_inst = resized.replace_expr(replacements)
            z3_expr = self._translator_z3.from_expr(expr_inst)
            solver.add(z3_expr == z3.BitVecVal(int(expected) & mask, size))

        if solver.check() != z3.sat:
            return None

        model = solver.model()
        solved: Dict[ExprId, int] = {}
        for placeholder in placeholders:
            bv = z3.BitVec(placeholder.name, size)
            solved[placeholder] = model.eval(bv, model_completion=True).as_long() & mask
        return solved

    def try_synthesize(
        self,
        subtree: Expr,
        unified_subtree: Expr,
        unification_dict: Dict[Expr, Expr],
    ) -> Optional[Expr]:
        """
        Attempts to synthesize constants for a unified subtree using templates.

        The method:
        1) Evaluates the unified subtree on the template oracle inputs.
        2) Looks up candidate templates by truncated key; falls back to all
           templates if no key matches.
        3) Solves placeholder constants with Z3 and reconstructs the candidate.
        4) Validates the candidate against the subtree by counterexample-guided
           I/O sampling; a disagreeing sample feeds the next solve. A candidate
           that survives the sample set is returned. CEGIS does NOT prove
           equivalence itself -- that is the caller's suitability gate's job
           (see module docstring), so an unprovable-but-sample-consistent
           candidate is returned here and accepted/rejected upstream.
        5) Reverses unification (pN -> original terminals) before returning.

        Args:
            subtree: Original subtree (used for size and final mapping).
            unified_subtree: Subtree after unification (p0, p1, ...).
            unification_dict: Map from original terminals to pN variables.

        Returns:
            Candidate expression with constants instantiated (sample-validated,
            not self-proven; equivalence is enforced by the caller), or None.

        Example:
            >>> from msynth.utils.unification import gen_unification_dict
            >>> subtree = v0 * ExprInt(0x47, 8) + ExprInt(0x13, 8)
            >>> udict = gen_unification_dict(subtree)        # {v0: p0}
            >>> unified = subtree.replace_expr(udict)         # p0*0x47 + 0x13
            >>> solver.try_synthesize(subtree, unified, udict)
            ExprOp('+', ExprOp('*', ExprInt(0x47, 8), v0), ExprInt(0x13, 8))
        """
        if len(unification_dict) > self.max_variables:
            return None

        num_vars = len(unification_dict)
        # Use the template oracle's fixed inputs as the initial sample set.
        base_inputs = list(self.template_oracle.inputs)
        # Compile the target once; every output / validation evaluation below
        # runs natively instead of via per-call ``expr_simp``.
        target_eval = _make_evaluator(unified_subtree)
        base_outputs = [target_eval(row) for row in base_inputs]

        # Structured bit-discriminating probes (see _structured_probe_inputs):
        # seeding the solver with these pins down constants whose effect is
        # masked on random inputs, so the bitwise / bit-serial solvers recover
        # the right value on the first solve instead of guessing an
        # underdetermined bit. Appended to every template's initial solve set.
        probe_inputs = self._structured_probe_inputs(num_vars, subtree.size)
        probe_outputs = [target_eval(row) for row in probe_inputs]
        # The structured probes carry the discriminating inputs; only a small
        # slice of the (often 32) oracle base inputs is needed to seed the
        # solvers, and trimming it directly shrinks the per-bit work of the
        # bit-serial lifter (its cost is linear in the input count). Refinement
        # and the bit-walk escalation add more inputs on demand when a solve is
        # underdetermined, so coverage is preserved.
        seed_base = base_inputs[: self._solve_base_inputs]
        solve_inputs = seed_base + probe_inputs
        solve_outputs = base_outputs[: self._solve_base_inputs] + probe_outputs

        # Tier 1 (cheap, shape-aware): templates whose skeleton matches the
        # target's skeleton. O(1) lookup, no Z3 cost for non-matching shapes.
        skeleton_templates = list(
            self.template_oracle.get_skeleton_templates(unified_subtree)
        )

        # Tier 2: I/O-behaviour-keyed lookup. NOTE: oracles built by
        # ``gen_runtime_oracle`` store every template under the synthetic "*"
        # bucket (not under an I/O key), so for the default runtime oracle this
        # tier returns nothing and only the skeleton tier (1) and full-scan
        # tier (3) carry the load. It still works for oracles populated by hand
        # via ``add_template(template, outputs)`` with real I/O keys.
        equiv_key = self.template_oracle.determine_equiv_key(base_outputs)
        keyed_templates = list(self.template_oracle.get_templates(equiv_key))

        # Tier 3: full template iteration. Only enter when both keyed paths come
        # up empty -- and then route through the truncated-I/O prefilter so only
        # templates whose low-bit behaviour *can* match the target are solved.
        # Skeleton-matched templates share the target's exact structure, so they
        # are solved as synthesis (licensing the unique-solve Z3-gate skip); the
        # rest carry the gate.
        skeleton_ids = set(id(t) for t in skeleton_templates)
        templates = skeleton_templates + keyed_templates
        if templates:
            # De-duplicate while preserving order: the same template can live in
            # both the skeleton bucket and the synthetic ``"*"`` bucket.
            seen_ids = set()
            deduped: List[Expr] = []
            for template in templates:
                tid = id(template)
                if tid in seen_ids:
                    continue
                seen_ids.add(tid)
                deduped.append(template)
            templates = deduped
            synthesis_ids = skeleton_ids
            if self.expand_templates:
                templates = self._expand_templates(templates, subtree.size)
                # _expand_templates copies the bases (verbatim, first) then adds
                # wrappers; only the verbatim skeleton copies keep the target's
                # shape, so re-derive the synthesis set by structural identity.
                synthesis_ids = set(
                    id(t)
                    for t in templates
                    if any(t == s for s in skeleton_templates)
                )
        else:
            templates = self._filtered_walk_templates(
                subtree.size, target_eval, unified_subtree, num_vars
            )
            synthesis_ids = set()

        # Cumulative wall-time budget across all per-template solves. The fast
        # affine/bitwise solvers cost a couple ms each, so this only ever trips
        # on a miss that keeps falling through to Z3 — bounding the worst case
        # while letting an early skeleton-matched template solve in time.
        solve_deadline = time.time() + self._z3_target_budget_ms / 1000.0

        for template in templates[: self.max_templates]:
            if time.time() > solve_deadline:
                break
            _evaluator, tpl_placeholders, _base, resized_template = self._cached_template(
                template, subtree.size
            )
            p_vars = self._placeholder_vars(resized_template, "p")
            if p_vars and int(p_vars[-1].name[1:]) >= num_vars:
                continue

            candidate = self._solve_validated(
                template,
                unified_subtree,
                num_vars,
                subtree.size,
                target_eval,
                solve_inputs,
                solve_outputs,
                is_synthesis=(id(template) in synthesis_ids),
            )
            if candidate is not None:
                return reverse_unification(candidate, unification_dict)

        # Structural synthesis tier: no static template matched. Abstract the
        # target's *own* constants into placeholders and solve that skeleton
        # directly — the bit-serial solver recovers the constants for an
        # arbitrary (clean) constant-bearing shape. The skeleton is registered so
        # later targets of the same shape are found instantly via the skeleton
        # index. (For an obfuscated subtree the static templates above already
        # found the simpler form first; this tier only fires when they did not.)
        #
        # Not gated on ``solve_deadline``: that budget bounds the *template walk*,
        # but the synthesis tier is the only path that solves a clean
        # never-before-seen shape, and it is self-bounded (one abstraction, a few
        # refinements, at most one escalation). Skipping it whenever the walk ran
        # long was exactly what dropped clean targets whose walk was slow.
        if self.structural_synthesis:
            template, ncols = self._abstract_constants(unified_subtree, subtree.size)
            if 0 < ncols <= self._synthesis_max_constants:
                candidate = self._solve_validated(
                    template,
                    unified_subtree,
                    num_vars,
                    subtree.size,
                    target_eval,
                    solve_inputs,
                    solve_outputs,
                    is_synthesis=True,
                )
                if candidate is not None:
                    # Remember the shape for future same-skeleton targets.
                    self.template_oracle._index_skeleton(template)
                    return reverse_unification(candidate, unification_dict)

        return None

    def _solve_validated(
        self,
        template: Expr,
        unified_subtree: Expr,
        num_vars: int,
        size: int,
        target_eval,
        solve_inputs: List[List[int]],
        solve_outputs: List[int],
        is_synthesis: bool = False,
    ) -> Optional[Expr]:
        """Solve a template's constants, validate, and refine to a sound
        candidate (unified, constants instantiated) or ``None``.

        Acceptance is gated by sampling **and** a final Z3 equivalence query
        (:meth:`_z3_counterexample`). A Z3 counterexample means a constant bit
        was underdetermined by the sample because masking hides it (the
        ``(x & c) & (-x)`` family). The first such miss triggers a one-time
        escalation: the full power-of-2 bit-walk is folded into the solve set so
        the next solve observes every reachable bit at once, instead of clawing
        bits back one Z3 counterexample at a time. Truly irrelevant constant
        bits (masked on *every* input) are left to any value — Z3 then proves
        equivalence regardless and the candidate is accepted.
        """
        inputs = list(solve_inputs)
        outputs = list(solve_outputs)
        resized = self._resize_expr(template, size)
        escalated = False
        for _ in range(max(1, self.refinement_iters)):
            solved = self._solve_template(template, outputs, inputs, size)
            solve_kind = self._last_solve_kind
            if not solved:
                return None
            repl = {k: ExprInt(v, size) for k, v in solved.items()}
            candidate = resized.replace_expr(repl)

            counterexample = self._validate_candidate(
                target_eval, candidate, num_vars, size
            )
            if counterexample is None:
                # Skip the Z3 equivalence gate only where the candidate is
                # provably equivalent to the target without it:
                #  - structurally identical to the target (the solve recovered
                #    the original constants) -> trivially equivalent, any tier;
                #  - in the synthesis / skeleton-matched tier the target IS this
                #    template at the true constants, so a solve the samples
                #    determine *uniquely* must have found those true constants:
                #    affine "solved" (a full-rank linear solution is unique),
                #    bitwise_unique, or bit_serial_unique all qualify.
                # In the general walk the template's structure differs from the
                # target, so a unique sample-fit can still diverge globally -> it
                # keeps the gate (cheap there: walk matches are linear / bitwise,
                # not the 64-bit bvmul shapes that make the gate expensive).
                if (
                    candidate == unified_subtree
                    # Globally exact regardless of tier: the affine solver only
                    # succeeds on an affine target (full-rank => unique solution)
                    # and the bitwise solver only succeeds on a bit-independent
                    # target; a *uniquely* determined solve there equals the
                    # target globally.
                    or solve_kind in ("affine", "bitwise_unique")
                    # bit-serial's uniqueness is only over the samples, which
                    # equals "global" solely when the template IS the target's
                    # shape (synthesis / skeleton match).
                    or (is_synthesis and solve_kind == "bit_serial_unique")
                ):
                    return candidate
                counterexample = self._z3_counterexample(
                    unified_subtree, candidate, num_vars, size
                )
                if counterexample is None:
                    return candidate
                # Z3 disproved a sample-passing candidate -> a masked constant
                # bit. Escalate once: fold the full bit-walk into the solve set
                # so the next bit-serial solve observes every reachable bit, and
                # additionally try a direct Z3 constant-solve over the enriched
                # I/O. The latter converges the coupled two-constant add/and/or
                # masks (e.g. ``(x+c0) & (x&c1)``) that Hensel lifting alone does
                # not resolve, while bit-serial still carries the bvmul shapes Z3
                # cannot. Whichever yields a Z3-equivalent candidate wins.
                if not escalated:
                    escalated = True
                    # Exact synthesis first: a single exists-forall query lifts
                    # the constants to a *proven*-equivalent assignment, which
                    # cracks the coupled masked two-constant shapes outright.
                    forall_cand = self._synthesize_constants_forall(
                        resized, unified_subtree, size
                    )
                    if forall_cand is not None:
                        return forall_cand
                    # Otherwise fold the full bit-walk into the solve set so the
                    # next bit-serial solve observes every reachable bit.
                    bitwalk = self._bitwalk_probe_inputs(num_vars, size)
                    inputs.extend(bitwalk)
                    outputs.extend(target_eval(row) for row in bitwalk)

            # Counterexample-guided refinement: add failing input/output.
            inputs.append(counterexample)
            outputs.append(target_eval(counterexample))

        return None

    @staticmethod
    def _is_low_bits_closed(expr: Expr) -> bool:
        """True if every op keeps the low bits of the result a function of only
        the low bits of the inputs (so an ``_io_k``-bit truncation is sound).

        ``+ - * & | ^`` and unary ``-``/``~`` qualify; a *constant* left shift
        qualifies; right shift, division and modulo do not (they pull high bits
        down). A target or template containing a disqualifying op skips the
        prefilter and is always tried.
        """
        unsafe = {">>", "a>>", ">>>", "/", "%", "sdiv", "udiv", "smod", "umod"}
        stack = [expr]
        while stack:
            node = stack.pop()
            if isinstance(node, ExprOp):
                if node.op in unsafe:
                    return False
                if node.op == "<<":
                    # Only a *constant* shift amount keeps it low-bits-closed.
                    if len(node.args) != 2 or not isinstance(node.args[1], ExprInt):
                        return False
                stack.extend(node.args)
            elif isinstance(node, ExprSlice):
                # A slice that drops low bits (start>0) is not low-bits-closed.
                if node.start != 0:
                    return False
                stack.append(node.arg)
            elif isinstance(node, (ExprCond, ExprCompose)):
                return False
        return True

    def _template_sig_set(self, evaluator, base: int, ncols: int) -> frozenset:
        """Enumerate a template's achievable ``_io_k``-bit behaviour set.

        For every ``_io_k``-bit constant assignment, evaluate the template at the
        fixed filter inputs and collect the tuple of truncated outputs. The
        resulting set is the complete set of behaviours the template can exhibit
        (low-bits-closed, so low-bit constants suffice).
        """
        k = self._io_k
        kmask = self._io_kmask
        rows = [
            (list(pv) + [0] * base)[:base] for pv in self._io_filter_inputs
        ]
        sigset = set()
        for combo in range(1 << (k * ncols)):
            cvals = [(combo >> (k * j)) & kmask for j in range(ncols)]
            try:
                sig = tuple(evaluator(row + cvals) & kmask for row in rows)
            except (ZeroDivisionError, ValueError, OverflowError):
                return frozenset()
            sigset.add(sig)
        return frozenset(sigset)

    def _filtered_walk_templates(
        self, size: int, target_eval, unified_subtree: Expr, num_vars: int
    ) -> List[Expr]:
        """Tier-3 walk set, narrowed by the truncated-I/O prefilter.

        Builds (once per bit-width) the full expanded template list together
        with each low-bits-closed template's achievable behaviour set, then
        returns only the templates whose set contains the target's truncated
        behaviour (plus every non-closed template, which cannot be filtered
        soundly). Falls back to the full list when the target itself is not
        low-bits-closed.
        """
        cache = self._walk_cache.get(size)
        if cache is None:
            base_templates = list(self.template_oracle.all_templates())
            if self.expand_templates:
                expanded = self._expand_templates(base_templates, size)
            else:
                expanded = [self._resize_expr(t, size) for t in base_templates]
            sig_sets: List[Optional[frozenset]] = []
            for template in expanded:
                if not self._is_low_bits_closed(template):
                    sig_sets.append(None)
                    continue
                evaluator, placeholders, base, _resized = self._cached_template(
                    template, size
                )
                ncols = len(placeholders)
                # Bound the enumeration; a template with too many constants is
                # always tried rather than enumerated (2^(k*ncols) blows up).
                if ncols > self._synthesis_max_constants:
                    sig_sets.append(None)
                    continue
                sig_sets.append(self._template_sig_set(evaluator, base, ncols))
            cache = (expanded, sig_sets)
            self._walk_cache[size] = cache

        expanded, sig_sets = cache
        if not self._is_low_bits_closed(unified_subtree):
            return expanded
        # Pad each probe to the target's variable count with zeros (matching the
        # zero-padding _template_sig_set applies), so a target with more
        # variables than the 3-wide probe tuples still evaluates.
        try:
            target_sig = tuple(
                target_eval((list(pv) + [0] * num_vars)[: max(len(pv), num_vars)])
                & self._io_kmask
                for pv in self._io_filter_inputs
            )
        except (ZeroDivisionError, ValueError, OverflowError, IndexError):
            return expanded
        return [
            template
            for template, sig_set in zip(expanded, sig_sets)
            if sig_set is None or target_sig in sig_set
        ]

    def _abstract_constants(self, expr: Expr, size: int):
        """Replace each ``ExprInt`` leaf with a fresh ``c`` placeholder.

        Returns ``(template, ncols)``. Every integer occurrence becomes its own
        placeholder so distinct (or coincidentally-equal) constants are solved
        independently. Used by the synthesis tier to treat the target's own
        skeleton as a template.
        """
        counter = [0]

        def walk(node: Expr) -> Expr:
            if isinstance(node, ExprInt):
                placeholder = ExprId(f"c{counter[0]}", node.size)
                counter[0] += 1
                return placeholder
            if isinstance(node, ExprOp):
                return ExprOp(node.op, *[walk(arg) for arg in node.args])
            if isinstance(node, ExprSlice):
                return ExprSlice(walk(node.arg), node.start, node.stop)
            if isinstance(node, ExprCond):
                return ExprCond(walk(node.cond), walk(node.src1), walk(node.src2))
            if isinstance(node, ExprCompose):
                return ExprCompose(*[walk(arg) for arg in node.args])
            return node

        template = walk(expr)
        return template, counter[0]
