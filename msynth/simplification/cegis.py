"""
CEGIS helpers for constant synthesis.

This module implements a runtime-only template oracle and a CEGIS solver.
It is designed to augment oracle-based simplification with constant recovery:

- Templates contain placeholders c0, c1, ... for unknown constants.
- Subtrees are unified to p0, p1, ... before solving.
- Z3 is used to solve the constants from a handful of I/O samples, and then
  again to *prove* the resulting candidate equal to the subtree for all
  inputs. A candidate is accepted only on an UNSAT (provably-equal) verdict;
  a SAT verdict yields a counterexample that refines the next solve, and an
  ``unknown`` (timeout) verdict rejects the candidate. I/O sampling alone is
  only a fast pre-filter and never certifies a result.

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
from msynth.utils.expr_utils import get_unique_variables
from msynth.utils.sampling import gen_inputs
from msynth.utils.unification import reverse_unification


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

        templates: List[Expr] = []
        if num_variables == 0:
            return templates

        def add(expr: Expr) -> None:
            templates.append(expr)

        p0 = vars_list[0]
        # One-variable affine / mask / shift style templates.
        add(p0 + c0)
        add(p0 - c0)
        add(p0 ^ c0)
        add(p0 * c0)
        add(p0 & c0)
        add(p0 | c0)
        add((p0 & c0) | c1)
        add((p0 & c0) ^ c1)
        add((p0 | c0) + c1)
        add((c0 * p0) + c1)
        add((p0 ^ c0) + c1)
        add((p0 + c0) ^ c1)
        add((p0 + c0) + c1)
        add(p0 << c0)

        if num_variables >= 2:
            p1 = vars_list[1]
            # Two-variable linear/bitwise templates plus constant mixing.
            add(p0 + p1)
            add(p0 ^ p1)
            add(p0 | p1)
            add(p0 & p1)
            add((p0 + p1) + c0)
            add((p0 ^ p1) ^ c0)
            add((p0 ^ p1) + c0)
            add((p0 & p1) + c0)
            add((p0 | p1) + c0)
            add((p0 + p1) ^ c0)
            add((p0 & c0) + p1)
            add((p0 ^ c0) + p1)
            add(p0 + (p1 & c0))
            add(p0 + (p1 ^ c0))
            add((p0 & c0) | (p1 & c1))
            add((p0 & c0) ^ (p1 & c1))
            add((p0 | c0) + (p1 | c1))
            add((p0 ^ c0) + (p1 ^ c1))

        if num_variables >= 3:
            p2 = vars_list[2]
            # Three-variable blends common in MBA patterns.
            add(p0 + p1 + p2)
            add(p0 ^ p1 ^ p2)
            add((p0 + p1) + c0)
            add((p0 ^ p1) + c0)
            add(p0 + (p1 & p2))
            add(p0 ^ (p1 | p2))
            add((p0 & p1) + p2)
            add((p0 | p1) + p2)
            add((p0 ^ p1) + p2)
            add((p0 + p1) ^ p2)

        if max_placeholders < 2:
            templates = [
                t.replace_expr({c1: ExprInt(0, template_bits)}) for t in templates
            ]

        return templates[:template_budget]

    @staticmethod
    def gen_runtime_oracle(
        template_bits: int = 8,
        num_variables: int = 3,
        num_samples: int = 32,
        max_placeholders: int = 2,
        template_budget: int = 80,
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
        max_templates: int = 50,
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
            seed: Seed for the validation-sampling RNG. Sampling is only a
                cheap pre-filter (acceptance is gated by a Z3 equivalence
                proof, see :meth:`_prove_equivalent`), but a fixed seed keeps
                the refinement path reproducible across runs.
        """
        self.template_oracle = template_oracle
        self.max_templates = max_templates
        self.solver_timeout = solver_timeout
        self.max_variables = max_variables
        self.refinement_iters = refinement_iters
        self.validation_samples = validation_samples
        self.expand_templates = expand_templates
        self.expansion_budget = expansion_budget
        self._translator_z3 = TranslatorZ3()
        self._rng = random.Random(seed)

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
            inputs.append(
                [self._rng.getrandbits(size) & mask for _ in range(num_vars)]
            )
        return inputs

    def _validate_candidate(
        self,
        unified_subtree: Expr,
        candidate: Expr,
        num_vars: int,
        size: int,
    ) -> Optional[List[int]]:
        """
        Validates a candidate and returns a counterexample input if found.

        Args:
            unified_subtree: Ground-truth unified expression.
            candidate: Candidate unified expression.
            num_vars: Number of variables.
            size: Bit-width.

        Returns:
            Counterexample input vector if validation fails, otherwise None.
        """
        for inputs in self._gen_validation_inputs(
            num_vars, size, self.validation_samples
        ):
            expected = self._evaluate_unified(unified_subtree, inputs)
            actual = self._evaluate_unified(candidate, inputs)
            if expected != actual:
                return inputs
        return None

    def _prove_equivalent(
        self,
        unified_subtree: Expr,
        candidate: Expr,
        num_vars: int,
    ) -> "tuple[str, Optional[List[int]]]":
        """
        Proves whether ``candidate`` is semantically equal to
        ``unified_subtree`` for *all* values of the unified ``pN`` variables.

        Sampling validation (:meth:`_validate_candidate`) only checks a finite
        set of inputs and can therefore accept a candidate that merely overfits
        the samples. This method closes that gap with a real Z3 query: it
        asserts the two expressions *differ* and inspects the result.

        Returns a ``(status, counterexample)`` pair:

        * ``("equivalent", None)`` — Z3 returned ``unsat``: the candidate is
          provably equal on every input. Only this verdict is safe to accept.
        * ``("counterexample", inputs)`` — Z3 returned ``sat``: ``inputs`` is a
          concrete assignment on which the two disagree (fed back into the
          refinement loop).
        * ``("unknown", None)`` — Z3 could not decide (timeout / translation
          failure). Treated as *not proven*; the candidate must NOT be
          accepted on this verdict.
        """
        try:
            z3_lhs = self._translator_z3.from_expr(unified_subtree)
            z3_rhs = self._translator_z3.from_expr(candidate)
        except Exception:
            # A translation failure means we cannot certify equivalence.
            return ("unknown", None)

        solver = z3.Solver()
        solver.set("timeout", self.solver_timeout * 1000)
        solver.add(z3_lhs != z3_rhs)
        result = solver.check()

        if result == z3.unsat:
            return ("equivalent", None)
        if result == z3.sat:
            model = solver.model()
            p_vars = self._placeholder_vars(unified_subtree, "p")
            p_vars += [v for v in self._placeholder_vars(candidate, "p") if v not in p_vars]
            counterexample = [0] * num_vars
            for p_var in p_vars:
                idx = int(p_var.name[1:])
                if idx >= num_vars:
                    continue
                bv = z3.BitVec(p_var.name, p_var.size)
                counterexample[idx] = model.eval(bv, model_completion=True).as_long()
            return ("counterexample", counterexample)
        # z3.unknown -- cannot certify; do not accept.
        return ("unknown", None)

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

    def _solve_template(
        self,
        template: Expr,
        outputs: List[int],
        inputs: List[List[int]],
        size: int,
    ) -> Optional[Dict[ExprId, int]]:
        """
        Solves constant placeholders for a template using Z3.

        Each placeholder (c0, c1, ...) is treated as a BitVec of `size` bits.
        For every I/O sample, the unified variables (p0, p1, ...) are substituted
        with the concrete inputs and the resulting expression is constrained to
        equal the expected output. If Z3 finds a model, it yields the constants.

        Args:
            template: Unified expression template with placeholders.
            outputs: Expected outputs for each input sample.
            inputs: Input sample list aligned with outputs.
            size: Bit-width to solve for.

        Returns:
            Mapping from placeholder ExprId to solved integer value, or None.
        """
        # Resize template to the subtree bit-width before solving.
        resized = self._resize_expr(template, size)
        placeholders = self._placeholder_vars(resized, "c")
        if not placeholders:
            return None

        solver = z3.Solver()
        solver.set("timeout", self.solver_timeout * 1000)
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
        4) Validates the candidate against the subtree: cheap sampling first,
           then a Z3 equivalence *proof* (:meth:`_prove_equivalent`). A
           candidate is only returned when Z3 proves it equal on every input;
           a Z3 counterexample feeds counterexample-guided refinement, and a
           Z3 ``unknown`` (timeout) is treated as "not proven" (candidate
           rejected). Sampling alone never certifies a result.
        5) Reverses unification (pN -> original terminals) before returning.

        Args:
            subtree: Original subtree (used for size and final mapping).
            unified_subtree: Subtree after unification (p0, p1, ...).
            unification_dict: Map from original terminals to pN variables.

        Returns:
            Candidate expression with constants instantiated (Z3-proven
            equivalent to ``subtree``), or None.

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
        base_outputs = self.template_oracle.get_outputs(unified_subtree)

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

        # Tier 3: full template iteration. Only enter when both keyed
        # paths come up empty.
        templates = skeleton_templates + keyed_templates
        if not templates:
            templates = list(self.template_oracle.all_templates())

        # De-duplicate while preserving order: the same template can live in
        # both the skeleton bucket and the synthetic ``"*"`` bucket; we only
        # want to spend Z3 on it once.
        seen_ids = set()
        deduped: List[Expr] = []
        for template in templates:
            tid = id(template)
            if tid in seen_ids:
                continue
            seen_ids.add(tid)
            deduped.append(template)
        templates = deduped

        if self.expand_templates:
            templates = self._expand_templates(templates, subtree.size)

        for template in templates[: self.max_templates]:
            resized_template = self._resize_expr(template, subtree.size)
            p_vars = self._placeholder_vars(resized_template, "p")
            if p_vars and int(p_vars[-1].name[1:]) >= num_vars:
                continue

            inputs = list(base_inputs)
            outputs = list(base_outputs)

            for _ in range(max(1, self.refinement_iters)):
                solved = self._solve_template(
                    template,
                    outputs,
                    inputs,
                    subtree.size,
                )
                if not solved:
                    break

                resized = self._resize_expr(template, subtree.size)
                repl = {k: ExprInt(v, subtree.size) for k, v in solved.items()}
                candidate = resized.replace_expr(repl)

                counterexample = self._validate_candidate(
                    unified_subtree, candidate, num_vars, subtree.size
                )
                if counterexample is None:
                    # Sampling found no disagreement -- now PROVE it with Z3
                    # before accepting (sampling alone can be overfit).
                    status, proof_cex = self._prove_equivalent(
                        unified_subtree, candidate, num_vars
                    )
                    if status == "equivalent":
                        return reverse_unification(candidate, unification_dict)
                    if status == "unknown":
                        # Cannot certify this template's candidate; a refined
                        # constant would not change the proof outcome, so move
                        # on to the next template rather than spin.
                        break
                    # status == "counterexample": Z3 found an input the cheap
                    # sampling missed; fall through to refine with it.
                    counterexample = proof_cex

                # Counterexample-guided refinement: add failing input/output.
                inputs.append(counterexample)
                outputs.append(self._evaluate_unified(unified_subtree, counterexample))

        return None
