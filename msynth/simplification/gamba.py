"""
GAMBA algebraic refinement engine for MBA simplification.

This is a Miasm-native reimplementation of the §5.2 algebraic-refinement
rule set from:
Benjamin Reichenwallner and Peter Meerwald-Stadler,
"Simplification of General Mixed Boolean-Arithmetic Expressions: GAMBA",
SP-Workshop 2023 (DLS), arXiv:2305.06763.

Reference implementation: https://github.com/DenuvoSoftwareSolutions/GAMBA

This module provides a fixpoint engine that applies msynth-native
:class:`~msynth.simplification.rewrites.RewriteRule` objects bottom-up
on an :class:`~miasm.expression.expression.Expr` tree, without delegating
to miasm's :data:`ExpressionSimplifier.PASS_COMMONS`. The motivation is
two-fold:

1. Isolate msynth's algebraic refinement from miasm's normalisation
   choices so that identities targeting MBA-obfuscation shapes fire
   deterministically on the input expression as the caller wrote it.
2. Make the rule set serve two distinct pipeline phases with different
   safety constraints:

   - The pre-SimBA stage must NOT grow the expression and must NOT
     distribute (distribution widens the cube SimBA reasons over).
     :class:`GambaPreprocessor` runs only ``guarded=False`` rules.
   - The post-SimBA stage may apply guarded rewrites (ring normalisation,
     deep factorisation) that accept their output only if it strictly
     shrinks the tree. :class:`GambaPostRewriter` adds those.

The engine is intentionally simple — a bottom-up walk with a per-op
rule index — because the rules themselves carry all the cleverness.
A ``max_iters`` cap guards against any rule-pair that oscillates;
the no-grow contract of safe rules makes oscillation rare in practice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from miasm.expression.expression import (
    Expr,
    ExprCompose,
    ExprCond,
    ExprOp,
    ExprSlice,
)

from msynth.simplification.rewrites import DEFAULT_RULES, RewriteRule


def _node_count(expr: Expr) -> int:
    return len(expr.graph().nodes())


@dataclass(frozen=True)
class _GambaEngine:
    """
    Bottom-up fixpoint engine over a fixed rule set.

    The walk visits children first (post-order). At each node, the
    engine tries every rule whose op-key matches; the first rule that
    returns a non-``None`` rewrite wins, and the walk restarts at the
    new node (allowing the same node to be reduced repeatedly within
    one bottom-up pass). The outer loop repeats the whole walk until
    a fixpoint is reached or ``max_iters`` is hit.

    ``max_iters`` is a safety cap, not a tuning knob — under normal
    operation (all rules ``guarded=False`` net-shrinkers, no
    oscillating rule-pairs) the engine converges in a handful of
    iterations.
    """

    rules: Tuple[RewriteRule, ...]
    rules_by_op: Dict[str, Tuple[RewriteRule, ...]]
    rules_any_op: Tuple[RewriteRule, ...]

    @classmethod
    def build(cls, rules: Sequence[RewriteRule]) -> "_GambaEngine":
        by_op: Dict[str, List[RewriteRule]] = {}
        any_op: List[RewriteRule] = []
        # The current msynth rule family is universally about ExprOp
        # patterns; we still keep a fallback bucket for rules that
        # might match without a known op handle (e.g., a future rule
        # rewriting a bare ExprCond shape).
        for rule in rules:
            target_ops = _candidate_ops_for_rule(rule)
            if not target_ops:
                any_op.append(rule)
            else:
                for op in target_ops:
                    by_op.setdefault(op, []).append(rule)
        return cls(
            rules=tuple(rules),
            rules_by_op={op: tuple(rs) for op, rs in by_op.items()},
            rules_any_op=tuple(any_op),
        )

    def normalize(self, expr: Expr, max_iters: int = 50) -> Expr:
        for _ in range(max_iters):
            try:
                new_expr = self._walk(expr)
            except RecursionError:
                # Deep XOR/OR/AND chains in pathological obfuscated inputs
                # can blow the recursive descent stack. Bailing out with the
                # current expression is sound (no rewrite is always sound)
                # and lets the upstream simplifier continue with whatever
                # collapses we managed before the limit was hit.
                return expr
            if new_expr == expr:
                return new_expr
            expr = new_expr
        return expr

    def _walk(self, expr: Expr) -> Expr:
        # Post-order: rewrite children first, then try rules at this node.
        if isinstance(expr, ExprOp):
            new_args = tuple(self._walk(arg) for arg in expr.args)
            if new_args != expr.args:
                expr = ExprOp(expr.op, *new_args)
        elif isinstance(expr, ExprSlice):
            inner = self._walk(expr.arg)
            if inner != expr.arg:
                expr = ExprSlice(inner, expr.start, expr.stop)
        elif isinstance(expr, ExprCompose):
            new_args = tuple(self._walk(arg) for arg in expr.args)
            if new_args != expr.args:
                expr = ExprCompose(*new_args)
        elif isinstance(expr, ExprCond):
            new_cond = self._walk(expr.cond)
            new_src1 = self._walk(expr.src1)
            new_src2 = self._walk(expr.src2)
            if (new_cond, new_src1, new_src2) != (expr.cond, expr.src1, expr.src2):
                expr = ExprCond(new_cond, new_src1, new_src2)
        # Atoms (ExprId, ExprInt, ExprMem) have no children to descend.

        return self._apply_at_node(expr)

    def _apply_at_node(self, expr: Expr) -> Expr:
        # Try op-targeted rules first, then any-op rules.
        candidates: Tuple[RewriteRule, ...] = ()
        if isinstance(expr, ExprOp):
            candidates = self.rules_by_op.get(expr.op, ())
        # Local rewrite loop: re-apply at the same node until quiescent,
        # since one rule's output may unlock another's pattern.
        changed = True
        while changed:
            changed = False
            for rule in candidates + self.rules_any_op:
                try:
                    rewritten = rule.apply(expr)
                except RecursionError:
                    # A rule's pattern walk may recurse on deeply nested
                    # operand chains. Skipping the rule on this node is
                    # sound (no rewrite is always sound); other rules and
                    # the outer fixpoint may still make progress.
                    continue
                if rewritten is None or rewritten == expr:
                    continue
                expr = rewritten
                if isinstance(expr, ExprOp):
                    candidates = self.rules_by_op.get(expr.op, ())
                else:
                    candidates = ()
                changed = True
                break
        return expr


def _candidate_ops_for_rule(rule: RewriteRule) -> Tuple[str, ...]:
    """
    Conservative static analysis of a rule's op affinity.

    The current rule catalogue follows a clear naming convention where
    the family or rule name hints at the op the rule expects. We index
    on a best-effort basis so the per-node cost is O(matching rules)
    rather than O(all rules); a rule whose op affinity cannot be
    inferred ends up in the "any op" bucket and is tried at every
    ExprOp node.
    """
    name = rule.name
    family = rule.family

    # Explicit hints baked into rule / family names.
    op_keywords = {
        "_and": "&",
        "_or": "|",
        "_xor": "^",
        "_add": "+",
        "_mul": "*",
        "_shift": None,  # both << and >>
    }

    # Family-level mapping — broad buckets that fire on multiple ops.
    family_to_ops: Dict[str, Tuple[str, ...]] = {
        "inverse_element": ("+",),
        "two_complement": ("+", "^"),
        "constant_merge": ("+",),
        "power_of_two": ("&", "|", "^"),
        "bitwise_flatten": ("|",),
        "demorgan": ("^",),  # ~(...) is encoded as ^ all_ones
        "absorption": ("|", "&"),
        "redundancy": ("|", "&"),
        "ring": ("+",),
        "factor": ("+",),
    }

    # Per-name override (more specific than family).
    name_to_ops: Dict[str, Tuple[str, ...]] = {
        "demorgan_and_to_or": ("^",),
        "demorgan_or_to_and": ("^",),
        "or_xor_split": ("|",),
        "ring_normalize": ("+",),
        "factor_common_subterm": ("+",),
    }

    if name in name_to_ops:
        return name_to_ops[name]
    if family in family_to_ops:
        return family_to_ops[family]

    # Conservative fallback: try the rule on every ExprOp.
    inferred: List[str] = []
    for keyword, op in op_keywords.items():
        if keyword in name and op is not None:
            inferred.append(op)
    return tuple(inferred) if inferred else ()


def _select_safe_rules(rules: Sequence[RewriteRule]) -> Tuple[RewriteRule, ...]:
    """Subset of ``rules`` that are no-grow (``guarded=False``)."""
    return tuple(r for r in rules if not r.guarded)


def _select_guarded_rules(rules: Sequence[RewriteRule]) -> Tuple[RewriteRule, ...]:
    """Subset of ``rules`` that need an external net-shrink guard."""
    return tuple(r for r in rules if r.guarded)


@dataclass(frozen=True)
class GambaPreprocessor:
    """
    Strict no-grow preprocessor for use BEFORE SimBA.

    Applies only :class:`RewriteRule` objects with ``guarded=False`` —
    no distributivity, no factorisation, only structural collapses
    and algebraic identities that always reduce or preserve tree size.
    Suitable for input shaping before SimBA's linear-MBA reconstruction
    because the rule set cannot widen SimBA's atom set or push the
    expression into a verbose canonical form.
    """

    rules: Tuple[RewriteRule, ...]
    engine: _GambaEngine
    max_iters: int = 50

    @classmethod
    def from_default(cls) -> "GambaPreprocessor":
        safe = _select_safe_rules(DEFAULT_RULES)
        return cls(rules=safe, engine=_GambaEngine.build(safe))

    def normalize(self, expr: Expr) -> Expr:
        return self.engine.normalize(expr, max_iters=self.max_iters)


@dataclass(frozen=True)
class GambaPostRewriter:
    """
    Full algebraic rewriter for use AFTER the simplifier loop.

    Applies all :class:`GambaPreprocessor` rules to fixed point, then
    runs the ``guarded=True`` rules (ring normalisation, deep
    factorisation) once each with their net-smaller checks intact.
    Suitable as a closing post-pass where shape rearrangement is
    desirable as long as it strictly shrinks the tree.
    """

    safe_rules: Tuple[RewriteRule, ...]
    guarded_rules: Tuple[RewriteRule, ...]
    engine: _GambaEngine
    max_iters: int = 50

    @classmethod
    def from_default(cls) -> "GambaPostRewriter":
        safe = _select_safe_rules(DEFAULT_RULES)
        guarded = _select_guarded_rules(DEFAULT_RULES)
        return cls(
            safe_rules=safe,
            guarded_rules=guarded,
            engine=_GambaEngine.build(safe),
        )

    def normalize(self, expr: Expr) -> Expr:
        # Drive the safe-rule fixpoint first.
        expr = self.engine.normalize(expr, max_iters=self.max_iters)
        # Then attempt each guarded rule once at the root, with its own
        # net-smaller check inside ``apply`` deciding whether to commit.
        for rule in self.guarded_rules:
            try:
                rewritten = rule.apply(expr)
            except RecursionError:
                # Guarded rules (ring/factor) walk operand chains
                # recursively; pathologically deep inputs can blow the
                # stack. Skipping the rule is sound.
                continue
            if rewritten is None or rewritten == expr:
                continue
            if _node_count(rewritten) < _node_count(expr):
                expr = self.engine.normalize(rewritten, max_iters=self.max_iters)
        return expr


# Module-level singletons. Constructing the engine builds the rule
# index, which is cheap (≈ number of rules); a singleton keeps the
# index alive across calls without re-paying that cost.
GAMBA_PREPROCESSOR: GambaPreprocessor = GambaPreprocessor.from_default()
GAMBA_POST_REWRITER: GambaPostRewriter = GambaPostRewriter.from_default()
