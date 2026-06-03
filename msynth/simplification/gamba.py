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
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

from miasm.expression.expression import (
    Expr,
    ExprCompose,
    ExprCond,
    ExprId,
    ExprInt,
    ExprMem,
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


# =============================================================================
# GAMBA general nonlinear MBA (paper §5.1 substitution + Layer-1 passes)
# =============================================================================
#
# Per user direction: everything related to GAMBA's general nonlinear pipeline
# (the §5.1 substitution loop, the Expand pass, the FactorizeSums pass) lives
# in this same gamba.py module rather than a separate gamba_general.py.
#
# Layering recap:
#   Layer 1 (this module): ExpandPass, FactorizeSumsPass — pipeline passes
#     appended to gamba_pipeline() in pipeline.py.
#   Layer 2 (this module): gamba_substitution() — invoked per-subtree from
#     simplifier.py's BFS loop. Replaces the plain subtree-SimBA tier.
#   Layer 3 (simba.py): _bitwise_refine helper (wired but currently no-op —
#     see comment in simba._lookup_bitwise_expression).


# ---------------------------------------------------------------------------
# Helpers shared between the new passes
# ---------------------------------------------------------------------------


_ATOM_TYPES = (ExprId, ExprInt, ExprMem, ExprSlice, ExprCompose, ExprCond)
"""Expression node kinds treated as opaque leaves by classification and
abstraction helpers below. Mirrors :data:`simba._PRIMARY_LEAVES` plus
``ExprInt`` (constants are atoms for §5.1 abstraction purposes)."""


def _is_atom(expr: Expr) -> bool:
    """True iff ``expr`` is an opaque leaf (no operator sub-structure)."""
    return isinstance(expr, _ATOM_TYPES)


def _is_arith_op(op: str) -> bool:
    """True iff ``op`` is a ring-arithmetic operator (paper's linear half)."""
    return op in {"+", "-", "*"}


def _is_bitwise_op(op: str) -> bool:
    """True iff ``op`` is a bitwise operator (paper's nonlinear half except *)."""
    return op in {"&", "|", "^"}


def classify_linear_nonlinear(expr: Expr) -> str:
    """
    Coarse linear/nonlinear classifier mirroring upstream GAMBA's
    ``Node.is_linear``. Returns ``"linear"`` if ``expr`` only uses
    ring-arithmetic and single-variable bitwise terms in a linear-MBA
    shape, ``"nonlinear"`` otherwise.

    The linear-MBA fragment per the paper:
      - constants and variables (atoms),
      - sums and differences of linear terms,
      - constant * linear term,
      - bitwise operations over atoms.

    Anything beyond — a product of two variable-dependent bitwise
    expressions, a power, an XOR inside a sum of products — is
    nonlinear and the §5.1 substitution loop should consider abstracting
    nonlinear sub-expressions to fresh variables.

    Args:
        expr: Expression to classify.

    Returns:
        ``"linear"`` if the expression fits the linear-MBA fragment;
        ``"nonlinear"`` otherwise.
    """
    if _is_atom(expr):
        return "linear"
    if not isinstance(expr, ExprOp):
        return "nonlinear"
    op = expr.op
    # Unary negation, addition, subtraction over linear sub-terms remain linear.
    if op in {"-", "+"}:
        return (
            "linear"
            if all(classify_linear_nonlinear(arg) == "linear" for arg in expr.args)
            else "nonlinear"
        )
    # Constant * linear-term is linear; variable * variable is not. The
    # linear-MBA fragment per the paper requires every product to have at
    # least one constant (``ExprInt``) operand — without that, the product
    # introduces an arity-2 dependence on the operand vector that the
    # linear classifier cannot represent.
    if op == "*":
        has_const = any(isinstance(arg, ExprInt) for arg in expr.args)
        if not has_const:
            return "nonlinear"
        if all(classify_linear_nonlinear(arg) == "linear" for arg in expr.args):
            return "linear"
        return "nonlinear"
    # Bitwise ops are linear in the linear-MBA fragment iff their arguments
    # are atoms (the classifier in simba.py performs the stronger check —
    # this routing classifier is intentionally coarser).
    if _is_bitwise_op(op):
        return "linear" if all(_is_atom(arg) for arg in expr.args) else "nonlinear"
    # Any other operator (shifts, division, etc.) — treat as nonlinear so the
    # §5.1 abstraction loop has a chance to wrap it in a fresh variable.
    return "nonlinear"


def _is_irreducibly_nonlinear(expr: Expr) -> bool:
    """
    True iff the *node itself* (ignoring its subtree) has a shape outside
    the linear-MBA fragment.

    Distinct from :func:`classify_linear_nonlinear`, which classifies the
    whole subtree. A sum ``x*y + a*3`` is subtree-nonlinear but the sum
    node itself is linear-shaped — the nonlinearity is inside ``x*y``.
    For the §5.1 substitution loop we need the inside node.
    """
    if not isinstance(expr, ExprOp):
        return False
    op = expr.op
    if op == "*":
        # Product of two non-constants is nonlinear (no constant coefficient).
        non_const = sum(1 for arg in expr.args if not isinstance(arg, ExprInt))
        return non_const >= 2
    # Shifts / division / mod / power — operators that the linear-MBA
    # classifier does not handle, so the §5.1 loop treats them as opaque
    # nonlinear leaves to be abstracted.
    if op in {"<<", ">>", "a>>", "/", "%", "**"}:
        return True
    return False


def nonlinear_leaves(expr: Expr) -> List[Expr]:
    """
    Enumerate every *irreducibly-nonlinear* sub-expression of ``expr``.

    Walking the whole tree (not stopping at the outermost nonlinear node),
    we collect each Expr whose own shape is nonlinear per
    :func:`_is_irreducibly_nonlinear`. Inner nonlinear sub-expressions that
    are shared between sibling nonlinear nodes (e.g. ``x*y`` inside both
    ``a*x*y`` and ``b*x*y``) appear in the list separately from their
    enclosing parents — which is what the §5.1 substitution loop needs: it
    may want to abstract ``x*y`` alone (so the parents collapse to a
    linear ``a*g + b*g`` SimBA can solve), or to abstract ``a*x*y`` and
    ``b*x*y`` as opaque atoms (so the sum becomes ``g0 + g1``). The
    enumeration leaves that policy decision to the caller.

    De-duplication is by structural equality (``Expr.__eq__``) so two
    syntactically-identical sub-expressions are counted once. The order
    is the post-order traversal sequence; the caller (see
    :func:`gamba_substitution`) does its own combinatorial walk over the
    returned list.

    Args:
        expr: Expression to scan.

    Returns:
        List of nonlinear sub-expressions (de-duplicated by structural
        equality).
    """
    found: List[Expr] = []
    seen: set[Expr] = set()

    def walk(node: Expr) -> None:
        if isinstance(node, ExprOp):
            for arg in node.args:
                walk(arg)
        if _is_irreducibly_nonlinear(node) and node not in seen:
            seen.add(node)
            found.append(node)

    walk(expr)
    return found


# ---------------------------------------------------------------------------
# Layer 1 — ExpandPass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpandPass:
    """
    Distributive expansion pass (GAMBA paper §5.2 distribution step).

    Applies ``a * (b + c) → a*b + a*c`` (and the symmetric `(b + c) * a`
    shape) bottom-up with a net-shrink guard: the rewrite is committed only
    when the distributed form has strictly fewer nodes than the input. This
    avoids the obvious "expansion always grows the tree" trap — we only
    distribute when downstream simplification will reduce the result enough
    to pay for the expansion.

    Designed to slot into :func:`pipeline.gamba_pipeline` after the
    ``GambaPostRewriterPass`` so that any common-factor extraction the
    post-rewriter has already done is not undone by indiscriminate
    distribution.

    Implementation note: the current pass is intentionally conservative —
    it only distributes through one level. Multi-level distribution
    (``a * (b * (c + d))``) is left for a follow-up pass that integrates
    with :class:`FactorizeSumsPass` so that distribute/factor cycles can
    converge via the BFS fixpoint instead of inside this pass.
    """

    max_iters: int = 50

    def run(self, expr: Expr) -> Expr:
        """Pipeline-pass entry point; matches the GambaPreprocessingPass shape."""
        for _ in range(self.max_iters):
            new_expr = _expand_once(expr)
            if new_expr is expr or new_expr == expr:
                return expr
            expr = new_expr
        return expr


def _expand_once(expr: Expr) -> Expr:
    """One bottom-up walk that distributes products over sums when net-shrink."""
    if not isinstance(expr, ExprOp):
        return expr
    # Recurse first so leaves are expanded before parents (bottom-up).
    new_args = tuple(_expand_once(arg) for arg in expr.args)
    rebuilt = ExprOp(expr.op, *new_args) if new_args != tuple(expr.args) else expr
    distributed = _try_distribute(rebuilt)
    return distributed if distributed is not None else rebuilt


def _try_distribute(expr: Expr) -> Expr | None:
    """
    Try the distributive rewrite at the root of ``expr``.

    Returns the distributed Expr only when it has strictly fewer nodes
    than ``expr``; otherwise ``None`` (caller keeps the original).
    """
    if not (isinstance(expr, ExprOp) and expr.op == "*" and len(expr.args) == 2):
        return None
    left, right = expr.args
    sum_side: Expr | None = None
    other: Expr | None = None
    if isinstance(right, ExprOp) and right.op == "+":
        sum_side, other = right, left
    elif isinstance(left, ExprOp) and left.op == "+":
        sum_side, other = left, right
    if sum_side is None or other is None:
        return None
    # Distribute over each sum child.
    distributed_terms = tuple(ExprOp("*", other, term) for term in sum_side.args)
    candidate = ExprOp("+", *distributed_terms)
    if _node_count(candidate) < _node_count(expr):
        return candidate
    return None


# ---------------------------------------------------------------------------
# Layer 1 — FactorizeSumsPass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorizeSumsPass:
    """
    Global sum-factoring pass — local-pattern version of upstream GAMBA's
    ``Node.factorize_sums`` (``utils/batch.Batch``).

    Walks every ``+`` node and looks for a common factor shared by ≥ 2 of
    its children; when found, extracts it as ``factor * (rest1 + rest2)``.
    Accepts the rewrite only when net-shrink (the standard guard).

    This is a *simplified* port: the upstream ``Batch`` partitioner tries
    every multiset combination of factors across all sum children and
    picks the one that maximises shared cardinality. The version below
    extracts the FIRST commonly-shared factor it finds (cheap, less
    optimal). The full multiset partitioner is a follow-up; the current
    version covers the common ``c*x + c*y → c*(x+y)`` shape that recurs
    in MBA outputs from SimBA reconstruction.
    """

    max_iters: int = 50

    def run(self, expr: Expr) -> Expr:
        """Pipeline-pass entry point."""
        for _ in range(self.max_iters):
            new_expr = _factorize_once(expr)
            if new_expr is expr or new_expr == expr:
                return expr
            expr = new_expr
        return expr


def _product_factors(expr: Expr) -> List[Expr]:
    """
    Split a product into its multiplicand factors.

    Returns ``[expr]`` for a non-product Expr — the convention is that
    every Expr has at least itself as a factor for factoring purposes.
    """
    if isinstance(expr, ExprOp) and expr.op == "*":
        return list(expr.args)
    return [expr]


def _factorize_once(expr: Expr) -> Expr:
    """One bottom-up walk that factors common sub-terms out of sums when net-shrink."""
    if not isinstance(expr, ExprOp):
        return expr
    new_args = tuple(_factorize_once(arg) for arg in expr.args)
    rebuilt = ExprOp(expr.op, *new_args) if new_args != tuple(expr.args) else expr
    factored = _try_factor_sum(rebuilt)
    return factored if factored is not None else rebuilt


def _try_factor_sum(expr: Expr) -> Expr | None:
    """
    Try to factor a common multiplicand shared by ≥ 2 of a ``+`` node's
    children. This is a partial-overlap version of upstream GAMBA's
    ``Node.factorize_sums`` — not the full multiset partitioner, but
    enough to handle the recurring shapes
    ``c*x + c*y + ...   → c*(x+y) + ...``
    ``a*x + a*y + b*x + b*y → a*(x+y) + b*x + b*y → ... → (a+b)*(x+y)``
    when the BFS fixpoint re-enters this pass.

    Returns the factored Expr only when it has strictly fewer nodes than
    ``expr``; otherwise ``None``.
    """
    if not (isinstance(expr, ExprOp) and expr.op == "+" and len(expr.args) >= 2):
        return None

    # Each term is a (possibly trivial) product; collect each term's factor list.
    factor_lists: List[List[Expr]] = [_product_factors(term) for term in expr.args]

    # Score every candidate factor by how many terms it appears in. We pick
    # the highest-scoring factor (ties broken by first-seen) — that's the
    # one whose extraction maximises the shared structure exposed.
    candidate_counts: Dict[Expr, int] = {}
    for factors in factor_lists:
        seen_in_term: set[int] = set()
        for factor in factors:
            # Each distinct factor counts at most once per term.
            if id(factor) in seen_in_term:
                continue
            seen_in_term.add(id(factor))
            candidate_counts[factor] = candidate_counts.get(factor, 0) + 1

    best_factor: Expr | None = None
    best_count = 1  # at least 2 → a real shared factor
    for factor, count in candidate_counts.items():
        if count >= 2 and count > best_count:
            best_factor = factor
            best_count = count

    if best_factor is None:
        return None

    matching_residues: List[Expr] = []
    untouched: List[Expr] = []
    for term, factors in zip(expr.args, factor_lists):
        if best_factor not in factors:
            untouched.append(term)
            continue
        remaining = list(factors)
        remaining.remove(best_factor)  # remove first occurrence only
        if not remaining:
            matching_residues.append(ExprInt(1, expr.size))
        elif len(remaining) == 1:
            matching_residues.append(remaining[0])
        else:
            matching_residues.append(ExprOp("*", *remaining))

    # Reassemble: factor * (residue1 + residue2 + ...) [+ untouched terms].
    factored_part: Expr = (
        ExprOp("*", best_factor, ExprOp("+", *matching_residues))
        if len(matching_residues) > 1
        else ExprOp("*", best_factor, matching_residues[0])
    )
    if untouched:
        candidate_expr = ExprOp("+", factored_part, *untouched)
    else:
        candidate_expr = factored_part
    if _node_count(candidate_expr) < _node_count(expr):
        return candidate_expr
    return None


# ---------------------------------------------------------------------------
# Layer 2 — §5.1 substitution loop (replaces the BFS-loop subtree-SimBA tier)
# ---------------------------------------------------------------------------


def _gated_max_k(n_leaves: int, max_k: int) -> int:
    """
    Combinatorial cap that mirrors upstream GAMBA's gating in
    ``simplify_general.__simplify_via_substitution``.

    The enumeration tries ``C(n_leaves, k)`` SimBA invocations per ``k``,
    so unbounded growth in either dimension is unacceptable. Upstream
    bounds the inner ``k`` based on the leaf count itself:

    - ``n_leaves <= 5``: allow up to the caller-supplied ``max_k``;
    - ``5 < n_leaves <= 9``: cap at ``min(max_k, 3)``;
    - ``n_leaves > 9``: cap at ``min(max_k, 2)``.

    Returns ``0`` when no escalation is possible (no leaves to abstract
    or ``max_k == 0``).
    """
    if n_leaves == 0 or max_k <= 0:
        return 0
    if n_leaves <= 5:
        return min(max_k, n_leaves)
    if n_leaves <= 9:
        return min(max_k, 3, n_leaves)
    return min(max_k, 2, n_leaves)


def gamba_substitution(
    subtree: Expr,
    simba_fn,
    *,
    max_k: int = 0,
) -> Expr | None:
    """
    §5.1 substitution loop — replaces the plain subtree-SimBA tier in
    :meth:`Simplifier.simplify`'s BFS loop with an escalating attempt:

    1. ``n = 0``: plain SimBA on the subtree as-is (subsumes the old
       direct subtree-SimBA tier exactly when ``max_k == 0``).
    2. ``n = 1..gated_max``: enumerate every ``C(n_leaves, k)``
       combination of nonlinear leaves, abstract them to fresh
       placeholder variables (``g0``, ``g1``, …), re-run SimBA on the
       linearised form, reverse-substitute and accept only when net-shrink.

    The ``gated_max`` cap (see :func:`_gated_max_k`) follows upstream
    GAMBA's combinatorial bounds to keep enumeration tractable on
    high-arity nonlinear sub-trees.

    Args:
        subtree: The expression to simplify.
        simba_fn: Callable ``Expr -> Expr | None`` that runs SimBA on
            the input. ``None`` (or returning the same Expr unchanged)
            signals "SimBA could not reduce". Caller supplies it so this
            module does not depend on simba.py.
        max_k: Maximum number of nonlinear leaves to abstract per
            attempt. Default ``0`` disables escalation (behavior-equal
            to plain subtree-SimBA). ``>= 1`` enables §5.1 escalation
            up to the gated cap.

    Returns:
        The simplified subtree when SimBA (at any escalation level)
        produced a strictly-smaller candidate; ``None`` otherwise. The
        caller commits via
        :meth:`Simplifier._is_suitable_simplification_candidate`.
    """
    original_nodes = _node_count(subtree)

    # n = 0: plain SimBA on the subtree as-is. ``simba_fn`` is expected to
    # return ``None`` (or the unchanged input) when SimBA's classifier
    # rejects the subtree — that's a miss and we escalate (or fall
    # through to the BFS loop's next tier when ``max_k == 0``).
    candidate = simba_fn(subtree)
    if candidate is not None and candidate != subtree:
        # n = 0 SimBA's output is already accepted by SimBA's own
        # net-shrink check on the conjunction-basis form; we don't
        # impose a second node-count guard here because that would
        # reject the canonicalised reconstruction even when it is
        # semantically the same as a previous oracle hit.
        return candidate

    # n >= 1: §5.1 abstraction loop. Each iteration:
    #   - pick ``k`` of the irreducibly-nonlinear leaves,
    #   - substitute them with fresh placeholder vars (``abstract_subexprs``),
    #   - run SimBA on the now-linearised form,
    #   - reverse the substitution and accept only on net-shrink.
    if max_k < 1:
        return None
    leaves = nonlinear_leaves(subtree)
    gated_max = _gated_max_k(len(leaves), max_k)
    if gated_max == 0:
        return None

    best: Expr | None = None
    best_nodes = original_nodes
    for k in range(1, gated_max + 1):
        for combo in combinations(leaves, k):
            abstracted, mapping = abstract_subexprs(subtree, list(combo))
            attempt = simba_fn(abstracted)
            # SimBA miss → try next combination.
            if attempt is None or attempt == abstracted:
                continue
            restored = reverse_abstract(attempt, mapping)
            restored_nodes = _node_count(restored)
            # Strict-shrink guard mirrors the BFS-loop suitability check
            # (whose node-count side is the cheap pre-filter). Equal-size
            # candidates are rejected because the BFS loop already exposed
            # the subtree to the same SimBA on a previous pass.
            if restored_nodes < best_nodes:
                best = restored
                best_nodes = restored_nodes
    return best


def abstract_subexprs(
    expr: Expr, targets: Sequence[Expr]
) -> Tuple[Expr, Dict[Expr, Expr]]:
    """
    Replace each occurrence of every Expr in ``targets`` with a fresh
    placeholder variable, returning the abstracted expression and the
    placeholder → original mapping for reverse substitution.

    Used by the §5.1 substitution loop (n ≥ 1) to abstract nonlinear
    sub-expressions to fresh variables before re-running the linear SimBA
    solver on the resulting linearised form.

    Args:
        expr: The expression containing nonlinear sub-expressions to abstract.
        targets: The sub-expressions to abstract. Each gets a unique fresh
            placeholder variable named ``g0``, ``g1``, … of matching size.

    Returns:
        A tuple ``(abstracted_expr, mapping)`` where ``abstracted_expr`` is
        the input with every ``target`` replaced and ``mapping`` records
        placeholder → original for use by :func:`reverse_abstract`.
    """
    mapping: Dict[Expr, Expr] = {}
    replacements: Dict[Expr, Expr] = {}
    for index, target in enumerate(targets):
        placeholder = ExprId(f"g{index}", target.size)
        mapping[placeholder] = target
        replacements[target] = placeholder
    abstracted = expr.replace_expr(replacements)
    return abstracted, mapping


def reverse_abstract(expr: Expr, mapping: Dict[Expr, Expr]) -> Expr:
    """
    Inverse of :func:`abstract_subexprs` — replace each placeholder var
    with its original sub-expression.

    Args:
        expr: An abstracted Expr whose placeholder vars match keys in
            ``mapping``.
        mapping: Placeholder → original Expr mapping, as produced by
            :func:`abstract_subexprs`.

    Returns:
        The expression with all placeholders restored to their originals.
    """
    return expr.replace_expr(mapping)
