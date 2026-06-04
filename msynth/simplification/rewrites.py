"""
Unified algebraic-rewrite framework for msynth.

This module is the single home for semantic-preserving rewrite rules applied
outside the synthesis/oracle path. A rule is a small object identifying a
match shape and producing a rewritten expression, organised into families
("ring", "demorgan", "inverse_element", ...).

Two integration modes are supported through one API:

- **Safe rules** (``guarded=False``) are always-apply, fixpoint-friendly
  rewrites. They are registered as custom passes on a local
  :class:`miasm.expression.simplifications.ExpressionSimplifier` instance
  that also carries Miasm's default passes (``PASS_COMMONS``). We never
  touch Miasm globals — the simplifier instance lives on the ``Rewriter``.
- **Guarded rules** (``guarded=True``) require a net-smaller check before
  acceptance and may grow the tree on intermediate states; they cannot
  ride inside Miasm's fixpoint without risking divergence. They run in
  :meth:`Rewriter.normalize` as a post-pass after the local
  :class:`ExpressionSimplifier`.

The rule's ``apply`` callable returns ``None`` when it does not match, or
the rewritten :class:`Expr` when it does. This matches Miasm's pass
contract closely enough that the adapter is one line; it also lets each
rule's match-and-rewrite logic stay in a single function.

History: this module supersedes ``msynth/simplification/ring.py``. The
ring-normalisation logic is now a single guarded rule with the same
net-smaller behaviour. :class:`msynth.simplification.simplifier.Simplifier`
calls :data:`DEFAULT_REWRITER`'s :meth:`Rewriter.normalize` as its final
post-pass.
"""

from __future__ import annotations

from collections import Counter as _Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from miasm.expression.expression import Expr, ExprInt, ExprOp
from miasm.expression.simplifications import ExpressionSimplifier


# ---------------------------------------------------------------------------
# Rule object + helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewriteRule:
    """
    A semantic-preserving algebraic rewrite.

    Attributes:
        name: Stable identifier (e.g. ``"ring_normalize"``,
            ``"demorgan_and_to_or"``). Used in tests and telemetry.
        family: Group tag (e.g. ``"ring"``, ``"demorgan"``,
            ``"inverse_element"``). Matches the family vocabulary used
            elsewhere in the codebase (mutate.py).
        guarded: ``False`` for safe rules that always reduce or preserve
            complexity and can run inside Miasm's fixpoint. ``True`` for
            rules that need a net-smaller check before acceptance.
        apply: Callable that returns ``None`` if the rule does not match
            the input shape, or the rewritten :class:`Expr` otherwise.
            Guarded rules must perform the net-smaller check themselves
            and return ``None`` when the rewrite is not strictly smaller.
    """

    name: str
    family: str
    guarded: bool
    apply: Callable[[Expr], Optional[Expr]]


def _node_count(expr: Expr) -> int:
    return len(expr.graph().nodes())


def _mask(size: int) -> int:
    return (1 << size) - 1 if size > 0 else 0


# ---------------------------------------------------------------------------
# Ring normalisation (ported from the old ring.py, behaviour-preserving)
# ---------------------------------------------------------------------------
#
# This implements the same algorithm as the previous ``ring.py``:
#     flatten nested ``+`` -> distribute ``c * (a + b + ...)`` -> collect
#     coefficients per atom -> rebuild as a sum.
# The result is accepted only if its Miasm graph node count is strictly
# smaller than the input's; otherwise the rule returns ``None`` (i.e. the
# rewrite is rejected and the caller keeps the input unchanged). The guard
# is what makes distribution safe to call unconditionally inside a
# post-pass — distribution cannot inflate the output because the inflated
# form is rejected.


def _has_sum_node(expr: Expr) -> bool:
    if isinstance(expr, ExprOp) and expr.op == "+":
        return True
    if isinstance(expr, ExprOp):
        return any(_has_sum_node(arg) for arg in expr.args)
    return False


def _as_const_times_sum(
    expr: Expr, size: int
) -> Optional[Tuple[int, Tuple[Expr, ...]]]:
    """If ``expr`` is ``c * (a + b + …)``, return ``(c, (a, b, …))``."""
    if not (isinstance(expr, ExprOp) and expr.op == "*"):
        return None
    mask = _mask(size)
    coeff = 1
    non_int_args: List[Expr] = []
    for arg in expr.args:
        if arg.is_int():
            coeff = (coeff * int(arg)) & mask
        else:
            non_int_args.append(arg)
    if len(non_int_args) != 1:
        return None
    inner = non_int_args[0]
    if not (isinstance(inner, ExprOp) and inner.op == "+"):
        return None
    return (coeff, tuple(inner.args))


def _split_coeff(expr: Expr, size: int) -> Tuple[int, Optional[Expr]]:
    """
    Return ``(coefficient, atom)`` for ``coefficient * atom``.

    Bare constant -> ``(value, None)``. Unary ``-X`` recurses with negated
    coefficient. ``c * X`` recurses into ``X`` so chained constant
    multiplications fold. Anything else -> ``(1, expr)``.
    """
    mask = _mask(size)
    if expr.is_int():
        return (int(expr) & mask, None)
    if isinstance(expr, ExprOp) and expr.op == "-" and len(expr.args) == 1:
        sub_coeff, sub_atom = _split_coeff(expr.args[0], size)
        return ((-sub_coeff) & mask, sub_atom)
    if isinstance(expr, ExprOp) and expr.op == "*":
        coeff = 1
        non_int_args: List[Expr] = []
        for arg in expr.args:
            if arg.is_int():
                coeff = (coeff * int(arg)) & mask
            else:
                non_int_args.append(arg)
        if len(non_int_args) == 1:
            inner_coeff, inner_atom = _split_coeff(non_int_args[0], size)
            return ((coeff * inner_coeff) & mask, inner_atom)
    return (1, expr)


def _mul_const(coeff: int, atom: Expr, size: int) -> Expr:
    coeff &= _mask(size)
    if coeff == 0:
        return ExprInt(0, size)
    if coeff == 1:
        return atom
    return ExprOp("*", ExprInt(coeff, size), atom)


def _ring_rewrite(expr: Expr) -> Expr:
    """
    Recursive normalisation: descend into children, then if this node is
    a ``+``, flatten + distribute + collect coefficients + rebuild.

    Returns an equivalent expression. The caller is responsible for the
    net-smaller acceptance check.
    """
    if not isinstance(expr, ExprOp):
        return expr

    new_args = tuple(_ring_rewrite(arg) for arg in expr.args)
    if new_args != tuple(expr.args):
        expr = ExprOp(expr.op, *new_args)

    if not (isinstance(expr, ExprOp) and expr.op == "+"):
        return expr

    size = expr.size
    mask = _mask(size)

    # 1) flatten nested +
    flat_operands: List[Expr] = []
    for arg in expr.args:
        if isinstance(arg, ExprOp) and arg.op == "+":
            flat_operands.extend(arg.args)
        else:
            flat_operands.append(arg)

    # 2) distribute c * (sum) -> c*a + c*b + …
    distributed: List[Expr] = []
    for operand in flat_operands:
        inner_sum = _as_const_times_sum(operand, size)
        if inner_sum is not None:
            coeff, inner_args = inner_sum
            for inner in inner_args:
                distributed.append(_mul_const(coeff, inner, size))
            continue
        distributed.append(operand)

    # 3) collect coefficients per atom; constants accumulate separately.
    constant_total = 0
    coefficients: Dict[Expr, int] = {}
    atom_order: List[Expr] = []
    for operand in distributed:
        coeff, atom = _split_coeff(operand, size)
        if atom is None:
            constant_total = (constant_total + coeff) & mask
            continue
        if atom not in coefficients:
            atom_order.append(atom)
            coefficients[atom] = 0
        coefficients[atom] = (coefficients[atom] + coeff) & mask

    # 4) rebuild — drop zero coefficients, fold like atoms back into a sum.
    terms: List[Expr] = []
    if constant_total:
        terms.append(ExprInt(constant_total, size))
    for atom in atom_order:
        coeff = coefficients[atom]
        if coeff == 0:
            continue
        terms.append(_mul_const(coeff, atom, size))

    if not terms:
        return ExprInt(0, size)
    if len(terms) == 1:
        return terms[0]
    return ExprOp("+", *terms)


def _ring_normalize_apply(expr: Expr) -> Optional[Expr]:
    """
    Guarded rule entry point. Returns ``None`` if the rewrite is not
    strictly smaller than the input (i.e. distribution did not pay off);
    otherwise returns the normalised expression.

    Preserves the historical contract of ``ring_normalize`` exactly: same
    early-exit for non-sum-rooted expressions, same net-smaller check via
    Miasm graph node count.
    """
    if not _has_sum_node(expr):
        return None
    rebuilt = _ring_rewrite(expr)
    if rebuilt is expr or rebuilt == expr:
        return None
    if _node_count(rebuilt) < _node_count(expr):
        return rebuilt
    return None


RING_NORMALIZE_RULE = RewriteRule(
    name="ring_normalize",
    family="ring",
    guarded=True,
    apply=_ring_normalize_apply,
)


def ring_normalize(expr: Expr) -> Expr:
    """
    Backwards-compatible alias preserving the old ``ring.py`` API.

    Returns the ring-normalised form of ``expr`` if strictly smaller by
    Miasm graph-node count, else ``expr`` unchanged. Equivalent to
    ``RING_NORMALIZE_RULE.apply(expr) or expr``.
    """
    result = _ring_normalize_apply(expr)
    return result if result is not None else expr


# ---------------------------------------------------------------------------
# Helpers for GAMBA rule patterns
# ---------------------------------------------------------------------------


def _is_const(expr: Expr) -> bool:
    return isinstance(expr, ExprInt)


def _const_value(expr: Expr) -> Optional[int]:
    return int(expr) if isinstance(expr, ExprInt) else None


def _is_not(expr: Expr) -> Optional[Expr]:
    """If ``expr`` is ``x ^ all_ones`` (Miasm's representation of ``~x``),
    return ``x``; otherwise ``None``."""
    if not (isinstance(expr, ExprOp) and expr.op == "^" and len(expr.args) == 2):
        return None
    a, b = expr.args
    if isinstance(b, ExprInt) and int(b) == _mask(expr.size):
        return a
    if isinstance(a, ExprInt) and int(a) == _mask(expr.size):
        return b
    return None


def _not_of(expr: Expr) -> Expr:
    """Return ``~expr`` as ``expr ^ all_ones``. If ``expr`` is itself a
    ``~y`` pattern, return the inner ``y`` (double-negation collapse)."""
    inner = _is_not(expr)
    if inner is not None:
        return inner
    return ExprOp("^", expr, ExprInt(_mask(expr.size), expr.size))


def _match_binary_bitwise(expr: Expr, op: str) -> Optional[Tuple[Expr, Expr]]:
    """If ``expr`` is ``ExprOp(op, a, b)`` (exactly two args), return
    ``(a, b)``; else ``None``. Used to recognise ``X & Y``, ``X | Y``,
    ``X ^ Y`` shapes for rule matching."""
    if isinstance(expr, ExprOp) and expr.op == op and len(expr.args) == 2:
        return expr.args[0], expr.args[1]
    return None


def _replace_pair_in_sum(
    expr: Expr,
    finder: Callable[[Expr, Expr], Optional[Expr]],
) -> Optional[Expr]:
    """
    Walk every ordered pair ``(args[i], args[j])`` of a sum ``expr``
    looking for a match. The ``finder`` callable receives the two args
    and returns either ``None`` (no match) or a single replacement
    :class:`Expr` that algebraically equals their sum. If found, return a
    new sum with those two args replaced by the single replacement;
    otherwise ``None``.

    The pair search is exhaustive over orderings because the GAMBA
    identities are not symmetric in argument order (``(X&Y)`` and
    ``(~X&Y)`` are structurally distinct), even though the sum operator
    is commutative.
    """
    if not (isinstance(expr, ExprOp) and expr.op == "+" and len(expr.args) >= 2):
        return None
    args = list(expr.args)
    for i in range(len(args)):
        for j in range(len(args)):
            if i == j:
                continue
            replacement = finder(args[i], args[j])
            if replacement is None:
                continue
            kept = [a for k, a in enumerate(args) if k != i and k != j]
            kept.append(replacement)
            if len(kept) == 1:
                return kept[0]
            return ExprOp("+", *kept)
    return None


# ---------------------------------------------------------------------------
# Family: inverse_element  --  collapse complementary pairs over ``+``
# ---------------------------------------------------------------------------
#
# These are GAMBA Section 5.2 identities. They are sound for arbitrary
# bit-vector values, fire on shapes Miasm does not natively recognise,
# and shrink the expression on every match -- safe to register as
# Miasm passes (``guarded=False``).


def _find_pq_pattern(
    x_args: Tuple[Expr, Expr], y_args: Tuple[Expr, Expr]
) -> Optional[Tuple[Expr, Expr]]:
    """
    For binary bitwise operands ``x`` and ``y`` already split into their
    arg pairs, find ``(p, q)`` such that ``x == op(~p, q)`` and
    ``y == op(p, q)`` (with the inner ``op`` left to the caller). Returns
    ``(p, q)`` if found, else ``None``.

    Match is by *position*, not set membership: the rewrite is only valid
    when one side has ``~p`` at one position and ``q`` at the other, and
    the matching side has ``p`` and ``q`` (in either order). Set-
    membership shortcuts spuriously fire when ``p == q``: e.g. for
    ``(~q & q) + (q & r)`` the first term is ``0`` and the whole sum is
    ``q & r``, not ``q``, but a membership check sees both ``q`` and
    ``q`` (== ``p`` in disguise) in ``y_args`` and returns ``q``.
    """
    # Each of x_args has two positions; try both as the negated half.
    for not_p_idx in (0, 1):
        not_p = x_args[not_p_idx]
        p = _is_not(not_p)
        if p is None:
            continue
        q = x_args[1 - not_p_idx]
        # y must equal (p, q) or (q, p) as an ordered tuple. This
        # explicit positional check is what avoids the set-membership
        # degeneracy when p == q.
        if y_args == (p, q) or y_args == (q, p):
            return p, q
    return None


def _inverse_xor_neg(x: Expr, y: Expr) -> Optional[Expr]:
    """``(~P & Q) + (P & Q) -> Q``.

    The caller (`_replace_pair_in_sum`) tries both ``(x, y)`` orderings,
    so this only needs to handle the direction where ``x`` carries the
    negated half.
    """
    ax = _match_binary_bitwise(x, "&")
    ay = _match_binary_bitwise(y, "&")
    if ax is None or ay is None:
        return None
    found = _find_pq_pattern(ax, ay)
    if found is None:
        return None
    _p, q = found
    return q


def _apply_inverse_xor_neg(expr: Expr) -> Optional[Expr]:
    return _replace_pair_in_sum(expr, _inverse_xor_neg)


def _inverse_or_neg(x: Expr, y: Expr) -> Optional[Expr]:
    """``(~P | Q) + (P | Q) -> Q + (-1)``.

    Holds because the two ORs cover every bit position in complementary
    patterns; their sum is ``Q`` (the always-present part) plus ``-1``
    (because every bit not in Q ends up set on exactly one of the two
    sides). The result is one node larger than bare ``Q`` because of
    the trailing ``-1``, but the pair still collapses from ``2 * (or-
    tree)`` to ``Q + const`` — a strict reduction in tree size.
    """
    ax = _match_binary_bitwise(x, "|")
    ay = _match_binary_bitwise(y, "|")
    if ax is None or ay is None:
        return None
    found = _find_pq_pattern(ax, ay)
    if found is None:
        return None
    _p, q = found
    size = q.size
    return ExprOp("+", q, ExprInt(_mask(size), size))


def _apply_inverse_or_neg(expr: Expr) -> Optional[Expr]:
    return _replace_pair_in_sum(expr, _inverse_or_neg)


def _inverse_xor_neg_xor(x: Expr, y: Expr) -> Optional[Expr]:
    """``(~P ^ Q) + (P ^ Q) -> -1``.

    The two XORs are bitwise complements, so they sum to all-ones in
    every position.
    """
    ax = _match_binary_bitwise(x, "^")
    ay = _match_binary_bitwise(y, "^")
    if ax is None or ay is None:
        return None
    found = _find_pq_pattern(ax, ay)
    if found is None:
        return None
    size = x.size
    return ExprInt(_mask(size), size)


def _apply_inverse_xor_neg_xor(expr: Expr) -> Optional[Expr]:
    return _replace_pair_in_sum(expr, _inverse_xor_neg_xor)


# ---------------------------------------------------------------------------
# Family: two_complement  --  collapse x + ~x and x ^ ~x
# ---------------------------------------------------------------------------


def _add_complement_pair(x: Expr, y: Expr) -> Optional[Expr]:
    """``x + ~x -> -1``.

    Match on either ordering: ``y`` is ``~x`` or ``x`` is ``~y``.
    """
    if _is_not(y) == x or _is_not(x) == y:
        size = x.size
        return ExprInt(_mask(size), size)
    return None


def _apply_add_complement_pair(expr: Expr) -> Optional[Expr]:
    return _replace_pair_in_sum(expr, _add_complement_pair)


def _apply_xor_complement_pair(expr: Expr) -> Optional[Expr]:
    """``x ^ ~x -> -1`` (any number of additional XOR args)."""
    if not (isinstance(expr, ExprOp) and expr.op == "^" and len(expr.args) >= 2):
        return None
    args = list(expr.args)
    for i in range(len(args)):
        for j in range(i + 1, len(args)):
            a, b = args[i], args[j]
            if _is_not(a) == b or _is_not(b) == a:
                size = expr.size
                minus_one = ExprInt(_mask(size), size)
                kept = [arg for k, arg in enumerate(args) if k != i and k != j]
                kept.append(minus_one)
                if len(kept) == 1:
                    return kept[0]
                return ExprOp("^", *kept)
    return None


# ---------------------------------------------------------------------------
# Family: constant_merge  --  merge disjoint-bit constants
# ---------------------------------------------------------------------------


def _const_and_with_atom(expr: Expr) -> Optional[Tuple[int, Expr]]:
    """If ``expr`` is ``c & X`` with one ExprInt and one other operand,
    return ``(int(c), X)``; otherwise ``None``."""
    if not (isinstance(expr, ExprOp) and expr.op == "&" and len(expr.args) == 2):
        return None
    a, b = expr.args
    if isinstance(a, ExprInt) and not isinstance(b, ExprInt):
        return int(a), b
    if isinstance(b, ExprInt) and not isinstance(a, ExprInt):
        return int(b), a
    return None


def _constant_merge_and(x: Expr, y: Expr) -> Optional[Expr]:
    """``(a & X) + (b & X) -> (a + b) & X`` when ``a & b == 0``."""
    fx = _const_and_with_atom(x)
    fy = _const_and_with_atom(y)
    if fx is None or fy is None:
        return None
    a, atom_x = fx
    b, atom_y = fy
    if atom_x != atom_y:
        return None
    size = x.size
    if (a & b) & _mask(size) != 0:
        return None
    merged_const = (a + b) & _mask(size)
    return ExprOp("&", ExprInt(merged_const, size), atom_x)


def _apply_constant_merge_and(expr: Expr) -> Optional[Expr]:
    return _replace_pair_in_sum(expr, _constant_merge_and)


# ---------------------------------------------------------------------------
# Family: power_of_two  --  factor 2^k from bitwise ops
# ---------------------------------------------------------------------------
#
# Pattern: ``(c1 * X) OP (c2 * Y)`` where c1 == c2 and c1 is a power of
# two -> ``c1 * (X OP Y)``. Sound for ``&``, ``|``, ``^``. Strictly
# shrinks the AST (two ``*`` nodes -> one).


def _const_times_atom(expr: Expr) -> Optional[Tuple[int, Expr]]:
    """If ``expr`` is ``c * X`` with one int and one other operand,
    return ``(int(c), X)``; otherwise ``None``."""
    if not (isinstance(expr, ExprOp) and expr.op == "*"):
        return None
    coeff = 1
    non_int = []
    for arg in expr.args:
        if isinstance(arg, ExprInt):
            coeff *= int(arg)
        else:
            non_int.append(arg)
    if len(non_int) != 1:
        return None
    return coeff & _mask(expr.size), non_int[0]


def _is_pow_of_two(value: int) -> bool:
    return value > 1 and (value & (value - 1)) == 0


def _factor_pow2_from_bitwise(op: str):
    def _apply(expr: Expr) -> Optional[Expr]:
        if not (isinstance(expr, ExprOp) and expr.op == op and len(expr.args) == 2):
            return None
        lhs, rhs = expr.args
        fl = _const_times_atom(lhs)
        fr = _const_times_atom(rhs)
        if fl is None or fr is None:
            return None
        coeff_l, atom_l = fl
        coeff_r, atom_r = fr
        if coeff_l != coeff_r:
            return None
        if not _is_pow_of_two(coeff_l):
            return None
        size = expr.size
        return ExprOp(
            "*",
            ExprInt(coeff_l, size),
            ExprOp(op, atom_l, atom_r),
        )

    return _apply


# ---------------------------------------------------------------------------
# Family: bitwise_flatten  --  split bitwise ops into linear sums
# ---------------------------------------------------------------------------


def _apply_or_xor_split(expr: Expr) -> Optional[Expr]:
    """``(X & Y) | (X ^ Y) -> (X & Y) + (X ^ Y)``.

    Sound because ``X & Y`` and ``X ^ Y`` are bitwise disjoint, so OR and
    addition coincide on this pair. Lets downstream like-term collection
    and ring-normalisation operate on the sum form.
    """
    if not (isinstance(expr, ExprOp) and expr.op == "|" and len(expr.args) == 2):
        return None
    lhs, rhs = expr.args

    def _match_and_xor(a: Expr, b: Expr) -> Optional[Tuple[Expr, Expr]]:
        and_args = _match_binary_bitwise(a, "&")
        xor_args = _match_binary_bitwise(b, "^")
        if and_args is None or xor_args is None:
            return None
        if set(and_args) == set(xor_args):
            return a, b
        return None

    pair = _match_and_xor(lhs, rhs) or _match_and_xor(rhs, lhs)
    if pair is None:
        return None
    and_expr, xor_expr = pair
    return ExprOp("+", and_expr, xor_expr)


# ---------------------------------------------------------------------------
# Family: demorgan  --  apply only when a NOT already exists to absorb
# ---------------------------------------------------------------------------
#
# DeMorgan applied unconditionally grows the tree (~(a&b) is 5 nodes;
# ~a|~b is 6 nodes) and would loop in Miasm's fixpoint with no benefit.
# Guard: fire only when at least one operand is itself ``~something``;
# the resulting ``~~something`` collapses via Miasm's existing
# double-negation pass, netting a strict reduction.


def _demorgan(op_in: str, op_out: str):
    def _apply(expr: Expr) -> Optional[Expr]:
        inner = _is_not(expr)
        if inner is None:
            return None
        if not (
            isinstance(inner, ExprOp) and inner.op == op_in and len(inner.args) == 2
        ):
            return None
        a, b = inner.args
        # Require at least one operand to already be a NOT pattern so the
        # ``~~`` will collapse downstream. This avoids growing the tree
        # in cases where the rewrite would not net-shrink.
        if _is_not(a) is None and _is_not(b) is None:
            return None
        return ExprOp(op_out, _not_of(a), _not_of(b))

    return _apply


# ---------------------------------------------------------------------------
# Family: absorption  --  GAMBA Section 5.2 boolean absorption identities
# ---------------------------------------------------------------------------
#
# These are not covered by Miasm's PASS_COMMONS (verified against
# miasm/expression/simplifications_common.py:simp_cst_propagation).
# Both shrink the tree by one node on every match, are commutative
# in the outer args, and are sound by truth-table inspection.


def _apply_absorption_or(expr: Expr) -> Optional[Expr]:
    """``a | (a & b) -> a`` (either operand order)."""
    if not (isinstance(expr, ExprOp) and expr.op == "|" and len(expr.args) == 2):
        return None
    x, y = expr.args
    # x | (x & b) -> x
    inner = _match_binary_bitwise(y, "&")
    if inner is not None and x in inner:
        return x
    # (x & b) | x -> x
    inner = _match_binary_bitwise(x, "&")
    if inner is not None and y in inner:
        return y
    return None


def _apply_absorption_and(expr: Expr) -> Optional[Expr]:
    """``a & (a | b) -> a`` (either operand order)."""
    if not (isinstance(expr, ExprOp) and expr.op == "&" and len(expr.args) == 2):
        return None
    x, y = expr.args
    inner = _match_binary_bitwise(y, "|")
    if inner is not None and x in inner:
        return x
    inner = _match_binary_bitwise(x, "|")
    if inner is not None and y in inner:
        return y
    return None


# ---------------------------------------------------------------------------
# Family: redundancy  --  GAMBA Section 5.2 self-complement identities
# ---------------------------------------------------------------------------
#
# ``a | ~a`` always has every bit set; ``a & ~a`` always has every bit
# clear. Miasm does NOT recognise these patterns through PASS_COMMONS
# (it has constant-folding for ``a & 0`` etc. but not the
# self-complement collapse, because that requires recognising the
# ``a ^ all_ones`` form as ``~a``).


def _apply_redundancy_or_not(expr: Expr) -> Optional[Expr]:
    """``a | ~a -> -1``  (i.e. all-ones at the expression's width)."""
    if not (isinstance(expr, ExprOp) and expr.op == "|" and len(expr.args) == 2):
        return None
    a, b = expr.args
    if _is_not(b) == a or _is_not(a) == b:
        return ExprInt(_mask(expr.size), expr.size)
    return None


def _apply_redundancy_and_not(expr: Expr) -> Optional[Expr]:
    """``a & ~a -> 0``."""
    if not (isinstance(expr, ExprOp) and expr.op == "&" and len(expr.args) == 2):
        return None
    a, b = expr.args
    if _is_not(b) == a or _is_not(a) == b:
        return ExprInt(0, expr.size)
    return None


# ---------------------------------------------------------------------------
# Family: factor  --  GAMBA Section 5.4 deep factorisation
# ---------------------------------------------------------------------------
#
# Pattern: ``(F * a_1 * a_2 ...) + (F * b_1 * b_2 ...) + ... -> F *
# ((a_1 * a_2 ...) + (b_1 * b_2 ...) + ...)`` for any non-constant
# common multiset of factors `F`. Generalises ring_normalize's
# coefficient collection (which handles the CONSTANT-only common
# factor case) to structural common factors.
#
# Guarded (``guarded=True``): the net-smaller check in
# :meth:`Rewriter.normalize` rejects rewrites that don't shrink the
# tree. This guards against:
# - ring_normalize having already DISTRIBUTED ``F * (a + b)`` -- factoring
#   back is then a no-op and the guard rejects (preventing oscillation).
# - Inflation when the residual sum doesn't compose tightly.


def _factors_of(arg: Expr) -> List[Expr]:
    """Decompose ``arg`` into a multiplicative factor list.

    ``a * b * c`` -> ``[a, b, c]``. Any other shape (a bare ``x``, a
    unary ``-x``, a sum, ...) is treated as a single opaque factor and
    returned as ``[arg]``. In particular a negation is *not* split into
    ``[x, -1]``: the common-factor matching only sees explicit ``*``
    products, so ``a*b + -(a*c)`` is left to miasm / ``ring_normalize``.
    """
    if isinstance(arg, ExprOp) and arg.op == "*":
        return list(arg.args)
    return [arg]


def _apply_factor_common_subterm(expr: Expr) -> Optional[Expr]:
    """``(F*a_1...) + (F*b_1...) + ... -> F * (a_1... + b_1... + ...)``.

    Requires at least one non-constant factor in the common multiset
    so we don't duplicate miasm's coefficient-collection /
    ring_normalize work. Returns ``None`` (i.e. no rewrite) if no
    common non-constant factor exists, or if the rewrite is not
    net-smaller than the input.
    """
    if not (isinstance(expr, ExprOp) and expr.op == "+" and len(expr.args) >= 2):
        return None

    factor_lists = [_factors_of(arg) for arg in expr.args]

    # Multiset intersection across all terms.
    common_counter = _Counter(factor_lists[0])
    for factors in factor_lists[1:]:
        common_counter &= _Counter(factors)

    common = list(common_counter.elements())
    if not common:
        return None
    # Defer pure-constant common factors to miasm / ring_normalize.
    if all(isinstance(f, ExprInt) for f in common):
        return None

    # Build the common factor product, deterministically ordered.
    common_sorted = sorted(common, key=lambda x: str(x))
    if len(common_sorted) == 1:
        common_factor = common_sorted[0]
    else:
        common_factor = ExprOp("*", *common_sorted)

    # Build the residual sum.
    residual_terms: List[Expr] = []
    for factors in factor_lists:
        remaining = list(factors)
        for c in common:
            remaining.remove(c)
        if not remaining:
            residual_terms.append(ExprInt(1, expr.size))
        elif len(remaining) == 1:
            residual_terms.append(remaining[0])
        else:
            residual_terms.append(ExprOp("*", *remaining))

    if len(residual_terms) == 1:
        residual_sum = residual_terms[0]
    else:
        residual_sum = ExprOp("+", *residual_terms)

    result = ExprOp("*", common_factor, residual_sum)

    # Net-smaller guard.
    if _node_count(result) < _node_count(expr):
        return result
    return None


FACTOR_COMMON_SUBTERM_RULE = RewriteRule(
    name="factor_common_subterm",
    family="factor",
    guarded=True,
    apply=_apply_factor_common_subterm,
)


# ---------------------------------------------------------------------------
# Family: idempotence + duplicate-pair elimination (n-ary normalisation)
# ---------------------------------------------------------------------------
#
# Core re-normalisation rules used by the downstream Miasm-free preprocessor.
# They duplicate behaviour Miasm's PASS_COMMONS already provides, so they
# behave as no-ops inside the Miasm-backed ``Rewriter`` pipeline; they exist
# so the pure-msynth fixpoint engine has access to the same identities.


def _drop_duplicates_apply(op: str):
    """``a OP a OP ... OP b → a OP b`` for idempotent ``OP`` (``&``/``|``).

    Drops every repeated occurrence of a structurally identical child. If
    only one distinct child remains, the operator collapses to that child.
    """

    def _apply(expr: Expr) -> Optional[Expr]:
        if not (isinstance(expr, ExprOp) and expr.op == op and len(expr.args) >= 2):
            return None
        seen: List[Expr] = []
        kept: List[Expr] = []
        for arg in expr.args:
            if arg in seen:
                continue
            seen.append(arg)
            kept.append(arg)
        if len(kept) == len(expr.args):
            return None
        if len(kept) == 1:
            return kept[0]
        return ExprOp(op, *kept)

    return _apply


def _apply_xor_self_cancel(expr: Expr) -> Optional[Expr]:
    """N-ary XOR: drop every pair of structurally identical children.

    ``a ^ a → 0``; ``a ^ b ^ a → b``; with an odd count of a single ``a``,
    one ``a`` survives. If everything cancels, returns the all-zero
    constant at the expression's width.
    """
    if not (isinstance(expr, ExprOp) and expr.op == "^" and len(expr.args) >= 2):
        return None
    counts: Dict[Expr, int] = {}
    order: List[Expr] = []
    for arg in expr.args:
        if arg not in counts:
            counts[arg] = 0
            order.append(arg)
        counts[arg] += 1
    kept: List[Expr] = []
    for arg in order:
        # Each pair of identical XOR operands cancels; what's left is the
        # parity of the count.
        if counts[arg] % 2:
            kept.append(arg)
    if len(kept) == len(expr.args):
        return None
    if not kept:
        return ExprInt(0, expr.size)
    if len(kept) == 1:
        return kept[0]
    return ExprOp("^", *kept)


# ---------------------------------------------------------------------------
# Family: double_negation -- ``~~x → x``
# ---------------------------------------------------------------------------


def _apply_double_negation(expr: Expr) -> Optional[Expr]:
    """``~~x → x`` where ``~y`` is the ``y ^ all_ones`` shape."""
    inner = _is_not(expr)
    if inner is None:
        return None
    inner2 = _is_not(inner)
    if inner2 is None:
        return None
    return inner2


# ---------------------------------------------------------------------------
# Family: const_fold -- absorbing/identity constants for bitwise + arith ops
# ---------------------------------------------------------------------------
#
# All rules are n-ary safe: they scan every child for the special constant.
# When the constant is an annihilator (``0`` for ``&`` and ``*``, ``-1`` for
# ``|``) the result is the constant itself. When it's an identity (``0``
# for ``|``/``^``/``+``, ``-1`` for ``&``, ``1`` for ``*``) the constant is
# dropped and the surviving children are returned as the same op (or as a
# single child if only one remains).


def _fold_const_bitwise(op: str, identity: int, annihilator: Optional[int]):
    def _apply(expr: Expr) -> Optional[Expr]:
        if not (isinstance(expr, ExprOp) and expr.op == op and len(expr.args) >= 2):
            return None
        size = expr.size
        mask = _mask(size)
        ident = identity & mask
        annih = None if annihilator is None else (annihilator & mask)
        kept: List[Expr] = []
        had_match = False
        for arg in expr.args:
            if isinstance(arg, ExprInt):
                val = int(arg) & mask
                if annih is not None and val == annih:
                    return ExprInt(annih, size)
                if val == ident:
                    had_match = True
                    continue
            kept.append(arg)
        if not had_match:
            return None
        if not kept:
            return ExprInt(ident, size)
        if len(kept) == 1:
            return kept[0]
        return ExprOp(op, *kept)

    return _apply


def _apply_const_fold_add_zero(expr: Expr) -> Optional[Expr]:
    """Drop ``0`` from n-ary ``+``. Empty sum → ``0``."""
    if not (isinstance(expr, ExprOp) and expr.op == "+" and len(expr.args) >= 2):
        return None
    size = expr.size
    kept: List[Expr] = []
    had_zero = False
    for arg in expr.args:
        if isinstance(arg, ExprInt) and int(arg) == 0:
            had_zero = True
            continue
        kept.append(arg)
    if not had_zero:
        return None
    if not kept:
        return ExprInt(0, size)
    if len(kept) == 1:
        return kept[0]
    return ExprOp("+", *kept)


def _apply_const_fold_mul_zero(expr: Expr) -> Optional[Expr]:
    """Any product containing a ``0`` constant collapses to ``0``."""
    if not (isinstance(expr, ExprOp) and expr.op == "*" and len(expr.args) >= 2):
        return None
    size = expr.size
    for arg in expr.args:
        if isinstance(arg, ExprInt) and int(arg) == 0:
            return ExprInt(0, size)
    return None


def _apply_const_fold_mul_one(expr: Expr) -> Optional[Expr]:
    """Drop ``1`` from n-ary ``*``. Empty product → ``1``."""
    if not (isinstance(expr, ExprOp) and expr.op == "*" and len(expr.args) >= 2):
        return None
    size = expr.size
    kept: List[Expr] = []
    had_one = False
    for arg in expr.args:
        if isinstance(arg, ExprInt) and (int(arg) & _mask(size)) == 1:
            had_one = True
            continue
        kept.append(arg)
    if not had_one:
        return None
    if not kept:
        return ExprInt(1, size)
    if len(kept) == 1:
        return kept[0]
    return ExprOp("*", *kept)


# ---------------------------------------------------------------------------
# Helpers for arithmetic-negation patterns (``-x`` shapes)
# ---------------------------------------------------------------------------


def _arith_neg_of(expr: Expr) -> Optional[Expr]:
    """If ``expr`` is ``-y`` in any of Miasm's representations, return ``y``.

    Recognises:
      * unary ``ExprOp("-", y)``
      * binary product where one factor is the constant ``-1`` (i.e.,
        ``all_ones`` at the expression's width).
    """
    if isinstance(expr, ExprOp) and expr.op == "-" and len(expr.args) == 1:
        return expr.args[0]
    if isinstance(expr, ExprOp) and expr.op == "*" and len(expr.args) == 2:
        a, b = expr.args
        size = expr.size
        mask = _mask(size)
        if isinstance(a, ExprInt) and (int(a) & mask) == mask:
            return b
        if isinstance(b, ExprInt) and (int(b) & mask) == mask:
            return a
    return None


def _arith_double_of(expr: Expr) -> Optional[Expr]:
    """If ``expr`` is ``2 * y`` in any of Miasm's representations,
    return ``y``."""
    if isinstance(expr, ExprOp) and expr.op == "*" and len(expr.args) == 2:
        a, b = expr.args
        size = expr.size
        mask = _mask(size)
        if isinstance(a, ExprInt) and (int(a) & mask) == 2 % (
            1 << size if size > 0 else 1
        ):
            return b
        if isinstance(b, ExprInt) and (int(b) & mask) == 2 % (
            1 << size if size > 0 else 1
        ):
            return a
    return None


def _is_xor_with_neg(expr: Expr) -> Optional[Expr]:
    """If ``expr`` is ``x ^ -x`` (in any order), return ``x``."""
    if not (isinstance(expr, ExprOp) and expr.op == "^" and len(expr.args) == 2):
        return None
    a, b = expr.args
    if _arith_neg_of(b) == a:
        return a
    if _arith_neg_of(a) == b:
        return b
    return None


# ---------------------------------------------------------------------------
# Family: bitwise_zero -- conjunctions that collapse to ``0``
# ---------------------------------------------------------------------------
#
# Two identities, both producing the all-zero constant:
#   ``x & -x & 2*x → 0`` -- the conjunction contains ``x`` and ``-x``,
#       which are bitwise complements in every bit not shared between
#       ``x`` and ``-x``; combined with ``2*x`` the bit-0 of ``2*x`` is
#       always 0, and the remaining bit positions can only set when both
#       ``x`` and ``-x`` set the same bit, which never happens above bit 0.
#   ``~(2*x) & -(x ^ -x) → 0`` -- ``x ^ -x`` is all-ones, so ``-(x^-x)``
#       is ``1``; ``~(2*x)`` has bit 0 set (since ``2*x`` always has bit 0
#       clear), but the conjunction with ``1`` keeps only bit 0, while
#       the rule is verified to collapse to zero across all valuations.


def _apply_conj_self_neg_double_zero(expr: Expr) -> Optional[Expr]:
    """``x & -x & 2*x → 0`` (n-ary conjunction, any operand order)."""
    if not (isinstance(expr, ExprOp) and expr.op == "&" and len(expr.args) >= 3):
        return None
    args = list(expr.args)
    # Find indices (i, j) of a (x, -x) pair, then a third index k with 2*x or 2*(-x).
    for i in range(len(args)):
        x = args[i]
        for j in range(len(args)):
            if j == i:
                continue
            negx = _arith_neg_of(args[j])
            if negx != x:
                continue
            for k in range(len(args)):
                if k == i or k == j:
                    continue
                doubled = _arith_double_of(args[k])
                if doubled == x or doubled == args[j]:
                    return ExprInt(0, expr.size)
    return None


def _apply_conj_neg_xor_zero(expr: Expr) -> Optional[Expr]:
    """``~(2*x) & -(x ^ -x) → 0`` (conjunction may contain extra children;
    pattern just needs both clauses to be present)."""
    if not (isinstance(expr, ExprOp) and expr.op == "&" and len(expr.args) >= 2):
        return None
    args = list(expr.args)
    for i in range(len(args)):
        inner_not = _is_not(args[i])
        if inner_not is None:
            continue
        x = _arith_double_of(inner_not)
        if x is None:
            continue
        # Find -(x ^ -x) or -(-x ^ x) in any other position.
        for j in range(len(args)):
            if j == i:
                continue
            neg_of = _arith_neg_of(args[j])
            if neg_of is None:
                continue
            inner_xor_arg = _is_xor_with_neg(neg_of)
            if inner_xor_arg is None:
                continue
            # Accept either matching base (since x ^ -x == -x ^ x).
            if inner_xor_arg == x or _arith_neg_of(x) == inner_xor_arg:
                return ExprInt(0, expr.size)
            # Also handle the case where the doubled term and the xor base
            # are arithmetic negatives of each other.
            if _arith_neg_of(inner_xor_arg) == x:
                return ExprInt(0, expr.size)
    return None


def _apply_conj_negated_xor_zero(expr: Expr) -> Optional[Expr]:
    """``2*x & ~(x ^ -x) → 0`` -- ``x ^ -x`` is all-ones so ``~(x^-x)``
    is zero, and the conjunction is therefore zero."""
    if not (isinstance(expr, ExprOp) and expr.op == "&" and len(expr.args) >= 2):
        return None
    args = list(expr.args)
    for i in range(len(args)):
        inner = _is_not(args[i])
        if inner is None:
            continue
        xor_base = _is_xor_with_neg(inner)
        if xor_base is None:
            continue
        for j in range(len(args)):
            if j == i:
                continue
            doubled = _arith_double_of(args[j])
            if doubled == xor_base or doubled == _arith_neg_of(xor_base):
                return ExprInt(0, expr.size)
    return None


# ---------------------------------------------------------------------------
# Family: bitwise_identity_clause -- drop a clause that's algebraically
# the identity element for the surrounding bitwise op
# ---------------------------------------------------------------------------


def _drop_clause(expr: Expr, op: str, drop_idx: int) -> Expr:
    """Helper: rebuild ``expr`` with the child at ``drop_idx`` removed."""
    kept = [a for k, a in enumerate(expr.args) if k != drop_idx]
    if len(kept) == 1:
        return kept[0]
    return ExprOp(op, *kept)


def _apply_conj_xor_identity(expr: Expr) -> Optional[Expr]:
    """``2*x & (x ^ -x) → 2*x``: ``x ^ -x`` is all-ones, so its presence
    in a conjunction is the identity element and the clause can be
    dropped. Requires a second clause whose value involves ``x`` (we
    detect via the ``2*x`` shape so the rule's match is conservative)."""
    if not (isinstance(expr, ExprOp) and expr.op == "&" and len(expr.args) >= 2):
        return None
    args = list(expr.args)
    for i in range(len(args)):
        xor_base = _is_xor_with_neg(args[i])
        if xor_base is None:
            continue
        for j in range(len(args)):
            if j == i:
                continue
            doubled = _arith_double_of(args[j])
            if doubled == xor_base or doubled == _arith_neg_of(xor_base):
                return _drop_clause(expr, "&", i)
    return None


def _apply_disj_xor_identity(expr: Expr) -> Optional[Expr]:
    """``2*x | -(x ^ -x) → 2*x``: ``-(x ^ -x)`` equals ``1`` and so its
    OR-contribution sits inside any non-zero ``2*x`` -- the clause can
    be dropped."""
    if not (isinstance(expr, ExprOp) and expr.op == "|" and len(expr.args) >= 2):
        return None
    args = list(expr.args)
    for i in range(len(args)):
        neg_of = _arith_neg_of(args[i])
        if neg_of is None:
            continue
        xor_base = _is_xor_with_neg(neg_of)
        if xor_base is None:
            continue
        for j in range(len(args)):
            if j == i:
                continue
            doubled = _arith_double_of(args[j])
            if doubled == xor_base or doubled == _arith_neg_of(xor_base):
                return _drop_clause(expr, "|", i)
    return None


# ---------------------------------------------------------------------------
# Family: nested_bitwise_absorb -- ``x op_outer -(... op_inner ... -x ...)``
# patterns where the negated nested sub-expression evaluates to the
# bitwise identity for the outer op and can be dropped
# ---------------------------------------------------------------------------


def _apply_disj_disj_negation_absorb(expr: Expr) -> Optional[Expr]:
    """``x | -((x & y) | -x) → x``.

    The negated inner expression evaluates to a value whose every bit
    is already covered by ``x``, so the second OR clause is absorbed.
    """
    if not (isinstance(expr, ExprOp) and expr.op == "|" and len(expr.args) >= 2):
        return None
    args = list(expr.args)
    for i in range(len(args)):
        x = args[i]
        for j in range(len(args)):
            if j == i:
                continue
            inner = _arith_neg_of(args[j])
            if inner is None:
                continue
            if not (
                isinstance(inner, ExprOp) and inner.op == "|" and len(inner.args) == 2
            ):
                continue
            # One side must equal -x; the other must contain x as an &-operand.
            for negx_idx in (0, 1):
                cand_negx = inner.args[negx_idx]
                cand_other = inner.args[1 - negx_idx]
                if _arith_neg_of(cand_negx) != x:
                    continue
                # cand_other must be either an AND containing x, or x itself.
                if cand_other == x:
                    return _drop_clause(expr, "|", j)
                if (
                    isinstance(cand_other, ExprOp)
                    and cand_other.op == "&"
                    and x in cand_other.args
                ):
                    return _drop_clause(expr, "|", j)
    return None


def _apply_conj_conj_negation_absorb(expr: Expr) -> Optional[Expr]:
    """``x & -((x | y) & -x) → x`` (dual of the disj/disj absorption)."""
    if not (isinstance(expr, ExprOp) and expr.op == "&" and len(expr.args) >= 2):
        return None
    args = list(expr.args)
    for i in range(len(args)):
        x = args[i]
        for j in range(len(args)):
            if j == i:
                continue
            inner = _arith_neg_of(args[j])
            if inner is None:
                continue
            if not (
                isinstance(inner, ExprOp) and inner.op == "&" and len(inner.args) == 2
            ):
                continue
            for negx_idx in (0, 1):
                cand_negx = inner.args[negx_idx]
                cand_other = inner.args[1 - negx_idx]
                if _arith_neg_of(cand_negx) != x:
                    continue
                if cand_other == x:
                    return _drop_clause(expr, "&", j)
                if (
                    isinstance(cand_other, ExprOp)
                    and cand_other.op == "|"
                    and x in cand_other.args
                ):
                    return _drop_clause(expr, "&", j)
    return None


def _apply_disj_conj_negation_absorb(expr: Expr) -> Optional[Expr]:
    """``-x | (~x & 2*x) → -x``.

    The conjunction clause is bitwise-disjoint from ``-x``'s bit pattern
    (and absorbed into it), so the OR drops the clause.
    """
    if not (isinstance(expr, ExprOp) and expr.op == "|" and len(expr.args) >= 2):
        return None
    args = list(expr.args)
    for i in range(len(args)):
        x = _arith_neg_of(args[i])
        if x is None:
            continue
        for j in range(len(args)):
            if j == i:
                continue
            inner = args[j]
            if not (
                isinstance(inner, ExprOp) and inner.op == "&" and len(inner.args) == 2
            ):
                continue
            # Both clause operands must reference the *same* base ``x``:
            # one side ``~x``, the other ``2*x``. Mixing bases (e.g.
            # ``~(-x) & 2*x``) is NOT absorbed by ``-x`` and would be unsound.
            for not_idx in (0, 1):
                cand_not = inner.args[not_idx]
                cand_double = inner.args[1 - not_idx]
                not_inner = _is_not(cand_not)
                if not_inner != x:
                    continue
                doubled = _arith_double_of(cand_double)
                if doubled == x:
                    return _drop_clause(expr, "|", j)
    return None


def _apply_disj_neg_disj_identity(expr: Expr) -> Optional[Expr]:
    """``x | -(-x | 2*x) → x``.

    Conservative variant of the broader GAMBA identity: the inner OR is
    of ``-x`` and ``2*x`` (``_arith_double_of`` only recognises the
    coefficient ``2``), so the negated result is fully covered by ``x``.
    """
    if not (isinstance(expr, ExprOp) and expr.op == "|" and len(expr.args) >= 2):
        return None
    args = list(expr.args)
    for i in range(len(args)):
        x = args[i]
        for j in range(len(args)):
            if j == i:
                continue
            inner = _arith_neg_of(args[j])
            if inner is None:
                continue
            if not (
                isinstance(inner, ExprOp) and inner.op == "|" and len(inner.args) == 2
            ):
                continue
            for neg_idx in (0, 1):
                cand_neg = inner.args[neg_idx]
                cand_double = inner.args[1 - neg_idx]
                if _arith_neg_of(cand_neg) != x:
                    continue
                doubled = _arith_double_of(cand_double)
                if doubled == x or doubled == cand_neg:
                    return _drop_clause(expr, "|", j)
    return None


# ---------------------------------------------------------------------------
# Family: xor_same_mult_collapse -- ``2*(x | -x) → x ^ -x``
# ---------------------------------------------------------------------------


def _apply_xor_same_mult_or(expr: Expr) -> Optional[Expr]:
    """``2*(x | -x) → x ^ -x`` (factor 2 form)."""
    if not (isinstance(expr, ExprOp) and expr.op == "*" and len(expr.args) == 2):
        return None
    size = expr.size
    mask = _mask(size)
    a, b = expr.args
    if isinstance(a, ExprInt) and (int(a) & mask) == (
        2 % (1 << size if size > 0 else 1)
    ):
        inner = b
    elif isinstance(b, ExprInt) and (int(b) & mask) == (
        2 % (1 << size if size > 0 else 1)
    ):
        inner = a
    else:
        return None
    if not (isinstance(inner, ExprOp) and inner.op == "|" and len(inner.args) == 2):
        return None
    ia, ib = inner.args
    if _arith_neg_of(ia) == ib or _arith_neg_of(ib) == ia:
        return ExprOp("^", ia, ib)
    return None


def _apply_xor_same_mult_and(expr: Expr) -> Optional[Expr]:
    """``-2*(x & -x) → x ^ -x`` (negated factor 2 form)."""
    if not (isinstance(expr, ExprOp) and expr.op == "*" and len(expr.args) == 2):
        return None
    size = expr.size
    mask = _mask(size)
    neg2 = (-2) & mask
    a, b = expr.args
    if isinstance(a, ExprInt) and (int(a) & mask) == neg2:
        inner = b
    elif isinstance(b, ExprInt) and (int(b) & mask) == neg2:
        inner = a
    else:
        return None
    if not (isinstance(inner, ExprOp) and inner.op == "&" and len(inner.args) == 2):
        return None
    ia, ib = inner.args
    if _arith_neg_of(ia) == ib or _arith_neg_of(ib) == ia:
        return ExprOp("^", ia, ib)
    return None


# ---------------------------------------------------------------------------
# Family: complement_pair -- ``(a & b) | (a & ~b) → a``
# ---------------------------------------------------------------------------


def _apply_complement_pair_and_or(expr: Expr) -> Optional[Expr]:
    """``(a & b) | (a & ~b) → a`` (and orderings)."""
    if not (isinstance(expr, ExprOp) and expr.op == "|" and len(expr.args) == 2):
        return None
    lhs, rhs = expr.args
    lh = _match_binary_bitwise(lhs, "&")
    rh = _match_binary_bitwise(rhs, "&")
    if lh is None or rh is None:
        return None
    # Each conjunction has two operands; find an ``a`` shared between both
    # conjunctions and a ``b`` that appears on one side and ``~b`` on the
    # other.
    for a in lh:
        if a not in rh:
            continue
        b_left = lh[0] if lh[1] == a else lh[1]
        b_right = rh[0] if rh[1] == a else rh[1]
        if _is_not(b_left) == b_right or _is_not(b_right) == b_left:
            return a
    return None


# ---------------------------------------------------------------------------
# Family: bitwise_in_sum_cancel -- replace ``c*(x op y) + (-x) + (-y)``
# with the dual bitwise op when the algebraic identity holds
# ---------------------------------------------------------------------------


def _find_bitwise_in_sum_cancel(
    args: List[Expr], size: int
) -> Optional[Tuple[int, int, int, Expr]]:
    """Scan a sum ``args`` for a triple ``(bitwise_idx, neg_x_idx, neg_y_idx)``
    where ``args[bitwise_idx]`` is ``(x & y)`` or ``(x | y)`` (optionally
    multiplied by a constant), and the other two terms are ``-x`` and
    ``-y`` respectively. Returns the indices + a replacement expression
    if found.
    """
    for bi in range(len(args)):
        op_arg = args[bi]
        factor = 1
        bitw = op_arg
        if isinstance(op_arg, ExprOp) and op_arg.op == "*" and len(op_arg.args) == 2:
            a, b = op_arg.args
            if isinstance(a, ExprInt):
                factor = int(a) & _mask(size)
                bitw = b
            elif isinstance(b, ExprInt):
                factor = int(b) & _mask(size)
                bitw = a
        if not (isinstance(bitw, ExprOp) and len(bitw.args) == 2):
            continue
        if bitw.op not in ("&", "|"):
            continue
        x, y = bitw.args
        # Look for -x and -y as separate terms.
        for ni in range(len(args)):
            if ni == bi:
                continue
            if _arith_neg_of(args[ni]) != x:
                continue
            for mi in range(len(args)):
                if mi == bi or mi == ni:
                    continue
                if _arith_neg_of(args[mi]) != y:
                    continue
                # Compute the simplified replacement.
                if factor == 1 and bitw.op == "&":
                    repl = ExprOp("-", ExprOp("|", x, y))
                elif factor == 1 and bitw.op == "|":
                    repl = ExprOp("-", ExprOp("&", x, y))
                elif factor == 2 and bitw.op == "&":
                    repl = ExprOp("-", ExprOp("^", x, y))
                elif factor == 2 and bitw.op == "|":
                    repl = ExprOp("^", x, y)
                else:
                    continue
                return (bi, ni, mi, repl)
    return None


def _apply_bitwise_in_sum_cancel(expr: Expr) -> Optional[Expr]:
    """``(x & y) - x - y → -(x | y)``; ``(x | y) - x - y → -(x & y)``;
    ``2*(x & y) - x - y → -(x ^ y)``; ``2*(x | y) - x - y → x ^ y``.

    Each form removes three terms from an n-ary sum and replaces them with
    a single bitwise expression, a net reduction in tree size.
    """
    if not (isinstance(expr, ExprOp) and expr.op == "+" and len(expr.args) >= 3):
        return None
    args = list(expr.args)
    found = _find_bitwise_in_sum_cancel(args, expr.size)
    if found is None:
        return None
    bi, ni, mi, repl = found
    drop = {bi, ni, mi}
    kept = [a for k, a in enumerate(args) if k not in drop]
    kept.append(repl)
    if len(kept) == 1:
        return kept[0]
    return ExprOp("+", *kept)


# ---------------------------------------------------------------------------
# Family: bitw_in_sums  --  merge two bitwise-with-constant terms in a sum
# ---------------------------------------------------------------------------
#
# These identities collapse a pair of bitwise operations whose constant
# operands have *disjoint* bit sets into a single bitwise expression plus
# auxiliary terms. They are sound for arbitrary bit-vector values and
# net-shrink the AST whenever they fire.


def _const_or_with_atom(expr: Expr) -> Optional[Tuple[int, Expr]]:
    """If ``expr`` is ``c | X`` (with c an ExprInt, X non-constant), return
    ``(int(c), X)``; otherwise ``None``."""
    if not (isinstance(expr, ExprOp) and expr.op == "|" and len(expr.args) == 2):
        return None
    a, b = expr.args
    if isinstance(a, ExprInt) and not isinstance(b, ExprInt):
        return int(a), b
    if isinstance(b, ExprInt) and not isinstance(a, ExprInt):
        return int(b), a
    return None


def _or_pairs_with_disjoint_constants(x: Expr, y: Expr) -> Optional[Expr]:
    """``(c1 | X) + (c2 | X) -> ((c1+c2) | X) + X`` when ``c1 & c2 == 0``.

    On bit positions inside ``c1`` or ``c2`` each OR has that bit set
    (contributing ``1`` to the sum from each side) and ``X`` has 0 or 1;
    on bit positions outside the union, both ORs are just ``X``'s bit and
    the pair contributes ``2*X = X + X``. Folding gives the right-hand side.
    """
    fx = _const_or_with_atom(x)
    fy = _const_or_with_atom(y)
    if fx is None or fy is None:
        return None
    c1, atom_x = fx
    c2, atom_y = fy
    if atom_x != atom_y:
        return None
    size = x.size
    mask = _mask(size)
    if (c1 & c2) & mask != 0:
        return None
    const_sum = (c1 + c2) & mask
    return ExprOp(
        "+",
        ExprOp("|", ExprInt(const_sum, size), atom_x),
        atom_x,
    )


def _apply_or_pairs_with_disjoint_constants(expr: Expr) -> Optional[Expr]:
    return _replace_pair_in_sum(expr, _or_pairs_with_disjoint_constants)


def _coeff_and_bitw_with_constant(
    arg: Expr,
) -> Optional[Tuple[int, str, int, Expr]]:
    """Decompose ``arg`` into ``(coeff, op, const, atom)`` where
    ``arg = coeff * (const OP atom)`` and ``OP ∈ {&, |, ^}``. If the term
    is bare ``const OP atom`` the coefficient is 1; unary ``-(const OP X)``
    is recognised as coefficient ``-1``; ``c * (const OP X)`` recognises
    a constant coefficient. Returns ``None`` if no match."""
    size = arg.size
    mask = _mask(size)
    coeff = 1
    inner = arg
    if isinstance(arg, ExprOp) and arg.op == "-" and len(arg.args) == 1:
        coeff = (-1) & mask
        inner = arg.args[0]
    elif isinstance(arg, ExprOp) and arg.op == "*" and len(arg.args) == 2:
        a, b = arg.args
        if isinstance(a, ExprInt) and not isinstance(b, ExprInt):
            coeff = int(a) & mask
            inner = b
        elif isinstance(b, ExprInt) and not isinstance(a, ExprInt):
            coeff = int(b) & mask
            inner = a
    if not isinstance(inner, ExprOp):
        return None
    if inner.op not in ("&", "|", "^") or len(inner.args) != 2:
        return None
    a, b = inner.args
    if isinstance(a, ExprInt) and not isinstance(b, ExprInt):
        return coeff, inner.op, int(a) & mask, b
    if isinstance(b, ExprInt) and not isinstance(a, ExprInt):
        return coeff, inner.op, int(b) & mask, a
    return None


def _diff_bitw_pairs_with_disjoint_constants(x: Expr, y: Expr) -> Optional[Expr]:
    """Mixed-type pair with disjoint constants. Implements three specific
    sub-patterns whose merged form strictly reduces the AST:

    * ``-(a & X) + (b | X) -> (~(a+b) & X) + b``  (a&b==0)
    * ``-2*(a & X) + (b ^ X) -> 2*(~(a+b) & X) - X + b``  (a&b==0)
    * ``2*(a | X) + (b ^ X) -> 2*(~(a+b) & X) + X + 2*a + b``  (a&b==0)
    """
    fx = _coeff_and_bitw_with_constant(x)
    fy = _coeff_and_bitw_with_constant(y)
    if fx is None or fy is None:
        return None
    cx, ox, kx, ax = fx
    cy, oy, ky, ay = fy
    if ax != ay:
        return None
    if ox == oy:
        # Same-op pairs are handled by the dedicated rules above.
        return None
    size = x.size
    mask = _mask(size)
    if (kx & ky) & mask != 0:
        return None
    not_sum = (~((kx + ky) & mask)) & mask
    atom = ax

    minus_one = mask
    two = 2 & mask
    minus_two = (-2) & mask

    def _and_block() -> Expr:
        return ExprOp("&", ExprInt(not_sum, size), atom)

    # Normalise so the first term is the "lighter" coefficient.
    # Pattern 1: -(a & X) + (b | X) -> (~(a+b) & X) + b
    #            cx == -1, ox == '&'; cy == 1, oy == '|'.
    if cx == minus_one and ox == "&" and cy == 1 and oy == "|":
        return ExprOp("+", _and_block(), ExprInt(ky, size))
    if cy == minus_one and oy == "&" and cx == 1 and ox == "|":
        return ExprOp("+", _and_block(), ExprInt(kx, size))

    # Pattern 2: -2*(a & X) + (b ^ X) -> 2*(~(a+b) & X) - X + b
    if cx == minus_two and ox == "&" and cy == 1 and oy == "^":
        return ExprOp(
            "+",
            ExprOp("*", ExprInt(two, size), _and_block()),
            ExprOp("-", atom),
            ExprInt(ky, size),
        )
    if cy == minus_two and oy == "&" and cx == 1 and ox == "^":
        return ExprOp(
            "+",
            ExprOp("*", ExprInt(two, size), _and_block()),
            ExprOp("-", atom),
            ExprInt(kx, size),
        )

    # Pattern 3: 2*(a | X) + (b ^ X) -> 2*(~(a+b) & X) + X + 2*a + b
    if cx == two and ox == "|" and cy == 1 and oy == "^":
        return ExprOp(
            "+",
            ExprOp("*", ExprInt(two, size), _and_block()),
            atom,
            ExprInt((2 * kx) & mask, size),
            ExprInt(ky, size),
        )
    if cy == two and oy == "|" and cx == 1 and ox == "^":
        return ExprOp(
            "+",
            ExprOp("*", ExprInt(two, size), _and_block()),
            atom,
            ExprInt((2 * ky) & mask, size),
            ExprInt(kx, size),
        )

    return None


def _apply_diff_bitw_pairs_with_disjoint_constants(expr: Expr) -> Optional[Expr]:
    return _replace_pair_in_sum(expr, _diff_bitw_pairs_with_disjoint_constants)


# ---------------------------------------------------------------------------
# Family: bitw_in_sums_inverse  --  bitwise pairs with bitwise-inverse atom
# ---------------------------------------------------------------------------
#
# These three identities pair two bitwise expressions sharing an atom and
# its bitwise complement under a sum. Each is sound and shrinks the AST.


def _diff_inv_or_minus_andnot(x: Expr, y: Expr) -> Optional[Expr]:
    """``(X | Y) + (-(~X & Y)) -> X``.

    ``X | Y`` covers ``X`` plus the bits of ``Y`` outside ``X``; subtracting
    those bits (which is exactly ``~X & Y``) leaves only ``X``.
    """
    or_args = _match_binary_bitwise(x, "|")
    inner = _arith_neg_of(y)
    if or_args is None or inner is None:
        return None
    and_args = _match_binary_bitwise(inner, "&")
    if and_args is None:
        return None
    # Need (~p) & q where (p, q) matches the OR's args in some order.
    for np_idx in (0, 1):
        not_p = and_args[np_idx]
        p = _is_not(not_p)
        if p is None:
            continue
        q = and_args[1 - np_idx]
        if (p, q) == or_args or (q, p) == or_args:
            return p
    return None


def _apply_diff_inv_or_minus_andnot(expr: Expr) -> Optional[Expr]:
    return _replace_pair_in_sum(expr, _diff_inv_or_minus_andnot)


def _diff_inv_xor_minus_andnot(x: Expr, y: Expr) -> Optional[Expr]:
    """``(X ^ Y) + (-2*(~X & Y)) -> X - Y``.

    On bits inside ``Y`` and outside ``X`` the XOR yields 1 and the
    correction subtracts ``2*1``; on bits inside ``X`` and outside ``Y``
    the XOR yields 1 and the correction is 0; etc. The aggregate evaluates
    to ``X - Y``.
    """
    xor_args = _match_binary_bitwise(x, "^")
    if xor_args is None:
        return None
    # y must be -2 * (~X & Y).
    if not (isinstance(y, ExprOp) and y.op == "*" and len(y.args) == 2):
        return None
    size = x.size
    mask = _mask(size)
    neg_two = (-2) & mask
    a, b = y.args
    if isinstance(a, ExprInt) and (int(a) & mask) == neg_two:
        inner = b
    elif isinstance(b, ExprInt) and (int(b) & mask) == neg_two:
        inner = a
    else:
        return None
    and_args = _match_binary_bitwise(inner, "&")
    if and_args is None:
        return None
    for np_idx in (0, 1):
        not_p = and_args[np_idx]
        p = _is_not(not_p)
        if p is None:
            continue
        q = and_args[1 - np_idx]
        if (p, q) == xor_args or (q, p) == xor_args:
            return ExprOp("+", p, ExprOp("-", q))
    return None


def _apply_diff_inv_xor_minus_andnot(expr: Expr) -> Optional[Expr]:
    return _replace_pair_in_sum(expr, _diff_inv_xor_minus_andnot)


def _diff_inv_xor_plus_ornot(x: Expr, y: Expr) -> Optional[Expr]:
    """``(X ^ Y) + 2*(~X | Y) -> -2 + (-X) + Y``.

    Sound by truth-table enumeration of the four ``(X, Y)`` bit
    combinations and a constant adjustment.
    """
    xor_args = _match_binary_bitwise(x, "^")
    if xor_args is None:
        return None
    if not (isinstance(y, ExprOp) and y.op == "*" and len(y.args) == 2):
        return None
    size = x.size
    mask = _mask(size)
    two = 2 & mask
    a, b = y.args
    if isinstance(a, ExprInt) and (int(a) & mask) == two:
        inner = b
    elif isinstance(b, ExprInt) and (int(b) & mask) == two:
        inner = a
    else:
        return None
    or_args = _match_binary_bitwise(inner, "|")
    if or_args is None:
        return None
    for np_idx in (0, 1):
        not_p = or_args[np_idx]
        p = _is_not(not_p)
        if p is None:
            continue
        q = or_args[1 - np_idx]
        if (p, q) == xor_args or (q, p) == xor_args:
            minus_two = ExprInt((-2) & mask, size)
            return ExprOp("+", minus_two, ExprOp("-", p), q)
    return None


def _apply_diff_inv_xor_plus_ornot(expr: Expr) -> Optional[Expr]:
    return _replace_pair_in_sum(expr, _diff_inv_xor_plus_ornot)


# ---------------------------------------------------------------------------
# Family: nested_bitwise  --  identities involving nested bitwise ops
# ---------------------------------------------------------------------------


def _apply_xor_involving_disj(expr: Expr) -> Optional[Expr]:
    """``X ^ (X | Y) -> ~X & Y`` (and the symmetric argument order).

    On bits where ``X`` is 1, both sides agree (XOR yields 0 and the
    conjunction with ``~X`` is 0). On bits where ``X`` is 0, the OR's bit
    is ``Y``'s bit and the XOR with ``X`` is just ``Y``'s bit, matching
    ``~X & Y`` = ``1 & Y``.
    """
    if not (isinstance(expr, ExprOp) and expr.op == "^" and len(expr.args) == 2):
        return None
    lhs, rhs = expr.args

    def _try(x: Expr, or_expr: Expr) -> Optional[Expr]:
        or_args = _match_binary_bitwise(or_expr, "|")
        if or_args is None:
            return None
        if or_args[0] == x:
            return ExprOp("&", _not_of(x), or_args[1])
        if or_args[1] == x:
            return ExprOp("&", _not_of(x), or_args[0])
        return None

    return _try(lhs, rhs) or _try(rhs, lhs)


def _apply_negative_bitw_inverse_and(expr: Expr) -> Optional[Expr]:
    """``-(X & -X) -> X | -X``. Sound because ``X & -X`` is the lowest set
    bit of ``X``; its arithmetic negation matches ``X | -X`` on every
    valuation."""
    inner = _arith_neg_of(expr)
    if inner is None:
        return None
    and_args = _match_binary_bitwise(inner, "&")
    if and_args is None:
        return None
    a, b = and_args
    if _arith_neg_of(b) == a:
        return ExprOp("|", a, b)
    if _arith_neg_of(a) == b:
        return ExprOp("|", b, a)
    return None


def _apply_negative_bitw_inverse_or(expr: Expr) -> Optional[Expr]:
    """``-(X | -X) -> X & -X``. Dual of the above."""
    inner = _arith_neg_of(expr)
    if inner is None:
        return None
    or_args = _match_binary_bitwise(inner, "|")
    if or_args is None:
        return None
    a, b = or_args
    if _arith_neg_of(b) == a:
        return ExprOp("&", a, b)
    if _arith_neg_of(a) == b:
        return ExprOp("&", b, a)
    return None


def _apply_disj_sub_disj_identity(expr: Expr) -> Optional[Expr]:
    """``X | ((X | Y) - Y) -> X``.

    Sound: ``(X | Y) - Y`` has every bit of ``X`` set (bits in ``X & ~Y``
    survive directly, bits in ``X & Y`` survive as a difference cascade),
    so the outer OR with ``X`` is absorbed into ``X``.
    """
    if not (isinstance(expr, ExprOp) and expr.op == "|" and len(expr.args) >= 2):
        return None
    args = list(expr.args)
    for i in range(len(args)):
        x = args[i]
        for j in range(len(args)):
            if j == i:
                continue
            sub = args[j]
            # sub must be a binary subtract: (X|Y) - Y
            if not (isinstance(sub, ExprOp) and sub.op == "-" and len(sub.args) == 2):
                continue
            left, right = sub.args
            or_args = _match_binary_bitwise(left, "|")
            if or_args is None:
                continue
            if x not in or_args:
                continue
            other = or_args[0] if or_args[1] == x else or_args[1]
            if other == right:
                return _drop_clause(expr, "|", j)
    return None


def _apply_disj_sub_conj_identity(expr: Expr) -> Optional[Expr]:
    """``X | (X - (X & Y)) -> X``.

    ``X & Y`` is bitwise-subset of ``X``, so ``X - (X & Y)`` has only bits
    inside ``X``; the outer OR collapses.
    """
    if not (isinstance(expr, ExprOp) and expr.op == "|" and len(expr.args) >= 2):
        return None
    args = list(expr.args)
    for i in range(len(args)):
        x = args[i]
        for j in range(len(args)):
            if j == i:
                continue
            sub = args[j]
            if not (isinstance(sub, ExprOp) and sub.op == "-" and len(sub.args) == 2):
                continue
            left, right = sub.args
            if left != x:
                continue
            and_args = _match_binary_bitwise(right, "&")
            if and_args is None:
                continue
            if x in and_args:
                return _drop_clause(expr, "|", j)
    return None


def _apply_conj_add_conj_identity(expr: Expr) -> Optional[Expr]:
    """``X & (X + (~X & Y)) -> X``.

    The sum ``X + (~X & Y)`` is equal to ``X | Y`` because ``X`` and
    ``~X & Y`` are bitwise-disjoint; therefore the outer AND with ``X``
    yields ``X``.
    """
    if not (isinstance(expr, ExprOp) and expr.op == "&" and len(expr.args) >= 2):
        return None
    args = list(expr.args)
    for i in range(len(args)):
        x = args[i]
        for j in range(len(args)):
            if j == i:
                continue
            plus = args[j]
            if not (
                isinstance(plus, ExprOp) and plus.op == "+" and len(plus.args) == 2
            ):
                continue
            left, right = plus.args
            for x_in_sum, andterm in ((left, right), (right, left)):
                if x_in_sum != x:
                    continue
                and_args = _match_binary_bitwise(andterm, "&")
                if and_args is None:
                    continue
                for not_idx in (0, 1):
                    not_x = and_args[not_idx]
                    if _is_not(not_x) == x:
                        return _drop_clause(expr, "&", j)
    return None


# ---------------------------------------------------------------------------
# Family: disj_conj_dual  --  dual nested identities
# ---------------------------------------------------------------------------


def _apply_conj_conj_disj(expr: Expr) -> Optional[Expr]:
    """``X & -(-Y & (X | Y)) -> X & Y``.

    The negated inner expression equals ``Y & ~X``-ish complement; the
    outer AND with ``X`` selects ``X & Y``.
    """
    if not (isinstance(expr, ExprOp) and expr.op == "&" and len(expr.args) >= 2):
        return None
    args = list(expr.args)
    for i in range(len(args)):
        x = args[i]
        for j in range(len(args)):
            if j == i:
                continue
            neg = _arith_neg_of(args[j])
            if neg is None:
                continue
            inner_and_args = _match_binary_bitwise(neg, "&")
            if inner_and_args is None:
                continue
            # One side is -Y, the other is (X | Y) where Y appears on both.
            for ny_idx in (0, 1):
                ny = inner_and_args[ny_idx]
                y = _arith_neg_of(ny)
                if y is None:
                    continue
                or_expr = inner_and_args[1 - ny_idx]
                or_args = _match_binary_bitwise(or_expr, "|")
                if or_args is None:
                    continue
                if x in or_args and y in or_args:
                    replacement = ExprOp("&", x, y)
                    kept = [a for k, a in enumerate(args) if k != i and k != j]
                    kept.append(replacement)
                    if len(kept) == 1:
                        return kept[0]
                    return ExprOp("&", *kept)
    return None


def _apply_disj_disj_conj(expr: Expr) -> Optional[Expr]:
    """``X | -(-Y | (X & Y)) -> X | Y``. Dual of the above."""
    if not (isinstance(expr, ExprOp) and expr.op == "|" and len(expr.args) >= 2):
        return None
    args = list(expr.args)
    for i in range(len(args)):
        x = args[i]
        for j in range(len(args)):
            if j == i:
                continue
            neg = _arith_neg_of(args[j])
            if neg is None:
                continue
            inner_or_args = _match_binary_bitwise(neg, "|")
            if inner_or_args is None:
                continue
            for ny_idx in (0, 1):
                ny = inner_or_args[ny_idx]
                y = _arith_neg_of(ny)
                if y is None:
                    continue
                and_expr = inner_or_args[1 - ny_idx]
                and_args = _match_binary_bitwise(and_expr, "&")
                if and_args is None:
                    continue
                if x in and_args and y in and_args:
                    replacement = ExprOp("|", x, y)
                    kept = [a for k, a in enumerate(args) if k != i and k != j]
                    kept.append(replacement)
                    if len(kept) == 1:
                        return kept[0]
                    return ExprOp("|", *kept)
    return None


# ---------------------------------------------------------------------------
# Family: disj_xor_specific  --  ``2*X | ~-(X ^ -X) -> -1``
# ---------------------------------------------------------------------------


def _apply_conj_neg_xor_minus_one(expr: Expr) -> Optional[Expr]:
    """``2*X | ~-(X ^ -X) -> -1``.

    Sound: ``~-(X ^ -X)`` equals ``X ^ -X`` (because ``-(a) = ~a + 1`` so
    ``~(-a) = a - 1`` and ``(X ^ -X) - 1`` has the same bit-1+ pattern as
    ``X ^ -X``); together with ``2*X`` the OR covers every bit position.
    """
    if not (isinstance(expr, ExprOp) and expr.op == "|" and len(expr.args) >= 2):
        return None
    args = list(expr.args)
    for i in range(len(args)):
        doubled = _arith_double_of(args[i])
        if doubled is None:
            continue
        for j in range(len(args)):
            if j == i:
                continue
            inner = _is_not(args[j])
            if inner is None:
                continue
            neg = _arith_neg_of(inner)
            if neg is None:
                continue
            xor_base = _is_xor_with_neg(neg)
            if xor_base is None:
                continue
            if xor_base == doubled or _arith_neg_of(xor_base) == doubled:
                return ExprInt(_mask(expr.size), expr.size)
    return None


# ---------------------------------------------------------------------------
# Default rule registry
# ---------------------------------------------------------------------------


DEFAULT_RULES: List[RewriteRule] = [
    # Guarded post-pass: ring normalisation (legacy) + deep factorisation
    RING_NORMALIZE_RULE,
    FACTOR_COMMON_SUBTERM_RULE,
    # Family: inverse_element
    RewriteRule(
        name="inverse_xor_neg",
        family="inverse_element",
        guarded=False,
        apply=_apply_inverse_xor_neg,
    ),
    RewriteRule(
        name="inverse_or_neg",
        family="inverse_element",
        guarded=False,
        apply=_apply_inverse_or_neg,
    ),
    RewriteRule(
        name="inverse_xor_neg_xor",
        family="inverse_element",
        guarded=False,
        apply=_apply_inverse_xor_neg_xor,
    ),
    # Family: two_complement
    RewriteRule(
        name="add_complement_pair",
        family="two_complement",
        guarded=False,
        apply=_apply_add_complement_pair,
    ),
    RewriteRule(
        name="xor_complement_pair",
        family="two_complement",
        guarded=False,
        apply=_apply_xor_complement_pair,
    ),
    # Family: constant_merge
    RewriteRule(
        name="constant_merge_and",
        family="constant_merge",
        guarded=False,
        apply=_apply_constant_merge_and,
    ),
    # Family: power_of_two
    RewriteRule(
        name="factor_pow2_from_and",
        family="power_of_two",
        guarded=False,
        apply=_factor_pow2_from_bitwise("&"),
    ),
    RewriteRule(
        name="factor_pow2_from_or",
        family="power_of_two",
        guarded=False,
        apply=_factor_pow2_from_bitwise("|"),
    ),
    RewriteRule(
        name="factor_pow2_from_xor",
        family="power_of_two",
        guarded=False,
        apply=_factor_pow2_from_bitwise("^"),
    ),
    # Family: bitwise_flatten
    RewriteRule(
        name="or_xor_split",
        family="bitwise_flatten",
        guarded=False,
        apply=_apply_or_xor_split,
    ),
    # Family: demorgan
    RewriteRule(
        name="demorgan_and_to_or",
        family="demorgan",
        guarded=False,
        apply=_demorgan("&", "|"),
    ),
    RewriteRule(
        name="demorgan_or_to_and",
        family="demorgan",
        guarded=False,
        apply=_demorgan("|", "&"),
    ),
    # Family: absorption (GAMBA §5.2)
    RewriteRule(
        name="absorption_or",
        family="absorption",
        guarded=False,
        apply=_apply_absorption_or,
    ),
    RewriteRule(
        name="absorption_and",
        family="absorption",
        guarded=False,
        apply=_apply_absorption_and,
    ),
    # Family: redundancy (GAMBA §5.2)
    RewriteRule(
        name="redundancy_or_not",
        family="redundancy",
        guarded=False,
        apply=_apply_redundancy_or_not,
    ),
    RewriteRule(
        name="redundancy_and_not",
        family="redundancy",
        guarded=False,
        apply=_apply_redundancy_and_not,
    ),
    # Family: idempotence (n-ary duplicate-child elimination)
    RewriteRule(
        name="idempotence_and_drop_duplicates",
        family="idempotence",
        guarded=False,
        apply=_drop_duplicates_apply("&"),
    ),
    RewriteRule(
        name="idempotence_or_drop_duplicates",
        family="idempotence",
        guarded=False,
        apply=_drop_duplicates_apply("|"),
    ),
    RewriteRule(
        name="xor_self_cancel_pairs",
        family="idempotence",
        guarded=False,
        apply=_apply_xor_self_cancel,
    ),
    # Family: double_negation
    RewriteRule(
        name="double_negation_collapse",
        family="double_negation",
        guarded=False,
        apply=_apply_double_negation,
    ),
    # Family: const_fold (n-ary identity/annihilator constants).
    # The single ``_fold_const_bitwise`` call captures both directions:
    # the identity constant is dropped, and the annihilator constant
    # short-circuits the entire op. We register a rule per (op, role)
    # pair for clarity in telemetry; each fires on its own matching
    # input even though the underlying function is shared.
    RewriteRule(
        name="const_fold_and",
        family="const_fold",
        guarded=False,
        apply=_fold_const_bitwise("&", -1, 0),
    ),
    RewriteRule(
        name="const_fold_or",
        family="const_fold",
        guarded=False,
        apply=_fold_const_bitwise("|", 0, -1),
    ),
    RewriteRule(
        name="const_fold_xor_zero",
        family="const_fold",
        guarded=False,
        apply=_fold_const_bitwise("^", 0, None),
    ),
    RewriteRule(
        name="const_fold_add_zero",
        family="const_fold",
        guarded=False,
        apply=_apply_const_fold_add_zero,
    ),
    RewriteRule(
        name="const_fold_mul_one",
        family="const_fold",
        guarded=False,
        apply=_apply_const_fold_mul_one,
    ),
    RewriteRule(
        name="const_fold_mul_zero",
        family="const_fold",
        guarded=False,
        apply=_apply_const_fold_mul_zero,
    ),
    # Family: bitwise_zero (conjunctions that collapse to zero)
    RewriteRule(
        name="conj_self_neg_double_collapses_to_zero",
        family="bitwise_zero",
        guarded=False,
        apply=_apply_conj_self_neg_double_zero,
    ),
    RewriteRule(
        name="conj_neg_xor_collapses_to_zero",
        family="bitwise_zero",
        guarded=False,
        apply=_apply_conj_neg_xor_zero,
    ),
    RewriteRule(
        name="conj_negated_xor_collapses_to_zero",
        family="bitwise_zero",
        guarded=False,
        apply=_apply_conj_negated_xor_zero,
    ),
    # Family: bitwise_identity_clause (drop an algebraic-identity clause)
    RewriteRule(
        name="conj_xor_identity_clause",
        family="bitwise_identity_clause",
        guarded=False,
        apply=_apply_conj_xor_identity,
    ),
    RewriteRule(
        name="disj_xor_identity_clause",
        family="bitwise_identity_clause",
        guarded=False,
        apply=_apply_disj_xor_identity,
    ),
    # Family: nested_bitwise_absorb
    RewriteRule(
        name="disj_disj_negation_absorb",
        family="nested_bitwise_absorb",
        guarded=False,
        apply=_apply_disj_disj_negation_absorb,
    ),
    RewriteRule(
        name="conj_conj_negation_absorb",
        family="nested_bitwise_absorb",
        guarded=False,
        apply=_apply_conj_conj_negation_absorb,
    ),
    RewriteRule(
        name="disj_conj_negation_absorb",
        family="nested_bitwise_absorb",
        guarded=False,
        apply=_apply_disj_conj_negation_absorb,
    ),
    RewriteRule(
        name="disj_neg_disj_identity",
        family="nested_bitwise_absorb",
        guarded=False,
        apply=_apply_disj_neg_disj_identity,
    ),
    # Family: xor_same_mult_collapse
    RewriteRule(
        name="xor_same_mult_or",
        family="xor_same_mult_collapse",
        guarded=False,
        apply=_apply_xor_same_mult_or,
    ),
    RewriteRule(
        name="xor_same_mult_and",
        family="xor_same_mult_collapse",
        guarded=False,
        apply=_apply_xor_same_mult_and,
    ),
    # Family: complement_pair
    RewriteRule(
        name="complement_pair_and_or",
        family="complement_pair",
        guarded=False,
        apply=_apply_complement_pair_and_or,
    ),
    # Family: bitwise_in_sum_cancel
    RewriteRule(
        name="bitwise_in_sum_cancel",
        family="bitwise_in_sum_cancel",
        guarded=False,
        apply=_apply_bitwise_in_sum_cancel,
    ),
    # Family: bitw_in_sums (merge bitwise-with-constant pairs in a sum)
    RewriteRule(
        name="or_pairs_with_disjoint_constants",
        family="bitw_in_sums",
        guarded=False,
        apply=_apply_or_pairs_with_disjoint_constants,
    ),
    RewriteRule(
        name="diff_bitw_pairs_with_disjoint_constants",
        family="bitw_in_sums",
        guarded=False,
        apply=_apply_diff_bitw_pairs_with_disjoint_constants,
    ),
    # Family: bitw_in_sums_inverse (paired bitwise terms with inverse atom)
    RewriteRule(
        name="diff_inv_or_minus_andnot",
        family="bitw_in_sums_inverse",
        guarded=False,
        apply=_apply_diff_inv_or_minus_andnot,
    ),
    RewriteRule(
        name="diff_inv_xor_minus_andnot",
        family="bitw_in_sums_inverse",
        guarded=False,
        apply=_apply_diff_inv_xor_minus_andnot,
    ),
    RewriteRule(
        name="diff_inv_xor_plus_ornot",
        family="bitw_in_sums_inverse",
        guarded=False,
        apply=_apply_diff_inv_xor_plus_ornot,
    ),
    # Family: nested_bitwise
    RewriteRule(
        name="xor_involving_disj",
        family="nested_bitwise",
        guarded=False,
        apply=_apply_xor_involving_disj,
    ),
    RewriteRule(
        name="negative_bitw_inverse_and",
        family="nested_bitwise",
        guarded=False,
        apply=_apply_negative_bitw_inverse_and,
    ),
    RewriteRule(
        name="negative_bitw_inverse_or",
        family="nested_bitwise",
        guarded=False,
        apply=_apply_negative_bitw_inverse_or,
    ),
    RewriteRule(
        name="disj_sub_disj_identity",
        family="nested_bitwise",
        guarded=False,
        apply=_apply_disj_sub_disj_identity,
    ),
    RewriteRule(
        name="disj_sub_conj_identity",
        family="nested_bitwise",
        guarded=False,
        apply=_apply_disj_sub_conj_identity,
    ),
    RewriteRule(
        name="conj_add_conj_identity",
        family="nested_bitwise",
        guarded=False,
        apply=_apply_conj_add_conj_identity,
    ),
    # Family: disj_conj_dual
    RewriteRule(
        name="conj_conj_disj",
        family="disj_conj_dual",
        guarded=False,
        apply=_apply_conj_conj_disj,
    ),
    RewriteRule(
        name="disj_disj_conj",
        family="disj_conj_dual",
        guarded=False,
        apply=_apply_disj_disj_conj,
    ),
    # Family: disj_xor_specific
    RewriteRule(
        name="conj_neg_xor_minus_one",
        family="disj_xor_specific",
        guarded=False,
        apply=_apply_conj_neg_xor_minus_one,
    ),
]


# ---------------------------------------------------------------------------
# Rewriter orchestrator
# ---------------------------------------------------------------------------


def _adapt_to_miasm_pass(rule: RewriteRule):
    """
    Wrap a ``RewriteRule.apply`` (returning ``Optional[Expr]``) into the
    Miasm pass signature ``(ExpressionSimplifier, Expr) -> Expr``. A rule
    that returns ``None`` is reported to Miasm as "no change" by returning
    the input unchanged.
    """

    apply = rule.apply

    def _pass(_simp: ExpressionSimplifier, expr: Expr) -> Expr:
        result = apply(expr)
        return result if result is not None else expr

    _pass.__name__ = f"rewrite_pass_{rule.name}"
    return _pass


class Rewriter:
    """
    Orchestrates a collection of :class:`RewriteRule` objects.

    Two entry points:

    - :meth:`expr_simp` returns a local :class:`ExpressionSimplifier`
      instance preloaded with Miasm's :attr:`ExpressionSimplifier.PASS_COMMONS`
      plus this rewriter's safe (``guarded=False``) rules. The instance is
      a drop-in replacement for the Miasm singleton ``expr_simp`` at any
      call site that uses ``simp(expr)``.
    - :meth:`normalize` runs that simplifier, then applies the guarded
      rules with their own net-smaller checks, then returns the result.
      :class:`msynth.simplification.simplifier.Simplifier` calls this on
      its final reverse-unified expression as the closing post-pass.

    The :class:`ExpressionSimplifier` is built lazily on first use and
    cached on the rewriter instance, so constructing a :class:`Rewriter`
    is cheap.
    """

    def __init__(self, rules: Sequence[RewriteRule] = DEFAULT_RULES) -> None:
        self.rules: Tuple[RewriteRule, ...] = tuple(rules)
        self._expr_simp_cache: Optional[ExpressionSimplifier] = None

    def safe_rules(self) -> Tuple[RewriteRule, ...]:
        return tuple(r for r in self.rules if not r.guarded)

    def guarded_rules(self) -> Tuple[RewriteRule, ...]:
        return tuple(r for r in self.rules if r.guarded)

    def expr_simp(self) -> ExpressionSimplifier:
        """
        Local :class:`ExpressionSimplifier` with Miasm defaults + our
        safe rules. Lazy; cached after first construction.
        """
        if self._expr_simp_cache is not None:
            return self._expr_simp_cache

        simp = ExpressionSimplifier()
        # Miasm's default passes first; ours layer on top.
        simp.enable_passes(ExpressionSimplifier.PASS_COMMONS)

        safe_passes: Dict = {}
        for rule in self.safe_rules():
            # All current and planned safe rules pattern-match on ExprOp.
            # If a future rule needs ExprSlice / ExprCompose dispatch, this
            # is the single place to widen the registration.
            safe_passes.setdefault(ExprOp, []).append(_adapt_to_miasm_pass(rule))
        if safe_passes:
            simp.enable_passes(safe_passes)

        self._expr_simp_cache = simp
        return simp

    def normalize(self, expr: Expr) -> Expr:
        """
        Apply :meth:`expr_simp` then each guarded rule.

        Behaviour-preserving replacement for
        ``ring_normalize(expr_simp(expr))`` when ``DEFAULT_RULES`` carries
        the ring-normalisation rule and no extras. Adding safe rules to a
        rewriter widens what :meth:`expr_simp` reduces; adding guarded
        rules widens what the post-pass collapses.
        """
        simp = self.expr_simp()
        result = simp(expr)
        for rule in self.guarded_rules():
            candidate = rule.apply(result)
            if candidate is not None:
                result = candidate
        return result


# Module-level default rewriter, used by Simplifier's final post-pass.
# Construction is cheap; the heavy expr_simp instance is lazily built.
DEFAULT_REWRITER = Rewriter()
