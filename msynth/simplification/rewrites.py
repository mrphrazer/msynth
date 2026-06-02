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

    ``a * b * c`` -> ``[a, b, c]``. Bare ``x`` -> ``[x]``. A unary
    ``-arg`` is treated as the factor list of ``arg`` extended with a
    ``-1`` constant so the common-factor matching works on the
    underlying product.
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
