"""
Ring-normalisation post-pass for the Simplifier output.

Miasm's ``expr_simp`` deliberately preserves factored forms like
``(v0 + v1) * 0x2`` and refuses to redistribute them, even when a
surrounding ``+`` has matching atoms that would collect under
distribution. The factored form is shorter in isolation, so the choice
is sound for a general-purpose simplifier — but inside an MBA-targeted
toolchain it blocks like-term collection on outputs from paths that
produce ``c * (sum)`` shapes (subtree-SiMBA refinement,
CEGIS-instantiated templates, hand-written inputs).

``ring_normalize`` runs after the main simplification loop. It walks
the AST, flattens nested ``+``, distributes ``c * (sum)`` over the
surrounding sum, collects coefficients per atom (using Miasm's
structural equality), and rebuilds. **The rebuilt expression is kept
only when its Miasm graph is strictly smaller than the input.** That
guard makes the pass safe to call unconditionally — distribution
cannot inflate the output, because the inflated form is rejected.

The pass operates over the unsigned modular ring of width ``expr.size``
(coefficients are reduced modulo ``2**size``), which matches the
semantics of every other arithmetic step in the Simplifier.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from miasm.expression.expression import Expr, ExprInt, ExprOp


def ring_normalize(expr: Expr) -> Expr:
    """
    Return a ring-normalised form of ``expr`` if strictly smaller than
    the input by Miasm graph-node count, else return ``expr`` unchanged.

    Safe to call on any ``Expr``; non-sum-rooted expressions are returned
    as-is. The net-smaller guard prevents the
    "distribution-explodes-term-count" failure mode.
    """
    if not _has_sum_node(expr):
        return expr
    rebuilt = _normalise(expr)
    if rebuilt is expr or rebuilt == expr:
        return expr
    if _node_count(rebuilt) < _node_count(expr):
        return rebuilt
    return expr


def _node_count(expr: Expr) -> int:
    return len(expr.graph().nodes())


def _has_sum_node(expr: Expr) -> bool:
    """Quick check: does ``expr`` contain any ``+`` operator anywhere?"""
    if isinstance(expr, ExprOp) and expr.op == "+":
        return True
    if isinstance(expr, ExprOp):
        return any(_has_sum_node(arg) for arg in expr.args)
    return False


def _normalise(expr: Expr) -> Expr:
    """
    Recursively normalise children, then if this node is a ``+``,
    flatten + distribute + collect coefficients + rebuild.
    """
    if not isinstance(expr, ExprOp):
        return expr

    # Recurse into children first so nested sums are flattened bottom-up.
    new_args = tuple(_normalise(arg) for arg in expr.args)
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

    # 2) distribute c * (sum)  ->  c*a + c*b + …
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
    atom_order: List[Expr] = []  # preserve first-seen order for stability
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


def _as_const_times_sum(
    expr: Expr, size: int
) -> Optional[Tuple[int, Tuple[Expr, ...]]]:
    """
    If ``expr`` is ``c * (a + b + …)`` (any number of constant factors and
    exactly one non-int operand which is itself a ``+``), return
    ``(c, (a, b, …))``; otherwise ``None``.
    """
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

    - For a bare constant: ``(value, None)``.
    - For an explicit ``c * X`` with one non-int operand: peeling recurses
      into ``X`` so chained constant multiplications fold into one coefficient
      and a bare atom.
    - For unary ``-X``: peeling recurses into ``X`` with the coefficient
      negated modulo ``2**size``.
    - For anything else: ``(1, expr)`` — atom is the whole expression.

    The recursion guarantees the returned atom is never itself a ``c * X``
    or a ``-X`` — i.e. the coefficient is fully extracted into the int.
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
    """Build ``coeff * atom`` while folding the trivial coefficients."""
    coeff &= _mask(size)
    if coeff == 0:
        return ExprInt(0, size)
    if coeff == 1:
        return atom
    return ExprOp("*", ExprInt(coeff, size), atom)


def _mask(size: int) -> int:
    return (1 << size) - 1 if size > 0 else 0
