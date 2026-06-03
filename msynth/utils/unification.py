from typing import Dict, Sequence, Tuple

from miasm.expression.expression import Expr, ExprId
from msynth.utils.expr_utils import get_unification_candidates


def invert_dict(d: Dict[Expr, Expr]) -> Dict[Expr, Expr]:
    """
    Inverts a dictionary by swapping keys and values
    (each key k becomes the value of its value v).

    Args:
        d: Dictionary of expressions to invert.

    Returns:
        Dictionary with inverted key/value mappings.
    """
    return {v: k for k, v in d.items()}


def gen_unification_dict(expr: Expr) -> Dict[Expr, Expr]:
    """
    Generates a dictionary of unificiation variables.

    For each unification candidate (terminal expressions such
    as registers or memory), we generate placeholder variables
    p<index> of the corresponding terminal expression size.

    The resulting dictionary maps termial expressions to their
    corresponding unification.

    Args:
        expr: Expression to generate unification variables for.

    Returns:
        Dictionary of expressions; terminals are mapped to unification variables.
    """
    return {
        # {x: p0, y: p1, ...,}
        unique_var: ExprId(f"p{index}", unique_var.size)
        for index, unique_var in enumerate(get_unification_candidates(expr))
    }


def reverse_unification(expr: Expr, unification_dict: Dict[Expr, Expr]) -> Expr:
    """
    Reverses the unification of an expression.

    This way, each unified variable in an expression is replaced with
    their corresponding terminal expression in the original expression.
    To achieve this, we first have to inverse the unification dictionary.

    Example: Given: {x: p0, y:p1} and expression p0 + p1. We invert
                the dictionary {p0: x, p1: y}. The expresion becomes
                x + y.

    Args:
        expr: Expression to reverse unification for.
        unification_dict: Dictionary of expressions containing unifications.

    Returns:
        Expression with reversed unification.
    """
    return expr.replace_expr(invert_dict(unification_dict))


def abstract_terms(
    expr: Expr, targets: Sequence[Expr], prefix: str = "p"
) -> Tuple[Expr, Dict[Expr, Expr]]:
    """
    Replace each occurrence of every Expr in ``targets`` with a fresh
    placeholder variable, returning the abstracted expression and the
    placeholder -> original mapping for reverse substitution.

    This is the general placeholder-substitution primitive underlying
    unification. :func:`gen_unification_dict` is the special case that
    auto-discovers all terminals; here the caller supplies the exact set of
    sub-expressions to abstract (e.g. GAMBA's nonlinear leaves) and chooses a
    placeholder ``prefix`` so distinct abstraction passes do not collide
    (``p`` for terminal unification, ``g`` for GAMBA subexpression abstraction).

    Args:
        expr: Expression containing the sub-expressions to abstract.
        targets: Sub-expressions to abstract. Each gets a unique fresh
            placeholder ``ExprId`` named ``{prefix}0``, ``{prefix}1``, ... of
            matching size.
        prefix: Placeholder name prefix.

    Returns:
        Tuple ``(abstracted_expr, mapping)`` where ``abstracted_expr`` is the
        input with every ``target`` replaced and ``mapping`` records
        placeholder -> original for use by :func:`reverse_abstraction`.
    """
    mapping: Dict[Expr, Expr] = {}
    replacements: Dict[Expr, Expr] = {}
    for index, target in enumerate(targets):
        placeholder = ExprId(f"{prefix}{index}", target.size)
        mapping[placeholder] = target
        replacements[target] = placeholder
    return expr.replace_expr(replacements), mapping


def reverse_abstraction(expr: Expr, mapping: Dict[Expr, Expr]) -> Expr:
    """
    Inverse of :func:`abstract_terms` — replace each placeholder var with its
    original sub-expression.

    Unlike :func:`reverse_unification` (which takes the forward
    original -> placeholder dict and inverts it), this takes the
    placeholder -> original mapping produced by :func:`abstract_terms`
    directly.

    Args:
        expr: An abstracted Expr whose placeholder vars match keys in
            ``mapping``.
        mapping: Placeholder -> original Expr mapping from
            :func:`abstract_terms`.

    Returns:
        The expression with all placeholders restored to their originals.
    """
    return expr.replace_expr(mapping)
