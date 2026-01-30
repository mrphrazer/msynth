from typing import Dict

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
