from typing import List, Iterator
from miasm.expression.expression import Expr


def get_unique_variables(expr: Expr) -> List[Expr]:
    """
    Get all unique variables in an expression.

    To provide deterministic behavior, the list is sorted.

    Args:
        expr: Expression to parse.

    Returns:
        Sorted list of unique variables.
    """
    l = set()

    def add_to_set(e: Expr) -> Expr:
        """
        Helper function to add variables to a set.

        Args:
            expr: Expression to add.

        Returns:
            Expression.
        """
        if e.is_id():
            l.add(e)
        return e

    expr.visit(add_to_set)

    return sorted(l, key=lambda x: str(x))


def get_unification_candidates(expr: Expr) -> List[Expr]:
    """
    Get all unification candidates in an expression.

    A unification candidate is a leaf in an abstract 
    syntax tree (variable, memory or label). Integers 
    are excluded.

    To provide deterministic behavior, the list is sorted.

    Args:
        expr: Expression to parse.

    Returns:
        Sorted list of unification candidates.
    """
    results = set()

    def add_to_set(e: Expr) -> Expr:
        """
        Helper function to add variables, memory 
        and labels to a set.

        Args:
            expr: Expression to add.

        Returns:
            Expression.
        """
        # memory
        if e.is_mem():
            results.add(e)
        # registers
        if e.is_id():
            results.add(e)
        # location IDs
        if e.is_loc():
            results.add(e)
        return e

    expr.visit(add_to_set)

    return sorted(list(results), key=lambda x: str(x))


def get_subexpressions(expr: Expr) -> Iterator[Expr]:
    """
    Get all subexpressions in descending order.

    This can be understood as a breadth-first search
    on an abstract syntax tree from top to bottom.

    Args:
        expr: Expr

    Returns:
        Iterator of expressions.
    """
    l = []

    def add_to_list(e: Expr) -> Expr:
        """
        Helper function to an expression
        to a list.

        Args:
            expr: Expression to add.

        Returns:
            Expression.
        """
        l.append(e)
        return e

    expr.visit(add_to_list)

    return reversed(l)
