from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from miasm.expression.expression import Expr

from msynth.simplification.ast import AbstractSyntaxTreeTranslator
from msynth.simplification.simba import SimbaPass


@dataclass(frozen=True)
class Preprocessor:
    """
    Composes preprocessing passes in order. A "pass" is duck-typed:
    any object with a ``run(expr: Expr) -> Expr`` method qualifies.
    """

    passes: Sequence[Any]

    def run(self, expr: Expr) -> Expr:
        for rewrite_pass in self.passes:
            expr = rewrite_pass.run(expr)
        return expr


@dataclass(frozen=True)
class AstNormalizationPass:
    """
    Convert variadic ``ExprOp`` / ``ExprCompose`` nodes to a strict
    binary tree. Load-bearing for the main simplifier loop, whose
    ``get_subexpressions`` walk exposes intermediate sub-pairs only when
    they exist as physical AST nodes. See ``ast.py`` for the translator.
    """

    name: str = "ast"

    def run(self, expr: Expr) -> Expr:
        return AbstractSyntaxTreeTranslator().from_expr(expr)


def default_preprocessor(
    extra_passes: Sequence[Any] | None = None,
) -> Preprocessor:
    """
    Standard preprocessing pipeline: ``[SimbaPass, *extras,
    AstNormalizationPass]``.

    - :class:`SimbaPass` runs linear-MBA reconstruction on whatever
      SimBA can classify. SimBA's classifier, cube evaluator, and
      reconstruction are all arity-tolerant — they operate on
      ``expr.args`` as a uniform iterable — so SimBA does not require
      binarised input.
    - :class:`AstNormalizationPass` at the *tail* binarises whatever
      SimBA emits (variadic by construction; see ``_sum`` / ``_or`` /
      ``_xor`` / ``_conjunction`` in simba.py) before the main
      simplifier loop's ``get_subexpressions`` walk runs. That walk
      only exposes intermediate sub-pair nodes when they exist as
      physical AST nodes, so binarising the tree at the boundary
      maximises the oracle's lookup surface on multi-arg sums.
    """
    return Preprocessor([SimbaPass(), *(extra_passes or ()), AstNormalizationPass()])
