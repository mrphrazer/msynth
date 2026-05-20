from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from miasm.expression.expression import Expr

from msynth.simplification.ast import AbstractSyntaxTreeTranslator
from msynth.simplification.simba import SimbaPass


class RewritePass(Protocol):
    """Protocol implemented by preprocessing rewrite passes."""

    name: str

    def run(self, expr: Expr) -> Expr:
        """Return the rewritten expression."""


@dataclass(frozen=True)
class Preprocessor:
    passes: Sequence[RewritePass]

    def run(self, expr: Expr) -> Expr:
        for rewrite_pass in self.passes:
            expr = rewrite_pass.run(expr)
        return expr


@dataclass(frozen=True)
class AstNormalizationPass:
    name: str = "ast"

    def run(self, expr: Expr) -> Expr:
        return AbstractSyntaxTreeTranslator().from_expr(expr)


def default_preprocessor(
    extra_passes: Sequence[RewritePass] | None = None,
) -> Preprocessor:
    return Preprocessor([AstNormalizationPass(), SimbaPass(), *(extra_passes or ())])
