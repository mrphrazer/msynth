from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from miasm.expression.expression import Expr

from msynth.simplification.ast import AbstractSyntaxTreeTranslator
from msynth.simplification.rewrites import DEFAULT_REWRITER
from msynth.simplification.simba import SimbaPass


@dataclass(frozen=True)
class GambaPreprocessingPass:
    """
    GAMBA-paper-derived algebraic preprocessor that runs BEFORE SimBA.

    Applies only the ``guarded=False`` algebraic rules from
    :data:`msynth.simplification.rewrites.DEFAULT_REWRITER` (via its
    local :class:`ExpressionSimplifier`). These rules always reduce or
    preserve structural complexity and are safe to apply inside Miasm's
    fixpoint:

    - **Absorption** (§5.2): ``a | (a & b) → a``, ``a & (a | b) → a``
    - **Redundancy** (§5.2): ``a | ~a → -1``, ``a & ~a → 0``
    - **Inverse-element** (§5.2): ``(X & Y) + (~X & Y) → Y`` etc.
    - **Two-complement** (§5.2): ``x + ~x → -1`` etc.
    - **Constant-merge** (§5.2): ``(a & X) + (b & X) → (a + b) & X``
    - **Power-of-two factor** (§5.2): ``(c·X) | (c·Y) → c · (X | Y)``
    - **OR↔+ split** (§5.2): ``(X & Y) | (X ^ Y) → (X & Y) + (X ^ Y)``
    - **De Morgan** (§5.2): ``~(a & b) → ~a | ~b`` (guarded by an
      already-existing NOT to ensure net shrink)
    - Miasm's :data:`ExpressionSimplifier.PASS_COMMONS` (idempotence,
      double-negation, constant folding, shift algebra, slice/compose
      elimination, coefficient collection)

    Deliberately EXCLUDES the ``guarded=True`` rules
    (``ring_normalize``, ``factor_common_subterm``) because both can
    DISTRIBUTE or FACTOR in ways that change the *shape* SimBA sees
    on the boolean cube. Distributing ``c·(a+b)`` to ``c·a + c·b``
    widens SimBA's atom set on the demo MBA shape (cf.
    ``test_simplifier_demo_mba_reaches_shortest_form_with_placeholder_guard``).
    Both belong in the post-pass after SimBA, where they already run
    via :meth:`Rewriter.normalize`.
    """

    name: str = "gamba_preprocessing"

    def run(self, expr: Expr) -> Expr:
        return DEFAULT_REWRITER.expr_simp()(expr)


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

    :class:`GambaPreprocessingPass` is intentionally NOT in the default
    pipeline. The §5.2 algebraic rules collapse structural noise into
    shapes SimBA classifies and reconstructs in conjunction basis,
    which then defeats oracle-template matching downstream (cf. the
    demo-MBA regression: 9 -> 25 nodes). Use it explicitly via
    ``extra_passes=[GambaPreprocessingPass()]`` when running SimBA-only
    pipelines without oracle help, or before SimBA in offline
    benchmarks.

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
