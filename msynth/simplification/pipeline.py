"""
Composable simplification pipeline.

A :class:`Pipeline` is an ordered sequence of *passes*. Each pass is any
object with a ``run(expr: Expr) -> Expr`` method (duck-typed) — see
:class:`SimbaPass`, :class:`GambaPreprocessingPass`,
:class:`GambaPostRewriterPass`, :class:`AstNormalizationPass`. The
pipeline runs each pass on the expression in order and returns the final
result. :class:`~msynth.simplification.simplifier.Simplifier` invokes the
pipeline once at the top of :meth:`~Simplifier.simplify`, before its
oracle-driven outer loop.

Two ready-made factories:

- :func:`default_pipeline` — ``[SimbaPass, AstNormalizationPass]``. The
  production setting; intentionally omits the GAMBA pre/post sandwich
  because the §5.2 algebraic identities collapse demo-MBA shapes into a
  conjunction basis that defeats oracle-template matching downstream.
- :func:`gamba_sandwich_pipeline` — ``[GambaPreprocessingPass, SimbaPass,
  GambaPostRewriterPass, AstNormalizationPass]``. The opt-in setting for
  SimBA-only / no-oracle runs, where the algebraic refinement on both
  sides of SimBA is a clean win on the corpus (see
  ``tmp/gamba_sweep_report.md`` for measurements).

Both factories accept an ``extra_passes`` sequence inserted just before
the closing :class:`AstNormalizationPass`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from miasm.expression.expression import Expr

from msynth.simplification.ast import AbstractSyntaxTreeTranslator
from msynth.simplification.gamba import GAMBA_POST_REWRITER, GAMBA_PREPROCESSOR
from msynth.simplification.simba import SimbaPass


@dataclass(frozen=True)
class GambaPreprocessingPass:
    """
    Algebraic preprocessor that runs BEFORE SimBA.

    Delegates to the self-contained, miasm-free
    :data:`~msynth.simplification.gamba.GAMBA_PREPROCESSOR`, which
    applies all 53 ``guarded=False`` rules from
    :data:`msynth.simplification.rewrites.DEFAULT_RULES` in a bottom-up
    fixpoint. Every safe rule is sound under Z3 (verified per-rule in
    :mod:`tests.simplification.test_rewrites`) and no-grow on its
    matching input shape. Rule families grouped by what they do:

    *Core boolean identities*
        - **idempotence** (3): ``a & a → a``, ``a | a → a``, ``a ^ a → 0``
          (n-ary; drops duplicate operands).
        - **double_negation** (1): ``~~a → a``.
        - **redundancy** (2): ``a | ~a → -1``, ``a & ~a → 0``.
        - **complement_pair** (1): ``(a & b) | (a & ~b) → a``
          (and orderings).
        - **absorption** (2): ``a | (a & b) → a``, ``a & (a | b) → a``.
        - **demorgan** (2): ``~(a & b) → ~a | ~b`` (and dual);
          gated by an existing NOT so the rewrite net-shrinks.

    *Arithmetic/mixed identities*
        - **two_complement** (2): ``x + ~x → -1``, ``x ^ ~x → -1``
          (any number of additional summands / XOR args).
        - **inverse_element** (3): ``(X & Y) + (~X & Y) → Y`` plus
          OR- and XOR-keyed variants.
        - **const_fold** (6): ``a & 0``, ``a | 0``, ``a ^ 0``, ``a + 0``,
          ``a * 1``, ``a * 0``.
        - **constant_merge** (1): ``(a & X) + (b & X) → (a + b) & X``.
        - **power_of_two** (3): ``(c·X) & (c·Y) → c · (X & Y)``,
          and OR/XOR variants — factor a common power-of-two coefficient.

    *Structural collapses*
        - **bitwise_flatten** (1): ``(X & Y) | (X ^ Y) → (X & Y) + (X ^ Y)``
          (OR↔+ split when arguments are disjoint).
        - **bitwise_zero** (3): three ``x & -x & 2*x → 0`` family
          identities that collapse contradictory conjunctions.
        - **bitwise_identity_clause** (2): ``2*x & (x ^ -x) → 2*x``
          (``x ^ -x`` is all-ones so the clause is an identity);
          OR-dual variant.
        - **bitwise_in_sum_cancel** (1): ``(x & y) − x − y → −(x | y)``
          (and conjugate forms).
        - **xor_same_mult_collapse** (2): ``2·(x | −x) → x ^ −x``
          and the negated-factor variant.

    *Tier 3 GAMBA additions (§5.2)*
        - **bitw_in_sums** (2): ``(c1 | X) + (c2 | X) → ((c1+c2) | X) + X``
          and the diff-of-bitwise-pairs reduction (constants must be
          disjoint).
        - **bitw_in_sums_inverse** (3): three ``−(X OP Y) + (~X AND-NOT Y)
          → ±Y`` family rules.
        - **nested_bitwise** (6): ``x ^ (x | y) → ~x & y``,
          ``−(x & −x) → x | −x``, ``x | ((x | y) − y) → x``,
          ``x | (x − (x & y)) → x``, ``x & (x + (~x & y)) → x``,
          and the OR-dual.
        - **nested_bitwise_absorb** (4): ``x | −((x & y) | −x) → x``,
          ``x & −((x | y) & −x) → x``, ``−x | (~x & 2*x) → −x``,
          ``x | −(−x | 2*x) → x``.
        - **disj_conj_dual** (2): ``x & −(−y & (x | y)) → x & y``
          and its OR/AND dual.
        - **disj_xor_specific** (1): ``2*x | ~−(x ^ −x) → −1``.

    The bottom-up engine in :mod:`~msynth.simplification.gamba` reaches
    a fixed point under these rules in ≤50 iterations (the safety cap;
    a no-grow contract on every rule makes oscillation rare in
    practice).

    Deliberately EXCLUDES the two ``guarded=True`` rules
    (``ring_normalize``, ``factor_common_subterm``) because both can
    DISTRIBUTE or FACTOR in ways that change the *shape* SimBA sees
    on the boolean cube. Distributing ``c·(a+b)`` to ``c·a + c·b``
    widens SimBA's atom set on the demo MBA shape (cf.
    ``test_simplifier_demo_mba_reaches_shortest_form_with_placeholder_guard``).
    Both belong in the post-pass after SimBA, where they run via
    :meth:`~msynth.simplification.gamba.GambaPostRewriter.normalize`
    with an external net-shrink guard.
    """

    name: str = "gamba_preprocessing"

    def run(self, expr: Expr) -> Expr:
        return GAMBA_PREPROCESSOR.normalize(expr)


@dataclass(frozen=True)
class GambaPostRewriterPass:
    """
    Algebraic post-rewriter that runs AFTER SimBA but BEFORE binarisation.

    Delegates to
    :data:`~msynth.simplification.gamba.GAMBA_POST_REWRITER`, which
    applies the same ``guarded=False`` rules as :class:`GambaPreprocessingPass`
    plus the ``guarded=True`` rules (``ring_normalize``,
    ``factor_common_subterm``) with their net-shrink checks intact.

    Placed BEFORE :class:`AstNormalizationPass` because the §5.2 algebraic
    rules pattern-match on n-ary :class:`ExprOp` shapes (e.g. multiple
    duplicate children in a single ``&`` node, or coefficient collection
    across many summands). Running this pass before binarisation lets
    those rules see the natural variadic form SimBA emits; running it
    after binarisation would force every rule to re-flatten internally.
    """

    name: str = "gamba_post_rewriter"

    def run(self, expr: Expr) -> Expr:
        return GAMBA_POST_REWRITER.normalize(expr)


@dataclass(frozen=True)
class Pipeline:
    """
    Composes simplification passes in order. A "pass" is duck-typed:
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


def default_pipeline(
    extra_passes: Sequence[Any] | None = None,
) -> Pipeline:
    """
    Standard simplification pipeline: ``[SimbaPass, *extras,
    AstNormalizationPass]``.

    :class:`GambaPreprocessingPass` and :class:`GambaPostRewriterPass` are
    intentionally NOT in the default pipeline. Their §5.2 algebraic rules
    collapse structural noise into shapes SimBA classifies and reconstructs
    in conjunction basis, which then defeats oracle-template matching
    downstream (cf. the demo-MBA regression: 9 -> 24 nodes when the
    pre/post sandwich is wired into the oracle path). Use them explicitly
    via :func:`gamba_sandwich_pipeline` below or via ``extra_passes`` when
    running SimBA-only pipelines without oracle help, or before SimBA in
    offline benchmarks.

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
    return Pipeline([SimbaPass(), *(extra_passes or ()), AstNormalizationPass()])


def gamba_sandwich_pipeline(
    extra_passes: Sequence[Any] | None = None,
) -> Pipeline:
    """
    Alternative pipeline with GAMBA pre/post sandwich around SimbaPass:
    ``[GambaPreprocessingPass, SimbaPass, GambaPostRewriterPass, *extras,
    AstNormalizationPass]``.

    The pre/post pair runs the §5.2 algebraic rule set before and after
    SimBA's cube reconstruction. The post pass runs BEFORE the binariser
    so the algebraic rules see the natural variadic shape SimBA emits.

    Use this for SimBA-only / no-oracle pipelines where the oracle
    pattern-matching argument doesn't apply. NOT the default because the
    GAMBA-pre transforms shapes into the SimBA-classifier accept set,
    which then produces verbose conjunction-basis output the oracle path
    cannot fold further.
    """
    return Pipeline(
        [
            GambaPreprocessingPass(),
            SimbaPass(),
            GambaPostRewriterPass(),
            *(extra_passes or ()),
            AstNormalizationPass(),
        ]
    )
