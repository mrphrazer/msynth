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

Three ready-made factories that align with the three :class:`PipelineMode`
values:

- :func:`default_pipeline` (``PipelineMode.AST``) — ``[AstNormalizationPass]``.
  Pure structural normalisation. No simplification work. Used when the
  caller does not want SimBA / GAMBA — for example oracle-only sweeps,
  or as a baseline against which other modes are measured.
- :func:`simba_pipeline` (``PipelineMode.SIMBA``) — ``[SimbaPass,
  AstNormalizationPass]``. Linear-MBA reconstruction followed by
  binarisation.
- :func:`gamba_pipeline` (``PipelineMode.GAMBA``) — ``[GambaPreprocessingPass,
  SimbaPass, GambaPostRewriterPass, AstNormalizationPass]``. GAMBA's
  §5.2 algebraic rewriter wraps SimBA on both sides; pre exposes more
  linear-MBA shapes, post collapses SimBA's verbose conjunction-basis
  output.

If a caller needs custom passes that don't fit any of the three modes,
they construct a :class:`Pipeline` directly and pass it to the
simplifier via :class:`Simplifier`'s ``pipeline=`` override. The
factories themselves take no parameters — every mode is a fixed
composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from miasm.expression.expression import Expr

from msynth.simplification.ast import AbstractSyntaxTreeTranslator
from msynth.simplification.gamba import (
    GAMBA_POST_REWRITER,
    GAMBA_PREPROCESSOR,
)
from msynth.simplification.simba import SimbaPass


class PipelineMode(str, Enum):
    """
    Pre-defined simplification-pipeline configurations.

    Each value maps to a single factory function and to a corresponding
    subtree-SimBA behaviour inside the simplifier:

    - :attr:`AST` — :func:`default_pipeline`; subtree-SimBA disabled.
    - :attr:`SIMBA` — :func:`simba_pipeline`; subtree-SimBA enabled,
      no GAMBA wrap.
    - :attr:`GAMBA` — :func:`gamba_pipeline`; subtree-SimBA enabled
      and wrapped with GAMBA pre/post (matching the global pipeline).

    The enum inherits from :class:`str` so callers may pass the bare
    literal (e.g. ``Simplifier(pipeline_mode="simba")``) when an enum
    import is inconvenient.
    """

    AST = "ast"
    SIMBA = "simba"
    GAMBA = "gamba"


@dataclass(frozen=True)
class GambaPreprocessingPass:
    """
    Algebraic preprocessor that runs BEFORE SimBA.

    Delegates to the self-contained, miasm-free
    :data:`~msynth.simplification.gamba.GAMBA_PREPROCESSOR`, which
    applies every ``guarded=False`` rule from
    :data:`msynth.simplification.rewrites.DEFAULT_RULES` in a bottom-up
    fixpoint (each rule is tried at every node). Every safe rule is sound
    under Z3 (verified per-rule in
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
    ``get_subexpressions`` walk exposes intermediate sub-pair nodes only
    when they exist as physical AST nodes. See ``ast.py`` for the
    translator.
    """

    name: str = "ast"

    def run(self, expr: Expr) -> Expr:
        return AbstractSyntaxTreeTranslator().from_expr(expr)


def default_pipeline() -> Pipeline:
    """
    Pure-normalisation pipeline (the ``PipelineMode.AST`` preset).

    Pipeline shape: ``[AstNormalizationPass]``.

    The single pass binarises variadic :class:`ExprOp` /
    :class:`ExprCompose` nodes into a strict binary tree. That binary
    form is load-bearing for the main simplifier loop: its
    ``get_subexpressions`` walk only exposes intermediate sub-pair nodes
    when they physically exist in the AST, and the oracle lookup is
    keyed on those sub-pair shapes.

    No simplification work is done here — this pipeline is the right
    choice when:

    - You only care about the oracle lookup (the simplifier loop will
      still query :class:`Simplifier.oracle` for every subtree).
    - You want a baseline against which SIMBA / GAMBA modes are
      measured.
    - You're running CEGIS-only and the SimBA pre-pass would just be
      wasted work on shapes SimBA can't classify.
    """
    return Pipeline([AstNormalizationPass()])


def simba_pipeline() -> Pipeline:
    """
    SimBA linear-MBA reconstruction pipeline (the ``PipelineMode.SIMBA``
    preset).

    Pipeline shape: ``[SimbaPass, AstNormalizationPass]``.

    Phase-by-phase rationale:

    1. **SimbaPass** — runs on the raw input. SimBA's classifier, cube
       evaluator, and reconstruction are all *arity-tolerant*: they
       operate on ``expr.args`` as a uniform iterable, so SimBA does not
       require binarised input. Running SimBA first means it sees the
       expression in its natural (possibly variadic) form, which keeps
       its atom set tight and avoids the overhead of an upstream
       binariser.

    2. **AstNormalizationPass** — runs LAST. SimBA's reconstruction
       helpers (``_sum`` / ``_or`` / ``_xor`` / ``_conjunction`` in
       ``simba.py``) emit variadic :class:`ExprOp` nodes; the binariser
       splits those into a strict binary tree so the simplifier loop's
       ``get_subexpressions`` walk can expose intermediate sub-pair
       nodes for oracle lookup.

    Nothing else belongs between these two. Any algebraic refinement of
    SimBA's output is the GAMBA post-rewriter's job and lives in the
    next preset up.
    """
    return Pipeline([SimbaPass(), AstNormalizationPass()])


def gamba_pipeline() -> Pipeline:
    """
    GAMBA pre/post sandwich around SimBA (the ``PipelineMode.GAMBA``
    preset).

    Pipeline shape: ``[GambaPreprocessingPass, SimbaPass,
    GambaPostRewriterPass, AstNormalizationPass]``.

    Phase-by-phase rationale:

    1. **GambaPreprocessingPass** — applies every ``guarded=False``
       algebraic rule (idempotence, absorption, two-complement,
       inverse-element, the Tier 3 GAMBA additions) in a bottom-up
       fixpoint, trying each rule at every node. Every safe rule is
       no-grow on its matching input, so
       this phase only ever collapses structure. The collapse exposes
       more linear-MBA shapes to SimBA's classifier — patterns that
       would have been rejected as "not linear" in their raw obfuscated
       form become classifiable after idempotence / De Morgan / etc.
       fire. Explicitly EXCLUDES the two guarded rules (ring/factor)
       because they distribute or factor in ways that widen SimBA's
       atom set on the cube.

    2. **SimbaPass** — same role as in :func:`simba_pipeline`, but now
       fed shapes the pre-pass has already collapsed. Empirically lifts
       reduction by +1.6 to +2.1pp on the corpus and exact-match by
       +6.2 to +6.4pp (see ``tmp/gamba_sweep_report.md``).

    3. **GambaPostRewriterPass** — runs the same safe-rule fixpoint as
       the pre-pass PLUS the two ``guarded=True`` rules
       (``ring_normalize``, ``factor_common_subterm``) with their
       net-shrink checks. Collapses SimBA's verbose conjunction-basis
       output back into compact forms. Must run BEFORE binarisation
       because the §5.2 algebraic rules pattern-match on n-ary
       :class:`ExprOp` shapes (multiple duplicate children, coefficient
       collection across many summands); running them after the
       binariser would force every rule to re-flatten internally.

    4. **AstNormalizationPass** — final binariser, same role as in
       :func:`default_pipeline` and :func:`simba_pipeline`.
    """
    return Pipeline(
        [
            GambaPreprocessingPass(),
            SimbaPass(),
            GambaPostRewriterPass(),
            AstNormalizationPass(),
        ]
    )
