"""
Regression tests for the wiring of GAMBA pre/post rewriters around every
SimBA invocation and for the :class:`PipelineMode` selector.

Three pipelines / modes are exercised:

1. ``default_pipeline()`` (``PipelineMode.AST``) — production default.
   Pure AST binarisation, no simplification work. Used when the caller
   either has no preference, runs only the oracle lookup, or stays in
   CEGIS-only mode.

2. ``simba_pipeline()`` (``PipelineMode.SIMBA``) — SimBA reconstruction
   plus binarisation. Subtree-SimBA is enabled but does NOT wrap with
   GAMBA pre/post — that's a GAMBA-only behaviour.

3. ``gamba_pipeline()`` (``PipelineMode.GAMBA``) — GAMBA pre + SimBA +
   GAMBA post + AST binarisation. Subtree-SimBA is enabled and applies
   the same GAMBA pre/post wrap that the global pipeline applies, so
   the subtree-level call mirrors the global one.
"""

from __future__ import annotations

import pytest
from miasm.expression.expression import Expr, ExprId, ExprInt, ExprOp

from msynth.simplification.gamba import (
    GAMBA_POST_REWRITER,
    GAMBA_PREPROCESSOR,
)
from msynth.simplification.pipeline import (
    AstNormalizationPass,
    Pipeline,
    PipelineMode,
    default_pipeline,
    gamba_pipeline,
    simba_pipeline,
)
from msynth.simplification.simplifier import Simplifier

_SIZE = 32
_MASK = (1 << _SIZE) - 1


def _nodes(expr: Expr) -> int:
    return len(expr.graph().nodes())


def _not_(x: Expr) -> Expr:
    return ExprOp("^", x, ExprInt(_MASK, _SIZE))


# ---------------------------------------------------------------------------
# Pipeline shape
# ---------------------------------------------------------------------------


def test_default_pipeline_is_ast_normalization_only() -> None:
    p = default_pipeline()
    classes = [type(s).__name__ for s in p.passes]
    assert classes == ["AstNormalizationPass"]


def test_simba_pipeline_runs_simba_then_ast() -> None:
    p = simba_pipeline()
    classes = [type(s).__name__ for s in p.passes]
    assert classes == ["SimbaPass", "AstNormalizationPass"]


def test_gamba_pipeline_has_pre_then_simba_then_post_then_ast() -> None:
    p = gamba_pipeline()
    classes = [type(s).__name__ for s in p.passes]
    assert classes == [
        "GambaPreprocessingPass",
        "SimbaPass",
        "GambaPostRewriterPass",
        "AstNormalizationPass",
    ]


def test_simplifier_pipeline_override_replaces_default() -> None:
    # ``pipeline=`` argument is a real override now (previously it was
    # spliced into default_pipeline as ``extra_passes``). The user's
    # passes land in the resulting pipeline verbatim, no wrapping.
    override = Pipeline([AstNormalizationPass()])
    sim = Simplifier(pipeline=override)
    assert sim.pipeline is override


def test_simplifier_pipeline_override_wins_over_mode() -> None:
    # Explicit ``pipeline=`` beats ``pipeline_mode=``. Useful escape
    # hatch when a caller needs a custom composition but still wants
    # subtree-SimBA enabled (which tracks the mode, not the pipeline).
    override = Pipeline([AstNormalizationPass()])
    sim = Simplifier(pipeline_mode=PipelineMode.GAMBA, pipeline=override)
    assert sim.pipeline is override
    # Subtree-SimBA still tracks the mode the caller declared.
    assert sim._subtree_simba_pass is not None


# ---------------------------------------------------------------------------
# Pre-pass behaviour around SimBA (GAMBA-mode pipeline)
# ---------------------------------------------------------------------------


def test_gamba_pre_collapses_idempotence_before_simba_sees_it() -> None:
    # `a & a` should collapse to `a` before SimBA classifies. Then SimBA
    # has nothing to do on a bare atom and the AST norm pass is a no-op.
    a = ExprId("a", _SIZE)
    p = gamba_pipeline()
    expr = ExprOp("&", a, a)
    out = p.run(expr)
    assert out == a


def test_gamba_pre_collapses_self_complement_to_zero() -> None:
    a = ExprId("a", _SIZE)
    p = gamba_pipeline()
    expr = ExprOp("&", a, _not_(a))
    out = p.run(expr)
    assert out == ExprInt(0, _SIZE)


# ---------------------------------------------------------------------------
# Post-pass behaviour after SimBA
# ---------------------------------------------------------------------------


def test_gamba_post_cleans_simba_conjunction_basis_duplicates() -> None:
    # Construct a sum the way SimBA's `_sum` helper would: variadic `+`
    # with a duplicated atomic operand and a paired complement. After
    # GAMBA-post: `a + a + ~a → a + (-1)` since `a + ~a == -1`.
    a = ExprId("a", _SIZE)
    expr = ExprOp("+", a, a, _not_(a))
    out = GAMBA_POST_REWRITER.normalize(expr)
    # Soundness check: same value as input for any `a`.
    from test_rewrites import _z3_equivalent

    assert _z3_equivalent(expr, out)
    assert _nodes(out) <= _nodes(expr)


def test_gamba_post_runs_before_ast_normalisation() -> None:
    # Pipeline ordering: the post-rewriter must see the variadic shape SimBA
    # emits, not the binarised shape, so GambaPostRewriterPass MUST precede
    # AstNormalizationPass. Assert that order structurally -- an equivalence
    # check alone cannot distinguish it (both orders are sound and, for this
    # input, both even produce the same result).
    pass_order = [type(p).__name__ for p in gamba_pipeline().passes]
    assert pass_order.index("GambaPostRewriterPass") < pass_order.index(
        "AstNormalizationPass"
    )

    # and the assembled pipeline is still sound on a representative input
    a, b = ExprId("a", _SIZE), ExprId("b", _SIZE)
    inner = ExprOp("+", ExprOp("&", a, b), ExprOp("|", a, b))
    out = gamba_pipeline().run(inner)
    from test_rewrites import _z3_equivalent

    assert _z3_equivalent(inner, out)


# ---------------------------------------------------------------------------
# Subtree-SimBA wiring (mode-aware)
# ---------------------------------------------------------------------------


def test_subtree_simba_wraps_with_gamba_under_gamba_mode() -> None:
    # Under GAMBA mode, _try_subtree_simba wraps its SimBA call with
    # GAMBA pre on input and GAMBA post on output (before binarisation).
    # We exercise a shape where pre would shrink the input meaningfully
    # — `(a & a)` and `(b & b)` collapse under pre via idempotence; the
    # remaining `(a & b) + (a | b)` is then a SimBA-friendly linear MBA.
    sim = Simplifier(pipeline_mode=PipelineMode.GAMBA)
    a, b = ExprId("a", _SIZE), ExprId("b", _SIZE)
    expr = ExprOp(
        "+",
        ExprOp("&", a, b),
        ExprOp("|", a, b),
        ExprOp("&", a, a),
        ExprOp("&", b, b),
    )
    out = sim.simplify(expr)
    from test_rewrites import _z3_equivalent

    assert _z3_equivalent(expr, out)


def test_subtree_simba_pre_collapses_duplicate_idempotence_under_gamba() -> None:
    sim = Simplifier(pipeline_mode=PipelineMode.GAMBA)
    a = ExprId("a", _SIZE)
    # Pure idempotence: 4-way ``a & a & a & a`` collapses to ``a``.
    expr = ExprOp("&", a, a, a, a)
    out = sim.simplify(expr)
    assert out == a


def test_subtree_simba_fires_under_simba_mode_without_gamba_wrap() -> None:
    # Under SIMBA mode, subtree-SimBA fires but does NOT wrap with
    # GAMBA pre/post. The expression's outer op (>>) keeps the global
    # SimbaPass from handling it, so only the inner subtree gets
    # rewritten — by SimBA without algebraic refinement around it.
    sim = Simplifier(pipeline_mode=PipelineMode.SIMBA)
    x = ExprId("x", _SIZE)
    y = ExprId("y", _SIZE)
    shift = ExprId("shift", _SIZE)
    inner = ExprOp("+", ExprOp("&", x, y), ExprOp("|", x, y))
    expr = ExprOp(">>", inner, shift)
    out = sim.simplify(expr)
    from test_rewrites import _z3_equivalent

    assert _z3_equivalent(expr, out)
    # Subtree-SimBA found a strictly-smaller candidate (x + y vs the
    # variadic (a&b)+(a|b) shape).
    assert _nodes(out) < _nodes(expr)


def test_subtree_simba_disabled_under_ast_mode() -> None:
    # AST mode: subtree-SimBA is off entirely. The inner SimBA-friendly
    # subtree therefore stays untouched.
    sim = Simplifier(pipeline_mode=PipelineMode.AST)
    assert sim._subtree_simba_pass is None
    x = ExprId("x", _SIZE)
    y = ExprId("y", _SIZE)
    shift = ExprId("shift", _SIZE)
    inner = ExprOp("+", ExprOp("&", x, y), ExprOp("|", x, y))
    expr = ExprOp(">>", inner, shift)
    out = sim.simplify(expr)
    assert out == expr


# ---------------------------------------------------------------------------
# Default-pipeline / mode regression
# ---------------------------------------------------------------------------


def test_default_pipeline_shape_via_simplifier() -> None:
    # ``Simplifier()`` defaults to ``PipelineMode.AST`` — pure binariser.
    sim = Simplifier()
    assert [type(s).__name__ for s in sim.pipeline.passes] == ["AstNormalizationPass"]


def test_simba_mode_pipeline_shape_via_simplifier() -> None:
    sim = Simplifier(pipeline_mode=PipelineMode.SIMBA)
    assert [type(s).__name__ for s in sim.pipeline.passes] == [
        "SimbaPass",
        "AstNormalizationPass",
    ]


def test_gamba_mode_pipeline_shape_via_simplifier() -> None:
    sim = Simplifier(pipeline_mode=PipelineMode.GAMBA)
    assert [type(s).__name__ for s in sim.pipeline.passes] == [
        "GambaPreprocessingPass",
        "SimbaPass",
        "GambaPostRewriterPass",
        "AstNormalizationPass",
    ]


# ---------------------------------------------------------------------------
# Engine convergence + soundness
# ---------------------------------------------------------------------------


def test_gamba_pipeline_converges_on_pathological_shapes() -> None:
    # Pre + post both fire on related patterns; this test catches
    # oscillation regressions. Build an expression that pre simplifies
    # and post also simplifies; both should converge in one pipeline pass
    # (the max_iters=50 cap is the safety net).
    a, b, c = ExprId("a", _SIZE), ExprId("b", _SIZE), ExprId("c", _SIZE)
    expr = ExprOp(
        "+",
        ExprOp("|", a, ExprOp("&", a, b)),  # absorption -> a
        ExprOp("&", c, ExprOp("|", c, a)),  # absorption -> c
        ExprOp("&", b, _not_(b)),  # redundancy -> 0
    )
    out = GAMBA_PREPROCESSOR.normalize(expr)
    # Expect `a + c` (with 0 dropped).
    from test_rewrites import _z3_equivalent

    assert _z3_equivalent(expr, out)
    assert _nodes(out) <= 4  # `+`, `a`, `c`, plus maybe one wrapper


@pytest.mark.parametrize("size", [8, 16, 32, 64])
def test_gamba_pipeline_idempotent_on_simple_inputs(size: int) -> None:
    a = ExprId("a", size)
    p = gamba_pipeline()
    once = p.run(a)
    twice = p.run(once)
    assert once == twice
    assert once == a


# ---------------------------------------------------------------------------
# Soundness on a small zoo of shapes
# ---------------------------------------------------------------------------


_ZOO: list[Expr] = []


def _build_zoo() -> list[Expr]:
    if _ZOO:
        return _ZOO
    a, b = ExprId("a", _SIZE), ExprId("b", _SIZE)
    _ZOO.extend(
        [
            ExprOp("+", ExprOp("&", a, b), ExprOp("|", a, b)),
            ExprOp("+", a, _not_(a)),
            ExprOp("&", a, ExprOp("|", a, b)),
            ExprOp("|", a, ExprOp("&", a, b)),
            ExprOp("^", a, _not_(a)),
            ExprOp(
                "+",
                ExprOp("*", ExprInt(7, _SIZE), a),
                ExprOp("*", ExprInt(3, _SIZE), a),
            ),
            ExprOp("+", ExprOp("&", _not_(a), b), ExprOp("&", a, b)),
            ExprOp("+", ExprOp("|", _not_(a), b), ExprOp("|", a, b)),
            ExprOp("|", ExprOp("&", a, b), ExprOp("^", a, b)),
            ExprOp(
                "+",
                ExprOp("*", ExprInt(2, _SIZE), a),
                ExprOp("*", ExprInt(2, _SIZE), b),
            ),
        ]
    )
    return _ZOO


@pytest.mark.parametrize("expr", _build_zoo(), ids=lambda e: str(e)[:50])
def test_gamba_pipeline_is_sound_on_zoo(expr: Expr) -> None:
    p = gamba_pipeline()
    out = p.run(expr)
    from test_rewrites import _z3_equivalent

    assert _z3_equivalent(expr, out), f"unsound rewrite: {expr} -> {out}"


# ---------------------------------------------------------------------------
# §5.1 substitution loop — end-to-end through the real Simplifier + SimbaPass
# ---------------------------------------------------------------------------
#
# The unit tests in test_gamba.py drive ``gamba_substitution`` with a synthetic
# ``simba_fn`` stub. These exercise the escalation through the *real* pipeline:
# ``Simplifier(pipeline_mode=GAMBA, gamba_substitution_max_k=k)`` builds the
# real GAMBA-wrapped SimBA ``_simba_fn`` and routes subtree-SimBA through the
# §5.1 abstraction loop.


def test_gamba_substitution_escalation_end_to_end_unlocks_nonlinear_atom() -> None:
    # ((x*y) ^ z) + 2*((x*y) & z) is the MBA identity for (x*y) + z. SimBA
    # REJECTS the product-of-two-variables x*y (not a linear-MBA shape), so at
    # max_k=0 the whole expression is a no-op and stays obfuscated. At max_k>=1
    # the §5.1 loop abstracts x*y to a fresh atom, SimBA reduces the linearised
    # form to g0 + z, and reverse-substitution restores (x*y) + z.
    from test_rewrites import _z3_equivalent

    x = ExprId("x", 8)
    y = ExprId("y", 8)
    z = ExprId("z", 8)
    p = ExprOp("*", x, y)
    expr = ExprOp("+", ExprOp("^", p, z), ExprOp("*", ExprInt(2, 8), ExprOp("&", p, z)))

    out0 = Simplifier(
        pipeline_mode=PipelineMode.GAMBA, gamba_substitution_max_k=0
    ).simplify(expr)
    out2 = Simplifier(
        pipeline_mode=PipelineMode.GAMBA, gamba_substitution_max_k=2
    ).simplify(expr)

    # Both must stay sound.
    assert _z3_equivalent(expr, out0)
    assert _z3_equivalent(expr, out2)
    # Escalation strictly helps: max_k=0 cannot reduce, max_k>=1 does.
    assert out2 != out0
    assert _nodes(out2) < _nodes(out0)
    # The recovered form is the de-obfuscated (x*y) + z.
    assert _z3_equivalent(out2, ExprOp("+", p, z))


def test_gamba_substitution_escalation_end_to_end_is_sound_when_no_reduction() -> None:
    # An expression with nonlinear leaves that the escalation cannot linearise
    # must still come back semantically unchanged (the loop drives the real
    # _simba_fn at every k and commits nothing).
    from test_rewrites import _z3_equivalent

    x = ExprId("x", 8)
    y = ExprId("y", 8)
    expr = ExprOp("+", ExprOp("*", x, y), x)  # a*x*y-style residue, no MBA collapse

    out = Simplifier(
        pipeline_mode=PipelineMode.GAMBA, gamba_substitution_max_k=3
    ).simplify(expr)
    assert _z3_equivalent(expr, out)
