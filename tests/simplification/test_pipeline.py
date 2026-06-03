from __future__ import annotations

from dataclasses import dataclass

import pytest
from miasm.expression.expression import Expr, ExprId, ExprInt, ExprOp

from msynth.simplification.pipeline import (
    AstNormalizationPass,
    Pipeline,
    PipelineMode,
    default_pipeline,
    gamba_pipeline,
    simba_pipeline,
)
from msynth.simplification.simplifier import Simplifier


# ---------------------------------------------------------------------------
# Pass + Pipeline mechanics
# ---------------------------------------------------------------------------


def test_ast_normalization_pass_splits_variadic_expression() -> None:
    x = ExprId("x", 8)
    y = ExprId("y", 8)
    z = ExprId("z", 8)

    rewritten = AstNormalizationPass().run(ExprOp("+", x, y, z))

    assert rewritten == ExprOp("+", ExprOp("+", x, y), z)


def test_pipeline_runs_passes_in_order() -> None:
    x = ExprId("x", 8)
    seen: list[str] = []

    @dataclass(frozen=True)
    class RecordingPass:
        name: str

        def run(self, expr: Expr) -> Expr:
            seen.append(self.name)
            return expr

    pipeline = Pipeline([RecordingPass("first"), RecordingPass("second")])

    assert pipeline.run(x) == x
    assert seen == ["first", "second"]


# ---------------------------------------------------------------------------
# PipelineMode enum + factory smoke tests
# ---------------------------------------------------------------------------


def test_pipeline_mode_enum_values() -> None:
    # The enum inherits from str so callers can pass string literals.
    assert PipelineMode.AST == "ast"
    assert PipelineMode.SIMBA == "simba"
    assert PipelineMode.GAMBA == "gamba"
    assert {m.value for m in PipelineMode} == {"ast", "simba", "gamba"}


def test_default_pipeline_is_ast_norm_only() -> None:
    pipeline = default_pipeline()
    assert [type(p).__name__ for p in pipeline.passes] == ["AstNormalizationPass"]


def test_simba_pipeline_factory_shape() -> None:
    pipeline = simba_pipeline()
    assert [type(p).__name__ for p in pipeline.passes] == [
        "SimbaPass",
        "AstNormalizationPass",
    ]


def test_gamba_pipeline_factory_shape() -> None:
    pipeline = gamba_pipeline()
    # ExpandPass and FactorizeSumsPass were added between the post-rewriter
    # and AST normalisation as part of the GAMBA-general integration; see
    # the docstring on :func:`pipeline.gamba_pipeline` for the rationale.
    assert [type(p).__name__ for p in pipeline.passes] == [
        "GambaPreprocessingPass",
        "SimbaPass",
        "GambaPostRewriterPass",
        "ExpandPass",
        "FactorizeSumsPass",
        "AstNormalizationPass",
    ]


def test_simba_pipeline_simplifies_linear_mba() -> None:
    # `(a & b) + (a | b) == a + b` — SimBA recognises this as a linear MBA
    # and reconstructs it. AstNorm at the tail leaves the binary `+` alone.
    x = ExprId("x", 8)
    y = ExprId("y", 8)
    out = simba_pipeline().run(ExprOp("+", ExprOp("&", x, y), ExprOp("|", x, y)))
    assert out == ExprOp("+", x, y)


# ---------------------------------------------------------------------------
# Simplifier wiring
# ---------------------------------------------------------------------------


def test_simplifier_default_pipeline_is_ast_norm_only() -> None:
    sim = Simplifier()
    assert [type(p).__name__ for p in sim.pipeline.passes] == ["AstNormalizationPass"]


@pytest.mark.parametrize(
    "mode, expected",
    [
        (PipelineMode.AST, ["AstNormalizationPass"]),
        (PipelineMode.SIMBA, ["SimbaPass", "AstNormalizationPass"]),
        (
            PipelineMode.GAMBA,
            [
                "GambaPreprocessingPass",
                "SimbaPass",
                "GambaPostRewriterPass",
                # ExpandPass + FactorizeSumsPass added for GAMBA-general
                # nonlinear MBA work; see gamba.py for implementations.
                "ExpandPass",
                "FactorizeSumsPass",
                "AstNormalizationPass",
            ],
        ),
    ],
)
def test_simplifier_pipeline_mode_selects_factory(
    mode: PipelineMode, expected: list[str]
) -> None:
    sim = Simplifier(pipeline_mode=mode)
    assert [type(p).__name__ for p in sim.pipeline.passes] == expected


def test_simplifier_pipeline_override_replaces_default() -> None:
    # Explicit ``pipeline=`` is a real override now (previously its
    # contents were spliced into ``default_pipeline`` as
    # ``extra_passes``, producing duplicated SimBA/AstNorm boundaries).
    @dataclass(frozen=True)
    class ConstantPass:
        name: str = "constant"

        def run(self, expr: Expr) -> Expr:
            _ = expr
            return ExprInt(7, 8)

    sim = Simplifier(pipeline=Pipeline([ConstantPass()]))
    assert [type(p).__name__ for p in sim.pipeline.passes] == ["ConstantPass"]
    # End-to-end: ConstantPass alone produces 7 for any input.
    assert sim.simplify(ExprId("x", 8)) == ExprInt(7, 8)


def test_simplifier_pipeline_override_wins_over_mode() -> None:
    @dataclass(frozen=True)
    class IdentityPass:
        name: str = "identity"

        def run(self, expr: Expr) -> Expr:
            return expr

    sim = Simplifier(
        pipeline_mode=PipelineMode.GAMBA,
        pipeline=Pipeline([IdentityPass()]),
    )
    assert [type(p).__name__ for p in sim.pipeline.passes] == ["IdentityPass"]
    # Subtree-SimBA still tracks the declared mode, not the override.
    assert sim._subtree_simba_pass is not None
