from __future__ import annotations

from dataclasses import dataclass

from miasm.expression.expression import Expr, ExprId, ExprInt, ExprOp

from msynth.simplification.pipeline import (
    AstNormalizationPass,
    Pipeline,
    default_pipeline,
)
from msynth.simplification.simba import SimbaPass
from msynth.simplification.simplifier import Simplifier


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


def test_default_pipeline_order() -> None:
    # Pipeline shape: [SimbaPass, *extras, AstNormalizationPass].
    # SimBA runs on the raw input (arity-tolerant; no binarisation
    # needed upstream). AstNorm at the *tail* binarises whatever SimBA
    # emits — SimBA's reconstruction helpers build variadic ops, and
    # the main simplifier loop's get_subexpressions walk needs the
    # binary form to expose intermediate sub-pair nodes for oracle
    # lookup.
    pipeline = default_pipeline()

    assert isinstance(pipeline.passes[0], SimbaPass)
    assert isinstance(pipeline.passes[-1], AstNormalizationPass)


def test_simplifier_applies_optional_pipeline() -> None:
    @dataclass(frozen=True)
    class ConstantPass:
        name: str = "constant"

        def run(self, expr: Expr) -> Expr:
            _ = expr
            return ExprInt(7, 8)

    simplifier = Simplifier(pipeline=Pipeline([ConstantPass()]))

    assert simplifier.simplify(ExprId("x", 8)) == ExprInt(7, 8)


def test_simplifier_uses_default_pipeline_order() -> None:
    simplifier = Simplifier()

    assert isinstance(simplifier.pipeline.passes[0], SimbaPass)
    assert isinstance(simplifier.pipeline.passes[-1], AstNormalizationPass)


def test_default_pipeline_runs_simba_after_ast_normalization() -> None:
    x = ExprId("x", 8)
    y = ExprId("y", 8)

    rewritten = default_pipeline().run(
        ExprOp("+", ExprOp("&", x, y), ExprOp("|", x, y))
    )

    assert rewritten == ExprOp("+", x, y)
