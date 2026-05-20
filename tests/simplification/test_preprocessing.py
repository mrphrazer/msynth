from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from miasm.expression.expression import Expr, ExprId, ExprInt, ExprOp

from msynth.simplification.oracle import SimplificationOracle
from msynth.simplification.preprocessing import (
    AstNormalizationPass,
    Preprocessor,
    default_preprocessor,
)
from msynth.simplification.simplifier import Simplifier


def _write_min_oracle(tmp_path: Path) -> Path:
    oracle = SimplificationOracle.__new__(SimplificationOracle)
    oracle.num_variables = 1
    oracle.num_samples = 3
    oracle.inputs = [[0], [1], [2]]
    oracle.oracle_map = {}

    path = tmp_path / "oracle.pkl"
    with open(path, "wb") as handle:
        pickle.dump(oracle, handle)
    return path


def test_ast_normalization_pass_splits_variadic_expression() -> None:
    x = ExprId("x", 8)
    y = ExprId("y", 8)
    z = ExprId("z", 8)

    rewritten = AstNormalizationPass().run(ExprOp("+", x, y, z))

    assert rewritten == ExprOp("+", ExprOp("+", x, y), z)


def test_preprocessor_runs_passes_in_order() -> None:
    x = ExprId("x", 8)
    seen: list[str] = []

    @dataclass(frozen=True)
    class RecordingPass:
        name: str

        def run(self, expr: Expr) -> Expr:
            seen.append(self.name)
            return expr

    preprocessor = Preprocessor([RecordingPass("first"), RecordingPass("second")])

    assert preprocessor.run(x) == x
    assert seen == ["first", "second"]


def test_default_preprocessor_starts_with_ast_normalization() -> None:
    preprocessor = default_preprocessor()

    assert isinstance(preprocessor.passes[0], AstNormalizationPass)


def test_simplifier_applies_optional_preprocessor(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class ConstantPass:
        name: str = "constant"

        def run(self, expr: Expr) -> Expr:
            _ = expr
            return ExprInt(7, 8)

    simplifier = Simplifier(
        _write_min_oracle(tmp_path), preprocessor=Preprocessor([ConstantPass()])
    )

    assert simplifier.simplify(ExprId("x", 8)) == ExprInt(7, 8)


def test_simplifier_uses_ast_preprocessor_by_default(tmp_path: Path) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path))

    assert isinstance(simplifier.preprocessor.passes[0], AstNormalizationPass)
