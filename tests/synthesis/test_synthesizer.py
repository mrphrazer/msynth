from __future__ import annotations

from miasm.expression.expression import ExprId, ExprInt, ExprOp

from msynth.synthesis.synthesizer import Synthesizer


def test_simplify_returns_synthesized_on_zero_score(monkeypatch) -> None:
    expr = ExprOp("+", ExprId("p0", 8), ExprInt(1, 8))
    synth = Synthesizer()
    synthesized = ExprInt(7, 8)

    monkeypatch.setattr(synth, "synthesize_from_expression", lambda _expr, _n: (synthesized, 0.0))

    assert synth.simplify(expr, num_samples=2) == synthesized


def test_simplify_returns_original_on_nonzero_score(monkeypatch) -> None:
    expr = ExprOp("+", ExprId("p0", 8), ExprInt(1, 8))
    synth = Synthesizer()
    synthesized = ExprInt(7, 8)

    monkeypatch.setattr(synth, "synthesize_from_expression", lambda _expr, _n: (synthesized, 1.0))

    assert synth.simplify(expr, num_samples=2) == expr


def test_synthesize_from_expression_parallel_returns_first_success(monkeypatch) -> None:
    expr = ExprOp("+", ExprId("p0", 8), ExprInt(1, 8))
    synth = Synthesizer()

    def fake_synthesize(_expr, _n):
        return ExprInt(7, 8), 0.0

    class FakeParallelizer:
        def __init__(self, _tasks):
            self.task_group_results = {}

        def execute(self):
            self.task_group_results[str(expr)] = (ExprInt(7, 8), 0.0)

    monkeypatch.setattr(synth, "synthesize_from_expression", fake_synthesize)
    monkeypatch.setattr("msynth.synthesis.synthesizer.Parallelizer", FakeParallelizer)

    result_expr, result_score = synth.synthesize_from_expression_parallel(expr, num_samples=2)

    assert result_expr == ExprInt(7, 8)
    assert result_score == 0.0
