from __future__ import annotations

from miasm.expression.expression import ExprId, ExprInt

import msynth.synthesis.oracle as synth_oracle
from msynth.synthesis.grammar import Grammar
from msynth.synthesis.mutations import Mutator
from msynth.synthesis.oracle import SynthesisOracle
from msynth.synthesis.state import SynthesisState
from msynth.synthesis.synthesizer import Synthesizer


def test_synthesize_from_expression_reapplies_unification(monkeypatch) -> None:
    seq = iter([1, 2, 3, 4])

    def fake_get_rand_input() -> int:
        return next(seq)

    monkeypatch.setattr(synth_oracle, "get_rand_input", fake_get_rand_input)

    expr = ExprId("x", 8)
    synth = Synthesizer()

    def fake_ils(_mutator, _oracle):
        return SynthesisState(ExprId("p0", 8)), 0.0

    monkeypatch.setattr(synth, "iterated_local_search", fake_ils)

    result_expr, score = synth.synthesize_from_expression(expr, num_samples=1)

    assert score == 0.0
    assert result_expr == expr


def test_synthesize_from_expression_zero_extends_on_size_mismatch(monkeypatch) -> None:
    seq = iter([1, 2, 3, 4])

    def fake_get_rand_input() -> int:
        return next(seq)

    monkeypatch.setattr(synth_oracle, "get_rand_input", fake_get_rand_input)

    expr = ExprId("p0", 8)
    synth = Synthesizer()

    def fake_ils(_mutator, _oracle):
        return SynthesisState(ExprId("p0", 4)), 0.0

    monkeypatch.setattr(synth, "iterated_local_search", fake_ils)

    result_expr, score = synth.synthesize_from_expression(expr, num_samples=1)

    assert score == 0.0
    assert result_expr.size == 8


def test_iterated_local_search_returns_zero_score(monkeypatch) -> None:
    p0 = ExprId("p0", 8)
    oracle = SynthesisOracle({(ExprInt(1, 8),): ExprInt(1, 8)})

    grammar = Grammar(8, [p0])
    mutator = Mutator(grammar)

    monkeypatch.setattr(SynthesisState, "get_score", lambda _self, _o, _v: 0.0)

    synth = Synthesizer()
    _state, score = synth.iterated_local_search(mutator, oracle, timeout=1)

    assert score == 0.0
