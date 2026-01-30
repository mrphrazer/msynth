from __future__ import annotations

from miasm.expression.expression import ExprId

import msynth.synthesis.grammar as grammar_mod
from msynth.synthesis.grammar import Grammar


def test_gen_expr_for_state_coerces_sizes_upcast(monkeypatch) -> None:
    g = Grammar(16, [ExprId("p0", 8), ExprId("p1", 16)])

    seq = iter([ExprId("v1", 8), ExprId("v2", 16)])
    monkeypatch.setattr(g, "gen_fresh_var", lambda: next(seq))

    coin_seq = iter([0, 0])  # upcast, then choose '+' op
    monkeypatch.setattr(grammar_mod, "getrandbits", lambda _n: next(coin_seq))

    expr, _repl = g.gen_expr_for_state()

    assert expr.size == 16


def test_gen_expr_for_state_coerces_sizes_downcast(monkeypatch) -> None:
    g = Grammar(16, [ExprId("p0", 8), ExprId("p1", 16)])

    seq = iter([ExprId("v1", 8), ExprId("v2", 16)])
    monkeypatch.setattr(g, "gen_fresh_var", lambda: next(seq))

    coin_seq = iter([1, 0])  # downcast, then choose '+' op
    monkeypatch.setattr(grammar_mod, "getrandbits", lambda _n: next(coin_seq))

    expr, _repl = g.gen_expr_for_state()

    assert expr.size == 8
