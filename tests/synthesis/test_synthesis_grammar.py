from __future__ import annotations

from miasm.expression.expression import ExprId

import msynth.synthesis.grammar as grammar_mod
from msynth.synthesis.grammar import Grammar


def test_grammar_gen_terminal_for_state_deterministic(monkeypatch) -> None:
    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 8)
    grammar = Grammar(8, [p0, p1])

    monkeypatch.setattr(grammar_mod, "choice", lambda seq: seq[0])

    expr, repl = grammar.gen_terminal_for_state()

    assert expr.size == 8
    assert expr in repl
    assert repl[expr] == p0


def test_grammar_gen_expr_for_state_shapes(monkeypatch) -> None:
    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 8)
    grammar = Grammar(8, [p0, p1])

    monkeypatch.setattr(grammar_mod, "choice", lambda seq: seq[0])
    monkeypatch.setattr(grammar_mod, "getrandbits", lambda _n: 0)

    expr, repl = grammar.gen_expr_for_state()

    assert expr.size == 8
    assert len(repl) in (1, 2)
    for fresh_var, mapped in repl.items():
        assert fresh_var.size == mapped.size


def test_grammar_gen_expr_for_state_mixed_sizes(monkeypatch) -> None:
    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 16)
    grammar = Grammar(16, [p0, p1])

    monkeypatch.setattr(grammar_mod, "choice", lambda seq: seq[0])
    monkeypatch.setattr(grammar_mod, "getrandbits", lambda _n: 0)

    expr, repl = grammar.gen_expr_for_state()

    assert expr.size in (8, 16)
    for fresh_var, mapped in repl.items():
        assert fresh_var.size == mapped.size
