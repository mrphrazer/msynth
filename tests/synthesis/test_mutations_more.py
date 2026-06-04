from __future__ import annotations

from miasm.expression.expression import ExprId, ExprInt, ExprOp

import msynth.synthesis.mutations as mutations_mod
from msynth.synthesis.grammar import Grammar
from msynth.synthesis.mutations import Mutator
from msynth.synthesis.state import SynthesisState


def test_replace_subexpression_with_expression_noop_on_size_mismatch(
    monkeypatch,
) -> None:
    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 8)
    expr = ExprOp("+", p0, p1)

    grammar = Grammar(8, [p0, p1])
    mutator = Mutator(grammar)
    state = SynthesisState(expr, {p0: p0, p1: p1})

    monkeypatch.setattr(
        grammar,
        "gen_expr_for_state",
        lambda: (ExprId("q0", 4), {ExprId("q0", 4): p0}),
    )

    before = state.expr_ast
    before_repl = state.replacements.copy()

    after = mutator.replace_subexpression_with_expression(state)

    assert after.expr_ast == before
    assert after.replacements == before_repl


def test_downcast_expression_applies_mask(monkeypatch) -> None:
    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 8)
    expr = ExprOp("+", p0, p1)

    grammar = Grammar(8, [p0, p1])
    mutator = Mutator(grammar)
    # a mask strictly narrower than the 8-bit subexpression -> genuine downcast
    mutator.sizes_casting = [0x0F]
    mutator.mutations.append(mutator.downcast_expression)

    state = SynthesisState(expr, {p0: p0, p1: p1})

    # deterministic choices: choose the root expr and the only mask
    monkeypatch.setattr(mutations_mod, "choice", lambda seq: seq[0])
    monkeypatch.setattr(mutations_mod, "get_subexpressions", lambda _e: [expr])

    mutated = mutator.downcast_expression(state)

    assert isinstance(mutated.expr_ast, ExprOp)
    assert mutated.expr_ast.op == "&"
    # the narrower mask is applied, genuinely reducing the value range
    assert mutated.expr_ast.args[1] == ExprInt(0x0F, 8)


def test_downcast_expression_skips_when_no_narrower_mask(monkeypatch) -> None:
    # A mask equal to the subexpression's full width is a no-op AND, so the
    # downcast must leave the expression unchanged rather than apply it.
    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 8)
    expr = ExprOp("+", p0, p1)

    grammar = Grammar(8, [p0, p1])
    mutator = Mutator(grammar)
    mutator.sizes_casting = [0xFF]  # == full 8-bit mask, not narrower
    mutator.mutations.append(mutator.downcast_expression)

    state = SynthesisState(expr, {p0: p0, p1: p1})

    monkeypatch.setattr(mutations_mod, "choice", lambda seq: seq[0])
    monkeypatch.setattr(mutations_mod, "get_subexpressions", lambda _e: [expr])

    mutated = mutator.downcast_expression(state)

    # unchanged: no spurious "& 0xFF" wrapper
    assert mutated.expr_ast == expr
