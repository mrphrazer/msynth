from __future__ import annotations

from miasm.expression.expression import ExprCompose, ExprId, ExprInt, ExprOp

import msynth.synthesis.mutations as mutations_mod
from msynth.synthesis.grammar import Grammar
from msynth.synthesis.mutations import Mutator
from msynth.synthesis.state import SynthesisState


def test_mutator_enable_casting_for_mixed_sizes() -> None:
    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 16)
    expr = ExprCompose(p0, p1)

    mutator = Mutator(Grammar(24, [p0, p1]))
    mutator.maybe_enable_casting(expr)

    assert mutator.sizes_casting == [0xFF, 0xFFFF, 0xFFFFFF]
    assert mutator.downcast_expression in mutator.mutations


def test_mutator_replace_subexpression_with_leaf_deterministic(monkeypatch) -> None:
    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 8)
    grammar = Grammar(8, [p0, p1])
    t1 = ExprId("t1", 8)
    t2 = ExprId("t2", 8)
    state = SynthesisState(ExprOp("+", t1, t2), {t1: p0, t2: p1})

    monkeypatch.setattr(mutations_mod, "choice", lambda seq: seq[0])
    monkeypatch.setattr(grammar, "get_rand_var_of_size", lambda _size: p0)

    mutated = Mutator(grammar).replace_subexpression_with_leaf(state)

    assert mutated.expr_ast.is_id()
    assert list(mutated.replacements.values()) == [p0]


def test_mutator_downcast_expression_adds_mask(monkeypatch) -> None:
    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 8)
    grammar = Grammar(8, [p0, p1])
    mutator = Mutator(grammar)
    mutator.sizes_casting = [0xFF]
    mutator.mutations.append(mutator.downcast_expression)

    t1 = ExprId("t1", 8)
    t2 = ExprId("t2", 8)
    state = SynthesisState(ExprOp("+", t1, t2), {t1: p0, t2: p1})

    monkeypatch.setattr(mutations_mod, "choice", lambda seq: seq[0])

    mutated = mutator.downcast_expression(state)

    assert mutated.expr_ast.is_op() and mutated.expr_ast.op == "&"
    assert isinstance(mutated.expr_ast.args[1], ExprInt)


def test_mutator_replace_subexpression_with_expression_size_mismatch(monkeypatch) -> None:
    p0 = ExprId("p0", 8)
    p1 = ExprId("p1", 8)
    grammar = Grammar(8, [p0, p1])
    mutator = Mutator(grammar)

    state = SynthesisState(ExprOp("+", ExprId("t1", 8), ExprId("t2", 8)), {})
    original = state.expr_ast

    monkeypatch.setattr(mutations_mod, "choice", lambda seq: seq[0])
    monkeypatch.setattr(grammar, "gen_expr_for_state", lambda: (ExprId("x", 16), {}))

    mutated = mutator.replace_subexpression_with_expression(state)

    assert mutated.expr_ast == original
