"""Coverage for SynthesisState.clone().

clone() must return an INDEPENDENT copy: the (immutable, interned) expr_ast is
shared by reference while the mutable `replacements` dict and the compiled-cache
are private to each state, so mutating one state never affects the other.

These tests pin that contract across many state shapes, and additionally guard that
clone() does not rebuild expr_ast via .copy() (a no-op for immutable interned Exprs,
since copy() just returns the same object).
"""

from __future__ import annotations

import pytest

from miasm.expression.expression import (
    ExprId,
    ExprInt,
    ExprOp,
    ExprSlice,
    ExprCond,
    ExprCompose,
    ExprMem,
)

from msynth.synthesis.state import SynthesisState
from msynth.utils.expr_utils import get_unique_variables

OP = ExprOp


def _state_cases():
    """(name, expr_ast, replacements) over a variety of AST shapes/sizes.
    expr_ast uses unique leaves t0.. ; replacements map them into the domain."""
    cases = []
    for sz in (8, 16, 32):
        t = [ExprId(f"t{i}_{sz}", sz) for i in range(6)]
        x, y, z = (ExprId(n, sz) for n in ("x", "y", "z"))
        cases += [
            (f"binary_add_{sz}", OP("+", t[0], t[1]), {t[0]: x, t[1]: y}),
            (
                f"repeated_leaf_{sz}",
                OP("*", OP("+", t[0], t[1]), t[2]),
                {t[0]: x, t[1]: y, t[2]: x},
            ),
            (f"single_leaf_{sz}", t[0], {t[0]: x}),
            (
                f"const_replacement_{sz}",
                OP("^", t[0], t[1]),
                {t[0]: x, t[1]: ExprInt(5, sz)},
            ),
            (
                f"deep_tree_{sz}",
                OP("+", OP("*", t[0], t[1]), OP("^", t[2], t[3])),
                {t[0]: x, t[1]: y, t[2]: z, t[3]: x},
            ),
            (
                f"nary_{sz}",
                OP("&", t[0], t[1], t[2], t[3]),
                {t[0]: x, t[1]: y, t[2]: z, t[3]: y},
            ),
            (
                f"slice_{sz}",
                ExprSlice(OP("+", t[0], t[1]), 0, sz // 2),
                {t[0]: x, t[1]: y},
            ),
            (
                f"cond_{sz}",
                ExprCond(ExprSlice(t[0], 0, 1), t[1], t[2]),
                {t[0]: x, t[1]: y, t[2]: z},
            ),
        ]
    # 32-bit-only structural shapes
    t = [ExprId(f"u{i}", 32) for i in range(4)]
    x, y = ExprId("x", 32), ExprId("y", 32)
    cases += [
        (
            "compose",
            ExprCompose(ExprSlice(t[0], 0, 16), ExprSlice(t[1], 0, 16)),
            {t[0]: x, t[1]: y},
        ),
        ("mem_ptr", ExprMem(OP("+", t[0], t[1]), 32), {t[0]: x, t[1]: y}),
        ("empty_replacements", OP("+", ExprId("a", 32), ExprId("b", 32)), {}),
    ]
    return cases


CASES = _state_cases()


@pytest.mark.parametrize("name,expr_ast,repl", CASES, ids=[c[0] for c in CASES])
def test_clone_is_correct_independent_copy(name, expr_ast, repl):
    state = SynthesisState(expr_ast, dict(repl))
    clone = state.clone()

    # value-preserving
    assert clone.expr_ast == state.expr_ast
    assert clone.replacements == state.replacements
    assert clone.get_expr() == state.get_expr()
    assert clone.get_expr_simplified() == state.get_expr_simplified()

    # expr_ast shared by reference (immutable -> safe; clone does not rebuild it)
    assert clone.expr_ast is state.expr_ast

    # replacements is a private copy, not the same dict object
    assert clone.replacements is not state.replacements

    # compiled cache is private to each state
    assert clone._compiled_cache is not state._compiled_cache


@pytest.mark.parametrize("name,expr_ast,repl", CASES, ids=[c[0] for c in CASES])
def test_clone_mutation_isolation(name, expr_ast, repl):
    state = SynthesisState(expr_ast, dict(repl))
    clone = state.clone()
    before = dict(state.replacements)

    # mutate the clone's replacements: rebind an existing key and add a new one
    sentinel = ExprId("SENTINEL", expr_ast.size)
    if clone.replacements:
        some_key = next(iter(clone.replacements))
        clone.replacements[some_key] = sentinel
    clone.replacements[ExprId("brand_new", 8)] = ExprInt(0, 8)

    # the original is unchanged
    assert state.replacements == before

    # and the reverse direction: mutating the original leaves the clone intact
    state2 = SynthesisState(expr_ast, dict(repl))
    clone2 = state2.clone()
    clone_before = dict(clone2.replacements)
    state2.replacements[ExprId("only_original", 8)] = ExprInt(1, 8)
    assert clone2.replacements == clone_before


@pytest.mark.parametrize("name,expr_ast,repl", CASES, ids=[c[0] for c in CASES])
def test_clone_cleanup_isolation(name, expr_ast, repl):
    """cleanup() reassigns self.replacements; doing it on the clone must not
    touch the original (and vice versa)."""
    if not set(get_unique_variables(expr_ast)).issubset(repl):
        pytest.skip("cleanup() requires every AST leaf to have a replacement")
    state = SynthesisState(expr_ast, dict(repl))
    clone = state.clone()
    # pollute the clone with a dead replacement, then clean it up
    clone.replacements[ExprId("dead", 8)] = ExprInt(0, 8)
    original_repl = dict(state.replacements)
    clone.cleanup()
    assert state.replacements == original_repl
    # clone no longer has the dead key; original never did
    assert ExprId("dead", 8) not in clone.replacements


def test_clone_does_not_rebuild_expr_ast(monkeypatch):
    """Regression guard: clone() must not call expr_ast.copy() (a no-op rebuild
    for immutable interned Exprs). Spies on the op's copy() and asserts 0 calls."""
    t0, t1 = ExprId("t0", 32), ExprId("t1", 32)
    x, y = ExprId("x", 32), ExprId("y", 32)
    state = SynthesisState(OP("+", t0, t1), {t0: x, t1: y})

    calls = {"n": 0}
    op_type = type(state.expr_ast)
    real_copy = op_type.copy

    def spy_copy(self):
        calls["n"] += 1
        return real_copy(self)

    monkeypatch.setattr(op_type, "copy", spy_copy)
    clone = state.clone()

    assert calls["n"] == 0, "clone() rebuilt expr_ast via copy() (regression)"
    assert clone.expr_ast is state.expr_ast
    assert clone.get_expr() == state.get_expr()


def test_clone_corpus_has_diverse_shapes():
    assert len(CASES) >= 20
