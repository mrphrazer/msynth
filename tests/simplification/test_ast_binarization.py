"""Binarization (fixed-arity) characteristic of AbstractSyntaxTreeTranslator.

The translator rewrites every n-ary ExprOp / ExprCompose into a tree of arity-2
nodes (e.g. a + b + c  ->  (a + b) + c), recursively, at every layer. These tests
pin that contract: for a wide variety of tree shapes we assert the "fully binary"
characteristic does NOT hold for the input (some op has arity > 2) and DOES hold
after translation, and that translation preserves the value.

These tests fail loudly if binarization regresses at any layer (e.g. if from_ExprOp
stops recursing into operands or stops folding n-ary nodes down to arity 2).
"""
from __future__ import annotations

import pytest
import z3

from miasm.expression.expression import (
    Expr, ExprId, ExprInt, ExprOp, ExprSlice, ExprCond, ExprCompose, ExprMem,
)
from miasm.ir.translators.z3_ir import TranslatorZ3

from msynth.simplification.ast import AbstractSyntaxTreeTranslator

# ---- vocabulary --------------------------------------------------------------
a, b, c, d, e = (ExprId(n, 32) for n in "abcde")
p, q, r, s = (ExprId(n, 8) for n in "pqrs")
base = ExprId("base", 32)
I = lambda v, sz=32: ExprInt(v, sz)


def _children(x: Expr):
    if isinstance(x, ExprOp):
        return x.args
    if isinstance(x, ExprCond):
        return (x.cond, x.src1, x.src2)
    if isinstance(x, ExprSlice):
        return (x.arg,)
    if isinstance(x, ExprCompose):
        return x.args
    if isinstance(x, ExprMem):
        return (x.ptr,)
    return ()


def max_op_arity(expr: Expr) -> int:
    """Largest arity of any ExprOp/ExprCompose node in the DAG (iterative)."""
    worst, stack, seen = 0, [expr], set()
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        if isinstance(x, (ExprOp, ExprCompose)):
            worst = max(worst, len(x.args))
        stack.extend(_children(x))
    return worst


def is_fully_binary(expr: Expr) -> bool:
    """The characteristic: no ExprOp/ExprCompose node has arity > 2."""
    return max_op_arity(expr) <= 2


def _equivalent(x: Expr, y: Expr) -> bool:
    """Z3-prove x == y (returns False if not provably equal)."""
    tz = TranslatorZ3()
    solver = z3.Solver()
    solver.add(tz.from_expr(x) != tz.from_expr(y))
    return solver.check() == z3.unsat


# ---- the corpus of shapes (each input has some op of arity > 2) --------------
# NOTE: Python's a + b + c builds an ALREADY-binary (a+b)+c (left-assoc), so we use
# explicit ExprOp(op, x, y, z, ...) to construct genuine n-ary nodes.
# (name, expr, check_equivalence?) -- equivalence skipped where z3 lacks a model
# (ExprMem); binarization + structure is still checked for those.
O = ExprOp
CASES = [
    # flat n-ary of each associative op, various arities
    ("op_add3", O("+", a, b, c), True),
    ("op_mul3", O("*", a, b, c), True),
    ("op_and3", O("&", a, b, c), True),
    ("op_or3", O("|", a, b, c), True),
    ("op_xor3", O("^", a, b, c), True),
    ("op_add4", O("+", a, b, c, d), True),
    ("op_mul5", O("*", a, b, c, d, e), True),
    ("op_add6_with_ints", O("+", a, b, c, I(7), I(9), I(0x1000)), True),
    ("op_arity10", O("+", a, b, c, d, e, a, b, c, d, e), True),
    ("op_ints_and_vars", O("*", I(3), a, b, I(5)), True),
    # nested SAME op (layered): the inner n-ary must also be binarized
    ("nested_same_outer2_inner3", O("+", a, O("+", a, b, c)), True),
    ("nested_same_two_naries", O("+", O("+", a, b, c), O("+", b, c, d)), True),
    ("nested_same_3level", O("+", O("+", O("+", a, b, c), d, e), b, c), True),
    ("nested_same_left_inner", O("+", O("+", a, b, c), d, e), True),
    # nested DIFFERENT ops (cross)
    ("cross_mul_of_adds", O("*", O("+", a, b, c), O("+", d, e, a)), True),
    ("cross_add_of_muls", O("+", O("*", a, b, c), O("*", d, e, a)), True),
    ("mixed_3level", O("+", O("^", O("+", a, b, c), O("|", d, e, a)), O("*", O("&", b, c, d), a)), True),
    ("multi_nary_siblings", O("|", O("+", a, b, c), O("*", d, e, a), O("^", b, c, d)), True),
    ("asymmetric_deep", O("+", a, O("*", b, O("+", c, d, e, a))), True),
    ("deep_mba_xor5", O("^", a, b, c, O("&", a, b, c), O("|", a, b, c)), True),
    # n-ary buried under other node types (unary / shift / slice / cond / compose / mem)
    ("unary_minus_of_nary", O("-", O("+", a, b, c)), True),
    ("shift_left_nary_operand", O("<<", O("+", a, b, c, d), I(3)), True),
    ("shift_right_nary_operand", O(">>", O("*", a, b, c), I(2)), True),
    ("slice_of_nary", ExprSlice(O("+", a, b, c, d), 0, 16), True),
    ("nary_inside_slice_operand", ExprSlice(O("+", a, b, c, e), 8, 24), True),
    ("cond_arms_nary", ExprCond(ExprSlice(O("+", a, b), 0, 1), O("+", a, b, c), O("^", d, e, a)), True),
    ("cond_condition_nary", ExprCond(ExprSlice(O("&", a, b, c), 0, 1), b, c), True),
    ("cond_everywhere_nary", ExprCond(ExprSlice(O("+", a, b, c), 0, 1), O("*", a, b, c), O("|", d, e, a)), True),
    ("compose3_bytes", ExprCompose(p, q, r), True),
    ("compose4_bytes", ExprCompose(p, q, r, s), True),
    ("compose_of_nary_halves", ExprCompose(ExprSlice(O("+", a, b, c, d), 0, 16), ExprSlice(a, 0, 16)), True),
    ("nary_inside_compose_arg", ExprCompose(ExprSlice(O("^", a, b, c, d), 0, 16), ExprSlice(e, 0, 16)), True),
    ("mem_ptr_nary", ExprMem(O("+", a, b, c, d), 32), False),
    ("mem_ptr_mixed_nary", ExprMem(O("+", base, a, b, c), 32), False),
    ("nary_with_mem_operand", O("+", a, ExprMem(base, 32), c), False),
    # wide-and-deep combos
    ("wide_deep_combo", O("+", O("*", a, b, c), O("^", d, e, a), O("&", b, c, d), O("|", a, d, e)), True),
    ("nary_at_multiple_depths", O("*", O("+", a, b, c), d, O("&", a, O("|", b, c, d), e)), True),
    # extra deterministic shapes (breadth; no randomization)
    ("very_wide_add16", O("+", a, b, c, d, e, a, b, c, d, e, a, b, c, d, e, a), True),
    ("very_wide_xor8", O("^", a, b, c, d, e, a, b, c), True),
    ("all_leaves_same_and4", O("&", a, a, a, a), True),
    ("deep_nest_4level_same", O("+", O("+", O("+", O("+", a, b, c), d, e), a, b), c, d), True),
    ("deep_nest_4level_cross", O("*", O("+", a, O("^", b, c, O("&", d, e, a))), b, c), True),
    ("ops_mix_with_shifts", O("+", O("<<", O("*", a, b, c), I(2)), O(">>", O("|", d, e, a), I(3)), c), True),
    ("binary_sub_of_naries", O("-", O("+", a, b, c), O("+", d, e, a)), True),
    # shared n-ary subtree (DAG): the same arity-3 node reused 3x, binarized once
    ("shared_nary_subtree_dag", O("+", O("^", a, b, c), O("*", O("^", a, b, c), d), O("^", a, b, c)), True),
    ("nested_compose_two_naries", ExprCompose(ExprSlice(O("+", a, b, c), 0, 16), ExprSlice(O("*", d, e, a), 0, 16)), True),
]


@pytest.mark.parametrize("name,expr,check_equiv", CASES, ids=[c[0] for c in CASES])
def test_binarization_before_and_after(name, expr, check_equiv):
    # characteristic must NOT hold beforehand (the input has an op of arity > 2)
    assert not is_fully_binary(expr), (
        f"{name}: input is already fully binary (max arity "
        f"{max_op_arity(expr)}) -- this case does not exercise binarization")
    assert max_op_arity(expr) >= 3

    translated = AbstractSyntaxTreeTranslator().from_expr(expr)

    # characteristic MUST hold afterwards (every op/compose node has arity <= 2)
    assert is_fully_binary(translated), (
        f"{name}: translated AST still has an op of arity "
        f"{max_op_arity(translated)} > 2")

    # translation must preserve the value
    if check_equiv:
        assert _equivalent(expr, translated), f"{name}: binarized AST not equivalent to input"


def test_corpus_has_diverse_shapes():
    # guard against the corpus silently shrinking / losing coverage
    assert len(CASES) >= 30
    # every input is genuinely non-binary, every output genuinely binary
    t = AbstractSyntaxTreeTranslator()
    for name, expr, _ in CASES:
        assert max_op_arity(expr) >= 3, name
        assert max_op_arity(t.from_expr(expr)) <= 2, name
