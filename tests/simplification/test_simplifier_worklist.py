"""Tests for the worklist subtree loop (lever C6).

The fixpoint loop processes a full pass over the subtrees, marking replaced
subtrees + descendants stale, instead of restarting the whole walk after each
hit. It must still reach a fixpoint where every independently-simplifiable
subtree is folded, soundly and no larger.
"""

from __future__ import annotations

from miasm.expression.expression import ExprId, ExprInt, ExprOp

from msynth import Simplifier
from msynth.simplification.pipeline import PipelineMode
from scripts.run_simplification_corpus import expressions_equivalent, node_count

SIZE = 64
X = ExprId("x", SIZE)
Y = ExprId("y", SIZE)
Z = ExprId("z", SIZE)


def _obf_add(a, k):
    # (a | k) + (a & k) == a + k
    return (a | k) + (a & k)


def test_worklist_folds_multiple_independent_subtrees_in_output() -> None:
    # Two independent obfuscated `+const` subtrees in one expression: both must
    # be folded (a single pass of the worklist handles siblings; the old restart
    # would take several passes but reach the same result).
    k1, k2 = ExprInt(0x1234, SIZE), ExprInt(0x5678, SIZE)
    expr = _obf_add(X, k1) + _obf_add(Y, k2)
    out = Simplifier(None, pipeline_mode=PipelineMode.GAMBA).simplify(expr)
    assert expressions_equivalent(expr, out) is not False
    assert node_count(out) < node_count(expr)
    # Neither obfuscation pattern survives (both `(a|k)+(a&k)` collapsed).
    text = str(out)
    assert "|" not in text and "&" not in text


def test_worklist_reaches_fixpoint_on_nested_obfuscation() -> None:
    # Ancestor simplification depends on a simplified child; the outer fixpoint
    # loop must reprocess the ancestor after the child is folded.
    inner = _obf_add(X, ExprInt(0x11, SIZE))  # == x + 0x11
    nested = _obf_add(inner, ExprInt(0x22, SIZE))  # == (x + 0x11) + 0x22
    out = Simplifier(None, pipeline_mode=PipelineMode.GAMBA).simplify(nested)
    assert expressions_equivalent(nested, out) is not False
    assert node_count(out) < node_count(nested)


def test_worklist_no_op_on_irreducible_expression() -> None:
    # A genuinely-irreducible mix must come back equivalent and no larger.
    expr = (X ^ Y) & (ExprOp(">>>", Z, ExprInt(7, SIZE)) | X)
    out = Simplifier(None, pipeline_mode=PipelineMode.GAMBA).simplify(expr)
    assert expressions_equivalent(expr, out) is not False
    assert node_count(out) <= node_count(expr)
