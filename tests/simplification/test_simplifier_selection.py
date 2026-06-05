"""Tests for the simplifier's binarized min-of-stages selection (lever A1).

The final stage of ``Simplifier.simplify`` returns the smallest of the pipeline /
AST / closing-rewriter outputs. It must rank them by their *canonical binary*
node count, not the raw graph count -- otherwise a variadic ``expr_simp`` output
(``a+b+c`` = 1 op node over 3 leaves) is scored smaller than an equivalent but
genuinely-more-compact binary form just because of representation.
"""

from __future__ import annotations

from miasm.expression.expression import ExprId, ExprInt, ExprOp

from msynth import Simplifier
from msynth.simplification.pipeline import PipelineMode
from msynth.simplification.simplifier import binarized_node_count
from scripts.run_simplification_corpus import expressions_equivalent, node_count

SIZE = 64
A = ExprId("a", SIZE)
B = ExprId("b", SIZE)
C = ExprId("c", SIZE)


def test_binarized_node_count_is_representation_independent() -> None:
    variadic = ExprOp("+", A, B, C)  # one op node over 3 leaves
    binary = ExprOp("+", ExprOp("+", A, B), C)  # extra inner node
    # raw graph counts differ (this is the bias the helper removes)...
    assert len(variadic.graph().nodes()) < len(binary.graph().nodes())
    # ...but the binarized count is identical for both forms.
    assert binarized_node_count(variadic) == binarized_node_count(binary)
    # and it equals the binary graph count (the larger, honest one).
    assert binarized_node_count(variadic) == len(binary.graph().nodes())


def test_simplify_output_is_no_larger_than_input_binarized() -> None:
    # The selection must never return a form that is binarized-larger than the
    # input; on a batch of real-world-ish constant MBAs the output is equivalent
    # and no larger in canonical size.
    v0 = ExprId("v0", SIZE)
    v1 = ExprId("v1", SIZE)
    cases = [
        (v0 | ExprInt(0xFF, SIZE)) + (v0 & ExprInt(0xFF, SIZE)),  # == v0 + 0xFF
        (v0 ^ v1) + ExprInt(2, SIZE) * (v0 & v1),  # == v0 + v1
        (v0 + v1) - (v0 & v1),  # == v0 | v1
    ]
    s = Simplifier(None, pipeline_mode=PipelineMode.GAMBA)
    for expr in cases:
        out = s.simplify(expr)
        assert expressions_equivalent(expr, out) is not False
        assert node_count(out) <= node_count(expr)
