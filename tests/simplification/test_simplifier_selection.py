"""Tests for the simplifier's binarized min-of-stages selection (lever A1).

The final stage of ``Simplifier.simplify`` returns the smallest of the pipeline /
AST / closing-rewriter outputs. It must rank them by their *canonical binary*
node count, not the raw graph count -- otherwise a variadic ``expr_simp`` output
(``a+b+c`` = 1 op node over 3 leaves) is scored smaller than an equivalent but
genuinely-more-compact binary form just because of representation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import z3
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


@pytest.mark.parametrize("mode", [PipelineMode.SIMBA, PipelineMode.GAMBA])
@pytest.mark.parametrize("bits,key", [(8, 9), (16, 0x5678), (32, 0x12345678), (64, 0x12345678)])
def test_strict_selection_preserves_rare_exception(mode, bits, key) -> None:
    x = ExprId("x", bits)
    t = x ^ ExprInt(key, bits)
    original = x + (((t | -t) >> ExprInt(bits - 1, bits)) ^ ExprInt(1, bits))
    simplifier = Simplifier(None, pipeline_mode=mode, enforce_equivalence=True)
    result = simplifier.simplify(original)
    assert simplifier.check_semantical_equivalence(original, result) == z3.unsat


@pytest.mark.parametrize("verdict", [z3.sat, z3.unknown, z3.unsat])
def test_strict_selection_checks_unique_candidate_once(monkeypatch, verdict) -> None:
    simplifier = Simplifier(None, enforce_equivalence=True)
    monkeypatch.setattr(type(simplifier.pipeline), "run", lambda self, expr: B)
    calls = []

    def check(original, candidate):
        calls.append((original, candidate))
        return verdict

    def no_sampling(*args):
        pytest.fail("Strict selection must not use the permissive gate")

    monkeypatch.setattr(simplifier, "check_semantical_equivalence", check)
    monkeypatch.setattr(simplifier, "_permissive_equivalent", no_sampling)
    result = simplifier.simplify(A)
    assert result == (B if verdict == z3.unsat else A)
    assert calls == [(A, B)]


def test_strict_selection_skips_identity_proof(monkeypatch) -> None:
    simplifier = Simplifier(None, enforce_equivalence=True)

    def no_proof(*args):
        pytest.fail("Structural identity needs no solver query")

    monkeypatch.setattr(simplifier, "check_semantical_equivalence", no_proof)
    assert simplifier.simplify(A) == A


def test_non_strict_selection_retains_sampling(monkeypatch) -> None:
    simplifier = Simplifier(None, enforce_equivalence=False)
    monkeypatch.setattr(type(simplifier.pipeline), "run", lambda self, expr: B)
    monkeypatch.setattr(simplifier, "_permissive_equivalent", lambda *args: True)

    def no_proof(*args):
        pytest.fail("Non-strict selection must not require SMT")

    monkeypatch.setattr(simplifier, "check_semantical_equivalence", no_proof)
    assert simplifier.simplify(A) == B


@pytest.mark.parametrize("accept_first", [False, True])
def test_strict_selection_ranks_and_proves_all_stage_outputs(monkeypatch, accept_first) -> None:
    simplifier = Simplifier(None, enforce_equivalence=True)
    monkeypatch.setattr(type(simplifier.pipeline), "run", lambda self, expr: B)
    monkeypatch.setattr(simplifier, "_reverse_global_unification", lambda *args: A + B)
    monkeypatch.setattr(
        "msynth.simplification.simplifier.DEFAULT_REWRITER",
        SimpleNamespace(normalize=lambda expr: A),
    )
    calls = []

    def check(original, candidate):
        assert original == C
        calls.append(candidate)
        if accept_first or candidate == A + B:
            return z3.unsat
        return z3.unknown if candidate == B else z3.sat

    monkeypatch.setattr(simplifier, "check_semantical_equivalence", check)
    result = simplifier.simplify(C)
    assert result == (B if accept_first else A + B)
    assert calls == ([B] if accept_first else [B, A, A + B])
