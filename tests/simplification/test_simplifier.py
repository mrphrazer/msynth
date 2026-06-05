from __future__ import annotations

import pickle
from pathlib import Path

import z3
from miasm.expression.expression import ExprId, ExprInt, ExprLoc, ExprOp, LocKey
from miasm.expression.simplifications import expr_simp

from msynth.simplification.oracle import SimplificationOracle
from msynth.simplification.pipeline import PipelineMode
from msynth.simplification.simplifier import Simplifier


def _write_min_oracle(tmp_path: Path) -> Path:
    """
    Pickle a degenerate :class:`SimplificationOracle` (empty ``oracle_map``,
    one variable, three samples) for the few tests that exercise oracle
    internals — ``determine_equivalence_class`` adding constant entries,
    the demo-MBA reduction against ``_FULL_ORACLE``, etc. Tests that only
    need the no-oracle fall-through path use ``Simplifier()`` directly
    instead and skip this helper.
    """
    oracle = SimplificationOracle.__new__(SimplificationOracle)
    oracle.num_variables = 1
    oracle.num_samples = 3
    oracle.inputs = [[0], [1], [2]]
    oracle.oracle_map = {}

    path = tmp_path / "oracle.pkl"
    with open(path, "wb") as f:
        pickle.dump(oracle, f)
    return path


def _nodes(expr) -> int:
    return len(expr.graph().nodes())


def test_skip_subtree_terminals() -> None:
    simplifier = Simplifier()

    assert simplifier._skip_subtree(ExprId("p0", 8))
    assert simplifier._skip_subtree(ExprInt(1, 8))
    assert simplifier._skip_subtree(ExprLoc(LocKey(1), 8))
    assert not simplifier._skip_subtree(ExprOp("+", ExprId("p0", 8), ExprInt(1, 8)))


def test_determine_equivalence_class_adds_constant(tmp_path: Path) -> None:
    # Needs a loaded oracle: ``determine_equivalence_class`` writes the
    # constant equiv-class entry back into ``self.oracle.oracle_map``.
    simplifier = Simplifier(_write_min_oracle(tmp_path))

    p0 = ExprId("p0", 8)
    expr = ExprOp("-", p0, p0)  # always zero
    equiv_class = simplifier.determine_equivalence_class(expr)

    assert equiv_class in simplifier.oracle.oracle_map
    assert simplifier.oracle.oracle_map[equiv_class] == [ExprInt(0, 8)]


def test_reverse_global_unification_iterative() -> None:
    simplifier = Simplifier()

    g0 = simplifier._gen_global_variable_replacement(0, 8)
    g1 = simplifier._gen_global_variable_replacement(1, 8)
    x = ExprId("x", 8)
    y = ExprId("y", 8)

    expr = ExprOp("+", g0, g1)
    unification = {g0: ExprOp("+", x, g1), g1: y}

    rewritten = simplifier._reverse_global_unification(expr, unification)

    assert rewritten == ExprOp("+", ExprOp("+", x, y), y)


def test_is_suitable_simplification_candidate_rejects_placeholder() -> None:
    simplifier = Simplifier()

    expr = ExprOp("+", ExprId("p0", 8), ExprInt(1, 8))
    simplified = ExprId("p0", 8)

    assert not simplifier._is_suitable_simplification_candidate(expr, simplified)


def test_is_suitable_simplification_candidate_rejects_expr_simp_equivalence() -> None:
    simplifier = Simplifier()

    p0 = ExprId("p0", 8)
    expr = ExprOp("+", p0, ExprInt(0, 8))

    assert not simplifier._is_suitable_simplification_candidate(expr, p0)


def test_is_suitable_simplification_candidate_enforce_equivalence(monkeypatch) -> None:
    simplifier = Simplifier(enforce_equivalence=True)

    expr = ExprOp("+", ExprId("p0", 8), ExprInt(1, 8))
    simplified = ExprOp("^", ExprId("p0", 8), ExprInt(1, 8))

    monkeypatch.setattr(
        simplifier, "check_semantical_equivalence", lambda _a, _b: z3.unknown
    )

    assert not simplifier._is_suitable_simplification_candidate(expr, simplified)


def test_is_suitable_simplification_candidate_accepts_unknown_without_enforce(
    monkeypatch,
) -> None:
    simplifier = Simplifier(enforce_equivalence=False)

    x = ExprId("x", 8)
    y = ExprId("y", 8)
    expr = ExprOp("+", ExprOp("&", x, y), ExprOp("|", x, y))
    simplified = ExprOp("+", x, y)

    monkeypatch.setattr(
        simplifier, "check_semantical_equivalence", lambda _a, _b: z3.unknown
    )

    assert simplifier._is_suitable_simplification_candidate(expr, simplified)


def test_is_suitable_simplification_candidate_rejects_larger_candidate(
    monkeypatch,
) -> None:
    simplifier = Simplifier(enforce_equivalence=False)

    x = ExprId("x", 8)
    expr = ExprOp("^", x, ExprOp("^", x, ExprOp("-", x)))
    simplified = ExprOp(
        "*",
        x,
        ExprOp("^", ExprOp("<<", ExprInt(2, 8), ExprOp("-", x)), ExprInt(0xFF, 8)),
    )

    monkeypatch.setattr(
        simplifier, "check_semantical_equivalence", lambda _a, _b: z3.unknown
    )

    assert not simplifier._is_suitable_simplification_candidate(expr, simplified)


def test_is_suitable_simplification_candidate_accepts_smaller_normalized_equivalent() -> (
    None
):
    simplifier = Simplifier(enforce_equivalence=True)

    x = ExprId("x", 64)
    y = ExprId("y", 64)
    expr = ExprOp(
        "&",
        x,
        ExprOp(
            "+",
            x,
            ExprOp("-", ExprOp("+", x, ExprOp("-", ExprOp("&", x, y)))),
        ),
    )
    simplified = ExprOp("&", x, y)

    assert expr_simp(expr) == expr_simp(simplified)
    assert simplifier._is_suitable_simplification_candidate(expr, simplified)


def test_is_suitable_simplification_candidate_rejects_unknown_counterexample(
    monkeypatch,
) -> None:
    simplifier = Simplifier(enforce_equivalence=False)

    x = ExprId("x", 8)
    y = ExprId("y", 8)
    expr = ExprOp("+", x, y)
    simplified = x

    monkeypatch.setattr(
        simplifier, "check_semantical_equivalence", lambda _a, _b: z3.unknown
    )

    assert not simplifier._is_suitable_simplification_candidate(expr, simplified)


def test_is_suitable_simplification_candidate_rejects_sat(monkeypatch) -> None:
    simplifier = Simplifier()

    expr = ExprOp("+", ExprId("p0", 8), ExprInt(1, 8))
    simplified = ExprOp("^", ExprId("p0", 8), ExprInt(1, 8))

    monkeypatch.setattr(
        simplifier, "check_semantical_equivalence", lambda _a, _b: z3.sat
    )

    assert not simplifier._is_suitable_simplification_candidate(expr, simplified)


def test_subtree_simba_fallback_simplifies_inner_linear_mba() -> None:
    # The default Simplifier holds an empty oracle, so no pre-computed
    # equivalence class can ever match. An inner linear MBA wrapped in a
    # non-constant right shift cannot be simplified by the global
    # SimbaPass either (the root op is outside the linear-MBA fragment).
    # The only path that can produce a simplification is the subtree-
    # level SiMBA fallback, applied to the inner subtree.
    size = 64
    x = ExprId("x", size)
    y = ExprId("y", size)
    shift = ExprId("shift", size)
    inner = ExprOp("+", ExprOp("&", x, y), ExprOp("|", x, y))
    expr = ExprOp(">>", inner, shift)

    simplifier = Simplifier(pipeline_mode=PipelineMode.SIMBA)
    simplified = simplifier.simplify(expr)

    # Inner (x & y) + (x | y) collapses to (x + y); outer >> shift remains.
    expected = ExprOp(">>", ExprOp("+", x, y), shift)
    assert simplified == expected
    assert _nodes(simplified) < _nodes(expr)


def test_ast_mode_leaves_inner_linear_mba_untouched() -> None:
    # Counterproof for the previous test: under PipelineMode.AST,
    # subtree-SiMBA is disabled. With no pre-computed oracle match and no
    # SimBA fallback, the expression must come back unchanged. (AST is no
    # longer the constructor default, so request it explicitly.)
    size = 64
    x = ExprId("x", size)
    y = ExprId("y", size)
    shift = ExprId("shift", size)
    inner = ExprOp("+", ExprOp("&", x, y), ExprOp("|", x, y))
    expr = ExprOp(">>", inner, shift)

    simplifier = Simplifier(pipeline_mode=PipelineMode.AST)
    simplified = simplifier.simplify(expr)

    assert simplified == expr


def test_subtree_simba_respects_op_whitelist() -> None:
    # The non-constant shift at the root is outside the operator whitelist,
    # so subtree-SiMBA must reject it regardless of size or variable count.
    # Combined with the empty default oracle, this proves the whitelist
    # is enforced: the only candidate subtree is a leaf chain and nothing
    # fires.
    size = 64
    x = ExprId("x", size)
    shift = ExprId("shift", size)
    expr = ExprOp(">>", x, shift)

    simplifier = Simplifier(pipeline_mode=PipelineMode.SIMBA)
    simplified = simplifier.simplify(expr)

    assert simplified == expr


def test_subtree_simba_skips_placeholder_terminals() -> None:
    # When the unification dict's keys are global_reg placeholders introduced
    # by an earlier BFS replacement, subtree-SiMBA must skip the subtree.
    #
    # SiMBA's cube reconstruction over the placeholder atoms is *sound* (the
    # linear-MBA theorem doesn't care that an atom stands in for another
    # expression), but it produces the conjunction-basis canonical form:
    # ``g0 + g1 + g2 + g0`` becomes ``2*g0 + g1 + g2``. Once that
    # coefficient-times-placeholder form is cemented as a new placeholder
    # body, ring_normalize's structural like-term collection cannot fold
    # the underlying atom-level terms together with sibling sub-expressions,
    # and the simplifier converges to a strictly-larger fixed point.
    #
    # See ``test_simplifier_demo_mba_reaches_shortest_form_with_placeholder_guard``
    # for the end-to-end regression that motivates this guard.
    simplifier = Simplifier(pipeline_mode=PipelineMode.SIMBA)

    size = 64
    g0 = simplifier._gen_global_variable_replacement(0, size)
    g1 = simplifier._gen_global_variable_replacement(1, size)
    g2 = simplifier._gen_global_variable_replacement(2, size)
    # A flat sum of placeholders with a repeated term — exactly the shape
    # that previously fired SiMBA and produced `2*g0 + g1 + g2` form.
    subtree = ExprOp("+", g0, g1, g2, g0)
    from msynth.utils.unification import gen_unification_dict

    result = simplifier._try_subtree_simba(subtree, gen_unification_dict(subtree))
    assert result is None


import pytest  # noqa: E402  (kept local to the slow regression below)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FULL_ORACLE = _REPO_ROOT / "oracle.pickle"


@pytest.mark.slow
@pytest.mark.skipif(
    not _FULL_ORACLE.is_file(),
    reason="requires the checked-in oracle.pickle (60MB precomputed library)",
)
def test_simplifier_demo_mba_reaches_shortest_form_with_placeholder_guard() -> None:
    """
    End-to-end regression for the placeholder-guard in ``_try_subtree_simba``.

    Setup: a 55-node hand-built MBA over three 32-bit variables
    ``v0, v1, v2`` (inlined below; same shape as the demo expression in
    ``scripts/simplify_expression.py``). It expands semantically to
    ``6*v0 + 6*v1 + 3*v2``. Under the SIMBA default (subtree-SiMBA active)
    the simplifier converges to that distributed form ``6*v0 + 6*v1 + 3*v2``
    (10 graph nodes); the even-shorter factored form ``v2*3 + (v0+v1)*6``
    (9 nodes) is what the oracle-only AST path emits.

    Why this test exists: when subtree-SiMBA is allowed to run on subtrees
    whose unification dict contains ``global_reg*`` placeholders left over
    from earlier oracle hits, it canonicalises them into the linear-MBA
    conjunction basis. That intermediate form introduces shifts and
    coefficient-times-placeholder terms (e.g. ``v0 << 1`` instead of
    ``v0 + v0``, ``0xFF * g_k`` instead of ``-g_k``) which the downstream
    ``ring_normalize`` pass cannot fold back together — like-term
    collection is structural, and once the shift is in place the two
    sibling ``v0`` contributions stop sharing a structural form. Without
    the guard the simplifier converges to
    ``2*v0 + 3*v2 + 6*v1 + 2*(v0 << 1)`` (13 graph nodes) — same
    coefficient combination, 44% larger, no further reduction possible.

    The check is therefore two-pronged: assert the result is semantically
    ``6*v0 + 6*v1 + 3*v2`` (via ``expr_simp`` for structural+algebraic
    normalisation), *and* assert the node count is ≤ 10 so that a future
    regression of the placeholder guard (which would blow the form up to
    the 13-node shifted version above) fails this test rather than
    silently degrading the output's compactness.

    Marked ``slow`` because it loads the 60MB precomputed oracle; the
    simplification itself runs in ~6 seconds on a typical worker.
    """
    size = 32
    v0 = ExprId("v0", size)
    v1 = ExprId("v1", size)
    v2 = ExprId("v2", size)
    one = ExprInt(0x1, size)

    # Three structurally repeated sub-MBAs, each a known identity over
    # ``v0, v1, v2``. The repetition is the load-bearing part of the
    # regression: the simplifier hits oracle simplifications on the
    # inner sub-MBAs, introducing ``global_reg*`` placeholders, and the
    # outer flat sum is then the subtree whose unification dict would
    # contain those placeholders.
    block_a = (~v0 | v2) - ~(
        (~((((one + v2) - one) | ~v0) - ~v0) & v2) + (v2 + (v2 & ~v2))
    )
    block_b = (
        (v0 & (((v0 & v1) + (v0 & v1)) + (v0 ^ v1)))
        + (v0 & (((v0 & v1) + (v0 & v1)) + (v0 ^ v1)))
    ) + (v0 ^ (((v0 & v1) + (v0 & v1)) + (v0 ^ v1)))
    block_c = (
        one
        + ~(
            ((v1 + (~v1 & v2)) | ((v1 + (~v1 & v2)) + (~(v1 + (~v1 & v2)) & v2)))
            - (v1 & (v1 + (~v1 & v2)))
        )
    ) + ((-(-v0)) + (((v1 + (~v1 & v2)) + (~(v1 + (~v1 & v2)) & v2)) & ~v2))
    expr = (
        block_a
        + block_b
        + block_c
        + block_a
        + block_b
        + block_c
        + block_a
        + block_b
        + block_c
    )

    simplifier = Simplifier(_FULL_ORACLE, pipeline_mode=PipelineMode.SIMBA)
    simplified = simplifier.simplify(expr)

    # Semantic check by concrete sampling — robust to the canonical
    # form msynth chooses to emit (factored vs distributed). We expect
    # the simplified expression to behave like ``6*v0 + 6*v1 + 3*v2``
    # for every input. A small deterministic sample set is sufficient
    # because a single disagreement on a 3-variable linear MBA fails
    # the algebraic identity globally.
    mask = (1 << size) - 1
    samples = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 2, 3),
        (0xDEAD_BEEF, 0x1234_5678, 0xCAFE_BABE),
        (mask, mask, mask),
        (mask - 1, 1, 2),
    ]
    for a, b, c in samples:
        env = {v0: ExprInt(a, size), v1: ExprInt(b, size), v2: ExprInt(c, size)}
        got = int(expr_simp(simplified.replace_expr(env)))
        want = (6 * a + 6 * b + 3 * c) & mask
        assert got == want, (
            f"semantic mismatch at v0={a:#x} v1={b:#x} v2={c:#x}: "
            f"got {got:#x}, want {want:#x}\n  simplified: {simplified}"
        )

    assert _nodes(simplified) <= 10, (
        f"shortest-form regression: got {_nodes(simplified)} nodes, "
        f"expected ≤ 10; result was {simplified!r}. "
        "If this fires, the placeholder guard in _try_subtree_simba was "
        "likely removed or weakened — see the comment there."
    )


def test_subtree_simba_respects_node_limit() -> None:
    # The inner linear MBA has 5 nodes; setting the node limit to 4 must
    # block subtree-SiMBA from firing, so the expression comes back
    # untouched even with subtree-SiMBA otherwise enabled.
    size = 64
    x = ExprId("x", size)
    y = ExprId("y", size)
    shift = ExprId("shift", size)
    inner = ExprOp("+", ExprOp("&", x, y), ExprOp("|", x, y))
    expr = ExprOp(">>", inner, shift)

    simplifier = Simplifier(
        pipeline_mode=PipelineMode.SIMBA,
        subtree_simba_max_nodes=4,
    )
    simplified = simplifier.simplify(expr)

    assert simplified == expr


# ---------------------------------------------------------------------------
# Constructor: oracle + pipeline-mode + subtree-SimBA coupling
# ---------------------------------------------------------------------------


def test_simplifier_default_oracle_is_empty_in_memory() -> None:
    # No oracle_path → in-memory empty oracle. Lookups never match;
    # simplifications come from the pipeline / CEGIS only.
    sim = Simplifier()
    assert isinstance(sim.oracle, SimplificationOracle)
    assert sim.oracle.oracle_map == {}


def test_simplifier_oracle_loaded_when_path_provided(tmp_path: Path) -> None:
    sim = Simplifier(_write_min_oracle(tmp_path))
    assert isinstance(sim.oracle, SimplificationOracle)
    assert sim.oracle.num_variables == 1
    assert sim.oracle.num_samples == 3


def test_simplifier_pipeline_mode_ast_disables_subtree_simba() -> None:
    sim = Simplifier(pipeline_mode=PipelineMode.AST)
    assert sim._subtree_simba_pass is None


def test_simplifier_pipeline_mode_simba_enables_subtree_simba() -> None:
    sim = Simplifier(pipeline_mode=PipelineMode.SIMBA)
    assert sim._subtree_simba_pass is not None
    assert sim._pipeline_mode == PipelineMode.SIMBA


def test_simplifier_pipeline_mode_gamba_enables_subtree_simba() -> None:
    sim = Simplifier(pipeline_mode=PipelineMode.GAMBA)
    assert sim._subtree_simba_pass is not None
    assert sim._pipeline_mode == PipelineMode.GAMBA


def test_simplifier_accepts_string_pipeline_mode_literals() -> None:
    # ``PipelineMode`` inherits from ``str`` so callers may pass the
    # bare literal when an enum import is inconvenient.
    sim = Simplifier(pipeline_mode="simba")
    assert sim._pipeline_mode == PipelineMode.SIMBA
    assert [type(p).__name__ for p in sim.pipeline.passes] == [
        "SimbaPass",
        "AstNormalizationPass",
    ]
