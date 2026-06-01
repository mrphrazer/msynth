from __future__ import annotations

import pickle
from pathlib import Path

import z3
from miasm.expression.expression import ExprId, ExprInt, ExprLoc, ExprOp, LocKey
from miasm.expression.simplifications import expr_simp

from msynth.simplification.oracle import SimplificationOracle
from msynth.simplification.simplifier import Simplifier


def _write_min_oracle(tmp_path: Path) -> Path:
    # create a minimal oracle instance without invoking multiprocessing-heavy init.
    oracle = SimplificationOracle.__new__(SimplificationOracle)
    oracle.num_variables = 1
    oracle.num_samples = 3
    oracle.inputs = [[0], [1], [2]]
    oracle.oracle_map = {}

    path = tmp_path / "oracle.pkl"
    with open(path, "wb") as f:
        pickle.dump(oracle, f)
    return path


def _write_empty_oracle(
    tmp_path: Path, num_variables: int = 3, num_samples: int = 8
) -> Path:
    # Empty oracle sized to hold equivalence-class queries with up to
    # ``num_variables`` unified terminals. Used to exercise paths that
    # must work without any pre-computed oracle entries.
    oracle = SimplificationOracle.__new__(SimplificationOracle)
    oracle.num_variables = num_variables
    oracle.num_samples = num_samples
    oracle.inputs = [
        [(s * 17 + v * 3 + 1) & 0xFF for v in range(num_variables)]
        for s in range(num_samples)
    ]
    oracle.oracle_map = {}

    path = tmp_path / "empty_oracle.pkl"
    with open(path, "wb") as f:
        pickle.dump(oracle, f)
    return path


def _nodes(expr) -> int:
    return len(expr.graph().nodes())


def test_skip_subtree_terminals(tmp_path: Path) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path))

    assert simplifier._skip_subtree(ExprId("p0", 8))
    assert simplifier._skip_subtree(ExprInt(1, 8))
    assert simplifier._skip_subtree(ExprLoc(LocKey(1), 8))
    assert not simplifier._skip_subtree(ExprOp("+", ExprId("p0", 8), ExprInt(1, 8)))


def test_determine_equivalence_class_adds_constant(tmp_path: Path) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path))

    p0 = ExprId("p0", 8)
    expr = ExprOp("-", p0, p0)  # always zero
    equiv_class = simplifier.determine_equivalence_class(expr)

    assert equiv_class in simplifier.oracle.oracle_map
    assert simplifier.oracle.oracle_map[equiv_class] == [ExprInt(0, 8)]


def test_reverse_global_unification_iterative(tmp_path: Path) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path))

    g0 = simplifier._gen_global_variable_replacement(0, 8)
    g1 = simplifier._gen_global_variable_replacement(1, 8)
    x = ExprId("x", 8)
    y = ExprId("y", 8)

    expr = ExprOp("+", g0, g1)
    unification = {g0: ExprOp("+", x, g1), g1: y}

    rewritten = simplifier._reverse_global_unification(expr, unification)

    assert rewritten == ExprOp("+", ExprOp("+", x, y), y)


def test_is_suitable_simplification_candidate_rejects_placeholder(
    tmp_path: Path,
) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path))

    expr = ExprOp("+", ExprId("p0", 8), ExprInt(1, 8))
    simplified = ExprId("p0", 8)

    assert not simplifier._is_suitable_simplification_candidate(expr, simplified)


def test_is_suitable_simplification_candidate_rejects_expr_simp_equivalence(
    tmp_path: Path,
) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path))

    p0 = ExprId("p0", 8)
    expr = ExprOp("+", p0, ExprInt(0, 8))

    assert not simplifier._is_suitable_simplification_candidate(expr, p0)


def test_is_suitable_simplification_candidate_enforce_equivalence(
    tmp_path: Path, monkeypatch
) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path), enforce_equivalence=True)

    expr = ExprOp("+", ExprId("p0", 8), ExprInt(1, 8))
    simplified = ExprOp("^", ExprId("p0", 8), ExprInt(1, 8))

    monkeypatch.setattr(
        simplifier, "check_semantical_equivalence", lambda _a, _b: z3.unknown
    )

    assert not simplifier._is_suitable_simplification_candidate(expr, simplified)


def test_is_suitable_simplification_candidate_accepts_unknown_without_enforce(
    tmp_path: Path, monkeypatch
) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path), enforce_equivalence=False)

    x = ExprId("x", 8)
    y = ExprId("y", 8)
    expr = ExprOp("+", ExprOp("&", x, y), ExprOp("|", x, y))
    simplified = ExprOp("+", x, y)

    monkeypatch.setattr(
        simplifier, "check_semantical_equivalence", lambda _a, _b: z3.unknown
    )

    assert simplifier._is_suitable_simplification_candidate(expr, simplified)


def test_is_suitable_simplification_candidate_rejects_larger_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path), enforce_equivalence=False)

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


def test_is_suitable_simplification_candidate_accepts_smaller_normalized_equivalent(
    tmp_path: Path,
) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path), enforce_equivalence=True)

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
    tmp_path: Path, monkeypatch
) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path), enforce_equivalence=False)

    x = ExprId("x", 8)
    y = ExprId("y", 8)
    expr = ExprOp("+", x, y)
    simplified = x

    monkeypatch.setattr(
        simplifier, "check_semantical_equivalence", lambda _a, _b: z3.unknown
    )

    assert not simplifier._is_suitable_simplification_candidate(expr, simplified)


def test_is_suitable_simplification_candidate_rejects_sat(
    tmp_path: Path, monkeypatch
) -> None:
    simplifier = Simplifier(_write_min_oracle(tmp_path))

    expr = ExprOp("+", ExprId("p0", 8), ExprInt(1, 8))
    simplified = ExprOp("^", ExprId("p0", 8), ExprInt(1, 8))

    monkeypatch.setattr(
        simplifier, "check_semantical_equivalence", lambda _a, _b: z3.sat
    )

    assert not simplifier._is_suitable_simplification_candidate(expr, simplified)


def test_subtree_simba_fallback_simplifies_on_empty_oracle(tmp_path: Path) -> None:
    # With an empty oracle, no pre-computed equivalence class can ever match.
    # An inner linear MBA wrapped in a non-constant right shift cannot be
    # simplified by the global SimbaPass either (the root op is outside the
    # linear-MBA fragment). The only path that can produce a simplification
    # is the subtree-level SiMBA fallback, applied to the inner subtree.
    size = 64
    x = ExprId("x", size)
    y = ExprId("y", size)
    shift = ExprId("shift", size)
    inner = ExprOp("+", ExprOp("&", x, y), ExprOp("|", x, y))
    expr = ExprOp(">>", inner, shift)

    simplifier = Simplifier(
        _write_empty_oracle(tmp_path), enable_subtree_simba=True
    )
    simplified = simplifier.simplify(expr)

    # Inner (x & y) + (x | y) collapses to (x + y); outer >> shift remains.
    expected = ExprOp(">>", ExprOp("+", x, y), shift)
    assert simplified == expected
    assert _nodes(simplified) < _nodes(expr)


def test_subtree_simba_disabled_leaves_inner_linear_mba_untouched(
    tmp_path: Path,
) -> None:
    # Counterproof for the previous test: with subtree-SiMBA disabled and an
    # empty oracle, no path can match the inner subtree, so the expression
    # must come back unchanged.
    size = 64
    x = ExprId("x", size)
    y = ExprId("y", size)
    shift = ExprId("shift", size)
    inner = ExprOp("+", ExprOp("&", x, y), ExprOp("|", x, y))
    expr = ExprOp(">>", inner, shift)

    simplifier = Simplifier(
        _write_empty_oracle(tmp_path), enable_subtree_simba=False
    )
    simplified = simplifier.simplify(expr)

    assert simplified == expr


def test_subtree_simba_respects_op_whitelist(tmp_path: Path) -> None:
    # The non-constant shift at the root is outside the operator whitelist,
    # so subtree-SiMBA must reject it regardless of size or variable count.
    # Combined with an empty oracle, this proves the whitelist is enforced:
    # the only candidate subtree is a leaf chain and nothing fires.
    size = 64
    x = ExprId("x", size)
    shift = ExprId("shift", size)
    expr = ExprOp(">>", x, shift)

    simplifier = Simplifier(
        _write_empty_oracle(tmp_path), enable_subtree_simba=True
    )
    simplified = simplifier.simplify(expr)

    assert simplified == expr


def test_subtree_simba_respects_node_limit(tmp_path: Path) -> None:
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
        _write_empty_oracle(tmp_path),
        enable_subtree_simba=True,
        subtree_simba_max_nodes=4,
    )
    simplified = simplifier.simplify(expr)

    assert simplified == expr
