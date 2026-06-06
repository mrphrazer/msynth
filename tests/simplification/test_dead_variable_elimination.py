"""
Tests for the probabilistic dead/opaque-variable elimination pass.

The pass detects variables that never affect the output, replaces them with 0,
and accepts the result only after a sampling gate + short-timeout Z3 gate
(SAT -> reject, UNSAT/UNKNOWN -> accept). It must:

  * find genuinely dead variables and prune them (equivalent, smaller);
  * leave every live variable in place -- including the hard "magic-constant
    gate" case where a variable is live only near a constant harvested from the
    AST;
  * never accept a non-equivalent pruning (the Z3 gate is the soundness
    backstop when sampling misses a witness);
  * recover the largest sound subset via block-splitting repair when the bulk
    replacement contains a false positive;
  * compose as a pipeline pass and wire into the Simplifier behind an opt-in
    flag without changing default behaviour.
"""

from __future__ import annotations

import random

import pytest
from miasm.expression.expression import (
    ExprCond,
    ExprId,
    ExprInt,
    ExprMem,
    ExprOp,
)

from msynth import Simplifier
from msynth.simplification.dead_variable_elimination import (
    DeadVariableEliminationConfig,
    DeadVariableEliminationPass,
    VariableStatus,
)
from msynth.simplification.pipeline import PipelineMode
from scripts.run_simplification_corpus import expressions_equivalent, node_count

SIZE = 32
X = ExprId("x", SIZE)
Y = ExprId("y", SIZE)
Z = ExprId("z", SIZE)
FAKE = ExprId("fake", SIZE)
JUNK = ExprId("junk", SIZE)


def _pass(**overrides) -> DeadVariableEliminationPass:
    cfg = DeadVariableEliminationConfig(enabled=True, **overrides)
    return DeadVariableEliminationPass(cfg)


def _names(variables) -> set:
    return {str(v) for v in variables}


# --------------------------------------------------------------------------- #
# Dead detection
# --------------------------------------------------------------------------- #
def test_eliminates_xor_self_and_zero_product() -> None:
    # fake ^ fake == 0, so (fake ^ fake) * y == 0 -> both fake and y are dead.
    expr = X + ExprOp("*", FAKE ^ FAKE, Y)
    pass_ = _pass()
    out = pass_.run(expr)
    assert _names(pass_.last_result.candidates) == {"fake", "y"}
    assert pass_.last_result.final_equivalent is True
    assert expressions_equivalent(expr, out) is not False
    assert node_count(out) < node_count(expr)


def test_zero_product_variable_is_dead() -> None:
    expr = X + ExprOp("*", JUNK, ExprInt(0, SIZE))
    pass_ = _pass()
    out = pass_.run(expr)
    assert "junk" in _names(pass_.last_result.accepted_variables)
    assert expressions_equivalent(expr, out) is not False


# --------------------------------------------------------------------------- #
# Live retention
# --------------------------------------------------------------------------- #
def test_all_live_variables_are_kept() -> None:
    expr = X + Y + Z
    pass_ = _pass()
    out = pass_.run(expr)
    assert pass_.last_result.candidate_count == 0
    # No dead variable -> the expression comes back untouched.
    assert out == expr
    statuses = {str(r.variable): r.status for r in pass_.last_result.variable_results}
    assert all(s == VariableStatus.LIVE for s in statuses.values())


def test_magic_constant_gate_keeps_variable_live() -> None:
    # output == x when y == 0xDEADBEEF, else x + 1.  ``y`` is live, but only at a
    # single magic value -- harvesting it from the AST is what lets the scan see
    # the dependency. ``y`` must NOT be pruned.
    gate = ExprCond(Y - ExprInt(0xDEADBEEF, SIZE), X + ExprInt(1, SIZE), X)
    pass_ = _pass()
    out = pass_.run(gate)
    assert "y" not in _names(pass_.last_result.candidates)
    assert out == gate
    assert expressions_equivalent(gate, out) is not False


# --------------------------------------------------------------------------- #
# Validation gate (soundness backstop)
# --------------------------------------------------------------------------- #
def test_check_rejects_dropping_a_live_variable() -> None:
    # Directly exercise the gate: x + y is NOT equivalent to x.
    pass_ = _pass()
    ok, counterexample = pass_._check(X + Y, X, [X, Y])
    assert ok is False
    assert counterexample is not None


def test_z3_backstop_rejects_false_positive_without_repair() -> None:
    # ``y`` is live only at a non-power-of-two prime that is NOT in the AST
    # (constants harvesting disabled), so the sampling scan flags it dead. With
    # repair off, the Z3 gate must reject the whole pruning and return the
    # original unchanged.
    prime = 7919
    expr = ExprCond(Y - ExprInt(prime, SIZE), X + ExprInt(1, SIZE), X)
    pass_ = _pass(use_ast_constants=False, repair=False)
    out = pass_.run(expr)
    assert out == expr
    assert pass_.last_result.final_equivalent is False
    assert "y" in _names(pass_.last_result.rejected_variables)


# --------------------------------------------------------------------------- #
# Repair
# --------------------------------------------------------------------------- #
def test_repair_keeps_sound_subset() -> None:
    # Two genuinely dead vars (fake^fake, junk*0) plus a false-positive ``y``
    # (live only at a non-harvested prime). Block-splitting must keep the two
    # dead vars and drop ``y``.
    prime = 7919
    gate = ExprCond(Y - ExprInt(prime, SIZE), X + ExprInt(1, SIZE), X)
    expr = gate + ExprOp("*", FAKE ^ FAKE, X) + ExprOp("*", JUNK, ExprInt(0, SIZE))
    pass_ = _pass(use_ast_constants=False, repair=True)
    out = pass_.run(expr)
    accepted = _names(pass_.last_result.accepted_variables)
    assert accepted == {"fake", "junk"}
    assert "y" in _names(pass_.last_result.rejected_variables)
    assert pass_.last_result.final_equivalent is True
    assert expressions_equivalent(expr, out) is not False
    assert node_count(out) < node_count(expr)


# --------------------------------------------------------------------------- #
# Config / guards
# --------------------------------------------------------------------------- #
def test_disabled_config_is_noop() -> None:
    expr = X + ExprOp("*", FAKE ^ FAKE, Y)
    out = DeadVariableEliminationPass(DeadVariableEliminationConfig(enabled=False)).run(
        expr
    )
    assert out == expr


def test_skips_below_min_variables() -> None:
    # Single variable: nothing to prune relative to "rest"; min_variables=2 skips.
    expr = X + ExprInt(5, SIZE)
    out = _pass().run(expr)
    assert out == expr


def test_skips_above_max_variables() -> None:
    expr = X + ExprOp("*", FAKE ^ FAKE, Y)
    out = _pass(max_variables=1).run(expr)
    assert out == expr


def test_handles_one_bit_variable_without_hanging() -> None:
    # Regression: _mutation_values' random-fill loop spun forever for a small
    # value space -- a 1-bit flag ({0, 1}) can never reach the fill budget, so
    # the scan never terminated. 1-bit flags are ubiquitous in real MBAs. The
    # pass must terminate (here even with the wall-time budget disabled) and
    # stay sound: the live flag is kept, the dead variables pruned.
    flag = ExprId("flag", 1)
    cond = ExprCond(flag, X + ExprInt(1, SIZE), X)  # flag is live
    expr = cond + ExprOp("*", FAKE ^ FAKE, Y)  # fake, y are dead
    pass_ = _pass(time_budget_s=None)
    out = pass_.run(expr)
    assert expressions_equivalent(expr, out) is not False
    assert "flag" not in _names(pass_.last_result.candidates)
    assert _names(pass_.last_result.accepted_variables) == {"fake", "y"}


def test_time_budget_abort_returns_original() -> None:
    # A zero budget forces an immediate abort; the pass must fall back to the
    # original expression soundly rather than partially transforming it.
    expr = X + ExprOp("*", FAKE ^ FAKE, Y) + Z
    out = _pass(time_budget_s=0.0).run(expr)
    assert out == expr


def test_non_compilable_expression_returns_original_without_crash() -> None:
    # An expression the fast evaluator may not support (memory deref) must not
    # crash the pass; it returns the original untouched (or an equivalent).
    mem = ExprMem(X, SIZE)
    expr = mem + ExprOp("*", FAKE ^ FAKE, Y)
    out = _pass().run(expr)
    assert expressions_equivalent(expr, out) is not False


# --------------------------------------------------------------------------- #
# Pipeline-pass contract
# --------------------------------------------------------------------------- #
def test_run_is_a_pure_expr_to_expr_pass() -> None:
    expr = X + ExprOp("*", FAKE ^ FAKE, Y) + Z
    out = _pass().run(expr)
    assert expressions_equivalent(expr, out) is not False
    assert node_count(out) <= node_count(expr)


# --------------------------------------------------------------------------- #
# Simplifier integration
# --------------------------------------------------------------------------- #
def test_simplifier_default_has_no_dead_var_pass() -> None:
    sim = Simplifier(None, pipeline_mode=PipelineMode.SIMBA)
    assert sim._dead_var_pass is None
    assert type(sim.pipeline.passes[0]).__name__ != "DeadVariableEliminationPass"


def test_simplifier_prepends_pass_when_enabled() -> None:
    sim = Simplifier(
        None,
        pipeline_mode=PipelineMode.SIMBA,
        enable_dead_variable_elimination=True,
    )
    assert isinstance(sim.pipeline.passes[0], DeadVariableEliminationPass)
    # The base SIMBA pipeline is preserved after the prepended pass.
    assert [type(p).__name__ for p in sim.pipeline.passes] == [
        "DeadVariableEliminationPass",
        "SimbaPass",
        "AstNormalizationPass",
    ]


def test_simplifier_flag_forces_config_enabled() -> None:
    # Passing a config with enabled=False but the flag on must still enable it.
    sim = Simplifier(
        None,
        enable_dead_variable_elimination=True,
        dead_variable_elimination_config=DeadVariableEliminationConfig(enabled=False),
    )
    assert sim.pipeline.passes[0].config.enabled is True


def test_simplifier_end_to_end_is_equivalent() -> None:
    expr = X + ExprOp("*", FAKE ^ FAKE, Y) + Z
    sim = Simplifier(
        None,
        pipeline_mode=PipelineMode.SIMBA,
        enable_dead_variable_elimination=True,
    )
    out = sim.simplify(expr)
    assert expressions_equivalent(expr, out) is not False
    assert node_count(out) <= node_count(expr)


# --------------------------------------------------------------------------- #
# Soundness battery
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(8))
def test_random_dead_live_mix_is_always_sound(seed: int) -> None:
    # Build expressions over a mix of live and provably-dead variables; the pass
    # output must be equivalent to the input on every seed (0 wrong acceptances).
    rng = random.Random(seed)
    width = 16
    live = [ExprId(f"l{i}", width) for i in range(3)]
    dead = [ExprId(f"d{i}", width) for i in range(3)]

    expr = live[0]
    for var in live[1:]:
        op = rng.choice(["+", "^", "&", "|"])
        expr = ExprOp(op, expr, var)
    # Add provably-dead contributions: var ^ var (==0) and var & 0.
    for var in dead:
        if rng.random() < 0.5:
            expr = expr + ExprOp("*", var ^ var, ExprId("w", width))
        else:
            expr = expr + ExprOp("&", var, ExprInt(0, width))

    pass_ = DeadVariableEliminationPass(
        DeadVariableEliminationConfig(enabled=True, random_seed=seed)
    )
    out = pass_.run(expr)
    assert expressions_equivalent(expr, out) is not False
