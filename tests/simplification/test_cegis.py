"""
Tests for the CEGIS constant-synthesis fallback.

The whole point of CEGIS is to handle subtrees the precomputed oracle
cannot. Every Simplifier-level test here therefore uses an **empty**
oracle so that *any* simplification observed must have come from the
CEGIS path. With CEGIS off + empty oracle, the simplifier produces the
input unchanged; with CEGIS on, semantically-equivalent recovery is
expected.
"""
from __future__ import annotations

import random
from pathlib import Path

from miasm.expression.expression import ExprId, ExprInt, ExprOp
from miasm.expression.simplifications import expr_simp

from msynth import Simplifier
from msynth.simplification.cegis import CegisSolver, TemplateOracle
from msynth.simplification.oracle import SimplificationOracle
from msynth.utils.unification import gen_unification_dict


def _eval(expr, variables, values):
    """Concrete-evaluate ``expr`` after substituting ``variables`` -> ``values``."""
    replacements = {
        variables[i]: ExprInt(values[i], variables[i].size)
        for i in range(len(variables))
    }
    return int(expr_simp(expr.replace_expr(replacements)))


def _semantically_equivalent(a, b, variables, *, seed: int, trials: int = 16) -> bool:
    """Random-sample equivalence check; cheap, sufficient for these tests."""
    random.seed(seed)
    size = variables[0].size
    mask = (1 << size) - 1
    for _ in range(trials):
        values = [random.getrandbits(size) & mask for _ in variables]
        if _eval(a, variables, values) != _eval(b, variables, values):
            return False
    return True


def test_cegis_off_by_default(write_empty_oracle) -> None:
    """Regression test: no opt-in -> no CEGIS solver, no behavioural change."""
    s = Simplifier(write_empty_oracle())
    assert s._cegis_solver is None


def test_cegis_recovers_constant_oracle_cannot(write_empty_oracle) -> None:
    """
    The empty oracle has no equivalence class for any subtree. The
    expression `v0 * 0x47 + 0x13` uses arbitrary constants no precomputed
    oracle could cover. CEGIS off -> input unchanged. CEGIS on -> the
    `p0 * c0 + c1` template solves the constants and the simplifier
    accepts an equivalent (semantically identical) candidate.
    """
    size = 8
    v0 = ExprId("v0", size)
    expr = v0 * ExprInt(0x47, size) + ExprInt(0x13, size)

    # Baseline: nothing happens without CEGIS on an empty oracle.
    baseline = Simplifier(write_empty_oracle(num_variables=1), enable_cegis=False)
    out_off = baseline.simplify(expr)
    assert expr_simp(out_off) == expr_simp(expr)

    # CEGIS on: must produce a semantically equivalent expression.
    cegis = Simplifier(
        write_empty_oracle(num_variables=1),
        enable_cegis=True,
        cegis_max_variables=1,
    )
    out_on = cegis.simplify(expr)
    assert _semantically_equivalent(out_on, expr, [v0], seed=1)


def test_cegis_solves_two_variable_template(write_empty_oracle) -> None:
    """
    Two-variable template `(p0 & c0) | (p1 & c1)` is in the runtime
    template set. CEGIS should recover the masks 0x0F and 0xF0 from the
    expression's I/O behaviour even though the oracle is empty.
    """
    size = 8
    v0 = ExprId("v0", size)
    v1 = ExprId("v1", size)
    expr = (v0 & ExprInt(0x0F, size)) | (v1 & ExprInt(0xF0, size))

    s = Simplifier(
        write_empty_oracle(num_variables=2),
        enable_cegis=True,
        cegis_max_variables=2,
    )
    out = s.simplify(expr)
    assert _semantically_equivalent(out, expr, [v0, v1], seed=2)


def test_cegis_refinement_recovers_constant() -> None:
    """
    Initial sample set has only one input ([0]). Z3 sees a single
    constraint and could pick any constant satisfying that one row.
    Validation must find a counter-example; refinement adds it; Z3
    re-solves and converges on the true constant.

    Exercises CegisSolver directly (not via Simplifier) to pin the
    refinement loop semantics independent of BFS scheduling.
    """
    size = 8
    v0 = ExprId("v0", size)
    expr = v0 * ExprInt(5, size)

    unification_dict = gen_unification_dict(expr)
    unified = expr.replace_expr(unification_dict)

    inputs = [[0]]  # deliberately under-constrained
    outputs = [
        SimplificationOracle.evaluate_expression(unified, row) for row in inputs
    ]

    template = ExprId("p0", size) * ExprId("c0", size)
    oracle = TemplateOracle(
        template_bits=size,
        num_variables=1,
        num_samples=len(inputs),
        inputs=inputs,
        oracle_map={},
    )
    oracle.oracle_map[oracle.determine_equiv_key(outputs)] = [template]

    solver = CegisSolver(
        oracle,
        max_templates=3,
        solver_timeout=1,
        max_variables=1,
        refinement_iters=3,
        validation_samples=4,
        expand_templates=False,
        harvest_templates=False,
    )
    candidate = solver.try_synthesize(expr, unified, unification_dict)
    assert candidate is not None
    assert _semantically_equivalent(candidate, expr, [v0], seed=3)


def test_cegis_harvest_reuses_template_within_run(write_empty_oracle) -> None:
    """
    Two structurally identical subtrees with different constants:
        v0 * 0x47 + 0x13  and  v1 * 0x91 + 0x25
    The first triggers full CEGIS + harvest; the second should be able to
    reuse the harvested template (`pX * c0 + c1`) since the template is now
    in the synthetic "*" bucket. We instrument `add_template` to observe
    that at least one harvest happened during the run.
    """
    size = 8
    v0 = ExprId("v0", size)
    v1 = ExprId("v1", size)
    expr = ExprOp(
        "+",
        v0 * ExprInt(0x47, size) + ExprInt(0x13, size),
        v1 * ExprInt(0x91, size) + ExprInt(0x25, size),
    )

    s = Simplifier(
        write_empty_oracle(num_variables=2),
        enable_cegis=True,
        cegis_max_variables=1,
        cegis_harvest_templates=True,
    )

    harvest_counter = {"n": 0}
    real_add = s._cegis_solver.template_oracle.add_template

    def counting(t):
        harvest_counter["n"] += 1
        real_add(t)

    s._cegis_solver.template_oracle.add_template = counting

    out = s.simplify(expr)
    assert _semantically_equivalent(out, expr, [v0, v1], seed=4)
    assert harvest_counter["n"] >= 1


def test_cegis_skips_when_too_many_variables(write_empty_oracle) -> None:
    """
    With ``cegis_max_variables=3`` and a subtree using 4 distinct
    terminals, CEGIS must bail without attempting synthesis. Combined
    with an empty oracle, this leaves the expression unchanged.
    """
    size = 8
    a = ExprId("a", size)
    b = ExprId("b", size)
    c = ExprId("c", size)
    d = ExprId("d", size)
    expr = ExprOp("+", a, b, c, d, ExprInt(0x99, size))

    s = Simplifier(
        write_empty_oracle(num_variables=4),
        enable_cegis=True,
        cegis_max_variables=3,
    )
    out = s.simplify(expr)
    assert expr_simp(out) == expr_simp(expr)
