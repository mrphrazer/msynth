"""Unit tests for CEGIS's uniform-width handling of mixed-width subtrees.

CEGIS only models a single uniform bit-width. Real-world MBAs mix widths via
slices / composes / comparisons (sub-register accesses, flags). These tests pin:

  * the width predicate ``_is_uniform_width``;
  * the re-unification (``_uniform_unification``) that turns width-changing nodes
    into opaque variables so the surrounding uniform-width arithmetic is solvable;
  * the *handled* end-to-end cases (obfuscated sub-register arithmetic recovered,
    soundly and strictly smaller);
  * the *declined* cases (mixed/bare/over-budget shapes) never crash.
"""

from __future__ import annotations

import random

from miasm.expression.expression import (
    ExprCompose,
    ExprId,
    ExprInt,
    ExprOp,
    ExprSlice,
)

from msynth import Simplifier
from msynth.simplification.cegis import CegisSolver, TemplateOracle
from msynth.utils.expr_utils import get_unique_variables
from msynth.utils.unification import gen_unification_dict, reverse_unification
from scripts.run_simplification_corpus import expressions_equivalent, node_count

SIZE = 64
X0 = ExprId("x0", SIZE)
X1 = ExprId("x1", SIZE)


def _solver(max_variables: int = 3) -> CegisSolver:
    oracle = TemplateOracle.gen_runtime_oracle(num_variables=3)
    return CegisSolver(oracle, max_variables=max_variables)


def _synth(solver: CegisSolver, expr):
    """Call try_synthesize the way the Simplifier does (base-var unification)."""
    udict = gen_unification_dict(expr)
    return solver.try_synthesize(expr, expr.replace_expr(udict), udict)


# --------------------------------------------------------------------------- #
# _is_uniform_width
# --------------------------------------------------------------------------- #


def test_is_uniform_width_accepts_uniform_arithmetic() -> None:
    expr = X0 * ExprInt(5, SIZE) + ExprInt(3, SIZE)
    assert CegisSolver._is_uniform_width(expr, SIZE) is True


def test_is_uniform_width_rejects_slice() -> None:
    expr = ExprSlice(X0, 0, 32) * ExprInt(5, 32) + ExprInt(3, 32)
    assert CegisSolver._is_uniform_width(expr, 32) is False


def test_is_uniform_width_rejects_comparison() -> None:
    # `==` yields a 1-bit result from 64-bit args -> not uniform at width 1.
    assert CegisSolver._is_uniform_width(ExprOp("==", X0, X1), 1) is False


# --------------------------------------------------------------------------- #
# _uniform_unification
# --------------------------------------------------------------------------- #


def test_uniform_unification_atomizes_slice_to_contiguous_var() -> None:
    solver = _solver()
    sliced = ExprSlice(X0, 0, 32)
    expr = sliced * ExprInt(5, 32) + ExprInt(3, 32)  # 32-bit, contains a slice
    result = solver._uniform_unification(expr)
    assert result is not None
    unified, udict = result
    # now uniform, exactly one fresh variable p0, constants left in place
    assert CegisSolver._is_uniform_width(unified, 32) is True
    assert [str(v) for v in get_unique_variables(unified)] == ["p0"]
    # reverse_unification restores the original slice exactly
    assert reverse_unification(unified, udict) == expr


def test_uniform_unification_dedups_repeated_slice() -> None:
    solver = _solver()
    sliced = ExprSlice(X0, 0, 32)
    expr = sliced + sliced  # the same slice twice -> a single variable
    result = solver._uniform_unification(expr)
    assert result is not None
    unified, _ = result
    assert len(get_unique_variables(unified)) == 1


def test_uniform_unification_atomizes_comparison() -> None:
    solver = _solver()
    expr = ExprOp("==", X0, X1)  # width-changing op -> atomized whole
    result = solver._uniform_unification(expr)
    assert result is not None
    unified, _ = result
    assert CegisSolver._is_uniform_width(unified, 1) is True
    assert len(get_unique_variables(unified)) == 1


def test_uniform_unification_returns_none_when_already_uniform() -> None:
    # No width-changer present: nothing for this pass to do (the caller already
    # unified the plain terminals).
    assert _solver()._uniform_unification(X0 + X1) is None


def test_uniform_unification_returns_none_over_max_variables() -> None:
    solver = _solver(max_variables=1)
    expr = ExprSlice(X0, 0, 32) + ExprSlice(X1, 0, 32)  # two distinct slice atoms
    assert solver._uniform_unification(expr) is None


# --------------------------------------------------------------------------- #
# Handled cases (end-to-end via try_synthesize)
# --------------------------------------------------------------------------- #


def test_try_synthesize_recovers_obfuscated_sub_register_affine() -> None:
    solver = _solver()
    sliced = ExprSlice(X0, 0, 32)
    inner = sliced * ExprInt(5, 32)
    k = ExprInt(3, 32)
    # (a ^ k) + 2*(a & k) == a + k, with a = x0[0:32]*5  ->  x0[0:32]*5 + 3
    obfuscated = (inner ^ k) + ExprInt(2, 32) * (inner & k)
    candidate = _synth(solver, obfuscated)
    assert candidate is not None
    assert expressions_equivalent(obfuscated, candidate) is not False
    assert node_count(candidate) < node_count(obfuscated)


def test_simplifier_recovers_sub_register_affine_end_to_end() -> None:
    sliced = ExprSlice(X0, 0, 32)
    inner = sliced * ExprInt(7, 32)
    k = ExprInt(0x11, 32)
    obfuscated = (inner ^ k) + ExprInt(2, 32) * (inner & k)
    out = Simplifier(None, enable_cegis=True).simplify(obfuscated)
    assert expressions_equivalent(obfuscated, out) is not False
    assert node_count(out) < node_count(obfuscated)


def test_try_synthesize_sub_register_is_sound_random() -> None:
    """The recovered candidate must agree with the input on random inputs."""
    sliced = ExprSlice(X0, 0, 16)
    inner = sliced * ExprInt(0x9, 16)
    k = ExprInt(0x55, 16)
    obfuscated = (inner ^ k) + ExprInt(2, 16) * (inner & k)
    candidate = _synth(_solver(), obfuscated)
    assert candidate is not None
    rng = random.Random(7)
    fa = obfuscated
    # exhaustive over the 16-bit slice's source low bits would be huge; sample.
    from msynth.utils.expr_utils import compile_expr_to_python
    from msynth.utils.sampling import _rename_variables_for_compilation

    variables = sorted(
        set(get_unique_variables(fa)) | set(get_unique_variables(candidate)), key=str
    )
    f_in = compile_expr_to_python(_rename_variables_for_compilation(fa, variables))
    f_out = compile_expr_to_python(
        _rename_variables_for_compilation(candidate, variables)
    )
    for _ in range(500):
        row = [rng.getrandbits(64) for _ in variables]
        assert f_in(row) == f_out(row)


# --------------------------------------------------------------------------- #
# Declined cases (must never crash; return None or an equivalent)
# --------------------------------------------------------------------------- #


def test_try_synthesize_bare_slice_no_crash_and_sound() -> None:
    # A bare sub-register read has no useful simplification; CEGIS may return an
    # equivalent-but-not-smaller candidate (e.g. ``x0[0:16] + 0``), which the
    # Simplifier's suitability gate rejects. The contract here is: no crash, and
    # any candidate is a sound equivalent.
    sliced = ExprSlice(X0, 0, 16)
    candidate = _synth(_solver(), sliced)
    if candidate is not None:
        assert expressions_equivalent(sliced, candidate) is not False


def test_try_synthesize_mixed_width_compose_no_crash_and_sound() -> None:
    composed = ExprCompose(ExprSlice(X0, 0, 8), ExprSlice(X1, 8, 64))  # 64-bit
    candidate = _synth(_solver(), composed)  # must not raise the size sanitycheck
    if candidate is not None:
        assert expressions_equivalent(composed, candidate) is not False


def test_simplifier_never_crashes_on_mixed_width_mba() -> None:
    # Pre-fix this raised miasm's "ExprOp args must have same size" from CEGIS's
    # template resizer. It must now simplify (or pass through) without raising.
    mixed = ExprCompose(ExprSlice(X0, 0, 8), ExprInt(0, 56)) * ExprInt(3, SIZE) + (
        X1 ^ ExprInt(0xDEAD, SIZE)
    )
    out = Simplifier(None, enable_cegis=True).simplify(mixed)
    assert expressions_equivalent(mixed, out) is not False
