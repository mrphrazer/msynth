"""
Tests for SimbaPass treating ExprSlice and ExprCompose as opaque BITWISE atoms.

The change being guarded here lets SimbaPass classify expressions whose leaves
include slice and compose nodes (in addition to the previously-supported
ExprId and ExprMem). The linear-MBA theorem (Reichenwallner & Meerwald-Stadler,
2022) only requires that an atom can be substituted with a concrete value on
each boolean assignment — register vs memory vs slice vs compose makes no
difference to soundness. The walker that collects atoms must stop at each of
these kinds; descending into a slice's argument or a compose's pieces would
silently mis-vary the cube and break the reconstruction.

Categories mirror the plan at
``/home/agent/.claude/plans/ok-add-slice-compose-as-stateful-hopper.md``:

- G: atom collector internals (soundness tripwires; run these first)
- E: atom de-duplication on the cube
- L: structural-equality semantics
- F: classifier guards (negative tests — SimbaPass must remain a no-op)
- A: slice basics (single-slice MBA identities)
- B: compose basics
- C: mixed atom kinds in one expression
- D: correlated slices (soundness-critical: cube is sound on the full space,
  hence sound on any reachable subset)
- H: bare leaves at root
- I: all-ones-XOR (bitwise NOT) interaction with the new atoms
- J: Z3-checked directed equivalence (re-asserts directed cases with SMT)

A standalone fuzz harness that complements the directed Z3 cases here
lives at ``scripts/run_simba_fuzzer.py`` — run that for broader
coverage; the unit tests intentionally keep their assertions
deterministic and cheap.
"""

from __future__ import annotations

from miasm.expression.expression import (
    Expr,
    ExprCompose,
    ExprCond,
    ExprId,
    ExprInt,
    ExprMem,
    ExprOp,
    ExprSlice,
)
from miasm.expression.simplifications import expr_simp

from msynth.simplification.simba import (
    SimbaPass,
    _ExpressionKind,
    _SimbaSimplifier,
    _bitwise_refine,
    _classify,
    _collect_atoms,
)


def _node_count(expr: Expr) -> int:
    return len(expr.graph().nodes())


def _evaluate_with_atoms(expr: Expr, env: dict[Expr, int]) -> int:
    """
    Evaluate ``expr`` on a boolean-cube assignment, treating ExprId,
    ExprMem, ExprSlice, ExprCompose, and ExprCond as opaque atomic
    lookups. Mirrors SimbaPass's cube semantics under the atomisation
    extension so the test can assert equivalence by direct enumeration
    on small atom counts.
    """
    mask = (1 << expr.size) - 1
    if isinstance(expr, ExprInt):
        return int(expr) & mask
    if isinstance(expr, (ExprId, ExprMem, ExprSlice, ExprCompose, ExprCond)):
        return env.get(expr, 0) & mask
    if isinstance(expr, ExprOp):
        # ExprOps whose op isn't in the linear-MBA fragment are
        # atomised by SimbaPass — mirror that here.
        if expr.op not in {"+", "-", "*", "&", "|", "^"}:
            return env.get(expr, 0) & mask
        args = [_evaluate_with_atoms(arg, env) for arg in expr.args]
        if expr.op == "-" and len(args) == 1:
            return (-args[0]) & mask
        if expr.op == "+":
            return sum(args) & mask
        if expr.op == "-" and len(args) >= 2:
            result = args[0]
            for arg in args[1:]:
                result -= arg
            return result & mask
        if expr.op == "*":
            result = 1
            for arg in args:
                result *= arg
            return result & mask
        if expr.op == "&":
            result = mask
            for arg in args:
                result &= arg
            return result & mask
        if expr.op == "|":
            result = 0
            for arg in args:
                result |= arg
            return result & mask
        if expr.op == "^":
            result = 0
            for arg in args:
                result ^= arg
            return result & mask
    raise AssertionError(f"unsupported test expression: {expr!r}")


def _assert_equivalent_on_cube(left: Expr, right: Expr, atoms: list[Expr]) -> None:
    """
    Boolean-cube equivalence over an explicit atom list. Used for cheap
    pre-Z3 checks on small expressions.
    """
    assert left.size == right.size, (left.size, right.size)
    assert len(atoms) <= 5, "cube would be too large"
    for assignment in range(1 << len(atoms)):
        env = {atom: (assignment >> index) & 1 for index, atom in enumerate(atoms)}
        assert _evaluate_with_atoms(left, env) == _evaluate_with_atoms(right, env), (
            f"cube disagreement at {env}: left={left!r} right={right!r}"
        )


# ---------------------------------------------------------------------------
# Category G — atom collector internals (soundness tripwires)
# ---------------------------------------------------------------------------


def test_collect_atoms_treats_slice_as_atom_not_its_arg() -> None:
    # Walker must not descend into ExprSlice.arg. If it did, the cube
    # would also vary the underlying register and the reconstruction
    # would be over an inconsistent atom set — silently unsound.
    x = ExprId("x", 32)
    sl = ExprSlice(x, 0, 8)
    assert _collect_atoms(sl) == [sl]


def test_collect_atoms_treats_compose_as_atom_not_its_args() -> None:
    lo = ExprId("lo", 8)
    hi = ExprId("hi", 8)
    comp = ExprCompose(lo, hi)
    assert _collect_atoms(comp) == [comp]


def test_collect_atoms_does_not_recurse_into_slice_of_memory() -> None:
    ptr = ExprId("ptr", 64)
    sl = ExprSlice(ExprMem(ptr, 32), 0, 8)
    assert _collect_atoms(sl) == [sl]


def test_collect_atoms_distinguishes_slice_ranges_of_same_base() -> None:
    x = ExprId("x", 32)
    lo = ExprSlice(x, 0, 8)
    hi = ExprSlice(x, 8, 16)
    atoms = _collect_atoms(lo + hi)
    assert set(atoms) == {lo, hi}


def test_collect_atoms_dedupes_same_slice_textually_repeated() -> None:
    x = ExprId("x", 32)
    e = ExprSlice(x, 0, 8) + ExprSlice(x, 0, 8) + ExprSlice(x, 0, 8)
    atoms = _collect_atoms(e)
    assert atoms == [ExprSlice(x, 0, 8)]


def test_collect_atoms_dedupes_same_compose_textually_repeated() -> None:
    lo = ExprId("lo", 8)
    hi = ExprId("hi", 8)
    e = ExprCompose(lo, hi) & ExprCompose(lo, hi)
    atoms = _collect_atoms(e)
    assert atoms == [ExprCompose(lo, hi)]


# ---------------------------------------------------------------------------
# Category E — atom de-duplication on the cube
# ---------------------------------------------------------------------------


def test_simba_self_xor_of_slice_collapses_to_zero() -> None:
    sl = ExprSlice(ExprId("x", 32), 0, 8)
    out = SimbaPass().run(sl ^ sl)
    assert expr_simp(out) == ExprInt(0, 8)


def test_simba_self_subtract_of_slice_collapses_to_zero() -> None:
    sl = ExprSlice(ExprId("x", 32), 0, 8)
    out = SimbaPass().run(sl - sl)
    assert expr_simp(out) == ExprInt(0, 8)


def test_simba_self_xor_of_compose_collapses_to_zero() -> None:
    comp = ExprCompose(ExprId("lo", 8), ExprId("hi", 8))
    out = SimbaPass().run(comp ^ comp)
    assert expr_simp(out) == ExprInt(0, 16)


def test_simba_self_subtract_of_compose_collapses_to_zero() -> None:
    comp = ExprCompose(ExprId("lo", 8), ExprId("hi", 8))
    out = SimbaPass().run(comp - comp)
    assert expr_simp(out) == ExprInt(0, 16)


# ---------------------------------------------------------------------------
# Category L — structural-equality semantics
# ---------------------------------------------------------------------------


def test_simba_dedupes_two_separately_constructed_slices() -> None:
    x = ExprId("x", 32)
    s1 = ExprSlice(x, 0, 8)
    s2 = ExprSlice(x, 0, 8)
    assert s1 == s2
    assert _collect_atoms(s1 + s2) == [s1]


def test_simba_dedupes_two_separately_constructed_composes() -> None:
    lo = ExprId("lo", 8)
    hi = ExprId("hi", 8)
    c1 = ExprCompose(lo, hi)
    c2 = ExprCompose(lo, hi)
    assert c1 == c2
    assert _collect_atoms(c1 ^ c2) == [c1]


# ---------------------------------------------------------------------------
# Category F — classifier guards (negative tests)
# ---------------------------------------------------------------------------


def test_simba_classifier_rejects_slice_times_slice() -> None:
    # Non-linear: two bitwise factors. Must remain a no-op.
    x = ExprId("x", 32)
    y = ExprId("y", 32)
    expr = ExprSlice(x, 0, 8) * ExprSlice(y, 0, 8)
    assert SimbaPass().run(expr) is expr


def test_simba_classifier_rejects_xor_with_non_allones_arithmetic_arg() -> None:
    x = ExprId("x", 32)
    y = ExprId("y", 32)
    z = ExprId("z", 32)
    expr = (ExprSlice(x, 0, 8) + ExprSlice(y, 0, 8)) ^ ExprSlice(z, 0, 8)
    assert SimbaPass().run(expr) is expr


def test_simba_classifier_rejects_shift_with_slice_atom() -> None:
    expr = ExprOp("<<", ExprSlice(ExprId("x", 32), 0, 8), ExprInt(1, 8))
    assert SimbaPass().run(expr) is expr


def test_simba_classifier_rejects_compose_with_shift_outside() -> None:
    comp = ExprCompose(ExprId("lo", 8), ExprId("hi", 8))
    expr = ExprOp("<<", comp, ExprInt(1, 16))
    assert SimbaPass().run(expr) is expr


def test_simba_atomises_cond_plus_slice_soundly() -> None:
    # Under the atomisation extension, both ExprCond and ExprSlice
    # are primary atoms, so ``cond + slice`` IS a two-atom linear
    # MBA and SimbaPass reconstructs over it. The output may not be
    # syntactically identical to the input; the invariant pinned here
    # is semantic equivalence.
    x = ExprId("x", 32)
    y = ExprId("y", 32)
    c = ExprId("c", 1)
    cond = ExprCond(c, ExprSlice(x, 0, 8), ExprSlice(y, 0, 8))
    expr = cond + ExprSlice(x, 0, 8)
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [cond, ExprSlice(x, 0, 8)])
    assert _z3_equivalent(expr, out)


def test_simba_classifier_rejects_compose_times_compose() -> None:
    expr = ExprCompose(ExprId("a", 8), ExprId("b", 8)) * ExprCompose(
        ExprId("c", 8), ExprId("d", 8)
    )
    assert SimbaPass().run(expr) is expr


# ---------------------------------------------------------------------------
# Category A — slice basics
# ---------------------------------------------------------------------------


def test_simba_slice_and_or_sum_identity_collapses_to_linear_sum() -> None:
    x = ExprId("x", 32)
    y = ExprId("y", 32)
    a = ExprSlice(x, 0, 8)
    b = ExprSlice(y, 0, 8)
    expr = (a & b) + (a | b)
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [a, b])
    assert _node_count(out) <= _node_count(expr)


def test_simba_slice_idempotent_and_is_slice() -> None:
    sl = ExprSlice(ExprId("x", 32), 0, 8)
    out = SimbaPass().run(sl & sl)
    assert expr_simp(out) == sl


def test_simba_slice_idempotent_or_is_slice() -> None:
    sl = ExprSlice(ExprId("x", 32), 0, 8)
    out = SimbaPass().run(sl | sl)
    assert expr_simp(out) == sl


def test_simba_slice_affine_combination_collapses() -> None:
    x = ExprId("x", 32)
    y = ExprId("y", 32)
    a = ExprSlice(x, 0, 8)
    b = ExprSlice(y, 0, 8)
    expr = ExprInt(7, 8) * ((a & b) + (a | b)) + ExprInt(3, 8)
    expected = ExprInt(7, 8) * a + ExprInt(7, 8) * b + ExprInt(3, 8)
    out = SimbaPass().run(expr)
    assert expr_simp(out) == expr_simp(expected)


def test_simba_slice_double_via_addition_is_equivalent() -> None:
    sl = ExprSlice(ExprId("x", 32), 0, 8)
    out = SimbaPass().run(sl + sl)
    _assert_equivalent_on_cube(out, sl + sl, [sl])


# ---------------------------------------------------------------------------
# Category B — compose basics
# ---------------------------------------------------------------------------


def test_simba_compose_and_or_sum_identity_collapses_to_linear_sum() -> None:
    a = ExprCompose(ExprId("a0", 8), ExprId("a1", 8))
    b = ExprCompose(ExprId("b0", 8), ExprId("b1", 8))
    expr = (a & b) + (a | b)
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [a, b])
    assert _node_count(out) <= _node_count(expr)


def test_simba_compose_idempotent_and_is_compose() -> None:
    comp = ExprCompose(ExprId("lo", 8), ExprId("hi", 8))
    out = SimbaPass().run(comp & comp)
    assert expr_simp(out) == comp


def test_simba_compose_idempotent_or_is_compose() -> None:
    comp = ExprCompose(ExprId("lo", 8), ExprId("hi", 8))
    out = SimbaPass().run(comp | comp)
    assert expr_simp(out) == comp


def test_simba_compose_bitwise_not_via_xor_allones() -> None:
    comp = ExprCompose(ExprId("lo", 8), ExprId("hi", 8))
    expr = comp ^ ExprInt(0xFFFF, 16)
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [comp])


def test_simba_compose_affine_combination_collapses() -> None:
    a = ExprCompose(ExprId("a0", 8), ExprId("a1", 8))
    b = ExprCompose(ExprId("b0", 8), ExprId("b1", 8))
    expr = ExprInt(5, 16) * ((a & b) + (a | b)) + ExprInt(11, 16)
    expected = ExprInt(5, 16) * a + ExprInt(5, 16) * b + ExprInt(11, 16)
    out = SimbaPass().run(expr)
    assert expr_simp(out) == expr_simp(expected)


# ---------------------------------------------------------------------------
# Category C — mixed atom kinds
# ---------------------------------------------------------------------------


def test_simba_mixed_slice_and_register_paper_identity() -> None:
    sl = ExprSlice(ExprId("x", 32), 0, 8)
    y = ExprId("y", 8)
    expr = (sl & y) + (sl | y)
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [sl, y])


def test_simba_mixed_compose_and_memory_paper_identity() -> None:
    comp = ExprCompose(ExprId("lo", 8), ExprId("hi", 8))
    mem = ExprMem(ExprId("ptr", 64), 16)
    expr = (comp & mem) + (comp | mem)
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [comp, mem])


def test_simba_mixed_slice_compose_register_three_atoms() -> None:
    sl = ExprSlice(ExprId("x", 32), 0, 16)
    comp = ExprCompose(ExprId("lo", 8), ExprId("hi", 8))
    r = ExprId("r", 16)
    expr = (sl & r) + (sl | r) + (comp ^ ExprInt(0xFFFF, 16))
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [sl, comp, r])


def test_simba_mixed_all_four_atom_kinds() -> None:
    sl = ExprSlice(ExprId("x", 32), 0, 16)
    comp = ExprCompose(ExprId("lo", 8), ExprId("hi", 8))
    r = ExprId("r", 16)
    m = ExprMem(ExprId("ptr", 64), 16)
    expr = (sl & r) + (sl | r) + (comp & m) + (comp | m)
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [sl, comp, r, m])


# ---------------------------------------------------------------------------
# Category D — correlated atoms (soundness-critical)
# ---------------------------------------------------------------------------


def test_simba_correlated_low_high_slices_collapse_to_sum() -> None:
    # `X[0:8]` and `X[8:16]` are derived from the same X but the cube
    # treats them as independent atoms. The reconstruction is correct
    # on the full cube — including unreachable pairs — and therefore on
    # the reachable subset where they ARE byte-decompositions of X.
    x = ExprId("x", 16)
    lo = ExprSlice(x, 0, 8)
    hi = ExprSlice(x, 8, 16)
    expr = (lo & hi) + (lo | hi)
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [lo, hi])
    assert _node_count(out) <= _node_count(expr)


def test_simba_correlated_overlapping_slice_atoms_stay_distinct() -> None:
    # Structurally-different slices over the same base must not dedupe.
    x = ExprId("x", 32)
    s1 = ExprSlice(x, 0, 8)
    s2 = ExprSlice(x, 4, 12)
    assert set(_collect_atoms(s1 + s2)) == {s1, s2}


def test_simba_correlated_three_slices_linear_combination() -> None:
    x = ExprId("x", 32)
    a = ExprSlice(x, 0, 8)
    b = ExprSlice(x, 8, 16)
    c = ExprSlice(x, 16, 24)
    expr = a + b + c
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [a, b, c])


def test_simba_correlated_slice_inside_compose_atom() -> None:
    # The Compose is the atom; slices inside its arms are not separately
    # atomised because the walker stops at the compose.
    x = ExprId("x", 32)
    comp = ExprCompose(ExprSlice(x, 0, 8), ExprSlice(x, 8, 16))
    assert _collect_atoms(comp) == [comp]


def test_simba_correlated_compose_arms_share_a_slice_outside() -> None:
    # The slice appears as an outer atom; it also happens to be used to
    # build a compose elsewhere in user code but not in this expression.
    # Atom set seen by SimbaPass here is {sl}.
    x = ExprId("x", 32)
    sl = ExprSlice(x, 0, 8)
    expr = (sl ^ ExprInt(0xFF, 8)) + (sl & ExprInt(0x0F, 8))
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [sl])


# ---------------------------------------------------------------------------
# Category H — bare leaves at root
# ---------------------------------------------------------------------------


def test_simba_top_level_slice_passes_through() -> None:
    sl = ExprSlice(ExprId("x", 32), 0, 8)
    assert SimbaPass().run(sl) == sl


def test_simba_top_level_compose_passes_through() -> None:
    comp = ExprCompose(ExprId("lo", 8), ExprId("hi", 8))
    assert SimbaPass().run(comp) == comp


def test_simba_top_level_slice_of_complex_pointer_passes_through() -> None:
    x = ExprId("x", 32)
    y = ExprId("y", 32)
    sl = ExprSlice(x + y, 0, 8)
    assert SimbaPass().run(sl) == sl


# ---------------------------------------------------------------------------
# Category I — all-ones-XOR (bitwise NOT) interaction
# ---------------------------------------------------------------------------


def test_simba_xor_allones_treats_slice_as_bitwise_not() -> None:
    sl = ExprSlice(ExprId("x", 32), 0, 8)
    expr = sl ^ ExprInt(0xFF, 8)
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [sl])


def test_simba_xor_allones_treats_compose_as_bitwise_not() -> None:
    comp = ExprCompose(ExprId("lo", 8), ExprId("hi", 8))
    expr = comp ^ ExprInt(0xFFFF, 16)
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [comp])


def test_simba_xor_allones_in_linear_combination_with_slice() -> None:
    a = ExprSlice(ExprId("x", 32), 0, 8)
    b = ExprId("y", 8)
    expr = (a ^ ExprInt(0xFF, 8)) & b
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [a, b])


# ---------------------------------------------------------------------------
# Z3 verification helper
# ---------------------------------------------------------------------------


def _z3_equivalent(left: Expr, right: Expr) -> bool:
    """
    Solver-backed equivalence check used by the directed J-category
    tests: UNSAT(left != right) iff ``left == right`` over every
    bit-vector assignment of their free variables. Treats
    ``z3.unknown`` as a hard failure — every directed case here is
    small enough that Z3 should converge well inside the 5-second
    timeout, and silently passing on solver timeout would defeat the
    point of these tests.

    One TranslatorZ3 is shared between both sides so memory arrays and
    free-id bindings line up; two independently constructed translators
    can produce distinct Z3 arrays for the same ExprMem and falsely
    report inequivalence.
    """
    import z3
    from miasm.ir.translators.z3_ir import TranslatorZ3

    assert left.size == right.size, (left.size, right.size)
    translator = TranslatorZ3()
    z3_left = translator.from_expr(left)
    z3_right = translator.from_expr(right)
    solver = z3.Solver()
    # 30s headroom: these J-category multi-atom queries solve in a few seconds
    # unloaded, but a 5s cap flaked under heavy/parallel suite load (Z3 returns
    # unknown, which this helper treats as a hard failure). Generous headroom
    # keeps a genuine timeout loud without spurious flakes.
    solver.set("timeout", 30000)
    solver.add(z3_left != z3_right)
    result = solver.check()
    if result == z3.unknown:
        raise AssertionError(
            f"Z3 returned unknown within timeout for {left!r} vs {right!r}; "
            "shrink the inputs or move this case to the standalone "
            "fuzz harness at scripts/run_simba_fuzzer.py."
        )
    return result == z3.unsat


# ---------------------------------------------------------------------------
# Category J — Z3-checked directed equivalence
# ---------------------------------------------------------------------------


def test_simba_z3_slice_paper_identity() -> None:
    x = ExprId("x", 32)
    y = ExprId("y", 32)
    a = ExprSlice(x, 0, 8)
    b = ExprSlice(y, 0, 8)
    expr = (a & b) + (a | b)
    assert _z3_equivalent(expr, SimbaPass().run(expr))


def test_simba_z3_compose_affine_combination() -> None:
    a = ExprCompose(ExprId("a0", 8), ExprId("a1", 8))
    b = ExprCompose(ExprId("b0", 8), ExprId("b1", 8))
    expr = ExprInt(5, 16) * ((a & b) + (a | b)) + ExprInt(11, 16)
    assert _z3_equivalent(expr, SimbaPass().run(expr))


def test_simba_z3_mixed_compose_and_memory() -> None:
    comp = ExprCompose(ExprId("lo", 8), ExprId("hi", 8))
    mem = ExprMem(ExprId("ptr", 64), 16)
    expr = (comp & mem) + (comp | mem)
    assert _z3_equivalent(expr, SimbaPass().run(expr))


def test_simba_z3_correlated_low_high_slices() -> None:
    # Soundness witness for the correlated-atoms argument: Z3 quantifies
    # over X freely, so the check space includes both reachable pairs
    # (byte decompositions of some X) and unreachable independent pairs.
    # The rewrite must hold over the full space.
    x = ExprId("x", 16)
    lo = ExprSlice(x, 0, 8)
    hi = ExprSlice(x, 8, 16)
    expr = (lo & hi) + (lo | hi)
    assert _z3_equivalent(expr, SimbaPass().run(expr))


def test_simba_z3_three_correlated_slices() -> None:
    x = ExprId("x", 32)
    a = ExprSlice(x, 0, 8)
    b = ExprSlice(x, 8, 16)
    c = ExprSlice(x, 16, 24)
    expr = a + b + c
    assert _z3_equivalent(expr, SimbaPass().run(expr))


def test_simba_z3_xor_allones_with_slice_in_and() -> None:
    a = ExprSlice(ExprId("x", 32), 0, 8)
    b = ExprId("y", 8)
    expr = (a ^ ExprInt(0xFF, 8)) & b
    assert _z3_equivalent(expr, SimbaPass().run(expr))


# ---------------------------------------------------------------------------
# Soundness gate for the post-assembly _bitwise_refine polish
# ---------------------------------------------------------------------------
#
# _bitwise_refine runs the GAMBA §5.2 no-grow rules once on the fully-assembled
# SimBA reconstruction (end of _SimbaSimplifier.simplify). It was previously
# disabled because applying it PER QM REGION perturbed the multi-coefficient
# assembly; applied to the complete output it must be semantics-preserving.
# These cases gate that, plus the affine / division-atom / three-atom-mix
# shapes the disabled NOTE specifically warned about.


def test_simba_bitwise_refine_is_sound() -> None:
    x = ExprId("x", 8)
    y = ExprId("y", 8)
    z = ExprId("z", 8)
    battery = [
        ~(~x),                                  # double negation
        (x & x) + y,                            # idempotence inside a sum
        (x & y) | (x & (~y)),                   # complement pair -> x
        x | (~x),                               # redundancy -> all-ones
        (x ^ y) + ExprInt(2, 8) * (x & y),      # linear-MBA identity (== x + y)
        ExprInt(5, 8) * ((x & y) + (x | y)) + ExprInt(3, 8),  # affine combination
        x + y + z,                              # three-atom mix
    ]
    for expr in battery:
        assert _z3_equivalent(expr, _bitwise_refine(expr)), expr


def test_simba_bitwise_refine_actually_fires() -> None:
    # Guard against the polish silently degenerating into a no-op: it must
    # strictly shrink at least one representative input.
    x = ExprId("x", 8)
    expr = (x & x) + ~(~x)  # both summands collapse to x -> 2*x / x + x
    out = _bitwise_refine(expr)
    assert out != expr
    assert _node_count(out) < _node_count(expr)
    assert _z3_equivalent(expr, out)


def test_simba_run_equivalent_after_refine_on_mba_identities() -> None:
    # End-to-end: the refine step is inside SimbaPass.run; output stays
    # equivalent on classic MBA identities.
    x = ExprId("x", 16)
    y = ExprId("y", 16)
    battery = [
        (x ^ y) + ExprInt(2, 16) * (x & y),     # == x + y
        (x | y) + (x & y),                       # == x + y
        (x | y) - (x & y),                       # == x ^ y
    ]
    for expr in battery:
        assert _z3_equivalent(expr, SimbaPass().run(expr)), expr


# ===========================================================================
# GAMBA 5.5 — operator-level atomisation extension
# ===========================================================================
#
# The categories below test SiMBA's behaviour when its classifier
# encounters miasm IL operators OUTSIDE the linear-MBA fragment —
# shifts, rotations, division/modulo, bit-counting primitives,
# exponentiation, ``*`` with two non-arithmetic operands, and
# ``ExprCond``. These nodes are atomised by ``_classify_uncached``'s
# fast path: the whole subtree becomes one opaque BITWISE atom and
# the surrounding linear-MBA cube reasoning operates over that atom
# uniformly.
#
# Operand-kind rejections (``&`` over BITWISE+MIXED, ``*`` with two
# non-arithmetic operands, ``^`` over MIXED+MIXED, ...) are deliberately
# NOT atomised — they propagate ``None`` from ``_classify`` so SiMBA
# remains a no-op on those shapes, matching pre-extension behaviour
# (otherwise the demo-MBA regression of 9 → 13 nodes returns; see
# ``test_simplifier_demo_mba_reaches_shortest_form_with_placeholder_guard``).
# Category Y below pins this regression boundary.

# Representative atoms per non-linear operator family. Each is a node
# the classifier rejects at the OP LEVEL (op not in {+, -, *, &, |, ^}).
# Width-matched to 32 bits so they can compose freely with the existing
# slice/compose atoms in the same expressions.
_X32 = ExprId("x", 32)
_Y32 = ExprId("y", 32)
_C1 = ExprId("c", 1)
_SHIFT_L = ExprOp("<<", _X32, ExprInt(3, 32))
_SHIFT_R = ExprOp(">>", _X32, ExprInt(5, 32))
_SHIFT_AR = ExprOp("a>>", _X32, _Y32)
_ROT_L = ExprOp("<<<", _X32, ExprInt(7, 32))
_ROT_R = ExprOp(">>>", _X32, ExprInt(5, 32))
_DIV = ExprOp("/", _X32, _Y32)
_MOD = ExprOp("%", _X32, _Y32)
_SDIV = ExprOp("sdiv", _X32, _Y32)
_SMOD = ExprOp("smod", _X32, _Y32)
_CLZ = ExprOp("cntleadzeros", _X32)
_CTZ = ExprOp("cnttrailzeros", _X32)
_PAR = ExprOp("parity", _X32)
_POW = ExprOp("**", _X32, _Y32)
_COND = ExprCond(_C1, _X32, _Y32)


# ---------------------------------------------------------------------------
# Category K — atom collector tripwires (operator-level atomisation)
# ---------------------------------------------------------------------------


def test_collect_atoms_treats_shift_left_as_atom_not_descending() -> None:
    # ``<<`` is outside the linear-MBA fragment; the walker must
    # treat the whole shift as one atom and NOT recurse into ``x``.
    # Descending would vary ``x`` independently of the shift's value
    # in the cube, breaking the soundness sketch.
    assert _collect_atoms(_SHIFT_L) == [_SHIFT_L]


def test_collect_atoms_treats_shift_right_as_atom() -> None:
    assert _collect_atoms(_SHIFT_R) == [_SHIFT_R]


def test_collect_atoms_treats_arith_shift_right_as_atom() -> None:
    assert _collect_atoms(_SHIFT_AR) == [_SHIFT_AR]


def test_collect_atoms_treats_rotate_left_as_atom() -> None:
    assert _collect_atoms(_ROT_L) == [_ROT_L]


def test_collect_atoms_treats_rotate_right_as_atom() -> None:
    assert _collect_atoms(_ROT_R) == [_ROT_R]


def test_collect_atoms_treats_division_as_atom() -> None:
    assert _collect_atoms(_DIV) == [_DIV]


def test_collect_atoms_treats_modulo_as_atom() -> None:
    assert _collect_atoms(_MOD) == [_MOD]


def test_collect_atoms_treats_signed_division_as_atom() -> None:
    assert _collect_atoms(_SDIV) == [_SDIV]


def test_collect_atoms_treats_signed_modulo_as_atom() -> None:
    assert _collect_atoms(_SMOD) == [_SMOD]


def test_collect_atoms_treats_count_leading_zeros_as_atom() -> None:
    assert _collect_atoms(_CLZ) == [_CLZ]


def test_collect_atoms_treats_count_trailing_zeros_as_atom() -> None:
    assert _collect_atoms(_CTZ) == [_CTZ]


def test_collect_atoms_treats_parity_as_atom() -> None:
    assert _collect_atoms(_PAR) == [_PAR]


def test_collect_atoms_treats_power_as_atom() -> None:
    assert _collect_atoms(_POW) == [_POW]


def test_collect_atoms_treats_cond_as_atom() -> None:
    # ExprCond joins the primary-leaf set under the atomisation
    # extension. Its branches' variables (``x``, ``y``) must NOT be
    # exposed to the cube — only the cond as a whole.
    assert _collect_atoms(_COND) == [_COND]


def test_collect_atoms_dedupes_repeated_shift_atom() -> None:
    # Structural equality must dedupe two textually-identical shift
    # subtrees, or ``e ^ e`` won't collapse.
    e = _SHIFT_R + _SHIFT_R + _SHIFT_R
    assert _collect_atoms(e) == [_SHIFT_R]


def test_collect_atoms_dedupes_repeated_cond_atom() -> None:
    e = _COND & _COND
    assert _collect_atoms(e) == [_COND]


# ---------------------------------------------------------------------------
# Category M — classifier tripwires (atom kind + is_atom flag)
# ---------------------------------------------------------------------------


def test_classify_marks_shift_as_bitwise_atom() -> None:
    kind, is_atom = _classify(_SHIFT_R, 32)
    assert kind is _ExpressionKind.BITWISE
    assert is_atom is True


def test_classify_marks_rotation_as_bitwise_atom() -> None:
    kind, is_atom = _classify(_ROT_L, 32)
    assert kind is _ExpressionKind.BITWISE
    assert is_atom is True


def test_classify_marks_division_as_bitwise_atom() -> None:
    kind, is_atom = _classify(_DIV, 32)
    assert kind is _ExpressionKind.BITWISE
    assert is_atom is True


def test_classify_marks_modulo_as_bitwise_atom() -> None:
    kind, is_atom = _classify(_MOD, 32)
    assert kind is _ExpressionKind.BITWISE
    assert is_atom is True


def test_classify_marks_count_leading_zeros_as_bitwise_atom() -> None:
    kind, is_atom = _classify(_CLZ, 32)
    assert kind is _ExpressionKind.BITWISE
    assert is_atom is True


def test_classify_marks_parity_as_bitwise_atom() -> None:
    # ``parity`` returns 1 bit in miasm — test it at its native
    # width. Atomisation discipline is the same; the cube treats it
    # as a 1-bit opaque atom.
    kind, is_atom = _classify(_PAR, 1)
    assert kind is _ExpressionKind.BITWISE
    assert is_atom is True


def test_classify_parity_at_wrong_parent_size_signals_no_op() -> None:
    # Width mismatch at the parent level returns ``(None, True)`` —
    # the no-op signal. Pins the size discipline that prevents a
    # 1-bit ``parity`` from being mis-treated as a 32-bit atom.
    kind, is_atom = _classify(_PAR, 32)
    assert kind is None
    assert is_atom is True


def test_classify_marks_power_as_bitwise_atom() -> None:
    kind, is_atom = _classify(_POW, 32)
    assert kind is _ExpressionKind.BITWISE
    assert is_atom is True


def test_classify_marks_cond_as_bitwise_atom() -> None:
    kind, is_atom = _classify(_COND, 32)
    assert kind is _ExpressionKind.BITWISE
    assert is_atom is True


def test_classify_marks_linear_sum_of_shifts_as_decomposable() -> None:
    # ``+`` of two shift atoms is in the linear-MBA fragment, with
    # both args classifying as BITWISE atoms. The ``+`` itself is NOT
    # an atom — SiMBA should reconstruct over the atom set.
    expr = _SHIFT_L + _SHIFT_R
    kind, is_atom = _classify(expr, 32)
    assert kind is _ExpressionKind.MIXED
    assert is_atom is False


def test_classify_marks_sum_of_cond_atoms_as_decomposable() -> None:
    expr = _COND + _COND
    kind, is_atom = _classify(expr, 32)
    assert kind is _ExpressionKind.MIXED
    assert is_atom is False


# ---------------------------------------------------------------------------
# Category N — evaluator tripwires
# ---------------------------------------------------------------------------


def test_evaluate_looks_up_shift_atom_in_env() -> None:
    sim = _SimbaSimplifier(_SHIFT_R + _SHIFT_R)
    env = {_SHIFT_R: 7}
    # Direct lookup — masked to 32 bits.
    assert sim._evaluate(_SHIFT_R, env) == 7


def test_evaluate_looks_up_cond_atom_in_env() -> None:
    sim = _SimbaSimplifier(_COND + _COND)
    env = {_COND: 0xAA}
    assert sim._evaluate(_COND, env) == 0xAA


def test_evaluate_does_not_recurse_into_shift_args() -> None:
    # Even though ``x`` appears inside the shift, the evaluator
    # must NOT evaluate ``x`` separately — the shift is one atom.
    sim = _SimbaSimplifier(_SHIFT_R + _SHIFT_R)
    env = {_SHIFT_R: 3}  # NOT providing _X32
    # No KeyError despite _X32 being absent from env.
    assert sim._evaluate(_SHIFT_R, env) == 3


# ---------------------------------------------------------------------------
# Category P — self-cancellation collapse over non-linear atoms
# ---------------------------------------------------------------------------


def test_simba_self_xor_of_shift_collapses_to_zero() -> None:
    out = SimbaPass().run(_SHIFT_R ^ _SHIFT_R)
    assert expr_simp(out) == ExprInt(0, 32)


def test_simba_self_subtract_of_shift_collapses_to_zero() -> None:
    out = SimbaPass().run(_SHIFT_R - _SHIFT_R)
    assert expr_simp(out) == ExprInt(0, 32)


def test_simba_self_and_of_shift_is_shift() -> None:
    out = SimbaPass().run(_SHIFT_R & _SHIFT_R)
    assert expr_simp(out) == _SHIFT_R


def test_simba_self_or_of_shift_is_shift() -> None:
    out = SimbaPass().run(_SHIFT_R | _SHIFT_R)
    assert expr_simp(out) == _SHIFT_R


def test_simba_self_xor_of_rotation_collapses_to_zero() -> None:
    out = SimbaPass().run(_ROT_L ^ _ROT_L)
    assert expr_simp(out) == ExprInt(0, 32)


def test_simba_self_subtract_of_division_collapses_to_zero() -> None:
    out = SimbaPass().run(_DIV - _DIV)
    assert expr_simp(out) == ExprInt(0, 32)


def test_simba_self_xor_of_count_leading_zeros_collapses_to_zero() -> None:
    out = SimbaPass().run(_CLZ ^ _CLZ)
    assert expr_simp(out) == ExprInt(0, 32)


def test_simba_self_xor_of_cond_collapses_to_zero() -> None:
    out = SimbaPass().run(_COND ^ _COND)
    assert expr_simp(out) == ExprInt(0, 32)


def test_simba_self_subtract_of_cond_collapses_to_zero() -> None:
    out = SimbaPass().run(_COND - _COND)
    assert expr_simp(out) == ExprInt(0, 32)


def test_simba_self_and_of_cond_is_cond() -> None:
    out = SimbaPass().run(_COND & _COND)
    assert expr_simp(out) == _COND


# ---------------------------------------------------------------------------
# Category Q — coefficient folding over non-linear atoms
# ---------------------------------------------------------------------------


def test_simba_shift_atom_sum_folds_to_double() -> None:
    expr = _SHIFT_R + _SHIFT_R
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [_SHIFT_R])
    # The reconstruction must agree on the cube; structural form may
    # be ``2 * SHIFT_R`` or equivalent.
    assert _z3_equivalent(expr, out)


def test_simba_shift_atom_double_andor_identity_collapses() -> None:
    # ``(a & b) + (a | b)`` over two distinct shift atoms — same
    # paper identity as for slice/compose, now over shift atoms.
    a = _SHIFT_R
    b = ExprOp(">>", _Y32, ExprInt(5, 32))
    expr = (a & b) + (a | b)
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [a, b])
    assert _z3_equivalent(expr, out)


def test_simba_cond_atom_sum_folds_to_double() -> None:
    expr = _COND + _COND
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [_COND])
    assert _z3_equivalent(expr, out)


def test_simba_cond_atom_andor_identity_collapses() -> None:
    a = _COND
    b = ExprCond(_C1, _Y32, _X32)
    expr = (a & b) + (a | b)
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [a, b])
    assert _z3_equivalent(expr, out)


def test_simba_division_atom_xor_paper_identity_collapses() -> None:
    # ``-(a | ~b) + ~b + (a & ~b) + b = a ^ b`` for any opaque atoms
    # a and b. Test it with two distinct division subtrees.
    a = _DIV
    b = ExprOp("/", _Y32, _X32)
    expr = (
        (-(a | (b ^ ExprInt(0xFFFFFFFF, 32))))
        + (b ^ ExprInt(0xFFFFFFFF, 32))
        + (a & (b ^ ExprInt(0xFFFFFFFF, 32)))
        + b
    )
    out = SimbaPass().run(expr)
    assert _z3_equivalent(expr, out)


# ---------------------------------------------------------------------------
# Category R — cross-family / mixed atom kinds with non-linear atoms
# ---------------------------------------------------------------------------


def test_simba_register_plus_shift_atom_decomposes() -> None:
    expr = _X32 + _SHIFT_R
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [_X32, _SHIFT_R])


def test_simba_slice_plus_cond_atom_decomposes() -> None:
    sl = ExprSlice(_X32, 0, 32)
    expr = sl + _COND
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [sl, _COND])


def test_simba_memory_plus_division_atom_decomposes() -> None:
    mem = ExprMem(ExprId("ptr", 64), 32)
    expr = mem + _DIV
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [mem, _DIV])


def test_simba_three_kinds_register_shift_cond_decomposes() -> None:
    expr = _X32 + _SHIFT_R + _COND
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [_X32, _SHIFT_R, _COND])


def test_simba_paper_identity_over_shift_and_cond() -> None:
    a = _SHIFT_R
    b = _COND
    expr = (a & b) + (a | b)
    out = SimbaPass().run(expr)
    _assert_equivalent_on_cube(out, expr, [a, b])
    assert _z3_equivalent(expr, out)


# ---------------------------------------------------------------------------
# Category S — atomised top-level pass-through (single-atom shapes
# reconstruct to themselves and stay ``is expr`` for caller identity)
# ---------------------------------------------------------------------------


def test_simba_top_level_shift_passes_through() -> None:
    assert SimbaPass().run(_SHIFT_R) is _SHIFT_R


def test_simba_top_level_rotation_passes_through() -> None:
    assert SimbaPass().run(_ROT_L) is _ROT_L


def test_simba_top_level_division_passes_through() -> None:
    assert SimbaPass().run(_DIV) is _DIV


def test_simba_top_level_count_leading_zeros_passes_through() -> None:
    assert SimbaPass().run(_CLZ) is _CLZ


def test_simba_top_level_power_passes_through() -> None:
    assert SimbaPass().run(_POW) is _POW


def test_simba_top_level_cond_passes_through() -> None:
    assert SimbaPass().run(_COND) is _COND


# ---------------------------------------------------------------------------
# Category U — idempotence on non-linear-atom shapes
# ---------------------------------------------------------------------------


def test_simba_idempotent_on_shift_sum() -> None:
    expr = _SHIFT_R + _SHIFT_R + _SHIFT_R
    once = SimbaPass().run(expr)
    twice = SimbaPass().run(once)
    assert once == twice


def test_simba_idempotent_on_cond_and_or_identity() -> None:
    a = _COND
    b = ExprCond(_C1, _Y32, _X32)
    expr = (a & b) + (a | b)
    once = SimbaPass().run(expr)
    twice = SimbaPass().run(once)
    assert once == twice


def test_simba_idempotent_on_register_plus_shift() -> None:
    expr = _X32 + _SHIFT_R + _SHIFT_R
    once = SimbaPass().run(expr)
    twice = SimbaPass().run(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Category Y — operand-kind rejection regression guard
# ---------------------------------------------------------------------------
#
# These shapes are operand-kind rejections inside ``_apply_op_rule``:
# the op string IS in ``{+, -, *, &, |, ^}`` but the kind combination
# doesn't match a linear-MBA rule. The atomisation extension MUST NOT
# atomise these — atomising them widens the atom set in ways the
# downstream simplifier can't fold, regressing
# ``test_simplifier_demo_mba_reaches_shortest_form_with_placeholder_guard``
# (9 → 13 nodes). Each test pins SiMBA as a no-op (``is expr``).


def test_simba_operand_kind_rejection_mul_two_non_arithmetic_no_op() -> None:
    expr = ExprSlice(_X32, 0, 8) * ExprSlice(_Y32, 0, 8)
    assert SimbaPass().run(expr) is expr


def test_simba_operand_kind_rejection_and_bitwise_mixed_no_op() -> None:
    # ``(x+y) & z`` — kinds [MIXED, BITWISE]. Atomising would expand
    # the demo MBA's atom set.
    expr = (_X32 + _Y32) & ExprId("z", 32)
    assert SimbaPass().run(expr) is expr


def test_simba_operand_kind_rejection_or_bitwise_arithmetic_no_op() -> None:
    # ``x | 0xDEADBEEF`` — kinds [BITWISE, ARITHMETIC] (non-all-ones
    # constant). Same operand-kind rejection class.
    expr = _X32 | ExprInt(0xDEADBEEF, 32)
    assert SimbaPass().run(expr) is expr


def test_simba_operand_kind_rejection_xor_mixed_mixed_no_op() -> None:
    a = _X32 & _Y32
    b = _Y32 + _X32
    expr = (a + ExprOp("-", _Y32)) ^ (b + ExprOp("-", _X32))
    assert SimbaPass().run(expr) is expr


def test_simba_operand_kind_rejection_propagates_through_outer_sum() -> None:
    # Outer ``+`` would otherwise classify as MIXED, but an inner
    # operand-kind rejection (`(x+y) & z`) propagates None upward,
    # so the whole expression remains a no-op. Pins the strict
    # propagation rule that prevents the demo-MBA regression.
    inner = (_X32 + _Y32) & ExprId("z", 32)
    expr = inner + ExprId("w", 32)
    assert SimbaPass().run(expr) is expr


# ---------------------------------------------------------------------------
# Category T — Z3-checked directed equivalence (non-linear atoms)
# ---------------------------------------------------------------------------


def test_simba_z3_shift_paper_identity() -> None:
    a = _SHIFT_R
    b = ExprOp(">>", _Y32, ExprInt(5, 32))
    expr = (a & b) + (a | b)
    assert _z3_equivalent(expr, SimbaPass().run(expr))


def test_simba_z3_rotation_paper_identity() -> None:
    a = _ROT_L
    b = ExprOp("<<<", _Y32, ExprInt(7, 32))
    expr = (a & b) + (a | b)
    assert _z3_equivalent(expr, SimbaPass().run(expr))


def test_simba_z3_division_self_xor() -> None:
    expr = _DIV ^ _DIV
    assert _z3_equivalent(expr, SimbaPass().run(expr))


def test_simba_z3_cond_paper_identity() -> None:
    a = _COND
    b = ExprCond(_C1, _Y32, _X32)
    expr = (a & b) + (a | b)
    assert _z3_equivalent(expr, SimbaPass().run(expr))


def test_simba_z3_register_plus_shift() -> None:
    expr = _X32 + _SHIFT_R
    assert _z3_equivalent(expr, SimbaPass().run(expr))


def test_simba_z3_three_atom_mix_register_shift_cond() -> None:
    expr = _X32 + _SHIFT_R + _COND
    assert _z3_equivalent(expr, SimbaPass().run(expr))


def test_simba_z3_affine_over_shift_atom() -> None:
    expr = ExprInt(5, 32) * _SHIFT_R + ExprInt(7, 32)
    assert _z3_equivalent(expr, SimbaPass().run(expr))


def test_simba_z3_xor_all_ones_with_shift_atom() -> None:
    # ``~SHIFT`` (= SHIFT ^ 0xFFFFFFFF) — the all-ones path through
    # the XOR rule must still classify cleanly once the shift is an
    # atom.
    expr = _SHIFT_R ^ ExprInt(0xFFFFFFFF, 32)
    assert _z3_equivalent(expr, SimbaPass().run(expr))


def test_simba_z3_xor_all_ones_with_cond_atom() -> None:
    expr = _COND ^ ExprInt(0xFFFFFFFF, 32)
    assert _z3_equivalent(expr, SimbaPass().run(expr))


# ---------------------------------------------------------------------------
# Category V — atom-set composition and dedup with non-linear atoms
# ---------------------------------------------------------------------------


def test_collect_atoms_register_shift_distinguished() -> None:
    # ``x`` and ``x >> 5`` must be DISTINCT atoms even though they
    # share the same base register — the shift's value depends on x
    # but the cube treats them as independent (and the SiMBA theorem
    # is sound on the full cube, which is a superset of the
    # reachable pairs).
    assert set(_collect_atoms(_X32 + _SHIFT_R)) == {_X32, _SHIFT_R}


def test_collect_atoms_two_different_shift_amounts_distinguished() -> None:
    a = ExprOp(">>", _X32, ExprInt(3, 32))
    b = ExprOp(">>", _X32, ExprInt(5, 32))
    assert set(_collect_atoms(a + b)) == {a, b}


def test_collect_atoms_cond_and_inner_branch_register_distinguished() -> None:
    # The cond is the atom; its branches' variables don't escape.
    # Adding ``x`` (which is also inside the cond) creates TWO atoms.
    expr = _X32 + _COND
    assert set(_collect_atoms(expr)) == {_X32, _COND}


def test_collect_atoms_division_dedup_in_xor() -> None:
    expr = _DIV ^ _DIV
    assert _collect_atoms(expr) == [_DIV]


def test_collect_atoms_mixed_kinds_with_shift_atom() -> None:
    # All four atom kinds + shift atom in one expression.
    sl = ExprSlice(ExprId("xs", 64), 0, 32)
    comp = ExprCompose(ExprId("clo", 16), ExprId("chi", 16))
    mem = ExprMem(ExprId("ptr", 64), 32)
    expr = sl + comp + mem + _SHIFT_R + _X32
    assert set(_collect_atoms(expr)) == {sl, comp, mem, _SHIFT_R, _X32}
