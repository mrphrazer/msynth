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

from msynth.parsing import parse_infix_expr
from msynth.simplification.simba import SimbaPass
from msynth.utils.expr_utils import get_unique_variables


def simplify(text: str, *, size: int = 8) -> Expr:
    return SimbaPass().run(parse_infix_expr(text, size=size))


def assert_simplifies_to(source: str, expected: str, *, size: int = 8) -> None:
    source_expr = parse_infix_expr(source, size=size)
    expected_expr = parse_infix_expr(expected, size=size)
    simplified = SimbaPass().run(source_expr)

    assert expr_simp(simplified) == expr_simp(expected_expr)
    assert_equivalent(source_expr, simplified)


def assert_equivalent(left: Expr, right: Expr) -> None:
    variables = sorted(
        set(get_unique_variables(left)) | set(get_unique_variables(right)),
        key=lambda expr: str(expr),
    )
    assert left.size == right.size
    assert len(variables) <= 5

    for assignment in range(1 << len(variables)):
        env = {
            variable: (assignment >> index) & 1
            for index, variable in enumerate(variables)
        }
        assert evaluate(left, env) == evaluate(right, env)


def evaluate(expr: Expr, env: dict[Expr, int]) -> int:
    mask = (1 << expr.size) - 1
    if isinstance(expr, ExprInt):
        return int(expr) & mask
    if isinstance(expr, ExprId):
        return env.get(expr, 0) & mask
    if isinstance(expr, ExprOp):
        args = [evaluate(arg, env) for arg in expr.args]
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


def node_count(expr: Expr) -> int:
    return len(expr.graph().nodes())


def test_simba_simplifies_and_or_sum_identity() -> None:
    assert_simplifies_to("(x & y) + (x | y)", "x + y")


def test_simba_simplifies_masked_or_subtraction() -> None:
    assert_simplifies_to("(x | y) - (~x & y) - (x & ~y)", "x & y")


def test_simba_simplifies_xor_refinement() -> None:
    assert_simplifies_to("-(a | ~b) + ~b + (a & ~b) + b", "a ^ b")


def test_simba_simplifies_representative_paper_expression() -> None:
    assert_simplifies_to(
        "2*(s&~t)+2*(s^t)-(s|t)+2*~(s^t)-~t-~(s&t)",
        "s",
    )


def test_simba_simplifies_constant_expression() -> None:
    assert_simplifies_to("(x ^ x) + 5", "5")


def test_simba_simplifies_bitwise_not() -> None:
    assert_simplifies_to("~x", "~x")


def test_simba_simplifies_arithmetic_negation_to_modular_coefficient() -> None:
    x = ExprId("x", 8)

    assert simplify("-x") == ExprOp("*", ExprInt(0xFF, 8), x)


def test_simba_simplifies_affine_output_encoding() -> None:
    assert_simplifies_to("7 * ((x & y) + (x | y)) + 3", "7*x + 7*y + 3")


def test_simba_simplifies_more_than_three_variable_generic_case() -> None:
    source = parse_infix_expr("((x & y) + (x | y)) + z + w", size=8)
    simplified = SimbaPass().run(source)

    assert expr_simp(simplified) == expr_simp(parse_infix_expr("x + y + z + w", size=8))
    assert_equivalent(source, simplified)


def test_simba_refines_after_expression_loses_extra_variables() -> None:
    assert_simplifies_to("(x ^ y) + (z - z) + (w - w)", "x ^ y")


def test_simba_reduces_representative_node_count() -> None:
    source = parse_infix_expr(
        "2*(s&~t)+2*(s^t)-(s|t)+2*~(s^t)-~t-~(s&t)",
        size=8,
    )
    simplified = SimbaPass().run(source)

    assert node_count(simplified) < node_count(source)


def test_simba_returns_non_linear_multiplication_unchanged() -> None:
    expr = parse_infix_expr("x * y", size=8)

    assert SimbaPass().run(expr) is expr


def test_simba_returns_mixed_bitwise_arithmetic_unchanged() -> None:
    expr = parse_infix_expr("(x + y) & z", size=8)

    assert SimbaPass().run(expr) is expr


def test_simba_returns_constants_inside_bitwise_operands_unchanged() -> None:
    expr = parse_infix_expr("x & 1", size=8)

    assert SimbaPass().run(expr) is expr


def test_simba_returns_shift_unchanged() -> None:
    expr = parse_infix_expr("x << 1", size=8)

    assert SimbaPass().run(expr) is expr


def test_simba_returns_mixed_width_slice_unchanged() -> None:
    x = ExprId("x", 8)
    expr = ExprSlice(x, 0, 4)

    assert SimbaPass().run(expr) is expr


def test_simba_returns_compose_unchanged() -> None:
    expr = ExprCompose(ExprId("x", 4), ExprId("y", 4))

    assert SimbaPass().run(expr) is expr


def test_simba_returns_memory_unchanged() -> None:
    expr = ExprMem(ExprId("ptr", 8), 8)

    assert SimbaPass().run(expr) is expr


def test_simba_returns_condition_unchanged() -> None:
    expr = ExprCond(ExprId("c", 1), ExprId("x", 8), ExprId("y", 8))

    assert SimbaPass().run(expr) is expr


def test_simba_atomises_cond_in_linear_combination_soundly() -> None:
    # Under the atomisation extension (GAMBA 5.5), ExprCond is a
    # primary atom on the cube, so this ``cond + 1`` expression is
    # within the linear-MBA fragment and SimbaPass will reconstruct
    # it. The reconstruction may take a different shape than the
    # input (e.g. ``-(cond ^ 0xFF)`` instead of ``cond + 1``); the
    # invariant we pin here is semantic equivalence — Z3 sees no
    # bit-vector input that distinguishes the two.
    cond = ExprCond(ExprId("c", 1), ExprId("x", 8), ExprId("y", 8))
    expr = ExprOp("+", cond, ExprInt(1, 8))

    out = SimbaPass().run(expr)
    assert _xor_z3_equivalent(expr, out), (
        f"unsound SimbaPass rewrite over ExprCond atom:\n"
        f"  source:    {expr}\n  rewritten: {out}"
    )


# ---------------------------------------------------------------------------
# Classifier-soundness regression: XOR over linear-MBA operands
# ---------------------------------------------------------------------------
#
# Discovered via the slice/compose fuzz harness: SimbaPass's _classify
# was returning ARITHMETIC for ``MIXED ^ MIXED`` (and for
# ``MIXED ^ non_allones_constant``) — categories that are NOT in the
# linear-MBA fragment. The cube reconstruction then extrapolated from
# boolean-cube samples to all bit-vector inputs, producing rewrites
# that agree with the source on {0,1}^n but disagree everywhere else.
#
# The tests below pin the classifier's correct behaviour: SimbaPass
# must remain a no-op on these shapes (or, at worst, produce a
# Z3-equivalent rewrite). They are intentionally small and free of
# slices/composes so the bug is clearly orthogonal to atom kind.


def _xor_z3_equivalent(left: Expr, right: Expr) -> bool:
    """Local helper duplicated from test_simba_atoms.py to avoid a
    cross-test-file import. UNSAT(left != right) iff sound rewrite."""
    import z3
    from miasm.ir.translators.z3_ir import TranslatorZ3

    assert left.size == right.size
    translator = TranslatorZ3()
    z3_left = translator.from_expr(left)
    z3_right = translator.from_expr(right)
    solver = z3.Solver()
    # Generous headroom: the hard 4-var multiply-by-constant corpus query
    # solves in ~3s unloaded, but a 5s cap flaked under full-suite CPU
    # contention. A genuine timeout must be a LOUD, distinct error — silently
    # returning ``unknown == unsat`` (False) would misreport a correct rewrite
    # as unsound.
    solver.set("timeout", 30000)
    solver.add(z3_left != z3_right)
    result = solver.check()
    if result == z3.unknown:
        raise AssertionError(
            f"Z3 returned unknown within timeout for {left!r} vs {right!r}; "
            "raise the timeout or move this case to scripts/run_simba_fuzzer.py."
        )
    return result == z3.unsat


def test_simba_classifier_rejects_xor_of_mixed_with_mixed() -> None:
    # ((a & b) + -b) is MIXED; (c + -b) is MIXED. Their XOR is NOT a
    # linear MBA — the rewrite must therefore be either ``is expr``
    # (classifier rejected) or semantically equivalent to the input.
    a = ExprId("a", 16)
    b = ExprId("b", 16)
    c = ExprId("c", 16)
    d = ExprId("d", 16)
    expr = (((a & b) + ExprOp("-", b)) ^ (c + ExprOp("-", b))) + ExprOp(
        "-", ExprOp("-", d)
    )
    out = SimbaPass().run(expr)
    assert _xor_z3_equivalent(expr, out), (
        f"unsound SimbaPass rewrite of MIXED^MIXED:\n"
        f"  source:    {expr}\n  rewritten: {out}"
    )


def test_simba_classifier_rejects_xor_of_mixed_with_non_allones_const() -> None:
    # MIXED ^ constant (non-all-ones) is bitwise XOR with a known bit
    # pattern — not a linear MBA when the MIXED operand isn't bitwise.
    a = ExprId("a", 16)
    b = ExprId("b", 16)
    expr = ((a & b) + ExprOp("-", b)) ^ ExprInt(0x5A5A, 16)
    out = SimbaPass().run(expr)
    assert _xor_z3_equivalent(expr, out), (
        f"unsound SimbaPass rewrite of MIXED^const:\n"
        f"  source:    {expr}\n  rewritten: {out}"
    )


def test_simba_classifier_rejects_xor_of_three_mixed_operands() -> None:
    # n-ary XOR over three MIXED operands. None of the operands is
    # bitwise or all-ones, so the classifier must still reject.
    a = ExprId("a", 16)
    b = ExprId("b", 16)
    c = ExprId("c", 16)
    expr = ExprOp(
        "^",
        a + ExprOp("-", b),
        b + ExprOp("-", c),
        c + ExprOp("-", a),
    )
    out = SimbaPass().run(expr)
    assert _xor_z3_equivalent(expr, out), (
        f"unsound SimbaPass rewrite of n-ary MIXED^...^MIXED:\n"
        f"  source:    {expr}\n  rewritten: {out}"
    )


def test_simba_classifier_still_accepts_mixed_xor_all_ones() -> None:
    # ``~MIXED`` (which is ``MIXED ^ all_ones``) IS a linear MBA
    # (``-MIXED - 1``). Coverage here must be preserved after the fix
    # tightens the XOR classification.
    a = ExprId("a", 16)
    b = ExprId("b", 16)
    expr = ((a & b) + ExprOp("-", b)) ^ ExprInt(0xFFFF, 16)
    out = SimbaPass().run(expr)
    assert _xor_z3_equivalent(expr, out), (
        f"unsound SimbaPass rewrite of ~MIXED:\n  source:    {expr}\n  rewritten: {out}"
    )


def test_simba_classifier_still_accepts_bitwise_xor_all_ones() -> None:
    # ``~B`` for B bitwise must still classify as BITWISE.
    a = ExprId("a", 16)
    expr = a ^ ExprInt(0xFFFF, 16)
    out = SimbaPass().run(expr)
    assert _xor_z3_equivalent(expr, out)


def test_simba_classifier_still_accepts_pure_constant_xor() -> None:
    # ``const ^ const`` is itself a constant; the classifier should
    # still call it ARITHMETIC and SimbaPass should reconstruct (or
    # leave it) cleanly.
    expr = ExprInt(0xAA, 8) ^ ExprInt(0x33, 8)
    out = SimbaPass().run(expr)
    assert _xor_z3_equivalent(expr, out)


def test_simba_simplifies_memory_paper_identity() -> None:
    # The (a & b) + (a | b) == a + b identity carried by a single
    # memory atom (b == a). Validates that ExprMem participates in the
    # cube reconstruction, not just that it is named as an atom.
    mem = ExprMem(ExprId("ptr", 8), 8)
    expr = (mem & mem) + (mem | mem)

    simplified = SimbaPass().run(expr)
    assert expr_simp(simplified) == expr_simp(mem + mem)


def test_simba_simplifies_memory_self_xor_to_zero() -> None:
    # A standard identity over an opaque atom; verifies the cube
    # evaluator uses structural equality so two textually identical
    # ExprMem nodes share the same atom on the cube.
    mem = ExprMem(ExprId("ptr", 8), 8)
    expr = mem ^ mem

    simplified = SimbaPass().run(expr)
    assert simplified == ExprInt(0, 8)


def test_simba_collapses_memory_andor_sum_to_linear_sum() -> None:
    # The canonical paper-style identity (a & b) + (a | b) == a + b
    # must hold when both atoms are memory loads, otherwise the cube
    # argument is silently treating one of them as something else.
    a = ExprMem(ExprId("p", 8), 8)
    b = ExprMem(ExprId("q", 8), 8)
    expr = (a & b) + (a | b)

    simplified = SimbaPass().run(expr)
    assert_equivalent_atoms(simplified, expr, [a, b])
    assert node_count(simplified) <= node_count(expr)


def test_simba_handles_mixed_register_and_memory_mba() -> None:
    # Mixed leaves exercise the atom-collector's deterministic ordering
    # (str-sorted across heterogeneous leaf kinds) and the cube
    # evaluator's lookup for both ExprId and ExprMem in the same env.
    x = ExprId("x", 8)
    m = ExprMem(ExprId("p", 8), 8)
    expr = (x & m) + (x | m)

    simplified = SimbaPass().run(expr)
    assert_equivalent_atoms(simplified, expr, [x, m])


def test_simba_does_not_atomise_memory_pointer_internals() -> None:
    # The pointer expression contains an ExprId, but the whole load is
    # one atom — we must not also enumerate the pointer's variables on
    # the cube, or the cube would mis-vary the load when only the
    # pointer's bits flip. Two different ExprMem nodes over the same
    # pointer subtree are the same atom; an inner-pointer variable
    # is not a separate atom.
    from msynth.simplification.simba import _collect_atoms

    mem = ExprMem(ExprId("ptr", 8) + ExprInt(4, 8), 8)
    atoms = _collect_atoms(mem)
    assert atoms == [mem]


def test_simba_collects_distinct_memory_loads_as_distinct_atoms() -> None:
    # Two structurally different memory loads must end up as two atoms;
    # collapsing them would silently treat the cube assignments as
    # symmetric across loads they aren't.
    from msynth.simplification.simba import _collect_atoms

    a = ExprMem(ExprId("p", 8), 8)
    b = ExprMem(ExprId("q", 8), 8)
    atoms = _collect_atoms(a + b)
    assert set(atoms) == {a, b}


def assert_equivalent_atoms(left: Expr, right: Expr, atoms: list[Expr]) -> None:
    """Boolean-cube equivalence check over an explicit atom list."""
    assert left.size == right.size
    assert len(atoms) <= 5
    for assignment in range(1 << len(atoms)):
        env = {atom: (assignment >> index) & 1 for index, atom in enumerate(atoms)}
        assert _evaluate_with_atoms(left, env) == _evaluate_with_atoms(right, env)


def _evaluate_with_atoms(expr: Expr, env: dict[Expr, int]) -> int:
    # Like ``evaluate`` above, but treats ExprId, ExprMem, ExprSlice,
    # and ExprCompose as atomic lookups so the test can mirror SiMBA's
    # atom semantics.
    mask = (1 << expr.size) - 1
    if isinstance(expr, ExprInt):
        return int(expr) & mask
    if isinstance(expr, (ExprId, ExprMem, ExprSlice, ExprCompose)):
        return env.get(expr, 0) & mask
    if isinstance(expr, ExprOp):
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


# --- Quine-McCluskey unit tests ---


def _qm_reconstruct_table(primes: list[tuple[int, int]], n_vars: int) -> int:
    """Reconstruct a packed truth table from a list of (value, mask) primes."""
    table = 0
    for row in range(1 << n_vars):
        for value, mask in primes:
            if (row & ~mask) == (value & ~mask):
                table |= 1 << row
                break
    return table


def test_qm_constant_zero():
    from msynth.simplification.simba import _qm_minimise

    assert _qm_minimise(0, 3) == []


def test_qm_constant_one():
    from msynth.simplification.simba import _qm_minimise

    # All 8 rows of a 3-var function are 1.
    terms = _qm_minimise(0xFF, 3)
    assert len(terms) == 1
    value, mask = terms[0]
    assert mask == 0b111  # all bits are don't-cares


def test_qm_single_variable_x():
    from msynth.simplification.simba import _qm_minimise

    # f(x, y) = x. Rows where x=1: 1, 3 -> table = 0b1010.
    terms = _qm_minimise(0b1010, 2)
    assert len(terms) == 1
    value, mask = terms[0]
    # bit 0 (x) is fixed to 1; bit 1 (y) is don't-care.
    assert mask == 0b10
    assert value & ~mask == 1


def test_qm_xor_two_vars():
    from msynth.simplification.simba import _qm_minimise

    # f(x, y) = x ^ y. table = 0b0110. Expect 2 terms covering 01 and 10.
    terms = _qm_minimise(0b0110, 2)
    assert len(terms) == 2
    # No combinable pair (each minterm differs from the other in two
    # bits), so both are single-minterm implicants.
    covered = {value & ~mask for value, mask in terms}
    assert covered == {1, 2}


def test_qm_and_three_vars():
    from msynth.simplification.simba import _qm_minimise

    # f(x, y, z) = x & y & z. table = 0b10000000 (only row 7).
    terms = _qm_minimise(0b10000000, 3)
    assert len(terms) == 1
    value, mask = terms[0]
    assert mask == 0
    assert value == 0b111


def test_qm_or_three_vars():
    from msynth.simplification.simba import _qm_minimise

    # f(x, y, z) = x | y | z. table = 0b11111110.
    terms = _qm_minimise(0b11111110, 3)
    assert len(terms) == 3
    # Each implicant should fix exactly one literal to 1.
    for value, mask in terms:
        fixed_bits = (~mask) & 0b111
        assert bin(fixed_bits).count("1") == 1
        assert value & fixed_bits == fixed_bits


def test_qm_majority():
    from msynth.simplification.simba import _qm_minimise

    # majority(x, y, z) = 1 iff at least 2 of (x, y, z) are 1.
    # Rows with popcount >= 2: 3, 5, 6, 7 -> table = 0b11101000.
    terms = _qm_minimise(0b11101000, 3)
    assert len(terms) == 3
    # Each prime should fix exactly two literals to 1.
    for value, mask in terms:
        fixed_bits = (~mask) & 0b111
        assert bin(fixed_bits).count("1") == 2
        assert value & fixed_bits == fixed_bits


def test_qm_4var_compact():
    from msynth.simplification.simba import _qm_minimise

    # f(x, y, z, w) = (x & y) | (z & w).
    # Minterms: rows where (bit0=1 & bit1=1) or (bit2=1 & bit3=1)
    #   = {3, 7, 11, 12, 13, 14, 15} (7 minterms - DNF would emit 7 terms)
    # QM should collapse to two prime implicants: (x&y) and (z&w).
    table = 0
    for row in [3, 7, 11, 12, 13, 14, 15]:
        table |= 1 << row
    terms = _qm_minimise(table, 4)
    assert len(terms) == 2
    # The two implicants are (value=0b0011, mask=0b1100) and
    # (value=0b1100, mask=0b0011).
    assert set(terms) == {(3, 12), (12, 3)}


def test_qm_round_trip():
    import random

    from msynth.simplification.simba import _qm_minimise

    rng = random.Random(0xC0DE)
    n_vars = 3
    rows = 1 << n_vars
    for _ in range(20):
        table = rng.randrange(0, 1 << rows)
        terms = _qm_minimise(table, n_vars)
        recovered = _qm_reconstruct_table(terms, n_vars)
        assert recovered == table, (
            f"QM round-trip mismatch: table=0x{table:X} terms={terms}"
        )


def test_qm_no_redundant_implicants():
    from msynth.simplification.simba import _qm_minimise

    # AND function: only one prime implicant (the single full-degree term).
    terms = _qm_minimise(0b10000000, 3)
    assert len(terms) == 1


def test_qm_terms_cover_all_minterms():
    from msynth.simplification.simba import _qm_minimise

    # Non-trivial 4-variable table; every 1-bit must be covered.
    table = 0b1011_0101_1100_1001
    n_vars = 4
    terms = _qm_minimise(table, n_vars)
    for row in range(1 << n_vars):
        if (table >> row) & 1:
            assert any((row & ~mask) == (value & ~mask) for value, mask in terms), (
                f"row {row} not covered by any prime implicant"
            )


def test_qm_terms_dont_cover_zero_bits():
    from msynth.simplification.simba import _qm_minimise

    # Same table; verify no implicant matches a 0-row.
    table = 0b1011_0101_1100_1001
    n_vars = 4
    terms = _qm_minimise(table, n_vars)
    for row in range(1 << n_vars):
        if not ((table >> row) & 1):
            for value, mask in terms:
                assert (row & ~mask) != (value & ~mask), (
                    f"prime ({value},{mask}) wrongly covers zero-row {row}"
                )


# --- SimbaPass + QM integration tests ---


def test_simba_4var_qm_produces_compact_form():
    # Sample drawn from cobra.jsonl.gz simba/e1_4vars suite
    # (case_061916, expected = ``3735936685 * ~x``).
    import gzip
    import json
    from pathlib import Path

    corpus = (
        Path(__file__).resolve().parents[2] / "datasets" / "corpora" / "cobra.jsonl.gz"
    )
    record = None
    with gzip.open(corpus, "rt") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["id"] == "case_061916":
                record = rec
                break
    assert record is not None, "expected corpus entry case_061916 not found"

    source = parse_infix_expr(record["expr_text"], size=record["size"])
    expected = parse_infix_expr(record["expected_text"], size=record["size"])
    simplified = SimbaPass().run(source)

    # The corpus entry expands a single ``coeff * ~x`` term across many
    # cube cells; the rewrite should collapse back to roughly the
    # expected node count (small affine-encoding slack allowed).
    assert node_count(simplified) <= node_count(expected) + 2
    assert _xor_z3_equivalent(source, simplified), (
        f"unsound SimbaPass rewrite of 4-var corpus entry:\n"
        f"  source:    {source}\n  rewritten: {simplified}"
    )


def test_simba_qm_z3_equivalent():
    # Three directed cases that exercise the QM fallback inside
    # _lookup_bitwise_expression. Each is a 3-variable bitwise function
    # whose truth table is not a single n-ary XOR/AND/OR (so QM is
    # actually invoked), wrapped in a coefficient to keep it on the
    # linear-MBA fragment SimbaPass actually rewrites.
    x = ExprId("x", 8)
    y = ExprId("y", 8)
    z = ExprId("z", 8)
    coeff = ExprInt(7, 8)

    majority = (x & y) | (x & z) | (y & z)
    case1 = ExprOp("*", coeff, majority)
    out1 = SimbaPass().run(case1)
    assert _xor_z3_equivalent(case1, out1), f"unsound: source={case1} rewritten={out1}"

    # f = (x & ~y) | (~x & z)
    case2 = ExprOp(
        "*",
        coeff,
        (x & (y ^ ExprInt(0xFF, 8))) | ((x ^ ExprInt(0xFF, 8)) & z),
    )
    out2 = SimbaPass().run(case2)
    assert _xor_z3_equivalent(case2, out2), f"unsound: source={case2} rewritten={out2}"

    # f = (x ^ y) | (y ^ z) (3-variable, not a single n-ary AND/OR/XOR)
    case3 = ExprOp("*", coeff, (x ^ y) | (y ^ z))
    out3 = SimbaPass().run(case3)
    assert _xor_z3_equivalent(case3, out3), f"unsound: source={case3} rewritten={out3}"
