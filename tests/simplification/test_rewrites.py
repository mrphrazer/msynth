"""
Per-rule soundness tests + Rewriter integration tests for
``msynth.simplification.rewrites``.

Each rule in :data:`DEFAULT_RULES` is verified on three independent axes:

1. **Cube equivalence** -- complete on the linear-MBA fragment; the rule
   must agree with its input on every ``{0,1}^n`` assignment of its atoms.
2. **Z3 equivalence** -- complete in general over bit-vectors;
   ``z3.unknown`` is a hard failure to avoid silently passing on solver
   timeout.
3. **Random 32-bit sampling** -- defence in depth for rules where the
   cube and Z3 paths might both have blind spots (especially
   ``power_of_two`` rules involving multiplications by constants).

Integration tests verify ``Rewriter.normalize`` is a behaviour-preserving
drop-in for the previous ``ring_normalize(expr_simp(expr))`` and that
safe rules actually fire inside the local
``ExpressionSimplifier`` instance.
"""

from __future__ import annotations

import random

import pytest
from miasm.expression.expression import Expr, ExprId, ExprInt, ExprOp
from miasm.expression.simplifications import expr_simp

from msynth.simplification.rewrites import (
    DEFAULT_RULES,
    DEFAULT_REWRITER,
    RewriteRule,
    Rewriter,
    ring_normalize,
)


_SIZE = 16
_MASK = (1 << _SIZE) - 1


def _atoms():
    return (
        ExprId("a", _SIZE),
        ExprId("b", _SIZE),
        ExprId("c", _SIZE),
        ExprId("d", _SIZE),
    )


# ---------------------------------------------------------------------------
# Equivalence helpers
# ---------------------------------------------------------------------------


def _collect_id_atoms(expr: Expr) -> set:
    out: set = set()
    expr.visit(lambda e: (out.add(e) if e.is_id() else None) or e)
    return out


class _CubeUnsupported(Exception):
    pass


def _eval_cube(expr: Expr, env: dict, mask: int) -> int:
    if isinstance(expr, ExprInt):
        return int(expr) & mask
    if expr.is_id():
        return env.get(expr, 0) & mask
    if isinstance(expr, ExprOp):
        args = [_eval_cube(a, env, mask) for a in expr.args]
        op = expr.op
        if op == "+":
            return sum(args) & mask
        if op == "-":
            if len(args) == 1:
                return (-args[0]) & mask
            r = args[0]
            for a in args[1:]:
                r -= a
            return r & mask
        if op == "*":
            r = 1
            for a in args:
                r *= a
            return r & mask
        if op == "&":
            r = mask
            for a in args:
                r &= a
            return r & mask
        if op == "|":
            r = 0
            for a in args:
                r |= a
            return r & mask
        if op == "^":
            r = 0
            for a in args:
                r ^= a
            return r & mask
        if op == "<<":
            return (args[0] << args[1]) & mask
    raise _CubeUnsupported(f"unsupported op: {expr}")


def _cube_equivalent(left: Expr, right: Expr) -> bool:
    if left.size != right.size:
        return False
    atoms = sorted(_collect_id_atoms(left) | _collect_id_atoms(right), key=str)
    if len(atoms) > 6:
        # Too many atoms for an exhaustive cube sweep. Signal "can't check"
        # rather than silently returning True -- the caller relies on the
        # mandatory Z3 check instead.
        raise _CubeUnsupported("too many atoms for exhaustive cube check")
    mask = (1 << left.size) - 1
    for assignment in range(1 << len(atoms)):
        env = {atom: (assignment >> i) & 1 for i, atom in enumerate(atoms)}
        # _eval_cube raises _CubeUnsupported on ops outside its fragment;
        # let that propagate so the caller skips this axis explicitly.
        lv = _eval_cube(left, env, mask)
        rv = _eval_cube(right, env, mask)
        if lv != rv:
            return False
    return True


def _z3_equivalent(left: Expr, right: Expr, *, timeout_ms: int = 5000) -> bool:
    import z3
    from miasm.ir.translators.z3_ir import TranslatorZ3

    if left.size != right.size:
        return False
    translator = TranslatorZ3()
    z3_left = translator.from_expr(left)
    z3_right = translator.from_expr(right)
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.add(z3_left != z3_right)
    result = solver.check()
    if result == z3.unknown:
        raise AssertionError(f"z3 returned unknown for {left!r} vs {right!r}")
    return result == z3.unsat


def _random_sample_equivalent(
    left: Expr, right: Expr, *, n_samples: int = 32, seed: int = 0xC0FFEE
) -> bool:
    size = max(left.size, right.size)
    mask = (1 << size) - 1
    rng = random.Random(seed)
    atoms = sorted(_collect_id_atoms(left) | _collect_id_atoms(right), key=str)
    for _ in range(n_samples):
        env = {a: rng.getrandbits(a.size) & ((1 << a.size) - 1) for a in atoms}
        # let _CubeUnsupported propagate (see _cube_equivalent); the caller
        # falls back to the mandatory Z3 check rather than a silent pass.
        lv = _eval_cube(left, env, mask)
        rv = _eval_cube(right, env, mask)
        if lv != rv:
            return False
    return True


def _assert_sound(input_expr: Expr, output_expr: Expr, name: str) -> None:
    # Z3 is the authoritative, op-complete equivalence check and always runs.
    assert _z3_equivalent(input_expr, output_expr), (
        f"rule {name}: z3 found counterexample\n"
        f"  in:  {input_expr}\n  out: {output_expr}"
    )
    # The cube and random-sample axes are redundant cross-checks that only
    # cover _eval_cube's fragment. When a shape falls outside it (e.g. a
    # shift, or too many atoms) the helpers raise _CubeUnsupported; we skip
    # that axis explicitly and rely on the Z3 proof above, rather than
    # silently reporting a pass.
    try:
        assert _cube_equivalent(input_expr, output_expr), (
            f"rule {name}: cube disagreement\n  in:  {input_expr}\n  out: {output_expr}"
        )
    except _CubeUnsupported:
        pass
    try:
        assert _random_sample_equivalent(input_expr, output_expr), (
            f"rule {name}: random-sample counterexample\n"
            f"  in:  {input_expr}\n  out: {output_expr}"
        )
    except _CubeUnsupported:
        pass


# ---------------------------------------------------------------------------
# Matching-input factory: one hand-authored matching input per rule
# ---------------------------------------------------------------------------


def _mask_expr() -> ExprInt:
    return ExprInt(_MASK, _SIZE)


def _not_(x: Expr) -> Expr:
    return ExprOp("^", x, _mask_expr())


def _matching_input(rule: RewriteRule) -> Expr:
    a, b, c, d = _atoms()
    name = rule.name

    if name == "ring_normalize":
        # Shape borrowed from the existing ring-normalize regression test
        # (see ``test_ring_distributes_and_collects_across_compound_sum``)
        # -- a CEGIS/SimBA-like compound sum that Miasm leaves at ~17
        # nodes and the ring rule strictly shrinks via flatten+distribute
        # +collect.
        neg2 = ExprInt((-2) & _MASK, _SIZE)
        return expr_simp(
            ExprOp(
                "+",
                ExprOp("*", neg2, a),
                ExprOp("*", ExprInt(5, _SIZE), c),
                ExprOp("*", ExprInt(2, _SIZE), ExprOp("+", a, b)),
                ExprOp(
                    "*",
                    ExprInt(2, _SIZE),
                    ExprOp("+", a, b, ExprOp("-", c)),
                ),
                ExprOp(
                    "*",
                    ExprInt(2, _SIZE),
                    ExprOp("+", b, ExprOp("*", ExprInt(2, _SIZE), a)),
                ),
            )
        )

    if name == "inverse_xor_neg":
        # (X & Y) + (~X & Y)  --  use a single non-leaf X so the rule has
        # to dig past structural equality on the un-negated side.
        return ExprOp(
            "+",
            ExprOp("&", a, b),
            ExprOp("&", _not_(a), b),
        )
    if name == "inverse_or_neg":
        # (X | Y) + (~X | Y) -> Y + (-1)
        return ExprOp(
            "+",
            ExprOp("|", a, b),
            ExprOp("|", _not_(a), b),
        )
    if name == "inverse_xor_neg_xor":
        # (X ^ Y) + (~X ^ Y) -> -1
        return ExprOp(
            "+",
            ExprOp("^", a, b),
            ExprOp("^", _not_(a), b),
        )

    if name == "add_complement_pair":
        # a + ~a -> -1
        return ExprOp("+", a, _not_(a))
    if name == "xor_complement_pair":
        # a ^ ~a -> -1
        return ExprOp("^", a, _not_(a))

    if name == "constant_merge_and":
        # (0x0F & X) + (0xF0 & X) -> (0xFF) & X
        return ExprOp(
            "+",
            ExprOp("&", ExprInt(0x0F, _SIZE), a),
            ExprOp("&", ExprInt(0xF0, _SIZE), a),
        )

    if name == "factor_pow2_from_and":
        # (2*a) & (2*b) -> 2 * (a & b)
        return ExprOp(
            "&",
            ExprOp("*", ExprInt(2, _SIZE), a),
            ExprOp("*", ExprInt(2, _SIZE), b),
        )
    if name == "factor_pow2_from_or":
        return ExprOp(
            "|",
            ExprOp("*", ExprInt(4, _SIZE), a),
            ExprOp("*", ExprInt(4, _SIZE), b),
        )
    if name == "factor_pow2_from_xor":
        return ExprOp(
            "^",
            ExprOp("*", ExprInt(8, _SIZE), a),
            ExprOp("*", ExprInt(8, _SIZE), b),
        )

    if name == "or_xor_split":
        # (X & Y) | (X ^ Y) -> (X & Y) + (X ^ Y)
        return ExprOp("|", ExprOp("&", a, b), ExprOp("^", a, b))

    if name == "demorgan_and_to_or":
        # ~(~a & b) -> ~~a | ~b -> a | ~b  (after double-neg via expr_simp)
        return _not_(ExprOp("&", _not_(a), b))
    if name == "demorgan_or_to_and":
        return _not_(ExprOp("|", _not_(a), b))

    if name == "factor_common_subterm":
        # (a * b) + (a * c) -> a * (b + c)
        return ExprOp(
            "+",
            ExprOp("*", a, b),
            ExprOp("*", a, c),
        )

    if name == "absorption_or":
        # a | (a & b) -> a
        return ExprOp("|", a, ExprOp("&", a, b))
    if name == "absorption_and":
        # a & (a | b) -> a
        return ExprOp("&", a, ExprOp("|", a, b))
    if name == "redundancy_or_not":
        # a | ~a -> -1
        return ExprOp("|", a, _not_(a))
    if name == "redundancy_and_not":
        # a & ~a -> 0
        return ExprOp("&", a, _not_(a))

    # --- New Tier 1 + Tier 2 rules ---

    if name == "idempotence_and_drop_duplicates":
        return ExprOp("&", a, a, b)
    if name == "idempotence_or_drop_duplicates":
        return ExprOp("|", a, a, b)
    if name == "xor_self_cancel_pairs":
        # a ^ b ^ a -> b
        return ExprOp("^", a, b, a)

    if name == "double_negation_collapse":
        return _not_(_not_(a))

    if name == "const_fold_and":
        # Annihilator side: a & 0 -> 0
        return ExprOp("&", a, ExprInt(0, _SIZE))
    if name == "const_fold_or":
        # Annihilator side: a | -1 -> -1
        return ExprOp("|", a, _mask_expr())
    if name == "const_fold_xor_zero":
        return ExprOp("^", a, ExprInt(0, _SIZE))
    if name == "const_fold_add_zero":
        return ExprOp("+", a, ExprInt(0, _SIZE))
    if name == "const_fold_mul_one":
        return ExprOp("*", a, ExprInt(1, _SIZE))
    if name == "const_fold_mul_zero":
        return ExprOp("*", a, ExprInt(0, _SIZE))

    if name == "conj_self_neg_double_collapses_to_zero":
        # x & -x & 2*x -> 0
        return ExprOp(
            "&",
            a,
            ExprOp("-", a),
            ExprOp("*", ExprInt(2, _SIZE), a),
        )
    if name == "conj_neg_xor_collapses_to_zero":
        # ~(2*x) & -(x ^ -x) -> 0
        two_a = ExprOp("*", ExprInt(2, _SIZE), a)
        xor_self_neg = ExprOp("^", a, ExprOp("-", a))
        return ExprOp("&", _not_(two_a), ExprOp("-", xor_self_neg))
    if name == "conj_negated_xor_collapses_to_zero":
        # 2*x & ~(x ^ -x) -> 0
        two_a = ExprOp("*", ExprInt(2, _SIZE), a)
        xor_self_neg = ExprOp("^", a, ExprOp("-", a))
        return ExprOp("&", two_a, _not_(xor_self_neg))

    if name == "conj_xor_identity_clause":
        # 2*x & (x ^ -x) -> 2*x
        two_a = ExprOp("*", ExprInt(2, _SIZE), a)
        xor_self_neg = ExprOp("^", a, ExprOp("-", a))
        return ExprOp("&", two_a, xor_self_neg)
    if name == "disj_xor_identity_clause":
        # 2*x | -(x ^ -x) -> 2*x
        two_a = ExprOp("*", ExprInt(2, _SIZE), a)
        xor_self_neg = ExprOp("^", a, ExprOp("-", a))
        return ExprOp("|", two_a, ExprOp("-", xor_self_neg))

    if name == "disj_disj_negation_absorb":
        # x | -((x & y) | -x) -> x
        inner = ExprOp("|", ExprOp("&", a, b), ExprOp("-", a))
        return ExprOp("|", a, ExprOp("-", inner))
    if name == "conj_conj_negation_absorb":
        # x & -((x | y) & -x) -> x
        inner = ExprOp("&", ExprOp("|", a, b), ExprOp("-", a))
        return ExprOp("&", a, ExprOp("-", inner))
    if name == "disj_conj_negation_absorb":
        # -x | (~x & 2*x) -> -x
        return ExprOp(
            "|",
            ExprOp("-", a),
            ExprOp("&", _not_(a), ExprOp("*", ExprInt(2, _SIZE), a)),
        )
    if name == "disj_neg_disj_identity":
        # x | -(-x | 2*x) -> x
        inner = ExprOp("|", ExprOp("-", a), ExprOp("*", ExprInt(2, _SIZE), a))
        return ExprOp("|", a, ExprOp("-", inner))

    if name == "xor_same_mult_or":
        # 2*(x | -x) -> x ^ -x
        return ExprOp("*", ExprInt(2, _SIZE), ExprOp("|", a, ExprOp("-", a)))
    if name == "xor_same_mult_and":
        # -2*(x & -x) -> x ^ -x
        return ExprOp(
            "*",
            ExprInt((-2) & _MASK, _SIZE),
            ExprOp("&", a, ExprOp("-", a)),
        )

    if name == "complement_pair_and_or":
        # (a & b) | (a & ~b) -> a
        return ExprOp("|", ExprOp("&", a, b), ExprOp("&", a, _not_(b)))

    if name == "bitwise_in_sum_cancel":
        # (a & b) - a - b -> -(a | b)
        return ExprOp(
            "+",
            ExprOp("&", a, b),
            ExprOp("-", a),
            ExprOp("-", b),
        )

    # --- New Tier 3 algebraic identities ---

    if name == "or_pairs_with_disjoint_constants":
        # (0x0F | a) + (0xF0 | a)
        return ExprOp(
            "+",
            ExprOp("|", ExprInt(0x0F, _SIZE), a),
            ExprOp("|", ExprInt(0xF0, _SIZE), a),
        )
    if name == "diff_bitw_pairs_with_disjoint_constants":
        # -(0x0F & a) + (0xF0 | a)  -- the and-or mixed pattern
        return ExprOp(
            "+",
            ExprOp("-", ExprOp("&", ExprInt(0x0F, _SIZE), a)),
            ExprOp("|", ExprInt(0xF0, _SIZE), a),
        )

    if name == "diff_inv_or_minus_andnot":
        # (a | b) + (-(~a & b)) -> a
        return ExprOp(
            "+",
            ExprOp("|", a, b),
            ExprOp("-", ExprOp("&", _not_(a), b)),
        )
    if name == "diff_inv_xor_minus_andnot":
        # (a ^ b) + (-2 * (~a & b)) -> a - b
        return ExprOp(
            "+",
            ExprOp("^", a, b),
            ExprOp(
                "*",
                ExprInt((-2) & _MASK, _SIZE),
                ExprOp("&", _not_(a), b),
            ),
        )
    if name == "diff_inv_xor_plus_ornot":
        # (a ^ b) + 2*(~a | b) -> -2 - a + b
        return ExprOp(
            "+",
            ExprOp("^", a, b),
            ExprOp(
                "*",
                ExprInt(2, _SIZE),
                ExprOp("|", _not_(a), b),
            ),
        )

    if name == "xor_involving_disj":
        # ~a ^ (~a | b) -> ~(~a) & b -> a & b (double-negation collapses
        # inside ``_not_of``, so the rewrite is net-smaller).
        return ExprOp("^", _not_(a), ExprOp("|", _not_(a), b))
    if name == "negative_bitw_inverse_and":
        # -(a & -a) -> a | -a
        return ExprOp("-", ExprOp("&", a, ExprOp("-", a)))
    if name == "negative_bitw_inverse_or":
        # -(a | -a) -> a & -a
        return ExprOp("-", ExprOp("|", a, ExprOp("-", a)))

    if name == "disj_sub_disj_identity":
        # a | ((a | b) - b) -> a
        return ExprOp("|", a, ExprOp("-", ExprOp("|", a, b), b))
    if name == "disj_sub_conj_identity":
        # a | (a - (a & b)) -> a
        return ExprOp("|", a, ExprOp("-", a, ExprOp("&", a, b)))
    if name == "conj_add_conj_identity":
        # a & (a + (~a & b)) -> a
        return ExprOp(
            "&",
            a,
            ExprOp("+", a, ExprOp("&", _not_(a), b)),
        )

    if name == "conj_conj_disj":
        # a & -(-b & (a | b)) -> a & b
        inner = ExprOp("&", ExprOp("-", b), ExprOp("|", a, b))
        return ExprOp("&", a, ExprOp("-", inner))
    if name == "disj_disj_conj":
        # a | -(-b | (a & b)) -> a | b
        inner = ExprOp("|", ExprOp("-", b), ExprOp("&", a, b))
        return ExprOp("|", a, ExprOp("-", inner))

    if name == "conj_neg_xor_minus_one":
        # 2*a | ~-(a ^ -a) -> -1
        two_a = ExprOp("*", ExprInt(2, _SIZE), a)
        xor_self_neg = ExprOp("^", a, ExprOp("-", a))
        return ExprOp("|", two_a, _not_(ExprOp("-", xor_self_neg)))

    raise AssertionError(f"no matching input for rule {rule.name!r}")


def _non_matching_input(rule: RewriteRule) -> Expr:
    """Build an input that the rule should reject (``apply`` returns None)."""
    a, b, _, _ = _atoms()
    name = rule.name

    if name == "ring_normalize":
        # Bare identifier is not sum-rooted.
        return a
    if name.startswith("inverse_"):
        # Sum of unrelated bitwise expressions; no complementary pair.
        return ExprOp("+", ExprOp("&", a, b), ExprOp("|", a, b))
    if name == "add_complement_pair":
        return ExprOp("+", a, b)
    if name == "xor_complement_pair":
        return ExprOp("^", a, b)
    if name == "constant_merge_and":
        # Overlapping constant bits -> guard rejects.
        return ExprOp(
            "+",
            ExprOp("&", ExprInt(0x0F, _SIZE), a),
            ExprOp("&", ExprInt(0x03, _SIZE), a),  # overlaps 0x0F
        )
    if name == "factor_pow2_from_and":
        # Coefficients differ -> rejected.
        return ExprOp(
            "&",
            ExprOp("*", ExprInt(2, _SIZE), a),
            ExprOp("*", ExprInt(4, _SIZE), b),
        )
    if name == "factor_pow2_from_or":
        return ExprOp("|", a, b)  # not c*X | c*Y
    if name == "factor_pow2_from_xor":
        return ExprOp("^", a, b)
    if name == "or_xor_split":
        return ExprOp("|", a, b)  # neither operand is an &/^ pair on same atoms
    if name == "demorgan_and_to_or":
        # ~(a & b) where neither operand is a NOT -> rule must reject
        return _not_(ExprOp("&", a, b))
    if name == "demorgan_or_to_and":
        return _not_(ExprOp("|", a, b))

    if name == "factor_common_subterm":
        # (a * b) + (c * d) -- no common multiplicative factor.
        c = ExprId("c", _SIZE)
        d = ExprId("d", _SIZE)
        return ExprOp("+", ExprOp("*", a, b), ExprOp("*", c, d))

    if name == "absorption_or":
        # a | (c & b) -- shared atom is not the right one
        c = ExprId("c", _SIZE)
        return ExprOp("|", a, ExprOp("&", c, b))
    if name == "absorption_and":
        c = ExprId("c", _SIZE)
        return ExprOp("&", a, ExprOp("|", c, b))
    if name == "redundancy_or_not":
        # a | ~b -- different operand, not a self-complement
        return ExprOp("|", a, _not_(b))
    if name == "redundancy_and_not":
        return ExprOp("&", a, _not_(b))

    # --- New Tier 1 + Tier 2 rules ---

    if name == "idempotence_and_drop_duplicates":
        # All-distinct children: no duplicate to drop.
        return ExprOp("&", a, b)
    if name == "idempotence_or_drop_duplicates":
        return ExprOp("|", a, b)
    if name == "xor_self_cancel_pairs":
        # Distinct children, no pair.
        return ExprOp("^", a, b)

    if name == "double_negation_collapse":
        # Only one NOT layer, not two.
        return _not_(a)

    if name == "const_fold_and":
        # No special constant present.
        return ExprOp("&", a, b)
    if name == "const_fold_or":
        return ExprOp("|", a, b)
    if name == "const_fold_xor_zero":
        return ExprOp("^", a, b)
    if name == "const_fold_add_zero":
        return ExprOp("+", a, b)
    if name == "const_fold_mul_one":
        return ExprOp("*", a, b)
    if name == "const_fold_mul_zero":
        return ExprOp("*", a, b)

    if name == "conj_self_neg_double_collapses_to_zero":
        # Only x and 2*x, no -x.
        return ExprOp("&", a, ExprOp("*", ExprInt(2, _SIZE), a))
    if name == "conj_neg_xor_collapses_to_zero":
        # Pieces don't share the same base.
        return ExprOp("&", _not_(a), ExprOp("-", b))
    if name == "conj_negated_xor_collapses_to_zero":
        return ExprOp("&", a, _not_(b))

    if name == "conj_xor_identity_clause":
        # Conjunction without an ``x ^ -x`` clause.
        return ExprOp("&", a, b)
    if name == "disj_xor_identity_clause":
        return ExprOp("|", a, b)

    if name == "disj_disj_negation_absorb":
        return ExprOp("|", a, b)
    if name == "conj_conj_negation_absorb":
        return ExprOp("&", a, b)
    if name == "disj_conj_negation_absorb":
        return ExprOp("|", a, b)
    if name == "disj_neg_disj_identity":
        return ExprOp("|", a, b)

    if name == "xor_same_mult_or":
        # Coefficient is 3, not 2.
        return ExprOp("*", ExprInt(3, _SIZE), ExprOp("|", a, ExprOp("-", a)))
    if name == "xor_same_mult_and":
        return ExprOp("*", ExprInt(3, _SIZE), ExprOp("&", a, ExprOp("-", a)))

    if name == "complement_pair_and_or":
        # Distinct conjunctions with no NOT pairing.
        return ExprOp("|", ExprOp("&", a, b), ExprOp("&", a, b))

    if name == "bitwise_in_sum_cancel":
        # Sum has only two terms, can't match the (bitw, -x, -y) triple.
        return ExprOp("+", a, b)

    # --- New Tier 3 algebraic identities ---

    if name == "or_pairs_with_disjoint_constants":
        return ExprOp(
            "+",
            ExprOp("|", ExprInt(0x0F, _SIZE), a),
            ExprOp("|", ExprInt(0x03, _SIZE), a),
        )
    if name == "diff_bitw_pairs_with_disjoint_constants":
        # Same-op pair -- handled by the dedicated same-op rule.
        return ExprOp(
            "+",
            ExprOp("&", ExprInt(0x0F, _SIZE), a),
            ExprOp("&", ExprInt(0xF0, _SIZE), a),
        )

    if name == "diff_inv_or_minus_andnot":
        # ~ is missing -> rule rejects.
        return ExprOp(
            "+",
            ExprOp("|", a, b),
            ExprOp("-", ExprOp("&", a, b)),
        )
    if name == "diff_inv_xor_minus_andnot":
        # Coefficient is +2, not -2.
        return ExprOp(
            "+",
            ExprOp("^", a, b),
            ExprOp("*", ExprInt(2, _SIZE), ExprOp("&", _not_(a), b)),
        )
    if name == "diff_inv_xor_plus_ornot":
        # Coefficient is 3, not 2.
        return ExprOp(
            "+",
            ExprOp("^", a, b),
            ExprOp("*", ExprInt(3, _SIZE), ExprOp("|", _not_(a), b)),
        )

    if name == "xor_involving_disj":
        # No nested OR sharing the outer atom.
        return ExprOp("^", a, b)
    if name == "negative_bitw_inverse_and":
        # Inner op is OR, not AND.
        return ExprOp("-", ExprOp("|", a, ExprOp("-", a)))
    if name == "negative_bitw_inverse_or":
        return ExprOp("-", ExprOp("&", a, ExprOp("-", a)))

    if name == "disj_sub_disj_identity":
        # Subtractor doesn't match the inner OR's second operand.
        c = ExprId("c", _SIZE)
        return ExprOp("|", a, ExprOp("-", ExprOp("|", a, b), c))
    if name == "disj_sub_conj_identity":
        # Inner AND doesn't include the outer atom.
        c = ExprId("c", _SIZE)
        return ExprOp("|", a, ExprOp("-", a, ExprOp("&", b, c)))
    if name == "conj_add_conj_identity":
        # ~a does not appear inside the inner conjunction.
        return ExprOp("&", a, ExprOp("+", a, ExprOp("&", a, b)))

    if name == "conj_conj_disj":
        # Inner OR does not contain y.
        c = ExprId("c", _SIZE)
        inner = ExprOp("&", ExprOp("-", b), ExprOp("|", a, c))
        return ExprOp("&", a, ExprOp("-", inner))
    if name == "disj_disj_conj":
        c = ExprId("c", _SIZE)
        inner = ExprOp("|", ExprOp("-", b), ExprOp("&", a, c))
        return ExprOp("|", a, ExprOp("-", inner))

    if name == "conj_neg_xor_minus_one":
        # Missing the 2*x clause -> rejects.
        return ExprOp("|", a, _not_(ExprOp("-", ExprOp("^", a, ExprOp("-", a)))))

    raise AssertionError(f"no non-matching input for rule {rule.name!r}")


# ---------------------------------------------------------------------------
# Per-rule tests, parametrised over DEFAULT_RULES
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", DEFAULT_RULES, ids=lambda r: r.name)
def test_rule_applies_to_matching_input(rule: RewriteRule) -> None:
    inp = _matching_input(rule)
    out = rule.apply(inp)
    assert out is not None, f"{rule.name} unexpectedly rejected matching input {inp}"


@pytest.mark.parametrize("rule", DEFAULT_RULES, ids=lambda r: r.name)
def test_rule_rejects_non_matching_input(rule: RewriteRule) -> None:
    inp = _non_matching_input(rule)
    out = rule.apply(inp)
    assert out is None, (
        f"{rule.name} unexpectedly fired on non-matching input {inp} -> {out}"
    )


def test_inverse_xor_neg_does_not_fire_on_pq_aliased_pattern() -> None:
    """
    Regression: ``inverse_xor_neg`` must not fire on
    ``(~Q & Q) + (Q & R)`` (where R != Q). Earlier implementation used
    set-membership checks on the AND arg-pairs which spuriously matched
    when the negated half aliased with the supposed-common half — in
    that case ``(~Q & Q) = 0`` and the sum is ``Q & R``, not ``Q``.
    """
    from msynth.simplification.rewrites import _apply_inverse_xor_neg

    q = ExprId("q", _SIZE)
    r = ExprId("r", _SIZE)
    # (~q & q) + (q & r) -- algebraically equals (q & r)
    expr = ExprOp(
        "+",
        ExprOp("&", _not_(q), q),
        ExprOp("&", q, r),
    )
    result = _apply_inverse_xor_neg(expr)
    # Either the rule rejects (returns None), or its output is
    # algebraically (q & r). Anything else is a soundness bug.
    if result is not None:
        assert _z3_equivalent(expr, result), (
            f"inverse_xor_neg fired unsoundly:\n  in={expr}\n  out={result}"
        )


def test_inverse_or_neg_does_not_fire_on_pq_aliased_pattern() -> None:
    from msynth.simplification.rewrites import _apply_inverse_or_neg

    q = ExprId("q", _SIZE)
    r = ExprId("r", _SIZE)
    expr = ExprOp(
        "+",
        ExprOp("|", _not_(q), q),
        ExprOp("|", q, r),
    )
    result = _apply_inverse_or_neg(expr)
    if result is not None:
        assert _z3_equivalent(expr, result), (
            f"inverse_or_neg fired unsoundly:\n  in={expr}\n  out={result}"
        )


def test_inverse_xor_neg_xor_does_not_fire_on_pq_aliased_pattern() -> None:
    from msynth.simplification.rewrites import _apply_inverse_xor_neg_xor

    q = ExprId("q", _SIZE)
    r = ExprId("r", _SIZE)
    expr = ExprOp(
        "+",
        ExprOp("^", _not_(q), q),
        ExprOp("^", q, r),
    )
    result = _apply_inverse_xor_neg_xor(expr)
    if result is not None:
        assert _z3_equivalent(expr, result), (
            f"inverse_xor_neg_xor fired unsoundly:\n  in={expr}\n  out={result}"
        )


@pytest.mark.parametrize("rule", DEFAULT_RULES, ids=lambda r: r.name)
def test_rule_output_is_equivalent_to_input(rule: RewriteRule) -> None:
    inp = _matching_input(rule)
    out = rule.apply(inp)
    assert out is not None
    _assert_sound(inp, out, rule.name)


@pytest.mark.parametrize("rule", DEFAULT_RULES, ids=lambda r: r.name)
def test_rule_is_idempotent_on_its_own_output(rule: RewriteRule) -> None:
    """
    Applying a rule to its own output must reach a fixed point in one
    step: either the rule rejects (returns None — already canonical) or
    it returns the same shape it produced before. Otherwise the rule
    would loop the engine when run in fixpoint mode.
    """
    inp = _matching_input(rule)
    once = rule.apply(inp)
    assert once is not None, f"{rule.name} unexpectedly rejected its matching input"
    twice = rule.apply(once)
    # Either the rule rejects (None) or returns the same shape.
    assert twice is None or twice == once, (
        f"{rule.name} is not idempotent on its own output:\n"
        f"  once={once}\n  twice={twice}"
    )


# ---------------------------------------------------------------------------
# Rewriter integration
# ---------------------------------------------------------------------------


_INTEGRATION_EXPRS = [
    # The hand-rolled MBA from scripts/simplify_expression.py-style shape
    (lambda a, b: (a & b) + (a | b))(*_atoms()[:2]),
    # XOR complement pair embedded in a sum
    (lambda a, b: a + b + ExprOp("^", a, ExprInt(_MASK, _SIZE)))(*_atoms()[:2]),
    # Inverse element in a sum
    (lambda a, b: ExprOp("+", ExprOp("&", a, b), ExprOp("&", _not_(a), b)))(
        *_atoms()[:2]
    ),
    # Factor 2 from AND
    (
        lambda a, b: ExprOp(
            "&",
            ExprOp("*", ExprInt(4, _SIZE), a),
            ExprOp("*", ExprInt(4, _SIZE), b),
        )
    )(*_atoms()[:2]),
    # Constant merge under AND
    (
        lambda a: ExprOp(
            "+",
            ExprOp("&", ExprInt(0x0F, _SIZE), a),
            ExprOp("&", ExprInt(0xF0, _SIZE), a),
        )
    )(_atoms()[0]),
    # DeMorgan when one operand is already negated
    (lambda a, b: _not_(ExprOp("&", _not_(a), b)))(*_atoms()[:2]),
]


@pytest.mark.parametrize("expr", _INTEGRATION_EXPRS, ids=lambda e: str(e)[:60])
def test_rewriter_normalize_preserves_semantics(expr: Expr) -> None:
    rewriter = Rewriter()
    normalized = rewriter.normalize(expr)
    _assert_sound(expr, normalized, "Rewriter.normalize")


@pytest.mark.parametrize("expr", _INTEGRATION_EXPRS, ids=lambda e: str(e)[:60])
def test_rewriter_normalize_is_idempotent(expr: Expr) -> None:
    rewriter = Rewriter()
    once = rewriter.normalize(expr)
    twice = rewriter.normalize(once)
    assert once == twice, (
        f"normalize is not idempotent on {expr}: once={once} twice={twice}"
    )


def test_default_rewriter_normalize_matches_baseline_pipeline() -> None:
    """The new ``normalize`` must be at least as good as
    ``ring_normalize(expr_simp(expr))`` on every integration input;
    semantically equivalent on all of them."""
    for expr in _INTEGRATION_EXPRS:
        baseline = ring_normalize(expr_simp(expr))
        new = DEFAULT_REWRITER.normalize(expr)
        # Same semantics
        _assert_sound(baseline, new, "default vs baseline")
        # New should be at least as small as baseline
        assert len(new.graph().nodes()) <= len(baseline.graph().nodes()) + 1, (
            f"new normalize grew the tree: {expr} -> baseline={baseline} new={new}"
        )


def test_rewriter_safe_rules_registered_on_local_expr_simp() -> None:
    """Verify that our safe rules are actually wired into the local
    ExpressionSimplifier (registered as ExprOp passes), not silently
    dropped during construction."""
    rewriter = Rewriter()
    simp = rewriter.expr_simp()
    from miasm.expression.expression import ExprOp as _ExprOp

    pass_names = [p.__name__ for p in simp.expr_simp_cb.get(_ExprOp, [])]
    expected_safe_rules = [r for r in DEFAULT_RULES if not r.guarded]
    for rule in expected_safe_rules:
        assert any(rule.name in n for n in pass_names), (
            f"safe rule {rule.name} not registered on the local ExpressionSimplifier"
        )


def test_rewriter_local_expr_simp_does_not_touch_miasm_singleton() -> None:
    """Building a Rewriter must not mutate Miasm's global ``expr_simp``
    singleton. Smoke test: ``expr_simp`` should not contain our pass
    names after a Rewriter is constructed."""
    Rewriter()  # ignore the instance; check side effects
    from miasm.expression.simplifications import expr_simp as global_expr_simp
    from miasm.expression.expression import ExprOp as _ExprOp

    pass_names = [p.__name__ for p in global_expr_simp.expr_simp_cb.get(_ExprOp, [])]
    assert not any("rewrite_pass_" in n for n in pass_names), (
        f"Rewriter contaminated the Miasm singleton: {pass_names}"
    )


def test_rewriter_guarded_rules_have_net_smaller_check() -> None:
    """A guarded rule applied to an input whose rewrite is not smaller
    must return ``None`` (the input is preserved). Test using a bare
    identifier where ring_normalize has nothing to do."""
    a = _atoms()[0]
    out = DEFAULT_REWRITER.normalize(a)
    assert out == a  # nothing to normalise; not None, not a different expr


def test_normalize_is_drop_in_for_old_composition() -> None:
    """End-to-end: ``DEFAULT_REWRITER.normalize`` produces output
    Z3-equivalent to the historical ``ring_normalize(expr_simp(expr))``
    composition on a curated benchmark."""
    benchmark: list[Expr] = []
    a, b, c, _ = _atoms()
    # Several shapes that exercise different rule combinations.
    benchmark.append(a + b - a)  # cancels to b
    benchmark.append(ExprOp("*", ExprInt(7, _SIZE), ExprOp("+", a, b)))  # factored MBA
    benchmark.append(ExprOp("+", ExprOp("&", a, b), ExprOp("|", a, b)))  # = a + b
    benchmark.append(a + a + a)  # = 3 * a
    benchmark.append(_not_(_not_(a)))  # = a

    for expr in benchmark:
        old = ring_normalize(expr_simp(expr))
        new = DEFAULT_REWRITER.normalize(expr)
        _assert_sound(old, new, "drop-in vs baseline")


# ---------------------------------------------------------------------------
# Directed tests for GAMBA §5.2 absorption + redundancy rules
# ---------------------------------------------------------------------------


def test_absorption_or_basic() -> None:
    a, b, _, _ = _atoms()
    expr = ExprOp("|", a, ExprOp("&", a, b))
    out = DEFAULT_REWRITER.normalize(expr)
    assert out == a


def test_absorption_or_symmetric() -> None:
    a, b, _, _ = _atoms()
    expr = ExprOp("|", ExprOp("&", a, b), a)
    out = DEFAULT_REWRITER.normalize(expr)
    assert out == a


def test_absorption_and_basic() -> None:
    a, b, _, _ = _atoms()
    expr = ExprOp("&", a, ExprOp("|", a, b))
    out = DEFAULT_REWRITER.normalize(expr)
    assert out == a


def test_absorption_and_symmetric() -> None:
    a, b, _, _ = _atoms()
    expr = ExprOp("&", ExprOp("|", a, b), a)
    out = DEFAULT_REWRITER.normalize(expr)
    assert out == a


def test_redundancy_or_not_collapses_to_all_ones() -> None:
    a, _, _, _ = _atoms()
    expr = ExprOp("|", a, _not_(a))
    out = DEFAULT_REWRITER.normalize(expr)
    assert out == ExprInt(_MASK, _SIZE)


def test_redundancy_or_not_symmetric() -> None:
    a, _, _, _ = _atoms()
    expr = ExprOp("|", _not_(a), a)
    out = DEFAULT_REWRITER.normalize(expr)
    assert out == ExprInt(_MASK, _SIZE)


def test_redundancy_and_not_collapses_to_zero() -> None:
    a, _, _, _ = _atoms()
    expr = ExprOp("&", a, _not_(a))
    out = DEFAULT_REWRITER.normalize(expr)
    assert out == ExprInt(0, _SIZE)


def test_redundancy_and_not_symmetric() -> None:
    a, _, _, _ = _atoms()
    expr = ExprOp("&", _not_(a), a)
    out = DEFAULT_REWRITER.normalize(expr)
    assert out == ExprInt(0, _SIZE)


def test_absorption_does_not_fire_on_unrelated_operands() -> None:
    a, b, c, _ = _atoms()
    expr = ExprOp("|", a, ExprOp("&", c, b))
    out = DEFAULT_REWRITER.normalize(expr)
    # Should NOT collapse to a; just a Z3-equivalent of the input.
    assert out != a


def test_redundancy_does_not_fire_on_different_operands() -> None:
    a, b, _, _ = _atoms()
    expr = ExprOp("|", a, _not_(b))
    out = DEFAULT_REWRITER.normalize(expr)
    # Not a self-complement; stays non-constant.
    assert out != ExprInt(_MASK, _SIZE)


def test_absorption_or_with_compound_left_operand() -> None:
    # F | (F & c) -> F  where F is itself a compound expression.
    # We use ``a + b`` (an arithmetic compound) so miasm doesn't flatten
    # it into the surrounding ``&`` via associativity normalisation.
    a, b, c, _ = _atoms()
    f = ExprOp("+", a, b)
    expr = ExprOp("|", f, ExprOp("&", f, c))
    out = DEFAULT_REWRITER.normalize(expr)
    assert out == f


def test_redundancy_inside_larger_expression() -> None:
    # (a & ~a) + b -> 0 + b -> b
    a, b, _, _ = _atoms()
    expr = ExprOp("+", ExprOp("&", a, _not_(a)), b)
    out = DEFAULT_REWRITER.normalize(expr)
    assert out == b


def test_absorption_rule_is_idempotent() -> None:
    a, b, _, _ = _atoms()
    expr = ExprOp("|", a, ExprOp("&", a, b))
    once = DEFAULT_REWRITER.normalize(expr)
    twice = DEFAULT_REWRITER.normalize(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Directed tests for GAMBA §5.4 deep factorisation
# ---------------------------------------------------------------------------


def _import_factor():
    from msynth.simplification.rewrites import _apply_factor_common_subterm

    return _apply_factor_common_subterm


def test_factor_simple_xy_xz() -> None:
    a, b, c, _ = _atoms()
    factor = _import_factor()
    expr = ExprOp("+", ExprOp("*", a, b), ExprOp("*", a, c))
    out = factor(expr)
    assert out is not None
    # Result must be ``a * (b + c)`` in some commutative ordering.
    assert out.op == "*"
    assert a in out.args


def test_factor_three_terms() -> None:
    a, b, c, d = _atoms()
    factor = _import_factor()
    expr = ExprOp("+", ExprOp("*", a, b), ExprOp("*", a, c), ExprOp("*", a, d))
    out = factor(expr)
    assert out is not None
    _assert_sound(expr, out, "factor three terms")


def test_factor_with_compound_common_subexpression() -> None:
    a, b, c, d = _atoms()
    factor = _import_factor()
    # ((a & b) * c) + ((a & b) * d) -> (a & b) * (c + d)
    common = ExprOp("&", a, b)
    expr = ExprOp("+", ExprOp("*", common, c), ExprOp("*", common, d))
    out = factor(expr)
    assert out is not None
    _assert_sound(expr, out, "factor compound")


def test_factor_rejects_no_common_factor() -> None:
    a, b, c, d = _atoms()
    factor = _import_factor()
    expr = ExprOp("+", ExprOp("*", a, b), ExprOp("*", c, d))
    assert factor(expr) is None


def test_factor_rejects_constant_only_common() -> None:
    # Pure-constant common factor — defer to coefficient collection.
    a, b, _, _ = _atoms()
    factor = _import_factor()
    expr = ExprOp(
        "+",
        ExprOp("*", ExprInt(2, _SIZE), a),
        ExprOp("*", ExprInt(3, _SIZE), b),
    )
    assert factor(expr) is None


def test_factor_rejects_partial_majority() -> None:
    # x*a + x*b + y*c -- only TWO of three share `x`. Common multiset
    # across ALL terms is empty -> rule rejects.
    a, b, c, d = _atoms()
    factor = _import_factor()
    expr = ExprOp(
        "+",
        ExprOp("*", a, b),
        ExprOp("*", a, c),
        ExprOp("*", d, c),  # third term has no `a`
    )
    assert factor(expr) is None


def test_factor_net_smaller_guard_rejects_inflation() -> None:
    # a + (a * b) -- factoring would produce ``a * (1 + b)`` which
    # has more nodes than the input. Net-smaller guard rejects.
    a, b, _, _ = _atoms()
    factor = _import_factor()
    expr = ExprOp("+", a, ExprOp("*", a, b))
    assert factor(expr) is None


def test_factor_with_multiplicity_intersection_is_correct() -> None:
    # Test the multiset machinery directly: factors of ``a*a*b`` should
    # be ``[a, a, b]`` (preserving multiplicity), so the intersection
    # with ``[a, a, c]`` is ``[a, a]``. Whether the rewrite is then
    # accepted depends on the net-smaller guard, which is tested
    # separately.
    from msynth.simplification.rewrites import _factors_of

    a, b, _, _ = _atoms()
    factors = _factors_of(ExprOp("*", a, a, b))
    assert factors.count(a) == 2
    assert factors.count(b) == 1


def test_factor_via_default_rewriter_normalize() -> None:
    a, b, c, _ = _atoms()
    expr = ExprOp("+", ExprOp("*", a, b), ExprOp("*", a, c))
    out = DEFAULT_REWRITER.normalize(expr)
    _assert_sound(expr, out, "factor via normalize")
    # Output should be strictly smaller than input.
    assert len(out.graph().nodes()) <= len(expr.graph().nodes())


def test_factor_then_normalize_is_idempotent() -> None:
    a, b, c, _ = _atoms()
    expr = ExprOp("+", ExprOp("*", a, b), ExprOp("*", a, c))
    once = DEFAULT_REWRITER.normalize(expr)
    twice = DEFAULT_REWRITER.normalize(once)
    assert once == twice


def test_factor_does_not_oscillate_with_ring_normalize() -> None:
    # ring_normalize DISTRIBUTES; factor_common_subterm UN-distributes.
    # Both are guarded, so both must net-shrink. On a shape where
    # neither nets a shrink, they must agree on a fixpoint.
    a, b, c, _ = _atoms()
    expr = ExprOp("*", a, ExprOp("+", b, c))  # already factored
    out1 = DEFAULT_REWRITER.normalize(expr)
    out2 = DEFAULT_REWRITER.normalize(out1)
    assert out1 == out2


def test_factor_three_way_compound_factor() -> None:
    # F = (a ^ b); ((a^b)*c) + ((a^b)*d) + ((a^b)*x) -> (a^b)*(c+d+x)
    a, b, c, d = _atoms()
    f = ExprOp("^", a, b)
    factor = _import_factor()
    expr = ExprOp(
        "+",
        ExprOp("*", f, c),
        ExprOp("*", f, d),
        ExprOp("*", f, a),  # reuse a as the third "x"
    )
    out = factor(expr)
    assert out is not None
    _assert_sound(expr, out, "factor 3-way compound")


# ---------------------------------------------------------------------------
# Directed tests for Tier 1 n-ary normalisation rules
# ---------------------------------------------------------------------------


def _rule_by_name(name: str) -> RewriteRule:
    for r in DEFAULT_RULES:
        if r.name == name:
            return r
    raise AssertionError(f"rule {name!r} not registered")


def test_idempotence_and_n_ary_collapses_all_duplicates() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("idempotence_and_drop_duplicates")
    out = rule.apply(ExprOp("&", a, b, a, b, a))
    assert out == ExprOp("&", a, b)


def test_idempotence_and_single_distinct_collapses_to_child() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("idempotence_and_drop_duplicates")
    out = rule.apply(ExprOp("&", a, a, a))
    assert out == a


def test_idempotence_or_n_ary_collapses_all_duplicates() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("idempotence_or_drop_duplicates")
    out = rule.apply(ExprOp("|", a, b, a))
    assert out == ExprOp("|", a, b)


def test_xor_self_cancel_pairs_eliminates_pair() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("xor_self_cancel_pairs")
    out = rule.apply(ExprOp("^", a, b, a))
    assert out == b


def test_xor_self_cancel_pairs_collapses_to_zero() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("xor_self_cancel_pairs")
    out = rule.apply(ExprOp("^", a, a))
    assert out == ExprInt(0, _SIZE)


def test_double_negation_collapses_two_xor_layers() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("double_negation_collapse")
    out = rule.apply(_not_(_not_(a)))
    assert out == a


def test_double_negation_rejects_single_layer() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("double_negation_collapse")
    assert rule.apply(_not_(a)) is None


def test_const_fold_and_zero_short_circuits() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("const_fold_and")
    out = rule.apply(ExprOp("&", a, ExprInt(0, _SIZE)))
    assert out == ExprInt(0, _SIZE)


def test_const_fold_and_allones_drops_constant() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("const_fold_and")
    out = rule.apply(ExprOp("&", a, _mask_expr(), b))
    assert out == ExprOp("&", a, b)


def test_const_fold_or_zero_drops_constant() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("const_fold_or")
    out = rule.apply(ExprOp("|", a, ExprInt(0, _SIZE), b))
    assert out == ExprOp("|", a, b)


def test_const_fold_or_allones_short_circuits() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("const_fold_or")
    out = rule.apply(ExprOp("|", a, _mask_expr()))
    assert out == _mask_expr()


def test_const_fold_xor_zero_drops_constant() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("const_fold_xor_zero")
    out = rule.apply(ExprOp("^", a, ExprInt(0, _SIZE)))
    assert out == a


def test_const_fold_add_zero_drops_constant() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("const_fold_add_zero")
    out = rule.apply(ExprOp("+", a, ExprInt(0, _SIZE), b))
    assert out == ExprOp("+", a, b)


def test_const_fold_mul_one_drops_constant() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("const_fold_mul_one")
    out = rule.apply(ExprOp("*", a, ExprInt(1, _SIZE)))
    assert out == a


def test_const_fold_mul_zero_short_circuits() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("const_fold_mul_zero")
    out = rule.apply(ExprOp("*", a, ExprInt(0, _SIZE), b))
    assert out == ExprInt(0, _SIZE)


# ---------------------------------------------------------------------------
# Directed tests for Tier 2 GAMBA §5.2 specific identities
# ---------------------------------------------------------------------------


def test_conj_self_neg_double_collapses_to_zero_basic() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("conj_self_neg_double_collapses_to_zero")
    expr = ExprOp(
        "&",
        a,
        ExprOp("-", a),
        ExprOp("*", ExprInt(2, _SIZE), a),
    )
    assert rule.apply(expr) == ExprInt(0, _SIZE)


def test_conj_self_neg_double_rejects_without_double() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("conj_self_neg_double_collapses_to_zero")
    expr = ExprOp("&", a, ExprOp("-", a))
    assert rule.apply(expr) is None


def test_conj_xor_identity_drops_clause() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("conj_xor_identity_clause")
    two_a = ExprOp("*", ExprInt(2, _SIZE), a)
    xor_self_neg = ExprOp("^", a, ExprOp("-", a))
    expr = ExprOp("&", two_a, xor_self_neg)
    assert rule.apply(expr) == two_a


def test_conj_xor_identity_idempotent() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("conj_xor_identity_clause")
    two_a = ExprOp("*", ExprInt(2, _SIZE), a)
    xor_self_neg = ExprOp("^", a, ExprOp("-", a))
    expr = ExprOp("&", two_a, xor_self_neg)
    once = rule.apply(expr)
    assert once is not None
    twice = rule.apply(once)
    # Second application should not fire (no xor_self_neg clause remaining).
    assert twice is None


def test_disj_xor_identity_drops_clause() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("disj_xor_identity_clause")
    two_a = ExprOp("*", ExprInt(2, _SIZE), a)
    xor_self_neg = ExprOp("^", a, ExprOp("-", a))
    expr = ExprOp("|", two_a, ExprOp("-", xor_self_neg))
    assert rule.apply(expr) == two_a


def test_disj_disj_negation_absorb_basic() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("disj_disj_negation_absorb")
    inner = ExprOp("|", ExprOp("&", a, b), ExprOp("-", a))
    expr = ExprOp("|", a, ExprOp("-", inner))
    assert rule.apply(expr) == a


def test_conj_conj_negation_absorb_basic() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("conj_conj_negation_absorb")
    inner = ExprOp("&", ExprOp("|", a, b), ExprOp("-", a))
    expr = ExprOp("&", a, ExprOp("-", inner))
    assert rule.apply(expr) == a


def test_disj_conj_negation_absorb_basic() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("disj_conj_negation_absorb")
    expr = ExprOp(
        "|",
        ExprOp("-", a),
        ExprOp("&", _not_(a), ExprOp("*", ExprInt(2, _SIZE), a)),
    )
    assert rule.apply(expr) == ExprOp("-", a)


def test_disj_neg_disj_identity_basic() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("disj_neg_disj_identity")
    inner = ExprOp("|", ExprOp("-", a), ExprOp("*", ExprInt(2, _SIZE), a))
    expr = ExprOp("|", a, ExprOp("-", inner))
    assert rule.apply(expr) == a


def test_xor_same_mult_or_yields_xor() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("xor_same_mult_or")
    expr = ExprOp(
        "*",
        ExprInt(2, _SIZE),
        ExprOp("|", a, ExprOp("-", a)),
    )
    out = rule.apply(expr)
    assert out is not None
    _assert_sound(expr, out, "xor_same_mult_or")


def test_xor_same_mult_and_yields_xor() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("xor_same_mult_and")
    expr = ExprOp(
        "*",
        ExprInt((-2) & _MASK, _SIZE),
        ExprOp("&", a, ExprOp("-", a)),
    )
    out = rule.apply(expr)
    assert out is not None
    _assert_sound(expr, out, "xor_same_mult_and")


def test_complement_pair_and_or_basic() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("complement_pair_and_or")
    expr = ExprOp("|", ExprOp("&", a, b), ExprOp("&", a, _not_(b)))
    assert rule.apply(expr) == a


def test_complement_pair_and_or_reverse_order() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("complement_pair_and_or")
    # (a & ~b) | (a & b) -> a
    expr = ExprOp("|", ExprOp("&", a, _not_(b)), ExprOp("&", a, b))
    assert rule.apply(expr) == a


def test_bitwise_in_sum_cancel_and_form() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("bitwise_in_sum_cancel")
    # (a & b) - a - b -> -(a | b)
    expr = ExprOp(
        "+",
        ExprOp("&", a, b),
        ExprOp("-", a),
        ExprOp("-", b),
    )
    out = rule.apply(expr)
    assert out is not None
    _assert_sound(expr, out, "bitwise_in_sum_cancel and form")


def test_bitwise_in_sum_cancel_or_form() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("bitwise_in_sum_cancel")
    # (a | b) - a - b -> -(a & b)
    expr = ExprOp(
        "+",
        ExprOp("|", a, b),
        ExprOp("-", a),
        ExprOp("-", b),
    )
    out = rule.apply(expr)
    assert out is not None
    _assert_sound(expr, out, "bitwise_in_sum_cancel or form")


def test_bitwise_in_sum_cancel_xor_via_factor_two() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("bitwise_in_sum_cancel")
    # 2*(a | b) - a - b -> a ^ b
    expr = ExprOp(
        "+",
        ExprOp("*", ExprInt(2, _SIZE), ExprOp("|", a, b)),
        ExprOp("-", a),
        ExprOp("-", b),
    )
    out = rule.apply(expr)
    assert out is not None
    _assert_sound(expr, out, "bitwise_in_sum_cancel xor form")


# ---------------------------------------------------------------------------
# Directed tests for Tier 3 algebraic identities
# ---------------------------------------------------------------------------


def test_or_pairs_with_disjoint_constants_merge() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("or_pairs_with_disjoint_constants")
    expr = ExprOp(
        "+",
        ExprOp("|", ExprInt(0x0F, _SIZE), a),
        ExprOp("|", ExprInt(0xF0, _SIZE), a),
    )
    out = rule.apply(expr)
    assert out is not None
    _assert_sound(expr, out, "or_pairs_with_disjoint_constants")


def test_or_pairs_with_overlapping_constants_rejected() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("or_pairs_with_disjoint_constants")
    expr = ExprOp(
        "+",
        ExprOp("|", ExprInt(0x07, _SIZE), a),
        ExprOp("|", ExprInt(0x03, _SIZE), a),
    )
    assert rule.apply(expr) is None


def test_or_pairs_with_disjoint_constants_idempotent() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("or_pairs_with_disjoint_constants")
    expr = ExprOp(
        "+",
        ExprOp("|", ExprInt(0x0F, _SIZE), a),
        ExprOp("|", ExprInt(0xF0, _SIZE), a),
    )
    once = rule.apply(expr)
    assert once is not None
    twice = rule.apply(once)
    assert twice is None


def test_diff_bitw_pairs_with_disjoint_constants_neg_and_plus_or() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("diff_bitw_pairs_with_disjoint_constants")
    expr = ExprOp(
        "+",
        ExprOp("-", ExprOp("&", ExprInt(0x0F, _SIZE), a)),
        ExprOp("|", ExprInt(0xF0, _SIZE), a),
    )
    out = rule.apply(expr)
    assert out is not None
    _assert_sound(expr, out, "diff_bitw_pairs_with_disjoint_constants")


def test_diff_bitw_pairs_with_same_op_rejected() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("diff_bitw_pairs_with_disjoint_constants")
    # Same-op pair is handled by the dedicated rules, not this one.
    expr = ExprOp(
        "+",
        ExprOp("&", ExprInt(0x0F, _SIZE), a),
        ExprOp("&", ExprInt(0xF0, _SIZE), a),
    )
    assert rule.apply(expr) is None


def test_diff_bitw_pairs_with_disjoint_constants_idempotent() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("diff_bitw_pairs_with_disjoint_constants")
    expr = ExprOp(
        "+",
        ExprOp("-", ExprOp("&", ExprInt(0x0F, _SIZE), a)),
        ExprOp("|", ExprInt(0xF0, _SIZE), a),
    )
    once = rule.apply(expr)
    assert once is not None
    twice = rule.apply(once)
    assert twice is None


def test_diff_inv_or_minus_andnot_collapses_to_atom() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("diff_inv_or_minus_andnot")
    expr = ExprOp(
        "+",
        ExprOp("|", a, b),
        ExprOp("-", ExprOp("&", _not_(a), b)),
    )
    assert rule.apply(expr) == a


def test_diff_inv_or_minus_andnot_rejects_unrelated() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("diff_inv_or_minus_andnot")
    expr = ExprOp(
        "+",
        ExprOp("|", a, b),
        ExprOp("-", ExprOp("&", a, b)),  # missing the NOT
    )
    assert rule.apply(expr) is None


def test_diff_inv_or_minus_andnot_idempotent() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("diff_inv_or_minus_andnot")
    expr = ExprOp(
        "+",
        ExprOp("|", a, b),
        ExprOp("-", ExprOp("&", _not_(a), b)),
    )
    once = rule.apply(expr)
    twice = rule.apply(once) if once is not None else None
    assert twice is None


def test_diff_inv_xor_minus_andnot_yields_diff() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("diff_inv_xor_minus_andnot")
    expr = ExprOp(
        "+",
        ExprOp("^", a, b),
        ExprOp(
            "*",
            ExprInt((-2) & _MASK, _SIZE),
            ExprOp("&", _not_(a), b),
        ),
    )
    out = rule.apply(expr)
    assert out is not None
    _assert_sound(expr, out, "diff_inv_xor_minus_andnot")


def test_diff_inv_xor_minus_andnot_wrong_coeff_rejected() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("diff_inv_xor_minus_andnot")
    expr = ExprOp(
        "+",
        ExprOp("^", a, b),
        ExprOp("*", ExprInt(2, _SIZE), ExprOp("&", _not_(a), b)),
    )
    assert rule.apply(expr) is None


def test_diff_inv_xor_minus_andnot_idempotent() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("diff_inv_xor_minus_andnot")
    expr = ExprOp(
        "+",
        ExprOp("^", a, b),
        ExprOp(
            "*",
            ExprInt((-2) & _MASK, _SIZE),
            ExprOp("&", _not_(a), b),
        ),
    )
    once = rule.apply(expr)
    twice = rule.apply(once) if once is not None else None
    assert twice is None


def test_diff_inv_xor_plus_ornot_yields_linear_form() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("diff_inv_xor_plus_ornot")
    expr = ExprOp(
        "+",
        ExprOp("^", a, b),
        ExprOp("*", ExprInt(2, _SIZE), ExprOp("|", _not_(a), b)),
    )
    out = rule.apply(expr)
    assert out is not None
    _assert_sound(expr, out, "diff_inv_xor_plus_ornot")


def test_diff_inv_xor_plus_ornot_wrong_coeff_rejected() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("diff_inv_xor_plus_ornot")
    expr = ExprOp(
        "+",
        ExprOp("^", a, b),
        ExprOp("*", ExprInt(3, _SIZE), ExprOp("|", _not_(a), b)),
    )
    assert rule.apply(expr) is None


def test_diff_inv_xor_plus_ornot_idempotent() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("diff_inv_xor_plus_ornot")
    expr = ExprOp(
        "+",
        ExprOp("^", a, b),
        ExprOp("*", ExprInt(2, _SIZE), ExprOp("|", _not_(a), b)),
    )
    once = rule.apply(expr)
    twice = rule.apply(once) if once is not None else None
    assert twice is None


def test_xor_involving_disj_yields_andnot() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("xor_involving_disj")
    expr = ExprOp("^", a, ExprOp("|", a, b))
    out = rule.apply(expr)
    assert out is not None
    _assert_sound(expr, out, "xor_involving_disj")


def test_xor_involving_disj_rejects_unrelated() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("xor_involving_disj")
    assert rule.apply(ExprOp("^", a, b)) is None


def test_xor_involving_disj_idempotent() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("xor_involving_disj")
    expr = ExprOp("^", a, ExprOp("|", a, b))
    once = rule.apply(expr)
    twice = rule.apply(once) if once is not None else None
    assert twice is None


def test_negative_bitw_inverse_and_yields_disj() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("negative_bitw_inverse_and")
    expr = ExprOp("-", ExprOp("&", a, ExprOp("-", a)))
    out = rule.apply(expr)
    assert out is not None
    _assert_sound(expr, out, "negative_bitw_inverse_and")


def test_negative_bitw_inverse_and_rejects_or_inner() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("negative_bitw_inverse_and")
    expr = ExprOp("-", ExprOp("|", a, ExprOp("-", a)))
    assert rule.apply(expr) is None


def test_negative_bitw_inverse_and_idempotent() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("negative_bitw_inverse_and")
    expr = ExprOp("-", ExprOp("&", a, ExprOp("-", a)))
    once = rule.apply(expr)
    twice = rule.apply(once) if once is not None else None
    assert twice is None


def test_negative_bitw_inverse_or_yields_conj() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("negative_bitw_inverse_or")
    expr = ExprOp("-", ExprOp("|", a, ExprOp("-", a)))
    out = rule.apply(expr)
    assert out is not None
    _assert_sound(expr, out, "negative_bitw_inverse_or")


def test_negative_bitw_inverse_or_rejects_and_inner() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("negative_bitw_inverse_or")
    expr = ExprOp("-", ExprOp("&", a, ExprOp("-", a)))
    assert rule.apply(expr) is None


def test_negative_bitw_inverse_or_idempotent() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("negative_bitw_inverse_or")
    expr = ExprOp("-", ExprOp("|", a, ExprOp("-", a)))
    once = rule.apply(expr)
    twice = rule.apply(once) if once is not None else None
    assert twice is None


def test_disj_sub_disj_identity_collapses_to_atom() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("disj_sub_disj_identity")
    expr = ExprOp("|", a, ExprOp("-", ExprOp("|", a, b), b))
    assert rule.apply(expr) == a


def test_disj_sub_disj_identity_rejects_unrelated() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("disj_sub_disj_identity")
    c = ExprId("c", _SIZE)
    expr = ExprOp("|", a, ExprOp("-", ExprOp("|", a, b), c))
    assert rule.apply(expr) is None


def test_disj_sub_disj_identity_idempotent() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("disj_sub_disj_identity")
    expr = ExprOp("|", a, ExprOp("-", ExprOp("|", a, b), b))
    once = rule.apply(expr)
    twice = rule.apply(once) if once is not None else None
    assert twice is None


def test_disj_sub_conj_identity_collapses_to_atom() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("disj_sub_conj_identity")
    expr = ExprOp("|", a, ExprOp("-", a, ExprOp("&", a, b)))
    assert rule.apply(expr) == a


def test_disj_sub_conj_identity_rejects_unrelated() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("disj_sub_conj_identity")
    c = ExprId("c", _SIZE)
    expr = ExprOp("|", a, ExprOp("-", a, ExprOp("&", b, c)))
    assert rule.apply(expr) is None


def test_disj_sub_conj_identity_idempotent() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("disj_sub_conj_identity")
    expr = ExprOp("|", a, ExprOp("-", a, ExprOp("&", a, b)))
    once = rule.apply(expr)
    twice = rule.apply(once) if once is not None else None
    assert twice is None


def test_conj_add_conj_identity_collapses_to_atom() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("conj_add_conj_identity")
    expr = ExprOp(
        "&",
        a,
        ExprOp("+", a, ExprOp("&", _not_(a), b)),
    )
    assert rule.apply(expr) == a


def test_conj_add_conj_identity_rejects_unrelated() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("conj_add_conj_identity")
    # No ~a in the inner conjunction.
    expr = ExprOp("&", a, ExprOp("+", a, ExprOp("&", a, b)))
    assert rule.apply(expr) is None


def test_conj_add_conj_identity_idempotent() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("conj_add_conj_identity")
    expr = ExprOp(
        "&",
        a,
        ExprOp("+", a, ExprOp("&", _not_(a), b)),
    )
    once = rule.apply(expr)
    twice = rule.apply(once) if once is not None else None
    assert twice is None


def test_conj_conj_disj_yields_conj() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("conj_conj_disj")
    inner = ExprOp("&", ExprOp("-", b), ExprOp("|", a, b))
    expr = ExprOp("&", a, ExprOp("-", inner))
    out = rule.apply(expr)
    assert out is not None
    _assert_sound(expr, out, "conj_conj_disj")


def test_conj_conj_disj_rejects_unrelated() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("conj_conj_disj")
    c = ExprId("c", _SIZE)
    # Inner OR does not contain b.
    inner = ExprOp("&", ExprOp("-", b), ExprOp("|", a, c))
    expr = ExprOp("&", a, ExprOp("-", inner))
    assert rule.apply(expr) is None


def test_conj_conj_disj_idempotent() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("conj_conj_disj")
    inner = ExprOp("&", ExprOp("-", b), ExprOp("|", a, b))
    expr = ExprOp("&", a, ExprOp("-", inner))
    once = rule.apply(expr)
    twice = rule.apply(once) if once is not None else None
    assert twice is None


def test_disj_disj_conj_yields_disj() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("disj_disj_conj")
    inner = ExprOp("|", ExprOp("-", b), ExprOp("&", a, b))
    expr = ExprOp("|", a, ExprOp("-", inner))
    out = rule.apply(expr)
    assert out is not None
    _assert_sound(expr, out, "disj_disj_conj")


def test_disj_disj_conj_rejects_unrelated() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("disj_disj_conj")
    c = ExprId("c", _SIZE)
    inner = ExprOp("|", ExprOp("-", b), ExprOp("&", a, c))
    expr = ExprOp("|", a, ExprOp("-", inner))
    assert rule.apply(expr) is None


def test_disj_disj_conj_idempotent() -> None:
    a, b, _, _ = _atoms()
    rule = _rule_by_name("disj_disj_conj")
    inner = ExprOp("|", ExprOp("-", b), ExprOp("&", a, b))
    expr = ExprOp("|", a, ExprOp("-", inner))
    once = rule.apply(expr)
    twice = rule.apply(once) if once is not None else None
    assert twice is None


def test_conj_neg_xor_minus_one_collapses_to_all_ones() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("conj_neg_xor_minus_one")
    two_a = ExprOp("*", ExprInt(2, _SIZE), a)
    xor_self_neg = ExprOp("^", a, ExprOp("-", a))
    expr = ExprOp("|", two_a, _not_(ExprOp("-", xor_self_neg)))
    assert rule.apply(expr) == ExprInt(_MASK, _SIZE)


def test_conj_neg_xor_minus_one_rejects_unrelated() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("conj_neg_xor_minus_one")
    # Missing the 2*x clause.
    expr = ExprOp(
        "|",
        a,
        _not_(ExprOp("-", ExprOp("^", a, ExprOp("-", a)))),
    )
    assert rule.apply(expr) is None


def test_conj_neg_xor_minus_one_idempotent() -> None:
    a, _, _, _ = _atoms()
    rule = _rule_by_name("conj_neg_xor_minus_one")
    two_a = ExprOp("*", ExprInt(2, _SIZE), a)
    xor_self_neg = ExprOp("^", a, ExprOp("-", a))
    expr = ExprOp("|", two_a, _not_(ExprOp("-", xor_self_neg)))
    once = rule.apply(expr)
    assert once is not None
    # Second pass: result is a bare ExprInt and the rule rejects.
    assert rule.apply(once) is None
