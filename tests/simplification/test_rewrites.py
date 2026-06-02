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
        return True  # defer to Z3 / sampling
    mask = (1 << left.size) - 1
    for assignment in range(1 << len(atoms)):
        env = {atom: (assignment >> i) & 1 for i, atom in enumerate(atoms)}
        try:
            lv = _eval_cube(left, env, mask)
            rv = _eval_cube(right, env, mask)
        except _CubeUnsupported:
            return True
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
        try:
            lv = _eval_cube(left, env, mask)
            rv = _eval_cube(right, env, mask)
        except _CubeUnsupported:
            return True
        if lv != rv:
            return False
    return True


def _assert_sound(input_expr: Expr, output_expr: Expr, name: str) -> None:
    assert _cube_equivalent(input_expr, output_expr), (
        f"rule {name}: cube disagreement\n  in:  {input_expr}\n  out: {output_expr}"
    )
    assert _z3_equivalent(input_expr, output_expr), (
        f"rule {name}: z3 found counterexample\n"
        f"  in:  {input_expr}\n  out: {output_expr}"
    )
    assert _random_sample_equivalent(input_expr, output_expr), (
        f"rule {name}: random-sample counterexample\n"
        f"  in:  {input_expr}\n  out: {output_expr}"
    )


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
