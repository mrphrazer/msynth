from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pytest
from miasm.expression.expression import (
    Expr,
    ExprCompose,
    ExprCond,
    ExprId,
    ExprInt,
    ExprMem,
    ExprOp,
    ExprSlice,
    TOK_EQUAL,
    TOK_INF_EQUAL_SIGNED,
    TOK_INF_EQUAL_UNSIGNED,
    TOK_INF_SIGNED,
    TOK_INF_UNSIGNED,
)
from miasm.expression.simplifications import expr_simp

from msynth.utils.expr_utils import compile_expr_to_python


def miasm_eval(expr: Expr, inputs: List[int]) -> int:
    replacements: Dict[Expr, Expr] = {}
    vars_set = set()

    def add_vars(e: Expr) -> Expr:
        if e.is_id():
            vars_set.add(e)
        return e

    expr.visit(add_vars)
    for v in vars_set:
        if not v.name.startswith("p") or not v.name[1:].isdigit():
            raise ValueError(f"Unexpected variable name: {v.name}")
        idx = int(v.name[1:])
        replacements[v] = ExprInt(inputs[idx], v.size)
    return int(expr_simp(expr.replace_expr(replacements)))


def build_cases() -> List[Dict[str, Any]]:
    p0_32 = ExprId("p0", 32)
    p1_32 = ExprId("p1", 32)
    p0_8 = ExprId("p0", 8)
    p1_8 = ExprId("p1", 8)
    p0_64 = ExprId("p0", 64)
    p1_64 = ExprId("p1", 64)

    cases: List[Dict[str, Any]] = []

    def add_case(
        name: str,
        expr: Expr,
        inputs: List[int],
        expected_fn: Optional[Callable[[Expr, List[int]], int]] = None,
        allow_compile_error: bool = False,
    ) -> None:
        cases.append(
            {
                "name": name,
                "expr": expr,
                "inputs": inputs,
                "expected_fn": expected_fn,
                "allow_compile_error": allow_compile_error,
            }
        )

    add_case(
        "sdiv_neg_pos",
        ExprOp("sdiv", ExprInt(0xFFFF_FFFD, 32), ExprInt(2, 32)),
        [0, 0],
    )
    add_case(
        "sdiv_neg_neg",
        ExprOp("sdiv", ExprInt(0xFFFF_FFFD, 32), ExprInt(0xFFFF_FFFE, 32)),
        [0, 0],
    )
    add_case(
        "smod_neg_pos",
        ExprOp("smod", ExprInt(0xFFFF_FFFD, 32), ExprInt(2, 32)),
        [0, 0],
    )
    add_case(
        "smod_neg_neg",
        ExprOp("smod", ExprInt(0xFFFF_FFFD, 32), ExprInt(0xFFFF_FFFE, 32)),
        [0, 0],
    )
    add_case(
        "cmp_signed",
        ExprOp("<s", ExprInt(0xFFFF_FFFF, 32), ExprInt(1, 32)),
        [0, 0],
    )
    add_case(
        "compose_simple",
        ExprCompose(ExprInt(0x34, 8), ExprInt(0x12, 8)),
        [0, 0],
    )
    add_case(
        "compose_vars",
        ExprCompose(ExprSlice(p0_32, 0, 8), ExprSlice(p1_32, 0, 8)),
        [0xAA55AA55, 0x12345678],
    )
    cond_expr = ExprCond(
        ExprSlice(p0_32, 0, 1), ExprInt(0x11111111, 32), ExprInt(0x22222222, 32)
    )
    add_case("cond_lowbit", cond_expr, [1, 0])
    add_case("cond_lowbit_zero", cond_expr, [0, 0])
    add_case(
        "shift_gt_size",
        ExprOp(">>", ExprInt(0xFFFFFFFF, 32), ExprInt(40, 32)),
        [0, 0],
    )
    add_case(
        "ashr_negative",
        ExprOp("a>>", ExprInt(0x8000_0000, 32), ExprInt(4, 32)),
        [0, 0],
    )
    add_case("slice_var", ExprSlice(p0_32, 4, 12), [0xF0F0F0F0, 0])
    add_case("parity_lowbyte", ExprOp("parity", ExprInt(0b1010_0001, 8)), [0, 0])
    add_case("clz", ExprOp("cntleadzeros", ExprInt(0x0000_0100, 32)), [0, 0])
    add_case("ctz", ExprOp("cnttrailzeros", ExprInt(0x0000_0100, 32)), [0, 0])
    add_case(
        "compose_slice_mixed",
        ExprCompose(ExprSlice(p0_32, 0, 8), ExprSlice(p0_32, 8, 16)),
        [0x11223344, 0],
    )
    add_case("add", ExprOp("+", p0_32, p1_32), [0x11111111, 0x22222222])
    add_case("sub", ExprOp("-", p0_32, p1_32), [0x11111111, 0x22222222])
    add_case("mul", ExprOp("*", p0_32, p1_32), [0x1234, 0x20])
    add_case("and", ExprOp("&", p0_32, p1_32), [0xF0F0F0F0, 0x0FF00FF0])
    add_case("or", ExprOp("|", p0_32, p1_32), [0xF0F0F0F0, 0x0FF00FF0])
    add_case("xor", ExprOp("^", p0_32, p1_32), [0xF0F0F0F0, 0x0FF00FF0])
    add_case("shl", ExprOp("<<", p0_32, ExprInt(4, 32)), [0x12345678, 0])
    add_case("shr", ExprOp(">>", p0_32, ExprInt(4, 32)), [0x12345678, 0])
    add_case("ashr_var", ExprOp("a>>", p0_32, ExprInt(1, 32)), [0x80000001, 0])
    add_case("rotl", ExprOp("<<<", p0_32, ExprInt(8, 32)), [0x12345678, 0])
    add_case("rotr", ExprOp(">>>", p0_32, ExprInt(8, 32)), [0x12345678, 0])
    add_case("eq", ExprOp("==", p0_32, p1_32), [0x1234, 0x1234])
    add_case("ne_eq", ExprOp("==", p0_32, p1_32), [0x1234, 0x5678])
    add_case("lt_u", ExprOp("<u", p0_32, p1_32), [1, 2])
    add_case("lte_u", ExprOp("<=u", p0_32, p1_32), [2, 2])
    add_case(
        "lt_s",
        ExprOp("<s", ExprInt(0xFFFF_FFFF, 32), ExprInt(1, 32)),
        [0, 0],
    )
    add_case(
        "lte_s",
        ExprOp("<=s", ExprInt(0xFFFF_FFFF, 32), ExprInt(0xFFFF_FFFF, 32)),
        [0, 0],
    )
    add_case("tok_eq", ExprOp(TOK_EQUAL, p0_32, p1_32), [1, 1])
    add_case(
        "tok_lt_s",
        ExprOp(TOK_INF_SIGNED, ExprInt(0xFFFF_FFFF, 32), ExprInt(1, 32)),
        [0, 0],
    )
    add_case(
        "tok_lt_u",
        ExprOp(TOK_INF_UNSIGNED, ExprInt(1, 32), ExprInt(2, 32)),
        [0, 0],
    )
    add_case(
        "tok_lte_s",
        ExprOp(
            TOK_INF_EQUAL_SIGNED,
            ExprInt(0xFFFF_FFFF, 32),
            ExprInt(0xFFFF_FFFF, 32),
        ),
        [0, 0],
    )
    add_case(
        "tok_lte_u",
        ExprOp(TOK_INF_EQUAL_UNSIGNED, ExprInt(1, 32), ExprInt(1, 32)),
        [0, 0],
    )
    add_case("udiv", ExprOp("udiv", ExprInt(10, 32), ExprInt(3, 32)), [0, 0])
    add_case("umod", ExprOp("umod", ExprInt(10, 32), ExprInt(3, 32)), [0, 0])
    add_case(
        "div_signed",
        ExprOp("/", ExprInt(0xFFFF_FFFD, 32), ExprInt(2, 32)),
        [0, 0],
    )
    add_case(
        "mod_signed",
        ExprOp("%", ExprInt(0xFFFF_FFFD, 32), ExprInt(2, 32)),
        [0, 0],
    )
    add_case("zeroext", ExprOp("zeroExt_16", ExprInt(0xAB, 8)), [0, 0])
    add_case("signext_pos", ExprOp("signExt_16", ExprInt(0x7F, 8)), [0, 0])
    add_case("signext_neg", ExprOp("signExt_16", ExprInt(0x80, 8)), [0, 0])
    # bitwise not is intentionally excluded from these tests
    add_case("unary_neg", ExprOp("-", ExprInt(1, 8)), [0, 0])
    add_case("signext_mask_src", ExprOp("signExt_16", ExprInt(0x1FF, 8)), [0, 0])
    add_case("zeroext_mask_src", ExprOp("zeroExt_16", ExprInt(0x1FF, 8)), [0, 0])
    add_case(
        "sdiv_large_64",
        ExprOp("sdiv", ExprInt(0x8000_0000_0000_0001, 64), ExprInt(3, 64)),
        [0, 0],
    )
    add_case(
        "smod_large_64",
        ExprOp("smod", ExprInt(0x8000_0000_0000_0001, 64), ExprInt(3, 64)),
        [0, 0],
    )
    add_case(
        "ashr_shift_ge_size",
        ExprOp("a>>", p0_32, ExprInt(64, 32)),
        [0x80000000, 0],
    )
    add_case(
        "compose_three_args",
        ExprCompose(ExprInt(0xAA, 8), ExprInt(0xBB, 8), ExprInt(0xCC, 8)),
        [0, 0],
    )

    # bitwise not op name is intentionally excluded from these tests

    # additional size variants and signed/unsigned comparisons
    add_case("add_8", ExprOp("+", p0_8, p1_8), [0x12, 0x34])
    add_case("add_16", ExprOp("+", ExprId("p0", 16), ExprId("p1", 16)), [0x1234, 0x00FF])
    add_case("sub_8", ExprOp("-", p0_8, p1_8), [0x01, 0x02])
    add_case("mul_16", ExprOp("*", ExprId("p0", 16), ExprId("p1", 16)), [0x00FF, 0x0003])
    add_case("and_8", ExprOp("&", p0_8, p1_8), [0xF0, 0x0F])
    add_case("or_16", ExprOp("|", ExprId("p0", 16), ExprId("p1", 16)), [0xF0F0, 0x0FF0])
    add_case("xor_64", ExprOp("^", p0_64, p1_64), [0xFFFF_FFFF_0000_0000, 0x0000_0000_FFFF_FFFF])
    add_case("add_64", ExprOp("+", p0_64, p1_64), [0x7FFF_FFFF_FFFF_FFFF, 0x1])
    add_case("sub_64", ExprOp("-", p0_64, p1_64), [0x0, 0x1])
    add_case("mul_64", ExprOp("*", p0_64, p1_64), [0x1234_5678_9ABC_DEF0, 0x10])
    add_case("and_64", ExprOp("&", p0_64, p1_64), [0xFFFF_0000_FFFF_0000, 0x0FFF_FFFF_0FFF_FFFF])
    add_case("or_64", ExprOp("|", p0_64, p1_64), [0x8000_0000_0000_0000, 0x7FFF_FFFF_FFFF_FFFF])
    add_case("shl_64", ExprOp("<<", p0_64, ExprInt(5, 64)), [0x1234_5678_9ABC_DEF0, 0])
    add_case("shr_64", ExprOp(">>", p0_64, ExprInt(5, 64)), [0x8000_0000_0000_0000, 0])
    add_case("ashr_64", ExprOp("a>>", p0_64, ExprInt(5, 64)), [0x8000_0000_0000_0000, 0])
    add_case("rotl_64", ExprOp("<<<", p0_64, ExprInt(17, 64)), [0x1234_5678_9ABC_DEF0, 0])
    add_case("rotr_64", ExprOp(">>>", p0_64, ExprInt(17, 64)), [0x1234_5678_9ABC_DEF0, 0])
    add_case("eq_64", ExprOp("==", p0_64, p1_64), [0x1234, 0x1234])
    add_case("lt_u_64", ExprOp("<u", p0_64, p1_64), [0x1, 0x2])
    add_case("lte_u_64", ExprOp("<=u", p0_64, p1_64), [0x2, 0x2])
    add_case("lt_s_64_neg", ExprOp("<s", ExprInt(0xFFFF_FFFF_FFFF_FFFF, 64), ExprInt(1, 64)), [0, 0])
    add_case("lte_s_64_eq", ExprOp("<=s", ExprInt(0x8000_0000_0000_0000, 64), ExprInt(0x8000_0000_0000_0000, 64)), [0, 0])
    add_case("udiv_64", ExprOp("udiv", ExprInt(10, 64), ExprInt(3, 64)), [0, 0])
    add_case("umod_64", ExprOp("umod", ExprInt(10, 64), ExprInt(3, 64)), [0, 0])
    add_case("sdiv_64", ExprOp("sdiv", ExprInt(0xFFFF_FFFF_FFFF_FFF7, 64), ExprInt(3, 64)), [0, 0])
    add_case("smod_64", ExprOp("smod", ExprInt(0xFFFF_FFFF_FFFF_FFF7, 64), ExprInt(3, 64)), [0, 0])
    add_case("shl_8", ExprOp("<<", p0_8, ExprInt(3, 8)), [0x12, 0])
    add_case("shr_8", ExprOp(">>", p0_8, ExprInt(3, 8)), [0x80, 0])
    add_case("ashr_8", ExprOp("a>>", p0_8, ExprInt(3, 8)), [0x80, 0])
    add_case("rotl_16", ExprOp("<<<", ExprId("p0", 16), ExprInt(9, 16)), [0x1234, 0])
    add_case("rotr_16", ExprOp(">>>", ExprId("p0", 16), ExprInt(9, 16)), [0x1234, 0])
    add_case("eq_8", ExprOp("==", p0_8, p1_8), [0x7F, 0x7F])
    add_case("lt_u_8", ExprOp("<u", p0_8, p1_8), [0x01, 0x02])
    add_case("lte_u_8", ExprOp("<=u", p0_8, p1_8), [0x02, 0x02])
    add_case("lt_s_8_neg", ExprOp("<s", ExprInt(0xFF, 8), ExprInt(0x01, 8)), [0, 0])
    add_case("lte_s_8_eq", ExprOp("<=s", ExprInt(0x80, 8), ExprInt(0x80, 8)), [0, 0])
    add_case("udiv_8", ExprOp("udiv", ExprInt(10, 8), ExprInt(3, 8)), [0, 0])
    add_case("umod_8", ExprOp("umod", ExprInt(10, 8), ExprInt(3, 8)), [0, 0])
    add_case("sdiv_8", ExprOp("sdiv", ExprInt(0xF7, 8), ExprInt(0x03, 8)), [0, 0])
    add_case("smod_8", ExprOp("smod", ExprInt(0xF7, 8), ExprInt(0x03, 8)), [0, 0])
    add_case("slice_1bit", ExprSlice(p0_32, 0, 1), [0x1, 0])
    add_case("slice_highbit", ExprSlice(p0_32, 31, 32), [0x80000000, 0])
    add_case("compose_4x4", ExprCompose(ExprSlice(p0_8, 0, 4), ExprSlice(p0_8, 4, 8)), [0xAB, 0])
    add_case("cond_true", ExprCond(ExprInt(1, 1), ExprInt(0x12, 8), ExprInt(0x34, 8)), [0, 0])
    add_case("cond_false", ExprCond(ExprInt(0, 1), ExprInt(0x12, 8), ExprInt(0x34, 8)), [0, 0])

    return cases


@pytest.mark.parametrize("case", build_cases(), ids=lambda c: c["name"])
def test_compile_expr_matches_miasm(case: Dict[str, Any]) -> None:
    expr = case["expr"]
    inputs = case["inputs"]
    expected_fn = case["expected_fn"]
    allow_compile_error = case["allow_compile_error"]

    if expected_fn is None:
        expected = miasm_eval(expr, inputs)
    else:
        expected = expected_fn(expr, inputs)

    if allow_compile_error:
        with pytest.raises(ValueError):
            compile_expr_to_python(expr)
        return

    func = compile_expr_to_python(expr)
    got = func(inputs)
    assert int(got) == int(expected)


def test_compile_masks_variable_inputs() -> None:
    p0_8 = ExprId("p0", 8)
    expr = ExprOp("+", p0_8, ExprInt(1, 8))
    func = compile_expr_to_python(expr)
    # 0x1FF masked to 0xFF -> 0xFF + 1 == 0x100 -> masked to 0
    assert func([0x1FF]) == 0


def test_compile_zeroext_and_signext_sizes() -> None:
    expr_zero = ExprOp("zeroExt_32", ExprInt(0xFF, 8))
    expr_sign_pos = ExprOp("signExt_32", ExprInt(0x7F, 8))
    expr_sign_neg = ExprOp("signExt_32", ExprInt(0x80, 8))
    func_zero = compile_expr_to_python(expr_zero)
    func_pos = compile_expr_to_python(expr_sign_pos)
    func_neg = compile_expr_to_python(expr_sign_neg)
    assert func_zero([0]) == miasm_eval(expr_zero, [0])
    assert func_pos([0]) == miasm_eval(expr_sign_pos, [0])
    assert func_neg([0]) == miasm_eval(expr_sign_neg, [0])


def test_compile_rotate_large_shift() -> None:
    p0_32 = ExprId("p0", 32)
    expr_rotl = ExprOp("<<<", p0_32, ExprInt(40, 32))
    expr_rotr = ExprOp(">>>", p0_32, ExprInt(40, 32))
    func_rotl = compile_expr_to_python(expr_rotl)
    func_rotr = compile_expr_to_python(expr_rotr)
    inputs = [0x12345678]
    assert func_rotl(inputs) == miasm_eval(expr_rotl, inputs)
    assert func_rotr(inputs) == miasm_eval(expr_rotr, inputs)


def test_compile_compose_and_slice_edges() -> None:
    p0_32 = ExprId("p0", 32)
    p1_32 = ExprId("p1", 32)
    expr_full_slice = ExprSlice(p0_32, 0, 32)
    expr_compose = ExprCompose(
        ExprSlice(p0_32, 0, 16), ExprSlice(p1_32, 0, 16)
    )
    inputs = [0xA5A5A5A5, 0x5A5A5A5A]
    func_slice = compile_expr_to_python(expr_full_slice)
    func_comp = compile_expr_to_python(expr_compose)
    assert func_slice(inputs) == miasm_eval(expr_full_slice, inputs)
    assert func_comp(inputs) == miasm_eval(expr_compose, inputs)


def test_compile_conditional_cases() -> None:
    p0_32 = ExprId("p0", 32)
    cond_expr = ExprCond(
        ExprSlice(p0_32, 31, 32), ExprInt(0x11111111, 32), ExprInt(0x22222222, 32)
    )
    func = compile_expr_to_python(cond_expr)
    assert func([0]) == miasm_eval(cond_expr, [0])
    assert func([0x80000000]) == miasm_eval(cond_expr, [0x80000000])


def test_compile_unsupported_variable_name() -> None:
    expr = ExprId("x0", 32)
    with pytest.raises(ValueError):
        compile_expr_to_python(expr)


def test_compile_unknown_op() -> None:
    expr = ExprOp("unknown_op", ExprInt(1, 8), ExprInt(2, 8))
    with pytest.raises(ValueError):
        compile_expr_to_python(expr)


def test_compile_unexpected_arity() -> None:
    expr = ExprOp("+", ExprInt(1, 8))
    with pytest.raises(ValueError):
        compile_expr_to_python(expr)


def test_compile_unsupported_expression_type() -> None:
    expr = ExprMem(ExprId("p0", 32), 32)
    with pytest.raises(ValueError):
        compile_expr_to_python(expr)


def test_compile_randomized_small_expressions() -> None:
    import random

    random.seed(0)
    ops = ["+", "-", "*", "&", "|", "^", "==", "<u", "<=u", "<s", "<=s"]
    shift_ops = ["<<", ">>", "a>>", "<<<", ">>>"]
    sizes = [1, 8, 16, 32, 64]
    for _ in range(300):
        size = random.choice(sizes)
        p0 = ExprId("p0", size)
        p1 = ExprId("p1", size)
        op = random.choice(ops + shift_ops)
        if op in shift_ops:
            shift = random.randint(0, size + 4)
            expr = ExprOp(op, p0, ExprInt(shift, size))
            inputs = [random.getrandbits(size)]
        else:
            expr = ExprOp(op, p0, p1)
            inputs = [random.getrandbits(size), random.getrandbits(size)]
        func = compile_expr_to_python(expr)
        assert func(inputs) == miasm_eval(expr, inputs)


def test_compile_randomized_signed_div_mod() -> None:
    import random

    random.seed(1)
    sizes = [8, 16, 32, 64]
    for _ in range(80):
        size = random.choice(sizes)
        a = ExprInt(random.getrandbits(size), size)
        b_val = random.getrandbits(size) or 1
        b = ExprInt(b_val, size)
        for op in ["sdiv", "smod"]:
            expr = ExprOp(op, a, b)
            func = compile_expr_to_python(expr)
            assert func([0, 0]) == miasm_eval(expr, [0, 0])


def test_compile_mixed_size_ops_and_masks() -> None:
    p0_8 = ExprId("p0", 8)
    p0_16 = ExprId("p0", 16)
    p0_32 = ExprId("p0", 32)
    exprs = [
        ExprOp("+", p0_8, ExprInt(0xFF, 8)),
        ExprOp("-", p0_8, ExprInt(0x80, 8)),
        ExprOp("*", p0_16, ExprInt(0x0101, 16)),
        ExprOp("&", p0_16, ExprInt(0x00FF, 16)),
        ExprOp("|", p0_32, ExprInt(0xFF00FF00, 32)),
        ExprOp("^", p0_32, ExprInt(0x0F0F0F0F, 32)),
    ]
    inputs = [0xFFFF_FFFF]
    for expr in exprs:
        func = compile_expr_to_python(expr)
        assert func(inputs) == miasm_eval(expr, inputs)


def test_compile_shift_edge_sizes() -> None:
    p0_1 = ExprId("p0", 1)
    p0_64 = ExprId("p0", 64)
    exprs = [
        ExprOp("<<", p0_1, ExprInt(1, 1)),
        ExprOp(">>", p0_1, ExprInt(1, 1)),
        ExprOp("a>>", p0_1, ExprInt(1, 1)),
        ExprOp("<<", p0_64, ExprInt(63, 64)),
        ExprOp(">>", p0_64, ExprInt(63, 64)),
        ExprOp("a>>", p0_64, ExprInt(63, 64)),
    ]
    inputs = [0x1]
    for expr in exprs:
        func = compile_expr_to_python(expr)
        assert func(inputs) == miasm_eval(expr, inputs)


def test_compile_randomized_compose_slice_cond() -> None:
    import random

    random.seed(2)
    sizes = [8, 16, 32, 64]
    for _ in range(120):
        size = random.choice(sizes)
        p0 = ExprId("p0", size)
        p1 = ExprId("p1", size)
        slice_start = random.randint(0, size - 1)
        slice_stop = random.randint(slice_start + 1, size)
        slice_expr = ExprSlice(p0, slice_start, slice_stop)
        cond_bit = ExprSlice(p1, 0, 1)
        cond_expr = ExprCond(cond_bit, slice_expr, ExprInt(0, slice_stop - slice_start))
        # compose two slices of equal size if possible
        if size >= 8:
            left = ExprSlice(p0, 0, size // 2)
            right = ExprSlice(p1, 0, size // 2)
            compose_expr = ExprCompose(left, right)
            exprs = [slice_expr, cond_expr, compose_expr]
        else:
            exprs = [slice_expr, cond_expr]
        inputs = [random.getrandbits(size), random.getrandbits(size)]
        for expr in exprs:
            func = compile_expr_to_python(expr)
            assert func(inputs) == miasm_eval(expr, inputs)


def test_compile_randomized_ext_ops() -> None:
    import random

    random.seed(3)
    ext_sizes = [(8, 16), (8, 32), (8, 64), (16, 32), (16, 64), (32, 64)]
    for _ in range(60):
        src, dst = random.choice(ext_sizes)
        val = random.getrandbits(src)
        zero_expr = ExprOp(f"zeroExt_{dst}", ExprInt(val, src))
        sign_expr = ExprOp(f"signExt_{dst}", ExprInt(val, src))
        for expr in [zero_expr, sign_expr]:
            func = compile_expr_to_python(expr)
            assert func([0]) == miasm_eval(expr, [0])

def test_compile_shared_subexpression_or_chain() -> None:
    expr = ExprId("p0", 32)
    for _ in range(25):
        expr = ExprOp("|", expr, expr)
    func = compile_expr_to_python(expr)
    inputs = [0x12345678]
    assert func(inputs) == miasm_eval(expr, inputs)
