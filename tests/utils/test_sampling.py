from __future__ import annotations

from typing import List

import pytest
from miasm.expression.expression import ExprId, ExprInt, ExprOp

import msynth.utils.sampling as sampling


def test_gen_inputs_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = iter([1, 2, 3, 4, 5, 6])

    def fake_get_rand_input() -> int:
        return next(seq)

    monkeypatch.setattr(sampling, "get_rand_input", fake_get_rand_input)
    inputs = sampling.gen_inputs(num_variables=2, num_samples=3)
    assert inputs == [[1, 2], [3, 4], [5, 6]]


def test_get_rand_input_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    # force coin outcomes 0..4 in order, and return deterministic values for each branch.
    coin_values = iter([0, 1, 2, 3, 4])
    value_map = {
        8: 0x12,
        16: 0x1234,
        32: 0x12345678,
        64: 0x1234_5678_9ABC_DEF0,
    }

    def fake_getrandbits(n: int) -> int:
        if n == 8:
            if fake_getrandbits.expect_value:
                fake_getrandbits.expect_value = False
                return value_map[n]
            coin = next(coin_values)
            if coin == 0:
                fake_getrandbits.expect_value = True
            return coin
        return value_map[n]

    fake_getrandbits.expect_value = False

    def fake_choice(vals: List[int]) -> int:
        # use a known special value to validate the special-values path.
        return vals[0]

    monkeypatch.setattr(sampling, "getrandbits", fake_getrandbits)
    monkeypatch.setattr(sampling, "choice", fake_choice)

    assert sampling.get_rand_input() == value_map[8]
    assert sampling.get_rand_input() == value_map[16]
    assert sampling.get_rand_input() == value_map[32]
    assert sampling.get_rand_input() == value_map[64]
    assert sampling.get_rand_input() == sampling.SPECIAL_VALUES[0]


def test_gen_inputs_array_uses_get_rand_input(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = iter([10, 20, 30])

    def fake_get_rand_input() -> int:
        return next(seq)

    monkeypatch.setattr(sampling, "get_rand_input", fake_get_rand_input)

    assert sampling.gen_inputs_array(3) == [10, 20, 30]


def test_gen_inputs_edge_cases() -> None:
    assert sampling.gen_inputs(num_variables=0, num_samples=3) == [[], [], []]
    assert sampling.gen_inputs(num_variables=2, num_samples=0) == []


def test_gen_adversarial_values_includes_wraparound_tail() -> None:
    values = sampling.gen_adversarial_values(8)

    assert 0 in values
    assert 1 in values
    assert 64 in values
    assert 0xFF in values
    assert 0xFE in values
    assert 0xC0 in values


def test_gen_adversarial_inputs_is_linear_in_variables() -> None:
    x = ExprId("x", 8)
    y = ExprId("y", 8)

    inputs = sampling.gen_adversarial_inputs([x, y])

    assert [] not in inputs
    assert [0, 0] in inputs
    assert [1, 1] in inputs
    assert [0xFF, 0] in inputs
    assert [1, 0xFF] in inputs


def test_has_adversarial_counterexample_catches_shift_collision() -> None:
    x = ExprId("x", 8)
    expr = ExprOp("^", x, ExprOp("^", x, ExprOp("-", x)))
    candidate = ExprOp(
        "*",
        x,
        ExprOp("^", ExprOp("<<", ExprInt(2, 8), ExprOp("-", x)), ExprInt(0xFF, 8)),
    )

    assert sampling.has_adversarial_counterexample(expr, candidate)


def test_has_adversarial_counterexample_returns_false_when_samples_match() -> None:
    x = ExprId("x", 8)
    y = ExprId("y", 8)

    assert not sampling.has_adversarial_counterexample(
        ExprOp("+", ExprOp("&", x, y), ExprOp("|", x, y)),
        ExprOp("+", x, y),
    )
