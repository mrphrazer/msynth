from __future__ import annotations

from typing import List

import pytest

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
