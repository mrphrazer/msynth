from random import choice, getrandbits
from typing import List

SPECIAL_VALUES = [
    0x0,
    0x1,
    0x2,
    0x80,
    0xFF,
    0x8000,
    0xFFFF,
    0x8000_0000,
    0xFFFF_FFFF,
    0x8000_0000_0000_0000,
    0xFFFF_FFFF_FFFF_FFFF,
]


def get_rand_input() -> int:
    """
    Generates a random value. It equally chooses
    between u8, u16, u32, u64 and special values
    for a better synthesis coverage.

    Returns:
        Random value.
    """
    coin = getrandbits(8) % 5
    # u8
    if coin == 0:
        return getrandbits(8)
    # u16
    elif coin == 1:
        return getrandbits(16)
    # u32
    elif coin == 2:
        return getrandbits(32)
    # u64
    elif coin == 3:
        return getrandbits(64)
    # special values
    elif coin == 4:
        return choice(SPECIAL_VALUES)
    else:
        raise NotImplementedError()


def gen_inputs_array(n: int) -> List[int]:
    """
    Returns an array of random values.

    Args:
        n: Number of random values.

    Returns:
        List of random values.
    """
    return [get_rand_input() for _ in range(n)]


def gen_inputs(num_variables: int, num_samples: int) -> List[List[int]]:
    """
    Generates the oracle inputs.

    The oracle inputs are a nested lists of ints:

    - the inner lists hold random values, one for each input variable in the synthesis function
    - the outer list represents `num_samples` independent input arrays,
      the number of I/O samples for synthesis

    Args:
        num_variables: Number of variables in the synthesis function
        num_samples: Number of independent oracle queries

    Returns:
        List of lists, inner lists hold random values.
    """
    return [gen_inputs_array(num_variables) for _ in range(num_samples)]
