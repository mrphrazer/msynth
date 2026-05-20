from random import choice, getrandbits
from typing import List

from miasm.expression.expression import Expr, ExprId

from msynth.utils.expr_utils import compile_expr_to_python, get_unique_variables

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

ADVERSARIAL_TAIL_VALUES = 64


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


def gen_adversarial_values(size: int) -> List[int]:
    """
    Generate a compact deterministic set of edge values for quick equivalence probes.

    These values are intentionally biased toward bit-vector edge cases that random
    oracle sampling can miss: small values, powers of two, all-ones, and values just
    below the modular wraparound point. The tail values are especially useful for
    catching expressions whose behavior changes when a negated value becomes a small
    shift count.
    """
    if size <= 0:
        raise ValueError("size must be positive")

    mask = (1 << size) - 1
    values = {
        0,
        1,
        2,
        3,
        4,
        7,
        8,
        15,
        16,
        31,
        32,
        63,
        64,
        mask,
    }
    values.update((mask - offset) & mask for offset in range(ADVERSARIAL_TAIL_VALUES))
    return sorted(values)


def gen_adversarial_inputs(variables: List[Expr]) -> List[List[int]]:
    """
    Generate low-cost input rows for expressions over the given variables.

    The generated rows vary one variable at a time while keeping the remaining
    variables at either zero or one. This keeps the number of probes linear in the
    number of variables, avoiding the exponential cost of exhaustive evaluation.
    """
    if not variables:
        return [[]]

    inputs = [[0 for _ in variables], [1 for _ in variables]]
    for index, variable in enumerate(variables):
        for value in gen_adversarial_values(variable.size):
            row = [0 for _ in variables]
            row[index] = value
            inputs.append(row)

            row = [1 for _ in variables]
            row[index] = value
            inputs.append(row)
    return inputs


def _rename_variables_for_compilation(expr: Expr, variables: List[Expr]) -> Expr:
    return expr.replace_expr(
        {
            variable: ExprId(f"p{index}", variable.size)
            for index, variable in enumerate(variables)
        }
    )


def has_adversarial_counterexample(expr: Expr, candidate: Expr) -> bool:
    """
    Return True if deterministic edge-case samples disprove equivalence.

    This is a cheap guard for permissive simplification paths when SMT returns
    UNKNOWN. It is deliberately incomplete: returning False means no counterexample
    was found in the small probe set, not that the expressions are equivalent.
    Unsupported expression forms fall back to the caller's existing checks.
    """
    variables = sorted(
        set(get_unique_variables(expr)) | set(get_unique_variables(candidate)),
        key=lambda variable: str(variable),
    )
    try:
        original_func = compile_expr_to_python(
            _rename_variables_for_compilation(expr, variables)
        )
        candidate_func = compile_expr_to_python(
            _rename_variables_for_compilation(candidate, variables)
        )
    except ValueError:
        return False

    for inputs in gen_adversarial_inputs(variables):
        if original_func(inputs) != candidate_func(inputs):
            return True
    return False
