"""
Standalone fuzz harness for SimbaPass over ExprSlice / ExprCompose atoms.

The unit tests in ``tests/simplification/test_simba_atoms.py``
cover the directed soundness shapes plus six Z3-checked directed cases.
This script complements them with a randomised harness over hundreds or
thousands of seeds — too slow to run in CI but useful when changing the
classifier, the atom collector, or the cube reconstruction.

Soundness witnesses, not theorems: for each seed we generate a
well-typed linear MBA from the grammar SimbaPass actually supports
(sum/diff of terms; each term is a constant, a bitwise expression over
the leaf pool, a ``const * bitwise``, or unary ``-`` of any of those).
We then run SimbaPass and assert ``input == rewritten`` either via Z3
(with a short timeout) or, when Z3 returns ``unknown``, via random
concrete sampling of every atom. A SAT result from Z3 or any sample
disagreement is a real counterexample and aborts immediately with the
offending seed.

Why constrained-grammar generation: a free-form random-MBA generator
also exposes a *pre-existing* classifier bug in the way ``^`` is
categorised when its arguments are MIXED (e.g. ``(x & y) + -y``).
That bug is orthogonal to slice/compose atomisation and would
obscure the signal here, so we stay inside the linear-MBA fragment
where SimbaPass's classifier is supposed to be sound.

Usage:
    PYTHONPATH=. python3 scripts/run_simba_fuzzer.py --seeds 1000
    PYTHONPATH=. python3 scripts/run_simba_fuzzer.py --seeds 5000 --depth 5
    PYTHONPATH=. python3 scripts/run_simba_fuzzer.py --seed 23   # reproduce one seed
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from miasm.expression.expression import (  # noqa: E402
    Expr,
    ExprCompose,
    ExprCond,
    ExprId,
    ExprInt,
    ExprMem,
    ExprOp,
    ExprSlice,
)

from msynth.simplification.simba import SimbaPass, _collect_atoms  # noqa: E402

_LINEAR_MBA_OPS = frozenset({"+", "-", "*", "&", "|", "^"})


def _node_count(expr: Expr) -> int:
    return len(expr.graph().nodes())


def _evaluate_atomic(expr: Expr, env: dict[Expr, int]) -> int:
    """Cube/sample evaluator that mirrors SimbaPass's atomisation rule.

    Treats every primary leaf (Id/Mem/Slice/Compose/Cond) AND every
    ExprOp whose op string is outside the linear-MBA fragment as
    opaque atomic lookups. Anything else (the fragment ops over
    well-typed args) is evaluated by recursion. Must stay in lockstep
    with ``_classify_uncached`` in ``msynth/simplification/simba.py``.
    """
    mask = (1 << expr.size) - 1
    if isinstance(expr, ExprInt):
        return int(expr) & mask
    if isinstance(expr, (ExprId, ExprMem, ExprSlice, ExprCompose, ExprCond)):
        return env.get(expr, 0) & mask
    if isinstance(expr, ExprOp):
        if expr.op not in _LINEAR_MBA_OPS:
            # Non-linear-fragment op — SimbaPass atomises this; mirror
            # that by looking up the whole node in env.
            return env.get(expr, 0) & mask
        args = [_evaluate_atomic(arg, env) for arg in expr.args]
        if expr.op == "-" and len(args) == 1:
            return (-args[0]) & mask
        if expr.op == "+":
            return sum(args) & mask
        if expr.op == "-":
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
    raise ValueError(f"unsupported expression: {expr!r}")


def _z3_check(left: Expr, right: Expr, *, timeout_ms: int) -> str:
    """Returns ``"equivalent"``, ``"counterexample"``, or ``"unknown"``."""
    import z3
    from miasm.ir.translators.z3_ir import TranslatorZ3

    translator = TranslatorZ3()
    z3_left = translator.from_expr(left)
    z3_right = translator.from_expr(right)
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.add(z3_left != z3_right)
    result = solver.check()
    if result == z3.unsat:
        return "equivalent"
    if result == z3.sat:
        return "counterexample"
    return "unknown"


def _random_concrete_check(
    left: Expr, right: Expr, atoms: list[Expr], rng: random.Random, *, samples: int
) -> bool:
    """Returns False iff any sample disagreement (real counterexample)."""
    size = left.size
    mask = (1 << size) - 1
    for _ in range(samples):
        env = {
            atom: rng.getrandbits(atom.size) & ((1 << atom.size) - 1) for atom in atoms
        }
        try:
            lv = _evaluate_atomic(left, env)
            rv = _evaluate_atomic(right, env)
        except KeyError:
            continue
        if (lv & mask) != (rv & mask):
            return False
    return True


def _leaf_pool() -> list[Expr]:
    """Five 16-bit atoms drawn from all four atom kinds SimbaPass supports."""
    return [
        ExprId("fx", 16),
        ExprId("fy", 16),
        ExprMem(ExprId("fptr", 64), 16),
        ExprSlice(ExprId("fbig", 32), 0, 16),
        ExprCompose(ExprId("flo", 8), ExprId("fhi", 8)),
    ]


def _nonlinear_leaf_pool() -> list[Expr]:
    """
    Extended 16-bit atom pool covering the GAMBA 5.5 operator-level
    atomisation cases: shifts (logical + arithmetic), rotations,
    division/modulo, count-leading/trailing-zeros, exponentiation,
    ``ExprCond``, plus the original four atom kinds. Every entry is
    a width-16 expression whose root operator (or whose node type)
    is atomised by SimbaPass — the cube reasoning sees them as
    opaque BITWISE atoms.
    """
    fx = ExprId("fx", 16)
    fy = ExprId("fy", 16)
    return [
        # Original four kinds.
        fx,
        fy,
        ExprMem(ExprId("fptr", 64), 16),
        ExprSlice(ExprId("fbig", 32), 0, 16),
        ExprCompose(ExprId("flo", 8), ExprId("fhi", 8)),
        # Operator-level atomisation candidates (GAMBA 5.5).
        ExprOp("<<", fx, ExprInt(3, 16)),
        ExprOp(">>", fx, ExprInt(5, 16)),
        ExprOp("a>>", fx, fy),
        ExprOp("<<<", fx, ExprInt(7, 16)),
        ExprOp(">>>", fx, ExprInt(5, 16)),
        ExprOp("/", fx, fy),
        ExprOp("%", fx, fy),
        ExprOp("cntleadzeros", fx),
        ExprOp("cnttrailzeros", fx),
        ExprCond(ExprId("fc", 1), fx, fy),
    ]


def _random_bitwise(
    rng: random.Random, leaves: list[Expr], size: int, depth: int
) -> Expr:
    mask = (1 << size) - 1
    if depth <= 0 or rng.random() < 0.40:
        return rng.choice(leaves)
    choice = rng.choice(["&", "|", "^", "not_"])
    if choice == "not_":
        return _random_bitwise(rng, leaves, size, depth - 1) ^ ExprInt(mask, size)
    a = _random_bitwise(rng, leaves, size, depth - 1)
    b = _random_bitwise(rng, leaves, size, depth - 1)
    return ExprOp(choice, a, b)


def _random_term(rng: random.Random, leaves: list[Expr], size: int, depth: int) -> Expr:
    mask = (1 << size) - 1
    shape = rng.choice(["const", "bitwise", "const_mul_bitwise", "neg"])
    if shape == "const":
        return ExprInt(rng.getrandbits(size) & mask, size)
    if shape == "bitwise":
        return _random_bitwise(rng, leaves, size, depth)
    if shape == "const_mul_bitwise":
        const = ExprInt(rng.getrandbits(size) & mask, size)
        b = _random_bitwise(rng, leaves, size, depth)
        return const * b if rng.random() < 0.5 else b * const
    sub = _random_term(rng, leaves, size, max(depth - 1, 0))
    return ExprOp("-", sub)


def _random_linear_mba(
    rng: random.Random, leaves: list[Expr], size: int, depth: int
) -> Expr:
    n_terms = rng.randint(1, 4)
    sub_depth = max(depth - 1, 1)
    terms = [_random_term(rng, leaves, size, sub_depth) for _ in range(n_terms)]
    expr = terms[0]
    for term in terms[1:]:
        expr = ExprOp(rng.choice(["+", "-"]), expr, term)
    return expr


def _random_wide_mba(
    rng: random.Random, leaves: list[Expr], size: int, depth: int
) -> Expr:
    """
    Unconstrained generator that combines any supported operator at any
    level. Used with --wide to stress the classifier; SimbaPass must
    either reject (``is source``) or produce an equivalent rewrite for
    every output of this generator.
    """
    mask = (1 << size) - 1
    if depth <= 0 or rng.random() < 0.30:
        return rng.choice(leaves)
    choice = rng.choice(["+", "-", "&", "|", "^", "*", "neg", "not_", "+", "^"])
    if choice == "*":
        const = ExprInt(rng.getrandbits(size) & mask, size)
        sub = _random_wide_mba(rng, leaves, size, depth - 1)
        return const * sub if rng.random() < 0.5 else sub * const
    if choice == "neg":
        return ExprOp("-", _random_wide_mba(rng, leaves, size, depth - 1))
    if choice == "not_":
        return _random_wide_mba(rng, leaves, size, depth - 1) ^ ExprInt(mask, size)
    a = _random_wide_mba(rng, leaves, size, depth - 1)
    b = _random_wide_mba(rng, leaves, size, depth - 1)
    return ExprOp(choice, a, b)


def _check_one(
    seed: int,
    *,
    depth: int,
    timeout_ms: int,
    samples: int,
    wide: bool,
    nonlinear_leaves: bool,
) -> str:
    """Returns one of ``"ok"``, ``"noop"``, ``"counter"``, ``"unknown"``."""
    rng = random.Random(seed)
    leaves = _nonlinear_leaf_pool() if nonlinear_leaves else _leaf_pool()
    generator = _random_wide_mba if wide else _random_linear_mba
    source = generator(rng, leaves, size=16, depth=depth)
    rewritten = SimbaPass().run(source)
    if rewritten is source:
        return "noop"
    status = _z3_check(source, rewritten, timeout_ms=timeout_ms)
    if status == "equivalent":
        return "ok"
    if status == "counterexample":
        sys.stderr.write(
            f"\nseed={seed} Z3 COUNTEREXAMPLE\n"
            f"  source:    {source}\n  rewritten: {rewritten}\n"
        )
        return "counter"
    # Fall back to random concrete sampling when Z3 times out.
    sample_rng = random.Random(seed ^ 0xDEADBEEF)
    atoms = sorted(
        set(_collect_atoms(source)) | set(_collect_atoms(rewritten)),
        key=lambda x: str(x),
    )
    if len(atoms) > 8:
        return "unknown"
    if not _random_concrete_check(
        source, rewritten, atoms, sample_rng, samples=samples
    ):
        sys.stderr.write(
            f"\nseed={seed} SAMPLE COUNTEREXAMPLE (Z3 timed out)\n"
            f"  source:    {source}\n  rewritten: {rewritten}\n"
        )
        return "counter"
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        default=1000,
        help="Number of seeds to try, starting at --start. Default 1000.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First seed to try (inclusive). Default 0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproduce a single seed (overrides --seeds/--start).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Maximum expression depth. Default 4.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=2000,
        help="Per-call Z3 timeout in milliseconds. Default 2000.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Random concrete samples on Z3 timeout. Default 64.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first counterexample.",
    )
    parser.add_argument(
        "--wide",
        action="store_true",
        help=(
            "Use an unconstrained random-MBA generator instead of the "
            "well-typed linear-MBA grammar. Stresses the classifier; "
            "any rewrite SimbaPass produces must still be sound."
        ),
    )
    parser.add_argument(
        "--nonlinear-leaves",
        action="store_true",
        help=(
            "Extend the leaf pool with operator-level atomisation "
            "candidates (shifts, rotations, division/modulo, "
            "count-zeros, ExprCond). Exercises GAMBA 5.5 paths in "
            "SimbaPass's classifier."
        ),
    )
    args = parser.parse_args()

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = range(args.start, args.start + args.seeds)

    start = time.time()
    counts = {"ok": 0, "noop": 0, "counter": 0, "unknown": 0}
    for seed in seeds:
        status = _check_one(
            seed,
            depth=args.depth,
            timeout_ms=args.timeout_ms,
            samples=args.samples,
            wide=args.wide,
            nonlinear_leaves=args.nonlinear_leaves,
        )
        counts[status] += 1
        if status == "counter" and args.fail_fast:
            break

    elapsed = time.time() - start
    sys.stdout.write(
        "checked={total} ok={ok} noop={noop} unknown={unknown} counter={counter} "
        "seconds={seconds:.1f}\n".format(
            total=sum(counts.values()),
            seconds=elapsed,
            **counts,
        )
    )
    return 1 if counts["counter"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
