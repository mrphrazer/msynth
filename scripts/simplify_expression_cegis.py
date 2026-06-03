"""
Example: CEGIS-driven simplification without an oracle.

Companion to ``simplify_expression.py``, which demonstrates the oracle
path. This script demonstrates the CEGIS fallback path, which kicks in
when the oracle does not have an entry for the subtree under inspection.

CEGIS synthesises constants for one of a fixed family of templates (see
:mod:`msynth.simplification.cegis`) using Z3, so the canonical motivating
case is an expression whose subtrees contain arbitrary constants the
precomputed oracle cannot cover — e.g. ``v0 * 0xDEADBEEF + 0x1337``.

What this example shows:

1. Construct a :class:`Simplifier` without an ``oracle_path`` — the
   simplifier skips the oracle-lookup branch entirely and relies on
   the pipeline + subtree SiMBA + CEGIS.
2. Build a small MBA whose simplest equivalent fits the CEGIS template
   ``(c0 * p0) + c1``. The pipeline can't fully fold it (the inner
   ``v0 * 0xDEADBEEF`` is not a SiMBA atom), so the simplifier loop
   misses on every pass. CEGIS then recognises the placeholder shape
   and synthesises the constants.
3. Run :class:`Simplifier` with ``enable_cegis=True`` and print the
   before/after.

Run with: ``PYTHONPATH=. python3 scripts/simplify_expression_cegis.py``.
"""

import logging
import time

from miasm.expression.expression import Expr, ExprId, ExprInt, ExprOp

from msynth import Simplifier

logger = logging.getLogger("msynth")


def setup_logging() -> None:
    """Setup logger"""
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(name)-18s - %(levelname)-8s: %(message)s")
    )
    logger.addHandler(console_handler)


def cegis_mba(size: int) -> Expr:
    """
    Construct an MBA whose simplest equivalent fits a CEGIS template.

    The output equals ``v0 * 0xDEADBEEF + 0x1337`` algebraically, via
    the identity ``x + y = (x | y) + (x & y)``. The product
    ``v0 * 0xDEADBEEF`` is not a bitwise leaf of SiMBA's grammar, so the
    SiMBA pass cannot classify the expression. Without an oracle the
    main loop has nothing to consult. CEGIS then unifies the variable
    as ``p0``, recognises the template ``(c0 * p0) + c1`` against the
    semantics, and synthesises ``c0 = 0xDEADBEEF, c1 = 0x1337``.
    """
    v0 = ExprId("v0", size)
    c = ExprInt(0xDEADBEEF, size)
    k = ExprInt(0x1337, size)
    prod = ExprOp("*", v0, c)
    return (prod | k) + (prod & k)


def main(expr: Expr) -> None:
    # No oracle — CEGIS is off by default, so we enable it explicitly.
    # The other cegis_* knobs use the simplifier defaults — see
    # Simplifier's docstring for the budget / template-count tuning.
    logger.debug("Initializing simplification engine with CEGIS enabled")
    simplifier = Simplifier(enable_cegis=True)
    logger.info("Simplifying expression")
    start = time.time()
    simplified = simplifier.simplify(expr)
    elapsed = time.time() - start
    print(f"initial:    {expr}")
    print(f"simplified: {simplified}")
    logger.info(f"Done in {round(elapsed, 2)} seconds")


if __name__ == "__main__":
    setup_logging()
    main(cegis_mba(32))
