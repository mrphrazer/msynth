"""
Example: SimBA- and GAMBA-mode simplification without an oracle.

Companion to ``simplify_expression.py`` (oracle path) and
``simplify_expression_cegis.py`` (CEGIS path). This script demonstrates
the ``PipelineMode.SIMBA`` and ``PipelineMode.GAMBA`` presets side by
side, both running against the default empty in-memory oracle so the
output is produced exclusively by the pipeline (no precomputed
equivalence-class table involved).

Two MBAs are exercised:

1. **SimBA-friendly linear MBA** — ``(v0 & v1) + (v0 | v1) + 5``,
   algebraically equal to ``v0 + v1 + 5``. SimBA's classifier
   recognises this as a linear combination of bitwise atoms.

2. **Nested-bitwise absorption identity** — ``x | -((x & y) | -x) → x``.
   Collapsed by the shared algebraic rewriter in every mode (not a
   GAMBA-exclusive reduction); see ``gamba_friendly_mba`` for details.

Both demos run against both modes side by side for an honest
comparison. The corpus-level advantage of GAMBA over SIMBA (+1.6 to
+2.1pp aggregate reduction, see ``tmp/gamba_sweep_report.md``) only
shows reliably across many cases — on individual tractable examples
both modes typically converge on the canonical form.

Run with: ``PYTHONPATH=. python3 scripts/simplify_expression_simba.py``.
"""

from __future__ import annotations

import logging
import time

from miasm.expression.expression import Expr, ExprId, ExprInt

from msynth import PipelineMode, Simplifier

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


def simba_friendly_mba(size: int) -> Expr:
    """
    Construct a linear MBA that SimBA's classifier accepts.

    The output equals ``v0 + v1 + 5`` algebraically via the identity
    ``(a & b) + (a | b) = a + b``. Every operand of the outer ``+`` is
    a bitwise function of ``{v0, v1}``, which is exactly the shape
    SimBA's grammar recognises as a linear MBA. SIMBA-mode pipeline
    suffices to reduce it to the canonical form.
    """
    v0 = ExprId("v0", size)
    v1 = ExprId("v1", size)
    return (v0 & v1) + (v0 | v1) + ExprInt(5, size)


def gamba_friendly_mba(size: int) -> Expr:
    """
    Construct a nested-bitwise shape that exercises the GAMBA post-
    rewriter's algebraic identities.

    ``x | -((x & y) | -x)`` matches one of the Tier 3 GAMBA
    ``nested_bitwise_absorb`` identities and simplifies to ``x``. SimBA's
    linear-MBA classifier on its own cannot recognise the outer ``|`` over
    a negated nested expression — it's outside the linear fragment — so the
    SimBA reconstruction stage leaves it alone.

    NOTE: the closing algebraic rewriter (``DEFAULT_REWRITER``, run at the
    end of *every* ``Simplifier.simplify`` regardless of mode) already
    collapses this shape, so in practice ALL pipeline modes (AST, SIMBA,
    GAMBA) converge on ``x`` here. This example therefore demonstrates the
    shared nested-bitwise absorption identity, not a GAMBA-exclusive
    capability. GAMBA's distinct advantage is structural (running the
    algebraic rules *before* SimBA and handling non-linear MBA via §5.1
    substitution) and shows up at the corpus level, not on this single
    tractable shape.
    """
    x = ExprId("x", size)
    y = ExprId("y", size)
    return x | (-((x & y) | (-x)))


def _run(label: str, mode: PipelineMode, expr: Expr) -> None:
    sim = Simplifier(pipeline_mode=mode)
    start = time.time()
    out = sim.simplify(expr)
    elapsed = time.time() - start
    print(f"[{label}] initial:    {expr}")
    print(f"[{label}] simplified: {out}")
    print(f"[{label}] elapsed:    {elapsed:.3f}s")
    print()


def main() -> None:
    size = 32

    # 1) SimBA-friendly linear MBA. SIMBA mode is sufficient; GAMBA mode
    #    produces the same result (the algebraic refinement has nothing
    #    extra to collapse on a shape SimBA already reconstructs cleanly).
    logger.info("Demo 1: SimBA-friendly linear MBA")
    mba1 = simba_friendly_mba(size)
    _run("SIMBA", PipelineMode.SIMBA, mba1)
    _run("GAMBA", PipelineMode.GAMBA, mba1)

    # 2) Nested-bitwise absorption identity. The closing algebraic rewriter
    #    runs in every mode, so AST/SIMBA/GAMBA all collapse this to ``x``;
    #    the demo illustrates the shared absorption identity, not a
    #    GAMBA-exclusive win.
    logger.info("Demo 2: nested-bitwise absorption (collapses in all modes)")
    mba2 = gamba_friendly_mba(size)
    _run("SIMBA", PipelineMode.SIMBA, mba2)
    _run("GAMBA", PipelineMode.GAMBA, mba2)


if __name__ == "__main__":
    setup_logging()
    main()
