from __future__ import annotations

import pytest
import z3
from miasm.analysis.machine import Machine
from miasm.core.bin_stream import bin_stream_str
from miasm.core.locationdb import LocationDB
from miasm.expression.expression import ExprId
from miasm.expression.simplifications import expr_simp
from miasm.ir.symbexec import SymbolicExecutionEngine

from msynth import Simplifier


@pytest.mark.parametrize(
    "bits,code,register_names",
    [
        (32, "89d8 09c8 21cb 01d8 c3", ("EAX", "EBX", "ECX")),
        (64, "4889d8 4809c8 4821cb 4801d8 c3", ("RAX", "RBX", "RCX")),
    ],
    ids=["x86_32", "x86_64"],
)
def test_lift_and_simplify_x86_mba(bits, code, register_names) -> None:
    # mov a, b; or a, c; and b, c; add a, b; ret computes (x | y) + (x & y).
    # Loading Machine also exercises SemBuilder's Python AST compatibility.
    machine = Machine(f"x86_{bits}")
    loc_db = LocationDB()
    disassembler = machine.dis_engine(
        bin_stream_str(bytes.fromhex(code)), loc_db=loc_db
    )
    asm_cfg = disassembler.dis_multiblock(0)
    lifter = machine.lifter_model_call(loc_db)
    ir_cfg = lifter.new_ircfg_from_asmcfg(asm_cfg)

    result_reg, left_reg, right_reg = (ExprId(name, bits) for name in register_names)
    x, y = ExprId("x", bits), ExprId("y", bits)
    engine = SymbolicExecutionEngine(lifter, {left_reg: x, right_reg: y})
    engine.run_block_at(ir_cfg, 0)
    lifted = engine.symbols[result_reg]
    assert lifted == expr_simp((x | y) + (x & y))

    simplifier = Simplifier(enforce_equivalence=True)
    simplified = simplifier.simplify(lifted)

    assert simplified == expr_simp(x + y)
    assert len(simplified.graph().nodes()) < len(lifted.graph().nodes())
    assert simplifier.check_semantical_equivalence(lifted, simplified) == z3.unsat
    assert simplifier.check_semantical_equivalence(lifted, x - y) == z3.sat
