from __future__ import annotations

import multiprocessing
import subprocess
import sys
from pathlib import Path

import pytest

from msynth import Synthesizer


class UnsuccessfulSynthesizer(Synthesizer):
    def synthesize_from_expression(self, expr, num_samples, timeout=60):
        return expr, 1.0


@pytest.mark.parametrize("start_method", multiprocessing.get_all_start_methods())
@pytest.mark.parametrize("succeeds", [True, False], ids=["success", "no-solution"])
@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="POSIX fork required"
)
def test_parallel_synthesis_with_real_processes(
    tmp_path, start_method, succeeds
) -> None:
    # A fresh interpreter keeps each start method independent of pytest's state.
    code = """
import math
import multiprocessing
import sys

import z3
from miasm.expression.expression import ExprId, ExprInt
from msynth import Simplifier, Synthesizer

sys.path.insert(0, sys.argv[1])
from test_synthesizer_parallel import UnsuccessfulSynthesizer

multiprocessing.set_start_method(sys.argv[2])
multiprocessing.cpu_count = lambda: 2
expr = ExprId("x", 8) + ExprInt(1, 8)
succeeds = sys.argv[3] == "True"
synth = Synthesizer() if succeeds else UnsuccessfulSynthesizer()
result, score = synth.synthesize_from_expression_parallel(expr, 16)
if succeeds:
    assert score == 0.0, score
    assert Simplifier().check_semantical_equivalence(expr, result) == z3.unsat
else:
    assert result == expr
    assert math.isinf(score)
assert multiprocessing.get_start_method() == sys.argv[2]
assert not multiprocessing.active_children(), multiprocessing.active_children()
"""
    # Match the training scripts: execute at module level without a main guard.
    script = tmp_path / "parallel_synthesis.py"
    script.write_text(code)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(Path(__file__).parent),
            start_method,
            str(succeeds),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
