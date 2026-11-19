# Copyright 2026 Maxence Faldor

import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from jolty import Simulation, Transient
from jolty.units import us
from sample_circuits import low_pass

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
"""Whether this interpreter is a free-threaded build, which may run without its GIL."""

THREADS = 8
"""Python threads that simulate at once, as many as the performance cores."""


@pytest.mark.skipif(not FREE_THREADED, reason="the interpreter has a GIL to keep")
def test_the_engine_leaves_a_free_threaded_interpreter_without_its_gil() -> None:
    """The extension declares it needs no GIL, so importing it keeps the GIL off.

    A module that does not declare so turns the GIL back on as it loads, and warns,
    which these tests make an error.
    """
    from jolty import _core  # noqa: PLC0415 - the import is what is tested

    assert _core.__name__ == "jolty._core"
    assert not sys._is_gil_enabled()  # noqa: SLF001 - CPython's only way to ask


def test_simulations_from_many_python_threads_find_what_one_finds() -> None:
    """Each thread's simulation finds the bits one thread finds alone.

    The engine releases the interpreter while it runs, so with a GIL the threads take
    turns in Python and run at once in Rust; without one, they run at once throughout.
    """
    circuit = low_pass()

    def run(_: int) -> tuple[bytes, bytes]:
        waveforms = Simulation(circuit).run(Transient(stop=5 * us))
        return waveforms.times.tobytes(), waveforms.voltage("out").tobytes()

    alone = run(0)
    with ThreadPoolExecutor(THREADS) as pool:
        together = list(pool.map(run, range(4 * THREADS)))
    assert all(found == alone for found in together)
    # At stop = 5 us (tau = 1 us), 1 - exp(-5) ~= 0.99326; tail exp(-5) = 6.74e-3.
    assert np.frombuffer(alone[1])[-1] == pytest.approx(1, abs=1e-2)
