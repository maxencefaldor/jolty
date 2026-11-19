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
"""Flag indicating a free-threaded Python build capable of running without the GIL."""

THREADS = 8
"""Python threads that simulate at once, as many as the performance cores."""


@pytest.mark.skipif(not FREE_THREADED, reason="the interpreter has a GIL to keep")
def test_the_engine_leaves_a_free_threaded_interpreter_without_its_gil() -> None:
    """Verifies importing the Rust extension keeps the GIL disabled on 3.15t builds.

    An extension without free-threading support re-enables the GIL on import and emits
    a warning, which pytest treats as an error.
    """
    from jolty import _core  # noqa: PLC0415 - the import is what is tested

    assert _core.__name__ == "jolty._core"
    assert not sys._is_gil_enabled()  # noqa: SLF001 - CPython's only way to ask


def test_simulations_from_many_python_threads_find_what_one_finds() -> None:
    """Verifies multithreaded Python runs match single-threaded results bit for bit.

    The engine releases the GIL during Rust execution, so threads run concurrently in
    Rust under GIL builds and concurrently throughout on free-threaded builds.
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
