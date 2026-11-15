# Copyright 2026 Maxence Faldor
"""Writes the circuits of the differential corpus and the examples, with their runs.

The engine evaluates every device as a kernel's instance, together with the others of
its kernel and shape in a group, and checks at every evaluation of a run that the groups
assemble the circuit's equations as the instances do evaluated one by one, each alone,
in the order of the devices, and that what the observer reads of each is what the
instance alone gives. It checks them on the circuits this script writes, to
``crates/jolty-engine/tests/data/kernels.json``: every circuit of the differential
corpus, ``tests/test_ngspice.py``, and of the examples, each with the analyses its tests
or its plan run on it. Run it from the repository root after changing either,
``uv run python tests/kernel_corpus.py``; with ``--check``, it writes nothing, and fails
when the file is not what it would write, as ``tests/test_kernel_corpus.py`` runs it.
"""

import importlib
import json
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from jolty import Simulation, _core, cli, simulate

ROOT: Final = Path(__file__).resolve().parent.parent
"""The repository."""

WRITTEN: Final = ROOT / "crates/jolty-engine/tests/data/kernels.json"
"""Where the circuits are written."""

type Record = dict[str, object]
"""A circuit, what it is elaborated in, and the analyses run on it, as written."""


class _Simulator:
    """An engine's simulator that records each analysis run on it in its record."""

    def __init__(self, inner: _core.Simulator, record: Record) -> None:
        self._inner = inner
        self._record = record

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def run(
        self, analysis: str, options: str, pool: _core.Pool
    ) -> _core.AnalysisResult:
        runs = self._record["runs"]
        assert isinstance(runs, list)
        runs.append([json.loads(analysis), json.loads(options)])
        return self._inner.run(analysis, options, pool)


class _Core:
    """The engine's module, its simulators recording what they are made of and run."""

    def __init__(self, records: list[Record]) -> None:
        self._records = records

    def __getattr__(self, name: str) -> object:
        return getattr(_core, name)

    def Simulator(  # noqa: N802 - stands in for the engine's class
        self, circuit: str, temperature: float, overrides: str, substitutes: str
    ) -> _Simulator:
        record: Record = {
            "circuit": json.loads(circuit),
            "temperature": temperature,
            "overrides": json.loads(overrides),
            "substitutes": json.loads(substitutes),
            "runs": [],
        }
        self._records.append(record)
        inner = _core.Simulator(circuit, temperature, overrides, substitutes)
        return _Simulator(inner, record)


@contextmanager
def _recording(records: list[Record]) -> Generator[None]:
    """Records every simulator made in `records`, and every analysis run on it."""
    original = simulate._core  # noqa: SLF001 - the engine the simulations call
    stand_in = _Core(records)
    simulate._core = stand_in  # noqa: SLF001  # ty: ignore[invalid-assignment] - a stand-in that records
    try:
        yield
    finally:
        simulate._core = original  # noqa: SLF001 - as it was


def _corpus(records: list[Record]) -> None:
    """Runs the differential corpus's tests, recording their simulations."""
    with _recording(records):
        code = pytest.main(["-q", "-n", "0", "tests/test_ngspice.py"])
    if code != 0:
        message = f"the differential corpus's tests failed: {code}"
        raise SystemExit(message)


def _examples(records: list[Record]) -> None:
    """Runs each example's plan, recording its simulations."""
    examples = ROOT / "examples"
    sys.path.insert(0, str(examples))
    for script in sorted(examples.glob("*.py")):
        module = importlib.import_module(script.stem)
        if not (hasattr(module, "build") and hasattr(module, "plan")):
            continue
        with _recording(records):
            Simulation(module.build()).run_plan(module.plan())


def _models(circuit: object) -> set[str]:
    """The models of every device instance of `circuit`, as written."""
    assert isinstance(circuit, dict)
    components = circuit["components"]
    assert isinstance(components, dict)
    return {
        instance["of"]["device"]
        for component in components.values()
        for instance in component["instances"]
        if "device" in instance["of"]
    }


def recorded() -> str:
    """Each circuit recorded with a device, once, with its runs, as written."""
    records: list[Record] = []
    _corpus(records)
    _examples(records)
    kept: dict[str, Record] = {}
    for record in records:
        if not record["runs"] or not _models(record["circuit"]):
            continue
        key = json.dumps(
            {name: record[name] for name in ("circuit", "temperature", "overrides")}
            | {"substitutes": record["substitutes"]},
            sort_keys=True,
        )
        found = kept.setdefault(key, {**record, "runs": []})
        runs, recorded = found["runs"], record["runs"]
        assert isinstance(runs, list)
        assert isinstance(recorded, list)
        runs.extend(run for run in recorded if run not in runs)
    written = json.dumps(list(kept.values()), separators=(",", ":"), allow_nan=False)
    return written + "\n"


class Corpus(cli.CommandLine):
    """Write the circuits of the corpus and the examples, with their runs."""

    model_config = SettingsConfigDict(cli_prog_name="tests/kernel_corpus.py")

    check: bool = Field(
        default=False, description="Fail unless the file holds the circuits."
    )

    def cli_cmd(self) -> None:
        """Writes the circuits, or with ``--check`` fails unless the file holds them."""
        written = recorded()
        if not self.check:
            WRITTEN.write_text(written, encoding="utf-8")
        elif WRITTEN.read_text(encoding="utf-8") != written:
            message = (
                f"{WRITTEN.relative_to(ROOT)} is not the corpus's circuits and runs: "
                "write it again with `uv run python tests/kernel_corpus.py`"
            )
            raise SystemExit(message)


def main() -> None:
    """Runs the command line."""
    cli.run(Corpus)


if __name__ == "__main__":
    main()
