# Copyright 2026 Maxence Faldor

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from differential import SKIP

BENCHMARKS = Path(__file__).parent.parent / "benchmarks"
COST_DOWN = BENCHMARKS / "cost_down"


@pytest.fixture(autouse=True)
def importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Imports the benchmarks' scripts, which import each other, as their runs do."""
    monkeypatch.syspath_prepend(str(BENCHMARKS))


def test_a_wrapper_logs_each_run_then_runs_the_program(tmp_path: Path) -> None:
    from run import wrap  # noqa: PLC0415 - importable only once the path holds it

    program = tmp_path / "simulator"
    program.write_text(
        '#!/bin/sh\nprintf "%s|" "$@"\necho "$PWD" >&2\nexit 3\n', encoding="utf-8"
    )
    program.chmod(0o755)
    wrappers = tmp_path / "bin"
    wrappers.mkdir()
    log = tmp_path / "simulations.jsonl"
    wrap(wrappers, program, log)
    arguments = ["-b", "a deck.cir", "it's", ""]
    for _ in range(2):
        ran = subprocess.run(  # noqa: S603 - the wrapper just written
            [str(wrappers / "simulator"), *arguments],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert ran.returncode == 3
        assert ran.stdout == "-b|a deck.cir|it's||"
        assert ran.stderr == f"{tmp_path.resolve()}\n"
    runs = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [(each["program"], each["arguments"]) for each in runs] == [
        ("simulator", arguments)
    ] * 2
    assert runs[0]["time"] <= runs[1]["time"]


@SKIP
def test_the_ngspice_arm_has_ngspice_and_python_but_no_jolty(tmp_path: Path) -> None:
    from arms import Arm  # noqa: PLC0415 - importable only once the path holds it
    from run import PACKAGES, SIMULATIONS, environment  # noqa: PLC0415 - importable only once the path holds it

    prepared = environment(Arm.NGSPICE, tmp_path)
    assert set(PACKAGES) <= set(prepared.packages)
    assert "jolty" not in prepared.packages
    path = prepared.variables["PATH"].split(os.pathsep)
    assert path[:2] == [str(tmp_path / "bin"), str(tmp_path / "python" / "bin")]
    assert not [each for each in path if (Path(each) / "jolty").exists()]

    def ran(*command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - the arm's own programs
            command,
            env=prepared.variables,
            capture_output=True,
            text=True,
            check=False,
        )

    probe = "import numpy, scipy, matplotlib, sys; print(sys.prefix)"
    assert ran("python", "-c", probe).stdout.strip() == str(tmp_path / "python")
    assert "No module named 'jolty'" in ran("python", "-c", "import jolty").stderr
    assert "ngspice" in ran("ngspice", "--version").stdout
    (logged,) = (tmp_path / SIMULATIONS).read_text(encoding="utf-8").splitlines()
    assert json.loads(logged)["arguments"] == ["--version"]


def test_the_jolty_arm_counts_its_command_lines_runs(tmp_path: Path) -> None:
    from arms import Arm  # noqa: PLC0415 - importable only once the path holds it
    from run import SIMULATIONS, environment  # noqa: PLC0415 - importable only once the path holds it

    prepared = environment(Arm.JOLTY, tmp_path)
    assert not prepared.packages
    path = prepared.variables["PATH"].split(os.pathsep)
    assert path[:2] == [str(tmp_path / "bin"), str(Path(sys.executable).parent)]
    ran = subprocess.run(
        ["jolty", "--version"],  # noqa: S607 - the wrapper, first on the arm's path
        env=prepared.variables,
        capture_output=True,
        text=True,
        check=True,
    )
    assert ran.stdout.startswith("jolty ")
    (logged,) = (tmp_path / SIMULATIONS).read_text(encoding="utf-8").splitlines()
    assert json.loads(logged)["program"] == "jolty"
