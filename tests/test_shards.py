# Copyright 2026 Maxence Faldor
"""A Monte Carlo analysis run in shards finds what one run of every sample finds.

Merging the shards of 1, 3 and 7 ranges of a plan's samples, run on three threads,
gives the report and the batch file of one run of every sample on one thread, bit for
bit, and the same batches at every condition. Shards of one circuit built by scripts
in two places merge. Shards of other fingerprints, and shards whose samples leave a gap
or overlap, do not merge, and the error names them. A shard
killed while it runs leaves no file, and run again writes the same file each time.
"""

import importlib.util
import itertools
import os
import runpy
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from jolty import files, results, runner, shards, simulate
from jolty.builder import circuit
from jolty.cli import main
from jolty.errors import JoltyError
from jolty.report import ShardReport
from jolty.service import SimulateRequest

SAMPLES = 70
"""The samples of the plan's Monte Carlo analysis."""

DIVIDER = """
from jolty import (
    Ac,
    Builder,
    Corner,
    MonteCarlo,
    OperatingPoint,
    Plan,
    Role,
    Specs,
    Transient,
    component,
    devices,
)
from jolty.plan import decade
from jolty.results import TransientResult
from jolty.specs import Measure, MeasureError, Reading, gain, value
from jolty.units import VOLTAGE, Hz, MHz, V, celsius, kohm, nF, us

SEED = 3


@component
def divider(c: Builder, supply=5 * V) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(supply), name="V1")
    upper = c.place(devices.resistor(1 * kohm, tolerance=0.05), name="R1")
    lower = c.place(devices.resistor(1 * kohm, tolerance=0.05), name="R2")
    held = c.place(devices.capacitor(1 * nF, tolerance=0.1), name="C1")
    c.connect(source.positive, upper.a, name="in")
    c.connect(upper.b, lower.a, held.a, name="out")
    c.connect(lower.b, held.b, source.negative, ground)


def build(supply=5 * V):
    return divider(supply=supply)


def late(result):
    # Read in Python: unread above 2.5 V, why naming the voltage, so runs differ.
    if not isinstance(result, TransientResult):
        raise TypeError(result)
    volts = float(result.voltage("out")[-1])
    if volts > 2.5:
        raise MeasureError(f"out ends at {volts:.4f} V")
    return Reading(volts, VOLTAGE)


def plan():
    return Plan(
        {
            "op": OperatingPoint(),
            "ac": Ac("V1", decade(1 * Hz, 10 * MHz, 3)),
            "step": Transient(stop=5 * us, breakpoints=[1 * us, 4 * us]),
        },
        corners={"hot": Corner(temperature=celsius(85)), "low": Corner(supply=4.5 * V)},
        monte_carlo=MonteCarlo(SAMPLES, seed=SEED),
    )


def specs():
    return Specs(
        [
            value("op", "out").between(2.1 * V, 2.6 * V),
            gain("ac", "out", at=1e5 * Hz) >= -6.5,
            Measure("step", "out at its end", None, late) >= 2.2 * V,
        ],
        yield_target=0.5,
    )
"""


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs each test in its own directory, where script paths are relative."""
    monkeypatch.chdir(tmp_path)


def _script(directory: Path, seed: int = 3, name: str = "divider.py") -> Path:
    path = directory / name
    text = DIVIDER.replace("SAMPLES", str(SAMPLES)).replace(
        "SEED = 3", f"SEED = {seed}"
    )
    path.write_text(text, encoding="utf-8")
    return path


def _built(outcome: runner.Outcome) -> runner.Built:
    assert isinstance(outcome, runner.Built), str(outcome)
    return outcome


def _shard(
    script: Path,
    directory: Path,
    start: int,
    stop: int,
    parallelism: runner.Thresholds | None = None,
) -> Path:
    """Runs the samples from `start` up to `stop` as a shard, and returns its file.

    The shard's runs spread their work as `parallelism` says, in place of the plan's.
    """
    job = runner.Job(
        script=script,
        plan=script,
        max_unknowns=10,
        samples=(start, stop),
        parallelism=parallelism or runner.Thresholds(),
    )
    work = directory / f"job-{start}-{stop}"
    work.mkdir(parents=True)
    found = _built(runner.run(job, work)).shard
    assert found is not None
    assert (found.start, found.stop, found.samples) == (start, stop, SAMPLES)
    path = directory / files.shard_name(start, stop)
    (work / runner.SHARD).rename(path)
    return path


def _bits(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array).tobytes()


def _same(found: results.BatchResult | None, other: results.BatchResult | None) -> None:
    """Checks that two batches hold the same bits, and the same of the rest."""
    assert (found is None) == (other is None)
    if found is None or other is None:
        return
    for array in ("values", "ended", "measured", "causes"):
        assert _bits(getattr(found, array)) == _bits(getattr(other, array)), array
    for name in ("parameters", "units", "reasons", "problems", "errors"):
        assert getattr(found, name) == getattr(other, name), name


def test_merging_1_3_and_7_shards_finds_one_run_s_bits(tmp_path: Path) -> None:
    script = _script(tmp_path)
    alone = tmp_path / "alone"
    alone.mkdir()
    with simulate.threads(1):
        job = runner.Job(script=script, plan=script, max_unknowns=10)
        whole = _built(runner.run(job, alone)).monte_carlo
    assert whole is not None
    assert whole.passing is not None
    assert 0 < whole.passing.passing < SAMPLES
    merged = []
    for count in (1, 3, 7):
        directory = tmp_path / f"{count}.shards"
        cuts = [SAMPLES * place // count for place in range(count + 1)]
        with simulate.threads(3):
            paths = [
                _shard(script, directory, start, stop)
                for start, stop in itertools.pairwise(cuts)
            ]
        # The merge takes the shards in any order.
        shard, ordered = shards.merge(paths[::-1])
        assert ordered == tuple(paths)
        assert (shard.start, shard.stop) == (0, SAMPLES)
        found, batch = shards.monte_carlo(shard)
        assert found == whole
        saved = tmp_path / f"{count}.batches.npz"
        files.save_batches(saved, {runner.MONTE_CARLO: batch})
        assert saved.read_bytes() == (alone / runner.BATCHES).read_bytes()
        merged.append(shard)
    first, *others = merged
    # Some runs read nothing of the measure Python reads, each for its own reason.
    plan_s = first.conditions[0]
    assert plan_s is not None
    assert len(plan_s.reasons) > 3
    for other in others:
        assert other.fingerprint == first.fingerprint
        for mine, theirs in zip(other.conditions, first.conditions, strict=True):
            _same(mine, theirs)


def test_shards_of_other_fingerprints_do_not_merge(tmp_path: Path) -> None:
    directory = tmp_path / "divider.shards"
    first = _shard(_script(tmp_path), directory, 0, 35)
    other = _script(tmp_path, seed=4, name="other.py")
    second = _shard(other, directory, 35, SAMPLES)
    with pytest.raises(JoltyError) as raised:
        shards.merge([first, second])
    message = str(raised.value)
    assert message.startswith(
        "the shards ran other circuits, plans, options, builds or platforms, as their "
        "fingerprints differ: shard-0-35.npz has "
    )
    assert "; shard-35-70.npz has " in message


def test_shards_of_one_circuit_built_in_two_places_merge(tmp_path: Path) -> None:
    directory = tmp_path / "divider.shards"
    places = []
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
        places.append(_script(tmp_path / name))
    # Each instance's source names the script that placed it, in its own directory.
    built = [circuit(runpy.run_path(str(place))["build"]()) for place in places]
    resistors = [each.find("R1") for each in built]
    assert [
        each.source_location.file for each in resistors if each and each.source_location
    ] == [
        "one/divider.py",
        "two/divider.py",
    ]
    assert files.circuit_hash(built[0]) == files.circuit_hash(built[1])
    first = _shard(places[0], directory, 0, 35)
    second = _shard(places[1], directory, 35, SAMPLES)
    shard, ordered = shards.merge([first, second])
    assert ordered == (first, second)
    assert (shard.start, shard.stop) == (0, SAMPLES)
    # A value changed is another circuit, wherever its script lies.
    text = places[0].read_text(encoding="utf-8")
    assert "capacitor(1 * nF" in text
    changed = tmp_path / "one" / "changed.py"
    changed.write_text(
        text.replace("capacitor(1 * nF", "capacitor(2 * nF"), encoding="utf-8"
    )
    other = _shard(changed, tmp_path / "changed.shards", 35, SAMPLES)
    with pytest.raises(JoltyError, match="as their fingerprints differ"):
        shards.merge([first, other])


def test_shards_of_other_interiors_do_not_merge(tmp_path: Path) -> None:
    # The interiors of a dissected factorization order its operations, so they join a
    # shard's fingerprint, where the thresholds that only spread work do not.
    script = _script(tmp_path)
    directory = tmp_path / "divider.shards"
    first = _shard(script, directory, 0, 35)
    spread = _shard(
        script, directory, 35, SAMPLES, runner.Thresholds(min_solves_per_task=1)
    )
    shard, _ = shards.merge([first, spread])
    assert (shard.start, shard.stop) == (0, SAMPLES)
    dissected = _shard(
        script,
        tmp_path / "dissected.shards",
        35,
        SAMPLES,
        runner.Thresholds(max_unknowns_per_interior=2),
    )
    with pytest.raises(JoltyError, match="fingerprints differ"):
        shards.merge([first, dissected])


def test_gaps_and_overlaps_name_the_shards(tmp_path: Path) -> None:
    script = _script(tmp_path)
    directory = tmp_path / "divider.shards"
    paths = [
        _shard(script, directory, start, stop)
        for start, stop in ((5, 20), (30, 50), (40, 45), (44, 60))
    ]
    with pytest.raises(JoltyError) as raised:
        shards.merge(paths)
    assert str(raised.value).splitlines() == [
        "no shard holds samples 0 up to 5, before shard-5-20.npz",
        (
            "no shard holds samples 20 up to 30, after shard-5-20.npz and before "
            "shard-30-50.npz"
        ),
        "shard-30-50.npz and shard-40-45.npz both hold samples 40 up to 45",
        "shard-30-50.npz and shard-44-60.npz both hold samples 44 up to 50",
        "no shard holds samples 60 up to 70, after shard-44-60.npz",
    ]
    with pytest.raises(JoltyError, match="there is no shard to merge"):
        shards.merge([])
    (directory / "shard-70-80.npz").write_bytes(b"not a shard")
    with pytest.raises(JoltyError, match="is not a jolty shard file of version 2"):
        shards.merge([directory / "shard-70-80.npz"])


def test_a_shard_killed_and_run_again_writes_the_same_file(tmp_path: Path) -> None:
    script = _script(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    command = [sys.executable, "-m", "jolty", "simulate", script.name]
    command += ["--samples", f"0:{SAMPLES}"]
    environment = {**os.environ, "TMPDIR": str(scratch)}
    killed = subprocess.Popen(  # noqa: S603 - this interpreter, running jolty
        command,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # The service writes the job before it starts the runner, which takes longer to
    # import jolty than this takes to kill it: the shard is still running.
    deadline = time.monotonic() + 60
    while not list(scratch.glob("jolty-*/job.json")):
        assert killed.poll() is None
        assert time.monotonic() < deadline
        time.sleep(0.005)
    os.killpg(killed.pid, signal.SIGKILL)
    killed.wait()
    shard = tmp_path / "divider.shards" / files.shard_name(0, SAMPLES)
    assert not shard.exists()
    written = []
    for _ in range(2):
        done = subprocess.run(  # noqa: S603 - this interpreter, running jolty
            command, env=environment, capture_output=True, check=False
        )
        assert done.returncode == 0, done.stderr
        written.append(shard.read_bytes())
    assert written[0] == written[1]
    assert [path.name for path in shard.parent.iterdir()] == [shard.name]


def test_the_command_line_runs_shards_and_merges_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _script(tmp_path)
    for samples in ("0:40", f"40:{SAMPLES}"):
        assert main(["simulate", script.name, "--samples", samples]) is None
    out = capsys.readouterr().out
    assert "saved divider.shards/shard-40-70.npz" in out
    assert "shard: samples 40 up to 70 of 70, at hot, at low; 0 failed;" in out
    assert not (tmp_path / "divider.schematic.png").exists()
    with pytest.raises(SystemExit) as exited:
        main(["merge", "divider.shards"])
    out = capsys.readouterr().out
    assert exited.value.code == 1  # The yield misses its target.
    assert out.startswith("merged 2 shards\n\nsaved ")
    assert "monte carlo: 70 samples of 3 toleranced parameters, at hot, at low" in out
    assert (tmp_path / "divider.batches.npz").is_file()
    assert (tmp_path / "divider.monte-carlo.plot.png").is_file()


def test_the_cluster_example_runs_the_missing_shards_and_merges_them(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    path = Path(__file__).parent.parent / "examples" / "cluster" / "launch.py"
    spec = importlib.util.spec_from_file_location("launch", path)
    assert spec is not None
    assert spec.loader is not None
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    script = _script(tmp_path)
    arguments = [str(script), "--samples", str(SAMPLES), "--shard", "30"]
    assert launch.main([*arguments, "--workers", "2"]) == 1  # The yield misses.
    out = capfd.readouterr().out
    for taken in ("0:30", "30:60", "60:70"):
        assert f"shard {taken} exited with 0" in out
    assert "merged 3 shards" in out
    (tmp_path / "divider.shards" / "shard-30-60.npz").unlink()
    assert launch.main(arguments) == 1
    out = capfd.readouterr().out
    assert out.startswith("shard 30:60 exited with 0\n")
    assert "shard 0:30" not in out


def test_a_shard_names_samples_of_the_plan_s_monte_carlo_analysis(
    tmp_path: Path,
) -> None:
    script = _script(tmp_path)
    with pytest.raises(ValidationError, match="from a start up to a later stop"):
        SimulateRequest(script=script, samples="5:5")
    with pytest.raises(ValidationError, match="should match pattern"):
        SimulateRequest(script=script, samples="5-9")
    with pytest.raises(ValidationError, match="a directory of shards holds samples"):
        SimulateRequest(script=script, shards=tmp_path)
    job = runner.Job(script=script, plan=script, max_unknowns=10, samples=(60, 71))
    failure = runner.run(job, tmp_path)
    assert isinstance(failure, runner.Failure)
    assert "draws 70 samples, not samples (60, 71)" in failure.message
    plain = tmp_path / "plain.py"
    plain.write_text(
        script.read_text(encoding="utf-8").replace(
            "monte_carlo=MonteCarlo(70, seed=SEED),", ""
        ),
        encoding="utf-8",
    )
    job = runner.Job(script=plain, plan=plain, max_unknowns=10, samples=(0, 1))
    failure = runner.run(job, tmp_path)
    assert isinstance(failure, runner.Failure)
    assert failure.message == (
        "the plan has no Monte Carlo analysis whose samples a shard runs"
    )


def test_a_shard_warns_when_every_sample_s_shards_pass_10_gb() -> None:
    report = ShardReport(
        start=0,
        stop=1000,
        samples=10_000_000,
        failed=0,
        size=1_200_000,
        total=12_000_000_000,
        fingerprint="0" * 64,
    )
    assert report.describe().endswith(
        "\nwarning: the shards of every sample hold 12 GB of arrays, more than 10 GB"
    )
    small = report.model_copy(update={"total": 10_000_000_000})
    assert "warning" not in small.describe()
