# Copyright 2026 Maxence Faldor
"""Draw the docs' images by simulating the examples, as an agent would.

Run from the repository root: ``uv run python docs/illustrate.py``. Each example runs
through `jolty.service.simulate`, the same function behind ``jolty simulate``, and the
component inside each of its instances through `jolty.service.draw`, behind
``jolty draw``. The images land in ``docs/images`` under the names the commands give
them, such as ``rc_filter.step.plot.png`` and ``rc_filter.X1.schematic.png``.
"""

import asyncio
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final, NamedTuple

from jolty import service
from jolty.report import OperatingPointReport
from jolty.settings import Settings

ROOT: Final = Path(__file__).resolve().parent.parent
IMAGES: Final = ROOT / "docs" / "images"


class Example(NamedTuple):
    """What to draw of an example."""

    probes: list[str]
    """The unknowns to plot, or none for the defaults."""
    inside: list[str]
    """The instances whose components to draw."""


EXAMPLES: Final = {
    "rc_filter": Example(probes=[], inside=["X1"]),
    "rectifier": Example(probes=["in", "out", "V1:current"], inside=["X1"]),
    "full_adder": Example(
        probes=["a", "b", "carry_in", "sum", "carry_out", "adder/half"],
        inside=["adder", "adder/U1"],
    ),
}


async def illustrate(name: str, example: Example, scratch: Path) -> None:
    """Simulates the example `name` and copies its images into the docs' images."""
    request = service.SimulateRequest(
        script=ROOT / "examples" / f"{name}.py",
        output=scratch / f"{name}.jolty.json",
        results=scratch / f"{name}.results.npz",
        probes=example.probes,
    )
    simulated = await service.simulate(request, Settings())
    pointed = any(
        isinstance(report, OperatingPointReport) for report in simulated.analyses
    )
    images = list(simulated.images())
    for instance in example.inside:
        inside = service.DrawRequest(
            circuit=simulated.circuit.path,
            instance=instance,
            results=simulated.results if pointed else None,
        )
        drawn = await service.draw(inside, Settings())
        images.extend(drawn.images())
    for image in images:
        target = IMAGES / image.name
        shutil.copyfile(image, target)
        print(f"wrote {target.relative_to(ROOT)}")


async def main() -> None:
    """Draws every image, replacing those drawn before."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    for stale in IMAGES.glob("*.png"):
        stale.unlink()
    with TemporaryDirectory() as scratch:
        for name, example in EXAMPLES.items():
            await illustrate(name, example, Path(scratch))


if __name__ == "__main__":
    asyncio.run(main())
