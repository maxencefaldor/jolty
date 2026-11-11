# Copyright 2026 Maxence Faldor
"""Draw the docs' images by simulating the examples, as an agent would.

Run from the repository root: ``uv run python docs/illustrate.py``. Each example runs
through `jolty.service.simulate`, the same function behind ``jolty simulate``, and its
plots land in ``docs/images`` as ``<example>.<analysis>.png``.
"""

import asyncio
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

from jolty import service
from jolty.settings import Settings

ROOT: Final = Path(__file__).resolve().parent.parent
IMAGES: Final = ROOT / "docs" / "images"
EXAMPLES: Final = {
    "rc_filter": [],
    "rectifier": ["in", "out", "V1:current"],
}
"""Each example to draw, with the unknowns to plot, or none for the defaults."""


async def draw(example: str, probes: list[str], scratch: Path) -> None:
    """Simulates `example` and copies its plots into the docs' images."""
    request = service.SimulateRequest(
        script=ROOT / "examples" / f"{example}.py",
        output=scratch / f"{example}.jolty.json",
        results=scratch / f"{example}.results.npz",
        probes=probes,
    )
    result = await service.simulate(request, Settings())
    for plot in result.plots:
        target = IMAGES / plot.path.name
        shutil.copyfile(plot.path, target)
        print(f"wrote {target.relative_to(ROOT)}")


async def main() -> None:
    """Draws every image."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as scratch:
        for example, probes in EXAMPLES.items():
            await draw(example, probes, Path(scratch))


if __name__ == "__main__":
    asyncio.run(main())
