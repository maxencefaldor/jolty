# Copyright 2026 Maxence Faldor
"""Foundries' process design kits: their model files, fetched by digest, and read.

A [`Kit`][jolty.spice.kits.Kit] pins one release of a kit's ngspice model files, each
file or archive fetched from where it is published and checked against its SHA-256
digest, and laid out as open_pdks installs the kit, so a kit installed elsewhere reads
alike. [`root`][jolty.spice.kits.root] fetches it into the cache the first time.

[`read`][jolty.spice.kits.read] reads a kit's model libraries at a process corner, one
section of each, as ngspice reads them, into a [`Library`][jolty.spice.Library]:
``.lib`` sections, ``.include``, ``.option scale``, parameters and their expressions,
``.if`` blocks, binned models, ``temper`` at the temperature it reads at, and the
statistical functions ``agauss``, ``gauss``, ``aunif``, ``unif`` and ``limit``, drawn
per instance from a sample's seed and index, or at their nominal values without one.
[`inventory`][jolty.spice.kits.inventory] states, for each of the kit's devices, the
models it needs and whether jolty has them.
"""

import shutil
import tarfile
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from jolty import _core
from jolty.errors import JoltyError
from jolty.settings import Settings
from jolty.spice.download import fetch
from jolty.spice.library import VERILOG_A, Library, Use


@dataclass(frozen=True, slots=True)
class Fetched:
    """One file of a kit's release: where it is published, its digest, and its place.

    An archive keeps its members under `within`, at their own paths; any other file is
    kept at `within`, a path within the kit's root.
    """

    url: str
    digest: str
    """Its SHA-256 digest, in hexadecimal."""
    within: PurePosixPath
    archive: bool = False
    """Whether it is a tar archive compressed with Zstandard, as ciel publishes kits."""


type Section = tuple[PurePosixPath, str]
"""A section of a model library: the library's path within its kit's root, and the
section's name."""


@dataclass(frozen=True, slots=True)
class Corners:
    """A model library of a kit, and its sections, each a process corner it reads at.

    The first section is the typical corner.
    """

    library: PurePosixPath
    """Its path within the kit's root."""
    sections: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Kit:
    """A foundry's process design kit, pinned at one release of its model files.

    ngspice reads the files `includes` names whole, then one section of each library of
    `corners`; together they define the kit's devices. A section of `statistics` draws
    the process's variation instead of a corner's.
    """

    name: str
    version: str
    """The commit of the release's sources."""
    files: tuple[Fetched, ...]
    includes: tuple[PurePosixPath, ...]
    corners: tuple[Corners, ...]
    statistics: tuple[Section, ...] = ()

    @property
    def typical(self) -> tuple[Section, ...]:
        """Each library's typical section."""
        return tuple((each.library, each.sections[0]) for each in self.corners)

    @property
    def sections(self) -> tuple[Section, ...]:
        """Every section of the kit's libraries, corners and statistics."""
        return (
            *(
                (each.library, section)
                for each in self.corners
                for section in each.sections
            ),
            *self.statistics,
        )


def root(kit: Kit, settings: Settings | None = None) -> Path:
    """The directory that holds `kit`: the settings' PDK root, or its pinned release.

    The pinned release's files are fetched into the settings' cache the first time,
    each checked against its digest, and read from it after.

    Raises:
        DownloadError: When a file cannot be downloaded, or its digest is wrong.
        JoltyError: When the settings' PDK root lacks one of the kit's libraries, or the
            settings are invalid.
    """
    chosen = Settings.load() if settings is None else settings
    needed = (*kit.includes, *(each.library for each in kit.corners))
    if chosen.pdk_root is not None:
        missing = [path for path in needed if not (chosen.pdk_root / path).is_file()]
        if missing:
            message = f"the PDK root {chosen.pdk_root} holds no {missing[0]}"
            raise JoltyError(message)
        return chosen.pdk_root
    kept = chosen.cache / "pdks" / f"{kit.name}-{kit.version}"
    if all((kept / path).is_file() for path in needed):
        return kept
    kept.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{kept.name}.", dir=kept.parent))
    try:
        for each in kit.files:
            _place(fetch(each.url, each.digest, chosen.cache), each, staging)
        try:
            staging.rename(kept)
        except OSError:
            # Another process fetching the release at the same time kept it first.
            if not all((kept / path).is_file() for path in needed):
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return kept


def _place(fetched: Path, each: Fetched, staging: Path) -> None:
    """Lays `fetched`, the file `each` names, out within `staging`."""
    if not each.archive:
        target = staging / each.within
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fetched, target)
        return
    within = f"{each.within}/"
    with tarfile.open(fetched, "r:zst") as packed:
        members = [
            member
            for member in packed
            if member.name.startswith(within) and (member.isfile() or member.isdir())
        ]
        packed.extractall(staging, members=members, filter="data")


def read(
    kit: Kit,
    sections: Sequence[Section] | None = None,
    temperature: float = _core.REFERENCE_TEMPERATURE,
    settings: Settings | None = None,
) -> Library:
    """`kit`'s model libraries, read at `sections`, as ngspice reads them.

    The files the kit includes come first, then each section, each library's typical
    one unless given; ``temper`` is `temperature`, in kelvins.

    Raises:
        DownloadError: When a file cannot be downloaded, or its digest is wrong.
        JoltyError: When the settings' PDK root lacks one of the kit's libraries, or the
            settings are invalid.
        SpiceError: When a file holds what jolty cannot read.
    """
    found = root(kit, settings)
    library = Library(temperature)
    for path in kit.includes:
        library.include(found / path)
    for path, section in kit.typical if sections is None else sections:
        library.include(found / path, section)
    return library


type Status = Literal["native", "verilog-a", "neither", "unread", "unvalued"]
"""What jolty has of a device's models: every one natively, ``native``; all but
Verilog-A models, which its compiler compiles rather than loads, ``verilog-a``; not
some SPICE model, ``neither``; or every model its elements need, but its reader cannot
read how the kit places some subcircuit, ``unread``, or some element reads a parameter
that has no value, which an instance gives, ``unvalued``."""


@dataclass(frozen=True, slots=True)
class Device:
    """A device a kit defines, a subcircuit or a model, and what jolty has of it."""

    name: str
    kind: Literal["subcircuit", "model"]
    """Whether it is a subcircuit, which an ``X`` element places, or a model, which an
    element of its type names."""
    uses: tuple[Use, ...]
    """Each element it holds, the model it needs, and why jolty cannot read it."""

    @property
    def models(self) -> tuple[str, ...]:
        """The models its elements need, each once, in order."""
        return tuple(dict.fromkeys(use.model for use in self.uses if use.model))

    @property
    def status(self) -> Status:
        """What jolty has of its models.

        An element that jolty cannot read for its model makes the device's status
        ``verilog-a`` if every such element is a Verilog-A model, ``neither`` if not;
        failing those, a placement it cannot read makes it ``unread``, and an element it
        cannot read for want of a parameter's value alone ``unvalued``.
        """
        problems = [use for use in self.uses if use.problem is not None]
        valued = [use for use in problems if use.undefined is None]
        models = [use for use in valued if use.model]
        if models:
            verilog = all(use.model.startswith(VERILOG_A) for use in models)
            return "verilog-a" if verilog else "neither"
        if valued:
            return "unread"
        return "unvalued" if problems else "native"


def inventory(
    kit: Kit,
    sections: Sequence[Section] | None = None,
    settings: Settings | None = None,
) -> tuple[Device, ...]:
    """Each device `kit` defines at `sections`, each library's typical one unless given.

    Its devices are the subcircuits and the models its libraries define outside any
    subcircuit, each read at its parameters' defaults, as
    [`Library.survey`][jolty.spice.Library.survey] reads it.

    Raises:
        DownloadError: When a file cannot be downloaded, or its digest is wrong.
        JoltyError: When the settings' PDK root lacks one of the kit's libraries, or the
            settings are invalid.
        SpiceError: When a file holds what jolty cannot read.
    """
    library = read(kit, sections, settings=settings)
    return (
        *(
            Device(name, "subcircuit", library.survey(name))
            for name in library.subcircuits
        ),
        *(
            Device(name, "model", (library.survey_model(name),))
            for name in library.models
        ),
    )
