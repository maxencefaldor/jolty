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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final, Literal

from jolty import _core
from jolty.errors import JoltyError
from jolty.settings import Settings
from jolty.spice import models
from jolty.spice.download import fetch
from jolty.spice.expressions import Name
from jolty.spice.library import SIZES, VERILOG_A, Library, Use
from jolty.spice.ngspice import Compatibility


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
    compatibility: Compatibility | None = None
    """The compatibility mode ngspice reads the kit in, its own unless given."""

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
    one unless given, in the kit's compatibility mode; ``temper`` is `temperature`, in
    kelvins.

    Raises:
        DownloadError: When a file cannot be downloaded, or its digest is wrong.
        JoltyError: When the settings' PDK root lacks one of the kit's libraries, or the
            settings are invalid.
        SpiceError: When a file holds what jolty cannot read.
    """
    found = root(kit, settings)
    library = Library(temperature, kit.compatibility)
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
    parameters: Mapping[str, float] = field(default_factory=dict)
    """The parameters it is read with, its defaults but these: a subcircuit's length
    and width, `l` and `w`, where its defaults are a size no bin holds; or, where its
    defaults read a parameter nothing defines, those the first placement of it within
    another subcircuit gives, or its size, where a parameter defaults to SPICE's length
    or width, `l` or `w`, at the size ngspice gives a resistor that gives none, 10 µm
    square."""

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


_DEFAULT_SIZES: Final = {
    "l": models.DEFAULT_RESISTOR_LENGTH,
    "w": models.DEFAULT_WIDTH,
}
"""The length and the width, in meters, ngspice gives a resistor whose element and card
give none, 10 µm each: the size the inventory reads a subcircuit at whose parameters
default to SPICE's length and width, `l` and `w`, that nothing defines, as GF180MCU's
resistors and capacitors do."""


def inventory(
    kit: Kit,
    sections: Sequence[Section] | None = None,
    settings: Settings | None = None,
) -> tuple[Device, ...]:
    """Each device `kit` defines at `sections`, each library's typical one unless given.

    Its devices are the subcircuits and the models its libraries define outside any
    subcircuit, each read at its parameters' defaults, as
    [`Library.survey`][jolty.spice.Library.survey] reads it; a subcircuit whose
    defaults are a size that no bin of a MOSFET it holds holds, but that gives its
    length `l` and width `w`, is read again at the middle of that model's first bin,
    in the units of the library's `scale`, and is stated so where every bin it selects
    then holds its size. A subcircuit whose defaults read a parameter nothing defines
    is read again as the first placement of it within another subcircuit places it, or,
    where none does, with each of its parameters that defaults to `l` or `w` at the size
    ngspice gives a resistor that gives none, 10 µm square, in the units of the
    library's `scale`, and is stated so where it then reads every parameter.

    Raises:
        DownloadError: When a file cannot be downloaded, or its digest is wrong.
        JoltyError: When the settings' PDK root lacks one of the kit's libraries, or the
            settings are invalid.
        SpiceError: When a file holds what jolty cannot read.
    """
    library = read(kit, sections, settings=settings)
    subcircuits = [_subcircuit(library, name) for name in library.subcircuits]
    placements: dict[str, Mapping[str, float]] = {}
    for device in subcircuits:
        for use in device.uses:
            if use.placed is not None:
                placed, parameters = use.placed
                placements.setdefault(placed.casefold(), parameters)
    return (
        *(_valued(library, device, placements) for device in subcircuits),
        *(
            Device(name, "model", (library.survey_model(name),))
            for name in library.models
        ),
    )


def _valued(
    library: Library, device: Device, placements: Mapping[str, Mapping[str, float]]
) -> Device:
    """`device`, read again with values for parameters it reads that nothing defines.

    It is read as `placements` place it, by subcircuit, or else with each parameter that
    defaults to SPICE's length or width at `_DEFAULT_SIZES`, in the units of the
    library's `scale`; and stated so where it then reads every parameter.
    """
    if device.status != "unvalued":
        return device
    subcircuit = library.subcircuit(device.name)
    if subcircuit is None:
        return device
    parameters = placements.get(device.name.casefold())
    if parameters is None:
        defined = library.values()
        scale = library.options.get("scale", 1.0)
        parameters = {
            parameter: _DEFAULT_SIZES[default.name] / scale
            for parameter, default in subcircuit.defaults.items()
            if isinstance(default, Name)
            and default.name in _DEFAULT_SIZES
            and default.name not in defined
        }
    if not parameters:
        return device
    uses = library.survey(device.name, **parameters)
    if any(use.problem is not None and use.undefined is not None for use in uses):
        return device
    return Device(device.name, "subcircuit", uses, parameters)


def _subcircuit(library: Library, name: str) -> Device:
    """The subcircuit `name` of `library`, read at a size a bin holds if it can be.

    It is read at its defaults; where a MOSFET's size is one no bin holds, and the
    subcircuit gives its length and width, `l` and `w`, again at the middle of the
    first bin of that MOSFET's model, in the units of the library's `scale`.
    """
    uses = library.survey(name)
    middle = next((use.middle for use in uses if use.middle is not None), None)
    subcircuit = library.subcircuit(name)
    sized = subcircuit is not None and subcircuit.defaults.keys() >= SIZES.keys()
    if middle is None or subcircuit is None or not sized:
        return Device(name, "subcircuit", uses)
    scale = library.options.get("scale", 1.0)
    length, width = middle
    parameters = {"l": length / scale, "w": width / scale}
    resized = library.survey(name, **parameters)
    if any(use.middle is not None for use in resized):
        return Device(name, "subcircuit", uses)
    return Device(name, "subcircuit", resized, parameters)
