# Copyright 2026 Maxence Faldor
"""Foundry process design kits: digest-verified model fetching and library loading.

A [`Kit`][jolty.spice.kits.Kit] pins one release of ngspice model files verified by
SHA-256 digest and arranged in open_pdks directory layout so local installations work
identically. [`root`][jolty.spice.kits.root] downloads and caches the pinned release
on first use.

[`read`][jolty.spice.kits.read] loads a kit's model libraries at selected process
corners into a [`Library`][jolty.spice.Library] with ngspice semantics. Supported
features include ``.lib`` sections, ``.include``, ``.option scale``, parameter
expressions, ``.if`` blocks, binned models, ``temper``, and statistical functions
(``agauss``, ``gauss``, ``aunif``, ``unif``, ``limit``).
[`inventory`][jolty.spice.kits.inventory] classifies every kit device by model coverage
and native simulation readiness.
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
    """Single file or archive belonging to a pinned PDK release.

    Archives extract members under `within` preserving relative paths; plain files
    copy directly to `within` inside the kit root.
    """

    url: str
    digest: str
    """Lowercase hexadecimal SHA-256 digest of the downloaded payload."""
    within: PurePosixPath
    archive: bool = False
    """True when the payload is a Zstandard-compressed tar archive from ciel."""


type Section = tuple[PurePosixPath, str]
"""Model library section identified by relative library path and section name."""


@dataclass(frozen=True, slots=True)
class Corners:
    """Model library paired with its available process corner section names.

    The first entry in `sections` is the typical corner.
    """

    library: PurePosixPath
    """Relative path of the library file within the kit root."""
    sections: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Kit:
    """Foundry process design kit pinned to a specific release of ngspice models.

    ngspice loads all files in `includes`, then one section per entry in `corners` to
    define the kit devices. Entries in `statistics` select process-variation sections.
    """

    name: str
    version: str
    """Pinned upstream commit hash for this kit release."""
    files: tuple[Fetched, ...]
    includes: tuple[PurePosixPath, ...]
    corners: tuple[Corners, ...]
    statistics: tuple[Section, ...] = ()
    compatibility: Compatibility | None = None
    """Optional ngspice compatibility mode required by the kit, or None for default."""

    @property
    def typical(self) -> tuple[Section, ...]:
        """Typical corner section for every library in `corners`."""
        return tuple((each.library, each.sections[0]) for each in self.corners)

    @property
    def sections(self) -> tuple[Section, ...]:
        """All corner and statistical sections across the kit libraries."""
        return (
            *(
                (each.library, section)
                for each in self.corners
                for section in each.sections
            ),
            *self.statistics,
        )


def root(kit: Kit, settings: Settings | None = None) -> Path:
    """Returns the root directory containing `kit`, downloading and caching if needed.

    Uses ``settings.pdk_root`` when configured; otherwise downloads, verifies, and
    caches the pinned release on first call.

    Args:
        kit: Process design kit specification to locate or fetch.
        settings: Optional runtime settings overriding `Settings.load()`.

    Returns:
        Filesystem directory containing the verified model files for `kit`.

    Raises:
        DownloadError: When a download fails or fails SHA-256 verification.
        JoltyError: When ``pdk_root`` lacks a required library or settings are invalid.
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
    """Loads `kit` model libraries at `sections` into a `Library` using ngspice rules.

    Includes top-level kit files first, then loads `sections` (defaulting to each
    library's typical section) in the kit's compatibility mode with ``temper`` set to
    `temperature` in kelvins.

    Args:
        kit: Process design kit specification to load.
        sections: Optional `(library, section)` pairs overriding `kit.typical`.
        temperature: Circuit temperature in kelvins assigned to SPICE ``temper``.
        settings: Optional runtime settings overriding `Settings.load()`.

    Returns:
        `Library` populated with `kit` includes and selected corner sections.

    Raises:
        DownloadError: When a download fails or fails SHA-256 verification.
        JoltyError: When ``pdk_root`` lacks a required library or settings are invalid.
        SpiceError: When a model file contains unsupported SPICE syntax.
    """
    found = root(kit, settings)
    library = Library(temperature, kit.compatibility)
    for path in kit.includes:
        library.include(found / path)
    for path, section in kit.typical if sections is None else sections:
        library.include(found / path, section)
    return library


type Status = Literal["native", "verilog-a", "neither", "unread", "unvalued"]
"""Native simulation readiness classification for a PDK device.

Values are ``native`` (all models built in), ``verilog-a`` (remaining models are
Verilog-A), ``neither`` (unsupported SPICE model), ``unread`` (unsupported element
syntax), and ``unvalued`` (missing required instance parameters).
"""


@dataclass(frozen=True, slots=True)
class Device:
    """Top-level subcircuit or model defined by a PDK and its native coverage status."""

    name: str
    kind: Literal["subcircuit", "model"]
    """Declaration kind: ``subcircuit`` (instantiated by ``X``) or ``model``."""
    uses: tuple[Use, ...]
    """Element usages, required models, and any translation blockers."""
    parameters: Mapping[str, float] = field(default_factory=dict)
    """Override parameters used during survey when subcircuit defaults are unbinned.

    Populated with bin-midpoint ``l``/``w``, caller placement parameters, or the 10 µm
    ngspice fallback resistor size when default parameters cannot be surveyed directly.
    """

    @property
    def models(self) -> tuple[str, ...]:
        """Deduplicated model names referenced by device elements, in order."""
        return tuple(dict.fromkeys(use.model for use in self.uses if use.model))

    @property
    def status(self) -> Status:
        """Readiness status derived from element model support and parameter resolution.

        Unrecognized models yield ``verilog-a`` if all are Verilog-A or ``neither``
        otherwise. Remaining translation errors yield ``unread``, and missing parameter
        values yield ``unvalued``; clean devices return ``native``.
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
"""Default 10 µm square length and width fallback for unvalued PDK subcircuits."""


def inventory(
    kit: Kit,
    sections: Sequence[Section] | None = None,
    settings: Settings | None = None,
) -> tuple[Device, ...]:
    """Surveys every top-level subcircuit and model defined by `kit` at `sections`.

    Defaults to typical sections when `sections` is omitted. Each top-level subcircuit
    is surveyed at its defaults via [`Library.survey`][jolty.spice.Library.survey].
    Subcircuits whose defaults miss MOSFET bins retry at the first bin midpoint in
    library ``scale`` units. Subcircuits with undefined defaults retry with parameters
    from their first call site or the 10 µm square fallback in ``scale`` units.

    Args:
        kit: Process design kit specification to survey.
        sections: Optional `(library, section)` pairs overriding `kit.typical`.
        settings: Optional runtime settings overriding `Settings.load()`.

    Returns:
        Surveyed `Device` records for every top-level subcircuit and global model.

    Raises:
        DownloadError: When a download fails or fails SHA-256 verification.
        JoltyError: When ``pdk_root`` lacks a required library or settings are invalid.
        SpiceError: When a model file contains unsupported SPICE syntax.
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
    """Re-surveys an unvalued `device` using call-site parameters or fallback sizes.

    Uses `placements` first, falling back to `_DEFAULT_SIZES` scaled by library
    ``scale`` for parameters defaulting to SPICE ``l`` or ``w``. Keeps the updated
    survey only when all undefined-parameter errors clear.
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
    """Surveys subcircuit `name` in `library`, retrying at a MOSFET bin midpoint.

    When default parameters fall outside every MOSFET model bin and the subcircuit
    accepts `l` and `w`, re-surveys at the first bin midpoint in library ``scale``
    units.
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
