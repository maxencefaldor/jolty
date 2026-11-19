# Copyright 2026 Maxence Faldor
"""Independent exact reference solutions for precision benchmark fixtures.

Reference values come exclusively from closed-form formulas or high-precision solutions
of circuit equations, never from another simulator run.

A linear circuit driven by a piecewise-linear source evolves each natural mode ``k`` by
``żₖ = λₖ zₖ + u`` and yields output ``p`` as ``Σₖ Rₚₖ zₖ`` (`Modal`). Over a linear
segment ``u = uⱼ + sⱼ τ``, each mode follows
``zₖ(τ) = zₖ(0) e^{λₖτ} + uⱼ τ φ₁(λₖτ) + sⱼ τ² φ₂(λₖτ)``.
The auxiliary functions are ``φ₁(x) = (eˣ - 1)/x`` and
``φ₂(x) = (eˣ - 1 - x)/x²``. mpmath computes modal rates,
residues, and segment-boundary states at `DIGITS` decimal digits before rounding to
doubles. Double-precision evaluation at simulator sample times also computes a rounding
bound capping claimable digits.

Small circuits use mpmath's general eigensolver (`linear`). Large RC ladders and random
trees exploit tree topology (`tree`) via zero-fill elimination: Newton iteration refines
LAPACK eigenvalue estimates, Sylvester's inertia verifies eigenvalue ordering, and
inverse iteration computes eigenvectors.
"""

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import mpmath
import numpy as np
import numpy.typing as npt
from mpmath import mp, mpf
from mpmath.ctx_mp_python import _mpc, _mpf

type Real = npt.NDArray[np.float64]
type Complex = npt.NDArray[np.complex128]
type Precise = _mpf | _mpc
"""A number in mpmath, real or complex: the classes its context makes its numbers of,
which it exports only as the context's attributes."""
type Points = Sequence[tuple[float, float]]
"""A piecewise-linear level: its points, each a time and a level, the times rising from
zero, the last level held."""

DIGITS: Final = 50
"""The decimal digits mpmath works with: 34 past the double each answer is rounded
to, so that what its elimination of a stiff network loses, five digits at most on the
networks here, never reaches the double's."""

mp.dps = DIGITS  # every mpmath number of the suite, the fixtures' equations among them

CONVERGED: Final = mpf(10) ** -(DIGITS - 10)
"""Relative Newton-step convergence tolerance for eigenvalue refinement. Set ten digits
above the 10⁻⁴⁵ determinant rounding floor and well below double precision."""

NEWTON: Final = 12
"""Maximum Newton iterations per eigenvalue from LAPACK estimates (under 10⁻⁸ initial
relative error, typically reaching `CONVERGED` in four quadratic steps)."""

EPSILON: Final = float(np.finfo(np.float64).eps)
"""IEEE 754 double-precision machine epsilon (2⁻⁵²), twice the unit roundoff."""

ROUNDINGS: Final = 16
"""Maximum per-term rounding units in a modal sum prior to summation. Accounts for
residue, state/level, and time conversions, up to 7x cancellation in direct ``φ``
evaluation above `SERIES`, and four operations per complex product."""

SERIES: Final = 1.0
"""Modulus threshold on ``λτ`` separating Taylor-series evaluation of ``φ₁`` and ``φ₂``
(cancellation-free) from direct formulas (at most 7x cancellation near ``λτ = -1``)."""

TERMS: Final = 21
"""Number of Taylor terms summed below `SERIES`, ending at ``1/22!`` (about 10⁻²¹)."""


@dataclass(frozen=True, slots=True)
class Piece:
    """Single linear segment of a piecewise-linear source starting at `start`.

    Stores the initial source level, segment slope, and modal state vector at `start`.
    """

    start: float
    level: float
    slope: float
    states: Complex


@dataclass(frozen=True, slots=True)
class Modal:
    """Modal representation of a linear circuit's exact piecewise-linear response."""

    outputs: tuple[str, ...]
    """Output signal names such as ``v(out)``."""
    rates: Complex
    """Modal rates ``λₖ`` with non-positive real parts (zero when undamped)."""
    residues: Complex
    """Modal residue matrix with one row per output and one column per mode."""
    pieces: tuple[Piece, ...]

    def at(self, times: Real) -> tuple[Real, Real]:
        """Evaluates outputs at `times` and returns values with rounding error bounds.

        Bounds absolute error by summing `ROUNDINGS` plus one summation rounding
        per mode times each term's magnitude. Propagates two rounding units in ``λₖτ``
        through condition numbers ``|λτ|`` for ``e^{λτ}``, ``|λτ eᵗ/(eᵗ - 1)| + 1`` for
        ``φ₁``, and ``|λτ (eᵗ - 1)/(eᵗ - 1 - λτ)| + 2`` for ``φ₂`` (``t = λτ``).
        """
        times = np.asarray(times, dtype=np.float64)
        starts = np.array([piece.start for piece in self.pieces])
        which = np.maximum(np.searchsorted(starts, times, side="right") - 1, 0)
        values = np.zeros((len(self.outputs), len(times)))
        bound = np.zeros_like(values)
        real = not np.any(self.rates.imag) and not np.any(self.residues.imag)
        residues = self.residues.real if real else self.residues
        magnitudes = np.abs(self.residues)
        for index, piece in enumerate(self.pieces):
            chosen = which == index
            if not chosen.any():
                continue
            elapsed = times[chosen] - piece.start
            arguments = np.outer(self.rates, elapsed)
            first, second = phi(arguments)
            decays = np.exp(arguments)
            held = np.abs(piece.states[:, None] * decays)
            ramped = abs(piece.level) * elapsed * np.abs(first)
            sloped = abs(piece.slope) * elapsed**2 * np.abs(second)
            modes = piece.states[:, None] * decays + piece.level * elapsed * first
            modes += piece.slope * elapsed**2 * second
            values[:, chosen] = (residues @ (modes.real if real else modes)).real
            sizes = held + ramped + sloped
            conditioned = _conditioned(arguments, decays, held, ramped, sloped)
            terms = (len(self.rates) + ROUNDINGS) * (magnitudes @ sizes)
            bound[:, chosen] = EPSILON * (terms + 2 * (magnitudes @ conditioned))
        return values, bound


def _conditioned(
    arguments: Complex, decays: Complex, held: Real, ramped: Real, sloped: Real
) -> Real:
    """Weights each modal term magnitude by its condition number in ``λτ``."""
    size = np.abs(arguments)
    grown = np.abs(np.expm1(arguments))
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.where(grown > 0, size * np.abs(decays) / grown, 1.0) + 1
        rest = np.abs(np.expm1(arguments) - arguments)
        second = np.where(rest > 0, size * grown / rest, 2.0) + 2
    return size * held + first * ramped + second * sloped


def phi(arguments: Complex) -> tuple[Complex, Complex]:
    """Evaluates ``φ₁`` and ``φ₂`` at `arguments`, using Taylor series near zero."""
    z = np.asarray(arguments, dtype=np.complex128)
    near = np.abs(z) < SERIES
    first, second = np.zeros_like(z), np.zeros_like(z)
    term = np.ones_like(z)
    for order in range(TERMS):
        term = term / (order + 1)
        first += term
        second += term / (order + 2)
        term = term * z
    far = np.where(near, 1.0, z)
    growth = np.expm1(far)
    direct_first = growth / far
    direct_second = (growth - far) / far**2
    return np.where(near, first, direct_first), np.where(near, second, direct_second)


def _phi(z: Precise) -> tuple[Precise, Precise]:
    """Evaluates ``φ₁`` and ``φ₂`` at `z` in mpmath precision.

    Sums Taylor series near zero to prevent cancellation in direct exponential formulas.
    """
    if abs(z) < SERIES:
        first, second, term = mpf(0), mpf(0), mpf(1)
        for order in range(4 * DIGITS):
            first += term / (order + 1)
            second += term / ((order + 1) * (order + 2))
            term = term * z / (order + 1)
            if abs(term) < mpf(10) ** -(2 * DIGITS):
                break
        return first, second  # ty: ignore[unsound-return] - mpmath is untyped
    growth = mpmath.expm1(z)
    return growth / z, (growth - z) / z**2  # ty: ignore[unsound-return] - mpmath is untyped


def modal(
    outputs: Sequence[str],
    rates: Sequence[Precise],
    residues: Sequence[Sequence[Precise]],
    level: Points,
) -> Modal:
    """Constructs a `Modal` response for `rates` and `residues` driven by `level`.

    Initializes every mode at operating point ``-u₀/λₖ`` and propagates segment-boundary
    states analytically in mpmath at `DIGITS` decimal digits.
    """
    times = [mpf(time) for time, _ in level]
    levels = [mpf(value) for _, value in level]
    slopes = [
        (b - a) / (t1 - t0)
        for t0, t1, a, b in zip(times, times[1:], levels, levels[1:], strict=False)
    ]
    slopes.append(mpf(0))
    states = [-levels[0] / rate for rate in rates]
    pieces = [Piece(0.0, float(levels[0]), float(slopes[0]), _rounded(states))]
    for index in range(1, len(times)):
        span = times[index] - times[index - 1]
        moved = []
        for rate, state in zip(rates, states, strict=True):
            first, second = _phi(rate * span)
            moved.append(
                state * mpmath.exp(rate * span)
                + levels[index - 1] * span * first
                + slopes[index - 1] * span**2 * second
            )
        states = moved
        pieces.append(
            Piece(
                float(times[index]),
                float(levels[index]),
                float(slopes[index]),
                _rounded(states),
            )
        )
    return Modal(
        tuple(outputs),
        _rounded(rates),
        np.array([_rounded(row) for row in residues]),
        tuple(pieces),
    )


def _rounded(values: Sequence[Precise]) -> Complex:
    """Rounds mpmath `values` to the nearest complex doubles."""
    return np.array([complex(value) for value in values])


def linear(
    outputs: Sequence[str],
    states: Sequence[Sequence[Precise]],
    inputs: Sequence[Precise],
    read: Sequence[Sequence[Precise]],
    level: Points,
) -> Modal:
    """Computes exact modal response for ``ẋ = A x + b u``, ``y = C x``.

    Takes `states` (``A``), `inputs` (``b``), and `read` (``C``) row by row in SI units
    as mpmath values, assuming distinct eigenvalues of ``A``. Uses mpmath eigenvectors
    ``V`` to compute input weights ``w = V⁻¹ b`` and residues ``(C V)ₚₖ wₖ``.
    """
    a = mpmath.matrix([[mpf(value) for value in row] for row in states])
    b = mpmath.matrix([mpf(value) for value in inputs])
    c = mpmath.matrix([[mpf(value) for value in row] for row in read])
    rates, vectors = mpmath.eig(a)
    weights = mpmath.lu_solve(vectors, b)
    seen = c * vectors
    residues = [
        [seen[p, k] * weights[k] for k in range(len(rates))]
        for p in range(len(outputs))
    ]
    return modal(outputs, list(rates), residues, level)


def poles(states: Sequence[Sequence[Precise]]) -> Complex:
    """Computes the eigenvalues of state matrix `states` (``A``) in mpmath."""
    a = mpmath.matrix([[mpf(value) for value in row] for row in states])
    found = mpmath.eig(a, left=False, right=False)
    return np.array([complex(value) for value in found])


@dataclass(frozen=True, slots=True)
class Tree:
    """A network of resistors and capacitors joined as a tree, under a voltage source.

    Node 0 joins the source through `source` siemens; each node ``i`` past it joins
    its parent, a node before it, through ``branches[i]`` siemens, ``branches[0]``
    unused; and each node holds ``capacitances[i]`` farads to the ground. A ladder is
    the tree whose every node's parent is the node before it.
    """

    parents: tuple[int, ...]
    branches: tuple[float, ...]
    capacitances: tuple[float, ...]
    source: float


def tree(network: Tree, outputs: Sequence[int], level: Points) -> Modal:
    """Computes exact modal response of RC tree `network` at `outputs` under `level`.

    Changing coordinates via ``v = C^{-1/2} y`` transforms ``C v̇ = -G v + g₀ e₀ u``
    into symmetric ``ẏ = -A y + C^{-1/2} g₀ e₀ u`` with ``A = C^{-1/2} G C^{-1/2}``.
    Eigenpairs ``(μₖ, qₖ)`` give modes of rate ``-μₖ`` and node residues
    ``qₖ[p] qₖ[0] g₀ / √(cₚ c₀)``. Summing residues divided by rates verifies ``-1``
    at every node because unit DC gain holds everywhere.

    Raises:
        ArithmeticError: When an eigenvalue fails to converge, violates Sylvester
            inertia ordering, or fails the unit DC-gain check.
    """
    rates, vectors = _modes(network)
    roots = [mpmath.sqrt(mpf(c)) for c in network.capacitances]
    source = mpf(network.source)
    weights = [vector[0] * source / roots[0] for vector in vectors]
    residues = [
        [
            vector[p] * weight / roots[p]
            for vector, weight in zip(vectors, weights, strict=True)
        ]
        for p in outputs
    ]
    for p, row in zip(outputs, residues, strict=True):
        gain = -sum((r / rate for r, rate in zip(row, rates, strict=True)), mpf(0))
        if abs(gain - 1) > CONVERGED:
            message = f"node {p}'s modes sum to a gain of {float(gain)!r}, not 1"
            raise ArithmeticError(message)
    return modal([f"v(n{p})" for p in outputs], rates, residues, level)


def _modes(network: Tree) -> tuple[list[_mpf], list[list[_mpf]]]:
    """Computes modal rates ``-μₖ`` (ascending ``μ``) and eigenvectors ``qₖ``.

    Raises:
        ArithmeticError: When an eigenvalue fails to converge or violates Sylvester
            inertia ordering.
    """
    count = len(network.capacitances)
    c = [mpf(value) for value in network.capacitances]
    diagonal = [mpf(0)] * count
    diagonal[0] += mpf(network.source)
    for node in range(1, count):
        g = mpf(network.branches[node])
        diagonal[node] += g
        diagonal[network.parents[node]] += g
    a = [diagonal[node] / c[node] for node in range(count)]
    off = [mpf(0)] + [
        -mpf(network.branches[node]) / mpmath.sqrt(c[node] * c[network.parents[node]])
        for node in range(1, count)
    ]
    elimination = _Elimination(network.parents, a, off)
    dense = np.diag([float(value) for value in a])
    for node in range(1, count):
        parent = network.parents[node]
        dense[node, parent] = dense[parent, node] = float(off[node])
    refined: list[_mpf] = []
    for estimate in np.linalg.eigvalsh(dense):
        mu = mpf(float(estimate))
        for _ in range(NEWTON):
            step = elimination.newton(mu)
            mu -= step
            if abs(step) <= CONVERGED * abs(mu):
                break
        else:
            message = f"the eigenvalue near {float(estimate)!r} did not converge"
            raise ArithmeticError(message)
        refined.append(mu)
    shifts = [
        refined[0] / 2,
        *((low + high) / 2 for low, high in itertools.pairwise(refined)),
        refined[-1] * 2,
    ]
    for place, shift in enumerate(shifts):
        below = elimination.below(shift)
        if below != place:
            message = (
                "the eigenvalues refined from LAPACK's are not the network's own: "
                f"{below} lie below {float(shift)!r}, where {place} should"
            )
            raise ArithmeticError(message)
    return [-mu for mu in refined], [elimination.vector(mu) for mu in refined]


@dataclass(frozen=True, slots=True)
class _Elimination:
    """The elimination of a tree's symmetric matrix shifted by ``μ``, leaves first.

    Each node is eliminated after its children, which follow it, so nothing fills:
    ``dᵢ(μ) = aᵢ - μ - Σ_{children j} oⱼ² / dⱼ(μ)``, the pivots of ``A - μI``.
    """

    parents: tuple[int, ...]
    diagonal: Sequence[_mpf]
    off: Sequence[_mpf]

    def pivots(self, mu: _mpf) -> tuple[list[_mpf], list[_mpf]]:
        """Computes elimination pivots ``dᵢ(μ)`` and their ``μ`` derivatives."""
        pivots = [value - mu for value in self.diagonal]
        slopes = [mpf(-1)] * len(pivots)
        for node in range(len(pivots) - 1, 0, -1):
            parent = self.parents[node]
            square = self.off[node] ** 2
            pivot = pivots[node]
            pivots[parent] -= square / pivot
            slopes[parent] += square * slopes[node] / pivot**2
        return pivots, slopes  # ty: ignore[invalid-explicit-return] - mpmath is untyped

    def newton(self, mu: _mpf) -> _mpf:
        """Computes the Newton step for ``det(A - μI) = Πᵢ dᵢ(μ)``."""
        pivots, slopes = self.pivots(mu)
        if pivots[0] == 0:
            # Exact root: lim_{d_0 -> 0} 1 / (s_0 / d_0 + finite) = 0.
            return mpf(0)
        logarithmic = sum(
            (slope / pivot for slope, pivot in zip(slopes, pivots, strict=True)), mpf(0)
        )
        return 1 / logarithmic  # ty: ignore[unsound-return] - mpmath is untyped

    def below(self, mu: _mpf) -> int:
        """Counts eigenvalues below `mu` from negative pivots via Sylvester inertia."""
        pivots, _ = self.pivots(mu)
        return sum(1 for pivot in pivots if pivot < 0)

    def vector(self, mu: _mpf) -> list[_mpf]:
        """Computes the unit eigenvector at `mu` via two inverse iterations."""
        pivots, _ = self.pivots(mu)
        count = len(pivots)
        if pivots[0] == 0:
            # Exact root of L D L^T with D_{0,0} = 0: nullspace solves L^T v = e_0.
            solved = [mpf(0)] * count
            solved[0] = mpf(1)
            for node in range(1, count):
                above = solved[self.parents[node]]
                solved[node] = -self.off[node] * above / pivots[node]
            norm = mpmath.sqrt(sum((value**2 for value in solved), mpf(0)))
            return [value / norm for value in solved]
        vector = [mpf(1) + mpf(node % 7) / 7 for node in range(count)]
        for _ in range(2):
            rest = list(vector)
            for node in range(count - 1, 0, -1):
                rest[self.parents[node]] -= self.off[node] / pivots[node] * rest[node]
            solved = [mpf(0)] * count
            solved[0] = rest[0] / pivots[0]
            for node in range(1, count):
                above = solved[self.parents[node]]
                solved[node] = (rest[node] - self.off[node] * above) / pivots[node]
            norm = mpmath.sqrt(sum((value**2 for value in solved), mpf(0)))
            vector = [value / norm for value in solved]
        return vector  # ty: ignore[unsound-return] - mpmath is untyped


def diode(
    levels: Real, resistance: float, saturation: float, thermal: _mpf, gmin: float
) -> Real:
    """Computes exact diode voltages across `levels` through `resistance` with `gmin`.

    Solves ``(Vs - V)/R = Is (e^{V/Vt} - 1) + gmin V`` in mpmath at `DIGITS`
    digits. Setting ``G = 1/R + gmin`` and ``a = (Vs/R + Is)/G`` yields
    ``V = a - Vt W(Is/(G Vt) e^{a/Vt})`` on the principal branch of Lambert's W.
    """
    r, i, vt, g = (mpf(value) for value in (resistance, saturation, thermal, gmin))
    conductance = 1 / r + g
    found = []
    for level in levels:
        a = (mpf(float(level)) / r + i) / conductance
        w = mpmath.lambertw(i / (conductance * vt) * mpmath.exp(a / vt))
        found.append(float(a - vt * mpmath.re(w)))
    return np.array(found)


def closed(form: Callable[[_mpf], object], abscissa: Real) -> Complex:
    """Evaluates closed-form callable `form` at `abscissa` in mpmath at `DIGITS` digits.

    Accepts real or complex mpmath return values and converts them to complex doubles.
    """
    return np.array([complex(mpmath.mpc(form(mpf(float(x))))) for x in abscissa])
