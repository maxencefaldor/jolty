# Copyright 2026 Maxence Faldor
"""Exact answers, computed independently of jolty.

They are closed forms, and linear circuits solved in higher precision.

A linear circuit under a piecewise-linear source, from the operating point of its first
level, moves its every natural mode ``k`` by ``żₖ = λₖ zₖ + u``, and reads output ``p``
as ``Σₖ Rₚₖ zₖ`` (`Modal`). Along a straight piece of the source, ``u = uⱼ + sⱼ τ`` a
time ``τ`` into it, each mode is exactly
``zₖ(τ) = zₖ(0) e^{λₖτ} + uⱼ τ φ₁(λₖτ) + sⱼ τ² φ₂(λₖτ)``, with
``φ₁(x) = (eˣ - 1)/x`` and ``φ₂(x) = (eˣ - 1 - x)/x²``, terms none of which outgrows the
mode it sums to. mpmath computes the rates, the residues and each mode's state where
each piece begins at `DIGITS` decimal digits, and rounds them to doubles; the sum is
then evaluated in double precision at the points a simulator produced, beside a bound
on its rounding, which caps the digits a score may claim.

Small circuits are solved by mpmath's general eigensolver (`linear`). The ladders and
random networks of resistors and capacitors, trees of up to thousands of nodes, are
solved by their own structure (`tree`): their symmetric pencil's eigenvalues by
Newton's iteration on its determinant from LAPACK's estimates, each checked by
Sylvester's inertia to be the one its place stands for, and its eigenvectors by
inverse iteration, both through the tree's elimination, which fills nothing.

No reference is ever a simulator's run: a closed form, or this higher-precision
solution of the circuit's own equations.
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
"""The relative Newton step at which an eigenvalue has converged: ten digits above the
rounding floor of its determinant's evaluation, which Newton's steps settle at near
10⁻⁴⁵ on the networks here, and far past the double each rate is rounded to."""

NEWTON: Final = 12
"""The most Newton iterations an eigenvalue takes from LAPACK's estimate, whose
relative error was under 10⁻⁸ on the networks here: quadratic convergence reaches
`CONVERGED` in four."""

EPSILON: Final = float(np.finfo(np.float64).eps)
"""The double's machine epsilon, 2⁻⁵², twice each rounding's relative error."""

ROUNDINGS: Final = 16
"""The roundings each term of a modal sum carries beside its sum's, at most: its
residue's, its state's or level's and its time's to doubles, ``φ``'s evaluation with
the cancellation of up to seven its direct form allows above `SERIES`, and a complex
product's four."""

SERIES: Final = 1.0
"""Below this modulus of ``λτ``, ``φ₁`` and ``φ₂`` sum their Taylor series, which
cancel nothing; above it their direct forms cancel at most a factor of seven of their
terms' magnitude, as ``φ₂`` does at ``λτ = -1``."""

TERMS: Final = 21
"""The Taylor terms summed below `SERIES`: the last, ``1/22!``, is 10⁻²¹ of the
first."""


@dataclass(frozen=True, slots=True)
class Piece:
    """A straight piece of the source, from `start`.

    It holds the source's level there, its slope, and each mode's state there.
    """

    start: float
    level: float
    slope: float
    states: Complex


@dataclass(frozen=True, slots=True)
class Modal:
    """A linear circuit's exact response to a piecewise-linear source, by its modes."""

    outputs: tuple[str, ...]
    """What each output is, such as ``v(out)``."""
    rates: Complex
    """Each mode's rate ``λₖ``, its real part negative or, undamped, zero."""
    residues: Complex
    """Each output's residue at each mode, one row per output."""
    pieces: tuple[Piece, ...]

    def at(self, times: Real) -> tuple[Real, Real]:
        """Each output at `times`, one row per output, and a bound on its rounding.

        The bound is on each value's absolute error: each term's `ROUNDINGS` and its
        sum's, one a mode, times the term's magnitude, and the rounding of ``λₖτ``,
        two of its own, carried into the term by its function's condition number:
        ``|λτ|`` for ``e^{λτ}``, ``|λτ eᵗ/(eᵗ - 1)| + 1`` for ``φ₁`` and
        ``|λτ (eᵗ - 1)/(eᵗ - 1 - λτ)| + 2`` for ``φ₂``, ``t = λτ``.
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
    """Each term's magnitude times its function's condition number in ``λτ``."""
    size = np.abs(arguments)
    grown = np.abs(np.expm1(arguments))
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.where(grown > 0, size * np.abs(decays) / grown, 1.0) + 1
        rest = np.abs(np.expm1(arguments) - arguments)
        second = np.where(rest > 0, size * grown / rest, 2.0) + 2
    return size * held + first * ramped + second * sloped


def phi(arguments: Complex) -> tuple[Complex, Complex]:
    """``φ₁`` and ``φ₂`` at each of `arguments`, by their Taylor series near zero."""
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
    """``φ₁`` and ``φ₂`` at `z` in mpmath.

    Near zero they sum their series, which keep every digit that the direct forms'
    cancellation would lose.
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
    """The response of modes of `rates`, read with `residues`, to the source `level`.

    Each mode starts at its operating point, ``-u₀/λₖ``, and reaches each piece's
    start by the exact form of `Modal` over the piece before, in mpmath at `DIGITS`
    digits.
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
    """`values`, each rounded to the complex double nearest."""
    return np.array([complex(value) for value in values])


def linear(
    outputs: Sequence[str],
    states: Sequence[Sequence[Precise]],
    inputs: Sequence[Precise],
    read: Sequence[Sequence[Precise]],
    level: Points,
) -> Modal:
    """The response of a small circuit ``ẋ = A x + b u``, read by ``y = C x``.

    `states` is ``A``, `inputs` ``b`` and `read` ``C``, row by row, in SI base units
    and in mpmath, so that they hold the circuit's values with no rounding;
    ``A`` must have distinct eigenvalues, as every circuit here does. mpmath's
    eigenvectors ``V`` give the inputs in their basis, ``w = V⁻¹ b``, and each residue
    ``(C V)ₚₖ wₖ``.
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
    """The eigenvalues of ``A``, a circuit's natural frequencies, by mpmath."""
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
    """The response of `network` to the source's `level`, at the nodes `outputs`.

    With ``v = C^{-1/2} y``, the network's equations ``C v̇ = -G v + g₀ e₀ u`` are
    ``ẏ = -A y + C^{-1/2} g₀ e₀ u`` for the symmetric ``A = C^{-1/2} G C^{-1/2}``,
    whose eigenpairs ``(μₖ, qₖ)`` are its modes, of rate ``-μₖ``, and node ``p``'s
    residue at mode ``k`` is ``qₖ[p] qₖ[0] g₀ / √(cₚ c₀)``. Every node settles at the
    source's level, so each node's residues over its modes' rates sum to ``-1``.

    Raises:
        ArithmeticError: When an eigenvalue does not converge, or is not the one its
            place stands for, or a node's residues do not sum to its gain.
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
    """The rates ``-μₖ`` of `network`'s modes, ascending in ``μ``, and the ``qₖ``.

    Raises:
        ArithmeticError: When an eigenvalue does not converge, or is not the one its
            place stands for.
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
    diagonal: list[_mpf]
    off: list[_mpf]

    def pivots(self, mu: _mpf) -> tuple[list[_mpf], list[_mpf]]:
        """The pivots ``dᵢ(μ)`` and their derivatives in ``μ``."""
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
        """Newton's step for ``det(A - μI) = Πᵢ dᵢ(μ)``: one over its log's slope."""
        pivots, slopes = self.pivots(mu)
        logarithmic = sum(
            (slope / pivot for slope, pivot in zip(slopes, pivots, strict=True)), mpf(0)
        )
        return 1 / logarithmic  # ty: ignore[unsound-return] - mpmath is untyped

    def below(self, mu: _mpf) -> int:
        """How many eigenvalues lie below `mu`: the negative pivots, by Sylvester."""
        pivots, _ = self.pivots(mu)
        return sum(1 for pivot in pivots if pivot < 0)

    def vector(self, mu: _mpf) -> list[_mpf]:
        """The unit eigenvector at the eigenvalue `mu`, by two inverse iterations."""
        pivots, _ = self.pivots(mu)
        count = len(pivots)
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
    """The voltage across a diode fed through `resistance` from each of `levels`.

    The diode carries ``Is (e^{V/Vt} - 1) + gmin V``, SPICE's junction with its
    conductance across, so ``(Vs - V)/R = Is (e^{V/Vt} - 1) + gmin V``; with
    ``G = 1/R + gmin`` and ``a = (Vs/R + Is)/G``, ``V = a - Vt W(Is/(G Vt) e^{a/Vt})``,
    Lambert's W on its principal branch, in mpmath at `DIGITS` digits.
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
    """`form`, a closed form, evaluated in mpmath at `DIGITS` digits at `abscissa`.

    The form returns an mpmath number, real or complex.
    """
    return np.array([complex(mpmath.mpc(form(mpf(float(x))))) for x in abscissa])
