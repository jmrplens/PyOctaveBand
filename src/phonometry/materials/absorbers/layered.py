#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Declarative layer stacks and the transfer-matrix absorber solver.

An absorber is declared as a list of layers ordered from the sound-incidence
side towards the termination and solved at one angle, in the same
:math:`e^{+j \omega t}` time convention as the element models of
:mod:`~phonometry.materials.absorbers.porous`, with the forward wave carried
by :math:`e^{-j k x}` (so a passive medium has
:math:`\operatorname{Im}(k) < 0`):

* **Transfer-matrix multilayer prediction**: each fluid layer contributes
  :math:`[[\cos(k_x d), jZ_x\sin(k_x d)], [j\sin(k_x d)/Z_x, \cos(k_x d)]]`
  with the in-depth wavenumber
  :math:`k_x = \sqrt{k^2 - k_0^2 \sin^2 \theta}` from Snell's law and
  :math:`Z_x = Z_\mathrm{c} k / k_x` (Cox & D'Antonio Eqs. (2.29)-(2.32); Bies
  Eq. (D.83); equivalent to the layer-recursion of Bies Eq. (D.95) and
  Mechel Sect. D.4). Thin resonant sheets (perforated plate, microperforated
  plate, limp membrane) enter as series transfer impedances
  :math:`[[1, z], [0, 1]]`. The stack is closed by a rigid wall, by free air
  or by an arbitrary termination impedance, giving the surface impedance,
  the oblique reflection factor and :math:`\alpha(\theta)`. This same
  layer transfer matrix underlies the critically-coupled perfect-absorber
  designs of Jiménez,
  Groby, Pagneux & Romero-García (2017, *Applied Sciences* 7(6), 618,
  doi:10.3390/app7060618) and, for a rigidly-backed high-porosity layer,
  Jiménez, Romero-García & Groby (2018, *Acta Acustica united with Acustica*
  104(3), 396-409, doi:10.3813/AAA.919183), where the critical-coupling
  condition on the surface impedance yields total single-frequency absorption.

* **Random incidence**: the random-incidence (Paris) integral follows Mechel
  Sect. D.5 Eqs. (9)-(10), with the closed form for locally reacting surfaces
  implemented in :func:`statistical_absorption` (its maximum over passive
  impedances is the published 0.951).

The elements a stack is built from live elsewhere: the equivalent fluid a
:class:`PorousLayer` carries and the sheet impedances the plate and membrane
layers evaluate come from
:mod:`~phonometry.materials.absorbers.porous`, and the three Biot waves a
:class:`PoroelasticLayer` carries come from
:mod:`~phonometry.materials.absorbers.biot`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ..._internal.validation import (
    require_non_negative,
    require_positive,
    require_positive_array,
)
from .porous import (
    _AIR_DENSITY,
    _AIR_VISCOSITY,
    _SPEED_OF_SOUND,
    Complex,
    PorousMediumResult,
    membrane_impedance,
    microperforated_plate_impedance,
    perforated_plate_impedance,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

    from ..._internal.types import Real

#: Discriminator tag of the term tuples built by :func:`_layer_terms`: a
#: fluid term carries ``(Zx, kx d)`` and takes the fluid-layer branch of the
#: admittance recursion and of the chain matrix; any other tag is a series
#: sheet transfer impedance.
_FLUID = "fluid"

#: Minimum accepted Gauss-Legendre quadrature order for the Paris-integral
#: evaluation of :func:`diffuse_field_absorption`.
_MIN_QUADRATURE_POINTS = 2

#: ``|Im(1/z)| = |g2|`` below which :func:`statistical_absorption` replaces
#: Mechel's arctan term, which cancels catastrophically as ``g2 -> 0``, with
#: its exact ``g2 -> 0`` series limit: the limit's ``O(g2^2)`` truncation
#: error is far below double precision at this threshold, while the direct
#: form is stable for every larger ``|g2|`` (see the comment at the use site).
_NEAR_REAL_EPS = 1e-30

__all__ = [
    "AirLayer",
    "DiffuseFieldAbsorptionResult",
    "LayeredAbsorberResult",
    "MembraneLayer",
    "MicroperforatedPlateLayer",
    "PerforatedPlateLayer",
    "PoroelasticLayer",
    "PorousLayer",
    "diffuse_field_absorption",
    "layered_absorber",
    "statistical_absorption",
]


class _DrawableLayer:
    """Shared geometry drawing for the layer dataclasses.

    ``plot()`` draws the layer as a one-layer stack cross-section, to scale,
    against the rigid backing; a full stack is drawn by
    :func:`~phonometry.materials.plot_absorber_stack` or by
    :meth:`LayeredAbsorberResult.plot_geometry`.
    """

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw this layer's cross-section to scale (dimensioned).

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.geometry import plot_absorber_stack

        check_language(language)
        return plot_absorber_stack(
            [cast("Layer", self)], ax=ax, language=language, **kwargs
        )


@dataclass(frozen=True)
class AirLayer(_DrawableLayer):
    """A plain air gap of ``thickness`` metres inside the stack."""

    thickness: float

    def __post_init__(self) -> None:
        """Reject a gap whose thickness is not a finite, non-negative number.

        Non-negative rather than positive because the solver treats a zero
        gap as no gap; NaN fails every comparison downstream, so an unpinned
        layer would draw a blank stack rather than refuse.

        :raises ValueError: if ``thickness`` is negative or not finite.
        """
        require_non_negative(self.thickness, "thickness")


@dataclass(frozen=True)
class PorousLayer(_DrawableLayer):
    """A porous layer of ``thickness`` metres described by *medium*.

    ``medium`` is a :class:`PorousMediumResult` (from :func:`delany_bazley`,
    :func:`miki`, :func:`johnson_champoux_allard`, or built directly from
    measured ``Zc``/``k`` data) evaluated on the same frequency vector that
    is passed to :func:`layered_absorber`.
    """

    thickness: float
    medium: PorousMediumResult


@dataclass(frozen=True)
class PerforatedPlateLayer(_DrawableLayer):
    """A rigid perforated plate (see :func:`perforated_plate_impedance`)."""

    thickness: float
    hole_radius: float
    open_area: float
    end_correction: float | None = None


@dataclass(frozen=True)
class MicroperforatedPlateLayer(_DrawableLayer):
    """A microperforated plate (see :func:`microperforated_plate_impedance`)."""

    thickness: float
    hole_radius: float
    open_area: float
    end_correction: float = 0.85


@dataclass(frozen=True)
class MembraneLayer(_DrawableLayer):
    """A limp impervious membrane (see :func:`membrane_impedance`)."""

    surface_density: float
    resistance: float = 0.0


@dataclass(frozen=True)
class PoroelasticLayer(_DrawableLayer):
    """A porous layer whose frame is elastic (full Biot theory).

    Where :class:`PorousLayer` collapses the material into a single wave in an
    equivalent fluid, this layer carries the three Biot waves of Allard &
    Atalla 2e chapter 6 - two compressional and one shear - so the frame can
    resonate. It is the only layer type that reproduces the quarter-wavelength
    frame resonance of :func:`~phonometry.materials.absorbers.biot.frame_quarter_wave_resonance`,
    and the only one for which an air gap behind the layer, a bonded backing or
    an oblique angle change the frame motion rather than only the pore fluid.

    ``medium`` is the **rigid-frame** equivalent fluid of the pores (normally a
    :func:`johnson_champoux_allard` result on the solver's frequency vector):
    the frame inertia is added by the Biot model itself, so a limp-corrected
    medium would count it twice. The remaining fields describe the frame.

    Adding one of these to a stack switches :func:`layered_absorber` to the
    global-matrix assembly of Allard & Atalla Sect. 11.5. Two adjacent
    poroelastic layers are coupled as *bonded* frames (their Eq. (11.67)); a
    sheet layer next to a poroelastic layer is coupled as a free, mechanically
    decoupled screen (air on both sides, their Sect. 11.3.6).
    """

    thickness: float
    medium: PorousMediumResult
    porosity: float
    tortuosity: float
    frame_density: float
    shear_modulus: complex
    poisson_ratio: float = 0.0


Layer = (
    AirLayer
    | PorousLayer
    | PerforatedPlateLayer
    | MicroperforatedPlateLayer
    | MembraneLayer
    | PoroelasticLayer
)


@dataclass(frozen=True)
class LayeredAbsorberResult:
    r"""Oblique-incidence prediction of a layered absorber.

    All arrays share the shape of ``frequency``. ``surface_impedance`` is the
    specific impedance :math:`Z_\mathrm{s} = p / u_n` at the front face (may be
    ``inf`` for a lossless-sheet stack over a rigid wall), ``reflection``
    the complex plane-wave reflection factor :math:`R(\theta)`,
    ``absorption`` the coefficient
    :math:`\alpha(\theta) = 1 - \lvert R \rvert^2` and ``transfer_matrix``
    the total chain matrix with shape ``(2, 2, len(frequency))``
    (unimodular: every layer is reciprocal).

    ``layers`` retains the layer sequence the stack was solved with (front
    layer first) so :meth:`plot_geometry` can draw the cross-section; it is
    appended after the original fields and defaults to ``None`` for
    hand-built results.
    """

    frequency: Real
    angle: float
    surface_impedance: Complex
    normalized_impedance: Complex
    reflection: Complex
    absorption: Real
    transfer_matrix: Complex
    layers: tuple[Layer, ...] | None = None

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        r"""Plot the absorption spectrum :math:`\alpha(f)` with
        :math:`\lvert R \rvert` overlaid.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.materials import plot_layered_absorber

        check_language(language)
        return plot_layered_absorber(self, ax=ax, language=language, **kwargs)

    def plot_geometry(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw the solved stack cross-section to scale (dimensioned).

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :raises ValueError: If the result does not retain its ``layers``.
        """
        from ..._i18n import check_language
        from ..._plot.geometry import plot_layered_absorber_geometry

        check_language(language)
        return plot_layered_absorber_geometry(self, ax=ax, language=language, **kwargs)


@dataclass(frozen=True)
class DiffuseFieldAbsorptionResult:
    r"""Random-incidence (Paris-integral) absorption of a layered absorber.

    ``absorption`` is :math:`\alpha_{\mathrm{dif}}(f)` from Mechel 2e
    Sect. D.5 Eq. (9): the plane-wave :math:`\alpha(\theta)` weighted by
    :math:`\cos(\theta) \sin(\theta)` and normalised by
    :math:`\sin^2(\theta_{\mathrm{limit}})`.
    """

    frequency: Real
    absorption: Real
    angle_limit: float

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        r"""Plot the random-incidence absorption spectrum
        :math:`\alpha_{\mathrm{dif}}(f)`.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.materials import plot_diffuse_field_absorption

        check_language(language)
        return plot_diffuse_field_absorption(self, ax=ax, language=language, **kwargs)


def _sheet_impedance(
    layer: Layer,
    f: Real,
    *,
    air_density: float,
    viscosity: float,
) -> Complex:
    """Series transfer impedance of a sheet layer on the grid *f*."""
    if isinstance(layer, PerforatedPlateLayer):
        return perforated_plate_impedance(
            f,
            thickness=layer.thickness,
            hole_radius=layer.hole_radius,
            open_area=layer.open_area,
            end_correction=layer.end_correction,
            air_density=air_density,
            viscosity=viscosity,
        )
    if isinstance(layer, MicroperforatedPlateLayer):
        return microperforated_plate_impedance(
            f,
            thickness=layer.thickness,
            hole_radius=layer.hole_radius,
            open_area=layer.open_area,
            end_correction=layer.end_correction,
            air_density=air_density,
            viscosity=viscosity,
        )
    if isinstance(layer, MembraneLayer):
        return membrane_impedance(
            f,
            surface_density=layer.surface_density,
            resistance=layer.resistance,
        )
    msg = f"not a sheet layer: {layer!r}"
    raise TypeError(msg)  # pragma: no cover


def _fluid_layer_terms(
    zc: Complex, k: Complex, thickness: float, k0_sin2: Real
) -> tuple[Complex, Complex]:
    r"""In-depth impedance ``Zx`` and phase ``kx d`` of an oblique layer.

    :math:`k_x = \sqrt{k^2 - k_0^2 \sin^2 \theta}` (Snell's law, Cox &
    D'Antonio 3e Eq. (2.30)) and the in-depth wave impedance
    :math:`Z_x = z_\mathrm{c} k / k_x`; the layer chain matrix (Eq. (2.29)) is built
    from ``cos``/``sin`` of :math:`k_x d` with these two terms.
    """
    kx = np.sqrt(k * k - k0_sin2)
    # Passive decay: keep the branch with non-positive imaginary part.
    kx = np.where(kx.imag > 0.0, -kx, kx)
    zx = zc * k / kx
    return (
        np.asarray(zx, dtype=np.complex128),
        np.asarray(kx * thickness, dtype=np.complex128),
    )


def _porous_layer_term(
    layer: PorousLayer, f: Real, k0_sin2: Real
) -> tuple[Complex, Complex] | None:
    """``(Zx, kx d)`` of a porous layer, or ``None`` when zero-thickness."""
    d = require_non_negative(layer.thickness, "PorousLayer.thickness")
    if d <= 0.0:
        return None
    medium = layer.medium
    _check_medium_grid(medium, f, "PorousLayer")
    return _fluid_layer_terms(
        np.asarray(medium.characteristic_impedance, dtype=np.complex128),
        np.asarray(medium.wavenumber, dtype=np.complex128),
        d,
        k0_sin2,
    )


def _layer_terms(
    layers: list[Layer] | tuple[Layer, ...],
    f: Real,
    *,
    k0: Real,
    k0_sin2: Real,
    rc: float,
    rho0: float,
    viscosity: float,
) -> list[tuple[str, Complex, Complex]]:
    """Evaluate each layer once: fluid layers as ``(Zx, kx d)``, sheets as z.

    Zero-thickness fluid layers contribute the identity matrix and are
    skipped (``require_non_negative`` guarantees ``d >= 0``, so the strict
    ``d > 0`` test keeps exactly the non-degenerate layers).
    """
    terms: list[tuple[str, Complex, Complex]] = []
    for layer in layers:
        if isinstance(layer, AirLayer):
            d = require_non_negative(layer.thickness, "AirLayer.thickness")
            if d > 0.0:
                zc = np.full(f.shape, rc, dtype=np.complex128)
                k = np.asarray(k0, dtype=np.complex128)
                terms.append((_FLUID, *_fluid_layer_terms(zc, k, d, k0_sin2)))
        elif isinstance(layer, PorousLayer):
            term = _porous_layer_term(layer, f, k0_sin2)
            if term is not None:
                terms.append((_FLUID, *term))
        else:
            z = _sheet_impedance(
                layer,
                f,
                air_density=rho0,
                viscosity=viscosity,
            )
            terms.append(("sheet", z, z))
    return terms


def _termination_admittance(
    termination: str | complex | ArrayLike,
    f: Real,
    *,
    cos_t: float,
    rc: float,
) -> Complex:
    r"""Admittance :math:`G = u/p` at the termination face of the stack.

    ``"free"`` is evaluated as the literal :math:`\cos(\theta) / \rho c` the
    recursion has always used, not as the reciprocal of the impedance the
    global-matrix assembly needs: :math:`1 / (\rho c / \cos \theta)` is a
    different double for about a quarter of the angles in
    :math:`[0, \pi/2)`, and the equivalent-fluid path must stay
    bit-identical.
    """
    if isinstance(termination, str) and termination == "free":
        return np.full(f.shape, cos_t / rc, dtype=np.complex128)
    zl_arr = _termination_impedance(termination, f, cos_t=cos_t, rc=rc)
    if zl_arr is None:
        return np.zeros_like(f, dtype=np.complex128)
    return np.asarray(np.ones_like(f) / zl_arr, dtype=np.complex128)


def _termination_impedance(
    termination: str | complex | ArrayLike,
    f: Real,
    *,
    cos_t: float,
    rc: float,
) -> Complex | None:
    r"""Impedance :math:`p/v_3` closing the stack, or ``None`` for a hard wall.

    The global-matrix assembly of Allard & Atalla Sect. 11.5 needs the
    termination as an impedance (their Eq. (11.84)); the admittance the
    recursion of :func:`_surface_admittance` consumes is its reciprocal, with
    the hard wall the one case that has no finite impedance.
    """
    if isinstance(termination, str):
        if termination == "rigid":
            return None
        if termination == "free":
            return np.full(f.shape, rc / cos_t, dtype=np.complex128)
        msg = "'termination' must be 'rigid', 'free' or a complex impedance."
        raise ValueError(msg)
    zl_arr = np.asarray(termination, dtype=np.complex128)
    if zl_arr.ndim > 0 and zl_arr.shape != f.shape:
        msg = (
            "'termination' impedance array must be scalar or match the shape "
            f"of 'frequency' {f.shape}; got shape {zl_arr.shape}."
        )
        raise ValueError(msg)
    if not np.all(np.abs(zl_arr) > 0.0):
        msg = "'termination' impedance must be non-zero."
        raise ValueError(msg)
    return np.asarray(np.broadcast_to(zl_arr, f.shape), dtype=np.complex128)


def _surface_admittance(
    terms: list[tuple[str, Complex, Complex]], g: Complex
) -> Complex:
    """Back-to-front admittance recursion from the termination admittance.

    Stable: ``tan`` saturates where the chain-matrix entries would overflow.
    """
    for kind, a, b in reversed(terms):
        if kind == _FLUID:
            zx, kxd = a, b
            t = np.tan(kxd)
            g = (g + 1j * t / zx) / (1.0 + 1j * zx * t * g)
        else:
            g = g / (1.0 + a * g)
    return g


def _chain_matrix(terms: list[tuple[str, Complex, Complex]], f: Real) -> Complex:
    """Raw front-to-back chain-matrix product of the evaluated layers.

    Informational; may overflow for extremely attenuating layers while the
    admittance recursion stays finite.
    """
    ones = np.ones_like(f, dtype=np.complex128)
    zeros = np.zeros_like(f, dtype=np.complex128)
    t11, t12, t21, t22 = ones, zeros, zeros, ones
    with np.errstate(over="ignore", invalid="ignore"):
        for kind, a, b in terms:
            if kind == _FLUID:
                zx, kxd = a, b
                cos_l, sin_l = np.cos(kxd), np.sin(kxd)
                m = (cos_l, 1j * zx * sin_l, 1j * sin_l / zx, cos_l)
            else:
                m = (ones, a, zeros, ones)
            m11, m12, m21, m22 = m
            t11, t12, t21, t22 = (
                t11 * m11 + t12 * m21,
                t11 * m12 + t12 * m22,
                t21 * m11 + t22 * m21,
                t21 * m12 + t22 * m22,
            )
    return np.asarray([[t11, t12], [t21, t22]], dtype=np.complex128)


def _check_medium_grid(medium: PorousMediumResult, f: Real, owner: str) -> None:
    """Reject a medium evaluated on a different frequency vector."""
    if not np.array_equal(np.asarray(medium.frequency), f):
        msg = (
            f"{owner}.medium was evaluated on a different frequency "
            "vector; rebuild the medium on the solver grid."
        )
        raise ValueError(msg)


def _split_fluid_run(
    terms: list[tuple[str, Complex, Complex]], budget: float, limit: int
) -> list[list[tuple[str, Complex, Complex]]]:
    r"""Group a fluid run into chain blocks of at most *budget* nepers.

    A fluid run that attenuates by ``b`` nepers has chain-matrix entries of
    order :math:`e^b` while the same block's back face is the identity, so
    the assembled system of Allard & Atalla Sect. 11.6 holds rows differing
    by :math:`e^b` and the elimination of the block loses about
    :math:`b / \ln(10)` digits; past :math:`b \sim 710` the entries overflow
    float64 outright. The split is algebraically exact, because a
    homogeneous fluid layer of phase :math:`k_x d` is the product of ``m``
    layers of phase :math:`k_x d / m`.

    Returns the run unchanged, as a single group, whenever it stays inside
    the budget, so ordinary stacks keep the exact chain product they had.
    Sheet layers carry no attenuation and never force a split of their own.

    :raises ValueError: when the run would need more than *limit* blocks.
    """
    losses = [
        float(np.max(np.abs(np.imag(b)))) if kind == _FLUID else 0.0
        for kind, _, b in terms
    ]
    attenuation = sum(losses)
    # Checked before the run is expanded, so an absurd input cannot build the
    # sub-layer list it would then be refused for.
    if attenuation > budget * limit:
        msg = (
            f"the fluid layers of the stack attenuate by {attenuation:.0f} "
            f"nepers, which the global-matrix assembly cannot resolve in "
            f"{limit} blocks. Reduce their thickness: nothing behind such a "
            "run contributes to the surface impedance."
        )
        raise ValueError(msg)

    parts: list[tuple[str, Complex, Complex, float]] = []
    for (kind, a, b), loss in zip(terms, losses, strict=True):
        pieces = max(1, int(np.ceil(loss / budget)))
        if pieces == 1:
            parts.append((kind, a, b, loss))
        else:
            parts.extend([(kind, a, b / pieces, loss / pieces)] * pieces)

    groups: list[list[tuple[str, Complex, Complex]]] = []
    current: list[tuple[str, Complex, Complex]] = []
    total = 0.0
    for kind, a, b, loss in parts:
        if current and total + loss > budget:
            groups.append(current)
            current, total = [], 0.0
        current.append((kind, a, b))
        total += loss
    if current:
        groups.append(current)
    # The pre-check bounds the total loss, not the packing. Next-fit needs up
    # to twice the optimal number of bins (alternating items just over half a
    # budget open a block each), so a run that clears the sum can still be cut
    # into more blocks than the assembly resolves. Refuse on what was actually
    # produced rather than on what was estimated.
    if len(groups) > limit:
        msg = (
            f"the fluid layers of the stack pack into {len(groups)} chain "
            f"blocks of at most {budget:.0f} nepers, more than the {limit} "
            "the global-matrix assembly can resolve. Reduce their thickness: "
            "nothing behind such a run contributes to the surface impedance."
        )
        raise ValueError(msg)
    return groups


def _stack_blocks(
    layers: list[Layer] | tuple[Layer, ...],
    f: Real,
    *,
    k0: Real,
    k0_sin2: Real,
    rc: float,
    rho0: float,
    viscosity: float,
    transverse_wavenumber: Real,
) -> list[Any]:
    """Split a stack into fluid blocks and poroelastic blocks.

    Consecutive fluid and sheet layers collapse into two-variable blocks
    carrying their chain-matrix product; each :class:`PoroelasticLayer` becomes
    a six-variable block of Allard & Atalla Sect. 11.3.3. A run or a layer
    that attenuates by more than ``biot._BLOCK_NEPERS`` is cut into several
    blocks first, which the global matrix handles exactly as it handles
    adjacent fluid layers and bonded halves of one poroelastic material.
    """
    from . import biot

    blocks: list[Any] = []
    pending: list[Layer] = []

    def flush() -> None:
        if not pending:
            return
        terms = _layer_terms(
            pending,
            f,
            k0=k0,
            k0_sin2=k0_sin2,
            rc=rc,
            rho0=rho0,
            viscosity=viscosity,
        )
        groups = _split_fluid_run(terms, biot._BLOCK_NEPERS, biot._MAX_BLOCKS)
        for group in groups:
            chain = _chain_matrix(group, f)
            blocks.append(biot._fluid_block(np.moveaxis(chain, -1, 0)))
        pending.clear()

    for layer in layers:
        if isinstance(layer, PoroelasticLayer):
            thickness = require_non_negative(
                layer.thickness, "PoroelasticLayer.thickness"
            )
            if thickness <= 0.0:
                continue
            _check_medium_grid(layer.medium, f, "PoroelasticLayer")
            flush()
            waves = biot.biot_waves(
                layer.medium,
                porosity=layer.porosity,
                tortuosity=layer.tortuosity,
                frame_density=layer.frame_density,
                shear_modulus=layer.shear_modulus,
                poisson_ratio=layer.poisson_ratio,
            )
            blocks.extend(
                biot._poroelastic_blocks(waves, thickness, transverse_wavenumber)
            )
        else:
            pending.append(layer)
    flush()
    return blocks


def layered_absorber(
    frequency: ArrayLike,
    layers: list[Layer] | tuple[Layer, ...],
    *,
    angle: float = 0.0,
    termination: str | complex | ArrayLike = "rigid",
    speed_of_sound: float = _SPEED_OF_SOUND,
    air_density: float = _AIR_DENSITY,
    viscosity: float = _AIR_VISCOSITY,
) -> LayeredAbsorberResult:
    r"""Transfer-matrix prediction of a layered absorber at one angle.

    The *layers* list is ordered from the sound-incidence side towards the
    *termination*. Fluid layers (:class:`AirLayer`, :class:`PorousLayer`)
    contribute the oblique chain matrix of Cox & D'Antonio 3e Eq. (2.29)
    (equivalently the impedance recursion of Bies 5e Eq. (D.95) and the
    scheme of Mechel 2e Sect. D.4); sheet layers (:class:`PerforatedPlateLayer`,
    :class:`MicroperforatedPlateLayer`, :class:`MembraneLayer`) enter as
    locally reacting series impedances; a :class:`PoroelasticLayer` carries the
    three Biot waves of its elastic frame and switches the whole stack to the
    six-variable global-matrix assembly of Allard & Atalla 2e Sect. 11.5, with
    the coupling matrices of Sect. 11.4. The chain is closed by a rigid wall
    (``termination="rigid"``), by radiation into free air behind
    (``termination="free"``, :math:`Z_L = \rho c / \cos(\theta)`) or by an
    arbitrary complex impedance. The reflection factor is
    :math:`R = (Z_\mathrm{s} \cos(\theta) - \rho c) / (Z_\mathrm{s} \cos(\theta) + \rho c)`
    and :math:`\alpha = 1 - \lvert R \rvert^2` (Mechel 2e Sect. D.3
    Eq. (2)).

    ``Zs``, ``R`` and ``alpha`` are evaluated with the numerically robust
    admittance recursion (algebraically identical to the chain product but
    immune to the :math:`e^{\lvert \operatorname{Im}(k_x) \rvert d}`
    overflow of the raw matrix entries for
    extremely attenuating layers); the raw chain matrix is still returned in
    ``transfer_matrix`` and may overflow in such extreme cases.

    :param frequency: Frequency vector ``f``, in hertz.
    :param layers: Layer stack from the incidence side to the termination.
    :param angle: Polar angle of incidence ``theta``, in radians
        (:math:`0 \le \theta < \pi/2 - 10^{-6}`; grazing incidence is
        excluded).
    :param termination: ``"rigid"`` (default), ``"free"``, or a non-zero
        complex impedance (scalar or per-frequency array), in Pa s/m.
    :param speed_of_sound: Speed of sound ``c`` in air, in m/s.
    :param air_density: Air density ``rho``, in kg/m3.
    :param viscosity: Dynamic viscosity of air, in Pa s (sheet layers).
    :return: A :class:`LayeredAbsorberResult`.
    """
    f = require_positive_array(frequency, "frequency")
    if not layers:
        msg = "'layers' must contain at least one layer."
        raise ValueError(msg)
    theta = float(angle)
    # The last ~3e-8 rad below pi/2 round sin(theta)**2 to 1.0, driving the
    # in-depth wavenumber of an air layer to exactly zero (inf * 0 = nan in
    # the recursion); reject effectively grazing input with a clear error.
    if not 0.0 <= theta < np.pi / 2.0 - 1e-6:
        msg = "'angle' must satisfy 0 <= angle < pi/2 - 1e-6."
        raise ValueError(msg)
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    rho0 = require_positive(air_density, "air_density")
    require_positive(viscosity, "viscosity")

    k0 = 2.0 * np.pi * f / c0
    k0_sin2 = np.asarray((k0 * np.sin(theta)) ** 2, dtype=np.float64)
    cos_t = float(np.cos(theta))
    rc = rho0 * c0

    if any(isinstance(layer, PoroelasticLayer) for layer in layers):
        from . import biot

        blocks = _stack_blocks(
            layers,
            f,
            k0=k0,
            k0_sin2=k0_sin2,
            rc=rc,
            rho0=rho0,
            viscosity=viscosity,
            transverse_wavenumber=np.asarray(k0 * np.sin(theta)),
        )
        if not blocks:
            msg = "'layers' must contain at least one layer."
            raise ValueError(msg)
        zs = biot._stack_surface_impedance(
            blocks, _termination_impedance(termination, f, cos_t=cos_t, rc=rc)
        )
        finite = np.isfinite(zs)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(finite, 1.0 / np.where(finite, zs, 1.0), 0.0 + 0j)
        tm = np.full((2, 2, f.size), np.nan + 0j, dtype=np.complex128)
    else:
        terms = _layer_terms(
            layers,
            f,
            k0=k0,
            k0_sin2=k0_sin2,
            rc=rc,
            rho0=rho0,
            viscosity=viscosity,
        )
        g = _surface_admittance(
            terms, _termination_admittance(termination, f, cos_t=cos_t, rc=rc)
        )
        # G = 0 (lossless stack over a rigid wall) maps to an infinite surface
        # impedance; everywhere else Zs = 1/G with a safe denominator.
        nonzero = np.abs(g) > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            zs = np.where(nonzero, 1.0 / np.where(nonzero, g, 1.0), np.inf + 0j)
        tm = _chain_matrix(terms, f)

    r = (cos_t - rc * g) / (cos_t + rc * g)
    alpha = 1.0 - np.abs(r) ** 2
    return LayeredAbsorberResult(
        frequency=f,
        angle=theta,
        surface_impedance=np.asarray(zs, dtype=np.complex128),
        normalized_impedance=np.asarray(zs / rc, dtype=np.complex128),
        reflection=np.asarray(r, dtype=np.complex128),
        absorption=np.asarray(alpha, dtype=np.float64),
        transfer_matrix=tm,
        layers=tuple(layers),
    )


def diffuse_field_absorption(
    frequency: ArrayLike,
    layers: list[Layer] | tuple[Layer, ...],
    *,
    angle_limit: float = np.pi / 2.0,
    quadrature_points: int = 64,
    termination: str | complex | ArrayLike = "rigid",
    speed_of_sound: float = _SPEED_OF_SOUND,
    air_density: float = _AIR_DENSITY,
    viscosity: float = _AIR_VISCOSITY,
) -> DiffuseFieldAbsorptionResult:
    r"""Random-incidence absorption by the Paris integral (Mechel Sect. D.5).

    .. math::

       \alpha_{\mathrm{dif}} = \frac{2}{\sin^2 \theta_{\mathrm{lim}}}
       \int_0^{\theta_{\mathrm{lim}}} \alpha(\theta) \cos(\theta)
       \sin(\theta) \, d\theta

    (Mechel 2e Sect. D.5 Eq. (9)), evaluated
    with fixed-order Gauss-Legendre quadrature over the bulk-reacting
    :math:`\alpha(\theta)` of :func:`layered_absorber` (Sect. D.6 notes the
    bulk integral generally must be evaluated numerically). Some references
    truncate the integral at 75-87 degrees instead of 90 (Sect. D.5); set
    ``angle_limit`` accordingly.

    :param frequency: Frequency vector ``f``, in hertz.
    :param layers: Layer stack, as in :func:`layered_absorber`.
    :param angle_limit: Upper integration angle ``theta_lim``, in radians
        (0 < theta_lim <= pi/2; default pi/2).
    :param quadrature_points: Gauss-Legendre order (default 64).
    :param termination: As in :func:`layered_absorber`.
    :param speed_of_sound: Speed of sound ``c`` in air, in m/s.
    :param air_density: Air density ``rho``, in kg/m3.
    :param viscosity: Dynamic viscosity of air, in Pa s.
    :return: A :class:`DiffuseFieldAbsorptionResult`.
    """
    f = require_positive_array(frequency, "frequency")
    lim = float(angle_limit)
    if not 0.0 < lim <= np.pi / 2.0:
        msg = "'angle_limit' must satisfy 0 < angle_limit <= pi/2."
        raise ValueError(msg)
    n = int(quadrature_points)
    if n < _MIN_QUADRATURE_POINTS:
        msg = "'quadrature_points' must be at least 2."
        raise ValueError(msg)
    nodes, weights = np.polynomial.legendre.leggauss(n)
    theta = 0.5 * lim * (nodes + 1.0)
    w = 0.5 * lim * weights
    total = np.zeros_like(f, dtype=np.float64)
    for th, wt in zip(theta, w, strict=True):
        res = layered_absorber(
            f,
            layers,
            angle=float(th),
            termination=termination,
            speed_of_sound=speed_of_sound,
            air_density=air_density,
            viscosity=viscosity,
        )
        total += wt * res.absorption * np.cos(th) * np.sin(th)
    alpha_dif = 2.0 * total / np.sin(lim) ** 2
    return DiffuseFieldAbsorptionResult(
        frequency=f,
        absorption=np.asarray(alpha_dif, dtype=np.float64),
        angle_limit=lim,
    )


def statistical_absorption(
    normalized_impedance: ArrayLike,
    *,
    angle_limit: float = np.pi / 2.0,
) -> Real:
    r"""Closed-form Paris integral for a locally reacting plane.

    With the normalised surface admittance :math:`Z_0 G = g_1 + j g_2 = 1/z`
    (Mechel 2e Sect. D.5 Eq. (10)):

    .. math::

       \alpha_{\mathrm{dif}} = \frac{8 g_1}{\sin^2 T} \left[1 - \cos T
       + \frac{g_1^2 - g_2^2}{g_2}
       \left(\arctan\frac{1 + g_1}{g_2}
       - \arctan\frac{g_1 + \cos T}{g_2}\right)
       + g_1 \ln\frac{g_1^2 + g_2^2 + 2 g_1 \cos T + \cos^2 T}
       {1 + g_1^2 + g_2^2 + 2 g_1}\right]

    reducing for :math:`T = \pi/2` to Eq. (4) and, for real admittance, to
    the printed :math:`g_2 = 0` special case. The maximum over passive
    impedances is 0.951 (the published bound for locally reacting absorbers,
    Sect. D.5).

    :param normalized_impedance: Normalised surface impedance
        :math:`z = Z_\mathrm{s} / (\rho c)` (complex scalar or array), with
        :math:`\operatorname{Re}(z) > 0`.
    :param angle_limit: Upper integration angle ``theta_lim``, in radians
        (0 < theta_lim <= pi/2; default pi/2).
    :return: Statistical absorption coefficient ``alpha_dif``.
    """
    z = np.asarray(normalized_impedance, dtype=np.complex128)
    if np.any(z.real <= 0.0):
        msg = "'normalized_impedance' must have a positive real part."
        raise ValueError(msg)
    lim = float(angle_limit)
    if not 0.0 < lim <= np.pi / 2.0:
        msg = "'angle_limit' must satisfy 0 < angle_limit <= pi/2."
        raise ValueError(msg)
    g = 1.0 / z
    g1 = g.real
    g2 = g.imag
    cos_t = np.cos(lim)
    sin2_t = np.sin(lim) ** 2
    log_term = np.log(
        (g1**2 + g2**2 + 2.0 * g1 * cos_t + cos_t**2) / (1.0 + g1**2 + g2**2 + 2.0 * g1)
    )
    # Mechel prints (g1^2 - g2^2)/g2 * [arctan((1+g1)/g2) -
    # arctan((g1+cosT)/g2)], which cancels catastrophically as g2 -> 0
    # (a difference of two values near +-pi/2 amplified by 1/g2). With
    # a = 1 + g1 > 0 and b = g1 + cosT > 0 the identity
    # arctan(a/g2) - arctan(b/g2) = arctan(g2 (a - b) / (g2^2 + a b))
    # (valid because (a/g2)(b/g2) > 0) evaluates the same quantity
    # stably for every non-zero g2. Expanding arctan(x/g2) about
    # g2 = 0 (arctan(x/g2) = sgn(g2) pi/2 - g2/x + O(g2^3)) gives the
    # exact limit of the whole term,
    # g1^2 (1 - cos T) / ((g1 + cos T)(1 + g1)), with an O(g2^2)
    # truncation error - far below double precision at the switch
    # threshold, while the direct form is stable for every larger |g2|.
    a = 1.0 + g1
    b = g1 + cos_t
    near_real = np.abs(g2) < _NEAR_REAL_EPS
    g2_safe = np.where(near_real, 1.0, g2)
    atan_term = np.where(
        near_real,
        g1**2 * (1.0 - cos_t) / (b * a),
        (g1**2 - g2_safe**2)
        / g2_safe
        * np.arctan(g2_safe * (a - b) / (g2_safe**2 + a * b)),
    )
    alpha = 8.0 * g1 / sin2_t * (1.0 - cos_t + atan_term + g1 * log_term)
    return np.asarray(alpha, dtype=np.float64)
