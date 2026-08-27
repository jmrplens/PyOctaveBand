#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Radiation of a rigid circular piston set in an infinite baffle.

The baffled circular piston is the canonical acoustic radiator: a flat rigid
disc of radius ``a`` vibrating with a uniform normal velocity in an otherwise
rigid infinite plane. It is the model behind a loudspeaker cone in a large
cabinet, the open end of a duct, and the reference source for the radiation
efficiency of any finite vibrating surface, so its two results -- the
**radiation impedance** the air presents to the piston and the **directivity**
of the far field -- are the base of the electroacoustics domain (Beranek &
Mellow, *Acoustics: Sound Fields, Transducers and Vibration* 2nd ed., §4.4;
Bies, Hansen & Howard, *Engineering Noise Control* 5th ed.).

**Radiation impedance.** The reaction force of the air on the piston is
:math:`F = Z_\mathrm{r} u` with the mechanical radiation impedance

.. math::

   Z_\mathrm{r} = \rho c S \left( R_1(2ka) + j X_1(2ka) \right),
   \qquad S = \pi a^2

where :math:`k = \omega / c` is the wavenumber, :math:`\rho c` the
characteristic
impedance of air and ``S`` the piston area. The dimensionless **piston
resistance** and **reactance** functions are (Beranek & Mellow Eq. (4.30))

.. math::

   R_1(x) = 1 - \frac{2 J_1(x)}{x},
   \qquad X_1(x) = \frac{2 H_1(x)}{x}

with ``J1`` the Bessel function of the first kind and ``H1`` the Struve
function, both of order one, evaluated at :math:`x = 2ka`.

* **Low frequency** (:math:`ka \ll 1`): :math:`R_1 \to (ka)^2 / 2` so the
  radiated power
  rises as :math:`f^2`, and :math:`X_1 \to (8 / 3\pi) ka`. The reactance is
  mass-like,
  :math:`X_\mathrm{r} = \rho c S X_1 = \omega M_\mathrm{r}` with the **radiation (accreted)
  mass** :math:`M_\mathrm{r} = 8 \rho a^3 / 3`
  (Beranek & Mellow Eq. (4.32)): the piston drags an extra
  :math:`8 \rho a^3 / 3` of
  air, equivalent to a layer :math:`8a / 3\pi` thick over its face.
* **High frequency** (:math:`ka \gg 1`): :math:`R_1 \to 1` and
  :math:`X_1 \to 0`, so
  :math:`Z_\mathrm{r} \to \rho c S` -- the piston radiates as if into an infinite tube
  and the
  air loads it purely resistively.

**Directivity.** The far-field pressure of the baffled piston varies with the
polar angle ``theta`` from the axis as (Beranek & Mellow Eq. (4.42))

.. math::

   D(\theta) = \frac{2 J_1(ka \sin\theta)}{ka \sin\theta},
   \qquad D(0) = 1

The main lobe narrows as ``ka`` grows; its first null is at
:math:`ka \sin\theta = 3.8317` (the first zero of ``J1``), which exists only
once
:math:`ka > 3.8317`. The **directivity factor** ``Q`` (on-axis intensity over
the
intensity of a point source of equal power radiating into the full sphere) and
the **directivity index** :math:`DI = 10 \log_{10} Q` follow from integrating
:math:`|D|^2` over the radiating hemisphere,

.. math::

   Q = \frac{2}{\int_0^{\pi/2} |D(\theta)|^2 \sin\theta \, d\theta}

which tends to :math:`Q = 2` (:math:`DI = 3.01` dB, the half-space baffle
gain) at low
``ka`` and to :math:`Q \sim (ka)^2` (:math:`DI \sim 20 \log_{10} ka`) at
high ``ka``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import special

from .._internal.validation import (
    require_finite_fields,
    require_positive,
    require_ranks,
    require_same_length,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.projections.polar import PolarAxes
    from numpy.typing import ArrayLike, NDArray

#: First zero of the Bessel function ``J1``; the first directivity null of the
#: baffled piston sits at ``ka sin theta = J1_FIRST_ZERO`` (Beranek & Mellow).
J1_FIRST_ZERO = 3.8317059702075125

#: Reference characteristic impedance factors of air at 20 degC, 101.325 kPa.
_RHO_AIR = 1.206
_C_AIR = 343.0


def piston_resistance(x: ArrayLike) -> np.ndarray | float:
    r"""Piston resistance function :math:`R_1(x) = 1 - 2 J_1(x) / x`.

    The real part of the normalized radiation impedance of a baffled circular
    piston, as a function of :math:`x = 2ka` (Beranek & Mellow Eq. (4.30)).
    It rises as :math:`x^2 / 8 = (ka)^2 / 2` at low ``x`` and tends to 1 at
    high ``x``.

    :param x: Argument :math:`x = 2ka` (scalar or array), dimensionless.
    :return: ``R1(x)`` (float for scalar input, else an array).
    """
    arr = np.asarray(x, dtype=np.float64)
    if np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
        msg = "'x' must be non-negative and finite."
        raise ValueError(msg)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 1.0 - 2.0 * special.j1(arr) / arr
    # J1(x)/x -> 1/2 as x -> 0, so R1 -> 0; the input is validated finite, so
    # the 0/0 at x = 0 is the only non-finite point and is set to that limit.
    out = np.where(np.isfinite(out), out, 0.0)
    return out[()] if out.ndim == 0 else out


def piston_reactance(x: ArrayLike) -> np.ndarray | float:
    r"""Piston reactance function :math:`X_1(x) = 2 H_1(x) / x` (``H1`` Struve
    order 1).

    The imaginary part of the normalized radiation impedance of a baffled
    circular piston (Beranek & Mellow Eq. (4.30)). It rises as
    :math:`(8 / 3\pi) ka` (mass-like) at low :math:`x = 2ka` and decays to 0
    at high ``x``.

    :param x: Argument :math:`x = 2ka` (scalar or array), dimensionless.
    :return: ``X1(x)`` (float for scalar input, else an array).
    """
    arr = np.asarray(x, dtype=np.float64)
    if np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
        msg = "'x' must be non-negative and finite."
        raise ValueError(msg)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 2.0 * special.struve(1, arr) / arr
    # H1(x)/x -> 0 as x -> 0 (H1 ~ 2 x^2 / 3 pi), so X1 -> 0; the 0/0 at x = 0
    # is the only non-finite point (the input is validated finite).
    out = np.where(np.isfinite(out), out, 0.0)
    return out[()] if out.ndim == 0 else out


def piston_directivity(ka: ArrayLike, theta: ArrayLike) -> np.ndarray | float:
    r"""Far-field directivity :math:`D = 2 J_1(ka \sin\theta) / (ka \sin\theta)`.

    The pressure amplitude of a baffled circular piston relative to its
    on-axis value (Beranek & Mellow Eq. (4.42)), normalized so
    :math:`D(0) = 1`.

    :param ka: Wavenumber-radius product ``ka`` (scalar or array).
    :param theta: Polar angle from the axis, rad (scalar or array). Broadcast
        against ``ka``.
    :return: ``D`` (float for scalar inputs, else an array).
    """
    ka_arr = np.asarray(ka, dtype=np.float64)
    theta_arr = np.asarray(theta, dtype=np.float64)
    if np.any(ka_arr < 0.0) or not np.all(np.isfinite(ka_arr)):
        msg = "'ka' must be non-negative and finite."
        raise ValueError(msg)
    if not np.all(np.isfinite(theta_arr)):
        msg = "'theta' must be finite."
        raise ValueError(msg)
    try:
        np.broadcast_shapes(ka_arr.shape, theta_arr.shape)
    except ValueError:
        msg = (
            "'ka' and 'theta' must broadcast together; got shapes "
            f"{ka_arr.shape} and {theta_arr.shape}."
        )
        raise ValueError(msg) from None
    u = ka_arr * np.sin(theta_arr)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 2.0 * special.j1(u) / u
    # 2 J1(u)/u -> 1 as u -> 0 (on-axis, or ka = 0): the 0/0 there is the only
    # non-finite point and is set to that limit.
    out = np.where(np.isfinite(out), out, 1.0)
    return out[()] if out.ndim == 0 else out


#: Default polar-angle grid of the directivity pattern: 361 points spanning the
#: front hemisphere ``-90 deg`` to ``+90 deg`` (the baffle blocks the rear),
#: 0.5 deg apart, so the beam pattern is smooth even for a narrow high-``ka``
#: main lobe.
_DEFAULT_DIRECTIVITY_ANGLES = np.radians(np.linspace(-90.0, 90.0, 361))


@dataclass(frozen=True)
class PistonDirectivity:
    r"""Far-field directivity pattern of a baffled circular piston.

    Bundles the far-field directivity
    :math:`D(\theta) = 2 J_1(ka \sin\theta) / (ka \sin\theta)` (Beranek &
    Mellow
    Eq. (4.42)) of one or more baffled circular pistons over a shared
    polar-angle grid, so the classic beam pattern can be drawn with
    :meth:`plot`. The maths is :func:`piston_directivity`; this is a thin,
    plottable bundle around it.

    :ivar angles: Polar angles ``theta`` from the axis, rad.
    :ivar ka: Wavenumber-radius products ``ka``, one per pattern (a 1-D array).
    :ivar directivity: Linear directivity :math:`D(\theta)`, normalized so
        :math:`D(0) = 1`, as a ``(len(ka), len(angles))`` array; row ``i`` is
        the pattern for ``ka[i]``.
    :ivar directivity_db: Directivity in dB,
        :math:`20 \log_{10} \lvert D\rvert`, same shape as
        :attr:`directivity` (the side-lobe nulls floor at a large negative
        value rather than ``-inf``).
    """

    angles: np.ndarray
    ka: np.ndarray
    directivity: np.ndarray
    directivity_db: np.ndarray

    def __post_init__(self) -> None:
        """Reject a bundle whose pattern matrix does not match its two grids.

        The matrix is indexed by position on both sides: :meth:`plot` loops
        over :attr:`ka` and reads row ``i`` as the pattern of ``ka[i]``,
        drawing its columns over :attr:`angles`. Neither pairing is recorded
        anywhere else, and only one of the ways they can disagree is quiet.
        A matrix one row short runs that loop off the end of it, and the row
        that is not there comes back as an ``IndexError`` out of NumPy; a
        column count that differs in either direction is refused by matplotlib
        with ``x and y must have same first dimension``. Both come from inside
        the drawing code and name no field. A matrix one row too many is the
        silent one: the loop stops at the last ``ka``, so the extra pattern is
        never drawn -- no curve, no legend entry and no complaint.

        :raises ValueError: if the pattern matrix disagrees with either grid.
        """
        require_ranks(self, angles=1, ka=1, directivity=2, directivity_db=2)
        require_same_length(self, "ka", "directivity", "directivity_db", axis="pattern")
        require_same_length(
            self,
            "angles",
            ("directivity", 1),
            ("directivity_db", 1),
            axis="polar angle",
        )

    def plot(
        self, ax: PolarAxes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the far-field directivity (beam) pattern on a polar axes.

        Draws the directivity in dB against the polar angle: one curve per
        ``ka`` value as a single family (still one concept, the directivity
        pattern). A polar axes is created when ``ax`` is ``None``. Requires
        matplotlib (``pip install phonometry[plot]``).

        :param ax: Existing (polar) axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the ``Axes.plot`` call when the result
            holds a single ``ka``; ignored for a multi-``ka`` family so one
            user color or label cannot collapse the per-curve styling.
        :return: The axes.
        """
        from .._i18n import check_language
        from .._plot.electroacoustics import plot_piston_directivity

        check_language(language)
        return plot_piston_directivity(self, ax=ax, language=language, **kwargs)


def piston_directivity_pattern(
    ka: ArrayLike,
    angles: ArrayLike | None = None,
) -> PistonDirectivity:
    r"""Far-field directivity pattern of one or more baffled circular pistons.

    Samples the directivity
    :math:`D(\theta) = 2 J_1(ka \sin\theta) / (ka \sin\theta)`
    (Beranek & Mellow Eq. (4.42)) at every ``ka`` over a polar-angle grid and
    bundles it into a :class:`PistonDirectivity` that exposes ``.plot()``. The
    main lobe narrows as ``ka`` grows; its first null appears once ``ka`` passes
    the first zero of ``J1`` (:math:`ka \sin\theta = 3.8317`).

    :param ka: Wavenumber-radius product(s) ``ka`` (scalar or 1-D array), each
        non-negative.
    :param angles: Polar angles ``theta`` from the axis, rad (1-D). ``None``
        (default) uses 361 points spanning the front hemisphere ``-90 deg`` to
        ``+90 deg``, 0.5 deg apart.
    :return: A :class:`PistonDirectivity`.
    """
    ka_arr = np.atleast_1d(np.asarray(ka, dtype=np.float64))
    if ka_arr.ndim != 1 or ka_arr.size == 0:
        msg = "'ka' must be a non-empty scalar or 1-D array."
        raise ValueError(msg)
    if np.any(ka_arr < 0.0) or not np.all(np.isfinite(ka_arr)):
        msg = "'ka' must be non-negative and finite."
        raise ValueError(msg)

    if angles is None:
        angle_arr = _DEFAULT_DIRECTIVITY_ANGLES.copy()
    else:
        angle_arr = np.atleast_1d(np.asarray(angles, dtype=np.float64))
        if angle_arr.ndim != 1 or angle_arr.size == 0:
            msg = "'angles' must be a non-empty 1-D array."
            raise ValueError(msg)
        if not np.all(np.isfinite(angle_arr)):
            msg = "'angles' must be finite."
            raise ValueError(msg)

    directivity = np.asarray(
        piston_directivity(ka_arr[:, None], angle_arr[None, :]),
        dtype=np.float64,
    ).reshape(ka_arr.size, angle_arr.size)
    tiny = np.finfo(np.float64).tiny
    directivity_db = 20.0 * np.log10(np.maximum(np.abs(directivity), tiny))
    return PistonDirectivity(
        angles=angle_arr,
        ka=ka_arr,
        directivity=directivity,
        directivity_db=directivity_db,
    )


def _directivity_index(ka: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Directivity index :math:`DI = 10 \log_{10} Q` per ``ka`` by
    hemisphere quadrature.

    :math:`Q = 2 / \int_0^{\pi/2} \lvert D \rvert^2 \sin\theta \,
    d\theta` (radiation into the half-space in front of the baffle). Uses
    a fine trapezoid grid; :math:`ka \to 0` gives the half-space baffle
    gain :math:`DI = 10 \log_{10} 2 = 3.01` dB and high ``ka`` tends to
    :math:`Q = (ka)^2` (:math:`DI = 20 \log_{10} ka`).
    """
    theta = np.linspace(0.0, 0.5 * np.pi, 2001)
    sin_t = np.sin(theta)
    d = np.asarray(piston_directivity(ka[:, None], theta[None, :]), dtype=np.float64)
    integrand = d**2 * sin_t[None, :]
    integral = np.trapezoid(integrand, theta, axis=-1)
    return np.asarray(10.0 * np.log10(2.0 / integral), dtype=np.float64)


@dataclass(frozen=True)
class RadiatingPistonResult:
    r"""Radiation impedance and directivity of a baffled circular piston.

    :ivar frequencies: Frequencies ``f``, Hz.
    :ivar ka: Wavenumber-radius product ``ka`` at each frequency.
    :ivar resistance: Normalized piston resistance :math:`R_1(2ka)` (real
        part of :math:`Z_\mathrm{r} / (\rho c S)`).
    :ivar reactance: Normalized piston reactance :math:`X_1(2ka)`
        (imaginary part of :math:`Z_\mathrm{r} / (\rho c S)`).
    :ivar radiation_resistance: Mechanical radiation resistance
        :math:`\rho c S R_1`, N s/m.
    :ivar radiation_reactance: Mechanical radiation reactance
        :math:`\rho c S X_1`, N s/m.
    :ivar radiation_mass: Low-frequency accreted air mass
        :math:`M_\mathrm{r} = 8 \rho a^3/3`, kg (a single value; the mass limit of
        ``radiation_reactance / omega``).
    :ivar directivity_index: Directivity index
        :math:`DI = 10 \log_{10} Q`, dB.
    :ivar angles: Polar angles of ``directivity``, rad, or ``None`` if not
        requested.
    :ivar directivity: Far-field directivity :math:`D(\theta)` as a
        ``(n_freq, n_angle)`` array, or ``None`` if ``angles`` was not given.
    :ivar radius: Piston radius ``a``, m.
    :ivar speed_of_sound: Speed of sound ``c``, m/s.
    :ivar density: Air density :math:`\rho`, kg/m3.
    """

    frequencies: np.ndarray
    ka: np.ndarray
    resistance: np.ndarray
    reactance: np.ndarray
    radiation_resistance: np.ndarray
    radiation_reactance: np.ndarray
    radiation_mass: float
    directivity_index: np.ndarray
    angles: np.ndarray | None
    directivity: np.ndarray | None
    radius: float
    speed_of_sound: float
    density: float

    def __post_init__(self) -> None:
        """Reject a piston result whose quantities disagree along an axis.

        The impedance quantities are one curve each over the same frequency
        axis, and :meth:`plot` draws two of them against a third; the
        directivity pattern crosses that axis with the angle grid, carrying
        the frequencies down its rows and the angles across its columns, and
        :meth:`plot_geometry` picks one row by ``frequency_index`` to overlay
        the far-field lobe. That row is a positional lookup into an array that
        states nowhere how long it ought to be, so a pattern short of the
        frequency count draws the lobe of one frequency under the ``ka`` of
        another, silently and to scale. The angle axis is the loud half: a
        column count that disagrees with :attr:`angles` is already refused
        further down, by ``'angles' and 'directivity' must match.`` from the
        drawing helper. The check keeps it here so that one is raised at
        construction, against the result that carries it, instead of on the
        way to a figure.

        :raises ValueError: if a per-frequency quantity, or the pattern's
            angle axis, disagrees with the rest.
        """
        require_ranks(
            self,
            frequencies=1,
            ka=1,
            resistance=1,
            reactance=1,
            radiation_resistance=1,
            radiation_reactance=1,
            directivity_index=1,
            angles=1,
            directivity=2,
        )
        require_same_length(
            self,
            "frequencies",
            "ka",
            "resistance",
            "reactance",
            "radiation_resistance",
            "radiation_reactance",
            "directivity_index",
            "directivity",
            axis="frequency",
        )
        require_same_length(self, "angles", ("directivity", 1), axis="polar angle")
        # Finiteness is a construction invariant, not a rendering question:
        # :func:`piston_directivity` validates its inputs and patches the
        # on-axis 0/0 to its limit, so no producer emits a non-finite
        # pattern; one smuggled NaN would silently drop the whole lobe from
        # :meth:`plot_geometry` (the lobe's peak turns NaN and its `> 0`
        # gate turns False). The shared guard is what runs it, so that a
        # field holding something that is not a number at all is refused by
        # name too, instead of reaching ``np.isfinite`` as an anonymous
        # ``TypeError`` raised from inside numpy.
        require_finite_fields(self, "angles", "directivity")

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the normalized piston resistance and reactance against ``ka``.

        Reproduces the classic Beranek & Mellow figure: ``R1`` rising to 1 and
        ``X1`` peaking then decaying, over the ``ka`` range of the result.
        Requires matplotlib (``pip install phonometry[plot]``).

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.electroacoustics import plot_piston_impedance

        check_language(language)
        return plot_piston_impedance(self, ax=ax, language=language, **kwargs)

    def plot_geometry(
        self,
        ax: Axes | None = None,
        *,
        frequency_index: int = -1,
        language: str = "en",
        **kwargs: Any,
    ) -> Axes:
        """Draw the baffled piston to scale, with its directivity lobe.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param frequency_index: Index of the frequency whose far-field lobe
            is overlaid (default: the highest); ignored when the result
            carries no directivity.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the piston rectangle.
        """
        from .._i18n import check_language
        from .._plot.geometry import plot_piston_result_geometry

        check_language(language)
        return plot_piston_result_geometry(
            self,
            ax=ax,
            frequency_index=frequency_index,
            language=language,
            **kwargs,
        )


def radiating_piston(
    radius: float,
    frequencies: ArrayLike,
    *,
    speed_of_sound: float = _C_AIR,
    density: float = _RHO_AIR,
    angles: ArrayLike | None = None,
) -> RadiatingPistonResult:
    r"""Radiation impedance and directivity of a rigid baffled circular
    piston.

    Evaluates the piston resistance :math:`R_1(2ka)` and reactance
    :math:`X_1(2ka)`, the mechanical radiation impedance
    :math:`\rho c S (R_1 + j X_1)`, the low-frequency radiation mass
    :math:`8 \rho a^3 / 3` and the directivity index over the given
    frequencies (Beranek & Mellow §4.4). Pass ``angles`` to also sample
    the far-field directivity pattern :math:`D(\theta)`.

    :param radius: Piston radius ``a``, m.
    :param frequencies: Frequencies ``f``, Hz (scalar or 1-D array), all > 0.
    :param speed_of_sound: Speed of sound ``c``, m/s (default 343).
    :param density: Air density :math:`\rho`, kg/m3 (default 1.206).
    :param angles: Optional polar angles ``theta`` from the axis, rad, at which
        to sample the directivity pattern.
    :return: A :class:`RadiatingPistonResult`.
    """
    a = require_positive(radius, "radius")
    c = require_positive(speed_of_sound, "speed_of_sound")
    rho = require_positive(density, "density")
    f = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if f.ndim != 1 or f.size == 0:
        msg = "'frequencies' must be a non-empty 1-D array."
        raise ValueError(msg)
    if np.any(f <= 0.0) or not np.all(np.isfinite(f)):
        msg = "'frequencies' must be positive and finite."
        raise ValueError(msg)

    omega = 2.0 * np.pi * f
    k = omega / c
    ka = k * a
    area = np.pi * a**2
    r1 = np.asarray(piston_resistance(2.0 * ka), dtype=np.float64)
    x1 = np.asarray(piston_reactance(2.0 * ka), dtype=np.float64)
    rho_c_s = rho * c * area
    radiation_mass = 8.0 * rho * a**3 / 3.0
    di = _directivity_index(ka)

    directivity: np.ndarray | None = None
    angle_arr: np.ndarray | None = None
    if angles is not None:
        angle_arr = np.atleast_1d(np.asarray(angles, dtype=np.float64))
        if angle_arr.ndim != 1 or angle_arr.size == 0:
            msg = "'angles' must be a non-empty 1-D array."
            raise ValueError(msg)
        if not np.all(np.isfinite(angle_arr)):
            msg = "'angles' must be finite."
            raise ValueError(msg)
        directivity = np.asarray(
            piston_directivity(ka[:, None], angle_arr[None, :]),
            dtype=np.float64,
        )

    return RadiatingPistonResult(
        frequencies=f,
        ka=ka,
        resistance=r1,
        reactance=x1,
        radiation_resistance=rho_c_s * r1,
        radiation_reactance=rho_c_s * x1,
        radiation_mass=radiation_mass,
        directivity_index=di,
        angles=angle_arr,
        directivity=directivity,
        radius=a,
        speed_of_sound=c,
        density=rho,
    )
