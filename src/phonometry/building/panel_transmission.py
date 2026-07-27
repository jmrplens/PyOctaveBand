#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Predicted airborne sound reduction index of panels (Bies, Hansen & Howard
2017, Engineering Noise Control 5e, Section 7.2; Sharp 1973).

Where EN 12354-1 (:mod:`phonometry.building.building_prediction`) takes the
element sound reduction index ``R`` as a *measured* input, this module
**predicts** ``R(f)`` from the physical properties of the construction: the mass
per unit area, bending stiffness (through the coincidence frequency) and loss
factor. The prediction feeds the same ISO 717-1 weighting
(:func:`phonometry.weighted_rating`) as the measured quantities, closing the
chain from panel physics to the single-number ``Rw``.

**Mass law (Bies Eq. 7.40/7.42).** A non-stiff panel transmits by forced motion;
the transmission coefficient of an infinite limp panel gives the normal- and
field-incidence transmission loss::

    TL_normal = 10 lg(1 + (pi f m'' / (rho0 c0))**2)
    TL_field  = TL_normal - dB(band)

with ``m''`` the mass per unit area, ``rho0 c0`` the characteristic impedance of
air and the field-incidence correction ``dB = 5.5`` dB for one-third-octave or
``4.0`` dB for octave bands (Eq. 7.42). The mass law rises 6 dB per octave and
6 dB per doubling of mass.

**Single panel, Sharp's method (Bies 7.2.4.1).** Below the coincidence region
the field-incidence mass law holds; from the coincidence frequency ``fc``
upwards the loss factor ``eta`` controls the transmission (Eq. 7.44)::

    TL = 10 lg(1 + (pi f m'' / rho0 c0)**2) + 10 lg(2 eta f / (pi fc))

and between ``fc/2`` and ``fc`` the curve is a straight line on ``TL`` versus
``log10 f``. The coincidence dip at ``fc`` sits ``10 lg(2 eta / pi)`` below the
extrapolated mass law (Bies design-chart point B,
``TL = 20 lg(fc m'') + 10 lg eta - 44``).

**Double wall (Bies 7.2.6, Eq. 7.62-7.64).** Two leaves ``m1``, ``m2`` separated
by a gap ``d`` behave as a mass-spring-mass system. Below the resonance
``f0 = (1/2 pi) sqrt(s'' (m1 + m2)/(m1 m2))`` the pair follows the mass law of
the combined mass ``m1 + m2``; above it the two mass laws add, boosted by the
cavity (Eq. 7.64)::

    TL = TL_M                              , f <= f0
    TL = TL_1 + TL_2 + 20 lg(2 k d)        , f0 < f < f_l   (k = 2 pi f / c0)
    TL = TL_1 + TL_2 + 6                   , f >= f_l = c0 / (2 pi d)

The cavity stiffness ``s''`` is ``rho0 c0**2 / d`` for an empty (adiabatic) air
gap; a porous fill (a :class:`~phonometry.materials.PorousMediumResult` from
:mod:`phonometry.materials.porous_absorber`) lowers the resonance through its
softer, near-isothermal effective bulk modulus and damps the cavity so the
mid-band slope is realised without standing-wave dips.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from .._internal.validation import require_choice, require_positive
from ..vibration.radiation_efficiency import coincidence_frequency

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..materials.porous_absorber import PorousMediumResult
    from .insulation import WeightedRatingResult

#: Default speed of sound in air ``c0``, m/s (20 degC).
_SPEED_OF_SOUND: float = 343.0
#: Default air density ``rho0``, kg/m^3 (Bies 5e Appendix D: 1,205 at 20 degC).
_AIR_DENSITY: float = 1.205
#: Ratio of specific heats of air ``gamma``.
_GAMMA: float = 1.4
#: Field-incidence correction ``dB`` (Bies Eq. 7.42), keyed by band width.
_FIELD_CORRECTION: dict[str, float] = {"third": 5.5, "octave": 4.0}
#: Error message for a non-positive frequency (shared by the module funcs).
_FREQ_POSITIVE_MSG = "'frequency' must be positive."

__all__ = [
    "SoundReductionResult",
    "double_wall_transmission_loss",
    "field_incidence_correction",
    "mass_law_transmission_loss",
    "mass_spring_mass_resonance",
    "single_panel_transmission_loss",
]


def field_incidence_correction(band: str = "third") -> float:
    """Field-incidence mass-law correction ``dB`` (Bies Eq. 7.42).

    :param band: ``"third"`` (5.5 dB) or ``"octave"`` (4.0 dB).
    :return: The correction subtracted from the normal-incidence mass law, dB.
    :raises ValueError: for an unknown band width.
    """
    band = require_choice(band, "band", tuple(_FIELD_CORRECTION))
    return _FIELD_CORRECTION[band]


def mass_law_transmission_loss(
    frequency: ArrayLike,
    mass_per_area: float,
    *,
    incidence: str = "field",
    band: str = "third",
    speed_of_sound: float = _SPEED_OF_SOUND,
    air_density: float = _AIR_DENSITY,
) -> np.ndarray:
    """Mass-law transmission loss of a limp panel (Bies Eq. 7.40/7.42).

    ``TL_normal = 10 lg(1 + (pi f m'' / rho0 c0)**2)``; the field-incidence
    value subtracts the band correction of :func:`field_incidence_correction`.

    :param frequency: Frequency ``f``, in hertz (scalar or array, > 0).
    :param mass_per_area: Mass per unit area ``m''``, in kg/m^2 (> 0).
    :param incidence: ``"normal"`` or ``"field"`` (Default: ``"field"``).
    :param band: Band width for the field correction (``"third"``/``"octave"``).
    :param speed_of_sound: Speed of sound in air ``c0`` (Default: 343 m/s).
    :param air_density: Air density ``rho0`` (Default: 1.205 kg/m^3).
    :return: The transmission loss ``TL``, in dB.
    :raises ValueError: for a non-positive input or unknown incidence/band.
    """
    incidence = require_choice(incidence, "incidence", ("normal", "field"))
    m2 = require_positive(mass_per_area, "mass_per_area")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    rho0 = require_positive(air_density, "air_density")
    f = np.asarray(frequency, dtype=np.float64)
    if np.any(f <= 0.0):
        raise ValueError(_FREQ_POSITIVE_MSG)
    ratio = np.pi * f * m2 / (rho0 * c0)
    tl = 10.0 * np.log10(1.0 + ratio**2)
    if incidence == "field":
        tl = tl - field_incidence_correction(band)
    return np.asarray(tl, dtype=np.float64)


@dataclass(frozen=True)
class SoundReductionResult:
    """Predicted airborne sound reduction index ``R(f)`` of a construction.

    :ivar frequencies: Band centre frequencies, in hertz.
    :ivar transmission_loss: Sound reduction index ``R`` per band, in dB.
    :ivar model: Prediction model (e.g. ``"sharp-single"``, ``"double-wall"``).
    :ivar critical_frequency: Coincidence frequency ``fc``, in hertz, or
        ``None`` (double wall reports the mass-spring-mass resonance instead).
    :ivar resonance_frequency: Mass-spring-mass resonance ``f0``, in hertz, or
        ``None`` (single panel).
    :ivar mass1: First-leaf surface density, in kg/m2, retained (with
        ``mass2`` and ``gap``) by the double-wall constructor so
        :meth:`plot_geometry` can draw the section; ``None`` otherwise.
    :ivar mass2: Second-leaf surface density, in kg/m2, or ``None``.
    :ivar gap: Cavity depth, in metres, or ``None``.
    """

    frequencies: np.ndarray
    transmission_loss: np.ndarray
    model: str
    critical_frequency: float | None = None
    resonance_frequency: float | None = None
    mass1: float | None = None
    mass2: float | None = None
    gap: float | None = None

    @property
    def transmission_coefficient(self) -> np.ndarray:
        """Transmission coefficient ``tau = 10**(-R/10)`` per band."""
        r = np.asarray(self.transmission_loss, dtype=np.float64)
        return np.asarray(10.0 ** (-r / 10.0), dtype=np.float64)

    def rating(self, bands: str | None = None) -> WeightedRatingResult:
        """Single-number weighted rating ``Rw`` of the predicted ``R(f)``.

        Delegates to :func:`phonometry.weighted_rating` (ISO 717-1); requires
        the spectrum to be on the 16 one-third-octave bands (100 Hz to
        3150 Hz) or the 5 octave bands (125 Hz to 2000 Hz).

        :param bands: Band set forwarded to :func:`phonometry.weighted_rating`.
        :return: The :class:`~phonometry.building.insulation.WeightedRatingResult`.
        """
        from .insulation import weighted_rating

        return weighted_rating(self.transmission_loss, bands)

    def report(self, path: str, **kwargs: Any) -> str:
        """Render the ISO 717-1 Annex C rating fiche of ``R(f)`` to a PDF.

        Convenience wrapper delegating to
        :meth:`~phonometry.building.insulation.WeightedRatingResult.report`
        on :meth:`rating`; requires the predicted spectrum to be on the 16
        one-third-octave bands (100 Hz to 3150 Hz) or the 5 octave bands
        (125 Hz to 2000 Hz).

        :param path: Destination path of the PDF file.
        :param kwargs: Forwarded to
            :meth:`~phonometry.building.insulation.WeightedRatingResult.report`
            (e.g. ``engine``).
        :return: The written ``path`` as a :class:`str`.
        """
        return self.rating().report(path, **kwargs)

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the predicted sound reduction index ``R(f)``.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.building import plot_sound_reduction

        check_language(language)
        return plot_sound_reduction(self, ax=ax, language=language, **kwargs)

    def plot_geometry(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw the mass-spring-mass cross-section to scale.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :raises ValueError: If the result does not retain its geometry.
        """
        from .._i18n import check_language
        from .._plot.geometry import plot_double_wall_result_geometry

        check_language(language)
        return plot_double_wall_result_geometry(self, ax=ax, language=language, **kwargs)


def single_panel_transmission_loss(
    frequency: ArrayLike,
    mass_per_area: float,
    *,
    critical_frequency: float | None = None,
    bending_stiffness: float | None = None,
    loss_factor: float = 0.01,
    band: str = "third",
    speed_of_sound: float = _SPEED_OF_SOUND,
    air_density: float = _AIR_DENSITY,
) -> SoundReductionResult:
    """Sound reduction index of a single panel, Sharp's method (Bies 7.2.4.1).

    Field-incidence mass law up to ``fc/2``, Eq. 7.44 from ``fc`` upwards, and a
    straight line in ``log10 f`` across the coincidence region between them.

    Provide the coincidence frequency directly through *critical_frequency*, or
    let it be computed from *bending_stiffness* and *mass_per_area* through
    :func:`~phonometry.vibration.radiation_efficiency.coincidence_frequency`.

    :param frequency: Band centre frequencies ``f``, in hertz (array, > 0).
    :param mass_per_area: Mass per unit area ``m''``, in kg/m^2 (> 0).
    :param critical_frequency: Coincidence frequency ``fc``, in hertz (> 0).
    :param bending_stiffness: Bending stiffness per unit width ``B'``, in N.m,
        used to compute ``fc`` when *critical_frequency* is not given.
    :param loss_factor: Total loss factor ``eta`` (> 0, Default: 0.01).
    :param band: Band width for the field correction (``"third"``/``"octave"``).
    :param speed_of_sound: Speed of sound in air ``c0`` (Default: 343 m/s).
    :param air_density: Air density ``rho0`` (Default: 1.205 kg/m^3).
    :return: A :class:`SoundReductionResult` (model ``"sharp-single"``).
    :raises ValueError: for a non-positive input, or if neither
        *critical_frequency* nor *bending_stiffness* is given.
    """
    m2 = require_positive(mass_per_area, "mass_per_area")
    eta = require_positive(loss_factor, "loss_factor")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    rho0 = require_positive(air_density, "air_density")
    f = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
    if f.ndim != 1 or f.size == 0:
        raise ValueError("'frequency' must be a non-empty 1-D array.")
    if np.any(f <= 0.0):
        raise ValueError(_FREQ_POSITIVE_MSG)
    if critical_frequency is not None:
        fc = require_positive(critical_frequency, "critical_frequency")
    elif bending_stiffness is not None:
        fc = coincidence_frequency(m2, bending_stiffness, speed_of_sound=c0)
    else:
        raise ValueError(
            "provide 'critical_frequency' or 'bending_stiffness' to locate the "
            "coincidence frequency."
        )

    def _tl_normal(freq: np.ndarray) -> np.ndarray:
        return mass_law_transmission_loss(
            freq, m2, incidence="normal",
            speed_of_sound=c0, air_density=rho0,
        )

    correction = field_incidence_correction(band)
    tl = np.empty_like(f)
    below = f <= 0.5 * fc
    above = f >= fc
    middle = ~below & ~above
    # Field-incidence mass law below the coincidence region.
    tl[below] = _tl_normal(f[below]) - correction
    # Eq. 7.44 from fc upwards.
    tl[above] = _tl_normal(f[above]) + 10.0 * np.log10(
        2.0 * eta * f[above] / (np.pi * fc)
    )
    # Straight line on TL vs log10(f) across fc/2 .. fc.
    if np.any(middle):
        f_lo, f_hi = 0.5 * fc, fc
        tl_lo = _tl_normal(np.array([f_lo])) - correction
        tl_hi = _tl_normal(np.array([f_hi])) + 10.0 * np.log10(2.0 * eta / np.pi)
        frac = (np.log10(f[middle]) - np.log10(f_lo)) / (
            np.log10(f_hi) - np.log10(f_lo)
        )
        tl[middle] = tl_lo[0] + frac * (tl_hi[0] - tl_lo[0])
    return SoundReductionResult(
        frequencies=f,
        transmission_loss=np.asarray(tl, dtype=np.float64),
        model="sharp-single",
        critical_frequency=fc,
    )


def mass_spring_mass_resonance(
    mass1: float,
    mass2: float,
    gap: float,
    *,
    cavity_medium: PorousMediumResult | None = None,
    speed_of_sound: float = _SPEED_OF_SOUND,
    air_density: float = _AIR_DENSITY,
) -> float:
    """Mass-spring-mass resonance ``f0`` of a double wall (Bies Eq. 7.62).

    ``f0 = (1/2 pi) sqrt(s'' (m1 + m2)/(m1 m2))`` with the cavity stiffness per
    unit area ``s''``. For an empty air gap ``s'' = rho0 c0**2 / d`` (adiabatic,
    Hopkins Eq. 4.72); with a porous *cavity_medium* the fill's effective
    (near-isothermal) bulk modulus at the lowest supplied frequency sets a
    softer ``s'' = Re(K_e) / d``, lowering ``f0``.

    :param mass1: Surface density of leaf 1 ``m1``, in kg/m^2 (> 0).
    :param mass2: Surface density of leaf 2 ``m2``, in kg/m^2 (> 0).
    :param gap: Cavity depth ``d``, in m (> 0).
    :param cavity_medium: Optional porous fill (a
        :class:`~phonometry.materials.PorousMediumResult`) whose effective bulk
        modulus sets the cavity stiffness.
    :param speed_of_sound: Speed of sound in air ``c0`` (Default: 343 m/s).
    :param air_density: Air density ``rho0`` (Default: 1.205 kg/m^3).
    :return: The mass-spring-mass resonance ``f0``, in hertz.
    :raises ValueError: for a non-positive input.
    """
    m1 = require_positive(mass1, "mass1")
    m2 = require_positive(mass2, "mass2")
    d = require_positive(gap, "gap")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    rho0 = require_positive(air_density, "air_density")
    if cavity_medium is None:
        stiffness = rho0 * c0**2 / d
    else:
        bulk = np.atleast_1d(
            np.asarray(cavity_medium.bulk_modulus, dtype=np.complex128)
        )
        stiffness = float(np.real(bulk.flat[0])) / d
        if stiffness <= 0.0:
            raise ValueError("'cavity_medium' bulk modulus must be positive.")
    reduced = (m1 + m2) / (m1 * m2)
    return float(np.sqrt(stiffness * reduced) / (2.0 * np.pi))


def double_wall_transmission_loss(
    frequency: ArrayLike,
    mass1: float,
    mass2: float,
    gap: float,
    *,
    loss_factor: float = 0.1,
    cavity_medium: PorousMediumResult | None = None,
    band: str = "third",
    speed_of_sound: float = _SPEED_OF_SOUND,
    air_density: float = _AIR_DENSITY,
) -> SoundReductionResult:
    """Sound reduction index of a double wall (Bies 7.2.6, Eq. 7.64).

    Piecewise Sharp model: below the mass-spring-mass resonance ``f0`` the pair
    behaves as the mass law of the combined mass; between ``f0`` and the
    limiting frequency ``f_l = c0/(2 pi d)`` the two mass laws add plus
    ``20 lg(2 k d)``; above ``f_l`` they add plus 6 dB. The curve is continuous
    at ``f_l`` (``20 lg(2 k d) = 6`` there).

    :param frequency: Band centre frequencies ``f``, in hertz (array, > 0).
    :param mass1: Surface density of leaf 1 ``m1``, in kg/m^2 (> 0).
    :param mass2: Surface density of leaf 2 ``m2``, in kg/m^2 (> 0).
    :param gap: Cavity depth ``d``, in m (> 0).
    :param loss_factor: Leaf loss factor ``eta`` (> 0, Default: 0.1); reserved
        for the coincidence extension and reported for reference.
    :param cavity_medium: Optional porous fill; see
        :func:`mass_spring_mass_resonance`.
    :param band: Band width for the field correction (``"third"``/``"octave"``).
    :param speed_of_sound: Speed of sound in air ``c0`` (Default: 343 m/s).
    :param air_density: Air density ``rho0`` (Default: 1.205 kg/m^3).
    :return: A :class:`SoundReductionResult` (model ``"double-wall"``).
    :raises ValueError: for a non-positive input.
    """
    m1 = require_positive(mass1, "mass1")
    m2 = require_positive(mass2, "mass2")
    d = require_positive(gap, "gap")
    require_positive(loss_factor, "loss_factor")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    rho0 = require_positive(air_density, "air_density")
    f = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
    if f.ndim != 1 or f.size == 0:
        raise ValueError("'frequency' must be a non-empty 1-D array.")
    if np.any(f <= 0.0):
        raise ValueError(_FREQ_POSITIVE_MSG)

    f0 = mass_spring_mass_resonance(
        m1, m2, d, cavity_medium=cavity_medium,
        speed_of_sound=c0, air_density=rho0,
    )
    f_l = c0 / (2.0 * np.pi * d)

    def _ml(freq: np.ndarray, mass: float) -> np.ndarray:
        return mass_law_transmission_loss(
            freq, mass, incidence="field", band=band,
            speed_of_sound=c0, air_density=rho0,
        )

    tl = np.empty_like(f)
    # Strict partition by precedence, so lightweight leaves with a wide gap
    # (which can push f0 above f_l, collapsing the transition band) never make
    # the masks overlap and silently overwrite each other: below the resonance
    # first, then the saturated high branch, then whatever transition remains.
    below = f <= f0
    high = (f >= f_l) & ~below
    mid = ~below & ~high
    tl[below] = _ml(f[below], m1 + m2)
    tl1 = _ml(f, m1)
    tl2 = _ml(f, m2)
    k = 2.0 * np.pi * f / c0
    tl[mid] = tl1[mid] + tl2[mid] + 20.0 * np.log10(2.0 * k[mid] * d)
    tl[high] = tl1[high] + tl2[high] + 6.0
    return SoundReductionResult(
        frequencies=f,
        transmission_loss=np.asarray(tl, dtype=np.float64),
        model="double-wall",
        resonance_frequency=f0,
        mass1=float(mass1),
        mass2=float(mass2),
        gap=float(gap),
    )
