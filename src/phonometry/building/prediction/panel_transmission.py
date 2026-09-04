#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Predicted airborne sound reduction index of panels (Bies, Hansen & Howard
2017, Engineering Noise Control 5e, Section 7.2; Sharp 1973).

Where EN 12354-1 (:mod:`phonometry.building.prediction.simplified_model`) takes the
element sound reduction index ``R`` as a *measured* input, this module
**predicts** ``R(f)`` from the physical properties of the construction: the mass
per unit area, bending stiffness (through the coincidence frequency) and loss
factor. The prediction feeds the same ISO 717-1 weighting
(:func:`phonometry.building.weighted_rating`) as the measured quantities,
closing the chain from panel physics to the single-number ``Rw``.

**Mass law (Bies Eq. 7.40/7.42).** A non-stiff panel transmits by forced motion;
the transmission coefficient of an infinite limp panel gives the normal- and
field-incidence transmission loss:

.. math::

   \mathrm{TL}_{\mathrm{normal}} = 10 \log_{10}\!\left[ 1 +
   \left( \frac{\pi f m''}{\rho_0 c_0} \right)^{2} \right]

   \mathrm{TL}_{\mathrm{field}} = \mathrm{TL}_{\mathrm{normal}}
   - \mathrm{dB}(\text{band})

with ``m''`` the mass per unit area, :math:`\rho_0 c_0` the characteristic
impedance of air and the field-incidence correction
:math:`\mathrm{dB} = 5.5` dB for one-third-octave or ``4.0`` dB for octave
bands (Eq. 7.42). The mass law rises 6 dB per octave and 6 dB per doubling of
mass.

**Single panel, Sharp's method (Bies 7.2.4.1).** Below the coincidence region
the field-incidence mass law holds; from the coincidence frequency ``fc``
upwards the loss factor ``eta`` controls the transmission (Eq. 7.44):

.. math::

   \mathrm{TL} = 10 \log_{10}\!\left[ 1 +
   \left( \frac{\pi f m''}{\rho_0 c_0} \right)^{2} \right]
   + 10 \log_{10}\frac{2 \eta f}{\pi f_\mathrm{c}}

and between :math:`f_\mathrm{c}/2` and :math:`f_\mathrm{c}` the curve is a straight line on
:math:`\mathrm{TL}` versus :math:`\log_{10} f`. The coincidence dip at
:math:`f_\mathrm{c}` sits :math:`10 \log_{10}(2\eta/\pi)` below the extrapolated mass law
(Bies design-chart point B,
:math:`\mathrm{TL} = 20 \log_{10}(f_\mathrm{c} m'') + 10 \log_{10}\eta - 44`).

**Double wall (Bies 7.2.6, Eq. 7.62-7.64).** Two leaves ``m1``, ``m2`` separated
by a gap ``d`` behave as a mass-spring-mass system. Below the resonance
:math:`f_0 = \frac{1}{2\pi} \sqrt{s'' (m_1 + m_2)/(m_1 m_2)}` the pair
follows the mass law of the combined mass :math:`m_1 + m_2`; above it the two
mass laws add, boosted by the cavity (Eq. 7.64):

.. math::

   \mathrm{TL} = \mathrm{TL}_M, \qquad f \le f_0

   \mathrm{TL} = \mathrm{TL}_1 + \mathrm{TL}_2 + 20 \log_{10}(2 k d),
   \qquad f_0 < f < f_\mathrm{l}, \quad k = 2 \pi f / c_0

   \mathrm{TL} = \mathrm{TL}_1 + \mathrm{TL}_2 + 6,
   \qquad f \ge f_\mathrm{l} = \frac{c_0}{2 \pi d}

The cavity stiffness ``s''`` is :math:`\rho_0 c_0^{2} / d` for an empty
(adiabatic) air gap; a porous fill (a
:class:`~phonometry.materials.PorousMediumResult` from
:mod:`phonometry.materials.absorbers.porous`) lowers the resonance through its
softer, near-isothermal effective bulk modulus and damps the cavity so the
mid-band slope is realised without standing-wave dips.

**Orthotropic panels (Bies 7.2.4.5; Vigran, Building Acoustics, 3.7.3 and
6.5.3).** Ribbed and corrugated cladding is stiff along the corrugations and
limp across them, so a single coincidence frequency no longer exists: the panel
has a *range* :math:`f_{\mathrm{c}1} \le f \le f_{\mathrm{c}2}` bounded by the stiffest and the
least stiff direction (Vigran Eq. 6.107). The bending-wave impedance then
depends on the azimuth ``theta`` as well as the incidence angle ``phi``
(Heckl 1960; Hansen
1993; Vigran Eq. 6.108 = Bies Eq. 7.30), and the diffuse-field average is a
double integral (Vigran Eq. 6.111 = Bies Eq. 7.38). The consequence is the
whole point of the model: over one to two decades the resonant transmission
dominates and ``R`` flattens far below the mass law of a flat plate of the same
mass. See :func:`orthotropic_transmission_loss`,
:func:`orthotropic_critical_frequencies`, :func:`corrugated_plate_stiffness`
and :func:`orthotropic_plate_resonance`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from scipy.integrate import quad
from scipy.special import ellipe

from ..._internal.validation import (
    require_choice,
    require_non_negative,
    require_positive,
    require_ranks,
    require_same_length,
)
from ...materials.absorbers.porous import PUBLISHED_AIR
from ...vibration.structural.point_mobility import plate_bending_stiffness
from ...vibration.structural.radiation_efficiency import coincidence_frequency

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

    from ...fluids import Fluid
    from ...materials.absorbers.porous import PorousMediumResult
    from ..measurement.insulation import WeightedRatingResult

#: Default speed of sound in air ``c0``, m/s, of the single-property entry
#: points: the published air's own, so the module spells 343 m/s once.
_SPEED_OF_SOUND: float = PUBLISHED_AIR.speed_of_sound
#: Field-incidence correction ``dB`` (Bies Eq. 7.42), keyed by band width.
_FIELD_CORRECTION: dict[str, float] = {"third": 5.5, "octave": 4.0}
#: Error message for a non-positive frequency (shared by the module funcs).
_FREQ_POSITIVE_MSG = "'frequency' must be positive."
#: Error message for a malformed frequency axis (shared by the module funcs).
_FREQ_1D_MSG = "'frequency' must be a non-empty 1-D array."

#: Norton & Karczub (2003) Table 3.1: plateau-method data for common
#: materials, as ``(surface density in kg/m2 per mm of thickness, coincidence
#: plateau height in dB, frequency ratio B/A)``.
PLATEAU_MATERIALS: dict[str, tuple[float, float, float]] = {
    "aluminium": (2.66, 29.0, 11.0),
    "brick": (2.10, 37.0, 4.5),
    "concrete": (2.28, 38.0, 4.5),
    "glass": (2.47, 27.0, 10.0),
    "lead": (11.20, 56.0, 4.0),
    "plaster": (1.71, 30.0, 8.0),
    "plywood": (0.57, 19.0, 6.5),
    "steel": (7.60, 40.0, 11.0),
}
#: Field-incidence correction of Norton Eq. (3.106): a flat 5 dB below the
#: normal-incidence mass law (a diffuse field limited to 78 degrees).
_NORTON_FIELD_CORRECTION: float = 5.0
#: Grazing incidence, degrees: the open upper bound of the fixed limiting
#: angle ``theta_L`` of the field-incidence diffuse integral (at 90 degrees
#: the upper limit :math:`\sin^{2}\theta_\mathrm{L}` of Bies Eq. (7.38)
#: reaches 1).
_GRAZING_INCIDENCE_DEG: float = 90.0


def _band_axis(frequency: ArrayLike) -> np.ndarray:
    """Validate a 1-D array of strictly positive band centre frequencies."""
    f = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
    if f.ndim != 1 or f.size == 0:
        raise ValueError(_FREQ_1D_MSG)
    if np.any(f <= 0.0):
        raise ValueError(_FREQ_POSITIVE_MSG)
    return f


__all__ = [
    "PLATEAU_MATERIALS",
    "SoundReductionResult",
    "corrugated_plate_mass_factor",
    "corrugated_plate_stiffness",
    "double_wall_transmission_loss",
    "field_incidence_correction",
    "mass_law_transmission_loss",
    "mass_spring_mass_resonance",
    "orthotropic_critical_frequencies",
    "orthotropic_plate_resonance",
    "orthotropic_transmission_loss",
    "plateau_transmission_loss",
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
    field_correction: float | None = None,
    fluid: Fluid = PUBLISHED_AIR,
) -> np.ndarray:
    r"""Mass-law transmission loss of a limp panel (Bies Eq. 7.40/7.42).

    :math:`\mathrm{TL}_{\mathrm{normal}} = 10 \log_{10}[1 +
    (\pi f m'' / \rho_0 c_0)^{2}]`; the field-incidence value subtracts the
    band correction of :func:`field_incidence_correction`,
    or the explicit *field_correction* when one is given (Norton & Karczub
    Eq. 3.106 uses a flat 5 dB, the line :func:`plateau_transmission_loss`
    builds its estimate on).

    :param frequency: Frequency ``f``, in hertz (scalar or array, > 0).
    :param mass_per_area: Mass per unit area ``m''``, in kg/m^2 (> 0).
    :param incidence: ``"normal"`` or ``"field"`` (Default: ``"field"``).
    :param band: Band width for the field correction (``"third"``/``"octave"``).
    :param field_correction: Explicit field-incidence correction, in dB
        (>= 0), overriding the band table (Default: ``None``).
    :param fluid: The medium, a :class:`~phonometry.fluids.Fluid` (Default:
        :data:`PUBLISHED_AIR`, the air these models are published with).
    :return: The transmission loss ``TL``, in dB.
    :raises ValueError: for a non-positive input or unknown incidence/band.
    """
    incidence = require_choice(incidence, "incidence", ("normal", "field"))
    m2 = require_positive(mass_per_area, "mass_per_area")
    c0 = fluid.speed_of_sound
    rho0 = fluid.density
    f = np.asarray(frequency, dtype=np.float64)
    if np.any(f <= 0.0):
        raise ValueError(_FREQ_POSITIVE_MSG)
    ratio = np.pi * f * m2 / (rho0 * c0)
    tl = 10.0 * np.log10(1.0 + ratio**2)
    if incidence == "field":
        tl = tl - _resolve_field_correction(band, field_correction)
    return np.asarray(tl, dtype=np.float64)


def _resolve_field_correction(band: str, override: float | None) -> float:
    """The field-incidence correction to subtract, in dB."""
    if override is None:
        return field_incidence_correction(band)
    value = float(override)
    if not np.isfinite(value) or value < 0.0:
        msg = "'field_correction' must be finite and non-negative."
        raise ValueError(msg)
    return value


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
    :ivar plateau_height: Height of the coincidence plateau, in dB, or
        ``None`` (only the plateau model sets these three).
    :ivar plateau_start: Frequency of point A, where the mass-law line meets
        the plateau, in hertz, or ``None``.
    :ivar plateau_end: Frequency of point B, where the 10 dB/octave recovery
        starts, in hertz, or ``None``.
    :ivar critical_frequency_upper: Upper coincidence frequency ``fc2`` of an
        orthotropic panel, in hertz, or ``None``; ``critical_frequency`` then
        carries the lower bound ``fc1`` and the pair spans the flattened
        coincidence range.
    """

    frequencies: np.ndarray
    transmission_loss: np.ndarray
    model: str
    critical_frequency: float | None = None
    resonance_frequency: float | None = None
    mass1: float | None = None
    mass2: float | None = None
    gap: float | None = None
    plateau_height: float | None = None
    plateau_start: float | None = None
    plateau_end: float | None = None
    critical_frequency_upper: float | None = None

    def __post_init__(self) -> None:
        """Reject a predicted spectrum that does not match its band centres.

        :meth:`rating` and :meth:`report` hand ``transmission_loss`` alone to
        the ISO 717-1 curve fit, which recognises the band set by how many
        values arrive: 16 are read as the one-third-octave bands from 100 Hz
        and 5 as the octave bands from 125 Hz, whatever ``frequencies`` says
        the prediction covers. A spectrum longer than its own band centres is
        therefore rated against a reference curve for bands the construction
        was never evaluated on, and the rating fiche prints the standard
        centres beside it, so nothing on the sheet recalls the axis the
        prediction was actually made over.

        :raises ValueError: if the spectrum and the band centres disagree.
        """
        require_ranks(self, frequencies=1, transmission_loss=1)
        require_same_length(self, "frequencies", "transmission_loss")

    @property
    def transmission_coefficient(self) -> np.ndarray:
        r"""Transmission coefficient :math:`\tau = 10^{-R/10}` per band."""
        r = np.asarray(self.transmission_loss, dtype=np.float64)
        return np.asarray(10.0 ** (-r / 10.0), dtype=np.float64)

    def rating(self, bands: str | None = None) -> WeightedRatingResult:
        """Single-number weighted rating ``Rw`` of the predicted ``R(f)``.

        Delegates to :func:`phonometry.building.weighted_rating` (ISO 717-1);
        requires the spectrum to be on the 16 one-third-octave bands (100 Hz to
        3150 Hz) or the 5 octave bands (125 Hz to 2000 Hz).

        :param bands: Band set forwarded to
            :func:`phonometry.building.weighted_rating`.
        :return: The :class:`~phonometry.building.measurement.insulation.WeightedRatingResult`.
        """
        from ..measurement.insulation import weighted_rating

        return weighted_rating(self.transmission_loss, bands)

    def report(self, path: str, **kwargs: Any) -> str:
        """Render the ISO 717-1 Annex C rating fiche of ``R(f)`` to a PDF.

        Convenience wrapper delegating to
        :meth:`~phonometry.building.measurement.insulation.WeightedRatingResult.report`
        on :meth:`rating`; requires the predicted spectrum to be on the 16
        one-third-octave bands (100 Hz to 3150 Hz) or the 5 octave bands
        (125 Hz to 2000 Hz).

        :param path: Destination path of the PDF file.
        :param kwargs: Forwarded to
            :meth:`~phonometry.building.measurement.insulation.WeightedRatingResult.report`
            (e.g. ``engine``).
        :return: The written ``path`` as a :class:`str`.
        """
        return self.rating().report(path, **kwargs)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the predicted sound reduction index ``R(f)``.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_sound_reduction

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
        from ..._i18n import check_language
        from ..._plot.geometry import plot_double_wall_result_geometry

        check_language(language)
        return plot_double_wall_result_geometry(
            self, ax=ax, language=language, **kwargs
        )


def single_panel_transmission_loss(
    frequency: ArrayLike,
    mass_per_area: float,
    *,
    critical_frequency: float | None = None,
    bending_stiffness: float | None = None,
    loss_factor: float = 0.01,
    band: str = "third",
    coincidence_model: str = "sharp",
    field_correction: float | None = None,
    fluid: Fluid = PUBLISHED_AIR,
) -> SoundReductionResult:
    r"""Sound reduction index of a single panel, Sharp's method (Bies 7.2.4.1).

    Field-incidence mass law up to :math:`f_\mathrm{c}/2`, Eq. 7.44 from ``fc``
    upwards, and a straight line in :math:`\log_{10} f` across the
    coincidence region between them.

    With ``coincidence_model="cremer"`` the region above ``fc`` follows Cremer's
    empirical relationship instead (Norton & Karczub Eq. 3.110),

    .. math::

       \mathrm{TL} = \mathrm{TL}_0 + 10 \log_{10}(f/f_\mathrm{c} - 1)
       + 10 \log_{10}\eta - 2~\text{dB}

    which also rises at 10 dB per octave far above coincidence but starts from
    the singularity at ``fc`` itself rather than from a finite value. Norton
    pairs it with the field-incidence mass law below ``fc`` and treats the two
    as the whole model, so there is no interpolated bridge: the mass law runs
    all the way to ``fc``.

    The empirical line is floored at :math:`\mathrm{TL} = 0` dB, which is
    where it lands at :math:`f = f_\mathrm{c}`: Norton's Eq. (3.109) has
    :math:`\theta_{\mathrm{CO}} = 90` degrees there and the
    panel "offers no resistance to incident sound waves", :math:`\tau = 1`.
    It is also the hard bound of a passive panel, so without the floor a band
    centre landing on ``fc`` would report an arbitrarily large negative TL
    and a transmission coefficient above one.

    Provide the coincidence frequency directly through *critical_frequency*, or
    let it be computed from *bending_stiffness* and *mass_per_area* through
    :func:`~phonometry.vibration.structural.radiation_efficiency.coincidence_frequency`.

    :param frequency: Band centre frequencies ``f``, in hertz (array, > 0).
    :param mass_per_area: Mass per unit area ``m''``, in kg/m^2 (> 0).
    :param critical_frequency: Coincidence frequency ``fc``, in hertz (> 0).
    :param bending_stiffness: Bending stiffness per unit width ``B'``, in N.m,
        used to compute ``fc`` when *critical_frequency* is not given.
    :param loss_factor: Total loss factor ``eta`` (> 0, Default: 0.01).
    :param band: Band width for the field correction (``"third"``/``"octave"``).
    :param coincidence_model: ``"sharp"`` (Default, Bies Eq. 7.44 above ``fc``
        with the interpolated bridge from :math:`f_\mathrm{c}/2`) or ``"cremer"``
        (Norton Eq. 3.110, mass law right up to ``fc``).
    :param field_correction: Explicit field-incidence correction of the mass-law
        region, in dB (>= 0), overriding the band table (Default: ``None``;
        Norton's Eq. 3.106 uses a flat 5 dB).
    :param fluid: The medium, a :class:`~phonometry.fluids.Fluid` (Default:
        :data:`PUBLISHED_AIR`, the air these models are published with).
    :return: A :class:`SoundReductionResult` (model ``"sharp-single"`` or
        ``"cremer-single"``).
    :raises ValueError: for a non-positive input, an unknown coincidence model,
        or if neither *critical_frequency* nor *bending_stiffness* is given.
    """
    model = require_choice(coincidence_model, "coincidence_model", ("sharp", "cremer"))
    m2 = require_positive(mass_per_area, "mass_per_area")
    eta = require_positive(loss_factor, "loss_factor")
    c0 = fluid.speed_of_sound
    f = _band_axis(frequency)
    if critical_frequency is not None:
        fc = require_positive(critical_frequency, "critical_frequency")
    elif bending_stiffness is not None:
        fc = coincidence_frequency(m2, bending_stiffness, speed_of_sound=c0)
    else:
        msg = (
            "provide 'critical_frequency' or 'bending_stiffness' to locate the "
            "coincidence frequency."
        )
        raise ValueError(msg)

    def _tl_normal(freq: np.ndarray) -> np.ndarray:
        return mass_law_transmission_loss(
            freq,
            m2,
            incidence="normal",
            fluid=fluid,
        )

    correction = _resolve_field_correction(band, field_correction)
    tl = np.empty_like(f)
    if model == "cremer":
        # Norton Eqs. (3.104) below fc and (3.110) at and above it.
        above = f >= fc
        tl[~above] = _tl_normal(f[~above]) - correction
        # Eq. (3.110) is singular at f = fc, where 10 lg(f/fc - 1) is -inf.
        # Norton's own Eq. (3.109) covers that band: at the critical frequency
        # theta_CO = 90 degrees and "the panel offers no resistance to incident
        # sound waves", i.e. tau = 1 and TL = 0 dB. That is also the hard bound
        # of a passive panel (tau <= 1), so the empirical line is floored there
        # instead of running off to arbitrarily large negative values.
        with np.errstate(divide="ignore"):
            cremer = (
                _tl_normal(f[above])
                + 10.0 * np.log10(f[above] / fc - 1.0)
                + 10.0 * np.log10(eta)
                - 2.0
            )
        tl[above] = np.maximum(cremer, 0.0)
        return SoundReductionResult(
            frequencies=f,
            transmission_loss=np.asarray(tl, dtype=np.float64),
            model="cremer-single",
            critical_frequency=fc,
        )
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


def _resolve_plateau_panel(
    material: str | None,
    thickness_mm: float | None,
    mass_per_area: float | None,
    plateau_height: float | None,
    frequency_ratio: float | None,
) -> tuple[float, float, float]:
    """The plateau construction's three numbers, from the table or explicit.

    Returns ``(mass_per_area, plateau_height, frequency_ratio)``; an explicit
    value always wins over the :data:`PLATEAU_MATERIALS` entry.

    :raises ValueError: for an unknown material or an under-specified panel.
    """
    if material is not None:
        key = require_choice(material, "material", tuple(PLATEAU_MATERIALS))
        density_per_mm, table_height, table_ratio = PLATEAU_MATERIALS[key]
        if mass_per_area is None:
            if thickness_mm is None:
                msg = (
                    "give 'thickness_mm' with 'material', or pass "
                    "'mass_per_area' directly."
                )
                raise ValueError(msg)
            mass_per_area = density_per_mm * require_positive(
                thickness_mm, "thickness_mm"
            )
        plateau_height = table_height if plateau_height is None else plateau_height
        frequency_ratio = table_ratio if frequency_ratio is None else frequency_ratio
    if mass_per_area is None or plateau_height is None or frequency_ratio is None:
        msg = (
            "the plateau construction needs 'mass_per_area', 'plateau_height' "
            "and 'frequency_ratio'; give a tabulated 'material' (with "
            "'thickness_mm') or all three explicitly."
        )
        raise ValueError(msg)
    ratio = require_positive(frequency_ratio, "frequency_ratio")
    if ratio <= 1.0:
        msg = "'frequency_ratio' must be greater than 1."
        raise ValueError(msg)
    return (
        require_positive(mass_per_area, "mass_per_area"),
        require_positive(plateau_height, "plateau_height"),
        ratio,
    )


@overload
def plateau_transmission_loss(
    frequency: ArrayLike,
    *,
    material: str,
    thickness_mm: float,
    plateau_height: float | None = ...,
    frequency_ratio: float | None = ...,
    field_correction: float = ...,
    fluid: Fluid = ...,
) -> SoundReductionResult: ...


@overload
def plateau_transmission_loss(
    frequency: ArrayLike,
    *,
    material: str,
    mass_per_area: float,
    thickness_mm: float | None = ...,
    plateau_height: float | None = ...,
    frequency_ratio: float | None = ...,
    field_correction: float = ...,
    fluid: Fluid = ...,
) -> SoundReductionResult: ...


@overload
def plateau_transmission_loss(
    frequency: ArrayLike,
    *,
    mass_per_area: float,
    plateau_height: float,
    frequency_ratio: float,
    field_correction: float = ...,
    fluid: Fluid = ...,
) -> SoundReductionResult: ...


def plateau_transmission_loss(
    frequency: ArrayLike,
    *,
    material: str | None = None,
    thickness_mm: float | None = None,
    mass_per_area: float | None = None,
    plateau_height: float | None = None,
    frequency_ratio: float | None = None,
    field_correction: float = _NORTON_FIELD_CORRECTION,
    fluid: Fluid = PUBLISHED_AIR,
) -> SoundReductionResult:
    r"""Plateau-method estimate of a single panel's TL (Norton 3.9.1).

    The plateau (Watters) construction is the empirical shortcut practitioners
    draw by hand, and it approximates the whole curve from three numbers per
    material (Norton & Karczub Table 3.1, tabulated in
    :data:`PLATEAU_MATERIALS`):

    1. the **field-incidence mass law**
       :math:`\mathrm{TL} = 10 \log_{10}[1 + (\pi f m''/\rho_0 c_0)^{2}] - 5`
       (Eqs. 3.104/3.106), rising 6 dB per octave;
    2. a horizontal **coincidence plateau** at the material's plateau height;
       point **A** is where the mass-law line reaches it;
    3. point **B** at ``frequency_ratio x fA``, above which the estimate
       recovers at **10 dB per octave**.

    Unlike the physical model of :func:`single_panel_transmission_loss` it
    needs neither the bending stiffness nor the loss factor: the material's
    tabulated plateau absorbs both. The price is that it is only an estimate,
    and it assumes a diffuse field on both sides of a panel whose length and
    width are at least twenty times its thickness.

    Give a tabulated *material* with its *thickness_mm* (the surface density
    then follows from the table), or give *mass_per_area* together with
    *plateau_height* and *frequency_ratio*. An explicit *mass_per_area*,
    *plateau_height* or *frequency_ratio* always overrides the table.

    :param frequency: Band centre frequencies ``f``, in hertz (array, > 0).
    :param material: Key into :data:`PLATEAU_MATERIALS` (Default: ``None``).
    :param thickness_mm: Panel thickness, in **millimetres** (> 0), used with
        *material* to get the surface density.
    :param mass_per_area: Mass per unit area ``m''``, in kg/m^2 (> 0).
    :param plateau_height: Coincidence plateau height, in dB (> 0).
    :param frequency_ratio: Ratio :math:`B/A` locating the 10 dB/octave
        recovery (> 1).
    :param field_correction: Field-incidence correction of the mass-law line,
        in dB (Default: 5.0, Norton Eq. 3.106).
    :param fluid: The medium, a :class:`~phonometry.fluids.Fluid` (Default:
        :data:`PUBLISHED_AIR`, the air these models are published with).
    :return: A :class:`SoundReductionResult` (model ``"plateau"``) carrying
        :attr:`~SoundReductionResult.plateau_height`,
        :attr:`~SoundReductionResult.plateau_start` (point A) and
        :attr:`~SoundReductionResult.plateau_end` (point B).
    :raises ValueError: for a non-positive input, an unknown material, or an
        under-specified panel.
    """
    m2, height, ratio = _resolve_plateau_panel(
        material, thickness_mm, mass_per_area, plateau_height, frequency_ratio
    )
    correction = _resolve_field_correction("third", field_correction)
    c0 = fluid.speed_of_sound
    rho0 = fluid.density
    f = _band_axis(frequency)

    # Point A: invert the field-incidence mass law at the plateau height,
    # 10 lg(1 + (pi f m''/rho0 c0)^2) = height + correction.
    target = 10.0 ** ((height + correction) / 10.0) - 1.0
    if target <= 0.0:
        msg = (
            "the plateau height sits below the mass law at every frequency; "
            "check 'plateau_height' and 'field_correction'."
        )
        raise ValueError(msg)
    f_a = rho0 * c0 * math.sqrt(target) / (math.pi * m2)
    f_b = ratio * f_a

    tl = mass_law_transmission_loss(
        f,
        m2,
        incidence="field",
        field_correction=correction,
        fluid=fluid,
    )
    on_plateau = (f >= f_a) & (f <= f_b)
    above = f > f_b
    tl[on_plateau] = height
    # 10 dB per octave projected upwards from point B.
    tl[above] = height + 10.0 * np.log2(f[above] / f_b)
    return SoundReductionResult(
        frequencies=f,
        transmission_loss=np.asarray(tl, dtype=np.float64),
        model="plateau",
        plateau_height=height,
        plateau_start=float(f_a),
        plateau_end=float(f_b),
    )


# ---------------------------------------------------------------------------
# Orthotropic (ribbed / corrugated) panels
# ---------------------------------------------------------------------------
def corrugated_plate_mass_factor(
    corrugation_amplitude: float, corrugation_wavelength: float
) -> float:
    r"""Surface-density increase of a sine-corrugated plate.

    Corrugating a sheet does not change its thickness, so its mass per unit
    area grows in proportion to the **developed** length of the profile.
    For a sinusoid of amplitude ``H`` and wavelength ``L`` the developed length
    per period, divided by the period, is the closed form

    .. math::

       \frac{m''}{m''_{\mathrm{flat}}} = \frac{2}{\pi} \sqrt{1 + q^{2}}\,
       E\!\left( \frac{q^{2}}{1 + q^{2}} \right),
       \qquad q = \frac{2 \pi H}{L}

    with ``E`` the complete elliptic integral of the second kind. Vigran, in
    the worked example following his Eq. (3.115) (printed p. 96), warns that
    "we have to take into account the fact that the mass per unit area will
    increase when making the corrugations", and it is exactly this factor that
    reproduces his published eigenfrequencies.

    :param corrugation_amplitude: Corrugation amplitude ``H``, in m (> 0); the
        total peak-to-trough depth of the profile is :math:`2H`.
    :param corrugation_wavelength: Corrugation wavelength ``L``, in m (> 0).
    :return: The factor (>= 1) multiplying the flat-sheet surface density.
    :raises ValueError: for a non-positive input.
    """
    amplitude = require_positive(corrugation_amplitude, "corrugation_amplitude")
    wavelength = require_positive(corrugation_wavelength, "corrugation_wavelength")
    q = 2.0 * math.pi * amplitude / wavelength
    q2 = q * q
    return float(2.0 / math.pi * math.sqrt(1.0 + q2) * ellipe(q2 / (1.0 + q2)))


def corrugated_plate_stiffness(
    thickness: float,
    corrugation_amplitude: float,
    corrugation_wavelength: float,
    *,
    youngs_modulus: float,
    poisson_ratio: float = 0.3,
) -> tuple[float, float, float]:
    r"""Equivalent orthotropic stiffnesses of a "wavy" corrugated plate.

    Timoshenko & Woinowsky-Krieger's (1959) equivalent bending stiffnesses of a
    plate of thickness ``h`` whose profile is a sinusoid of amplitude ``H`` and
    wavelength ``L``, as transcribed by Vigran Eq. (3.115) (printed p. 96):

    .. math::

       B_x = \frac{E h^{3}}{12 (1 - \nu^{2})
       \left[ 1 + (\pi H / L)^{2} \right]}

       B_z = \frac{E H^{2} h}{2}
       \left[ 1 - \frac{0.81}{1 + 2.5\,(H/L)^{2}} \right]

       B_{xz} = \frac{E h^{3}}{12 (1 + \nu)}
       \left[ 1 + (\pi H / L)^{2} \right]

    ``Bx`` is the stiffness **across** the corrugations (slightly *below* the
    flat-plate value), ``Bz`` the stiffness **along** them (larger by orders of
    magnitude: that is what corrugating buys) and ``Bxz`` the twisting term
    Eq. (3.113) needs. Vigran's footnote records that the same equations appear
    in Blevins (1979) "unfortunately, with a misprint in the expression for
    ``Bz``".

    Feed ``(Bx, Bz)`` to :func:`orthotropic_critical_frequencies` for the
    coincidence range and all three to :func:`orthotropic_plate_resonance` for
    the eigenfrequencies. Remember to scale the surface density by
    :func:`corrugated_plate_mass_factor`.

    :param thickness: Sheet thickness ``h``, in m (> 0).
    :param corrugation_amplitude: Corrugation amplitude ``H``, in m (> 0).
    :param corrugation_wavelength: Corrugation wavelength ``L``, in m (> 0).
    :param youngs_modulus: Young's modulus ``E``, in Pa (> 0).
    :param poisson_ratio: Poisson's ratio ``nu`` (Default: 0.3).
    :return: The triple ``(Bx, Bz, Bxz)`` in N.m.
    :raises ValueError: for a non-positive input or
        :math:`\lvert\nu\rvert \ge 1`.
    """
    h = require_positive(thickness, "thickness")
    amplitude = require_positive(corrugation_amplitude, "corrugation_amplitude")
    wavelength = require_positive(corrugation_wavelength, "corrugation_wavelength")
    e = require_positive(youngs_modulus, "youngs_modulus")
    if not -1.0 < poisson_ratio < 1.0:
        msg = "'poisson_ratio' must lie in (-1, 1)."
        raise ValueError(msg)
    nu = float(poisson_ratio)
    shape = (math.pi * amplitude / wavelength) ** 2
    flat = e * h**3 / 12.0
    # B_x is the flat-plate bending stiffness divided by the profile factor, so
    # it is `plate_bending_stiffness` rather than a second spelling of it. B_xz
    # below shares only the E h^3 / 12, since its denominator is (1 + nu).
    b_x = plate_bending_stiffness(e, h, nu) / (1.0 + shape)
    b_z = (
        0.5
        * e
        * amplitude**2
        * h
        * (1.0 - 0.81 / (1.0 + 2.5 * (amplitude / wavelength) ** 2))
    )
    b_xz = flat / (1.0 + nu) * (1.0 + shape)
    return float(b_x), float(b_z), float(b_xz)


def orthotropic_plate_resonance(
    mode_x: int,
    mode_z: int,
    *,
    length_x: float,
    length_z: float,
    mass_per_area: float,
    bending_stiffness_x: float,
    bending_stiffness_z: float,
    bending_stiffness_xz: float,
) -> float:
    r"""Eigenfrequency of a simply supported orthotropic plate (Vigran 3.113).

    .. math::

       f_{i,n} = \frac{\pi}{2 \sqrt{m''}}
       \sqrt{ \frac{i^{4} B_x}{a^{4}} + \frac{n^{4} B_z}{b^{4}}
       + \frac{2 i^{2} n^{2} B_{xz}}{a^{2} b^{2}} }

    (Vigran Eq. (3.113), printed p. 95; identical to Bies Eq. (7.27) after
    Hearmon 1959). It collapses to the isotropic Eq. (3.109) when
    :math:`B_x = B_z = B` and :math:`B_{xz} = B\,(\nu + 2 (1 - \nu)/2) = B`.

    The lowest eigenfrequency :math:`f_{1,1}` matters to the transmission-loss
    prediction because the infinite-panel models of
    :func:`orthotropic_transmission_loss` and
    :func:`single_panel_transmission_loss` are only valid above about
    :math:`1.5 f_{1,1}` (Bies, Sect. 7.2.4).

    :param mode_x: Mode order ``i`` along ``a`` (integer >= 1).
    :param mode_z: Mode order ``n`` along ``b`` (integer >= 1).
    :param length_x: Plate dimension ``a``, in m (> 0), along the axis whose
        bending stiffness is *bending_stiffness_x*.
    :param length_z: Plate dimension ``b``, in m (> 0).
    :param mass_per_area: Mass per unit area ``m''``, in kg/m^2 (> 0).
    :param bending_stiffness_x: ``Bx``, in N.m (> 0).
    :param bending_stiffness_z: ``Bz``, in N.m (> 0).
    :param bending_stiffness_xz: ``Bxz``, in N.m (> 0).
    :return: The eigenfrequency, in hertz.
    :raises ValueError: for a non-positive input or a mode order below 1.
    """
    i = int(mode_x)
    n = int(mode_z)
    if i < 1 or n < 1:
        msg = "'mode_x' and 'mode_z' must be integers >= 1."
        raise ValueError(msg)
    a = require_positive(length_x, "length_x")
    b = require_positive(length_z, "length_z")
    m2 = require_positive(mass_per_area, "mass_per_area")
    b_x = require_positive(bending_stiffness_x, "bending_stiffness_x")
    b_z = require_positive(bending_stiffness_z, "bending_stiffness_z")
    b_xz = require_positive(bending_stiffness_xz, "bending_stiffness_xz")
    total = (
        i**4 * b_x / a**4 + n**4 * b_z / b**4 + 2.0 * i**2 * n**2 * b_xz / (a**2 * b**2)
    )
    return float(math.pi / (2.0 * math.sqrt(m2)) * math.sqrt(total))


def orthotropic_critical_frequencies(
    mass_per_area: float,
    bending_stiffness_1: float,
    bending_stiffness_2: float,
    *,
    speed_of_sound: float = _SPEED_OF_SOUND,
) -> tuple[float, float]:
    r"""Coincidence range ``(fc1, fc2)`` of orthotropic panels (Vigran 6.107).

    :math:`f_\mathrm{c} = \frac{c_0^{2}}{2 \pi} \sqrt{m'' / B}` evaluated for both
    principal bending stiffnesses (Vigran Eq. (6.107), printed p. 252; the
    same closed form as the isotropic
    :func:`~phonometry.vibration.structural.radiation_efficiency.coincidence_frequency`).
    The stiffest direction gives the **lowest** coincidence frequency, so the
    returned pair is sorted: ``fc1`` from the larger stiffness, ``fc2`` from the
    smaller. For a corrugated sheet ``fc1`` can sit at a few hundred hertz while
    ``fc2`` reaches 15 kHz to 30 kHz, and the resonant transmission then
    dominates over most of the useful frequency range.

    :param mass_per_area: Mass per unit area ``m''``, in kg/m^2 (> 0),
        including the developed-length increase of a corrugated sheet (see
        :func:`corrugated_plate_mass_factor`).
    :param bending_stiffness_1: One principal bending stiffness, in N.m (> 0).
    :param bending_stiffness_2: The other principal bending stiffness, in N.m
        (> 0). The argument order does not matter.
    :param speed_of_sound: Speed of sound in air ``c0`` (Default: 343 m/s).
    :return: The pair ``(fc1, fc2)`` in hertz, with
        :math:`f_{\mathrm{c}1} \le f_{\mathrm{c}2}`.
    :raises ValueError: for a non-positive input.
    """
    m2 = require_positive(mass_per_area, "mass_per_area")
    b1 = require_positive(bending_stiffness_1, "bending_stiffness_1")
    b2 = require_positive(bending_stiffness_2, "bending_stiffness_2")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    scale = c0**2 / (2.0 * math.pi) * math.sqrt(m2)
    pair = (scale / math.sqrt(b1), scale / math.sqrt(b2))
    return (min(pair), max(pair))


def _limiting_sin_squared(
    area: float | None, limiting_angle: float, wavelength: float
) -> float:
    r"""Upper limit :math:`\sin^{2}\theta_\mathrm{L}` of the diffuse-field integral.

    With an *area* the Davy (2009) limit of Bies Eq. (7.36) applies
    (:math:`\cos^{2}\theta_\mathrm{L} = \min(\lambda / (2 \pi \sqrt{A}), 0.9)`, the
    finite-size correction Vigran writes as Eq. (6.113)); otherwise the fixed
    *limiting_angle* is used (Sharp's 78 degrees, the value Vigran also fixes
    at :math:`\sin^{2}\theta_\mathrm{L} = 0.96`).
    """
    if area is None:
        return float(math.sin(math.radians(limiting_angle)) ** 2)
    cos2 = min(wavelength / (2.0 * math.pi * math.sqrt(area)), 0.9)
    return float(1.0 - cos2)


def _orthotropic_tau(
    frequency: float,
    mass_per_area: float,
    fc1: float,
    fc2: float,
    eta: float,
    z0: float,
    upper: float,
) -> float:
    r"""Diffuse-field ``tau`` of an infinite orthotropic panel (Vigran 6.111).

    The double integral of Vigran Eq. (6.111) / Bies Eq. (7.38) over the
    azimuth ``theta`` and :math:`x = \sin^{2}\phi`, with the wall impedance of
    Vigran Eq. (6.108) / Bies Eq. (7.30) and the angular transmission
    coefficient of Eq. (6.109) / Eq. (7.31).

    Above the coincidence range and at a low loss factor the inner integrand
    is a narrow spike at the angle where the bracket of Eq. (6.108) vanishes,
    :math:`x = 1/\sqrt{\mathrm{ratio}}`, whose width is set by ``eta``. That
    location is
    known in closed form, so it is handed to the quadrature as a break point
    (``points``, i.e. QUADPACK's QAGP) instead of being left for the adaptive
    subdivision to stumble on.
    """
    omega_m = 2.0 * math.pi * frequency * mass_per_area

    def azimuth(theta: float) -> float:
        # Vigran Eq. (6.108): the two critical frequencies enter through the
        # azimuth-weighted ratio, squared.
        ratio = (
            frequency / fc1 * math.cos(theta) ** 2
            + frequency / fc2 * math.sin(theta) ** 2
        ) ** 2

        def integrand(x: float) -> float:
            z_w = 1j * omega_m * (1.0 - ratio * (1.0 + 1j * eta) * x * x)
            return 1.0 / abs(1.0 + z_w * math.sqrt(1.0 - x) / (2.0 * z0)) ** 2

        resonance = 1.0 / math.sqrt(ratio) if ratio > 0.0 else math.inf
        if 0.0 < resonance < upper:
            return float(quad(integrand, 0.0, upper, points=(resonance,), limit=200)[0])
        return float(quad(integrand, 0.0, upper, limit=200)[0])

    return float(2.0 / math.pi * quad(azimuth, 0.0, 0.5 * math.pi, limit=200)[0])


def _heckl_transmission_loss(
    f: np.ndarray,
    m2: float,
    fc1: float,
    fc2: float,
    *,
    fluid: Fluid,
    correction: float,
) -> np.ndarray:
    r"""Heckl's (1960) piecewise orthotropic estimate (Bies 7.2.4.5).

    Field-incidence mass law below :math:`f_{\mathrm{c}1}/2`, Bies Eq. (7.59) (= the
    first of Vigran Eq. (6.112)) from ``fc1`` to :math:`f_{\mathrm{c}2}/2`, Bies
    Eq. (7.60) (= the second) above :math:`2 f_{\mathrm{c}2}`, and straight lines in
    :math:`\log_{10} f` across the two
    gaps, as Bies Figure 7.9(b) draws them.
    """
    z0 = fluid.density * fluid.speed_of_sound

    def mass_law(freq: np.ndarray) -> np.ndarray:
        return (
            mass_law_transmission_loss(freq, m2, incidence="normal", fluid=fluid)
            - correction
        )

    def coincidence(freq: np.ndarray) -> np.ndarray:
        """Bies Eq. (7.59): the flattened plateau of the coincidence range."""
        tau = z0 / (2.0 * np.pi**2 * m2) * fc1 / freq**2 * np.log(4.0 * freq / fc1) ** 2
        return np.asarray(-10.0 * np.log10(tau), dtype=np.float64)

    def recovered(freq: np.ndarray) -> np.ndarray:
        """Bies Eq. (7.60): 6 dB per octave again above the whole range."""
        tau = z0 / (2.0 * m2) * math.sqrt(fc1 * fc2) / freq**2
        return np.asarray(-10.0 * np.log10(tau), dtype=np.float64)

    # The four construction points of Bies Figure 7.9(b), in order.
    knots = np.array([0.5 * fc1, fc1, 0.5 * fc2, 2.0 * fc2])
    anchors = np.array(
        [
            float(mass_law(knots[:1])[0]),
            float(coincidence(knots[1:2])[0]),
            float(coincidence(knots[2:3])[0]),
            float(recovered(knots[3:4])[0]),
        ]
    )
    tl = np.empty_like(f)
    plateau = (f >= knots[1]) & (f <= knots[2])
    below = f < knots[0]
    above = f > knots[3]
    tl[below] = mass_law(f[below])
    tl[plateau] = coincidence(f[plateau])
    tl[above] = recovered(f[above])
    for mask, lo, hi in (
        (~below & (f < knots[1]), 0, 1),
        (~above & (f > knots[2]), 2, 3),
    ):
        if np.any(mask):
            frac = (np.log10(f[mask]) - np.log10(knots[lo])) / (
                np.log10(knots[hi]) - np.log10(knots[lo])
            )
            tl[mask] = anchors[lo] + frac * (anchors[hi] - anchors[lo])
    return np.asarray(tl, dtype=np.float64)


def orthotropic_transmission_loss(
    frequency: ArrayLike,
    mass_per_area: float,
    *,
    critical_frequency_lower: float,
    critical_frequency_upper: float,
    loss_factor: float = 0.01,
    method: str = "integral",
    area: float | None = None,
    limiting_angle: float = 78.0,
    band: str = "third",
    fluid: Fluid = PUBLISHED_AIR,
) -> SoundReductionResult:
    r"""Orthotropic-panel sound reduction index (Vigran 6.5.3, Bies 7.2.4.5).

    A ribbed or corrugated sheet is stiff along the corrugations and limp across
    them, so instead of one coincidence dip it has a whole **coincidence range**
    ``fc1`` to ``fc2`` (see :func:`orthotropic_critical_frequencies`). Over that
    range the resonant transmission dominates and ``R`` flattens well below the
    mass law of a flat plate of the same surface density, which is the price
    paid for the strength-to-weight ratio.

    Two prediction routes, both from the same wall impedance (Heckl 1960;
    Hansen 1993; Vigran Eq. (6.108) = Bies Eq. (7.30))

    .. math::

       Z_\mathrm{w} = j \omega m'' \left[ 1
       - \left( (f/f_{\mathrm{c}1}) \cos^{2}\theta
       + (f/f_{\mathrm{c}2}) \sin^{2}\theta \right)^{2}
       (1 + j \eta) \sin^{4}\phi \right]

    * ``method="integral"`` (Default) averages the angular transmission
      coefficient
      :math:`\tau = \lvert 1 + Z_\mathrm{w} \cos\phi / (2 \rho_0 c_0) \rvert^{-2}`
      (Vigran Eq. (6.109) = Bies Eq. (7.31)) over azimuth and incidence
      angle, :math:`\tau_\mathrm{F} = \frac{2}{\pi} \int_0^{\pi/2}
      \int_0^{\sin^{2}\theta_\mathrm{L}} \tau \,d(\sin^{2}\phi)\, d\theta`
      (Vigran Eq. (6.111) = Bies Eq. (7.38)), numerically. The
      near-grazing angles are excluded by the limiting angle: pass *area* for
      the size-dependent limit of Bies Eq. (7.36) (the correction Vigran writes
      as Eq. (6.113)) or leave it out for the fixed *limiting_angle*. This is
      the only route that responds to the loss factor.
    * ``method="heckl"`` is Heckl's closed-form approximation for
      :math:`\eta = 0`, the design chart of Bies Figure 7.9(b):
      field-incidence mass law below :math:`f_{\mathrm{c}1}/2`, Eq. (7.59) (the first
      of Vigran Eq. (6.112)) from ``fc1`` to :math:`f_{\mathrm{c}2}/2`, Eq. (7.60)
      (the second) above :math:`2 f_{\mathrm{c}2}`, and straight lines in
      :math:`\log_{10} f` across the two gaps. It is cheap and it needs no
      loss factor, but it cannot show the depth of the coincidence region and
      it requires :math:`f_{\mathrm{c}2} > 4 f_{\mathrm{c}1}` for its four construction points
      to stay ordered.

    The two routes are not interchangeable. Above :math:`2 f_{\mathrm{c}2}` they
    converge as the loss factor falls: with :math:`\eta \to 0` the integral
    lands within about 0.3 dB of Eq. (7.60), which is a useful independent
    check on both
    transcriptions. Across the coincidence range Eq. (7.59) is a much rougher
    approximation and stays a few decibels above the integral even at
    :math:`\eta \to 0`, as Vigran's Figure 6.27 shows for its own worked case.

    Both models are infinite-panel models, valid above roughly
    :math:`1.5 f_{1,1}` (:func:`orthotropic_plate_resonance`). Bies also
    notes two systematic departures of the Heckl branch from measurement:
    below about :math:`0.7 f_{\mathrm{c}1}` it underestimates ``R`` on small panels,
    and real corrugated panels show a dip of up to 5 dB between 2 kHz and
    4 kHz caused by resonances of the panel sections between the ribs, which
    no smooth model predicts.

    :param frequency: Band centre frequencies ``f``, in hertz (array, > 0).
    :param mass_per_area: Mass per unit area ``m''``, in kg/m^2 (> 0).
    :param critical_frequency_lower: Lower coincidence frequency ``fc1``, in
        hertz (> 0), from the stiffest direction.
    :param critical_frequency_upper: Upper coincidence frequency ``fc2``, in
        hertz (> ``fc1``).
    :param loss_factor: Total loss factor ``eta`` (> 0, Default: 0.01); used
        only by ``method="integral"``, but validated on both routes.
    :param method: ``"integral"`` (Default) or ``"heckl"``.
    :param area: Panel area ``S``, in m^2 (> 0), selecting the size-dependent
        limiting angle of Bies Eq. (7.36) (Default: ``None``); used only by
        ``method="integral"``, but validated on both routes.
    :param limiting_angle: Fixed limiting angle ``theta_L``, in degrees
        (:math:`0 < \theta_\mathrm{L} < 90`, Default: 78.0), used when *area* is
        ``None`` and only by ``method="integral"``, but validated on both
        routes.
    :param band: Band width for the field correction of the Heckl mass-law
        branch (``"third"``/``"octave"``).
    :param fluid: The medium, a :class:`~phonometry.fluids.Fluid` (Default:
        :data:`PUBLISHED_AIR`, the air these models are published with).
    :return: A :class:`SoundReductionResult` (model ``"orthotropic-integral"``
        or ``"orthotropic-heckl"``) carrying ``fc1`` in
        :attr:`~SoundReductionResult.critical_frequency` and ``fc2`` in
        :attr:`~SoundReductionResult.critical_frequency_upper`.
    :raises ValueError: for a non-positive input, an unknown method, a
        coincidence range that is not increasing, or a Heckl construction whose
        points would be out of order.
    """
    chosen = require_choice(method, "method", ("integral", "heckl"))
    m2 = require_positive(mass_per_area, "mass_per_area")
    fc1 = require_positive(critical_frequency_lower, "critical_frequency_lower")
    fc2 = require_positive(critical_frequency_upper, "critical_frequency_upper")
    if fc2 <= fc1:
        msg = "'critical_frequency_upper' must exceed 'critical_frequency_lower'."
        raise ValueError(msg)
    eta = require_positive(loss_factor, "loss_factor")
    c0 = fluid.speed_of_sound
    rho0 = fluid.density
    # Validated on both routes, so that an out-of-range argument is rejected
    # whichever method it was passed with, even where the method ignores it.
    if area is not None:
        require_positive(area, "area")
    elif not 0.0 < limiting_angle < _GRAZING_INCIDENCE_DEG:
        msg = "'limiting_angle' must lie in (0, 90) degrees."
        raise ValueError(msg)
    f = _band_axis(frequency)
    z0 = rho0 * c0

    if chosen == "heckl":
        if fc2 <= 4.0 * fc1:
            msg = (
                "the Heckl construction needs 'critical_frequency_upper' above "
                "four times 'critical_frequency_lower' so its points stay "
                "ordered; use method='integral' for a narrow coincidence range."
            )
            raise ValueError(msg)
        tl = _heckl_transmission_loss(
            f,
            m2,
            fc1,
            fc2,
            fluid=fluid,
            correction=field_incidence_correction(band),
        )
        return SoundReductionResult(
            frequencies=f,
            transmission_loss=tl,
            model="orthotropic-heckl",
            critical_frequency=fc1,
            critical_frequency_upper=fc2,
        )

    tau = np.array(
        [
            _orthotropic_tau(
                float(freq),
                m2,
                fc1,
                fc2,
                eta,
                z0,
                _limiting_sin_squared(area, limiting_angle, c0 / float(freq)),
            )
            for freq in f
        ],
        dtype=np.float64,
    )
    return SoundReductionResult(
        frequencies=f,
        transmission_loss=np.asarray(-10.0 * np.log10(tau), dtype=np.float64),
        model="orthotropic-integral",
        critical_frequency=fc1,
        critical_frequency_upper=fc2,
    )


def mass_spring_mass_resonance(
    mass1: float,
    mass2: float,
    gap: float,
    *,
    cavity_medium: PorousMediumResult | None = None,
    tie_stiffness_per_area: float = 0.0,
    fluid: Fluid = PUBLISHED_AIR,
) -> float:
    r"""Mass-spring-mass resonance ``f0`` of a double wall (Bies Eq. 7.62).

    :math:`f_0 = \frac{1}{2 \pi} \sqrt{s'' (m_1 + m_2)/(m_1 m_2)}` with the
    cavity stiffness per unit area ``s''``. For an empty air gap
    :math:`s'' = \rho_0 c_0^{2} / d` (adiabatic,
    Hopkins Eq. 4.72); with a porous *cavity_medium* the fill's effective
    (near-isothermal) bulk modulus at the lowest supplied frequency sets a
    softer :math:`s'' = \operatorname{Re}(K_\mathrm{e}) / d`, lowering ``f0``.

    An array of mechanical connections across the cavity (wall ties in a
    masonry cavity wall, resilient mounts under a floating floor) acts as a
    spring **in parallel** with the cavity, adding :math:`N k / S` to ``s''``
    (Hopkins Eq. 4.89). Pass that term as *tie_stiffness_per_area*; the helper
    :func:`phonometry.building.wall_tie_stiffness_per_area` builds it from a
    tie density and Hopkins' Table A4.

    :param mass1: Surface density of leaf 1 ``m1``, in kg/m^2 (> 0).
    :param mass2: Surface density of leaf 2 ``m2``, in kg/m^2 (> 0).
    :param gap: Cavity depth ``d``, in m (> 0).
    :param cavity_medium: Optional porous fill (a
        :class:`~phonometry.materials.PorousMediumResult`) whose effective bulk
        modulus sets the cavity stiffness.
    :param tie_stiffness_per_area: Stiffness per unit area :math:`N k / S` of a
        connection array bridging the cavity, in N/m^3 (>= 0, Default: 0).
    :param fluid: The medium, a :class:`~phonometry.fluids.Fluid` (Default:
        :data:`PUBLISHED_AIR`, the air these models are published with).
    :return: The mass-spring-mass resonance ``f0``, in hertz.
    :raises ValueError: for a non-positive input.
    """
    m1 = require_positive(mass1, "mass1")
    m2 = require_positive(mass2, "mass2")
    d = require_positive(gap, "gap")
    ties = require_non_negative(tie_stiffness_per_area, "tie_stiffness_per_area")
    c0 = fluid.speed_of_sound
    rho0 = fluid.density
    if cavity_medium is None:
        stiffness = rho0 * c0**2 / d
    else:
        bulk = np.atleast_1d(
            np.asarray(cavity_medium.bulk_modulus, dtype=np.complex128)
        )
        stiffness = float(np.real(bulk.flat[0])) / d
        if stiffness <= 0.0:
            msg = "'cavity_medium' bulk modulus must be positive."
            raise ValueError(msg)
    reduced = (m1 + m2) / (m1 * m2)
    return float(np.sqrt((stiffness + ties) * reduced) / (2.0 * np.pi))


def double_wall_transmission_loss(
    frequency: ArrayLike,
    mass1: float,
    mass2: float,
    gap: float,
    *,
    loss_factor: float = 0.1,
    cavity_medium: PorousMediumResult | None = None,
    tie_stiffness_per_area: float = 0.0,
    band: str = "third",
    fluid: Fluid = PUBLISHED_AIR,
) -> SoundReductionResult:
    r"""Sound reduction index of a double wall (Bies 7.2.6, Eq. 7.64).

    Piecewise Sharp model: below the mass-spring-mass resonance ``f0`` the pair
    behaves as the mass law of the combined mass; between ``f0`` and the
    limiting frequency :math:`f_\mathrm{l} = c_0/(2 \pi d)` the two mass laws add plus
    :math:`20 \log_{10}(2 k d)`; above ``f_l`` they add plus 6 dB. The curve is
    continuous at ``f_l`` (:math:`20 \log_{10}(2 k d) = 6` there).

    Ties or mounts bridging the cavity stiffen it (Hopkins Eq. 4.89), pushing
    ``f0`` up and extending the combined-mass branch; pass their stiffness per
    unit area as *tie_stiffness_per_area* (see
    :func:`phonometry.building.wall_tie_stiffness_per_area`).

    :param frequency: Band centre frequencies ``f``, in hertz (array, > 0).
    :param mass1: Surface density of leaf 1 ``m1``, in kg/m^2 (> 0).
    :param mass2: Surface density of leaf 2 ``m2``, in kg/m^2 (> 0).
    :param gap: Cavity depth ``d``, in m (> 0).
    :param loss_factor: Leaf loss factor ``eta`` (> 0, Default: 0.1); reserved
        for the coincidence extension and reported for reference.
    :param cavity_medium: Optional porous fill; see
        :func:`mass_spring_mass_resonance`.
    :param tie_stiffness_per_area: Stiffness per unit area :math:`N k / S` of a
        connection array bridging the cavity, in N/m^3 (>= 0, Default: 0).
    :param band: Band width for the field correction (``"third"``/``"octave"``).
    :param fluid: The medium, a :class:`~phonometry.fluids.Fluid` (Default:
        :data:`PUBLISHED_AIR`, the air these models are published with).
    :return: A :class:`SoundReductionResult` (model ``"double-wall"``).
    :raises ValueError: for a non-positive input.
    """
    m1 = require_positive(mass1, "mass1")
    m2 = require_positive(mass2, "mass2")
    d = require_positive(gap, "gap")
    require_positive(loss_factor, "loss_factor")
    c0 = fluid.speed_of_sound
    f = _band_axis(frequency)

    f0 = mass_spring_mass_resonance(
        m1,
        m2,
        d,
        cavity_medium=cavity_medium,
        tie_stiffness_per_area=tie_stiffness_per_area,
        fluid=fluid,
    )
    f_l = c0 / (2.0 * np.pi * d)

    def _ml(freq: np.ndarray, mass: float) -> np.ndarray:
        return mass_law_transmission_loss(
            freq,
            mass,
            incidence="field",
            band=band,
            fluid=fluid,
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
