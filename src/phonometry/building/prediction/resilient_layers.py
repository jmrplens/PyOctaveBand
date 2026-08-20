#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Prediction of resilient-layer performance: tapping force, floor coverings,
floating floors.

The measurement modules of this domain report what a resilient layer *achieved*
(:mod:`phonometry.building.measurement.floor_covering_improvement` for the ISO 16251-1
mock-up, :mod:`phonometry.materials.resilient.dynamic_stiffness` for the EN 29052-1
dynamic stiffness). This module is their **prediction** counterpart: it walks
the physical chain from the material data to the improvement spectrum, so a
covering or a floating floor can be sized before anything is built.

The chain is one story told in three steps.

**1. The excitation (Hopkins 3.6.3).** The ISO tapping machine drops a 0.5 kg
hammer from 40 mm, ten impacts per second, so the impact velocity is
:math:`v_o = \sqrt{2 g h} = 0.886` m/s (Eq. 3.85) and, for a short impact, the
peak force per Fourier line is :math:`|F_n| = 2 m v_o/T_i` (Eq. 3.90), giving
the band mean-square force :math:`F_\mathrm{rms}^{2} = 3.9 B` (Eq. 3.92). Real floors
are not that simple: the hammer, the contact stiffness ``K`` it deforms and the
floor's driving-point impedance ``Zdp`` form a mass-spring-dashpot (Fig. 3.28)
whose force pulse (:func:`force_pulse`, Eqs. 3.95/3.96) is **over-critical**
when :math:`K m \ge 4 Z_\mathrm{dp}^{2}` (a single positive pulse, no rebound) and
**under-critical** otherwise (a rebound; only the first positive lobe is
transformed). Its spectrum (:func:`tapping_force_spectrum`) is flat up to the
cut-off ``fco`` (Eqs. 3.101/3.102) and falls above it, and it asymptotes at low
frequency between :math:`|F_n|_{\text{lower}} = m v_o/T_i` and
:math:`|F_n|_{\text{upper}} = 2 m v_o/T_i`, 6 dB apart in mean square
(Eqs. 3.99/3.100).

**2. Soft floor coverings (Hopkins 4.4.3.1).** A soft covering on a heavyweight
floor changes nothing but the force input, so its improvement is the force
ratio :math:`\Delta L = 20 \log_{10}(|F_n|_{\text{without}}/|F_n|_{\text{with}})`
(Eq. 4.114). The covering's contact stiffness :math:`K = E \pi r^{2}/d`
(Eq. 3.98) sets its cut-off, against the bare plate's
:math:`K = 2 r E/(1 - \nu^{2})` (Eq. 3.97), which is why a two-line estimate,
:math:`\Delta L \approx 0` below ``fco`` and 12 dB/octave above it, captures
the whole design question (:func:`covering_improvement`).

**3. Floating floors (Hopkins 4.4.4, ISO 12354-2 Annex C, Vigran 8.4).** Above
the mass-spring resonance :math:`f_\mathrm{o} = 160 \sqrt{s'/m'}` (Formula C.2) the
improvement follows one of three laws
(:func:`floating_floor_improvement_spectrum`): the infinite-plate result of
Cremer, :math:`\Delta L = 40 \log_{10}(f/f_\mathrm{o})` (Eq. 4.119, Vigran Eq. 8.40), the
empirical :math:`\Delta L = 30 \log_{10}(f/f_\mathrm{o})` that EN 12354-2 adopted for
sand-cement screeds (Formula C.1, Eq. 4.124), and the same 40 lg law with the
hammer-impedance term :math:`10 \log_{10}[1 + (f/f_\mathrm{limit})^{2}]` that a lightweight
walking surface needs (Eq. 4.123, Vigran Eq. 8.48). A floating floor on
discrete mounts instead of a continuous layer is a two-subsystem SEA problem
(:func:`resilient_mount_improvement`, Vér's model as Hopkins Eq. 4.118 and
Vigran Eq. 8.45) and rises at 30 dB/decade, not 40. Two floating floors stacked
give two resonances (:func:`double_floating_floor_resonances`, Eq. 4.125), and
the weighted single number follows from ``m'`` and ``s'`` directly
(:func:`weighted_floating_floor_improvement`, Formulae C.4/C.5).

The wall and ceiling linings of ISO 12354-1 Annex D, which used to close this
file, are in :mod:`phonometry.building.prediction.linings`: they are added to a
wall rather than laid on a floor, and the annex rates them from a resonance
frequency instead of from the tapping machine's force.

Citations are to ISO 12354-2:2017, to Hopkins, *Sound Insulation* (2007) and to
Vigran, *Building Acoustics* (2008). Where the two books state the same model in
different algebra the test suite pins the identity rather than either
transcription. One printed defect is relevant here and is recorded in
``docs/ERRATA.md``: the carpet stiffness in the caption of Vigran's Fig. 8.37.

One relation used here carries no published worked example, so it is
implemented as printed and checked only for self-consistency: the asphalt fit
of Formula (C.5). The guide "Predicting Resilient-Layer Performance" says which
pieces have an oracle and which do not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from ..._internal.validation import (
    require_choice,
    require_positive,
    require_positive_array,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

# The ISO 12354-1 Annex D lining machinery now lives in :mod:`.linings`. Until
# every caller in the tree reads it from there, the names this module used to
# define stay importable from this path, unchanged; the ``__all__`` below
# re-exports them.
from .linings import (
    LiningImprovementResult,
    lining_improvement,
    lining_improvement_in_situ,
    lining_resonance_frequency,
    weighted_lining_improvement,
)

# The Annex D names above are listed here so they keep resolving through this
# module; everything else is defined below.
__all__ = [
    "TAPPING_DROP_HEIGHT",
    "TAPPING_HAMMER_MASS",
    "TAPPING_HAMMER_RADIUS",
    "TAPPING_IMPACT_RATE",
    "CoveringImprovementResult",
    "FloatingFloorImprovementResult",
    "LiningImprovementResult",
    "TappingForceResult",
    "combined_dynamic_stiffness",
    "covering_contact_stiffness",
    "covering_improvement",
    "double_floating_floor_resonances",
    "floating_floor_improvement_spectrum",
    "floating_floor_resonance_frequency",
    "force_pulse",
    "hammer_impact_velocity",
    "hammer_limiting_frequency",
    "lining_improvement",
    "lining_improvement_in_situ",
    "lining_resonance_frequency",
    "plate_contact_stiffness",
    "resilient_mount_improvement",
    "short_pulse_mean_square_force",
    "tapping_cut_off_frequency",
    "tapping_force_spectrum",
    "weighted_floating_floor_improvement",
    "weighted_lining_improvement",
]

#: Mass ``m`` of one ISO tapping-machine hammer, in kg (ISO 10140-5; Hopkins
#: 3.6.3: "a free-falling mass of 0,5 kg").
TAPPING_HAMMER_MASS: float = 0.5

#: Drop height ``h`` of the hammer, in m (Hopkins 3.6.3: "a drop-height of
#: 0,04 m").
TAPPING_DROP_HEIGHT: float = 0.04

#: Impact repetition rate ``fi`` of the machine, in Hz: five hammers giving
#: ten impacts per second, so ``Ti = 1/fi = 0,1 s`` (Hopkins 3.6.3).
TAPPING_IMPACT_RATE: float = 10.0

#: Radius ``r`` of the circular contact area assumed for the hammer, in m
#: (Hopkins 3.6.3.1: "The hammer radius, 15 mm, can be used as an estimate").
#: Vigran Eq. (8.51) uses the equivalent hammer area ``Sh = 7 cm²``.
TAPPING_HAMMER_RADIUS: float = 0.015

#: Standard acceleration of free fall ``g``, in m/s²; reproduces the printed
#: impact velocity ``vo = 0,886 m/s`` of Hopkins 3.6.3.1.
_GRAVITY: float = 9.81

#: Band-width factors ``B`` of Hopkins Eq. (3.91): ``B = 0,23 f`` for
#: one-third-octave bands and ``B = 0,707 f`` for octave bands.
_BANDWIDTH_FACTOR: dict[str, float] = {"third": 0.23, "octave": 0.707}

#: Half-band ratios ``fupper/fcentre`` of the base-ten band system, used to
#: collect the tapping machine's Fourier lines into each band.
_HALF_BAND_RATIO: dict[str, float] = {"third": 10.0**0.05, "octave": 10.0**0.15}

#: Coefficient of the short-duration mean-square force ``F²rms = 3,9 B``
#: (Hopkins Eq. 3.92); the exact value from Eqs. (3.90)/(3.91) is 3,925.
_SHORT_PULSE_COEFFICIENT: float = 3.9

#: Reference power of the power-input level, in W.
_POWER_REFERENCE: float = 1e-12

#: Constant of ISO 12354-2:2017 Formula (C.2), ``fo = 160 √(s'/m')`` with ``s'``
#: in MN/m³. The exact mass-spring value is ``1000/(2 π) = 159,15``; the
#: standard's rounded 160 is kept because its own Annex G example prints
#: ``fo = 52,8 Hz`` for ``s' = 8 MN/m³`` and ``m' = 73,5 kg/m²``.
_ISO_RESONANCE_CONSTANT: float = 160.0

#: Slopes of the floating-floor improvement laws, in dB per decade.
_SLOPE_EN12354: float = 30.0
_SLOPE_CREMER: float = 40.0

#: ISO 12354-2:2017 Formula (C.4), screeds: ``13 lg(m') − 14,2 lg(s') + 20,8``.
_DELTA_LW_SCREED = (13.0, -14.2, 20.8)

#: ISO 12354-2:2017 Formula (C.5), asphalt / dry floors:
#: ``(−0,21 m' − 5,45) lg(s') + 0,46 m' + 23,8``.
_DELTA_LW_ASPHALT = (-0.21, -5.45, 0.46, 23.8)

#: Band width of the mean-square force (Hopkins Eq. 3.91).
BandWidth = Literal["third", "octave"]
#: Floating-floor construction of ISO 12354-2:2017 Annex C.
FloorType = Literal["screed", "asphalt"]
#: Improvement law of a floating floor above its resonance.
FloatingFloorModel = Literal["en12354", "cremer", "cremer_hammer"]


def _band_factor(band: BandWidth) -> float:
    r"""Band-width factor :math:`B/f` of Hopkins Eq. (3.91)."""
    require_choice(band, "band", ("third", "octave"))
    return _BANDWIDTH_FACTOR[band]


# --------------------------------------------------------------------------- #
# 1. The ISO tapping machine as a mass-spring-dashpot (Hopkins 3.6.3)
# --------------------------------------------------------------------------- #
def hammer_impact_velocity(
    drop_height: float = TAPPING_DROP_HEIGHT, *, gravity: float = _GRAVITY
) -> float:
    r"""Hammer velocity at impact :math:`v_o = \sqrt{2 g h}` (Hopkins
    Eq. 3.85).

    The ISO tapping machine's nominal 40 mm drop gives
    :math:`v_o = 0.886` m/s.

    :param drop_height: Drop height ``h``, in m (Default: 0,04).
    :param gravity: Acceleration of free fall ``g``, in m/s² (Default: 9,81).
    :return: The impact velocity ``vo``, in m/s.
    :raises ValueError: If an input is not positive and finite.
    """
    h = require_positive(drop_height, "drop_height")
    g = require_positive(gravity, "gravity")
    return float(np.sqrt(2.0 * g * h))


def plate_contact_stiffness(
    youngs_modulus: float,
    *,
    poisson_ratio: float = 0.2,
    radius: float = TAPPING_HAMMER_RADIUS,
) -> float:
    r"""Contact stiffness of a plate material
    :math:`K = 2 r E/(1 - \nu^{2})` (Eq. 3.97).

    The stiffness the hammer deforms when it lands on the bare walking surface
    (Timoshenko and Goodier's Hertzian contact for a flat circular punch), as
    opposed to :func:`covering_contact_stiffness` for a soft covering laid on
    top of it.

    :param youngs_modulus: Young's modulus ``E`` of the plate, in Pa.
    :param poisson_ratio: Poisson's ratio ``ν`` of the plate (Default: 0,2,
        the value Hopkins Table A2 estimates for concrete and masonry).
    :param radius: Contact radius ``r``, in m (Default: 0,015).
    :return: The contact stiffness ``K``, in N/m.
    :raises ValueError: If an input is not positive and finite, or
        :math:`\lvert \nu \rvert \ge 1`.
    """
    e = require_positive(youngs_modulus, "youngs_modulus")
    r = require_positive(radius, "radius")
    if not -1.0 < poisson_ratio < 1.0:
        msg = "'poisson_ratio' must lie in (-1, 1)."
        raise ValueError(msg)
    return float(2.0 * r * e / (1.0 - poisson_ratio**2))


def covering_contact_stiffness(
    youngs_modulus: float,
    thickness: float,
    *,
    radius: float = TAPPING_HAMMER_RADIUS,
) -> float:
    r"""Contact stiffness of a soft floor covering
    :math:`K = E \pi r^{2}/d` (Eq. 3.98).

    The covering is treated as a linear spring of area :math:`\pi r^{2}`
    under the hammer, so only the ratio :math:`E/d` matters. Vigran's
    Eq. (8.51) is the same expression written with the hammer area ``Sh``,
    quoted there as 7 cm² against the 7.07 cm² of a 15 mm radius.

    :param youngs_modulus: Young's modulus ``E`` of the covering, in Pa.
    :param thickness: Covering thickness ``d``, in m.
    :param radius: Contact radius ``r``, in m (Default: 0,015).
    :return: The contact stiffness ``K``, in N/m.
    :raises ValueError: If an input is not positive and finite.
    """
    e = require_positive(youngs_modulus, "youngs_modulus")
    d = require_positive(thickness, "thickness")
    r = require_positive(radius, "radius")
    return float(e * np.pi * r**2 / d)


def _is_over_critical(stiffness: float, impedance: float, mass: float) -> bool:
    r"""``True`` for an over-critical oscillation,
    :math:`K m \ge 4 Z_\mathrm{dp}^{2}` (Eq. 3.95).
    """
    return stiffness * mass >= 4.0 * impedance**2


def tapping_cut_off_frequency(
    contact_stiffness: float,
    impedance: float,
    *,
    mass: float = TAPPING_HAMMER_MASS,
) -> float:
    r"""Cut-off frequency ``fco`` of the force spectrum (Eqs. 3.101/3.102).

    Above ``fco`` the tapping machine's force spectrum is no longer flat and
    the force falls away. For an under-critical oscillation
    (:math:`K m < 4 Z_\mathrm{dp}^{2}`, the case of a concrete slab with or without
    a soft covering) it is the undamped mass-spring value
    :math:`f_\mathrm{co} = \sqrt{K/m}/(2 \pi)` (Eq. 3.102); for an over-critical one
    (a lightweight walking surface) it is the lower root
    :math:`[K/(2 Z_\mathrm{dp}) - \sqrt{(K/(2 Z_\mathrm{dp}))^{2} - K/m}]/(2 \pi)`
    (Eq. 3.101).

    :param contact_stiffness: Contact stiffness ``K``, in N/m (see
        :func:`plate_contact_stiffness` / :func:`covering_contact_stiffness`).
    :param impedance: Driving-point impedance ``Zdp`` of the floor, in N.s/m
        (for a homogeneous plate,
        :func:`phonometry.vibration.infinite_plate_impedance`).
    :param mass: Hammer mass ``m``, in kg (Default: 0,5).
    :return: The cut-off frequency ``fco``, in Hz.
    :raises ValueError: If an input is not positive and finite.
    """
    k = require_positive(contact_stiffness, "contact_stiffness")
    z = require_positive(impedance, "impedance")
    m = require_positive(mass, "mass")
    if not _is_over_critical(k, z, m):
        return float(np.sqrt(k / m) / (2.0 * np.pi))
    a = k / (2.0 * z)
    return float((a - np.sqrt(a**2 - k / m)) / (2.0 * np.pi))


def hammer_limiting_frequency(
    impedance: float, *, mass: float = TAPPING_HAMMER_MASS
) -> float:
    r"""Limiting frequency :math:`f_\mathrm{limit} = Z_\mathrm{dp}/(2 \pi m)` (Hopkins
    Eq. 3.106).

    The frequency at which the floor's driving-point impedance equals the
    magnitude of the hammer's own mass impedance :math:`|Z_\mathrm{h}| = \omega m`;
    above it the hammer mass, not the floor, limits the injected power, and
    the power input stops rising at 3 dB per doubling of frequency. Vigran's
    Eq. (8.48) writes the same frequency as
    :math:`f_z = 4 \sqrt{m_1 B_1}/(\pi m_\mathrm{h})`.

    :param impedance: Driving-point impedance ``Zdp``, in N.s/m.
    :param mass: Hammer mass ``m``, in kg (Default: 0,5).
    :return: The limiting frequency ``flimit``, in Hz.
    :raises ValueError: If an input is not positive and finite.
    """
    z = require_positive(impedance, "impedance")
    m = require_positive(mass, "mass")
    return float(z / (2.0 * np.pi * m))


def force_pulse(
    time: ArrayLike,
    contact_stiffness: float,
    impedance: float,
    *,
    mass: float = TAPPING_HAMMER_MASS,
    impact_velocity: float | None = None,
) -> np.ndarray:
    r"""Force pulse ``F1(t)`` of a single hammer impact (Eqs. 3.95/3.96).

    Lindblad's solution of the mass-spring-dashpot of Hopkins Fig. 3.28, the
    hammer mass ``m`` on the contact stiffness ``K`` in series with the floor's
    driving-point impedance ``Zdp``. For an **over-critical** oscillation
    (:math:`K m \ge 4 Z_\mathrm{dp}^{2}`) the pulse decays to zero without changing
    sign (Eq. 3.95); for an **under-critical** one it is a decaying sinusoid
    (Eq. 3.96) whose first positive lobe is the impact proper. Hopkins's rule
    is stated in terms of the sign of the force rather than of any mechanism:
    "only the initial force pulse that has zero or positive force values is
    used to determine the force spectrum, with all subsequent values of F1(t)
    due to the oscillations set to zero before taking the Fourier transform",
    the hammer having rebounded from the plate. That truncation is applied
    here, so the under-critical pulse is returned as zero beyond its
    first zero crossing at :math:`t = \pi/\beta`; it is the same truncation
    :func:`tapping_force_spectrum` transforms, so integrating this pulse over
    :math:`0 \le t \le \pi/\beta` reproduces that spectrum.

    The over-critical pulse has no such cut and decays for all ``t``; it is
    evaluated in a form that stays finite over the whole 0,1 s between
    impacts rather than one that overflows partway through it.

    :param time: Time ``t`` since the impact, in s (scalar or array,
        :math:`\ge 0`).
    :param contact_stiffness: Contact stiffness ``K``, in N/m.
    :param impedance: Driving-point impedance ``Zdp``, in N.s/m.
    :param mass: Hammer mass ``m``, in kg (Default: 0,5).
    :param impact_velocity: Impact velocity ``vo``, in m/s
        (Default: :func:`hammer_impact_velocity`).
    :return: The force ``F1(t)``, in N, with the same shape as ``time``; never
        negative.
    :raises ValueError: If an input is not positive and finite, or ``time``
        contains a negative value.
    """
    k = require_positive(contact_stiffness, "contact_stiffness")
    z = require_positive(impedance, "impedance")
    m = require_positive(mass, "mass")
    v0 = (
        hammer_impact_velocity()
        if impact_velocity is None
        else require_positive(impact_velocity, "impact_velocity")
    )
    t = np.asarray(time, dtype=np.float64)
    if not np.all(np.isfinite(t)) or np.any(t < 0.0):
        msg = "'time' must contain only finite, non-negative values."
        raise ValueError(msg)
    decay = k / (2.0 * z)
    omega0_sq = k / m
    if _is_over_critical(k, z, m):
        gamma = np.sqrt(decay**2 - omega0_sq)
        if gamma <= 0.0:
            # A zero discriminant is the critically damped limit, v0 K t e^-at,
            # which is also the γ → 0 limit of the expression below.
            return np.asarray(v0 * k * t * np.exp(-decay * t), dtype=np.float64)
        # e^(-a t) sinh(γ t)/γ is regrouped as
        # e^(-(a-γ) t) (1 − e^(-2 γ t))/(2 γ) rather than evaluated as written.
        # Both a − γ and a + γ are positive (a² − γ² = ωo² > 0), so neither
        # exponential ever grows, whereas the literal form multiplies an
        # e^(-a t) that underflows to zero by a sinh that overflows to
        # infinity and returns NaN well inside the machine's own 0,1 s impact
        # period. ``expm1`` keeps the near-critical case (γ t ≪ 1, where the
        # two exponentials nearly cancel) accurate to full precision.
        pulse = (
            v0
            * k
            * np.exp(-(decay - gamma) * t)
            * -np.expm1(-2.0 * gamma * t)
            / (2.0 * gamma)
        )
        return np.asarray(pulse, dtype=np.float64)
    beta = np.sqrt(omega0_sq - decay**2)
    pulse = v0 * k * np.exp(-decay * t) * np.sin(beta * t) / beta
    return np.asarray(np.where(t <= np.pi / beta, pulse, 0.0), dtype=np.float64)


def short_pulse_mean_square_force(
    frequencies: ArrayLike, *, band: BandWidth = "third"
) -> np.ndarray:
    r"""Band mean-square force of a short impact
    :math:`F_\mathrm{rms}^{2} = 3.9 B` (Eq. 3.92).

    The limiting case in which the impact is short enough that the hammer's
    momentum alone sets the force: combining :math:`|F_n| = 2 m v_o/T_i`
    (Eq. 3.90) with :math:`F_\mathrm{rms}^{2} = |F_n|^{2} B/(2 f_i)` (Eq. 3.91)
    gives 3.925 B, printed as 3.9 B. Hopkins finds it adequate for bare
    concrete slabs of at least 100 mm.

    :param frequencies: Band centre frequencies ``f``, in Hz.
    :param band: ``"third"`` (:math:`B = 0.23 f`) or ``"octave"``
        (:math:`B = 0.707 f`).
    :return: The band mean-square force ``F²rms``, in N².
    :raises ValueError: If an input is not positive and finite, or ``band`` is
        unknown.
    """
    f = require_positive_array(frequencies, "frequencies")
    return np.asarray(_SHORT_PULSE_COEFFICIENT * _band_factor(band) * f)


@dataclass(frozen=True)
class TappingForceResult:
    r"""Force spectrum of the ISO tapping machine on one walking surface.

    :ivar frequencies: Band centre frequencies ``f``, in Hz.
    :ivar peak_force: Magnitude of the Fourier force component ``|Fn|``, in N
        (Hopkins Fig. 3.32).
    :ivar mean_square_force: Band mean-square force ``F²rms``, in N²
        (Eq. 3.91).
    :ivar power_input: Power injected into the floor
        :math:`W_\mathrm{in} = F_\mathrm{rms}^{2}/Z_\mathrm{dp}`, in W (Eq. 3.103).
    :ivar cut_off_frequency: Cut-off frequency ``fco``, in Hz
        (Eqs. 3.101/3.102).
    :ivar limiting_frequency: Limiting frequency ``flimit``, in Hz (Eq. 3.106).
    :ivar over_critical: ``True`` when :math:`K m \ge 4 Z_\mathrm{dp}^{2}`, i.e. the
        hammer does not rebound.
    :ivar contact_stiffness: Contact stiffness ``K`` used, in N/m.
    :ivar impedance: Driving-point impedance ``Zdp`` used, in N.s/m.
    :ivar lower_limit: Low-frequency asymptote
        :math:`\lvert F_n \rvert_{\text{lower}} = m v_o/T_i`, in N (Eq. 3.99).
    :ivar upper_limit: Low-frequency asymptote
        :math:`\lvert F_n \rvert_{\text{upper}} = 2 m v_o/T_i`, in N
        (Eq. 3.100); 6 dB above ``lower_limit`` in mean square.
    :ivar band: Band width used for ``mean_square_force``.
    """

    frequencies: np.ndarray
    peak_force: np.ndarray
    mean_square_force: np.ndarray
    power_input: np.ndarray
    cut_off_frequency: float
    limiting_frequency: float
    over_critical: bool
    contact_stiffness: float
    impedance: float
    lower_limit: float
    upper_limit: float
    band: str = "third"

    @property
    def power_input_level(self) -> np.ndarray:
        r"""Power input level :math:`10 \log_{10}(W_\mathrm{in}/1~\text{pW})`, in dB
        (Hopkins Fig. 3.33).
        """
        return np.asarray(10.0 * np.log10(self.power_input / _POWER_REFERENCE))

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the force spectrum ``|Fn|`` with its asymptotes and ``fco``.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_tapping_force

        check_language(language)
        return plot_tapping_force(self, ax=ax, language=language, **kwargs)


def tapping_force_spectrum(
    frequencies: ArrayLike,
    contact_stiffness: float,
    impedance: float,
    *,
    mass: float = TAPPING_HAMMER_MASS,
    impact_rate: float = TAPPING_IMPACT_RATE,
    impact_velocity: float | None = None,
    band: BandWidth = "third",
) -> TappingForceResult:
    r"""Force spectrum of the ISO tapping machine on a floor (Hopkins 3.6.3.1).

    The Fourier transform of the single-impact force pulse
    (:func:`force_pulse`), scaled by the impact repetition rate. Writing
    :math:`a = K/(2 Z_\mathrm{dp})` and :math:`\omega_o^{2} = K/m`, the transform of
    Eqs. (3.95)/(3.96) is the same rational function in both critical cases,
    :math:`\hat{F}(\omega) = v_o K/(\omega_o^{2} - \omega^{2} + 2 i a \omega)`,
    multiplied for the under-critical case by
    :math:`1 + e^{-a \pi/\beta} e^{-i \omega \pi/\beta}` because only the
    first positive lobe (of duration :math:`\pi/\beta`) is transformed. That
    truncation is what produces the deep troughs at :math:`n f_\mathrm{co}`,
    :math:`n = 3, 5, 7` that Hopkins notes below Fig. 4.64; they vanish once
    the covering's internal damping is included and the spectrum is averaged
    into bands.

    The transform is normalised so that the low-frequency asymptote is
    :math:`m v_o/T_i` for an over-critical impact (no rebound) and
    :math:`2 m v_o/T_i` for a lightly damped under-critical one (full
    rebound), the two limits of Eqs. (3.99)/(3.100).

    :param frequencies: Band centre frequencies ``f``, in Hz.
    :param contact_stiffness: Contact stiffness ``K``, in N/m.
    :param impedance: Driving-point impedance ``Zdp`` of the floor, in N.s/m.
    :param mass: Hammer mass ``m``, in kg (Default: 0,5).
    :param impact_rate: Impact repetition rate ``fi``, in Hz (Default: 10).
    :param impact_velocity: Impact velocity ``vo``, in m/s
        (Default: :func:`hammer_impact_velocity`).
    :param band: ``"third"`` or ``"octave"``, the band width of Eq. (3.91).
    :return: A :class:`TappingForceResult`.
    :raises ValueError: If an input is not positive and finite, or ``band`` is
        unknown.
    """
    f = require_positive_array(frequencies, "frequencies")
    k = require_positive(contact_stiffness, "contact_stiffness")
    z = require_positive(impedance, "impedance")
    m = require_positive(mass, "mass")
    fi = require_positive(impact_rate, "impact_rate")
    factor = _band_factor(band)
    v0 = (
        hammer_impact_velocity()
        if impact_velocity is None
        else require_positive(impact_velocity, "impact_velocity")
    )

    omega = 2.0 * np.pi * f
    decay = k / (2.0 * z)
    omega0_sq = k / m
    spectrum = v0 * k / (omega0_sq - omega**2 + 2.0j * decay * omega)
    over_critical = _is_over_critical(k, z, m)
    if not over_critical:
        beta = np.sqrt(omega0_sq - decay**2)
        duration = np.pi / beta
        spectrum = spectrum * (
            1.0 + np.exp(-decay * duration) * np.exp(-1.0j * omega * duration)
        )
    peak = np.abs(spectrum) * fi
    mean_square = peak**2 * (factor * f) / (2.0 * fi)
    return TappingForceResult(
        frequencies=f,
        peak_force=np.asarray(peak, dtype=np.float64),
        mean_square_force=np.asarray(mean_square, dtype=np.float64),
        power_input=np.asarray(mean_square / z, dtype=np.float64),
        cut_off_frequency=tapping_cut_off_frequency(k, z, mass=m),
        limiting_frequency=hammer_limiting_frequency(z, mass=m),
        over_critical=over_critical,
        contact_stiffness=k,
        impedance=z,
        lower_limit=float(m * v0 * fi),
        upper_limit=float(2.0 * m * v0 * fi),
        band=band,
    )


# --------------------------------------------------------------------------- #
# 2. Soft floor coverings on a heavyweight floor (Hopkins 4.4.3.1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CoveringImprovementResult:
    r"""Predicted improvement ``ΔL`` of a soft floor covering
    (Hopkins 4.4.3.1).

    :ivar frequencies: Band centre frequencies ``f``, in Hz.
    :ivar improvement: Band improvement ``ΔL``, in dB: Eq. (4.114) evaluated
        over the tapping machine's Fourier lines and summed in mean square
        across each band, :math:`10 \log_{10}(\sum \lvert F_n
        \rvert^{2}_{\text{without}}/\sum \lvert F_n
        \rvert^{2}_{\text{with}})`.
    :ivar two_line: The two-line estimate, in dB: 0 below ``fco`` and
        12 dB/octave (40 dB/decade) above it.
    :ivar cut_off_frequency: Cut-off frequency ``fco`` of the covered floor,
        in Hz.
    :ivar bare_cut_off_frequency: Cut-off frequency of the bare plate, in Hz.
    :ivar lines: Fourier line frequencies :math:`n f_i` of the tapping
        machine, in Hz, covering every band in ``frequencies``.
    :ivar line_improvement: The per-line ratio
        :math:`\Delta L = 20 \log_{10}(\lvert F_n \rvert_{\text{without}}/\lvert
        F_n \rvert_{\text{with}})` of Eq. (4.114) at ``lines``, in
        dB. It carries the deep troughs at odd multiples of ``fco`` that
        Hopkins notes below Fig. 4.64, which are an artefact of the undamped
        model and disappear from ``improvement``.
    :ivar bare: The bare-plate :class:`TappingForceResult`, at ``lines``.
    :ivar covered: The :class:`TappingForceResult` with the covering, at
        ``lines``.
    """

    frequencies: np.ndarray
    improvement: np.ndarray
    two_line: np.ndarray
    cut_off_frequency: float
    bare_cut_off_frequency: float
    lines: np.ndarray
    line_improvement: np.ndarray
    bare: TappingForceResult
    covered: TappingForceResult

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot ``ΔL(f)`` from the force ratio beside the two-line estimate.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_covering_improvement

        check_language(language)
        return plot_covering_improvement(self, ax=ax, language=language, **kwargs)


def covering_improvement(
    frequencies: ArrayLike,
    covering_stiffness: float,
    plate_stiffness: float,
    impedance: float,
    *,
    mass: float = TAPPING_HAMMER_MASS,
    impact_rate: float = TAPPING_IMPACT_RATE,
    band: BandWidth = "third",
) -> CoveringImprovementResult:
    r"""Improvement of impact sound insulation by a soft covering (Eq. 4.114).

    On a heavyweight base floor a soft covering has a negligible effect on the
    mass, bending stiffness and total loss factor of the slab, so it alters
    only the force the hammer injects. The improvement is then the ratio of
    the two force spectra,
    :math:`\Delta L = 20 \log_{10}(|F_n|_{\text{without}}/|F_n|_{\text{with}})`,
    computed here from :func:`tapping_force_spectrum` with the covering's
    contact stiffness (Eq. 3.98) and with the plate's (Eq. 3.97).

    The tapping machine excites a **line** spectrum, at multiples of the 10 Hz
    impact rate, so Eq. (4.114) is a statement about one Fourier component and
    the band value is the ratio of the band mean-square forces (Eq. 3.91),
    that is the sum over the lines that fall in the band. ``improvement`` is
    that band value and ``line_improvement`` is the per-line ratio. The
    distinction matters: the undamped model's transform has exact nulls at odd
    multiples of ``fco``, so a band centre that happens to land on one reads
    tens of dB high. With the 100 Hz cut-off of Hopkins's covering No. 2, the
    line ratio at 500 Hz is 66.8 dB against a two-line estimate of 27.9 dB,
    while the band value is 33.3 dB. Hopkins notes below Fig. 4.64 that the
    troughs vanish once the covering's internal damping is included and the
    spectrum is averaged into bands.

    ``two_line`` is Hopkins's design estimate: :math:`\Delta L \approx 0`
    below the covering's cut-off and a straight 12 dB/octave above it, that
    is :math:`40 \log_{10}(f/f_\mathrm{co})`.
    Real coverings behave as non-linear springs under the tapping machine's
    high force and show two or three slopes between 5 and 22 dB/octave, so the
    model identifies the general features rather than replacing a measurement.

    :param frequencies: Band centre frequencies ``f``, in Hz.
    :param covering_stiffness: Contact stiffness ``K`` of the covering, in N/m
        (:func:`covering_contact_stiffness`).
    :param plate_stiffness: Contact stiffness ``K`` of the bare plate, in N/m
        (:func:`plate_contact_stiffness`).
    :param impedance: Driving-point impedance ``Zdp`` of the base floor, in
        N.s/m; unchanged by the covering.
    :param mass: Hammer mass ``m``, in kg (Default: 0,5).
    :param impact_rate: Impact repetition rate ``fi``, in Hz (Default: 10);
        it sets the spacing of the Fourier lines the bands average over.
    :param band: ``"third"`` or ``"octave"``.
    :return: A :class:`CoveringImprovementResult`.
    :raises ValueError: If an input is not positive and finite, or ``band`` is
        unknown.
    """
    f = require_positive_array(frequencies, "frequencies")
    fi = require_positive(impact_rate, "impact_rate")
    require_choice(band, "band", ("third", "octave"))
    ratio = _HALF_BAND_RATIO[band]
    count = int(np.ceil(float(f.max()) * ratio / fi))
    lines = np.arange(1, count + 1, dtype=np.float64) * fi
    bare = tapping_force_spectrum(
        lines, plate_stiffness, impedance, mass=mass, impact_rate=fi, band=band
    )
    covered = tapping_force_spectrum(
        lines, covering_stiffness, impedance, mass=mass, impact_rate=fi, band=band
    )
    line_improvement = 20.0 * np.log10(bare.peak_force / covered.peak_force)
    inside = (lines[None, :] >= f[:, None] / ratio) & (
        lines[None, :] <= f[:, None] * ratio
    )
    bare_ms = inside @ bare.peak_force**2
    covered_ms = inside @ covered.peak_force**2
    # A band narrower than the line spacing holds no line; there the single
    # Fourier component at the band centre is the whole band.
    empty = ~inside.any(axis=1)
    if empty.any():
        edge = tapping_force_spectrum(
            f[empty],
            plate_stiffness,
            impedance,
            mass=mass,
            impact_rate=fi,
            band=band,
        )
        edge_covered = tapping_force_spectrum(
            f[empty],
            covering_stiffness,
            impedance,
            mass=mass,
            impact_rate=fi,
            band=band,
        )
        bare_ms[empty] = edge.peak_force**2
        covered_ms[empty] = edge_covered.peak_force**2
    improvement = 10.0 * np.log10(bare_ms / covered_ms)
    fco = covered.cut_off_frequency
    two_line = np.where(f > fco, _SLOPE_CREMER * np.log10(f / fco), 0.0)
    return CoveringImprovementResult(
        frequencies=f,
        improvement=np.asarray(improvement, dtype=np.float64),
        two_line=np.asarray(two_line, dtype=np.float64),
        cut_off_frequency=fco,
        bare_cut_off_frequency=bare.cut_off_frequency,
        lines=lines,
        line_improvement=np.asarray(line_improvement, dtype=np.float64),
        bare=bare,
        covered=covered,
    )


# --------------------------------------------------------------------------- #
# 3. Floating floors (ISO 12354-2 Annex C, Hopkins 4.4.4, Vigran 8.4)
# --------------------------------------------------------------------------- #
def floating_floor_resonance_frequency(
    dynamic_stiffness: float, mass_per_area: float
) -> float:
    r"""Resonance :math:`f_\mathrm{o} = 160 \sqrt{s'/m'}` of a floating floor
    (Formula C.2).

    ISO 12354-2:2017 Formula (C.2), with ``s'`` the EN 29052-1 dynamic
    stiffness per unit area of the resilient layer measured without pre-load
    and ``m'`` the mass per unit area of the floating floor. The printed
    constant 160 rounds the exact mass-spring value
    :math:`1000/(2 \pi) = 159.15` that
    :func:`phonometry.materials.natural_frequency` applies, so the two differ
    by 0.5 %; this function reproduces the standard, whose own Annex G
    example prints :math:`f_\mathrm{o} = 52.8` Hz for :math:`s' = 8` MN/m³,
    :math:`m' = 73.5` kg/m².

    :param dynamic_stiffness: Dynamic stiffness per unit area ``s'``, in N/m³
        (i.e. 8e6 for the 8 MN/m³ of the standard's example).
    :param mass_per_area: Mass per unit area ``m'`` of the floating floor, in
        kg/m².
    :return: The resonance frequency ``fo``, in Hz.
    :raises ValueError: If an input is not positive and finite.
    """
    s = require_positive(dynamic_stiffness, "dynamic_stiffness")
    m = require_positive(mass_per_area, "mass_per_area")
    return float(_ISO_RESONANCE_CONSTANT * np.sqrt(s * 1e-6 / m))


def combined_dynamic_stiffness(layers: ArrayLike) -> float:
    r"""Total dynamic stiffness of stacked resilient layers (Formula C.6).

    :math:`s'_\mathrm{tot} = (\sum 1/s'_i)^{-1}`, springs in series (Hopkins
    Eq. 4.121 states the same rule). ISO 12354-2:2017 warns that it holds only
    if every layer covers the whole floor without cuts for pipes or electrical
    devices.

    :param layers: Dynamic stiffnesses per unit area ``s'i``, in N/m³
        (any 1-D array-like).
    :return: The total dynamic stiffness ``s'tot``, in N/m³.
    :raises ValueError: If ``layers`` is empty or holds a non-positive value.
    """
    values = require_positive_array(layers, "layers")
    return float(1.0 / np.sum(1.0 / values))


def double_floating_floor_resonances(
    lower_stiffness: float,
    lower_mass_per_area: float,
    upper_stiffness: float,
    upper_mass_per_area: float,
) -> tuple[float, float]:
    r"""The two resonances of a double floating floor (Hopkins Eq. 4.125).

    One floating floor on top of another over a heavyweight base is a
    mass-spring-mass-spring system with

    .. math::

       f_\mathrm{msms} = \frac{1}{2^{3/2} \pi}
       \sqrt{X \pm \sqrt{X^{2} - \frac{4 s_1 s_2}{\rho_{\mathrm{s}1} \rho_{\mathrm{s}2}}}},
       \qquad
       X = \frac{s_1}{\rho_{\mathrm{s}1}} + \frac{s_2}{\rho_{\mathrm{s}1}} +
       \frac{s_2}{\rho_{\mathrm{s}2}}

    subscript 1 being the lower floating floor (on the resilient layer that
    rests on the base) and 2 the upper one. The double floor avoids the single
    floor's dip at ``fms``, but the steep rise in ``ΔL`` only starts above the
    higher of the two resonances. For two identical floors the roots are
    :math:`f_\mathrm{ms} \sqrt{(3 \pm \sqrt{5})/2}`, that is :math:`0.618 f_\mathrm{ms}`
    and :math:`1.618 f_\mathrm{ms}`.

    :param lower_stiffness: Dynamic stiffness per unit area ``s1`` of the lower
        resilient layer, in N/m³.
    :param lower_mass_per_area: Mass per unit area ``ρs1`` of the lower
        floating floor, in kg/m².
    :param upper_stiffness: Dynamic stiffness per unit area ``s2`` of the upper
        resilient layer, in N/m³.
    :param upper_mass_per_area: Mass per unit area ``ρs2`` of the upper
        floating floor, in kg/m².
    :return: ``(lower, upper)`` resonance frequencies, in Hz.
    :raises ValueError: If an input is not positive and finite.
    """
    s1 = require_positive(lower_stiffness, "lower_stiffness")
    m1 = require_positive(lower_mass_per_area, "lower_mass_per_area")
    s2 = require_positive(upper_stiffness, "upper_stiffness")
    m2 = require_positive(upper_mass_per_area, "upper_mass_per_area")
    x = s1 / m1 + s2 / m1 + s2 / m2
    root = np.sqrt(x**2 - 4.0 * s1 * s2 / (m1 * m2))
    scale = 1.0 / (2.0**1.5 * np.pi)
    return float(scale * np.sqrt(x - root)), float(scale * np.sqrt(x + root))


def weighted_floating_floor_improvement(
    mass_per_area: float,
    dynamic_stiffness: float,
    *,
    floor: FloorType = "screed",
) -> float:
    r"""Weighted improvement ``ΔLw`` of a floating floor (Formulae C.4/C.5).

    The single number that feeds the simplified prediction
    (:func:`phonometry.building.predicted_impact_insulation`), read directly
    from the floating floor's mass per unit area and the resilient layer's
    dynamic stiffness. ISO 12354-2:2017 gives it as the two nomograms of
    Figures C.1 and C.2 and prints the fits:

    * ``floor="screed"`` (sand-cement or calcium-sulfate screeds, Formula C.4):
      :math:`\Delta L_\mathrm{w} = 13 \log_{10}(m') - 14.2 \log_{10}(s') + 20.8`;
    * ``floor="asphalt"`` (asphalt or dry floating floors, Formula C.5):
      :math:`\Delta L_\mathrm{w} = (-0.21 m' - 5.45) \log_{10}(s') + 0.46 m' + 23.8`.

    :param mass_per_area: Mass per unit area ``m'`` of the floating floor, in
        kg/m².
    :param dynamic_stiffness: Dynamic stiffness per unit area ``s'``, in N/m³.
    :param floor: ``"screed"`` (Formula C.4) or ``"asphalt"`` (Formula C.5).
    :return: The weighted improvement ``ΔLw``, in dB.
    :raises ValueError: If an input is not positive and finite, or ``floor`` is
        unknown.
    """
    m = require_positive(mass_per_area, "mass_per_area")
    s = require_positive(dynamic_stiffness, "dynamic_stiffness") * 1e-6
    require_choice(floor, "floor", ("screed", "asphalt"))
    if floor == "screed":
        a, b, c = _DELTA_LW_SCREED
        return float(a * np.log10(m) + b * np.log10(s) + c)
    a, b, c, d = _DELTA_LW_ASPHALT
    return float((a * m + b) * np.log10(s) + c * m + d)


@dataclass(frozen=True)
class FloatingFloorImprovementResult:
    """Predicted improvement ``ΔL(f)`` of a floating floor.

    :ivar frequencies: Band centre frequencies ``f``, in Hz.
    :ivar improvement: Improvement ``ΔL`` per band, in dB (0 at and below
        ``resonance_frequency``).
    :ivar resonance_frequency: Mass-spring resonance ``fo``, in Hz.
    :ivar model: ``"en12354"``, ``"cremer"`` or ``"cremer_hammer"``.
    :ivar slope: Slope of the law, in dB per decade (30 or 40).
    :ivar limiting_frequency: Limiting frequency ``flimit`` of the hammer term,
        in Hz, or ``None`` when the term is not applied.
    :ivar delta_lw: Weighted improvement ``ΔLw``, in dB, or ``None`` when the
        floor data were not supplied: Formula (C.4) for the ``"en12354"``
        model, Formula (C.5) for the other two.
    """

    frequencies: np.ndarray
    improvement: np.ndarray
    resonance_frequency: float
    model: str
    slope: float
    limiting_frequency: float | None = None
    delta_lw: float | None = None

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot ``ΔL(f)`` with the resonance and the asymptotic slope marked.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_floating_floor_improvement

        check_language(language)
        return plot_floating_floor_improvement(self, ax=ax, language=language, **kwargs)


@overload
def floating_floor_improvement_spectrum(
    frequencies: ArrayLike,
    *,
    resonance_frequency: float,
    model: Literal["cremer_hammer"],
    limiting_frequency: float,
    mass_per_area: float,
    dynamic_stiffness: float,
) -> FloatingFloorImprovementResult: ...


@overload
def floating_floor_improvement_spectrum(
    frequencies: ArrayLike,
    *,
    resonance_frequency: float,
    model: Literal["cremer_hammer"],
    limiting_frequency: float,
) -> FloatingFloorImprovementResult: ...


@overload
def floating_floor_improvement_spectrum(
    frequencies: ArrayLike,
    *,
    resonance_frequency: float,
    model: Literal["en12354", "cremer"] = ...,
    mass_per_area: float,
    dynamic_stiffness: float,
) -> FloatingFloorImprovementResult: ...


@overload
def floating_floor_improvement_spectrum(
    frequencies: ArrayLike,
    *,
    resonance_frequency: float,
    model: Literal["en12354", "cremer"] = ...,
) -> FloatingFloorImprovementResult: ...


def floating_floor_improvement_spectrum(
    frequencies: ArrayLike,
    *,
    resonance_frequency: float,
    model: FloatingFloorModel = "en12354",
    limiting_frequency: float | None = None,
    mass_per_area: float | None = None,
    dynamic_stiffness: float | None = None,
) -> FloatingFloorImprovementResult:
    r"""Improvement ``ΔL(f)`` of a floating floor on a heavyweight base floor.

    Three laws share the same anchor, the mass-spring resonance ``fo`` of the
    walking surface on the resilient layer (:func:`floating_floor_resonance_frequency`),
    and all give :math:`\Delta L = 0` at and below it (in the band containing
    ``fo``, ``ΔL`` is in practice between −5 dB and 0 dB):

    * ``"cremer"``: :math:`\Delta L = 40 \log_{10}(f/f_\mathrm{o})`, Cremer's 1952 result for
      two infinite, locally reacting plates coupled by a spring layer (Hopkins
      Eq. 4.119, Vigran Eq. 8.40), i.e. 12 dB per octave. It holds for
      constructions with high internal damping, such as asphalt screeds, and is
      the branch ISO 12354-2 Formula (C.3) prescribes for asphalt and dry
      floating floors.
    * ``"en12354"`` (default): :math:`\Delta L = 30 \log_{10}(f/f_\mathrm{o})`, the empirical
      law of ISO 12354-2 Formula (C.1) for sand-cement and calcium-sulfate
      screeds (Hopkins Eq. 4.124). Sand-cement screeds have a low internal
      loss factor and act as finite plates with a reverberant bending field,
      for which the 40 lg law overestimates ``ΔL``.
    * ``"cremer_hammer"``:
      :math:`\Delta L = 40 \log_{10}(f/f_\mathrm{o}) + 10 \log_{10}[1 + (f/f_\mathrm{limit})^{2}]`, the
      40 lg law with the reduction in power input above the limiting frequency
      of the hammer's own impedance (Hopkins Eq. 4.123, Vigran Eq. 8.48). A
      lightweight walking surface such as chipboard needs it, and tends to
      18 dB per octave well above ``flimit``.

    The laws are stated as valid above ``fo``, and Cremer's derivation is
    reported to hold in :math:`f_\mathrm{o} < f < 4 f_\mathrm{o}`.

    :param frequencies: Band centre frequencies ``f``, in Hz.
    :param resonance_frequency: Mass-spring resonance ``fo``, in Hz.
    :param model: ``"en12354"``, ``"cremer"`` or ``"cremer_hammer"``.
    :param limiting_frequency: Limiting frequency ``flimit``, in Hz; required
        by ``"cremer_hammer"`` and ignored otherwise
        (:func:`hammer_limiting_frequency`).
    :param mass_per_area: Optional ``m'`` of the floating floor, in kg/m².
    :param dynamic_stiffness: Optional ``s'`` of the resilient layer, in N/m³;
        supplied together with ``mass_per_area`` it adds ``ΔLw`` to the result,
        from Formula (C.4) for ``"en12354"`` (screeds) and Formula (C.5) for
        the other two models (asphalt and dry floating floors).
    :return: A :class:`FloatingFloorImprovementResult`.
    :raises ValueError: If an input is not positive and finite, ``model`` is
        unknown, or ``"cremer_hammer"`` is used without a limiting frequency.
    """
    f = require_positive_array(frequencies, "frequencies")
    f0 = require_positive(resonance_frequency, "resonance_frequency")
    require_choice(model, "model", ("en12354", "cremer", "cremer_hammer"))
    slope = _SLOPE_EN12354 if model == "en12354" else _SLOPE_CREMER
    above = f > f0
    improvement = np.where(above, slope * np.log10(f / f0), 0.0)
    f_limit: float | None = None
    if model == "cremer_hammer":
        if limiting_frequency is None:
            msg = "'limiting_frequency' is required by model='cremer_hammer'."
            raise ValueError(msg)
        f_limit = require_positive(limiting_frequency, "limiting_frequency")
        improvement = improvement + np.where(
            above, 10.0 * np.log10(1.0 + (f / f_limit) ** 2), 0.0
        )
    delta_lw: float | None = None
    if (mass_per_area is None) != (dynamic_stiffness is None):
        msg = (
            "'mass_per_area' and 'dynamic_stiffness' are the two halves of the "
            "weighted improvement 'delta_lw'; give both or neither, since one "
            "alone would return it as None without saying why"
        )
        raise ValueError(msg)
    if mass_per_area is not None and dynamic_stiffness is not None:
        # The weighted fit follows the construction the law belongs to:
        # Formula (C.4) for the sand-cement screeds of the 30 lg law, and
        # Formula (C.5) for the asphalt and dry floating floors of the 40 lg
        # law, which are also the lightweight walking surfaces the hammer term
        # applies to.
        delta_lw = weighted_floating_floor_improvement(
            mass_per_area,
            dynamic_stiffness,
            floor="screed" if model == "en12354" else "asphalt",
        )
    return FloatingFloorImprovementResult(
        frequencies=f,
        improvement=np.asarray(improvement, dtype=np.float64),
        resonance_frequency=f0,
        model=model,
        slope=slope,
        limiting_frequency=f_limit,
        delta_lw=delta_lw,
    )


def resilient_mount_improvement(
    frequencies: ArrayLike,
    *,
    impedance: float,
    mass_per_area: float,
    loss_factor: ArrayLike,
    mount_stiffness: float,
    mount_density: float,
) -> np.ndarray:
    r"""Improvement of a floating floor on discrete resilient mounts (Vér).

    Vér's two-subsystem SEA model of a walking surface carrying a reverberant
    bending-wave field, connected to a heavyweight base floor by ``N`` mounts
    per unit area of stiffness ``k`` each, with all transmission through the
    mounts and none through the cavity. Hopkins Eq. (4.118) writes it as

    .. math::

       \Delta L \approx 10 \log_{10}\!\left(
       \frac{2.3 \rho_{\mathrm{s}1}^{2} c_{\mathrm{L}1} h_1 \eta_1 S_1 \omega^{3}}{N k^{2}}
       \right)

    where ``k`` is the dynamic stiffness of each mount, ``N`` the **number** of
    mounts and ``S1`` the area of the walking surface. Since
    :math:`2.3 \rho_{\mathrm{s}1}^{2} c_{\mathrm{L}1} h_1 = Z_{\mathrm{dp}1} \rho_{\mathrm{s}1}` for
    :math:`Z_{\mathrm{dp}1} = 2.3 \rho c_\mathrm{L} h^{2}` (Eq. 2.190), the same expression
    reads :math:`10 \log_{10}(Z_{\mathrm{dp}1} \rho_{\mathrm{s}1} \eta_1 \omega^{3}/(N/S_1 \cdot
    k^{2}))`, which is the form evaluated here: this function takes the mount
    **density** ``N/S1``, not the count.

    Vigran's Eq. (8.45) is a sum of three terms, :math:`Z_1/Z_2`,
    :math:`m_1 \eta_1/(m_2 \eta_2)` and
    :math:`Z_1 \eta_1 N f^{3}/(2 \pi m_1 f_\mathrm{o}^{4})` with
    :math:`f_\mathrm{o} = \sqrt{N k/m_1}/(2 \pi)`. Only the
    **third** of them is the model implemented here, and that term is
    algebraically identical to Hopkins Eq. (4.118); the first two are the
    low-frequency floor, negligible once the third dominates, which is the
    regime Vigran states the 9 dB per octave slope for. The dominant-term form
    used here therefore rises at **30 dB per decade** (9 dB per octave),
    against the 40 dB per decade of a continuous resilient layer: fewer mounts,
    a thicker walking surface or more internal damping all raise ``ΔL``.

    Vigran's simplified Eq. (8.46) inserts ``Z1`` into that third term and
    prints the coefficient as :math:`2/(\sqrt{3} \pi) = 0.3676`, which is the
    same number as the :math:`2.3094/(2 \pi) = 0.3676` the substitution
    gives; the two forms agree.

    :param frequencies: Band centre frequencies ``f``, in Hz.
    :param impedance: Driving-point impedance ``Zdp1`` of the walking surface,
        in N.s/m.
    :param mass_per_area: Mass per unit area ``ρs1`` of the walking surface, in
        kg/m².
    :param loss_factor: Total loss factor ``η1`` of the walking surface
        (scalar or per band).
    :param mount_stiffness: Dynamic stiffness ``k`` of one mount, in N/m.
    :param mount_density: Number of mounts per unit area ``N/S1``, in 1/m²
        (Vigran's ``N``, which is already a density).
    :return: The improvement ``ΔL`` per band, in dB, and 0 dB at and below
        ``fo``.
    :raises ValueError: If an input is not positive and finite.
    """
    f = require_positive_array(frequencies, "frequencies")
    z1 = require_positive(impedance, "impedance")
    m1 = require_positive(mass_per_area, "mass_per_area")
    eta = require_positive_array(loss_factor, "loss_factor")
    k = require_positive(mount_stiffness, "mount_stiffness")
    n = require_positive(mount_density, "mount_density")
    omega = 2.0 * np.pi * f
    delta = 10.0 * np.log10(z1 * m1 * eta * omega**3 / (n * k**2))
    # Both books state the law above the mass-spring resonance, where this
    # term dominates the two Vigran keeps beside it. Below it the dominant
    # term alone falls without bound and turns negative, which a resilient
    # mounting cannot do, so it is floored at 0 dB the way Hopkins treats the
    # bands below fms ("it is simplest to assume that DeltaL = 0 dB in all
    # frequency bands below the band containing fms", printed p. 521) and the
    # way :func:`floating_floor_improvement_spectrum` already does.
    f0 = np.sqrt(n * k / m1) / (2.0 * np.pi)
    return np.asarray(np.where(f > f0, delta, 0.0), dtype=np.float64)
