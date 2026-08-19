#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Detailed per-band building prediction (EN/ISO 12354-1/-2:2017).

This is the **detailed model** of the building-prediction chain, the per-band
counterpart of the simplified single-number model implemented in
:mod:`phonometry.building.prediction.simplified_model`. Where the simplified model
combines the weighted ratings of the elements (``Rw``, ``ΔRw``, ``Kij``) into a
single ``R'w`` / ``L'n,w``, the detailed model carries every quantity through
the one-third-octave (or octave) bands, converts the laboratory element data to
their **in-situ** values, forms each transmission path per band and only then
rates the result through ISO 717. It is what a consultant runs when the element
spectra are known and the dominant path per band matters.

**Chain, airborne (ISO 12354-1:2017, Clause 4.2).**

1. Element data per band. For homogeneous elements the sound reduction index
   follows from the material properties with the Annex B model: the radiation
   factor for free bending waves ``σ`` (Formulae B.4 to B.6,
   :func:`bending_radiation_factor`), the radiation factor for forced waves ``σf``
   (Formula B.3, :func:`forced_radiation_factor`) and the three-branch
   transmission factor of Formula (B.2)
   (:func:`calculated_sound_reduction_index`).
2. In-situ conversion (Clause 4.2.2). The total loss factor in situ follows
   from Annex C Formula (C.1),

   .. math::

      \eta_\mathrm{tot} = \eta_\mathrm{int}
      + \frac{2 \rho_\mathrm{o} c_\mathrm{o} \sigma}{2 \pi f m'}
      + \frac{c_\mathrm{o}}{\pi^2 S \sqrt{f f_\mathrm{c}}} \sum_k l_k \alpha_k

   (:func:`in_situ_total_loss_factor`), with the perimeter absorption
   coefficients deduced from the junctions' vibration reduction indices
   (Formula C.4, :func:`perimeter_absorption_coefficient`). From it come the
   structural reverberation time :math:`T_\mathrm{s} = 2.2/(f \eta_\mathrm{tot})`
   (:func:`structural_reverberation_time`), the in-situ index
   :math:`R_\mathrm{situ} = R - 10 \log_{10}(T_\mathrm{s,situ}/T_\mathrm{s,lab})` (Formula 9,
   :func:`in_situ_reduction_index`) and the equivalent absorption length
   :math:`a_\mathrm{situ} = 2.2\,\pi^2 S \sqrt{f_\mathrm{ref}/f}/(c_\mathrm{o} T_\mathrm{s,situ})`
   (Formula 11).
3. Junctions (Formula 10).
   :math:`D_{v,ij,\mathrm{situ}} = K_{ij}
   - 10 \log_{10}(l_{ij}/\sqrt{a_{i,\mathrm{situ}} a_{j,\mathrm{situ}}})`,
   floored at 0 dB (:func:`in_situ_velocity_level_difference`).
4. Paths. The direct path is
   :math:`R_\mathrm{Dd} = R_\mathrm{s,situ} + \Delta R_\mathrm{D,situ} + \Delta R_\mathrm{d,situ}`
   (Formula 14) and each flanking path (Formula 15) is
   :math:`R_{ij} = R_{i,\mathrm{situ}}/2 + \Delta R_{i,\mathrm{situ}} + R_{j,\mathrm{situ}}/2
   + \Delta R_{j,\mathrm{situ}} + D_{v,ij,\mathrm{situ}} + T`
   with the geometry term :math:`T = 10 \log_{10}(S_\mathrm{s}/\sqrt{S_i S_j})`
   (:func:`flanking_reduction_index`).
5. Assembly. :math:`R' = -10 \log_{10}(\sum 10^{-R/10})` over the direct path and
   all flanking paths (Formulae 1 to 4), then ``R'w (C; Ctr)`` per ISO 717-1
   (:func:`detailed_airborne_prediction`).

**Chain, impact (ISO 12354-2:2017, Clause 4.2).** The bare floor's normalized
impact sound pressure level per band follows from Annex B Formula (B.2),
:math:`L_\mathrm{n} = 155 - 30 \log_{10}(m') + 10 \log_{10}(T_\mathrm{s}) + 10 \log_{10}(\sigma)
+ 10 \log_{10}(f/f_\mathrm{ref})`
(:func:`bare_floor_impact_level`); the direct path is
:math:`L_\mathrm{n,d} = L_\mathrm{n,situ} - \Delta L_\mathrm{situ} - \Delta L_\mathrm{d,situ}`
(Formula 11) and each flanking path (Formula 12) is
:math:`L_{\mathrm{n},ij} = L_\mathrm{n,situ} - \Delta L_\mathrm{situ} + (R_{i,\mathrm{situ}} - R_{j,\mathrm{situ}})/2
- \Delta R_{j,\mathrm{situ}} - D_{v,ij,\mathrm{situ}} - 10 \log_{10}(S_i/\sqrt{S_i S_j})`
(:func:`flanking_impact_level`), combined
energetically into ``L'n`` and rated ``L'n,w (CI)`` per ISO 717-2
(:func:`detailed_impact_prediction`).

The two parts share the same in-situ machinery, so a building is described once
(:class:`HomogeneousElement` per element, :func:`in_situ_element` per band) and
both the airborne and the impact chain read the same
:class:`InSituElementResult`.

**Type A and Type B elements.** :class:`HomogeneousElement` and
:func:`in_situ_element` describe a **Type A** element, one whose structural
reverberation time is set by the elements connected to it. For a **Type B**
element the standard takes :math:`T_\mathrm{s,situ} = T_\mathrm{s,lab}` (so no in-situ
transfer is
needed) and describes the junction with the normalized direction-averaged
velocity level difference ``Dv,ij,n`` instead of ``Kij``, or with a laboratory
measurement of the flanking level difference ``Dn,f``. Those branches are
:func:`flanking_reduction_index_from_normalized_difference` (Formula 17),
:func:`flanking_impact_level_from_normalized_difference` (Part 2, Formula 14)
and :func:`flanking_reduction_index_from_flanking_level` (Formula 16), with
:func:`resonant_sound_reduction_index` for the Annex B.1 correction their
element indices need below ``fc``.

Clause and formula citations refer to ISO 12354-1:2017 (airborne) or
ISO 12354-2:2017 (impact). The worked example of ISO 12354-1:2017 Annex L and
ISO 12354-2:2017 Annex G (one heavy homogeneous building driving both parts) is
reproduced band by band in the test suite; the defects found in its printed
tables are recorded in ``docs/ERRATA.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from ..._internal.validation import (
    require_choice,
    require_positive,
    require_positive_array,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..._report.metadata import ReportMetadata
    from ..measurement.insulation import ImpactRatingResult, WeightedRatingResult

#: Speed of sound in air ``co`` used throughout EN/ISO 12354 (Annex A): 340 m/s.
SPEED_OF_SOUND: float = 340.0

#: Density of air ``ρo`` of the Annex B transmission model, in kg/m³.
AIR_DENSITY: float = 1.29

#: Reference frequency ``fref`` of Formulae (11), (C.4) and Part 2 (B.2): 1 kHz.
REFERENCE_FREQUENCY: float = 1000.0

#: Reference coupling length ``lo`` (Clause 4.2.2.3): 1 m.
REFERENCE_LENGTH: float = 1.0

#: Upper bound on every radiation factor ("In all cases σ ≤ 2,0", Annex B).
MAX_RADIATION_FACTOR: float = 2.0

#: Constant ``2,2`` of the structural reverberation time ``Ts = 2,2/(f ηtot)``
#: (Formula C.1) and of the equivalent absorption length (Formula 11).
_STRUCTURAL_CONSTANT: float = 2.2

#: Constant term of the calculated impact level, Part 2 Formula (B.2).
_IMPACT_LEVEL_CONSTANT: float = 155.0

#: Coefficient of ``lg(m')`` in Part 2 Formula (B.2).
_IMPACT_MASS_COEFFICIENT: float = 30.0

#: Laboratory total loss factor of Formula (C.3): ``ηint + m'/(485 √f)``.
_LAB_LOSS_CONSTANT: float = 485.0

#: Constants of the high-frequency plateau transmission factor (Formula B.10):
#: ``τplateau = (4 ρo co/(1,1 ρ cL))² · 0,02/ηtot``.
_PLATEAU_FACTOR: float = 1.1
_PLATEAU_CONSTANT: float = 0.02

#: Half-band ratios used to decide which band contains the critical frequency
#: (the ``f ≈ fc`` branch of Formula B.2).
_HALF_BAND: dict[str, float] = {"third": 2.0 ** (1.0 / 6.0), "octave": 2.0**0.5}

#: Constants of the reciprocity relation ``R + Ln = C + 30 lg f`` between the
#: airborne index and the impact level of a homogeneous floor (Part 2, B.3/B.4).
_RECIPROCITY_CONSTANT: dict[str, float] = {"third": 38.0, "octave": 43.0}

#: Band sets ISO 717-1/-2 rate over, in hertz.
_RATING_BANDS: dict[str, tuple[float, ...]] = {
    "third": (
        100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0,
        1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0,
    ),
    "octave": (125.0, 250.0, 500.0, 1000.0, 2000.0),
}

_BAND_CHOICES = ("third", "octave")

BandType = Literal["third", "octave"]
FlankingKind = Literal["Ff", "Df", "Fd"]


# --------------------------------------------------------------------------- #
# Private helpers
# --------------------------------------------------------------------------- #
def _band_array(
    values: ArrayLike, n_bands: int, name: str, *, positive: bool = False
) -> np.ndarray:
    """Coerce *values* to ``n_bands`` finite floats, broadcasting a scalar."""
    data = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if data.ndim != 1 or data.size == 0:
        raise ValueError(f"'{name}' must be a non-empty 1-D array.")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"'{name}' must be finite.")
    if positive and np.any(data <= 0.0):
        raise ValueError(f"'{name}' must be strictly positive.")
    if data.size == 1:
        return np.full(n_bands, data[0], dtype=np.float64)
    if data.size != n_bands:
        raise ValueError(
            f"'{name}' must have one value per band ({n_bands}) or a single value."
        )
    return data


def _ordered_sides(length1: float, length2: float) -> tuple[float, float]:
    r"""Return the side lengths as ``(l1, l2)`` with :math:`l_1 \ge l_2`."""
    a = require_positive(length1, "length1")
    b = require_positive(length2, "length2")
    return (a, b) if a >= b else (b, a)


def _time_ratio(situ: ArrayLike, laboratory: ArrayLike) -> np.ndarray:
    """Validated ``Ts,situ / Ts,lab`` ratio shared by Formulae (9) and (5)."""
    ts = require_positive_array(situ, "situ_reverberation_time")
    tl = require_positive_array(laboratory, "laboratory_reverberation_time")
    n = max(ts.size, tl.size)
    numerator = _band_array(ts, n, "situ_reverberation_time", positive=True)
    denominator = _band_array(tl, n, "laboratory_reverberation_time", positive=True)
    return np.asarray(numerator / denominator, dtype=np.float64)


def _rating_slice(
    frequencies: np.ndarray, values: np.ndarray, bands: BandType
) -> np.ndarray | None:
    """Return the ISO 717 rating range of *values*, or ``None`` if incomplete."""
    wanted = _RATING_BANDS[require_choice(bands, "bands", _BAND_CHOICES)]
    indices = []
    for target in wanted:
        match = np.flatnonzero(np.isclose(frequencies, target, rtol=0.03))
        if match.size == 0:
            return None
        indices.append(int(match[0]))
    return np.asarray(values[indices], dtype=np.float64)


def _check_report_request(engine: str, language: str) -> None:
    """Reject an unknown engine or language before a fiche is rendered."""
    from ..._i18n import check_language

    check_language(language)
    if engine != "reportlab":
        raise ValueError(
            f"Unknown report engine {engine!r}; only 'reportlab' is supported."
        )


# --------------------------------------------------------------------------- #
# Radiation factors (ISO 12354-1:2017, Annex B)
# --------------------------------------------------------------------------- #
def bending_radiation_factor(
    frequencies: ArrayLike,
    *,
    critical_frequency: float,
    length1: float,
    length2: float,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> np.ndarray:
    r"""Radiation factor for free bending waves ``σ`` (Formulae B.4 to B.6).

    The three candidate factors of Formula (B.4) are

    - :math:`\sigma_1 = 1/\sqrt{1 - f_\mathrm{c}/f}` (above the critical frequency),
    - :math:`\sigma_2 = 4 l_1 l_2 (f/c_\mathrm{o})^2` (the plate acting as a small
      piston),
    - :math:`\sigma_3 = \sqrt{2 \pi f (l_1 + l_2)/(16 c_\mathrm{o})}` (corner and
      edge modes),

    and the first plate mode
    :math:`f_{11} = c_\mathrm{o}^2/(4 f_\mathrm{c}) \cdot (1/l_1^2 + 1/l_2^2)` selects
    between the two regimes. For :math:`f_{11} \le f_\mathrm{c}/2` the element is mode
    dense at its critical frequency and Formula (B.5) applies:
    :math:`\sigma = \sigma_1` at and above ``fc``, and below it the
    edge/corner sum
    :math:`\sigma = 2(l_1+l_2)/(l_1 l_2) \cdot (c_\mathrm{o}/f_\mathrm{c}) \cdot \delta_1
    + \delta_2` with :math:`\lambda = \sqrt{f/f_\mathrm{c}}` and ``δ2``
    vanishing above ``fc/2``. For :math:`f_{11} > f_\mathrm{c}/2` Formula (B.6) picks
    ``σ3`` unless ``σ2`` (below ``fc``) or ``σ1`` (above ``fc``) is smaller.
    Every branch is capped at :math:`\sigma \le 2.0`.

    These relations hold for a plate in an infinite baffle; the standard notes
    that walls and floors surrounded by orthogonal elements radiate 2 (edge
    modes) to 4 (corner modes) times more efficiently well below ``fc``.

    :param frequencies: Band centre frequencies ``f``, in Hz.
    :param critical_frequency: Critical frequency
        :math:`f_\mathrm{c} = c_\mathrm{o}^2/(1.8\,c_\mathrm{L} t)`, Hz.
    :param length1: One side length of the rectangular element, in m.
    :param length2: The other side length, in m.
    :param speed_of_sound: Speed of sound in air ``co``, in m/s
        (Default: 340 m/s, the value ISO 12354-1 Annex A fixes).
    :return: The radiation factor ``σ`` per band (dimensionless).
    :raises ValueError: If any input is not positive and finite.
    """
    f = require_positive_array(frequencies, "frequencies")
    fc = require_positive(critical_frequency, "critical_frequency")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    l1, l2 = _ordered_sides(length1, length2)

    sigma2 = 4.0 * l1 * l2 * (f / c0) ** 2
    sigma3 = np.sqrt(2.0 * np.pi * f * (l1 + l2) / (16.0 * c0))
    f11 = c0**2 / (4.0 * fc) * (1.0 / l1**2 + 1.0 / l2**2)

    above = f > fc
    # sigma1 diverges as f approaches fc from above, so it is evaluated only
    # where it applies and pinned to the cap exactly at fc.
    safe = np.where(above, f, 2.0 * fc)
    sigma1 = np.where(
        above, 1.0 / np.sqrt(1.0 - fc / safe), MAX_RADIATION_FACTOR
    )

    if f11 <= fc / 2.0:
        sigma = _mode_dense_radiation_factor(f, fc, l1, l2, c0, sigma1)
        if f11 < fc / 2.0:
            sigma = np.where((f < f11) & (sigma > sigma2), sigma2, sigma)
    else:
        sigma = np.where((f < fc) & (sigma2 < sigma3), sigma2, sigma3)
        sigma = np.where(above & (sigma1 < sigma3), sigma1, sigma)
    return np.asarray(np.minimum(sigma, MAX_RADIATION_FACTOR), dtype=np.float64)


def _mode_dense_radiation_factor(
    f: np.ndarray, fc: float, l1: float, l2: float, c0: float, sigma1: np.ndarray
) -> np.ndarray:
    r"""The :math:`f_{11} \le f_\mathrm{c}/2` branch of Formula (B.5) (edge/corner)."""
    below = f < fc
    lam = np.sqrt(np.where(below, f, 0.5 * fc) / fc)
    one_minus = 1.0 - lam**2
    delta1 = (one_minus * np.log((1.0 + lam) / (1.0 - lam)) + 2.0 * lam) / (
        4.0 * np.pi**2 * one_minus**1.5
    )
    corner = f <= fc / 2.0
    delta2 = np.where(
        corner,
        8.0
        * c0**2
        * (1.0 - 2.0 * lam**2)
        / (fc**2 * np.pi**4 * l1 * l2 * lam * np.sqrt(one_minus)),
        0.0,
    )
    low = 2.0 * (l1 + l2) / (l1 * l2) * (c0 / fc) * delta1 + delta2
    return np.asarray(np.where(below, low, sigma1), dtype=np.float64)


def forced_radiation_factor(
    frequencies: ArrayLike,
    *,
    length1: float,
    length2: float,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> np.ndarray:
    r"""Radiation factor for forced waves ``σf`` (Formula B.3).

    :math:`\sigma_\mathrm{f} = 0.5 (\ln(k_\mathrm{o} \sqrt{l_1 l_2}) - \Lambda)` capped at
    :math:`\sigma_\mathrm{f} \le 2`, with :math:`k_\mathrm{o} = 2 \pi f / c_\mathrm{o}` and, for
    :math:`l_1 > l_2`,

    .. math::

       \Lambda = -0.964 - \left(0.5 + \frac{l_2}{\pi l_1}\right)
       \ln\frac{l_2}{l_1} + \frac{5 l_2}{2 \pi l_1} - E

    with :math:`E = 1/(4 \pi l_1 l_2 k_\mathrm{o}^2)`.

    ISO 12354-1:2017 Table B.1 tabulates :math:`10 \log_{10} \sigma_\mathrm{f}` for the
    two standard laboratory openings (2 m² and 10 m²), which this
    implementation reproduces.

    :param frequencies: Band centre frequencies ``f``, in Hz.
    :param length1: One side length of the rectangular element, in m.
    :param length2: The other side length, in m.
    :param speed_of_sound: Speed of sound in air ``co``, in m/s
        (Default: 340 m/s).
    :return: The forced radiation factor ``σf`` per band (dimensionless),
        clipped to :math:`0 \le \sigma_\mathrm{f} \le 2` (the standard prints only the
        upper bound;
        the lower one guards the deep low-frequency extrapolation, where the
        logarithm turns negative).
    :raises ValueError: If any input is not positive and finite.
    """
    f = require_positive_array(frequencies, "frequencies")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    l1, l2 = _ordered_sides(length1, length2)

    k0 = 2.0 * np.pi * f / c0
    lam = (
        -0.964
        - (0.5 + l2 / (np.pi * l1)) * np.log(l2 / l1)
        + 5.0 * l2 / (2.0 * np.pi * l1)
        - 1.0 / (4.0 * np.pi * l1 * l2 * k0**2)
    )
    sigma_f = 0.5 * (np.log(k0 * np.sqrt(l1 * l2)) - lam)
    return np.asarray(np.clip(sigma_f, 0.0, MAX_RADIATION_FACTOR), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Calculated element performance (Annex B of both parts)
# --------------------------------------------------------------------------- #
def calculated_sound_reduction_index(
    frequencies: ArrayLike,
    *,
    mass_per_area: float,
    critical_frequency: float,
    total_loss_factor: ArrayLike,
    radiation_factor: ArrayLike,
    forced_radiation_factor: ArrayLike,
    bands: BandType = "third",
    resonant_only: bool = False,
    density: float | None = None,
    longitudinal_velocity: float | None = None,
    speed_of_sound: float = SPEED_OF_SOUND,
    air_density: float = AIR_DENSITY,
) -> np.ndarray:
    r"""Sound reduction index of a homogeneous element (Formulae B.2, B.10).

    :math:`R = -10 \log_{10} \tau` with the three-branch transmission factor

    - :math:`f > f_\mathrm{c}`:
      :math:`\tau = (2 \rho_\mathrm{o} c_\mathrm{o}/(2 \pi f m'))^2
      \cdot \pi f_\mathrm{c} \sigma^2/(2 f \eta_\mathrm{tot})`,
    - :math:`f \approx f_\mathrm{c}`:
      :math:`\tau = (2 \rho_\mathrm{o} c_\mathrm{o}/(2 \pi f m'))^2
      \cdot \pi \sigma^2/(2 \eta_\mathrm{tot})`,
    - :math:`f < f_\mathrm{c}`:
      :math:`\tau = (2 \rho_\mathrm{o} c_\mathrm{o}/(2 \pi f m'))^2 \cdot (F + R)` with the
      forced term :math:`F = 2 \sigma_\mathrm{f} [1 - f^2/f_\mathrm{c}^2]^{-2}` and the
      resonant term :math:`R = 2 (\pi f_\mathrm{c}/(4 f)) \sigma^2/\eta_\mathrm{tot}`.

    The :math:`f \approx f_\mathrm{c}` branch is applied to the band whose limits
    straddle the
    critical frequency, which is how the Annex L worked example selects it.

    Below the critical frequency the first term is the *forced* contribution.
    Annex B.1 requires flanking paths to use the **resonant** transmission
    only; ``resonant_only=True`` drops that term (Annex B.3: "the contribution
    of forced transmission can be neglected for flanking paths"). The Annex L
    worked example keeps it on every path, so the default is ``False``.

    **High-frequency plateau (Formula B.10).** At high frequency the index of
    a thick element stops growing; the standard bounds the transmission factor
    from below by
    :math:`\tau_\mathrm{plateau} = (4 \rho_\mathrm{o} c_\mathrm{o}/(1.1\,\rho c_\mathrm{L}))^2
    \cdot 0.02/\eta_\mathrm{tot}`. Supplying
    both ``density`` and ``longitudinal_velocity`` applies that floor,
    :math:`\tau = \max(\tau, \tau_\mathrm{plateau})`, as the Annex L example does
    from about 1250 Hz
    upwards on its lightweight blockwork.

    :param frequencies: Band centre frequencies ``f``, in Hz.
    :param mass_per_area: Mass per unit area ``m'``, in kg/m².
    :param critical_frequency: Critical frequency ``fc``, in Hz.
    :param total_loss_factor: Total loss factor ``ηtot`` per band (laboratory
        or in situ, matching the situation being described).
    :param radiation_factor: Radiation factor for free bending waves ``σ`` per
        band (see :func:`bending_radiation_factor`).
    :param forced_radiation_factor: Radiation factor for forced waves ``σf``
        per band (see :func:`forced_radiation_factor`); ignored when
        ``resonant_only`` is set.
    :param bands: ``"third"`` (default) or ``"octave"``, setting the band
        limits used to locate the :math:`f \approx f_\mathrm{c}` branch.
    :param resonant_only: Drop the forced-transmission term below ``fc``.
    :param density: Density ``ρ`` of the material, in kg/m³; with
        ``longitudinal_velocity`` it enables the Formula (B.10) plateau.
    :param longitudinal_velocity: Quasi-longitudinal phase velocity ``cL`` of
        the material, in m/s.
    :param speed_of_sound: Speed of sound in air ``co``, in m/s.
    :param air_density: Density of air ``ρo``, in kg/m³.
    :return: The sound reduction index ``R`` per band, in dB.
    :raises ValueError: If an input is not positive/finite or the per-band
        arrays do not share the band count.
    """
    f = require_positive_array(frequencies, "frequencies")
    m = require_positive(mass_per_area, "mass_per_area")
    fc = require_positive(critical_frequency, "critical_frequency")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    rho0 = require_positive(air_density, "air_density")
    eta = _band_array(total_loss_factor, f.size, "total_loss_factor", positive=True)
    sigma = _band_array(radiation_factor, f.size, "radiation_factor", positive=True)
    sigma_f = _band_array(
        forced_radiation_factor, f.size, "forced_radiation_factor"
    )
    half = _HALF_BAND[require_choice(bands, "bands", _BAND_CHOICES)]

    mass_term = (2.0 * rho0 * c0 / (2.0 * np.pi * f * m)) ** 2
    at_fc = (f / half <= fc) & (fc <= f * half)
    below = (~at_fc) & (f < fc)

    resonant = np.pi * fc * sigma**2 / (2.0 * f * eta)
    # The forced term is only formed below fc; elsewhere the ratio f/fc would
    # make the [1 - f^2/fc^2]^-2 factor singular, so it is masked out first.
    forced = np.zeros_like(f)
    if not resonant_only:
        ratio = np.where(below, f / fc, 0.0)
        forced = np.where(below, 2.0 * sigma_f / (1.0 - ratio**2) ** 2, 0.0)
    tau = mass_term * np.where(
        at_fc, np.pi * sigma**2 / (2.0 * eta), resonant + forced
    )
    if density is not None and longitudinal_velocity is not None:
        rho = require_positive(density, "density")
        c_l = require_positive(longitudinal_velocity, "longitudinal_velocity")
        plateau = (4.0 * rho0 * c0 / (_PLATEAU_FACTOR * rho * c_l)) ** 2 * (
            _PLATEAU_CONSTANT / eta
        )
        tau = np.maximum(tau, plateau)
    return np.asarray(-10.0 * np.log10(tau), dtype=np.float64)


def bare_floor_impact_level(
    frequencies: ArrayLike,
    *,
    mass_per_area: float,
    structural_reverberation_time: ArrayLike,
    radiation_factor: ArrayLike,
) -> np.ndarray:
    r"""Normalized impact level of a bare monolithic floor (Part 2, F. B.2).

    .. math::

       L_\mathrm{n} = 155 - 30 \log_{10}(m'/1\,\mathrm{kg/m^2}) + 10 \log_{10}(T_\mathrm{s}/1\,\mathrm{s})
       + 10 \log_{10} \sigma + 10 \log_{10}(f/f_\mathrm{ref})

    with :math:`f_\mathrm{ref} = 1000` Hz, the closed form obtained with
    the force level of the standard tapping machine on a low-mobility floor.
    Supplying the *in-situ* structural reverberation time and radiation factor
    returns ``Ln,situ`` directly.

    The reciprocity relation of Part 2 Formulae (B.3)/(B.4),
    :math:`R + L_\mathrm{n} = 38 + 30 \log_{10} f` in one-third-octave bands (43 in octave
    bands),
    holds where forced transmission is negligible and gives an independent
    check on the pair.

    :param frequencies: Band centre frequencies ``f``, in Hz.
    :param mass_per_area: Mass per unit area ``m'`` of the floor, in kg/m².
    :param structural_reverberation_time: Structural reverberation time ``Ts``
        per band, in s.
    :param radiation_factor: Radiation factor for free bending waves ``σ`` per
        band.
    :return: The normalized impact sound pressure level ``Ln`` per band, in dB.
    :raises ValueError: If an input is not positive/finite or the per-band
        arrays do not share the band count.
    """
    f = require_positive_array(frequencies, "frequencies")
    m = require_positive(mass_per_area, "mass_per_area")
    ts = _band_array(
        structural_reverberation_time,
        f.size,
        "structural_reverberation_time",
        positive=True,
    )
    sigma = _band_array(radiation_factor, f.size, "radiation_factor", positive=True)
    return np.asarray(
        _IMPACT_LEVEL_CONSTANT
        - _IMPACT_MASS_COEFFICIENT * np.log10(m)
        + 10.0 * np.log10(ts)
        + 10.0 * np.log10(sigma)
        + 10.0 * np.log10(f / REFERENCE_FREQUENCY),
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# Structural reverberation time and loss factors (Annex C)
# --------------------------------------------------------------------------- #
def perimeter_absorption_coefficient(
    critical_frequencies: ArrayLike, vibration_reduction_indices: ArrayLike
) -> float:
    r"""Absorption coefficient for bending waves at one border (Formula C.4).

    :math:`\alpha_k = \sum_j \sqrt{f_{\mathrm{c},j}/f_\mathrm{ref}} \cdot 10^{-K_{ij}/10}`
    summed over the elements ``j``
    connected to the considered element at border ``k`` (the standard sums
    over at most three). Multiplied by the border length and summed over the
    perimeter it gives the :math:`\sum l_k \alpha_k` that
    :func:`in_situ_total_loss_factor` takes. Annex C.3 places the in-situ
    coefficients between 0,05 and 0,5.

    :param critical_frequencies: Critical frequency ``fc,j`` of each connected
        element, in Hz.
    :param vibration_reduction_indices: Vibration reduction index ``Kij`` of
        the path to each connected element, in dB (same order and length).
    :return: The absorption coefficient ``αk`` at that border
        (dimensionless).
    :raises ValueError: If the two sequences differ in length, a critical
        frequency is not positive, or an index is not finite.
    """
    fc = require_positive_array(critical_frequencies, "critical_frequencies")
    kij = np.atleast_1d(np.asarray(vibration_reduction_indices, dtype=np.float64))
    if kij.shape != fc.shape:
        raise ValueError(
            "'critical_frequencies' and 'vibration_reduction_indices' must "
            "have the same length (one value per connected element)."
        )
    if not np.all(np.isfinite(kij)):
        raise ValueError("'vibration_reduction_indices' must be finite.")
    return float(np.sum(np.sqrt(fc / REFERENCE_FREQUENCY) * 10.0 ** (-kij / 10.0)))


def in_situ_total_loss_factor(
    frequencies: ArrayLike,
    *,
    internal_loss_factor: float,
    mass_per_area: float,
    area: float,
    critical_frequency: float,
    radiation_factor: ArrayLike,
    perimeter_absorption: float,
    speed_of_sound: float = SPEED_OF_SOUND,
    air_density: float = AIR_DENSITY,
) -> np.ndarray:
    r"""Total loss factor in situ ``ηtot,situ`` (Formula C.1).

    :math:`\eta_\mathrm{tot} = \eta_\mathrm{int} + 2 \rho_\mathrm{o} c_\mathrm{o} \sigma/(2 \pi f m')
    + c_\mathrm{o}/(\pi^2 S \sqrt{f f_\mathrm{c}}) \cdot \sum_k l_k \alpha_k`: the
    internal losses of the material, the losses by radiation into the air
    and the losses at the perimeter of the element.
    :math:`\sum l_k \alpha_k` is the junction-length-weighted sum of the
    Formula (C.4) absorption coefficients (see
    :func:`perimeter_absorption_coefficient`).

    :param frequencies: Band centre frequencies ``f``, in Hz.
    :param internal_loss_factor: Internal loss factor ``ηint`` of the material
        (about 0,01 for common homogeneous building materials).
    :param mass_per_area: Mass per unit area ``m'``, in kg/m².
    :param area: Element area ``S``, in m².
    :param critical_frequency: Critical frequency ``fc``, in Hz.
    :param radiation_factor: Radiation factor ``σ`` per band.
    :param perimeter_absorption: :math:`\sum l_k \alpha_k` over the element's
        perimeter, in m (may be zero for a free-edged element).
    :param speed_of_sound: Speed of sound in air ``co``, in m/s.
    :param air_density: Density of air ``ρo``, in kg/m³.
    :return: The total loss factor ``ηtot,situ`` per band (dimensionless).
    :raises ValueError: If an input is not positive/finite, the perimeter sum
        is negative, or the band counts disagree.
    """
    f = require_positive_array(frequencies, "frequencies")
    eta_int = require_positive(internal_loss_factor, "internal_loss_factor")
    m = require_positive(mass_per_area, "mass_per_area")
    s = require_positive(area, "area")
    fc = require_positive(critical_frequency, "critical_frequency")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    rho0 = require_positive(air_density, "air_density")
    sigma = _band_array(radiation_factor, f.size, "radiation_factor", positive=True)
    perimeter_sum = float(perimeter_absorption)
    if not np.isfinite(perimeter_sum) or perimeter_sum < 0.0:
        raise ValueError("'perimeter_absorption' must be finite and non-negative.")

    radiation = 2.0 * rho0 * c0 * sigma / (2.0 * np.pi * f * m)
    perimeter = c0 / (np.pi**2 * s * np.sqrt(f * fc)) * perimeter_sum
    return np.asarray(eta_int + radiation + perimeter, dtype=np.float64)


def laboratory_total_loss_factor(
    frequencies: ArrayLike,
    *,
    mass_per_area: float,
    internal_loss_factor: float = 0.01,
) -> np.ndarray:
    r"""Total loss factor in the laboratory ``ηtot,lab`` (Formula C.3).

    :math:`\eta_\mathrm{tot,lab} \approx \eta_\mathrm{int} + m'/(485 \sqrt{f})`, the
    estimate for the heavy test frame
    of an ISO 10140 facility. The relation holds for elements below
    :math:`m' = 800` kg/m² and ``ηint`` can normally be taken as 0.01.

    :param frequencies: Band centre frequencies ``f``, in Hz.
    :param mass_per_area: Mass per unit area ``m'``, in kg/m².
    :param internal_loss_factor: Internal loss factor ``ηint``
        (Default: 0,01).
    :return: The laboratory total loss factor per band (dimensionless).
    :raises ValueError: If an input is not positive and finite.
    """
    f = require_positive_array(frequencies, "frequencies")
    m = require_positive(mass_per_area, "mass_per_area")
    eta_int = require_positive(internal_loss_factor, "internal_loss_factor")
    return np.asarray(
        eta_int + m / (_LAB_LOSS_CONSTANT * np.sqrt(f)), dtype=np.float64
    )


def structural_reverberation_time(
    frequencies: ArrayLike, total_loss_factor: ArrayLike
) -> np.ndarray:
    r"""Structural reverberation time :math:`T_\mathrm{s} = 2.2/(f \eta_\mathrm{tot})` (C.1).

    :param frequencies: Band centre frequencies ``f``, in Hz.
    :param total_loss_factor: Total loss factor ``ηtot`` per band.
    :return: The structural reverberation time ``Ts`` per band, in s.
    :raises ValueError: If an input is not positive/finite or the band counts
        disagree.
    """
    f = require_positive_array(frequencies, "frequencies")
    eta = _band_array(total_loss_factor, f.size, "total_loss_factor", positive=True)
    return np.asarray(_STRUCTURAL_CONSTANT / (f * eta), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Transfer of the input data to in-situ values (Clause 4.2.2)
# --------------------------------------------------------------------------- #
def in_situ_reduction_index(
    sound_reduction_index: ArrayLike,
    situ_reverberation_time: ArrayLike,
    laboratory_reverberation_time: ArrayLike,
) -> np.ndarray:
    r"""In-situ sound reduction index ``Rsitu`` (Formula 9).

    :math:`R_\mathrm{situ} = R - 10 \log_{10}(T_\mathrm{s,situ}/T_\mathrm{s,lab})`: an element that is
    better damped in
    the building than in the test frame radiates less and gains index. The
    standard notes that :math:`R_\mathrm{situ} = R` is a usable first approximation,
    and the
    correction is exactly zero for Type B elements (Clause 4.2.2.3).

    :param sound_reduction_index: Laboratory index ``R`` per band, in dB.
    :param situ_reverberation_time: In-situ structural reverberation time
        ``Ts,situ`` per band, in s.
    :param laboratory_reverberation_time: Laboratory structural reverberation
        time ``Ts,lab`` per band, in s.
    :return: The in-situ index ``Rsitu`` per band, in dB.
    :raises ValueError: If a reverberation time is not positive/finite or the
        band counts disagree.
    """
    r = np.atleast_1d(np.asarray(sound_reduction_index, dtype=np.float64))
    ratio = _time_ratio(situ_reverberation_time, laboratory_reverberation_time)
    return np.asarray(r - 10.0 * np.log10(ratio), dtype=np.float64)


def in_situ_impact_level(
    impact_level: ArrayLike,
    situ_reverberation_time: ArrayLike,
    laboratory_reverberation_time: ArrayLike,
) -> np.ndarray:
    r"""In-situ normalized impact level ``Ln,situ`` (Part 2, Formula 5).

    :math:`L_\mathrm{n,situ} = L_\mathrm{n} + 10 \log_{10}(T_\mathrm{s,situ}/T_\mathrm{s,lab})`, the sign
    opposite to
    :func:`in_situ_reduction_index`: a floor that rings longer in the building
    than in the laboratory radiates more impact sound.

    :param impact_level: Laboratory level ``Ln`` per band, in dB.
    :param situ_reverberation_time: ``Ts,situ`` per band, in s.
    :param laboratory_reverberation_time: ``Ts,lab`` per band, in s.
    :return: The in-situ level ``Ln,situ`` per band, in dB.
    :raises ValueError: If a reverberation time is not positive/finite or the
        band counts disagree.
    """
    ln = np.atleast_1d(np.asarray(impact_level, dtype=np.float64))
    ratio = _time_ratio(situ_reverberation_time, laboratory_reverberation_time)
    return np.asarray(ln + 10.0 * np.log10(ratio), dtype=np.float64)


def in_situ_equivalent_absorption_length(
    frequencies: ArrayLike,
    *,
    area: float,
    situ_reverberation_time: ArrayLike,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> np.ndarray:
    r"""In-situ equivalent absorption length ``asitu`` (Formula 11).

    :math:`a_\mathrm{situ} = 2.2\,\pi^2 S \sqrt{f_\mathrm{ref}/f}/(c_\mathrm{o} T_\mathrm{s,situ})` with
    :math:`f_\mathrm{ref} = 1000` Hz. Note
    the :math:`\sqrt{f_\mathrm{ref}/f}` dependence: the absorption length grows as
    the element
    rings shorter at high frequency. For a Type B element the standard
    replaces it by the element area, :math:`a_\mathrm{situ} = S/l_o` (Formula 13).

    This is the ISO 10848 Formula (12) quantity
    (:func:`phonometry.building.equivalent_absorption_length`) evaluated with
    the ISO 12354 value :math:`c_\mathrm{o} = 340` m/s.

    :param frequencies: Band centre frequencies ``f``, in Hz.
    :param area: Element area ``S``, in m².
    :param situ_reverberation_time: ``Ts,situ`` per band, in s.
    :param speed_of_sound: Speed of sound in air ``co``, in m/s.
    :return: The equivalent absorption length ``asitu`` per band, in m.
    :raises ValueError: If an input is not positive/finite or the band counts
        disagree.
    """
    from ..measurement.flanking_transmission import equivalent_absorption_length

    f = require_positive_array(frequencies, "frequencies")
    ts = _band_array(
        situ_reverberation_time, f.size, "situ_reverberation_time", positive=True
    )
    return np.asarray(
        equivalent_absorption_length(area, ts, f, speed_of_sound=speed_of_sound),
        dtype=np.float64,
    )


def in_situ_velocity_level_difference(
    vibration_reduction_index: ArrayLike,
    *,
    coupling_length: float,
    absorption_length_i: ArrayLike,
    absorption_length_j: ArrayLike,
) -> np.ndarray:
    r"""In-situ velocity level difference ``Dv,ij,situ`` (Formula 10).

    :math:`D_{v,ij,\mathrm{situ}} = K_{ij} - 10 \log_{10}(l_{ij}/\sqrt{a_{i,\mathrm{situ}}
    a_{j,\mathrm{situ}}})`, floored at 0 dB as
    the formula prescribes. It converts the situation-invariant junction
    descriptor ``Kij`` (ISO 12354-1 Annex E, or measured per ISO 10848) into
    the level drop the junction actually produces between the two elements as
    built.

    :param vibration_reduction_index: ``Kij`` per band (or a single value
        broadcast to all bands), in dB.
    :param coupling_length: Common coupling length ``lij``, in m.
    :param absorption_length_i: ``ai,situ`` per band, in m.
    :param absorption_length_j: ``aj,situ`` per band, in m.
    :return: ``Dv,ij,situ`` per band, in dB (never negative).
    :raises ValueError: If a length is not positive/finite or the band counts
        disagree.
    """
    a_i_raw = require_positive_array(absorption_length_i, "absorption_length_i")
    a_j_raw = require_positive_array(absorption_length_j, "absorption_length_j")
    lij = require_positive(coupling_length, "coupling_length")
    n = max(a_i_raw.size, a_j_raw.size)
    a_i = _band_array(a_i_raw, n, "absorption_length_i", positive=True)
    a_j = _band_array(a_j_raw, n, "absorption_length_j", positive=True)
    kij = _band_array(vibration_reduction_index, n, "vibration_reduction_index")
    dv = kij - 10.0 * np.log10(lij / np.sqrt(a_i * a_j))
    return np.asarray(np.maximum(dv, 0.0), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Transmission paths (Clause 4.2.3)
# --------------------------------------------------------------------------- #
def direct_reduction_index(
    separating_index: ArrayLike,
    *,
    delta_r_source: ArrayLike = 0.0,
    delta_r_receiving: ArrayLike = 0.0,
) -> np.ndarray:
    r"""Sound reduction index of the direct path ``RDd`` (Formula 14).

    :math:`R_\mathrm{Dd} = R_\mathrm{s,situ} + \Delta R_\mathrm{D,situ} + \Delta R_\mathrm{d,situ}`: the
    in-situ index of the
    separating element plus the improvement of any lining on its source and
    receiving faces (for the in-situ improvement the standard accepts the
    laboratory value, Formula 8).

    :param separating_index: ``Rs,situ`` per band, in dB.
    :param delta_r_source: ``ΔRD,situ`` on the source side, per band, in dB.
    :param delta_r_receiving: ``ΔRd,situ`` on the receiving side, in dB.
    :return: ``RDd`` per band, in dB.
    """
    r = np.atleast_1d(np.asarray(separating_index, dtype=np.float64))
    return np.asarray(
        r
        + np.asarray(delta_r_source, dtype=np.float64)
        + np.asarray(delta_r_receiving, dtype=np.float64),
        dtype=np.float64,
    )


def flanking_reduction_index(
    *,
    index_i: ArrayLike,
    index_j: ArrayLike,
    velocity_level_difference: ArrayLike,
    separating_area: float,
    area_i: float,
    area_j: float,
    delta_r_i: ArrayLike = 0.0,
    delta_r_j: ArrayLike = 0.0,
) -> np.ndarray:
    r"""Flanking sound reduction index ``Rij`` per band (Formula 15).

    :math:`R_{ij} = R_{i,\mathrm{situ}}/2 + \Delta R_{i,\mathrm{situ}} + R_{j,\mathrm{situ}}/2
    + \Delta R_{j,\mathrm{situ}} + D_{v,ij,\mathrm{situ}} + T`
    for ``ij = Ff, Fd, Df``, with the geometry term
    :math:`T = 10 \log_{10}(S_\mathrm{s}/\sqrt{S_i S_j})`. For diagonal transmission the
    standard fixes :math:`S_\mathrm{s} = 10` m².

    The element indices depend on the path: ``Ff`` takes the flanking element
    on both sides, ``Fd`` the flanking element as ``i`` and the separating
    element as ``j``, and ``Df`` the separating element as ``i`` and the
    flanking element as ``j``.

    :param index_i: ``Ri,situ`` of the element excited in the source room, in
        dB per band.
    :param index_j: ``Rj,situ`` of the radiating element in the receiving
        room, per band, in dB.
    :param velocity_level_difference: ``Dv,ij,situ`` per band, in dB (see
        :func:`in_situ_velocity_level_difference`; for a Type B junction pass
        the Formula (12) value derived from ``Dv,ij,n``).
    :param separating_area: Separating-element area ``Ss``, in m².
    :param area_i: Area ``Si`` of element ``i``, in m².
    :param area_j: Area ``Sj`` of element ``j``, in m².
    :param delta_r_i: ``ΔRi,situ`` on element ``i``, per band, in dB.
    :param delta_r_j: ``ΔRj,situ`` on element ``j``, per band, in dB.
    :return: ``Rij`` per band, in dB.
    :raises ValueError: If an area is not positive and finite.
    """
    ss = require_positive(separating_area, "separating_area")
    si = require_positive(area_i, "area_i")
    sj = require_positive(area_j, "area_j")
    r_i = np.atleast_1d(np.asarray(index_i, dtype=np.float64))
    r_j = np.atleast_1d(np.asarray(index_j, dtype=np.float64))
    dv = np.atleast_1d(np.asarray(velocity_level_difference, dtype=np.float64))
    return np.asarray(
        r_i / 2.0
        + np.asarray(delta_r_i, dtype=np.float64)
        + r_j / 2.0
        + np.asarray(delta_r_j, dtype=np.float64)
        + dv
        + 10.0 * np.log10(ss / np.sqrt(si * sj)),
        dtype=np.float64,
    )


def flanking_reduction_index_from_normalized_difference(
    *,
    index_i: ArrayLike,
    index_j: ArrayLike,
    normalized_velocity_level_difference: ArrayLike,
    separating_area: float,
    coupling_length: float,
    delta_r_i: ArrayLike = 0.0,
    delta_r_j: ArrayLike = 0.0,
) -> np.ndarray:
    r"""Flanking index of a Type B junction ``Rij`` (Formula 17).

    :math:`R_{ij} = R_{i,\mathrm{situ}}/2 + \Delta R_{i,\mathrm{situ}} + R_{j,\mathrm{situ}}/2
    + \Delta R_{j,\mathrm{situ}} + D_{v,ij,\mathrm{n}} + T` with
    the geometry term :math:`T = 10 \log_{10}(S_\mathrm{s}/(l_o l_{ij}))` and the reference
    length :math:`l_o = 1` m. It is
    Formula (15) with Formula (12) substituted, so the junction is described by
    the *normalized* direction-averaged velocity level difference ``Dv,ij,n``
    (ISO 12354-1 Annex F) rather than by ``Kij``: the form used for lightweight
    double-leaf constructions, where the indices refer either to the double
    element as a whole or to its inner leaf and should relate to resonant
    transmission only (see :func:`resonant_sound_reduction_index`).

    :param index_i: ``Ri,situ`` of the element excited in the source room, in
        dB per band.
    :param index_j: ``Rj,situ`` of the radiating element, per band, in dB.
    :param normalized_velocity_level_difference: ``Dv,ij,n`` per band, in dB.
    :param separating_area: Separating-element area ``Ss``, in m².
    :param coupling_length: Common coupling length ``lij``, in m.
    :param delta_r_i: ``ΔRi,situ`` on element ``i``, per band, in dB.
    :param delta_r_j: ``ΔRj,situ`` on element ``j``, per band, in dB.
    :return: ``Rij`` per band, in dB.
    :raises ValueError: If a geometry value is not positive and finite.
    """
    ss = require_positive(separating_area, "separating_area")
    lij = require_positive(coupling_length, "coupling_length")
    r_i = np.atleast_1d(np.asarray(index_i, dtype=np.float64))
    r_j = np.atleast_1d(np.asarray(index_j, dtype=np.float64))
    dv = np.atleast_1d(
        np.asarray(normalized_velocity_level_difference, dtype=np.float64)
    )
    return np.asarray(
        r_i / 2.0
        + np.asarray(delta_r_i, dtype=np.float64)
        + r_j / 2.0
        + np.asarray(delta_r_j, dtype=np.float64)
        + dv
        + 10.0 * np.log10(ss / (REFERENCE_LENGTH * lij)),
        dtype=np.float64,
    )


def flanking_reduction_index_from_flanking_level(
    flanking_level_difference: ArrayLike,
    *,
    separating_area: float,
    coupling_length: float,
    laboratory_coupling_length: float,
    reference_absorption_area: float = 10.0,
) -> np.ndarray:
    r"""Flanking index from a measured ``Dn,f`` (Formula 16).

    :math:`R_{ij} = D_{\mathrm{n,f},ij,\mathrm{situ}} + 10 \log_{10}(S_\mathrm{s} l_\mathrm{lab}/(A_o l_{ij}))` with
    :math:`A_o = 10` m², the
    route used when the flanking construction is characterised as a whole by a
    laboratory measurement of the flanking normalized level difference
    (ISO 10848). ISO 12354-1 Clause 4.4.2 gives the usual laboratory coupling
    lengths: 4,5 m for horizontal flanking elements such as ceilings, 2,5 m
    for vertical ones such as facades.

    :param flanking_level_difference: ``Dn,f,ij,situ`` per band, in dB.
    :param separating_area: Separating-element area ``Ss``, in m².
    :param coupling_length: In-situ coupling length ``lij``, in m.
    :param laboratory_coupling_length: Laboratory coupling length ``llab``, m.
    :param reference_absorption_area: ``Ao``, in m² (Default: 10 m²).
    :return: ``Rij`` per band, in dB.
    :raises ValueError: If a geometry value is not positive and finite.
    """
    ss = require_positive(separating_area, "separating_area")
    lij = require_positive(coupling_length, "coupling_length")
    llab = require_positive(laboratory_coupling_length, "laboratory_coupling_length")
    a0 = require_positive(reference_absorption_area, "reference_absorption_area")
    dnf = np.atleast_1d(np.asarray(flanking_level_difference, dtype=np.float64))
    return np.asarray(dnf + 10.0 * np.log10(ss * llab / (a0 * lij)), dtype=np.float64)


def resonant_sound_reduction_index(
    sound_reduction_index: ArrayLike,
    frequencies: ArrayLike,
    *,
    critical_frequency: float,
    correction: float = 8.0,
) -> np.ndarray:
    r"""Correct a measured ``R`` to resonant transmission only (Formula B.1).

    :math:`R^* = R + 10 \log_{10}(\sigma_\mathrm{a}/\sigma_\mathrm{s})`. No standardized method
    exists to measure the
    two radiation factors, so Annex B.2 gives the estimate this function
    applies: no correction for elements separated by one or two cavities, and
    a fixed correction (8 dB, the standard's figure for single homogeneous or
    layered wood or steel frame elements without a cavity) **below the
    critical frequency only**. Above ``fc`` the laboratory index already
    describes resonant transmission and is returned unchanged.

    :param sound_reduction_index: Measured index ``R`` per band, in dB.
    :param frequencies: Band centre frequencies, in Hz.
    :param critical_frequency: Critical frequency ``fc``, in Hz.
    :param correction: Correction applied below ``fc``, in dB (Default: 8 dB;
        Annex B.2 caps the estimate of Formula (B.8) at this value, and the
        Annex L lightweight example reduces it around the cavity resonance).
    :return: The resonant-only index ``R*`` per band, in dB.
    :raises ValueError: If an input is not positive/finite or the band counts
        disagree.
    """
    f = require_positive_array(frequencies, "frequencies")
    fc = require_positive(critical_frequency, "critical_frequency")
    r = _band_array(sound_reduction_index, f.size, "sound_reduction_index")
    delta = _band_array(correction, f.size, "correction")
    return np.asarray(np.where(f < fc, r + delta, r), dtype=np.float64)


def reciprocity_impact_level(
    sound_reduction_index: ArrayLike,
    frequencies: ArrayLike,
    *,
    bands: BandType = "third",
) -> np.ndarray:
    r"""Impact level of a homogeneous floor by reciprocity (Part 2, B.3/B.4).

    :math:`R + L_\mathrm{n} = 38 + 30 \log_{10}(f/1\,\mathrm{Hz})` in one-third-octave bands
    and
    :math:`R + L_\mathrm{n} = 43 + 30 \log_{10}(f/1\,\mathrm{Hz})` in octave bands: for a
    homogeneous floor
    the sum of the airborne index and the normalized impact level depends only
    on frequency, provided forced transmission is negligible (normally up to
    about 1 kHz, above which the stiffness of the floor's top layer matters).

    :param sound_reduction_index: ``R`` of the floor per band, in dB.
    :param frequencies: Band centre frequencies, in Hz.
    :param bands: ``"third"`` (default, constant 38) or ``"octave"`` (43).
    :return: The normalized impact sound pressure level ``Ln`` per band, dB.
    :raises ValueError: If an input is not positive/finite or the band counts
        disagree.
    """
    f = require_positive_array(frequencies, "frequencies")
    r = _band_array(sound_reduction_index, f.size, "sound_reduction_index")
    constant = _RECIPROCITY_CONSTANT[require_choice(bands, "bands", _BAND_CHOICES)]
    return np.asarray(constant + 30.0 * np.log10(f) - r, dtype=np.float64)


def direct_impact_level(
    floor_level: ArrayLike,
    *,
    delta_l: ArrayLike = 0.0,
    delta_l_ceiling: ArrayLike = 0.0,
) -> np.ndarray:
    r"""Normalized impact level of the direct path ``Ln,d`` (Part 2, F. 11).

    :math:`L_\mathrm{n,d} = L_\mathrm{n,situ} - \Delta L_\mathrm{situ} - \Delta L_\mathrm{d,situ}`: the
    in-situ level of the bare
    floor reduced by the floor covering and by any additional layer on the
    receiving side (a suspended ceiling).

    :param floor_level: ``Ln,situ`` of the bare separating floor, per band, dB.
    :param delta_l: Improvement of the floor covering ``ΔLsitu``, per band, dB.
    :param delta_l_ceiling: Improvement ``ΔLd,situ`` of a layer on the
        receiving side, per band, in dB.
    :return: ``Ln,d`` per band, in dB.
    """
    ln = np.atleast_1d(np.asarray(floor_level, dtype=np.float64))
    return np.asarray(
        ln
        - np.asarray(delta_l, dtype=np.float64)
        - np.asarray(delta_l_ceiling, dtype=np.float64),
        dtype=np.float64,
    )


def flanking_impact_level(
    *,
    floor_level: ArrayLike,
    index_i: ArrayLike,
    index_j: ArrayLike,
    velocity_level_difference: ArrayLike,
    area_i: float,
    area_j: float,
    delta_l: ArrayLike = 0.0,
    delta_r_j: ArrayLike = 0.0,
) -> np.ndarray:
    r"""Flanking normalized impact level ``Ln,ij`` per band (Part 2, F. 12).

    :math:`L_{\mathrm{n},ij} = L_\mathrm{n,situ} - \Delta L_\mathrm{situ}
    + (R_{i,\mathrm{situ}} - R_{j,\mathrm{situ}})/2 - \Delta R_{j,\mathrm{situ}} - D_{v,ij,\mathrm{situ}} - T`
    with the geometry term :math:`T = 10 \log_{10}(S_i/\sqrt{S_i S_j})`, ``i`` the
    excited floor
    and ``j`` the flanking element radiating in the receiving room.

    :param floor_level: ``Ln,situ`` of the excited floor, per band, in dB.
    :param index_i: ``Ri,situ`` of the excited floor, per band, in dB.
    :param index_j: ``Rj,situ`` of the flanking element, per band, in dB.
    :param velocity_level_difference: ``Dv,ij,situ`` per band, in dB.
    :param area_i: Area ``Si`` of the excited floor, in m².
    :param area_j: Area ``Sj`` of the flanking element, in m².
    :param delta_l: Improvement of the floor covering ``ΔLsitu``, in dB.
    :param delta_r_j: Improvement ``ΔRj,situ`` of a lining on the flanking
        element, per band, in dB.
    :return: ``Ln,ij`` per band, in dB.
    :raises ValueError: If an area is not positive and finite.
    """
    si = require_positive(area_i, "area_i")
    sj = require_positive(area_j, "area_j")
    ln = np.atleast_1d(np.asarray(floor_level, dtype=np.float64))
    r_i = np.atleast_1d(np.asarray(index_i, dtype=np.float64))
    r_j = np.atleast_1d(np.asarray(index_j, dtype=np.float64))
    dv = np.atleast_1d(np.asarray(velocity_level_difference, dtype=np.float64))
    return np.asarray(
        ln
        - np.asarray(delta_l, dtype=np.float64)
        + (r_i - r_j) / 2.0
        - np.asarray(delta_r_j, dtype=np.float64)
        - dv
        - 10.0 * np.log10(si / np.sqrt(si * sj)),
        dtype=np.float64,
    )


def flanking_impact_level_from_normalized_difference(
    *,
    floor_level: ArrayLike,
    index_i: ArrayLike,
    index_j: ArrayLike,
    normalized_velocity_level_difference: ArrayLike,
    area_i: float,
    coupling_length: float,
    delta_l: ArrayLike = 0.0,
    delta_r_j: ArrayLike = 0.0,
) -> np.ndarray:
    r"""Flanking impact level of a Type B junction (Part 2, Formula 14).

    :math:`L_{\mathrm{n},ij} = L_{\mathrm{n},ii} - \Delta L_i + (R_i - R_j)/2 - \Delta R_j
    - D_{v,ij,\mathrm{n}} - 10 \log_{10}(S_i/(l_o l_{ij}))`
    with the reference length :math:`l_o = 1` m: Formula (12) with the
    junction described by the
    normalized direction-averaged velocity level difference instead of
    ``Kij``, the form used for lightweight constructions.

    :param floor_level: ``Ln,ii`` of the excited bare floor, per band, in dB.
    :param index_i: ``Ri`` of the excited floor, per band, in dB.
    :param index_j: ``Rj`` of the flanking element, per band, in dB.
    :param normalized_velocity_level_difference: ``Dv,ij,n`` per band, in dB.
    :param area_i: Area ``Si`` of the excited floor, in m².
    :param coupling_length: Common coupling length ``lij``, in m.
    :param delta_l: Improvement of the floor covering ``ΔLi``, per band, dB.
    :param delta_r_j: Improvement ``ΔRj`` of a lining on the flanking element,
        per band, in dB.
    :return: ``Ln,ij`` per band, in dB.
    :raises ValueError: If a geometry value is not positive and finite.
    """
    si = require_positive(area_i, "area_i")
    lij = require_positive(coupling_length, "coupling_length")
    ln = np.atleast_1d(np.asarray(floor_level, dtype=np.float64))
    r_i = np.atleast_1d(np.asarray(index_i, dtype=np.float64))
    r_j = np.atleast_1d(np.asarray(index_j, dtype=np.float64))
    dv = np.atleast_1d(
        np.asarray(normalized_velocity_level_difference, dtype=np.float64)
    )
    return np.asarray(
        ln
        - np.asarray(delta_l, dtype=np.float64)
        + (r_i - r_j) / 2.0
        - np.asarray(delta_r_j, dtype=np.float64)
        - dv
        - 10.0 * np.log10(si / (REFERENCE_LENGTH * lij)),
        dtype=np.float64,
    )


def flanking_impact_level_from_flanking_level(
    normalized_flanking_impact_level: ArrayLike,
    *,
    area: float,
    laboratory_area: float,
    coupling_length: float,
    laboratory_coupling_length: float,
) -> np.ndarray:
    r"""Flanking impact level from a measured ``Ln,f`` (Part 2, Formula 13).

    :math:`L_{\mathrm{n},ij} = L_{\mathrm{n,f},ij,\mathrm{situ}} - 10 \log_{10}(S_i l_\mathrm{lab}/(S_{i,\mathrm{lab}}
    l_{ij}))`, the impact twin of
    the airborne :func:`flanking_reduction_index_from_flanking_level`: the
    route used when the flanking construction is characterised as a whole by a
    laboratory measurement of the normalized flanking impact sound pressure
    level (ISO 10848) instead of by the properties of its elements. The
    laboratory measurement is transferred to the field situation first, as
    ISO 12354-2:2017, Annex D indicates.

    :param normalized_flanking_impact_level: ``Ln,f,ij,situ`` per band, in dB.
    :param area: In-situ area ``Si`` of the excited floor, in m².
    :param laboratory_area: Laboratory area ``Si,lab`` of the excited floor,
        in m².
    :param coupling_length: In-situ coupling length ``lij``, in m.
    :param laboratory_coupling_length: Laboratory coupling length ``llab``,
        in m. ISO 12354-1 Clause 4.4.2 gives the usual values: 4,5 m for
        horizontal flanking elements, 2,5 m for vertical ones.
    :return: ``Ln,ij`` per band, in dB.
    :raises ValueError: If a geometry value is not positive and finite.
    """
    si = require_positive(area, "area")
    si_lab = require_positive(laboratory_area, "laboratory_area")
    lij = require_positive(coupling_length, "coupling_length")
    llab = require_positive(laboratory_coupling_length, "laboratory_coupling_length")
    ln = np.atleast_1d(
        np.asarray(normalized_flanking_impact_level, dtype=np.float64)
    )
    return np.asarray(
        ln - 10.0 * np.log10(si * llab / (si_lab * lij)), dtype=np.float64
    )


def floating_floor_improvement(
    frequencies: ArrayLike,
    *,
    resonance_frequency: float,
    slope: float = 30.0,
) -> np.ndarray:
    r"""Improvement of a floating floor ``ΔL`` per band (Part 2, Formula C.1).

    :math:`\Delta L = 30 \log_{10}(f/f_\mathrm{o})` for sand/cement or calcium-sulfate
    screeds and
    :math:`\Delta L = 40 \log_{10}(f/f_\mathrm{o})` (``slope=40``, Formula C.3) for asphalt
    or dry
    floating floors, with the system resonance
    :math:`f_\mathrm{o} = 160 \sqrt{s'/m'}`
    (Formula C.2) and no improvement at or below it. The Annex L airborne
    example reuses the same curve as ``ΔR``, noting explicitly that assuming
    :math:`\Delta R = \Delta L` is rough.

    :param frequencies: Band centre frequencies ``f``, in Hz.
    :param resonance_frequency: Resonance frequency ``fo``, in Hz.
    :param slope: 30 (screed, Formula C.1) or 40 (asphalt/dry, Formula C.3).
    :return: The improvement ``ΔL`` per band, in dB (0 at and below ``fo``).
    :raises ValueError: If an input is not positive and finite.
    """
    f = require_positive_array(frequencies, "frequencies")
    f0 = require_positive(resonance_frequency, "resonance_frequency")
    require_positive(slope, "slope")
    return np.asarray(
        np.where(f > f0, slope * np.log10(f / f0), 0.0), dtype=np.float64
    )


# --------------------------------------------------------------------------- #
# Element description and its in-situ evaluation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HomogeneousElement:
    r"""A Type A homogeneous element of the detailed model.

    :ivar label: Human-readable element name, e.g. ``"separating floor"``.
    :ivar area: Element area ``S``, in m².
    :ivar length1: One side length of the rectangular element, in m.
    :ivar length2: The other side length, in m.
    :ivar mass_per_area: Mass per unit area ``m'``, in kg/m².
    :ivar critical_frequency: Critical frequency ``fc``, in Hz.
    :ivar internal_loss_factor: Internal loss factor ``ηint`` of the material
        (about 0,01 for common homogeneous building materials; ISO 12354-1
        Table B.3 tabulates it per material).
    :ivar perimeter_absorption: :math:`\sum l_k \alpha_k` over the element's
        perimeter, in m (Formula C.1; build it from
        :func:`perimeter_absorption_coefficient` times the border lengths).
    :ivar density: Density ``ρ`` of the material, in kg/m³; supplied together
        with ``longitudinal_velocity`` it enables the high-frequency plateau
        of Formula (B.10). ``None`` (the default) leaves the plateau off.
    :ivar longitudinal_velocity: Quasi-longitudinal phase velocity ``cL`` of
        the material, in m/s (ISO 12354-1 Table B.3).
    """

    label: str
    area: float
    length1: float
    length2: float
    mass_per_area: float
    critical_frequency: float
    internal_loss_factor: float = 0.01
    perimeter_absorption: float = 0.0
    density: float | None = None
    longitudinal_velocity: float | None = None


@dataclass(frozen=True)
class InSituElementResult:
    """Per-band in-situ description of one element (Clause 4.2.2).

    :ivar label: The element name.
    :ivar frequencies: Band centre frequencies, in Hz.
    :ivar area: Element area ``S``, in m².
    :ivar radiation_factor: Radiation factor for free bending waves ``σ``.
    :ivar forced_radiation_factor: Radiation factor for forced waves ``σf``.
    :ivar total_loss_factor: In-situ total loss factor ``ηtot,situ``.
    :ivar reverberation_time: In-situ structural reverberation time
        ``Ts,situ``, in s.
    :ivar absorption_length: In-situ equivalent absorption length ``asitu``,
        in m.
    :ivar sound_reduction_index: In-situ sound reduction index ``Rsitu``, dB.
    :ivar impact_level: In-situ normalized impact level ``Ln,situ`` of the
        bare element, in dB (meaningful for the excited floor).
    """

    label: str
    frequencies: np.ndarray
    area: float
    radiation_factor: np.ndarray
    forced_radiation_factor: np.ndarray
    total_loss_factor: np.ndarray
    reverberation_time: np.ndarray
    absorption_length: np.ndarray
    sound_reduction_index: np.ndarray
    impact_level: np.ndarray

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the in-situ ``Rsitu`` and ``Ln,situ`` spectra of the element.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_in_situ_element

        check_language(language)
        return plot_in_situ_element(self, ax=ax, language=language, **kwargs)


def in_situ_element(
    element: HomogeneousElement,
    frequencies: ArrayLike,
    *,
    bands: BandType = "third",
    resonant_only: bool = False,
    speed_of_sound: float = SPEED_OF_SOUND,
    air_density: float = AIR_DENSITY,
) -> InSituElementResult:
    r"""Evaluate one homogeneous element in situ, per band (Clause 4.2.2).

    Runs the whole Annex B / Annex C chain in one call: the two radiation
    factors, the in-situ total loss factor and structural reverberation time,
    the equivalent absorption length, the calculated in-situ sound reduction
    index and, for a floor, the calculated in-situ normalized impact level.

    Because the element performance is *calculated from material properties*,
    the in-situ loss factor enters Formula (B.2) directly and no
    :math:`10 \log_{10}(T_\mathrm{s,situ}/T_\mathrm{s,lab})` transfer is needed
    (Annex B.3). Use :func:`in_situ_reduction_index` instead when the element
    data come from a laboratory measurement.

    :param element: The :class:`HomogeneousElement` description.
    :param frequencies: Band centre frequencies, in Hz.
    :param bands: ``"third"`` (default) or ``"octave"``.
    :param resonant_only: Drop the forced-transmission term of Formula (B.2)
        below ``fc`` (Annex B.1, flanking paths).
    :param speed_of_sound: Speed of sound in air ``co``, in m/s.
    :param air_density: Density of air ``ρo``, in kg/m³.
    :return: The :class:`InSituElementResult`.
    :raises ValueError: If any element property is not positive and finite.
    """
    f = require_positive_array(frequencies, "frequencies")
    sigma = bending_radiation_factor(
        f,
        critical_frequency=element.critical_frequency,
        length1=element.length1,
        length2=element.length2,
        speed_of_sound=speed_of_sound,
    )
    sigma_f = forced_radiation_factor(
        f,
        length1=element.length1,
        length2=element.length2,
        speed_of_sound=speed_of_sound,
    )
    eta = in_situ_total_loss_factor(
        f,
        internal_loss_factor=element.internal_loss_factor,
        mass_per_area=element.mass_per_area,
        area=element.area,
        critical_frequency=element.critical_frequency,
        radiation_factor=sigma,
        perimeter_absorption=element.perimeter_absorption,
        speed_of_sound=speed_of_sound,
        air_density=air_density,
    )
    ts = structural_reverberation_time(f, eta)
    r = calculated_sound_reduction_index(
        f,
        mass_per_area=element.mass_per_area,
        critical_frequency=element.critical_frequency,
        total_loss_factor=eta,
        radiation_factor=sigma,
        forced_radiation_factor=sigma_f,
        bands=bands,
        resonant_only=resonant_only,
        density=element.density,
        longitudinal_velocity=element.longitudinal_velocity,
        speed_of_sound=speed_of_sound,
        air_density=air_density,
    )
    ln = bare_floor_impact_level(
        f,
        mass_per_area=element.mass_per_area,
        structural_reverberation_time=ts,
        radiation_factor=sigma,
    )
    a_situ = in_situ_equivalent_absorption_length(
        f,
        area=element.area,
        situ_reverberation_time=ts,
        speed_of_sound=speed_of_sound,
    )
    return InSituElementResult(
        label=element.label,
        frequencies=f,
        area=require_positive(element.area, "area"),
        radiation_factor=sigma,
        forced_radiation_factor=sigma_f,
        total_loss_factor=eta,
        reverberation_time=ts,
        absorption_length=a_situ,
        sound_reduction_index=r,
        impact_level=ln,
    )


# --------------------------------------------------------------------------- #
# Path assembly and totals (Clause 4.1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BandPath:
    """One transmission path of the detailed model, per band.

    :ivar label: Human-readable path name, e.g. ``"ext wall 1-Df"``.
    :ivar kind: ``"Dd"``, ``"Ff"``, ``"Df"`` or ``"Fd"``.
    :ivar values: ``Rij`` (airborne) or ``Ln,ij`` (impact) per band, in dB.
    """

    label: str
    kind: str
    values: np.ndarray


def _path_matrix(paths: Sequence[BandPath], n_bands: int) -> np.ndarray:
    """Stack the paths' per-band values into a ``paths x bands`` matrix."""
    return np.vstack(
        [
            _band_array(p.values, n_bands, f"path {p.label!r}")
            for p in paths
        ]
    )


def airborne_flanking_path(
    *,
    label: str,
    kind: FlankingKind,
    element_i: InSituElementResult,
    element_j: InSituElementResult,
    vibration_reduction_index: ArrayLike,
    coupling_length: float,
    separating_area: float,
    delta_r_i: ArrayLike = 0.0,
    delta_r_j: ArrayLike = 0.0,
) -> BandPath:
    """Build one airborne flanking path from two in-situ elements (Formula 15).

    The junction velocity level difference is formed from the two elements'
    equivalent absorption lengths with Formula (10), then Formula (15) gives
    ``Rij`` per band.

    :param label: Human-readable path name.
    :param kind: ``"Ff"``, ``"Df"`` or ``"Fd"``.
    :param element_i: The element excited in the source room.
    :param element_j: The element radiating in the receiving room.
    :param vibration_reduction_index: ``Kij`` of this path (per band or a
        single value), in dB.
    :param coupling_length: Common coupling length ``lij``, in m.
    :param separating_area: Separating-element area ``Ss``, in m².
    :param delta_r_i: ``ΔRi,situ`` on element ``i``, per band, in dB.
    :param delta_r_j: ``ΔRj,situ`` on element ``j``, per band, in dB.
    :return: The :class:`BandPath` carrying ``Rij``.
    :raises ValueError: If ``kind`` is unknown or a geometry value is not
        positive.
    """
    require_choice(kind, "kind", ("Ff", "Df", "Fd"))
    dv = in_situ_velocity_level_difference(
        vibration_reduction_index,
        coupling_length=coupling_length,
        absorption_length_i=element_i.absorption_length,
        absorption_length_j=element_j.absorption_length,
    )
    values = flanking_reduction_index(
        index_i=element_i.sound_reduction_index,
        index_j=element_j.sound_reduction_index,
        velocity_level_difference=dv,
        separating_area=separating_area,
        area_i=element_i.area,
        area_j=element_j.area,
        delta_r_i=delta_r_i,
        delta_r_j=delta_r_j,
    )
    return BandPath(label=label, kind=kind, values=values)


def impact_flanking_path(
    *,
    label: str,
    floor: InSituElementResult,
    element_j: InSituElementResult,
    vibration_reduction_index: ArrayLike,
    coupling_length: float,
    delta_l: ArrayLike = 0.0,
    delta_r_j: ArrayLike = 0.0,
) -> BandPath:
    """Build one impact flanking path ``Df`` (Part 2, Formula 12).

    :param label: Human-readable path name.
    :param floor: The excited separating floor, in situ.
    :param element_j: The flanking element radiating in the receiving room.
    :param vibration_reduction_index: ``Kij`` of this path, in dB.
    :param coupling_length: Common coupling length ``lij``, in m.
    :param delta_l: Improvement of the floor covering ``ΔLsitu``, in dB.
    :param delta_r_j: ``ΔRj,situ`` of a lining on the flanking element, in dB.
    :return: The :class:`BandPath` carrying ``Ln,ij``.
    :raises ValueError: If a geometry value is not positive and finite.
    """
    dv = in_situ_velocity_level_difference(
        vibration_reduction_index,
        coupling_length=coupling_length,
        absorption_length_i=floor.absorption_length,
        absorption_length_j=element_j.absorption_length,
    )
    values = flanking_impact_level(
        floor_level=floor.impact_level,
        index_i=floor.sound_reduction_index,
        index_j=element_j.sound_reduction_index,
        velocity_level_difference=dv,
        area_i=floor.area,
        area_j=element_j.area,
        delta_l=delta_l,
        delta_r_j=delta_r_j,
    )
    return BandPath(label=label, kind="Df", values=values)


@dataclass(frozen=True)
class DetailedAirborneResult:
    """Per-band apparent sound reduction index ``R'`` (ISO 12354-1, 4.2).

    :ivar frequencies: Band centre frequencies, in Hz.
    :ivar paths: Every transmission path (the direct path first), each with
        its ``Rij`` per band.
    :ivar r_prime: Apparent sound reduction index ``R'`` per band, in dB.
    :ivar fractions: Share of the transmitted energy carried by each path per
        band (paths x bands), summing to 1 in every band.
    :ivar rating: ``R'w (C; Ctr)`` per ISO 717-1, or ``None`` when the bands
        supplied do not cover the rating range.
    """

    frequencies: np.ndarray
    paths: tuple[BandPath, ...]
    r_prime: np.ndarray
    fractions: np.ndarray
    rating: WeightedRatingResult | None

    @property
    def dominant(self) -> tuple[str, ...]:
        """Label of the path carrying most energy in each band."""
        return tuple(self.paths[k].label for k in np.argmax(self.fractions, axis=0))

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the per-band path contributions and the resulting ``R'``.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_detailed_airborne_prediction

        check_language(language)
        return plot_detailed_airborne_prediction(
            self, ax=ax, language=language, **kwargs
        )

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render a detailed airborne prediction fiche (EN/ISO 12354-1).

        Writes a one-page **prediction** report for the per-band detailed
        model: a basis line naming ISO 12354-1:2017 Clause 4.2, an optional
        metadata header, a two-panel body with the per-path energy-share table
        beside the per-band path-contribution figure, the boxed predicted
        ``R'w``, the prediction statement and, when a requirement is supplied,
        a PASS/FAIL verdict, followed by the footer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` the path table also gives the band in
            which each path contributes most.
        :param language: ``"en"`` (default) or ``"es"``.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine``/``language`` is unknown or the
            result carries no ISO 717-1 rating.
        :raises ImportError: If reportlab or matplotlib is missing.
        """
        _check_report_request(engine, language)
        from ..._report.iso12354 import render_iso12354_detailed_airborne_report

        return render_iso12354_detailed_airborne_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


@dataclass(frozen=True)
class DetailedImpactResult:
    """Per-band apparent impact level ``L'n`` (ISO 12354-2, 4.2).

    :ivar frequencies: Band centre frequencies, in Hz.
    :ivar paths: Every transmission path (the direct path first), each with
        its ``Ln,ij`` per band.
    :ivar l_prime_n: Apparent normalized impact level ``L'n`` per band, in dB.
    :ivar fractions: Share of the radiated energy carried by each path per
        band (paths x bands), summing to 1 in every band.
    :ivar rating: ``L'n,w (CI)`` per ISO 717-2, or ``None`` when the bands
        supplied do not cover the rating range.
    """

    frequencies: np.ndarray
    paths: tuple[BandPath, ...]
    l_prime_n: np.ndarray
    fractions: np.ndarray
    rating: ImpactRatingResult | None

    @property
    def dominant(self) -> tuple[str, ...]:
        """Label of the path carrying most energy in each band."""
        return tuple(self.paths[k].label for k in np.argmax(self.fractions, axis=0))

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the per-band path contributions and the resulting ``L'n``.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_detailed_impact_prediction

        check_language(language)
        return plot_detailed_impact_prediction(
            self, ax=ax, language=language, **kwargs
        )

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render a detailed impact prediction fiche (EN/ISO 12354-2).

        The impact counterpart of :meth:`DetailedAirborneResult.report`: the
        per-band detailed model of ISO 12354-2:2017 Clause 4.2, with the boxed
        predicted ``L'n,w`` and a PASS/FAIL verdict against a requirement (a
        lower level passing).

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` the path table also gives the band in
            which each path contributes most.
        :param language: ``"en"`` (default) or ``"es"``.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine``/``language`` is unknown or the
            result carries no ISO 717-2 rating.
        :raises ImportError: If reportlab or matplotlib is missing.
        """
        _check_report_request(engine, language)
        from ..._report.iso12354 import render_iso12354_detailed_impact_report

        return render_iso12354_detailed_impact_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def detailed_airborne_prediction(
    frequencies: ArrayLike,
    *,
    direct_index: ArrayLike,
    flanking_paths: Sequence[BandPath] = (),
    direct_label: str = "Dd",
    bands: BandType = "third",
) -> DetailedAirborneResult:
    r"""Combine direct and flanking paths into ``R'`` per band (F. 1 to 4).

    :math:`R' = -10 \log_{10}(\sum 10^{-R/10})` over the direct path ``RDd`` and
    every
    flanking path ``Rij``. The result exposes each path's share of the
    transmitted energy in every band, which is what identifies the path to
    treat first, and the ISO 717-1 rating of the resulting spectrum whenever
    the bands cover the rating range (100 Hz to 3150 Hz in one-third octaves,
    125 Hz to 2000 Hz in octaves).

    :param frequencies: Band centre frequencies, in Hz.
    :param direct_index: ``RDd`` per band, in dB (see
        :func:`direct_reduction_index`).
    :param flanking_paths: The flanking paths (see
        :func:`airborne_flanking_path`); may be empty.
    :param direct_label: Label of the direct path (Default: ``"Dd"``).
    :param bands: ``"third"`` (default) or ``"octave"``, selecting the
        ISO 717-1 rating range.
    :return: The :class:`DetailedAirborneResult`.
    :raises ValueError: If a path does not match the band count.
    """
    from ..measurement.insulation import weighted_rating

    f = require_positive_array(frequencies, "frequencies")
    direct = BandPath(
        label=direct_label,
        kind="Dd",
        values=_band_array(direct_index, f.size, "direct_index"),
    )
    paths = (direct, *flanking_paths)
    tau = 10.0 ** (-_path_matrix(paths, f.size) / 10.0)
    total = np.sum(tau, axis=0)
    r_prime = np.asarray(-10.0 * np.log10(total), dtype=np.float64)
    rating_values = _rating_slice(f, r_prime, bands)
    return DetailedAirborneResult(
        frequencies=f,
        paths=paths,
        r_prime=r_prime,
        fractions=np.asarray(tau / total, dtype=np.float64),
        rating=None if rating_values is None else weighted_rating(rating_values),
    )


def detailed_impact_prediction(
    frequencies: ArrayLike,
    *,
    direct_level: ArrayLike | None = None,
    flanking_paths: Sequence[BandPath] = (),
    direct_label: str = "Dd",
    bands: BandType = "third",
) -> DetailedImpactResult:
    r"""Combine direct and flanking paths into ``L'n`` per band (Part 2, (1)).

    :math:`L'_\mathrm{n} = 10 \log_{10}(\sum 10^{L_\mathrm{n}/10})` over the direct impact path
    ``Ln,d`` and
    every flanking path ``Ln,ij``, with the ISO 717-2 rating of the resulting
    spectrum whenever the bands cover the rating range. For rooms next to each
    other there is no direct impact path and the sum runs over the flanking
    paths only (Part 2, Formula 2): leave ``direct_level`` out and the result
    carries no direct path at all.

    :param frequencies: Band centre frequencies, in Hz.
    :param direct_level: ``Ln,d`` per band, in dB (see
        :func:`direct_impact_level`), or ``None`` for the Formula (2) case of
        two rooms next to each other, which has no direct path.
    :param flanking_paths: The flanking paths (see
        :func:`impact_flanking_path`); may be empty when ``direct_level`` is
        given.
    :param direct_label: Label of the direct path (Default: ``"Dd"``).
    :param bands: ``"third"`` (default) or ``"octave"``.
    :return: The :class:`DetailedImpactResult`.
    :raises ValueError: If a path does not match the band count, or if neither
        a direct level nor any flanking path is given.
    """
    from ..measurement.insulation import weighted_impact_rating

    f = require_positive_array(frequencies, "frequencies")
    if direct_level is None and not flanking_paths:
        raise ValueError(
            "'detailed_impact_prediction' needs a 'direct_level', at least one "
            "flanking path, or both."
        )
    paths: tuple[BandPath, ...] = tuple(flanking_paths)
    if direct_level is not None:
        direct = BandPath(
            label=direct_label,
            kind="Dd",
            values=_band_array(direct_level, f.size, "direct_level"),
        )
        paths = (direct, *paths)
    energy = 10.0 ** (_path_matrix(paths, f.size) / 10.0)
    total = np.sum(energy, axis=0)
    l_prime_n = np.asarray(10.0 * np.log10(total), dtype=np.float64)
    rating_values = _rating_slice(f, l_prime_n, bands)
    return DetailedImpactResult(
        frequencies=f,
        paths=paths,
        l_prime_n=l_prime_n,
        fractions=np.asarray(energy / total, dtype=np.float64),
        rating=(
            None if rating_values is None else weighted_impact_rating(rating_values)
        ),
    )


__all__ = [
    "BandPath",
    "DetailedAirborneResult",
    "DetailedImpactResult",
    "HomogeneousElement",
    "InSituElementResult",
    "airborne_flanking_path",
    "bare_floor_impact_level",
    "bending_radiation_factor",
    "calculated_sound_reduction_index",
    "detailed_airborne_prediction",
    "detailed_impact_prediction",
    "direct_impact_level",
    "direct_reduction_index",
    "flanking_impact_level",
    "flanking_impact_level_from_flanking_level",
    "flanking_impact_level_from_normalized_difference",
    "flanking_reduction_index",
    "flanking_reduction_index_from_flanking_level",
    "flanking_reduction_index_from_normalized_difference",
    "floating_floor_improvement",
    "forced_radiation_factor",
    "impact_flanking_path",
    "in_situ_element",
    "in_situ_equivalent_absorption_length",
    "in_situ_impact_level",
    "in_situ_reduction_index",
    "in_situ_total_loss_factor",
    "in_situ_velocity_level_difference",
    "laboratory_total_loss_factor",
    "perimeter_absorption_coefficient",
    "reciprocity_impact_level",
    "resonant_sound_reduction_index",
    "structural_reverberation_time",
]
