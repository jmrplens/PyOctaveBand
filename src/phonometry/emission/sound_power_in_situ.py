#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Sound power and sound energy levels of a noise source determined in situ
by comparison with a reference sound source: ISO 3747:2010 (engineering
grade 2 and survey grade 3).

The source stays where it works. A calibrated reference sound source (RSS)
of known octave-band sound power ``LW(RSS)`` is set beside it and the same
three or four microphone positions listen to each source in turn, in the
part of the room where the field is reverberant, that is where the excess of
sound pressure level over the free field, :math:`\Delta L_f`, is at least
7 dB (clause 4.1, Annex A). Both sources then see the same room and the room
drops out of the algebra: the sound power level of the source under test
(ST) in each octave band is the calibrated power of the RSS carried across
by the difference of the two mean corrected levels (clause 8.3.1),

.. math::

   L_W = L_{W(\mathrm{RSS})} - \overline{L_{p(\mathrm{RSS})}}
   + \overline{L_{p(\mathrm{ST})}} \tag{Eq. 11}

where the mean corrected levels are the energy averages over the ``n``
microphone positions (Eq. 8, 9) of the levels corrected position by
position for background noise (clause 8.1),

.. math::

   K_{1i} = -10 \log_{10}\!\left(1 - 10^{-0.1\,\Delta L_{pi}}\right),
   \qquad \Delta L_{pi} = L'_{pi(\mathrm{ST})} - L_{pi(\mathrm{B})} \tag{Eq. 7}

with three rules around it: a margin above 15 dB needs no correction, a
margin between 6 dB and 15 dB takes Eq. (7), and a margin below 6 dB caps
the correction at 1,3 dB and turns the band into an upper bound that the
report must flag as not meeting the background requirement. A determination
that carries no background reading at all cannot meet that requirement
either, since 8.1 declares a measurement valid only where the margin is at
least 6 dB and 7.5 has the background obtained once at each position. When the RSS is
run at ``m`` locations around a large source the calibrated powers and the
per-location means are each energy-averaged over the locations before the
subtraction (clause 8.3.2, Eq. 12).

An impulsive source is described by its sound energy level instead. The
single event levels measured at each position, either ``N`` events one at a
time (Eq. 13, 15) or one measurement encompassing ``N`` events (Eq. 16, 17),
are background-corrected with the same rule (Eq. 14), reduced to the level
of one event and averaged over positions (Eq. 18); Eq. (19) and (20) are
then Eq. (11) and (12) with :math:`\overline{L_{E(\mathrm{ST})}}` in place of
:math:`\overline{L_{p(\mathrm{ST})}}` (clause 8.5). Eq. (14) subtracts the
time-averaged background level from a time-integrated event level, exactly
as ISO 3741:2010 (9.2.2) and ISO 3744:2010 (8.3.4) print it; the text only
asks that both be measured over the same integration time ``T``. As printed
the difference is a true signal-to-background margin for ``T`` = 1 s; the
optional ``integration_time`` carries the background to the event's interval
(:math:`+10 \log_{10}(T/T_0)`, clause 3.4 NOTE 1) before the subtraction.

Annex C carries either level to the reference meteorological conditions of
101,325 kPa and 23,0 °C with the radiation-impedance correction
:math:`C_2 = -10 \log_{10}(p_\mathrm{s}/p_{\mathrm{s},0}) + 15 \log_{10}((273.15 + \theta)/296)`,
the same ``C2`` as ISO 3741:2010 clause 9.1.4, reused from that module; the
whole ISO 3740 family prints :math:`\theta_\mathrm{ref}` = 296 K beside a
23,0 °C reference, so at the reference conditions ``C2`` is +0,003 3 dB
rather than zero. Eq. (C.2) estimates the static pressure from the altitude
of the site. Annex D forms the A-weighted totals from the Table D.1 band
corrections, which are the ISO 3744 Annex E octave values digit for digit.

Clause 9 estimates the uncertainty as :math:`\sigma_\mathrm{tot} =
\sqrt{\sigma_{R0}^2 + \sigma_\mathrm{omc}^2}` (Eq. 22) and
:math:`U = k\,\sigma_\mathrm{tot}` (Eq. 23), with the typical upper bound of
the reproducibility :math:`\sigma_{R0}` read from Table 2 by grade: 1,5 dB
for grade 2, which needs :math:`\Delta L_{f\mathrm{A}} \ge 7` dB at every
microphone position and a source directivity range within ±7 dB, and 4,0 dB
for grade 3 otherwise. Table 1, the zoning of the test environment by lines
of sight, is normative (7.4.2 divides the environment into zones with a
*shall*, and 7.4.3 draws the microphone positions from them), but it
constrains where the sources and microphones go, not what is computed from
them: the positions arrive here already chosen, so the zoning is not
evaluated.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

from .._internal.levels_math import energy_mean, energy_sum
from .._internal.validation import (
    _as_float64,
    require_choice,
    require_equal_counts,
    require_positive,
    require_ranks,
    require_same_length,
)
from ._shared import (
    _CK_OCTAVE,
    _PS0,
    Grade,
    SoundPowerWarning,
    _a_weighting_corrections,
    _c2_correction,
    _validate_meteorology,
)

#: Background margin at or above which the measurement is valid (ISO 3747:2010, 8.1).
_K1_VALID_DB = 6.0
#: Background margin above which the background is negligible and K1 = 0 (8.1).
_K1_NEGLIGIBLE_DB = 15.0
#: Largest background correction the standard allows, in decibels (8.1).
_K1_MAX_DB = 1.3
#: Excess of sound pressure level that marks the reverberant region, in
#: decibels (4.1, 7.4.1, Table 2).
_EXCESS_LEVEL_MIN_DB = 7.0
#: Source directivity range above which grade 2 cannot be reported, in
#: decibels (7.2, Table 2).
_DIRECTIVITY_RANGE_MAX_DB = 7.0
#: Typical upper bound of the standard deviation of reproducibility of the
#: method per grade of accuracy, in decibels (Table 2).
_SIGMA_R0_DB: dict[Grade, float] = {"engineering": 1.5, "survey": 4.0}
#: Coverage factor of the two-sided 95 % interval, normal distribution (9.5).
_COVERAGE_TWO_SIDED = 2.0
#: Reference distance of Eq. (A.1), in metres.
_R0_M = 1.0
#: Spherical-spreading constant of Eq. (A.1), in decibels.
_SPHERICAL_CONSTANT_DB = 11.0
#: Altitude coefficients of Eq. (C.2): ``a`` in reciprocal metres, ``b``
#: dimensionless (Annex C).
_ALTITUDE_A_PER_M = 2.2560e-5
_ALTITUDE_B = 5.2553
#: Reference duration of the single event level, in seconds (3.4, NOTE 1).
_T0_S = 1.0
#: Fewest and most microphone positions the procedure uses (7.4.1: "in total,
#: three or four microphone positions are to be used").
_MIN_POSITIONS = 3
_MAX_POSITIONS = 4
#: Fewest single events of an impulsive determination (7.6: N >= 5).
_MIN_EVENTS = 5


@dataclass(frozen=True)
class InSituSoundPowerResult:
    r"""Result of an ISO 3747:2010 in situ determination by comparison.

    ``quantity`` says which of the two determinations this is: ``'power'``
    carries the octave-band sound power level ``LW`` (Eq. 11 or 12) in
    ``sound_power_level`` with ``sound_energy_level`` all ``NaN``, and
    ``'energy'`` the sound energy level ``LJ`` (Eq. 19 or 20) in
    ``sound_energy_level`` with ``sound_power_level`` all ``NaN``. Both are
    at the meteorological conditions of the test; the properties
    :attr:`sound_power_level_ref` and :attr:`sound_energy_level_ref` add the
    Annex C correction ``c2`` (Eq. C.1, C.3).

    ``mean_source_level`` is the mean corrected level of the source under
    test, :math:`\overline{L_{p(\mathrm{ST})}}` (Eq. 8) or
    :math:`\overline{L_{E(\mathrm{ST})}}` (Eq. 18); ``reference_levels`` the
    mean corrected level of the reference sound source at each of its ``m``
    locations (Eq. 9, 10) and ``mean_reference_level`` their energy mean
    (the second term of Eq. 12, equal to Eq. 9 for one location);
    ``reference_power_level`` the calibrated power of the reference source,
    energy-averaged over its locations (the first term of Eq. 12).

    ``background_correction`` is ``K1i`` at each microphone position and band
    for the source under test (Eq. 7; for ``N`` events measured one at a
    time it is the per-position shift the per-event corrections of Eq. 13
    produce in the mean of Eq. 15), ``background_correction_ref`` the same for
    the reference source at each location (Eq. 9, 10), and
    ``background_requirement_met`` is ``True`` only where a background level
    reached every position (7.5) and every margin over it was at least 6 dB.
    Clause 8.1 writes that margin for the source under test alone
    (:math:`\Delta L_{pi} = L'_{pi}(\mathrm{ST}) - L_{pi}(\mathrm{B})`); the
    flag extends the same test to the reference source, whose level enters
    Eq. 11 carrying a ``K1`` of its own, rather than call a band sound on a
    correction the standard had to cap. It is ``False`` in a band where
    either margin fell below 6 dB, and ``False`` throughout when no
    background levels were supplied at all, since nothing was measured
    against; either way the level is an upper bound to be reported as such
    (8.1).

    ``grade`` is the accuracy grade Table 2 grants (``'engineering'`` or
    ``'survey'``) and ``sigma_r0`` its typical reproducibility; ``sigma_omc``,
    ``sigma_tot`` and ``expanded_uncertainty`` are the operating-and-mounting
    deviation, Eq. (22) and Eq. (23) for ``coverage_factor``, ``NaN`` when no
    ``sigma_omc`` was supplied. ``sound_power_level_a`` and
    ``sound_energy_level_a`` are the Annex D A-weighted totals of the level
    that was determined (``NaN`` for the other).
    """

    frequencies: np.ndarray
    sound_power_level: np.ndarray
    sound_energy_level: np.ndarray
    mean_source_level: np.ndarray
    mean_reference_level: np.ndarray
    reference_levels: np.ndarray
    reference_power_level: np.ndarray
    background_correction: np.ndarray
    background_correction_ref: np.ndarray
    background_requirement_met: np.ndarray
    c2: float
    grade: str
    sigma_r0: float
    sigma_omc: float
    sigma_tot: float
    expanded_uncertainty: float
    coverage_factor: float
    sound_power_level_a: float
    sound_energy_level_a: float
    quantity: str

    def __post_init__(self) -> None:
        """Reject a result whose per-band, per-position or per-location
        quantities disagree, or whose tags are not the two the standard has.

        Every reader of this result indexes its arrays alongside each other:
        the plot draws one bar per ``frequencies`` entry from the level of
        the same index and hatches it by ``background_requirement_met``, and
        the two correction grids are read per position and per location. A
        quantity one entry short raises a bare ``IndexError`` downstream and
        one entry long is silently truncated, so the shapes are pinned here.

        :raises ValueError: if ``quantity`` or ``grade`` is not one of its
            two values, ``coverage_factor`` is not positive, or any array
            disagrees with the rest in bands, positions or locations.
        """
        require_choice(self.quantity, "quantity", ("power", "energy"))
        require_choice(self.grade, "grade", ("engineering", "survey"))
        require_positive(self.coverage_factor, "coverage_factor")
        require_ranks(
            self,
            frequencies=1,
            sound_power_level=1,
            sound_energy_level=1,
            mean_source_level=1,
            mean_reference_level=1,
            reference_levels=2,
            reference_power_level=1,
            background_correction=2,
            background_correction_ref=3,
            background_requirement_met=1,
        )
        require_same_length(
            self,
            "frequencies",
            "sound_power_level",
            "sound_energy_level",
            "mean_source_level",
            "mean_reference_level",
            ("reference_levels", 1),
            "reference_power_level",
            ("background_correction", 1),
            ("background_correction_ref", 2),
            "background_requirement_met",
        )
        owner = type(self).__name__
        require_equal_counts(
            owner,
            {
                "reference_levels": int(np.shape(self.reference_levels)[0]),
                "background_correction_ref": int(
                    np.shape(self.background_correction_ref)[0]
                ),
            },
            axis="reference source location",
        )
        require_equal_counts(
            owner,
            {
                "background_correction": int(np.shape(self.background_correction)[0]),
                "background_correction_ref": int(
                    np.shape(self.background_correction_ref)[1]
                ),
            },
            axis="microphone position",
        )

    @property
    def sound_power_level_ref(self) -> np.ndarray:
        """``LW`` under the reference meteorological conditions, ``LW + C2``
        (ISO 3747:2010 Annex C, Eq. C.1); ``NaN`` for an energy determination.
        """
        return np.asarray(self.sound_power_level + self.c2, dtype=np.float64)

    @property
    def sound_energy_level_ref(self) -> np.ndarray:
        """``LJ`` under the reference meteorological conditions, ``LJ + C2``
        (ISO 3747:2010 Annex C, Eq. C.3); ``NaN`` for a power determination.
        """
        return np.asarray(self.sound_energy_level + self.c2, dtype=np.float64)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the determined spectrum with the A-weighted total annotated.

        One bar per octave band of ``LW`` (or ``LJ`` for an energy
        determination); a band whose background margin fell below 6 dB is
        hatched, because its level is an upper bound (8.1). Requires
        matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.emission import plot_in_situ_sound_power

        check_language(language)
        return plot_in_situ_sound_power(self, ax=ax, language=language, **kwargs)


def static_pressure_from_altitude(altitude: float) -> float:
    r"""Static pressure at the altitude of the test site (ISO 3747:2010 Annex C,
    Eq. C.2).

    .. math::

       p_\mathrm{s} = p_{\mathrm{s},0}\,(1 - a H_\mathrm{a})^{b},
       \qquad a = 2{,}2560 \times 10^{-5}\ \mathrm{m}^{-1},
       \quad b = 5{,}255\,3

    Annex C prints :math:`p_{\mathrm{s},0}` = 1,013 25 x 10^5 Pa and states
    the quantity in pascals. The result here is in kilopascals so that it feeds
    ``static_pressure`` of :func:`sound_power_in_situ` directly, matching
    ISO 3741, ISO 3744 and ISO 3745, which do print kilopascals. The pressure
    reaches ``C2`` only as :math:`p_\mathrm{s}/p_{\mathrm{s},0}`, so the two
    unit conventions give the same correction. A site below sea level is admissible (the base exceeds one);
    the formula stops meaning anything where the base reaches zero, some
    44 km up, and that is refused.

    :param altitude: Altitude of the test site ``Ha``, in metres, one site at
        a time.
    :return: The static pressure ``ps``, in kilopascals.
    :raises ValueError: if ``altitude`` is not a single finite number or
        ``1 - a Ha`` is not positive.
    """
    height = _as_float64(altitude, "altitude")
    if height.ndim != 0:
        msg = "'altitude' must be a single value, the altitude of the test site."
        raise ValueError(msg)
    base = 1.0 - _ALTITUDE_A_PER_M * float(height)
    if not np.isfinite(height) or base <= 0.0:
        msg = (
            "'altitude' must be finite and below the height where Eq. (C.2) "
            "vanishes (about 44 300 m)."
        )
        raise ValueError(msg)
    return float(_PS0 * base**_ALTITUDE_B)


def excess_sound_pressure_level(
    level: ArrayLike, lw_ref: ArrayLike, distance: ArrayLike
) -> np.ndarray | float:
    r"""Excess of sound pressure level over the free field at a distance from
    the reference sound source (ISO 3747:2010 Annex A, Eq. A.1).

    .. math::

       \Delta L_f(r) = L_{p(\mathrm{RSS}),r} - L_{W(\mathrm{RSS})} + 11\ \mathrm{dB}
       + 20 \log_{10}\frac{r}{r_0}, \qquad r_0 = 1\ \mathrm{m}

    The 11 dB is the spherical free-field relation :math:`L_p = L_W - 20
    \log_{10}(r/r_0) - 11` dB, so :math:`\Delta L_f` is zero in a free field
    and grows with the reverberant contribution; the microphone positions of
    the method must lie where it is at least 7 dB (4.1, 7.4.1). Measured
    with A-weighted levels the quantity is :math:`\Delta L_{f\mathrm{A}}`,
    the indicator Table 2 grades the determination by. The three arguments
    broadcast against each other, so one calibrated power serves a whole
    traverse of levels and distances.

    :param level: Sound pressure level(s) ``Lp(RSS),r`` measured at distance
        ``r`` from the reference sound source, in decibels.
    :param lw_ref: Calibrated sound power level ``LW(RSS)`` of the reference
        source, in decibels (per band, or A-weighted).
    :param distance: Distance(s) ``r`` from the microphone to the reference
        source, in metres.
    :return: The excess ``dLf(r)``, in decibels, as a float for scalar input
        or an array of the broadcast shape.
    :raises ValueError: if any input is empty or not finite, any ``distance``
        is not positive, or the three shapes do not broadcast against each
        other.
    """
    lp = _as_float64(level, "level")
    lw = _as_float64(lw_ref, "lw_ref")
    r = _as_float64(distance, "distance")
    for arr, name in ((lp, "level"), (lw, "lw_ref"), (r, "distance")):
        if arr.size == 0:
            msg = f"'{name}' must not be empty."
            raise ValueError(msg)
    if not np.all(np.isfinite(lp)):
        msg = "'level' must be finite."
        raise ValueError(msg)
    if not np.all(np.isfinite(lw)):
        msg = "'lw_ref' must be finite."
        raise ValueError(msg)
    if not np.all(np.isfinite(r)) or np.any(r <= 0.0):
        msg = "'distance' must be finite and positive."
        raise ValueError(msg)
    try:
        np.broadcast_shapes(lp.shape, lw.shape, r.shape)
    except ValueError as exc:
        msg = (
            "'level', 'lw_ref' and 'distance' must broadcast against each "
            f"other; got shapes {lp.shape}, {lw.shape} and {r.shape}."
        )
        raise ValueError(msg) from exc
    excess = lp - lw + _SPHERICAL_CONSTANT_DB + 20.0 * np.log10(r / _R0_M)
    if excess.ndim == 0:
        return float(excess)
    return np.asarray(excess, dtype=np.float64)


def _background_correction(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""``K1`` from the signal-to-background margin, elementwise, with the
    three rules of ISO 3747:2010 clause 8.1 around Eq. (7).

    Above 15 dB the correction is zero; from 6 dB to 15 dB it is Eq. (7);
    below 6 dB it is the smaller of Eq. (7) and the 1,3 dB cap the clause
    sets, and the second array returned says where the margin was at least
    6 dB, so the caller can report the rest as upper bounds. Eq. (7) has no
    value at or below a zero margin (the bracket is not positive); the cap
    takes over there, which is what the clause asks for any margin below
    6 dB.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = -10.0 * np.log10(1.0 - 10.0 ** (-0.1 * delta))
    raw = np.where(np.isnan(raw), np.inf, raw)
    k1 = np.minimum(raw, _K1_MAX_DB)
    k1 = np.where(delta > _K1_NEGLIGIBLE_DB, 0.0, k1)
    return np.asarray(k1, dtype=np.float64), np.asarray(
        delta >= _K1_VALID_DB, dtype=bool
    )


def _finite_grid(value: ArrayLike, name: str, ndim: tuple[int, ...]) -> np.ndarray:
    """A finite float array of one of the admissible ranks, or a refusal
    naming the argument.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim not in ndim or arr.size == 0:
        shapes = " or ".join(f"{d}D" for d in ndim)
        msg = f"'{name}' must be a non-empty {shapes} array of levels in decibels."
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = f"'{name}' must be finite."
        raise ValueError(msg)
    return arr


def _checked_frequencies(frequencies: ArrayLike, n_bands: int) -> np.ndarray:
    """The octave mid-band frequencies, one per band and all in Table D.1."""
    freqs = np.asarray(frequencies, dtype=np.float64)
    if freqs.shape != (n_bands,):
        msg = f"'frequencies' must carry one value per band ({n_bands})."
        raise ValueError(msg)
    if not np.all(np.isfinite(freqs)) or any(
        round(float(f)) not in _CK_OCTAVE for f in freqs
    ):
        msg = (
            "'frequencies' must be nominal octave mid-band frequencies from "
            "63 Hz to 8 kHz (ISO 3747:2010, Table D.1)."
        )
        raise ValueError(msg)
    # Eq. (D.1) sums over k = kmin..kmax, one distinct band per k, so a
    # repeated or out-of-order centre would weight the A-weighted total wrong
    # (Annex D, Table D.1).
    nominal = [round(float(f)) for f in freqs]
    if any(b <= a for a, b in pairwise(nominal)):
        msg = (
            "'frequencies' must be distinct octave mid-band frequencies in "
            "ascending order (ISO 3747:2010, Annex D)."
        )
        raise ValueError(msg)
    return freqs


def _reference_grids(
    levels_ref: ArrayLike, lw_ref: ArrayLike, n_positions: int, n_bands: int
) -> tuple[np.ndarray, np.ndarray]:
    """The reference-source levels as ``(m, n, bands)`` and its calibrated
    power as ``(m, bands)``, for one location (2D levels, 1D power) or ``m``.
    """
    ref = _finite_grid(levels_ref, "levels_ref", (2, 3))
    if ref.ndim == 2:  # noqa: PLR2004
        ref = ref[None, :, :]
    if ref.shape[1:] != (n_positions, n_bands):
        msg = (
            "'levels_ref' must be measured at the same microphone positions "
            f"and bands as 'levels' ({n_positions} positions, {n_bands} bands), "
            "as (positions, bands) or (locations, positions, bands)."
        )
        raise ValueError(msg)
    m = ref.shape[0]
    power = _finite_grid(lw_ref, "lw_ref", (1, 2))
    if power.ndim == 1:
        if power.shape != (n_bands,):
            msg = f"'lw_ref' must carry one value per band ({n_bands})."
            raise ValueError(msg)
        power = np.broadcast_to(power, (m, n_bands)).copy()
    elif power.shape != (m, n_bands):
        msg = (
            "'lw_ref' must be one spectrum, or one per reference source "
            f"location ({m} locations, {n_bands} bands)."
        )
        raise ValueError(msg)
    return ref, power


def _background_grid(
    background: ArrayLike | None, name: str, n_positions: int, n_bands: int
) -> np.ndarray | None:
    """The background levels as ``(n, bands)``: one spectrum for every
    position, or one per position; ``None`` when none were given.
    """
    if background is None:
        return None
    bg = _finite_grid(background, name, (1, 2))
    if bg.ndim == 1:
        if bg.shape != (n_bands,):
            msg = f"'{name}' must carry one value per band ({n_bands})."
            raise ValueError(msg)
        return np.broadcast_to(bg, (n_positions, n_bands)).copy()
    if bg.shape != (n_positions, n_bands):
        msg = (
            f"'{name}' must be one spectrum or one per microphone position "
            f"({n_positions} positions, {n_bands} bands)."
        )
        raise ValueError(msg)
    return bg


@dataclass(frozen=True)
class GradeConditions:
    r"""The two conditions Table 2 puts on engineering grade 2.

    They travel together because they answer one question between them, which
    row of Table 2 the determination may claim, and neither answers it alone:
    the grade is engineering only when the field is reverberant enough at
    *every* microphone position **and** the source is not too directional.
    Either one missing leaves the determination at survey grade 3, which is
    the grade the standard grants when the evidence is not there.

    :param excess_levels: A-weighted excess of sound pressure level over the
        free field, :math:`\Delta L_{f\mathrm{A}}`, one finite value per
        microphone position, in decibels (clause 4.1, Annex A). Engineering
        grade needs at least 7 dB at every position.
    :param directivity_range: Range of the A-weighted directivity survey of
        the source under test, in decibels (clause 7.2). Engineering grade
        needs it within 7 dB.
    """

    excess_levels: ArrayLike | None = None
    directivity_range: float | None = None


def _accuracy_grade(conditions: GradeConditions | None, n_positions: int) -> Grade:
    """The grade Table 2 grants: engineering only with the excess of sound
    pressure level at least 7 dB at every position and a directivity range
    within 7 dB; survey whenever either indicator fails or was not determined.

    Each indicator is checked on its own as soon as it is supplied, so a
    mis-shaped or nonsensical one is refused rather than silently downgrading
    the determination to grade 3; only a genuinely undetermined indicator
    (``None``) falls through to survey.
    """
    if conditions is None:
        return "survey"
    excess_levels = conditions.excess_levels
    directivity_range = conditions.directivity_range
    excess: np.ndarray | None = None
    if excess_levels is not None:
        excess = _as_float64(excess_levels, "excess_levels")
        if excess.shape != (n_positions,) or not np.all(np.isfinite(excess)):
            msg = (
                "'excess_levels' must carry one finite value per microphone "
                f"position ({n_positions})."
            )
            raise ValueError(msg)
    if directivity_range is not None and (
        not np.isfinite(directivity_range) or directivity_range < 0.0
    ):
        msg = "'directivity_range' must be finite and non-negative."
        raise ValueError(msg)
    if excess is None or directivity_range is None:
        return "survey"
    reverberant = bool(np.all(excess >= _EXCESS_LEVEL_MIN_DB))
    if reverberant and directivity_range <= _DIRECTIVITY_RANGE_MAX_DB:
        return "engineering"
    return "survey"


def _uncertainty(
    grade: Grade, sigma_omc: float | None, coverage_factor: float
) -> tuple[float, float, float, float]:
    """``sigma_R0`` from Table 2, ``sigma_omc`` as given, ``sigma_tot`` of
    Eq. (22) and ``U`` of Eq. (23); the last three ``NaN`` without ``sigma_omc``.
    """
    require_positive(coverage_factor, "coverage_factor")
    sigma_r0 = _SIGMA_R0_DB[grade]
    if sigma_omc is None:
        return sigma_r0, float("nan"), float("nan"), float("nan")
    if not np.isfinite(sigma_omc) or sigma_omc < 0.0:
        msg = "'sigma_omc' must be finite and non-negative."
        raise ValueError(msg)
    sigma_tot = float(np.sqrt(sigma_r0**2 + sigma_omc**2))
    return sigma_r0, float(sigma_omc), sigma_tot, coverage_factor * sigma_tot


def _position_advisory(n_positions: int) -> None:
    """Warn when the number of positions is outside the three or four the
    procedure uses (7.4.1 bounds the count on both sides).
    """
    if not _MIN_POSITIONS <= n_positions <= _MAX_POSITIONS:
        warnings.warn(
            f"{n_positions} microphone position(s) were supplied; the "
            "procedure uses three or four, distributed as evenly as possible "
            "around the source (ISO 3747:2010, 7.4.1).",
            SoundPowerWarning,
            stacklevel=3,
        )


def _background_advisory(background_levels: ArrayLike | None) -> None:
    """Warn when no background levels reached the determination at all.

    Clause 7.5 has the background obtained once at each microphone position,
    and 8.1 declares the measurement valid only where the margin over it is
    at least 6 dB. With nothing measured no band can be shown to meet that
    requirement, so ``background_requirement_met`` is ``False`` throughout.
    """
    if background_levels is None:
        warnings.warn(
            "No background levels were supplied; the procedure obtains them "
            "once at each microphone position (ISO 3747:2010, 7.5), so no "
            "band can be reported as meeting the background requirement of "
            "8.1 and every level is returned as an upper bound.",
            SoundPowerWarning,
            stacklevel=3,
        )


@dataclass(frozen=True)
class _Comparison:
    """Everything the reference source and the environment contribute.

    Both entry points forward these ten arguments to :func:`_determine`
    unchanged: they describe the source being compared against and the
    conditions of the test, not the quantity under determination. Keeping
    them together is what lets the two routes share the second half of the
    procedure verbatim, which is what the standard does as well, since
    clauses 8.5 and 9 read the same for power and for energy.
    """

    levels_ref: ArrayLike
    lw_ref: ArrayLike
    frequencies: ArrayLike
    background_levels_ref: ArrayLike | None
    temperature: float
    static_pressure: float
    conditions: GradeConditions | None
    sigma_omc: float | None
    coverage_factor: float


def _event_shape(arr: np.ndarray, events: int | None) -> tuple[bool, int, int, int]:
    """Read the two forms of clause 8.4 off the array, and refuse the mixture.

    Measured one event at a time the array is ``(positions, N, bands)`` and
    carries ``N`` itself, so ``events`` would be a second, contradictable
    source for the same number and is refused. Measured once over ``N``
    events the array is ``(positions, bands)``, ``N`` is nowhere in it, and
    Eq. (17) cannot be applied without it.

    :return: ``(one_at_a_time, n_positions, n_events, n_bands)``.
    """
    one_at_a_time = arr.ndim == 3  # noqa: PLR2004
    if one_at_a_time:
        if events is not None:
            msg = (
                "'events' is counted from the second axis of a 3D 'event_levels'; "
                "pass it only with a 2D (positions, bands) measurement."
            )
            raise ValueError(msg)
        n_positions, n_events, n_bands = arr.shape
        return True, n_positions, n_events, n_bands
    if (
        events is None
        or isinstance(events, bool)
        or not isinstance(events, (int, np.integer))
        or events < 1
    ):
        msg = (
            "'events' must be a positive integer, the number N of events the "
            "2D 'event_levels' measurement encompasses (ISO 3747:2010, Eq. 17)."
        )
        raise ValueError(msg)
    n_positions, n_bands = arr.shape
    return False, n_positions, int(events), n_bands


def _event_background(
    arr: np.ndarray,
    bg: np.ndarray | None,
    *,
    integration_time: float | None,
    one_at_a_time: bool,
    n_bands: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``K1``, the per-band verdict and the per-position levels of clause 8.4.

    With no background measured there is nothing to correct and nothing that
    can be declared valid, so ``K1`` is zero and the verdict is ``False``
    throughout. With one measured, Eq. (14) compares a time-integrated event
    level against a time-averaged background, which is a ratio of energies
    only over a one-second window; ``integration_time`` carries the
    background to the window actually used before the comparison.

    :return: ``(K1 per position and band, verdict per band, per-position
        levels)``.
    """
    n_positions = arr.shape[0]
    if bg is None:
        return (
            np.zeros((n_positions, n_bands), dtype=np.float64),
            np.zeros(n_bands, dtype=bool),
            arr if not one_at_a_time else energy_mean(arr, axis=1),
        )
    bg_event = bg
    if integration_time is not None:
        bg_event = bg + 10.0 * np.log10(integration_time / _T0_S)
    if one_at_a_time:
        k1_events, met_grid = _background_correction(arr - bg_event[:, None, :])
        per_position = energy_mean(arr - k1_events, axis=1)  # Eq. (13), (15)
        return (
            energy_mean(arr, axis=1) - per_position,
            np.all(met_grid, axis=(0, 1)),
            per_position,
        )
    k1, met_grid = _background_correction(arr - bg_event)
    return k1, np.all(met_grid, axis=0), arr - k1  # Eq. (16)


def _determine(
    *,
    quantity: str,
    mean_source: np.ndarray,
    background_correction: np.ndarray,
    met_source: np.ndarray,
    background_levels: np.ndarray | None,
    comparison: _Comparison,
) -> InSituSoundPowerResult:
    """The part of both determinations that runs on the reference source:
    Eq. (9), (10) and the two energy means of Eq. (12) / (20), then Annex C,
    Annex D and clause 9.
    """
    n_positions, n_bands = background_correction.shape
    freqs = _checked_frequencies(comparison.frequencies, n_bands)
    ref, power = _reference_grids(
        comparison.levels_ref, comparison.lw_ref, n_positions, n_bands
    )
    m = ref.shape[0]
    # One background reading serves both sources (7.5): the reference-source
    # measurement takes the source's background unless it brought its own.
    bg_ref = _background_grid(
        comparison.background_levels_ref,
        "background_levels_ref",
        n_positions,
        n_bands,
    )
    if bg_ref is None:
        bg_ref = background_levels
    if bg_ref is None:
        # Nothing was measured, so 8.1 cannot declare the band valid (7.5).
        k1_ref = np.zeros((m, n_positions, n_bands), dtype=np.float64)
        met_ref = np.zeros(n_bands, dtype=bool)
    else:
        # Eq. (10) prints K1i(RSS) without the location index, but the margin
        # it comes from is per location: evaluated per (j, i).
        k1_ref, met = _background_correction(ref - bg_ref[None, :, :])
        met_ref = np.all(met, axis=(0, 1))
    _validate_meteorology(comparison.temperature, comparison.static_pressure)
    corrected_ref = ref - k1_ref
    per_location = energy_mean(corrected_ref, axis=1)  # Eq. (9) / (10)
    mean_ref = energy_mean(per_location, axis=0)  # second term of Eq. (12)
    ref_power = energy_mean(power, axis=0)  # first term of Eq. (12)
    level = np.asarray(ref_power - mean_ref + mean_source, dtype=np.float64)
    nan_band = np.full(n_bands, np.nan, dtype=np.float64)
    total = energy_sum(level + _a_weighting_corrections(freqs))  # Eq. (D.1) / (D.2)
    grade = _accuracy_grade(comparison.conditions, n_positions)
    sigma_r0, omc, sigma_tot, expanded = _uncertainty(
        grade, comparison.sigma_omc, comparison.coverage_factor
    )
    is_power = quantity == "power"
    return InSituSoundPowerResult(
        frequencies=freqs,
        sound_power_level=level if is_power else nan_band,
        sound_energy_level=nan_band if is_power else level,
        mean_source_level=np.asarray(mean_source, dtype=np.float64),
        mean_reference_level=np.asarray(mean_ref, dtype=np.float64),
        reference_levels=np.asarray(per_location, dtype=np.float64),
        reference_power_level=np.asarray(ref_power, dtype=np.float64),
        background_correction=np.asarray(background_correction, dtype=np.float64),
        background_correction_ref=np.asarray(k1_ref, dtype=np.float64),
        background_requirement_met=np.asarray(met_source & met_ref, dtype=bool),
        c2=_c2_correction(comparison.temperature, comparison.static_pressure),
        grade=grade,
        sigma_r0=sigma_r0,
        sigma_omc=omc,
        sigma_tot=sigma_tot,
        expanded_uncertainty=expanded,
        coverage_factor=comparison.coverage_factor,
        sound_power_level_a=total if is_power else float("nan"),
        sound_energy_level_a=float("nan") if is_power else total,
        quantity=quantity,
    )


def sound_power_in_situ(
    levels: ArrayLike,
    levels_ref: ArrayLike,
    lw_ref: ArrayLike,
    frequencies: ArrayLike,
    *,
    background_levels: ArrayLike | None = None,
    background_levels_ref: ArrayLike | None = None,
    temperature: float = 23.0,
    static_pressure: float = 101.325,
    conditions: GradeConditions | None = None,
    sigma_omc: float | None = None,
    coverage_factor: float = _COVERAGE_TWO_SIDED,
) -> InSituSoundPowerResult:
    r"""Sound power level of a steady or non-steady source in situ, by
    comparison with a reference sound source (ISO 3747:2010, clause 8.3).

    The time-averaged octave-band levels of the source under test at the
    ``n`` microphone positions are corrected for background noise position
    by position (Eq. 7 with the rules of 8.1) and energy-averaged (Eq. 8);
    the reference source's levels at the same positions are treated the same
    way (Eq. 9), or per location and then energy-averaged over the ``m``
    locations together with its calibrated powers (Eq. 10, 12). The sound
    power level in each band is then

    .. math::

       L_W = L_{W(\mathrm{RSS})} - \overline{L_{p(\mathrm{RSS})}}
       + \overline{L_{p(\mathrm{ST})}} \tag{Eq. 11}

    at the meteorological conditions of the test; the returned ``c2`` and
    the ``sound_power_level_ref`` property carry it to the reference
    conditions of Annex C, and ``sound_power_level_a`` is the Annex D total.

    :param levels: Measured (uncorrected) octave-band time-averaged levels
        ``L'pi(ST)`` of the source under test, ``(n, bands)``, one row per
        microphone position, in decibels.
    :param levels_ref: The same for the reference sound source, ``L'pi(RSS)``
        already corrected for speed, temperature and static pressure per its
        manufacturer but not for background: ``(n, bands)`` for one
        location, or ``(m, n, bands)`` for ``m`` locations (Eq. 10).
    :param lw_ref: Calibrated octave-band sound power level ``LW(RSS)`` of the
        reference source, ``(bands,)``, or ``(m, bands)`` when each location
        was calibrated in its own similar position (Eq. 12), in decibels.
    :param frequencies: Nominal octave mid-band frequencies, one per band,
        from 63 Hz to 8 kHz (Table D.1), in hertz.
    :param background_levels: Octave-band time-averaged background levels
        ``Lpi(B)``, ``(n, bands)`` or one ``(bands,)`` spectrum for every
        position, in decibels; ``None`` applies no correction, warns, and
        leaves ``background_requirement_met`` ``False`` in every band, since
        7.5 has the background measured at each position and 8.1 needs the
        margin to declare the measurement valid.
    :param background_levels_ref: Background for the reference-source
        measurement, same shapes; ``None`` reuses ``background_levels``,
        since the procedure takes one background reading (7.5).
    :param temperature: Air temperature at the test, in degrees Celsius.
    :param static_pressure: Static pressure at the test, in kilopascals
        (see :func:`static_pressure_from_altitude`). Annex C prints this
        quantity in pascals, with :math:`p_{\mathrm{s},0}` = 1,013 25 x 10^5
        Pa, and is alone in its family in doing so: ISO 3741:2010,
        ISO 3744:2010 and ISO 3745:2012 all print kilopascals. This argument
        follows the three, so that one unit serves the whole ISO 3740 family;
        ``C2`` carries a pressure term and a temperature term, and the
        pressure enters only as the ratio
        :math:`p_\mathrm{s}/p_{\mathrm{s},0}`, so converting both together
        cannot move a result.
    :param conditions: The :class:`GradeConditions` Table 2 reads to decide
        the accuracy grade: the excess of sound pressure level at each
        microphone position (Annex A) and the range of the directivity survey
        of the source (7.2). ``None``, or either condition left out, leaves
        the determination at survey grade.
    :param sigma_omc: Standard deviation of the operating and mounting
        conditions of the source (9.2, E.3), in decibels; ``None`` leaves
        ``sigma_tot`` and the expanded uncertainty ``NaN``.
    :param coverage_factor: ``k`` of Eq. (23): 2 for the two-sided 95 %
        interval (default), 1,6 for a one-sided comparison with a limit.
    :return: :class:`InSituSoundPowerResult` with ``quantity='power'``.
    :raises ValueError: if ``levels`` is not a finite ``(n, bands)`` array,
        ``levels_ref``, ``lw_ref`` or either background does not match it,
        ``frequencies`` are not the octave centres of Table D.1,
        ``temperature`` or ``static_pressure`` is out of range,
        ``conditions.excess_levels`` is supplied and is not one finite value
        per position, ``conditions.directivity_range`` or ``sigma_omc`` is
        supplied and is
        negative, or ``coverage_factor`` is not positive.
    """
    arr = _finite_grid(levels, "levels", (2,))
    n_positions, n_bands = arr.shape
    _position_advisory(n_positions)
    _background_advisory(background_levels)
    bg = _background_grid(background_levels, "background_levels", n_positions, n_bands)
    if bg is None:
        k1 = np.zeros_like(arr)
        met = np.zeros(n_bands, dtype=bool)
    else:
        k1, met_grid = _background_correction(arr - bg)
        met = np.all(met_grid, axis=0)
    mean_source = energy_mean(arr - k1, axis=0)  # Eq. (8)
    return _determine(
        quantity="power",
        mean_source=mean_source,
        background_correction=k1,
        met_source=met,
        background_levels=bg,
        comparison=_Comparison(
            levels_ref=levels_ref,
            lw_ref=lw_ref,
            frequencies=frequencies,
            background_levels_ref=background_levels_ref,
            temperature=temperature,
            static_pressure=static_pressure,
            conditions=conditions,
            sigma_omc=sigma_omc,
            coverage_factor=coverage_factor,
        ),
    )


def sound_energy_in_situ(
    event_levels: ArrayLike,
    levels_ref: ArrayLike,
    lw_ref: ArrayLike,
    frequencies: ArrayLike,
    *,
    events: int | None = None,
    background_levels: ArrayLike | None = None,
    background_levels_ref: ArrayLike | None = None,
    integration_time: float | None = None,
    temperature: float = 23.0,
    static_pressure: float = 101.325,
    conditions: GradeConditions | None = None,
    sigma_omc: float | None = None,
    coverage_factor: float = _COVERAGE_TWO_SIDED,
) -> InSituSoundPowerResult:
    r"""Sound energy level of an impulsive source in situ, by comparison with
    a reference sound source (ISO 3747:2010, clauses 8.4 and 8.5).

    The single event levels are given in one of the two forms clause 8.4
    admits. Measured one event at a time, ``event_levels`` is ``(n, N,
    bands)``: each event is corrected for background (Eq. 13, 14) and the
    ``N`` corrected levels are energy-averaged into the mean single event
    level of the position (Eq. 15). Measured once over ``N`` successive
    events, ``event_levels`` is ``(n, bands)`` with ``events=N``: the level
    is corrected (Eq. 16) and reduced by :math:`10 \log_{10} N` to one event
    (Eq. 17). Either way the per-position levels are energy-averaged (Eq. 18)
    and the sound energy level in each band is

    .. math::

       L_J = L_{W(\mathrm{RSS})} - \overline{L_{p(\mathrm{RSS})}}
       + \overline{L_{E(\mathrm{ST})}} \tag{Eq. 19}

    or its ``m``-location form (Eq. 20). The reference source is measured
    time-averaged, over 30 s (7.6), exactly as for a steady source.

    :param event_levels: Measured (uncorrected) octave-band single event
        levels ``L'Ei,q(ST)`` as ``(n, N, bands)``, or ``L'Ei,N(ST)`` of one
        measurement encompassing ``events`` events as ``(n, bands)``, in
        decibels.
    :param levels_ref: Time-averaged levels of the reference sound source,
        ``(n, bands)`` or ``(m, n, bands)``, as in :func:`sound_power_in_situ`.
    :param lw_ref: Calibrated sound power level of the reference source,
        ``(bands,)`` or ``(m, bands)``, in decibels.
    :param frequencies: Nominal octave mid-band frequencies, one per band.
    :param events: The number ``N`` of events a 2D ``event_levels`` contains
        (Eq. 17); must be ``None`` with the 3D form, which counts them.
    :param background_levels: Octave-band time-averaged background levels
        ``Lpi(B)``, ``(n, bands)`` or ``(bands,)``, in decibels; ``None``
        applies no correction, warns, and leaves
        ``background_requirement_met`` ``False`` in every band (7.5, 8.1).
    :param background_levels_ref: Background for the reference-source
        measurement; ``None`` reuses ``background_levels`` (7.5).
    :param integration_time: The integration time ``T`` of the event
        measurement, in seconds. ``None`` applies Eq. (14) as printed,
        subtracting the time-averaged background from the single event level;
        a value carries the background to the same interval first,
        :math:`L_{pi(\mathrm{B})} + 10 \log_{10}(T/T_0)` with ``T0`` = 1 s
        (3.4, NOTE 1), so that the margin compares like with like. The two
        coincide at ``T`` = 1 s.
    :param temperature: Air temperature at the test, in degrees Celsius.
    :param static_pressure: Static pressure at the test, in kilopascals, as
        for :func:`sound_power_in_situ`; Annex C itself prints pascals.
    :param conditions: The :class:`GradeConditions` of the determination, as
        for :func:`sound_power_in_situ`.
    :param sigma_omc: Operating-and-mounting standard deviation, in decibels.
    :param coverage_factor: ``k`` of Eq. (23), 2 by default.
    :return: :class:`InSituSoundPowerResult` with ``quantity='energy'``.
    :raises ValueError: if ``event_levels`` is not a finite 2D or 3D array,
        ``events`` is given with the 3D form or missing or not a positive
        integer with the 2D form, ``integration_time`` is not positive, or
        any of the refusals of :func:`sound_power_in_situ` applies.
    """
    arr = _finite_grid(event_levels, "event_levels", (2, 3))
    one_at_a_time, n_positions, n_events, n_bands = _event_shape(arr, events)
    _position_advisory(n_positions)
    if n_events < _MIN_EVENTS:
        warnings.warn(
            f"Only {n_events} single event(s) were measured; the procedure asks "
            f"for at least {_MIN_EVENTS} (ISO 3747:2010, 7.6).",
            SoundPowerWarning,
            stacklevel=2,
        )
    _background_advisory(background_levels)
    bg = _background_grid(background_levels, "background_levels", n_positions, n_bands)
    if integration_time is not None:
        require_positive(integration_time, "integration_time")
    k1, met, per_position = _event_background(
        arr,
        bg,
        integration_time=integration_time,
        one_at_a_time=one_at_a_time,
        n_bands=n_bands,
    )
    if not one_at_a_time:
        per_position = per_position - 10.0 * np.log10(n_events)  # Eq. (17)
    mean_source = energy_mean(per_position, axis=0)  # Eq. (18)
    return _determine(
        quantity="energy",
        mean_source=mean_source,
        background_correction=np.asarray(k1, dtype=np.float64),
        met_source=met,
        background_levels=bg,
        comparison=_Comparison(
            levels_ref=levels_ref,
            lw_ref=lw_ref,
            frequencies=frequencies,
            background_levels_ref=background_levels_ref,
            temperature=temperature,
            static_pressure=static_pressure,
            conditions=conditions,
            sigma_omc=sigma_omc,
            coverage_factor=coverage_factor,
        ),
    )
