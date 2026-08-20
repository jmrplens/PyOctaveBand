#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Spanish noise regulation: the corrected level LKeq (Real Decreto 1367/2007).

Real Decreto 1367/2007 develops Ley 37/2003 del Ruido on acoustic zoning,
quality objectives and emitter limit values. Its assessment chain is built on
one index the ISO 1996 family does not define: the **corrected equivalent
continuous level** :math:`L_{\mathrm{Keq},T} = L_{\mathrm{Aeq},T} + K_\mathrm{t} + K_\mathrm{f} + K_\mathrm{i}`
(Annex I A.2 c), where the
three corrections penalise emergent tonal components, low-frequency components
and impulsive character. Each is 0, 3 or 6 dB and their sum is capped at 9 dB
(Annex IV A.3.3).

**Corrections (Annex IV A.3.3).** The reference procedures are:

* ``Kt``: unweighted one-third-octave analysis; for the band ``f`` holding the
  tone, :math:`L_\mathrm{t} = L_f - L_s` with ``Ls`` the *arithmetic* mean of the two
  adjacent
  band levels. ``Kt`` is 0/3/6 dB by the thresholds 8/12 dB (20 to 125 Hz),
  5/8 dB (160 to 400 Hz) and 3/5 dB (500 Hz to 10 kHz); with several emergent
  tones the largest applies.
* ``Kf``: :math:`L_f = L_{\mathrm{Ceq},Ti} - L_{\mathrm{Aeq},Ti}` (background-corrected),
  giving 0 dB for
  :math:`L_f \le 10`, 3 dB for :math:`10 < L_f \le 15` and 6 dB above.
* ``Ki``: :math:`L_\mathrm{i} = L_{\mathrm{AIeq},Ti} - L_{\mathrm{Aeq},Ti}` (background-corrected),
  with the same
  0/3/6 dB thresholds as ``Kf``.

**Relationship with the ISO 1996 procedures already in the library.** They are
*relatives, not the same procedure*, so this module implements the RD's own
variants rather than delegating:

* :func:`~phonometry.environment.assessment.measurement.tonal_audibility` /
  :func:`~phonometry.environment.assessment.measurement.tonal_adjustment` are the
  ISO 1996-2 Annex C engineering method: a critical-band audibility
  ``ΔLta`` mapped to a *continuous* ``Kt`` in 0 to 6 dB. The RD works on
  one-third-octave band differences and yields 0/3/6 dB only.
  :func:`~phonometry.environment.assessment.measurement.tonal_seeking_survey` is the
  closest relative (ISO 1996-2:2017 Annex K also splits the spectrum at
  125/400 Hz), but it requires the band to exceed *both* neighbours by
  15/8/5 dB and returns a boolean flag, whereas the RD compares against the
  *mean* of the neighbours with 8/5/3 dB thresholds and grades the result.
* :func:`~phonometry.environment.assessment.impulsive_sound.impulsive_sound_adjustment`
  is the ISO/PAS 1996-3 onset-rate method on a calibrated time signal. The
  RD's ``Ki`` is the classic :math:`L_\mathrm{AIeq} - L_\mathrm{Aeq}` impulse-vs-fast
  difference read
  off a sound level meter.
* No ISO 1996 counterpart exists for ``Kf``: the :math:`L_\mathrm{Ceq} - L_\mathrm{Aeq}`
  difference is
  specific to the RD.

**Evaluation periods and integration.** Day 07:00-19:00 (12 h), evening
19:00-23:00 (4 h) and night 23:00-07:00 (8 h) (Annex I A.1). A period whose
emission varies is split into *noise phases* ``Ti`` of steady level, and the
period level is the energy mean weighted by phase duration (Annex IV
A.3.4.2 b):

.. math::

   L_{\mathrm{Keq},T} = 10 \log_{10}\left[ (1/T) \sum_i T_i \cdot 10^{L_{\mathrm{Keq},Ti}/10} \right]

The result is
rounded by adding 0.5 dB and taking the integer part. The long-term index
``LK,x`` is the energy mean of the daily ``LKeq,x`` over a year (Annex I
A.2 d).

**Limit tables.** Annex II holds the acoustic quality objectives (Table A
outdoor by area type, as amended by RD 1038/2012; Table B indoor by room; Table
C vibration), Annex III the emitter immission limits (Table A1/A2 new road,
rail and airport infrastructure; Table B1 port infrastructure and activities;
Table B2 noise transmitted to acoustically adjacent premises).

**Compliance.** For activities and port infrastructure (Article 25.1 b) the
limits are respected when no annual average ``LK,x`` exceeds the table value,
no daily ``LKeq,x`` exceeds it by more than 3 dB and no measured ``LKeq,Ti``
exceeds it by more than 5 dB. For the inspection of an activity already in
operation (Article 25.2) only the last two apply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes

    from ..._report.metadata import ReportMetadata

__all__ = [
    "ACOUSTIC_AREA_TYPES",
    "RD1367_CORRECTION_VALUES",
    "RD1367_EVALUATION_PERIODS",
    "RD1367_MAX_CORRECTION",
    "RD1367_PERIOD_CLOCK_LIMITS",
    "RD1367_PERIOD_HOURS",
    "ActivityAssessment",
    "NoisePhase",
    "PeriodAssessment",
    "RegulationLimits",
    "TonalCorrectionResult",
    "activity_limits",
    "adjacent_premises_limits",
    "assess_activity",
    "corrected_level",
    "evaluation_period_level",
    "impulsive_correction",
    "indoor_quality_objectives",
    "infrastructure_limits",
    "long_term_corrected_level",
    "low_frequency_correction",
    "max_infrastructure_limit",
    "outdoor_quality_objectives",
    "round_reported_level",
    "tonal_correction",
    "total_correction",
    "vibration_quality_objective",
]

#: The three daily evaluation periods of Annex I A.1.
RD1367_EVALUATION_PERIODS: tuple[str, str, str] = ("day", "evening", "night")

#: Default duration of each evaluation period, in hours (Annex I A.1 a).
RD1367_PERIOD_HOURS: Mapping[str, float] = MappingProxyType(
    {"day": 12.0, "evening": 4.0, "night": 8.0}
)

#: Local-time start/end hour of each evaluation period (Annex I A.1 b).
RD1367_PERIOD_CLOCK_LIMITS: Mapping[str, tuple[int, int]] = MappingProxyType(
    {
        "day": (7, 19),
        "evening": (19, 23),
        "night": (23, 7),
    }
)

#: Maximum of the summed correction ``Kt + Kf + Ki``, in dB (Annex IV A.3.3).
RD1367_MAX_CORRECTION = 9.0

#: The acoustic area types of Article 7 of Ley 37/2003, keyed by their letter.
ACOUSTIC_AREA_TYPES: Mapping[str, str] = MappingProxyType(
    {
        "e": "sanitary, educational and cultural land use requiring special protection",
        "a": "residential land use",
        "d": "tertiary land use other than type c",
        "c": "recreational and public-entertainment land use",
        "b": "industrial land use",
        "f": "general transport-infrastructure systems and public facilities",
    }
)

#: Accepted spellings of an acoustic area type, mapped to its letter code.
_AREA_ALIASES: dict[str, str] = {
    "sanitary": "e",
    "educational": "e",
    "cultural": "e",
    "sanitary_educational_cultural": "e",
    "residential": "a",
    "tertiary": "d",
    "recreational": "c",
    "entertainment": "c",
    "industrial": "b",
    "transport": "f",
    "general_systems": "f",
}

# --------------------------------------------------------------------------- #
# Limit tables
# --------------------------------------------------------------------------- #
#: Annex II Table A, as amended by RD 1038/2012 (existing urbanised areas).
_OUTDOOR_OBJECTIVES: dict[str, tuple[float, float, float]] = {
    "e": (60.0, 60.0, 50.0),
    "a": (65.0, 65.0, 55.0),
    "d": (70.0, 70.0, 65.0),
    "c": (73.0, 73.0, 63.0),
    "b": (75.0, 75.0, 65.0),
}

#: Annex II Table B: indoor quality objectives, keyed by (use, room type).
_INDOOR_OBJECTIVES: dict[tuple[str, str], tuple[float, float, float]] = {
    ("residential", "living"): (45.0, 45.0, 35.0),
    ("residential", "bedrooms"): (40.0, 40.0, 30.0),
    ("sanitary", "living"): (45.0, 45.0, 35.0),
    ("sanitary", "bedrooms"): (40.0, 40.0, 30.0),
    ("educational", "classrooms"): (40.0, 40.0, 40.0),
    ("educational", "reading_rooms"): (35.0, 35.0, 35.0),
}

#: Annex II Table C: vibration quality objectives ``Law``, in dB.
_VIBRATION_OBJECTIVES: dict[str, float] = {
    "residential": 75.0,
    "sanitary": 72.0,
    "educational": 72.0,
}

#: Annex III Table A1: new road, rail and airport infrastructure.
_INFRASTRUCTURE_LIMITS: dict[str, tuple[float, float, float]] = {
    "e": (55.0, 55.0, 45.0),
    "a": (60.0, 60.0, 50.0),
    "d": (65.0, 65.0, 55.0),
    "c": (68.0, 68.0, 58.0),
    "b": (70.0, 70.0, 60.0),
}

#: Annex III Table A2: maximum ``LAmax`` for rail and airport infrastructure.
_MAX_INFRASTRUCTURE_LIMITS: dict[str, float] = {
    "e": 80.0,
    "a": 85.0,
    "d": 88.0,
    "c": 90.0,
    "b": 90.0,
}

#: Annex III Table B1: port infrastructure and activities.
_ACTIVITY_LIMITS: dict[str, tuple[float, float, float]] = {
    "e": (50.0, 50.0, 40.0),
    "a": (55.0, 55.0, 45.0),
    "d": (60.0, 60.0, 50.0),
    "c": (63.0, 63.0, 53.0),
    "b": (65.0, 65.0, 55.0),
}

#: Annex III Table B2: noise transmitted by activities to adjacent premises.
_ADJACENT_LIMITS: dict[tuple[str, str], tuple[float, float, float]] = {
    ("residential", "living"): (40.0, 40.0, 30.0),
    ("residential", "bedrooms"): (35.0, 35.0, 25.0),
    ("office", "professional_offices"): (35.0, 35.0, 35.0),
    ("office", "offices"): (40.0, 40.0, 40.0),
    ("sanitary", "living"): (40.0, 40.0, 30.0),
    ("sanitary", "bedrooms"): (35.0, 35.0, 25.0),
    ("educational", "classrooms"): (35.0, 35.0, 35.0),
    ("educational", "reading_rooms"): (30.0, 30.0, 30.0),
}

#: Accepted spellings of a building use in the indoor tables.
_USE_ALIASES: dict[str, str] = {
    "dwelling": "residential",
    "housing": "residential",
    "hospital": "sanitary",
    "hospitalario": "sanitary",
    "health": "sanitary",
    "cultural": "educational",
    "administrative": "office",
    "offices": "office",
}

#: Accepted spellings of a room type in the indoor tables.
_ROOM_ALIASES: dict[str, str] = {
    "living_rooms": "living",
    "living_areas": "living",
    "estancias": "living",
    "bedroom": "bedrooms",
    "dormitorios": "bedrooms",
    "classroom": "classrooms",
    "aulas": "classrooms",
    "reading_room": "reading_rooms",
    "professional_office": "professional_offices",
}

#: Allowance, in dB, added to the table limit for a daily ``LKeq,x``
#: (Article 25.1 b ii).
_DAILY_ALLOWANCE = 3.0
#: Allowance, in dB, added to the table limit for a phase ``LKeq,Ti``
#: (Article 25.1 b iii).
_PHASE_ALLOWANCE = 5.0

#: One-third-octave band ranges and (lower, upper) ``Lt`` thresholds of the
#: ``Kt`` table (Annex IV A.3.3): ``Lt < lower`` gives 0 dB, ``lower <= Lt <=
#: upper`` gives 3 dB and ``Lt > upper`` gives 6 dB.
_KT_BANDS: tuple[tuple[float, float, float, float], ...] = (
    (20.0, 125.0, 8.0, 12.0),
    (160.0, 400.0, 5.0, 8.0),
    (500.0, 10000.0, 3.0, 5.0),
)

#: ``Kf`` / ``Ki`` thresholds (Annex IV A.3.3): 0 dB up to 10 dB, 3 dB from
#: there to 15 dB, 6 dB above.
_LEVEL_DIFFERENCE_LOW = 10.0
_LEVEL_DIFFERENCE_HIGH = 15.0

#: Band-centre matching tolerance (fractional) for the ``Kt`` band ranges.
_BAND_TOLERANCE = 0.06

#: Fewest one-third-octave bands the ``Kt`` procedure (Annex IV A.3.3) can
#: read: each candidate band is compared against the arithmetic mean of its
#: two neighbours, so the minimum spectrum is a centre band plus both.
_MIN_KT_BANDS = 3

#: Absolute floating-point tolerance, in hours, on the Annex IV A.3.4.2 b
#: requirement that the noise-phase durations sum to the evaluation-period
#: duration (sum Ti = T).
_HOURS_SUM_TOL = 1e-9


def _finite(value: float, name: str) -> float:
    """Return ``value`` as a float, rejecting non-finite input."""
    scalar = float(value)
    if not math.isfinite(scalar):
        msg = f"'{name}' must be finite."
        raise ValueError(msg)
    return scalar


def _positive(value: float, name: str) -> float:
    """Return ``value`` as a float, rejecting non-positive or non-finite input."""
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        msg = f"'{name}' must be a positive, finite number."
        raise ValueError(msg)
    return scalar


def _normalise(value: str, aliases: Mapping[str, str], name: str) -> str:
    """Lower-case ``value``, replace spaces by underscores and resolve aliases."""
    if not isinstance(value, str):
        msg = f"'{name}' must be a string; got {value!r}."
        raise ValueError(msg)  # noqa: TRY004 - ValueError keeps the module validation errors uniform
    key = value.strip().lower().replace(" ", "_").replace("-", "_")
    return aliases.get(key, key)


def _check_period(period: str) -> str:
    """Validate an evaluation-period name."""
    key = str(period).strip().lower()
    if key not in RD1367_EVALUATION_PERIODS:
        msg = f"'period' must be one of 'day', 'evening', 'night'; got {period!r}."
        raise ValueError(msg)
    return key


def round_reported_level(value: float) -> int:
    """Round a level the way RD 1367/2007 prescribes (Annex IV A.3.4.2).

    "El valor del nivel sonoro resultante se redondeará incrementándolo en
    0,5 dB(A), tomando la parte entera como valor resultante": add 0,5 dB and
    take the integer part. This is a half-up rounding towards ``+inf``, which
    for a negative level differs from Python's banker's :func:`round`.

    :param value: Sound level, in dB.
    :return: The rounded level, as an :class:`int`.
    :raises ValueError: If ``value`` is not finite.
    """
    return math.floor(_finite(value, "value") + 0.5)


# --------------------------------------------------------------------------- #
# Corrections Kt, Kf, Ki (Annex IV A.3.3)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TonalCorrectionResult:
    r"""Tonal correction ``Kt`` of RD 1367/2007 (Annex IV A.3.3).

    :ivar frequencies: One-third-octave band centre frequencies, in Hz.
    :ivar levels: Unweighted band sound pressure levels, in dB.
    :ivar differences: :math:`L_\mathrm{t} = L_f - L_s` per band, in dB, where ``Ls``
        is the
        arithmetic mean of the two adjacent band levels. ``NaN`` for bands
        that cannot be evaluated (the two end bands, and bands outside the
        20 Hz to 10 kHz range of the table).
    :ivar band_corrections: The ``Kt`` each band would contribute, in dB
        (0, 3 or 6); ``NaN`` where ``differences`` is ``NaN``.
    :ivar correction: The governing ``Kt``, in dB: the largest band
        contribution (Annex IV A.3.3 d), or 0 dB if no band qualifies.
    :ivar governing_frequency: Centre frequency of the governing band, in Hz,
        or ``None`` when ``correction`` is 0 dB.
    """

    frequencies: np.ndarray
    levels: np.ndarray
    differences: np.ndarray
    band_corrections: np.ndarray
    correction: float
    governing_frequency: float | None

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the band spectrum with the emergent-tone differences ``Lt``."""
        from ..._i18n import check_language
        from ..._plot.environment import plot_tonal_correction_rd1367

        return plot_tonal_correction_rd1367(
            self, ax=ax, language=check_language(language), **kwargs
        )


def _kt_thresholds(frequency: float) -> tuple[float, float] | None:
    """The (lower, upper) ``Lt`` thresholds for a band centre, or ``None``."""
    for low, high, lower, upper in _KT_BANDS:
        if frequency >= low * (1.0 - _BAND_TOLERANCE) and frequency <= high * (
            1.0 + _BAND_TOLERANCE
        ):
            return lower, upper
    return None


def _validate_kt_spectrum(band_levels: np.ndarray, freqs: np.ndarray) -> None:
    """Reject a spectrum the Annex IV A.3.3 procedure cannot read.

    :param band_levels: Unweighted one-third-octave band levels, dB.
    :param freqs: Band centre frequencies, Hz, strictly ascending.
    :raises ValueError: For a shape, a value or an order the procedure needs.
    """
    if band_levels.ndim != 1 or freqs.ndim != 1:
        msg = "'levels' and 'frequencies' must be one-dimensional."
        raise ValueError(msg)
    if band_levels.shape != freqs.shape:
        msg = (
            "'levels' and 'frequencies' must have the same length; got "
            f"{band_levels.size} and {freqs.size}."
        )
        raise ValueError(msg)
    if band_levels.size < _MIN_KT_BANDS:
        msg = (
            "At least three one-third-octave bands are needed: the procedure "
            "compares a band against its two neighbours."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(band_levels)):
        msg = "'levels' must contain finite values."
        raise ValueError(msg)
    if not np.all(np.isfinite(freqs)) or np.any(freqs <= 0.0):
        msg = "'frequencies' must contain positive, finite values."
        raise ValueError(msg)
    if np.any(np.diff(freqs) <= 0.0):
        msg = "'frequencies' must be strictly ascending."
        raise ValueError(msg)


def _band_tonal_corrections(
    band_levels: np.ndarray, freqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Steps b and c of Annex IV A.3.3, band by band.

    Interior bands only: the first and the last have no pair of neighbours to
    average, and a band outside the tabulated ranges has no threshold, so both
    stay ``nan`` in each output.

    :param band_levels: Unweighted one-third-octave band levels, dB.
    :param freqs: Band centre frequencies, Hz.
    :return: ``(differences, band_corrections)``, both aligned to the input.
    """
    differences = np.full(band_levels.shape, np.nan, dtype=np.float64)
    band_kt = np.full(band_levels.shape, np.nan, dtype=np.float64)
    for i in range(1, band_levels.size - 1):
        thresholds = _kt_thresholds(float(freqs[i]))
        if thresholds is None:
            continue
        neighbour_mean = 0.5 * (band_levels[i - 1] + band_levels[i + 1])
        lt = float(band_levels[i] - neighbour_mean)
        differences[i] = lt
        lower, upper = thresholds
        if lt < lower:
            band_kt[i] = 0.0
        elif lt <= upper:
            band_kt[i] = 3.0
        else:
            band_kt[i] = 6.0
    return differences, band_kt


def tonal_correction(
    levels: Sequence[float] | np.ndarray,
    frequencies: Sequence[float] | np.ndarray,
) -> TonalCorrectionResult:
    r"""Tonal correction ``Kt`` from a one-third-octave spectrum (Annex IV A.3.3).

    The spectrum must be **unweighted** (no frequency weighting applied, as
    required by step a). For every interior band ``f`` the procedure forms
    :math:`L_\mathrm{t} = L_f - L_s` with ``Ls`` the arithmetic mean of the levels of
    the bands
    immediately above and below (step b), and reads ``Kt`` off the table of
    step c: with 20 Hz to 125 Hz bands :math:`L_\mathrm{t} < 8` gives 0 dB,
    :math:`8 \le L_\mathrm{t} \le 12`
    gives 3 dB and :math:`L_\mathrm{t} > 12` gives 6 dB; the thresholds are 5/8 dB over
    160 Hz to 400 Hz and 3/5 dB over 500 Hz to 10 kHz. With more than one
    emergent tone the largest ``Kt`` governs (step d).

    .. note::
       This is *not* the ISO 1996-2 tonal adjustment. The closest ISO relative
       is the Annex K survey method
       (:func:`~phonometry.environment.assessment.measurement.tonal_seeking_survey`),
       which compares the band against *both* neighbours with 15/8/5 dB
       thresholds and only flags prominence; the RD compares against their
       arithmetic mean with 8/5/3 dB thresholds and grades the result 0/3/6 dB.

    :param levels: Unweighted one-third-octave band levels, in dB.
    :param frequencies: Band centre frequencies, in Hz, in ascending order
        (one per level).
    :return: A :class:`TonalCorrectionResult`.
    :raises ValueError: If fewer than three bands are given, the shapes differ,
        the frequencies are not positive and strictly ascending, or any value
        is not finite.
    """
    band_levels = np.asarray(levels, dtype=np.float64)
    freqs = np.asarray(frequencies, dtype=np.float64)
    _validate_kt_spectrum(band_levels, freqs)
    differences, band_kt = _band_tonal_corrections(band_levels, freqs)

    if np.all(np.isnan(band_kt)):
        correction = 0.0
        governing: float | None = None
    else:
        best = int(np.nanargmax(band_kt))
        correction = float(band_kt[best])
        governing = float(freqs[best]) if correction > 0.0 else None
    return TonalCorrectionResult(
        # Copy so a later mutation of the caller's arrays cannot rewrite the
        # spectrum this result reports.
        frequencies=freqs.copy(),
        levels=band_levels.copy(),
        differences=differences,
        band_corrections=band_kt,
        correction=correction,
        governing_frequency=governing,
    )


def _graded_correction(difference: float) -> float:
    """0/3/6 dB from a level difference on the ``Kf`` / ``Ki`` thresholds."""
    if difference <= _LEVEL_DIFFERENCE_LOW:
        return 0.0
    if difference <= _LEVEL_DIFFERENCE_HIGH:
        return 3.0
    return 6.0


def low_frequency_correction(lceq: float, laeq: float) -> float:
    r"""Low-frequency correction ``Kf`` (Annex IV A.3.3).

    From the C- and A-weighted equivalent levels of the same noise phase, both
    already corrected for background noise,
    :math:`L_f = L_{\mathrm{Ceq},Ti} - L_{\mathrm{Aeq},Ti}` gives
    :math:`K_\mathrm{f} = 0` for :math:`L_f \le 10` dB, :math:`K_\mathrm{f} = 3` for
    :math:`10 < L_f \le 15` dB and
    :math:`K_\mathrm{f} = 6` above.

    .. note::
       The printed table reads "Si 10 >Lf <=15" for the 3 dB row, a misprint
       for :math:`10 < L_f \le 15`; the bracketing rows (:math:`L_f \le 10`
       and
       :math:`L_f > 15`) leave no other consistent reading. See
       ``docs/ERRATA.md``.

    :param lceq: C-weighted equivalent continuous level ``LCeq,Ti``, in dB.
    :param laeq: A-weighted equivalent continuous level ``LAeq,Ti``, in dB.
    :return: ``Kf`` in dB (0, 3 or 6).
    :raises ValueError: If either level is not finite.
    """
    return _graded_correction(_finite(lceq, "lceq") - _finite(laeq, "laeq"))


def impulsive_correction(laieq: float, laeq: float) -> float:
    r"""Impulsive correction ``Ki`` (Annex IV A.3.3).

    From the impulse- and fast-time-weighted equivalent levels of the same
    noise phase, both already corrected for background noise,
    :math:`L_\mathrm{i} = L_{\mathrm{AIeq},Ti} - L_{\mathrm{Aeq},Ti}` gives :math:`K_\mathrm{i} = 0` for
    :math:`L_\mathrm{i} \le 10` dB,
    :math:`K_\mathrm{i} = 3` for :math:`10 < L_\mathrm{i} \le 15` dB and :math:`K_\mathrm{i} = 6` above.

    .. note::
       This is the classic sound-level-meter route. The onset-rate method of
       :func:`~phonometry.environment.assessment.impulsive_sound.impulsive_sound_adjustment`
       (ISO/PAS 1996-3) is a different, signal-based procedure and its ``KI``
       is not interchangeable with this ``Ki``. The same misprint as in
       :func:`low_frequency_correction` affects the 3 dB row of the printed
       table.

    :param laieq: Impulse-weighted equivalent level ``LAIeq,Ti``, in dB.
    :param laeq: A-weighted equivalent continuous level ``LAeq,Ti``, in dB.
    :return: ``Ki`` in dB (0, 3 or 6).
    :raises ValueError: If either level is not finite.
    """
    return _graded_correction(_finite(laieq, "laieq") - _finite(laeq, "laeq"))


#: The only values a single correction can take (Annex IV A.3.3): each of the
#: three tables grades its parameter 0, 3 or 6 dB and nothing in between.
RD1367_CORRECTION_VALUES: tuple[float, float, float] = (0.0, 3.0, 6.0)


def total_correction(kt: float = 0.0, kf: float = 0.0, ki: float = 0.0) -> float:
    r"""Summed correction :math:`K = K_\mathrm{t} + K_\mathrm{f} + K_\mathrm{i}`, capped at 9 dB
    (Annex IV A.3.3).

    Each of the three tables of Annex IV A.3.3 grades its parameter 0, 3 or
    6 dB, so any other value is rejected rather than silently accepted: a
    correction of, say, 4.5 dB is not a reading the regulation can produce.

    :param kt: Tonal correction, in dB (0, 3 or 6).
    :param kf: Low-frequency correction, in dB (0, 3 or 6).
    :param ki: Impulsive correction, in dB (0, 3 or 6).
    :return: The summed correction, in dB, never above
        :data:`RD1367_MAX_CORRECTION`.
    :raises ValueError: If a correction is not one of 0, 3 or 6 dB.
    """
    total = 0.0
    for value, name in ((kt, "kt"), (kf, "kf"), (ki, "ki")):
        scalar = _finite(value, name)
        if not any(math.isclose(scalar, step) for step in RD1367_CORRECTION_VALUES):
            msg = (
                f"'{name}' must be 0, 3 or 6 dB, the only values the Annex IV "
                f"A.3.3 tables grade; got {value!r}."
            )
            raise ValueError(msg)
        total += scalar
    return min(total, RD1367_MAX_CORRECTION)


def corrected_level(
    laeq: float, *, kt: float = 0.0, kf: float = 0.0, ki: float = 0.0
) -> float:
    r"""Corrected equivalent continuous level ``LKeq,T`` (Annex I A.2 c).

    :math:`L_{\mathrm{Keq},T} = L_{\mathrm{Aeq},T} + K_\mathrm{t} + K_\mathrm{f} + K_\mathrm{i}` with the sum of the
    corrections capped
    at 9 dB. Although it is derived from an A-weighted level, the index is
    expressed in dB by definition.

    :param laeq: A-weighted equivalent continuous level ``LAeq,T``, in dB,
        already corrected for background noise.
    :param kt: Tonal correction, in dB.
    :param kf: Low-frequency correction, in dB.
    :param ki: Impulsive correction, in dB.
    :return: ``LKeq,T`` in dB.
    :raises ValueError: If ``laeq`` is not finite or a correction is negative.
    """
    return _finite(laeq, "laeq") + total_correction(kt, kf, ki)


# --------------------------------------------------------------------------- #
# Noise phases and period integration (Annex IV A.3.4.2 b)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NoisePhase:
    """A noise phase ``Ti`` of steady emission within an evaluation period.

    RD 1367/2007 (Annex IV A.3.4.2 b) splits an evaluation period whose
    emission varies into phases in which the sound pressure level at the
    assessment point is perceived uniformly, measures ``LAeq,Ti`` over at
    least 5 s in each, and corrects it for tonal, low-frequency and impulsive
    character.

    :ivar hours: Duration ``Ti`` of the phase, in hours (positive).
    :ivar laeq: Background-corrected ``LAeq,Ti``, in dB. A phase in which the
        activity is shut down carries the level actually measured; the worked
        examples of the Spanish literature enter it as 0 dB.
    :ivar kt: Tonal correction of the phase, in dB.
    :ivar kf: Low-frequency correction of the phase, in dB.
    :ivar ki: Impulsive correction of the phase, in dB.
    :ivar label: Optional free-text description of the phase.
    :raises ValueError: If ``hours`` is not positive and finite, ``laeq`` is
        not finite, or a correction is negative.
    """

    hours: float
    laeq: float
    kt: float = 0.0
    kf: float = 0.0
    ki: float = 0.0
    label: str | None = None

    def __post_init__(self) -> None:
        """Validate the duration, level and corrections."""
        _positive(self.hours, "hours")
        _finite(self.laeq, "laeq")
        total_correction(self.kt, self.kf, self.ki)

    @property
    def correction(self) -> float:
        r"""Summed correction :math:`K = K_\mathrm{t} + K_\mathrm{f} + K_\mathrm{i}`, capped at 9 dB."""
        return total_correction(self.kt, self.kf, self.ki)

    @property
    def lkeq(self) -> float:
        """The corrected level ``LKeq,Ti`` of the phase, in dB."""
        return self.laeq + self.correction


def evaluation_period_level(
    phases: Sequence[NoisePhase], *, hours: float | None = None
) -> float:
    r"""Evaluation-period level ``LKeq,T`` from its noise phases.

    :math:`L_{\mathrm{Keq},T} = 10 \log_{10}\left[ (1/T) \sum_i T_i \cdot
    10^{L_{\mathrm{Keq},Ti}/10} \right]` (Annex IV
    A.3.4.2 b): the duration-weighted energy mean of the phase levels. The
    returned value is **not** rounded; apply :func:`round_reported_level` for the value
    the regulation asks to report.

    :param phases: The noise phases of the period.
    :param hours: Total period duration ``T``, in hours. ``None`` (default)
        uses the sum of the phase durations, as the regulation requires
        :math:`\sum T_i = T`.
    :return: ``LKeq,T`` in dB.
    :raises ValueError: If ``phases`` is empty, ``hours`` is not positive, or
        the phase durations do not sum to ``hours``.
    """
    items = list(phases)
    if not items:
        msg = "At least one noise phase is required."
        raise ValueError(msg)
    durations = np.array([p.hours for p in items], dtype=np.float64)
    levels = np.array([p.lkeq for p in items], dtype=np.float64)
    total = float(np.sum(durations))
    if hours is None:
        period = total
    else:
        period = _positive(hours, "hours")
        if abs(total - period) > _HOURS_SUM_TOL:
            msg = (
                "The phase durations must sum to the period duration "
                f"(sum Ti = T); got {total!r} h for T = {period!r} h."
            )
            raise ValueError(msg)
    return float(10.0 * np.log10(np.sum(durations * 10.0 ** (levels / 10.0)) / period))


def long_term_corrected_level(
    daily_levels: Sequence[float] | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float:
    r"""Long-term index ``LK,x`` from the daily period levels (Annex I A.2 d).

    :math:`L_{\mathrm{K},x} = 10 \log_{10}\left[ (1/n) \sum_j 10^{L_{\mathrm{Keq},x,j}/10} \right]`:
    the energy mean of the
    daily corrected levels of the same evaluation period over a year. With
    ``weights`` the mean is weighted, which lets a whole block of identical
    days be entered once (e.g. 303 operating days at one level and 62 closed
    days at another).

    :param daily_levels: Daily ``LKeq,x`` values, in dB.
    :param weights: Optional number of days each level represents (positive).
    :return: ``LK,x`` in dB, unrounded.
    :raises ValueError: If the input is empty, non-finite, or the weights do
        not match the levels or are not positive.
    """
    levels = np.asarray(daily_levels, dtype=np.float64).ravel()
    if levels.size == 0:
        msg = "At least one daily level is required."
        raise ValueError(msg)
    if not np.all(np.isfinite(levels)):
        msg = "'daily_levels' must contain finite values."
        raise ValueError(msg)
    if weights is None:
        counts = np.ones_like(levels)
    else:
        counts = np.asarray(weights, dtype=np.float64).ravel()
        if counts.shape != levels.shape:
            msg = (
                "'weights' must have one value per daily level; got "
                f"{counts.size} for {levels.size} levels."
            )
            raise ValueError(msg)
        if not np.all(np.isfinite(counts)) or np.any(counts <= 0.0):
            msg = "'weights' must contain positive, finite values."
            raise ValueError(msg)
    return float(
        10.0 * np.log10(np.sum(counts * 10.0 ** (levels / 10.0)) / np.sum(counts))
    )


# --------------------------------------------------------------------------- #
# Limit-table lookups
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegulationLimits:
    """A day/evening/night limit triple read from RD 1367/2007.

    :ivar day: Limit of the day period, in dB.
    :ivar evening: Limit of the evening period, in dB.
    :ivar night: Limit of the night period, in dB.
    :ivar index: Noise index the limits apply to (e.g. ``"LK,x"``, ``"Lx"``).
    :ivar reference: The table the values were read from.
    :ivar description: The row of that table, in plain words.
    """

    day: float
    evening: float
    night: float
    index: str
    reference: str
    description: str

    def __getitem__(self, period: str) -> float:
        """The limit of one evaluation period, in dB."""
        return float(getattr(self, _check_period(period)))

    def as_dict(self) -> dict[str, float]:
        """The three limits keyed by evaluation period."""
        return {period: self[period] for period in RD1367_EVALUATION_PERIODS}


def _limits(
    table: Mapping[Any, tuple[float, float, float]],
    key: Any,
    *,
    index: str,
    reference: str,
    description: str,
    offset: float = 0.0,
) -> RegulationLimits:
    """Build a :class:`RegulationLimits` from a table row, plus an offset."""
    day, evening, night = table[key]
    return RegulationLimits(
        day=day + offset,
        evening=evening + offset,
        night=night + offset,
        index=index,
        reference=reference,
        description=description,
    )


def _area_key(area_type: str, table: Mapping[str, Any]) -> str:
    """Resolve and validate an acoustic area type against a table."""
    key = _normalise(area_type, _AREA_ALIASES, "area_type")
    if key not in table:
        msg = (
            "'area_type' must be one of "
            f"{sorted(table)} (or a documented alias); got {area_type!r}."
        )
        raise ValueError(msg)
    return key


def outdoor_quality_objectives(
    area_type: str, *, urbanisation: str = "existing"
) -> RegulationLimits:
    """Outdoor acoustic quality objectives ``Ld``/``Le``/``Ln`` (Annex II Table A).

    The values are those of Table A as amended by RD 1038/2012. For an area
    urbanised **after** the regulation entered into force (24 October 2007) the
    objective is the same table reduced by 5 dB (Article 14.2); the same 5 dB
    reduction defines the objective of quiet areas (Article 14.4).

    Area type ``"f"`` (general transport-infrastructure systems) carries no
    numeric objective: footnote (2) refers it to the adjoining areas.

    :param area_type: Acoustic area type: the letter ``"e"``, ``"a"``, ``"d"``,
        ``"c"`` or ``"b"``, or an alias such as ``"residential"``.
    :param urbanisation: ``"existing"`` (default) for areas already urbanised
        on 24 October 2007, or ``"new"`` for the rest (5 dB stricter).
    :return: The applicable :class:`RegulationLimits` for the index ``Lx``.
    :raises ValueError: For an unknown area type or urbanisation state.
    """
    key = _area_key(area_type, _OUTDOOR_OBJECTIVES)
    state = str(urbanisation).strip().lower()
    if state not in ("existing", "new"):
        msg = f"'urbanisation' must be 'existing' or 'new'; got {urbanisation!r}."
        raise ValueError(msg)
    offset = 0.0 if state == "existing" else -5.0
    suffix = (
        "existing urbanised areas"
        if state == "existing"
        else "areas urbanised after 24 October 2007 (Article 14.2, -5 dB)"
    )
    return _limits(
        _OUTDOOR_OBJECTIVES,
        key,
        index="Lx",
        reference="RD 1367/2007 Annex II Table A (as amended by RD 1038/2012)",
        description=f"Area type {key}: {ACOUSTIC_AREA_TYPES[key]}, {suffix}",
        offset=offset,
    )


def _indoor_key(
    building_use: str, room_type: str, table: Mapping[tuple[str, str], Any]
) -> tuple[str, str]:
    """Resolve and validate a (building use, room type) pair against a table."""
    use = _normalise(building_use, _USE_ALIASES, "building_use")
    room = _normalise(room_type, _ROOM_ALIASES, "room_type")
    if (use, room) not in table:
        msg = (
            "Unknown (building_use, room_type) combination "
            f"{(building_use, room_type)!r}; the table holds {sorted(table)}."
        )
        raise ValueError(msg)
    return use, room


def indoor_quality_objectives(building_use: str, room_type: str) -> RegulationLimits:
    """Indoor acoustic quality objectives ``Ld``/``Le``/``Ln`` (Annex II Table B).

    The objectives that the *whole set* of emitters reaching a habitable room
    must respect: dwellings and residential uses, hospitals, and educational or
    cultural buildings.

    :param building_use: ``"residential"``, ``"sanitary"`` (alias
        ``"hospital"``) or ``"educational"``.
    :param room_type: ``"living"`` (living areas), ``"bedrooms"``,
        ``"classrooms"`` or ``"reading_rooms"``.
    :return: The applicable :class:`RegulationLimits` for the index ``Lx``.
    :raises ValueError: For an unknown combination.
    """
    key = _indoor_key(building_use, room_type, _INDOOR_OBJECTIVES)
    return _limits(
        _INDOOR_OBJECTIVES,
        key,
        index="Lx",
        reference="RD 1367/2007 Annex II Table B",
        description=f"{key[0]} building, {key[1].replace('_', ' ')}",
    )


def vibration_quality_objective(building_use: str) -> float:
    """Indoor vibration quality objective ``Law``, in dB (Annex II Table C).

    :param building_use: ``"residential"``, ``"sanitary"`` (alias
        ``"hospital"``) or ``"educational"``.
    :return: The ``Law`` objective, in dB.
    :raises ValueError: For an unknown building use.
    """
    use = _normalise(building_use, _USE_ALIASES, "building_use")
    if use not in _VIBRATION_OBJECTIVES:
        msg = (
            "'building_use' must be one of "
            f"{sorted(_VIBRATION_OBJECTIVES)}; got {building_use!r}."
        )
        raise ValueError(msg)
    return _VIBRATION_OBJECTIVES[use]


def infrastructure_limits(area_type: str) -> RegulationLimits:
    """Immission limits for new transport infrastructure (Annex III Table A1).

    New road, rail and airport infrastructure must not transmit to the outdoor
    environment of an acoustic area levels above these ``Ld``/``Le``/``Ln``
    values (Article 23.1).

    :param area_type: Acoustic area type letter, or an alias.
    :return: The applicable :class:`RegulationLimits` for the index ``Lx``.
    :raises ValueError: For an unknown area type.
    """
    key = _area_key(area_type, _INFRASTRUCTURE_LIMITS)
    return _limits(
        _INFRASTRUCTURE_LIMITS,
        key,
        index="Lx",
        reference="RD 1367/2007 Annex III Table A1",
        description=f"Area type {key}: {ACOUSTIC_AREA_TYPES[key]}",
    )


def max_infrastructure_limit(area_type: str) -> float:
    """Maximum ``LAmax`` for rail and airport infrastructure (Annex III Table A2).

    The limit is respected when, over a year, 97 % of all daily values stay at
    or below it (Article 25.1 a iii).

    :param area_type: Acoustic area type letter, or an alias.
    :return: The ``LAmax`` limit, in dB.
    :raises ValueError: For an unknown area type.
    """
    key = _area_key(area_type, _MAX_INFRASTRUCTURE_LIMITS)
    return _MAX_INFRASTRUCTURE_LIMITS[key]


def activity_limits(area_type: str) -> RegulationLimits:
    """Outdoor immission limits for activities and ports (Annex III Table B1).

    The ``LK,d``/``LK,e``/``LK,n`` an installation, establishment or activity
    (industrial, commercial, storage, sports, recreational or leisure) and port
    infrastructure must not exceed in the outdoor environment of the acoustic
    area they sit in (Article 24.1).

    :param area_type: Acoustic area type letter, or an alias.
    :return: The applicable :class:`RegulationLimits` for the index ``LK,x``.
    :raises ValueError: For an unknown area type.
    """
    key = _area_key(area_type, _ACTIVITY_LIMITS)
    return _limits(
        _ACTIVITY_LIMITS,
        key,
        index="LK,x",
        reference="RD 1367/2007 Annex III Table B1",
        description=f"Area type {key}: {ACOUSTIC_AREA_TYPES[key]}",
    )


def adjacent_premises_limits(building_use: str, room_type: str) -> RegulationLimits:
    """Limits on noise transmitted to adjacent premises (Annex III Table B2).

    Two premises are acoustically adjacent when noise never travels between
    emitter and receiver through the outdoor environment (Article 24.3): the
    workshop on the ground floor and the flat above it are, the building across
    the street is not.

    :param building_use: ``"residential"``, ``"office"`` (alias
        ``"administrative"``), ``"sanitary"`` or ``"educational"``.
    :param room_type: ``"living"``, ``"bedrooms"``, ``"professional_offices"``,
        ``"offices"``, ``"classrooms"`` or ``"reading_rooms"``.
    :return: The applicable :class:`RegulationLimits` for the index ``LK,x``.
    :raises ValueError: For an unknown combination.
    """
    key = _indoor_key(building_use, room_type, _ADJACENT_LIMITS)
    return _limits(
        _ADJACENT_LIMITS,
        key,
        index="LK,x",
        reference="RD 1367/2007 Annex III Table B2",
        description=f"{key[0]} premises, {key[1].replace('_', ' ')}",
    )


# --------------------------------------------------------------------------- #
# Compliance assessment (Article 25)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PeriodAssessment:
    """The assessment of one evaluation period against its limit.

    :ivar period: ``"day"``, ``"evening"`` or ``"night"``.
    :ivar phases: The noise phases the period was split into.
    :ivar duration_hours: The period duration ``T``, in hours.
    :ivar evaluation_period_level: ``LKeq,x`` of the period, in dB, unrounded.
    :ivar reported_level: ``LKeq,x`` rounded per Annex IV A.3.4.2.
    :ivar long_term_corrected_level: Annual ``LK,x``, in dB, unrounded, or ``None`` when
        no annual information was supplied.
    :ivar reported_long_term: ``LK,x`` rounded, or ``None``.
    :ivar limit: The table limit of the period, in dB.
    :ivar max_phase_level: The largest ``LKeq,Ti`` of the period, in dB.
    :ivar phase_pass: Whether every ``LKeq,Ti`` stays within ``limit + 5`` dB.
    :ivar daily_pass: Whether ``LKeq,x`` stays within ``limit + 3`` dB.
    :ivar long_term_pass: Whether ``LK,x`` stays at or below ``limit``, or
        ``None`` when the criterion was not evaluated.
    """

    period: str
    phases: tuple[NoisePhase, ...]
    duration_hours: float
    evaluation_period_level: float
    reported_level: int
    long_term_corrected_level: float | None
    reported_long_term: int | None
    limit: float
    max_phase_level: float
    phase_pass: bool
    daily_pass: bool
    long_term_pass: bool | None

    @property
    def phase_limit(self) -> float:
        """The phase limit ``limit + 5`` dB (Article 25.1 b iii)."""
        return self.limit + _PHASE_ALLOWANCE

    @property
    def daily_limit(self) -> float:
        """The daily limit ``limit + 3`` dB (Article 25.1 b ii)."""
        return self.limit + _DAILY_ALLOWANCE

    @property
    def complies(self) -> bool:
        """Whether every evaluated criterion of this period is met."""
        return (
            self.phase_pass and self.daily_pass and (self.long_term_pass is not False)
        )


@dataclass(frozen=True)
class ActivityAssessment:
    """Compliance of an activity or port infrastructure (Article 25).

    :ivar periods: The per-period assessments, in day/evening/night order.
    :ivar limits: The limit table row the assessment was made against.
    :ivar new_activity: ``True`` when the activity is *new* in the sense of the
        regulation, so the annual criterion of Article 25.1 b i applies; for
        the inspection of an activity already in operation only the daily and
        phase criteria apply (Article 25.2).
    """

    periods: tuple[PeriodAssessment, ...]
    limits: RegulationLimits
    new_activity: bool

    @property
    def complies(self) -> bool:
        """Whether every period meets every evaluated criterion."""
        return all(p.complies for p in self.periods)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the per-period indices against their RD 1367/2007 limits."""
        from ..._i18n import check_language
        from ..._plot.environment import plot_activity_assessment

        return plot_activity_assessment(
            self, ax=ax, language=check_language(language), **kwargs
        )

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        verbose: bool = False,
        language: str = "es",
    ) -> str:
        """Render a one-page noise inspection fiche (``acta``) to ``path``.

        The layout follows the Spanish municipal / accredited-laboratory noise
        inspection report: an identification header, the per-phase measurement
        table with the ``Kt``/``Kf``/``Ki`` corrections, the per-period
        assessment against the applicable limits and the boxed verdict.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`ReportMetadata` supplying the header
            and footer identity fields.
        :param verbose: Add the per-phase duration and correction breakdown
            columns.
        :param language: Fiche language: ``"es"`` (default, the language of the
            regulation) or ``"en"``.
        :return: The written ``path``.
        :raises ImportError: If reportlab (or, for the figure, matplotlib) is
            not installed.
        """
        from ..._i18n import check_language
        from ..._report.rd1367 import render_activity_report

        return render_activity_report(
            self,
            path,
            metadata=metadata,
            verbose=verbose,
            language=check_language(language),
        )


def _period_durations(
    period_hours: Mapping[str, float] | None,
) -> dict[str, float]:
    """The period durations ``T``, with any caller override applied.

    :param period_hours: Durations in hours keyed by period, or ``None``.
    :return: A duration for every RD 1367 evaluation period.
    :raises ValueError: For an unknown period key or a non-positive duration.
    """
    durations = dict(RD1367_PERIOD_HOURS)
    if period_hours is not None:
        for name, value in period_hours.items():
            durations[_check_period(name)] = _positive(value, "period_hours")
    return durations


def _validate_annual_inputs(
    operating_days: int | None, year_days: int, closed_level: float
) -> None:
    """Reject the annual parameters the Article 25.1 b i criterion cannot use.

    :param operating_days: Days a year the activity operates, or ``None``.
    :param year_days: Days in the year considered.
    :param closed_level: Level representing a day with no operation, dB.
    :raises ValueError: For a fractional or out-of-range count of days.
    """
    if int(year_days) != year_days or int(year_days) <= 0:
        msg = f"'year_days' must be a positive whole number; got {year_days!r}."
        raise ValueError(msg)
    if operating_days is None:
        return
    if int(operating_days) != operating_days:
        msg = (
            f"'operating_days' must be a whole number of days; got {operating_days!r}."
        )
        raise ValueError(msg)
    if not 0 < int(operating_days) <= int(year_days):
        msg = (
            "'operating_days' must be a positive integer no larger than "
            f"'year_days'; got {operating_days!r} of {year_days!r}."
        )
        raise ValueError(msg)
    _finite(closed_level, "closed_level")


def _annual_index(
    reported: int,
    period: str,
    *,
    long_term_levels: Mapping[str, float] | None,
    operating_days: int | None,
    year_days: int,
    closed_level: float,
) -> float | None:
    """The annual index ``LK,x`` of a period: given, derived, or unavailable.

    :param reported: The reported daily ``LKeq,x`` of the period, dB.
    :param period: Evaluation period name.
    :param long_term_levels: Annual index per period when the caller knows it.
    :param operating_days: Days a year the activity operates, or ``None``.
    :param year_days: Days in the year considered.
    :param closed_level: Level of a day with no operation, dB.
    :return: The annual index in dB, or ``None`` when neither input is given.
    """
    if long_term_levels is not None and period in long_term_levels:
        return _finite(long_term_levels[period], "long_term_levels")
    if operating_days is None:
        return None
    closed = int(year_days) - int(operating_days)
    weights = [float(operating_days)] + ([float(closed)] if closed else [])
    values = [float(reported)] + ([closed_level] if closed else [])
    return long_term_corrected_level(values, weights=weights)


def _assess_period(
    name: str,
    phases: tuple[NoisePhase, ...],
    duration: float,
    limit: float,
    *,
    long_term_levels: Mapping[str, float] | None,
    operating_days: int | None,
    year_days: int,
    closed_level: float,
    new_activity: bool,
) -> PeriodAssessment:
    """One evaluation period against the three criteria of Article 25.1 b.

    :param name: Evaluation period name.
    :param phases: The noise phases measured in the period.
    :param duration: Period duration ``T``, hours.
    :param limit: The applicable limit value, dB.
    :param long_term_levels: Annual index per period when the caller knows it.
    :param operating_days: Days a year the activity operates, or ``None``.
    :param year_days: Days in the year considered.
    :param closed_level: Level of a day with no operation, dB.
    :param new_activity: Whether the annual criterion applies.
    :return: The assessment of the period.
    :raises ValueError: If the period carries no noise phases.
    """
    if not phases:
        msg = f"Period {name!r} has no noise phases."
        raise ValueError(msg)
    level = evaluation_period_level(phases, hours=duration)
    reported = round_reported_level(level)
    annual = _annual_index(
        reported,
        name,
        long_term_levels=long_term_levels,
        operating_days=operating_days,
        year_days=year_days,
        closed_level=closed_level,
    )
    max_phase = max(p.lkeq for p in phases)
    return PeriodAssessment(
        period=name,
        phases=phases,
        duration_hours=duration,
        evaluation_period_level=level,
        reported_level=reported,
        long_term_corrected_level=annual,
        reported_long_term=None if annual is None else round_reported_level(annual),
        limit=limit,
        max_phase_level=max_phase,
        phase_pass=bool(max_phase <= limit + _PHASE_ALLOWANCE + 1e-9),
        daily_pass=bool(reported <= limit + _DAILY_ALLOWANCE + 1e-9),
        long_term_pass=(
            None
            if annual is None or not new_activity
            else bool(round_reported_level(annual) <= limit + 1e-9)
        ),
    )


def assess_activity(
    measurements: Mapping[str, Sequence[NoisePhase]],
    limits: RegulationLimits,
    *,
    long_term_levels: Mapping[str, float] | None = None,
    operating_days: int | None = None,
    year_days: int = 365,
    closed_level: float = 0.0,
    new_activity: bool = True,
    period_hours: Mapping[str, float] | None = None,
) -> ActivityAssessment:
    """Assess an activity against the RD 1367/2007 limit values (Article 25).

    For every evaluation period supplied the function integrates the noise
    phases into ``LKeq,x`` (Annex IV A.3.4.2 b), derives the annual ``LK,x``
    when annual information is given, and checks the three criteria of Article
    25.1 b: no annual ``LK,x`` above the table limit, no daily ``LKeq,x`` more
    than 3 dB above it, and no measured ``LKeq,Ti`` more than 5 dB above it.
    With ``new_activity=False`` only the last two apply (Article 25.2, the
    inspection of an activity in operation).

    The annual index can be supplied directly through ``long_term_levels`` or
    derived from ``operating_days``: the reported daily level then represents
    ``operating_days`` days of the year and ``closed_level`` the remaining
    ``year_days - operating_days``.

    :param measurements: Noise phases keyed by evaluation period (``"day"``,
        ``"evening"``, ``"night"``). Periods that are absent are not assessed.
    :param limits: The applicable limit row, from :func:`activity_limits` or
        :func:`adjacent_premises_limits`.
    :param long_term_levels: Annual ``LK,x`` per period, in dB, when known.
    :param operating_days: Number of days a year the activity operates; used
        to derive ``LK,x`` from the daily ``LKeq,x`` when
        ``long_term_levels`` is not given.
    :param year_days: Days in the year considered (default 365).
    :param closed_level: Level, in dB, representing a day on which the
        activity does not operate (default 0 dB, as in the Spanish worked
        literature).
    :param new_activity: Whether the annual criterion applies.
    :param period_hours: Period durations ``T``, in hours, overriding
        :data:`RD1367_PERIOD_HOURS`.
    :return: An :class:`ActivityAssessment`.
    :raises ValueError: For an unknown period key, empty phase list,
        inconsistent durations, or invalid annual parameters.
    """
    if not measurements:
        msg = "At least one evaluation period must be supplied."
        raise ValueError(msg)
    durations = _period_durations(period_hours)
    _validate_annual_inputs(operating_days, year_days, closed_level)

    assessments = [
        _assess_period(
            name,
            tuple(measurements[name]),
            durations[name],
            limits[name],
            long_term_levels=long_term_levels,
            operating_days=operating_days,
            year_days=year_days,
            closed_level=closed_level,
            new_activity=new_activity,
        )
        for name in RD1367_EVALUATION_PERIODS
        if name in measurements
    ]
    if not assessments:
        msg = (
            "None of the supplied keys is an evaluation period; expected "
            f"any of {RD1367_EVALUATION_PERIODS}."
        )
        raise ValueError(msg)
    if new_activity and all(p.long_term_pass is None for p in assessments):
        # Article 25.1 b i is mandatory for a new activity, so an assessment
        # that never evaluates it cannot conclude anything about compliance.
        # Checked after the structural validation so the more specific error
        # about the measurements themselves wins.
        msg = (
            "A new activity is assessed against all three criteria of Article "
            "25.1 b, so the annual index LK,x is required: supply "
            "'long_term_levels' or 'operating_days'. For the inspection of an "
            "activity already in operation, which Article 25.2 subjects only "
            "to the daily and phase criteria, pass new_activity=False."
        )
        raise ValueError(msg)
    return ActivityAssessment(
        periods=tuple(assessments), limits=limits, new_activity=new_activity
    )
