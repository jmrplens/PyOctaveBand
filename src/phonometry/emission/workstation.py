#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""What the ISO 11200 group shares: the emission sound pressure level at a work
station, its two corrections and its uncertainty.

The sound power level says how much noise a machine makes. The **emission sound
pressure level** says how much of it reaches the person working at it, and it is
the number a machine is declared and bought by. ISO 4871 already has this
library declaring :math:`L_{p\mathrm{A}}`; nothing computed it until now.

Five standards determine it, and they differ only in how they get rid of the
room:

======================  =====================================================
ISO 11201:2010          Free field over a reflecting plane, so there is no
                        room to get rid of and :math:`K_3 = 0`.
ISO 11202:2010          Two approximate routes to :math:`K_3`, one for a
                        machine with a dominating source and one from the
                        directivity the work station sees.
ISO 11203:1995          No sound-pressure measurement at the work station:
                        the level is derived from the sound power level,
                        which was itself determined somehow.
ISO 11204:2010          The same piecewise :math:`K_3` as ISO 11202 method
                        A.2, reached accurately rather than approximately.
ISO 11205:2003          By sound intensity, and not implemented here.
======================  =====================================================

Everything they share is in this module, transcribed once, and each part's own
method sits beside it in its own module.

**The quantity.** ISO 11201:2010 Equation (7), ISO 11202:2010 Equation (10) and
ISO 11204:2010 Equation (9) print one law three times,

.. math::

   L_p = L'_p - K_1 - K_3

where :math:`L'_p` is what the meter read, :math:`K_1` removes the background
noise and :math:`K_3` removes the reflections the room sent back. ISO 11201
prints it without the :math:`K_3` term because its environment is qualified so
that the term is negligible, which is the same equation with a zero in it.

**Peak levels take no correction at all.** ISO 11204:2010 clause 7 and
ISO 11202:2010 clause 8 both say so: :math:`L_{p\mathrm{C,peak}}` is reported as
measured. A correction derived from mean-square pressures has no meaning for a
single largest excursion. Nothing here can tell a peak level from any other,
since both arrive as a number of decibels, so this is a rule for the caller and
not a guard: do not put a peak level through
:func:`emission_sound_pressure_level`.

**The background correction** is the same expression the sound-power side
already uses, ISO 3744:2010 Equation (16), but this group sets its own
thresholds: 15 dB of margin makes it negligible, and 6 dB (grade 2) or 3 dB
(grade 3) is as far down as a result may be claimed. Below that the correction
is clamped and the level becomes an upper bound, which is why
:func:`background_noise_correction_at_workstation` returns the clamp rather than
raising: the reading is still worth reporting, it just stops being a
determination.

**The local environmental correction** is where the group divides. Both
ISO 11202 Equation (A.5) and ISO 11204 Equations (A.2)/(A.5) print the same
piecewise function of one dimensionless ratio :math:`z`,

.. math::

   K_3 = \begin{cases}
     7\ \mathrm{dB}, & z \le 0{,}2 \\
     -10 \lg z\ \mathrm{dB}, & 0{,}2 < z \le 1 \\
     0\ \mathrm{dB}, & z > 1
   \end{cases}

and differ only in how :math:`z` is reached. The two branches meet: at
:math:`z = 0{,}2` the middle branch gives :math:`-10 \lg 0{,}2 = 6{,}99` dB, so
the 7 dB cap is the curve's own value rounded, not a discontinuity.

**The uncertainty** is one pair of equations in all three measuring parts
(ISO 11201 Equations (10) and (11), ISO 11202 (13) and (14), ISO 11204 (12) and
(13)):

.. math::

   \sigma_\mathrm{tot} = \sqrt{\sigma_{R0}^2 + \sigma_\mathrm{omc}^2},
   \qquad U = k\,\sigma_\mathrm{tot}

with :math:`\sigma_{R0}` the reproducibility of the method and
:math:`\sigma_\mathrm{omc}` the instability of the machine itself, estimated
from repeated measurements by Equation (C.1).

.. note::

   Equation (C.1) prints the **sample** standard deviation, with
   :math:`1/(N-1)`, and that is what :func:`operating_standard_deviation`
   computes. The two worked examples of ISO 11200:2014 Annex B do not agree
   with each other on this: Table B.1 divides by :math:`N` and Table B.3 by
   :math:`N-1`. See ``docs/ERRATA.md``.

Sources (clean-room, implemented from the standard texts): ISO 11201:2010,
ISO 11202:2010 and its Amendment 1:2020, ISO 11203:1995 and its Amendment
1:2020, ISO 11204:2010, with the worked examples of ISO 11200:2014 Annex B.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import (
    require_non_negative,
    require_positive,
    require_positive_array,
)
from ._shared import Grade, SoundPowerWarning, _check_grade

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

#: Margin, in decibels, above which the background is negligible and
#: :math:`K_1 = 0` (ISO 11201:2010 5.4.2, ISO 11202:2010 6.4.2,
#: ISO 11204:2010 5.4.2).
NEGLIGIBLE_BACKGROUND_MARGIN_DB = 15.0

#: Smallest signal-to-background margin, in decibels, a result of each grade may
#: be determined from. Below it the correction is clamped there and the level
#: becomes an upper bound (ISO 11202:2010 6.4.1 and Annex B).
MINIMUM_BACKGROUND_MARGIN_DB: dict[Grade, float] = {
    "engineering": 6.0,
    "survey": 3.0,
}

#: Largest local environmental correction, in decibels, that still earns
#: accuracy grade 2. Above it the result is grade 3 (ISO 11202:2010 A.1.3, and
#: the same boundary reached through Condition (A.6) in A.2.5).
GRADE_2_MAX_K3_DB = 4.0

#: Cap on :math:`K_3`, in decibels, and the ratio it applies below
#: (ISO 11202:2010 Eq. (A.5), ISO 11204:2010 Eq. (A.2)).
MAX_K3_DB = 7.0
_K3_CAP_RATIO = 0.2

#: Coverage factor the ISO 11200 group uses when it prints an expanded
#: uncertainty (ISO 11200:2014 Annex B, Tables B.1 to B.4).
DEFAULT_COVERAGE_FACTOR = 1.6

#: Fewest repeated readings Equation (C.1) can be evaluated on.
_MIN_REPEATS = 2


def _like(values: NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Return *values* as a number when they are one, and as an array otherwise.

    The array guards coerce a scalar to a one-element array, and a caller who
    handed over one number wants one number back: every quantity here is
    equally at home as an overall A-weighted value and as a spectrum, so the
    shape of the answer follows the shape of the question.

    The question is the *broadcast* of every argument, not of the first one. A
    scalar environmental correction against a per-band directivity index is one
    value per band, and reading the rank off the correction alone returned the
    first band and dropped the rest, silently.
    """
    return float(values.reshape(-1)[0]) if values.size == 1 else values


@dataclass(frozen=True)
class EmissionPressureResult:
    r"""An emission sound pressure level and the two corrections behind it.

    :ivar level_db: Emission sound pressure level :math:`L_p`, in decibels
        re 20 uPa: what the meter read, less both corrections.
    :ivar measured_level_db: The uncorrected reading :math:`L'_p`, in decibels.
    :ivar background_correction_db: :math:`K_1`, in decibels.
    :ivar local_correction_db: :math:`K_3`, in decibels.
    :ivar grade: Accuracy grade the determination earns.
    :ivar upper_bound: ``True`` when the background margin fell below the
        grade's minimum, so the level is an upper bound rather than a
        determination.
    :ivar standard: The part of the group the determination followed.
    """

    level_db: float | NDArray[np.float64]
    measured_level_db: float | NDArray[np.float64]
    background_correction_db: float | NDArray[np.float64]
    local_correction_db: float | NDArray[np.float64]
    grade: Grade
    upper_bound: bool
    standard: str

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the reading, the two corrections and what is left of them."""
        from .._i18n import check_language
        from .._plot.emission import plot_emission_pressure

        return plot_emission_pressure(
            self, ax=ax, language=check_language(language), **kwargs
        )


def background_noise_correction_at_workstation(
    measured_level_db: ArrayLike,
    background_level_db: ArrayLike,
    *,
    grade: Grade = "engineering",
) -> tuple[float | NDArray[np.float64], bool]:
    r"""Background-noise correction :math:`K_1` for the ISO 11200 group.

    .. math::

       K_1 = -10 \lg \left( 1 - 10^{-0,1 \Delta L} \right)\ \mathrm{dB},
       \qquad \Delta L = L'_p - L_p(B)

    ISO 11201:2010 Equation (5), ISO 11202:2010 Equation (8) and
    ISO 11204:2010 Equation (7), one expression printed three times. It is the
    same closed form as ISO 3744:2010 Equation (16) on the sound-power side, and
    this group puts its own thresholds around it: above
    :data:`NEGLIGIBLE_BACKGROUND_MARGIN_DB` the correction is taken as zero, and
    below the grade's entry in :data:`MINIMUM_BACKGROUND_MARGIN_DB` it is held
    at the value it has there. A held correction does not fail the measurement;
    it makes the level an upper bound, which the second return value reports and
    the caller must carry into what it publishes.

    :param measured_level_db: The reading with the machine running,
        :math:`L'_p`, in decibels; a scalar or one value per band.
    :param background_level_db: The reading with it stopped, :math:`L_p(B)`, in
        decibels, of the same shape.
    :param grade: ``'engineering'`` (grade 2, 6 dB) or ``'survey'`` (grade 3,
        3 dB), which sets how far down a determination may be claimed.
    :return: The correction in decibels, and whether any value was clamped.
    :raises ValueError: If the two arguments do not have the same shape, or the
        grade is neither of the two.
    """
    checked = _check_grade(grade)
    measured = np.asarray(measured_level_db, dtype=np.float64)
    background = np.asarray(background_level_db, dtype=np.float64)
    if measured.shape != background.shape:
        msg = (
            f"'measured_level_db' {measured.shape} and 'background_level_db' "
            f"{background.shape} must have the same shape: one background "
            "reading per source reading."
        )
        raise ValueError(msg)

    floor = MINIMUM_BACKGROUND_MARGIN_DB[checked]
    delta = measured - background
    clamped = np.clip(delta, floor, None)
    correction = -10.0 * np.log10(1.0 - np.power(10.0, -0.1 * clamped))
    correction = np.where(delta > NEGLIGIBLE_BACKGROUND_MARGIN_DB, 0.0, correction)

    held = bool(np.any(delta < floor))
    if held:
        warnings.warn(
            f"the background is within {floor:g} dB of the source in at least "
            f"one band, so K1 is held at its value there and the emission "
            "level is an upper bound, not a determination "
            "(ISO 11202:2010, 6.4.1).",
            SoundPowerWarning,
            stacklevel=2,
        )
    return _like(np.asarray(correction, dtype=np.float64)), held


def local_environmental_correction(ratio: ArrayLike) -> float | NDArray[np.float64]:
    r"""Local environmental correction :math:`K_3` from the ratio :math:`z`.

    .. math::

       K_3 = \begin{cases}
         7\ \mathrm{dB}, & z \le 0{,}2 \\
         -10 \lg z\ \mathrm{dB}, & 0{,}2 < z \le 1 \\
         0\ \mathrm{dB}, & z > 1
       \end{cases}

    ISO 11202:2010 Equation (A.5) and ISO 11204:2010 Equations (A.2) and (A.5),
    the same three lines printed three times; only the route to :math:`z`
    differs between them, and that is :func:`environmental_ratio_from_k2` and
    :func:`environmental_ratio_from_absorption`.

    The cap is the curve's own value, not a separate rule:
    :math:`-10 \lg 0{,}2 = 6{,}99` dB, so the function is continuous where the
    7 dB takes over. The upper branch is a floor for the same reason a
    correction cannot be negative: the room can only add to the reading.

    :param ratio: The dimensionless :math:`z`, strictly positive; a scalar or
        one value per band.
    :return: :math:`K_3` in decibels, of the same shape.
    :raises ValueError: If any value is not strictly positive.
    """
    z = require_positive_array(ratio, "ratio")
    correction = np.where(
        z <= _K3_CAP_RATIO,
        MAX_K3_DB,
        np.where(z <= 1.0, -10.0 * np.log10(np.clip(z, _K3_CAP_RATIO, None)), 0.0),
    )
    # -10 lg 1 is -0.0, and a correction of minus nothing reads as a defect on
    # a report; the two zeros are the same number, so publish the one with the
    # sign a reader expects.
    correction = correction + 0.0
    return _like(np.asarray(correction, dtype=np.float64))


def environmental_ratio_from_k2(
    environmental_correction_db: ArrayLike,
    directivity_index_db: ArrayLike = 0.0,
) -> float | NDArray[np.float64]:
    r"""The ratio :math:`z` from the environmental correction of the test room.

    .. math::

       z = 1 - \left( 1 - 10^{-0,1 K_2} \right) 10^{-0,1 D^*_{I,\mathrm{op}}}

    ISO 11202:2010 Equation (A.4) and ISO 11204:2010 Equation (A.3). :math:`K_2`
    is the average environmental correction of the reference measurement
    surface, the quantity :func:`~phonometry.emission.environmental_correction`
    computes for the sound-power methods, and :math:`D^*_{I,\mathrm{op}}` is the
    apparent directivity index the work station sees.

    With no directivity to speak of the expression collapses to
    :math:`z = 10^{-0,1 K_2}`, so :math:`K_3 = K_2`: a work station that sees
    the machine no more strongly than the measurement surface does needs the
    same correction the surface needed. That holds until the cap bites, at
    :math:`K_2 = 7` dB; above it the local correction stays at
    :data:`MAX_K3_DB` however large the environmental one grows, which is the
    piecewise function of :func:`local_environmental_correction` and not a
    limitation here.

    :param environmental_correction_db: :math:`K_2` in decibels, non-negative.
    :param directivity_index_db: :math:`D^*_{I,\mathrm{op}}` in decibels
        (default 0, no directivity).
    :return: The ratio :math:`z`, of the broadcast shape.
    """
    k2 = np.asarray(environmental_correction_db, dtype=np.float64)
    di = np.asarray(directivity_index_db, dtype=np.float64)
    if np.any(k2 < 0.0):
        msg = "'environmental_correction_db' is a correction and cannot be negative."
        raise ValueError(msg)
    ratio = 1.0 - (1.0 - np.power(10.0, -0.1 * k2)) * np.power(10.0, -0.1 * di)
    return _like(np.asarray(ratio, dtype=np.float64))


def environmental_ratio_from_absorption(
    absorption_area_m2: ArrayLike,
    measurement_surface_m2: float,
    directivity_index_db: ArrayLike = 0.0,
) -> float | NDArray[np.float64]:
    r"""The ratio :math:`z` from the equivalent sound absorption area.

    .. math::

       z = 1 - \frac{1}{1 + A / (4 S_M)}\, 10^{-0,1 D^*_{I,\mathrm{op}}}

    ISO 11204:2010 Equation (A.6). It is the same quantity
    :func:`environmental_ratio_from_k2` returns, reached without going through
    :math:`K_2`: under the ISO 3744 definition
    :math:`K_2 = 10 \lg (1 + 4 S_M / A)` the two are identically equal, which is
    why ISO 11204 A.1.2 says the two routes rest on the same assumptions.

    :param absorption_area_m2: Equivalent sound absorption area :math:`A` of the
        test room, in square metres, strictly positive.
    :param measurement_surface_m2: Area :math:`S_M` of the reference measurement
        surface, in square metres, strictly positive.
    :param directivity_index_db: :math:`D^*_{I,\mathrm{op}}` in decibels
        (default 0).
    :return: The ratio :math:`z`.
    :raises ValueError: If either area is not strictly positive.
    """
    a = require_positive_array(absorption_area_m2, "absorption_area_m2")
    s_m = require_positive(measurement_surface_m2, "measurement_surface_m2")
    di = np.asarray(directivity_index_db, dtype=np.float64)
    ratio = 1.0 - np.power(10.0, -0.1 * di) / (1.0 + a / (4.0 * s_m))
    return _like(np.asarray(ratio, dtype=np.float64))


def emission_sound_pressure_level(
    measured_level_db: ArrayLike,
    *,
    background_correction_db: ArrayLike = 0.0,
    local_correction_db: ArrayLike = 0.0,
) -> float | NDArray[np.float64]:
    r"""The emission sound pressure level, reading less both corrections.

    .. math::

       L_p = L'_p - K_1 - K_3

    ISO 11201:2010 Equation (7) (which prints no :math:`K_3` because its
    environment makes the term negligible), ISO 11202:2010 Equation (10) and
    ISO 11204:2010 Equation (9).

    Never call this for a peak level. ISO 11202:2010 clause 8 and ISO 11204:2010
    clause 7 both forbid correcting :math:`L_{p\mathrm{C,peak}}`, which is
    reported exactly as measured: neither correction has a meaning for a single
    largest excursion, both being derived from mean-square pressures. A peak
    level reaches this function as an ordinary number of decibels and cannot be
    recognised, so the rule is the caller's to keep.

    :param measured_level_db: The uncorrected reading :math:`L'_p`, in decibels.
    :param background_correction_db: :math:`K_1` in decibels (default 0).
    :param local_correction_db: :math:`K_3` in decibels (default 0).
    :return: :math:`L_p` in decibels, of the broadcast shape.
    """
    measured = np.asarray(measured_level_db, dtype=np.float64)
    k1 = np.asarray(background_correction_db, dtype=np.float64)
    k3 = np.asarray(local_correction_db, dtype=np.float64)
    level = np.asarray(measured - k1 - k3, dtype=np.float64)
    return float(level) if level.ndim == 0 else level


def operating_standard_deviation(levels_db: ArrayLike) -> float:
    r"""Standard deviation of the operating and mounting conditions.

    .. math::

       \sigma_\mathrm{omc} = \sqrt{\frac{1}{N-1}
       \sum_{j=1}^{N} \left( L'_{p,j} - \overline{L'_p} \right)^2}

    Equation (C.1), identical in ISO 11201:2010, ISO 11202:2010 and
    ISO 11204:2010. It is the sample standard deviation of levels measured under
    the same nominal conditions, and it answers how repeatable the machine is
    rather than how good the method is: the measurements are made in situ, so
    the readings need no correction before going in.

    The divisor is :math:`N-1`, as printed. The two worked examples of
    ISO 11200:2014 Annex B disagree with each other about that, and
    ``docs/ERRATA.md`` records it; this library follows the equation.

    :param levels_db: Repeated readings under the same conditions, in decibels;
        at least two.
    :return: :math:`\sigma_\mathrm{omc}` in decibels.
    :raises ValueError: If fewer than two readings are given.
    """
    levels = np.asarray(levels_db, dtype=np.float64).ravel()
    if levels.size < _MIN_REPEATS:
        msg = (
            f"'levels_db' holds {levels.size} reading(s); the standard deviation "
            "of Equation (C.1) needs at least two."
        )
        raise ValueError(msg)
    return float(np.std(levels, ddof=1))


def total_standard_deviation(
    reproducibility_db: float, operating_db: float = 0.0
) -> float:
    r"""Total standard deviation of the determination.

    .. math::

       \sigma_\mathrm{tot} = \sqrt{\sigma_{R0}^2 + \sigma_\mathrm{omc}^2}

    ISO 11201:2010 Equation (10), ISO 11202:2010 Equation (13) and
    ISO 11204:2010 Equation (12). The two components are taken as statistically
    independent, which is what lets them add in quadrature: one is a property of
    the method and the other of the machine.

    :param reproducibility_db: :math:`\sigma_{R0}` of the method, in decibels.
    :param operating_db: :math:`\sigma_\mathrm{omc}` of the machine, in decibels
        (default 0, a source whose emission does not wander).
    :return: :math:`\sigma_\mathrm{tot}` in decibels.
    :raises ValueError: If either component is negative.
    """
    sigma_r0 = require_non_negative(reproducibility_db, "reproducibility_db")
    sigma_omc = require_non_negative(operating_db, "operating_db")
    return math.hypot(sigma_r0, sigma_omc)


def emission_expanded_uncertainty(
    total_standard_deviation_db: float,
    coverage_factor: float = DEFAULT_COVERAGE_FACTOR,
) -> float:
    r"""Expanded uncertainty :math:`U = k\,\sigma_\mathrm{tot}`.

    ISO 11201:2010 Equation (11), ISO 11202:2010 Equation (14) and
    ISO 11204:2010 Equation (13). The coverage factor is the caller's to choose:
    :math:`k = 2` gives the two-sided 95 % interval of a normal distribution,
    while the worked examples of ISO 11200:2014 Annex B all print
    :math:`k = 1{,}6`, which is the one-sided factor used when the result is
    compared with a limit value, and is this function's default.

    :param total_standard_deviation_db: :math:`\sigma_\mathrm{tot}` in decibels.
    :param coverage_factor: :math:`k` (default
        :data:`DEFAULT_COVERAGE_FACTOR`).
    :return: :math:`U` in decibels.
    :raises ValueError: If either argument is negative.
    """
    sigma = require_non_negative(
        total_standard_deviation_db, "total_standard_deviation_db"
    )
    k = require_non_negative(coverage_factor, "coverage_factor")
    return sigma * k


def subinterval_level(levels_db: ArrayLike, durations_s: ArrayLike) -> float:
    r"""One level for a cycle made of operating periods of different lengths.

    .. math::

       L_p = 10 \lg \left[ \frac{1}{T} \sum_{i=1}^{N}
       T_i\, 10^{0,1 L_{p,T_i}} \right]\ \mathrm{dB},
       \qquad T = \sum_i T_i

    ISO 11201:2010 Equation (8), ISO 11202:2010 Equation (11) and
    ISO 11204:2010 Equation (10). A machine that idles, cuts and returns spends
    a different length of time in each state, so the states are energy-averaged
    weighted by how long each lasts, not by how many there are.

    :param levels_db: The level of each sub-interval, in decibels.
    :param durations_s: How long each lasted, in seconds, strictly positive and
        of the same length.
    :return: The level of the whole interval, in decibels.
    :raises ValueError: If the two are of different lengths, or a duration is
        not strictly positive.
    """
    levels = np.asarray(levels_db, dtype=np.float64).ravel()
    durations = require_positive_array(durations_s, "durations_s").ravel()
    if levels.size != durations.size:
        msg = (
            f"'levels_db' holds {levels.size} value(s) and 'durations_s' "
            f"{durations.size}: one duration per sub-interval."
        )
        raise ValueError(msg)
    total = float(np.sum(durations))
    return float(
        10.0 * np.log10(np.sum(durations * np.power(10.0, 0.1 * levels)) / total)
    )


def grade_from_local_correction(local_correction_db: ArrayLike) -> Grade:
    r"""The accuracy grade a local environmental correction earns.

    ISO 11202:2010 A.1.3 puts the boundary at
    :data:`GRADE_2_MAX_K3_DB`: a greatest possible :math:`K_3` of 4 dB or less
    is grade 2 (engineering), and more than that is grade 3 (survey). Method A.2
    reaches the same boundary by a different road, Condition (A.6), which is
    algebraically the same 4 dB once (A.4) and (A.5) are substituted into it.

    The worst band decides, since a determination is only as good as its
    weakest part.

    :param local_correction_db: :math:`K_3` in decibels; a scalar or one value
        per band.
    :return: ``'engineering'`` or ``'survey'``.
    """
    k3 = np.asarray(local_correction_db, dtype=np.float64)
    worst = float(np.max(k3)) if k3.size else 0.0
    return "engineering" if worst <= GRADE_2_MAX_K3_DB else "survey"
