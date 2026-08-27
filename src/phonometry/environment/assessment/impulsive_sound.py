#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Prominence of impulsive sounds and the ``LAeq`` adjustment (NT ACOU 112:2002,
ISO/PAS 1996-3:2022).

Noise with prominent impulses is more annoying than a steady sound of the same
equivalent level, so both methods add an adjustment ``KI`` to the measured
``LAeq``. They share the prominence and adjustment formulae, which both take
from Pedersen's method, and differ in what the caller supplies.

**NT ACOU 112:2002** is the closed form: the caller brings the onset rate and
the level difference of the impulse, and the **predicted prominence**

.. math::

   P = 3 \log_{10}(\text{onset rate}) + 2 \log_{10}(\text{level difference})
   \tag{Formula 1}

follows (clause 7). From the impulse with the highest prominence over a
30-minute period, a graduated adjustment follows (clause 8):

.. math::

   K_\mathrm{I} = 1.8 \, (P - 5)~\text{dB} \quad \text{for } P > 5,
   \text{ else } 0 \tag{Formula 2}

and the rating level over a reference time interval combines the adjusted
sub-interval levels (clause 8, Note 1). An impulse qualifies when its onset
rate exceeds 10 dB/s (clauses 4.5-4.7); non-qualifying level rises receive
no adjustment (clause 8 applies only "for sounds with onset rates larger
than 10 dB/s").

**ISO/PAS 1996-3:2022** is the measurement chain that reads those same two
quantities from a calibrated time signal, and categorises the source by the
adjustment it earns (typically 0.0 dB to 9.0 dB):

* the A frequency-weighted, F time-weighted sound pressure level ``LpAF`` is
  computed from the signal and sampled at 10-25 ms intervals (Clause 4);
* an *onset* is a contiguous part of the positive slope of ``LpAF`` where the
  gradient exceeds 10 dB/s; its **starting** and **end** points are found from
  procedures a) to d) of Clause 4, merging events separated by less than 50 ms
  (Clause 3.3, Figure 2);
* for each onset the **level difference** :math:`\mathrm{LD} = L_\mathrm{e} - L_\mathrm{s}`
  and the **onset rate** ``OR`` (the least-squares slope over the onset) are
  measured (Clauses 3.4, 3.5, Figures 1 and 2);
* the prominence (Clause 5, Formula 2) and the adjustment
  :math:`K_\mathrm{I} = 1.8 \cdot (P - 5)` dB for :math:`P > 5` (Clause 6, Formula 3)
  are the NT ACOU 112 formulae above;
* the source is categorised (Clause 7) as *not impulsive* (:math:`K_\mathrm{I} = 0`),
  *regular impulsive* (:math:`0 < K_\mathrm{I} \le 5`) or *highly impulsive*
  (:math:`K_\mathrm{I} > 5`).

The ISO/PAS method for determining ``KI`` is not sensitive to the absolute
calibration of the equipment (Clause 8): onset rate and level difference are
level *differences*, so the adjustment is unchanged by a constant offset. Only
the reported ``LAeq`` and the adjusted ``LAeq`` depend on calibration.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..._internal.validation import (
    check_engine,
    require_equal_shapes,
    require_ranks,
    require_same_length,
)
from ..._internal.warnings import PhonometryWarning
from ...io._resolve import SignalInput, resolve_fs, resolve_samples

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

    from ..._report.metadata import ReportMetadata


class ImpulseProminenceWarning(PhonometryWarning):
    """A supplied level rise does not qualify as an impulse (clause 4.5)."""


# ---------------------------------------------------------------------------
# Normative constants (NT ACOU 112:2002).
# ---------------------------------------------------------------------------

#: Coefficients ``(k1, k2)`` of ``P = k1*lg(onset_rate) + k2*lg(LD)`` (Formula 1).
PROMINENCE_COEFFS: tuple[float, float] = (3.0, 2.0)

#: Adjustment slope ``k3`` in ``KI = k3*(P - k4)`` (Formula 2).
ADJUSTMENT_SLOPE: float = 1.8

#: Prominence threshold ``k4`` below which no adjustment is made (Formula 2).
ADJUSTMENT_THRESHOLD: float = 5.0

#: Minimum onset rate, in dB/s, for a level rise to count as an impulse (4.5).
ONSET_RATE_LIMIT: float = 10.0

#: Default assessment time interval, in minutes, over which the governing
#: impulse is taken: "The default assessment time interval is 30 min"
#: (ISO/PAS 1996-3:2022, Clause 5). Another interval may be assessed, so it is
#: a default rather than a fixed value.
DEFAULT_ASSESSMENT_PERIOD_MIN: float = 30.0


@dataclass(frozen=True)
class ImpulseProminenceResult:
    """Prominence of a set of candidate impulses (NT ACOU 112:2002).

    :ivar onset_rates: Onset rate of each impulse, in dB/s.
    :ivar level_differences: Level difference of each impulse, in dB.
    :ivar per_impulse: Predicted prominence ``P`` of each impulse (Formula 1).
    :ivar qualifies: Whether each event qualifies as an impulse: onset rate
        above 10 dB/s (clause 4.5; clause 8 applies the adjustment "for
        sounds with onset rates larger than 10 dB/s" only).
    :ivar prominence: The governing prominence: the highest ``P`` among the
        qualifying impulses (clause 7), or the highest overall (informational)
        when none qualifies.
    :ivar adjustment: The LAeq adjustment ``KI``, in dB, of the governing
        qualifying impulse (Formula 2); 0 dB when no event qualifies.
    :ivar assessment_period_min: The assessment time interval the impulses were
        selected over, in minutes (Clause 5; 30 min by default).
    """

    onset_rates: np.ndarray
    level_differences: np.ndarray
    per_impulse: np.ndarray
    qualifies: np.ndarray
    prominence: float
    adjustment: float
    assessment_period_min: float = DEFAULT_ASSESSMENT_PERIOD_MIN

    def __post_init__(self) -> None:
        """Reject a set of impulses whose per-impulse columns disagree.

        The fiche prints one row per impulse and, above them, a prominence
        taken over the whole set, so a column short of a row leaves a sheet
        whose headline covers an impulse that is not in the table.

        The set must hold at least one impulse: four length-0 columns agree
        with each other, and the fiche then dies hunting the governing row in
        numpy's bare "argmax of an empty sequence", naming neither the field
        nor the result. :func:`impulse_prominence` refuses empty input for the
        same reason.

        The values must be finite: the producer accepts only positive, finite
        onset rates and level differences, so ``P`` and ``KI`` are finite by
        construction, and a NaN smuggled into a hand-built result would
        otherwise render an affirmative prominence note around a ``nan``
        (or crash the verdict row anonymously when a requirement is set).

        :raises ValueError: if the per-impulse columns disagree, are empty,
            or carry a non-finite value.
        """
        require_ranks(
            self,
            onset_rates=1,
            level_differences=1,
            per_impulse=1,
            qualifies=1,
        )
        require_same_length(
            self,
            "onset_rates",
            "level_differences",
            "per_impulse",
            "qualifies",
            axis="impulse",
        )
        if np.asarray(self.onset_rates).size == 0:
            msg = (
                "ImpulseProminenceResult: 'onset_rates' must carry at least "
                "one impulse; all per-impulse columns are empty."
            )
            raise ValueError(msg)
        for name in ("onset_rates", "level_differences", "per_impulse"):
            if not np.all(np.isfinite(np.asarray(getattr(self, name), np.float64))):
                msg = (
                    f"ImpulseProminenceResult: '{name}' must contain only "
                    "finite values."
                )
                raise ValueError(msg)
        for name in ("prominence", "adjustment"):
            value = getattr(self, name)
            if not math.isfinite(float(value)):
                msg = (
                    f"ImpulseProminenceResult: '{name}' must be finite; got {value!r}."
                )
                raise ValueError(msg)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the adjustment curve ``KI(P)`` with the impulses marked.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.environment import plot_impulse_prominence

        return plot_impulse_prominence(
            self, ax=ax, language=check_language(language), **kwargs
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
        """Render an impulsive-sound prominence assessment fiche to a PDF.

        Writes a one-page assessment report following NT ACOU 112:2002 (carried
        into ISO/PAS 1996-3:2022): the standard-basis line, an optional metadata
        header (source/situation, client, measurement position, instrumentation
        and date, always followed by this result's ``assessment_period_min``), a
        full-width per-impulse table (onset rate, level difference, predicted
        prominence ``P`` and whether the onset qualifies as an impulse) above the
        adjustment-curve plot ``KI(P)`` with the candidate impulses marked, the
        boxed governing prominence ``P`` and the derived ``LAeq`` adjustment
        ``KI`` (Formula 2), an optional verdict row and a prominence-category
        note, and a footer with the fixed disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional
            :class:`~phonometry.ReportMetadata`; ``None`` produces a bare
            assessment fiche (body, result and disclaimer only). A supplied
            ``requirement`` is read as the maximum acceptable governing
            prominence ``P`` (a lower prominence passes).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for signature parity with the other fiches; the
            per-impulse table already shows every candidate, so it has no effect.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``language`` is not one of the supported
            languages, or if ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from ..._i18n import check_language

        check_language(language)
        check_engine(engine)
        from ..._report.iso1996_impulse import render_impulse_prominence_report

        return render_impulse_prominence_report(
            self,
            path,
            metadata=metadata,
            verbose=verbose,
            language=check_language(language),
        )


def predicted_prominence(
    onset_rate: ArrayLike, level_difference: ArrayLike
) -> np.ndarray:
    r"""Predicted prominence ``P`` of an impulse (NT ACOU 112, clause 7).

    :math:`P = 3 \log_{10}(\text{onset rate}) + 2 \log_{10}(\text{level difference})`
    (Formula 1), with :math:`\log_{10}`
    the base-10 logarithm. Both quantities are read from the A-weighted,
    time-weighting-F level history: the onset rate is the slope of the onset in
    dB/s and the level difference is the level rise over the onset in dB
    (clauses 4.6-4.7). An impulse qualifies when its onset rate exceeds
    :data:`ONSET_RATE_LIMIT` (10 dB/s).

    :param onset_rate: Onset rate(s), in dB/s (> 0).
    :param level_difference: Level difference(s), in dB (> 0).
    :return: The predicted prominence ``P`` (scalar inputs give a 0-d array).
    :raises ValueError: for a non-positive onset rate or level difference.
    """
    orate = np.asarray(onset_rate, dtype=np.float64)
    ld = np.asarray(level_difference, dtype=np.float64)
    if not (np.all(np.isfinite(orate)) and np.all(np.isfinite(ld))):
        msg = "onset_rate and level_difference must be finite."
        raise ValueError(msg)
    if np.any(orate <= 0.0) or np.any(ld <= 0.0):
        msg = "onset_rate and level_difference must be positive."
        raise ValueError(msg)
    k1, k2 = PROMINENCE_COEFFS
    return k1 * np.log10(orate) + k2 * np.log10(ld)


def impulse_adjustment(prominence: ArrayLike) -> np.ndarray:
    r"""Adjustment ``KI`` to ``LAeq`` from the prominence (clause 8, Formula 2).

    :math:`K_\mathrm{I} = 1.8 \, (P - 5)` dB for :math:`P > 5`, else 0 dB. The
    adjustment is made
    to ``LAeq,30min`` on the basis of the single impulse with the highest ``P``.
    This helper applies the bare Formula 2; the clause 8 onset-rate
    qualification (> 10 dB/s, clause 4.5) is enforced by
    :func:`impulse_prominence`.

    :param prominence: Predicted prominence ``P``.
    :return: The adjustment ``KI``, in dB, clamped at zero.
    """
    p = np.asarray(prominence, dtype=np.float64)
    return np.maximum(ADJUSTMENT_SLOPE * (p - ADJUSTMENT_THRESHOLD), 0.0)


def impulse_prominence(
    onset_rates: ArrayLike,
    level_differences: ArrayLike,
    *,
    assessment_period_min: float = DEFAULT_ASSESSMENT_PERIOD_MIN,
) -> ImpulseProminenceResult:
    """Governing prominence and adjustment of a set of impulses (clauses 7-8).

    Evaluates the predicted prominence of each candidate impulse (Formula 1),
    takes the highest among the *qualifying* impulses as the governing
    prominence (clause 7) and derives its ``LAeq`` adjustment (Formula 2).
    An event qualifies as an impulse only when its onset rate exceeds
    10 dB/s (clause 4.5); clause 8 applies the adjustment "for sounds with
    onset rates larger than 10 dB/s" only, so non-qualifying events cannot
    produce a ``KI`` (an :class:`ImpulseProminenceWarning` reports them and
    the adjustment is 0 dB when no event qualifies).

    :param onset_rates: Onset rate of each impulse, in dB/s (> 0).
    :param level_differences: Level difference of each impulse, in dB (> 0).
    :param assessment_period_min: The assessment time interval the impulses
        were selected over, in minutes; the standard's default is 30 min
        (Clause 5), and the value is carried through to the fiche.
    :return: An :class:`ImpulseProminenceResult` with the per-impulse and governing
        values and ``.plot()``.
    :raises ValueError: for empty input, mismatched lengths, a non-positive
        onset rate or level difference, or an assessment period that is not
        positive and finite.
    """
    if not math.isfinite(assessment_period_min) or assessment_period_min <= 0.0:
        msg = (
            f"assessment_period_min must be positive and finite; got "
            f"{assessment_period_min}."
        )
        raise ValueError(msg)
    orate = np.asarray(onset_rates, dtype=np.float64).ravel()
    ld = np.asarray(level_differences, dtype=np.float64).ravel()
    if orate.size == 0:
        msg = "at least one impulse is required."
        raise ValueError(msg)
    require_equal_shapes(
        "impulse_prominence",
        {"onset_rates": orate.shape, "level_differences": ld.shape},
        "impulse",
    )
    per_impulse = predicted_prominence(orate, ld)
    qualifies = orate > ONSET_RATE_LIMIT
    if not np.all(qualifies):
        warnings.warn(
            "One or more level rises have onset rates at or below 10 dB/s: "
            "they do not qualify as impulses (NT ACOU 112, clauses 4.5/8) "
            "and cannot produce a KI adjustment.",
            ImpulseProminenceWarning,
            stacklevel=2,
        )
    if np.any(qualifies):
        governing = float(np.max(per_impulse[qualifies]))
        adjustment = float(impulse_adjustment(governing))
    else:
        governing = float(np.max(per_impulse))  # informational only
        adjustment = 0.0
    return ImpulseProminenceResult(
        onset_rates=orate,
        level_differences=ld,
        per_impulse=per_impulse,
        qualifies=qualifies,
        prominence=governing,
        adjustment=adjustment,
        assessment_period_min=float(assessment_period_min),
    )


def rating_level(
    laeq: ArrayLike,
    adjustment: ArrayLike,
    durations: ArrayLike,
    reference_time: float,
) -> float:
    r"""Rating level over a reference time interval (clause 8, Note 1).

    Combines the impulse-adjusted equivalent levels of the measurement
    sub-intervals into a single rating level:

    .. math::

       L_{\mathrm{Ar},T} = 10 \log_{10}\!\left[ \frac{1}{T}
       \sum_N \Delta t_N \, 10^{(L_{\mathrm{Aeq},N} + K_{\mathrm{I},N})/10} \right]

    :param laeq: Equivalent level ``LAeq,N`` of each sub-interval, in dB.
    :param adjustment: Adjustment ``KI,N`` of each sub-interval, in dB.
    :param durations: Duration ``dt_N`` of each sub-interval (any time unit,
        consistent with ``reference_time``).
    :param reference_time: Reference time interval ``T`` (same unit as
        ``durations``); commonly the sum of the durations.
    :return: The rating level ``LAr,T``, in dB.
    :raises ValueError: for mismatched lengths or a non-positive time.
    """
    le = np.atleast_1d(np.asarray(laeq, dtype=np.float64))
    ki = np.atleast_1d(np.asarray(adjustment, dtype=np.float64))
    dt = np.atleast_1d(np.asarray(durations, dtype=np.float64))
    require_equal_shapes(
        "rating_level",
        {"laeq": le.shape, "adjustment": ki.shape, "durations": dt.shape},
        "sub-interval",
    )
    if le.size == 0:
        msg = "at least one sub-interval is required."
        raise ValueError(msg)
    if not (
        np.all(np.isfinite(le)) and np.all(np.isfinite(ki)) and np.all(np.isfinite(dt))
    ):
        msg = "laeq, adjustment and durations must be finite."
        raise ValueError(msg)
    if not math.isfinite(reference_time) or reference_time <= 0.0 or np.any(dt <= 0.0):
        msg = "reference_time and durations must be positive."
        raise ValueError(msg)
    energy = np.sum(dt * 10.0 ** ((le + ki) / 10.0))
    return float(10.0 * np.log10(energy / reference_time))


class ImpulsiveSoundWarning(PhonometryWarning):
    """No qualifying onset (gradient > 10 dB/s) was found in the interval."""


# ---------------------------------------------------------------------------
# Normative constants (ISO/PAS 1996-3:2022).
# ---------------------------------------------------------------------------

#: Reference sound pressure, in pascal (20 uPa).
REFERENCE_PRESSURE: float = 2e-5

#: Gradient, in dB/s, that the positive slope of ``LpAF`` must exceed for a
#: point to belong to an onset (Clauses 3.3 and 4, procedures a-d).
ONSET_GRADIENT_LIMIT: float = 10.0

#: F time-weighting time constant, in seconds (Clause 4, Formula 1).
TIME_WEIGHTING_F_TAU: float = 0.125

#: Shortest separation, in seconds, below which successive onsets are merged
#: and short irregularities on an onset are excluded (Clauses 3.3 and 4 d).
MERGE_GAP: float = 0.050

#: Permitted range of the ``LpAF`` sampling interval, in seconds (Clause 4).
SAMPLE_INTERVAL_RANGE: tuple[float, float] = (0.010, 0.025)

#: Default ``LpAF`` sampling interval, in seconds (within the permitted range).
DEFAULT_SAMPLE_INTERVAL: float = 0.020

#: Upper bound of the *regular impulsive* category (Clause 7); above it the
#: source is *highly impulsive*.
HIGHLY_IMPULSIVE_LIMIT: float = 5.0

#: The three source categories of Clause 7, which are both the value reported in
#: ``ImpulsiveSoundResult.category`` and the key of their plot translation.
_CATEGORY_NOT_IMPULSIVE = "not impulsive"
_CATEGORY_REGULAR_IMPULSIVE = "regular impulsive"
_CATEGORY_HIGHLY_IMPULSIVE = "highly impulsive"

#: Fewest level samples that determine a slope: two points are the minimum for
#: the least-squares onset-rate fit (Clause 3.5) and for the gradient of the
#: level history (Clause 4).
_MIN_SLOPE_SAMPLES: int = 2

_OnsetRateMethod = Literal["least_squares", "upper_half"]


@dataclass(frozen=True)
class ImpulseOnset:
    r"""A single detected onset of ``LpAF`` (ISO/PAS 1996-3, Clause 3).

    :ivar index_start: Sample index of the starting point ``s``.
    :ivar index_end: Sample index of the end point ``e``.
    :ivar time_start: Time of the starting point, in seconds.
    :ivar time_end: Time of the end point, in seconds.
    :ivar level_start: Level ``Ls`` at the starting point, in dB.
    :ivar level_end: Level ``Le`` at the end point, in dB.
    :ivar level_difference: Level difference :math:`\mathrm{LD} = L_\mathrm{e} - L_\mathrm{s}`,
        in dB (3.4).
    :ivar onset_rate: Onset rate ``OR``, in dB/s, the least-squares slope over
        the onset (3.5).
    :ivar prominence: Predicted prominence ``P`` of this onset (Formula 2).
    :ivar qualifies: Whether the onset rate exceeds 10 dB/s, so the onset can
        contribute an adjustment (Clause 6).
    """

    index_start: int
    index_end: int
    time_start: float
    time_end: float
    level_start: float
    level_end: float
    level_difference: float
    onset_rate: float
    prominence: float
    qualifies: bool


@dataclass(frozen=True)
class ImpulsiveSoundResult:
    """Objective prominence of an impulsive interval (ISO/PAS 1996-3:2022).

    :ivar times: Time of each ``LpAF`` sample, in seconds.
    :ivar levels: A-weighted, F time-weighted level ``LpAF``, in dB.
    :ivar dt: Sampling interval of ``levels``, in seconds.
    :ivar onsets: The detected onsets, ordered in time (Clause 4).
    :ivar prominence: Governing prominence ``P``: the highest ``P`` among the
        qualifying onsets (Clause 5); ``nan`` when none qualifies.
    :ivar adjustment: The ``LAeq`` adjustment ``KI``, in dB (Formula 3); 0 dB
        when no onset qualifies.
    :ivar category: Source category (Clause 7): ``"not impulsive"``,
        ``"regular impulsive"`` or ``"highly impulsive"``.
    :ivar laeq: A-weighted equivalent level of the interval, in dB.
    :ivar adjusted_laeq: ``laeq + adjustment``, in dB.
    """

    times: np.ndarray
    levels: np.ndarray
    dt: float
    onsets: tuple[ImpulseOnset, ...]
    prominence: float
    adjustment: float
    category: str
    laeq: float
    adjusted_laeq: float

    def __post_init__(self) -> None:
        """Reject a trace whose levels and time axis do not line up.

        The figure is the one place either array is read again, in the call
        that opens it, ``ax.plot(result.times, result.levels, ...)``, and the
        two ways of disagreeing land very differently there.

        A length mismatch is not the silent one. Measured without this guard,
        matplotlib refuses it in both directions, "x and y must have same
        first dimension, but have shapes (100,) and (99,)", and since the
        trace is the first thing drawn nothing beyond it is reached: no onset
        mark, no governing annotation, no labelled axes. What the guard adds
        there is the name of the field that is wrong, at the point the result
        is built, in place of two bare shapes raised from inside a drawing
        call by whoever happened to plot it.

        The silent one is the extra axis. A ``levels`` of shape ``(n, 2)``
        draws without a word, one line per column, so the figure comes back
        with a second trace over the onsets that is no level history at all,
        both entries in the legend named ``LpAF``, and the level axis widened
        to take the intruder in.

        :raises ValueError: if ``times`` and ``levels`` disagree.
        """
        require_ranks(self, times=1, levels=1)
        require_same_length(self, "times", "levels", axis="sample")

    @property
    def governing_onset(self) -> ImpulseOnset | None:
        """The qualifying onset with the highest prominence, or ``None``."""
        qualifying = [o for o in self.onsets if o.qualifies]
        if not qualifying:
            return None
        return max(qualifying, key=lambda o: o.prominence)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot ``LpAF`` versus time with the detected onsets marked.

        Draws the level history, the starting/end points of each onset, the
        least-squares onset line and the level difference of the governing
        impulse, annotated with the prominence, adjustment and category.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        return _plot_impulsive_sound(self, ax=ax, language=language, **kwargs)


# ---------------------------------------------------------------------------
# Sound pressure level time history (Clause 4).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelHistory:
    """A frequency- and time-weighted level trace, and the axis it lives on.

    What :func:`sound_pressure_level_history` computes is ``LpAF``: the
    A-weighted, F time-weighted sound pressure level sampled every ``dt``
    seconds. It came back as a bare ``(times, levels)`` pair, which meant
    the one thing every other result in this library offers, a plot that
    knows what it is drawing, had to be written by hand each time.

    For backward compatibility with that pair, the dataclass is iterable
    and unpacks as ``times, levels = sound_pressure_level_history(...)``,
    the same way :class:`~phonometry.room.DecayCurve` replaced its own
    tuple return.

    :ivar times: Time of each sample, in seconds from the start.
    :ivar levels: The level at each of those times, in dB.
    :ivar dt: Sampling interval of ``levels``, in seconds.
    """

    times: np.ndarray
    levels: np.ndarray
    dt: float

    def __post_init__(self) -> None:
        """Reject a trace whose levels and time axis are of different lengths.

        The history is handed on as a pair, and it unpacks as one, so whoever
        receives it puts the two arrays together again: a window taken as
        ``levels[times > t]`` needs them to index alike, and the plot draws
        one against the other. A pair whose halves disagree is not a level
        history at all, so it is refused where it is made rather than at each
        place it is opened.

        :raises ValueError: if ``times`` and ``levels`` disagree.
        """
        require_ranks(self, times=1, levels=1)
        require_same_length(self, "times", "levels", axis="sample")

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield ``times`` then ``levels`` so the result unpacks like a tuple."""
        yield self.times
        yield self.levels

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the level trace against time.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        return _plot_level_history(self, ax=ax, language=language, **kwargs)


def sound_pressure_level_history(
    signal: SignalInput,
    fs: float | None = None,
    *,
    dt: float = DEFAULT_SAMPLE_INTERVAL,
    reference_pressure: float = REFERENCE_PRESSURE,
    calibration_offset: float = 0.0,
) -> LevelHistory:
    r"""A frequency-weighted, F time-weighted level history ``LpAF`` (Clause 4).

    The signal is A-weighted (IEC 61672-1), F time-weighted
    (:math:`\tau = 125` ms)
    and sampled at intervals ``dt`` in the 10-25 ms range required by the
    standard.

    :param signal: Calibrated sound pressure signal, in pascal. Accepts a
        :class:`phonometry.io.Signal`, whose calibration is applied to the
        samples: that factor is what puts them in pascal, and it is a
        different knob from ``calibration_offset``, which shifts the
        finished level in decibels for a record that never was.
    :param fs: Sampling rate of ``signal``, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param dt: Target sampling interval of ``LpAF``, in seconds (10-25 ms).
    :param reference_pressure: Reference pressure, in pascal (default 20 uPa).
    :param calibration_offset: Level offset added to ``LpAF``, in dB, for
        signals recorded on a scale other than pascal.
    :return: A :class:`LevelHistory`, which unpacks as ``(times, levels)``
        for the callers that always did: the sample times in seconds and ``LpAF`` in
        dB. The realised interval is ``times[1] - times[0]`` and may differ
        slightly from ``dt`` because it is an integer number of samples.
    :raises ValueError: for a non-positive ``fs`` or ``dt`` outside 10-25 ms.
    """
    from ...filters.weighting import time_weighting, weighting_filter

    fs = resolve_fs(signal, fs, name="signal")
    x = np.asarray(resolve_samples(signal), dtype=np.float64).ravel()
    if not math.isfinite(fs) or fs <= 0.0:
        msg = "fs must be positive."
        raise ValueError(msg)
    lo, hi = SAMPLE_INTERVAL_RANGE
    if not lo <= dt <= hi:
        msg = f"dt must be within {lo * 1e3:g}-{hi * 1e3:g} ms (Clause 4); got {dt * 1e3:g} ms."
        raise ValueError(msg)
    if x.size == 0:
        msg = "signal must not be empty."
        raise ValueError(msg)

    weighted = np.asarray(weighting_filter(x, round(fs), curve="A"), dtype=np.float64)
    # Warm-start the F integrator at the first window's mean square so a steady
    # interval does not show a spurious onset from the processing start-up.
    window = min(weighted.size, max(1, round(TIME_WEIGHTING_F_TAU * fs)))
    warm_start = float(np.mean(weighted[:window] ** 2))
    mean_square = np.asarray(
        time_weighting(weighted, round(fs), mode="fast", initial_state=warm_start),
        dtype=np.float64,
    )

    step = max(1, round(fs * dt))
    idx = np.arange(0, mean_square.size, step)
    sampled = mean_square[idx]
    floor = np.finfo(np.float64).tiny
    levels = (
        10.0 * np.log10(np.maximum(sampled, floor) / reference_pressure**2)
        + calibration_offset
    )
    times = idx / fs
    # The realised interval, not the target: the step is quantised to whole
    # samples, so storing the request would put a dt in the result that its
    # own times disagree with.
    realised = float(times[1] - times[0]) if times.size > 1 else float(dt)
    return LevelHistory(times=times, levels=levels, dt=realised)


def _equivalent_level(
    signal: np.ndarray, fs: float, reference_pressure: float, calibration_offset: float
) -> float:
    """A-weighted equivalent continuous level ``LAeq`` of the interval, in dB."""
    from ...filters.weighting import weighting_filter

    weighted = np.asarray(
        weighting_filter(signal, round(fs), curve="A"), dtype=np.float64
    )
    mean_square = float(np.mean(weighted**2))
    floor = np.finfo(np.float64).tiny
    return (
        float(10.0 * np.log10(max(mean_square, floor) / reference_pressure**2))
        + calibration_offset
    )


# ---------------------------------------------------------------------------
# Onset detection (Clause 4, procedures a-d).
# ---------------------------------------------------------------------------


def _raw_runs(gradient: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, end)`` sample-index pairs of maximal rising runs.

    A rising run is a maximal set of consecutive gradient samples exceeding the
    limit; the onset spans level samples ``start .. end`` inclusive.
    """
    above = gradient > ONSET_GRADIENT_LIMIT
    runs: list[tuple[int, int]] = []
    n = above.size
    i = 0
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            runs.append((i, j + 1))  # levels[i .. j+1] is the rising segment
            i = j + 1
        else:
            i += 1
    return runs


def _merge_runs(
    runs: list[tuple[int, int]], times: np.ndarray, levels: np.ndarray
) -> list[tuple[int, int]]:
    """Merge onsets per procedure d) (Clause 4, Figure 2).

    When a new starting point occurs within 50 ms after the previous end point
    and both the end-to-end and start-to-start slopes exceed 10 dB/s, the
    intermediate points are absorbed and the onset extends to the later end
    point. This also excludes irregularities shorter than 50 ms (Clause 3.3).
    """
    if not runs:
        return runs
    merged = [runs[0]]
    for s2, e2 in runs[1:]:
        s1, e1 = merged[-1]
        gap = times[s2] - times[e1]
        end_slope = (levels[e2] - levels[e1]) / (times[e2] - times[e1])
        start_slope = (levels[s2] - levels[s1]) / (times[s2] - times[s1])
        if (
            0.0 <= gap <= MERGE_GAP
            and end_slope > ONSET_GRADIENT_LIMIT
            and start_slope > ONSET_GRADIENT_LIMIT
        ):
            merged[-1] = (s1, e2)
        else:
            merged.append((s2, e2))
    return merged


def _onset_rate(
    times: np.ndarray, levels: np.ndarray, method: _OnsetRateMethod
) -> float:
    """Least-squares slope of the onset, in dB/s (Clause 3.5)."""
    t = times
    lev = levels
    if method == "upper_half":
        # Pass-by variant (3.5, Note 1): fit over the upper half of the slope,
        # levels from Le - (Le - Ls)/2 to Le.
        threshold = lev[-1] - (lev[-1] - lev[0]) / 2.0
        mask = lev >= threshold
        if np.count_nonzero(mask) >= _MIN_SLOPE_SAMPLES:
            t = t[mask]
            lev = lev[mask]
    if t.size < _MIN_SLOPE_SAMPLES:
        return float((levels[-1] - levels[0]) / (times[-1] - times[0]))
    slope = np.polyfit(t - t[0], lev, 1)[0]
    return float(slope)


def detect_onsets(
    levels: ArrayLike,
    dt: float,
    *,
    onset_rate_method: _OnsetRateMethod = "least_squares",
) -> tuple[ImpulseOnset, ...]:
    r"""Detect the onsets in an ``LpAF`` level history (Clause 4).

    Applies procedures a) to d) of the standard: the starting point is the
    first sample where the gradient exceeds 10 dB/s, the end point the first
    later sample where it drops below 10 dB/s, and onsets separated by less
    than 50 ms are merged. Each onset carries its level difference
    :math:`\mathrm{LD} = L_\mathrm{e} - L_\mathrm{s}` (3.4), its onset rate (3.5) and its
    prominence ``P``.

    :param levels: A-weighted, F time-weighted level history ``LpAF``, in dB,
        uniformly sampled with interval ``dt``.
    :param dt: Sampling interval of ``levels``, in seconds (must be positive).
    :param onset_rate_method: ``"least_squares"`` (default) fits the whole
        onset; ``"upper_half"`` fits the upper half of the slope, the variant
        for pass-bys of road vehicles, trains or aircraft (3.5, Note 1).
    :return: The detected onsets, ordered in time (empty when none is found).
    :raises ValueError: for a non-positive ``dt`` or fewer than two samples.
    """
    lev = np.asarray(levels, dtype=np.float64).ravel()
    if not math.isfinite(dt) or dt <= 0.0:
        msg = "dt must be positive."
        raise ValueError(msg)
    if lev.size < _MIN_SLOPE_SAMPLES:
        msg = "at least two level samples are required."
        raise ValueError(msg)
    times = np.arange(lev.size) * dt
    gradient = np.diff(lev) / dt

    runs = _merge_runs(_raw_runs(gradient), times, lev)

    onsets: list[ImpulseOnset] = []
    for s, e in runs:
        ld = float(lev[e] - lev[s])
        orate = _onset_rate(times[s : e + 1], lev[s : e + 1], onset_rate_method)
        qualifies = orate > ONSET_GRADIENT_LIMIT and ld > 0.0
        if orate > 0.0 and ld > 0.0:
            prominence = float(predicted_prominence(orate, ld))
        else:
            prominence = float("nan")
        onsets.append(
            ImpulseOnset(
                index_start=int(s),
                index_end=int(e),
                time_start=float(times[s]),
                time_end=float(times[e]),
                level_start=float(lev[s]),
                level_end=float(lev[e]),
                level_difference=ld,
                onset_rate=orate,
                prominence=prominence,
                qualifies=bool(qualifies),
            )
        )
    return tuple(onsets)


# ---------------------------------------------------------------------------
# Objective adjustment (Clauses 5-7).
# ---------------------------------------------------------------------------


def _categorise(adjustment: float) -> str:
    """Source category from the adjustment (Clause 7)."""
    if adjustment <= 0.0:
        return _CATEGORY_NOT_IMPULSIVE
    if adjustment <= HIGHLY_IMPULSIVE_LIMIT:
        return _CATEGORY_REGULAR_IMPULSIVE
    return _CATEGORY_HIGHLY_IMPULSIVE


def impulsive_sound_adjustment(
    signal: SignalInput,
    fs: float | None = None,
    *,
    dt: float = DEFAULT_SAMPLE_INTERVAL,
    reference_pressure: float = REFERENCE_PRESSURE,
    calibration_offset: float = 0.0,
    onset_rate_method: _OnsetRateMethod = "least_squares",
    laeq: float | None = None,
) -> ImpulsiveSoundResult:
    """Objective prominence adjustment of an impulsive interval (ISO/PAS 1996-3).

    Computes the ``LpAF`` history from the calibrated signal (Clause 4), detects
    the onsets, evaluates the prominence of each and returns the governing
    adjustment ``KI`` (Formula 3) of the most prominent qualifying impulse,
    together with the source category (Clause 7) and the adjusted ``LAeq``.

    :param signal: Calibrated sound pressure signal of the candidate event,
        in pascal. Accepts a
        :class:`phonometry.io.Signal`, whose calibration is applied to the
        samples: that factor is what puts them in pascal, and it is a
        different knob from ``calibration_offset``, which shifts the
        finished level in decibels for a record that never was.
    :param fs: Sampling rate of ``signal``, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param dt: Target ``LpAF`` sampling interval, in seconds (10-25 ms).
    :param reference_pressure: Reference pressure, in pascal (default 20 uPa).
    :param calibration_offset: Level offset, in dB, for signals not scaled to
        pascal. The adjustment ``KI`` is unaffected by it (Clause 8); only the
        reported levels shift.
    :param onset_rate_method: ``"least_squares"`` (default) or ``"upper_half"``
        for pass-bys (3.5, Note 1).
    :param laeq: Equivalent level of the interval, in dB; when omitted it is
        computed from the A-weighted signal energy.
    :return: An :class:`ImpulsiveSoundResult` with the level history, onsets,
        prominence, adjustment, category and adjusted ``LAeq``.
    :raises ValueError: for invalid ``fs``, ``dt`` or an empty signal.
    """
    fs = resolve_fs(signal, fs, name="signal")
    x = np.asarray(resolve_samples(signal), dtype=np.float64).ravel()
    times, levels = sound_pressure_level_history(
        x,
        fs,
        dt=dt,
        reference_pressure=reference_pressure,
        calibration_offset=calibration_offset,
    )
    realised_dt = float(times[1] - times[0]) if times.size > 1 else dt
    onsets = detect_onsets(levels, realised_dt, onset_rate_method=onset_rate_method)

    qualifying = [o for o in onsets if o.qualifies]
    if qualifying:
        prominence = max(o.prominence for o in qualifying)
        adjustment = float(impulse_adjustment(prominence))
    else:
        prominence = float("nan")
        adjustment = 0.0
        warnings.warn(
            "No onset with a gradient above 10 dB/s was found; the interval is "
            "not impulsive and the adjustment is 0 dB (ISO/PAS 1996-3, Clause 6).",
            ImpulsiveSoundWarning,
            stacklevel=2,
        )

    if laeq is None:
        laeq = _equivalent_level(x, fs, reference_pressure, calibration_offset)

    return ImpulsiveSoundResult(
        times=times,
        levels=levels,
        dt=realised_dt,
        onsets=onsets,
        prominence=prominence,
        adjustment=adjustment,
        category=_categorise(adjustment),
        laeq=float(laeq),
        adjusted_laeq=float(laeq + adjustment),
    )


# ---------------------------------------------------------------------------
# Plotting (lazy matplotlib, self-contained EN/ES labels).
# ---------------------------------------------------------------------------

_PLOT_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "xlabel": "Time [s]",
        "ylabel": "$L_{pAF}$ [dB]",
        "level": "$L_{pAF}$",
        "onset": "onset",
        "start": "start",
        "end": "end",
        "ld": "LD",
        "title": "Impulsive-sound prominence (ISO/PAS 1996-3)",
        _CATEGORY_NOT_IMPULSIVE: _CATEGORY_NOT_IMPULSIVE,
        _CATEGORY_REGULAR_IMPULSIVE: _CATEGORY_REGULAR_IMPULSIVE,
        _CATEGORY_HIGHLY_IMPULSIVE: _CATEGORY_HIGHLY_IMPULSIVE,
        "summary": "P = {p}, K_I = {k} dB ({cat})",
        "nosummary": "no qualifying onset: K_I = 0 dB (not impulsive)",
    },
    "es": {
        "xlabel": "Tiempo [s]",
        "ylabel": "$L_{pAF}$ [dB]",
        "level": "$L_{pAF}$",
        "onset": "inicio",
        "start": "principio",
        "end": "final",
        "ld": "LD",
        "title": "Prominencia de sonido impulsivo (ISO/PAS 1996-3)",
        _CATEGORY_NOT_IMPULSIVE: "no impulsivo",
        _CATEGORY_REGULAR_IMPULSIVE: "impulsivo regular",
        _CATEGORY_HIGHLY_IMPULSIVE: "altamente impulsivo",
        "summary": "P = {p}, K_I = {k} dB ({cat})",
        "nosummary": "sin inicio válido: K_I = 0 dB (no impulsivo)",
    },
}


def _plot_level_history(
    result: LevelHistory,
    *,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """The bare ``LpAF`` trace, without the onset marks of the full assessment.

    The same axes as :func:`_plot_impulsive_sound` draws underneath its
    prominence marks, so the two figures read the same way when a reader
    puts them side by side.
    """
    from ..._i18n import check_language, localize_axes

    lang = check_language(language)
    labels = _PLOT_LABELS[lang]

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    kwargs.setdefault("lw", 1.0)
    ax.plot(result.times, result.levels, label=labels["level"], **kwargs)
    ax.set_xlabel(labels["xlabel"])
    ax.set_ylabel(labels["ylabel"])
    ax.grid(True, alpha=0.3)
    localize_axes(ax, lang)
    return ax


def _plot_impulsive_sound(
    result: ImpulsiveSoundResult,
    *,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    from ..._i18n import check_language

    lang = check_language(language)
    labels = _PLOT_LABELS[lang]

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    # The level history is drawn neutral so the coloured onset marks read on
    # top of it, but a fixed grey can only be neutral on one page: "0.3" is a
    # dark grey on the white theme and all but gone on the dark one. Mix the
    # page towards its own ink instead, which darkens on white and lightens on
    # black and keeps the same distance from the page on both.
    from ..._plot.common import theme_line

    ink = theme_line(ax.xaxis.label.get_color(), ax, quiet=0.7)
    ax.plot(
        result.times, result.levels, color=ink, lw=1.2, label=labels["level"], **kwargs
    )

    for onset in result.onsets:
        color = "tab:red" if onset.qualifies else "tab:orange"
        ax.plot(onset.time_start, onset.level_start, "o", color=color, ms=6)
        ax.plot(onset.time_end, onset.level_end, "s", color=color, ms=6)
        # Least-squares onset line across the onset span. Anchor the fitted
        # slope at the span midpoint (the regression line passes through the
        # sample centroid, not necessarily the start sample), so the drawn
        # trend does not overshoot the marked endpoints.
        t_mid = 0.5 * (onset.time_start + onset.time_end)
        l_mid = 0.5 * (onset.level_start + onset.level_end)
        line_t = np.array([onset.time_start, onset.time_end])
        line_l = l_mid + onset.onset_rate * (line_t - t_mid)
        ax.plot(line_t, line_l, color=color, lw=1.6, ls="--")

    governing = result.governing_onset
    if governing is not None:
        ax.annotate(
            "",
            xy=(governing.time_start, governing.level_end),
            xytext=(governing.time_start, governing.level_start),
            arrowprops={"arrowstyle": "<->", "color": "tab:blue"},
        )
        ax.text(
            governing.time_start,
            (governing.level_start + governing.level_end) / 2.0,
            f" {labels['ld']} = {governing.level_difference:.1f} dB",
            color="tab:blue",
            va="center",
        )
        summary = labels["summary"].format(
            p=f"{result.prominence:.2f}",
            k=f"{result.adjustment:.2f}",
            cat=labels[result.category],
        )
    else:
        summary = labels["nosummary"]

    ax.set_xlabel(labels["xlabel"])
    ax.set_ylabel(labels["ylabel"])
    ax.set_title(f"{labels['title']}\n{summary}")
    ax.legend(loc="best")
    ax.grid(True, ls=":", alpha=0.5)
    return ax
