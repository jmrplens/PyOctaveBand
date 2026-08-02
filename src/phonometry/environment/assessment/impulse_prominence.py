#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Prominence of impulsive sounds and the LAeq adjustment (NT ACOU 112:2002).

Noise with prominent impulses is more annoying than a steady sound of the same
equivalent level, so the Nordtest method adds an adjustment ``KI`` to the
measured ``LAeq``. The audibility of an impulse is captured by the **predicted
prominence** ``P``, a logarithmic measure of the onset rate and the level
difference of the impulse (clause 7):

.. math::

   P = 3 \log_{10}(\text{onset rate}) + 2 \log_{10}(\text{level difference})
   \tag{Formula 1}

From the impulse with the highest prominence over a 30-minute period, a
graduated adjustment follows (clause 8):

.. math::

   K_I = 1.8 \, (P - 5)~\text{dB} \quad \text{for } P > 5,
   \text{ else } 0 \tag{Formula 2}

and the rating level over a reference time interval combines the adjusted
sub-interval levels (clause 8, Note 1). An impulse qualifies when its onset
rate exceeds 10 dB/s (clauses 4.5-4.7); non-qualifying level rises receive
no adjustment (clause 8 applies only "for sounds with onset rates larger
than 10 dB/s").
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._internal.warnings import PhonometryWarning

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..._report.metadata import ReportMetadata

from numpy.typing import ArrayLike


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

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the adjustment curve ``KI(P)`` with the impulses marked.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.environmental import plot_impulse_prominence

        return plot_impulse_prominence(self, ax=ax, language=check_language(language), **kwargs)

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
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
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
    if np.any(orate <= 0.0) or np.any(ld <= 0.0):
        raise ValueError("onset_rate and level_difference must be positive.")
    k1, k2 = PROMINENCE_COEFFS
    return k1 * np.log10(orate) + k2 * np.log10(ld)


def impulse_adjustment(prominence: ArrayLike) -> np.ndarray:
    r"""Adjustment ``KI`` to ``LAeq`` from the prominence (clause 8, Formula 2).

    :math:`K_I = 1.8 \, (P - 5)` dB for :math:`P > 5`, else 0 dB. The
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
        raise ValueError(
            f"assessment_period_min must be positive and finite; got "
            f"{assessment_period_min}."
        )
    orate = np.asarray(onset_rates, dtype=np.float64).ravel()
    ld = np.asarray(level_differences, dtype=np.float64).ravel()
    if orate.size == 0:
        raise ValueError("at least one impulse is required.")
    if orate.shape != ld.shape:
        raise ValueError(
            f"onset_rates and level_differences must have the same shape; "
            f"got {orate.shape} and {ld.shape}."
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

       L_{Ar,T} = 10 \log_{10}\!\left[ \frac{1}{T}
       \sum_N \Delta t_N \, 10^{(L_{Aeq,N} + K_{I,N})/10} \right]

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
    if not le.shape == ki.shape == dt.shape:
        raise ValueError("laeq, adjustment and durations must have equal length.")
    if le.size == 0:
        raise ValueError("at least one sub-interval is required.")
    if reference_time <= 0.0 or np.any(dt <= 0.0):
        raise ValueError("reference_time and durations must be positive.")
    energy = np.sum(dt * 10.0 ** ((le + ki) / 10.0))
    return float(10.0 * np.log10(energy / reference_time))
