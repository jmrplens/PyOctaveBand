#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Objective prominence of impulsive sounds and the ``LAeq`` adjustment
(ISO/PAS 1996-3:2022).

ISO/PAS 1996-3 objectively categorises a source by how prominently its
impulsive sound is perceived and derives an adjustment ``KI`` (typically in the
range 0,0 dB to 9,0 dB) that is added to ``LAeq``. Unlike the closed-form
:mod:`~phonometry.environmental.impulse_prominence` helpers (NT ACOU 112, which
take the onset rate and level difference as inputs), this module implements the
*objective measurement chain* that reads those quantities directly from a
calibrated time signal.

The chain follows the standard:

* the A frequency-weighted, F time-weighted sound pressure level ``LpAF`` is
  computed from the signal and sampled at 10-25 ms intervals (Clause 4);
* an *onset* is a contiguous part of the positive slope of ``LpAF`` where the
  gradient exceeds 10 dB/s; its **starting** and **end** points are found from
  procedures a) to d) of Clause 4, merging events separated by less than 50 ms
  (Clause 3.3, Figure 2);
* for each onset the **level difference** ``LD = Le - Ls`` and the **onset
  rate** ``OR`` (the least-squares slope over the onset) are measured
  (Clauses 3.4, 3.5, Figures 1 and 2);
* the **prominence** ``P = 3*lg(OR) + 2*lg(LD)`` follows (Clause 5, Formula 2)
  and the impulse with the highest ``P`` gives the **adjustment**
  ``KI = 1.8*(P - 5)`` dB for ``P > 5``, else 0 dB (Clause 6, Formula 3);
* the source is categorised (Clause 7) as *not impulsive* (``KI = 0``),
  *regular impulsive* (``0 < KI <= 5``) or *highly impulsive* (``KI > 5``).

The prominence and adjustment formulae are shared with NT ACOU 112 (both derive
from Pedersen's method) and are reused from
:mod:`~phonometry.environmental.impulse_prominence`. The method for determining
``KI`` is not sensitive to the absolute calibration of the equipment
(Clause 8): onset rate and level difference are level *differences*, so the
adjustment is unchanged by a constant offset. Only the reported ``LAeq`` and the
adjusted ``LAeq`` depend on calibration.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from .._internal.warnings import PhonometryWarning
from .impulse_prominence import impulse_adjustment, predicted_prominence

if TYPE_CHECKING:
    from matplotlib.axes import Axes


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

_OnsetRateMethod = Literal["least_squares", "upper_half"]


@dataclass(frozen=True)
class ImpulseOnset:
    """A single detected onset of ``LpAF`` (ISO/PAS 1996-3, Clause 3).

    :ivar index_start: Sample index of the starting point ``s``.
    :ivar index_end: Sample index of the end point ``e``.
    :ivar time_start: Time of the starting point, in seconds.
    :ivar time_end: Time of the end point, in seconds.
    :ivar level_start: Level ``Ls`` at the starting point, in dB.
    :ivar level_end: Level ``Le`` at the end point, in dB.
    :ivar level_difference: Level difference ``LD = Le - Ls``, in dB (3.4).
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

    @property
    def governing_onset(self) -> ImpulseOnset | None:
        """The qualifying onset with the highest prominence, or ``None``."""
        qualifying = [o for o in self.onsets if o.qualifies]
        if not qualifying:
            return None
        return max(qualifying, key=lambda o: o.prominence)

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
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


def sound_pressure_level_history(
    signal: ArrayLike,
    fs: float,
    *,
    dt: float = DEFAULT_SAMPLE_INTERVAL,
    reference_pressure: float = REFERENCE_PRESSURE,
    calibration_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A frequency-weighted, F time-weighted level history ``LpAF`` (Clause 4).

    The signal is A-weighted (IEC 61672-1), F time-weighted (``tau = 125 ms``)
    and sampled at intervals ``dt`` in the 10-25 ms range required by the
    standard.

    :param signal: Calibrated sound pressure signal, in pascal.
    :param fs: Sampling rate of ``signal``, in Hz.
    :param dt: Target sampling interval of ``LpAF``, in seconds (10-25 ms).
    :param reference_pressure: Reference pressure, in pascal (default 20 uPa).
    :param calibration_offset: Level offset added to ``LpAF``, in dB, for
        signals recorded on a scale other than pascal.
    :return: ``(times, levels)``, the sample times in seconds and ``LpAF`` in
        dB. The realised interval is ``times[1] - times[0]`` and may differ
        slightly from ``dt`` because it is an integer number of samples.
    :raises ValueError: for a non-positive ``fs`` or ``dt`` outside 10-25 ms.
    """
    from ..metrology.parametric_filters import time_weighting, weighting_filter

    x = np.asarray(signal, dtype=np.float64).ravel()
    if fs <= 0.0:
        raise ValueError("fs must be positive.")
    lo, hi = SAMPLE_INTERVAL_RANGE
    if not lo <= dt <= hi:
        raise ValueError(
            f"dt must be within {lo * 1e3:g}-{hi * 1e3:g} ms (Clause 4); got {dt * 1e3:g} ms."
        )
    if x.size == 0:
        raise ValueError("signal must not be empty.")

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
    levels = 10.0 * np.log10(np.maximum(sampled, floor) / reference_pressure**2) + calibration_offset
    times = idx / fs
    return times, levels


def _equivalent_level(
    signal: np.ndarray, fs: float, reference_pressure: float, calibration_offset: float
) -> float:
    """A-weighted equivalent continuous level ``LAeq`` of the interval, in dB."""
    from ..metrology.parametric_filters import weighting_filter

    weighted = np.asarray(weighting_filter(signal, round(fs), curve="A"), dtype=np.float64)
    mean_square = float(np.mean(weighted**2))
    floor = np.finfo(np.float64).tiny
    return float(10.0 * np.log10(max(mean_square, floor) / reference_pressure**2)) + calibration_offset


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
        if np.count_nonzero(mask) >= 2:
            t = t[mask]
            lev = lev[mask]
    if t.size < 2:
        return float((levels[-1] - levels[0]) / (times[-1] - times[0]))
    slope = np.polyfit(t - t[0], lev, 1)[0]
    return float(slope)


def detect_onsets(
    levels: ArrayLike,
    dt: float,
    *,
    onset_rate_method: _OnsetRateMethod = "least_squares",
) -> tuple[ImpulseOnset, ...]:
    """Detect the onsets in an ``LpAF`` level history (Clause 4).

    Applies procedures a) to d) of the standard: the starting point is the
    first sample where the gradient exceeds 10 dB/s, the end point the first
    later sample where it drops below 10 dB/s, and onsets separated by less
    than 50 ms are merged. Each onset carries its level difference
    ``LD = Le - Ls`` (3.4), its onset rate (3.5) and its prominence ``P``.

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
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if lev.size < 2:
        raise ValueError("at least two level samples are required.")
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
        return "not impulsive"
    if adjustment <= HIGHLY_IMPULSIVE_LIMIT:
        return "regular impulsive"
    return "highly impulsive"


def impulsive_sound_adjustment(
    signal: ArrayLike,
    fs: float,
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

    :param signal: Calibrated sound pressure signal of the candidate event, in
        pascal.
    :param fs: Sampling rate of ``signal``, in Hz.
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
    x = np.asarray(signal, dtype=np.float64).ravel()
    times, levels = sound_pressure_level_history(
        x, fs, dt=dt, reference_pressure=reference_pressure, calibration_offset=calibration_offset
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
        "not impulsive": "not impulsive",
        "regular impulsive": "regular impulsive",
        "highly impulsive": "highly impulsive",
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
        "not impulsive": "no impulsivo",
        "regular impulsive": "impulsivo regular",
        "highly impulsive": "altamente impulsivo",
        "summary": "P = {p}, K_I = {k} dB ({cat})",
        "nosummary": "sin inicio válido: K_I = 0 dB (no impulsivo)",
    },
}


def _plot_impulsive_sound(
    result: ImpulsiveSoundResult,
    *,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    from .._i18n import check_language

    lang = check_language(language)
    labels = _PLOT_LABELS[lang]

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(result.times, result.levels, color="0.3", lw=1.2, label=labels["level"], **kwargs)

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
            p=f"{result.prominence:.2f}", k=f"{result.adjustment:.2f}", cat=labels[result.category]
        )
    else:
        summary = labels["nosummary"]

    ax.set_xlabel(labels["xlabel"])
    ax.set_ylabel(labels["ylabel"])
    ax.set_title(f"{labels['title']}\n{summary}")
    ax.legend(loc="best")
    ax.grid(True, ls=":", alpha=0.5)
    return ax
