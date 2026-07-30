#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Open-plan-office spatial metrics per ISO 3382-3:2012.

From a line of measurement positions across workstations (ISO 3382-3:2012,
5.2.2) this module derives the single-number quantities of Clause 4:

- the spatial decay rate of the A-weighted sound pressure level of speech
  ``D2,S`` and the nominal A-weighted speech level at 4 m ``Lp,A,S,4m``
  (Clause 6.2), obtained by a least-squares fit of the A-weighted speech
  level against ``lg(r/r0)`` (logarithmic distance axis, ``r0 = 1 m``)
  using only positions in the 2 m to 16 m range (Equation (5)); and
- the distraction distance ``rD`` (STI = 0.50) and privacy distance
  ``rP`` (STI = 0.20) (Clause 3.6, 3.7, 6.3), obtained from a linear
  regression of the speech transmission index against distance on a
  linear axis.

The A-weighted speech levels ``Lp,A,S,n`` (Clause 6.2, Equation (4)) and
the per-position STI (full IEC 60268-16 method, spatially averaged
background noise per Clause 6.3) are taken as inputs; this module performs
only the regressions and threshold read-offs of Clause 6.

The distraction and privacy distances are read from the fitted STI line,
extrapolating beyond the measured range when necessary (the regression-line
method of Clause 6.3, Figure 3 b). A distance is reported as ``nan`` when
the STI does not decrease with distance (non-negative fitted slope) or when
the crossing would fall at or before the source, realising the standard's
note that it "can prove impossible to determine the privacy distance if
STI > 0.20 in all positions" (Clause 6.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

#: Reference distance for the logarithmic distance axis of D2,S
#: (ISO 3382-3:2012, 6.2, Equation (5)): r0 = 1 m.
_R0 = 1.0

#: Distance range (m) of the positions used for D2,S and Lp,A,S,4m
#: (ISO 3382-3:2012, 5.2.2 and 6.2): only 2 m to 16 m.
_DECAY_RANGE_M = (2.0, 16.0)

#: Distance (m) at which the nominal A-weighted speech level is reported,
#: Lp,A,S,4m (ISO 3382-3:2012, 3.3, 6.2).
_LP_DISTANCE_M = 4.0

#: STI thresholds for the distraction and privacy distances
#: (ISO 3382-3:2012, 3.6, 3.7).
_STI_DISTRACTION = 0.50
_STI_PRIVACY = 0.20

#: Minimum number of measurement positions (ISO 3382-3:2012, 5.2.2).
_MIN_POSITIONS = 4


@dataclass(frozen=True)
class OpenPlanResult:
    """Single-number open-plan-office quantities (ISO 3382-3:2012, Cl. 4).

    :ivar d2s: Spatial decay rate of the A-weighted SPL of speech in dB
        (per distance doubling), from the least-squares fit of Clause 6.2,
        Equation (5). ``nan`` when fewer than two positions lie in the
        2 m to 16 m range.
    :ivar lp_as_4m: Nominal A-weighted speech level at 4 m in dB, read off
        the same regression line (Clause 3.3, 6.2). ``nan`` under the same
        condition as ``d2s``.
    :ivar rd: Distraction distance in m, where the linear STI-vs-distance
        regression crosses 0.50 (Clause 3.6, 6.3). ``nan`` when the fitted
        STI does not decrease with distance or the crossing is non-positive.
    :ivar rp: Privacy distance in m, where the same regression crosses 0.20
        (Clause 3.7, 6.3), possibly extrapolated beyond the measured range.
        ``nan`` under the same condition as ``rd``.
    :ivar positions_m: The microphone distances the metrics were fitted
        on, in metres, retained so :meth:`plot_geometry` can draw the
        line; ``None`` for hand-built results.
    """

    d2s: float
    lp_as_4m: float
    rd: float
    rp: float
    positions_m: np.ndarray | None = None

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the spatial decay of speech with ``rD``/``rP`` marked.

        Redraws the Clause 6.2 regression line from ``d2s`` and
        ``lp_as_4m`` and marks the distraction and privacy distances.
        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.room import plot_open_plan

        check_language(language)
        return plot_open_plan(self, ax=ax, language=language, **kwargs)

    def plot_geometry(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw the measurement line to scale, rD and rP marked.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :raises ValueError: If the result does not retain its geometry.
        """
        from .._i18n import check_language
        from .._plot.geometry import plot_open_plan_result_geometry

        check_language(language)
        return plot_open_plan_result_geometry(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an open-plan office acoustics fiche to a PDF (ISO 3382-3).

        Writes a one-page report laid out like an open-plan-office
        speech-privacy measurement report: the standard-basis line, an optional
        metadata header block (client, office/zone, description, floor area,
        source and measurement positions, climate ...), a compact metrics table
        of the four single-number quantities of Clause 4 (the spatial decay
        rate ``D2,S``, the nominal 4 m A-weighted speech level ``Lp,A,S,4m``,
        the distraction distance ``rD`` and the privacy distance ``rP``), the
        result's own spatial-decay plot (:meth:`plot`, the Clause 6.2
        regression on the logarithmic distance axis with the 4 m read-off and
        the ``rD`` / ``rP`` crossings), the boxed ``D2,S``, an optional verdict
        row and a footer with the fixed disclaimer.

        ISO 3382-3 characterises a space rather than defining an intrinsic
        pass/fail, so the verdict row appears only when a target spatial decay
        rate is supplied through ``metadata.requirement`` (read as the minimum
        acceptable ``D2,S`` in dB, reflecting the informative quality ranges of
        Annex A where a larger spatial decay is better; the room passes at or
        above it).

        :param path: Destination path of the PDF file.
        :param metadata: Optional
            :class:`~phonometry.ReportMetadata`; ``None`` produces a bare
            characterisation fiche (metrics, plot and result only). The
            open-plan-specific fields ``area`` (floor area), ``source_positions``
            and ``receiver_positions`` (the number of measurement positions)
            populate the header, alongside ``client``, ``test_room`` (the
            office or zone), ``specimen`` (the description and furnishing
            state), ``instrumentation``, ``temperature``, ``relative_humidity``,
            ``pressure``, ``measurement_standard``, ``test_date``,
            ``laboratory``, ``operator``, ``report_id`` and ``notes``;
            ``requirement`` is read as the minimum acceptable ``D2,S``.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for parity with the other fiches; the fiche
            has a single stacked body layout, so it has no effect.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab or, for the embedded spatial-decay
            chart, matplotlib is not installed. The fiche embeds the result's
            own plot whenever the regression is defined, so both are required
            (``pip install "phonometry[report,plot]"``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.iso3382_3 import render_iso3382_3_report

        return render_iso3382_3_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ordinary least-squares fit ``y = slope*x + intercept``.

    :param x: Independent variable samples.
    :param y: Dependent variable samples (same length as ``x``).
    :return: ``(slope, intercept)``.
    """
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _sti_crossing(
    slope: float, intercept: float, threshold: float
) -> float:
    """Distance where the STI regression line reaches ``threshold``.

    Solves ``slope*r + intercept = threshold`` (ISO 3382-3:2012, 6.3).
    Returns ``nan`` when the STI does not decrease with distance
    (``slope >= 0``) or the crossing is at or before the source
    (``r <= 0``); otherwise the (possibly extrapolated) crossing distance.
    """
    if slope >= 0.0:
        return float("nan")
    distance = (threshold - intercept) / slope
    if distance <= 0.0:
        return float("nan")
    return distance


def open_plan_metrics(
    positions_m: list[float] | np.ndarray,
    spl_a_speech: list[float] | np.ndarray,
    sti_values: list[float] | np.ndarray,
) -> OpenPlanResult:
    """
    Open-plan-office single-number quantities per ISO 3382-3:2012.

    Computes the spatial decay rate ``D2,S`` and the nominal A-weighted
    speech level at 4 m ``Lp,A,S,4m`` (Clause 6.2, Equation (5)) from a
    least-squares fit of the A-weighted speech level against ``lg(r/r0)``
    (``r0 = 1 m``) restricted to positions in the 2 m to 16 m range, and
    the distraction distance ``rD`` (STI = 0.50) and privacy distance
    ``rP`` (STI = 0.20) from a linear regression of STI against distance
    (Clause 3.6, 3.7, 6.3).

    :param positions_m: Distances ``r`` from the source to each
        measurement position, in metres (ISO 3382-3:2012, 5.2.3 d).
    :param spl_a_speech: A-weighted speech level ``Lp,A,S,n`` at each
        position, in dB (Clause 6.2, Equation (4)).
    :param sti_values: Speech transmission index at each position (full
        IEC 60268-16 method with spatially averaged background noise,
        Clause 6.3).
    :return: :class:`OpenPlanResult` with ``d2s``, ``lp_as_4m``, ``rd``
        and ``rp``.
    :raises ValueError: If fewer than four positions are given
        (Clause 5.2.2) or the three arrays differ in length.
    """
    r = np.asarray(positions_m, dtype=np.float64)
    lp = np.asarray(spl_a_speech, dtype=np.float64)
    sti = np.asarray(sti_values, dtype=np.float64)

    if not (r.shape == lp.shape == sti.shape):
        raise ValueError(
            "positions_m, spl_a_speech and sti_values must have the same "
            "length."
        )
    if r.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")
    if r.size < _MIN_POSITIONS:
        raise ValueError(
            "ISO 3382-3 requires at least 4 measurement positions "
            f"(got {r.size})."
        )
    if not (
        np.all(np.isfinite(r))
        and np.all(np.isfinite(lp))
        and np.all(np.isfinite(sti))
    ):
        raise ValueError(
            "positions_m, spl_a_speech and sti_values must be finite "
            "(no NaN or Inf)."
        )
    if np.any(r <= 0.0):
        raise ValueError("Distances 'positions_m' must be positive.")

    d2s = float("nan")
    lp_as_4m = float("nan")
    in_range = (r >= _DECAY_RANGE_M[0]) & (r <= _DECAY_RANGE_M[1])
    if int(in_range.sum()) >= 2:
        x = np.log10(r[in_range] / _R0)
        slope, intercept = _linear_fit(x, lp[in_range])
        # D2,S is the decay per distance doubling: slope is dB per unit of
        # lg(r), so scale by lg(2) and negate (positive when level falls).
        d2s = -np.log10(2.0) * slope
        lp_as_4m = intercept + slope * np.log10(_LP_DISTANCE_M / _R0)

    sti_slope, sti_intercept = _linear_fit(r, sti)
    rd = _sti_crossing(sti_slope, sti_intercept, _STI_DISTRACTION)
    rp = _sti_crossing(sti_slope, sti_intercept, _STI_PRIVACY)

    return OpenPlanResult(
        d2s=d2s, lp_as_4m=lp_as_4m, rd=rd, rp=rp,
        positions_m=np.asarray(positions_m, dtype=np.float64),
    )

