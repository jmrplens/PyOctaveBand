#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
The sonar equation (passive and active), in decibels.

Combines the sonar performance terms -- source level ``SL``, propagation loss
``PL``, noise level ``NL``, directivity index ``DI``, detection threshold ``DT``,
target strength ``TS`` and reverberation level ``RL`` -- into the signal excess
``SE``, the signal-to-noise ratio and the figure of merit (the maximum allowable
propagation loss at the detection limit :math:`\mathrm{SE} = 0`):

* :func:`passive_sonar_equation` --
  :math:`\mathrm{SE} = \mathrm{SL} - \mathrm{PL} -
  (\mathrm{NL} - \mathrm{DI}) - \mathrm{DT}`.
* :func:`active_sonar_equation` -- monostatic, noise-limited
  :math:`\mathrm{SE} = \mathrm{SL} - 2\,\mathrm{PL} + \mathrm{TS} -
  (\mathrm{NL} - \mathrm{DI}) - \mathrm{DT}` or, when a reverberation level
  is given, reverberation-limited
  :math:`\mathrm{SE} = \mathrm{SL} - 2\,\mathrm{PL} + \mathrm{TS} -
  \mathrm{RL} - \mathrm{DT}`.

All quantities are in dB (levels re a plane wave of 1 µPa rms; the terms are
spectrum levels, i.e. referred to a 1 Hz band). Source: Urick, *Principles of
Underwater Sound*, via Etter (2003), Table 10.2. The loss term is the
propagation loss :math:`N_\mathrm{PL} = L_\mathrm{S} - L_p(x)` of ISO 18405:2017,
3.4.1.4, which is also the term its own passive and active sonar equations
(3.6.2.7 and 3.6.2.11) are written with.

The figure of merit is the *maximum allowable propagation loss*, so inverting
a propagation-loss law at :math:`\mathrm{PL} = \mathrm{FOM}` gives the
**detection range**, the
range at which the detection probability is 50 %:

* :func:`detection_range` inverts the closed-form loss of
  :mod:`phonometry.underwater.propagation.closed_form` (spreading plus volume absorption),
  which is strictly increasing with range and therefore has a single crossing;
* :func:`detection_range_from_curve` reads the crossing off any computed loss
  curve -- a normal-mode, parabolic-equation or Weston-regime prediction --
  where the oscillatory loss of a real waveguide can cross the figure of merit
  more than once (Ainslie, *Principles of Sonar Performance Modelling*, §11.2.8
  makes exactly that point about convergence zones).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


def _finite(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"'{name}' must be a finite number.")
    return scalar


def _finite_array(values: NDArray[np.float64] | list[float] | float, name: str) -> NDArray[np.float64]:
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"'{name}' must be finite and non-empty.")
    return arr


@dataclass(frozen=True)
class SonarEquationResult:
    r"""Sonar-equation solution.

    :ivar mode: ``"passive"`` or ``"active"``.
    :ivar signal_excess: Signal excess ``SE`` per propagation loss, in dB
        (detection when ``SE >= 0``).
    :ivar snr: Signal-to-noise (or signal-to-reverberation) ratio, in dB
        (:math:`\mathrm{SE} + \mathrm{DT}`).
    :ivar figure_of_merit: Maximum allowable (one-way) propagation loss at
        the detection limit :math:`\mathrm{SE} = 0`, in dB.
    :ivar propagation_loss: The propagation-loss values, in dB.
    :ivar source_level: Source level ``SL``, in dB.
    :ivar noise_level: Background noise level ``NL`` input, in dB. The masking
        term is :math:`\mathrm{NL} - \mathrm{DI}`, except when
        ``reverberation_limited`` is true, where the reverberation level
        ``RL`` masks instead.
    :ivar directivity_index: Receiver directivity index ``DI``, in dB.
    :ivar detection_threshold: Detection threshold ``DT``, in dB.
    :ivar target_strength: Target strength ``TS``, in dB (``None`` for passive).
    :ivar reverberation_limited: Whether the active case is reverberation-limited.
    """

    mode: str
    signal_excess: NDArray[np.float64]
    snr: NDArray[np.float64]
    figure_of_merit: float
    propagation_loss: NDArray[np.float64]
    source_level: float
    noise_level: float
    directivity_index: float
    detection_threshold: float
    target_strength: float | None
    reverberation_limited: bool

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot signal excess versus propagation loss with the detection limit."""
        from .._i18n import check_language
        from .._plot.underwater import plot_sonar_equation

        return plot_sonar_equation(self, ax=ax, language=check_language(language), **kwargs)


def passive_sonar_equation(
    source_level: float,
    propagation_loss: NDArray[np.float64] | list[float] | float,
    noise_level: float,
    *,
    directivity_index: float = 0.0,
    detection_threshold: float = 0.0,
) -> SonarEquationResult:
    r"""Passive sonar equation :math:`\mathrm{SE} = \mathrm{SL} -
    \mathrm{PL} - (\mathrm{NL} - \mathrm{DI}) - \mathrm{DT}`.

    :param source_level: Source level ``SL`` (of the target), in dB.
    :param propagation_loss: One-way propagation loss ``PL``, in dB (scalar or
        array).
    :param noise_level: Background noise level ``NL``, in dB.
    :param directivity_index: Receiver directivity index ``DI``, in dB.
    :param detection_threshold: Detection threshold ``DT``, in dB.
    :return: A :class:`SonarEquationResult`.
    :raises ValueError: If an input is not finite.
    """
    sl = _finite(source_level, "source_level")
    nl = _finite(noise_level, "noise_level")
    di = _finite(directivity_index, "directivity_index")
    dt = _finite(detection_threshold, "detection_threshold")
    pl = _finite_array(propagation_loss, "propagation_loss")
    masking = nl - di
    snr = sl - pl - masking
    signal_excess = snr - dt
    fom = sl - masking - dt
    return SonarEquationResult(
        mode="passive",
        signal_excess=signal_excess,
        snr=snr,
        figure_of_merit=float(fom),
        propagation_loss=pl,
        source_level=sl,
        noise_level=nl,
        directivity_index=di,
        detection_threshold=dt,
        target_strength=None,
        reverberation_limited=False,
    )


def active_sonar_equation(
    source_level: float,
    propagation_loss: NDArray[np.float64] | list[float] | float,
    target_strength: float,
    noise_level: float,
    *,
    directivity_index: float = 0.0,
    detection_threshold: float = 0.0,
    reverberation_level: float | None = None,
) -> SonarEquationResult:
    r"""Monostatic active sonar equation with a two-way propagation loss.

    Noise-limited: :math:`\mathrm{SE} = \mathrm{SL} - 2\,\mathrm{PL} +
    \mathrm{TS} - (\mathrm{NL} - \mathrm{DI}) - \mathrm{DT}`. When
    ``reverberation_level`` is given, reverberation-limited:
    :math:`\mathrm{SE} = \mathrm{SL} - 2\,\mathrm{PL} + \mathrm{TS} -
    \mathrm{RL} - \mathrm{DT}` (``DI`` does not apply to reverberation).

    :param source_level: Source level ``SL``, in dB.
    :param propagation_loss: One-way propagation loss ``PL``, in dB (scalar or
        array); the equation applies :math:`2\,\mathrm{PL}`.
    :param target_strength: Target strength ``TS``, in dB.
    :param noise_level: Background noise level ``NL``, in dB.
    :param directivity_index: Receiver directivity index ``DI``, in dB.
    :param detection_threshold: Detection threshold ``DT``, in dB.
    :param reverberation_level: Reverberation level ``RL`` in dB; when given, the
        case is reverberation-limited.
    :return: A :class:`SonarEquationResult`.
    :raises ValueError: If an input is not finite.
    """
    sl = _finite(source_level, "source_level")
    ts = _finite(target_strength, "target_strength")
    nl = _finite(noise_level, "noise_level")
    di = _finite(directivity_index, "directivity_index")
    dt = _finite(detection_threshold, "detection_threshold")
    pl = _finite_array(propagation_loss, "propagation_loss")
    if reverberation_level is not None:
        masking = _finite(reverberation_level, "reverberation_level")
        reverb = True
    else:
        masking = nl - di
        reverb = False
    snr = sl - 2.0 * pl + ts - masking
    signal_excess = snr - dt
    fom = 0.5 * (sl + ts - masking - dt)
    return SonarEquationResult(
        mode="active",
        signal_excess=signal_excess,
        snr=snr,
        figure_of_merit=float(fom),
        propagation_loss=pl,
        source_level=sl,
        noise_level=nl,
        directivity_index=di,
        detection_threshold=dt,
        target_strength=ts,
        reverberation_limited=reverb,
    )


@dataclass(frozen=True)
class DetectionRangeResult:
    r"""Detection range obtained by inverting a propagation-loss law.

    :ivar detection_range: Range at which ``PL`` equals the figure of merit, in
        metres. ``inf`` when the loss never reaches it inside ``max_range``
        (detectable throughout) and ``0.0`` when it already exceeds it at the
        search floor (detectable nowhere).
    :ivar figure_of_merit: The figure of merit inverted, in dB.
    :ivar frequency: Acoustic frequency, in Hz.
    :ivar range_m: Range grid over which the loss was evaluated, in metres.
    :ivar propagation_loss: Propagation loss at each range, in dB.
    :ivar absorption_coefficient: Absorption coefficient :math:`\alpha`, in
        dB/km.
    :ivar law: The spreading law used.
    :ivar model: The absorption model used.
    """

    detection_range: float
    figure_of_merit: float
    frequency: float
    range_m: NDArray[np.float64]
    propagation_loss: NDArray[np.float64]
    absorption_coefficient: float
    law: str
    model: str

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the propagation loss against the figure of merit."""
        from .._i18n import check_language
        from .._plot.underwater import plot_detection_range

        return plot_detection_range(self, ax=ax, language=check_language(language), **kwargs)


def detection_range(
    figure_of_merit: float,
    frequency_hz: float,
    *,
    law: str = "spherical",
    transition_range: float | None = None,
    temperature: float = 10.0,
    salinity: float = 35.0,
    depth: float = 0.0,
    ph: float = 8.0,
    model: str = "francois-garrison",
    max_range: float = 500_000.0,
    n_points: int = 400,
) -> DetectionRangeResult:
    r"""Range at which the closed-form propagation loss equals the figure of
    merit.

    Solves :math:`\mathrm{PL}(r) = \mathrm{FOM}` for the loss of
    :func:`~phonometry.underwater.propagation.propagation_loss`, which is
    strictly increasing in range, so the root is unique. A **one-way** figure of
    merit works for both sonar modes: the active figure of merit returned by
    :func:`active_sonar_equation` is already the maximum allowable one-way loss.

    :param figure_of_merit: Maximum allowable one-way propagation loss, in dB.
    :param frequency_hz: Acoustic frequency, in Hz.
    :param law: Spreading law (see
        :func:`~phonometry.underwater.propagation.spreading_loss`).
    :param transition_range: Transition range for the ``"practical"`` law, in m.
    :param temperature: Temperature ``T``, in degrees Celsius.
    :param salinity: Salinity ``S``, in parts per thousand.
    :param depth: Depth, in metres.
    :param ph: Acidity (default 8).
    :param model: Absorption model (see
        :func:`~phonometry.underwater.propagation.seawater_absorption`).
    :param max_range: Upper bound of the search, in metres.
    :param n_points: Number of ranges kept on the returned loss curve.
    :return: A :class:`DetectionRangeResult`.
    :raises ValueError: If an input is invalid.
    """
    from scipy.optimize import brentq

    from .propagation.closed_form import propagation_loss

    fom = _finite(figure_of_merit, "figure_of_merit")
    rmax = _finite(max_range, "max_range")
    if rmax <= 1.0:
        raise ValueError("'max_range' must exceed 1 m.")
    if int(n_points) < 2:
        raise ValueError("'n_points' must be at least 2.")
    options = {
        "law": law, "temperature": temperature, "salinity": salinity,
        "depth": depth, "ph": ph, "model": model,
        "transition_range": transition_range,
    }

    def _loss(r: float) -> float:
        return float(propagation_loss(r, frequency_hz, **options).pl[0])  # type: ignore[arg-type]

    lo = 1e-3
    if _loss(rmax) < fom:
        root = float("inf")
    elif _loss(lo) > fom:
        root = 0.0  # the loss already exceeds the figure of merit at 1 mm
    else:
        root = float(brentq(lambda r: _loss(r) - fom, lo, rmax, xtol=1e-6, rtol=1e-12))
    upper = rmax if not np.isfinite(root) else min(rmax, max(2.0 * root, 10.0))
    grid = np.linspace(max(1.0, upper / 1000.0), upper, int(n_points))
    curve = propagation_loss(grid, frequency_hz, **options)  # type: ignore[arg-type]
    return DetectionRangeResult(
        detection_range=root,
        figure_of_merit=fom,
        frequency=curve.frequency,
        range_m=curve.range_m,
        propagation_loss=curve.pl,
        absorption_coefficient=curve.absorption_coefficient,
        law=curve.law,
        model=curve.model,
    )


def detection_range_from_curve(
    figure_of_merit: float,
    range_m: NDArray[np.float64] | list[float],
    propagation_loss: NDArray[np.float64] | list[float],
    *,
    crossing: str = "first",
) -> float:
    """Detection range read off a computed propagation-loss curve.

    Finds where ``PL(r)`` crosses the figure of merit from below, interpolating
    linearly between the two bracketing samples. Real waveguides oscillate, so
    ``crossing`` selects which crossing to report.

    :param figure_of_merit: Maximum allowable propagation loss, in dB.
    :param range_m: Ranges, in metres (1-D, strictly increasing).
    :param propagation_loss: Loss at each range, in dB (same length).
    :param crossing: ``"first"`` (default) or ``"last"`` upward crossing.
    :return: The detection range, in metres. Two limiting cases carry no
        crossing and are distinguished by the loss at the **last** sample:
        ``inf`` when the loss is still below the figure of merit there (the
        target stays detectable past the end of the grid) and ``0.0`` when the
        loss exceeds it there, which without an upward crossing means it
        exceeded it at every sample and the target is detectable nowhere.
        :func:`detection_range` returns the same two values for the same two
        situations.
    :raises ValueError: If the inputs are invalid.
    """
    fom = _finite(figure_of_merit, "figure_of_merit")
    r = _finite_array(range_m, "range_m")
    pl = _finite_array(propagation_loss, "propagation_loss")
    if r.shape != pl.shape:
        raise ValueError("'propagation_loss' must have the same length as 'range_m'.")
    if r.size < 2 or np.any(np.diff(r) <= 0.0):
        raise ValueError("'range_m' must be strictly increasing with at least two samples.")
    key = crossing.strip().lower()
    if key not in ("first", "last"):
        raise ValueError(f"'crossing' must be 'first' or 'last', got {crossing!r}.")
    below = pl <= fom
    up = np.flatnonzero(below[:-1] & ~below[1:])
    if up.size == 0:
        # No upward crossing. Either the loss is below the figure of merit at
        # the far end -- detectable past the grid, so the range is unbounded --
        # or it is above it there, and since reaching "above" from "below"
        # would have produced an upward crossing it was above at every sample:
        # the target is detectable nowhere and the range is zero, which is what
        # ``detection_range`` returns for the same situation.
        return float("inf") if bool(below[-1]) else 0.0
    i = int(up[0] if key == "first" else up[-1])
    # At an upward crossing pl[i] <= fom < pl[i+1], so the span is strictly
    # positive and the linear interpolation cannot divide by zero.
    frac = (fom - pl[i]) / (pl[i + 1] - pl[i])
    return float(r[i] + frac * (r[i + 1] - r[i]))
