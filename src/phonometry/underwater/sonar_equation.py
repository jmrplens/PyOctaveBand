#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""The sonar equation (passive and active), in decibels.

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

Two of those terms have their own model here rather than having to be supplied
from outside: :func:`array_directivity_index` gives ``DI`` from the length of a
line array and the wavelength, which is also its array gain when the noise is
isotropic, and :func:`detection_threshold` gives ``DT`` from the false-alarm
probability alone. Both are Ainslie (2010).

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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import (
    require_axis_rank,
    require_equal_shapes,
    require_positive,
    require_same_shape,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

#: The false-alarm probability at which Equation (11.22) diverges: its inner
#: logarithm is log2(1/(2 p_fa)), which is zero here. Half the empty beams are
#: already being declared detections, so no threshold makes the decision.
_DIVERGENT_FALSE_ALARM_PROBABILITY = 0.5

#: Below this argument, sigma(x)/x of Equation (6.54) is taken from its
#: series. The first omitted term, 2 x^4 / 225, is worth 9e-19 here, three
#: orders below the last bit of a double.
_SIGMA_SERIES_LIMIT = 1e-4

#: Fewest points a range grid may have: two, the least that define a curve
#: segment between two bracketing samples.
_MIN_CURVE_POINTS = 2


def _finite(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        msg = f"'{name}' must be a finite number."
        raise ValueError(msg)
    return scalar


def _finite_array(
    values: NDArray[np.float64] | list[float] | float, name: str
) -> NDArray[np.float64]:
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        msg = f"'{name}' must be finite and non-empty."
        raise ValueError(msg)
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

    def __post_init__(self) -> None:
        """Reject a solution whose three loss-axis quantities disagree.

        The sonar equation is solved one propagation loss at a time: both
        entry points take the loss array and subtract the same term balance
        from every entry of it, so :attr:`snr` and :attr:`signal_excess` are
        that axis written down twice more. Nothing pairs the three afterwards
        except position.

        The mistake is loud in one direction only. :meth:`plot` sorts by
        propagation loss and reads the signal excess through that sort order,
        so a short ``signal_excess`` arrives at numpy as an out-of-range fancy
        index: ``IndexError: index 3 is out of bounds for axis 0 with size
        3``, an axis and a size, naming neither the field nor the result it
        came from. A long one says nothing at all. The sort order holds one
        entry per loss, so the curve is drawn from as many signal-excess
        values as there are losses and the tail is dropped in silence -- and
        the tail is the half that matters, because signal excess falls as
        loss grows and its last values are the negative ones. Five excesses
        from 20 dB down to -20 dB plotted against three losses drew
        ``[20, 10, 0]``: a curve that stops dead on the detection limit and
        never crosses it, under the ``SE = 0`` line and the figure-of-merit
        line drawn across it as usual, in a figure whose whole subject is
        where that crossing falls.

        :attr:`snr` has no reader in the library, which is why it is checked
        here rather than left to one. It is the same excess before the
        detection threshold comes off, read as ``res.snr[i]`` beside
        ``res.propagation_loss[i]``, so an offset one is wrong by an ordinary
        number of decibels with nothing downstream to disagree.

        The three are held to one shape rather than to one length, and no rank
        is pinned. A loss over a grid of depth and range is exactly what
        ``gaussian_beams`` and ``parabolic_equation`` hand back, and feeding it
        here is how a detection footprint is drawn, so refusing a second axis
        would refuse the library's own output. A length check is not enough
        either: ``_finite_array`` widens a scalar and never narrows a grid, so
        a ``(3, 2)`` loss beside a ``(3, 4)`` excess agrees on the only axis
        such a check counts and disagrees about every value.

        What the shape check does not cover is the figure. ``.plot()`` sorts by
        propagation loss and cannot draw a grid: it indexes the ``(3, 2)`` pair
        into a ``(3, 2, 2)`` one and matplotlib refuses with ``x and y can be
        no greater than 2D``. That is a limit of the plot, not a disagreement
        between the fields, and a detection map is read as numbers rather than
        drawn as a curve.

        :raises ValueError: if the three quantities disagree in length, or
            one carries an axis the loss axis does not.
        """
        require_same_shape(
            self,
            "propagation_loss",
            "signal_excess",
            "snr",
            quantity="propagation loss",
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot signal excess versus propagation loss with the detection limit."""
        from .._i18n import check_language
        from .._plot.underwater import plot_sonar_equation

        return plot_sonar_equation(
            self, ax=ax, language=check_language(language), **kwargs
        )


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

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the propagation loss against the figure of merit."""
        from .._i18n import check_language
        from .._plot.underwater import plot_detection_range

        return plot_detection_range(
            self, ax=ax, language=check_language(language), **kwargs
        )


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
        msg = "'max_range' must exceed 1 m."
        raise ValueError(msg)
    if int(n_points) < _MIN_CURVE_POINTS:
        msg = "'n_points' must be at least 2."
        raise ValueError(msg)
    options = {
        "law": law,
        "temperature": temperature,
        "salinity": salinity,
        "depth": depth,
        "ph": ph,
        "model": model,
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
    # Rank before shape: the coercion widens a scalar and never narrows a
    # grid, so two equal two-dimensional inputs agree on their shape and reach
    # the crossing search, where the interpolation of a bracketing pair ends in
    # numpy's "only 0-dimensional arrays can be converted to Python scalars",
    # which names neither the argument nor this function. A curve is 1-D, as
    # the signature says.
    for name, value in (("range_m", r), ("propagation_loss", pl)):
        require_axis_rank(value, "detection_range_from_curve", name, 1)
    require_equal_shapes(
        "detection_range_from_curve",
        {"range_m": r.shape, "propagation_loss": pl.shape},
        "sample",
    )
    if r.size < _MIN_CURVE_POINTS or np.any(np.diff(r) <= 0.0):
        msg = "'range_m' must be strictly increasing with at least two samples."
        raise ValueError(msg)
    key = crossing.strip().lower()
    if key not in ("first", "last"):
        msg = f"'crossing' must be 'first' or 'last', got {crossing!r}."
        raise ValueError(msg)
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


def array_directivity_index(
    array_length_m: float,
    wavelength_m: float,
    *,
    steer_angle_rad: float = 0.0,
) -> float:
    r"""Directivity index of an unshaded line array (Ainslie 2010, Eq. (6.49)).

    :math:`\mathrm{DI} = 10 \log_{10} G_\mathrm{D}` with the directivity factor
    :math:`G_\mathrm{D} = 4\pi / \delta\Omega` the reciprocal of the solid-angle
    footprint of the beam. For a steered unshaded line array, Equation (6.56)
    on printed folio 267 gives that footprint in closed form:

    .. math::

       \delta\Omega = \frac{4}{G_0} \left\{
       \sigma\!\left[\frac{\pi G_0}{2}(1 - \sin\psi)\right] +
       \sigma\!\left[\frac{\pi G_0}{2}(1 + \sin\psi)\right] \right\}

    with :math:`\sigma(x) = \int_0^x \mathrm{d}u \sin^2 u / u^2
    = \mathrm{Si}(2x) - \sin^2 x / x` (Eq. (6.54)) and
    :math:`G_0 = 2L/\lambda` (Eq. (6.57)), the high-frequency limit of the
    broadside directivity factor.

    This is the **array gain** whenever the noise is isotropic and the signal a
    plane wave, which is the case the sonar equation is written for
    (Section 6.1.3.1): the two coincide in that limit, so this is what
    :func:`passive_sonar_equation` wants for its ``directivity_index``.

    The book states three limits, and they are what the implementation is
    checked against: :math:`10 \log_{10}(2L/\lambda)` at high frequency for
    every steer direction but endfire, :math:`10 \log_{10}(4L/\lambda)` near
    endfire, where the footprint halves, and 0 dB as :math:`L/\lambda \to 0`,
    where the array stops resolving anything. That last one is a limit and not
    a cutoff: a finite array a wavelength long still returns 3.45 dB, and half
    a wavelength 1.11 dB. It is reached exactly only where the ratio itself
    underflows, and the value there is 0 dB rather than an error.

    :param array_length_m: Array length ``L``, in metres (> 0).
    :param wavelength_m: Acoustic wavelength ``lambda``, in metres (> 0).
    :param steer_angle_rad: Steer angle ``psi`` from broadside, in radians
        (Default: 0, broadside). Only its sine enters, so the two sides of
        broadside give the same index.
    :return: The directivity index ``DI``, in dB (>= 0).
    :raises ValueError: for a non-positive or non-finite length or wavelength,
        or a non-finite steer angle.
    """
    from scipy.special import sici

    length = require_positive(array_length_m, "array_length_m")
    wavelength = require_positive(wavelength_m, "wavelength_m")
    psi = _finite(steer_angle_rad, "steer_angle_rad")

    def sigma_ratio(x: float) -> float:
        """``sigma(x) / x`` of Equation (6.54), which is one at the origin.

        The ratio rather than ``sigma`` itself, because dividing the footprint
        by ``G0`` afterwards is what breaks: ``G0 = 2L/lambda`` underflows to
        nought for a short enough array, and ``4 / G0`` then raises rather than
        returning the low-frequency limit. Both factors of ``G0`` cancel by
        hand, so it never appears in a denominator.

        Below ``x = 1e-4`` the series :math:`1 - x^2/9 + 2x^4/225` is used. The
        omitted term is worth 9e-19 there, three orders below the last bit, and
        the closed form is the one that suffers: ``Si(2x)`` and ``x sinc(x)^2``
        agree to their leading ``x``, so the subtraction cancels a digit that
        the series never computes. It also puts ``sigma(0)/0`` at its limit
        without comparing a float for equality, which an endfire steer needs:
        it lands on ``x = 0`` exactly.
        """
        if x < _SIGMA_SERIES_LIMIT:
            return 1.0 - x * x / 9.0
        sinc = float(np.sinc(x / math.pi))
        return (float(sici(2.0 * x)[0]) - x * sinc * sinc) / x

    half = math.pi * length / wavelength
    sin_psi = math.sin(psi)
    # delta_Omega / (4 pi) with G0 cancelled: the beam's share of the sphere.
    beam_share = (
        (1.0 - sin_psi) * sigma_ratio(half * (1.0 - sin_psi))
        + (1.0 + sin_psi) * sigma_ratio(half * (1.0 + sin_psi))
    ) / 2.0
    # Plus nought, so that the low-frequency limit reads 0.0 rather than -0.0.
    return float(-10.0 * math.log10(beam_share) + 0.0)


def detection_threshold(false_alarm_probability: float) -> float:
    r"""Detection threshold at 50 % detection probability (Ainslie Eq. (11.22)).

    .. math::

       \mathrm{DT}_{50}(p_\mathrm{fa}) \approx
       10 \log_{10}\left(\log_2 \frac{1}{2 p_\mathrm{fa}}\right) - 0.8 \ \mathrm{dB}

    printed on folio 581. ``DT`` is :math:`10 \log_{10} R_{50}`, the
    signal-to-noise ratio after all processing that a 50 % detection
    probability needs (Eq. (3.31)); this closed form estimates it from the
    false-alarm probability alone.

    The logarithm inside is **base two**, not a square. The book states the
    approximation is accurate to +/- 0.1 dB for :math:`p_\mathrm{fa} < 10^{-2}`
    with one-dominant-plus-Rayleigh signal statistics, which is the
    intermediate choice to make when the target statistics are unknown, and
    that assuming those statistics anyway costs no more than 0.8 dB even for a
    stable signal or a fully Rayleigh one.

    :param false_alarm_probability: ``p_fa``, the probability of declaring a
        detection with no target present, in (0, 1/2).
    :return: The detection threshold ``DT``, in dB.
    :raises ValueError: for a non-finite ``p_fa``, or one outside (0, 1/2).
        At :math:`p_\mathrm{fa} = 1/2` the inner logarithm is zero and the
        threshold diverges: half the empty beams are already called detections.
    """
    p_fa = _finite(false_alarm_probability, "false_alarm_probability")
    if not 0.0 < p_fa < _DIVERGENT_FALSE_ALARM_PROBABILITY:
        msg = (
            "'false_alarm_probability' must lie in (0, 1/2), got "
            f"{p_fa!r}. At 1/2 the threshold diverges, since half the empty "
            "beams are already declared detections."
        )
        raise ValueError(msg)
    # ``-log2(2 p)`` rather than ``log2(1 / (2 p))``: the reciprocal overflows
    # to infinity for the smallest subnormal, where the threshold is a perfectly
    # finite 1073 bits' worth, and the two are the same number everywhere else.
    return float(10.0 * math.log10(-math.log2(2.0 * p_fa)) - 0.8)
