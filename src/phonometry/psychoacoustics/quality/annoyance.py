#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Psychoacoustic annoyance (PA) after Fastl & Zwicker.

Psychoacoustic annoyance combines four hearing sensations -- loudness,
sharpness, fluctuation strength and roughness -- into a single figure that
tracks annoyance ratings from listening experiments. The model is due to
Widmann (1992) and is given in Fastl & Zwicker, *Psychoacoustics: Facts and
Models* (Equations 16.2-16.4):

.. math::

   PA = N_5 \left( 1 + \sqrt{w_S^2 + w_{FR}^2} \right) \tag{Eq. 16.2}

with the percentile loudness ``N5`` in sone and the two loudness-weighted
terms

.. math::

   w_S = (S - 1.75) \cdot 0.25 \cdot \log_{10}(N_5 + 10)
   \quad \text{for } S > 1.75~\text{acum, else } 0 \tag{Eq. 16.3}

   w_{FR} = \frac{2.18}{N_5^{0.4}} \left( 0.4 F + 0.6 R \right)
   \tag{Eq. 16.4}

describing sharpness ``S`` (acum) and the joint influence of fluctuation
strength ``F`` (vacil) and roughness ``R`` (asper). Note the "1 +" sits
*outside* the radical (Fastl & Zwicker 2006, Eq. (16.2), p. 328). There is
no ISO standard for PA; the formula is exact, verified against a
hand-computed worked tuple, and matches the combination implemented by the
open SQAT reference implementation.

:func:`psychoacoustic_annoyance` evaluates the model from the four quantities
directly. :func:`psychoacoustic_annoyance_from_signal` is a convenience that
derives them from a calibrated pressure signal using the library's existing
models -- N5/S from ISO 532-1 Zwicker loudness and DIN 45692 sharpness, R from
ECMA-418-2 roughness and F from :func:`~phonometry.psychoacoustics.quality.fluctuation_strength.
fluctuation_strength`. That composite mixes model families (Zwicker N5/S,
Sottek R, Osses F); the original PA model was calibrated with Zwicker-family
sensations, so the signal convenience is an engineering estimate, while
:func:`psychoacoustic_annoyance` is the exact model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: Sharpness threshold (acum) below which the sharpness term wS vanishes (16.3).
_SHARPNESS_THRESHOLD = 1.75
#: Scale of the sharpness term wS (Equation 16.3).
_SHARPNESS_SCALE = 0.25
#: Loudness offset inside the lg() of the sharpness term wS (Equation 16.3).
_SHARPNESS_LOUDNESS_OFFSET = 10.0
#: Scale of the fluctuation/roughness term wFR (Equation 16.4).
_WFR_SCALE = 2.18
#: Loudness exponent of the fluctuation/roughness term wFR (Equation 16.4).
_WFR_LOUDNESS_EXPONENT = 0.4
#: Weight of fluctuation strength in the wFR term (Equation 16.4).
_WFR_FLUCTUATION_WEIGHT = 0.4
#: Weight of roughness in the wFR term (Equation 16.4).
_WFR_ROUGHNESS_WEIGHT = 0.6


def _nonnegative(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"'{name}' must be a non-negative, finite number.")
    return scalar


@dataclass(frozen=True)
class PsychoacousticAnnoyanceResult:
    """Psychoacoustic annoyance and its contributing terms (Fastl & Zwicker).

    :ivar annoyance: Psychoacoustic annoyance ``PA`` (Equation 16.2),
        dimensionless.
    :ivar n5: Percentile loudness ``N5`` used, in sone.
    :ivar sharpness: Sharpness ``S`` used, in acum.
    :ivar fluctuation_strength: Fluctuation strength ``F`` used, in vacil.
    :ivar roughness: Roughness ``R`` used, in asper.
    :ivar w_s: Sharpness term ``wS`` (Equation 16.3).
    :ivar w_fr: Fluctuation/roughness term ``wFR`` (Equation 16.4).
    """

    annoyance: float
    n5: float
    sharpness: float
    fluctuation_strength: float
    roughness: float
    w_s: float
    w_fr: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the PA value and the ``wS`` / ``wFR`` term contributions."""
        from ..._i18n import check_language
        from ..._plot.psychoacoustics import plot_psychoacoustic_annoyance

        return plot_psychoacoustic_annoyance(self, ax=ax, language=check_language(language), **kwargs)


def psychoacoustic_annoyance(
    n5: float,
    sharpness: float,
    fluctuation_strength: float,
    roughness: float,
) -> PsychoacousticAnnoyanceResult:
    r"""Psychoacoustic annoyance from the four hearing sensations (16.2-16.4).

    :math:`PA = N_5 (1 + \sqrt{w_S^2 + w_{FR}^2})` with the loudness-weighted
    sharpness term ``wS`` (Equation 16.3) and the fluctuation/roughness term
    ``wFR`` (Equation 16.4). The sharpness term is zero for
    :math:`S \le 1.75` acum.

    :param n5: Percentile loudness ``N5``, in sone (the loudness exceeded 5 %
        of the time; :attr:`~phonometry.ZwickerLoudness.n5`).
    :param sharpness: Sharpness ``S``, in acum (DIN 45692).
    :param fluctuation_strength: Fluctuation strength ``F``, in vacil.
    :param roughness: Roughness ``R``, in asper (ECMA-418-2).
    :return: A :class:`PsychoacousticAnnoyanceResult`.
    :raises ValueError: If any quantity is negative or non-finite.
    """
    n5_v = _nonnegative(n5, "n5")
    s_v = _nonnegative(sharpness, "sharpness")
    f_v = _nonnegative(fluctuation_strength, "fluctuation_strength")
    r_v = _nonnegative(roughness, "roughness")

    if s_v > _SHARPNESS_THRESHOLD:
        w_s = (
            (s_v - _SHARPNESS_THRESHOLD)
            * _SHARPNESS_SCALE
            * np.log10(n5_v + _SHARPNESS_LOUDNESS_OFFSET)
        )
    else:
        w_s = 0.0

    if n5_v > 0.0:
        w_fr = (
            _WFR_SCALE
            / n5_v**_WFR_LOUDNESS_EXPONENT
            * (_WFR_FLUCTUATION_WEIGHT * f_v + _WFR_ROUGHNESS_WEIGHT * r_v)
        )
    else:
        w_fr = 0.0

    annoyance = n5_v * (1.0 + np.sqrt(w_s**2 + w_fr**2))  # Eq. (16.2)
    return PsychoacousticAnnoyanceResult(
        annoyance=float(annoyance),
        n5=n5_v,
        sharpness=s_v,
        fluctuation_strength=f_v,
        roughness=r_v,
        w_s=float(w_s),
        w_fr=float(w_fr),
    )


def psychoacoustic_annoyance_from_signal(
    x: list[float] | np.ndarray,
    fs: int,
    *,
    field: Literal["free", "diffuse"] = "free",
    calibration_factor: float = 1.0,
) -> PsychoacousticAnnoyanceResult:
    """Psychoacoustic annoyance from a calibrated pressure signal (convenience).

    Derives the four sensations from the library's models and combines them
    with :func:`psychoacoustic_annoyance`: ``N5`` from the ISO 532-1 Zwicker
    time-varying loudness, ``S`` from DIN 45692 sharpness, ``R`` from
    ECMA-418-2 roughness and ``F`` from
    :func:`~phonometry.psychoacoustics.quality.fluctuation_strength.fluctuation_strength`.

    .. note::
        This composite mixes model families (Zwicker ``N5``/``S``, Sottek
        ``R``, Osses ``F``); the PA model was calibrated with Zwicker-family
        sensations, so treat the signal convenience as an engineering estimate.
        For exact, reproducible results pass the four quantities to
        :func:`psychoacoustic_annoyance` directly.

        ``field`` selects the sound field for the loudness, sharpness and
        roughness front-ends; the fluctuation strength ``F`` is always computed
        in the free field, because the Osses 2016 model has no diffuse-field
        variant (see :func:`~phonometry.psychoacoustics.quality.fluctuation_strength.fluctuation_strength`).

    :param x: Calibrated sound-pressure signal (1-D), in Pa after
        ``calibration_factor``.
    :param fs: Sample rate, in Hz.
    :param field: ``'free'`` (default) or ``'diffuse'`` sound field for the
        loudness/sharpness/roughness front-ends (``F`` is always free-field).
    :param calibration_factor: Digital-units-to-Pa factor applied to ``x``.
    :return: A :class:`PsychoacousticAnnoyanceResult`.
    """
    from ..loudness.zwicker import loudness_zwicker
    from .fluctuation_strength import fluctuation_strength as _fluctuation_strength
    from .roughness_ecma import roughness_ecma
    from .sharpness import sharpness_din_from_specific

    signal = np.asarray(x, dtype=np.float64) * float(calibration_factor)
    loud = loudness_zwicker(signal, fs, field=field, stationary=False)
    # N5: the 5 % percentile loudness of the time-varying trace; fall back to
    # the total loudness for signals too short to yield a percentile.
    n5 = loud.n5 if loud.n5 is not None else loud.loudness
    s = sharpness_din_from_specific(loud.specific)
    r = roughness_ecma(signal, float(fs), field=field).roughness
    # The Osses 2016 fluctuation-strength model is free-field only.
    f = _fluctuation_strength(signal, float(fs)).fluctuation_strength
    return psychoacoustic_annoyance(n5, s, f, r)
