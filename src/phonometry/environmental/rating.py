#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Environmental noise descriptors per ISO 1996-1:2016.

Day-evening-night (Lden, 3.6.4) and day-night (Ldn, 3.6.5) sound levels,
and composite whole-day rating levels (6.5, Formulae 5-6).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def composite_rating_level(periods: Iterable[tuple[float, float, float]]) -> float:
    """
    Composite whole-day rating level (ISO 1996-1:2016, 6.5).

    Generalizes Formulae (5) and (6): each period contributes its rating
    level plus adjustment, weighted by its share of the 24 h day:

    ``10*lg[ sum_i (h_i / 24) * 10^(0.1*(L_i + K_i)) ]``

    :param periods: Iterable of ``(level_db, hours, adjustment_db)`` tuples.
        ``level_db`` is the period's (rating) equivalent continuous level,
        ``hours`` its duration and ``adjustment_db`` the time-of-day or
        source adjustment K (e.g. ISO 1996-1 Table A.1: evening 5 dB,
        night 10 dB). Hours must be positive and sum to 24.
    :return: Composite rating level in dB.
    """
    periods = list(periods)
    if not periods:
        raise ValueError("At least one period is required.")
    hours = np.array([h for _, h, _ in periods], dtype=np.float64)
    if not np.all(np.isfinite(hours)) or np.any(hours <= 0):
        raise ValueError("Every period duration must be a positive, finite number.")
    total = float(np.sum(hours))
    if abs(total - 24.0) > 1e-9:
        raise ValueError(f"Period durations must sum to 24 h; got {total!r}.")
    acc = sum(
        h / 24.0 * 10 ** (0.1 * (level + k)) for level, h, k in periods
    )
    return float(10 * np.log10(acc))


def lden(
    lday: float,
    levening: float,
    lnight: float,
    hours: tuple[float, float, float] = (12.0, 4.0, 8.0),
) -> float:
    r"""
    Day-evening-night sound level Lden (ISO 1996-1:2016, 3.6.4).

    .. math::

       L_{den} = 10 \lg\left\{ (1/24) \left[ t_d \cdot 10^{0.1 L_{day}}
       + t_e \cdot 10^{0.1 (L_{evening}+5)}
       + t_n \cdot 10^{0.1 (L_{night}+10)} \right] \right\}

    :param lday: LAeq over the day period, in dB.
    :param levening: LAeq over the evening period, in dB (+5 dB weighting).
    :param lnight: LAeq over the night period, in dB (+10 dB weighting).
    :param hours: ``(t_day, t_evening, t_night)`` in hours, summing to 24.
        Default (12, 4, 8); countries may define the periods differently
        (3.6.4 Note 1).
    :return: Lden in dB.
    """
    td, te, tn = hours
    return composite_rating_level(
        [(lday, td, 0.0), (levening, te, 5.0), (lnight, tn, 10.0)]
    )


def ldn(
    lday: float,
    lnight: float,
    hours: tuple[float, float] = (15.0, 9.0),
) -> float:
    r"""
    Day-night sound level Ldn (ISO 1996-1:2016, 3.6.5).

    .. math::

       L_{dn} = 10 \lg\left\{ (1/24) \left[ t_d \cdot 10^{0.1 L_{day}}
       + t_n \cdot 10^{0.1 (L_{night}+10)} \right] \right\}

    :param lday: LAeq over the day period, in dB.
    :param lnight: LAeq over the night period, in dB (+10 dB weighting).
    :param hours: ``(t_day, t_night)`` in hours, summing to 24.
        Default (15, 9).
    :return: Ldn in dB.
    """
    td, tn = hours
    return composite_rating_level([(lday, td, 0.0), (lnight, tn, 10.0)])
