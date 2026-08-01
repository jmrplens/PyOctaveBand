#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Ship radiated noise and equivalent monopole source level (ISO 17208-1/-2).

A surface ship measured in deep water is characterised by its **radiated noise
level** and then by an **equivalent monopole source level** referred to a point
source below the sea surface:

* :func:`radiated_noise_level` --
  :math:`L_{\mathrm{RN}} = 20 \lg(p_{\mathrm{rms}}/p_0) + 20 \lg(r/r_0)`
  dB re 1 µPa·m (ISO 17208-1), the level of the product of the far-field RMS
  pressure and the source distance.
* :func:`monopole_source_level` -- converts ``LRN`` to the source level
  :math:`L_s = L_{\mathrm{RN}} + \Delta L` with the Lloyd's-mirror surface
  correction :math:`\Delta L` of ISO 17208-2 Formula 3, for a nominal source
  depth :math:`d_s = 0.7 D` (Formula 1).

Supporting helpers give the ISO 17208-1 three-hydrophone measurement depths
(:func:`hydrophone_depths`) and the ISO 17208-2 tabulated source-level
uncertainty (:func:`source_level_uncertainty`). The conversion assumes an ideal
pressure-release sea surface and ignores wind; the reported source level is an
*equivalent monopole broadside* value and must be quoted with its source depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .acoustics import UNDERWATER_REFERENCE_PRESSURE, _positive

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

#: Default speed of sound in sea water (m/s).
_DEFAULT_SOUND_SPEED = 1500.0
#: Nominal source depth as a fraction of the ship draught (ISO 17208-2 Formula 1).
_SOURCE_DEPTH_FRACTION = 0.7
#: Reference distance ``r₀`` for radiated noise level (m).
_REFERENCE_DISTANCE = 1.0
#: ISO 17208-1 standard depression angles (degrees) for the three hydrophones.
_STANDARD_ANGLES = (15.0, 30.0, 45.0)


def radiated_noise_level(rms_pressure: float, distance: float) -> float:
    r"""Radiated noise level ``LRN`` (ISO 17208-1), dB re 1 µPa·m.

    :math:`L_{\mathrm{RN}} = 20 \lg(p_{\mathrm{rms}}/p_0) + 20 \lg(r/r_0)` --
    the level of the product of the far-field RMS sound pressure and the
    source distance, referred to 1 µPa·m.

    :param rms_pressure: Far-field RMS sound pressure ``p_rms``, in Pa.
    :param distance: Distance ``r`` from the ship reference point, in m.
    :return: Radiated noise level, in dB re 1 µPa·m.
    :raises ValueError: If the pressure or distance is not positive.
    """
    p = _positive(rms_pressure, "rms_pressure")
    r = _positive(distance, "distance")
    return float(
        20.0 * np.log10(p / UNDERWATER_REFERENCE_PRESSURE)
        + 20.0 * np.log10(r / _REFERENCE_DISTANCE)
    )


def hydrophone_depths(
    cpa_distance: float, angles: tuple[float, ...] = _STANDARD_ANGLES
) -> NDArray[np.float64]:
    r"""Hydrophone depths for the ISO 17208-1 deep-water geometry.

    At the closest point of approach the three hydrophones sit at depression
    angles from the sea surface seen from the ship reference point; at a
    horizontal range equal to ``cpa_distance`` the depth of each is
    :math:`d = d_{\mathrm{CPA}} \tan(\mathrm{angle})`.

    :param cpa_distance: Horizontal distance at the closest point of approach,
        in m (:math:`d_{\mathrm{CPA}} = \max(100~\text{m},
        \text{ship length})`).
    :param angles: Depression angles, in degrees (default 15°, 30°, 45°).
    :return: The hydrophone depths, in m.
    :raises ValueError: If the distance or any angle is out of range.
    """
    cpa = _positive(cpa_distance, "cpa_distance")
    ang = np.asarray(angles, dtype=np.float64)
    if (
        ang.size == 0
        or not np.all(np.isfinite(ang))
        or np.any(ang <= 0.0)
        or np.any(ang >= 90.0)
    ):
        raise ValueError(
            "'angles' must be finite and lie in the open interval (0, 90) degrees."
        )
    return np.asarray(cpa * np.tan(np.radians(ang)), dtype=np.float64)


def source_level_uncertainty(frequency: float) -> float:
    """Tabulated expanded source-level uncertainty (ISO 17208-2 §5), in dB.

    5 dB for the low-frequency bands (the standard lists 10 Hz-100 Hz; values
    below 10 Hz reuse it), 3 dB for the mid-frequency bands (125 Hz-16 kHz)
    and 4 dB for the high-frequency bands. Part 2 prints its high band as
    "> 20 000 Hz", which leaves the 20 kHz band itself unassigned; Part 1,
    from which the values are taken, prints ">= 20 000 Hz". This
    implementation follows Part 1, so the 4 dB high-band value applies from
    the 20 kHz band upwards (the cut is written at 16 kHz because no
    one-third-octave band lies between the two). These are representative
    values, not exact.

    :param frequency: One-third-octave band centre frequency, in Hz.
    :return: The representative expanded uncertainty, in dB.
    :raises ValueError: If the frequency is not positive.
    """
    f = _positive(frequency, "frequency")
    if f <= 100.0:
        return 5.0
    if f <= 16000.0:
        return 3.0
    return 4.0


def _surface_correction(
    frequency: NDArray[np.float64], source_depth: float, sound_speed: float
) -> NDArray[np.float64]:
    """Lloyd's-mirror RNL-to-source-level correction ΔL (ISO 17208-2 Formula 3)."""
    k = 2.0 * np.pi * frequency / sound_speed
    u = k * source_depth
    u2 = u**2
    u4 = u2**2
    ratio = (2.0 * u4 + 14.0 * u2) / (14.0 + 2.0 * u2 + u4)
    return np.asarray(-10.0 * np.log10(ratio), dtype=np.float64)


@dataclass(frozen=True)
class ShipSourceLevelResult:
    r"""Equivalent monopole source level of a ship (ISO 17208-2).

    :ivar frequencies: Frequencies, in Hz.
    :ivar radiated_noise_level: Input RNL per frequency, in dB re 1 µPa·m.
    :ivar surface_correction: Lloyd's-mirror correction :math:`\Delta L` per
        frequency, dB.
    :ivar source_level: Equivalent monopole source level
        :math:`L_s = L_{\mathrm{RN}} + \Delta L`, in dB re 1 µPa·m.
    :ivar source_depth: Nominal source depth :math:`d_s = 0.7 D`, in m.
    :ivar sound_speed: Speed of sound used, in m/s.
    """

    frequencies: NDArray[np.float64]
    radiated_noise_level: NDArray[np.float64]
    surface_correction: NDArray[np.float64]
    source_level: NDArray[np.float64]
    source_depth: float
    sound_speed: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot RNL, source level and the ΔL surface correction vs frequency."""
        from .._i18n import check_language
        from .._plot.underwater import plot_ship_source_level

        return plot_ship_source_level(self, ax=ax, language=check_language(language), **kwargs)


def monopole_source_level(
    rnl: float | NDArray[np.float64] | list[float],
    frequency: float | NDArray[np.float64] | list[float],
    draught: float,
    *,
    c: float = _DEFAULT_SOUND_SPEED,
) -> ShipSourceLevelResult:
    r"""Equivalent monopole source level from radiated noise level
    (ISO 17208-2).

    :math:`L_s = L_{\mathrm{RN}} + \Delta L` with the surface correction

    .. math::

       \Delta L = -10 \lg \frac{2 u^4 + 14 u^2}{14 + 2 u^2 + u^4}
       \tag{Formula 3}

    where :math:`u = k d_s`, :math:`k = 2 \pi f/c` and the nominal source
    depth is :math:`d_s = 0.7 D` (Formula 1).

    :param rnl: Radiated noise level per frequency, in dB re 1 µPa·m (scalar or
        array; array length must match ``frequency``).
    :param frequency: Frequency or frequencies, in Hz.
    :param draught: Ship draught ``D`` (mean of bow and stern), in m.
    :param c: Speed of sound in sea water, in m/s (default 1500).
    :return: A :class:`ShipSourceLevelResult`.
    :raises ValueError: If the inputs are invalid or the shapes mismatch.
    """
    d = _positive(draught, "draught")
    speed = _positive(c, "c")
    freqs = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
    rnl_arr = np.atleast_1d(np.asarray(rnl, dtype=np.float64))
    if np.any(freqs <= 0.0) or not np.all(np.isfinite(freqs)):
        raise ValueError("'frequency' must be positive and finite.")
    if not np.all(np.isfinite(rnl_arr)):
        raise ValueError("'rnl' must be finite.")
    if rnl_arr.size == 1 and freqs.size > 1:
        rnl_arr = np.full(freqs.shape, rnl_arr[0])
    if rnl_arr.shape != freqs.shape:
        raise ValueError("'rnl' and 'frequency' must have the same length.")
    source_depth = _SOURCE_DEPTH_FRACTION * d
    delta_l = _surface_correction(freqs, source_depth, speed)
    return ShipSourceLevelResult(
        frequencies=freqs,
        radiated_noise_level=rnl_arr,
        surface_correction=delta_l,
        source_level=rnl_arr + delta_l,
        source_depth=source_depth,
        sound_speed=speed,
    )
