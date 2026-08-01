#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Predicted source-level spectrum of shipping traffic (semi-empirical models).

When no measured spectrum is available, the underwater radiated-noise source
level of a ship can be *estimated* from readily available traffic parameters
(vessel class, speed, length) with published semi-empirical models. This module
implements three, selectable through ``model``:

* ``"jomopans-echo"`` -- the JOMOPANS-ECHO reference spectrum (MacGillivray &
  de Jong 2021), per **vessel class** with class reference speeds; validated
  against 1862 source-level measurements (ECHO programme), σ ≈ 6 dB. Default.
* ``"randi"`` -- the RANDI 3.1 semi-empirical model: an "average ship" baseline
  spectrum scaled by speed and length.
* ``"wales-heitmeyer"`` -- the Wales & Heitmeyer (2002) ensemble merchant-ship
  spectrum (no speed/length dependence), valid 30 Hz-1200 Hz.

All three return an equivalent-monopole source spectral-density level (dB re
1 µPa²/Hz at 1 m, source depth 6 m) and the decidecade-band source level
(dB re 1 µPa m). The predicted spectrum can be used as the ``shipping`` input of
:func:`phonometry.underwater.ocean_ambient_noise.ocean_ambient_noise` or placed at range
with :func:`phonometry.underwater.propagation.transmission_loss`.

Source (clean-room, implemented from the equations, validated against the
authors' own Excel reference implementation, File S1): MacGillivray, A.;
de Jong, C. (2021), "A Reference Spectrum Model for Estimating Source Levels of
Marine Shipping Based on Automated Identification System Data", J. Mar. Sci.
Eng. 9(4), 369, https://doi.org/10.3390/jmse9040369 (CC-BY) -- which also
reproduces RANDI 3.1 [Breeding et al.] and Wales & Heitmeyer (2002).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import require_positive, require_positive_array

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

#: Reference length ``l0`` = 300 ft, expressed in metres.
_L_REF_M = 300.0 / 3.28084

_MODELS = ("jomopans-echo", "randi", "wales-heitmeyer")

#: JOMOPANS-ECHO per-class parameters: ``class -> (V_C [kn], cargo, D_lo, D_hi)``
#: (Table 1 and sheet *Parameters* of File S1). ``D_lo`` is the damping of the
#: cargo low-frequency hump; ``D_hi`` the damping of the main spectrum.
_VESSEL_CLASSES: dict[str, tuple[float, bool, float, float]] = {
    "bulker": (13.9, True, 0.8, 3.0),
    "containership": (18.0, True, 0.8, 3.0),
    "cruise": (17.1, False, 1.0, 4.0),
    "dredger": (9.5, False, 1.0, 3.0),
    "fishing": (6.4, False, 1.0, 3.0),
    "government/research": (8.0, False, 1.0, 3.0),
    "naval": (11.1, False, 1.0, 3.0),
    "other": (7.4, False, 1.0, 3.0),
    "passenger": (9.7, False, 1.0, 3.0),
    "recreational": (10.6, False, 1.0, 3.0),
    "tanker": (12.4, True, 1.0, 3.0),
    "tug": (3.7, False, 1.0, 3.0),
    "vehicle carrier": (15.8, True, 1.0, 3.0),
}

#: Available JOMOPANS-ECHO vessel classes (for discovery and validation).
VESSEL_CLASSES: tuple[str, ...] = tuple(sorted(_VESSEL_CLASSES))


def _default_bands() -> NDArray[np.float64]:
    """Decidecade band centres 10 Hz-31.5 kHz (indices -20..15), in Hz."""
    i = np.arange(-20, 16, dtype=np.float64)
    return np.asarray(10.0 ** (i / 10.0) * 1000.0, dtype=np.float64)


def _clean_frequency(frequency_hz: NDArray[np.float64] | list[float] | None) -> NDArray[np.float64]:
    if frequency_hz is None:
        return _default_bands()
    return require_positive_array(frequency_hz, "frequency_hz")


def _jomopans_echo(
    f: NDArray[np.float64], vessel_class: str, speed_knots: float, length_m: float
) -> NDArray[np.float64]:
    key = vessel_class.strip().lower()
    if key not in _VESSEL_CLASSES:
        raise ValueError(f"'vessel_class' must be one of {VESSEL_CLASSES}, got {vessel_class!r}.")
    v_c, cargo, d_lo, d_hi = _VESSEL_CLASSES[key]
    v = require_positive(speed_knots, "speed_knots")
    length = require_positive(length_m, "length_m")
    hump = cargo & (f < 100.0)
    k = np.where(hump, 208.0, 191.0)
    j = np.where(hump, 2.0, 0.0)
    d = np.where(hump, d_lo, d_hi)
    f1 = np.where(hump, 600.0, 480.0) / v_c
    psd = (
        k
        - 10.0 * (j + 2.0) * np.log10(f1)
        + 5.0 * j * np.log10(f)
        - 10.0 * np.log10((1.0 - (f / f1) ** (0.5 * (j + 2.0))) ** 2 + d**2)
        + 60.0 * np.log10(v / v_c)
        + 20.0 * np.log10(length / _L_REF_M)
    )
    return np.asarray(psd, dtype=np.float64)


def _randi(f: NDArray[np.float64], speed_knots: float, length_m: float) -> NDArray[np.float64]:
    v = require_positive(speed_knots, "speed_knots")
    length_ft = require_positive(length_m, "length_m") * 3.28084
    lf = -10.0 * np.log10(
        10.0 ** (-1.06 * np.log10(f) - 14.34) + 10.0 ** (3.32 * np.log10(f) - 21.425)
    )
    base = np.where(f < 500.0, lf, 173.2 - 18.0 * np.log10(f))
    d_l = length_ft**1.15 / 3643.0
    d_f = np.where(
        f <= 28.4,
        8.1,
        np.where(f <= 191.6, 22.3 - 9.77 * np.log10(f), 0.0),
    )
    psd = base + 60.0 * np.log10(v / 12.0) + 20.0 * np.log10(length_ft / 300.0) + d_f * d_l + 3.0
    return np.asarray(psd, dtype=np.float64)


def _wales_heitmeyer(f: NDArray[np.float64]) -> NDArray[np.float64]:
    psd = 230.0 - 10.0 * np.log10(f**3.594) + 10.0 * np.log10((1.0 + (f / 340.0) ** 2) ** 0.917)
    return np.asarray(psd, dtype=np.float64)


@dataclass(frozen=True)
class ShipTrafficSpectrum:
    r"""Predicted ship source-level spectrum.

    :ivar frequency: Frequencies, in Hz.
    :ivar source_psd: Source pressure spectral-density level, in dB re 1 µPa²/Hz
        at 1 m (equivalent monopole).
    :ivar band_level: Decidecade-band source level, in dB re 1 µPa m
        (``source_psd`` plus :math:`10 \log_{10}(0.231 f)`).
    :ivar model: The model used.
    :ivar vessel_class: The vessel class (JOMOPANS-ECHO only; else ``None``).
    :ivar speed_knots: Speed used, in knots (``None`` if the model ignores it).
    :ivar length_m: Length used, in metres (``None`` if the model ignores it).
    """

    frequency: NDArray[np.float64]
    source_psd: NDArray[np.float64]
    band_level: NDArray[np.float64]
    model: str
    vessel_class: str | None
    speed_knots: float | None
    length_m: float | None

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the predicted source spectral-density level versus frequency."""
        from .._i18n import check_language
        from .._plot.underwater import plot_ship_traffic_spectrum

        return plot_ship_traffic_spectrum(self, ax=ax, language=check_language(language), **kwargs)


def ship_source_spectrum(
    speed_knots: float = 12.0,
    length_m: float = 100.0,
    *,
    vessel_class: str = "containership",
    model: str = "jomopans-echo",
    frequency_hz: NDArray[np.float64] | list[float] | None = None,
) -> ShipTrafficSpectrum:
    """Predicted underwater source-level spectrum of a ship.

    :param speed_knots: Vessel speed, in knots (used by ``"jomopans-echo"`` and
        ``"randi"``; ignored by ``"wales-heitmeyer"``).
    :param length_m: Vessel length, in metres (used by ``"jomopans-echo"`` and
        ``"randi"``; ignored by ``"wales-heitmeyer"``).
    :param vessel_class: JOMOPANS-ECHO vessel class (see :data:`VESSEL_CLASSES`);
        used only by ``"jomopans-echo"``.
    :param model: ``"jomopans-echo"`` (default), ``"randi"`` or
        ``"wales-heitmeyer"``.
    :param frequency_hz: Frequencies, in Hz; defaults to the decidecade bands
        10 Hz-31.5 kHz of the JOMOPANS-ECHO validation range.
    :return: A :class:`ShipTrafficSpectrum`.
    :raises ValueError: If ``model``/``vessel_class`` is unknown or an input is
        invalid.
    """
    f = _clean_frequency(frequency_hz)
    key = model.strip().lower()
    used_class: str | None = None
    used_speed: float | None = None
    used_length: float | None = None
    if key == "jomopans-echo":
        psd = _jomopans_echo(f, vessel_class, speed_knots, length_m)
        used_class = vessel_class.strip().lower()
        used_speed = float(speed_knots)
        used_length = float(length_m)
    elif key == "randi":
        psd = _randi(f, speed_knots, length_m)
        used_speed = float(speed_knots)
        used_length = float(length_m)
    elif key == "wales-heitmeyer":
        psd = _wales_heitmeyer(f)
    else:
        raise ValueError(f"'model' must be one of {_MODELS}, got {model!r}.")
    band = psd + 10.0 * np.log10(0.231 * f)
    return ShipTrafficSpectrum(
        frequency=f,
        source_psd=psd,
        band_level=np.asarray(band, dtype=np.float64),
        model=key,
        vessel_class=used_class,
        speed_knots=used_speed,
        length_m=used_length,
    )
