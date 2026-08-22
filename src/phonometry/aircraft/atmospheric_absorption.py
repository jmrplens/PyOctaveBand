#  Copyright (c) 2026. Jose Manuel Requena Plens
"""One-third-octave-band atmospheric absorption for aircraft noise (SAE ARP 5534).

Aircraft noise certification (14 CFR Part 36, ICAO Annex 16 Vol. I) works with
one-third-octave-band spectra, and correcting a measured flyover to reference
atmospheric conditions requires the band attenuation over the propagation path.
The pure-tone attenuation coefficient is the ISO 9613-1 one (identical, per ARP
5534 §3.1) already provided by :func:`~phonometry.environment.propagation.air_absorption.air_attenuation`;
this module adds the **SAE Method** (ARP 5534 §3.2.2), a regression that turns
the pure-tone mid-band path-length attenuation into the one-third-octave-band
attenuation and stays consistent with the ISO/ANSI Exact Method well beyond the
50 dB limit of the older Approximate Method.

* :func:`sae_band_attenuation` -- the one-third-octave-band attenuation over a
  path, returned as a :class:`AircraftBandAttenuation` with a ``.plot()``.

Source (clean-room, implemented from the standard text): SAE ARP 5534 (2021),
*Application of Pure-Tone Atmospheric Absorption Losses to One-Third-Octave-Band
Data*, Eqs. 7-10; the pure-tone coefficient is ISO 9613-1:1993.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import (
    require_non_negative,
    require_positive_array,
    require_ranks,
    require_same_length,
)
from ..environment.propagation.air_absorption import air_attenuation

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

# SAE Method regression constants (ARP 5534 §3.2.2, Eqs. 7-8).
_A = 0.867942
_B = 0.111761
_C = 0.95824
_D = 0.008191
_E = 1.6
_F = 9.2
_G = 0.765
#: Mid-band attenuation (dB) at which the piecewise SAE Method switches branch.
_SPLIT_DB = 150.0


def _sae_band(delta_t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map pure-tone mid-band attenuation ``δ_t`` (dB) to band attenuation (dB)."""
    # np.where evaluates both branches; clamp the low branch's input to the split
    # point so its power base stays positive (it turns negative near 1209 dB) and
    # never produces a NaN for the discarded δ_t >= 150 dB samples.
    dt_low = np.minimum(delta_t, _SPLIT_DB)
    low = _A * dt_low * (1.0 + _B * (_C - _D * dt_low)) ** _E
    high = _F + _G * delta_t
    return np.asarray(np.where(delta_t < _SPLIT_DB, low, high), dtype=np.float64)


@dataclass(frozen=True)
class AircraftBandAttenuation:
    r"""One-third-octave-band atmospheric attenuation over a path (SAE ARP 5534).

    :ivar frequency: Nominal one-third-octave-band centre frequencies, in Hz.
    :ivar band_attenuation: SAE-Method band attenuation ``δ_B`` per band, in dB.
    :ivar midband_attenuation: Pure-tone mid-band path-length attenuation
        :math:`\delta_\mathrm{t} = \alpha \cdot s` per band, in dB (ISO 9613-1
        coefficient).
    :ivar coefficient: Pure-tone mid-band attenuation coefficient ``α`` per band,
        in dB/m.
    :ivar path_length: Propagation path length ``s``, in metres.
    :ivar temperature: Air temperature, in degrees Celsius.
    :ivar relative_humidity: Relative humidity, in percent.
    :ivar pressure: Ambient atmospheric pressure, in kPa.
    """

    frequency: NDArray[np.float64]
    band_attenuation: NDArray[np.float64]
    midband_attenuation: NDArray[np.float64]
    coefficient: NDArray[np.float64]
    path_length: float
    temperature: float
    relative_humidity: float
    pressure: float

    def __post_init__(self) -> None:
        """Reject an attenuation whose arrays do not all run over the same bands.

        Every array here is one value per one-third-octave band of the same
        spectrum: the band attenuation the certification correction subtracts,
        the pure-tone mid-band attenuation it was regressed from, and the
        coefficient behind that. An array of another length either stops the
        figure, which draws two of them against ``frequency`` on one pair of
        axes and gets matplotlib's complaint about an x and a y that name
        neither field, or -- where it has collapsed to a single value -- is
        broadcast by numpy over the whole spectrum, correcting every band of a
        measured flyover by one band's attenuation without a word.

        :raises ValueError: if a per-band array disagrees with the rest.
        """
        require_ranks(
            self,
            frequency=1,
            band_attenuation=1,
            midband_attenuation=1,
            coefficient=1,
        )
        require_same_length(
            self,
            "frequency",
            "band_attenuation",
            "midband_attenuation",
            "coefficient",
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the band and pure-tone mid-band attenuation versus frequency."""
        from .._i18n import check_language
        from .._plot.aircraft import plot_aircraft_band_attenuation

        return plot_aircraft_band_attenuation(
            self, ax=ax, language=check_language(language), **kwargs
        )


def sae_band_attenuation(
    frequencies: NDArray[np.float64] | list[float],
    path_length: float,
    *,
    temperature: float = 25.0,
    relative_humidity: float = 70.0,
    pressure: float = 101.325,
) -> AircraftBandAttenuation:
    r"""One-third-octave-band atmospheric attenuation (SAE ARP 5534, SAE Method).

    Computes the pure-tone attenuation coefficient at each band's exact mid-band
    frequency (ISO 9613-1, :math:`f_{\mathrm{m},i} = 10^{i/10}`), forms the mid-band
    path-length attenuation :math:`\delta_\mathrm{t} = \alpha \cdot s` and maps it to
    the band attenuation
    ``δ_B`` with the SAE-Method regression (Eqs. 7-8).

    :param frequencies: Nominal one-third-octave-band centre frequencies, in Hz
        (standard range 50 Hz-10 kHz; the method extends to 25 Hz-20 kHz).
    :param path_length: Propagation path length ``s``, in metres (``>= 0``).
    :param temperature: Air temperature, in degrees Celsius (SAE window
        ~6-32 °C; default 25 °C, the ARP 5534 reference point).
    :param relative_humidity: Relative humidity, in percent (SAE window
        ~20-95 %; default 70 %).
    :param pressure: Ambient atmospheric pressure, in kPa (default 101.325).
    :return: An :class:`AircraftBandAttenuation`.
    :raises ValueError: If the inputs are invalid.
    """
    f = require_positive_array(frequencies, "frequencies")
    s = require_non_negative(path_length, "path_length")

    # Pure-tone coefficient at the exact mid-band frequency (ISO 9613-1).
    alpha = air_attenuation(
        f, temperature, relative_humidity, pressure, exact_midband=True
    )
    delta_t = alpha * s
    delta_b = _sae_band(delta_t)
    return AircraftBandAttenuation(
        frequency=f,
        band_attenuation=delta_b,
        midband_attenuation=np.asarray(delta_t, dtype=np.float64),
        coefficient=np.asarray(alpha, dtype=np.float64),
        path_length=s,
        temperature=float(temperature),
        relative_humidity=float(relative_humidity),
        pressure=float(pressure),
    )
