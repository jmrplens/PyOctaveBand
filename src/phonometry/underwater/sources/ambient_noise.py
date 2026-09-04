#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Ocean ambient-noise spectrum levels (Wenz framework).

Deep-water ambient-noise **spectrum levels** (dB re 1 µPa²/Hz) from the two
physically grounded components of the Wenz curves:

* :func:`wind_noise_spectrum` -- wind / sea-surface (Knudsen) noise via Wenz's
  "rule of fives",
  :math:`\mathrm{NL} = 51.02 - (5/3) \cdot 10 (\log_{10} f - \log_{10}(U/5))`
  (``f`` in kHz, ``U`` in knots; the historical 25 dB anchor is re 20 µPa and
  becomes :math:`25 + 20 \log_{10}(20)` re 1 µPa), valid over roughly 500 Hz-5 kHz.
* :func:`thermal_noise_spectrum` -- the molecular thermal-noise limit (Mellen
  1952), :math:`\langle p^2(f) \rangle = 4 \pi k T \rho f^2 / c` (Pa²/Hz),
  dominant above ~50 kHz.

:func:`ocean_ambient_noise` energy-sums the enabled components (and an optional
caller-supplied shipping spectrum) into a composite :class:`AmbientNoiseResult`
with a ``.plot()`` of the Wenz-style curves.

The low-frequency turbulence band and a built-in distant-shipping model are out
of scope: Wenz (1962) and Carey & Evans note these bands are strongly variable
and shipping-dependent, with no single fixed analytic parametrisation; a shipping
spectrum may be supplied by the caller. Source: Carey & Evans, *Ocean Ambient
Noise* (2011) -- the rule of fives (p. 2) and the thermal-noise derivation
(Appendix F).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._internal.validation import (
    require_above_absolute_zero,
    require_equal_shapes,
    require_finite_array,
    require_non_negative,
    require_positive,
    require_positive_array,
    require_ranks,
    require_same_length,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

#: Boltzmann constant ``k`` (J/K).
_BOLTZMANN = 1.380649e-23
#: Underwater reference pressure ``p₀`` (Pa), i.e. 1 µPa.
_P_REF = 1e-6
#: Wind-noise reference wind speed for the rule of fives (knots).
_WIND_REF_KNOTS = 5.0
#: Rule-of-fives anchor at 1 kHz / 5 kn, re 1 µPa²/Hz. Wenz/Knudsen state
#: "25 dB (5 × 5)" **re 0.0002 dyn/cm² = 20 µPa** (the historical reference
#: pressure); converting to the ISO 18405 1 µPa reference adds
#: ``20·lg(20) ≈ 26.02`` dB.
_WIND_ANCHOR_DB = 25.0 + 20.0 * np.log10(20.0)


def wind_noise_spectrum(
    frequency_hz: NDArray[np.float64] | list[float] | float, wind_speed_knots: float
) -> NDArray[np.float64]:
    r"""Wind / sea-surface noise spectrum level (Wenz rule of fives), dB re
    1 µPa²/Hz.

    :math:`\mathrm{NL}(f, U) = 51.02 - (5/3) \cdot 10 (\log_{10} f - \log_{10}(U/5))`
    with ``f`` in kHz and ``U``
    in knots: −5 dB per octave and +5 dB per doubling of wind speed about the
    canonical anchor, which Wenz/Knudsen state as "25 dB (5 × 5)" at 1 kHz for
    5 knots **re 0.0002 dyn/cm² (20 µPa)**, i.e.
    :math:`25 + 20 \log_{10}(20) \approx 51.02` dB
    once referenced to the ISO 18405 1 µPa. Valid over roughly 500 Hz-5 kHz
    and winds of 2.5-40 knots (the stated range of the wind-doubling law);
    outside both the formula extrapolates.

    :param frequency_hz: Frequency, in Hz (scalar or array).
    :param wind_speed_knots: Wind speed ``U``, in knots. A calm sea (``0``) has
        no wind-driven noise, returning ``-inf`` (zero contribution in the
        energy sum).
    :return: Wind-noise spectrum level per frequency, in dB re 1 µPa²/Hz.
    :raises ValueError: If the inputs are invalid (a negative wind speed).
    """
    f_khz = require_positive_array(frequency_hz, "frequency_hz") / 1000.0
    u = require_non_negative(wind_speed_knots, "wind_speed_knots")
    if u <= 0.0:  # calm sea: no wind-driven noise (u is non-negative here)
        return np.full(f_khz.shape, -np.inf, dtype=np.float64)
    nl = _WIND_ANCHOR_DB - (5.0 / 3.0) * 10.0 * (
        np.log10(f_khz) - np.log10(u / _WIND_REF_KNOTS)
    )
    return np.asarray(nl, dtype=np.float64)


def thermal_noise_spectrum(
    frequency_hz: NDArray[np.float64] | list[float] | float,
    *,
    temperature: float = 16.85,
    density: float = 1025.0,
    sound_speed: float = 1500.0,
) -> NDArray[np.float64]:
    r"""Molecular thermal-noise spectrum level (Mellen 1952), dB re 1 µPa²/Hz.

    :math:`\langle p^2(f) \rangle = 4 \pi k T \rho f^2 / c` (Pa²/Hz); the
    level is :math:`10 \log_{10}(\langle p^2 \rangle / p_0^2)`.

    :param frequency_hz: Frequency, in Hz (scalar or array).
    :param temperature: Water temperature, in degrees Celsius (default 16.85 °C
        = 290 K).
    :param density: Water density :math:`\rho`, in kg/m³ (default 1025).
    :param sound_speed: Sound speed ``c``, in m/s (default 1500).
    :return: Thermal-noise spectrum level per frequency, in dB re 1 µPa²/Hz.
    :raises ValueError: If the inputs are invalid.
    """
    f = require_positive_array(frequency_hz, "frequency_hz")
    t_kelvin = require_above_absolute_zero(float(temperature), "temperature") + 273.15
    rho = require_positive(density, "density")
    c = require_positive(sound_speed, "sound_speed")
    p2 = 4.0 * np.pi * _BOLTZMANN * t_kelvin * rho * f**2 / c
    return np.asarray(10.0 * np.log10(p2 / _P_REF**2), dtype=np.float64)


@dataclass(frozen=True)
class AmbientNoiseResult:
    """Composite ambient-noise spectrum (Wenz framework).

    :ivar frequency: Frequencies, in Hz.
    :ivar spectrum_level: Composite spectrum level (energy sum of the enabled
        components), in dB re 1 µPa²/Hz.
    :ivar wind: Wind-noise component per frequency, in dB re 1 µPa²/Hz.
    :ivar thermal: Thermal-noise component per frequency, in dB re 1 µPa²/Hz.
    :ivar shipping: Caller-supplied shipping component, or ``None``.
    :ivar wind_speed_knots: The wind speed used, in knots.
    """

    frequency: NDArray[np.float64]
    spectrum_level: NDArray[np.float64]
    wind: NDArray[np.float64]
    thermal: NDArray[np.float64]
    shipping: NDArray[np.float64] | None
    wind_speed_knots: float

    def __post_init__(self) -> None:
        """Reject a spectrum whose components do not run over its frequencies.

        A Wenz chart is read by where its curves meet -- the frequency at
        which the wind curve gives way to the thermal floor is taken straight
        off the picture -- and the composite level drawn above them is stated
        to be the energy sum of exactly the components drawn beneath it.
        Nothing in the result records which frequency an entry belongs to
        except its position, so a component of another length was summed over
        a different set of frequencies than the ones the chart puts it under.

        :func:`ocean_ambient_noise` already measures ``shipping`` against the
        frequencies it was handed, but a frozen result is a value that gets
        rebuilt: substituting a measured shipping spectrum through
        :func:`dataclasses.replace` re-enters this constructor and nothing
        else, and the composite level that comes through with it was summed
        without the curve now beside it.

        :raises ValueError: if a component disagrees with the frequency axis.
        """
        require_ranks(
            self, frequency=1, spectrum_level=1, wind=1, thermal=1, shipping=1
        )
        require_same_length(
            self,
            "frequency",
            "spectrum_level",
            "wind",
            "thermal",
            "shipping",
            axis="frequency",
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the composite spectrum and its components versus frequency."""
        from ..._i18n import check_language
        from ..._plot.underwater import plot_ambient_noise

        return plot_ambient_noise(
            self, ax=ax, language=check_language(language), **kwargs
        )


def ocean_ambient_noise(
    frequency_hz: NDArray[np.float64] | list[float],
    *,
    wind_speed_knots: float,
    shipping: NDArray[np.float64] | list[float] | None = None,
    temperature: float = 16.85,
    density: float = 1025.0,
    sound_speed: float = 1500.0,
) -> AmbientNoiseResult:
    """Composite deep-water ambient-noise spectrum (wind + thermal [+ shipping]).

    Energy-sums the wind-noise (rule of fives) and thermal-noise (Mellen)
    components, plus an optional caller-supplied shipping spectrum.

    :param frequency_hz: Frequencies, in Hz (1-D, strictly positive).
    :param wind_speed_knots: Wind speed ``U``, in knots.
    :param shipping: Optional shipping-noise spectrum level per frequency, in dB
        re 1 µPa²/Hz (same length as ``frequency_hz``), or ``None``.
    :param temperature: Water temperature, in degrees Celsius.
    :param density: Water density, in kg/m³.
    :param sound_speed: Sound speed, in m/s.
    :return: An :class:`AmbientNoiseResult`.
    :raises ValueError: If the inputs are invalid.
    """
    f = require_positive_array(frequency_hz, "frequency_hz")
    wind = wind_noise_spectrum(f, wind_speed_knots)
    thermal = thermal_noise_spectrum(
        f, temperature=temperature, density=density, sound_speed=sound_speed
    )
    energies = 10.0 ** (wind / 10.0) + 10.0 ** (thermal / 10.0)
    ship_arr: NDArray[np.float64] | None = None
    if shipping is not None:
        ship_arr = require_finite_array(shipping, "shipping")
        require_equal_shapes(
            "ocean_ambient_noise",
            {"frequency_hz": f.shape, "shipping": ship_arr.shape},
            "frequency",
        )
        energies = energies + 10.0 ** (ship_arr / 10.0)
    spectrum = 10.0 * np.log10(energies)
    return AmbientNoiseResult(
        frequency=f,
        spectrum_level=np.asarray(spectrum, dtype=np.float64),
        wind=wind,
        thermal=thermal,
        shipping=ship_arr,
        wind_speed_knots=float(wind_speed_knots),
    )
