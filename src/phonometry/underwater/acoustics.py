#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Underwater-acoustics reference levels (ISO 18405:2017).

Underwater sound levels use a reference pressure of **1 µPa** (not the 20 µPa of
airborne acoustics) and a reference sound exposure of **1 µPa²·s**. This module
realises the ISO 18405 terminology as the shared level primitives used by the
ship-radiated-noise and pile-driving modules:

* :func:`sound_pressure_level` -- the mean-square sound pressure level,
  :math:`\mathrm{SPL} = 10 \log_{10}(\langle p^2 \rangle / p_0^2)` dB re 1 µPa.
* :func:`sound_exposure_level` -- the time-integrated exposure level,
  :math:`\mathrm{SEL} = 10 \log_{10}\!\left( \int p^2 \, dt / E_0 \right)` dB re
  1 µPa²·s.
* :func:`peak_sound_pressure_level` -- the zero-to-peak level
  :math:`20 \log_{10}(\max \lvert p \rvert / p_0)` dB re 1 µPa.

:func:`underwater_to_in_air_spl` / :func:`in_air_to_underwater_spl` convert a
level between the two reference pressures (a
:math:`20 \log_{10}(20) \approx 26.02` dB reference
change, **not** an energy/intensity equivalence, which would additionally
involve the media impedances). For background-noise subtraction of a measured
level, reuse the ISO 3744 ``background_noise_correction`` (``K1``) helper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Underwater reference sound pressure ``p₀`` (Pa), i.e. 1 µPa (ISO 18405).
UNDERWATER_REFERENCE_PRESSURE = 1e-6
#: Underwater reference sound exposure ``E₀`` (Pa²·s), i.e. 1 µPa²·s (ISO 18405).
UNDERWATER_REFERENCE_EXPOSURE = 1e-12
#: In-air reference sound pressure (Pa), i.e. 20 µPa.
_IN_AIR_REFERENCE_PRESSURE = 20e-6
#: Level offset between the 20 µPa and 1 µPa references, ``20·lg(20)`` dB.
_REFERENCE_OFFSET_DB = 20.0 * np.log10(_IN_AIR_REFERENCE_PRESSURE / UNDERWATER_REFERENCE_PRESSURE)


def _positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"'{name}' must be a positive, finite number.")
    return scalar


def _validate_pressure(
    pressure: NDArray[np.float64] | list[float], *, min_samples: int = 1
) -> NDArray[np.float64]:
    sig = np.asarray(pressure, dtype=np.float64)
    if sig.ndim != 1:
        raise ValueError("'pressure' must be one-dimensional.")
    if sig.size < min_samples:
        plural = "sample" if min_samples == 1 else "samples"
        raise ValueError(f"'pressure' must contain at least {min_samples} {plural}.")
    if not np.all(np.isfinite(sig)):
        raise ValueError("'pressure' must be finite.")
    return sig


def sound_pressure_level(
    pressure: NDArray[np.float64] | list[float],
    *,
    reference: float = UNDERWATER_REFERENCE_PRESSURE,
) -> float:
    r"""Mean-square sound pressure level (ISO 18405 / ISO 18406 Formula 7).

    :math:`\mathrm{SPL} = 10 \log_{10}(\langle p^2 \rangle / p_0^2)` dB, with ``p``
    in pascals and the underwater reference :math:`p_0 = 1` µPa by default.

    :param pressure: Sound-pressure time series (1-D), in Pa.
    :param reference: Reference pressure :math:`p_0`, in Pa (default 1 µPa).
    :return: Sound pressure level, in dB re the reference.
    :raises ValueError: If the signal is invalid or has no energy.
    """
    sig = _validate_pressure(pressure)
    p0 = _positive(reference, "reference")
    mean_square = float(np.mean(sig**2))
    if mean_square <= 0.0:
        raise ValueError("'pressure' has no energy.")
    return float(10.0 * np.log10(mean_square / p0**2))


def sound_exposure_level(
    pressure: NDArray[np.float64] | list[float],
    fs: float,
    *,
    reference: float = UNDERWATER_REFERENCE_EXPOSURE,
) -> float:
    r"""Sound exposure level (ISO 18405 / ISO 18406 Formulae 3-4).

    :math:`\mathrm{SEL} = 10 \log_{10}(E/E_0)` dB re 1 µPa²·s, with the sound
    exposure :math:`E = \int p^2 \, dt \approx (1/f_s) \sum p^2` over the
    record.

    :param pressure: Sound-pressure time series (1-D), in Pa.
    :param fs: Sample rate, in Hz.
    :param reference: Reference exposure :math:`E_0`, in Pa²·s (default
        1 µPa²·s).
    :return: Sound exposure level, in dB re the reference.
    :raises ValueError: If the inputs are invalid or the signal has no energy.
    """
    sig = _validate_pressure(pressure)
    fs_v = _positive(fs, "fs")
    e0 = _positive(reference, "reference")
    exposure = float(np.sum(sig**2) / fs_v)
    if exposure <= 0.0:
        raise ValueError("'pressure' has no energy.")
    return float(10.0 * np.log10(exposure / e0))


def peak_sound_pressure_level(
    pressure: NDArray[np.float64] | list[float],
    *,
    reference: float = UNDERWATER_REFERENCE_PRESSURE,
) -> float:
    r"""Zero-to-peak sound pressure level (ISO 18406 6.4.2.1.3).

    :math:`L_{p,\mathrm{pk}} = 20 \log_{10}(\max \lvert p \rvert / p_0)` dB re
    1 µPa.

    :param pressure: Sound-pressure time series (1-D), in Pa.
    :param reference: Reference pressure :math:`p_0`, in Pa (default 1 µPa).
    :return: Peak sound pressure level, in dB re the reference.
    :raises ValueError: If the signal is invalid or is all zero.
    """
    sig = _validate_pressure(pressure)
    p0 = _positive(reference, "reference")
    peak = float(np.max(np.abs(sig)))
    if peak <= 0.0:
        raise ValueError("'pressure' has no energy.")
    return float(20.0 * np.log10(peak / p0))


def underwater_to_in_air_spl(level: float) -> float:
    r"""Re-reference an underwater SPL (re 1 µPa) to the in-air 20 µPa
    reference.

    Subtracts :math:`20 \log_{10}(20) \approx 26.02` dB. This is a
    **reference-pressure change only** -- it is not the in-water/in-air
    intensity equivalence, which also involves the media characteristic
    impedances.

    :param level: Level in dB re 1 µPa.
    :return: The same pressure expressed in dB re 20 µPa.
    """
    return float(float(level) - _REFERENCE_OFFSET_DB)


def in_air_to_underwater_spl(level: float) -> float:
    r"""Re-reference an in-air SPL (re 20 µPa) to the underwater 1 µPa
    reference.

    Adds :math:`20 \log_{10}(20) \approx 26.02` dB (a reference-pressure change
    only; see :func:`underwater_to_in_air_spl`).

    :param level: Level in dB re 20 µPa.
    :return: The same pressure expressed in dB re 1 µPa.
    """
    return float(float(level) + _REFERENCE_OFFSET_DB)
