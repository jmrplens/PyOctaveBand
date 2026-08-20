#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Radiated underwater sound from percussive pile driving (ISO 18406:2017).

Percussive pile driving radiates a train of impulsive acoustic pulses, one per
hammer strike. ISO 18406 characterises them with:

* :func:`single_strike_sel` -- the single-strike sound exposure level
  ``SEL_ss`` of one pulse (Formulae 3-4), reusing the 1 µPa²·s reference.
* :func:`cumulative_sel` / :func:`cumulative_sel_identical` -- the cumulative
  sound exposure level over N strikes (Formulae 8-9); for N identical strikes
  :math:`\mathrm{SEL}_{\mathrm{cum}} =
  \mathrm{SEL}_{\mathrm{ss}} + 10 \log_{10} N`.
* :func:`pile_strike_metrics` -- a :class:`PileStrikeResult` bundling the
  single-strike SEL, the peak sound pressure level, the SPL/Leq and the
  90 %-energy pulse duration for one recorded strike, with a ``.plot()``.
* :func:`strike_sel_spectrum` -- the same single-strike SEL resolved into
  fractional-octave bands (ISO 18406 6.4.2.2), the input a marine-mammal
  assessment needs: feed it to
  :func:`~phonometry.underwater.bioacoustics.weighting.weighted_exposure` to
  obtain the weighted cumulative SEL of a piling campaign and its margin
  against the regulatory injury and TTS criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ...io._resolve import SignalInput, resolve_fs
from ..acoustics import (
    _positive,
    _validate_pressure,
    peak_sound_pressure_level,
    sound_exposure_level,
    sound_pressure_level,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

#: Lower/upper cumulative-energy fractions defining the 90 % pulse duration.
_ENERGY_LOW = 0.05
_ENERGY_HIGH = 0.95


def single_strike_sel(pressure: SignalInput, fs: float | None = None) -> float:
    """Single-strike sound exposure level ``SEL_ss`` (ISO 18406 Formulae 3-4).

    The sound exposure level of one hammer-strike pulse, integrated over the
    pulse, in dB re 1 µPa²·s.

    :param pressure: Sound-pressure time series of one strike (1-D), in
        Pa. Accepts a
        :class:`phonometry.io.Signal`, whose calibration is applied to the
        samples: this quantity is a pressure, and the underwater reference
        of 1 uPa changes what the decibel is counted from, not what the
        samples have to be in.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :return: Single-strike SEL, in dB re 1 µPa²·s.
    :raises ValueError: If the inputs are invalid.
    """
    return sound_exposure_level(pressure, fs)


def cumulative_sel(single_sels: NDArray[np.float64] | list[float]) -> float:
    r"""Cumulative sound exposure level over N strikes (ISO 18406
    Formulae 8-9).

    :math:`\mathrm{SEL}_{\mathrm{cum}} =
    10 \log_{10} \sum_n 10^{\mathrm{SEL}_n/10}` -- the energy sum of the per-strike
    single-strike SELs.

    :param single_sels: Per-strike single-strike SELs, in dB re 1 µPa²·s.
    :return: Cumulative SEL, in dB re 1 µPa²·s.
    :raises ValueError: If the sequence is empty or non-finite.
    """
    sels = np.asarray(single_sels, dtype=np.float64)
    if sels.ndim != 1 or sels.size < 1:
        msg = "'single_sels' must be a non-empty 1-D sequence."
        raise ValueError(msg)
    if not np.all(np.isfinite(sels)):
        msg = "'single_sels' must be finite."
        raise ValueError(msg)
    return float(10.0 * np.log10(np.sum(10.0 ** (sels / 10.0))))


def cumulative_sel_identical(sel_ss: float, n_strikes: int) -> float:
    r"""Cumulative SEL of ``n_strikes`` identical strikes:
    :math:`\mathrm{SEL}_{\mathrm{ss}} + 10 \log_{10} N`.

    :param sel_ss: Single-strike SEL, in dB re 1 µPa²·s.
    :param n_strikes: Number of (identical) strikes, :math:`N \ge 1`.
    :return: Cumulative SEL, in dB re 1 µPa²·s.
    :raises ValueError: If ``n_strikes`` is not a whole number
        :math:`\ge 1`.
    """
    n_float = float(n_strikes)
    if not n_float.is_integer():
        msg = "'n_strikes' must be a whole number of strikes."
        raise ValueError(msg)
    n = int(n_float)
    if n < 1:
        msg = "'n_strikes' must be at least 1."
        raise ValueError(msg)
    if not np.isfinite(sel_ss):
        msg = "'sel_ss' must be finite."
        raise ValueError(msg)
    return float(float(sel_ss) + 10.0 * np.log10(n))


def _pulse_duration(pressure: NDArray[np.float64], fs: float) -> float:
    """90 %-energy pulse duration: the time between the 5 % and 95 % energy points."""
    energy = np.cumsum(pressure**2)
    total = float(energy[-1])
    if total <= 0.0:
        return 0.0
    cum = energy / total
    lo = int(np.searchsorted(cum, _ENERGY_LOW))
    hi = int(np.searchsorted(cum, _ENERGY_HIGH))
    return float((hi - lo) / fs)


@dataclass(frozen=True)
class PileStrikeResult:
    """Per-strike pile-driving metrics (ISO 18406).

    :ivar single_strike_sel: Single-strike SEL, in dB re 1 µPa²·s.
    :ivar peak_spl: Zero-to-peak sound pressure level, in dB re 1 µPa.
    :ivar spl: Sound pressure level (Leq over the record), in dB re 1 µPa.
    :ivar pulse_duration: 90 %-energy pulse duration, in s.
    :ivar pressure: The strike pressure waveform, in Pa.
    :ivar fs: Sample rate, in Hz.
    """

    single_strike_sel: float
    peak_spl: float
    spl: float
    pulse_duration: float
    pressure: NDArray[np.float64]
    fs: float

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | NDArray[Any]:
        """Plot the strike waveform and its cumulative energy."""
        from ..._i18n import check_language
        from ..._plot.underwater import plot_pile_strike

        return plot_pile_strike(
            self, ax=ax, language=check_language(language), **kwargs
        )


@dataclass(frozen=True)
class StrikeSelSpectrum:
    """Single-strike sound exposure level resolved into fractional-octave bands.

    :ivar frequencies: Nominal band centre frequencies, in Hz.
    :ivar band_sel: Per-band single-strike SEL, in dB re 1 µPa²·s. A band that
        contains no discrete-spectrum bin -- which happens whenever the band is
        narrower than the FFT bin spacing ``fs/n``, i.e. in the lowest bands of
        a short record -- holds no energy at all and is reported as ``-inf``,
        the level of zero exposure. That is the neutral element of an energy
        sum, so such bands pass straight through
        :func:`~phonometry.underwater.bioacoustics.weighting.weighted_exposure`
        without contributing.
    :ivar total_sel: Energy sum of ``band_sel`` over the covered bands, in dB
        re 1 µPa²·s.
    :ivar broadband_sel: The broadband single-strike SEL of the whole record,
        in dB re 1 µPa²·s (equal to ``total_sel`` when the bands span the
        signal's whole occupied spectrum).
    :ivar fraction: Bandwidth fraction (1 for octaves, 3 for one-third octaves).
    :ivar fs: Sample rate, in Hz.
    """

    frequencies: NDArray[np.float64]
    band_sel: NDArray[np.float64]
    total_sel: float
    broadband_sel: float
    fraction: int
    fs: float

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the per-band single-strike SEL."""
        from ..._i18n import check_language
        from ..._plot.underwater import plot_strike_sel_spectrum

        return plot_strike_sel_spectrum(
            self, ax=ax, language=check_language(language), **kwargs
        )


def strike_sel_spectrum(
    pressure: SignalInput,
    fs: float | None = None,
    *,
    fraction: int = 3,
    limits: tuple[float, float] = (10.0, 20_000.0),
) -> StrikeSelSpectrum:
    r"""Band-resolved single-strike sound exposure level.

    The sound exposure :math:`E = \int p^2 \, dt` is split between
    fractional-octave bands by integrating the discrete power spectrum over
    each band (Parseval), so
    the energy sum of the returned band levels reproduces the broadband
    :func:`single_strike_sel` of the same record to within the energy that
    falls outside ``limits``.

    Bands narrower than the FFT bin spacing ``fs/n`` contain no bin and are
    reported as ``-inf`` dB (see :class:`StrikeSelSpectrum`); the result can be
    handed straight to
    :func:`~phonometry.underwater.bioacoustics.weighting.weighted_exposure`.

    :param pressure: Sound-pressure time series of one strike (1-D), in
        Pa. Accepts a
        :class:`phonometry.io.Signal`, whose calibration is applied to the
        samples: this quantity is a pressure, and the underwater reference
        of 1 uPa changes what the decibel is counted from, not what the
        samples have to be in.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param fraction: Bandwidth fraction: 1 (octave) or 3 (one-third octave).
    :param limits: Lower and upper band-centre limits, in Hz.
    :return: A :class:`StrikeSelSpectrum`.
    :raises ValueError: If the inputs are invalid.
    """
    from ...filters.frequencies import nominal_frequencies

    fs = resolve_fs(pressure, fs, name="pressure")
    sig = _validate_pressure(pressure, min_samples=2)
    fs_v = _positive(fs, "fs")
    if int(fraction) not in (1, 3):
        msg = "'fraction' must be 1 (octave) or 3 (one-third octave)."
        raise ValueError(msg)
    lo, hi = (float(limits[0]), float(limits[1]))
    if not (np.isfinite(lo) and np.isfinite(hi)) or not (0.0 < lo < hi):
        msg = "'limits' must be a finite, increasing, positive pair."
        raise ValueError(msg)

    centres, lower, upper, _ = nominal_frequencies(int(fraction), [lo, hi])
    n = sig.size
    spectrum = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_v)
    # Parseval on the one-sided spectrum: E = (1/(fs·n))·Σ|X_k|², with the
    # interior bins counted twice because the negative half is not stored.
    weight = np.full(freqs.size, 2.0)
    weight[0] = 1.0
    if n % 2 == 0:
        weight[-1] = 1.0
    energy = weight * np.abs(spectrum) ** 2 / (fs_v * n)

    fc = np.asarray(centres, dtype=np.float64)
    band_energy = np.array(
        [
            float(energy[(freqs >= f_lo) & (freqs < f_hi)].sum())
            for f_lo, f_hi in zip(lower, upper, strict=True)
        ]
    )
    e0 = 1e-12  # 1 µPa²·s in Pa²·s
    # An empty band (narrower than the bin spacing fs/n) carries no energy, so
    # its level is -inf: the neutral element of the energy sum downstream.
    with np.errstate(divide="ignore"):
        band_sel = 10.0 * np.log10(band_energy / e0)
    total = float(band_energy.sum())
    if total <= 0.0:
        msg = "'pressure' has no energy inside the requested bands."
        raise ValueError(msg)
    return StrikeSelSpectrum(
        frequencies=fc,
        band_sel=np.asarray(band_sel, dtype=np.float64),
        total_sel=float(10.0 * np.log10(total / e0)),
        broadband_sel=sound_exposure_level(sig, fs_v),
        fraction=int(fraction),
        fs=fs_v,
    )


def pile_strike_metrics(
    pressure: SignalInput, fs: float | None = None
) -> PileStrikeResult:
    """Full per-strike pile-driving metrics (ISO 18406).

    Bundles the single-strike SEL, the peak sound pressure level, the SPL/Leq
    and the 90 %-energy pulse duration of one recorded hammer strike.

    :param pressure: Sound-pressure time series of one strike (1-D), in
        Pa. Accepts a
        :class:`phonometry.io.Signal`, whose calibration is applied to the
        samples: this quantity is a pressure, and the underwater reference
        of 1 uPa changes what the decibel is counted from, not what the
        samples have to be in.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :return: A :class:`PileStrikeResult`.
    :raises ValueError: If the inputs are invalid.
    """
    fs = resolve_fs(pressure, fs, name="pressure")
    sig = _validate_pressure(pressure, min_samples=2)
    fs_v = _positive(fs, "fs")
    return PileStrikeResult(
        single_strike_sel=sound_exposure_level(sig, fs_v),
        peak_spl=peak_sound_pressure_level(sig),
        spl=sound_pressure_level(sig),
        pulse_duration=_pulse_duration(sig, fs_v),
        pressure=sig,
        fs=fs_v,
    )
