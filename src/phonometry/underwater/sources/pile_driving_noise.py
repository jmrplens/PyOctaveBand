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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._internal.levels_math import energy_sum
from ..._internal.validation import (
    require_finite_fields,
    require_ranks,
    require_same_length,
)
from ...io._resolve import SignalInput, resolve_fs
from ..acoustics import (
    UNDERWATER_REFERENCE_PRESSURE,
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


def _peak_level(pressure: NDArray[np.float64]) -> float:
    r"""Zero-to-peak level of *pressure*, in dB re 1 µPa.

    :func:`~phonometry.underwater.acoustics.peak_sound_pressure_level` is the
    measurement and refuses a silent record outright, which is right there and
    wrong in a guard: what a guard has to tell apart is a level that restates
    its trace from one that contradicts it, and an exception raised about the
    trace answers neither. A silent trace peaks at ``-inf``, the level of zero
    pressure -- the same neutral value :attr:`StrikeSelSpectrum.band_sel`
    carries for a band that holds no energy, so the two guards in this module
    treat an undetermined level the same way.

    :param pressure: The strike waveform, in Pa (non-empty).
    :return: :math:`20 \log_{10}(\max\lvert p \rvert / 1\,\mu\mathrm{Pa})`,
        ``-inf`` for an all-zero trace.
    """
    peak = float(np.max(np.abs(np.asarray(pressure, dtype=np.float64))))
    with np.errstate(divide="ignore"):
        return float(20.0 * np.log10(peak / UNDERWATER_REFERENCE_PRESSURE))


def _band_energy_sum(band_sel: NDArray[np.float64]) -> float:
    r"""Energy sum of the band levels *band_sel*, in dB re 1 µPa²·s.

    :math:`10 \log_{10} \sum_b 10^{\mathrm{SEL}_b/10}`, the only sum a column
    of decibels has: adding the decibels themselves totals a strike at some
    thousands of dB. A band at ``-inf`` contributes an exact zero, so a
    spectrum with an empty band sums as though the band were not there; a
    spectrum with *every* band empty sums to ``-inf``, the level of no
    exposure at all, which is the truthful total over such a column.

    :param band_sel: Per-band single-strike SELs, in dB re 1 µPa²·s.
    :return: Their energy sum, in dB re 1 µPa²·s.
    """
    with np.errstate(divide="ignore"):
        return energy_sum(np.asarray(band_sel, dtype=np.float64))


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

    def __post_init__(self) -> None:
        """Reject a strike whose peak level is not the peak of its own trace.

        :attr:`peak_spl` is not a measurement of its own: ISO 18406 6.4.2.1.3
        takes it off the very waveform stored here, and
        :func:`pile_strike_metrics` computes it as
        :func:`~phonometry.underwater.acoustics.peak_sound_pressure_level` of
        the trace it goes on to store beside it. The two cannot part company
        except by hand or by :func:`dataclasses.replace`, which is how a
        variant of a frozen result is built -- substitute a filtered or
        de-spiked waveform and the peak level that travels through with it is
        the old trace's.

        Parted, they are still drawn together. The figure marks the sample at
        ``argmax(|pressure|)`` and labels that marker with ``peak_spl``, so a
        peak level from another trace prints itself over the one sample that
        disproves it. The number is also the one half of the marine-mammal
        dual-metric rule that is compared unweighted, against a fixed
        peak-pressure injury threshold: a strike that clears the threshold on
        its own waveform is failed against it, or the reverse, with the
        waveform beneath saying otherwise.

        The comparison allows a billionth of a decibel so a caller who
        recomputed the peak along another floating-point path is not refused
        over the last bit.

        :raises ValueError: if ``pressure`` is not a non-empty, finite,
            one-dimensional trace, or ``peak_spl`` is not its zero-to-peak
            level.
        """
        trace = np.asarray(self.pressure)
        # The rank helper waives its pin when every field it was given is a
        # bare number, an exemption meant for the entry points that answer in
        # scalars. Here 'pressure' is the only field listed, so a lone number
        # satisfies the whole set and walks past it, with a size of one and a
        # peak of its own that the comparison below is happy to confirm. A
        # trace of one sample is a trace; a number is not one, so the rank is
        # pinned here in its own right.
        if trace.ndim != 1:
            msg = (
                "PileStrikeResult: 'pressure' must be a one-dimensional "
                f"waveform; got shape {trace.shape}."
            )
            raise ValueError(msg)
        require_ranks(self, pressure=1)
        if trace.size == 0:
            msg = (
                "PileStrikeResult: 'pressure' must carry at least one sample; "
                "a strike with no waveform has no peak to state."
            )
            raise ValueError(msg)
        require_finite_fields(self, "pressure")
        peak = _peak_level(self.pressure)
        if not math.isclose(self.peak_spl, peak, rel_tol=0.0, abs_tol=1e-9):
            msg = (
                "PileStrikeResult: 'peak_spl' must be the zero-to-peak level of "
                "'pressure', 20 lg(max|p| / 1 uPa), the trace it summarises; "
                f"got {self.peak_spl!r} where the trace peaks at {peak!r} dB "
                "re 1 uPa."
            )
            raise ValueError(msg)

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

    def __post_init__(self) -> None:
        r"""Reject a band spectrum whose levels do not match its band centres.

        The two arrays state one measurement twice over -- a nominal centre
        frequency, and the exposure the band at that frequency carries -- and
        only position pairs them. :attr:`total_sel` is then a stored number
        claiming to be the energy sum of exactly those bands, which no reader
        can re-derive: a ``band_sel`` longer than ``frequencies`` keeps a
        total that counted a band the centre list never reaches, and one
        shorter keeps a total missing a band the table will print.

        The pairing is also what the assessment downstream consumes.
        :func:`~phonometry.underwater.bioacoustics.weighting.weighted_exposure`
        takes both arrays and weights each band by the hearing group's
        sensitivity at whatever centre landed opposite it; those weighting
        curves fall by tens of decibels across a decade, so a misaligned
        spectrum returns an injury margin that reads like any other.

        :attr:`total_sel` is then held to the column it totals. It is the
        *energy* sum of ``band_sel``, :math:`10 \log_{10} \sum_b
        10^{\mathrm{SEL}_b/10}`, which is what :func:`strike_sel_spectrum`
        computes by summing the band energies before taking the logarithm
        once; the arithmetic sum of the same decibels totals a strike at some
        thousands of dB and is simply a different number. The figure rules a
        dashed line across the axes at ``total_sel`` and labels it
        :math:`\mathrm{SEL}_{\mathrm{ss}}`, so a total kept from another
        column -- a mitigated spectrum substituted through
        :func:`dataclasses.replace` is exactly that -- is drawn as a broadband
        level the bands beneath it never reach.

        A band at ``-inf`` is admitted, and only ``-inf``: it is the level of
        zero exposure this class documents for a band narrower than the FFT
        bin spacing, and it is the neutral element of the energy sum, so it
        passes through the total without disturbing it. Nothing else
        non-finite is a level, and a ``NaN`` band would otherwise be reported
        as a wrong total rather than as the wrong band it is.

        :attr:`broadband_sel` is left unpinned deliberately. It is the SEL of
        the whole record, :math:`\int p^2 dt` over the waveform, and the
        waveform is not stored here; it exceeds ``total_sel`` by exactly the
        energy falling outside ``limits``, which nothing in hand can measure.

        The total's comparison allows a billionth of a decibel so a caller who
        re-summed the bands along another floating-point path is not refused
        over the last bit.

        :raises ValueError: if the band levels disagree with the centres, if a
            band is non-finite otherwise than ``-inf``, or if ``total_sel`` is
            not the energy sum of ``band_sel``.
        """
        require_ranks(self, frequencies=1, band_sel=1)
        require_same_length(self, "frequencies", "band_sel")
        band_sel = np.asarray(self.band_sel, dtype=np.float64)
        undetermined = np.flatnonzero(~(np.isfinite(band_sel) | np.isneginf(band_sel)))
        if undetermined.size:
            i = int(undetermined[0])
            msg = (
                "StrikeSelSpectrum: 'band_sel' must be finite, or -inf for a "
                "band that holds no energy at all; the "
                f"{float(np.asarray(self.frequencies)[i]):g} Hz band states "
                f"{float(band_sel[i])!r}."
            )
            raise ValueError(msg)
        total = _band_energy_sum(band_sel)
        if not math.isclose(self.total_sel, total, rel_tol=0.0, abs_tol=1e-9):
            msg = (
                "StrikeSelSpectrum: 'total_sel' must be the energy sum of "
                "'band_sel', 10 lg(sum 10^(SEL/10)) over the bands beside it "
                f"and not their arithmetic sum; got {self.total_sel!r} where "
                f"the bands sum to {total!r}."
            )
            raise ValueError(msg)

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
    # The value the caller passed, not its truncation: int() would silently
    # turn 3.9 into one-third octaves and 1.4 into octaves.
    if fraction not in (1, 3):
        msg = "'fraction' must be 1 (octave) or 3 (one-third octave)."
        raise ValueError(msg)
    # Unpacking rejects the short, the over-long and the non-iterable in one
    # move, where indexing would escape as an IndexError or TypeError that
    # names nothing and a third element would be silently ignored.
    try:
        lo, hi = (float(v) for v in limits)
    except (TypeError, ValueError):
        msg = "'limits' must be a (lower, upper) pair of frequencies in Hz."
        raise ValueError(msg) from None
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
