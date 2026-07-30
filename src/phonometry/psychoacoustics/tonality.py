#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Prominent discrete tone assessment per ECMA-418-1:2024 (3rd edition).

Implements the tone-to-noise ratio (TNR, clause 11) and prominence ratio
(PR, clause 12) methods on Hann-windowed, RMS-averaged FFT spectra
(clauses 11.1 / 12.1), with the clause 10 critical-band model.

The ``prominent`` verdict is the **numeric criterion only** (Formulae
(12)/(13) resp. (25)/(26)). The standard additionally requires prominent
tones to be confirmed by aural examination (clauses 11.8/12.8) and to pass
the clause 8/9 lower-threshold-of-hearing screen (its Formula (1), which
needs calibrated absolute levels); both audibility requirements are the
caller's responsibility.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.utils import _typesignal
from .._internal.warnings import PhonometryWarning
from ..metrology.spectra import _welch_autospectrum

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class TonalityWarning(PhonometryWarning):
    """Warns about biased tonality estimates (e.g. coarse FFT resolution)."""


def _warn_coarse_resolution(dfc: float, df: float, ft: float) -> None:
    """Warn when the FFT resolution is too coarse to resolve the tone band.

    The tone band is 15 % of the critical bandwidth ``dfc`` (half-width
    ``0.075*dfc``). Below ~3 bins across the half-width the tone/noise split
    is biased (ECMA-418-1 keeps the tone band within 15 % of the critical
    bandwidth). At 250 Hz a 4 Hz bin already costs ~0.3 dB of TNR."""
    bins = 0.075 * dfc / df
    if bins < 3.0:
        warnings.warn(
            f"Frequency resolution {df:.3g} Hz is coarse for the tone at "
            f"{ft:.0f} Hz: the tone band spans only {bins:.1f} bins (< 3), "
            "which biases TNR/PR. Use a finer resolution_hz.",
            TonalityWarning,
            stacklevel=3,
        )

# Frequency range of interest for discrete tones (clauses 11.5 / 12.6).
# NOTE: clause 4.1.2 prints the range as "89,1 Hz and 11 220 Hz inclusive",
# but every formula and table span of the standard uses 11 200 Hz (Tables
# 2/3 end at 11 200; Formulae (13)/(26) treat the upper end as exclusive).
# The printed 11 220 Hz is a standard-side typo (see docs/ERRATA.md); the
# consistent 89.1 Hz - 11 200 Hz reading is used here.
_F_MIN = 89.1
_F_MAX = 11200.0

#: Band power (Pa^2-like units) below which a spectrum is numeric noise: a
#: silence or DC input yields formally finite TNR/PR values that carry no
#: meaning, so a warning is raised (about -100 dB SPL over a critical band).
_NOISE_FLOOR_POWER = 4e-20


def _warn_degenerate(ft: float, band_power: float, df: float) -> None:
    """Warn on verdicts that are numerically valid but perceptually void.

    * A peak within one bin of the 89.1 Hz / 11.2 kHz range edge can snap to
      the wrong side at coarse resolution (a tone at exactly 89.1 Hz lands on
      an 89.0 Hz bin at 1 Hz resolution and is reported non-prominent
      regardless of its ratio).
    * A critical band whose power sits at numeric-noise level (silence, DC)
      produces a meaningless ratio.
    """
    if abs(ft - _F_MIN) <= df or abs(ft - _F_MAX) <= df:
        warnings.warn(
            f"The assessed peak at {ft:.1f} Hz lies within one bin of the "
            "89.1 Hz - 11.2 kHz range-of-interest edge; the prominence "
            "verdict can flip with the bin snapping. Use a finer "
            "resolution_hz or check the tone frequency.",
            TonalityWarning,
            stacklevel=3,
        )
    if band_power < _NOISE_FLOOR_POWER:
        warnings.warn(
            "The critical-band power is at numeric-noise level (silence or "
            "DC input); the TNR/PR value is meaningless.",
            TonalityWarning,
            stacklevel=3,
        )


@dataclass(frozen=True)
class ToneAssessment:
    """Result of a discrete-tone prominence assessment.

    ``ratio_db`` is the tone-to-noise ratio or the prominence ratio in
    decibels depending on the producing function; ``criterion_db`` is the
    prominence limit at ``frequency`` and ``prominent`` the verdict.

    ``prominent`` applies the numeric criterion only; the standard's
    audibility requirements (aural examination per clauses 11.8/12.8 and
    the clause 8/9 lower-threshold-of-hearing screen, which needs
    calibrated absolute levels) are the caller's responsibility.
    """

    frequency: float
    ratio_db: float
    criterion_db: float
    prominent: bool

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the assessed tone against the ECMA-418-1 prominence criterion.

        Draws the frequency-dependent criterion curve over the 89.1 Hz -
        11.2 kHz range of interest with this tone at its own frequency and
        the margin to the criterion marked; a tone on or above the curve is
        prominent. The criterion family (tone-to-noise ratio, clause 11, or
        prominence ratio, clause 12) is recovered from ``criterion_db``.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :param ax: Existing axes to draw on, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the assessed-tone marker ``plot`` call.
        :return: The axes.
        """
        from .._i18n import check_language
        from .._plot.psychoacoustics import plot_tone_assessment

        return plot_tone_assessment(
            self, ax=ax, language=check_language(language), **kwargs
        )


def _critical_band(f0: float) -> tuple[float, float, float]:
    """Clause 10: critical bandwidth and band-edge frequencies at ``f0``.

    Formula (2): dfc = 25,0 + 75,0*(1,0 + 1,4*(f0/1000)^2)^0,69.
    Edges: Formulae (4)-(5) (arithmetic) for f0 <= 500 Hz and (7)-(8)
    (geometric) above.
    """
    dfc = 25.0 + 75.0 * (1.0 + 1.4 * (f0 / 1000.0) ** 2) ** 0.69
    if f0 <= 500.0:
        f1 = f0 - dfc / 2.0
        f2 = f0 + dfc / 2.0
    else:
        f1 = -dfc / 2.0 + np.sqrt(dfc**2 + 4.0 * f0**2) / 2.0
        f2 = f1 + dfc
    return f1, f2, dfc


def _proximity_spacing(fp: float) -> float:
    """Clause 11.6 Formula (14): proximity spacing for multiple tones.

    For fp >= 1000 Hz the spacing always exceeds half the critical band,
    so proximity is always met (the formula only applies below 1 kHz).
    """
    return float(21.0 * 10 ** (1.2 * abs(np.log10(fp / 212.0)) ** 1.8))


def _averaged_spectrum(
    x: np.ndarray, fs: int, resolution_hz: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Hann-windowed, RMS-averaged power spectrum (clauses 11.1 / 12.1).

    Returns (frequencies, bin powers in Pa^2-like units, bin spacing).
    """
    nperseg = round(fs / resolution_hz)
    if nperseg < 16:
        raise ValueError(
            f"resolution_hz={resolution_hz!r} is too coarse for fs={fs}: it "
            "leaves fewer than 16 samples per FFT segment."
        )
    if x.shape[-1] < nperseg:
        raise ValueError(
            f"Signal too short for the requested {resolution_hz} Hz "
            f"resolution: need at least {nperseg} samples, got {x.shape[-1]}."
        )
    freqs, psd = _welch_autospectrum(x, float(fs), nperseg, nperseg // 2)
    df = float(freqs[1] - freqs[0])
    return freqs, psd * df, df


def _find_peak(freqs: np.ndarray, power: np.ndarray, tone_freq: float | None) -> int:
    if tone_freq is None:
        in_range = (freqs >= _F_MIN) & (freqs <= _F_MAX)
        if not np.any(in_range):
            raise ValueError("Spectrum has no bins in the 89.1 Hz - 11.2 kHz range.")
        return int(np.flatnonzero(in_range)[np.argmax(power[in_range])])
    _, _, dfc = _critical_band(tone_freq)
    near = np.abs(freqs - tone_freq) <= dfc / 2.0
    if not np.any(near):
        raise ValueError(f"No spectrum bins near tone_freq={tone_freq!r} Hz.")
    return int(np.flatnonzero(near)[np.argmax(power[near])])


def _tone_band(power: np.ndarray, peak: int, half_width_bins: int) -> tuple[int, int]:
    """Clause 11.2: tone-band edges at the minimal points on both sides of
    the peak within 15 % of the critical band; if the minimum falls on the
    window edge, the nearest local minimum beyond it is used."""

    def _edge(direction: int) -> int:
        last = power.size - 1
        limit = max(0, min(peak + direction * half_width_bins, last))
        idx = peak
        while idx != limit:
            nxt = idx + direction
            if power[nxt] >= power[idx] and idx != peak:
                return idx  # local minimum inside the window
            idx = nxt
        # Minimum sits on the window edge: continue to the nearest local
        # minimum beyond the 15 % window (clause 11.2).
        while 0 < idx < last and power[idx + direction] < power[idx]:
            idx += direction
        return idx

    return _edge(-1), _edge(+1)


def _band_power(freqs: np.ndarray, power: np.ndarray, f1: float, f2: float) -> tuple[float, int]:
    mask = (freqs > f1) & (freqs <= f2)
    n = int(np.count_nonzero(mask))
    if n == 0:
        raise ValueError(
            f"No spectrum bins fall in the {f1:.1f}-{f2:.1f} Hz band; use a "
            "finer resolution_hz."
        )
    return float(np.sum(power[mask])), n


def _tnr_criterion(ft: float) -> float:
    """Clause 11.5 Formulae (12)-(13): TNR prominence limit in dB."""
    if ft < 1000.0:
        return float(8.0 + 8.33 * np.log10(1000.0 / ft))
    return 8.0


def _pr_criterion(ft: float) -> float:
    """Clause 12.6 Formulae (25)-(26): PR prominence limit in dB."""
    if ft < 1000.0:
        return float(9.0 + 10.0 * np.log10(1000.0 / ft))
    return 9.0


def tone_to_noise_ratio(
    x: list[float] | np.ndarray,
    fs: int,
    tone_freq: float | None = None,
    resolution_hz: float = 1.0,
) -> ToneAssessment:
    """
    Tone-to-noise ratio of a discrete tone (ECMA-418-1:2024, clause 11).

    The spectrum is Hann-windowed and RMS-averaged (clause 11.1). The tone
    level ``Lt`` is the tone-band level above the line connecting the band
    edges (Formula 9); the masking-noise level ``Ln`` is the remaining
    critical-band level rescaled to the full critical bandwidth
    (Formula 10); TNR = Lt - Ln (Formula 11). Proximate secondary tones in
    the same critical band are combined per clause 11.6 (Formulae 14-16).

    :param x: Input signal (1D).
    :param fs: Sample rate in Hz.
    :param tone_freq: Approximate tone frequency in Hz. Default (None)
        assesses the highest spectral peak in the 89.1 Hz - 11.2 kHz range
        of interest. For harmonic complexes, call once per component
        (clause 11.7).
    :param resolution_hz: FFT bin spacing (default 1.0 Hz; the tone band
        must stay within 15 % of the critical bandwidth, clause 11.2).
    :return: :class:`ToneAssessment` with ``ratio_db`` = TNR in dB.
    """
    x_proc = np.atleast_1d(np.asarray(_typesignal(x), dtype=np.float64))
    if x_proc.ndim != 1:
        raise ValueError("tone_to_noise_ratio expects a 1D signal.")
    if fs <= 0:
        raise ValueError("Sample rate 'fs' must be positive.")
    if resolution_hz <= 0:
        raise ValueError("'resolution_hz' must be positive.")
    if tone_freq is not None and tone_freq <= 0:
        raise ValueError("'tone_freq' must be positive.")

    freqs, power, df = _averaged_spectrum(x_proc, fs, resolution_hz)
    peak = _find_peak(freqs, power, tone_freq)
    ft = float(freqs[peak])
    f1, f2, dfc = _critical_band(ft)
    _warn_coarse_resolution(dfc, df, ft)

    half_width_bins = max(1, round(0.075 * dfc / df))
    lo, hi = _tone_band(power, peak, half_width_bins)
    n_tone = hi - lo + 1
    # Formula (9): subtract the line connecting the band-edge levels.
    p_tone_total = float(np.sum(power[lo : hi + 1]))
    p_tone = p_tone_total - (power[lo] + power[hi]) * n_tone / 2.0
    p_tone = max(p_tone, np.finfo(float).tiny)

    # Clause 11.6: combine a proximate secondary tone in the critical band.
    p_secondary = 0.0
    in_band = (freqs > f1) & (freqs <= f2)
    band_idx = np.flatnonzero(in_band)
    outside_tone = band_idx[(band_idx < lo) | (band_idx > hi)]
    if outside_tone.size >= 3:
        rel = power[outside_tone]
        sec_local = outside_tone[np.argmax(rel)]
        # A secondary "tone" must be a local spectral peak, not noise floor:
        # require it to stand at least 6 dB above its immediate neighbours' mean.
        neigh = power[max(0, sec_local - 3)] + power[min(power.size - 1, sec_local + 3)]
        if power[sec_local] > 2.0 * neigh:
            fsec = float(freqs[sec_local])
            proximate = ft >= 1000.0 or abs(fsec - ft) < _proximity_spacing(ft)
            s_lo, s_hi = _tone_band(power, int(sec_local), half_width_bins)
            n_sec = s_hi - s_lo + 1
            p_sec = float(np.sum(power[s_lo : s_hi + 1])) - (
                power[s_lo] + power[s_hi]
            ) * n_sec / 2.0
            p_sec = max(p_sec, 0.0)
            if proximate:
                p_tone += p_sec  # Formula (15); (16) subtracts it via p_tone
            else:
                p_secondary = p_sec  # subtracted from the noise only (16)

    p_tot, n_tot = _band_power(freqs, power, f1, f2)
    _warn_degenerate(ft, p_tot, df)
    df_tot = n_tot * df
    # Formula (10)/(16): masking noise rescaled to the critical bandwidth.
    p_noise = max(p_tot - p_tone - p_secondary, np.finfo(float).tiny) * (dfc / df_tot)

    tnr = float(10 * np.log10(p_tone / p_noise))
    criterion = float(_tnr_criterion(ft))
    prominent = tnr >= criterion and _F_MIN <= ft < _F_MAX
    return ToneAssessment(ft, tnr, criterion, prominent)


# ECMA-418-1:2024 Table 2 (p. 17): f_1,L = C0 + C1*ft + C2*ft^2 (Formula 21).
# Formula (21) is printed "f_1,L = C_L,0 + C_L,0 ft + C_L,2 ft^2", repeating
# the constant term where the linear coefficient belongs; its own where-list
# declares C_L,0/C_L,1/C_L,2, Table 2 tabulates a C_L,1 column and the parallel
# Formula (22) prints C_U,1 correctly. Read as printed the middle range would
# return -149.5 - 149.5*ft - 6.90e-5*ft**2, negative everywhere. See ERRATA.
_LOWER_EDGE_COEFFS = (
    (89.1, 171.4, (20.0, 0.0, 0.0)),
    (171.4, 1600.0, (-149.5, 1.001, -6.90e-5)),
    (1600.0, 11200.0, (6.8, 0.806, -8.20e-6)),
)
# ECMA-418-1:2024 Table 3 (p. 18): f_2,U = C0 + C1*ft + C2*ft^2 (Formula 22)
_UPPER_EDGE_COEFFS = (
    (89.1, 1600.0, (149.5, 1.035, 7.70e-5)),
    (1600.0, 11200.0, (3.3, 1.215, 2.16e-5)),
)


def _fitted_edge(
    ft: float, table: tuple[tuple[float, float, tuple[float, float, float]], ...]
) -> float:
    # Peaks marginally outside the range of interest (the verdict already
    # reports them as non-prominent) are clipped to the table span.
    ft = min(max(ft, _F_MIN), _F_MAX)
    for lo, hi, (c0, c1, c2) in table:
        if lo <= ft <= hi:
            return float(c0 + c1 * ft + c2 * ft**2)
    raise AssertionError("unreachable: ft clipped to the table span")


def prominence_ratio(
    x: list[float] | np.ndarray,
    fs: int,
    tone_freq: float | None = None,
    resolution_hz: float = 1.0,
) -> ToneAssessment:
    """
    Prominence ratio of a discrete tone (ECMA-418-1:2024, clause 12).

    Compares the level of the critical band centred on the tone (``LM``)
    with the mean of the two contiguous critical bands (``LL``, ``LU``,
    band edges from the fitted Formulae 21-22 with Tables 2-3). For tones
    at or below 171.4 Hz the lower band is truncated at 20 Hz and rescaled
    to a 100 Hz bandwidth (Formula 24; Formula 23 otherwise, per the
    clause 12.5 prose - the inline condition printed next to Formula 23 in
    the PDF is a typo).

    :param x: Input signal (1D).
    :param fs: Sample rate in Hz.
    :param tone_freq: Approximate tone frequency in Hz (default: highest
        peak in the range of interest).
    :param resolution_hz: FFT bin spacing (default 1.0 Hz).
    :return: :class:`ToneAssessment` with ``ratio_db`` = PR in dB.
    """
    x_proc = np.atleast_1d(np.asarray(_typesignal(x), dtype=np.float64))
    if x_proc.ndim != 1:
        raise ValueError("prominence_ratio expects a 1D signal.")
    if fs <= 0:
        raise ValueError("Sample rate 'fs' must be positive.")
    if resolution_hz <= 0:
        raise ValueError("'resolution_hz' must be positive.")
    if tone_freq is not None and tone_freq <= 0:
        raise ValueError("'tone_freq' must be positive.")

    freqs, power, df = _averaged_spectrum(x_proc, fs, resolution_hz)
    peak = _find_peak(freqs, power, tone_freq)
    ft = float(freqs[peak])

    # Middle critical band (12.2).
    f1_m, f2_m, dfc_m = _critical_band(ft)
    _warn_coarse_resolution(dfc_m, df, ft)
    # Lower band (12.3): truncated at 20 Hz below 171.4 Hz.
    f1_l = max(_fitted_edge(ft, _LOWER_EDGE_COEFFS), 20.0) if ft <= 171.4 else _fitted_edge(ft, _LOWER_EDGE_COEFFS)
    # Upper band (12.4).
    f2_u = _fitted_edge(ft, _UPPER_EDGE_COEFFS)

    p_m, _ = _band_power(freqs, power, f1_m, f2_m)
    _warn_degenerate(ft, p_m, df)
    p_l, n_l = _band_power(freqs, power, f1_l, f1_m)
    p_u, _ = _band_power(freqs, power, f2_m, f2_u)
    p_m = max(p_m, np.finfo(float).tiny)
    p_l = max(p_l, np.finfo(float).tiny)
    p_u = max(p_u, np.finfo(float).tiny)

    if ft <= 171.4:
        # Formula (24): rescale the truncated lower band to 100 Hz.
        df_l = n_l * df
        pr = 10 * np.log10(p_m) - 10 * np.log10((100.0 / df_l * p_l + p_u) * 0.5)
    else:
        # Formula (23).
        pr = 10 * np.log10(p_m) - 10 * np.log10((p_l + p_u) * 0.5)

    criterion = float(_pr_criterion(ft))
    prominent = float(pr) >= criterion and _F_MIN <= ft < _F_MAX
    return ToneAssessment(ft, float(pr), criterion, prominent)
