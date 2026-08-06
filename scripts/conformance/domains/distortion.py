#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Electroacoustics: distortion & frequency response (IEC 60268-3 / Bendat).

Transducer and amplifier metrics measured on synthesized signals whose
distortion is known by construction: total harmonic distortion and its
variants, intermodulation distortion (IEC 60268-3 and the difference-frequency
method), the IEC 60268-4/-5 microphone and loudspeaker quantities, the
ITU-R BS.468-4 weighted noise measurement and the AES17 conventions.

The oracle for a distortion figure is the polynomial that produced it: a
signal built as ``x + a2 x^2 + a3 x^3`` has harmonic amplitudes given in closed
form, so the expected value is arithmetic rather than a published number.
"""

from __future__ import annotations

import math

import numpy as np
import reference_data as ref
from scipy import signal as sg

import phonometry as ph

from ..registry import Outcome, numeric, register

_ELECTRO = "Electroacoustics: distortion & frequency response"


def _electro_fs() -> int:
    return 48000


def _electro_tone(t: np.ndarray, freq: float, amp: float) -> np.ndarray:
    """A single sine of amplitude ``amp`` at ``freq`` over the time base ``t``."""
    return amp * np.sin(2.0 * np.pi * freq * t)


def _electro_harmonic_signal() -> np.ndarray:
    fs = _electro_fs()
    t = np.arange(fs) / fs  # 1 s, tones on integer bins
    a1, a2, a3, a4 = ref.DISTORTION_HARMONICS
    sig = (
        _electro_tone(t, 1000.0, a1)
        + _electro_tone(t, 2000.0, a2)
        + _electro_tone(t, 3000.0, a3)
        + _electro_tone(t, 4000.0, a4)
    )
    return np.asarray(sig, dtype=np.float64)


@register(
    _ELECTRO,
    "IEC 60268-3:2013 (14.12.3.2)",
    "THD (rel. total RMS, the R convention the clause defines)",
)
def _chk_thd_r() -> Outcome:
    value = ph.thd(_electro_harmonic_signal(), _electro_fs(), 1000.0, kind="R")
    return numeric(ref.DISTORTION_THD_R, value, 1e-4, places=6)


@register(
    _ELECTRO,
    "Closed-form harmonic synthesis (THD_F convention)",
    "THD (rel. fundamental, the widespread datasheet convention)",
)
def _chk_thd_f() -> Outcome:
    value = ph.thd(_electro_harmonic_signal(), _electro_fs(), 1000.0, kind="F")
    return numeric(ref.DISTORTION_THD_F, value, 1e-4, places=6)


def _loudspeaker_flat_response() -> tuple[np.ndarray, np.ndarray]:
    """Flat 90 dB on-axis response with ramps crossing 80 dB at 50/18000 Hz."""
    f = np.geomspace(20.0, 20000.0, 400)
    spl = np.full_like(f, 90.0)
    below = f < 80.0
    spl[below] = 90.0 - 10.0 * (np.log2(80.0 / f[below]) / np.log2(80.0 / 50.0))
    above = f > 15000.0
    spl[above] = 90.0 - 10.0 * (np.log2(f[above] / 15000.0) / np.log2(18000.0 / 15000.0))
    return f, spl


@register(
    _ELECTRO,
    "IEC 60268-5:2003 (20.3/20.4)",
    "Characteristic sensitivity level, 1 W into 8 ohm at 1 m (flat 90 dB)",
)
def _chk_loudspeaker_sensitivity() -> Outcome:
    f, spl = _loudspeaker_flat_response()
    result = ph.loudspeaker_characteristics(f, spl, 8.0, sensitivity_band=(200.0, 4000.0))
    # A flat 90 dB response driven at sqrt(8) V (1 W) at 1 m has a
    # characteristic sensitivity level of 90 dB exactly (the corrections vanish).
    return numeric(90.0, result.sensitivity_level_db, 1e-6, unit="dB", places=6)


@register(
    _ELECTRO,
    "IEC 60268-5:2003 (21.2)",
    "Effective frequency range = -10 dB crossings (50 Hz / 18 kHz)",
)
def _chk_loudspeaker_effective_range() -> Outcome:
    f, spl = _loudspeaker_flat_response()
    result = ph.loudspeaker_characteristics(f, spl, 8.0, sensitivity_band=(200.0, 4000.0))
    lo, hi = result.effective_range
    ok = abs(lo - 50.0) <= 5e-3 and abs(hi - 18000.0) <= 2.0
    return Outcome(
        expected="50 Hz / 18000 Hz (ref -10 dB crossings)",
        computed=f"{lo:.3f} Hz / {hi:.1f} Hz",
        delta=f"{lo - 50.0:.3f} / {hi - 18000.0:.3f} Hz",
        passed=ok,
    )


@register(
    _ELECTRO,
    "IEC 60268-3:2013 (14.12.5)",
    "2nd-order harmonic distortion d2 (rel. total)",
)
def _chk_harmonic_d2() -> Outcome:
    value = ph.harmonic_distortion(_electro_harmonic_signal(), _electro_fs(), 1000.0, 2)
    return numeric(ref.DISTORTION_D2, value, 1e-4, places=6)


def _microphone_flat_response() -> tuple[np.ndarray, np.ndarray]:
    """Flat 0 dB relative response with ramps crossing -3 dB at 40/18000 Hz."""
    f = np.geomspace(20.0, 20000.0, 400)
    rel = np.zeros_like(f)
    below = f < 63.0
    rel[below] = -3.0 * (np.log2(63.0 / f[below]) / np.log2(63.0 / 40.0))
    above = f > 15000.0
    rel[above] = -3.0 * (np.log2(f[above] / 15000.0) / np.log2(18000.0 / 15000.0))
    return f, rel


@register(
    _ELECTRO,
    "IEC 60268-4:2014 (11.1/11.3)",
    "Microphone sensitivity level, 12.5 mV/Pa -> 20 lg 0.0125 dB re 1 V/Pa",
)
def _chk_microphone_sensitivity_level() -> Outcome:
    f, rel = _microphone_flat_response()
    result = ph.microphone_characteristics(f, rel, 12.5, tolerance_db=3.0)
    # Hand-computed: 20 lg 0.0125 = -38.061800 dB re 1 V/Pa.
    return numeric(-38.061800, result.sensitivity_level_db, 1e-5, unit="dB", places=6)


@register(
    _ELECTRO,
    "IEC 60268-4:2014 (12.2)",
    "Effective frequency range = +/-3 dB tolerance crossings (40 Hz / 18 kHz)",
)
def _chk_microphone_effective_range() -> Outcome:
    f, rel = _microphone_flat_response()
    result = ph.microphone_characteristics(f, rel, 12.5, tolerance_db=3.0)
    lo, hi = result.effective_range
    ok = abs(lo - 40.0) <= 5e-3 and abs(hi - 18000.0) <= 2.0
    return Outcome(
        expected="40 Hz / 18000 Hz (+/-3 dB tolerance crossings)",
        computed=f"{lo:.3f} Hz / {hi:.1f} Hz",
        delta=f"{lo - 40.0:.3f} / {hi - 18000.0:.3f} Hz",
        passed=ok,
    )


@register(
    _ELECTRO,
    "IEC 60268-4:2014 (13.2.2)",
    "Directivity index of the ideal cardioid, 10 lg 3 dB (11.2.2 a integral)",
)
def _chk_microphone_cardioid_di() -> Outcome:
    f, rel = _microphone_flat_response()
    angles = np.linspace(0.0, 179.9, 1800)
    pattern = 20.0 * np.log10((1.0 + np.cos(np.radians(angles))) / 2.0)
    result = ph.microphone_characteristics(
        f, rel, 12.5, tolerance_db=3.0,
        directivity=ph.MicrophoneDirectivity(polar=(angles, pattern)),
    )
    di = result.directivity_index_db
    if di is None:
        return Outcome(
            expected="4.771213 dB", computed="None", delta="n/a", passed=False
        )
    # Closed form: D = 10 lg 3 = 4.771213 dB.
    return numeric(4.771213, di, 5e-3, unit="dB", places=6)


@register(
    _ELECTRO,
    "IEC 60268-4:2014 (17.2)",
    "Equivalent noise level, 2.5 uV over 12.5 mV/Pa -> 200 uPa = 20 dB SPL",
)
def _chk_microphone_equivalent_noise() -> Outcome:
    f, rel = _microphone_flat_response()
    result = ph.microphone_characteristics(
        f, rel, 12.5, tolerance_db=3.0, noise=ph.MicrophoneNoise(voltage=2.5e-6)
    )
    noise = result.equivalent_noise_level_db
    if noise is None:
        return Outcome(
            expected="20 dB SPL", computed="None", delta="n/a", passed=False
        )
    return numeric(20.0, noise, 1e-9, unit="dB SPL", places=6)


def _electro_smpte_signal() -> tuple[np.ndarray, float, float]:
    fs = _electro_fs()
    t = np.arange(fs) / fs
    fl, fh = 250.0, 8000.0
    x = (
        _electro_tone(t, fl, 1.0)
        + _electro_tone(t, fh, 0.25)
        + _electro_tone(t, fh + fl, 0.02)
        + _electro_tone(t, fh - fl, 0.02)
        + _electro_tone(t, fh + 2 * fl, 0.01)
        + _electro_tone(t, fh - 2 * fl, 0.01)
    )
    return x, fl, fh


@register(
    _ELECTRO,
    "IEC 60268-3:2013 (14.12.7.2 g)",
    "Modulation distortion d_m,2 (arithmetic sideband sum over U_2,f2)",
)
def _chk_modulation_d2() -> Outcome:
    x, fl, fh = _electro_smpte_signal()
    # Sidebands 0.02 + 0.02 over the 0.25 carrier: d_m,2 = 0.16 exactly.
    value = ph.modulation_distortion(x, _electro_fs(), fl, fh).d2
    return numeric(0.16, value, 1e-4, places=6)


@register(
    _ELECTRO,
    "IEC 60268-3:2013 (14.12.7.2 h)",
    "Modulation distortion d_m,3 (arithmetic sideband sum over U_2,f2)",
)
def _chk_modulation_d3() -> Outcome:
    x, fl, fh = _electro_smpte_signal()
    # Sidebands 0.01 + 0.01 over the 0.25 carrier: d_m,3 = 0.08 exactly.
    value = ph.modulation_distortion(x, _electro_fs(), fl, fh).d3
    return numeric(0.08, value, 1e-4, places=6)


def _electro_dfd_signal() -> tuple[np.ndarray, float, float]:
    fs = _electro_fs()
    t = np.arange(fs) / fs
    f1, f2 = 13000.0, 14000.0
    x = (
        _electro_tone(t, f1, 0.5)
        + _electro_tone(t, f2, 0.5)
        + _electro_tone(t, f2 - f1, 0.03)
        + _electro_tone(t, 2 * f1 - f2, 0.02)
        + _electro_tone(t, 2 * f2 - f1, 0.02)
    )
    return x, f1, f2


@register(
    _ELECTRO,
    "IEC 60268-3:2013 (14.12.8.1 a)",
    "Difference-frequency distortion d_d,2 (over U_2,ref = 2 U_2,f2)",
)
def _chk_dfd_d2() -> Outcome:
    x, f1, f2 = _electro_dfd_signal()
    # Product 0.03 over the tone-amplitude sum 1.0: d_d,2 = 0.03 exactly.
    value = ph.difference_frequency_distortion(x, _electro_fs(), f1, f2, order=2)
    return numeric(0.03, value, 1e-4, places=6)


@register(
    _ELECTRO,
    "IEC 60268-3:2013 (14.12.8.1 b)",
    "Difference-frequency distortion d_d,3 (arithmetic product sum)",
)
def _chk_dfd_d3() -> Outcome:
    x, f1, f2 = _electro_dfd_signal()
    # Products 0.02 + 0.02 over the tone-amplitude sum 1.0: d_d,3 = 0.04.
    value = ph.difference_frequency_distortion(x, _electro_fs(), f1, f2, order=3)
    return numeric(0.04, value, 1e-4, places=6)


@register(
    _ELECTRO,
    "IEC 60268-3:2013 (14.12.10)",
    "Total difference-frequency distortion (8 kHz / 11.95 kHz tones)",
)
def _chk_tdfd() -> Outcome:
    fs = _electro_fs()
    t = np.arange(fs) / fs
    f1, f2 = 8000.0, 11950.0
    x = (
        _electro_tone(t, f1, 0.5)
        + _electro_tone(t, f2, 0.5)
        + _electro_tone(t, f2 - f1, 0.02)
        + _electro_tone(t, 2 * f1 - f2, 0.03)
    )
    # Only the in-band products at f0 -/+ delta (3950/4050 Hz) enter:
    # d_TDFD = sqrt(0.02^2 + 0.03^2) / (0.5 + 0.5) = sqrt(0.0013).
    value = ph.total_difference_frequency_distortion(x, fs)
    return numeric(0.03605551275463989, value, 1e-4, places=8)


@register(
    _ELECTRO,
    "ITU-R BS.468-4 Table 1",
    "Weighting network response at the 6.3 kHz peak (14.12.11 network)",
)
def _chk_itu_468_peak() -> Outcome:
    value = float(ph.itu_r_468_weighting([6300.0])[0])
    return numeric(12.2, value, 1e-9, unit="dB", places=2)


@register(
    _ELECTRO,
    "IEC 60268-3:2013 (14.12.9)",
    "DIM of the 15 kHz / 3.15 kHz signal (Table 2, 9 products)",
)
def _chk_dim() -> Outcome:
    fs = _electro_fs()
    t = np.arange(fs) / fs
    fsine, fsq = 15000.0, 3150.0
    comps = sorted(
        round(abs(k * fsq - fsine), 6) for k in range(1, 10) if abs(k * fsq - fsine) < fsine
    )
    amps = [0.01 * (i + 1) for i in range(len(comps))]
    # 15 kHz sine + the strong 3.15 kHz fundamental + the nine products.
    x = _electro_tone(t, fsine, 1.0) + _electro_tone(t, fsq, 0.8)
    for c, a in zip(comps, amps):
        x = x + _electro_tone(t, c, a)
    expected = math.sqrt(sum(a**2 for a in amps))
    return numeric(expected, ph.dynamic_intermodulation_distortion(x, fs), 1e-4, places=6)


@register(
    _ELECTRO,
    "Bendat & Piersol, Random Data 4e",
    "H1 recovers a known first-order IIR gain at 1 kHz",
)
def _chk_h1_gain() -> Outcome:
    fs = _electro_fs()
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200000)
    b, a = sg.butter(1, 2000.0 / (fs / 2.0), btype="low")
    y = sg.lfilter(b, a, x)
    res = ph.transfer_function(x, y, fs, estimator="H1")
    _, h = sg.freqz(b, a, worN=res.frequencies, fs=fs)
    idx = int(np.argmin(np.abs(res.frequencies - 1000.0)))
    return numeric(
        float(np.abs(h[idx])), float(np.abs(res.response[idx])), 0.02, rel=True, places=4
    )


@register(
    _ELECTRO,
    "Bendat & Piersol, Random Data 4e",
    "Ordinary coherence = 1 for a noiseless LTI path",
)
def _chk_coherence_unity() -> Outcome:
    fs = _electro_fs()
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200000)
    b, a = sg.butter(1, 2000.0 / (fs / 2.0), btype="low")
    y = sg.lfilter(b, a, x)
    f, g = ph.coherence(x, y, fs)
    band = (f > 100.0) & (f < 5000.0)
    return numeric(1.0, float(np.mean(g[band])), 1e-3, places=6)


@register(
    _ELECTRO,
    "AES17-2015 (6.4.2 / 5.2.7)",
    "Idle channel noise, 1 kHz -20 dBFS tone (CCIR-RMS -5.63 dB offset)",
)
def _chk_aes17_idle_noise() -> Outcome:
    fs = _electro_fs()
    t = np.arange(fs) / fs
    # 468 is 0 dB at 1 kHz, so CCIR-RMS reads -5.63 dB there: a -20 dBFS tone
    # measures -25.63 dBFS CCIR-RMS in closed form.
    sig = _electro_tone(t, 1000.0, 10.0 ** (-20.0 / 20.0))
    return numeric(-25.63, ph.idle_channel_noise(sig, fs), 1e-2, unit="dB", places=2)


@register(
    _ELECTRO,
    "AES17-2015 (6.4.1)",
    "Dynamic range, full-scale reference over a -40 dBFS residual at 2 kHz",
)
def _chk_aes17_dynamic_range() -> Outcome:
    fs = _electro_fs()
    t = np.arange(fs) / fs
    # 997 Hz test tone at -60 dBFS plus a lone 2 kHz residual at -40 dBFS: the
    # CCIR-RMS filter is unity at 2 kHz and the 997 Hz notch is negligible
    # there, so the ratio of the full-scale sine to the residual is ~40 dB
    # (a small notch lift aside).
    sig = _electro_tone(t, 997.0, 10.0 ** (-60.0 / 20.0))
    sig = sig + _electro_tone(t, 2000.0, 10.0 ** (-40.0 / 20.0))
    return numeric(40.0, ph.dynamic_range(sig, fs, 997.0), 0.6, unit="dB", places=2)
