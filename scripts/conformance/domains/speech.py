#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Domain 4 - Speech transmission, and the measurement chain under it.

The Speech Transmission Index of IEC 60268-16: the modulation transfer
function, the masking and reception-threshold corrections, the band
transmission indices and the weighted STI, plus the Edition 5 STIPA
verification-signal parameters restated from the standard rather than from the
implementation.

The system-measurement checks (Golay complementary pairs, Kirkeby & Nelson
regularized inversion, Mueller-Massarani shaped sweeps) sit in the same module
because they are how the impulse response an STI measurement rests on is
obtained: an error in the excitation or the deconvolution shows up as a
modulation-transfer error, and the two are only ever read together.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
import reference_data as ref
from scipy import signal as sg

import phonometry as ph
from phonometry.speech.sti import _sti_from_mtf

from ..registry import _DELTA_PLACES, Outcome, _fmt, numeric, register
from .levels import _FS

if TYPE_CHECKING:
    from phonometry.signals import InverseFilterResult

_NUM_STI_BANDS = 7
_NUM_MOD_FREQS = 14


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 A.2.2",
    "STI weighting-factor pair (500 Hz + 1 kHz bands)",
)
def _chk_sti_weighting_pair() -> Outcome:
    mtf = np.zeros((_NUM_STI_BANDS, _NUM_MOD_FREQS))
    mtf[[2, 3], :] = 1.0
    computed = float(_sti_from_mtf(mtf).sti)
    return numeric(0.398, computed, 0.001, places=4)


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 A.3.1.2",
    "Uniform MTF m=0.5 maps to STI=0.5",
)
def _chk_sti_uniform() -> Outcome:
    mtf = np.full((_NUM_STI_BANDS, _NUM_MOD_FREQS), 0.5)
    computed = float(_sti_from_mtf(mtf).sti)
    return numeric(0.5, computed, 0.01, places=4)


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16 Annex M",
    "Full-STI worked example: printed MTF + speech/noise spectra -> STI",
)
def _chk_sti_annex_m() -> Outcome:
    """End-to-end A.5.3-A.5.6 chain on the annex's own worked example."""
    mtf = np.asarray(ref.IEC60268_16_ANNEX_M_MTF, dtype=float).T
    result = _sti_from_mtf(
        mtf,
        level=np.asarray(ref.IEC60268_16_ANNEX_M_LEVEL, dtype=float),
        ambient=np.asarray(ref.IEC60268_16_ANNEX_M_AMBIENT, dtype=float),
    )
    mti_delta = float(
        np.max(
            np.abs(
                np.round(result.mti, 2)
                - np.asarray(ref.IEC60268_16_ANNEX_M_MTI, dtype=float)
            )
        )
    )
    outcome = numeric(ref.IEC60268_16_ANNEX_M_STI, float(result.sti), 0.005, places=3)
    return Outcome(
        expected=f"STI {ref.IEC60268_16_ANNEX_M_STI} (MTI row of step 4c)",
        computed=f"STI {result.sti:.3f} (max MTI dev {mti_delta:.2f})",
        delta=outcome.delta,
        passed=outcome.passed and mti_delta <= 0.01,
    )


# Ed.5 STIPA verification-signal parameters (restating the standard, not
# the implementation): Table B.1 modulation pairs, A.6.1 male spectrum.
_STI_CENTERS = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
_STI_F1 = [1.60, 1.00, 0.63, 2.00, 1.25, 0.80, 2.50]
_STI_F2 = [8.00, 5.00, 3.15, 10.00, 6.25, 4.00, 12.50]
_STI_LEVELS = [-2.5, 0.5, 0.0, -6.0, -12.0, -18.0, -24.0]
# m <-> STI staircase of the Ed.5 verification bench (m in 0,1 steps;
# TI = (10 lg(m/(1-m)) + 15)/30, rounded as published).
_STI_STAIRCASE = {
    0.0: 0.00,
    0.1: 0.18,
    0.2: 0.30,
    0.3: 0.38,
    0.4: 0.44,
    0.5: 0.50,
    0.6: 0.56,
    0.7: 0.62,
    0.8: 0.70,
    0.9: 0.82,
    1.0: 1.00,
}


def _stipa_sine_signal(
    m: float,
    seconds: float = 16.0,
    bands: tuple[int, ...] | None = None,
    edge_carriers: bool = False,
    flat_levels: bool = False,
) -> np.ndarray:
    """Ed.5 Formula (C.1)-style verification signal with sine carriers.

    A_k(t) = g_k c_k(t) sqrt(0,5 (1 + 0,55 m (sin 2 pi f1_k t -
    sin 2 pi f2_k t))). ``bands`` limits the modulation (m = 0 elsewhere,
    as in the A.2.2 pair signals); ``edge_carriers`` uses the two
    half-octave edge carriers fc 2^(+/-1/4) per band (A.3.1.2);
    ``flat_levels`` uses g_k = 1 as the A.2.2/A.3.1.2 bench does.
    """
    t = np.arange(round(seconds * _FS)) / _FS
    x = np.zeros(t.size)
    for k, (fc, fa, fb, level) in enumerate(
        zip(_STI_CENTERS, _STI_F1, _STI_F2, _STI_LEVELS, strict=True)
    ):
        mk = m if bands is None or k in bands else 0.0
        env = 0.5 * (
            1.0 + 0.55 * mk * (np.sin(2 * np.pi * fa * t) - np.sin(2 * np.pi * fb * t))
        )
        if edge_carriers:
            carrier = np.sin(2 * np.pi * fc / 2**0.25 * t) + np.sin(
                2 * np.pi * fc * 2**0.25 * t
            )
        else:
            carrier = np.sin(2 * np.pi * fc * t)
        gain = 1.0 if flat_levels else 10.0 ** (level / 20.0)
        x += gain * carrier * np.sqrt(np.maximum(env, 0.0))
    return x


def _chk_stipa_staircase(m: float) -> Outcome:
    # Ed.5 modulation-depth verification: the Formula (C.1) signal with
    # sinusoidal carriers at the A.6.1 male levels and modulation scale m
    # must read the published staircase STI through the full stipa()
    # audio path (octave bank, envelopes, correlation, TI chain). The
    # certified stipa.info WAV bench (same construction) measures a worst
    # deviation of 0,0031 STI; tolerance +/-0,01.
    computed = float(ph.speech.stipa(_stipa_sine_signal(m), _FS).sti)
    return numeric(_STI_STAIRCASE[m], computed, 0.01, places=4)


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 C.3.2",
    "STIPA direct method, Formula (C.1) signal at m=0.2",
)
def _chk_stipa_staircase_m02() -> Outcome:
    return _chk_stipa_staircase(0.2)


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 C.3.2",
    "STIPA direct method, Formula (C.1) signal at m=0.5",
)
def _chk_stipa_staircase_m05() -> Outcome:
    return _chk_stipa_staircase(0.5)


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 C.3.2",
    "STIPA direct method, Formula (C.1) signal at m=0.8",
)
def _chk_stipa_staircase_m08() -> Outcome:
    return _chk_stipa_staircase(0.8)


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 C.3.3",
    "Indirect method: exponential decay RT60=1 s vs Schroeder MTF",
)
def _chk_sti_indirect_expdecay() -> Outcome:
    # Ed.5 indirect-method verification: sine carriers in all seven bands
    # with an amplitude decay 1000^(-t/RT60) (60 dB intensity decay in
    # RT60) must reproduce the closed-form Schroeder MTF
    # m(F) = 1/sqrt(1 + (2 pi F RT60 / (6 ln 10))^2) through the octave
    # bank and Schroeder integral. Certified WAV bench worst deviation:
    # 0,0002 STI; tolerance +/-0,005.
    rt60 = 1.0
    t = np.arange(int(3.0 * _FS)) / _FS
    x = np.sum([np.sin(2 * np.pi * fc * t) for fc in _STI_CENTERS], axis=0) * 10.0 ** (
        -3.0 * t / rt60
    )
    # The 14 modulation frequencies 0,63 - 12,5 Hz (Ed.5 A.2.2).
    mod_freqs = np.array(
        [
            0.63,
            0.80,
            1.00,
            1.25,
            1.60,
            2.00,
            2.50,
            3.15,
            4.00,
            5.00,
            6.30,
            8.00,
            10.0,
            12.5,
        ]
    )
    m_formula = 1.0 / np.sqrt(
        1.0 + (2.0 * np.pi * mod_freqs * rt60 / (6.0 * np.log(10.0))) ** 2
    )
    expected = float(_sti_from_mtf(np.tile(m_formula, (_NUM_STI_BANDS, 1))).sti)
    computed = float(ph.speech.sti_from_impulse_response(x, _FS).sti)
    return numeric(expected, computed, 0.005, places=4)


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 C.4.2",
    "Filter-bank slope: +41 dB unmodulated tone one octave below 125 Hz",
)
def _chk_sti_filter_slope() -> Outcome:
    # Ed.5 filter-slope verification (worst case of the certified bench
    # pre-fix): a fully modulated 125 Hz carrier plus an unmodulated
    # 62,5 Hz tone 41 dB louder must still read m >= 0,5 in the 125 Hz
    # band, i.e. the analysis filter must be > 41 dB down one octave out
    # (steeper than IEC 61260-1 class 1; the zero-phase bank achieves
    # m >= 0,94 on all 14 certified signals).
    t = np.arange(int(16.0 * _FS)) / _FS
    env = 0.5 * (
        1.0 + 0.55 * (np.sin(2 * np.pi * 1.60 * t) - np.sin(2 * np.pi * 8.00 * t))
    )
    x = 10.0 ** (-41.0 / 20.0) * np.sin(2 * np.pi * 125.0 * t) * np.sqrt(
        np.maximum(env, 0.0)
    ) + np.sin(2 * np.pi * 62.5 * t)
    with warnings.catch_warnings():
        # Bands other than 125 Hz are legitimately (near-)empty here.
        warnings.simplefilter("ignore", UserWarning)
        m_observed = float(np.min(ph.speech.stipa(x, _FS).mtf[0]))
    return Outcome(
        expected="m >= 0.5 (C.4.2 pass criterion)",
        computed=_fmt(m_observed, places=4),
        delta=_fmt(m_observed - 0.5, "", _DELTA_PLACES),
        passed=m_observed >= 0.5,
    )


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 A.2.2 (audio path)",
    "Weighting factors: modulated 500 Hz + 1 kHz pair through stipa()",
)
def _chk_sti_weighting_pair_audio() -> Outcome:
    # Ed.5 weighting-factor verification through the full audio path:
    # sine carriers in all seven bands, only the 500/1000 Hz pair
    # modulated (m = 1) -> STI = alpha_3 + alpha_4 - beta_3 = 0,398.
    # Certified WAV bench worst deviation vs the identity: 0,0002 STI.
    x = _stipa_sine_signal(1.0, bands=(2, 3), flat_levels=True)
    computed = float(ph.speech.stipa(x, _FS).sti)
    return numeric(0.398, computed, 0.005, places=4)


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 A.3.1.2 (audio path)",
    "Filter-bank phase: half-octave edge carriers at TI=0.9",
)
def _chk_sti_edge_carriers() -> Outcome:
    # Ed.5 filter-bank phase-distortion verification: two sine carriers
    # per band at fc 2^(+/-1/4) (the extremes of the half-octave STIPA
    # generation band), modulation depth m = 0,94065 (TI = 0,9). The
    # normative criterion is |STI bias| < 0,01 over TI = 0,1 .. 0,9; the
    # zero-phase bank measures -0,0029 worst on the certified WAVs.
    x = _stipa_sine_signal(0.94065, edge_carriers=True, flat_levels=True)
    computed = float(ph.speech.stipa(x, _FS).sti)
    return numeric(0.9, computed, 0.01, places=4)


# ===========================================================================
# System measurement (Golay pairs / regularized inversion / shaped sweeps)
# ===========================================================================
_SYSMEAS = "System measurement (Golay / Kirkeby / Mueller-Massarani)"


@register(
    _SYSMEAS,
    "Havelock 2008 Part I Ch. 6 (Xiang), Eq. (2)",
    "Golay pair: sum of periodic autocorrelations = 2L*delta (L = 4096)",
)
def _chk_golay_complementary() -> Outcome:
    a, b = ph.room.golay_pair(12)
    length = a.size
    acorr = np.fft.irfft(
        np.abs(np.fft.rfft(a)) ** 2 + np.abs(np.fft.rfft(b)) ** 2, length
    )
    peak_ok = abs(acorr[0] - 2.0 * length) <= 1e-8
    sidelobe = float(np.max(np.abs(acorr[1:])))
    if not peak_ok:
        return numeric(2.0 * length, float(acorr[0]), 1e-8, places=4)
    return numeric(
        0.0,
        sidelobe,
        1e-10,
        places=6,
        expected_label="0 (algebraic identity, +/-1e-10)",
    )


@register(
    _SYSMEAS,
    "Havelock 2008 Part I Ch. 6 (Xiang), Eq. (4)",
    "Golay chain recovers a delay+gain system IR (noiseless, exact)",
)
def _chk_golay_recovery() -> Outcome:
    pair = ph.room.golay_pair(10)
    gain, delay = 0.75, 37
    res = ph.room.golay_impulse_response(
        gain * np.roll(pair[0], delay), gain * np.roll(pair[1], delay), pair
    )
    expected = np.zeros(pair[0].size)
    expected[delay] = gain
    err = float(np.max(np.abs(np.asarray(res) - expected)))
    return numeric(
        0.0, err, 1e-13, places=6, expected_label="0 (machine precision, +/-1e-13)"
    )


def _sysmeas_inverse() -> InverseFilterResult:
    b, a = sg.butter(2, [100.0, 8000.0], btype="bandpass", fs=float(_FS))
    imp = np.zeros(1024)
    imp[0] = 1.0
    return ph.signals.regularized_inverse_filter(
        sg.lfilter(b, a, imp), float(_FS), f_range=(200.0, 4000.0)
    )


@register(
    _SYSMEAS,
    "Kirkeby & Nelson 1999 Eq. (17) / Mueller-Massarani 2001 Sec. 3.1",
    "In-band equalization residue equals eps/(|H|^2 + eps) bin by bin",
)
def _chk_kirkeby_in_band() -> Outcome:
    res = _sysmeas_inverse()
    band = (res.frequencies >= 200.0) & (res.frequencies <= 4000.0)
    product = np.abs(res.response_spectrum * res.spectrum)[band]
    power = np.abs(res.response_spectrum[band]) ** 2
    residue = res.regularization[band] / (power + res.regularization[band])
    err = float(np.max(np.abs((1.0 - product) - residue)))
    return numeric(
        0.0, err, 1e-12, places=6, expected_label="0 (closed form, +/-1e-12)"
    )


@register(
    _SYSMEAS,
    "Kirkeby & Nelson 1999 (max of x/(x^2+eps) = 1/(2*sqrt(eps)))",
    "Out-of-band inverse-filter gain within the regularization cap",
)
def _chk_kirkeby_out_of_band() -> Outcome:
    res = _sysmeas_inverse()
    cap_db = -20.0 * math.log10(2.0)  # eps_outside = 1.0 -> -6.021 dB
    headroom = cap_db - res.max_gain_db
    return Outcome(
        expected=f"<= {cap_db:.3f} dB (analytic cap)",
        computed=f"{res.max_gain_db:.3f} dB",
        delta=f"headroom {headroom:+.3f} dB",
        passed=headroom >= 0.0,
    )


@register(
    _SYSMEAS,
    "Mueller-Massarani 2001 Secs. 4.2-4.3 (group-delay synthesis)",
    "Shaped sweep's Welch spectrum follows the pink target, in-band",
)
def _chk_shaped_sweep_pink() -> Outcome:
    res = ph.room.shaped_sweep_signal(_FS, 50.0, 5000.0, 2.0, target="pink")
    nperseg = 8192
    freqs, psd = sg.welch(
        np.asarray(res),
        fs=float(_FS),
        nperseg=nperseg,
        noverlap=3 * nperseg // 4,
    )
    third = 2.0 ** (1.0 / 3.0)
    band = (freqs >= 50.0 * third) & (freqs <= 5000.0 / third)
    diff = 10.0 * np.log10(psd[band]) + 10.0 * np.log10(freqs[band] / 50.0)
    dev = float(np.max(np.abs(diff - np.median(diff))))
    return numeric(
        0.0,
        dev,
        0.5,
        unit="dB",
        places=4,
        expected_label="0 dB in-band deviation (+/-0.5 dB)",
    )
