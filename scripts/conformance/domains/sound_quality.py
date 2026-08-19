#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Sound quality: prominent tones, tonal audibility and annoyance.

What a sound is like rather than how loud it is. The prominent-discrete-tone
procedure of ECMA-418-1 and the tonal audibility of ISO/PAS 20065 both decide
whether a tone stands out of its masking noise, from opposite directions;
psychoacoustic annoyance and fluctuation strength (Fastl & Zwicker) combine
loudness, sharpness, roughness and fluctuation into a single annoyance figure.

No standard publishes worked values for the Fastl & Zwicker models, so those
checks anchor on the reference conditions the textbook defines the units by
(1 vacil, 1 asper) and on the published implementations that agree with it.
"""

from __future__ import annotations

import numpy as np
import reference_data as ref

import phonometry as ph

from ..registry import Outcome, numeric, register

_TONES = "Prominent discrete tones (ECMA-418-1)"


@register(_TONES, "ECMA-418-1:2024 Clause 10 Formula (2)", "Critical band at 1 kHz (f1,c / f2,c / dfc)")
def _chk_ecma418_1_critical_band() -> Outcome:
    from phonometry.psychoacoustics.quality.tonality import _critical_band

    f1, f2, dfc = _critical_band(1000.0)
    # 0.05 Hz = half a unit in the last printed digit (the clause EXAMPLE
    # values are given to one decimal: 922,2 / 1084,4 / 162,2 Hz).
    out = numeric(ref.ECMA418_1_DFC_1KHZ, float(dfc), 0.05, unit="Hz", places=2)
    edges_ok = (
        abs(float(f1) - ref.ECMA418_1_F1_1KHZ) <= 0.05
        and abs(float(f2) - ref.ECMA418_1_F2_1KHZ) <= 0.05
    )
    return Outcome(
        expected=(
            f"dfc {out.expected}; edges {ref.ECMA418_1_F1_1KHZ:g}"
            f"-{ref.ECMA418_1_F2_1KHZ:g} Hz"
        ),
        computed=f"dfc {out.computed}; edges {float(f1):.1f}-{float(f2):.1f} Hz",
        delta=out.delta,
        passed=out.passed and edges_ok,
    )


@register(_TONES, "ECMA-418-1:2024 Clause 11.6 Formula (14)", "Proximity spacing dfprox at 150 / 850 Hz")
def _chk_ecma418_1_proximity_spacing() -> Outcome:
    from phonometry.psychoacoustics.quality.tonality import _proximity_spacing

    v150 = float(_proximity_spacing(150.0))
    v850 = float(_proximity_spacing(850.0))
    # 0.5 Hz = half a unit in the last printed digit of the coarser EXAMPLE
    # value (the standard prints 23 Hz with no decimals; 63,8 Hz with one).
    ok = (
        abs(v150 - ref.ECMA418_1_PROX_150HZ) <= 0.5
        and abs(v850 - ref.ECMA418_1_PROX_850HZ) <= 0.5
    )
    return Outcome(
        expected=(
            f"{ref.ECMA418_1_PROX_150HZ:g} Hz @ 150 Hz; "
            f"{ref.ECMA418_1_PROX_850HZ:g} Hz @ 850 Hz (+/-0.5 Hz)"
        ),
        computed=f"{v150:.1f} Hz; {v850:.1f} Hz",
        delta=(
            f"{v150 - ref.ECMA418_1_PROX_150HZ:+.3f}; "
            f"{v850 - ref.ECMA418_1_PROX_850HZ:+.3f} Hz"
        ),
        passed=ok,
    )


_TONE_AUD = "Tonal audibility (ISO/PAS 20065)"


@register(_TONE_AUD, "ISO/PAS 20065:2016 Formulae (12)-(14)", "Audibility at 137.3 Hz, Annex E spectrum 1")
def _chk_iso20065_audibility() -> Outcome:
    fT, ls, lt, expected = ref.ISO20065_ANNEX_E_TONES[1]  # 137.3 Hz tone
    value = ph.psychoacoustics.tone_audibility(
        lt, ls, fT, ref.ISO20065_LINE_SPACING
    )
    # 0.05 dB absorbs the standard's 2-decimal table rounding of LS/LT/LG/av.
    return numeric(expected, value, 0.05, unit="dB", places=2)


@register(_TONE_AUD, "ISO/PAS 20065:2016 Formula (13)", "Masking index av at 137.3 / 592.2 Hz")
def _chk_iso20065_masking_index() -> Outcome:
    av137 = ph.psychoacoustics.masking_index(137.3)
    av592 = ph.psychoacoustics.masking_index(592.2)
    ok = (
        abs(av137 - ref.ISO20065_AV_137) <= 0.005
        and abs(av592 - ref.ISO20065_AV_592) <= 0.005
    )
    return Outcome(
        expected=f"{ref.ISO20065_AV_137:g} dB @ 137.3 Hz; "
        f"{ref.ISO20065_AV_592:g} dB @ 592.2 Hz (+/-0.005 dB)",
        computed=f"{av137:.3f} dB; {av592:.3f} dB",
        delta=f"{av137 - ref.ISO20065_AV_137:+.3f}; "
        f"{av592 - ref.ISO20065_AV_592:+.3f} dB",
        passed=ok,
    )


@register(_TONE_AUD, "ISO/PAS 20065:2016 Formula (20)", "Mean audibility of the five spectra, Annex E")
def _chk_iso20065_mean_audibility() -> Outcome:
    value = ph.psychoacoustics.mean_audibility(
        ref.ISO20065_DECISIVE_AUDIBILITIES
    )
    # 0.05 dB absorbs the 2-decimal rounding of the tabulated decisive values.
    return numeric(ref.ISO20065_MEAN_AUDIBILITY, value, 0.05, unit="dB", places=2)


@register(_TONE_AUD, "ISO/PAS 20065:2016 Formula (6)", "Mean narrow-band level LS from spectrum, Table E.1")
def _chk_iso20065_mean_narrowband_level() -> Outcome:
    value = ph.psychoacoustics.mean_narrowband_level(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, 137.3
    )
    # Iterative Formula (6) with Hanning correction; 0.02 dB absorbs rounding.
    return numeric(ref.ISO20065_E1_LS, value, 0.02, unit="dB", places=2)


@register(_TONE_AUD, "ISO/PAS 20065:2016 Clause 6", "Extended uncertainty U of the 137.3 Hz tone, Table E.2")
def _chk_iso20065_uncertainty() -> Outcome:
    res = ph.psychoacoustics.analyze_spectrum(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, ref.ISO20065_LINE_SPACING
    )
    assert res.extended_uncertainties is not None
    by_freq = dict(zip(res.tone_frequencies, res.extended_uncertainties))
    # Table E.2, run index k = 2: U = 2.79 dB (90 % bilateral coverage).
    return numeric(ref.ISO20065_E2_U[1], float(by_freq[137.3]), 0.02, unit="dB", places=2)


@register(_TONE_AUD, "ISO/PAS 20065:2016 Formulae (28)-(29)", "Extended uncertainty of the mean audibility, Annex E Step 4")
def _chk_iso20065_mean_uncertainty() -> Outcome:
    u_j = [row[6] for row in ref.ISO20065_E4_DECISIVE_ROWS]
    value = ph.psychoacoustics.mean_audibility_uncertainty(
        ref.ISO20065_DECISIVE_AUDIBILITIES, u_j
    )
    return numeric(
        ref.ISO20065_E4_MEAN_UNCERTAINTY, value, 0.01, unit="dB", places=2
    )


@register(_TONE_AUD, "ISO/PAS 20065:2016 Formula (8)", "Tone level LT from spectrum, Table E.1")
def _chk_iso20065_tone_level() -> Outcome:
    ls = ph.psychoacoustics.mean_narrowband_level(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, 137.3
    )
    value = ph.psychoacoustics.tone_level(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, 137.3, ls
    )
    return numeric(ref.ISO20065_E1_LT, value, 0.02, unit="dB", places=2)


@register(_TONE_AUD, "ISO/PAS 20065:2016 Clause 5.3.8", "Tone detection over the spectrum, Table E.1")
def _chk_iso20065_peak_detection() -> Outcome:
    result = ph.psychoacoustics.analyze_spectrum(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, ref.ISO20065_LINE_SPACING
    )
    assert result.group_sizes is not None
    singles = result.group_sizes == 1
    found = sorted(round(float(f), 1) for f in result.tone_frequencies[singles])
    expected = sorted(ref.ISO20065_E1_TONE_FREQUENCIES)
    ok = found == expected
    return Outcome(
        expected=f"tones at {expected} Hz",
        computed=f"tones at {found} Hz",
        delta="exact" if ok else "mismatch",
        passed=ok,
    )


@register(
    _TONE_AUD,
    "ISO/PAS 20065:2016 Clause 5.3.8 Step 3",
    "Same-band FG combination inside analyze_spectrum, Table E.2 row 2 FG",
)
def _chk_iso20065_step3_fg() -> Outcome:
    result = ph.psychoacoustics.analyze_spectrum(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, ref.ISO20065_LINE_SPACING
    )
    assert result.group_sizes is not None
    fg = result.group_sizes > 1
    value = float(result.tone_levels[fg][0]) if int(fg.sum()) == 1 else float("nan")
    return numeric(ref.ISO20065_E1_LT_FG, value, 0.02, unit="dB", places=2)


@register(_TONE_AUD, "ISO/PAS 20065:2016 Formula (17)", "Multi-tone FG combination, Table E.1")
def _chk_iso20065_fg_combination() -> Outcome:
    value = ph.psychoacoustics.combined_tone_level(
        ref.ISO20065_E1_LEVELS,
        ref.ISO20065_E1_FREQUENCIES,
        ref.ISO20065_E1_TONE_FREQUENCIES,
        ref.ISO20065_E1_TONE_LS,
    )
    return numeric(ref.ISO20065_E1_LT_FG, value, 0.02, unit="dB", places=2)


@register(
    _TONE_AUD,
    "ISO/PAS 20065:2016 Formulae (18)/(19)",
    "Two-tone separation fD (DIN 45681 Annex J), 137.3 / 212 Hz",
)
def _chk_iso20065_two_tone_separation() -> Outcome:
    fd_137 = ph.psychoacoustics.two_tone_separation_frequency(137.3)
    fd_212 = ph.psychoacoustics.two_tone_separation_frequency(212.0)
    # Annex E consistency: the 118.4/137.3 Hz pair is combined, not separated
    # (|Δf| = 18.9 Hz < fD ≈ 24 Hz at the more prominent tone).
    annex_e_combined = not ph.psychoacoustics.resolve_tones_separately(
        118.4, 137.3, 4.0, 5.0
    )
    ok = (
        round(fd_137, 2) == ref.ISO20065_FD_137
        and round(fd_212, 2) == ref.ISO20065_FD_212
        and annex_e_combined
    )
    return Outcome(
        expected=(
            f"fD(137.3)={ref.ISO20065_FD_137}, fD(212)={ref.ISO20065_FD_212} Hz; "
            "Annex E pair combined"
        ),
        computed=(
            f"fD(137.3)={fd_137:.2f}, fD(212)={fd_212:.2f} Hz; "
            f"Annex E pair {'combined' if annex_e_combined else 'separated'}"
        ),
        delta="exact" if ok else "mismatch",
        passed=ok,
    )


# ===========================================================================
# Psychoacoustic annoyance & fluctuation strength (Fastl & Zwicker)
# ===========================================================================
_PA_FS = "Psychoacoustic annoyance & fluctuation strength (Fastl & Zwicker)"


@register(
    _PA_FS,
    "Fastl & Zwicker Eqs (16.2)-(16.4)",
    "Psychoacoustic annoyance, worked (N5,S,F,R) tuple",
)
def _chk_psychoacoustic_annoyance() -> Outcome:
    n5, s, f, r = ref.PA_WORKED_INPUT
    value = ph.psychoacoustics.psychoacoustic_annoyance(n5, s, f, r).annoyance
    return numeric(ref.PA_WORKED_VALUE, value, 1e-3, places=4)


@register(
    _PA_FS,
    "Fastl & Zwicker Eq (10.2)",
    "Fluctuation strength of AM broadband noise (60 dB, m=1, 4 Hz)",
)
def _chk_fluctuation_strength_am_noise() -> Outcome:
    value = ph.psychoacoustics.fluctuation_strength_am_noise(60.0, 1.0, 4.0)
    return numeric(ref.FS_BBN_60_1_4, value, 1e-3, unit="vacil", places=4)


@register(
    _PA_FS,
    "Fastl & Zwicker Ch. 10 / Osses et al. 2016",
    "Fluctuation-strength calibration: 1 kHz / 60 dB / m=1 / 4 Hz AM tone",
)
def _chk_fluctuation_strength_calibration() -> Outcome:
    # The signal model is anchored (clean-room) so the 1-vacil reference tone
    # returns 1.00 vacil through its own front-end. No numeric standard exists.
    fs = 48000
    t = np.arange(int(fs * 2.0)) / fs
    tone = (1.0 + np.sin(2.0 * np.pi * 4.0 * t)) * np.sin(2.0 * np.pi * 1000.0 * t)
    tone = tone / np.sqrt(np.mean(tone**2)) * 2e-5 * 10.0 ** (60.0 / 20.0)
    value = ph.psychoacoustics.fluctuation_strength(
        tone, fs
    ).fluctuation_strength
    return numeric(
        ref.FS_CALIBRATION_VACIL, value, 0.05, unit="vacil", places=3
    )
