#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Speech intelligibility predicted from spectra and from signals.

Two families that answer the same question with different inputs. The Speech
Intelligibility Index of ANSI S3.5-1997 works from band levels and the
importance function; its oracles are the reference test cases published with
the standard by the ASA working group (the ``.TST`` files), read here in the
same form the tests read them.

The short-time objective intelligibility measures STOI and ESTOI work from the
degraded signal itself. They have no standard behind them, so they are checked
against the properties their papers state - the correlation limits, the
monotonicity in signal-to-noise ratio - on signals synthesized to have a known
answer.
"""

from __future__ import annotations

import numpy as np
import reference_data as ref

import phonometry as ph

from ..registry import Outcome, numeric, register

_SII = "Speech intelligibility (ANSI S3.5-1997)"


@register(_SII, "ANSI S3.5-1997 Table 3", "Band-importance function normalisation")
def _chk_sii_band_importance_sum() -> Outcome:
    total = float(ph.speech.sii.BAND_IMPORTANCE.sum())
    return numeric(ref.ANSIS3_5_BAND_IMPORTANCE_SUM, total, 1e-9, places=6)


@register(_SII, "ASA WG S3-79 SII.C (clause 5.4)", "Equivalent masking spectrum level at 200 Hz")
def _chk_sii_masking() -> Outcome:
    result = ph.speech.speech_intelligibility_index("normal")
    return numeric(ref.ANSIS3_5_MASKING_Z_200HZ, float(result.masking[1]), 1e-3, places=3)


@register(_SII, "ANSI S3.5-1997 clause 5.6", "Equivalent disturbance in quiet at 5000 Hz")
def _chk_sii_disturbance_quiet() -> Outcome:
    # In quiet Di = max(Zi, Xi') = Xi' = -23.6 dB (Table 3) at 5000 Hz; an
    # energy-sum disturbance would read above the reference internal noise.
    result = ph.speech.sii.speech_intelligibility_index("normal")
    return numeric(
        ref.ANSIS3_5_DISTURBANCE_5000HZ, float(result.disturbance[15]), 1e-2,
        unit="dB", places=2,
    )


@register(_SII, "ASA WG S3-79 SII.C (clause 6)", "SII, noise 30 dB plus hearing loss 40 dB")
def _chk_sii_noise_plus_loss() -> Outcome:
    result = ph.speech.sii.speech_intelligibility_index(
        "normal",
        noise_spectrum=np.full(18, 30.0),
        threshold=np.full(18, 40.0),
    )
    return numeric(ref.ANSIS3_5_NOISE_PLUS_LOSS, result.sii, 1e-6, places=6)


@register(_SII, "ANSI S3.5-1997 Annex C.2", "Worked example (SII.C / R CRAN, errata applied)")
def _chk_sii_annex_c2() -> Outcome:
    result = ph.speech.sii.speech_intelligibility_index(
        np.full(18, 54.0),
        np.array([40.0, 30.0, 20.0] + [0.0] * 15),
        threshold=np.zeros(18),
    )
    return numeric(ref.ANSIS3_5_ANNEX_C2, result.sii, 1e-6, places=6)


@register(_SII, "ANSI S3.5-1997 Table C.2 (errata)", "Masking Zi at 200 Hz, corrected worksheet")
def _chk_sii_annex_c2_masking() -> Outcome:
    # The officially corrected first-row slope Ci = -46.59 (WG S3-79 errata;
    # printed -45.59) is required to reproduce the printed Z2 = 34.66 dB.
    result = ph.speech.sii.speech_intelligibility_index(
        np.full(18, 54.0),
        np.array([40.0, 30.0, 20.0] + [0.0] * 15),
        threshold=np.zeros(18),
    )
    return numeric(
        ref.ANSIS3_5_ANNEX_C2_MASKING[1], float(result.masking[1]), 5e-3,
        unit="dB", places=2,
    )


@register(_SII, "ASA WG S3-79 SII.C (clause 6)", "SII, standard speech in quiet, normal hearing")
def _chk_sii_standard_quiet() -> Outcome:
    result = ph.speech.speech_intelligibility_index("normal")
    return numeric(ref.ANSIS3_5_STANDARD_QUIET, result.sii, 1e-6, places=8)


def _sii_official_case(
    method: str,
    speech: tuple[float, ...],
    noise: tuple[float, ...],
    threshold: tuple[float, ...],
    published: float,
    importance: tuple[float, ...] | None = None,
) -> Outcome:
    """Run one official ASA WG S3-79 ``.TST`` case against its published SII.

    The published results are printed to three decimals in the DevelopmentKit
    readme, hence the 5e-4 tolerance; the tests pin the same eight cases to the
    full precision of the committee code.
    """
    result = ph.speech.sii.speech_intelligibility_index(
        np.array(speech), np.array(noise), threshold=np.array(threshold),
        method=method,
        band_importance=None if importance is None else np.array(importance),
    )
    return numeric(published, result.sii, 5e-4, places=3)


@register(_SII, "ASA WG S3-79 TO.TST", "Official one-third-octave test case")
def _chk_sii_wg_to_tst() -> Outcome:
    return _sii_official_case(
        "one-third-octave", ref.ANSIS3_5_WG_TO_SPEECH, ref.ANSIS3_5_WG_TO_NOISE,
        ref.ANSIS3_5_WG_TO_THRESHOLD, ref.ANSIS3_5_WG_TO_SII,
    )


@register(_SII, "ASA WG S3-79 TO_1.TST", "Official test case, alternative importance")
def _chk_sii_wg_to1_tst() -> Outcome:
    return _sii_official_case(
        "one-third-octave", ref.ANSIS3_5_WG_TO_SPEECH, ref.ANSIS3_5_WG_TO_NOISE,
        ref.ANSIS3_5_WG_TO_THRESHOLD, ref.ANSIS3_5_WG_TO1_SII,
        ref.ANSIS3_5_WG_TO1_IMPORTANCE,
    )


@register(_SII, "ASA WG S3-79 CB.TST", "Official critical-band test case")
def _chk_sii_wg_cb_tst() -> Outcome:
    return _sii_official_case(
        "critical-band", ref.ANSIS3_5_WG_CB_SPEECH, ref.ANSIS3_5_WG_CB_NOISE,
        ref.ANSIS3_5_WG_CB_THRESHOLD, ref.ANSIS3_5_WG_CB_SII,
    )


@register(_SII, "ASA WG S3-79 CB_1.TST", "Critical band, alternative importance")
def _chk_sii_wg_cb1_tst() -> Outcome:
    return _sii_official_case(
        "critical-band", ref.ANSIS3_5_WG_CB_SPEECH, ref.ANSIS3_5_WG_CB_NOISE,
        ref.ANSIS3_5_WG_CB_THRESHOLD, ref.ANSIS3_5_WG_CB1_SII,
        ref.ANSIS3_5_WG_CB1_IMPORTANCE,
    )


@register(_SII, "ASA WG S3-79 ECB.TST", "Official equally-contributing test case")
def _chk_sii_wg_ecb_tst() -> Outcome:
    return _sii_official_case(
        "equally-contributing", ref.ANSIS3_5_WG_ECB_SPEECH,
        ref.ANSIS3_5_WG_ECB_NOISE, ref.ANSIS3_5_WG_ECB_THRESHOLD,
        ref.ANSIS3_5_WG_ECB_SII,
    )


@register(_SII, "ASA WG S3-79 ECB_1.TST", "Equally contributing, alternative importance")
def _chk_sii_wg_ecb1_tst() -> Outcome:
    return _sii_official_case(
        "equally-contributing", ref.ANSIS3_5_WG_ECB_SPEECH,
        ref.ANSIS3_5_WG_ECB_NOISE, ref.ANSIS3_5_WG_ECB_THRESHOLD,
        ref.ANSIS3_5_WG_ECB1_SII, ref.ANSIS3_5_WG_ECB1_IMPORTANCE,
    )


@register(_SII, "ASA WG S3-79 OCTAVE.TST", "Official octave-band test case")
def _chk_sii_wg_octave_tst() -> Outcome:
    return _sii_official_case(
        "octave", ref.ANSIS3_5_WG_OCTAVE_SPEECH, ref.ANSIS3_5_WG_OCTAVE_NOISE,
        ref.ANSIS3_5_WG_OCTAVE_THRESHOLD, ref.ANSIS3_5_WG_OCTAVE_SII,
    )


@register(_SII, "ASA WG S3-79 OCTAVE_1.TST", "Octave band, alternative importance")
def _chk_sii_wg_octave1_tst() -> Outcome:
    return _sii_official_case(
        "octave", ref.ANSIS3_5_WG_OCTAVE_SPEECH, ref.ANSIS3_5_WG_OCTAVE_NOISE,
        ref.ANSIS3_5_WG_OCTAVE_THRESHOLD, ref.ANSIS3_5_WG_OCTAVE1_SII,
        ref.ANSIS3_5_WG_OCTAVE1_IMPORTANCE,
    )


@register(_SII, "ANSI S3.5-1997 Annex C.1", "Octave-band worked example (SII.C)")
def _chk_sii_annex_c1() -> Outcome:
    return _sii_official_case(
        "octave", ref.ANSIS3_5_ANNEX_C1_SPEECH, ref.ANSIS3_5_ANNEX_C1_NOISE,
        (0.0,) * 6, ref.ANSIS3_5_ANNEX_C1,
    )


@register(_SII, "ANSI S3.5-1997 Table C.1 (errata)", "Level distortion Li, row i = 5")
def _chk_sii_annex_c1_level_distortion() -> Outcome:
    # Table C.1's Li column with the official WG S3-79 erratum applied (the
    # printed 0.10 should be 1.00); clause 5.7 gives 0.99581 for that row.
    result = ph.speech.sii.speech_intelligibility_index(
        np.array(ref.ANSIS3_5_ANNEX_C1_SPEECH),
        np.array(ref.ANSIS3_5_ANNEX_C1_NOISE),
        method="octave",
    )
    return numeric(
        ref.ANSIS3_5_ANNEX_C1_LEVEL_DISTORTION_I5,
        float(result.level_distortion[4]), 5e-3, places=2,
    )


@register(_SII, "ANSI S3.5-1997 Table 1", "Critical-band importance normalisation")
def _chk_sii_critical_importance_sum() -> Outcome:
    total = float(
        ph.speech.sii_procedure("critical-band").band_importance.sum()
    )
    return numeric(ref.ANSIS3_5_CRITICAL_IMPORTANCE_SUM, total, 1e-9, places=6)


@register(_SII, "ANSI S3.5-1997 Table 2", "Equally-contributing importance, 17 x 0.0588")
def _chk_sii_equal_importance_sum() -> Outcome:
    total = float(
        ph.speech.sii_procedure("equally-contributing").band_importance.sum()
    )
    return numeric(ref.ANSIS3_5_EQUAL_IMPORTANCE_SUM, total, 1e-9, places=6)


@register(_SII, "ANSI S3.5-1997 Table 4", "Octave-band importance normalisation")
def _chk_sii_octave_importance_sum() -> Outcome:
    total = float(ph.speech.sii_procedure("octave").band_importance.sum())
    return numeric(ref.ANSIS3_5_OCTAVE_IMPORTANCE_SUM, total, 1e-9, places=6)


@register(_SII, "ANSI S3.5-1997 Table 4", "Octave-band Ui and Xi equal Table 3's")
def _chk_sii_octave_matches_table3() -> Outcome:
    # Both are spectrum (per-hertz) levels, so Table 4 repeats Table 3's Ui and
    # Xi at all six shared centre frequencies. Reported as the largest
    # disagreement over the twelve cells.
    octave = ph.speech.sii_procedure("octave")
    third = ph.speech.sii_procedure("one-third-octave")
    worst = 0.0
    for fc, speech, noise in ref.ANSIS3_5_OCTAVE_TABLE4_SHARED:
        k = int(np.flatnonzero(np.isclose(octave.frequencies, fc))[0])
        j = int(np.flatnonzero(np.isclose(third.frequencies, fc))[0])
        worst = max(
            worst,
            abs(float(octave.speech_spectrum[k]) - speech),
            abs(float(octave.internal_noise[k]) - noise),
            abs(float(octave.speech_spectrum[k]) - float(third.speech_spectrum[j])),
            abs(float(octave.internal_noise[k]) - float(third.internal_noise[j])),
        )
    return numeric(0.0, worst, 1e-9, unit="dB", places=2)


@register(_SII, "ANSI S3.5-1997 Table 1", "Critical-band table, all 21 rows")
def _chk_sii_critical_table1() -> Outcome:
    # Every cell of Table 1 as shipped: centre, both limits, Ii, Ui and Xi.
    # Reported as the largest absolute disagreement over the 126 cells.
    proc = ph.speech.sii_procedure("critical-band")
    worst = 0.0
    for i, (fc, lo, hi, imp, speech, noise) in enumerate(
        ref.ANSIS3_5_CRITICAL_TABLE1
    ):
        worst = max(
            worst,
            abs(float(proc.frequencies[i]) - fc),
            abs(float(proc.band_edges[i]) - lo),
            abs(float(proc.band_edges[i + 1]) - hi),
            abs(float(proc.band_importance[i]) - imp),
            abs(float(proc.speech_spectrum[i]) - speech),
            abs(float(proc.internal_noise[i]) - noise),
        )
    return numeric(0.0, worst, 1e-9, places=4)


@register(_SII, "ASA WG S3-79 SII.C (clause 6)", "Flat-input cases, all four procedures")
def _chk_sii_flat_cases() -> Outcome:
    # Flat speech/noise inputs that bring every band's Ui (loud regime) and Xi
    # (quiet regime) into the chain, which the eight .TST cases leave inert.
    # Reported as the largest deviation from the committee code over the twelve.
    worst = 0.0
    for method, _regime, speech, noise, committee in ref.ANSIS3_5_WG_FLAT_CASES:
        n = ph.speech.sii_procedure(method).frequencies.size
        result = ph.speech.sii.speech_intelligibility_index(
            np.full(n, speech), np.full(n, noise), method=method
        )
        worst = max(worst, abs(result.sii - committee))
    return numeric(0.0, worst, 1e-9, places=10)


@register(_SII, "ANSI S3.5-1997 Table 3", "Loud-effort speech spectrum level at 1 kHz")
def _chk_sii_loud_spectrum() -> Outcome:
    from phonometry.speech.sii import standard_speech_spectrum

    value = float(standard_speech_spectrum("loud")[8])
    return numeric(ref.ANSIS3_5_LOUD_1KHZ, value, 1e-9, unit="dB", places=2)


# ---------------------------------------------------------------------------
# Short-time objective intelligibility (STOI / ESTOI)
# ---------------------------------------------------------------------------
_STOI = "Objective intelligibility (STOI / ESTOI)"


def _stoi_speech_like(seed: int) -> np.ndarray:
    """A deterministic speech-like signal at the 10 kHz STOI internal rate."""
    fs = ph.speech.objective_intelligibility.SAMPLE_RATE
    rng = np.random.default_rng(seed)
    t = np.arange(3 * fs) / fs
    sig = np.zeros_like(t)
    for f0 in (200.0, 400.0, 700.0, 1100.0, 1800.0, 2600.0):
        depth = 0.5 * (1.0 + np.sin(2.0 * np.pi * rng.uniform(2.0, 6.0) * t
                                    + rng.uniform(0.0, 2.0 * np.pi)))
        sig += depth * np.sin(2.0 * np.pi * f0 * t + rng.uniform(0.0, 2.0 * np.pi))
    return np.asarray(sig, dtype=np.float64)


@register(
    _STOI,
    "Taal et al. 2011 (Eq. 6, degenerate)",
    "STOI of a signal against itself = 1 (perfect correlation)",
)
def _chk_stoi_identity() -> Outcome:
    x = _stoi_speech_like(1)
    return numeric(
        1.0,
        ph.speech.stoi(
            x, x, ph.speech.objective_intelligibility.SAMPLE_RATE
        ).value,
        1e-6,
        places=6,
    )


@register(
    _STOI,
    "Jensen & Taal 2016 (Eq. 8, degenerate)",
    "ESTOI of a signal against itself = 1 (perfect spectral correlation)",
)
def _chk_estoi_identity() -> Outcome:
    x = _stoi_speech_like(1)
    fs = ph.speech.objective_intelligibility.SAMPLE_RATE
    return numeric(
        1.0, ph.speech.stoi(x, x, fs, extended=True).value, 1e-6, places=6
    )


@register(
    _STOI,
    "Taal et al. 2011 (monotonicity with SNR)",
    "STOI rises from -15 dB to +25 dB SNR speech-shaped noise",
)
def _chk_stoi_monotonic() -> Outcome:
    fs = ph.speech.objective_intelligibility.SAMPLE_RATE
    x = _stoi_speech_like(2)
    rng = np.random.default_rng(10)
    noise = rng.standard_normal(x.size)
    scale = np.sqrt(np.mean(x**2)) / np.sqrt(np.mean(noise**2))
    lo = ph.speech.stoi(
        x, x + scale * 10.0 ** (15.0 / 20.0) * noise, fs
    ).value  # -15 dB
    hi = ph.speech.stoi(
        x, x + scale * 10.0 ** (-25.0 / 20.0) * noise, fs
    ).value  # +25 dB
    return Outcome(
        expected="STOI(+25 dB) - STOI(-15 dB) > 0.2",
        computed=f"{hi - lo:.3f} ({lo:.3f} -> {hi:.3f})",
        delta="0",
        passed=(hi - lo) > 0.2,
    )
