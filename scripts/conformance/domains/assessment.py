#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Judging a measured noise, and the confidence and consequence of the judgement.

The metrics that turn a level into a verdict: impulsive-sound prominence
(NT ACOU 112 and ISO/PAS 1996-3) and the ANSI S12.2-2019 room-noise ratings.
With them the quantities a verdict is only meaningful against - the hearing
threshold by age and its reference levels (ISO 7029, ISO 389-7), the
measurement uncertainty machinery of the GUM and its Supplement 1, and the
noise-induced hearing loss statistics of ISO 1999.

The ISO 2631-5 multiple-shock whole-body vibration checks and the EN 12354-6
equivalent absorption area close the module: both are assessment quantities
computed from a measured series or a surface list, and both were registered
here in the order the report reads them.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
import reference_data as ref
from scipy import signal as sg

import phonometry as ph

from ..registry import Outcome, numeric, register

if TYPE_CHECKING:
    from phonometry.environment import ImpulseOnset
    from phonometry.metrology import UncertaintyResult

_NTA = "Impulsive-sound prominence (NT ACOU 112)"


@register(
    _NTA, "NT ACOU 112:2002 Formula 1", "Predicted prominence, OR=1000 dB/s, LD=30 dB"
)
def _chk_impulse_prominence() -> Outcome:
    value = float(ph.environment.predicted_prominence(1000.0, 30.0))
    return numeric(ref.NTACOU112_PROMINENCE, value, 1e-4, places=4)


@register(
    _NTA, "NT ACOU 112:2002 Formula 2", "Adjustment KI to LAeq at prominence P=10"
)
def _chk_impulse_adjustment() -> Outcome:
    value = float(ph.environment.impulse_adjustment(10.0))
    return numeric(ref.NTACOU112_ADJUSTMENT_P10, value, 1e-9, unit="dB", places=3)


_ISO1996_3 = "Impulsive-sound prominence (ISO/PAS 1996-3)"


def _iso1996_3_ramp_onset() -> ImpulseOnset:
    """Detected onset of a 30 dB LpAF ramp over 0.30 s (dt = 20 ms)."""
    from phonometry.environment.assessment.impulsive_sound import detect_onsets

    dt = 0.02
    pre = np.full(round(0.2 / dt), 40.0)
    rise = 40.0 + 30.0 * (np.arange(1, round(0.3 / dt) + 1) / round(0.3 / dt))
    post = np.full(round(0.3 / dt), 70.0)
    return detect_onsets(np.concatenate([pre, rise, post]), dt)[0]


@register(
    _ISO1996_3, "ISO/PAS 1996-3:2022 3.5", "Onset rate of a 30 dB ramp over 0.30 s"
)
def _chk_iso1996_3_onset_rate() -> Outcome:
    return numeric(
        ref.ISO1996_3_RAMP_ONSET_RATE,
        _iso1996_3_ramp_onset().onset_rate,
        1e-6,
        unit="dB/s",
    )


@register(
    _ISO1996_3, "ISO/PAS 1996-3:2022 Formula 3", "Adjustment KI of the ramp onset"
)
def _chk_iso1996_3_adjustment() -> Outcome:
    value = float(ph.environment.impulse_adjustment(_iso1996_3_ramp_onset().prominence))
    return numeric(ref.ISO1996_3_RAMP_ADJUSTMENT, value, 1e-6, unit="dB", places=4)


_RN = "Room noise (ANSI S12.2-2019)"


@register(_RN, "ANSI S12.2-2019 Table 1", "NC-40 curve, tangency self-consistency")
def _chk_rn_nc_self() -> Outcome:
    rating = ph.room.noise_criterion(ph.room.nc_curve(40.0)).tangency_rating
    return numeric(ref.ANSIS12_2_NC40_SELF, rating, 1e-9, places=3)


@register(_RN, "ANSI S12.2-2019 Table D.1", "RC-31 Mark II curve, 63 Hz level")
def _chk_rn_rc_curve() -> Outcome:
    return numeric(
        ref.ANSIS12_2_RC31_63HZ,
        float(ph.room.rc_curve(31.0)[2]),
        1e-9,
        places=3,
    )


@register(_RN, "ANSI S12.2-2019 clause D.4", "RC-35 curve, mid-frequency average LMF")
def _chk_rn_rc_lmf() -> Outcome:
    return numeric(
        ref.ANSIS12_2_RC35_LMF,
        ph.room.room_criterion(ph.room.rc_curve(35.0)).lmf,
        1e-9,
        places=3,
    )


_HEAR = "Hearing threshold (ISO 7029 / ISO 389-7)"


@register(_HEAR, "ISO 7029:2017 Table 1", "Median threshold, male age 60 at 4 kHz")
def _chk_hearing_median() -> Outcome:
    value = float(ph.hearing.age_threshold(60, "male", 0.5).median[8])
    return numeric(ref.ISO7029_MEDIAN_MALE_60_4KHZ, value, 1e-3, unit="dB", places=3)


@register(_HEAR, "ISO 7029:2017 Table 2", "Upper spread su, male age 60 at 1 kHz")
def _chk_hearing_spread() -> Outcome:
    value = float(ph.hearing.age_threshold(60, "male", 0.5).spread_upper[4])
    return numeric(ref.ISO7029_SU_MALE_60_1KHZ, value, 1e-3, unit="dB", places=3)


@register(_HEAR, "ISO 389-7:2005 Table 1", "Free-field reference threshold at 1 kHz")
def _chk_hearing_reference() -> Outcome:
    value = float(ph.hearing.reference_threshold("free-field")[4])
    return numeric(ref.ISO389_7_REF_FREE_1KHZ, value, 1e-9, unit="dB", places=3)


_GUM = "Measurement uncertainty (GUM / Supplement 1)"


@register(
    _GUM, "ISO/IEC Guide 98-3-1 clause 9.2", "Combined uncertainty, additive model"
)
def _chk_gum_additive() -> Outcome:
    quantities = [ph.metrology.Quantity(0.0, 1.0) for _ in range(4)]
    result = ph.metrology.combine_uncertainty(
        lambda a, b, c, d: a + b + c + d, quantities
    )
    return numeric(ref.GUM_ADDITIVE_UC, result.combined_uncertainty, 1e-9, places=4)


@register(_GUM, "ISO/IEC Guide 98-3 Table G.2", "Coverage factor, p=0.99, v=16")
def _chk_gum_coverage() -> Outcome:
    from phonometry.metrology.uncertainty import coverage_factor

    return numeric(ref.GUM_COVERAGE_K99_16, coverage_factor(0.99, 16), 5e-3, places=3)


@register(_GUM, "ISO/IEC Guide 98-3 Annex G.4", "Welch-Satterthwaite effective dof")
def _chk_gum_welch() -> Outcome:
    quantities = [ph.metrology.Quantity(0.0, 1.0, dof=10) for _ in range(4)]
    result = ph.metrology.combine_uncertainty(
        lambda a, b, c, d: a + b + c + d, quantities
    )
    return numeric(ref.GUM_WELCH_VEFF, result.effective_dof, 1e-6, places=3)


def _gum_h1_result() -> UncertaintyResult:
    quantities = [
        ph.metrology.Quantity(v, unc, dof=dof) for v, unc, dof in ref.GUM_H1_INPUTS
    ]
    with warnings.catch_warnings():
        # alphaS and theta are genuinely flat directions at the H.1 estimates.
        warnings.simplefilter("ignore")
        return ph.metrology.combine_uncertainty(
            lambda ls, d, a_s, th, da, dth: ls + d - ls * (da * th + a_s * dth),
            quantities,
        )


@register(_GUM, "ISO/IEC Guide 98-3 Annex H.1", "End-gauge combined uncertainty uc, nm")
def _chk_gum_h1_uc() -> Outcome:
    result = _gum_h1_result()
    return numeric(
        ref.GUM_H1_UC, result.combined_uncertainty, 0.01, unit="nm", places=2
    )


@register(
    _GUM, "ISO/IEC Guide 98-3 Annex H.1", "End-gauge expanded uncertainty U99, nm"
)
def _chk_gum_h1_u99() -> Outcome:
    result = _gum_h1_result()
    _, big = result.expanded(0.99)
    return numeric(ref.GUM_H1_U99, big, 0.1, unit="nm", places=1)


@register(
    _GUM,
    "ISO/IEC Guide 98-3 Annex H.2 (Table H.3)",
    "Correlated V/I/phi budget: uc(R), ohm",
)
def _chk_gum_h2_correlated() -> Outcome:
    obs = np.array(ref.GUM_H2_OBSERVATIONS)
    obs[:, 1] *= 1e-3  # mA -> A
    means = obs.mean(axis=0)
    u_means = obs.std(axis=0, ddof=1) / math.sqrt(obs.shape[0])
    r = np.corrcoef(obs.T)
    quantities = [
        ph.metrology.Quantity(m, s) for m, s in zip(means, u_means, strict=True)
    ]
    result = ph.metrology.combine_uncertainty(
        lambda v, i, p: v / i * math.cos(p), quantities, correlation=r
    )
    return numeric(
        ref.GUM_H2_RESULTS["R"][1],
        result.combined_uncertainty,
        1e-3,
        unit="ohm",
        places=3,
    )


@register(
    _GUM,
    "ISO/IEC Guide 98-3-1 Table 3 (clause 9.2.3)",
    "Seeded Monte Carlo, rectangular sum: 95 % interval endpoint",
)
def _chk_gum_s1_table3_monte_carlo() -> Outcome:
    quantities = [ph.metrology.Quantity(0.0, 1.0, "rectangular") for _ in range(4)]
    mc = ph.metrology.monte_carlo(
        lambda a, b, c, d: a + b + c + d,
        quantities,
        trials=1_000_000,
        coverage=0.95,
        seed=1996,
    )
    endpoint = 0.5 * (mc.interval[1] - mc.interval[0])
    ok_u = abs(mc.standard_uncertainty - ref.GUMS1_TABLE3_U) <= 0.01
    outcome = numeric(ref.GUMS1_TABLE3_INTERVAL_95, endpoint, 0.03, places=3)
    return Outcome(
        expected=f"+/-{ref.GUMS1_TABLE3_INTERVAL_95} (u = {ref.GUMS1_TABLE3_U})",
        computed=f"+/-{endpoint:.3f} (u = {mc.standard_uncertainty:.3f})",
        delta=outcome.delta,
        passed=outcome.passed and ok_u,
    )


_NIHL = "Noise-induced hearing loss (ISO 1999)"


@register(_NIHL, "ISO 1999:2013 Table D.2", "Median NIPTS, 4 kHz, 90 dB, 20 yr")
def _chk_nihl_median() -> Outcome:
    value = float(ph.hearing.nipts(90.0, 20.0, 0.5).value[4])
    return numeric(ref.ISO1999_N50_4K_90_20, value, 0.5, unit="dB", places=1)


@register(_NIHL, "ISO 1999:2013 Table D.2", "Worst-10 % NIPTS, 4 kHz, 90 dB, 20 yr")
def _chk_nihl_fractile() -> Outcome:
    value = float(ph.hearing.nipts(90.0, 20.0, 0.9).value[4])
    return numeric(ref.ISO1999_N10_4K_90_20, value, 0.5, unit="dB", places=1)


@register(_NIHL, "ISO 1999:2013 Table D.4", "Worst-10 % NIPTS, 3 kHz, 100 dB, 40 yr")
def _chk_nihl_high() -> Outcome:
    value = float(ph.hearing.nipts(100.0, 40.0, 0.9).value[3])
    return numeric(ref.ISO1999_N10_3K_100_40, value, 0.5, unit="dB", places=1)


@register(
    _NIHL,
    "ISO 1999:2013 Annex C, Formulae (C.6) to (C.8)",
    "NIPTS at 1/2/4 kHz, 90 dB, 30 yr, Q = 10 % (annex inputs)",
)
def _chk_nihl_annex_c_nipts() -> Outcome:
    """The Table D.2 shifts the Annex C example takes as its noise input."""
    value = np.round(
        ph.hearing.nipts(90.0, 30.0, 0.9, frequencies=[1000.0, 2000.0, 4000.0]).value
    )
    expected = np.asarray(ref.ISO1999_ANNEX_C_N, dtype=float)
    delta = float(np.max(np.abs(value - expected)))
    return Outcome(
        expected=", ".join(f"{v:.0f}" for v in expected) + " dB",
        computed=", ".join(f"{v:.0f}" for v in value) + " dB",
        delta=f"{delta:.0f} dB",
        passed=delta <= 0.0,
    )


@register(
    _NIHL,
    "ISO 1999:2013 Annex C, Formula (C.5)",
    "Compressed 4 kHz shift, Formula (1) with the annex's H = 36 dB",
)
def _chk_nihl_annex_c_compression() -> Outcome:
    """The annex reduces the 19 dB shift at 4 kHz by the H*N/120 term."""
    h = ref.ISO1999_ANNEX_C_H[2]
    n = ref.ISO1999_ANNEX_C_N[2]
    # H' - H is what Formula (1) leaves of the noise component at that band.
    value = float(ph.hearing.combine_age_and_noise(h, n)) - h
    return numeric(
        ref.ISO1999_ANNEX_C_N_4K_COMPRESSED, value, 0.05, unit="dB", places=1
    )


@register(
    _NIHL,
    "ISO 1999:2013 Annex C, Formula (C.11)",
    "Hearing threshold level with age and noise, 1/2/4 kHz mean, Q = 10 %",
)
def _chk_nihl_annex_c_htlan() -> Outcome:
    """The end of the annex chain: the mean H plus the mean compressed shift.

    The age component is the annex's own Table A.3 selection rather than the
    library's ISO 7029:2017 evaluation, so this exercises the Formula (1)
    combination on the standard's stated inputs (see the source note the
    ISO 1999 fiche prints). The annex applies the compression term only where
    it matters, taking the shift straight from Table D.2 "when (H + N) < 40 dB"
    (its Formula (C.4) approximation); here that leaves only the 4 kHz band
    compressed.
    """
    h = np.asarray(ref.ISO1999_ANNEX_C_H, dtype=float)
    n = np.asarray(ref.ISO1999_ANNEX_C_N, dtype=float)
    compressed = ph.hearing.combine_age_and_noise(h, n) - h
    noise = np.where(h + n > ref.ISO1999_ANNEX_C_COMPRESSION_FENCE, compressed, n)
    value = float(np.mean(h) + np.mean(noise))
    return numeric(ref.ISO1999_ANNEX_C_HTLAN, value, 0.05, unit="dB", places=1)


_HPD = "Hearing protectors (ISO 4869-2)"


@register(
    _HPD,
    "ISO 4869-2:2018 Annex A, Table A.1",
    "Assumed protection: mean and spread over 16 subjects, 8 bands",
)
def _chk_hpd_annex_a() -> Outcome:
    """Formula (1) against the printed distribution of the worked example.

    The annex's own ``APV`` row is the difference of the rounded ``m_f`` and
    ``s_f`` it displays, which is 0,1 dB from Formula (1) applied to the
    attenuations in three of the eight bands, so what is pinned here is the
    pair the annex computes from, band for band.
    """
    result = ph.hearing.assumed_protection_value(ref.ISO4869_2_ATTENUATION)
    mean = np.asarray(ref.ISO4869_2_MEAN, dtype=float)
    spread = np.asarray(ref.ISO4869_2_STANDARD_DEVIATION, dtype=float)
    worst = max(
        float(np.max(np.abs(np.round(result.mean_attenuation, 1) - mean))),
        float(np.max(np.abs(np.round(result.standard_deviation, 1) - spread))),
    )
    return Outcome(
        expected="m_f and s_f equal to Table A.1 at 1 dp, all 8 bands",
        computed=f"max deviation {worst:.3f} dB",
        delta=f"{worst:.3f} dB",
        passed=worst <= 1e-9,
    )


@register(
    _HPD,
    "ISO 4869-2:2018 Formula (2), Annex B",
    "Octave-band method: Table B.1 net levels and L'p,A84",
)
def _chk_hpd_octave_band() -> Outcome:
    """The most faithful of the three methods, on the annex's own spectrum."""
    result = ph.hearing.octave_band_protected_level(
        ref.ISO4869_2_ANNEX_B_NOISE, ref.ISO4869_2_APV84_PRINTED
    )
    assert result.band_levels is not None
    net = np.asarray(ref.ISO4869_2_ANNEX_B_NET, dtype=float)
    band_worst = float(np.max(np.abs(np.round(result.band_levels, 1) - net)))
    level = float(result.effective_level)
    passed = (
        band_worst <= 1e-9
        and abs(level - ref.ISO4869_2_ANNEX_B_EFFECTIVE) <= 0.05
        and result.reported_level == ref.ISO4869_2_ANNEX_B_REPORTED
    )
    return Outcome(
        expected=f"Table B.1 rows exact; L'p,A84 = {ref.ISO4869_2_ANNEX_B_EFFECTIVE} dB",
        computed=f"rows within {band_worst:.3f} dB; {level:.1f} dB",
        delta=f"{level - ref.ISO4869_2_ANNEX_B_EFFECTIVE:+.3f} dB",
        passed=passed,
    )


@register(
    _HPD,
    "ISO 4869-2:2018 Formulae (12) to (15), Annex C",
    "HML method: 16 subject triples, statistics and H84/M84/L84",
)
def _chk_hpd_hml() -> Outcome:
    """All 48 printed per-subject values plus the six statistics they reduce to.

    The two cells Table C.1 misprints against Table 2 are what the sixth
    reference noise is sensitive to, and Table 2 is the reading that
    reproduces the annex (see docs/ERRATA.md).
    """
    rating = ph.hearing.hml_rating(ref.ISO4869_2_ATTENUATION)
    per_subject = max(
        float(np.max(np.abs(np.round(values, 1) - np.asarray(printed, dtype=float))))
        for values, printed in (
            (rating.subject_h, ref.ISO4869_2_ANNEX_C_H),
            (rating.subject_m, ref.ISO4869_2_ANNEX_C_M),
            (rating.subject_l, ref.ISO4869_2_ANNEX_C_L),
            (rating.predicted_reduction[:, 5], ref.ISO4869_2_ANNEX_C_PNR_NOISE6),
        )
    )
    passed = per_subject <= 1e-9 and rating.reported == ref.ISO4869_2_ANNEX_C_HML84
    return Outcome(
        expected=f"Table C.2 exact; H84/M84/L84 = {ref.ISO4869_2_ANNEX_C_HML84} dB",
        computed=f"within {per_subject:.3f} dB; {rating.reported} dB",
        delta=f"{per_subject:.3f} dB",
        passed=passed,
    )


@register(
    _HPD,
    "ISO 4869-2:2018 Formulae (16) to (24)",
    "HML and SNR applications land on the annexes' 82 dB",
)
def _chk_hpd_applications() -> Outcome:
    """The same protector in the same noise, by the two rating methods.

    Clause 7.2 and Clause 8.2 both round their ratings to the nearest integer
    before the application formulas see them, which is what makes Annex C's
    PNR84 come out at 22,5 dB rather than at the unrounded fit.
    """
    attenuation = ref.ISO4869_2_ATTENUATION
    hml = ph.hearing.hml_protected_level(
        ref.ISO4869_2_ANNEX_B_LPA,
        ref.ISO4869_2_ANNEX_B_LPC,
        ph.hearing.hml_rating(attenuation),
    )
    snr_rating = ph.hearing.snr_rating(attenuation)
    by_c = ph.hearing.snr_protected_level(snr_rating, l_p_c=ref.ISO4869_2_ANNEX_B_LPC)
    by_a = ph.hearing.snr_protected_level(
        snr_rating,
        l_p_a=ref.ISO4869_2_ANNEX_B_LPA,
        c_minus_a=ref.ISO4869_2_ANNEX_B_LPC - ref.ISO4869_2_ANNEX_B_LPA,
    )
    passed = (
        abs(hml.noise_reduction - ref.ISO4869_2_ANNEX_C_PNR84) <= 1e-9
        and hml.reported_level == ref.ISO4869_2_ANNEX_C_REPORTED
        and snr_rating.reported == ref.ISO4869_2_ANNEX_D_SNR84
        and by_c.reported_level == ref.ISO4869_2_ANNEX_D_REPORTED
        and by_a.reported_level == ref.ISO4869_2_ANNEX_D_REPORTED
    )
    return Outcome(
        expected="PNR84 = 22,5 dB; SNR84 = 21 dB; both report 82 dB",
        computed=(
            f"{hml.noise_reduction:.1f} dB; {snr_rating.reported} dB; "
            f"{hml.reported_level} and {by_c.reported_level} dB"
        ),
        delta=f"{hml.noise_reduction - ref.ISO4869_2_ANNEX_C_PNR84:+.3f} dB",
        passed=passed,
    )


_MSV = "Multiple-shock whole-body vibration (ISO 2631-5)"


@register(
    _MSV, "ISO 2631-5:2018 Formula 3", "Daily acceleration dose, 5 x 40 m/s2 peaks"
)
def _chk_multiple_shock_dose() -> Outcome:
    value = ph.vibration.dose_from_peaks([40.0] * 5)
    return numeric(ref.ISO2631_5_DZD_MALE, value, 0.01, unit="m/s2", places=2)


@register(
    _MSV, "ISO 2631-5:2018 Formula C.3", "Stress variable R, Annex C male example"
)
def _chk_multiple_shock_risk() -> Outcome:
    sd = ph.vibration.compression_dose(ph.vibration.dose_from_peaks([40.0] * 5))
    value = ph.vibration.injury_risk(
        sd, start_age=20, years=20, days_per_year=120, sex="male"
    )
    return numeric(ref.ISO2631_5_R_MALE, value, 0.01, places=2)


@register(
    _MSV, "ISO 2631-5:2018 Formula C.5", "Injury probability, Annex C male example"
)
def _chk_multiple_shock_probability() -> Outcome:
    sd = ph.vibration.compression_dose(ph.vibration.dose_from_peaks([40.0] * 5))
    r = ph.vibration.injury_risk(
        sd, start_age=20, years=20, days_per_year=120, sex="male"
    )
    return numeric(
        ref.ISO2631_5_PI_MALE,
        float(ph.vibration.injury_probability(r)),
        0.01,
        places=2,
    )


@register(
    _MSV, "ISO 2631-5:2018 Annex C NOTE 5", "Compressive stress Sd, female example"
)
def _chk_multiple_shock_female_sd() -> Outcome:
    from phonometry.vibration.human.multiple_shock import MZ_FEMALE

    sd = ph.vibration.compression_dose(
        ph.vibration.dose_from_peaks([40.0] * 5), mz=MZ_FEMALE
    )
    return numeric(ref.ISO2631_5_SD_FEMALE, sd, 0.01, unit="MPa", places=2)


@register(_MSV, "ISO 2631-5:2018 Annex C NOTE 5", "Stress variable R, female example")
def _chk_multiple_shock_female_r() -> Outcome:
    from phonometry.vibration.human.multiple_shock import MZ_FEMALE

    sd = ph.vibration.compression_dose(
        ph.vibration.dose_from_peaks([40.0] * 5), mz=MZ_FEMALE
    )
    r = ph.vibration.injury_risk(
        sd, start_age=20, years=20, days_per_year=120, sex="female"
    )
    return numeric(ref.ISO2631_5_R_FEMALE, r, 0.01, places=2)


@register(
    _MSV,
    "ISO 2631-5:2018 Formula 1 vs Annex D Table D.1",
    "Seat-to-spine transfer vs the 256 Hz digital filter (0,5-80 Hz)",
)
def _chk_multiple_shock_annex_d_filter() -> Outcome:
    freqs = np.array([0.5, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0])
    formula = np.abs(ph.vibration.seat_to_spine_transfer(freqs))
    _, h = sg.freqz(
        ref.ISO2631_5_ANNEX_D_B,
        ref.ISO2631_5_ANNEX_D_A,
        worN=2.0 * np.pi * freqs / ref.ISO2631_5_ANNEX_D_FS,
    )
    worst = float(np.max(np.abs(formula - np.abs(h))))
    return numeric(
        0.0,
        worst,
        0.04,
        places=3,
        expected_label="max abs(Formula 1 - filter) ≤ 0,04",
    )


_ABS = "Sound absorption in enclosed spaces (EN 12354-6)"


@register(
    _ABS, "EN 12354-6:2003 Formula 1", "Equivalent absorption area, Annex E bare room"
)
def _chk_enclosed_space_area() -> Outcome:
    value = float(
        ph.room.equivalent_absorption_area(ref.EN12354_6_ANNEX_E_BARE_SURFACES)
    )
    return numeric(ref.EN12354_6_A_BARE, value, 0.01, unit="m2", places=2)


@register(_ABS, "EN 12354-6:2003 Formula 5", "Reverberation time, Annex E bare room")
def _chk_enclosed_space_rt() -> Outcome:
    area = ph.room.equivalent_absorption_area(ref.EN12354_6_ANNEX_E_BARE_SURFACES)
    value = float(ph.room.reverberation_time(area, ref.EN12354_6_ANNEX_E_VOLUME))
    return numeric(ref.EN12354_6_T_BARE, value, 0.05, unit="s", places=1)
