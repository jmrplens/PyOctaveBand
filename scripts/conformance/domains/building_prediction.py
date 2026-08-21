#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Domain 7 - Building prediction & uncertainty.

The EN 12354 series predicting what Domain 6 measures: airborne insulation
between rooms and its flanking paths (Part 1), impact insulation (Part 2),
facade insulation against outdoor sound (Part 3) and the sound radiated by a
building into the outdoors (Part 4), each checked against the worked example
of its own annex.

The ISO 12999 uncertainty checks belong here rather than with the measurements
because what they qualify is the comparison: how far a predicted or measured
single-number rating may sit from another before the difference means
anything.
"""

from __future__ import annotations

import numpy as np
import reference_data as ref

import phonometry as ph

from ..registry import Outcome, numeric, register


def _annex_h3_paths() -> list[ph.building.FlankingPath]:
    """The EN 12354-1 Annex H.3 flanking paths from the shared input table."""
    ss = ref.EN12354_1_ANNEX_H3_SEPARATING_AREA
    paths: list[ph.building.FlankingPath] = []
    for label, rw, kff, kfd, lf in ref.EN12354_1_ANNEX_H3_ELEMENTS:
        ff, df, fd = ph.building.flanking_element(
            label=label,
            r_flanking=rw,
            r_separating=ref.EN12354_1_ANNEX_H3_R_DIRECT,
            k_ff=kff,
            k_fd=kfd,
            k_df=kfd,
            separating_area=ss,
            coupling_length=lf,
        )
        paths += [ff, df, fd]
    return paths


@register(
    "Building prediction & uncertainty",
    "EN 12354-1:2000 Annex H.3",
    "Airborne prediction R'w (direct + 12 flanking paths)",
)
def _chk_en12354_1_airborne() -> Outcome:
    res = ph.building.predicted_airborne_insulation(
        r_direct=ref.EN12354_1_ANNEX_H3_R_DIRECT, flanking_paths=_annex_h3_paths()
    )
    expected = ref.EN12354_1_ANNEX_H3_RPRIME_W
    computed = float(res.r_prime_w)
    paths_ok = len(res.paths) == ref.EN12354_1_ANNEX_H3_NUM_PATHS
    return Outcome(
        expected=f"R'w {expected} dB ({ref.EN12354_1_ANNEX_H3_NUM_PATHS} paths)",
        computed=f"R'w {round(computed)} dB ({len(res.paths)} paths, {computed:.2f})",
        delta=f"{computed - expected:+.2f} dB",
        passed=paths_ok and round(computed) == expected,
    )


@register(
    "Building prediction & uncertainty",
    "EN 12354-1:2000 Annex H.3 (paths)",
    "All 12 printed flanking-path values Rij,w",
)
def _chk_en12354_1_h3_paths() -> Outcome:
    res = ph.building.predicted_airborne_insulation(
        r_direct=ref.EN12354_1_ANNEX_H3_R_DIRECT, flanking_paths=_annex_h3_paths()
    )
    by_label = {p.label: p.r_w for p in res.paths}
    worst = 0.0
    for element, (r_ff, r_cross) in ref.EN12354_1_ANNEX_H3_PATH_RW.items():
        for suffix, expected in (("Ff", r_ff), ("Fd", r_cross), ("Df", r_cross)):
            worst = max(worst, abs(by_label[f"{element}-{suffix}"] - expected))
    return numeric(
        0.0,
        worst,
        0.05,
        unit="dB",
        places=3,
        expected_label="max abs(Rij,w - printed) <= 0,05 dB",
    )


@register(
    "Building prediction & uncertainty",
    "EN 12354-1:2000 Formula (5b) / Annex H.3",
    "DnT,w closure from R'w (both H.3 examples -> 54 dB)",
)
def _chk_en12354_1_dnt_closure() -> Outcome:
    v = ref.EN12354_1_ANNEX_H3_VOLUME
    ss = ref.EN12354_1_ANNEX_H3_SEPARATING_AREA
    first = float(ph.building.standardized_level_difference(52.2, v, ss))
    second = float(ph.building.standardized_level_difference(52.7, v, ss))
    ok = (
        round(first) == ref.EN12354_1_ANNEX_H3_DNT_W
        and round(second) == ref.EN12354_1_ANNEX_H3_DNT_W_SECOND
        # The printed 53,8 dB uses the standard's own V/(3 S) rounding of the
        # exact 0,32 V/Ss factor (0,18 dB apart).
        and abs(first - ref.EN12354_1_ANNEX_H3_DNT_W_PRINTED) <= 0.2
    )
    return Outcome(
        expected=f"DnT,w {ref.EN12354_1_ANNEX_H3_DNT_W} dB (printed 53,8/54,3)",
        computed=f"DnT,w {first:.2f} / {second:.2f} dB",
        delta=f"{first - ref.EN12354_1_ANNEX_H3_DNT_W_PRINTED:+.2f} dB vs printed",
        passed=ok,
    )


@register(
    "Building prediction & uncertainty",
    "EN 12354-2:2000 Annex E.3",
    "Impact prediction L'n,w = Ln,w,eq - dLw + K",
)
def _chk_en12354_2_impact() -> Outcome:
    ln_eq = ph.building.equivalent_impact_level(ref.EN12354_2_ANNEX_E3_MASS)
    k = ph.building.impact_flanking_correction(
        ref.EN12354_2_ANNEX_E3_MASS, ref.EN12354_2_ANNEX_E3_FLANKING_MEAN_MASS
    )
    res = ph.building.predicted_impact_insulation(
        ln_w_eq=round(ln_eq),
        delta_l_w=ref.EN12354_2_ANNEX_E3_DELTA_LW,
        k_correction=k,
    )
    k_ok = int(k) == ref.EN12354_2_ANNEX_E3_K
    computed = float(res.l_prime_n_w)
    out = numeric(
        ref.EN12354_2_ANNEX_E3_LPRIME_N_W, computed, 1e-9, unit="dB", places=6
    )
    return Outcome(out.expected, out.computed, out.delta, out.passed and k_ok)


@register(
    "Building prediction & uncertainty",
    "EN 12354-2:2000 Formula (3) / Annex E.3",
    "Standardized impact level L'nT,w (exact 0,032 V form -> 43 dB)",
)
def _chk_en12354_2_standardized() -> Outcome:
    # Exact Formula (3): 45 - 10 lg(0,032 x 50) = 42,96 dB. The E.3 chain's own
    # "10 lg(V/30)" rounding gives 42,8 dB; both round to 43 dB.
    lnt = float(ph.building.standardized_impact_level(45.0, 50.0))
    ok = round(lnt) == 43 and abs(lnt - 42.96) <= 0.01
    return Outcome(
        expected="L'nT,w 43 dB (exact 42,96; E.3 prints 42,8)",
        computed=f"L'nT,w {lnt:.2f} dB",
        delta=f"{lnt - 42.96:+.3f} dB",
        passed=ok,
    )


def _en12354_3_annex_f() -> ph.building.FacadePredictionResult:
    """The EN 12354-3 Annex F facade prediction from the shared input table."""
    elements = [
        ph.building.FacadeElement(name=name, area=area, r=r)
        for name, area, r in ref.EN12354_3_ANNEX_F_ELEMENTS
    ]
    elements.append(
        ph.building.FacadeElement(name="inlet", dn_e=ref.EN12354_3_ANNEX_F_INLET_DNE)
    )
    return ph.building.facade_sound_reduction(
        elements,
        area=ref.EN12354_3_ANNEX_F_AREA,
        volume=ref.EN12354_3_ANNEX_F_VOLUME,
        frequencies=ref.EN12354_3_ANNEX_F_BANDS,
        bands="octave",
    )


@register(
    "Building prediction & uncertainty",
    "EN 12354-3:2000 Annex F",
    "Facade airborne prediction (R'tr,s,w / D2m,nT,w single numbers)",
)
def _chk_en12354_3_facade() -> Outcome:
    res = _en12354_3_annex_f()
    # Anchor on the digit-exact low bands and the single-number ratings.
    low_ok = np.allclose(
        np.asarray(res.r_prime)[:3], ref.EN12354_3_ANNEX_F_RPRIME_LOW, atol=0.05
    )
    nums_ok = (
        res.r_tr_s_w == ref.EN12354_3_ANNEX_F_RTRS_W
        and res.c_tr == ref.EN12354_3_ANNEX_F_CTR
        and res.d_2m_nt_w == ref.EN12354_3_ANNEX_F_D2MNT_W
    )
    return Outcome(
        expected=(
            f"R'tr,s,w {ref.EN12354_3_ANNEX_F_RTRS_W} "
            f"(Ctr {ref.EN12354_3_ANNEX_F_CTR}); D2m,nT,w {ref.EN12354_3_ANNEX_F_D2MNT_W} dB"
        ),
        computed=f"R'tr,s,w {res.r_tr_s_w} (Ctr {res.c_tr}); D2m,nT,w {res.d_2m_nt_w} dB",
        delta="0",
        passed=bool(low_ok and nums_ok),
    )


@register(
    "Building prediction & uncertainty",
    "EN 12354-4:2000 Annex G / Formula (2)",
    "Radiated LW of a wall+door segment (side 1, low bands)",
)
def _chk_en12354_4_radiated() -> Outcome:
    res = ph.building.radiated_sound_power(
        [
            ph.building.FacadeElement(
                name="wall",
                area=ref.EN12354_4_ANNEX_G_SEGMENT_AREA
                - ref.EN12354_4_ANNEX_G_DOOR_AREA,
                r=ref.EN12354_4_ANNEX_G_CONCRETE_R,
            ),
            ph.building.FacadeElement(
                name="door",
                area=ref.EN12354_4_ANNEX_G_DOOR_AREA,
                r=ref.EN12354_4_ANNEX_G_DOOR_R,
            ),
        ],
        lp_in=ref.EN12354_4_ANNEX_G_LP_IN,
        area=ref.EN12354_4_ANNEX_G_SEGMENT_AREA,
        c_d=ref.EN12354_4_ANNEX_G_CD,
        r_prime_cap=ref.EN12354_4_ANNEX_G_RPRIME_CAP,
        octave_bands=[int(f) for f in ref.EN12354_4_ANNEX_G_BANDS],
    )
    rp_ok = np.allclose(
        np.asarray(res.r_prime)[:3], ref.EN12354_4_ANNEX_G_SIDE1_RPRIME_LOW, atol=0.05
    )
    lw = np.asarray(res.l_w)[:2]
    exp = np.asarray(ref.EN12354_4_ANNEX_G_SIDE1_LW_LOW)
    out = numeric(0.0, float(np.max(np.abs(lw - exp))), 0.1, unit="dB", places=3)
    return Outcome(
        expected=f"LW 63/125 Hz {ref.EN12354_4_ANNEX_G_SIDE1_LW_LOW} dB (+/-0.1)",
        computed=f"LW {np.round(lw, 1).tolist()} dB",
        delta=out.delta,
        passed=bool(rp_ok and out.passed),
    )


@register(
    "Building prediction & uncertainty",
    "EN 12354-4:2000 Annex E / Table G.9",
    "Exterior level of all four Table G.9 reception cells",
)
def _chk_en12354_4_propagation() -> Outcome:
    # (width, height, distance, printed A'tot, side LWA, printed Lp).
    cells = [
        (
            *ref.EN12354_4_ANNEX_G_ATTENUATION[0],
            ref.EN12354_4_ANNEX_G_SIDE1_LWA,
            ref.EN12354_4_ANNEX_G_LP_SIDE1_D5,
        ),
        (
            *ref.EN12354_4_ANNEX_G_ATTENUATION[1],
            ref.EN12354_4_ANNEX_G_SIDE1_LWA,
            ref.EN12354_4_ANNEX_G_LP_SIDE1_D25,
        ),
        (
            *ref.EN12354_4_ANNEX_G_ATTENUATION[2],
            ref.EN12354_4_ANNEX_G_SIDE4_LWA,
            ref.EN12354_4_ANNEX_G_LP_SIDE4_D5,
        ),
        (
            *ref.EN12354_4_ANNEX_G_ATTENUATION[3],
            ref.EN12354_4_ANNEX_G_SIDE4_LWA,
            ref.EN12354_4_ANNEX_G_LP_SIDE4_D25,
        ),
    ]
    worst = 0.0
    computed_lp = []
    for w, h, d, a_tot, lwa, lp_expected in cells:
        att = float(ph.building.outdoor_attenuation(w, h, d))
        lp = float(ph.building.outdoor_level(lwa, att))
        worst = max(worst, abs(att - a_tot), abs(lp - lp_expected))
        computed_lp.append(lp)
    return Outcome(
        expected="Lp 36,6 / 28,5 / 44,6 / 37,3 dB (+/-0,05)",
        computed="Lp " + " / ".join(f"{v:.1f}" for v in computed_lp) + " dB",
        delta=f"{worst:.3f} dB",
        passed=worst <= 0.05,
    )


@register(
    "Building prediction & uncertainty",
    "ISO 12999-1:2020 Table 2",
    "Airborne band uncertainty, situation A @ 1 kHz",
)
def _chk_iso12999_table2_band() -> Outcome:
    res = ph.building.band_uncertainty("airborne", "A")
    idx = list(res.frequencies).index(1000)
    computed = float(res.uncertainties[idx])
    return numeric(
        ref.ISO12999_1_TABLE2_AIRBORNE_A_1000HZ, computed, 1e-9, unit="dB", places=3
    )


@register(
    "Building prediction & uncertainty",
    "ISO 12999-1:2020 Annex B, Table B.2",
    "One-decimal single numbers Rw / Rw+C50-5000 / Rw+Ctr,50-5000",
)
def _chk_iso12999_annex_b_values() -> Outcome:
    res = ph.building.weighted_rating_extended(
        ref.ISO12999_1_ANNEX_B_RI,
        ref.ISO12999_1_ANNEX_B_FREQ,
        one_decimal=True,
    )
    assert res.c_50_5000 is not None
    assert res.ctr_50_5000 is not None
    rw = float(res.rating)
    rw_c = rw + float(res.c_50_5000)
    rw_ctr = rw + float(res.ctr_50_5000)
    ok = (
        abs(rw - ref.ISO12999_1_ANNEX_B_RW) <= 1e-9
        and abs(rw_c - ref.ISO12999_1_ANNEX_B_RW_C50_5000) <= 1e-9
        and abs(rw_ctr - ref.ISO12999_1_ANNEX_B_RW_CTR50_5000) <= 1e-9
    )
    return Outcome(
        expected=(
            f"{ref.ISO12999_1_ANNEX_B_RW} / {ref.ISO12999_1_ANNEX_B_RW_C50_5000}"
            f" / {ref.ISO12999_1_ANNEX_B_RW_CTR50_5000} dB"
        ),
        computed=f"{rw:.1f} / {rw_c:.1f} / {rw_ctr:.1f} dB",
        delta=f"{rw - ref.ISO12999_1_ANNEX_B_RW:+.2f} dB",
        passed=ok,
    )


@register(
    "Building prediction & uncertainty",
    "ISO 12999-1:2020 Annex B, Formulae (B.2)/(B.6)",
    "Single-number uncertainties (uncorrelated 0,6/0,8; correlated u(Rw) 1,9)",
)
def _chk_iso12999_annex_b_uncertainties() -> Outcome:
    from phonometry.building.measurement.insulation import (
        _SPECTRUM1_50_5000,
        _SPECTRUM2_50_5000,
    )

    ri = np.asarray(ref.ISO12999_1_ANNEX_B_RI, dtype=float)
    ui = np.asarray(ref.ISO12999_1_ANNEX_B_UI, dtype=float)
    u_c = float(
        ph.building.single_number_uncertainty_uncorrelated(
            ui, np.asarray(_SPECTRUM1_50_5000, dtype=float) - ri
        )
    )
    u_ctr = float(
        ph.building.single_number_uncertainty_uncorrelated(
            ui, np.asarray(_SPECTRUM2_50_5000, dtype=float) - ri
        )
    )
    up = ph.building.weighted_rating_extended(
        ri + ui, ref.ISO12999_1_ANNEX_B_FREQ, one_decimal=True
    ).rating
    down = ph.building.weighted_rating_extended(
        ri - ui, ref.ISO12999_1_ANNEX_B_FREQ, one_decimal=True
    ).rating
    u_rw = (float(up) - float(down)) / 2.0
    ok = (
        round(u_c, 1) == ref.ISO12999_1_ANNEX_B_U_UNCORR_C
        and round(u_ctr, 1) == ref.ISO12999_1_ANNEX_B_U_UNCORR_CTR
        and abs(u_rw - ref.ISO12999_1_ANNEX_B_U_CORR_RW) <= 1e-9
    )
    return Outcome(
        expected=(
            f"u_uncorr {ref.ISO12999_1_ANNEX_B_U_UNCORR_C} / "
            f"{ref.ISO12999_1_ANNEX_B_U_UNCORR_CTR} dB; "
            f"u_corr(Rw) {ref.ISO12999_1_ANNEX_B_U_CORR_RW} dB"
        ),
        computed=f"{u_c:.2f} / {u_ctr:.2f} dB; {u_rw:.2f} dB",
        delta=f"{u_rw - ref.ISO12999_1_ANNEX_B_U_CORR_RW:+.2f} dB",
        passed=ok,
    )


@register(
    "Building prediction & uncertainty",
    "ISO 12999-1:2020 Clause 8 / Table 8",
    "Expanded uncertainty U = 1.96 u (95 % two-sided, Rw sit. A)",
)
def _chk_iso12999_expanded() -> Outcome:
    u = ref.ISO12999_1_RW_A_STANDARD_UNCERTAINTY
    expected = ref.ISO12999_1_COVERAGE_K_95 * u
    computed = float(ph.building.insulation_expanded_uncertainty(u, coverage=0.95))
    return numeric(expected, computed, 1e-9, unit="dB", places=6)


@register(
    "Building prediction & uncertainty",
    "ISO 12999-2:2020 Table 4 / Formula (1)",
    "Absorption coefficient +/-U (k=2), reproducibility, 20 x 1/3-oct bands",
)
def _chk_iso12999_2_table4() -> Outcome:
    res = ph.materials.sound_absorption_coefficient_uncertainty(
        ref.ISO12999_2_TABLE4_ALPHA_S, ref.ISO12999_2_TABLE4_FREQ, confidence=0.95
    )
    got = res.reported_expanded_uncertainty
    expected = np.asarray(ref.ISO12999_2_TABLE4_U_K2, dtype=float)
    ok = bool(np.array_equal(got, expected))
    return Outcome(
        expected=f"U(k=2) = {expected.tolist()}",
        computed=f"U(k=2) = {got.tolist()}",
        delta="exact",
        passed=ok,
    )


@register(
    "Building prediction & uncertainty",
    "ISO 12999-2:2020 Table 5 / Formula (4)",
    "Practical coefficient +/-U (k=2), reproducibility, 5 octave bands",
)
def _chk_iso12999_2_table5() -> Outcome:
    res = ph.materials.practical_coefficient_uncertainty(
        ref.ISO12999_2_TABLE5_ALPHA_P, ref.ISO12999_2_TABLE5_FREQ
    )
    got = res.reported_expanded_uncertainty
    expected = np.asarray(ref.ISO12999_2_TABLE5_U_K2, dtype=float)
    ok = bool(np.array_equal(got, expected))
    return Outcome(
        expected=f"U(k=2) = {expected.tolist()}",
        computed=f"U(k=2) = {got.tolist()}",
        delta="exact",
        passed=ok,
    )


@register(
    "Building prediction & uncertainty",
    "ISO 12999-2:2020 Clause 7, Examples 1/2",
    "Single-number U (k=2): alpha_w and DLalpha,NRD",
)
def _chk_iso12999_2_single_numbers() -> Outcome:
    u_aw = float(
        ph.materials.weighted_coefficient_uncertainty(
            ref.ISO12999_2_ALPHA_W_EXAMPLE
        ).reported_expanded_uncertainty[0]
    )
    u_dl = float(
        ph.materials.single_number_rating_uncertainty(
            ref.ISO12999_2_DLALPHA_EXAMPLE
        ).reported_expanded_uncertainty[0]
    )
    ok = u_aw == ref.ISO12999_2_ALPHA_W_U_K2 and u_dl == ref.ISO12999_2_DLALPHA_U_K2
    return Outcome(
        expected=f"alpha_w +/-{ref.ISO12999_2_ALPHA_W_U_K2}, "
        f"DLalpha +/-{ref.ISO12999_2_DLALPHA_U_K2} dB",
        computed=f"alpha_w +/-{u_aw}, DLalpha +/-{u_dl} dB",
        delta="exact",
        passed=ok,
    )
