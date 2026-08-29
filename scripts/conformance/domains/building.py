#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Domain 6 - Room & building acoustics, as measured.

Everything the standards ask a laboratory or a field team to measure and rate:
reverberation time and its evaluation ranges (ISO 3382-2, ISO 18233), the
single-number ratings of airborne and impact insulation (ISO 717-1/-2),
absorption in a reverberation room (ISO 354), speech level distribution in
open-plan offices (ISO 3382-3), the field and laboratory insulation methods
(ISO 10140, ISO 16283, ISO 10052, ISO 15186-1, ISO 16251-1) and the flanking
measurement of ISO 10848.

With them come the structural quantities the prediction models need as inputs
and that have their own measurement standards: dynamic stiffness of resilient
materials (EN 29052-1), mechanical mobility and receptance (ISO 7626), dynamic
transfer stiffness of resilient elements (ISO 10846), sound power from surface
vibration (ISO/TS 7849) and structure-borne sound power of building equipment
(EN 15657, EN 12354-5). Suspended-ceiling plenum flanking is anchored on
published ASTM E1414 laboratory reports.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import reference_data as ref
from scipy import signal as sg

import phonometry as ph

from ..registry import Outcome, numeric, record, register
from .levels import _FS

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from phonometry.building import InSituElementResult

_A60 = 6.0 * math.log(10.0)


def _exponential_ir(t60: float, seconds: float) -> np.ndarray:
    t = np.arange(round(seconds * _FS)) / _FS
    return np.asarray(np.exp(-0.5 * _A60 * t / t60))


@register(
    "Room & building acoustics",
    "ISO 3382-2:2008 5.3.3",
    "T30 from a synthetic exponential decay (T=1.0 s)",
)
def _chk_room_t30() -> Outcome:
    t60 = 1.0
    res = ph.room.room_parameters(_exponential_ir(t60, 3.0 * t60), _FS, limits=None)
    return numeric(t60, float(res.t30[0]), 0.01, unit="s", rel=True, places=4)


@register(
    "Room & building acoustics",
    "ISO 18233:2006 (swept-sine method)",
    "Sweep deconvolution recovers a known IIR response",
)
def _chk_iso18233_sweep_deconvolution() -> Outcome:
    # Closed-form identity: an exponential sweep through a known Butterworth
    # band-pass, deconvolved back, must reproduce the filter's freqz response.
    b, a = sg.butter(4, [200.0, 2000.0], btype="band", fs=_FS)
    x = ph.room.sweep_signal(_FS, 20.0, 20000.0, 2.0)
    y = sg.lfilter(b, a, x)
    ir = np.asarray(ph.room.impulse_response(y, x, _FS, length=16384))
    freqs = np.fft.rfftfreq(ir.size, d=1.0 / _FS)
    h_est = np.fft.rfft(ir)
    _, h_true = sg.freqz(b, a, worN=freqs, fs=_FS)
    mask = (freqs >= 300.0) & (freqs <= 1500.0)
    worst = float(
        np.max(
            np.abs(
                20.0 * np.log10(np.abs(h_est[mask]))
                - 20.0 * np.log10(np.abs(h_true[mask]))
            )
        )
    )
    # Linear deconvolution is exact in-band up to windowing/regularisation
    # leakage; 0.1 dB is the demonstrated in-band bound (tests/room/test_impulse_response.py)
    # with the same 300-1500 Hz evaluation band, well inside the sweep edges.
    return numeric(
        0.0,
        worst,
        0.1,
        unit="dB",
        places=4,
        expected_label="0 dB in-band error (+/-0.1 dB)",
    )


@register(
    "Room & building acoustics",
    "ISO 717-1 Annex C, Table C.1",
    "Weighted sound reduction index Rw (C;Ctr)",
)
def _chk_iso717_rw() -> Outcome:
    exp = ref.ISO717_1_ANNEX_C_EXPECTED
    res = ph.building.weighted_rating(ref.ISO717_1_ANNEX_C_R)
    # Three integers read off one printed table, compared exactly: the standard
    # does the rounding itself, so a tolerance here would be a second opinion
    # about a number the standard has already settled. The unfavourable sum is
    # a second quantity, not a deviation, so it travels as one.
    return record(
        {"Rw": exp["rw"], "C": exp["c"], "Ctr": exp["ctr"]},
        {"Rw": res.rating, "C": res.c, "Ctr": res.ctr},
        label=f"Rw {exp['rw']} (C {exp['c']}; Ctr {exp['ctr']})",
        computed_label=(
            f"Rw {res.rating} (C {res.c}; Ctr {res.ctr}), "
            f"unfavourable sum {res.unfavourable_sum:.1f} dB"
        ),
    )


@register(
    "Room & building acoustics",
    "ISO 717-1:2020 Annex C, Table C.2",
    "Enlarged range 50-5000 Hz: Rw (C; Ctr; C50-5000; Ctr,50-5000)",
)
def _chk_iso717_1_extended() -> Outcome:
    exp = ref.ISO717_1_ANNEX_C2_EXPECTED
    freqs = [
        50,
        63,
        80,
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
        5000,
    ]
    res = ph.building.weighted_rating_extended(ref.ISO717_1_ANNEX_C2_R_50_5000, freqs)
    ok = (
        res.rating == exp["rw"]
        and res.c == exp["c"]
        and res.ctr == exp["ctr"]
        and res.c_50_5000 == exp["c_50_5000"]
        and res.ctr_50_5000 == exp["ctr_50_5000"]
    )
    # Not a `record` check, unlike its Table C.1 sibling above: the two
    # enlarged-range terms are optional in the result type, and a `None` there
    # is a different failure from a wrong number. Converting it means giving
    # that its own verdict first.
    return Outcome(
        expected=(
            f"Rw {exp['rw']} (C {exp['c']}; Ctr {exp['ctr']}; "
            f"C50-5000 {exp['c_50_5000']}; Ctr,50-5000 {exp['ctr_50_5000']})"
        ),
        computed=(
            f"Rw {res.rating:g} (C {res.c:g}; Ctr {res.ctr:g}; "
            f"C50-5000 {res.c_50_5000:g}; Ctr,50-5000 {res.ctr_50_5000:g})"
        ),
        delta="exact",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 717-2 Annex C, Table C.1",
    "Weighted impact sound pressure level Ln,w (CI)",
)
def _chk_iso717_2_lnw() -> Outcome:
    # Worked example: Ln,w = 79 dB, CI = -11 dB, unfavourable sum 28,0 dB.
    # CI = -11 is the ISO 717-2:2013 Annex C print; the 2020 reprint of this
    # example is internally inconsistent with its own A.2.1 (it sums the
    # 3 150 Hz band into Ln,sum and prints CI = -10).
    # Integer ratings and CI must match exactly; the unfavourable sum is a
    # one-decimal tabulated intermediate, so 1e-9 = exact up to float noise.
    exp = ref.ISO717_2_ANNEX_C1_EXPECTED
    res = ph.building.weighted_impact_rating(ref.ISO717_2_ANNEX_C1_LN)
    sum_ok = abs(res.unfavourable_sum - exp["unfavourable_sum"]) <= 1e-9
    ok = res.rating == exp["ln_w"] and res.ci == exp["ci"] and sum_ok
    return Outcome(
        expected=f"Ln,w {exp['ln_w']} (CI {exp['ci']}; sum {exp['unfavourable_sum']:.1f} dB)",
        computed=f"Ln,w {res.rating} (CI {res.ci}; sum {res.unfavourable_sum:.1f} dB)",
        delta=f"{res.rating - exp['ln_w']:+d} dB",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 717-2 Annex C, Table C.1 (covered)",
    "Weighted impact level of the floor WITH covering Ln,w (CI)",
)
def _chk_iso717_2_lnw_covered() -> Outcome:
    exp = ref.ISO717_2_ANNEX_C1_COVERED_EXPECTED
    res = ph.building.weighted_impact_rating(ref.ISO717_2_ANNEX_C1_COVERED_LN)
    sum_ok = abs(res.unfavourable_sum - exp["unfavourable_sum"]) <= 1e-9
    ok = res.rating == exp["ln_w"] and res.ci == exp["ci"] and sum_ok
    return Outcome(
        expected=f"Ln,w {exp['ln_w']} (CI {exp['ci']}; sum {exp['unfavourable_sum']:.1f} dB)",
        computed=f"Ln,w {res.rating} (CI {res.ci}; sum {res.unfavourable_sum:.1f} dB)",
        delta=f"{res.rating - exp['ln_w']:+d} dB",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 717-2 Annex C, Table C.2",
    "Floor-covering improvement ΔLw and CI,Δ (Formulae (2)/(A.4); CI,Δ from"
    " the normative Table 4 floor, not the 2020 print's misprinted C.2 chain)",
)
def _chk_iso717_2_c2_improvement() -> Outcome:
    dlw = ph.building.weighted_impact_improvement(ref.ISO717_2_ANNEX_C2_DELTA_L)
    ci_d = ph.building.impact_improvement_adaptation_term(ref.ISO717_2_ANNEX_C2_DELTA_L)
    ok = (
        dlw == ref.ISO717_2_ANNEX_C2_DELTA_LW and ci_d == ref.ISO717_2_ANNEX_C2_CI_DELTA
    )
    return Outcome(
        expected=(
            f"ΔLw {ref.ISO717_2_ANNEX_C2_DELTA_LW} dB; "
            f"CI,Δ {ref.ISO717_2_ANNEX_C2_CI_DELTA} dB (Table 4 reference floor)"
        ),
        computed=f"ΔLw {dlw} dB; CI,Δ {ci_d} dB",
        delta=f"{dlw - ref.ISO717_2_ANNEX_C2_DELTA_LW:+d} dB",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 354:2003 Eq. 5/8",
    "Sabine inversion recovers absorption area",
)
def _chk_iso354_absorption() -> Outcome:
    v, c, t = 200.0, 343.0, 3.5
    expected = 55.3 * v / (c * t)
    computed = float(
        np.asarray(ph.materials.absorption_area(t, v, speed_of_sound=c))[()]
    )
    return numeric(expected, computed, 1e-9, unit="m^2", places=6)


@register(
    "Room & building acoustics",
    "ISO 3382-3:2012 Clause 6.2",
    "Open-plan spatial decay rate D2,S (-6 dB/doubling)",
)
def _chk_open_plan_d2s() -> Outcome:
    r = np.array([2.0, 4.0, 8.0, 16.0])
    lp = 70.0 - 6.0 * np.log2(r)
    sti = 0.6 - 0.02 * r
    res = ph.room.open_plan_metrics(r, lp, sti)
    return numeric(6.0, float(res.d2s), 1e-9, unit="dB", places=6)


@register(
    "Room & building acoustics",
    "ISO 16283-3:2016 Clause 3.12",
    "Facade R'45 isolates the -1.5 dB incidence correction (S=A)",
)
def _chk_facade_r45() -> Outcome:
    # With S = A the 10 lg(S/A) coupling term vanishes, so R' = L1,s - L2 - 1,5.
    n = 3
    res = ph.building.facade_insulation(
        np.full(n, 55.0),
        np.full(n, ref.ISO16283_3_R45_RECEIVE_LEVEL_DB),
        np.full(n, ref.ISO16283_3_R45_REVERB_TIME_S),
        area=ref.ISO16283_3_R45_AREA_M2,
        volume=ref.ISO16283_3_R45_VOLUME_M3,
        surface_level=np.full(n, ref.ISO16283_3_R45_SURFACE_LEVEL_DB),
    )
    assert res.r_prime is not None
    # The expected value is rebuilt from its components, so the -1,5 dB
    # oblique-incidence correction constant is exercised explicitly.
    expected = (
        ref.ISO16283_3_R45_SURFACE_LEVEL_DB
        - ref.ISO16283_3_R45_RECEIVE_LEVEL_DB
        - ref.ISO16283_3_R45_LOUDSPEAKER_CORRECTION_DB
    )
    assert expected == ref.ISO16283_3_R45_EXPECTED_DB
    computed = float(np.asarray(res.r_prime)[0])
    return numeric(expected, computed, 1e-9, unit="dB", places=6)


@register(
    "Room & building acoustics",
    "ISO 10140-2:2010 Formula (2)",
    "Lab airborne R on the ISO 717-1 reference shape -> Rw = 54",
)
def _chk_lab_airborne_rw() -> Outcome:
    # S = A (A = 0,16*50/0,8 = 10 = area) => R = L1 - L2 = the reference curve.
    ref_r = np.asarray(ref.ISO10140_2_REF_AIRBORNE_R, dtype=float)
    res = ph.building.lab_airborne_insulation(
        np.full(16, 90.0), 90.0 - ref_r, np.full(16, 0.8), area=10.0, volume=50.0
    )
    assert res.rating is not None
    # R lands exactly on the reference; guard that before reading the rating.
    on_curve = bool(np.allclose(np.asarray(res.r), ref_r))
    expected = ref.ISO10140_2_REF_AIRBORNE_RW
    return Outcome(
        expected=f"Rw {expected} dB",
        computed=f"Rw {res.rating.rating} dB",
        delta=f"{res.rating.rating - expected:+d} dB",
        passed=on_curve and res.rating.rating == expected,
    )


@register(
    "Room & building acoustics",
    "ISO 10140-5:2010+A1 Annex B, Table B.1",
    "Reference elements end-to-end: printed Rw (C; Ctr) of all three",
)
def _chk_iso10140_5_reference_elements() -> Outcome:
    rows = [
        (ref.ISO10140_5_B1_HEAVY_WALL_R, ref.ISO10140_5_B1_HEAVY_WALL_RATING),
        (ref.ISO10140_5_B1_HEAVY_FLOOR_R, ref.ISO10140_5_B1_HEAVY_FLOOR_RATING),
        (ref.ISO10140_5_B1_LIGHT_WALL_R, ref.ISO10140_5_B1_LIGHT_WALL_RATING),
    ]
    computed = []
    ok = True
    for r, expected in rows:
        # S = A (10 m2) so the ISO 10140-2 chain returns R = L1 - L2 exactly.
        res = ph.building.lab_airborne_insulation(
            np.full(16, 90.0),
            90.0 - np.asarray(r, dtype=float),
            np.full(16, 0.8),
            area=10.0,
            volume=50.0,
        )
        assert res.rating is not None
        got = (res.rating.rating, res.rating.c, res.rating.ctr)
        computed.append(got)
        ok = ok and got == expected
    return Outcome(
        expected="Rw(C;Ctr) = 53(-1;-5) / 52(-1;-5) / 33(-1;-2)",
        computed=" / ".join(f"{rw}({c};{ctr})" for rw, c, ctr in computed),
        delta="exact",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 10140-5:2010+A1 Annex C, Table C.1",
    "Reference floors end-to-end: printed Ln,t,r,0,w (CI) of both",
)
def _chk_iso10140_5_reference_floors() -> Outcome:
    rows = [
        (ref.ISO10140_5_C1_FLOOR_C1C2_LN, ref.ISO10140_5_C1_FLOOR_C1C2_RATING),
        (ref.ISO10140_5_C1_FLOOR_C3_LN, ref.ISO10140_5_C1_FLOOR_C3_RATING),
    ]
    computed = []
    ok = True
    for ln, expected in rows:
        # A = A0 (V = 31,25 m3, T = 0,5 s) so Ln equals the receiving level.
        res = ph.building.lab_impact_insulation(
            np.asarray(ln, dtype=float), np.full(16, 0.5), volume=31.25
        )
        assert res.rating is not None
        got = (res.rating.rating, res.rating.ci)
        computed.append(got)
        ok = ok and got == expected
    return Outcome(
        expected="Ln,t,r,0,w(CI) = 72(0) / 75(-3)",
        computed=" / ".join(f"{lnw}({ci})" for lnw, ci in computed),
        delta="exact",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 15186-1:2000 Formula (7)",
    "Intensity RI on the ISO 717-1 reference shape -> RI,w = 30",
)
def _chk_intensity_ri_rw() -> Outcome:
    # Hand-computed scalar anchor pinning the Formula (7) constants (the
    # curve construction below inverts the same formula, so -6 dB and
    # 10 lg(Sm/S) would cancel there): Lp1 = 80, LIn = 40, Sm = S
    # -> RI = 80 - 6 - 40 - 0 = 34 dB exactly.
    scalar = ph.building.intensity_sound_reduction(
        [80.0], [40.0], measurement_area=10.0, area=10.0
    )
    scalar_ok = abs(float(scalar.r_i[0]) - 34.0) <= 1e-9
    # Choose LIn so that RI = Lp1 - 6 - [LIn + 10 lg(Sm/S)] lands exactly on
    # the ISO 717-1 Annex C curve; the rating engine must then return Rw = 30.
    ref_ri = np.asarray(ref.ISO15186_1_REF_RI, dtype=float)
    lp1, sm, s = ref.ISO15186_1_REF_LP1, ref.ISO15186_1_REF_SM, ref.ISO15186_1_REF_S
    lin = lp1 - 6.0 - 10.0 * np.log10(sm / s) - ref_ri
    res = ph.building.intensity_sound_reduction(
        np.full(16, lp1), lin, measurement_area=sm, area=s
    )
    assert res.rating is not None
    on_curve = bool(np.allclose(np.asarray(res.r_i), ref_ri))
    expected = ref.ISO15186_1_REF_RIW
    return Outcome(
        expected=f"RI,w {expected} dB (scalar anchor RI = 34 dB)",
        computed=f"RI,w {res.rating.rating} dB (RI = {float(scalar.r_i[0]):g} dB)",
        delta=f"{res.rating.rating - expected:+d} dB",
        passed=scalar_ok and on_curve and res.rating.rating == expected,
    )


@register(
    "Room & building acoustics",
    "ISO 15186-1:2000 Annex B, Table B.1",
    "Adaptation term Kc: all 21 printed rows; (B.1) reduces to (B.2)",
)
def _chk_intensity_kc_annexb() -> Outcome:
    # The printed Table B.1 (21 one-third-octave rows, one decimal) is the
    # independent oracle; additionally Formula (B.1) with Sb2 = 117 m²,
    # V2 = 81 m³, c = 340 m/s must reduce to (B.2) Kc = 10 lg(1 + 61,4/f).
    b2 = ph.building.adaptation_term_kc(ref.ISO15186_1_KC_BANDS)
    b1 = ph.building.adaptation_term_kc(
        ref.ISO15186_1_KC_BANDS, boundary_area=117.0, volume=81.0
    )
    printed = np.asarray(ref.ISO15186_1_KC_B1_PRINTED, dtype=float)
    worst = float(np.max(np.abs(b2 - printed)))
    delta = float(np.max(np.abs(b1 - b2)))
    passed = worst <= 0.05 and delta <= 1e-3
    return Outcome(
        expected="max abs(Kc - Table B.1) <= 0,05 dB (1 dp print)",
        computed=f"{worst:.3f} dB (B.1 vs B.2: {delta:.2e} dB)",
        delta=f"{worst:.3f} dB",
        passed=passed,
    )


@register(
    "Room & building acoustics",
    "ISO 10052:2021 Clause 3.6",
    "Survey R' applies the V/7,5 minimum-area rule",
)
def _chk_survey_rprime_area_rule() -> Outcome:
    # V/7,5 = 120/7,5 = 16 m^2 > S = 5 m^2, so the larger value replaces S.
    # With k = 0 (T = T0), R' = D + 10 lg(16 * T0 / (0,16 V)).
    v, s, d = 120.0, 5.0, 30.0
    res = ph.building.survey_airborne_insulation(
        np.full(5, 70.0), np.full(5, 40.0), np.zeros(5), volume=v, area=s
    )
    assert res.r_prime is not None
    s_eff = v / 7.5
    expected = d + 10.0 * np.log10(s_eff * 0.5 / (0.16 * v))
    return numeric(expected, float(res.r_prime[0]), 1e-9, unit="dB", places=6)


@register(
    "Room & building acoustics",
    "ISO 10052:2021 Clause 3.16",
    "Service-equipment LXY is the 3-position energy average",
)
def _chk_survey_service_equipment() -> Outcome:
    # Energy average of 35 / 30 / 32 dB(A), then standardized by k.
    levels = [35.0, 30.0, 32.0]
    res = ph.building.survey_service_equipment_level(levels, 3.0, volume=50.0)
    expected = 10.0 * np.log10(sum(10.0 ** (0.1 * x) for x in levels) / 3.0)
    return numeric(expected, float(np.asarray(res.l_xy)[()]), 1e-9, unit="dB", places=6)


@register(
    "Room & building acoustics",
    "ISO 10052:2021 Table 4",
    "Reverberation-index estimate (35 <= V < 60, type g)",
)
def _chk_survey_reverberation_estimate() -> Outcome:
    # Table 4 row 'g' for 35 <= V < 60: 4,5 / 5 / 5,5 / 5,5 / 5,5 dB.
    expected = [4.5, 5.0, 5.5, 5.5, 5.5]
    got = np.asarray(ph.building.estimate_reverberation_index(50.0, "g"), dtype=float)
    ok = bool(np.array_equal(got, expected))
    return Outcome(
        expected=f"k = {expected} dB",
        computed=f"k = {got.tolist()} dB",
        delta="exact",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 717-2:2020 Table 4 / Clause 5.2",
    "Reference-floor weighted level Ln,r,0,w and CI (ISO 16251-1 ΔLw anchor)",
)
def _chk_iso717_2_reference_floor() -> Outcome:
    res = ph.building.weighted_impact_rating(ref.ISO717_2_REFERENCE_FLOOR_LN_R0)
    ok = (
        res.rating == ref.ISO717_2_REFERENCE_FLOOR_LN_R0_W
        and res.ci == ref.ISO717_2_REFERENCE_FLOOR_CI
    )
    return Outcome(
        expected=f"Ln,r,0,w = {ref.ISO717_2_REFERENCE_FLOOR_LN_R0_W} dB, "
        f"CI = {ref.ISO717_2_REFERENCE_FLOOR_CI} dB",
        computed=f"Ln,r,0,w = {res.rating} dB, CI = {res.ci} dB",
        delta="exact",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 16251-1:2014 / ISO 717-2 Formula (2)",
    "Floor-covering ΔLw: zero improvement gives ΔLw = 0",
)
def _chk_iso16251_zero_improvement() -> Outcome:

    dlw = ph.building.weighted_impact_improvement(np.zeros(16))
    return Outcome(
        expected="ΔLw = 0 dB (ΔL = 0 -> Ln,r = Ln,r,0)",
        computed=f"ΔLw = {dlw} dB",
        delta="exact",
        passed=(dlw == 0),
    )


@register(
    "Room & building acoustics",
    "ISO 16251-1 / ISO 717-2 (Foret et al. 2011, carpet)",
    "Measured textile-carpet improvement rates to ΔLw = 29 dB",
)
def _chk_iso16251_carpet_foret2011() -> Outcome:
    # Real measurement: the Delta-L spectrum digitized from Figure 4 (vector
    # chart) of Foret, Chene & Guigou-Carter, Forum Acusticum 2011, run through
    # the full ISO 16251-1 improvement path (a flat bare-plate level minus the
    # measured Delta-L reconstructs the specimen level), then rated per ISO
    # 717-2 on the 100-3150 Hz sub-range. See tests/reference_data/ for the
    # provenance and the +/- 0,5 dB digitization tolerance.
    bare = np.full(len(ref.FORET2011_CARPET_FREQ), 100.0)
    delta_l = np.asarray(ref.FORET2011_CARPET_ISO16251_DELTA_L, dtype=float)
    res = ph.building.impact_improvement(
        bare, bare - delta_l, ref.FORET2011_CARPET_FREQ
    )
    dlw = res.delta_lw
    return Outcome(
        expected=f"ΔLw = {ref.FORET2011_CARPET_ISO16251_DELTA_LW} dB (paper, ISO 16251-1)",
        computed=f"ΔLw = {dlw} dB",
        delta=f"{(dlw or 0) - ref.FORET2011_CARPET_ISO16251_DELTA_LW:+d} dB",
        passed=(dlw == ref.FORET2011_CARPET_ISO16251_DELTA_LW),
    )


@register(
    "Room & building acoustics",
    "ISO 10848-1:2006 Formula (14)",
    "Flanking Kij (simplified) matches closed form",
)
def _chk_iso10848_kij_simplified() -> Outcome:
    # No worked example in the standard; anchor on the closed form recomputed
    # independently here (delta is "exact" to keep the report byte-stable).
    res = ph.building.vibration_reduction_index(
        [ref.ISO10848_KIJ_DBAR],
        ref.ISO10848_KIJ_LIJ,
        ref.ISO10848_KIJ_AREA,
        ref.ISO10848_KIJ_AREA,
    )
    computed = float(res.k_ij[0])
    expected = ref.ISO10848_KIJ_DBAR + 10.0 * math.log10(
        ref.ISO10848_KIJ_LIJ / math.sqrt(ref.ISO10848_KIJ_AREA**2)
    )
    return Outcome(
        expected=f"Kij = {expected:.4f} dB",
        computed=f"Kij = {computed:.4f} dB",
        delta="exact",
        passed=abs(computed - expected) < 1e-9,
    )


@register(
    "Room & building acoustics",
    "ISO 10848-1:2006 Formula (12)",
    "Flanking equivalent absorption length aj at f_ref",
)
def _chk_iso10848_absorption_length() -> Outcome:
    a = ph.building.equivalent_absorption_length(
        ref.ISO10848_ABS_AREA,
        ref.ISO10848_ABS_TS,
        [1000.0],
        speed_of_sound=ref.ISO10848_ABS_C0,
    )
    computed = float(a[0])
    # aj at f = f_ref (sqrt(f_ref/f) = 1): aj = 2,2·π²·S/(Ts·c0).
    expected = (
        2.2
        * math.pi**2
        * ref.ISO10848_ABS_AREA
        / (ref.ISO10848_ABS_TS * ref.ISO10848_ABS_C0)
    )
    return Outcome(
        expected=f"aj = {expected:.4f} m",
        computed=f"aj = {computed:.4f} m",
        delta="exact",
        passed=abs(computed - expected) < 1e-9,
    )


@register(
    "Room & building acoustics",
    "ISO 10848-1:2006 Clause 7.3.1",
    "Flanking total loss factor η = 2,2/(f·Ts)",
)
def _chk_iso10848_loss_factor() -> Outcome:
    eta = ph.building.total_loss_factor([1000.0], [0.5])
    computed = float(eta[0])
    expected = 2.2 / (1000.0 * 0.5)
    return Outcome(
        expected=f"η = {expected:.4f}",
        computed=f"η = {computed:.4f}",
        delta="exact",
        passed=abs(computed - expected) < 1e-12,
    )


@register(
    "Room & building acoustics",
    "ISO 12354-1:2017 Formula (20) vs Hopkins Eq. 2.201 (6 mm glass)",
    "Flanking critical frequency (c0²/1,8·cL·h) vs plate coincidence "
    "(c0²/2π · sqrt(m''/B'))",
)
def _chk_flanking_critical_frequency() -> Outcome:
    # The 1,8 constant rounds 2π/√12, so for a plate whose bending stiffness
    # and mass are mutually consistent the two independent formulas must
    # agree to within that rounding (< 1 %).
    e, rho, nu, h, c0 = 6.2e10, 2500.0, 0.24, 0.006, 343.0
    c_l = math.sqrt(e / (rho * (1.0 - nu**2)))
    fc_flank = ph.building.critical_frequency(c_l, h, speed_of_sound=c0)
    fc_coinc = ph.vibration.coincidence_frequency(
        rho * h,
        ph.vibration.plate_bending_stiffness(e, h, nu),
        speed_of_sound=c0,
    )
    return numeric(fc_coinc, fc_flank, 0.01, rel=True, unit="Hz", places=1)


# --- Dynamic stiffness of resilient materials (EN 29052-1:1992) ---
@register(
    "Room & building acoustics",
    "EN 29052-1:1992 Formula 4",
    "Apparent dynamic stiffness s't = 4π²·m't·fr²  (m't=200 kg/m², fr=25 Hz)",
)
def _chk_en29052_apparent() -> Outcome:
    computed = float(ph.materials.apparent_dynamic_stiffness(25.0, 200.0)) / 1e6
    expected = 4.0 * math.pi**2 * 200.0 * 25.0**2 / 1e6
    return numeric(expected, computed, 1e-6, unit="MN/m³", places=6)


@register(
    "Room & building acoustics",
    "EN 29052-1:1992 clause 8.2 NOTE",
    "Enclosed-gas stiffness s'a·d = 111 MN·mm/m³ (p₀=0,1 MPa, ε=0,9)",
)
def _chk_en29052_enclosed_gas() -> Outcome:
    # NOTE: s'a = 111/d MN/m3 for d in mm; the closed form gives 100/0,9 = 111.11.
    sa_mn = float(ph.materials.enclosed_gas_stiffness(0.020, 0.9)) / 1e6  # d = 20 mm
    return numeric(111.111111 / 20.0, sa_mn, 1e-4, unit="MN/m³", places=5)


@register(
    "Room & building acoustics",
    "EN 29052-1:1992 Formula 2",
    "Floating-floor natural frequency f0 = (1/2π)√(s'/m')  (s'=10 MN/m³, m'=100 kg/m²)",
)
def _chk_en29052_resonance() -> Outcome:
    computed = float(ph.materials.natural_frequency(10.0e6, 100.0))
    expected = math.sqrt(10.0e6 / 100.0) / (2.0 * math.pi)
    return numeric(expected, computed, 1e-6, unit="Hz", places=5)


# --- Mechanical mobility (ISO 7626-1:2011) ---
# Closed-form SDOF resonator (consistent with the ISO 7626-1 Table 1 / 3.1.2
# FRF definitions): m=2 kg, k=8000 N/m, c=5 N.s/m; f0 = sqrt(k/m)/(2pi).
_MOB_M, _MOB_K, _MOB_C = 2.0, 8000.0, 5.0
_MOB_F0 = math.sqrt(_MOB_K / _MOB_M) / (2.0 * math.pi)


@register(
    "Room & building acoustics",
    "ISO 7626-1:2011 Table 1 / 3.1.2",
    "Closed-form SDOF driving-point mobility peak mag(Y(f0)) = 1/c  (c=5 N·s/m)",
)
def _chk_iso7626_mobility_peak() -> Outcome:
    y0 = complex(ph.vibration.sdof_mobility(_MOB_F0, _MOB_M, _MOB_K, _MOB_C))
    return numeric(1.0 / _MOB_C, abs(y0), 1e-6, unit="m/(N·s)", places=6)


@register(
    "Room & building acoustics",
    "ISO 7626-1:2011 Table 1 / 3.1.2",
    "Closed-form SDOF static receptance H(0) = 1/k  (k=8000 N/m)",
)
def _chk_iso7626_static_receptance() -> Outcome:
    h = complex(ph.vibration.sdof_receptance(1e-6, _MOB_M, _MOB_K, _MOB_C))
    return numeric(1.0 / _MOB_K, h.real, 1e-6, unit="m/N", rel=True, places=8)


@register(
    "Room & building acoustics",
    "ISO 7626-1:2011 Table 1",
    "FRF reciprocity: impedance × mobility = 1  (at 37 Hz)",
)
def _chk_iso7626_reciprocity() -> Outcome:
    y = complex(ph.vibration.sdof_mobility(37.0, _MOB_M, _MOB_K, _MOB_C))
    z = complex(ph.vibration.convert_frf(y, 37.0, "mobility", "impedance"))
    return numeric(1.0, abs(z * y), 1e-9, expected_label="1 (= Z·Y)")


# --- Heavy and soft impact sources (ISO 16283-2 / JIS A 1418-2 / ISO 717-2) ---
#: ISO 717-2:2020 Table D.4 (printed p. 22): a field measurement in octave
#: bands, Li,Fmax at 63/125/250/500 Hz, rated LiA,Fmax = 55,350 66... = 55 dB.
_ISO717_2_D4_LEVELS = (65.3, 64.5, 58.0, 55.8)


@register(
    "Room & building acoustics",
    "ISO 717-2:2020 Table D.4",
    "A-weighted maximum impact level LiA,Fmax of the Annex D worked example",
)
def _chk_iso717_2_annex_d_rating() -> Outcome:
    res = ph.building.a_weighted_maximum_impact_level(_ISO717_2_D4_LEVELS)
    return (
        numeric(
            55.350_667,
            res.unrounded,
            1e-4,
            unit="dB",
            places=6,
            expected_label="55,350 66... dB (rated 55 dB)",
        )
        if res.rating == 55
        else Outcome(
            expected="55,350 66... dB (rated 55 dB)",
            computed=f"{res.unrounded:.6f} dB (rated {res.rating} dB)",
            delta=f"{res.unrounded - 55.350_667:+.3f} dB",
            passed=False,
        )
    )


@register(
    "Room & building acoustics",
    "ISO 16283-2:2020 Table A.1 / JIS A 1418-2:2019 Table A.2",
    "Rubber-ball impact force exposure level LFE, five octave bands",
)
def _chk_iso16283_2_rubber_ball_spectrum() -> Outcome:
    # ISO 16283-2 Table A.1, ISO 10140-5 Table F.1 and JIS A 1418-2 Table A.2
    # print the same five values; the tolerance band is +/-1,0 / 1,5 / 1,5 /
    # 2,0 / 2,0 dB, and a source on the nominal must conform in every band.
    freqs, lower, upper = ph.building.heavy_impact_source_limits("rubber_ball")
    nominal = 0.5 * (lower + upper)
    check = ph.building.check_heavy_impact_source(nominal)
    printed = "39,0 / 31,0 / 23,0 / 17,0 / 12,5 dB re 1 N"
    return Outcome(
        expected=f"{printed} at 31,5 to 500 Hz",
        computed=" / ".join(f"{v:g}".replace(".", ",") for v in nominal) + " dB re 1 N",
        delta=f"max |dev| {float(np.max(np.abs(check.deviation))):.3f} dB",
        passed=bool(
            check.passed
            and np.allclose(freqs, ph.building.HEAVY_IMPACT_OCTAVE_BANDS)
            and np.allclose(upper - lower, [2.0, 3.0, 3.0, 4.0, 4.0])
        ),
    )


@register(
    "Room & building acoustics",
    "ISO 16283-2:2020 Formulae (4), (5), (6)",
    "Standardized maximum impact level reduces to 10 lg(V/V0) at T = T0",
)
def _chk_iso16283_2_standardization_identity() -> Outcome:
    # Li,Fmax = 70 dB in a 100 m3 room at the reference T0 = 0,5 s: the Fast
    # correction term is identically zero, leaving 10 lg(100/50) = 3,0103 dB.
    res = ph.building.standardized_maximum_impact_level([70.0], 100.0, 0.5)
    return numeric(
        70.0 + 10.0 * math.log10(2.0),
        float(res.standardized[0]),
        1e-9,
        unit="dB",
        places=6,
        expected_label="73,0103 dB (= 70 + 10 lg(100/50))",
    )


# --- Suspended-ceiling plenum flanking (ASTM E1414/E413, ISO 140-9, Vigran) ---
#: Acoustic Laboratories Australia report ALA 16-091-4 (2016), tested to
#: ASTM E1414/E1414M-11a: the printed one-third-octave Dn,c of a 28 mm plaster
#: acoustic tile, 125 Hz to 4 kHz, rated CAC 34 in the report.
_ALA_DNC = (
    14.4,
    18.6,
    21.7,
    24.1,
    23.4,
    30.3,
    33.7,
    35.2,
    41.6,
    44.2,
    42.1,
    36.8,
    35.7,
    36.0,
    36.9,
    37.9,
)
#: Intertek report J7488.04-113-11-R0 (2019), tested to ASTM E1414: printed
#: Dn,c of ceiling planks, rated CAC 25 with a sum of deficiencies of 24 dB.
_INTERTEK_DNC = (8, 13, 15, 15, 19, 23, 24, 21, 23, 26, 26, 27, 29, 32, 34, 36)


@register(
    "Room & building acoustics",
    "ASTM E413-22 clause 5 (ASTM E1414 CAC)",
    "Ceiling attenuation class of two accredited E1414 test reports",
)
def _chk_astm_e413_cac() -> Outcome:
    ala = ph.building.ceiling_attenuation_class(_ALA_DNC)
    intertek = ph.building.ceiling_attenuation_class(_INTERTEK_DNC)
    ok = (
        ala.rating == 34
        and intertek.rating == 25
        and abs(intertek.deficiency_sum - 24.0) < 1e-9
    )
    return Outcome(
        expected="CAC 34 (ALA 16-091-4); CAC 25, sum 24 dB (Intertek J7488.04)",
        computed=(
            f"CAC {ala.rating}; CAC {intertek.rating}, "
            f"sum {intertek.deficiency_sum:.1f} dB"
        ),
        delta="exact",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 140-9:1985 clause 3.3",
    "Normalized ceiling attenuation Dn,c = D - 10 lg(A/A0), A0 = 10 m2",
)
def _chk_iso140_9_normalization() -> Outcome:
    dnc = ph.building.normalized_ceiling_attenuation([90.0], [50.0], 5.0)
    return numeric(40.0 + 10.0 * math.log10(2.0), float(dnc[0]), 1e-9, unit="dB")


@register(
    "Room & building acoustics",
    "Vigran (2008) Eqs. (9.18)-(9.20)",
    "Plenum model: Eq. (9.18) converges to Eq. (9.20) as the damping vanishes",
)
def _chk_vigran_plenum_convergence() -> Outcome:
    # Vigran prints the geometry of his own example (LS = LR = 4,75 m,
    # h = 0,43 m, eps = 2) but no numeric output anywhere in Section 9.2.3, so
    # the check is structural rather than a transcribed value: the full
    # attenuated form of Eq. (9.18) has to reproduce its own small-attenuation
    # limit, Eq. (9.20). That fails outright on the printed reading of the
    # receiving-side coefficient (see docs/ERRATA.md).
    undamped = ph.building.plenum_flanking_reduction_index(
        [50.0], [100.0], ceiling_length=4.75, plenum_height=0.43
    )
    attenuated = ph.building.plenum_flanking_reduction_index(
        [50.0],
        [100.0],
        ceiling_length=4.75,
        plenum_height=0.43,
        attenuation_source=[1e-5],
        attenuation_receiving=[1e-5],
    )
    return numeric(
        float(undamped.reduction_index[0]),
        float(attenuated.reduction_index[0]),
        1e-3,
        unit="dB",
        places=4,
        expected_label="Eq. (9.20) value, reproduced by Eq. (9.18)",
    )


# --- Masonry cavity-wall ties (Hopkins 2007) ---
@register(
    "Room & building acoustics",
    "Hopkins (2007) Eq. 4.89 / Fig. 4.35",
    "Mass-spring-mass resonance of a masonry cavity wall without and with ties",
)
def _chk_hopkins_wall_tie_resonance() -> Outcome:
    # Fig. 4.35: two 140 kg/m2 leaves, empty 75 mm cavity, then 2,5 ties/m2 of
    # s_75mm = 2e6 N/m. The caption prints fmsm = 26 Hz and fmsm = 50 Hz.
    ties = ph.building.wall_tie_stiffness_per_area(2.5, 2.0e6)
    untied = ph.building.mass_spring_mass_resonance(140.0, 140.0, 0.075)
    tied = ph.building.mass_spring_mass_resonance(
        140.0, 140.0, 0.075, tie_stiffness_per_area=ties
    )
    ok = round(untied) == 26 and round(tied) == 50
    return Outcome(
        expected="26 Hz (no ties) / 50 Hz (2,5 ties/m2, k = 2 MN/m)",
        computed=f"{untied:.2f} Hz / {tied:.2f} Hz",
        delta=f"{untied - 26.0:+.2f} / {tied - 50.0:+.2f} Hz",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "Hopkins (2007) Table A4",
    "Dynamic stiffness of four wall ties (butterfly, double-triangle, twist)",
)
def _chk_hopkins_wall_tie_table() -> Outcome:
    printed = {
        "butterfly": (0.050, 1.7e6),
        "double_triangle": (0.050, 16.1e6),
        "vertical_twist": (0.050, 94.0e6),
        "vertical_twist_100mm": (0.100, 43.4e6),
    }
    computed = {name: ph.building.wall_tie_stiffness(name) for name in printed}
    ok = all(
        abs(computed[n][0] - x) < 1e-12 and abs(computed[n][1] - k) < 1e-3
        for n, (x, k) in printed.items()
    )
    return Outcome(
        expected="1,7 / 16,1 / 94,0 MN/m at 50 mm; 43,4 MN/m at 100 mm",
        computed=" / ".join(f"{computed[n][1] / 1e6:g}" for n in printed) + " MN/m",
        delta="exact",
        passed=ok,
    )


# --- Dynamic transfer stiffness of resilient elements (ISO 10846) ---
@register(
    "Room & building acoustics",
    "ISO 10846-2:2008 3.17",
    "Transfer-stiffness level Lk = 20 lg(|k|/k0), k0 = 1 N/m  (|k| = 1 MN/m)",
)
def _chk_iso10846_level() -> Outcome:
    lk = float(ph.vibration.transfer_stiffness_level(1.0e6))
    return numeric(120.0, lk, 1e-6, unit="dB")


@register(
    "Room & building acoustics",
    "ISO 10846-3:2002 Formula (1)",
    "Indirect method k2,1 = -(2πf)²·m2·T  (f=500 Hz, m2=10 kg, T=0,01)",
)
def _chk_iso10846_indirect() -> Outcome:
    f, m2, t = 500.0, 10.0, 0.01
    expected = -((2.0 * math.pi * f) ** 2) * m2 * t
    computed = complex(ph.vibration.transfer_stiffness_indirect(f, t + 0j, m2)).real
    return numeric(expected, computed, 1e-3, rel=True, unit="N/m", places=1)


@register(
    "Room & building acoustics",
    "ISO 10846-1:2008 Table A.2",
    "FRF relation k = jω·Z at 250 Hz  (|k| recovered from impedance)",
)
def _chk_iso10846_stiffness_impedance() -> Outcome:
    f = 250.0
    w = 2.0 * math.pi * f
    k = 1.0e6 + 1j * 5.0e4
    z = complex(ph.vibration.convert_frf(k, f, "dynamic_stiffness", "impedance"))
    return numeric(abs(k), abs(1j * w * z), 1e-6, rel=True, unit="N/m", places=1)


@register(
    "Room & building acoustics",
    "ISO 7626-2:2015 7.5.2",
    "Rigid-mass calibration: accelerance mag(A) = 1/m  (m=10 kg)",
)
def _chk_iso7626_2_rigid_mass_accelerance() -> Outcome:
    res = ph.vibration.rigid_mass_calibration_check(
        [ref.ISO7626_2_CAL_ACCELERANCE], [100.0], ref.ISO7626_2_CAL_MASS_KG
    )
    value = float(res.expected[0]) if res.passed else math.nan
    return numeric(ref.ISO7626_2_CAL_ACCELERANCE, value, 1e-9, unit="1/kg", places=3)


@register(
    "Room & building acoustics",
    "ISO 7626-2:2015 7.5.2",
    "Rigid-mass calibration: mobility mag(Y) = 1/(2πf·m) at 100 Hz  (m=10 kg)",
)
def _chk_iso7626_2_rigid_mass_mobility() -> Outcome:
    res = ph.vibration.rigid_mass_calibration_check(
        [ref.ISO7626_2_CAL_MOBILITY_100HZ],
        [100.0],
        ref.ISO7626_2_CAL_MASS_KG,
        quantity="mobility",
    )
    value = float(res.expected[0]) if res.passed else math.nan
    return numeric(
        ref.ISO7626_2_CAL_MOBILITY_100HZ,
        value,
        1e-5,
        rel=True,
        unit="m/(N·s)",
        places=7,
    )


@register(
    "Room & building acoustics",
    "ISO 7626-2:2015 Annex A",
    "Normalized random error ε = √((1−γ²)/(2nγ²)): γ²=0,8, n=75 → 4,08 % (< 5 %)",
)
def _chk_iso7626_2_random_error() -> Outcome:
    eps = float(ph.vibration.random_error_percent(0.8, 75))
    return numeric(ref.ISO7626_2_RANDOM_ERROR_PCT, eps, 0.005, unit="%", places=2)


@register(
    "Room & building acoustics",
    "ISO 7626-1:2011 Table 1",
    "Rigid 1 kg mass at ω = 1000 rad/s: mobility 1e-3, compliance 1e-6 (decades)",
)
def _chk_iso7626_decade_identity() -> Outcome:
    f = ref.ISO7626_1_DECADE_FREQ_HZ
    y = abs(complex(ph.vibration.convert_frf(1.0, f, "apparent_mass", "mobility")))
    h = abs(complex(ph.vibration.convert_frf(1.0, f, "apparent_mass", "receptance")))
    ok = abs(h - ref.ISO7626_1_DECADE_COMPLIANCE) <= 1e-15
    return numeric(
        ref.ISO7626_1_DECADE_MOBILITY,
        y if ok else math.nan,
        1e-9,
        rel=True,
        unit="m/(N·s)",
        places=4,
    )


@register(
    "Room & building acoustics",
    "ISO 10846-3:2002 6.1 Inequality (2)",
    "Indirect-method validity limit mag(T) = 0,1 ↔ ΔL1,2 = 20 dB",
)
def _chk_iso10846_3_validity_threshold() -> Outcome:
    delta_l = 20.0 * math.log10(1.0 / ph.vibration.TRANSMISSIBILITY_LIMIT)
    return numeric(ref.ISO10846_3_LIMIT_DELTA_L_DB, delta_l, 1e-9, unit="dB", places=1)


@register(
    "Room & building acoustics",
    "ISO 10846-3:2002 6.1",
    "Model bias at the validity limit: k_ind/k = 1,1 (0,83 dB ≤ 1 dB, 10 % ≤ 12 %)",
)
def _chk_iso10846_3_validity_bias() -> Outcome:
    # Undamped mass-spring model at omega^2 m = 11 k, i.e. T = -0,1 exactly.
    k, m = 1.0e6, 1.0
    f = math.sqrt(11.0 * k / m) / (2.0 * math.pi)
    t = complex(ph.vibration.base_transmissibility(f, m, k))
    k_ind = abs(complex(ph.vibration.transfer_stiffness_indirect(f, t, m)))
    ratio = k_ind / k
    bias_ok = (
        20.0 * math.log10(ratio) <= ref.ISO10846_3_ACCURACY_DB
        and ratio - 1.0 <= ref.ISO10846_3_ACCURACY_FRACTION
    )
    return numeric(
        ref.ISO10846_3_LIMIT_BIAS_RATIO,
        ratio if bias_ok else math.nan,
        1e-9,
        rel=True,
        places=4,
    )


@register(
    "Room & building acoustics",
    "ISO 10846-1:2008 Equation (6)",
    "Delivered/blocking force F2/F2,b = 1/1,1 at mag(k2,2/kt) = 0,1 (within 10 %)",
)
def _chk_iso10846_1_blocking_force() -> Outcome:
    value = abs(complex(ph.vibration.blocking_force_ratio(1.0e5, 1.0e6)))
    return numeric(ref.ISO10846_1_EQ6_FORCE_RATIO, value, 1e-9, places=4)


@register(
    "Room & building acoustics",
    "ISO 10846-2:2008 / -3:2002 7.6",
    "Linearity: ΔLk ≤ 1,5 dB for input spectra 10 dB apart (linear element: 0)",
)
def _chk_iso10846_linearity() -> Outcome:
    k, u_a = 1.0e6 + 3.0e4j, 1.0e-6 + 0j
    u_b = u_a * 10.0 ** (-ref.ISO10846_LINEARITY_STEP_DB / 20.0)
    lk_a = float(
        ph.vibration.transfer_stiffness_level(
            ph.vibration.transfer_stiffness_direct(k * u_a, u_a)
        )
    )
    lk_b = float(
        ph.vibration.transfer_stiffness_level(
            ph.vibration.transfer_stiffness_direct(k * u_b, u_b)
        )
    )
    return numeric(
        0.0,
        abs(lk_a - lk_b),
        ref.ISO10846_LINEARITY_TOL_DB,
        unit="dB",
        places=3,
        expected_label="ΔLk ≤ 1,5 dB (7.6 c)",
    )


# --- Sound power from surface vibration (ISO/TS 7849-1/-2) ---
@register(
    "Room & building acoustics",
    "ISO/TS 7849-1:2009 Formula (8)",
    "Calibration L_v from â = 9,81 m/s² at 100 Hz  (standard's EXAMPLE)",
)
def _chk_iso7849_calibration() -> Outcome:
    lv = float(ph.emission.velocity_level_from_acceleration(9.81, 100.0))
    return numeric(106.9, lv, 0.05, unit="dB", places=1)


@register(
    "Room & building acoustics",
    "ISO/TS 7849-2:2009 Formula (15)",
    "L_W from L_v via measured radiation factor = 10 lg(P/P0)  (round-trip)",
)
def _chk_iso7849_power_round_trip() -> Outcome:
    p, s, v2 = 3.0e-4, 2.0, (1.0e-3) ** 2
    eps = float(ph.emission.radiation_factor(p, s, v2))
    lv = float(ph.emission.velocity_level(math.sqrt(v2)))
    lw = float(ph.emission.radiated_sound_power_level(lv, s, radiation_factor=eps))
    return numeric(10.0 * math.log10(p / 1e-12), lw, 1e-6, unit="dB", places=3)


@register(
    "Room & building acoustics",
    "ISO/TS 7849-1:2009 Formula (12)",
    "Impedance term: L_W − L_v = 10 lg(411/400) at ε = 1, S = S0",
)
def _chk_iso7849_impedance_term() -> Outcome:
    lw = float(ph.emission.radiated_sound_power_level(80.0, 1.0))
    return numeric(10.0 * math.log10(411.0 / 400.0), lw - 80.0, 1e-9, unit="dB")


# --- Structure-borne sound power of building equipment (EN 15657) ---
@register(
    "Room & building acoustics",
    "EN 15657:2018 Formula (14)",
    "Reception-plate L_Ws = resonant-plate power P = ωη(mS)⟨v²⟩  (round-trip)",
)
def _chk_en15657_power_balance() -> Outcome:
    lv, f, m, s, eta = 82.0, 800.0, 15.0, 1.5, 0.02
    lw = float(ph.building.structure_borne_power_level(lv, f, m, s, eta))
    v2 = (1e-9) ** 2 * 10.0 ** (0.1 * lv)
    p = 2.0 * math.pi * f * eta * (m * s) * v2
    return numeric(10.0 * math.log10(p / 1e-12), lw, 1e-6, unit="dB", places=3)


@register(
    "Room & building acoustics",
    "EN 15657:2018 Formula (13)",
    "Plate loss factor η = 2,2/(f·Ts) at 1 kHz, Ts = 0,3 s",
)
def _chk_en15657_loss_factor() -> Outcome:
    eta = float(ph.building.plate_loss_factor([1000.0], 0.3)[0])
    return numeric(2.2 / (1000.0 * 0.3), eta, 1e-9)


@register(
    "Room & building acoustics",
    "EN 15657:2018 Formulae (15)/(17) + EN 12354-5 Annex I.3",
    "Source conversion chain reproduces Table I.8 (wall, installed)",
)
def _chk_en15657_conversion_chain() -> Outcome:
    # Measured plate power (Y_plate = 5,34e-6) -> blocked force (15) ->
    # characteristic reception-plate level (17, Y_R,inf,low = 5e-6) ->
    # Annex I mobility correction to the wall (Y_wall = 24,1e-6). The printed
    # Table I.8 row is the oracle (one-decimal intermediates, +/-0,15 dB).
    lwsn = ph.building.characteristic_reception_plate_power(
        ph.building.equivalent_blocked_force_level(
            ref.EN12354_5_I8_WALL_LWS, ref.EN12354_5_I8_PLATE_MOBILITY
        )
    )
    installed = ph.building.installed_power_from_reception_plate(
        lwsn, ref.EN12354_5_I8_Y_WALL
    )
    worst = float(
        np.max(
            np.abs(np.asarray(installed) - np.asarray(ref.EN12354_5_I8_WALL_INSTALLED))
        )
    )
    return numeric(
        0.0,
        worst,
        ref.EN12354_5_ANNEX_I_TOL,
        unit="dB",
        places=3,
        expected_label="max abs(L_Ws,inst - Table I.8) <= 0,15 dB",
    )


@register(
    "Room & building acoustics",
    "ISO 9611:1996 eq. (9)",
    "Mean free velocity level (energy mean, v0 = 5e-8 m/s)",
)
def _chk_iso9611_mean_velocity() -> Outcome:
    computed = float(ph.building.mean_free_velocity_level(ref.ISO9611_MEAN_LEVELS))
    return numeric(ref.ISO9611_MEAN_EXPECTED, computed, 1e-9, unit="dB", places=4)


# --- Detailed per-band building prediction (ISO 12354-1/-2:2017) ---
def _iso12354_detailed_situ() -> tuple[
    np.ndarray, dict[str, InSituElementResult], np.ndarray
]:
    """The Annex L / Annex G building evaluated in situ, per band."""
    import iso12354_building as bld

    bands = np.asarray(ref.ISO12354_ANNEX_L_BANDS, dtype=np.float64)
    situ = {k: ph.building.in_situ_element(e, bands) for k, e in bld.elements().items()}
    delta = ph.building.floating_floor_improvement(
        bands, resonance_frequency=bld.floating_floor_resonance()
    )
    return bands, situ, delta


@register(
    "Room & building acoustics",
    "ISO 12354-1:2017 Annex L, Tables L.2 to L.4",
    "In-situ element chain: 10 lg sigma, 10 lg sigma_f, eta_tot, Rsitu, a_situ "
    "(21 bands x 5 elements)",
)
def _chk_iso12354_annex_l_elements() -> Outcome:
    # Every quantity is compared on a decibel scale so the reported worst
    # deviation carries one unit: the two radiation factors and the loss
    # factor are dimensionless, so they enter as 10 lg of the ratio, and the
    # absorption length as 10 lg of its ratio to the printed value.
    _bands, situ, _delta = _iso12354_detailed_situ()

    def _ratio_db(computed: ArrayLike, printed: ArrayLike) -> float:
        return float(
            np.max(np.abs(10.0 * np.log10(np.asarray(computed) / np.asarray(printed))))
        )

    worst = 0.0
    for label, printed in ref.ISO12354_ANNEX_L2_SIGMA.items():
        worst = max(worst, _ratio_db(situ[label].radiation_factor, printed))
    for label, printed in ref.ISO12354_ANNEX_L2_SIGMA_F.items():
        worst = max(worst, _ratio_db(situ[label].forced_radiation_factor, printed))
    for label, printed in ref.ISO12354_ANNEX_L3_ETA.items():
        worst = max(worst, _ratio_db(situ[label].total_loss_factor, printed))
    for label, printed in ref.ISO12354_ANNEX_L4_ABSORPTION.items():
        worst = max(worst, _ratio_db(situ[label].absorption_length, printed))
    for label, printed in ref.ISO12354_ANNEX_L3_R_SITU.items():
        worst = max(
            worst,
            float(
                np.max(np.abs(situ[label].sound_reduction_index - np.asarray(printed)))
            ),
        )
    return numeric(0.0, worst, 0.1, unit="dB", places=3)


@register(
    "Room & building acoustics",
    "ISO 12354-1:2017 Annex L, Table L.1",
    "Detailed airborne model: 13 paths + R' per band, R'w = 57 dB",
)
def _chk_iso12354_annex_l1() -> Outcome:
    import iso12354_building as bld

    bands, situ, delta = _iso12354_detailed_situ()
    result = ph.building.detailed_airborne_prediction(
        bands,
        direct_index=ph.building.direct_reduction_index(
            situ["floor"].sound_reduction_index, delta_r_source=delta
        ),
        flanking_paths=bld.airborne_paths(situ, delta),
    )
    worst = max(
        float(
            np.max(np.abs(p.values - np.asarray(ref.ISO12354_ANNEX_L1_PATHS[p.label])))
        )
        for p in result.paths
    )
    worst = max(
        worst,
        float(
            np.max(np.abs(result.r_prime - np.asarray(ref.ISO12354_ANNEX_L1_R_PRIME)))
        ),
    )
    rating = result.rating.rating if result.rating is not None else None
    ok = worst <= 0.1 and rating == ref.ISO12354_ANNEX_L1_R_PRIME_W
    return Outcome(
        expected=f"max path/total dev <= 0,1 dB; R'w = "
        f"{ref.ISO12354_ANNEX_L1_R_PRIME_W} dB",
        computed=f"{worst:.3f} dB; {rating} dB",
        delta=f"{worst:.3f} dB",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 12354-2:2017 Annex G, Tables G.3, G.4 and G.1",
    "Detailed impact model: Ln,situ, Ln,Dd, Ln,Df, L'n per band, L'n,w = 41 dB",
)
def _chk_iso12354_annex_g1() -> Outcome:
    import iso12354_building as bld

    bands, situ, delta = _iso12354_detailed_situ()
    result = ph.building.detailed_impact_prediction(
        bands,
        direct_level=ph.building.direct_impact_level(
            situ["floor"].impact_level, delta_l=delta
        ),
        flanking_paths=bld.impact_paths(situ, delta),
    )
    worst = float(
        np.max(
            np.abs(
                situ["floor"].impact_level - np.asarray(ref.ISO12354_ANNEX_G3_LN_SITU)
            )
        )
    )
    by_label = {p.label: np.asarray(p.values) for p in result.paths}
    worst = max(
        worst,
        float(np.max(np.abs(by_label["Dd"] - np.asarray(ref.ISO12354_ANNEX_G4_LN_DD)))),
    )
    worst = max(
        worst,
        float(
            np.max(np.abs(by_label["Df1"] - np.asarray(ref.ISO12354_ANNEX_G4_LN_DF)))
        ),
    )
    # Table G.1's flanking columns disagree with Table G.4 below 100 Hz; see
    # docs/ERRATA.md. The total is compared over the same 100 Hz to 5 kHz range.
    worst = max(
        worst,
        float(
            np.max(
                np.abs(
                    result.l_prime_n[3:]
                    - np.asarray(ref.ISO12354_ANNEX_G1_L_PRIME_N[3:])
                )
            )
        ),
    )
    rating = result.rating
    ok = (
        worst <= 0.1
        and rating is not None
        and rating.rating == ref.ISO12354_ANNEX_G1_L_PRIME_N_W
        and rating.ci == ref.ISO12354_ANNEX_G1_CI
    )
    printed = f"{ref.ISO12354_ANNEX_G1_L_PRIME_N_W} ({ref.ISO12354_ANNEX_G1_CI})"
    got = "n/a" if rating is None else f"{rating.rating} ({rating.ci})"
    return Outcome(
        expected=f"max path/total dev <= 0,1 dB; L'n,w (CI) = {printed} dB",
        computed=f"{worst:.3f} dB; {got} dB",
        delta=f"{worst:.3f} dB",
        passed=ok,
    )


# --- Resilient-layer prediction (tapping force, coverings, floating floors) ---
def _hopkins_plate(
    density: float, longitudinal: float, poisson: float, thickness: float
) -> tuple[float, float]:
    """``(contact stiffness, driving-point impedance)`` of a Hopkins Table A2 plate."""
    modulus = density * longitudinal**2 * (1.0 - poisson**2)
    stiffness = float(
        ph.building.plate_contact_stiffness(modulus, poisson_ratio=poisson)
    )
    impedance = float(
        ph.vibration.infinite_plate_impedance(
            ph.vibration.plate_bending_stiffness(modulus, thickness, poisson),
            density * thickness,
        )
    )
    return stiffness, impedance


@register(
    "Room & building acoustics",
    "Hopkins (2007) 3.6.3.1 / 4.4.3.1, printed pp. 276-282 and 513-514",
    "Tapping machine: vo, cut-off frequencies fco of a bare slab and two soft "
    "coverings (7 000 / 2 300 / 100 Hz)",
)
def _chk_hopkins_tapping_cut_off() -> Outcome:
    _stiffness, impedance = _hopkins_plate(2200.0, 3800.0, 0.2, 0.14)
    worst = abs(float(ph.building.hammer_impact_velocity()) / 0.886 - 1.0)
    printed = ((None, 7000.0), (1.5e11, 2300.0), (2.8e8, 100.0))
    for modulus_over_thickness, expected in printed:
        if modulus_over_thickness is None:
            stiffness = _stiffness
        else:
            stiffness = float(
                ph.building.covering_contact_stiffness(
                    modulus_over_thickness * 0.005, 0.005
                )
            )
        computed = float(ph.building.tapping_cut_off_frequency(stiffness, impedance))
        worst = max(worst, abs(computed / expected - 1.0))
    return numeric(0.0, worst, 0.02, rel=False, places=4)


@register(
    "Room & building acoustics",
    "Hopkins (2007) Figs. 3.30/3.31 and 4.73, printed pp. 281 and 524",
    "Over/under-critical case of four walking surfaces; double floating-floor "
    "resonances 74 Hz and 195 Hz",
)
def _chk_hopkins_resilient_layers() -> Outcome:
    plates = (
        ("concrete", 2200.0, 3800.0, 0.2, 0.14, False),
        ("screed", 2000.0, 3250.0, 0.2, 0.065, False),
        ("chipboard", 760.0, 2200.0, 0.3, 0.022, True),
        ("osb", 590.0, 2570.0, 0.3, 0.015, True),
    )
    cases_ok = True
    for _label, rho, c_l, nu, thickness, over in plates:
        stiffness, impedance = _hopkins_plate(rho, c_l, nu, thickness)
        result = ph.building.tapping_force_spectrum([100.0], stiffness, impedance)
        cases_ok = cases_ok and result.over_critical is over
    mass_per_area = 710.0 * 0.018
    lower, upper = ph.building.double_floating_floor_resonances(
        7.25e6, mass_per_area, 7.25e6, mass_per_area
    )
    worst = max(abs(lower / 74.0 - 1.0), abs(upper / 195.0 - 1.0))
    return Outcome(
        expected="4/4 critical cases; fmsms = 74 / 195 Hz (+/-2%)",
        computed=f"{'4/4' if cases_ok else 'mismatch'}; {lower:.1f} / {upper:.1f} Hz",
        delta=f"{worst * 100.0:.2f}%",
        passed=cases_ok and worst <= 0.02,
    )


@register(
    "Room & building acoustics",
    "ISO 12354-2:2017 Annex C / Annex G Table G.4",
    "Floating floor: fo = 160 sqrt(s'/m') = 52,8 Hz, DeltaL = 30 lg(f/fo) over "
    "21 bands, DeltaLw = 32,2 dB",
)
def _chk_iso12354_2_floating_floor() -> Outcome:
    bands = np.asarray(ref.ISO12354_ANNEX_L_BANDS, dtype=np.float64)
    stiffness = ref.ISO12354_ANNEX_L_FLOATING_STIFFNESS * 1e6
    mass = ref.ISO12354_ANNEX_L_FLOATING_MASS
    f0 = float(ph.building.floating_floor_resonance_frequency(stiffness, mass))
    spectrum = ph.building.floating_floor_improvement_spectrum(
        bands, resonance_frequency=f0
    )
    printed = np.asarray(ref.ISO12354_ANNEX_G4_DELTA_L)
    worst = float(np.max(np.abs(spectrum.improvement - printed)))
    # The printed table is quoted to 0,1 dB, so a residual under 0,05 dB is
    # the most any correct model can be asked for; requiring the computed
    # values to round *onto* the printed ones is the exact statement, and
    # stops the row passing on a near-miss inside the rounding quantum.
    if not np.array_equal(np.round(spectrum.improvement, 1), printed):
        worst = float("inf")
    delta_lw = float(ph.building.weighted_floating_floor_improvement(mass, stiffness))
    worst = max(worst, abs(delta_lw - ref.ISO12354_ANNEX_G10_DELTA_LW))
    # The resonance frequency is in hertz, so it is verified separately rather
    # than folded into the decibel residual reported below.
    if abs(f0 - ref.ISO12354_ANNEX_L_FLOATING_F0) > 0.05:
        worst = float("inf")
    return numeric(0.0, worst, 0.05, unit="dB", places=3)


@register(
    "Room & building acoustics",
    "ISO 12354-1:2017 Annex D / Hopkins (2007) Fig. 4.48, printed p. 486",
    "Lining resonance (Formula D.1) 542 Hz and the Table D.1 improvement branches",
)
def _chk_iso12354_1_annex_d() -> Outcome:
    f0 = float(
        ph.building.lining_resonance_frequency(51.0, 6.3, dynamic_stiffness=65e6)
    )
    worst = abs(f0 / 542.0 - 1.0)
    table_ok = all(
        ph.building.weighted_lining_improvement(resonance, 45.0) == expected
        for resonance, expected in (
            (200.0, -1.0),
            (250.0, -3.0),
            (315.0, -5.0),
            (400.0, -7.0),
            (500.0, -9.0),
            (1000.0, -10.0),
            (2000.0, -5.0),
        )
    )
    branch = float(ph.building.weighted_lining_improvement(100.0, 40.0))
    table_ok = table_ok and abs(branch - (74.4 - 40.0 - 20.0)) <= 1e-9
    return Outcome(
        expected="fo = 542 Hz (+/-1%); 8/8 Table D.1 rows",
        computed=f"{f0:.1f} Hz; {'8/8' if table_ok else 'mismatch'}",
        delta=f"{worst * 100.0:.2f}%",
        passed=table_ok and worst <= 0.01,
    )


# --- Installed structure-borne sound from equipment (EN 12354-5) ---
@register(
    "Room & building acoustics",
    "EN 12354-5:2009 Formula (19b/19c)",
    "Coupling term → force-source limit 10 lg(mag(Ys)/Re{Yi}) as mag(Ys) ≫ mag(Yi)",
)
def _chk_en12354_5_coupling_limit() -> Outcome:
    ys, yi = 1e-3 + 0j, 1e-7 + 0j
    dc = float(ph.building.coupling_term(ys, yi))
    limit = float(ph.building.coupling_term_force_source(ys, yi))
    return numeric(limit, dc, 1e-2, unit="dB", places=3)


@register(
    "Room & building acoustics",
    "EN 12354-5:2009 Annex I.3, Table I.9",
    "Flushing cistern: four paths + Formula (17) total -> 29 dB(A)",
)
def _chk_en12354_5_annex_i9() -> Outcome:
    # The standard's own end-to-end worked example (replaces the former
    # formula-restatement checks of Formulae (18a)/(18b), which could not
    # catch a mistranscribed constant): both power components through
    # D_C (Table I.9), Formula (18a) per path and the energetic total.
    tol = ref.EN12354_5_ANNEX_I_TOL
    inst_wall = ph.building.installed_structure_borne_power_level(
        ref.EN12354_5_I8_WALL_LWSC, ref.EN12354_5_I9_DC_WALL
    )
    inst_floor = ph.building.installed_structure_borne_power_level(
        ref.EN12354_5_I8_FLOOR_LWSC, ref.EN12354_5_I9_DC_FLOOR
    )
    paths = [
        (
            inst_wall,
            ref.EN12354_5_I9_DSA_WALL,
            ref.EN12354_5_I9_R_WALL_FLOOR,
            ref.EN12354_5_I9_S_WALL,
            ref.EN12354_5_I9_LNS_WALL_FLOOR,
        ),
        (
            inst_wall,
            ref.EN12354_5_I9_DSA_WALL,
            ref.EN12354_5_I9_R_WALL_WALL,
            ref.EN12354_5_I9_S_WALL,
            ref.EN12354_5_I9_LNS_WALL_WALL,
        ),
        (
            inst_floor,
            ref.EN12354_5_I9_DSA_FLOOR,
            ref.EN12354_5_I9_R_FLOOR_FLOOR,
            ref.EN12354_5_I9_S_FLOOR,
            ref.EN12354_5_I9_LNS_FLOOR_FLOOR,
        ),
        (
            inst_floor,
            ref.EN12354_5_I9_DSA_FLOOR,
            ref.EN12354_5_I9_R_FLOOR_WALL,
            ref.EN12354_5_I9_S_FLOOR,
            ref.EN12354_5_I9_LNS_FLOOR_WALL,
        ),
    ]
    worst = 0.0
    rows = []
    for inst, dsa, rij, s_i, expected in paths:
        lns = ph.building.structure_borne_pressure_level_path(inst, dsa, rij, s_i)
        worst = max(worst, float(np.max(np.abs(lns - np.asarray(expected)))))
        rows.append(np.asarray(lns))
    total = ph.building.total_structure_borne_pressure_level(np.vstack(rows))
    worst = max(
        worst, float(np.max(np.abs(total - np.asarray(ref.EN12354_5_I9_LNS_TOTAL))))
    )
    a_weights = np.array([-26.2, -16.1, -8.6, -3.2, 0.0, 1.2])
    lns_a = float(10.0 * np.log10(np.sum(10.0 ** (0.1 * (total + a_weights)))))
    ok = worst <= tol and round(lns_a) == ref.EN12354_5_I9_LNS_TOTAL_A
    return Outcome(
        expected=f"max path/total dev <= {tol} dB; total "
        f"{ref.EN12354_5_I9_LNS_TOTAL_A} dB(A)",
        computed=f"{worst:.3f} dB; {lns_a:.1f} dB(A)",
        delta=f"{worst:.3f} dB",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "EN 12354-5:2009 Annex I.2, Table I.6a",
    "Whirlpool floor component: mobility correction + path 11",
)
def _chk_en12354_5_annex_i6a() -> Outcome:
    tol = ref.EN12354_5_ANNEX_I_TOL
    inst = ph.building.installed_power_from_reception_plate(
        ref.EN12354_5_I6A_LWSN_FLOOR, ref.EN12354_5_I6A_Y_FLOOR
    )
    dev_inst = float(
        np.max(np.abs(np.asarray(inst) - np.asarray(ref.EN12354_5_I6A_LWSN_INST_FLOOR)))
    )
    lns = ph.building.structure_borne_pressure_level_path(
        inst, ref.EN12354_5_I6A_DSA_FLOOR, ref.EN12354_5_I6A_R11, 10.0
    )
    dev_path = float(
        np.max(np.abs(np.asarray(lns) - np.asarray(ref.EN12354_5_I6A_LNS_11)))
    )
    worst = max(dev_inst, dev_path)
    return numeric(
        0.0,
        worst,
        tol,
        unit="dB",
        places=3,
        expected_label="max abs(dev vs Table I.6a) <= 0,15 dB",
    )
