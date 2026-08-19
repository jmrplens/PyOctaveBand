#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Human vibration (ISO 8041-1 / ISO 2631 / ISO 5349 / Directive 2002/44/EC).

Vibration as a person receives it: the ISO 8041-1 frequency weightings and
their tolerance bands, the whole-body quantities of ISO 2631, the hand-arm
quantities of ISO 5349-1/-2, and the daily exposure A(8) that Directive
2002/44/EC sets its action and limit values on.
"""

from __future__ import annotations

import reference_data as ref

import phonometry as ph

from ..registry import Outcome, numeric, register

_HUMAN_VIB = "Human vibration (ISO 8041 / 2631 / 5349)"


def _true_centre(n: int) -> float:
    """True IEC 61260 one-third-octave centre ``10^(n/10)`` Hz."""
    return float(10.0 ** (n / 10.0))


@register(_HUMAN_VIB, "ISO 8041-1:2017 Table B.8", "Wk design-goal factor at 6,31 Hz")
def _chk_iso8041_wk_annex_b() -> Outcome:
    factor = float(ph.vibration.weighting_factors("Wk", _true_centre(8))[0])
    return numeric(ref.ISO8041_1_WK_FACTOR_6P31HZ, factor, 1e-3, rel=True, places=4)


@register(_HUMAN_VIB, "ISO 8041-1:2017 Table B.9", "Wm design-goal factor at 1,585 Hz")
def _chk_iso8041_wm_annex_b() -> Outcome:
    factor = float(ph.vibration.weighting_factors("Wm", _true_centre(2))[0])
    return numeric(ref.ISO8041_1_WM_FACTOR_1P585HZ, factor, 1e-3, rel=True, places=4)


@register(_HUMAN_VIB, "ISO 8041-1:2017 Table 1", "Wh factor at the 500 rad/s reference")
def _chk_iso8041_wh_reference() -> Outcome:
    factor = float(
        ph.vibration.weighting_factors("Wh", ref.ISO8041_1_WH_REF_FREQ_HZ)[0]
    )
    return numeric(ref.ISO8041_1_WH_REF_FACTOR, factor, 1.5e-3, rel=True, places=4)


@register(_HUMAN_VIB, "ISO 8041-1:2017 Table B.1", "Wb design-goal factor at 6,31 Hz")
def _chk_iso8041_wb_annex_b() -> Outcome:
    factor = float(ph.vibration.weighting_factors("Wb", _true_centre(8))[0])
    return numeric(ref.ISO8041_1_WB_FACTOR_6P31HZ, factor, 1e-3, rel=True, places=4)


@register(
    _HUMAN_VIB, "ISO 8041-1:2017 Table B.1", "Wb design-goal factors at 1 / 100 Hz"
)
def _chk_iso8041_wb_annex_b_edges() -> Outcome:
    worst = max(
        abs(
            float(ph.vibration.weighting_factors("Wb", _true_centre(n))[0])
            / expected
            - 1.0
        )
        for n, expected in (
            (0, ref.ISO8041_1_WB_FACTOR_1HZ),
            (20, ref.ISO8041_1_WB_FACTOR_100HZ),
        )
    )
    return numeric(
        0.0, worst, 1e-3, places=6, expected_label="max rel dev ≤ 0,1 %"
    )


@register(_HUMAN_VIB, "ISO 8041-1:2017 Table 1", "Wc factor at the 100 rad/s reference")
def _chk_iso8041_wc_reference() -> Outcome:
    factor = float(
        ph.vibration.weighting_factors("Wc", ref.ISO8041_1_WBV_REF_FREQ_HZ)[0]
    )
    return numeric(ref.ISO8041_1_WC_REF_FACTOR, factor, 1e-3, rel=True, places=4)


@register(
    _HUMAN_VIB,
    "ISO 8041-1:2017 Table 1 + Table B.3",
    "Wd factors at the 100 rad/s reference and 1 Hz",
)
def _chk_iso8041_wd_reference_and_annex_b() -> Outcome:
    worst = max(
        abs(
            float(ph.vibration.weighting_factors("Wd", freq)[0]) / expected
            - 1.0
        )
        for freq, expected in (
            (ref.ISO8041_1_WBV_REF_FREQ_HZ, ref.ISO8041_1_WD_REF_FACTOR),
            (_true_centre(0), ref.ISO8041_1_WD_FACTOR_1HZ),
        )
    )
    return numeric(
        0.0, worst, 1e-3, places=6, expected_label="max rel dev ≤ 0,1 %"
    )


@register(_HUMAN_VIB, "ISO 8041-1:2017 Table B.4", "We design-goal factor at 8 Hz")
def _chk_iso8041_we_annex_b() -> Outcome:
    factor = float(ph.vibration.weighting_factors("We", _true_centre(9))[0])
    return numeric(ref.ISO8041_1_WE_FACTOR_8HZ, factor, 1e-3, rel=True, places=4)


@register(
    _HUMAN_VIB, "ISO 8041-1:2017 Table B.5", "Wf design-goal factors at 0,1585 / 0,1 Hz"
)
def _chk_iso8041_wf_annex_b() -> Outcome:
    worst = max(
        abs(
            float(ph.vibration.weighting_factors("Wf", _true_centre(n))[0])
            / expected
            - 1.0
        )
        for n, expected in (
            (-8, ref.ISO8041_1_WF_FACTOR_0P1585HZ),
            (-10, ref.ISO8041_1_WF_FACTOR_0P1HZ),
        )
    )
    return numeric(
        0.0, worst, 1e-3, places=6, expected_label="max rel dev ≤ 0,1 %"
    )


@register(
    _HUMAN_VIB, "ISO 8041-1:2017 Table B.7", "Wj design-goal factors at 6,31 / 8 Hz"
)
def _chk_iso8041_wj_annex_b() -> Outcome:
    worst = max(
        abs(
            float(ph.vibration.weighting_factors("Wj", _true_centre(n))[0])
            / expected
            - 1.0
        )
        for n, expected in (
            (8, ref.ISO8041_1_WJ_FACTOR_6P31HZ),
            (9, ref.ISO8041_1_WJ_FACTOR_8HZ),
        )
    )
    return numeric(
        0.0, worst, 1e-3, places=6, expected_label="max rel dev ≤ 0,1 %"
    )


@register(
    _HUMAN_VIB,
    "ISO 8041-1:2017 Table 5 + Annex B",
    "All nine weightings inside the tolerance envelope (318 printed bands)",
)
def _chk_iso8041_table5_envelope() -> Outcome:
    violations = 0
    for name, rows in ref.ISO8041_1_ANNEX_B_FACTORS.items():
        ft1, ft2, ft3, ft4 = ref.ISO8041_1_TABLE4_TRANSITIONS[name]
        for n, printed in rows:
            freq = _true_centre(n)
            if freq <= ft1:
                region = 0
            elif freq < ft2:
                region = 1
            elif freq <= ft3:
                region = 2
            elif freq < ft4:
                region = 3
            else:
                region = 4
            upper, lower = ref.ISO8041_1_TABLE5_TOLERANCES[region]
            ratio = (
                float(ph.vibration.weighting_factors(name, freq)[0]) / printed
                - 1.0
            )
            if not -lower <= ratio <= upper:
                violations += 1
    return numeric(
        0.0, float(violations), 0.0, places=0,
        expected_label="0 bands outside the Table 5 tolerances",
    )


@register(_HUMAN_VIB, "ISO 5349-2:2001 Example E.2.1", "Single-tool daily exposure A(8)")
def _chk_iso5349_e21() -> Outcome:
    a8 = ph.vibration.daily_exposure(7.4, 2.5 * 3600.0)
    return numeric(ref.ISO5349_2_E21_A8, a8, 0.05, unit="m/s^2", places=2)


@register(_HUMAN_VIB, "ISO 5349-2:2001 Example E.3", "Forestry three-task A(8)")
def _chk_iso5349_e3() -> Outcome:
    a8 = ph.vibration.hav_daily_exposure(
        [4.6, 6.0, 3.6], [2 * 3600.0, 1 * 3600.0, 2 * 3600.0]
    )
    return numeric(ref.ISO5349_2_E3_A8, a8, 0.05, unit="m/s^2", places=2)


@register(_HUMAN_VIB, "ISO 5349-1:2001 Eq. (C.1)", "VWF 10 % lifetime Dy at A(8)=7")
def _chk_iso5349_vwf() -> Outcome:
    dy = ph.vibration.hav_vwf_lifetime_years(ref.ISO5349_1_VWF_A8)
    return numeric(ref.ISO5349_1_VWF_DY_YEARS, dy, 0.1, unit="yr", places=2)


@register(_HUMAN_VIB, "Directive 2002/44/EC Art. 3", "HAV/WBV action & limit values")
def _chk_directive_2002_44() -> Outcome:
    hav = ph.vibration.exposure_assessment(1.0, kind="hav")
    wbv = ph.vibration.exposure_assessment(0.1, kind="wbv")
    ok = (
        hav.action_value == ref.DIRECTIVE_2002_44_HAV_EAV
        and hav.limit_value == ref.DIRECTIVE_2002_44_HAV_ELV
        and wbv.action_value == ref.DIRECTIVE_2002_44_WBV_EAV
        and wbv.limit_value == ref.DIRECTIVE_2002_44_WBV_ELV
    )
    exp = (
        f"HAV {ref.DIRECTIVE_2002_44_HAV_EAV}/{ref.DIRECTIVE_2002_44_HAV_ELV}, "
        f"WBV {ref.DIRECTIVE_2002_44_WBV_EAV}/{ref.DIRECTIVE_2002_44_WBV_ELV} m/s^2"
    )
    got = (
        f"HAV {hav.action_value}/{hav.limit_value}, "
        f"WBV {wbv.action_value}/{wbv.limit_value} m/s^2"
    )
    return Outcome(expected=exp, computed=got, delta="0", passed=ok)
