#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Environmental noise sources: road traffic and wind turbines.

The CNOSSOS-EU road source of Directive 2002/49/EC Annex II 2.2 - rolling and
propulsion emission per vehicle category, the road-surface corrections and the
traffic-flow assembly - checked against the CIRCABC emission test set the
Commission publishes with the method.

The IEC 61400-11 wind-turbine quantities close the module: the apparent sound
power level referred to the rotor centre and the tonal-audibility chain, both
closed forms of the standard.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import reference_data as ref

import phonometry as ph

from ..registry import Outcome, numeric, register

_CNOSSOS_ROAD = "CNOSSOS-EU road source (Directive 2002/49/EC Annex II)"


def _cnossos_road_2015_inputs() -> tuple[Any, dict[str, Any]]:
    """The superseded (EU) 2015/996 Appendix F database the workbook used.

    Wrapped in ``tests/cnossos_road_oracle.py`` so this report and the test
    suite read the same database; two copies of the wrapping could drift and
    leave the two validating against different tables.
    """
    import cnossos_road_oracle as oracle

    return oracle.coefficients_2015(), oracle.surfaces_2015()


@register(
    _CNOSSOS_ROAD,
    "CIRCABC CNOSSOS-EU road emission test set",
    "Line power of the 60 committed cases of the 4 875-case published test set, 8 octave bands each, dB re 1 pW/m",
)
def _chk_cnossos_road_workbook() -> Outcome:
    """Worst per-band deviation from the published test workbook.

    The European Commission source-module test set was computed with the
    Appendix F tables of Commission Directive (EU) 2015/996, so the shipped
    equations are fed that superseded database. The workbook prints two
    decimals, hence the 0,01 dB budget.
    """
    coefficients, surfaces = _cnossos_road_2015_inputs()
    worst = 0.0
    for case in ref.cnossos_road_workbook_cases():
        traffic = [
            ph.environment.RoadTraffic(
                ph.environment.RoadVehicleCategory(c),
                float(case[f"q_{c}"]),
                float(case[f"v_{c}"]),
                studded_fraction=0.5 if c == "1" else 0.0,
            )
            for c in ("1", "2", "3", "4a", "4b")
        ]
        result = ph.environment.road_source_power(
            traffic,
            surface=surfaces[case["surface"]],
            temperature=float(case["temperature_c"]),
            gradient=float(case["gradient_pct"]),
            studded_months=float(case["studded_months"]),
            junction_distance=float(case["junction_distance_m"]),
            junction_type=ph.environment.JunctionType(int(case["junction_type"])),
            coefficients=coefficients,
        )
        for got, band in zip(
            result.total_line_power, ref.CNOSSOS_ROAD_BANDS, strict=True
        ):
            worst = max(worst, abs(float(got) - float(case[f"lw_{band}"])))
    return numeric(
        0.0,
        worst,
        0.01,
        unit="dB",
        places=4,
        expected_label="<= 0.01 dB on 480 published band levels (60 cases)",
    )


@register(
    _CNOSSOS_ROAD,
    "Directive (EU) 2021/1226 Annex pt (19)(a), Table F-1",
    "Rolling and propulsion coefficients, 5 categories x 4 rows x 8 bands",
)
def _chk_cnossos_road_table_f1() -> Outcome:
    bad = 0
    for category, expected in ref.CNOSSOS_ROAD_TABLE_F1.items():
        pairs = (
            (
                ph.environment.ROAD_COEFFICIENTS.rolling_a[category],
                expected["AR"],
            ),
            (
                ph.environment.ROAD_COEFFICIENTS.rolling_b[category],
                expected["BR"],
            ),
            (
                ph.environment.ROAD_COEFFICIENTS.propulsion_a[category],
                expected["AP"],
            ),
            (
                ph.environment.ROAD_COEFFICIENTS.propulsion_b[category],
                expected["BP"],
            ),
        )
        bad += sum(
            1 for got, want in pairs for a, b in zip(got, want, strict=True) if a != b
        )
    return numeric(
        0.0,
        float(bad),
        0.0,
        unit="mismatches",
        places=0,
        expected_label="160 coefficients identical",
    )


@register(
    _CNOSSOS_ROAD,
    "Directive (EU) 2021/1226 Annex pt (19)(b), Table F-4",
    "Road-surface coefficients, 15 surfaces x 5 categories x (8 alpha + beta)",
)
def _chk_cnossos_road_table_f4() -> Outcome:
    bad = 0
    for surface in ph.environment.RoadSurface:
        row = ph.environment.road_surface_coefficients(surface)
        expected = ref.CNOSSOS_ROAD_TABLE_F4[surface.value]
        for category in ("1", "2", "3", "4a", "4b"):
            key = category if category in expected else "4a/4b"
            bad += sum(
                1
                for a, b in zip(row.alpha[category], expected[key][0], strict=True)
                if a != b
            )
            bad += int(row.beta[category] != expected[key][1])
    return numeric(
        0.0,
        float(bad),
        0.0,
        unit="mismatches",
        places=0,
        expected_label="675 stored coefficients identical",
    )


@register(
    _CNOSSOS_ROAD,
    "Directive (EU) 2015/996 Appendix F, Tables F-2 and F-3",
    "Studded-tyre and junction coefficients, unchanged since 2015",
)
def _chk_cnossos_road_tables_f2_f3() -> Outcome:
    bad = sum(
        1
        for got, want in (
            (
                ph.environment.ROAD_COEFFICIENTS.studded_a,
                ref.CNOSSOS_ROAD_TABLE_F2["ai"],
            ),
            (
                ph.environment.ROAD_COEFFICIENTS.studded_b,
                ref.CNOSSOS_ROAD_TABLE_F2["bi"],
            ),
        )
        for a, b in zip(got, want, strict=True)
        if a != b
    )
    for category, expected in ref.CNOSSOS_ROAD_TABLE_F3.items():
        bad += int(
            ph.environment.ROAD_COEFFICIENTS.junction_c[category]
            != (expected[1], expected[2])
        )
    return numeric(
        0.0,
        float(bad),
        0.0,
        unit="mismatches",
        places=0,
        expected_label="36 coefficients identical",
    )


@register(
    _CNOSSOS_ROAD,
    "Directive (EU) 2015/996 Annex II 2.2.4 / 2.2.11",
    "Sound power at v_ref = 70 km/h under reference conditions, dB re 1 pW",
)
def _chk_cnossos_road_reference_conditions() -> Outcome:
    """At the reference conditions every correction vanishes identically, so
    the rolling and propulsion powers are the Table F-1 coefficients A_R, A_P.
    """
    worst = 0.0
    for category in ("1", "2", "3", "4a", "4b"):
        rolling = ph.environment.road_rolling_noise(category, 70.0)
        propulsion = ph.environment.road_propulsion_noise(category, 70.0)
        for got, want in (
            (rolling, ph.environment.ROAD_COEFFICIENTS.rolling_a[category]),
            (
                propulsion,
                ph.environment.ROAD_COEFFICIENTS.propulsion_a[category],
            ),
        ):
            worst = max(
                worst, max(abs(float(a) - b) for a, b in zip(got, want, strict=True))
            )
    return numeric(
        0.0,
        worst,
        0.0,
        unit="dB",
        places=6,
        expected_label="exactly A_R,i,m and A_P,i,m",
    )


@register(
    _CNOSSOS_ROAD,
    "Directive (EU) 2021/1226 Annex pt (8)(b)",
    "Octave-band A-weighting AWC_f,i prescribed by 2.5.5, dB",
)
def _chk_cnossos_a_weighting() -> Outcome:
    bad = sum(
        1
        for a, b in zip(
            ph.environment.CNOSSOS_A_WEIGHTING,
            ref.CNOSSOS_A_WEIGHTING_TABLE,
            strict=True,
        )
        if a != b
    )
    return numeric(
        0.0,
        float(bad),
        0.0,
        unit="mismatches",
        places=0,
        expected_label="8 values identical",
    )


# ===========================================================================
# Wind-turbine noise (IEC 61400-11)
# ===========================================================================
_WIND_TURBINE = "Wind-turbine noise (IEC 61400-11)"


@register(
    _WIND_TURBINE,
    "IEC 61400-11:2012 Formula 30",
    "Critical bandwidth about a 500 Hz tone, Hz",
)
def _chk_wt_critical_bandwidth() -> Outcome:
    from phonometry.environment.sources.wind_turbine import critical_bandwidth

    expected = 25.0 + 75.0 * (1.0 + 1.4 * (500.0 / 1000.0) ** 2) ** 0.69
    return numeric(expected, critical_bandwidth(500.0), 1e-6, unit="Hz", places=3)


@register(
    _WIND_TURBINE,
    "IEC 61400-11:2012 Formula 26",
    "Apparent sound power level of a single band, dB re 1 pW",
)
def _chk_wt_apparent_power() -> Outcome:
    r1 = 150.0
    expected = 100.0 - 6.0 + 10.0 * math.log10(4.0 * math.pi * r1**2)
    return numeric(
        expected,
        ph.environment.apparent_sound_power_level([100.0], r1),
        1e-4,
        unit="dB",
        places=4,
    )


@register(
    _WIND_TURBINE,
    "IEC 61400-11:2012 Formulae 31-34",
    "Tonal audibility of a synthetic clean tone, dB",
)
def _chk_wt_tonal_audibility() -> Outcome:
    df = 2.0
    freqs = np.arange(440.0, 560.0 + df, df)
    levels = np.full(freqs.size, 30.0)
    levels[int(np.argmin(np.abs(freqs - 500.0)))] = 60.0
    res = ph.environment.wind_turbine_tonality(levels, freqs)
    return numeric(16.38, res.tonal_audibility, 6e-2, unit="dB", places=2)
