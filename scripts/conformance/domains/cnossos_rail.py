#  Copyright (c) 2026. Jose Manuel Requena Plens
"""CNOSSOS-EU railway source (Directive 2002/49/EC Annex II, 2.3).

The railway emission model: rolling noise from the combined rail and wheel
roughness through the contact filter and the transfer functions, traction and
aerodynamic noise, impact noise from joints and switches, braking and idling,
assembled into the two equivalent source lines the method defines.

The oracle is the CIRCABC emission test set published with the method, read
through the same wrapper the test suite reads it through so the two cannot
validate against different tables.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import reference_data as ref

import phonometry as ph

from ..registry import Outcome, numeric, register

_CNOSSOS_RAIL = "CNOSSOS-EU railway source (Directive 2002/49/EC Annex II)"


def _cnossos_rail_case(case: dict[str, str]) -> Any:
    """One committed workbook case through the shipped model.

    The Commission's source-module test set was computed with the 2015
    coefficient database and the 2015 text of curve squeal, bridges and the
    vertical directivity, so the shipped equations are fed that superseded
    database, the squeal and bridge excesses are read from the case data and
    the roughness speed floor is switched off, which is what the reference
    program does.
    """
    wavelength = ref.cnossos_rail_2015_wavelength_tables()
    frequency = ref.cnossos_rail_2015_frequency_tables()
    vehicle = ref.cnossos_rail_2015_vehicles()[case["vehicle"]]
    lam = [float(w) for w in ref.CNOSSOS_RAIL_2015_WAVELENGTHS]
    idling = case["condition"] == "idling"
    traction = "traction_idling" if idling else "traction_constant"
    stock = ph.environment.RollingStock(
        axles=int(vehicle["axles"]),
        wheel_roughness=(
            lam,
            wavelength[("wheel_roughness", vehicle["wheel_roughness"])],
        ),
        contact_filter=(lam, wavelength[("contact_filter", vehicle["contact_filter"])]),
        wheel_transfer=frequency[("wheel_transfer", vehicle["wheel_transfer"], "")],
        superstructure_transfer=frequency[
            ("superstructure_transfer", case["superstructure_transfer"], "")
        ],
        traction=(
            frequency[(traction, vehicle["traction"], "A")],
            frequency[(traction, vehicle["traction"], "B")],
        ),
        aerodynamic=(
            frequency[("aerodynamic", vehicle["aerodynamic"], "A")],
            frequency[("aerodynamic", vehicle["aerodynamic"], "B")],
        ),
        aerodynamic_alpha=float(case["aero_alpha"]),
    )
    track = ph.environment.RailwayTrack(
        rail_roughness=(lam, wavelength[("rail_roughness", case["rail_roughness"])]),
        track_transfer=frequency[("track_transfer", case["track_transfer"], "")],
        impact_roughness=(
            (lam, wavelength[("impact_roughness", case["impact_roughness"])])
            if case["impact_roughness"]
            else None
        ),
        joint_density=float(case["joint_density_per_m"]),
        squeal_excess=float(case["squeal_excess_db"])
        + float(case["bridge_constant_db"]),
    )
    return ph.environment.railway_source_power(
        ph.environment.RailwayVehicle(
            stock=stock,
            flow_rate=float(case["flow_veh_per_h"]),
            speed=float(case["speed_kmh"]),
            condition=(
                ph.environment.RunningCondition.IDLING
                if idling
                else ph.environment.RunningCondition.CONSTANT
            ),
            idling_time=float(case["idling_time_h"]),
        ),
        track,
        psi=float(case["psi_deg"]),
        phi=float(case["phi_deg"]),
        reference_time=12.0,
        minimum_speed=0.0,
        directivity_edition=ph.environment.DirectivityEdition.ORIGINAL_2015,
    )


@register(
    _CNOSSOS_RAIL,
    "CIRCABC CNOSSOS-EU railway emission test set",
    "Line power of the 123 committed cases of the published test set, both source heights, 8 octave bands each, dB re 1 pW/m",
)
def _chk_cnossos_rail_workbook() -> Outcome:
    """Worst per-band deviation from the published test workbook."""
    worst = 0.0
    bands = ("63", "125", "250", "500", "1000", "2000", "4000", "8000")
    for case in ref.cnossos_rail_workbook_cases():
        result = _cnossos_rail_case(case)
        row = result.line_power[0 if case["source_height"] == "A" else 1]
        for got, band in zip(row, bands):
            worst = max(worst, abs(float(got) - float(case[f"lw_{band}"])))
    return numeric(
        0.0,
        worst,
        0.01,
        unit="dB",
        places=4,
        expected_label="<= 0.01 dB on 984 published band levels (123 cases)",
    )


@register(
    _CNOSSOS_RAIL,
    "Appendix G Tables G-1a and G-1b (roughness)",
    "Wheel roughness by brake type (3 x 32) and rail roughness by class (2 x 35), dB",
)
def _chk_cnossos_rail_roughness_tables() -> Outcome:
    bad = 0
    for brake, expected in (
        ("c", ref.CNOSSOS_RAIL_G1A_CAST_IRON),
        ("k", ref.CNOSSOS_RAIL_G1A_COMPOSITE),
        ("n", ref.CNOSSOS_RAIL_G1A_NON_TREAD),
    ):
        _, levels = ph.environment.wheel_roughness(brake)
        bad += sum(1 for a, b in zip(levels, expected) if a != b)
    for cls, expected in (("E", ref.CNOSSOS_RAIL_G1B_E), ("M", ref.CNOSSOS_RAIL_G1B_M)):
        _, levels = ph.environment.rail_roughness(cls)
        bad += sum(1 for a, b in zip(levels, expected) if a != b)
    return numeric(
        0.0,
        float(bad),
        0.0,
        unit="mismatches",
        places=0,
        expected_label="166 coefficients identical",
    )


@register(
    _CNOSSOS_RAIL,
    "Directive (EU) 2021/1226 Annex pt (20)(b), Table G-2",
    "Contact filter A3 for 5 wheel load and diameter combinations x 35 wavelengths, dB",
)
def _chk_cnossos_rail_table_g2() -> Outcome:
    bad = 0
    for key, expected in ref.CNOSSOS_RAIL_G2.items():
        _, levels = ph.environment.contact_filter(key)
        bad += sum(1 for a, b in zip(levels, expected) if a != b)
    return numeric(
        0.0,
        float(bad),
        0.0,
        unit="mismatches",
        places=0,
        expected_label="175 coefficients identical",
    )


@register(
    _CNOSSOS_RAIL,
    "Appendix G Table G-3 (transfer functions)",
    "Track transfer (8 x 24), wheel transfer (4 x 24) and superstructure transfer (24), dB per axle",
)
def _chk_cnossos_rail_table_g3() -> Outcome:
    bad = 0
    for key, expected in ref.CNOSSOS_RAIL_G3A.items():
        bad += sum(
            1 for a, b in zip(ph.environment.track_transfer(key), expected) if a != b
        )
    for diameter, expected in ref.CNOSSOS_RAIL_G3B.items():
        bad += sum(
            1
            for a, b in zip(ph.environment.wheel_transfer(diameter), expected)
            if a != b
        )
    bad += sum(
        1
        for a, b in zip(ph.environment.superstructure_transfer(), ref.CNOSSOS_RAIL_G3C)
        if a != b
    )
    return numeric(
        0.0,
        float(bad),
        0.0,
        unit="mismatches",
        places=0,
        expected_label="312 coefficients identical",
    )


@register(
    _CNOSSOS_RAIL,
    "Appendix G Tables G-4 to G-7",
    "Impact roughness (35), traction (5 x 2 x 24), aerodynamic (2 x 24) and bridge (2 x 24), dB",
)
def _chk_cnossos_rail_source_tables() -> Outcome:
    bad = 0
    _, impact = ph.environment.impact_roughness_single()
    bad += sum(1 for a, b in zip(impact, ref.CNOSSOS_RAIL_G4) if a != b)
    for key, (low, high) in ref.CNOSSOS_RAIL_G5.items():
        got_low, got_high = ph.environment.traction_sound_power(key)
        bad += sum(1 for a, b in zip(got_low, low) if a != b)
        bad += sum(1 for a, b in zip(got_high, high) if a != b)
    got_low, got_high = ph.environment.aerodynamic_sound_power()
    bad += sum(1 for a, b in zip(got_low, ref.CNOSSOS_RAIL_G6_A) if a != b)
    bad += sum(1 for a, b in zip(got_high, ref.CNOSSOS_RAIL_G6_B) if a != b)
    for key, expected in ref.CNOSSOS_RAIL_G7.items():
        bad += sum(
            1 for a, b in zip(ph.environment.bridge_transfer(key), expected) if a != b
        )
    return numeric(
        0.0,
        float(bad),
        0.0,
        unit="mismatches",
        places=0,
        expected_label="371 coefficients identical",
    )


@register(
    _CNOSSOS_RAIL,
    "Annex II 2.3.2, formula (2.3.15)",
    "Horizontal dipole directivity along the track: 10 lg(0,01) at phi = 0",
)
def _chk_cnossos_rail_horizontal_directivity() -> Outcome:
    got = float(ph.environment.horizontal_directivity(0.0)[0])
    return numeric(10.0 * math.log10(0.01), got, 1e-12, unit="dB")


@register(
    _CNOSSOS_RAIL,
    "Annex II 2.3.2, formulae (2.3.13) and (2.3.14)",
    "Aerodynamic speed law at v0 = 300 km/h reduces to Table G-6 verbatim",
)
def _chk_cnossos_rail_aerodynamic_law() -> Outcome:
    low, high = ph.environment.aerodynamic_sound_power(600.0)
    step = ref.CNOSSOS_RAIL_G6_ALPHA * math.log10(2.0)
    worst = max(
        max(abs(float(a) - b - step) for a, b in zip(low, ref.CNOSSOS_RAIL_G6_A)),
        max(abs(float(a) - b - step) for a, b in zip(high, ref.CNOSSOS_RAIL_G6_B)),
    )
    return numeric(
        0.0,
        worst,
        1e-12,
        unit="dB",
        places=6,
        expected_label="50 lg 2 = 15.051 dB on every band",
    )


@register(
    _CNOSSOS_RAIL,
    "Annex II 2.3.2, formula (2.3.12)",
    "Impact roughness at the tabulated joint density n_l = 0,01 per m",
)
def _chk_cnossos_rail_joint_density() -> Outcome:
    _, single = ph.environment.impact_roughness_single()
    frequency_grid = np.asarray(single[:24])
    got = ph.environment.impact_roughness(frequency_grid, 0.01)
    worst = float(np.max(np.abs(got - frequency_grid)))
    return numeric(
        0.0, worst, 1e-12, unit="dB", places=6, expected_label="Table G-4 verbatim"
    )
