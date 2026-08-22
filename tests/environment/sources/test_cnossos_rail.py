#  Copyright (c) 2026. Jose Manuel Requena Plens
"""CNOSSOS-EU railway source emission (Directive 2002/49/EC Annex II, 2.3).

Three kinds of assertion, kept deliberately separate because they carry very
different evidential weight:

1. **Table transcription.** Every shipped Appendix G table is pinned against
   ``tests/reference_data/``, whose constants were extracted mechanically
   from the Official Journal text of the instrument that enacted each table:
   Commission Delegated Directive (EU) 2021/1226 (OJ L 269, 28.7.2021) for
   Tables G-1b, G-2, G-3a, G-4 and G-7, Commission Directive (EU) 2015/996
   (OJ L 168, 1.7.2015) for the tables it did not replace.

2. **End to end against a published result set.** Annex II prints no worked
   example for the railway source, so the equations that combine the tables are
   pinned against the emission test workbook published by the European
   Commission with the CNOSSOS-EU source module. Because that workbook was
   computed with the **2015** coefficient database and the 2015 text of squeal,
   bridges and the vertical directivity, the cases are run with the committed
   2015 catalogue, with the squeal excess read from the data and with
   :class:`DirectivityEdition.ORIGINAL_2015`. What is verified is therefore the
   chain of equations, not the current tables, which item 1 covers.

3. **Closed form and exact identity.** Everything the Directive fixes exactly:
   the directivity at the cardinal angles, the aerodynamic law at ``v_0``, the
   joint-density law, the octave summation, the reference conditions at which a
   correction vanishes.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import reference_data as ref

from phonometry.environment.sources.cnossos_rail import (
    AERODYNAMIC_REFERENCE_SPEED,
    RAILWAY_MINIMUM_SPEED,
    RAILWAY_OCTAVE_BANDS,
    RAILWAY_THIRD_OCTAVE_BANDS,
    REFERENCE_JOINT_DENSITY,
    TRAM_MINIMUM_SPEED,
    BrakeType,
    BridgeType,
    ContactFilter,
    DirectivityEdition,
    RailPad,
    RailRoughnessClass,
    RailwayEmissionResult,
    RailwayTrack,
    RailwayVehicle,
    RollingStock,
    RoughnessInterpolation,
    RunningCondition,
    TrackBase,
    TrackCurvature,
    TrackDescriptor,
    TrackTransferClass,
    TractionVehicle,
    VehicleDescriptor,
    VehicleType,
    WheelDiameter,
    WheelMeasure,
    aerodynamic_sound_power,
    bridge_transfer,
    contact_filter,
    curve_squeal_excess,
    horizontal_directivity,
    impact_roughness,
    impact_roughness_single,
    octave_bands_from_third_octaves,
    rail_roughness,
    railway_source_power,
    rolling_sound_power,
    roughness_to_frequency,
    superstructure_transfer,
    total_effective_roughness,
    track_transfer,
    traction_sound_power,
    vertical_directivity,
    wheel_roughness,
    wheel_transfer,
)

# ---------------------------------------------------------------------------
# 1. Appendix G transcription
# ---------------------------------------------------------------------------


def test_wheel_roughness_matches_table_g1a() -> None:
    """Table G-1a, on the wavelength grid the Directive prints for it."""
    for brake, expected in (
        (BrakeType.CAST_IRON, ref.CNOSSOS_RAIL_G1A_CAST_IRON),
        (BrakeType.COMPOSITE, ref.CNOSSOS_RAIL_G1A_COMPOSITE),
        (BrakeType.NON_TREAD, ref.CNOSSOS_RAIL_G1A_NON_TREAD),
    ):
        lam, levels = wheel_roughness(brake)
        assert lam.tolist() == list(ref.CNOSSOS_RAIL_WHEEL_WAVELENGTHS)
        assert levels.tolist() == list(expected)


def test_rail_roughness_matches_table_g1b() -> None:
    """Table G-1b as replaced by (EU) 2021/1226 Annex point (20)(a)."""
    for cls, expected in (
        (RailRoughnessClass.WELL_MAINTAINED, ref.CNOSSOS_RAIL_G1B_E),
        (RailRoughnessClass.NORMAL, ref.CNOSSOS_RAIL_G1B_M),
    ):
        lam, levels = rail_roughness(cls)
        assert lam.tolist() == list(ref.CNOSSOS_RAIL_WAVELENGTHS)
        assert levels.tolist() == list(expected)


def test_contact_filter_matches_table_g2() -> None:
    """Table G-2 as replaced by (EU) 2021/1226 Annex point (20)(b)."""
    assert len(ContactFilter) == len(ref.CNOSSOS_RAIL_G2)
    for member in ContactFilter:
        lam, levels = contact_filter(member)
        assert lam.tolist() == list(ref.CNOSSOS_RAIL_WAVELENGTHS)
        assert levels.tolist() == list(ref.CNOSSOS_RAIL_G2[member.value])


def test_transfer_functions_match_table_g3() -> None:
    """Tables G-3a, G-3b and G-3c, per axle."""
    assert len(TrackTransferClass) == len(ref.CNOSSOS_RAIL_G3A)
    for member in TrackTransferClass:
        assert track_transfer(member).tolist() == list(
            ref.CNOSSOS_RAIL_G3A[member.value]
        )
    assert len(WheelDiameter) == len(ref.CNOSSOS_RAIL_G3B)
    for diameter in WheelDiameter:
        assert wheel_transfer(diameter).tolist() == list(
            ref.CNOSSOS_RAIL_G3B[diameter.value]
        )
    assert superstructure_transfer().tolist() == list(ref.CNOSSOS_RAIL_G3C)


def test_impact_traction_aerodynamic_and_bridge_tables() -> None:
    """Tables G-4, G-5, G-6 and G-7."""
    lam, impact = impact_roughness_single()
    assert lam.tolist() == list(ref.CNOSSOS_RAIL_WAVELENGTHS)
    assert impact.tolist() == list(ref.CNOSSOS_RAIL_G4)

    assert len(TractionVehicle) == len(ref.CNOSSOS_RAIL_G5)
    for vehicle in TractionVehicle:
        low, high = traction_sound_power(vehicle)
        expected_low, expected_high = ref.CNOSSOS_RAIL_G5[vehicle.value]
        assert low.tolist() == list(expected_low)
        assert high.tolist() == list(expected_high)

    low, high = aerodynamic_sound_power()
    assert low.tolist() == list(ref.CNOSSOS_RAIL_G6_A)
    assert high.tolist() == list(ref.CNOSSOS_RAIL_G6_B)

    assert len(BridgeType) == len(ref.CNOSSOS_RAIL_G7)
    for bridge in BridgeType:
        assert bridge_transfer(bridge).tolist() == list(
            ref.CNOSSOS_RAIL_G7[bridge.value]
        )


def test_table_g5_carries_the_2021_correction_at_6300_hz() -> None:
    """(EU) 2021/1226 point (20)(f) replaced 31,4 / 30,7 by 81,4 / 80,7.

    The 2015 values are about 50 dB below both neighbours of the same column,
    which is what makes them a typographical error rather than a measurement.
    """
    low, high = traction_sound_power(TractionVehicle.DIESEL_LOCO_2200)
    index = RAILWAY_THIRD_OCTAVE_BANDS.index(6300.0)
    assert (low[index], high[index]) == (81.4, 80.7)
    assert abs(low[index] - low[index - 1]) < 10.0
    assert abs(low[index] - low[index + 1]) < 10.0


def test_wheel_roughness_keeps_the_non_standard_wavelength_grid() -> None:
    """Table G-1a still prints 120, 12, 3,2 and 1,2 mm.

    (EU) 2021/1226 put the amended tables on the standard 1/3-octave series but
    left Table G-1a alone, so the two grids differ at four steps. Each table is
    therefore resampled on its own grid.
    """
    wheel = set(ref.CNOSSOS_RAIL_WHEEL_WAVELENGTHS)
    standard = set(ref.CNOSSOS_RAIL_WAVELENGTHS)
    assert wheel - standard == {120.0, 12.0, 3.2, 1.2}
    assert {125.0, 12.5, 3.15, 1.25} <= standard


# ---------------------------------------------------------------------------
# 2. End to end against the CIRCABC emission test workbook
# ---------------------------------------------------------------------------


def _catalogue_2015() -> tuple[dict, dict, dict]:
    return (
        ref.cnossos_rail_2015_wavelength_tables(),
        ref.cnossos_rail_2015_frequency_tables(),
        ref.cnossos_rail_2015_vehicles(),
    )


def _wavelengths_2015() -> list[float]:
    return [float(w) for w in ref.CNOSSOS_RAIL_2015_WAVELENGTHS]


def _case_result(case: dict[str, str]) -> RailwayEmissionResult:
    """Run one workbook case through the shipped model."""
    wavelength, frequency, vehicles = _catalogue_2015()
    lam = _wavelengths_2015()
    vehicle = vehicles[case["vehicle"]]
    idling = case["condition"] == "idling"
    condition = RunningCondition.IDLING if idling else RunningCondition.CONSTANT
    traction_key = "traction_idling" if idling else "traction_constant"
    speed = float(case["speed_kmh"])
    stock = RollingStock(
        axles=int(vehicle["axles"]),
        wheel_roughness=(
            lam,
            list(wavelength[("wheel_roughness", vehicle["wheel_roughness"])]),
        ),
        contact_filter=(
            lam,
            list(wavelength[("contact_filter", vehicle["contact_filter"])]),
        ),
        wheel_transfer=np.asarray(
            frequency[("wheel_transfer", vehicle["wheel_transfer"], "")]
        ),
        superstructure_transfer=np.asarray(
            frequency[("superstructure_transfer", case["superstructure_transfer"], "")]
        ),
        traction=(
            np.asarray(frequency[(traction_key, vehicle["traction"], "A")]),
            np.asarray(frequency[(traction_key, vehicle["traction"], "B")]),
        ),
        aerodynamic=(
            np.asarray(frequency[("aerodynamic", vehicle["aerodynamic"], "A")]),
            np.asarray(frequency[("aerodynamic", vehicle["aerodynamic"], "B")]),
        ),
        aerodynamic_alpha=float(case["aero_alpha"]),
    )
    track = RailwayTrack(
        rail_roughness=(
            lam,
            list(wavelength[("rail_roughness", case["rail_roughness"])]),
        ),
        track_transfer=np.asarray(
            frequency[("track_transfer", case["track_transfer"], "")]
        ),
        # A joint density of zero means no joint at all, so the workbook's
        # impact spectrum for that case is never read and is not committed.
        impact_roughness=(
            (lam, list(wavelength[("impact_roughness", case["impact_roughness"])]))
            if case["impact_roughness"]
            else None
        ),
        joint_density=float(case["joint_density_per_m"]),
        # The 2015 text made the bridge a constant added to the rolling noise;
        # (EU) 2021/1226 replaced it by the Table G-7 transfer function. The
        # workbook predates that, so the constant travels with the squeal
        # excess in the single rolling-noise excess the model accepts.
        squeal_excess=float(case["squeal_excess_db"])
        + float(case["bridge_constant_db"]),
        length=100.0,
    )
    return railway_source_power(
        RailwayVehicle(
            stock=stock,
            flow_rate=float(case["flow_veh_per_h"]),
            speed=speed,
            condition=condition,
            idling_time=float(case["idling_time_h"]),
        ),
        track,
        psi=float(case["psi_deg"]),
        phi=float(case["phi_deg"]),
        reference_time=12.0,
        # The reference module applies no speed floor to the roughness, and the
        # caller is expected to impose the 50 km/h of 2.3.2 itself.
        minimum_speed=0.0,
        directivity_edition=DirectivityEdition.ORIGINAL_2015,
    )


@pytest.mark.parametrize(
    "case",
    ref.cnossos_rail_workbook_cases(),
    ids=lambda c: f"v{c['vehicle']}-{c['case']}{c['source_height']}",
)
def test_workbook_cases(case: dict[str, str]) -> None:
    """Reproduce the CIRCABC railway emission workbook band by band.

    The workbook prints two decimals, so 0,01 dB is its own resolution; the
    tolerance is that rounding and nothing more.
    """
    result = _case_result(case)
    height = 0 if case["source_height"] == "A" else 1
    expected = [
        float(case[f"lw_{b}"])
        for b in ("63", "125", "250", "500", "1000", "2000", "4000", "8000")
    ]
    np.testing.assert_allclose(result.line_power[height], expected, atol=0.006)


def test_workbook_total_is_the_energy_sum_of_its_own_bands() -> None:
    """The workbook's total column is redundant with its band columns.

    Checking it is a transcription check on the committed extract itself: a
    mistyped band would no longer sum to the printed total.
    """
    for case in ref.cnossos_rail_workbook_cases():
        bands = np.array(
            [
                float(case[f"lw_{b}"])
                for b in ("63", "125", "250", "500", "1000", "2000", "4000", "8000")
            ]
        )
        total = 10.0 * np.log10(np.sum(10.0 ** (bands / 10.0)))
        assert total == pytest.approx(float(case["lw_total"]), abs=0.01)


def test_committed_cases_span_the_workbook_grid() -> None:
    """The committed extract must not quietly lose a level of any factor."""
    cases = ref.cnossos_rail_workbook_cases()
    assert len(cases) >= 120
    assert {c["source_height"] for c in cases} == {"A", "B"}
    assert {c["condition"] for c in cases} == {"constant", "idling"}
    assert {c["speed_kmh"] for c in cases} == {"30", "120", "260"}
    assert len({c["vehicle"] for c in cases}) == 20
    assert len({c["track_transfer"] for c in cases}) == 9
    assert len({c["rail_roughness"] for c in cases}) == 4
    assert len({c["impact_roughness"] for c in cases}) == 3  # two densities + none
    assert len({(c["phi_deg"], c["psi_deg"]) for c in cases}) == 4
    assert {c["curve_radius_m"] for c in cases} >= {"250", "500", "5000"}


def test_every_committed_catalogue_row_is_exercised_unmasked() -> None:
    """No committed coefficient row may sit behind a louder term in every case.

    A row that only ever appears where something 20 dB louder covers it is data
    the end-to-end agreement does not test, however green the run looks. Each
    kind of row therefore has to appear in the regime that exposes it: the
    rolling tables at source A under constant speed, the constant-speed traction
    at source B below 200 km/h where it is alone, the idling traction in an
    idling case, and the aerodynamic spectra above 200 km/h.
    """
    cases = ref.cnossos_rail_workbook_cases()
    vehicles = ref.cnossos_rail_2015_vehicles()
    exposed: set[tuple[str, str]] = set()
    for case in cases:
        vehicle = vehicles[case["vehicle"]]
        fast = float(case["speed_kmh"]) > 200.0
        if case["condition"] == "idling":
            exposed.add(("traction_idling", vehicle["traction"]))
        elif case["source_height"] == "A":
            exposed.add(("wheel_roughness", vehicle["wheel_roughness"]))
            exposed.add(("contact_filter", vehicle["contact_filter"]))
            exposed.add(("wheel_transfer", vehicle["wheel_transfer"]))
            exposed.add(("rail_roughness", case["rail_roughness"]))
            exposed.add(("track_transfer", case["track_transfer"]))
            exposed.add(("superstructure_transfer", case["superstructure_transfer"]))
            if case["impact_roughness"]:
                exposed.add(("impact_roughness", case["impact_roughness"]))
            if fast:
                exposed.add(("aerodynamic", vehicle["aerodynamic"]))
        elif fast:
            exposed.add(("aerodynamic", vehicle["aerodynamic"]))
        else:
            exposed.add(("traction_constant", vehicle["traction"]))

    committed = {
        (table, key) for table, key in ref.cnossos_rail_2015_wavelength_tables()
    }
    committed |= {
        (table, key) for table, key, _ in ref.cnossos_rail_2015_frequency_tables()
    }
    assert committed <= exposed, sorted(committed - exposed)


def test_workbook_exercises_every_physical_source() -> None:
    """Rolling, impact, squeal, traction, aerodynamic and the bridge term all
    have to be switched on somewhere in the committed extract, or the end-to-end
    agreement would be measuring less than it appears to.
    """
    cases = ref.cnossos_rail_workbook_cases()
    assert any(c["condition"] == "constant" for c in cases)  # rolling
    assert any(float(c["joint_density_per_m"]) > 0.0 for c in cases)  # impact
    assert any(float(c["squeal_excess_db"]) > 0.0 for c in cases)  # squeal
    assert any(c["condition"] == "idling" for c in cases)  # traction
    assert any(float(c["speed_kmh"]) > 200.0 for c in cases)  # aerodynamic
    assert any(float(c["bridge_constant_db"]) > 0.0 for c in cases)  # bridge
    # and every one of those with both source heights present
    for predicate in (
        lambda c: c["condition"] == "constant",
        lambda c: float(c["speed_kmh"]) > 200.0,
    ):
        assert {c["source_height"] for c in cases if predicate(c)} == {"A", "B"}


# ---------------------------------------------------------------------------
# 3. Closed form and exact identity
# ---------------------------------------------------------------------------


def test_horizontal_directivity_cardinal_values() -> None:
    """(2.3.15) is a dipole: 0 dB broadside and 10 lg 0,01 along the track."""
    assert horizontal_directivity(90.0)[0] == pytest.approx(0.0, abs=1e-12)
    assert horizontal_directivity(0.0)[0] == pytest.approx(-20.0, abs=1e-12)
    assert horizontal_directivity(180.0)[0] == pytest.approx(-20.0, abs=1e-12)
    # symmetric about the track axis and about broadside
    for angle in (17.0, 45.0, 63.0, 112.0):
        assert horizontal_directivity(angle)[0] == pytest.approx(
            horizontal_directivity(-angle)[0]
        )
        assert horizontal_directivity(angle)[0] == pytest.approx(
            horizontal_directivity(180.0 - angle)[0]
        )
    # frequency-independent, so every band carries the same value
    assert len(set(horizontal_directivity(30.0).tolist())) == 1


def test_vertical_directivity_current_text_is_zero_below_the_horizon() -> None:
    """(2.3.16) as replaced by (EU) 2021/1226 point (4)(d)."""
    freqs = np.asarray(RAILWAY_THIRD_OCTAVE_BANDS)
    assert np.all(vertical_directivity(0.0) == 0.0)
    for psi in (-1.0, -30.0, -45.0, -89.0):
        assert np.all(vertical_directivity(psi) == 0.0)
    for psi in (5.0, 30.0, 60.0, 85.0):
        expected = (
            (40.0 / 3.0)
            * (
                (2.0 / 3.0) * math.sin(2 * math.radians(psi))
                - math.sin(math.radians(psi))
            )
            * np.log10((freqs + 600.0) / 200.0)
        )
        np.testing.assert_allclose(vertical_directivity(psi), expected, atol=1e-12)


def test_vertical_directivity_2015_edition_differs_over_the_lower_half_space() -> None:
    """The superseded form is the same expression inside absolute-value bars.

    The two editions therefore disagree wherever the bracket is negative: the
    whole of ``psi < 0``, and ``psi`` above the angle where
    ``(2/3) sin 2psi = sin psi``, that is ``cos psi = 3/4``, about 41,4 deg.
    """
    for psi in (-45.0, -20.0, -1.0, 45.0, 60.0, 89.0):
        legacy = vertical_directivity(psi, edition=DirectivityEdition.ORIGINAL_2015)
        current = vertical_directivity(psi)
        assert np.all(legacy >= 0.0)
        assert not np.allclose(legacy, current)
    # and agree where the bracket is already non-negative
    for psi in (0.0, 10.0, 20.0, 40.0):
        np.testing.assert_allclose(
            vertical_directivity(psi, edition=DirectivityEdition.ORIGINAL_2015),
            vertical_directivity(psi),
            atol=1e-12,
        )
    crossover = math.degrees(math.acos(0.75))
    assert 41.0 < crossover < 42.0


def test_vertical_directivity_source_b() -> None:
    """(2.3.17): the aerodynamic source at 4,0 m only, and only below 0."""
    assert np.all(vertical_directivity(-40.0, height=2) == 0.0)
    assert np.all(vertical_directivity(40.0, height=2, aerodynamic=True) == 0.0)
    value = vertical_directivity(-60.0, height=2, aerodynamic=True)
    assert np.all(
        value == pytest.approx(10.0 * math.log10(math.cos(math.radians(60.0)) ** 2))
    )
    assert vertical_directivity(0.0, height=2, aerodynamic=True)[0] == pytest.approx(
        0.0
    )


def test_aerodynamic_law_is_exact_at_the_reference_speed() -> None:
    """(2.3.13)/(2.3.14) reduce to Table G-6 at ``v_0`` and follow 50 lg(v/v_0)."""
    low, high = aerodynamic_sound_power(AERODYNAMIC_REFERENCE_SPEED)
    assert low.tolist() == list(ref.CNOSSOS_RAIL_G6_A)
    assert high.tolist() == list(ref.CNOSSOS_RAIL_G6_B)
    doubled_low, doubled_high = aerodynamic_sound_power(
        2.0 * AERODYNAMIC_REFERENCE_SPEED
    )
    step = ref.CNOSSOS_RAIL_G6_ALPHA * math.log10(2.0)
    np.testing.assert_allclose(doubled_low - low, step, atol=1e-12)
    np.testing.assert_allclose(doubled_high - high, step, atol=1e-12)


def test_impact_roughness_joint_density_law() -> None:
    """(2.3.12): the table verbatim at ``n_l = 0,01`` and ``+10 lg(n_l/0,01)``."""
    _, single = impact_roughness_single()
    frequency_grid = np.asarray(single[:24])
    np.testing.assert_allclose(
        impact_roughness(frequency_grid, REFERENCE_JOINT_DENSITY),
        frequency_grid,
        atol=1e-12,
    )
    doubled = impact_roughness(frequency_grid, 2.0 * REFERENCE_JOINT_DENSITY)
    np.testing.assert_allclose(
        doubled - frequency_grid, 10.0 * math.log10(2.0), atol=1e-12
    )
    assert np.all(np.isneginf(impact_roughness(frequency_grid, 0.0)))


def test_total_effective_roughness_and_rolling_addition() -> None:
    """(2.3.7) is an energy sum plus the contact filter; (2.3.8) is an addition."""
    rail = np.full(24, 10.0)
    wheel = np.full(24, 10.0)
    filt = np.zeros(24)
    np.testing.assert_allclose(
        total_effective_roughness(rail, wheel, filt),
        10.0 + 10.0 * math.log10(2.0),
        atol=1e-12,
    )
    # a contact filter of -3 dB removes exactly 3 dB
    np.testing.assert_allclose(
        total_effective_roughness(rail, wheel, filt - 3.0),
        10.0 + 10.0 * math.log10(2.0) - 3.0,
        atol=1e-12,
    )
    # (2.3.8): four axles are exactly +6,0206 dB over one
    roughness = np.linspace(-5.0, 5.0, 24)
    transfer = np.linspace(80.0, 110.0, 24)
    np.testing.assert_allclose(
        rolling_sound_power(roughness, transfer, 4)
        - rolling_sound_power(roughness, transfer, 1),
        10.0 * math.log10(4.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rolling_sound_power(roughness, transfer, 1),
        roughness + transfer,
        atol=1e-12,
    )


def test_octave_summation_of_third_octaves() -> None:
    """Three equal 1/3-octave bands make one octave band 10 lg 3 dB higher."""
    octaves = octave_bands_from_third_octaves(np.full(24, 70.0))
    np.testing.assert_allclose(octaves, 70.0 + 10.0 * math.log10(3.0), atol=1e-12)
    assert len(octaves) == len(RAILWAY_OCTAVE_BANDS)
    # a single loud 1/3-octave band lands in exactly one octave band
    spectrum = np.full(24, -np.inf)
    spectrum[13] = 100.0  # 1 000 Hz
    octaves = octave_bands_from_third_octaves(spectrum)
    assert octaves[4] == pytest.approx(100.0)
    assert np.all(np.isneginf(np.delete(octaves, 4)))


def test_roughness_conversion_invariants() -> None:
    """The wavelength-to-frequency resampling of 2.3.2."""
    lam = [1000.0, 100.0, 10.0, 1.0]
    flat = [7.0, 7.0, 7.0, 7.0]
    # a flat spectrum is invariant under any speed and either rule
    for rule in RoughnessInterpolation:
        for speed in (30.0, 80.0, 250.0):
            np.testing.assert_allclose(
                roughness_to_frequency(flat, lam, speed, interpolation=rule),
                7.0,
                atol=1e-12,
            )
    # reading the table at f = v/lambda: at 36 km/h (10 m/s) the 100 Hz band
    # wants lambda = 0,1 m = 100 mm, which is tabulated, so no interpolation
    levels = [0.0, 12.0, -6.0, 3.0]
    got = roughness_to_frequency(levels, lam, 36.0, frequencies=[100.0])
    assert got[0] == pytest.approx(12.0)
    # beyond the ends of the table the end value is held
    assert roughness_to_frequency(levels, lam, 36.0, frequencies=[0.001])[0] == 0.0
    assert roughness_to_frequency(levels, lam, 36.0, frequencies=[1e9])[0] == 3.0
    # the two rules bracket each other: the energy rule never reads lower
    mid = roughness_to_frequency(levels, lam, 36.0, frequencies=[40.0])
    mid_energy = roughness_to_frequency(
        levels,
        lam,
        36.0,
        frequencies=[40.0],
        interpolation=RoughnessInterpolation.ENERGY,
    )
    assert mid_energy[0] > mid[0]


def test_curve_squeal_rule_of_2021() -> None:
    """The excess table (EU) 2021/1226 Annex point (4)(b) substituted."""
    assert curve_squeal_excess(300.0) == 8.0
    assert curve_squeal_excess(299.0) == 8.0
    assert curve_squeal_excess(301.0) == 5.0
    assert curve_squeal_excess(500.0) == 5.0
    assert curve_squeal_excess(501.0) == 0.0
    # at least 50 m of curve, except on a switch turnout
    assert curve_squeal_excess(250.0, track_length=49.0) == 0.0
    assert curve_squeal_excess(250.0, track_length=50.0) == 8.0
    assert curve_squeal_excess(250.0, track_length=1.0, turnout=True) == 8.0
    assert curve_squeal_excess(301.0, track_length=1.0, turnout=True) == 0.0
    # trams follow their own rule
    assert curve_squeal_excess(200.0, tram=True) == 5.0
    assert curve_squeal_excess(201.0, tram=True) == 0.0
    assert curve_squeal_excess(250.0, tram=True) == 0.0


# ---------------------------------------------------------------------------
# Descriptors, assembly and behaviour of the top-level function
# ---------------------------------------------------------------------------


def test_vehicle_and_track_descriptors_round_trip() -> None:
    """Tables [2.3.a] and [2.3.b]."""
    vehicle = VehicleDescriptor.from_code("a4cn")
    assert vehicle.vehicle_type is VehicleType.FREIGHT
    assert vehicle.axles == 4
    assert vehicle.brake is BrakeType.CAST_IRON
    assert vehicle.measure is WheelMeasure.NONE
    assert vehicle.code == "a4cn"
    assert VehicleDescriptor.from_code("h16nd").axles == 16

    track = TrackDescriptor.from_code("BMSNNH")
    assert track.base is TrackBase.BALLAST
    assert track.roughness is RailRoughnessClass.NORMAL
    assert track.pad is RailPad.SOFT
    assert track.curvature is TrackCurvature.HIGH
    assert track.code == "BMSNNH"


def test_descriptor_rejects_bad_codes() -> None:
    for code in ("", "a4", "z4cn", "a4xn", "a4cz"):
        with pytest.raises(ValueError, match="Table \\[2.3.a\\]"):
            VehicleDescriptor.from_code(code)
    for code in ("BMSNN", "BMSNNNN", "ZMSNNN"):
        with pytest.raises(ValueError, match="Table \\[2.3.b\\]"):
            TrackDescriptor.from_code(code)


def _reference_stock(**overrides: object) -> RollingStock:
    fields: dict[str, object] = {
        "axles": 4,
        "wheel_roughness": wheel_roughness(BrakeType.CAST_IRON),
        "contact_filter": contact_filter(ContactFilter.LOAD_50_DIAMETER_920),
        "wheel_transfer": wheel_transfer(WheelDiameter.MM_920),
        "traction": traction_sound_power(TractionVehicle.ELECTRIC_LOCO),
    }
    fields.update(overrides)
    return RollingStock(**fields)  # type: ignore[arg-type]


def _reference_track(**overrides: object) -> RailwayTrack:
    fields: dict[str, object] = {
        "rail_roughness": rail_roughness(RailRoughnessClass.NORMAL),
        "track_transfer": track_transfer(TrackTransferClass.MONOBLOCK_MEDIUM),
        "impact_roughness": impact_roughness_single(),
    }
    fields.update(overrides)
    return RailwayTrack(**fields)  # type: ignore[arg-type]


def test_flow_term_is_exact() -> None:
    """(2.3.2): doubling the flow raises the line power by exactly 10 lg 2."""
    track = _reference_track(impact_roughness=None)
    single = railway_source_power(
        RailwayVehicle(_reference_stock(), flow_rate=10.0, speed=100.0), track
    )
    doubled = railway_source_power(
        RailwayVehicle(_reference_stock(), flow_rate=20.0, speed=100.0), track
    )
    np.testing.assert_allclose(
        doubled.total_line_power - single.total_line_power,
        10.0 * math.log10(2.0),
        atol=1e-12,
    )


def test_idling_excludes_rolling_and_uses_the_idling_flow_term() -> None:
    """(2.3.4) and "when a train is idling, rolling noise shall be excluded"."""
    track = _reference_track()
    idling = railway_source_power(
        RailwayVehicle(
            _reference_stock(),
            condition=RunningCondition.IDLING,
            idling_time=1.0,
        ),
        track,
        reference_time=12.0,
    )
    assert np.all(np.isneginf(idling.components["rolling"][0]))
    low, _ = traction_sound_power(TractionVehicle.ELECTRIC_LOCO)
    expected = octave_bands_from_third_octaves(
        np.asarray(low) + 10.0 * math.log10(1.0 / (12.0 * 100.0))
    )
    np.testing.assert_allclose(
        idling.third_octave_line_power[0],
        np.asarray(low) + 10.0 * math.log10(1.0 / 1200.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(idling.line_power[0], expected, atol=1e-12)


def test_minimum_speed_floor_and_impact_exclusion() -> None:
    """2.3.2: a 50 km/h floor on the roughness, 30 km/h for a tram, and no
    impact noise below it.
    """
    plain = _reference_track(impact_roughness=None)
    slow = railway_source_power(
        RailwayVehicle(_reference_stock(), flow_rate=10.0, speed=20.0), plain
    )
    floored = railway_source_power(
        RailwayVehicle(_reference_stock(), flow_rate=10.0, speed=RAILWAY_MINIMUM_SPEED),
        plain,
    )
    # the roughness is frozen at the floor, so only the flow term moves
    shift = 10.0 * math.log10(RAILWAY_MINIMUM_SPEED / 20.0)
    np.testing.assert_allclose(
        slow.components["rolling"][0],
        floored.components["rolling"][0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        slow.total_line_power - floored.total_line_power,
        shift,
        atol=1e-12,
    )
    # below the floor the impact term is dropped altogether, at it is not above
    jointed = _reference_track()
    for speed, expect_impact in (
        (20.0, False),
        (49.9, False),
        (RAILWAY_MINIMUM_SPEED, True),
        (120.0, True),
    ):
        with_joints = railway_source_power(
            RailwayVehicle(_reference_stock(), flow_rate=10.0, speed=speed), jointed
        )
        without = railway_source_power(
            RailwayVehicle(_reference_stock(), flow_rate=10.0, speed=speed), plain
        )
        differs = bool(
            np.any(
                with_joints.components["rolling"][0]
                > without.components["rolling"][0] + 1e-9
            )
        )
        assert differs is expect_impact
    # a tram keeps its impact noise down to 30 km/h
    tram = railway_source_power(
        RailwayVehicle(
            _reference_stock(tram=True), flow_rate=10.0, speed=TRAM_MINIMUM_SPEED
        ),
        jointed,
    )
    tram_without = railway_source_power(
        RailwayVehicle(
            _reference_stock(tram=True), flow_rate=10.0, speed=TRAM_MINIMUM_SPEED
        ),
        plain,
    )
    assert np.any(tram.components["rolling"][0] > tram_without.components["rolling"][0])


def test_aerodynamic_noise_only_above_200_kmh() -> None:
    track = _reference_track(impact_roughness=None)
    aero = (np.asarray(ref.CNOSSOS_RAIL_G6_A), np.asarray(ref.CNOSSOS_RAIL_G6_B))
    below = railway_source_power(
        RailwayVehicle(_reference_stock(aerodynamic=aero), flow_rate=1.0, speed=200.0),
        track,
    )
    above = railway_source_power(
        RailwayVehicle(_reference_stock(aerodynamic=aero), flow_rate=1.0, speed=201.0),
        track,
    )
    assert np.all(np.isneginf(below.components["aerodynamic"][0]))
    assert np.all(np.isfinite(above.components["aerodynamic"][0]))


def test_bridge_is_a_separate_omnidirectional_source_at_height_a() -> None:
    """(2.3.18) as replaced by (EU) 2021/1226, with point (4)(c) omni-directionality."""
    bridge = bridge_transfer(BridgeType.PLUS_10_DBA)
    track = _reference_track(bridge_transfer=bridge, impact_roughness=None)
    on_axis = railway_source_power(
        RailwayVehicle(_reference_stock(), flow_rate=10.0, speed=100.0),
        track,
        phi=0.0,
        psi=30.0,
    )
    broadside = railway_source_power(
        RailwayVehicle(_reference_stock(), flow_rate=10.0, speed=100.0),
        track,
        phi=90.0,
        psi=0.0,
    )
    # the bridge component itself does not depend on the angles
    np.testing.assert_allclose(
        on_axis.components["bridge"][0],
        broadside.components["bridge"][0],
        atol=1e-12,
    )
    # and it survives the -20 dB dipole null that removes everything else
    without = railway_source_power(
        RailwayVehicle(_reference_stock(), flow_rate=10.0, speed=100.0),
        _reference_track(impact_roughness=None),
        phi=0.0,
        psi=30.0,
    )
    assert np.all(on_axis.total_line_power > without.total_line_power)


def test_superstructure_is_optional_and_freight_only() -> None:
    """(2.3.10) applies to vehicle type ``a`` only, and its table is all zeros."""
    track = _reference_track(impact_roughness=None)
    plain = railway_source_power(
        RailwayVehicle(_reference_stock(), flow_rate=10.0, speed=100.0), track
    )
    freight = railway_source_power(
        RailwayVehicle(
            _reference_stock(superstructure_transfer=superstructure_transfer()),
            flow_rate=10.0,
            speed=100.0,
        ),
        track,
    )
    assert np.all(freight.components["rolling"][0] >= plain.components["rolling"][0])
    assert np.any(freight.components["rolling"][0] > plain.components["rolling"][0])


def test_traffic_mix_is_summed_energetically() -> None:
    track = _reference_track(impact_roughness=None)
    one = RailwayVehicle(_reference_stock(), flow_rate=10.0, speed=100.0)
    single = railway_source_power(one, track)
    pair = railway_source_power([one, one], track)
    np.testing.assert_allclose(
        pair.total_line_power - single.total_line_power,
        10.0 * math.log10(2.0),
        atol=1e-12,
    )


def test_invalid_inputs() -> None:
    track = _reference_track()
    stock = _reference_stock()
    with pytest.raises(ValueError, match="at least one vehicle"):
        railway_source_power([], track)
    # the vehicles are built outside the raises blocks, so each block holds
    # exactly the one call whose exception is under test
    standing = RailwayVehicle(stock, flow_rate=1.0, speed=0.0)
    negative_flow = RailwayVehicle(stock, flow_rate=-1.0, speed=50.0)
    running = RailwayVehicle(stock, flow_rate=1.0, speed=50.0)
    zero_length = _reference_track(length=0.0)
    with pytest.raises(ValueError, match="positive number of km/h"):
        railway_source_power(standing, track)
    with pytest.raises(ValueError, match="non-negative number per hour"):
        railway_source_power(negative_flow, track)
    with pytest.raises(ValueError, match="positive period"):
        railway_source_power(running, track, reference_time=0.0)
    with pytest.raises(ValueError, match="positive number of metres"):
        railway_source_power(running, zero_length)
    with pytest.raises(ValueError, match="height"):
        vertical_directivity(10.0, height=3)
    with pytest.raises(ValueError, match="Unknown brake type"):
        wheel_roughness("z")
    with pytest.raises(ValueError, match="no spectrum in Table G-1b"):
        rail_roughness(RailRoughnessClass.BAD)
    with pytest.raises(ValueError, match="Unknown contact filter"):
        contact_filter((7.0, 7.0))
    with pytest.raises(ValueError, match="Unknown track transfer"):
        track_transfer("Z/Z")
    with pytest.raises(ValueError, match="Unknown wheel diameter"):
        wheel_transfer(1.0)
    with pytest.raises(ValueError, match="Unknown traction vehicle"):
        traction_sound_power("steam")
    with pytest.raises(ValueError, match="Unknown bridge type"):
        bridge_transfer("wooden")
    short_roughness = np.zeros(3)
    roughness = np.zeros(24)
    with pytest.raises(ValueError, match="must hold 24 values"):
        total_effective_roughness(short_roughness, roughness, roughness)
    with pytest.raises(ValueError, match="positive number of axles"):
        rolling_sound_power(roughness, roughness, 0)
    with pytest.raises(ValueError, match="non-negative number of m"):
        impact_roughness(roughness, -1.0)
    with pytest.raises(ValueError, match="same length"):
        roughness_to_frequency([1.0, 2.0], [1.0], 50.0)
    with pytest.raises(ValueError, match="'wavelengths' must all be positive"):
        roughness_to_frequency([1.0, 2.0], [1.0, 0.0], 50.0)
    with pytest.raises(ValueError, match="'frequencies' must all be positive"):
        roughness_to_frequency([1.0, 2.0], [1.0, 2.0], 50.0, frequencies=[0.0])
    with pytest.raises(ValueError, match="positive number of metres"):
        curve_squeal_excess(0.0)


def _emission_result() -> RailwayEmissionResult:
    """One emission as the public entry point hands it back, to be bent."""
    return railway_source_power(
        RailwayVehicle(_reference_stock(), flow_rate=10.0, speed=100.0),
        _reference_track(),
    )


def test_a_component_on_the_octave_grid_is_refused() -> None:
    """The breakdown carries no grid of its own: it is read against the 24 bands.

    Nothing in the library opens :attr:`RailwayEmissionResult.components`, so a
    component resampled onto the eight octave bands builds and draws without a
    word -- the chart is made from the octave fields and comes out with its two
    marker lines exactly as before -- and reading the eight values against
    ``third_octave_frequencies`` quotes the bridge from 50 to 250 Hz and stops
    there, sixteen bands short of the grid it is being read on.
    """
    result = _emission_result()
    octave = np.full(result.frequencies.size, 90.0)
    components = {**result.components, "bridge": (octave, octave)}
    with pytest.raises(ValueError, match=r"'components\['bridge'\]\[0\]'"):
        dataclasses.replace(result, components=components)


def test_a_component_missing_a_source_height_is_refused() -> None:
    """Each component holds the pair (source A, source B), and only the pair.

    A pair holding source A alone is short in no reading that says so: summed
    energetically over the heights it hands back source A by itself, which for
    the traction of the reference train is 1.8 to 6.3 dB below the sum of the
    two, band by band, and every one of those bands is an ordinary level.
    """
    result = _emission_result()
    traction = (result.components["traction"][0],)
    components = {**result.components, "traction": traction}
    with pytest.raises(ValueError, match=r"'components\['traction'\]'.*source height"):
        dataclasses.replace(result, components=components)


def test_a_collapsed_third_octave_table_is_refused() -> None:
    """Losing the source-height axis loses no number, so nothing looks amiss.

    A 1/3-octave table collapsed to source A alone still holds 24 ordinary
    levels, and no reader in the library opens it: the chart is drawn from the
    octave fields and comes out unchanged. The heights are then read off the
    wrong axis, and row 1, meant to be the spectrum of source B, is the 63 Hz
    level of source A, 62.0 dB, where source B's own spectrum starts at
    52.2 dB.
    """
    result = _emission_result()
    collapsed = result.third_octave_line_power[0]
    with pytest.raises(ValueError, match="'third_octave_line_power' must have 2 axes"):
        dataclasses.replace(result, third_octave_line_power=collapsed)


def test_result_plot_draws_both_heights() -> None:
    import matplotlib as mpl

    mpl.use("Agg")

    result = railway_source_power(
        RailwayVehicle(_reference_stock(), flow_rate=10.0, speed=100.0),
        _reference_track(),
    )
    ax = result.plot()
    assert ax.get_ylabel() == r"$L^{\prime}_{W,\mathrm{eq,line}}$ [dB re 1 pW/m]"
    assert len(ax.get_lines()) == 2
    ax_es = result.plot(language="es")
    assert "ferroviaria" in ax_es.get_title()
