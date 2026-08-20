#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the predicted panel sound reduction index (Bies 5e Section 7.2).

Anchored on: the exact mass law (Bies Eq. 7.40) rising 6 dB per octave and 6 dB
per doubling of surface mass; the field-incidence correction of 5.5 dB (1/3
octave) / 4.0 dB (octave) (Eq. 7.42); Sharp's coincidence-dip design-chart point
B ``TL = 20 lg(fc m'') + 10 lg eta - 44`` (Fig. 7.9a); the mass-air-mass
resonance ``f0 = 60 sqrt((m1 + m2)/(m1 m2 d))`` (Bies Eq. 7.62 / Hopkins
Eq. 4.73); the double-wall low-frequency limit (below f0 = mass law of the
combined mass, Eq. 7.64); and the measured 6 mm glass / 12.5 mm plasterboard
curves of Hopkins Fig. 4.8 / 4.9 as a few-dB sanity check in the mass-law range.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import building, vibration
from phonometry.materials import miki

# ISO 717-1 one-third-octave band centres, 100 Hz to 3150 Hz.
BANDS = np.array(
    [
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
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Mass law (Bies Eq. 7.40 / 7.42).
# ---------------------------------------------------------------------------
def test_mass_law_six_db_per_octave() -> None:
    lo = building.mass_law_transmission_loss(500.0, 20.0, incidence="normal")
    hi = building.mass_law_transmission_loss(1000.0, 20.0, incidence="normal")
    assert float(hi - lo) == pytest.approx(6.02, abs=0.01)


def test_mass_law_six_db_per_mass_doubling() -> None:
    light = building.mass_law_transmission_loss(500.0, 20.0, incidence="normal")
    heavy = building.mass_law_transmission_loss(500.0, 40.0, incidence="normal")
    assert float(heavy - light) == pytest.approx(6.02, abs=0.01)


def test_field_incidence_correction_values() -> None:
    assert building.field_incidence_correction("third") == 5.5
    assert building.field_incidence_correction("octave") == 4.0
    normal = building.mass_law_transmission_loss(500.0, 20.0, incidence="normal")
    field = building.mass_law_transmission_loss(
        500.0, 20.0, incidence="field", band="third"
    )
    assert float(normal - field) == pytest.approx(5.5)


def test_mass_law_hand_value() -> None:
    # TL_normal = 10 lg(1 + (pi f m / rho0 c0)**2).
    f, m = 500.0, 20.0
    ratio = math.pi * f * m / (1.205 * 343.0)
    expected = 10.0 * math.log10(1.0 + ratio**2)
    assert building.mass_law_transmission_loss(
        f, m, incidence="normal"
    ) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Single panel, Sharp's method (Bies 7.2.4.1).
# ---------------------------------------------------------------------------
def test_single_panel_coincidence_dip_point_b() -> None:
    m, eta = 15.0, 0.024
    bp = vibration.plate_bending_stiffness(6.2e10, 0.006, 0.24)
    fc = vibration.coincidence_frequency(m, bp)
    tl_fc = building.single_panel_transmission_loss(
        [fc], m, critical_frequency=fc, loss_factor=eta
    ).transmission_loss[0]
    point_b = 20.0 * math.log10(fc * m) + 10.0 * math.log10(eta) - 44.0
    assert float(tl_fc) == pytest.approx(point_b, abs=0.5)


def test_single_panel_dip_at_coincidence() -> None:
    m = 15.0
    bp = vibration.plate_bending_stiffness(6.2e10, 0.006, 0.24)
    fc = vibration.coincidence_frequency(m, bp)
    res = building.single_panel_transmission_loss(
        BANDS, m, critical_frequency=fc, loss_factor=0.024
    )
    # A local dip must sit near the coincidence frequency.
    dip_band = BANDS[int(np.argmin(np.abs(BANDS - fc)))]
    below = res.transmission_loss[BANDS == 1000.0][0]
    at_dip = res.transmission_loss[dip_band == BANDS][0]
    assert at_dip < below


def test_single_panel_glass_rating_realistic() -> None:
    # 6 mm float glass: catalogue Rw is about 31-32 dB.
    m = 15.0
    bp = vibration.plate_bending_stiffness(6.2e10, 0.006, 0.24)
    fc = vibration.coincidence_frequency(m, bp)
    res = building.single_panel_transmission_loss(
        BANDS, m, critical_frequency=fc, loss_factor=0.024
    )
    assert 29 <= res.rating().rating <= 34


def test_single_panel_matches_glass_measurement_masslaw_range() -> None:
    # Hopkins Fig. 4.8, 6 mm glass, below coincidence, within a few dB.
    m = 15.0
    bp = vibration.plate_bending_stiffness(6.2e10, 0.006, 0.24)
    fc = vibration.coincidence_frequency(m, bp)
    freqs = np.array([250.0, 500.0, 1000.0])
    measured = np.array([25.0, 28.5, 33.0])
    res = building.single_panel_transmission_loss(
        freqs, m, critical_frequency=fc, loss_factor=0.024
    )
    np.testing.assert_allclose(res.transmission_loss, measured, atol=4.0)


def test_single_panel_from_bending_stiffness() -> None:
    m = 15.0
    bp = vibration.plate_bending_stiffness(6.2e10, 0.006, 0.24)
    a = building.single_panel_transmission_loss(BANDS, m, bending_stiffness=bp)
    b = building.single_panel_transmission_loss(
        BANDS, m, critical_frequency=vibration.coincidence_frequency(m, bp)
    )
    np.testing.assert_allclose(a.transmission_loss, b.transmission_loss)


def test_single_panel_requires_fc_or_stiffness() -> None:
    with pytest.raises(ValueError, match="critical_frequency"):
        building.single_panel_transmission_loss(BANDS, 15.0)


# ---------------------------------------------------------------------------
# Double wall (Bies 7.2.6).
# ---------------------------------------------------------------------------
def test_mass_spring_mass_resonance_closed_form() -> None:
    m1, m2, d = 12.16, 12.16, 0.1
    f0 = building.mass_spring_mass_resonance(m1, m2, d)
    # Bies/Hopkins round the air-cavity constant to 60; the exact rho0 c0**2
    # form gives 59.9, so the design constant agrees to about 0.2 %.
    assert f0 == pytest.approx(60.0 * math.sqrt((m1 + m2) / (m1 * m2 * d)), rel=5e-3)
    # Exact closed form f0 = (1/2pi) sqrt(rho0 c0**2 / d * (m1+m2)/(m1 m2)).
    stiffness = 1.205 * 343.0**2 / d
    exact = math.sqrt(stiffness * (m1 + m2) / (m1 * m2)) / (2.0 * math.pi)
    assert f0 == pytest.approx(exact, rel=1e-9)


def test_double_wall_below_f0_is_total_mass_law() -> None:
    m1, m2, d = 12.16, 12.16, 0.1
    f0 = building.mass_spring_mass_resonance(m1, m2, d)
    fb = 0.5 * f0
    dw = building.double_wall_transmission_loss([fb], m1, m2, d).transmission_loss[0]
    ml = building.mass_law_transmission_loss(fb, m1 + m2)
    assert float(dw) == pytest.approx(float(ml))


def test_double_wall_continuous_at_limiting_frequency() -> None:
    m1, m2, d = 12.16, 12.16, 0.1
    f_l = 343.0 / (2.0 * math.pi * d)
    lo = building.double_wall_transmission_loss(
        [f_l * 0.999], m1, m2, d
    ).transmission_loss[0]
    hi = building.double_wall_transmission_loss(
        [f_l * 1.001], m1, m2, d
    ).transmission_loss[0]
    # 20 lg(2 k d) = 6.02 dB at f_l, the +6 dB high branch: continuous to ~0.05 dB.
    assert abs(float(hi - lo)) < 0.1


def test_double_wall_beats_single_leaf_above_resonance() -> None:
    m1, m2, d = 12.16, 12.16, 0.1
    dw = building.double_wall_transmission_loss([500.0], m1, m2, d).transmission_loss[0]
    single = building.mass_law_transmission_loss(500.0, m1 + m2)
    assert float(dw) > float(single) + 10.0


def test_double_wall_porous_fill_lowers_resonance() -> None:
    m1, m2, d = 12.16, 12.16, 0.1
    f0_air = building.mass_spring_mass_resonance(m1, m2, d)
    # Flow resistivity chosen so f/sigma stays within the Miki fit range at f0.
    medium = miki([f0_air], 7000.0)
    f0_fill = building.mass_spring_mass_resonance(m1, m2, d, cavity_medium=medium)
    assert f0_fill < f0_air


def test_double_wall_degenerate_f0_above_fl_is_partitioned() -> None:
    # Very light leaves with a wide gap push f0 above f_l = c0/(2 pi d); the
    # transition band collapses but the masks must stay a strict partition
    # (no silent overwrite), giving finite, monotone-ish values everywhere.
    m1 = m2 = 0.3  # kg/m2 (thin membranes)
    d = 0.3  # 300 mm gap
    f0 = building.mass_spring_mass_resonance(m1, m2, d)
    f_l = 343.0 / (2.0 * math.pi * d)
    assert f0 > f_l  # the degenerate regime
    res = building.double_wall_transmission_loss(BANDS, m1, m2, d)
    assert np.all(np.isfinite(res.transmission_loss))
    # Below f0 it is still the combined-mass law; above f0 it is the +6 branch.
    lo = float(res.transmission_loss[f0 >= BANDS][0])
    assert lo == pytest.approx(
        float(building.mass_law_transmission_loss(BANDS[f0 >= BANDS][0], m1 + m2))
    )


def test_double_wall_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="gap"):
        building.double_wall_transmission_loss(BANDS, 10.0, 10.0, -0.1)


def test_plot_language_spanish_and_validation() -> None:
    # The .plot() renderer accepts a language option: Spanish localises the
    # axis labels and title, and an unknown code is rejected up front.
    m = 15.0
    bp = vibration.plate_bending_stiffness(6.2e10, 0.006, 0.24)
    fc = vibration.coincidence_frequency(m, bp)
    res = building.single_panel_transmission_loss(
        BANDS, m, critical_frequency=fc, loss_factor=0.024
    )
    ax = res.plot(language="es")
    assert "Frecuencia" in ax.get_xlabel()
    assert "reducción acústica" in ax.get_ylabel()
    assert "Aislamiento" in ax.get_title()
    # English (the default) stays byte-for-byte the original text.
    ax_en = res.plot()
    assert ax_en.get_xlabel() == "Frequency [Hz]"
    assert ax_en.get_ylabel() == "Sound reduction index $R$ [dB]"
    with pytest.raises(ValueError, match="Unknown language"):
        res.plot(language="xx")


# ---------------------------------------------------------------------------
# Plateau method and Cremer coincidence recovery (Norton & Karczub 2003)
#
# M. P. Norton and D. G. Karczub, *Fundamentals of Noise and Vibration Analysis
# for Engineers* (2nd ed., CUP 2003), Section 3.9.1 (Eqs. 3.104, 3.106, 3.110
# and Table 3.1), with the published answers to problems 3.11 and 3.14.
#
# Clean-room oracle:
#
# * **Problem 3.11** (printed p. 580; answer p. 611). An 8 m x 3 m solid brick
#   wall, 110 mm thick, 2,1 kg/m^2 per mm: printed octave-band TL
#   35,8 / 37 / 37 / 42,5 / 52,5 / 62,5 / 72,5 dB from 63 Hz to 4 kHz. The
#   63 Hz value is the field-incidence mass law and the 125/250 Hz values are
#   the tabulated brick plateau height, so all three are reproduced exactly.
#   The four values above point B are read off the design chart of Fig. 3.24
#   and sit 0,8 dB below the analytic construction; what *is* exact there is
#   the 10 dB per octave slope, which the printed values carry to the digit.
# * **Problem 3.14** (printed p. 580; answer p. 611). A 20 mm particle-board
#   panel: printed "diffuse field" TL 6,4 / 12,2 / 18,1 / 24,1 / 30,1 / 36,1 /
#   42,1 / 48,1 dB from 31,5 Hz to 4 kHz, then 27 dB at 8 kHz and 38,6 dB at
#   16 kHz. With Norton's Appendix 4 particle board (750 kg/m^3, fc.t =
#   97,7 m/s) and rho0 c0 = 415 rayl, the first eight follow Eq. (3.104) and
#   the last two Eq. (3.110). See docs/ERRATA.md for the loss factor.
#
# Norton uses rho0 = 1,21 kg/m^3 with c0 = 343 m/s throughout.
# ---------------------------------------------------------------------------

NORTON_AIR_DENSITY = 1.21

#: Norton octave bands of problem 3.11.
P311_BANDS = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
#: Printed answer to problem 3.11, in dB.
P311_TL = np.array([35.8, 37.0, 37.0, 42.5, 52.5, 62.5, 72.5])

#: Norton octave bands of problem 3.14.
P314_BANDS = np.array(
    [31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
)
#: Printed "diffuse field" answer to problem 3.14, in dB.
P314_TL = np.array([6.4, 12.2, 18.1, 24.1, 30.1, 36.1, 42.1, 48.1, 27.0, 38.6])


def _brick_wall() -> building.SoundReductionResult:

    return building.plateau_transmission_loss(
        P311_BANDS,
        material="brick",
        thickness_mm=110.0,
        air_density=NORTON_AIR_DENSITY,
    )


#: Printed Table 3.1 (p. 241) in full: surface density in kg/m^2 per mm of
#: thickness, coincidence plateau height in dB, and the frequency ratio B/A.
#: The table *is* the feature, so every row is pinned, not just a sample.
NORTON_TABLE_31 = {
    "aluminium": (2.66, 29.0, 11.0),
    "brick": (2.10, 37.0, 4.5),
    "concrete": (2.28, 38.0, 4.5),
    "glass": (2.47, 27.0, 10.0),
    "lead": (11.20, 56.0, 4.0),
    "plaster": (1.71, 30.0, 8.0),
    "plywood": (0.57, 19.0, 6.5),
    "steel": (7.60, 40.0, 11.0),
}


def test_plateau_table_matches_norton_table_31() -> None:
    # Table 3.1: surface density per mm, plateau height and the B/A ratio of
    # the eight tabulated materials, transcribed from the printed table. The
    # whole mapping is compared, so neither a corrupted number nor an extra
    # material can slip through.

    assert building.PLATEAU_MATERIALS == NORTON_TABLE_31


@pytest.mark.parametrize(("material", "row"), sorted(NORTON_TABLE_31.items()))
def test_plateau_table_row_matches_the_printed_row(
    material: str, row: tuple[float, float, float]
) -> None:
    # One case per material, so a corrupted row names itself in the failure.

    assert building.PLATEAU_MATERIALS[material] == row


def test_plateau_problem_311_mass_law_and_plateau_exactly() -> None:
    # 63 Hz sits on the field-incidence mass law and 125/250 Hz on the brick
    # plateau: all three reproduce the printed answer digit for digit.
    res = _brick_wall()
    np.testing.assert_allclose(res.transmission_loss[:3], P311_TL[:3], atol=0.05)
    assert res.model == "plateau"
    assert res.plateau_height == 37.0


def test_plateau_problem_311_recovery_slope_is_exactly_10_db_per_octave() -> None:
    # Above point B the printed answers rise by exactly 10 dB per octave; so
    # does the construction, and the two agree to within the 0,8 dB offset of
    # reading Fig. 3.24 by eye.
    res = _brick_wall()
    above = res.transmission_loss[3:]
    np.testing.assert_allclose(np.diff(above), 10.0, atol=1e-9)
    np.testing.assert_allclose(np.diff(P311_TL[3:]), 10.0, atol=1e-9)
    np.testing.assert_allclose(above, P311_TL[3:], atol=1.0)


def test_plateau_construction_points() -> None:
    # Point A is where the mass-law line reaches the plateau height and point B
    # is B/A times higher, so the whole plateau spans exactly the tabulated
    # frequency ratio and its two ends sit at the plateau height.
    res = _brick_wall()
    assert res.plateau_start is not None
    assert res.plateau_end is not None
    assert res.plateau_end / res.plateau_start == pytest.approx(4.5, rel=1e-12)
    mass_law = building.mass_law_transmission_loss(
        res.plateau_start,
        231.0,
        incidence="field",
        field_correction=5.0,
        air_density=NORTON_AIR_DENSITY,
    )
    assert float(mass_law) == pytest.approx(37.0, abs=1e-9)


def test_plateau_degenerates_to_the_mass_law_below_point_a() -> None:
    # Well below the plateau the estimate *is* the field-incidence mass law:
    # 6 dB per octave and 6 dB per doubling of surface mass.

    low = np.array([20.0, 40.0])
    res = building.plateau_transmission_loss(
        low,
        material="brick",
        thickness_mm=110.0,
        air_density=NORTON_AIR_DENSITY,
    )
    reference = building.mass_law_transmission_loss(
        low,
        231.0,
        incidence="field",
        field_correction=5.0,
        air_density=NORTON_AIR_DENSITY,
    )
    np.testing.assert_allclose(res.transmission_loss, reference, rtol=1e-12)
    assert res.transmission_loss[1] - res.transmission_loss[0] == pytest.approx(
        6.0, abs=0.05
    )


def test_plateau_explicit_panel_matches_the_table() -> None:
    # Giving the three plateau numbers by hand must equal the tabulated route.

    by_table = _brick_wall()
    by_hand = building.plateau_transmission_loss(
        P311_BANDS,
        mass_per_area=2.10 * 110.0,
        plateau_height=37.0,
        frequency_ratio=4.5,
        air_density=NORTON_AIR_DENSITY,
    )
    np.testing.assert_allclose(
        by_hand.transmission_loss, by_table.transmission_loss, rtol=1e-12
    )


@pytest.mark.parametrize(("material", "row"), sorted(NORTON_TABLE_31.items()))
def test_plateau_construction_uses_every_column_of_the_named_row(
    material: str, row: tuple[float, float, float]
) -> None:
    # The named route must read all three columns of its own row: the surface
    # density feeds the mass-law line (hence point A), the plateau height is
    # reported as such, and the B/A ratio places point B.

    density_per_mm, height, ratio = row
    thickness_mm = 10.0
    bands = np.array([31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    res = building.plateau_transmission_loss(
        bands,
        material=material,
        thickness_mm=thickness_mm,
        air_density=NORTON_AIR_DENSITY,
    )
    assert res.plateau_height == height
    assert res.plateau_end == pytest.approx(ratio * res.plateau_start, rel=1e-12)
    # Point A is where the field-incidence mass law of this row's surface
    # density reaches the plateau height, so the two must agree there.
    at_a = building.mass_law_transmission_loss(
        np.array([res.plateau_start]),
        density_per_mm * thickness_mm,
        incidence="field",
        field_correction=5.0,
        air_density=NORTON_AIR_DENSITY,
    )
    assert at_a[0] == pytest.approx(height, abs=1e-9)


def test_cremer_problem_314_printed_answers() -> None:
    # Field-incidence mass law below fc (Eq. 3.104) and Cremer's Eq. (3.110)
    # above it, against the printed "diffuse field" column of problem 3.14.
    res = building.single_panel_transmission_loss(
        P314_BANDS,
        750.0 * 0.020,
        critical_frequency=97.7 / 0.020,
        loss_factor=1.5e-3,
        coincidence_model="cremer",
        field_correction=5.0,
        air_density=NORTON_AIR_DENSITY,
    )
    assert res.model == "cremer-single"
    np.testing.assert_allclose(res.transmission_loss, P314_TL, atol=0.1)


def test_cremer_is_floored_at_the_coincidence_frequency() -> None:
    # Eq. (3.110) is singular at f = fc. Eq. (3.109) covers that band: at the
    # critical frequency theta_CO = 90 deg and the panel "offers no resistance
    # to incident sound waves", tau = 1, TL = 0 dB. A passive panel can never
    # do worse, so no band may report tau > 1.
    fc = 2000.0
    res = building.single_panel_transmission_loss(
        np.array([0.999 * fc, fc, 1.0005 * fc, 1.05 * fc]),
        15.0,
        critical_frequency=fc,
        loss_factor=0.01,
        coincidence_model="cremer",
    )
    assert res.transmission_loss[1] == 0.0
    assert np.all(res.transmission_loss >= 0.0)
    assert np.all(res.transmission_coefficient <= 1.0)
    # Just above the floor the empirical line is back in charge and rising.
    assert res.transmission_loss[3] == pytest.approx(12.6, abs=0.5)


def test_cremer_rises_ten_db_per_octave_far_above_coincidence() -> None:
    # Eq. (3.110) adds 6 dB/octave of mass law to 10 lg(f/fc - 1), which tends
    # to 3 dB/octave far above fc, hence the 10 dB/octave Norton quotes.
    fc = 500.0
    f = fc * np.array([64.0, 128.0])
    res = building.single_panel_transmission_loss(
        f,
        15.0,
        critical_frequency=fc,
        loss_factor=0.01,
        coincidence_model="cremer",
    )
    assert np.diff(res.transmission_loss)[0] == pytest.approx(9.0, abs=0.15)


def test_plateau_and_physical_model_agree_in_the_mass_law_region() -> None:
    # The whole point of the plateau shortcut: below coincidence it is the same
    # curve as the physical model, and it only replaces the coincidence dip.

    thickness_mm, fc = 6.0, 2033.0  # 6 mm glass, Norton problem 3.13
    mass = 2.47 * thickness_mm
    low = np.array([125.0, 250.0])
    quick = building.plateau_transmission_loss(
        low,
        material="glass",
        thickness_mm=thickness_mm,
        field_correction=5.5,
    )
    physical = building.single_panel_transmission_loss(
        low,
        mass,
        critical_frequency=fc,
        loss_factor=0.02,
    )
    np.testing.assert_allclose(
        quick.transmission_loss, physical.transmission_loss, atol=0.3
    )


def test_plateau_plot_shades_the_coincidence_plateau() -> None:
    res = _brick_wall()
    ax = res.plot()
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("plateau" in label for label in labels)
    ax_es = res.plot(language="es")
    labels_es = [t.get_text() for t in ax_es.get_legend().get_texts()]
    assert any("meseta" in label for label in labels_es)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"material": "granite", "thickness_mm": 100.0},
        {"material": "brick"},
        {"mass_per_area": 231.0, "plateau_height": 37.0},
        {"mass_per_area": 231.0, "plateau_height": 37.0, "frequency_ratio": 1.0},
        {"mass_per_area": -1.0, "plateau_height": 37.0, "frequency_ratio": 4.5},
        {"mass_per_area": 231.0, "plateau_height": 0.0, "frequency_ratio": 4.5},
    ],
)
def test_plateau_validation(kwargs: dict[str, object]) -> None:

    with pytest.raises(ValueError):
        building.plateau_transmission_loss(P311_BANDS, **kwargs)  # type: ignore[arg-type]


def test_plateau_rejects_a_height_that_underflows_point_a() -> None:
    # Point A inverts 10 lg(1 + (pi f m''/rho0 c0)^2) = height + correction.
    # A plateau height so small that 10**((height + correction)/10) rounds to
    # exactly 1.0 leaves nothing to invert and would put point A at 0 Hz, i.e.
    # the whole spectrum on the plateau. The guard is what stops that.

    with pytest.raises(ValueError, match="sits below the mass law"):
        building.plateau_transmission_loss(
            P311_BANDS,
            mass_per_area=231.0,
            plateau_height=1e-300,
            frequency_ratio=4.5,
            field_correction=0.0,
        )


def test_plateau_rejects_bad_frequency_axis() -> None:

    with pytest.raises(ValueError, match="must be positive"):
        building.plateau_transmission_loss(
            [100.0, -1.0], material="brick", thickness_mm=110.0
        )
    with pytest.raises(ValueError, match="non-empty"):
        building.plateau_transmission_loss([], material="brick", thickness_mm=110.0)


def test_negative_field_correction_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        building.mass_law_transmission_loss(500.0, 15.0, field_correction=-1.0)


def test_unknown_coincidence_model_rejected() -> None:
    with pytest.raises(ValueError):
        building.single_panel_transmission_loss(
            BANDS, 15.0, critical_frequency=2000.0, coincidence_model="watters"
        )
