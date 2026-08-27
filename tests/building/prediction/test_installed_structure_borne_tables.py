#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the Annex D and Annex F tables of EN 12354-5:2009.

Table D.1 "Estimations for the mobility of typical construction elements"
(BS EN 12354-5:2009, PDF page 48, printed folio 46) and Table F.1 "Force level
LF re 1 pN for the ISO tapping machine in octave bands" (PDF page 61, printed
folio 59), together with the two Annex F terms Formula (18a) takes: the
adjustment term of Formula (F.3) and the multi-junction adjustment of clause
F.1.

Every tabulated cell is pinned against the printed page. Table F.1 is pinned
twice: once cell by cell, and once against the mechanics of the machine that
produces it, which is also what establishes its reference force.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import building
from phonometry.vibration.structural.point_mobility import infinite_plate_mobility

# ---------------------------------------------------------------------------
# Annex D, Table D.1 (PDF page 48, printed folio 46)
# ---------------------------------------------------------------------------
#: The "Describing quantities" column, row by row, as printed.
_TABLE_D1_COLUMN_TWO = {
    "mass": ("mass",),
    "bar_end": ("density", "longitudinal_velocity", "area"),
    "beam": ("density", "longitudinal_velocity", "thickness", "width"),
    "plate": ("density", "longitudinal_velocity", "thickness"),
    "pipe": ("density", "longitudinal_velocity", "thickness", "radius"),
    "mass_spring": ("mass", "stiffness", "loss_factor"),
}


def test_table_d1_has_the_six_printed_rows() -> None:
    """Table D.1 prints six rows, in this order, with these quantities."""
    assert list(building.TABLE_D1_QUANTITIES) == [
        "mass",
        "bar_end",
        "beam",
        "plate",
        "pipe",
        "mass_spring",
    ]
    assert building.TABLE_D1_QUANTITIES == _TABLE_D1_COLUMN_TWO


def test_table_d1_mass_row() -> None:
    """Row "Mass": |Y| = [2 pi f M]^-1."""
    assert building.typical_element_mobility(
        "mass", frequency=125.0, mass=25.0
    ) == pytest.approx(5.092958178940651e-05)


def test_table_d1_bar_end_row() -> None:
    """Row "Bar end": |Y| = [rho cL S]^-1, and no frequency in it."""
    assert building.typical_element_mobility(
        "bar_end", density=7800.0, longitudinal_velocity=5100.0, area=0.002
    ) == pytest.approx(1.2569130216189039e-05)


def test_table_d1_beam_row() -> None:
    """Row "Beam": |Y| = [7,6 rho t w sqrt(cL t f)]^-1 (radical included)."""
    assert building.typical_element_mobility(
        "beam",
        frequency=250.0,
        density=600.0,
        longitudinal_velocity=4400.0,
        thickness=0.08,
        width=0.12,
    ) == pytest.approx(7.700564949552747e-05)


def test_table_d1_plate_row() -> None:
    """Row "Plate": |Y| = [2,3 cL rho t^2]^-1, for a 140 mm concrete slab."""
    assert building.typical_element_mobility(
        "plate", density=2300.0, longitudinal_velocity=3800.0, thickness=0.14
    ) == pytest.approx(2.5380762194441e-06)


def test_table_d1_pipe_row() -> None:
    """Row "Pipe": |Y| = [63 rho t r sqrt(cL r f)]^-1 (radical included)."""
    assert building.typical_element_mobility(
        "pipe",
        frequency=500.0,
        density=7800.0,
        longitudinal_velocity=5100.0,
        thickness=0.003,
        radius=0.05,
    ) == pytest.approx(3.799430427426427e-05)


def test_table_d1_mass_spring_row() -> None:
    """Row "Mass-spring": the square root of the sum of two squares."""
    assert building.typical_element_mobility(
        "mass_spring",
        frequency=50.0,
        mass=200.0,
        stiffness=1.0e6,
        loss_factor=0.1,
    ) == pytest.approx(0.00029676786941383365)


def test_mass_spring_row_bottoms_out_at_its_own_resonance() -> None:
    """The second bracket vanishes at f0 = sqrt(s(1+eta^2)/M)/(2 pi).

    The row is the series sum of a spring and a mass mobility, so the two
    reactances cancel at f0 and only the damping term is left: a minimum of
    |Y|, not a maximum.
    """
    mass, stiffness, eta = 200.0, 1.0e6, 0.1
    f0 = math.sqrt(stiffness * (1.0 + eta**2) / mass) / (2.0 * math.pi)
    at_resonance = float(
        building.typical_element_mobility(
            "mass_spring",
            frequency=f0,
            mass=mass,
            stiffness=stiffness,
            loss_factor=eta,
        )
    )
    assert at_resonance == pytest.approx(
        2.0 * math.pi * f0 * eta / (stiffness * (1.0 + eta**2))
    )
    sweep = building.typical_element_mobility(
        "mass_spring",
        frequency=np.linspace(0.2 * f0, 5.0 * f0, 401),
        mass=mass,
        stiffness=stiffness,
        loss_factor=eta,
    )
    assert at_resonance <= float(np.min(sweep))


def test_mass_spring_row_is_the_series_sum_of_spring_and_mass() -> None:
    """|jw/(s(1+j eta)) + 1/(jwM)| is what the printed expression writes out."""
    bands = np.array([25.0, 50.0, 100.0, 200.0])
    mass, stiffness, eta = 200.0, 1.0e6, 0.1
    omega = 2.0 * math.pi * bands
    series = 1j * omega / (stiffness * (1.0 + 1j * eta)) + 1.0 / (1j * omega * mass)
    assert building.typical_element_mobility(
        "mass_spring",
        frequency=bands,
        mass=mass,
        stiffness=stiffness,
        loss_factor=eta,
    ) == pytest.approx(np.abs(series))


def test_plate_row_is_formula_f4_in_other_symbols() -> None:
    """[2,3 cL rho t^2]^-1 is Y = 1/(8 sqrt(m B')) of Formula (F.4).

    Annex F Formula (F.4) writes the same infinite-plate mobility as
    ``1/(8 sqrt(m B'))``; with ``B' = rho cL^2 t^3/12`` that is
    ``[8/sqrt(12) cL rho t^2]^-1``, and ``8/sqrt(12) = 2,309`` is the 2,3 of
    Table D.1 rounded to its two printed figures. The two therefore agree to
    0,4 %, which is that rounding and nothing else.
    """
    rho, c_l, t = 2300.0, 3800.0, 0.14
    bending = rho * c_l**2 * t**3 / 12.0
    formula_f4 = infinite_plate_mobility(bending, rho * t)
    table_d1 = float(
        building.typical_element_mobility(
            "plate", density=rho, longitudinal_velocity=c_l, thickness=t
        )
    )
    assert table_d1 == pytest.approx(float(np.real(formula_f4)), rel=5e-3)
    assert table_d1 * 2.3 == pytest.approx(
        float(np.real(formula_f4)) * 8.0 / math.sqrt(12.0)
    )


@pytest.mark.parametrize("structure", list(building.TABLE_D1_QUANTITIES))
def test_table_d1_rejects_a_quantity_that_does_not_describe_the_row(
    structure: str,
) -> None:
    """Only the quantities of the row's own second column are accepted."""
    supplied: dict[str, float] = dict.fromkeys(
        building.TABLE_D1_QUANTITIES[structure], 1.0
    )
    surplus = next(
        k for k in ("mass", "area", "stiffness", "radius", "width") if k not in supplied
    )
    kwargs: dict[str, float] = {**supplied, surplus: 1.0}
    if structure in ("mass", "beam", "pipe", "mass_spring"):
        kwargs["frequency"] = 100.0
    with pytest.raises(ValueError, match="does not describe it"):
        building.typical_element_mobility(structure, **kwargs)


def test_table_d1_names_the_missing_describing_quantity() -> None:
    with pytest.raises(
        ValueError, match=r"Table D\.1 row 'plate' .* missing longitudinal_velocity"
    ):
        building.typical_element_mobility("plate", density=2300.0, thickness=0.14)


def test_frequency_independent_rows_reject_a_frequency() -> None:
    """Two of the six expressions carry no f; passing one is an error."""
    with pytest.raises(
        ValueError,
        match=r"Table D\.1 row 'plate' is frequency-independent; do not pass 'frequency'",
    ):
        building.typical_element_mobility(
            "plate",
            frequency=125.0,
            density=2300.0,
            longitudinal_velocity=3800.0,
            thickness=0.14,
        )


def test_frequency_dependent_rows_require_a_frequency() -> None:
    with pytest.raises(ValueError, match="depends on frequency"):
        building.typical_element_mobility("mass", mass=25.0)


def test_unknown_row_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"'structure' must be one of"):
        building.typical_element_mobility("slab", density=1.0)


def test_table_d1_broadcasts_over_bands() -> None:
    bands = np.array([125.0, 250.0, 500.0])
    y = building.typical_element_mobility("mass", frequency=bands, mass=25.0)
    assert y.shape == (3,)
    # A mass mobility halves per octave.
    assert y[1] == pytest.approx(y[0] / 2.0)
    assert y[2] == pytest.approx(y[0] / 4.0)


# ---------------------------------------------------------------------------
# Annex F, Table F.1 (PDF page 61, printed folio 59)
# ---------------------------------------------------------------------------
def test_table_f1_cells_as_printed() -> None:
    """The eight cells of Table F.1, header row and value row.

    The header prints the first centre as "31"; it is the nominal 31,5 Hz
    octave band.
    """
    assert building.TABLE_F1_OCTAVE_BANDS == (
        31.5,
        63.0,
        125.0,
        250.0,
        500.0,
        1000.0,
        2000.0,
        4000.0,
    )
    assert building.TABLE_F1_FORCE_LEVEL == (
        139.0,
        142.0,
        145.0,
        148.0,
        151.0,
        154.0,
        156.0,
        156.0,
    )
    assert building.tapping_machine_force_level().tolist() == list(
        building.TABLE_F1_FORCE_LEVEL
    )


def test_table_f1_closed_form_reproduces_the_table_up_to_1_khz() -> None:
    """ "Up till about 1000 Hz this corresponds to LF = 10 lg 2,5f/10^-12"."""
    bands = np.array(building.TABLE_F1_OCTAVE_BANDS[:6])
    assert np.round(building.tapping_machine_force_level_estimate(bands)).tolist() == [
        139.0,
        142.0,
        145.0,
        148.0,
        151.0,
        154.0,
    ]


def test_table_f1_departs_from_the_closed_form_above_1_khz() -> None:
    """Above the stated limit the table flattens and the closed form does not."""
    high = np.array([2000.0, 4000.0])
    closed = building.tapping_machine_force_level_estimate(high)
    assert np.round(closed).tolist() == [157.0, 160.0]
    assert building.tapping_machine_force_level()[-2:].tolist() == [
        156.0,
        156.0,
    ]


def test_third_octave_closed_form_is_the_octave_one_less_10_lg_3() -> None:
    """0,8 f against 2,5 f: the three one-third octaves of an octave band."""
    difference = float(
        building.tapping_machine_force_level_estimate(1000.0)
        - building.tapping_machine_force_level_estimate(1000.0, bandwidth="third")
    )
    assert difference == pytest.approx(10.0 * math.log10(2.5 / 0.8))
    assert difference == pytest.approx(10.0 * math.log10(3.0), abs=0.2)


def test_table_f1_is_referred_to_1e_6_newton_not_1_piconewton() -> None:
    """The tabulated levels are the machine's own force spectrum re 1e-6 N.

    The ISO tapping machine drops five 0,5 kg hammers from 40 mm at ten
    impacts per second (ISO 10140-5), so each impact transfers a momentum
    ``m sqrt(2 g h)`` and the force is a 10 Hz impulse train whose every
    harmonic carries an r.m.s. force of ``sqrt(2) m sqrt(2 g h) n``. Summing
    the harmonics that fall in each octave band reproduces the tabulated
    values to 0,5 dB from 31,5 Hz to 1 kHz **only** when the level is read re
    1e-6 N, the reference force of ISO 1683 that EN 15657:2018 Formula (15)
    also uses. Read re the 1 pN printed in the caption of Table F.1 the same
    cells would be 120 dB away from the machine that produces them; see
    docs/ERRATA.md.
    """
    hammer, drop, rate, gravity = (
        building.TAPPING_HAMMER_MASS,
        0.04,
        10.0,
        9.81,
    )
    momentum = hammer * math.sqrt(2.0 * gravity * drop)
    harmonic = math.sqrt(2.0) * momentum * rate
    for centre, printed in zip(
        building.TABLE_F1_OCTAVE_BANDS[:6],
        building.TABLE_F1_FORCE_LEVEL[:6],
        strict=True,
    ):
        harmonics = centre * (math.sqrt(2.0) - 1.0 / math.sqrt(2.0)) / rate
        force = harmonic * math.sqrt(harmonics)
        assert 20.0 * math.log10(force / 1.0e-6) == pytest.approx(printed, abs=0.5)


def test_formula_d9a_is_flat_at_about_115_db_per_third_octave() -> None:
    """ "LWs,c = LF - 5 - 10 lg f ~ 115 dB re 1 pW per 1/3-octave" (D.9a).

    The characteristic power of the tapping machine is frequency-independent,
    which is the flat thick curve of Figure D.3 (PDF page 49, printed folio
    47) whose A-weighted total the key gives as 124 dB.
    """
    thirds = np.array([100.0, 200.0, 400.0, 800.0, 1600.0, 3150.0])
    lw = building.tapping_machine_characteristic_power_level(
        thirds,
        building.tapping_machine_force_level_estimate(thirds, bandwidth="third"),
    )
    assert np.ptp(lw) == pytest.approx(0.0, abs=1e-9)
    assert float(lw[0]) == pytest.approx(115.0, abs=1.0)


def test_formula_d9b_is_formula_19b_for_a_mass_like_source() -> None:
    """D.9b is Formula (19b) with Ys = 1/(j omega M): D_C agrees exactly."""
    bands = np.array([125.0, 250.0, 500.0, 1000.0])
    y_i = 2.5e-6  # a heavy concrete floor
    source = 1.0 / (1j * 2.0 * math.pi * bands * building.TAPPING_HAMMER_MASS)
    assert building.tapping_machine_coupling_term(bands, y_i) == pytest.approx(
        building.coupling_term(source, np.full(bands.shape, y_i, dtype=complex))
    )


def test_tapping_machine_coupling_term_takes_a_hammer_mass() -> None:
    doubled = building.tapping_machine_coupling_term(500.0, 2.5e-6, hammer_mass=1.0)
    default = building.tapping_machine_coupling_term(500.0, 2.5e-6)
    assert float(doubled) < float(default)


# ---------------------------------------------------------------------------
# Annex F: the terms Formula (18a) takes
# ---------------------------------------------------------------------------
def test_formula_f3_adjustment_term_hand_values() -> None:
    """Dsa = 10 lg(400 fc sigma / (m f^2)) for the 92 kg/m2 gypsum wall.

    The wall of the Annex I sanitary example (100 mm gypsum blocks,
    m' = 92 kg/m2), with the radiation factor of a homogeneous element below
    its 200 Hz critical frequency.
    """
    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0])
    sigma = np.minimum(1.0, np.sqrt(bands / 200.0))
    dsa = building.structure_to_airborne_adjustment(
        bands, 200.0, 92.0, radiation_factor=sigma
    )
    assert np.round(dsa, 1).tolist() == [-9.1, -13.6, -18.6, -24.6, -30.6, -36.6]


def test_formula_f3_is_negative_and_falls_20_lg_f_above_fc() -> None:
    """Above fc the radiation factor saturates and only the f^-2 is left."""
    dsa = building.structure_to_airborne_adjustment([500.0, 1000.0], 200.0, 92.0)
    assert float(dsa[0]) < 0.0
    assert float(dsa[0]) - float(dsa[1]) == pytest.approx(20.0 * math.log10(2.0))


def test_formula_f3_falls_3_db_when_the_mass_doubles() -> None:
    single = building.structure_to_airborne_adjustment(500.0, 200.0, 92.0)
    double = building.structure_to_airborne_adjustment(500.0, 200.0, 184.0)
    assert float(single) - float(double) == pytest.approx(10.0 * math.log10(2.0))


def test_clause_f1_multi_junction_adjustment_values() -> None:
    """ "dK = 4 dB for two junctions and dK = 6 dB for three junctions or more"."""
    assert building.multi_junction_adjustment(1) == 0.0
    assert building.multi_junction_adjustment(2) == 4.0
    assert building.multi_junction_adjustment(3) == 6.0
    assert building.multi_junction_adjustment(7) == 6.0


def test_clause_f1_kij_floor() -> None:
    """ "the resulting value for Kij should normally not become less than -5 dB"."""
    assert building.MINIMUM_MULTI_JUNCTION_KIJ == -5.0


def test_multi_junction_adjustment_rejects_zero_junctions() -> None:
    with pytest.raises(ValueError, match=r"'junctions' must be at least"):
        building.multi_junction_adjustment(0)
