#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Sound power radiated into a duct by fans, in-duct method: ISO 5136:2003.

Standard anchors, every one read from the rendered page:
- Table D.1 (Annex D), C3,4 for d = 0,5 m at U = +/-5, +/-15 and +/-30 m/s
  in all 27 bands: the printed oracle of the Annex A polynomial, Eq. (7),
  through Table A.4. 162 cells, each to the printed 0,1 dB.
- Eqs (D.1) to (D.3): the two-coefficient expression (1,85 + 0,038 U) dB and
  its values 2,42 and 1,28 at +/-15 m/s, printed as 2,4 and 1,3.
- Eq. (8), clause 5.3.4.3: the nose-cone and foam-ball correction
  10 lg[1/(1 - U/c)^2] with c = 340 m/s, positive on the outlet side.
- Eqs (9) to (12), clause 8: the energy average over positions, the combined
  correction C = C1 + C2 + C3,4 and the plane-wave relation with
  S = pi d^2 / 4, S0 = 1 m^2 and (rho c)_0 = 400 N s/m^3.
- Annex C, Eq. (C.1) and Table C.1: the A-weighted total over 27 bands.
- Clause 4, Table 2 and Table 3, and clause 9.2: sigma_R per band and the
  expanded uncertainty 2 sigma_R.
- Clause 1.1: the scope in duct diameter, temperature and flow velocity per
  shield, refused at the door.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import reference_data as ref

from phonometry import emission
from phonometry.emission.sound_power_in_duct import (
    _TABLE_A1,
    _TABLE_A2,
    _TABLE_A3,
    _TABLE_A5,
    _TABLE_A6,
    _annex_a_rows,
)

_BANDS = np.asarray(ref.ISO5136_BANDS, dtype=float)

#: Every printed cell of Table D.1 as (band, velocity, printed value).
_D1_CELLS = [
    (float(band), velocity, value)
    for band, row in zip(ref.ISO5136_BANDS, ref.ISO5136_TABLE_D1, strict=True)
    for velocity, value in zip(ref.ISO5136_TABLE_D1_VELOCITIES, row, strict=True)
]


# ---------------------------------------------------------------------------
# Annex D: Table D.1 cell by cell, and the worked example
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("band", "velocity", "printed"), _D1_CELLS)
def test_table_d1_reproduces_every_printed_cell(
    band: float, velocity: float, printed: float
) -> None:
    """Eq. (7) with the Table A.4 coefficients lands on every cell of Table D.1.

    The table prints C3,4 to 0,1 dB, so a cell reproduces when the polynomial
    is within half of that of the print. Four cells sit exactly on a decimal
    half (0,085; -0,185; 0,355; -0,455 dB in the 50 Hz to 250 Hz rows) and
    the print rounds them away from zero, which is why this is a tolerance
    and not a rounding rule.
    """
    computed = emission.flow_modal_correction(
        [band], velocity, ref.ISO5136_ANNEX_D_DIAMETER
    )
    assert computed.shape == (1,)
    assert abs(float(computed[0]) - printed) <= ref.ISO5136_TABLE_D1_TOLERANCE_DB + 1e-9


def test_table_d1_as_one_spectrum_per_velocity() -> None:
    """The whole band axis at once gives the same 27 values as one band at a time."""
    for column, velocity in enumerate(ref.ISO5136_TABLE_D1_VELOCITIES):
        spectrum = emission.flow_modal_correction(
            _BANDS, velocity, ref.ISO5136_ANNEX_D_DIAMETER
        )
        printed = np.asarray([row[column] for row in ref.ISO5136_TABLE_D1])
        assert spectrum.shape == _BANDS.shape
        np.testing.assert_allclose(
            spectrum, printed, atol=ref.ISO5136_TABLE_D1_TOLERANCE_DB + 1e-9
        )


@pytest.mark.parametrize(
    ("velocity", "printed"),
    [ref.ISO5136_ANNEX_D_OUTLET, ref.ISO5136_ANNEX_D_INLET],
)
def test_annex_d_worked_example(velocity: float, printed: float) -> None:
    """Eqs (D.2) and (D.3): (1,85 + 0,038 U) dB at U = +15 and -15 m/s.

    Written out: 1,85 + 0,038 x 15 = 1,85 + 0,57 = 2,42 dB, which the print
    rounds to 2,4; 1,85 + 0,038 x (-15) = 1,85 - 0,57 = 1,28 dB, printed 1,3.
    The exact products are checked to 1e-9 and the printed figures to their
    0,1 dB.
    """
    exact = ref.ISO5136_ANNEX_D_A0 + ref.ISO5136_ANNEX_D_A1 * velocity
    computed = float(
        emission.flow_modal_correction(
            [ref.ISO5136_ANNEX_D_FREQUENCY], velocity, ref.ISO5136_ANNEX_D_DIAMETER
        )[0]
    )
    assert computed == pytest.approx(exact, abs=1e-9)
    assert computed == pytest.approx(printed, abs=0.05)


# ---------------------------------------------------------------------------
# Annex A: table selection and a hand-evaluated row of each table
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("diameter", "band", "velocity", "expected"),
    [
        # Table A.1 (0,15 <= d < 0,2), "<= 630" row serving 50 Hz and 630 Hz:
        # -5,00e-02 + 2,70e-02 x 10 = -0,05 + 0,27 = 0,22 dB.
        (0.15, 50.0, 10.0, 0.22),
        (0.15, 630.0, 10.0, 0.22),
        # Table A.1, 800 Hz: the a0 cell is empty in the print and the
        # footnote makes it zero, so only 2,97e-02 x 10 = 0,297 dB remains.
        (0.15, 800.0, 10.0, 0.297),
        # Table A.2 (0,2 <= d < 0,3), 1 000 Hz at U = -20 m/s:
        # 1,75e-01 + 4,08e-02 x (-20) = 0,175 - 0,816 = -0,641 dB.
        (0.25, 1000.0, -20.0, -0.641),
        # Table A.3 (0,3 <= d < 0,5), 2 500 Hz at U = 10 m/s:
        # 2,46 + 7,49e-02 x 10 + 5,64e-04 x 100 - 3,11e-06 x 1000
        # = 2,46 + 0,749 + 0,0564 - 0,00311 = 3,26229 dB.
        (0.4, 2500.0, 10.0, 3.26229),
        # Table A.5 (0,8 <= d < 1,25), 200 Hz at U = 40 m/s:
        # -1,04 + 2,35e-02 x 40 = -1,04 + 0,94 = -0,10 dB.
        (1.0, 200.0, 40.0, -0.10),
        # Table A.6 (1,25 <= d <= 2), 125 Hz at U = 20 m/s:
        # -1,24 + 2,05e-02 x 20 = -1,24 + 0,41 = -0,83 dB.
        (1.5, 125.0, 20.0, -0.83),
    ],
)
def test_annex_a_rows_evaluated_by_hand(
    diameter: float, band: float, velocity: float, expected: float
) -> None:
    """One row of each Annex A table, multiplied out in the parameter list."""
    computed = float(emission.flow_modal_correction([band], velocity, diameter)[0])
    assert computed == pytest.approx(expected, abs=1e-9)


def test_table_a5_5000_hz_reads_the_missing_digit_as_one() -> None:
    """Table A.5 prints the a3 of 5 000 Hz as "- ,24 x 10-05", digit missing.

    The library reads it as -1,24e-05, the value between the -1,17e-05 of
    Table A.4 and the -1,27e-05 of Table A.6 at the same band; see
    docs/ERRATA.md. This pins that reading rather than the print, which
    cannot be pinned: at U = 10 m/s the row multiplies out to
    6,00 + 1,54e-01 x 10 + 1,74e-03 x 100 - 1,24e-05 x 1000 - 2,32e-07 x 10000
    = 6,00 + 1,54 + 0,174 - 0,0124 - 0,00232 = 7,69928 dB.
    """
    with pytest.warns(emission.SoundPowerWarning, match="without its leading digit"):
        computed = float(emission.flow_modal_correction([5000.0], 10.0, 1.0)[0])
    assert computed == pytest.approx(7.69928, abs=1e-9)
    assert dict(_TABLE_A5)[5000][3] == pytest.approx(-1.24e-05)


def test_the_reconstructed_coefficient_warns_only_for_its_own_cell() -> None:
    """The warning marks one cell, not the table and not the band.

    A caller is entitled to know when a coefficient is a reading rather than
    a transcription, and equally entitled not to be warned about the 161
    cells that are transcribed. The warning therefore needs both halves of
    the cell: the 5 000 Hz band and a diameter Table A.5 serves.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Same band, a diameter Table A.4 serves.
        emission.flow_modal_correction([5000.0], 10.0, 0.5)
        # Same table, a band whose whole row is printed.
        emission.flow_modal_correction([4000.0, 6300.0], 10.0, 1.0)


def test_table_a2_20_khz_row_reads_the_last_column_as_a9() -> None:
    """Table A.2 heads its a9 coefficient column "a9_0" (docs/ERRATA.md).

    The column is a_9: A.1 and A.3 to A.6 head it so, and the NOTE under
    every table sums a_i U^i from i = 0 to 10. The 20 000 Hz row is the only
    row of A.2 that populates it, and at U = 10 m/s its ten printed
    coefficients multiply out to
    1,18e+01 + 4,59e-01 x 1e1 + 1,81e-02 x 1e2 - 4,24e-04 x 1e3
    - 3,60e-05 x 1e4 + 3,70e-07 x 1e5 + 3,06e-08 x 1e6 - 1,94e-10 x 1e7
    - 8,76e-12 x 1e8 + 4,09e-14 x 1e9
    = 11,8 + 4,59 + 1,81 - 0,424 - 0,36 + 0,037 + 0,0306 - 0,00194
      - 0,000876 + 0,0000409 = 17,480 824 9 dB.
    """
    row = dict(_TABLE_A2)[20000]
    assert len(row) == 10
    assert row[9] == pytest.approx(4.09e-14)
    computed = float(emission.flow_modal_correction([20000.0], 10.0, 0.25)[0])
    assert computed == pytest.approx(17.4808249, abs=1e-9)


@pytest.mark.parametrize(
    ("diameter", "rows"),
    [
        (0.15, _TABLE_A1),
        (0.2, _TABLE_A2),
        (0.3, _TABLE_A3),
        (0.8, _TABLE_A5),
        (1.25, _TABLE_A6),
        (2.0, _TABLE_A6),
    ],
)
def test_annex_a_diameter_edges(diameter: float, rows: object) -> None:
    """Each table's range is "d_low <= d < d_high", the last one "<= 2 m".

    The printed titles: A.1 for 0,15 <= d < 0,2; A.2 for 0,2 <= d < 0,3;
    A.3 for 0,3 <= d < 0,5; A.5 for 0,8 <= d < 1,25; A.6 for 1,25 <= d <= 2,
    so a diameter on a lower edge belongs to the higher table and 2 m
    itself is still served.
    """
    assert _annex_a_rows(diameter) is rows


def test_lowest_row_serves_every_band_at_or_below_it() -> None:
    """Table A.6's "<= 100" row: 50 Hz, 63 Hz, 80 Hz and 100 Hz agree."""
    values = emission.flow_modal_correction([50.0, 63.0, 80.0, 100.0], 25.0, 1.6)
    np.testing.assert_allclose(values, values[0])
    # And the next band up is its own row: -9,02e-01 + 2,28e-02 x 25 = -0,332.
    above = float(emission.flow_modal_correction([160.0], 25.0, 1.6)[0])
    assert above == pytest.approx(-0.332, abs=1e-9)


def test_inlet_and_outlet_are_not_mirror_images() -> None:
    """U < 0 on the inlet side, U > 0 on the outlet side (Table 1 NOTE 2).

    Eq. (7) is an odd-and-even polynomial, so the same speed reads as a
    different correction on the two sides: at 1 kHz and d = 0,5 m the outlet
    figure is 1,14 dB above the inlet one (2,42 - 1,28, Eqs (D.2), (D.3)).
    """
    outlet = float(emission.flow_modal_correction([1000.0], 15.0, 0.5)[0])
    inlet = float(emission.flow_modal_correction([1000.0], -15.0, 0.5)[0])
    assert outlet - inlet == pytest.approx(1.14, abs=1e-9)


# ---------------------------------------------------------------------------
# Eq. (8): nose cone and foam ball
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shield", ["nose-cone", "foam-ball"])
@pytest.mark.parametrize(
    ("velocity", "expected"),
    [
        # -20 lg(1 - 15/340) = -20 lg(0,955882...) = +0,39186 dB
        (15.0, -20.0 * math.log10(1.0 - 15.0 / 340.0)),
        # -20 lg(1 + 15/340) = -20 lg(1,044118...) = -0,37500 dB
        (-15.0, -20.0 * math.log10(1.0 + 15.0 / 340.0)),
    ],
)
def test_eq8_omnidirectional_shields(
    shield: str, velocity: float, expected: float
) -> None:
    """10 lg[1/(1 - U/c)^2] = -20 lg(1 - U/c), c = 340 m/s, in every band.

    Positive on the outlet side and negative on the inlet side, as the
    equation is printed (the prose of 5.3.4.3 calls the correction negative;
    see docs/ERRATA.md).
    """
    values = emission.flow_modal_correction(
        _BANDS,
        velocity,
        0.5,
        shield=shield,  # type: ignore[arg-type]
    )
    np.testing.assert_allclose(values, expected, atol=1e-12)
    assert (values > 0).all() == (velocity > 0)


def test_eq8_nose_cone_at_the_scope_limit() -> None:
    """U = 20 m/s, the nose cone's limit: -20 lg(1 - 20/340) = 0,52658 dB."""
    value = float(
        emission.flow_modal_correction([1000.0], 20.0, 0.5, shield="nose-cone")[0]
    )
    assert value == pytest.approx(-20.0 * math.log10(1.0 - 20.0 / 340.0), abs=1e-12)
    assert value == pytest.approx(0.5266, abs=5e-5)


def test_eq8_speed_of_sound_is_the_one_supplied() -> None:
    """Eq. (8) is a Mach-number formula, so c moves the correction."""
    at_340 = float(
        emission.flow_modal_correction([1000.0], 20.0, 0.5, shield="nose-cone")[0]
    )
    at_343 = float(
        emission.flow_modal_correction(
            [1000.0], 20.0, 0.5, shield="nose-cone", speed_of_sound=343.2
        )[0]
    )
    assert at_343 == pytest.approx(-20.0 * math.log10(1.0 - 20.0 / 343.2), abs=1e-12)
    assert at_343 < at_340


# ---------------------------------------------------------------------------
# Clause 8: Eqs (9) to (12)
# ---------------------------------------------------------------------------
def _flat(level: float, positions: int = 3) -> np.ndarray:
    return np.full((positions, _BANDS.size), level)


def test_eq9_energy_average_over_positions() -> None:
    """Eq. (9) written out: 10 lg[(1/n) sum 10^(0,1 Lpi)] over the positions.

    Three positions at 80, 83 and 86 dB in every band:
    (10^8 + 10^8,3 + 10^8,6) / 3 = (1,000 + 1,995 + 3,981) x 10^8 / 3
    = 2,3254 x 10^8, whose level is 83,665 dB.
    """
    levels = np.stack([np.full(_BANDS.size, v) for v in (80.0, 83.0, 86.0)])
    res = emission.sound_power_in_duct(levels, _BANDS, 0.5, 0.0)
    expected = 10.0 * math.log10((1e8 + 10**8.3 + 10**8.6) / 3.0)
    np.testing.assert_allclose(res.mean_pressure_level, expected, atol=1e-9)
    assert expected == pytest.approx(83.665, abs=5e-4)


def test_eq11_averaged_spectrum_equals_identical_positions() -> None:
    """A (bands,) spectrum is the Lpm of Eq. (11): three equal rows give it."""
    one = emission.sound_power_in_duct(np.full(_BANDS.size, 78.0), _BANDS, 0.5, 12.0)
    three = emission.sound_power_in_duct(_flat(78.0), _BANDS, 0.5, 12.0)
    np.testing.assert_allclose(one.sound_power_level, three.sound_power_level)
    np.testing.assert_allclose(one.mean_pressure_level, 78.0)


def test_eq10_combined_correction_is_the_sum_of_three() -> None:
    """C = C1 + C2 + C3,4, and the corrected level is the mean level plus C."""
    c1 = np.linspace(-0.3, 0.3, _BANDS.size)
    c2 = 0.5
    res = emission.sound_power_in_duct(
        _flat(80.0),
        _BANDS,
        0.5,
        15.0,
        microphone_correction=c1,
        shield_correction=c2,
    )
    c34 = emission.flow_modal_correction(_BANDS, 15.0, 0.5)
    np.testing.assert_allclose(res.microphone_correction, c1)
    np.testing.assert_allclose(res.shield_correction, c2)
    np.testing.assert_allclose(res.flow_modal_correction, c34)
    np.testing.assert_allclose(res.combined_correction, c1 + c2 + c34)
    np.testing.assert_allclose(
        res.corrected_pressure_level, res.mean_pressure_level + res.combined_correction
    )


def test_eq12_area_and_impedance_terms() -> None:
    """LW - Lp = 10 lg(S/S0) - 10 lg(rho c / 400), S = pi d^2 / 4.

    For d = 0,5 m, S = pi x 0,25 / 4 = 0,196350 m^2 and
    10 lg(0,196350) = -7,0697 dB; the impedance term is read against the
    result's own rho c, whose value is pinned by the next test.
    """
    res = emission.sound_power_in_duct(_flat(80.0), _BANDS, 0.5, 0.0)
    area = math.pi * 0.5**2 / 4.0
    assert res.duct_area == pytest.approx(area)
    assert 10.0 * math.log10(area / ref.ISO5136_S0) == pytest.approx(-7.0697, abs=5e-5)
    expected = 10.0 * math.log10(area / ref.ISO5136_S0) - 10.0 * math.log10(
        res.characteristic_impedance / ref.ISO5136_RHO_C_0
    )
    np.testing.assert_allclose(
        res.sound_power_level - res.corrected_pressure_level, expected, atol=1e-9
    )


def test_duct_air_impedance_at_20_degrees() -> None:
    """rho c of the duct air at 20 degC and 101,325 kPa.

    c = 20,05 sqrt(273 + 20) = 343,20 m/s (the ISO 3741 form the package
    uses) and rho = p / (R T) = 101 325 / (287,05 x 293,15) = 1,2041 kg/m^3
    (the 1,204 kg/m^3 the intensity module documents for 20 degC), so
    rho c = 413,25 N s/m^3 and the impedance term of Eq. (12) is
    -10 lg(413,25 / 400) = -0,1416 dB.
    """
    res = emission.sound_power_in_duct(
        _flat(80.0), _BANDS, 0.5, 0.0, temperature=20.0, static_pressure=101.325
    )
    assert res.speed_of_sound == pytest.approx(20.05 * math.sqrt(293.0), abs=1e-9)
    assert res.characteristic_impedance == pytest.approx(413.25, abs=0.01)
    term = -10.0 * math.log10(res.characteristic_impedance / ref.ISO5136_RHO_C_0)
    assert term == pytest.approx(-0.1416, abs=5e-5)


def test_rho_c_scales_with_pressure_and_temperature() -> None:
    """Halving the static pressure halves rho and takes 3,01 dB off the term.

    The ideal gas: rho is proportional to p, so -10 lg(rho c / 400) rises by
    10 lg 2 = 3,0103 dB when p halves; c depends on temperature alone.
    """
    full = emission.sound_power_in_duct(_flat(80.0), _BANDS, 0.5, 0.0)
    half = emission.sound_power_in_duct(
        _flat(80.0), _BANDS, 0.5, 0.0, static_pressure=101.325 / 2.0
    )
    assert half.speed_of_sound == full.speed_of_sound
    assert half.characteristic_impedance == pytest.approx(
        full.characteristic_impedance / 2.0
    )
    np.testing.assert_allclose(
        half.sound_power_level - full.sound_power_level, 10.0 * math.log10(2.0)
    )


def test_end_to_end_single_band_by_hand() -> None:
    """One band, one position, the whole chain multiplied out.

    1 kHz, d = 0,5 m, U = +15 m/s, Lp = 80 dB, C1 = 0,2 dB, C2 = -0,4 dB,
    20 degC and 101,325 kPa:
    C3,4 = 2,42 dB (Eq. (D.2)); C = 0,2 - 0,4 + 2,42 = 2,22 dB;
    Lp corrected = 82,22 dB; 10 lg(S) = -7,0697 dB;
    -10 lg(413,25/400) = -0,1416 dB; LW = 82,22 - 7,0697 - 0,1416 = 75,009 dB.
    The A-weighted level of a lone 1 kHz band is the band level (C_j = 0).
    """
    res = emission.sound_power_in_duct(
        [80.0],
        [1000.0],
        0.5,
        15.0,
        microphone_correction=0.2,
        shield_correction=-0.4,
    )
    assert float(res.sound_power_level[0]) == pytest.approx(75.009, abs=1e-3)
    assert res.sound_power_level_a == pytest.approx(float(res.sound_power_level[0]))


# ---------------------------------------------------------------------------
# Annex C: the A-weighted total
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("band", "cj"), list(zip(ref.ISO5136_BANDS, ref.ISO5136_TABLE_C1, strict=True))
)
def test_table_c1_every_band(band: int, cj: float) -> None:
    """Eq. (C.1) on one band is LW + C_j, so LWA - LW reads the table back."""
    res = emission.sound_power_in_duct([80.0], [float(band)], 0.5, 0.0)
    assert res.sound_power_level_a - float(res.sound_power_level[0]) == pytest.approx(
        cj, abs=1e-9
    )


def test_eq_c1_energy_sum_over_27_bands() -> None:
    """A flat 80 dB spectrum: LWA = 80 + 10 lg sum 10^(0,1 C_j) over j = 1..27.

    The sum of 10^(C_j/10) over Table C.1 is 15,61, so the A-weighted total
    of a flat 80 dB spectrum is 80 + 11,94 = 91,94 dB. The band levels
    themselves are equal only if the corrections are, so C3,4 is switched
    off by taking U = 0 with the one Annex A table whose a0 is zero in no
    band; the levels are read back from the result instead of assumed.
    """
    res = emission.sound_power_in_duct(_flat(80.0), _BANDS, 0.5, 0.0)
    cj = np.asarray(ref.ISO5136_TABLE_C1)
    expected = 10.0 * math.log10(np.sum(10.0 ** (0.1 * (res.sound_power_level + cj))))
    assert res.sound_power_level_a == pytest.approx(expected, abs=1e-9)
    assert 10.0 * math.log10(np.sum(10.0 ** (0.1 * cj))) == pytest.approx(
        11.94, abs=0.01
    )


# ---------------------------------------------------------------------------
# Clause 4 and 9.2: reproducibility and the expanded uncertainty
# ---------------------------------------------------------------------------
def _table_2_cells() -> list[tuple[int, float]]:
    cells = []
    for low, high, sigma in ref.ISO5136_TABLE_2_SIGMA_R:
        cells.extend((band, sigma) for band in ref.ISO5136_BANDS if low <= band <= high)
    return cells


@pytest.mark.parametrize(("band", "sigma"), _table_2_cells())
def test_table_2_every_band(band: int, sigma: float) -> None:
    """Table 2 unrolled: the two printed ranges cover 80-100 Hz and 125-4 000 Hz."""
    assert float(emission.in_duct_reproducibility([float(band)])[0]) == sigma


@pytest.mark.parametrize(("band", "sigma"), ref.ISO5136_TABLE_3_SIGMA_R)
def test_table_3_extrapolated_bands(band: int, sigma: float) -> None:
    """Table 3 above 10 kHz, which clause 4 suggests without adopting."""
    assert float(emission.in_duct_reproducibility([float(band)])[0]) == sigma


def test_table_2_covers_24_bands_and_table_3_the_other_three() -> None:
    assert len(_table_2_cells()) == 24
    assert len(ref.ISO5136_TABLE_3_SIGMA_R) == 3


def test_expanded_uncertainty_is_twice_sigma_r() -> None:
    """Clause 9.2: U95 = 2 sigma_R, per band, on the result."""
    res = emission.sound_power_in_duct(_flat(80.0), _BANDS, 0.5, 15.0)
    sigma = emission.in_duct_reproducibility(_BANDS)
    np.testing.assert_allclose(res.reproducibility_standard_deviation, sigma)
    np.testing.assert_allclose(
        res.expanded_uncertainty, ref.ISO5136_COVERAGE_FACTOR * sigma
    )
    assert float(res.expanded_uncertainty[0]) == 7.0  # 50 Hz: 2 x 3,5


def test_table_2_is_reported_for_every_shield_and_says_so() -> None:
    """Clause 4 NOTE 5 expects larger figures for other shields and gives none.

    So the number reported for a nose cone or a foam ball is the sampling
    tube's, which makes it a lower bound rather than the shield's own
    reproducibility. The result cannot carry a better figure, because the
    standard publishes none, but it can say which of the two it is holding.
    """
    tube = emission.sound_power_in_duct(_flat(80.0), _BANDS, 0.5, 10.0)
    with pytest.warns(emission.SoundPowerWarning, match="sampling tube only"):
        cone = emission.sound_power_in_duct(
            _flat(80.0), _BANDS, 0.5, 10.0, shield="nose-cone"
        )
    np.testing.assert_allclose(
        cone.reproducibility_standard_deviation, tube.reproducibility_standard_deviation
    )


def test_the_sampling_tube_reports_its_reproducibility_without_a_warning() -> None:
    """The shield Table 2 does cover is the one that stays quiet."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        emission.sound_power_in_duct(_flat(80.0), _BANDS, 0.5, 10.0)


# ---------------------------------------------------------------------------
# The information-only flag
# ---------------------------------------------------------------------------
def test_information_only_bands_above_10_khz() -> None:
    """Bands above 10 kHz are flagged, the 24 normative ones are not."""
    res = emission.sound_power_in_duct(_flat(80.0), _BANDS, 0.5, 15.0)
    expected = _BANDS > 10000.0
    np.testing.assert_array_equal(res.information_only_band, expected)
    assert int(res.information_only_band.sum()) == 3


def test_information_only_when_sampling_tube_exceeds_40_m_s() -> None:
    """Between 40 and 60 m/s the Annex A coefficients are for information only."""
    within = emission.sound_power_in_duct(np.full((3, 5), 80.0), _BANDS[:5], 0.5, 40.0)
    beyond = emission.sound_power_in_duct(np.full((3, 5), 80.0), _BANDS[:5], 0.5, 40.5)
    assert not within.information_only_band.any()
    assert beyond.information_only_band.all()


# ---------------------------------------------------------------------------
# Refusals and warnings at the door
# ---------------------------------------------------------------------------
def test_refuses_a_band_that_is_not_nominal() -> None:
    with pytest.raises(ValueError, match="'frequencies' must be nominal"):
        emission.flow_modal_correction([1100.0], 10.0, 0.5)
    with pytest.raises(ValueError, match="'frequencies' must be nominal"):
        emission.in_duct_reproducibility([25000.0])
    with pytest.raises(ValueError, match="'frequencies' must be nominal"):
        emission.sound_power_in_duct([80.0], [40.0], 0.5, 10.0)


def test_refuses_a_non_positive_band() -> None:
    with pytest.raises(ValueError, match="'frequencies' must be strictly positive"):
        emission.flow_modal_correction([0.0], 10.0, 0.5)


@pytest.mark.parametrize("diameter", [0.1, 0.149, 2.01, 7.1])
def test_refuses_a_duct_outside_the_scope(diameter: float) -> None:
    """Clause 1.1: 0,15 m to 2 m; Annexes H and I are not implemented."""
    with pytest.raises(ValueError, match="'duct_diameter' must be between"):
        emission.flow_modal_correction([1000.0], 10.0, diameter)
    with pytest.raises(ValueError, match="'duct_diameter' must be between"):
        emission.sound_power_in_duct([80.0], [1000.0], diameter, 10.0)


def test_refuses_a_non_positive_duct_diameter() -> None:
    with pytest.raises(ValueError, match="'duct_diameter' must be positive"):
        emission.flow_modal_correction([1000.0], 10.0, 0.0)


@pytest.mark.parametrize(
    ("shield", "velocity"),
    [
        ("sampling-tube", 60.5),
        ("sampling-tube", -61.0),
        ("nose-cone", 20.5),
        ("nose-cone", -21.0),
        ("foam-ball", 15.5),
        ("foam-ball", -16.0),
    ],
)
def test_refuses_a_velocity_beyond_the_shield_limit(
    shield: str, velocity: float
) -> None:
    """Clause 1.1 per shield; 60 m/s for the sampling tube, where Annex A ends."""
    with pytest.raises(ValueError, match="'flow_velocity' must satisfy"):
        emission.flow_modal_correction(
            [1000.0],
            velocity,
            0.5,
            shield=shield,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="'flow_velocity' must satisfy"):
        emission.sound_power_in_duct(
            [80.0],
            [1000.0],
            0.5,
            velocity,
            shield=shield,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("shield", "velocity"),
    [("sampling-tube", 60.0), ("nose-cone", -20.0), ("foam-ball", 15.0)],
)
def test_accepts_a_velocity_on_the_shield_limit(shield: str, velocity: float) -> None:
    values = emission.flow_modal_correction(
        [1000.0],
        velocity,
        0.5,
        shield=shield,  # type: ignore[arg-type]
    )
    assert np.isfinite(values).all()


def test_refuses_a_non_finite_velocity() -> None:
    with pytest.raises(ValueError, match="'flow_velocity' must be finite"):
        emission.flow_modal_correction([1000.0], float("nan"), 0.5)


@pytest.mark.parametrize("band", [12500.0, 16000.0, 20000.0])
def test_refuses_above_10_khz_beyond_40_m_s(band: float) -> None:
    """The two informative extensions of the Annex A footnote do not compose.

    The rows for 12,5 kHz to 20 kHz sit under their own "|U| <= 40 m/s" band
    header (Tables A.1 to A.6), and the footnote extends only the frequency
    range, not the velocity. Past 40 m/s those polynomials diverge, so the
    combination is refused rather than answered.
    """
    with pytest.raises(ValueError, match="'flow_velocity' must satisfy"):
        emission.flow_modal_correction([band], 40.5, 0.5)
    with pytest.raises(ValueError, match="'flow_velocity' must satisfy"):
        emission.sound_power_in_duct([80.0], [band], 0.5, -60.0)


def test_the_informative_velocity_range_survives_below_10_khz() -> None:
    """40 m/s to 60 m/s stays available for the bands the footnote extends."""
    values = emission.flow_modal_correction([1000.0, 10000.0], 45.0, 0.5)
    assert np.isfinite(values).all()
    on_the_edge = emission.flow_modal_correction([20000.0], 40.0, 0.5)
    assert np.isfinite(on_the_edge).all()


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("duct_diameter", {"duct_diameter": np.array([0.5, 0.6])}),
        ("flow_velocity", {"flow_velocity": np.array([10.0, 20.0])}),
        ("temperature", {"temperature": np.array([20.0, 21.0])}),
        ("static_pressure", {"static_pressure": np.array([101.0, 102.0])}),
    ],
)
def test_refuses_a_non_scalar_where_one_measurement_is_meant(
    name: str, kwargs: dict[str, object]
) -> None:
    """A per-band array where a single number belongs names its parameter."""
    call: dict[str, object] = {
        "duct_diameter": 0.5,
        "flow_velocity": 10.0,
        **kwargs,
    }
    with pytest.raises(ValueError, match=f"'{name}' must be a real number"):
        emission.sound_power_in_duct([80.0], [1000.0], **call)  # type: ignore[arg-type]


def test_refuses_an_unknown_shield() -> None:
    with pytest.raises(ValueError, match="'shield' must be one of"):
        emission.flow_modal_correction([1000.0], 10.0, 0.5, shield="windscreen")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="'shield' must be one of"):
        emission.sound_power_in_duct([80.0], [1000.0], 0.5, 10.0, shield="windscreen")  # type: ignore[arg-type]


def test_refuses_a_non_positive_speed_of_sound() -> None:
    with pytest.raises(ValueError, match="'speed_of_sound' must be positive"):
        emission.flow_modal_correction(
            [1000.0], 10.0, 0.5, shield="nose-cone", speed_of_sound=0.0
        )


def test_refuses_a_sonic_flow_for_the_omnidirectional_shields() -> None:
    with pytest.raises(ValueError, match="'flow_velocity' must be smaller"):
        emission.flow_modal_correction(
            [1000.0], 15.0, 0.5, shield="foam-ball", speed_of_sound=15.0
        )


@pytest.mark.parametrize("temperature", [-50.5, 70.5, float("nan")])
def test_refuses_a_temperature_outside_the_scope(temperature: float) -> None:
    """Clause 1.1: -50 degC to +70 degC."""
    with pytest.raises(ValueError, match="'temperature' must be between"):
        emission.sound_power_in_duct(
            [80.0], [1000.0], 0.5, 10.0, temperature=temperature
        )


def test_accepts_the_temperature_edges() -> None:
    for temperature in (-50.0, 70.0):
        res = emission.sound_power_in_duct(
            [80.0], [1000.0], 0.5, 10.0, temperature=temperature
        )
        assert np.isfinite(res.sound_power_level).all()


def test_refuses_a_non_positive_static_pressure() -> None:
    with pytest.raises(ValueError, match="'static_pressure' must be positive"):
        emission.sound_power_in_duct([80.0], [1000.0], 0.5, 10.0, static_pressure=0.0)


def test_refuses_levels_of_the_wrong_band_count() -> None:
    with pytest.raises(ValueError, match="'levels' must carry one value per band"):
        emission.sound_power_in_duct(np.full((3, 4), 80.0), _BANDS[:5], 0.5, 10.0)
    with pytest.raises(ValueError, match="'levels' must carry one value per band"):
        emission.sound_power_in_duct(np.full(4, 80.0), _BANDS[:5], 0.5, 10.0)


def test_refuses_levels_of_the_wrong_rank_or_empty() -> None:
    with pytest.raises(ValueError, match="'levels' must be a"):
        emission.sound_power_in_duct(np.full((2, 3, 5), 80.0), _BANDS[:5], 0.5, 10.0)
    with pytest.raises(ValueError, match="'levels' must be a"):
        emission.sound_power_in_duct(80.0, [1000.0], 0.5, 10.0)
    with pytest.raises(ValueError, match="'levels' must be a"):
        emission.sound_power_in_duct(np.zeros((0, 5)), _BANDS[:5], 0.5, 10.0)


def test_refuses_non_finite_or_non_numeric_levels() -> None:
    levels = np.full((3, 5), 80.0)
    levels[1, 2] = np.nan
    with pytest.raises(ValueError, match="'levels' must contain only finite"):
        emission.sound_power_in_duct(levels, _BANDS[:5], 0.5, 10.0)
    with pytest.raises(ValueError, match="'levels' must be numeric"):
        emission.sound_power_in_duct(["a", "b"], _BANDS[:2], 0.5, 10.0)


def test_refuses_a_correction_of_the_wrong_length() -> None:
    with pytest.raises(
        ValueError, match="'microphone_correction' must be a scalar or carry"
    ):
        emission.sound_power_in_duct(
            _flat(80.0), _BANDS, 0.5, 10.0, microphone_correction=[0.1, 0.2]
        )
    with pytest.raises(ValueError, match=r"per band \(27 in 'frequencies'\)"):
        emission.sound_power_in_duct(
            _flat(80.0), _BANDS, 0.5, 10.0, microphone_correction=[0.1, 0.2]
        )
    with pytest.raises(
        ValueError, match="'shield_correction' must be a scalar or carry"
    ):
        emission.sound_power_in_duct(
            _flat(80.0), _BANDS, 0.5, 10.0, shield_correction=np.zeros(26)
        )


def test_refuses_a_non_finite_correction() -> None:
    with pytest.raises(
        ValueError, match="'microphone_correction' must contain only finite"
    ):
        emission.sound_power_in_duct(
            _flat(80.0), _BANDS, 0.5, 10.0, microphone_correction=float("inf")
        )
    with pytest.raises(
        ValueError, match="'shield_correction' must contain only finite"
    ):
        emission.sound_power_in_duct(
            _flat(80.0), _BANDS, 0.5, 10.0, shield_correction=float("nan")
        )


def test_warns_below_three_positions() -> None:
    """Clause 6.2.2 asks for at least three circumferential positions."""
    with pytest.warns(emission.SoundPowerWarning, match="at least 3"):
        emission.sound_power_in_duct(_flat(80.0, positions=2), _BANDS, 0.5, 10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        emission.sound_power_in_duct(_flat(80.0, positions=3), _BANDS, 0.5, 10.0)
        emission.sound_power_in_duct(np.full(_BANDS.size, 80.0), _BANDS, 0.5, 10.0)


# ---------------------------------------------------------------------------
# The result object
# ---------------------------------------------------------------------------
def test_result_refuses_mismatched_bands() -> None:
    res = emission.sound_power_in_duct(_flat(80.0), _BANDS, 0.5, 10.0)
    with pytest.raises(ValueError, match="'sound_power_level'"):
        dataclasses.replace(res, sound_power_level=res.sound_power_level[:-1])


def test_result_refuses_an_unknown_shield() -> None:
    res = emission.sound_power_in_duct(_flat(80.0), _BANDS, 0.5, 10.0)
    with pytest.raises(ValueError, match="'shield' must be one of"):
        dataclasses.replace(res, shield="windscreen")


def test_result_refuses_a_non_finite_scalar() -> None:
    res = emission.sound_power_in_duct(_flat(80.0), _BANDS, 0.5, 10.0)
    with pytest.raises(ValueError, match="'duct_area' must be finite"):
        dataclasses.replace(res, duct_area=float("nan"))


@pytest.mark.parametrize(
    "field",
    [
        "microphone_correction",
        "shield_correction",
        "flow_modal_correction",
        "combined_correction",
    ],
)
def test_result_refuses_a_non_finite_correction_column(field: str) -> None:
    """The record of clause 9.1 f) prints all four columns as plain numbers."""
    res = emission.sound_power_in_duct(_flat(80.0), _BANDS, 0.5, 10.0)
    blank = np.full(_BANDS.size, np.nan)
    with pytest.raises(ValueError, match=f"'{field}' must contain only finite"):
        dataclasses.replace(res, **{field: blank})


def test_result_records_the_inputs() -> None:
    res = emission.sound_power_in_duct(
        _flat(80.0), _BANDS, 0.63, -12.0, shield="sampling-tube"
    )
    assert res.duct_diameter == 0.63
    assert res.flow_velocity == -12.0
    assert res.shield == "sampling-tube"
    np.testing.assert_array_equal(res.frequencies, _BANDS)


def test_plot_draws_one_bar_per_band_with_lwa_in_the_title() -> None:
    res = emission.sound_power_in_duct(_flat(80.0), _BANDS, 0.5, 15.0)
    ax = res.plot()
    heights = [p.get_height() for p in ax.patches]
    np.testing.assert_allclose(heights, res.sound_power_level)
    assert "ISO 5136" in ax.get_title()
    assert f"{res.sound_power_level_a:.1f}" in ax.get_title()
    plt.close("all")


def test_plot_in_spanish_and_refuses_an_unknown_language() -> None:
    res = emission.sound_power_in_duct(_flat(80.0), _BANDS, 0.5, 15.0)
    ax = res.plot(language="es")
    assert "Nivel de potencia acústica" in ax.get_ylabel()
    assert "espectro de potencia acústica" in ax.get_title()
    plt.close("all")
    with pytest.raises(ValueError, match="Unknown language"):
        res.plot(language="xx")
