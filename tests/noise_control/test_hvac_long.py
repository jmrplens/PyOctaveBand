#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the duct-path element models of Long, Chapters 13 and 14.

Oracle: Long, *Architectural Acoustics* 2nd ed. (Academic Press, 2014):

* Ch. 13, Eq. 13.1 with Table 13.5 (level correction ``K_F``), Table 13.6
  (efficiency correction ``C_EFF``), Table 13.7 (blade frequency increment
  ``C_BFI``) and Table 13.8 (fan housing attenuation), printed pages 504-506;
* Ch. 14, Table 14.1 (unlined circular ducts), Eqs. 14.9-14.11 (unlined
  rectangular ducts), Eq. 14.12 with Table 14.2 (lined rectangular ducts),
  Eq. 14.13 with Table 14.3 (lined circular ducts), Table 14.4 (lined flexible
  duct), Tables 14.5-14.7 (elbows), Eqs. 14.14-14.16 (end reflection),
  Eq. 14.17 (split loss), Eq. 14.31 with Table 14.8 (silencer self-noise),
  printed pages 534-547;

plus ASHRAE (2019) *HVAC Applications Handbook*, Chapter 49, Table 9 (maximum
recommended neck velocities) and Table 10 (damper throttling correction), and
Bies, Hansen & Howard, *Engineering Noise Control* 5th ed., §8.10.5 Eq. (8.241)
(combination of the airways of a splitter silencer).

Every expected value is either a printed table entry or the printed equation
evaluated here with plain arithmetic, never a value read back from the module.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry.noise_control import hvac

_IN = 0.0254
_FT = 0.3048
_CFM = 0.0004719474432
_IN_WG = 249.0


# ---------------------------------------------------------------------------
# Fan sound power (Ch. 13)
# ---------------------------------------------------------------------------
def test_fan_sound_power_at_the_reference_operating_point() -> None:
    """Eq. 13.1 at Q = Q_REF and P = P_REF: only K_F and C_BFI remain.

    Long Table 13.5, forward-curved, all wheel diameters:
    53/53/43/36/36/31/26/21 dB, plus the Table 13.7 increment of 2 dB in the
    500 Hz octave, and no efficiency correction at peak efficiency.
    """
    res = hvac.fan_sound_power(
        0.472e-3, _IN_WG, fan_type="forward_curved", relative_efficiency=100.0
    )
    assert np.allclose(res.values, [53, 53, 43, 38, 36, 31, 26, 21], atol=1e-9)
    assert res.quantity == "sound_power_level"


def test_fan_sound_power_duty_terms() -> None:
    """Long's worked operating point, 5000 cfm at 2 in w.g., 80 per cent.

    ``10 lg 5000 + 10 lg 2 = 10 lg 10000 = 40 dB`` exactly, ``C_EFF = 6 dB``
    (Table 13.6, 75-85 per cent of peak) and ``C_BFI = 2 dB`` at 500 Hz, so
    Eq. 13.1 gives 99/99/89/84/82/77/72/67 dB.

    Long's Table 14.9 prints 90/86/82/79/77/75/71/61 dB for the same duty and
    fan type; that row does not come from this equation (see the warning in
    :mod:`phonometry.noise_control.hvac`), and this test pins what the printed
    equation actually gives.
    """
    res = hvac.fan_sound_power(5000.0 * _CFM, 2.0 * _IN_WG)
    assert np.allclose(res.values, [99, 99, 89, 84, 82, 77, 72, 67], atol=1e-9)


def test_fan_sound_power_places_the_blade_increment_by_frequency() -> None:
    res = hvac.fan_sound_power(
        0.472e-3, _IN_WG, relative_efficiency=100.0, blade_frequency=1000.0
    )
    # The 2 dB increment moves from the 500 Hz to the 1 kHz octave.
    assert np.allclose(res.values, [53, 53, 43, 36, 38, 31, 26, 21], atol=1e-9)


@pytest.mark.parametrize(
    ("efficiency", "correction"),
    [(95.0, 0.0), (87.0, 3.0), (80.0, 6.0), (70.0, 9.0), (60.0, 12.0),
     (52.0, 15.0), (40.0, 16.0)],
)
def test_efficiency_correction_table_13_6(
    efficiency: float, correction: float
) -> None:
    assert hvac.fan_efficiency_correction(efficiency) == correction


def test_blade_passing_frequency_eq_13_4() -> None:
    assert hvac.blade_passing_frequency(1200.0, 12) == pytest.approx(240.0)


def test_fan_casing_attenuation_table_13_8() -> None:
    res = hvac.fan_casing_attenuation()
    assert np.allclose(res.values, [0, 0, 5, 10, 15, 20, 22, 25])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fan_type": "turbine"}, "fan_type"),
        ({"relative_efficiency": 0.0}, "relative_efficiency"),
        ({"relative_efficiency": 101.0}, "100"),
    ],
)
def test_fan_sound_power_validation(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        hvac.fan_sound_power(1.0, 500.0, **kwargs)  # type: ignore[arg-type]


def test_blade_passing_frequency_validation() -> None:
    with pytest.raises(ValueError, match="blades"):
        hvac.blade_passing_frequency(1200.0, 0)


# ---------------------------------------------------------------------------
# Straight ducts (Ch. 14)
# ---------------------------------------------------------------------------
def test_unlined_circular_duct_table_14_1() -> None:
    """Table 14.1: 0.03 dB/ft to 250 Hz, 0.05 at 500 Hz, 0.07 above."""
    res = hvac.unlined_circular_duct_attenuation(None, 10.0 * _FT)
    assert np.allclose(res.values, [0.3, 0.3, 0.3, 0.5, 0.7, 0.7, 0.7, 0.7])


def test_unlined_rectangular_duct_eqs_14_9_and_14_11() -> None:
    """Eq. 14.9 (P/S >= 3) below 250 Hz and Eq. 14.11 above it.

    An 18 x 12 in duct 6 ft long has ``P / S = 2 (1.5 + 1) / 1.5 = 10/3 ft^-1``.
    """
    bands = hvac.OCTAVE_BANDS
    res = hvac.unlined_rectangular_duct_attenuation(
        bands, 18.0 * _IN, 12.0 * _IN, 6.0 * _FT
    )
    ps, length = 10.0 / 3.0, 6.0
    expected = [
        17.0 * ps**0.25 * f**-0.85 * length if f <= 250.0
        else 0.02 * ps**0.8 * length
        for f in bands
    ]
    assert np.allclose(res.values, expected)


def test_unlined_rectangular_duct_eq_14_10_below_three() -> None:
    """Eq. 14.10 applies below ``P / S = 3``; a 36 x 24 in duct is 5/3 ft^-1."""
    res = hvac.unlined_rectangular_duct_attenuation(
        [63.0], 36.0 * _IN, 24.0 * _IN, 5.0 * _FT
    )
    ps = 5.0 / 3.0
    assert res.values[0] == pytest.approx(1.64 * ps**0.73 * 63.0**-0.58 * 5.0)


def test_unlined_rectangular_duct_wrapping_doubles_the_low_bands() -> None:
    args = ([63.0, 500.0], 36.0 * _IN, 24.0 * _IN, 5.0 * _FT)
    bare = hvac.unlined_rectangular_duct_attenuation(*args)
    wrapped = hvac.unlined_rectangular_duct_attenuation(*args, wrapped=True)
    assert wrapped.values[0] == pytest.approx(2.0 * bare.values[0])
    assert wrapped.values[1] == pytest.approx(bare.values[1])


def test_lined_rectangular_duct_eq_14_12() -> None:
    """Eq. 14.12 ``R = B (P/S)^C t^D l`` with the Table 14.2 constants."""
    res = hvac.lined_rectangular_duct_attenuation(
        None, 18.0 * _IN, 12.0 * _IN, 6.0 * _FT, 1.0 * _IN
    )
    b = [0.0133, 0.0574, 0.2710, 1.0147, 1.7700, 1.3920, 1.5180, 1.5810]
    c = [1.959, 1.410, 0.824, 0.500, 0.695, 0.802, 0.451, 0.219]
    ps, length = 10.0 / 3.0, 6.0
    # t = 1 in, so t^D = 1 for every band and D drops out of this case.
    expected = [bi * ps**ci * length for bi, ci in zip(b, c)]
    assert np.allclose(res.values, expected)


def test_lined_rectangular_duct_thickness_exponent() -> None:
    """The Table 14.2 ``D`` column, checked at a 2 in lining and 63 Hz."""
    thin = hvac.lined_rectangular_duct_attenuation(
        [63.0], 18.0 * _IN, 12.0 * _IN, 6.0 * _FT, 1.0 * _IN
    )
    thick = hvac.lined_rectangular_duct_attenuation(
        [63.0], 18.0 * _IN, 12.0 * _IN, 6.0 * _FT, 2.0 * _IN
    )
    assert thick.values[0] / thin.values[0] == pytest.approx(2.0**0.917)


def test_lined_rectangular_duct_adds_the_unlined_contribution() -> None:
    args = (None, 18.0 * _IN, 12.0 * _IN, 6.0 * _FT, 1.0 * _IN)
    lined = hvac.lined_rectangular_duct_attenuation(*args)
    total = hvac.lined_rectangular_duct_attenuation(*args, include_unlined=True)
    unlined = hvac.unlined_rectangular_duct_attenuation(
        hvac.OCTAVE_BANDS, 18.0 * _IN, 12.0 * _IN, 6.0 * _FT
    )
    assert np.allclose(total.values, lined.values + unlined.values)


def test_lined_duct_attenuation_is_capped_by_flanking() -> None:
    """Long: flanking limits a lined duct run to 40 dB."""
    res = hvac.lined_rectangular_duct_attenuation(
        None, 12.0 * _IN, 12.0 * _IN, 60.0 * _FT, 2.0 * _IN
    )
    assert np.all(res.values <= 40.0)
    assert np.any(res.values == 40.0)


def test_lined_circular_duct_eq_14_13() -> None:
    """Eq. 14.13 with the Table 14.3 constants at 250 Hz, d = 12 in, t = 2 in."""
    res = hvac.lined_circular_duct_attenuation(
        [250.0], 12.0 * _IN, 10.0 * _FT, 2.0 * _IN
    )
    a, b, c = 0.3652, 0.7900, -0.1157
    d, e, g = -1.834e-2, -1.211e-4, 2.681e-6
    t, dia, length = 2.0, 12.0, 10.0
    expected = (a + b * t + c * t**2 + d * dia + e * dia**2 + g * dia**3) * length
    assert res.values[0] == pytest.approx(expected)


def test_lined_circular_duct_clips_negative_regression_values() -> None:
    # The 63 Hz regression goes negative for a large duct with a thin lining.
    res = hvac.lined_circular_duct_attenuation(
        [63.0], 60.0 * _IN, 10.0 * _FT, 1.0 * _IN
    )
    assert res.values[0] >= 0.0


@pytest.mark.parametrize(
    ("diameter_in", "length_ft", "expected"),
    [
        (4.0, 3.0, [2, 3, 3, 8, 9, 11, 7]),
        (8.0, 9.0, [6, 8, 16, 25, 28, 28, 18]),
        (12.0, 6.0, [3, 5, 10, 15, 17, 16, 9]),
        (16.0, 12.0, [2, 4, 9, 23, 28, 23, 9]),
    ],
)
def test_flexible_duct_table_14_4_nodes(
    diameter_in: float, length_ft: float, expected: list[int]
) -> None:
    res = hvac.flexible_duct_insertion_loss(
        None, diameter_in * _IN, length_ft * _FT
    )
    assert np.allclose(res.values, expected, atol=1e-9)
    assert np.allclose(res.frequencies, [63, 125, 250, 500, 1000, 2000, 4000])


def test_flexible_duct_interpolates_between_nodes() -> None:
    res = hvac.flexible_duct_insertion_loss(None, 12.0 * _IN, 7.5 * _FT)
    six = hvac.flexible_duct_insertion_loss(None, 12.0 * _IN, 6.0 * _FT)
    nine = hvac.flexible_duct_insertion_loss(None, 12.0 * _IN, 9.0 * _FT)
    assert np.allclose(res.values, 0.5 * (six.values + nine.values))


def test_flexible_duct_stops_at_4_khz() -> None:
    with pytest.raises(ValueError, match="octave-band centres"):
        hvac.flexible_duct_insertion_loss([8000.0], 12.0 * _IN, 6.0 * _FT)


# ---------------------------------------------------------------------------
# Elbows: Long Tables 14.5-14.7 read through the frequency-width product
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        # Table 14.5, unlined and lined square elbows without turning vanes.
        ({"bend_type": "square"}, [0, 1, 5, 8, 4, 3]),
        ({"bend_type": "square", "lined": True}, [0, 1, 6, 11, 10, 10]),
        # Table 14.6, with turning vanes.
        ({"bend_type": "square", "vanes": True}, [0, 1, 4, 6, 4, 4]),
        ({"bend_type": "square", "vanes": True, "lined": True},
         [0, 1, 4, 7, 7, 7]),
        # Table 14.7, round elbows.
        ({"bend_type": "round"}, [0, 1, 2, 3, 3, 3]),
    ],
)
def test_elbow_tables_by_frequency_width_product(
    kind: dict[str, object], expected: list[int]
) -> None:
    """Long's ``f w`` index (kHz x in) equals ``0.074 W / lambda``.

    The elbow rows are checked at the midpoint of every ``f w`` band of
    Tables 14.5-14.7 for a 24 in elbow: ``f w`` = 1, 3, 5, 10, 20 and 40.
    """
    width_in = 24.0
    fw = np.array([1.0, 3.0, 5.0, 10.0, 20.0, 40.0])
    frequencies = fw / width_in * 1000.0
    res = hvac.elbow_insertion_loss(frequencies, width_in * _IN, **kind)  # type: ignore[arg-type]
    assert np.allclose(res.values, expected)


# ---------------------------------------------------------------------------
# End reflection, splits and equivalent diameter
# ---------------------------------------------------------------------------
def test_end_reflection_closed_form_eqs_14_14_and_14_15() -> None:
    c, diameter = 343.0, 0.3
    free = hvac.end_reflection_loss_closed_form(
        [125.0], diameter, termination="free", speed_of_sound=c
    )
    flush = hvac.end_reflection_loss_closed_form(
        [125.0], diameter, termination="flush", speed_of_sound=c
    )
    ratio = c / (math.pi * 125.0 * diameter)
    assert free.values[0] == pytest.approx(
        10.0 * math.log10(1.0 + ratio**1.88)
    )
    assert flush.values[0] == pytest.approx(
        10.0 * math.log10(1.0 + (0.8 * ratio) ** 1.88)
    )


def test_end_reflection_method_long_selects_the_closed_form() -> None:
    through = hvac.end_reflection_loss([125.0], 0.3, method="long")
    direct = hvac.end_reflection_loss_closed_form([125.0], 0.3)
    assert np.allclose(through.values, direct.values)


def test_end_reflection_methods_agree_within_a_few_decibels() -> None:
    bands = [63.0, 125.0, 250.0, 500.0, 1000.0]
    bies = hvac.end_reflection_loss(bands, 0.3, method="bies")
    long = hvac.end_reflection_loss(bands, 0.3, method="long")
    assert np.max(np.abs(bies.values - long.values)) < 3.0


def test_end_reflection_unknown_method() -> None:
    with pytest.raises(ValueError, match="method"):
        hvac.end_reflection_loss([125.0], 0.3, method="ashrae")


def test_equivalent_diameter_eq_14_16() -> None:
    area = 36.0 * 24.0 * _IN**2
    assert hvac.equivalent_diameter(area) == pytest.approx(
        math.sqrt(4.0 * area / math.pi)
    )


def test_split_loss_eq_14_17_area_matched() -> None:
    """With ``sum S_i = S_m`` the reflection term vanishes: -10 lg(S_i/sum S_i)."""
    main = 0.6
    loss = hvac.split_loss(main, [0.15, 0.15, 0.15, 0.15], branch=2)
    assert loss == pytest.approx(-10.0 * math.log10(0.25))


def test_split_loss_eq_14_17_with_area_mismatch() -> None:
    main, branches = 0.6, [0.2, 0.5]
    loss = hvac.split_loss(main, branches, branch=0)
    total = sum(branches)
    expected = -10.0 * math.log10(
        1.0 - ((total - main) / (total + main)) ** 2
    ) - 10.0 * math.log10(0.2 / total)
    assert loss == pytest.approx(expected)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((0.0, [0.1]), "main_area"),
        ((0.6, []), "branch_areas"),
        ((0.6, [0.1, -0.2]), "positive"),
    ],
)
def test_split_loss_validation(args: tuple[object, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        hvac.split_loss(*args)  # type: ignore[arg-type]


def test_split_loss_branch_index_out_of_range() -> None:
    with pytest.raises(ValueError, match="branch"):
        hvac.split_loss(0.6, [0.3, 0.3], branch=2)


# ---------------------------------------------------------------------------
# Silencers and terminal devices
# ---------------------------------------------------------------------------
def test_silencer_self_noise_eq_14_31_and_table_14_8() -> None:
    velocity, passages, height_mm = 10.0, 4, 900.0
    res = hvac.silencer_self_noise(None, velocity, passages, height_mm / 1000.0)
    overall = (
        55.0 * math.log10(velocity)
        + 10.0 * math.log10(passages)
        + 10.0 * math.log10(height_mm)
        - 45.0
    )
    corrections = [4, 4, 6, 8, 13, 18, 23, 28]
    assert np.allclose(res.values, [overall - k for k in corrections])
    assert res.quantity == "sound_power_level"


def test_silencer_self_noise_velocity_exponent() -> None:
    """55 lg V: doubling the airway velocity adds 55 lg 2 = 16.6 dB."""
    slow = hvac.silencer_self_noise([63.0], 5.0, 4, 0.9)
    fast = hvac.silencer_self_noise([63.0], 10.0, 4, 0.9)
    assert fast.values[0] - slow.values[0] == pytest.approx(
        55.0 * math.log10(2.0)
    )


def test_silencer_self_noise_requires_passages() -> None:
    with pytest.raises(ValueError, match="passages"):
        hvac.silencer_self_noise(None, 10.0, 0, 0.9)


def test_splitter_silencer_identical_airways_equal_one_passage() -> None:
    """Bies §8.10.5: identical airways give the loss of a single passage."""
    one = hvac.splitter_silencer_insertion_loss(None, 0.6, 1.5, 0.1, 0.2)
    four = hvac.splitter_silencer_insertion_loss(
        None, 0.6, 1.5, [0.1, 0.1, 0.1, 0.1], 0.2
    )
    assert np.allclose(one.values, four.values)


def test_splitter_silencer_airway_is_a_lined_duct_of_half_the_splitter() -> None:
    one = hvac.splitter_silencer_insertion_loss(None, 0.6, 1.5, 0.1, 0.2)
    duct = hvac.lined_rectangular_duct_attenuation(None, 0.1, 0.6, 1.5, 0.1)
    assert np.allclose(one.values, duct.values)


def test_splitter_silencer_combines_by_eq_8_241() -> None:
    widths = [0.1, 0.3]
    res = hvac.splitter_silencer_insertion_loss(None, 0.6, 1.5, widths, 0.2)
    per_airway = np.array(
        [
            hvac.lined_rectangular_duct_attenuation(None, w, 0.6, 1.5, 0.1).values
            for w in widths
        ]
    )
    expected = -10.0 * np.log10(np.mean(10.0 ** (-per_airway / 10.0), axis=0))
    assert np.allclose(res.values, expected)
    # The leakiest airway dominates: the total lies between the worst airway
    # and that airway plus 10 lg N, never near the best one.
    worst = per_airway.min(axis=0)
    assert np.all(res.values >= worst - 1e-9)
    assert np.all(res.values <= worst + 10.0 * math.log10(len(widths)) + 1e-9)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((None, 0.6, 1.5, [], 0.2), "airway_widths"),
        ((None, 0.6, 1.5, [0.1, 0.0], 0.2), "positive"),
        ((None, 0.6, 1.5, 0.1, 0.0), "splitter_thickness"),
    ],
)
def test_splitter_silencer_validation(
    args: tuple[object, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        hvac.splitter_silencer_insertion_loss(*args)  # type: ignore[arg-type]


def test_diffuser_sound_power_reproduces_the_table_14_9_row() -> None:
    """Eqs. 13.27-13.33 against the supply diffuser of Long Table 14.9.

    The sheet's row is "Rectangular Diffuser, 312 cfm, 0.05 in pd" with the
    self-noise spectrum 33/32/29/23/15/4/0/0 dB. Read as a 24 x 24 in face,
    the printed equations give it back: the approach velocity is
    ``312 / (60 x 4 ft2) = 1.3 ft/s``, so the peak frequency
    ``f_P = 48.8 U_G`` falls in the 63 Hz octave and the spectrum decays from
    there by the Eq. 13.31 shape. The sheet floors its own rows at 0 dB, so
    only the six bands that carry a level are compared. Agreement is better
    than 1 dB in five of those six (+0.4/+0.4/+0.2/+0.7/+0.9 dB from 63 Hz to
    1 kHz); the 2 kHz band is +1.9 dB, which is why the two assertions below
    carry different tolerances.
    """
    res = hvac.diffuser_sound_power(
        None, (24.0 * _IN) ** 2, 312.0 * _CFM, 0.05 * _IN_WG
    )
    printed = np.array([33.0, 32.0, 29.0, 23.0, 15.0, 4.0])
    assert np.max(np.abs(res.values[:6] - printed)) < 2.0
    assert np.max(np.abs(res.values[:5] - printed[:5])) < 1.0
    assert res.quantity == "sound_power_level"


def test_diffuser_sound_power_follows_the_printed_equations() -> None:
    """Eqs. 13.27 to 13.33 evaluated here with plain arithmetic."""
    area_ft2, flow_cfm, drop_in_wg = 4.0, 312.0, 0.05
    velocity = flow_cfm / (60.0 * area_ft2)
    xi = 334.9 * drop_in_wg / (0.075 * velocity**2)
    overall = (
        10.0 * math.log10(area_ft2)
        + 30.0 * math.log10(xi)
        + 60.0 * math.log10(velocity)
        - 31.3
    )
    # Eq. 13.32: f_P = 48.8 U_G = 63.4 Hz, i.e. band number 1 (63 Hz).
    assert 48.8 * velocity == pytest.approx(63.44, abs=0.01)
    res = hvac.diffuser_sound_power(
        None, area_ft2 * _FT**2, flow_cfm * _CFM, drop_in_wg * _IN_WG
    )
    for band, value in enumerate(res.values):
        a = 1.0 - (band + 1)  # N_B(f_P) - N_B(f), 63 Hz being band 1
        assert value == pytest.approx(overall - 11.82 - 0.15 * a - 1.13 * a**2)


def test_diffuser_sound_power_round_shape_is_six_decibels_louder() -> None:
    """Eq. 13.30 against Eq. 13.31: -5.82 instead of -11.82."""
    args = (None, (24.0 * _IN) ** 2, 312.0 * _CFM, 0.05 * _IN_WG)
    rectangular = hvac.diffuser_sound_power(*args)
    round_ = hvac.diffuser_sound_power(*args, shape="round")
    assert np.allclose(round_.values - rectangular.values, 6.0)


def test_diffuser_sound_power_velocity_terms_cancel_at_fixed_pressure_drop() -> None:
    """Long's own note under Eq. 13.28: 60 lg U and the U in xi cancel.

    The velocity dependence of Eq. 13.27 is carried entirely by the pressure
    drop, so at a fixed drop the overall level does not move with the flow.
    Both cases below peak in the 250 Hz octave, so the same band compares the
    same point of the spectrum.
    """
    area, drop = 4.0 * _FT**2, 0.05 * _IN_WG
    slow = hvac.diffuser_sound_power([250.0], area, 1100.0 * _CFM, drop)
    fast = hvac.diffuser_sound_power([250.0], area, 1400.0 * _CFM, drop)
    assert slow.values[0] == pytest.approx(fast.values[0])


def test_diffuser_sound_power_velocity_and_area_exponents() -> None:
    """Long: 18 dB per doubling of velocity, -15 dB per doubling of face area.

    Both statements hold once the pressure drop is allowed to follow the
    velocity pressure, ``dP ~ U^2``, which is what a real device does. Each
    case is read in its own peak octave, so only the overall level of
    Eq. 13.27 is being compared.
    """
    area, drop = 4.0 * _FT**2, 0.05 * _IN_WG
    # 1248 cfm through 4 ft2 is 5.2 ft/s, peaking in the 250 Hz octave.
    base = hvac.diffuser_sound_power([250.0], area, 1248.0 * _CFM, drop)
    # Twice the velocity through the same face: four times the pressure drop,
    # peak in the 500 Hz octave.
    faster = hvac.diffuser_sound_power(
        [500.0], area, 2496.0 * _CFM, 4.0 * drop
    )
    assert faster.values[0] - base.values[0] == pytest.approx(
        60.0 * math.log10(2.0), abs=1e-9
    )
    # Twice the face area at the same flow: half the velocity, a quarter of
    # the pressure drop, peak in the 125 Hz octave.
    wider = hvac.diffuser_sound_power(
        [125.0], 2.0 * area, 1248.0 * _CFM, 0.25 * drop
    )
    assert wider.values[0] - base.values[0] == pytest.approx(
        10.0 * math.log10(2.0) - 60.0 * math.log10(2.0), abs=1e-9
    )


def test_diffuser_sound_power_counts_identical_devices() -> None:
    args = (None, (24.0 * _IN) ** 2, 312.0 * _CFM, 0.05 * _IN_WG)
    one = hvac.diffuser_sound_power(*args)
    four = hvac.diffuser_sound_power(*args, count=4)
    assert np.allclose(four.values - one.values, 10.0 * math.log10(4.0))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"face_area": 0.0}, "face_area"),
        ({"volume_flow": 0.0}, "volume_flow"),
        ({"pressure_drop": -1.0}, "pressure_drop"),
        ({"shape": "slot"}, "shape"),
        ({"count": 0}, "count"),
    ],
)
def test_diffuser_sound_power_validation(
    kwargs: dict[str, object], match: str
) -> None:
    call: dict[str, object] = {
        "frequencies": None,
        "face_area": 0.37,
        "volume_flow": 0.15,
        "pressure_drop": 12.0,
    }
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        hvac.diffuser_sound_power(**call)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("criterion", "opening", "expected"),
    [(45, "supply", 3.2), (30, "supply", 2.2), (25, "supply", 1.8),
     (45, "return", 3.8), (30, "return", 2.5), (25, "return", 2.2)],
)
def test_air_terminal_velocity_limit_ashrae_table_9(
    criterion: int, opening: str, expected: float
) -> None:
    assert hvac.air_terminal_velocity_limit(criterion, opening=opening) == expected


def test_air_terminal_velocity_limit_validation() -> None:
    with pytest.raises(ValueError, match="design_criterion"):
        hvac.air_terminal_velocity_limit(20)
    with pytest.raises(ValueError, match="opening"):
        hvac.air_terminal_velocity_limit(30, opening="exhaust")


@pytest.mark.parametrize(
    ("ratio", "location", "expected"),
    [
        (1.5, "diffuser_neck", 5.0), (3.0, "diffuser_neck", 15.0),
        (6.0, "diffuser_neck", 24.0), (2.0, "plenum_inlet", 3.0),
        (6.0, "plenum_inlet", 9.0), (2.5, "supply_duct", 0.0),
        (4.0, "supply_duct", 3.0),
    ],
)
def test_air_terminal_damper_correction_ashrae_table_10(
    ratio: float, location: str, expected: float
) -> None:
    assert hvac.air_terminal_damper_correction(
        ratio, location=location
    ) == pytest.approx(expected)


def test_air_terminal_damper_correction_validation() -> None:
    with pytest.raises(ValueError, match="pressure_ratio"):
        hvac.air_terminal_damper_correction(0.0)
    with pytest.raises(ValueError, match="location"):
        hvac.air_terminal_damper_correction(3.0, location="fan")


# ---------------------------------------------------------------------------
# Room effect
# ---------------------------------------------------------------------------
def test_room_effect_is_the_steady_state_room_relation() -> None:
    """Long Eq. 14.40: the positive form of ``10 lg[Q/(4 pi r^2) + 4/R]``."""
    distance, constant, directivity = 3.0, 50.0, 2.0
    value = hvac.room_effect(distance, constant, directivity=directivity)
    expected = -10.0 * math.log10(
        directivity / (4.0 * math.pi * distance**2) + 4.0 / constant
    )
    assert value == pytest.approx(expected)


def test_room_effect_accepts_a_per_band_room_constant() -> None:
    constants = np.array([20.0, 40.0, 80.0])
    values = hvac.room_effect(3.0, constants)
    assert isinstance(values, np.ndarray)
    assert values.shape == (3,)
    # A more absorbent room drops the level further.
    assert np.all(np.diff(values) > 0.0)


def test_room_effect_validation() -> None:
    with pytest.raises(ValueError, match="distance"):
        hvac.room_effect(0.0, 50.0)
    with pytest.raises(ValueError, match="room_constant"):
        hvac.room_effect(3.0, 0.0)
