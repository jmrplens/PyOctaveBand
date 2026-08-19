#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for EN 15657:2018 structure-borne sound power (reception-plate method).

Anchored on the closed-form reception-plate relations: the energetic spatial
mean velocity level (Formula 12), the loss factor eta = 2.2/(f Ts) (Formula 13),
and the characteristic power level L_Ws = 10 lg(2 pi f eta m S) + L_v - 60 dB
(Formula 14). The physical oracle is the resonant-plate power balance
P = omega eta (m S) <v^2>, whose level equals L_Ws exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import building

V0, P0 = 1.0e-9, 1.0e-12


# ---------------------------------------------------------------------------
# Spatial mean velocity level (Formula 12)
# ---------------------------------------------------------------------------
def test_spatial_mean_is_energetic_average() -> None:
    levels = np.array([78.0, 80.0, 82.0, 79.0, 81.0, 80.0])
    expected = 10.0 * math.log10(np.mean(10.0 ** (0.1 * levels)))
    assert building.spatial_mean_velocity_level(levels) == pytest.approx(
        expected
    )


def test_spatial_mean_uniform_levels() -> None:
    assert building.spatial_mean_velocity_level(
        [75.0, 75.0, 75.0]
    ) == pytest.approx(75.0)


# ---------------------------------------------------------------------------
# Loss factor (Formula 13) == ISO 10848 total loss factor
# ---------------------------------------------------------------------------
def test_loss_factor_formula() -> None:
    f, ts = np.array([500.0, 1000.0]), 0.3
    assert np.allclose(building.plate_loss_factor(f, ts), 2.2 / (f * ts))


def test_loss_factor_matches_iso10848() -> None:
    f, ts = np.array([250.0, 500.0, 1000.0]), 0.4
    assert np.allclose(
        building.plate_loss_factor(f, ts), building.total_loss_factor(f, ts)
    )


# ---------------------------------------------------------------------------
# Power level (Formula 14) and the resonant-plate power balance
# ---------------------------------------------------------------------------
def test_power_level_hand_value() -> None:
    lw = float(
        building.structure_borne_power_level(80.0, 1000.0, 10.0, 1.0, 0.01)
    )
    assert lw == pytest.approx(47.982, abs=1e-3)


def test_power_level_equals_dissipated_power() -> None:
    """L_Ws equals the level of P = omega eta (m S) <v^2>."""
    lv, f, m, s, eta = 82.0, 800.0, 15.0, 1.5, 0.02
    lw = float(building.structure_borne_power_level(lv, f, m, s, eta))
    v2 = V0**2 * 10.0 ** (0.1 * lv)
    p = 2.0 * math.pi * f * eta * (m * s) * v2
    assert lw == pytest.approx(10.0 * math.log10(p / P0))


def test_power_level_offset_is_minus_60() -> None:
    # With v0 = 1e-9 and P0 = 1e-12 the reference term is exactly -60 dB.
    # L_Ws - L_v - 10 lg(2 pi f eta m S) = -60
    f, m, s, eta = 1000.0, 10.0, 1.0, 0.01
    lw = float(building.structure_borne_power_level(0.0, f, m, s, eta))
    log_term = 10.0 * math.log10(2.0 * math.pi * f * eta * m * s)
    assert lw - log_term == pytest.approx(-60.0)


def test_power_level_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="mass_per_area"):
        building.structure_borne_power_level(80.0, 1000.0, 0.0, 1.0, 0.01)
    with pytest.raises(ValueError, match="area"):
        building.structure_borne_power_level(80.0, 1000.0, 10.0, 0.0, 0.01)
    with pytest.raises(ValueError, match="frequency"):
        building.structure_borne_power_level(80.0, 0.0, 10.0, 1.0, 0.01)


# ---------------------------------------------------------------------------
# Reception-plate constructor / result
# ---------------------------------------------------------------------------
def test_reception_plate_from_reverberation_time() -> None:
    f = np.array([500.0, 1000.0, 2000.0])
    res = building.reception_plate_power(
        np.array([80.0, 82.0, 79.0]), f, mass_per_area=25.0, area=1.2,
        reverberation_time=0.4,
    )
    assert isinstance(res, building.StructureBornePowerResult)
    assert np.allclose(res.loss_factor, 2.2 / (f * 0.4))
    assert np.allclose(
        res.power_level,
        building.structure_borne_power_level(
            res.velocity_level, f, 25.0, 1.2, res.loss_factor
        ),
    )


def test_reception_plate_explicit_loss_factor_broadcasts() -> None:
    f = np.array([500.0, 1000.0, 2000.0])
    res = building.reception_plate_power(
        80.0, f, mass_per_area=25.0, area=1.2, loss_factor=0.02
    )
    assert res.loss_factor.shape == (3,)
    assert np.all(res.loss_factor == 0.02)
    assert np.all(res.velocity_level == 80.0)


def test_reception_plate_total_level() -> None:
    res = building.reception_plate_power(
        np.array([80.0, 82.0, 79.0]), np.array([500.0, 1000.0, 2000.0]),
        mass_per_area=25.0, area=1.2, loss_factor=0.01,
    )
    expected = 10.0 * math.log10(np.sum(10.0 ** (0.1 * res.power_level)))
    assert res.total_level == pytest.approx(expected)


def test_reception_plate_requires_eta_or_ts() -> None:
    with pytest.raises(ValueError, match="loss_factor"):
        building.reception_plate_power(80.0, 1000.0, 25.0, 1.2)


def test_reception_plate_velocity_shape_mismatch() -> None:
    # a non-broadcastable velocity_level vs frequency length raises
    with pytest.raises(ValueError):
        building.reception_plate_power(
            [80.0, 82.0, 79.0], [500.0, 1000.0, 2000.0, 4000.0],
            mass_per_area=25.0, area=1.2, loss_factor=0.01,
        )


def test_plot_returns_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    res = building.reception_plate_power(
        np.array([80.0, 82.0, 79.0]), np.array([500.0, 1000.0, 2000.0]),
        mass_per_area=25.0, area=1.2, loss_factor=0.01,
    )
    assert res.plot() is not None


# ---------------------------------------------------------------------------
# EN 15657 source-quantity conversion chain (Formulae 15/17/18/19)
# ---------------------------------------------------------------------------

def test_blocked_force_level_formula_15() -> None:
    """L_Fb,eq = L_Ws,low - 10 lg(Re{Y}/Y0), dB re 1e-6 N."""

    lfb = building.equivalent_blocked_force_level(61.7, 5.34e-6)
    assert float(lfb) == pytest.approx(61.7 - 10.0 * math.log10(5.34e-6))
    # A complex plate mobility uses its real part (Formula 16).
    lfb_c = building.equivalent_blocked_force_level(61.7, 5.34e-6 + 2e-6j)
    assert float(lfb_c) == pytest.approx(float(lfb))
    with pytest.raises(ValueError, match="positive real part"):
        building.equivalent_blocked_force_level(61.7, -1e-6)


def test_characteristic_reception_plate_power_formula_17() -> None:
    """L_Wsn = L_Fb,eq + 10 lg(Y_R,inf,low/Y0), Y_R,inf,low = 5e-6 m/(N.s)."""

    lfb = building.equivalent_blocked_force_level(61.7, 5.34e-6)
    lwsn = building.characteristic_reception_plate_power(lfb)
    assert float(lwsn) == pytest.approx(
        61.7 + 10.0 * math.log10(5.0e-6 / 5.34e-6)
    )
    with pytest.raises(ValueError, match="characteristic_mobility"):
        building.characteristic_reception_plate_power(
            100.0, characteristic_mobility=0.0
        )


def test_conversion_chain_reproduces_annex_i8_wall() -> None:
    """(15)+(17) then the Annex I mobility correction reproduce Table I.8."""
    import reference_data as ref


    lwsn = building.characteristic_reception_plate_power(
        building.equivalent_blocked_force_level(
            ref.EN12354_5_I8_WALL_LWS, ref.EN12354_5_I8_PLATE_MOBILITY
        )
    )
    installed = building.installed_power_from_reception_plate(
        lwsn, ref.EN12354_5_I8_Y_WALL
    )
    np.testing.assert_allclose(
        installed, ref.EN12354_5_I8_WALL_INSTALLED,
        atol=ref.EN12354_5_ANNEX_I_TOL,
    )


def test_free_velocity_level_formula_18_real_mobility() -> None:
    """For a real plate mobility, (18) reduces to L_Ws + 10 lg(Y) + 60.

    Independent anchor from the physics rather than the code: for a real Y
    the injected power is P = v_f**2 / Y, so v_f**2 = P*Y and
    L_vf = 10 lg(P*Y/(1e-9)**2) with P in watts.
    """

    p_watt = 1e-12 * 10.0 ** (70.0 / 10.0)
    y = 1.0e-2
    expected = 10.0 * math.log10(p_watt * y / 1e-9**2)
    lvf = building.equivalent_free_velocity_level(70.0, y)
    assert float(lvf) == pytest.approx(expected)
    assert float(lvf) == pytest.approx(70.0 + 10.0 * math.log10(y) + 60.0)
    with pytest.raises(ValueError, match="positive real part"):
        building.equivalent_free_velocity_level(70.0, -1.0e-2)


def test_formulas_15_18_19_are_mutually_consistent() -> None:
    """(15) + (18) must combine through (19) into |Y_S,eq| = v_f / F_b.

    Synthesize a source of known free velocity and blocked force measured on
    ideal low- and high-mobility plates, then recover the source mobility.
    """

    y_source = 2.5e-4          # true |Y_S|, m/(N s)
    v_free = 3.2e-6            # true free velocity, m/s
    f_blocked = v_free / y_source
    y_low = 5.0e-6             # low-mobility plate: F ~ F_b, P = F_b**2 Re{Y}
    lw_low = 10.0 * math.log10(f_blocked**2 * y_low / 1e-12)
    y_high = 1.0e-2            # high-mobility plate: v ~ v_f, P = v_f**2 / Y
    lw_high = 10.0 * math.log10(v_free**2 / y_high / 1e-12)
    lfb = building.equivalent_blocked_force_level(lw_low, y_low)
    lvf = building.equivalent_free_velocity_level(lw_high, y_high)
    y_rec = building.source_mobility_from_levels(lvf, lfb)
    assert float(y_rec) == pytest.approx(y_source, rel=1e-9)


def test_source_mobility_formula_19_round_trip() -> None:
    """(19) recovers a known source mobility from the level pair.

    Physical identity: a source of free velocity v_f and blocked force F_b
    has |Y_S| = v_f/F_b; expressing both as levels re 1e-9 m/s and 1e-6 N
    must return exactly that ratio.
    """

    v_f, f_b = 1.0e-4, 2.0e-2  # m/s, N -> |Y_S| = 5e-3 m/(N.s)
    lvf = 20.0 * math.log10(v_f / 1.0e-9)
    lfb = 20.0 * math.log10(f_b / 1.0e-6)
    y_s = float(building.source_mobility_from_levels(lvf, lfb))
    assert y_s == pytest.approx(v_f / f_b, rel=1e-12)


def test_iso9611_mean_free_velocity_level() -> None:
    """ISO 9611:1996 eq. (9): the energy mean over positions."""
    import reference_data as ref

    from phonometry.building.measurement.structure_borne_power import (
        FREE_VELOCITY_REFERENCE,
    )

    assert FREE_VELOCITY_REFERENCE == ref.ISO9611_FREE_VELOCITY_REFERENCE
    assert building.mean_free_velocity_level(
        ref.ISO9611_MEAN_LEVELS
    ) == pytest.approx(ref.ISO9611_MEAN_EXPECTED)
    # Uniform levels are a fixed point of the energy mean.
    assert building.mean_free_velocity_level([66.0, 66.0]) == pytest.approx(
        66.0
    )


def test_power_level_rejects_nonpositive_loss_factor() -> None:
    with pytest.raises(ValueError, match="loss_factor"):
        building.structure_borne_power_level(80.0, 1000.0, 10.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="loss_factor"):
        building.structure_borne_power_level(
            80.0, 1000.0, 10.0, 1.0, [0.01, -0.01]
        )
