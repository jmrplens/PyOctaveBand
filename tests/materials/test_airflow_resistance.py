#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for ISO 9053-1:2018 (static) and ISO 9053-2:2020 (alternating) airflow
resistance.

Normative anchors:
- ISO 9053-1:2018, 3.1: R = dp / q_v            [Pa*s/m3].
- ISO 9053-1:2018, 3.2: R_s = R * A = dp / u     [Pa*s/m]  (NOT Pa*s/m2).
- ISO 9053-1:2018, 3.3: sigma = R_s / d          [Pa*s/m2].
- ISO 9053-1:2018, 3.4: u = q_v / A              [m/s].
- ISO 9053-1:2018, 7.5: stepwise dp = a*u + b*u**2 through the origin, evaluated
  at u = 0.5e-3 m/s; highest velocity <= 15e-3 m/s.
- ISO 9053-2:2020, 6.2: q_v = 2*pi*f*h*A_P.
- ISO 9053-2:2020, Formula (2): R = kappa'*P_S/(2*pi*f*V)*(h_t/h_s)*10**((L_ps-L_pt)/20).
- ISO 9053-2:2020, Formula (3)/(4): validity criteria.

Validation strategy: closed-form hand computation, independent of the module's
own control flow.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from reference_data import (
    ISO9053_2_ANNEX_A_BOUNDARY_LAYER,
    ISO9053_2_ANNEX_A_FREQUENCY,
    ISO9053_2_ANNEX_A_KAPPA_PRIME,
    ISO9053_2_ANNEX_A_SURFACE,
    ISO9053_2_ANNEX_A_VOLUME,
)

from phonometry.materials.airflow_resistance import (
    AirflowResistanceWarning,
    StaticAirflowResult,
    airflow_resistance,
    airflow_resistivity,
    alternating_airflow_resistance,
    effective_kappa,
    linear_airflow_velocity,
    piston_volume_flow_rate,
    specific_airflow_resistance,
    static_airflow_resistance,
    thermal_boundary_layer_thickness,
)

# --- 3.1-3.4: quantity chain and units --------------------------------------

def test_airflow_resistance_ratio() -> None:
    # R = dp / q_v = 20 / 0.001 = 20000 Pa*s/m3.
    assert airflow_resistance(20.0, 0.001) == pytest.approx(20000.0)


def test_linear_velocity() -> None:
    # u = q_v / A = 0.001 / 0.008 = 0.125 m/s.
    assert linear_airflow_velocity(0.001, 0.008) == pytest.approx(0.125)


def test_quantity_chain_round_trip() -> None:
    dp, q_v, area, thickness = 20.0, 0.001, 0.008, 0.05
    resistance = airflow_resistance(dp, q_v)  # 20000 Pa*s/m3
    velocity = linear_airflow_velocity(q_v, area)  # 0.125 m/s

    # R_s via both routes: R*A and dp/u must agree (Pa*s/m).
    r_s_from_r = specific_airflow_resistance(resistance, area)
    r_s_from_dp = specific_airflow_resistance(pressure_drop=dp, velocity=velocity)
    assert r_s_from_r == pytest.approx(160.0)  # 20000 * 0.008
    assert r_s_from_dp == pytest.approx(160.0)
    assert r_s_from_r == pytest.approx(r_s_from_dp)

    # sigma = R_s / d = 160 / 0.05 = 3200 Pa*s/m2.
    sigma = airflow_resistivity(r_s_from_r, thickness)
    assert sigma == pytest.approx(3200.0)


def test_specific_resistance_requires_exactly_one_route() -> None:
    with pytest.raises(ValueError):
        specific_airflow_resistance(20000.0, 0.008, pressure_drop=20.0, velocity=0.125)
    with pytest.raises(ValueError):
        specific_airflow_resistance()


@pytest.mark.parametrize(
    "call",
    [
        lambda: airflow_resistance(-1.0, 0.001),
        lambda: airflow_resistance(20.0, 0.0),
        lambda: airflow_resistivity(160.0, 0.0),
        lambda: linear_airflow_velocity(0.001, 0.0),
        lambda: specific_airflow_resistance(-1.0, 0.008),
        lambda: specific_airflow_resistance(pressure_drop=20.0, velocity=0.0),
    ],
)
def test_invalid_inputs_raise(call) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        call()


# --- 7.5: static-method through-origin regression ---------------------------

def test_static_regression_recovers_zero_velocity_intercept() -> None:
    # Synthetic data exactly on dp = a*u + b*u**2 with a known slope in R_s space:
    # R_s(u) = dp/u = a + b*u, so a is the zero-velocity specific resistance.
    a_true, b_true = 12000.0, 100000.0
    u = np.array([0.5e-3, 1.0e-3, 2.0e-3, 4.0e-3, 8.0e-3])
    dp = a_true * u + b_true * u**2
    area, thickness = 0.008, 0.05

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no range warning below 15 mm/s
        result = static_airflow_resistance(u, dp, area, thickness)

    assert isinstance(result, StaticAirflowResult)
    # Zero-velocity specific resistance = linear coefficient a.
    assert result.linear_coefficient == pytest.approx(a_true)
    assert result.quadratic_coefficient == pytest.approx(b_true)

    # Evaluated at the clause-7.5 reference u = 0.5e-3 m/s: R_s = a + b*u_eval.
    r_s_eval = a_true + b_true * 0.5e-3  # 12050 Pa*s/m
    assert result.evaluation_velocity == pytest.approx(0.5e-3)
    assert result.specific_resistance == pytest.approx(r_s_eval)
    assert result.resistance == pytest.approx(r_s_eval / area)
    assert result.resistivity == pytest.approx(r_s_eval / thickness)
    assert result.pressure_drop == pytest.approx(r_s_eval * 0.5e-3)


def test_static_resistivity_is_none_without_thickness() -> None:
    u = np.array([1.0e-3, 2.0e-3, 4.0e-3])
    dp = 12000.0 * u
    result = static_airflow_resistance(u, dp, 0.008)
    assert result.resistivity is None


def test_static_velocity_above_limit_warns() -> None:
    # Highest velocity 20 mm/s exceeds the 15 mm/s clause-7.5 limit.
    u = np.array([0.5e-3, 5.0e-3, 20.0e-3])
    dp = 12000.0 * u
    with pytest.warns(AirflowResistanceWarning, match="15"):
        static_airflow_resistance(u, dp, 0.008)


@pytest.mark.parametrize(
    "call",
    [
        lambda: static_airflow_resistance([1e-3], [12.0], 0.008),  # < 2 steps
        lambda: static_airflow_resistance([1e-3, 2e-3], [12.0], 0.008),  # length
        lambda: static_airflow_resistance([1e-3, -2e-3], [12.0, 24.0], 0.008),  # neg u
        lambda: static_airflow_resistance([1e-3, 2e-3], [12.0, 24.0], 0.0),  # area
    ],
)
def test_static_invalid_inputs_raise(call) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        call()


# --- 6.2 / Formula (2): alternating method ----------------------------------

def test_piston_volume_flow_rate() -> None:
    # q_v = 2*pi*f*h*A_P.
    f, h, a_p = 2.0, 1.4e-3, math.pi * (5.0e-3) ** 2  # piston diameter 10 mm
    expected = 2.0 * math.pi * f * h * a_p
    assert piston_volume_flow_rate(f, h, a_p) == pytest.approx(expected)


def test_alternating_formula_matches_hand_computation() -> None:
    # Independent replication of ISO 9053-2:2020 Formula (2).
    kappa, p_s, f, v = 1.4, 101325.0, 2.0, 1.0e-3
    h_t, h_s = 1.4e-3, 14.0e-3  # stroke ratio 0.1
    l_ps, l_pt = 60.0, 80.0     # level difference -20 dB -> factor 0.1
    expected = (
        kappa * p_s / (2.0 * math.pi * f * v)
        * (h_t / h_s)
        * 10.0 ** ((l_ps - l_pt) / 20.0)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # validity term 0.01 < 0.3, f in range
        result = alternating_airflow_resistance(
            l_ps,
            l_pt,
            piston_stroke_specimen=h_s,
            piston_stroke_termination=h_t,
            frequency=f,
            cavity_volume=v,
        )
    assert result == pytest.approx(expected)


def test_alternating_validity_term_warns() -> None:
    # (h_t/h_s)*10**((L_ps-L_pt)/20) = 0.5 * 1.0 = 0.5, not < 0.3 (Formula 3).
    with pytest.warns(AirflowResistanceWarning, match="Formula \\(3\\)"):
        alternating_airflow_resistance(
            70.0,
            70.0,
            piston_stroke_specimen=2.0e-3,
            piston_stroke_termination=1.0e-3,
            frequency=2.0,
            cavity_volume=1.0e-3,
        )


def test_alternating_background_margin_warns() -> None:
    # L_ps - L_pb = 5 dB, not > 10 dB (Formula 4).
    with pytest.warns(AirflowResistanceWarning, match="Formula \\(4\\)"):
        alternating_airflow_resistance(
            60.0,
            80.0,
            piston_stroke_specimen=14.0e-3,
            piston_stroke_termination=1.4e-3,
            frequency=2.0,
            cavity_volume=1.0e-3,
            background_level=55.0,
        )


def test_alternating_frequency_out_of_range_warns() -> None:
    with pytest.warns(AirflowResistanceWarning, match="6.2"):
        alternating_airflow_resistance(
            60.0,
            80.0,
            piston_stroke_specimen=14.0e-3,
            piston_stroke_termination=1.4e-3,
            frequency=6.0,  # outside 1-4 Hz
            cavity_volume=1.0e-3,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frequency": 0.0},
        {"cavity_volume": 0.0},
        {"piston_stroke_specimen": 0.0},
        {"piston_stroke_termination": 0.0},
        {"static_pressure": 0.0},
        {"kappa_prime": 0.0},
    ],
)
def test_alternating_invalid_inputs_raise(kwargs: dict[str, float]) -> None:
    base = {
        "piston_stroke_specimen": 14.0e-3,
        "piston_stroke_termination": 1.4e-3,
        "frequency": 2.0,
        "cavity_volume": 1.0e-3,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        alternating_airflow_resistance(60.0, 80.0, **base)


# --- Annex A (normative): effective ratio of specific heats kappa' -----------
# Worked example, ISO 9053-2:2020 Annex A.3 (page-verified oracle): closed cylinder
# 100 mm x 100 mm -> V = 7.854e-4 m3, S = 0.0471 m2; IEC 61094-2:2009 air properties
# at 23 C; f = 2 Hz give b = 1.83e-3 m and kappa' = kappa*0.978 = 1.370. The inputs
# and expected values are imported from tests/reference_data.py (shared with the
# CI conformance report).

def test_thermal_boundary_layer_thickness_annex_a_example() -> None:
    b = thermal_boundary_layer_thickness(frequency=ISO9053_2_ANNEX_A_FREQUENCY)
    assert b == pytest.approx(ISO9053_2_ANNEX_A_BOUNDARY_LAYER, abs=5e-6)


def test_effective_kappa_annex_a_example() -> None:
    kappa_prime = effective_kappa(
        cavity_surface=ISO9053_2_ANNEX_A_SURFACE,
        cavity_volume=ISO9053_2_ANNEX_A_VOLUME,
        frequency=ISO9053_2_ANNEX_A_FREQUENCY,
    )
    # Standard prints kappa' = kappa*0.978 = 1.370 (kappa = 1.4008).
    assert kappa_prime == pytest.approx(ISO9053_2_ANNEX_A_KAPPA_PRIME, abs=5e-4)
    assert kappa_prime == pytest.approx(1.4008 * 0.978, abs=1e-3)


def test_effective_kappa_below_adiabatic() -> None:
    # Heat conduction always lowers kappa below the adiabatic value.
    kappa_prime = effective_kappa(
        cavity_surface=ISO9053_2_ANNEX_A_SURFACE,
        cavity_volume=ISO9053_2_ANNEX_A_VOLUME,
        frequency=ISO9053_2_ANNEX_A_FREQUENCY,
    )
    assert kappa_prime < 1.4008


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cavity_surface": 0.0},
        {"cavity_volume": 0.0},
        {"frequency": 0.0},
        {"specific_heat_ratio": 0.0},
    ],
)
def test_effective_kappa_invalid_inputs_raise(kwargs: dict[str, float]) -> None:
    base = {"cavity_surface": 0.0471, "cavity_volume": 7.854e-4, "frequency": 2.0}
    base.update(kwargs)
    with pytest.raises(ValueError):
        effective_kappa(**base)


def test_public_exports() -> None:
    import phonometry

    for name in (
        "AirflowResistanceWarning", "StaticAirflowResult", "airflow_resistance",
        "specific_airflow_resistance", "airflow_resistivity", "linear_airflow_velocity",
        "static_airflow_resistance", "piston_volume_flow_rate",
        "alternating_airflow_resistance", "effective_kappa",
        "thermal_boundary_layer_thickness",
    ):
        assert hasattr(phonometry, name), name
