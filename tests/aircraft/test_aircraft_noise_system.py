#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the IEC 61265:1995 aircraft-noise measurement-system verifier.

The oracle is IEC 61265 Table 1 (microphone directional-response tolerances),
transcribed verbatim, plus the scalar tolerance limits of clauses 4.5-4.7.
"""

from __future__ import annotations

import pytest

from phonometry import aircraft
from phonometry.aircraft.measurement_system import _iec61265_directional_limit


def test_directional_limits_table1() -> None:
    # Table 1 verbatim spot checks.
    assert _iec61265_directional_limit(1000.0, 90.0) == 1.0  # 50-1600 Hz row
    assert _iec61265_directional_limit(4000.0, 90.0) == 2.0
    assert _iec61265_directional_limit(10000.0, 150.0) == 7.5
    assert _iec61265_directional_limit(6300.0, 120.0) == 4.0


def test_directional_intermediate_angle_uses_greater() -> None:
    # 75 deg is between 60 and 90 -> use the 90 deg limit (subclause 4.4.2).
    assert _iec61265_directional_limit(4000.0, 75.0) == _iec61265_directional_limit(4000.0, 90.0)


def test_directional_pass() -> None:
    meas = {4000.0: {30: 0.4, 60: 0.9, 90: 1.9, 120: 2.4, 150: 2.4}}
    result = aircraft.verify_aircraft_noise_system(directional=meas)
    assert result["passed"] is True
    assert all(c["ok"] for c in result["checks"])


def test_directional_fail() -> None:
    meas = {4000.0: {90: 2.5}}  # limit is 2.0 dB at 4 kHz / 90 deg
    result = aircraft.verify_aircraft_noise_system(directional=meas)
    assert result["passed"] is False


def test_scalar_checks() -> None:
    result = aircraft.verify_aircraft_noise_system(
        frequency_response={1000.0: 1.2, 8000.0: 1.6},  # 1.6 > 1.5 -> fail
        linearity={"reference": 0.3, "other": 0.6},  # 0.6 > 0.5 -> fail
        resolution=0.1,
    )
    by_q = {(c["quantity"], c.get("frequency")): c["ok"] for c in result["checks"]}
    assert by_q[("frequency_response", 1000.0)] is True
    assert by_q[("frequency_response", 8000.0)] is False
    assert result["passed"] is False


def test_out_of_range_frequency_raises() -> None:
    with pytest.raises(ValueError):
        aircraft.verify_aircraft_noise_system(directional={20.0: {90: 0.5}})


def test_out_of_range_angle_raises() -> None:
    with pytest.raises(ValueError):
        aircraft.verify_aircraft_noise_system(directional={4000.0: {0.0: 0.5}})
    with pytest.raises(ValueError):
        aircraft.verify_aircraft_noise_system(
            directional={4000.0: {160.0: 0.5}}
        )


def test_linearity_rejects_unknown_key() -> None:
    with pytest.raises(ValueError):
        aircraft.verify_aircraft_noise_system(linearity={"refernce": 0.3})


def test_resolution_rejects_negative() -> None:
    assert (
        aircraft.verify_aircraft_noise_system(resolution=-1.0)["passed"]
        is False
    )


def test_empty_call_not_passed() -> None:
    assert aircraft.verify_aircraft_noise_system()["passed"] is False
