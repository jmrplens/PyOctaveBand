#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for EN 29052-1:1992 dynamic stiffness of resilient materials.

Anchored on hand-computed closed-form values of the resonance relations
(Formulae 2-4) and on the standard's own worked NOTE for the enclosed-gas term
(``s'a = 111 / d`` MN/m3 for ``p0 = 0,1 MPa``, ``epsilon = 0,9``, clause 8.2) --
a genuine published numeric oracle.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import materials
from phonometry.materials.resilient.dynamic_stiffness import DynamicStiffnessWarning


# ---------------------------------------------------------------------------
# Apparent dynamic stiffness (Formula 4) and its resonance inverse (Formula 3)
# ---------------------------------------------------------------------------
def test_apparent_stiffness_hand_value() -> None:
    """s't = 4 pi^2 m't fr^2: 4 pi^2 * 200 * 25^2 = 4.934802 MN/m3."""
    st = materials.apparent_dynamic_stiffness(25.0, 200.0)
    assert st == pytest.approx(4_934_802.200545, rel=1e-9)
    assert st == pytest.approx(4.0 * math.pi**2 * 200.0 * 25.0**2, rel=1e-12)


def test_apparent_stiffness_inverts_the_resonance() -> None:
    """Recovering fr from s't via Formula 3 returns the input frequency."""
    st = materials.apparent_dynamic_stiffness(25.0, 200.0)
    assert materials.natural_frequency(st, 200.0) == pytest.approx(
        25.0, rel=1e-12
    )


def test_apparent_stiffness_vectorised() -> None:
    st = materials.apparent_dynamic_stiffness([20.0, 25.0, 30.0], 200.0)
    assert isinstance(st, np.ndarray)
    assert st.shape == (3,)
    assert np.all(np.diff(st) > 0.0)   # rises with fr^2


# ---------------------------------------------------------------------------
# Enclosed-gas stiffness (Formula 7) -- standard's 111/d NOTE oracle
# ---------------------------------------------------------------------------
def test_enclosed_gas_matches_standard_note() -> None:
    """s'a = p0/(d eps); with p0=0,1 MPa, eps=0,9 the NOTE gives ~111/d MN/m3.

    The closed form yields 100000/(0,9) = 111.11.../d MN/m3 (d in mm); the
    standard rounds the printed coefficient to 111.
    """
    for d_mm in (10.0, 20.0, 50.0):
        sa = materials.enclosed_gas_stiffness(
            d_mm / 1000.0, 0.9
        )  # thickness in metres
        assert sa == pytest.approx(1.0e5 / ((d_mm / 1000.0) * 0.9), rel=1e-12)
        # cross-check against the standard's printed 111/d relationship
        assert sa / 1e6 == pytest.approx(111.0 / d_mm, rel=2e-3)


def test_enclosed_gas_true_atmosphere() -> None:
    """A real 101 325 Pa can be passed instead of the standard's 0,1 MPa."""
    sa = materials.enclosed_gas_stiffness(
        0.02, 0.9, atmospheric_pressure=101_325.0
    )
    assert sa == pytest.approx(101_325.0 / (0.02 * 0.9), rel=1e-12)


def test_enclosed_gas_bad_porosity_raises() -> None:
    with pytest.raises(ValueError, match="porosity"):
        materials.enclosed_gas_stiffness(0.02, 0.0)
    with pytest.raises(ValueError, match="porosity"):
        materials.enclosed_gas_stiffness(0.02, 1.5)


# ---------------------------------------------------------------------------
# Natural frequency (Formula 2)
# ---------------------------------------------------------------------------
def test_natural_frequency_hand_value() -> None:
    """f0 = (1/2pi) sqrt(s'/m'): sqrt(10e6/100)/(2pi) = 50.329 Hz."""
    assert materials.natural_frequency(10.0e6, 100.0) == pytest.approx(
        50.3292121, rel=1e-6
    )


def test_natural_frequency_scales() -> None:
    """f0 scales with sqrt(s') and 1/sqrt(m')."""
    base = float(materials.natural_frequency(10.0e6, 100.0))
    assert materials.natural_frequency(40.0e6, 100.0) == pytest.approx(
        2.0 * base, rel=1e-9
    )
    assert materials.natural_frequency(10.0e6, 400.0) == pytest.approx(
        base / 2.0, rel=1e-9
    )


# ---------------------------------------------------------------------------
# Airflow-resistivity combination (clause 8.2)
# ---------------------------------------------------------------------------
def test_high_resistivity_uses_apparent_only() -> None:
    assert (
        materials.installed_dynamic_stiffness(20e6, 150.0, gas_stiffness=3e6)
        == 20e6
    )


def test_intermediate_resistivity_adds_gas() -> None:
    assert (
        materials.installed_dynamic_stiffness(20e6, 50.0, gas_stiffness=3e6)
        == 23e6
    )
    # boundary at 10 kPa.s/m2 is inclusive of the intermediate branch
    assert (
        materials.installed_dynamic_stiffness(20e6, 10.0, gas_stiffness=3e6)
        == 23e6
    )


def test_low_resistivity_negligible_gas_warns_and_uses_apparent() -> None:
    with pytest.warns(DynamicStiffnessWarning, match="disregarded"):
        s = materials.installed_dynamic_stiffness(
            20e6, 5.0, gas_stiffness=1e6
        )  # 5 % of s't
    assert s == 20e6


def test_low_resistivity_significant_gas_is_unresolvable() -> None:
    with pytest.warns(DynamicStiffnessWarning, match="cannot resolve"):
        s = materials.installed_dynamic_stiffness(
            20e6, 5.0, gas_stiffness=5e6
        )  # 25 % of s't
    assert math.isnan(s)


# ---------------------------------------------------------------------------
# Full chain
# ---------------------------------------------------------------------------
def test_floating_floor_resonance_chain() -> None:
    res = materials.floating_floor_resonance(
        25.0, 200.0, 100.0, airflow_resistivity=50.0, thickness=0.02, porosity=0.9
    )
    assert isinstance(res, materials.DynamicStiffnessResult)
    assert res.apparent_stiffness == pytest.approx(4_934_802.2, rel=1e-6)
    assert res.gas_stiffness == pytest.approx(1.0e5 / (0.02 * 0.9), rel=1e-9)
    assert res.dynamic_stiffness == pytest.approx(
        res.apparent_stiffness + res.gas_stiffness, rel=1e-12
    )
    assert res.natural_frequency == pytest.approx(
        math.sqrt(res.dynamic_stiffness / 100.0) / (2.0 * math.pi), rel=1e-9
    )


def test_chain_high_resistivity_ignores_gas() -> None:
    """With r >= 100 kPa.s/m2 the gas term is not needed."""
    res = materials.floating_floor_resonance(
        25.0, 200.0, 100.0
    )  # default r = inf
    assert res.gas_stiffness == 0.0
    assert res.dynamic_stiffness == res.apparent_stiffness


def test_chain_requires_gas_inputs_below_100() -> None:
    with pytest.raises(ValueError, match="thickness.*porosity"):
        materials.floating_floor_resonance(
            25.0, 200.0, 100.0, airflow_resistivity=50.0
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_non_positive_inputs_raise() -> None:
    with pytest.raises(ValueError, match="resonant_frequency"):
        materials.apparent_dynamic_stiffness(0.0, 200.0)
    with pytest.raises(ValueError, match="total_mass_per_area"):
        materials.apparent_dynamic_stiffness(25.0, 0.0)
    with pytest.raises(ValueError, match="mass_per_area"):
        materials.natural_frequency(10e6, 0.0)
    with pytest.raises(ValueError, match="airflow_resistivity"):
        materials.installed_dynamic_stiffness(20e6, 0.0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def test_plot_returns_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    res = materials.floating_floor_resonance(25.0, 200.0, 100.0)
    assert res.plot() is not None


# ---------------------------------------------------------------------------
# Floating-floor natural frequency — Vigran's published outcomes
# ---------------------------------------------------------------------------

# Vigran, Building Acoustics (2008), Sec. 8.4.4 "Examples" pp. 317-318,
# prints the constructions and the resulting natural frequencies (~40 Hz and
# ~90 Hz); the numeric inputs below are reconstructed from the same section:
# the total dynamic stiffness s' = 8.0 MPa/m of the 25 mm mineral-wool layer
# follows from the Table 8.3 dynamic E-modulus plus the enclosed-air
# stiffness, and the plate masses per unit area are nominal values for the
# stated build-ups.


def test_natural_frequency_vigran_concrete_floating_floor() -> None:
    # 50 mm concrete floating slab (nominal m' ~ 115 kg/m2) on the
    # s' = 8.0 MPa/m layer: the book publishes f0 ~= 40 Hz, rounded to the
    # nearest 10 Hz (the reconstruction gives 42.0 Hz exactly), so 3 Hz
    # covers the rounding plus the nominal slab mass.
    assert materials.natural_frequency(8.0e6, 115.0) == pytest.approx(
        40.0, abs=3.0
    )


def test_natural_frequency_vigran_lightweight_floating_floor() -> None:
    # Same layer under a lightweight top plate of 22 mm chipboard + 13 mm
    # plasterboard (nominal m' ~ 16.7 + 11.2 ~ 28 kg/m2): the book publishes
    # f0 ~= 90 Hz, rounded to the nearest 10 Hz (the reconstruction gives
    # 85.1 Hz), so 5.5 Hz covers the rounding plus the nominal plate masses.
    assert materials.natural_frequency(8.0e6, 28.0) == pytest.approx(
        90.0, abs=5.5
    )
