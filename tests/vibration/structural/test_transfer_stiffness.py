#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ISO 10846 dynamic transfer stiffness of resilient elements.

Anchored on closed-form identities: the level ``L_k = 20 lg(|k|/k0)`` re
1 N/m, the direct-method ratio ``k2,1 = F2,b/u1``, the indirect-method inertia
relation ``k2,1 = -(2 pi f)^2 (m2+mf) T`` (Equation 1), the loss factor
``eta = Im/Re``, and the Annex-A / Table-A.2 FRF relations ``k = j omega Z =
-omega^2 m_eff``. The physically-grounded oracle is a massless Kelvin-Voigt
element (``k + j omega c``) loaded by a mass: its base transmissibility fed
back through the indirect relation recovers the element stiffness at ``T << 1``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from phonometry import PhonometryWarning, vibration
from phonometry.vibration.structural.transfer_stiffness import TRANSMISSIBILITY_LIMIT


# ---------------------------------------------------------------------------
# Level (ISO 10846-2/-3, 3.17) — re k0 = 1 N/m
# ---------------------------------------------------------------------------
def test_level_reference_decade() -> None:
    assert vibration.transfer_stiffness_level(1.0) == pytest.approx(0.0)
    assert vibration.transfer_stiffness_level(1e6) == pytest.approx(120.0)
    assert vibration.transfer_stiffness_level(2e6) == pytest.approx(126.0206, abs=1e-3)


def test_level_uses_magnitude_of_complex() -> None:
    # |3+4j| = 5  ->  20 lg 5
    assert vibration.transfer_stiffness_level(3.0 + 4.0j) == pytest.approx(
        20.0 * math.log10(5.0)
    )


def test_level_custom_reference() -> None:
    assert vibration.transfer_stiffness_level(1e3, reference=1e3) == pytest.approx(0.0)


def test_level_rejects_non_positive_reference() -> None:
    with pytest.raises(ValueError, match="reference"):
        vibration.transfer_stiffness_level(1e6, reference=0.0)


def test_level_rejects_zero_stiffness() -> None:
    # A dead channel (|k| = 0) has no level; -inf must not propagate silently.
    with pytest.raises(ValueError, match="zero"):
        vibration.transfer_stiffness_level(0.0)
    with pytest.raises(ValueError, match="dead-channel"):
        vibration.transfer_stiffness_level([1e6, 0.0 + 0.0j])


# ---------------------------------------------------------------------------
# Loss factor (ISO 10846-1, 3.8)
# ---------------------------------------------------------------------------
def test_loss_factor_is_tan_phase() -> None:
    # k = k0 (1 + j eta)  ->  Im/Re = eta
    assert vibration.loss_factor(1e6 * (1.0 + 0.05j)) == pytest.approx(0.05)
    assert vibration.loss_factor(1e6 + 1j * 3e4) == pytest.approx(0.03)


def test_loss_factor_rejects_purely_imaginary_stiffness() -> None:
    # Re(k) = 0 -> eta = Im/Re is undefined, not inf.
    with pytest.raises(ValueError, match="purely imaginary"):
        vibration.loss_factor(1j * 1e6)
    with pytest.raises(ValueError, match="purely imaginary"):
        vibration.loss_factor([1e6 + 1e4j, 0.0 + 5e5j])


# ---------------------------------------------------------------------------
# Direct method (ISO 10846-2)
# ---------------------------------------------------------------------------
def test_direct_method_ratio() -> None:
    k = vibration.transfer_stiffness_direct(5.0 + 0j, 1e-6 + 0j)
    assert complex(k) == pytest.approx(5e6)


def test_direct_method_preserves_phase() -> None:
    # A phase lag in the force appears directly in k2,1.
    k = vibration.transfer_stiffness_direct(2.0 * np.exp(1j * 0.3), 1e-3 + 0j)
    assert np.angle(complex(k)) == pytest.approx(0.3)


def test_direct_method_rejects_zero_displacement() -> None:
    # A dead u1 channel must raise, not return an infinite stiffness.
    with pytest.raises(ValueError, match="input_displacement"):
        vibration.transfer_stiffness_direct(5.0 + 0j, 0.0 + 0j)
    with pytest.raises(ValueError, match="dead input channel"):
        vibration.transfer_stiffness_direct([5.0, 4.0], [1e-6, 0.0])


# ---------------------------------------------------------------------------
# Indirect method (ISO 10846-3, Equation 1)
# ---------------------------------------------------------------------------
def test_indirect_matches_equation_1() -> None:
    f, m2, t = 500.0, 10.0, 0.01 + 0j
    expected = -((2.0 * math.pi * f) ** 2) * m2 * t
    assert complex(vibration.transfer_stiffness_indirect(f, t, m2)) == pytest.approx(
        expected
    )


def test_indirect_includes_flange_mass() -> None:
    f, t = 300.0, 0.02 + 0j
    k_no_flange = vibration.transfer_stiffness_indirect(f, t, 8.0)
    k_flange = vibration.transfer_stiffness_indirect(f, t, 8.0, flange_mass=2.0)
    assert complex(k_flange) == pytest.approx(complex(k_no_flange) * 10.0 / 8.0)


def test_indirect_recovers_kelvin_voigt_at_high_frequency() -> None:
    """A massless (k + jwc) element loaded by m: T -> indirect recovers k+jwc."""
    k, c, m = 1e6, 200.0, 5.0
    f0 = math.sqrt(k / m) / (2.0 * math.pi)
    f = 30.0 * f0  # well into T << 1
    t = vibration.base_transmissibility(f, m, k, c)
    k_rec = complex(vibration.transfer_stiffness_indirect(f, t, m))
    k_true = k + 1j * (2.0 * math.pi * f) * c
    assert k_rec == pytest.approx(k_true, rel=5e-3)


def test_indirect_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="frequency"):
        vibration.transfer_stiffness_indirect(0.0, 0.01 + 0j, 10.0)
    with pytest.raises(ValueError, match="blocking_mass"):
        vibration.transfer_stiffness_indirect(500.0, 0.01 + 0j, 0.0)


# ---------------------------------------------------------------------------
# Validity of the T << 1 approximation (ISO 10846-3, 6.1, Inequality 2)
# ---------------------------------------------------------------------------
def test_indirect_warns_above_transmissibility_limit() -> None:
    # |T| = 0.5 violates Inequality (2) (DeltaL1,2 < 20 dB): a warning fires.
    with pytest.warns(PhonometryWarning, match="Inequality"):
        vibration.transfer_stiffness_indirect(50.0, 0.5 + 0j, 10.0)


def test_indirect_silent_within_transmissibility_limit() -> None:
    # |T| = 0.05 satisfies Inequality (2): no PhonometryWarning is emitted.
    with warnings.catch_warnings():
        warnings.simplefilter("error", PhonometryWarning)
        vibration.transfer_stiffness_indirect(500.0, 0.05 + 0j, 10.0)
        vibration.transfer_stiffness_indirect(500.0, TRANSMISSIBILITY_LIMIT + 0j, 10.0)


def test_result_helper_warns_above_transmissibility_limit() -> None:
    f = np.array([100.0, 200.0])
    t = np.array([0.5 + 0j, 0.02 + 0j])
    with pytest.warns(PhonometryWarning, match="ISO 10846-3"):
        vibration.indirect_transfer_stiffness_result(f, t, blocking_mass=8.0)


def test_indirect_bias_at_the_validity_limit_is_within_1_db() -> None:
    """At |T| = 0.1 (DeltaL1,2 = 20 dB) the undamped mass-spring model gives
    k_indirect = 1.1 k: a 0.83 dB (10 %) bias, inside the 1 dB (12 %) bound."""
    k, m = 1e6, 1.0
    f = math.sqrt(11.0 * k / m) / (2.0 * math.pi)  # omega^2 m = 11 k -> T = -0.1
    t = complex(vibration.base_transmissibility(f, m, k))
    assert abs(t) == pytest.approx(TRANSMISSIBILITY_LIMIT)
    k_rec = complex(vibration.transfer_stiffness_indirect(f, t, m))
    assert abs(k_rec) / k == pytest.approx(1.1)
    bias_db = 20.0 * math.log10(abs(k_rec) / k)
    assert bias_db == pytest.approx(0.828, abs=0.001)
    assert bias_db <= 1.0
    assert abs(k_rec) / k - 1.0 <= 0.12


def test_linearity_criterion_10_db_step() -> None:
    """ISO 10846-2/-3, 7.6: input spectra 10 dB apart must give transfer-
    stiffness levels within 1.5 dB - exact equality for a linear element."""
    u_a, step = 1e-6 + 0j, 10.0 ** (-10.0 / 20.0)
    k = 1e6 + 3e4j
    lk_a = float(
        vibration.transfer_stiffness_level(
            vibration.transfer_stiffness_direct(k * u_a, u_a)
        )
    )
    u_b = u_a * step
    lk_b = float(
        vibration.transfer_stiffness_level(
            vibration.transfer_stiffness_direct(k * u_b, u_b)
        )
    )
    assert abs(lk_a - lk_b) <= 1.5
    assert lk_a == pytest.approx(lk_b, abs=1e-9)


def test_indirect_still_recovers_kelvin_voigt_below_the_warning() -> None:
    """The valid-range identity of Formula (1) is unchanged by the guard."""
    k, c, m = 1e6, 200.0, 5.0
    f = 30.0 * math.sqrt(k / m) / (2.0 * math.pi)  # T << 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", PhonometryWarning)
        t = vibration.base_transmissibility(f, m, k, c)
        k_rec = complex(vibration.transfer_stiffness_indirect(f, t, m))
    assert k_rec == pytest.approx(k + 1j * (2.0 * math.pi * f) * c, rel=5e-3)


# ---------------------------------------------------------------------------
# Blocking-force approximation (ISO 10846-1, Equations 6 and 7)
# ---------------------------------------------------------------------------
def test_blocking_force_ratio_at_the_ten_percent_limit() -> None:
    """|k2,2/kt| = 0.1 gives F2/F2,b = 1/1.1 = 0.9091 - within 10 % (Eq. 7)."""
    ratio = complex(vibration.blocking_force_ratio(1e5, 1e6))
    assert ratio == pytest.approx(1.0 / 1.1)
    assert abs(abs(ratio) - 1.0) <= 0.10


def test_blocking_force_ratio_stiff_termination_is_unity() -> None:
    # An infinitely stiff receiver takes exactly the blocking force.
    assert complex(vibration.blocking_force_ratio(1e5, 1e12)) == pytest.approx(
        1.0, abs=1e-6
    )


def test_blocking_force_ratio_complex_and_validation() -> None:
    k22, kt = 1e5 + 1e4j, 2e6 + 5e5j
    assert complex(vibration.blocking_force_ratio(k22, kt)) == pytest.approx(
        1.0 / (1.0 + k22 / kt)
    )
    with pytest.raises(ValueError, match="termination_stiffness"):
        vibration.blocking_force_ratio(1e5, 0.0)


# ---------------------------------------------------------------------------
# Annex A / Table A.2 FRF relations (k = jwZ = -w^2 m_eff)
# ---------------------------------------------------------------------------
def test_stiffness_impedance_effective_mass_relations() -> None:
    f = 250.0
    w = 2.0 * math.pi * f
    k = 1e6 + 1j * 5e4
    z = vibration.convert_frf(k, f, "dynamic_stiffness", "impedance")
    m_eff = vibration.convert_frf(k, f, "dynamic_stiffness", "apparent_mass")
    assert complex(k) == pytest.approx(1j * w * complex(z))
    assert complex(k) == pytest.approx(-(w**2) * complex(m_eff))


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------
def test_result_bundle() -> None:
    f = np.array([100.0, 200.0, 400.0])
    t = np.array([0.05, 0.02, 0.008]) * np.exp(1j * 0.1)
    res = vibration.indirect_transfer_stiffness_result(f, t, blocking_mass=8.0)
    assert isinstance(res, vibration.TransferStiffnessResult)
    assert res.blocking_mass == 8.0
    # level is the level of the bundled stiffness
    assert np.allclose(
        res.level, vibration.transfer_stiffness_level(res.transfer_stiffness)
    )
    # .to("impedance") = k/(jw)
    z = res.to("impedance")
    assert np.allclose(z, res.transfer_stiffness / (1j * 2.0 * np.pi * f))
    # loss factor of a constant-phase transmissibility is tan(0.1)
    assert np.allclose(res.loss_factor, math.tan(0.1))


def test_plot_returns_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    f = np.logspace(1, 3, 50)
    t = vibration.base_transmissibility(f, 5.0, 1e6, 200.0)
    # The synthetic curve exceeds |T| = 0.1 near resonance, so the
    # ISO 10846-3 validity advisory is expected, as in the tests above.
    with pytest.warns(PhonometryWarning, match="ISO 10846-3"):
        res = vibration.indirect_transfer_stiffness_result(f, t, blocking_mass=5.0)
    assert res.plot() is not None
