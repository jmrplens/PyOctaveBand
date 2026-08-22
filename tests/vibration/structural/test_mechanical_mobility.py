#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ISO 7626-1:2011 mechanical mobility and the FRF family.

Anchored on the closed-form single-degree-of-freedom resonator (consistent
with the ISO 7626-1 Table 1 / 3.1.2 FRF definitions): at its resonance the
driving-point mobility is purely real and equal to ``1/c`` (the mobility peak
measures the damping), the static receptance is ``1/k``, and the Table-1 FRFs
are exact reciprocals / (jω)-powers of one another. The ISO 7626-2:2015
measurement-side acceptance criteria (7.5.2 rigid-mass calibration, Annex A
random error) are anchored on the standard's own values.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from phonometry import vibration

M, K, C = 2.0, 8000.0, 5.0  # mass [kg], stiffness [N/m], damping [N.s/m]
F0 = math.sqrt(K / M) / (2.0 * math.pi)  # ~10.066 Hz


# ---------------------------------------------------------------------------
# SDOF closed-form oracles (Table 1 / 3.1.2 definitions)
# ---------------------------------------------------------------------------
def test_resonance_frequency() -> None:
    assert vibration.resonance_frequency(M, K) == pytest.approx(F0, rel=1e-12)
    assert vibration.resonance_frequency(2.0, 8000.0) == pytest.approx(
        10.06578, abs=1e-4
    )


def test_mobility_peak_equals_inverse_damping() -> None:
    """At omega0 the driving-point mobility is real and |Y| = 1/c."""
    y0 = complex(vibration.sdof_mobility(F0, M, K, C))
    assert y0.real == pytest.approx(1.0 / C, rel=1e-9)
    assert y0.imag == pytest.approx(0.0, abs=1e-9)  # purely real at resonance


def test_static_receptance_is_inverse_stiffness() -> None:
    """As f -> 0 the receptance tends to the static compliance 1/k."""
    h = complex(vibration.sdof_receptance(1e-6, M, K, C))
    assert h.real == pytest.approx(1.0 / K, rel=1e-6)
    assert h.imag == pytest.approx(0.0, abs=1e-9)


def test_receptance_hand_value_off_resonance() -> None:
    """H(f) = 1/(k - w^2 m + j w c) at f = 20 Hz, hand-computed."""
    f = 20.0
    w = 2.0 * math.pi * f
    expected = 1.0 / (K - w**2 * M + 1j * w * C)
    assert complex(vibration.sdof_receptance(f, M, K, C)) == pytest.approx(
        expected, rel=1e-12
    )


# ---------------------------------------------------------------------------
# Table-1 conversion identities
# ---------------------------------------------------------------------------
def test_mobility_and_accelerance_are_jomega_powers_of_receptance() -> None:
    f = 37.0
    w = 2.0 * math.pi * f
    h = vibration.sdof_receptance(f, M, K, C)
    assert vibration.convert_frf(h, f, "receptance", "mobility") == pytest.approx(
        1j * w * h
    )
    assert vibration.convert_frf(h, f, "receptance", "accelerance") == pytest.approx(
        -(w**2) * h
    )


def test_reciprocals_multiply_to_one() -> None:
    f = 37.0
    h = vibration.sdof_receptance(f, M, K, C)
    y = vibration.convert_frf(h, f, "receptance", "mobility")
    a = vibration.convert_frf(h, f, "receptance", "accelerance")
    assert vibration.convert_frf(y, f, "mobility", "impedance") * y == pytest.approx(
        1.0
    )
    assert vibration.convert_frf(
        a, f, "accelerance", "apparent_mass"
    ) * a == pytest.approx(1.0)
    assert vibration.convert_frf(
        h, f, "receptance", "dynamic_stiffness"
    ) * h == pytest.approx(1.0)


def test_rigid_mass_decade_identity_at_1000_rad_s() -> None:
    """At omega = 1000 rad/s a rigid 1 kg mass spans exact decades:
    accelerance 1 1/kg <-> mobility 1e-3 m/(N.s) <-> compliance 1e-6 m/N.
    """
    f = 1000.0 / (2.0 * math.pi)
    y = vibration.convert_frf(1.0, f, "apparent_mass", "mobility")
    h = vibration.convert_frf(1.0, f, "apparent_mass", "receptance")
    a = vibration.convert_frf(1.0, f, "apparent_mass", "accelerance")
    assert abs(complex(y)) == pytest.approx(1.0e-3, rel=1e-12)
    assert abs(complex(h)) == pytest.approx(1.0e-6, rel=1e-12)
    assert abs(complex(a)) == pytest.approx(1.0, rel=1e-12)


def test_conversion_round_trip() -> None:
    f = np.array([10.0, 20.0, 50.0, 100.0])
    y = vibration.sdof_mobility(f, M, K, C)
    back = vibration.convert_frf(
        vibration.convert_frf(
            vibration.convert_frf(y, f, "mobility", "receptance"),
            f,
            "receptance",
            "accelerance",
        ),
        f,
        "accelerance",
        "mobility",
    )
    assert np.allclose(back, y)


def test_accelerance_matches_impedance_route() -> None:
    """Apparent mass at high frequency approaches the rigid mass m."""
    a = vibration.sdof_accelerance(5000.0, M, K, C)  # well above resonance
    apparent_mass = vibration.convert_frf(a, 5000.0, "accelerance", "apparent_mass")
    assert complex(apparent_mass).real == pytest.approx(M, rel=1e-2)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_unknown_frf_raises() -> None:
    with pytest.raises(ValueError, match="unknown source FRF"):
        vibration.convert_frf(1.0, 100.0, "velocity", "mobility")
    with pytest.raises(ValueError, match="unknown target FRF"):
        vibration.convert_frf(1.0, 100.0, "mobility", "velocity")


def test_zero_value_reciprocal_conversion_raises() -> None:
    # A dead channel cannot be converted through a force-per-motion reciprocal.
    with pytest.raises(ValueError, match="dead channel"):
        vibration.convert_frf(0.0, 100.0, "mobility", "impedance")
    with pytest.raises(ValueError, match="dead channel"):
        vibration.convert_frf([1e-3, 0.0], 100.0, "impedance", "mobility")
    # Motion-per-force to motion-per-force tolerates zeros (0 maps to 0).
    out = vibration.convert_frf(0.0, 100.0, "mobility", "accelerance")
    assert complex(out) == 0.0


def test_non_positive_frequency_raises() -> None:
    with pytest.raises(ValueError, match="frequency"):
        vibration.convert_frf(1.0, 0.0, "receptance", "mobility")
    with pytest.raises(ValueError, match="mass"):
        vibration.sdof_receptance(100.0, 0.0, K, C)
    with pytest.raises(ValueError, match="damping"):
        vibration.sdof_receptance(100.0, M, K, -1.0)


# ---------------------------------------------------------------------------
# Rigid-mass operational calibration (ISO 7626-2, 7.5.2)
# ---------------------------------------------------------------------------
def test_rigid_mass_calibration_accelerance_passes() -> None:
    """A 10 kg rigid block has |A| = 1/m = 0.100 1/kg at every frequency."""
    f = np.array([10.0, 100.0, 1000.0])
    res = vibration.rigid_mass_calibration_check(np.full(3, 0.1), f, 10.0)
    assert isinstance(res, vibration.RigidMassCalibrationResult)
    assert res.passed
    assert np.allclose(res.expected, 0.100)
    assert np.allclose(res.deviation, 0.0)


def test_rigid_mass_calibration_mobility_expected_value() -> None:
    """|Y| = 1/(2 pi f m): 1.59155e-4 m/(N.s) at 100 Hz for m = 10 kg."""
    y = 1.0 / (2.0 * math.pi * 100.0 * 10.0)
    res = vibration.rigid_mass_calibration_check(
        [y], [100.0], 10.0, quantity="mobility"
    )
    assert res.expected[0] == pytest.approx(1.59155e-4, rel=1e-5)
    assert res.passed


def test_rigid_mass_calibration_flags_out_of_tolerance_bands() -> None:
    # +4 % passes the +/-5 % criterion; -7 % fails, only in its band.
    f = np.array([50.0, 500.0])
    frf = np.array([0.104, 0.093])
    res = vibration.rigid_mass_calibration_check(frf, f, 10.0)
    assert res.within_tolerance.tolist() == [True, False]
    assert not res.passed
    assert res.deviation[0] == pytest.approx(0.04)


def test_rigid_mass_calibration_accepts_complex_frf() -> None:
    # The criterion applies to the magnitude; phase is irrelevant.
    frf = 0.1 * np.exp(1j * np.linspace(0.0, 1.0, 4))
    res = vibration.rigid_mass_calibration_check(frf, [10.0, 20.0, 40.0, 80.0], 10.0)
    assert res.passed


def test_rigid_mass_calibration_validation() -> None:
    with pytest.raises(ValueError, match="quantity"):
        vibration.rigid_mass_calibration_check(
            [0.1], [100.0], 10.0, quantity="receptance"
        )
    with pytest.raises(ValueError, match="mass"):
        vibration.rigid_mass_calibration_check([0.1], [100.0], 0.0)
    with pytest.raises(ValueError, match="same shape"):
        vibration.rigid_mass_calibration_check([0.1, 0.1], [100.0], 10.0)
    with pytest.raises(ValueError, match="frequency"):
        vibration.rigid_mass_calibration_check([0.1], [0.0], 10.0)


def test_a_tolerance_mask_short_of_its_frequencies_is_refused() -> None:
    """A mask that stops early would leave a pass standing over the rest.

    :attr:`within_tolerance` is what carries the 7.5.2 verdict frequency by
    frequency, and ``passed`` is stored rather than re-derived, so a mask
    covering the first frequencies of the range says nothing about the ones
    it never reached while the overall pass still reads true over all of
    them. Only :meth:`plot` protests, and then from inside numpy's boolean
    indexing, naming neither field.
    """
    f = np.geomspace(20.0, 2000.0, 12)
    res = vibration.rigid_mass_calibration_check(np.full(12, 0.1), f, mass=10.0)
    short = res.within_tolerance[:8]
    with pytest.raises(ValueError, match="'within_tolerance'"):
        dataclasses.replace(res, within_tolerance=short)


# ---------------------------------------------------------------------------
# Random error of averaged FRF estimates (ISO 7626-2, Annex A + 8.1.3)
# ---------------------------------------------------------------------------
def test_random_error_annex_a_example() -> None:
    """gamma2 = 0.8 with n = 75 averages gives 4.08 % < 5 % (Annex A)."""
    eps = float(vibration.random_error_percent(0.8, 75))
    assert eps == pytest.approx(4.08, abs=0.005)
    assert eps < 5.0


def test_random_error_perfect_coherence_is_zero() -> None:
    assert float(vibration.random_error_percent(1.0, 10)) == 0.0


def test_random_error_scales_with_averages() -> None:
    # eps ~ 1/sqrt(n): quadrupling n halves the error.
    one = vibration.random_error_percent(0.5, 10)
    four = vibration.random_error_percent(0.5, 40)
    assert float(one / four) == pytest.approx(2.0)


def test_random_error_validation() -> None:
    with pytest.raises(ValueError, match="n_averages"):
        vibration.random_error_percent(0.8, 0)
    with pytest.raises(ValueError, match="coherence"):
        vibration.random_error_percent(0.0, 10)
    with pytest.raises(ValueError, match="coherence"):
        vibration.random_error_percent(1.2, 10)


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------
def test_mobility_result_bundle() -> None:
    f = np.linspace(1.0, 50.0, 400)
    res = vibration.sdof_mobility_result(f, M, K, C)
    assert isinstance(res, vibration.MobilityResult)
    assert res.driving_point
    # the magnitude peak sits at the resonance
    peak_f = res.frequencies[int(np.argmax(res.magnitude))]
    assert peak_f == pytest.approx(F0, abs=0.2)
    # .to() converts to impedance = 1/Y
    z = res.to("impedance")
    assert np.allclose(z, 1.0 / res.mobility)


def test_a_mobility_short_of_its_frequencies_is_refused() -> None:
    """An FRF one value short is read position by position by everything.

    :meth:`plot` draws the mobility against the frequencies, the fiche reads
    the peak frequency at the peak index of ``|Y|``, and :meth:`to` divides
    one array by a function of the other; all three raise on lengths that
    disagree, out of matplotlib or numpy and naming neither field.
    """
    res = vibration.sdof_mobility_result(np.linspace(1.0, 50.0, 40), M, K, C)
    short = res.mobility[:-1]
    with pytest.raises(ValueError, match="'mobility'"):
        dataclasses.replace(res, mobility=short)


def test_plot_returns_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    res = vibration.sdof_mobility_result(np.linspace(1.0, 50.0, 200), M, K, C)
    assert res.plot() is not None


def test_rigid_mass_plot_two_panels_and_external_ax() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    f = np.logspace(np.log10(20.0), np.log10(2000.0), 40)
    # A calibration that drifts out of the +/-5 % band above ~1.2 kHz.
    meas = (1.0 / 10.0) * (1.0 + 0.02 * np.sin(f / 200.0) + 0.06 * (f > 1200.0))
    res = vibration.rigid_mass_calibration_check(meas, f, mass=10.0)
    axes = res.plot()
    assert len(axes) == 2  # magnitude + deviation panels
    _fig, ext = plt.subplots()
    assert res.plot(ax=ext) is ext  # single deviation panel on ax
    plt.close("all")


def test_rigid_mass_plot_rejects_unknown_language() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    res = vibration.rigid_mass_calibration_check([0.1], [100.0], 10.0)
    with pytest.raises(ValueError, match="Unknown language"):
        res.plot(language="xx")


# ---------------------------------------------------------------------------
# SDOF resonance — published tapping-machine/covering spot oracles
# ---------------------------------------------------------------------------

# These anchor the SDOF resonance relation f0 = (1/2pi) sqrt(k/m) on published
# building-acoustics worked values for the ISO tapping-machine hammer
# (m = 0.5 kg) resting on the contact stiffness of a floor covering — the
# input side of the ISO 16251-1 / ISO 10140-3 improvement prediction models.


def test_resonance_vigran_floor_covering_spot_values() -> None:
    # Vigran, Building Acoustics (2008), Sec. 8.4.5 pp. 319-321 (Figs. 8.36
    # and 8.37): hammer mass 0.5 kg on measured covering stiffnesses
    # s = 3.2e5 N/m (carpet squares) -> f0 ~= 130 Hz, and s = 5.2e6 N/m
    # (vinyl on felt) -> f0 ~= 510 Hz. The book quotes rounded frequencies
    # (exact values 127.3 and 513.3 Hz), so 4 Hz covers the rounding.
    # (The Fig. 8.37 caption prints the carpet stiffness as 3.2e6 N/m; the
    # body text value 3.2e5 N/m is the one that reproduces 130 Hz — a book
    # erratum, not one of the standard.)
    assert vibration.resonance_frequency(0.5, 3.2e5) == pytest.approx(130.0, abs=4.0)
    assert vibration.resonance_frequency(0.5, 5.2e6) == pytest.approx(510.0, abs=4.0)


def test_resonance_hopkins_contact_stiffness_cutoffs() -> None:
    # Hopkins, Sound Insulation (2007), Sec. 4.4.3.1 pp. 513-514 with the
    # model of Sec. 3.6.3.1 (Eqs. 3.97/3.98/3.102): ISO tapping machine
    # hammer m = 0.5 kg, contact radius r = 15 mm.
    m_hammer = 0.5
    r_contact = 0.015
    # Covering No. 2, E/d = 2.8e8 N/m3: K = (E/d)*pi*r^2 (Eq. 3.98) and
    # fco = (1/2pi) sqrt(K/m); Hopkins publishes fco = 100 Hz (exact
    # 100.1 Hz).
    k_covering_2 = 2.8e8 * math.pi * r_contact**2
    assert vibration.resonance_frequency(m_hammer, k_covering_2) == pytest.approx(
        100.0, abs=0.5
    )
    # Covering No. 1, E/d = 1.5e11 N/m3: Hopkins publishes fco ~= 2300 Hz
    # (exact 2318 Hz; the book rounds to two significant figures).
    k_covering_1 = 1.5e11 * math.pi * r_contact**2
    assert vibration.resonance_frequency(m_hammer, k_covering_1) == pytest.approx(
        2300.0, abs=25.0
    )
    # Bare 140 mm concrete slab: K = 2 r E / (1 - nu^2) (Eq. 3.97) with
    # E = cL^2 rho (1 - nu^2) from cL = 3800 m/s, rho = 2200 kg/m3,
    # nu = 0.2; Hopkins publishes fco ~= 7000 Hz (exact 6947 Hz, rounded to
    # one significant figure in the book).
    e_concrete = 3800.0**2 * 2200.0 * (1.0 - 0.2**2)
    k_plate = 2.0 * r_contact * e_concrete / (1.0 - 0.2**2)
    assert vibration.resonance_frequency(m_hammer, k_plate) == pytest.approx(
        7000.0, abs=100.0
    )
