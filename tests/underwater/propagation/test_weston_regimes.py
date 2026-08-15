#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for Weston's shallow-water propagation regimes.

Oracles: Ainslie, *Principles of Sonar Performance Modelling* (Springer 2010),
§9.1.1.2, printed pp. 452-458 -- the closed forms of Equations (9.42) to
(9.61), the seabed properties of Table 9.1 (printed p. 454) and the exact
energy-flux limit of an ideal waveguide, used here to cross-check the numerical
solvers of ``phonometry.underwater.propagation.numerical``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.special import erf

from phonometry.underwater.propagation.numerical import normal_modes, parabolic_equation
from phonometry.underwater.propagation.weston_regimes import (
    WESTON_REGIMES,
    WESTON_SEABEDS,
    WestonPropagationResult,
    WestonSeabed,
    critical_grazing_angle,
    effective_depth,
    loss_parameter,
    reflection_loss_gradient,
    waveguide_cutoff_frequency,
    weston_propagation_loss,
    weston_regime_boundaries,
)

# ---------------------------------------------------------------------------
# Table 9.1 (printed p. 454)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seabed", "beta", "expected", "half_digit"),
    [("sand", 0.88, 0.0161, 5e-5), ("mud", 0.09, 0.00165, 5e-6)],
)
def test_loss_parameter_reproduces_table_9_1(
    seabed: str, beta: float, expected: float, half_digit: float
) -> None:
    """ε = β_sed/(40·π·lg e), Ainslie Eq. (9.23); Table 9.1 prints 0.0161 and 0.00165.

    Tolerance: half a unit of the table's last printed digit.
    """
    assert WESTON_SEABEDS[seabed].attenuation_db_per_wavelength == beta
    assert loss_parameter(beta) == pytest.approx(expected, abs=half_digit)


def test_reflection_loss_gradient_sand_matches_table_9_1() -> None:
    """η_sand = 2·ε·(ρ_sed/ρ_w)·cos²ψc/sin³ψc (Eq. 9.51); Table 9.1 prints 0.28 Np/rad."""
    assert reflection_loss_gradient("sand") == pytest.approx(0.28, abs=0.005)


def test_reflection_loss_gradient_mud_matches_table_9_1_coefficient() -> None:
    """η_mud = 2·ω·ε/c' (Eq. 9.53); Table 9.1 prints ``0.021·f̂`` Np/rad."""
    assert reflection_loss_gradient("mud", frequency_hz=1.0) == pytest.approx(0.021, abs=5e-4)
    # Linear in frequency, as the table's f̂ factor states.
    assert reflection_loss_gradient("mud", frequency_hz=250.0) == pytest.approx(
        250.0 * reflection_loss_gradient("mud", frequency_hz=1.0), rel=1e-12
    )


def test_critical_angle_from_sound_speed_ratio() -> None:
    """ψc = arccos(c_w/c_sed): 33.56° for the sand ratio 1.20, none for mud (1.00)."""
    assert np.degrees(critical_grazing_angle(1.20)) == pytest.approx(33.557, abs=1e-3)
    assert critical_grazing_angle(1.00) == 0.0


# ---------------------------------------------------------------------------
# Equations (9.47), (9.50), (9.55), (9.57), (9.60)
# ---------------------------------------------------------------------------


def test_effective_depth_matches_equation_9_55() -> None:
    """He = H + (ρ_sed/ρ_w)/((ω/c_w)·sin ψc), recomputed by hand."""
    h, f, c = 50.0, 250.0, 1500.0
    k = 2.0 * np.pi * f / c
    psi_c = np.arccos(1.0 / 1.20)
    expected = h + 2.1 / (k * np.sin(psi_c))
    assert effective_depth(h, f, seabed="sand", sound_speed=c) == pytest.approx(expected, rel=1e-12)


def test_cutoff_frequency_matches_equation_9_60() -> None:
    """fc = (π − ρ_sed/ρ_w)/(2·π·sin ψc)·c/H, recomputed by hand."""
    h, c = 50.0, 1500.0
    psi_c = np.arccos(1.0 / 1.20)
    expected = (np.pi - 2.1) / (2.0 * np.pi * np.sin(psi_c)) * c / h
    assert waveguide_cutoff_frequency(h, seabed="sand", sound_speed=c) == pytest.approx(
        expected, rel=1e-12
    )


def test_cutoff_frequency_admits_exactly_one_mode() -> None:
    """At f = fc, Eq. (9.58) gives N = 1: (ω/c)·He·sin ψc = π."""
    h, c = 50.0, 1500.0
    fc = waveguide_cutoff_frequency(h, seabed="sand", sound_speed=c)
    bounds = weston_regime_boundaries(fc, h, seabed="sand", sound_speed=c)
    assert bounds.mode_count == pytest.approx(1.0, abs=1e-9)


def test_cylindrical_to_mode_stripping_boundary_equates_the_two_laws() -> None:
    """r_CS = π·H/(4·η·ψc²) (Eq. 9.50) is where Eq. (9.42) and Eq. (9.49) meet."""
    h, f = 50.0, 250.0
    b = weston_regime_boundaries(f, h, seabed="sand")
    expected = np.pi * h / (4.0 * b.reflection_loss_gradient * b.critical_angle**2)
    assert b.cylindrical_to_mode_stripping == pytest.approx(expected, rel=1e-12)
    r = b.cylindrical_to_mode_stripping
    f_cs = 2.0 * b.critical_angle / (r * h)
    f_ms = np.sqrt(np.pi / (b.reflection_loss_gradient * h)) * r**-1.5
    assert f_cs == pytest.approx(f_ms, rel=1e-12)


def test_mode_stripping_boundary_equates_theta_eff_with_mode_3_over_2() -> None:
    """r_MS is where θ_eff (Eq. 9.47) falls to θ_n (Eq. 9.56) at n = 3/2.

    This exercises the *published* rule, not the implementation formula: the
    effective angle is rebuilt from Equation (9.47) as printed, with the **true
    water depth H**, and the mode angle from Equation (9.56) as printed, with
    the **effective depth He**. Ainslie's printed Equation (9.57) reads
    ``k²·He³/(9·η)``, which is larger than what that rule gives by ``π·He/H``;
    this pins the derivation-consistent value (see ``docs/ERRATA.md``).
    """
    h, f, c = 50.0, 250.0, 1500.0
    b = weston_regime_boundaries(f, h, seabed="sand", sound_speed=c)
    k = 2.0 * np.pi * f / c
    eta = b.reflection_loss_gradient
    r = b.mode_stripping_to_single_mode
    # Equation (9.56) at n = 3/2, halfway between the first two mode angles.
    theta_32 = 1.5 * np.pi / (k * b.effective_depth)
    # Equation (9.47) as printed: the true depth h, not the effective depth.
    theta_eff = np.sqrt(np.pi * h / (4.0 * eta * r))
    assert theta_eff == pytest.approx(theta_32, rel=1e-12)
    # ...and the same rule solved symbolically for r.
    assert r == pytest.approx(k**2 * b.effective_depth**2 * h / (9.0 * np.pi * eta), rel=1e-12)
    # The printed Eq. (9.57) puts the transition where the effective angle has
    # already fallen below the *first* mode's own angle, which cannot be where
    # the second mode is stripped.
    r_printed = k**2 * b.effective_depth**3 / (9.0 * eta)
    theta_1 = np.pi / (k * b.effective_depth)
    assert r_printed == pytest.approx(r * np.pi * b.effective_depth / h, rel=1e-12)
    assert np.sqrt(np.pi * h / (4.0 * eta * r_printed)) < theta_1


def test_composite_loss_and_the_boundary_use_the_same_effective_angle() -> None:
    """θ_eff is Eq. (9.47) with H in both the multipath integral and the boundary.

    The multipath branch of :func:`weston_propagation_loss` and the
    mode-stripping boundary are two uses of the same printed equation, so at
    ``r_MS`` the angle implied by the boundary must be the angle the composite
    loss is evaluating.
    """
    h, f, c = 50.0, 250.0, 1500.0
    b = weston_regime_boundaries(f, h, seabed="sand", sound_speed=c)
    r = b.mode_stripping_to_single_mode
    res = weston_propagation_loss(r, f, h, seabed="sand", sound_speed=c)
    # Recover θ_eff from the printed Eq. (9.46) F_MP = (2·θ_eff/(r·H))·erf(...)
    # by inverting the closed form the module evaluates.
    theta_eff = np.sqrt(np.pi * h / (4.0 * b.reflection_loss_gradient * r))
    expected = (2.0 * theta_eff / (r * h)) * erf(np.sqrt(np.pi) * b.critical_angle / (2.0 * theta_eff))
    assert res.multipath[0] == pytest.approx(-10.0 * np.log10(expected), rel=1e-12)


def test_spherical_to_cylindrical_boundary() -> None:
    """1/r² and 2·ψc/(r·H) are equal at r = H/(2·ψc)."""
    h, f = 50.0, 250.0
    b = weston_regime_boundaries(f, h, seabed="sand")
    assert b.spherical_to_cylindrical == pytest.approx(h / (2.0 * b.critical_angle), rel=1e-12)


# ---------------------------------------------------------------------------
# Regime power laws
# ---------------------------------------------------------------------------


def test_regime_slopes_are_20_10_and_15_lg_r() -> None:
    """The three spreading regimes carry 20, 10 and 15 dB per decade of range."""
    h, f = 50.0, 250.0
    b = weston_regime_boundaries(f, h, seabed="sand")
    r = np.array([10.0, 100.0])
    res = weston_propagation_loss(r, f, h, seabed="sand")
    assert np.diff(res.spherical)[0] == pytest.approx(20.0, abs=1e-9)
    assert np.diff(res.cylindrical)[0] == pytest.approx(10.0, abs=1e-9)
    assert np.diff(res.mode_stripping)[0] == pytest.approx(15.0, abs=1e-9)
    assert b.cylindrical_to_mode_stripping > b.spherical_to_cylindrical


def test_multipath_integral_limits_to_the_two_regimes() -> None:
    """Eq. (9.46) tends to Eq. (9.42) at short range and Eq. (9.49) at long range."""
    h, f = 50.0, 250.0
    b = weston_regime_boundaries(f, h, seabed="sand")
    short = b.cylindrical_to_mode_stripping / 300.0
    long_ = b.cylindrical_to_mode_stripping * 300.0
    res = weston_propagation_loss([short, long_], f, h, seabed="sand")
    assert res.multipath[0] == pytest.approx(res.cylindrical[0], abs=0.05)
    assert res.multipath[1] == pytest.approx(res.mode_stripping[1], abs=0.05)


def test_composite_regime_labels_follow_the_boundaries() -> None:
    h, f = 50.0, 250.0
    b = weston_regime_boundaries(f, h, seabed="sand")
    ranges = np.array([
        b.spherical_to_cylindrical / 2.0,
        np.sqrt(b.spherical_to_cylindrical * b.cylindrical_to_mode_stripping),
        np.sqrt(b.cylindrical_to_mode_stripping * b.mode_stripping_to_single_mode),
        b.mode_stripping_to_single_mode * 2.0,
    ])
    res = weston_propagation_loss(ranges, f, h, seabed="sand")
    assert isinstance(res, WestonPropagationResult)
    assert list(res.regime) == list(WESTON_REGIMES)


def test_single_mode_regime_decays_exponentially() -> None:
    """Eq. (9.54) adds a constant dB per unit range beyond the last boundary."""
    h, f = 50.0, 250.0
    b = weston_regime_boundaries(f, h, seabed="sand")
    r0 = b.mode_stripping_to_single_mode
    res = weston_propagation_loss([r0, r0 + 1000.0, r0 + 2000.0], f, h, seabed="sand")
    d1 = res.single_mode[1] - res.single_mode[0]
    d2 = res.single_mode[2] - res.single_mode[1]
    # Beyond r_MS the exponential dominates the 1/r factor, so the increments
    # converge on the constant exp(-η λ²/(4 He³)) slope.
    assert d1 == pytest.approx(d2, rel=0.05)
    assert d2 > 0.0


def test_mud_is_lossier_than_sand_at_long_range() -> None:
    """Table 9.1 mud has no critical angle and a much larger η at 250 Hz."""
    h, f = 50.0, 250.0
    sand = weston_propagation_loss(20_000.0, f, h, seabed="sand")
    mud = weston_propagation_loss(20_000.0, f, h, seabed="mud", critical_angle=90.0)
    assert mud.propagation_loss[0] > sand.propagation_loss[0]


# ---------------------------------------------------------------------------
# Cross-checks against the numerical solvers
# ---------------------------------------------------------------------------

#: An ideal waveguide: pressure-release surface and bottom, no reflection loss,
#: so ψc = 90° and the flux result reduces to F = π/(r·H) exactly.
_IDEAL = {"critical_angle": 90.0, "reflection_loss_gradient_value": 0.0}


def test_ideal_waveguide_cylindrical_factor_is_pi_over_rh() -> None:
    """With ψc = π/2, Eq. (9.42) becomes F = π/(r·H) -- the exact modal limit."""
    h, r = 100.0, 25_000.0
    res = weston_propagation_loss(r, 100.0, h, **_IDEAL)
    assert res.propagation_factor[0] == pytest.approx(np.pi / (r * h), rel=1e-12)


@pytest.mark.parametrize("frequency", [50.0, 100.0, 200.0])
def test_normal_modes_range_average_matches_weston_cylindrical(frequency: float) -> None:
    """Depth- and range-averaged normal-mode loss converges on Weston's Eq. (9.42).

    The mean over range of the coherent modal field is the incoherent modal
    sum, and its continuum limit over many modes is exactly ``F = π/(r·H)``
    (the flux result with ψc = π/2). Averaging over receiver depth removes the
    ``sin²`` sampling bias of a single depth.
    """
    h, c = 100.0, 1500.0
    depths = np.array([0.0, h])
    speeds = np.array([c, c])
    ranges = np.linspace(20_000.0, 30_000.0, 2001)
    energies = [
        np.mean(10.0 ** (-normal_modes(
            frequency, depths, speeds, source_depth=41.0, receiver_depth=float(zr),
            ranges_m=ranges, bottom="pressure-release",
        ).propagation_loss / 10.0))
        for zr in np.linspace(10.0, 90.0, 9)
    ]
    numeric = -10.0 * np.log10(float(np.mean(energies)))
    flux = weston_propagation_loss(ranges, frequency, h, **_IDEAL)
    expected = -10.0 * np.log10(float(np.mean(10.0 ** (-flux.propagation_loss / 10.0))))
    assert numeric == pytest.approx(expected, abs=1.0)


def test_parabolic_equation_range_average_matches_weston_cylindrical() -> None:
    """The PE field converges on the same flux result, within its paraxial error.

    The standard PE is accurate within ~±15-20° of the horizontal, and an
    ideal 100 m waveguide at 50 Hz carries modes far steeper than that, so the
    PE under-propagates them and sits a couple of dB *lossier* than the flux
    law. The assertion is a one-sided band around that known bias rather than a
    symmetric tolerance: a symmetric band would also accept a PE that
    reproduced the flux law exactly, which for this geometry would be wrong.
    """
    h, c, f = 100.0, 1500.0, 50.0
    depths = np.array([0.0, h])
    speeds = np.array([c, c])
    pe = parabolic_equation(f, depths, speeds, source_depth=41.0, max_range=30_000.0,
                            range_step=5.0, n_depth_points=256)
    keep_r = pe.ranges >= 20_000.0
    keep_z = (pe.depths >= 10.0) & (pe.depths <= 90.0)
    numeric = -10.0 * np.log10(
        float(np.mean(10.0 ** (-pe.propagation_loss[np.ix_(keep_z, keep_r)] / 10.0)))
    )
    flux = weston_propagation_loss(pe.ranges[keep_r], f, h, **_IDEAL)
    expected = -10.0 * np.log10(float(np.mean(10.0 ** (-flux.propagation_loss / 10.0))))
    assert 1.5 < numeric - expected < 2.6


def test_ideal_waveguide_has_no_mode_stripping_or_single_mode_regime() -> None:
    res = weston_propagation_loss([10.0, 1e6], 100.0, 100.0, **_IDEAL)
    assert not np.isfinite(res.boundaries.cylindrical_to_mode_stripping)
    assert not np.isfinite(res.boundaries.mode_stripping_to_single_mode)
    assert set(res.regime) <= {"spherical", "cylindrical"}


# ---------------------------------------------------------------------------
# Validation and plotting
# ---------------------------------------------------------------------------


def test_plot_returns_axes() -> None:
    res = weston_propagation_loss(np.logspace(1.0, 5.0, 60), 250.0, 50.0, seabed="sand")
    assert res.plot() is not None
    assert res.plot(language="es") is not None
    plt.close("all")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seabed": "gravel"}, "seabed"),
        ({"sound_speed": 0.0}, "sound_speed"),
        ({"critical_angle": 0.0}, "critical_angle"),
        ({"critical_angle": 120.0}, "critical_angle"),
        ({"reflection_loss_gradient_value": -1.0}, "reflection_loss_gradient_value"),
        ({"source_depth": 1e6}, "source_depth"),
    ],
)
def test_invalid_arguments_raise(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        weston_propagation_loss(1000.0, 250.0, 50.0, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("ranges", [[], [0.0], [np.nan]])
def test_invalid_ranges_raise(ranges: list[float]) -> None:
    with pytest.raises(ValueError, match="range_m"):
        weston_propagation_loss(ranges, 250.0, 50.0)


def test_mud_without_critical_angle_needs_an_explicit_one() -> None:
    with pytest.raises(ValueError, match="no critical angle"):
        weston_propagation_loss(1000.0, 250.0, 50.0, seabed="mud")
    with pytest.raises(ValueError, match="critical angle"):
        effective_depth(50.0, 250.0, seabed="mud")
    with pytest.raises(ValueError, match="critical angle"):
        waveguide_cutoff_frequency(50.0, seabed="mud")


def test_refracting_branch_requires_a_frequency() -> None:
    with pytest.raises(ValueError, match="frequency_hz"):
        reflection_loss_gradient("mud")


def test_refracting_branch_requires_a_sediment_gradient() -> None:
    """Without a critical angle *and* without c', neither Eq. (9.51) nor (9.53) applies."""
    flat = WestonSeabed(
        name="flat", grain_size=8.0, sound_speed_ratio=1.0, density_ratio=1.4,
        attenuation_db_per_wavelength=0.09, loss_parameter=0.00165,
        sound_speed_gradient=0.0,
    )
    with pytest.raises(ValueError, match="sound_speed_gradient"):
        reflection_loss_gradient(flat, frequency_hz=250.0)


def test_negative_attenuation_rejected() -> None:
    with pytest.raises(ValueError, match="attenuation_db_per_wavelength"):
        loss_parameter(-1.0)
