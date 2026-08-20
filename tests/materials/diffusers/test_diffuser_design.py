#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Diffuser-design far-field prediction tests (Cox & D'Antonio Fraunhofer model).

The predictor has no numeric worked example to anchor on, so every test rests
on a closed-form geometry identity or an exact-by-physics behaviour:

- the quadratic residue sequence and depths reproduce Eqs. (10.2)/(10.3) by
  hand (``s = {0,1,4,2,2,4,1}`` for N = 7; ``d_max = s_max c / (2 N f0)``);
- a flat panel (all wells zero) collapses to a single specular lobe with a low
  raw diffusion, and normalises against itself to exactly zero (Formula (7));
- a well-designed QRD scatters far more evenly than the flat panel of the same
  footprint, so its predicted diffusion is markedly higher and its normalised
  coefficient is clearly positive;
- the explicit per-well ``reflection`` path reproduces the rigid-bottom
  ``depths`` path (``R_n = exp(-2 j k d_n)``);
- input-validation guards raise ``ValueError``.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry.materials.diffusers.design import (
    DEFAULT_POLAR_ANGLES,
    DiffuserPolarResponse,
    predict_diffuser_polar_response,
    predicted_diffusion_spectrum,
    qrd_well_depths,
    quadratic_residue_sequence,
)
from phonometry.materials.diffusers.scattering_diffusion import DiffusionSpectrum


# --- Geometry: quadratic residue sequence and depths (Eqs. (10.2)/(10.3)) ---
def test_quadratic_residue_sequence_n7_by_hand() -> None:
    np.testing.assert_array_equal(quadratic_residue_sequence(7), [0, 1, 4, 2, 2, 4, 1])


def test_quadratic_residue_sequence_n5_by_hand() -> None:
    np.testing.assert_array_equal(quadratic_residue_sequence(5), [0, 1, 4, 4, 1])


def test_qrd_depths_match_equation_10_3() -> None:
    # d_n = s_n * lambda0 / (2 N); lambda0 = c / f0.
    c, f0, prime = 343.0, 500.0, 7
    lambda0 = c / f0
    expected = quadratic_residue_sequence(prime) * lambda0 / (2 * prime)
    depths = qrd_well_depths(prime, f0, speed_of_sound=c)
    np.testing.assert_allclose(depths, expected)
    # Deepest well: s_max = 4 for N = 7 -> 4 * 0.686 / 14 = 0.196 m.
    assert float(depths.max()) == pytest.approx(0.196)


def test_qrd_depths_span_zero_to_half_wavelength() -> None:
    depths = qrd_well_depths(7, 500.0, speed_of_sound=343.0)
    lambda0 = 343.0 / 500.0
    assert float(depths.min()) == pytest.approx(0.0)
    assert float(depths.max()) < lambda0 / 2.0


# --- Exact-by-physics: flat panel is specular and self-normalises to zero -----
def test_flat_panel_normalizes_to_zero() -> None:
    spectrum = predicted_diffusion_spectrum(
        0.10, [500.0, 1000.0, 2000.0], depths=np.zeros(7), periods=5
    )
    assert spectrum.normalized is not None
    np.testing.assert_allclose(spectrum.normalized, 0.0, atol=1e-12)


def test_flat_panel_raw_diffusion_is_low() -> None:
    # A flat rigid panel throws a single specular lobe: low autocorrelation d.
    surface = predict_diffuser_polar_response(
        0.10, 2000.0, depths=np.zeros(7), periods=5
    )
    assert surface.coefficient < 0.05


# --- QRD scatters more evenly than the flat reference (the core anchor) -------
def test_qrd_diffusion_exceeds_flat_panel() -> None:
    freqs = [500.0, 1000.0, 2000.0]
    depths = qrd_well_depths(7, 500.0)
    qrd = predicted_diffusion_spectrum(0.10, freqs, depths=depths, periods=5)
    flat = predicted_diffusion_spectrum(0.10, freqs, depths=np.zeros(7), periods=5)
    # Markedly higher raw diffusion in every band, and clearly positive when
    # normalised against the flat reference.
    assert np.all(qrd.diffusion > flat.diffusion + 0.1)
    assert qrd.normalized is not None
    assert np.all(qrd.normalized > 0.05)


def test_qrd_normalized_diffusion_2k_value() -> None:
    depths = qrd_well_depths(7, 500.0, speed_of_sound=343.0)
    spectrum = predicted_diffusion_spectrum(0.10, [2000.0], depths=depths, periods=5)
    assert spectrum.normalized is not None
    assert float(spectrum.normalized[0]) == pytest.approx(0.208, abs=1e-3)


# --- Kirchhoff obliquity factor ----------------------------------------------
@pytest.mark.parametrize("psi", [0.0, 30.0, 60.0])
def test_obliquity_factor_is_symmetric_kirchhoff(psi: float) -> None:
    """The obliquity factor is the symmetric Kirchhoff (cos theta + cos psi)/2.

    Cox & D'Antonio Eq. (9.32) derive the factor for a normal-incidence
    source, (1 + cos theta) / 2; under the Kirchhoff approximation an oblique
    source at psi carries the symmetric (cos theta + cos psi) / 2, which
    reduces to Eq. (9.32) exactly at psi = 0 (so all normal-incidence anchors
    are untouched).
    """
    from phonometry.materials.diffusers.design import _scattered_pressure

    depths = qrd_well_depths(7, 500.0)
    freq, c = 1000.0, 343.0
    k = 2.0 * np.pi * freq / c
    reflection = np.exp(-2j * k * depths)
    angles = np.asarray(DEFAULT_POLAR_ANGLES, dtype=np.float64)
    common = {
        "source_angle": psi,
        "periods": 3,
        "speed_of_sound": c,
        "include_aperture": True,
    }
    with_factor = _scattered_pressure(
        reflection, 0.10, freq, angles, include_obliquity=True, **common
    )
    without = _scattered_pressure(
        reflection, 0.10, freq, angles, include_obliquity=False, **common
    )
    expected = (np.cos(np.radians(angles)) + np.cos(np.radians(psi))) / 2.0
    np.testing.assert_allclose(with_factor, without * expected)


# --- The explicit reflection path matches the rigid-bottom depths path --------
def test_reflection_path_matches_depths_path() -> None:
    depths = qrd_well_depths(7, 500.0)
    freq, c = 1500.0, 343.0
    k = 2.0 * np.pi * freq / c
    reflection = np.exp(-2j * k * depths)
    from_depths = predict_diffuser_polar_response(
        0.10, freq, depths=depths, periods=4, speed_of_sound=c
    )
    from_reflection = predict_diffuser_polar_response(
        0.10, freq, reflection=reflection, periods=4, speed_of_sound=c
    )
    assert from_reflection.coefficient == pytest.approx(from_depths.coefficient)
    np.testing.assert_allclose(from_reflection.levels, from_depths.levels)


def test_polar_levels_peak_referenced_to_zero() -> None:
    surface = predict_diffuser_polar_response(
        0.10, 1000.0, depths=qrd_well_depths(7, 500.0), periods=5
    )
    assert float(surface.levels.max()) == pytest.approx(0.0)
    assert surface.angles.shape == surface.levels.shape


def test_default_polar_angles_are_iso_semicircle() -> None:
    assert DEFAULT_POLAR_ANGLES[0] == -90
    assert DEFAULT_POLAR_ANGLES[-1] == 90
    assert len(DEFAULT_POLAR_ANGLES) == 37


def test_predict_returns_result_dataclass() -> None:
    surface = predict_diffuser_polar_response(
        0.10, 1000.0, depths=qrd_well_depths(7, 500.0)
    )
    assert isinstance(surface, DiffuserPolarResponse)
    assert surface.frequency == pytest.approx(1000.0)


def test_spectrum_returns_diffusion_spectrum() -> None:
    spectrum = predicted_diffusion_spectrum(
        0.10, [500.0, 1000.0], depths=qrd_well_depths(7, 500.0)
    )
    assert isinstance(spectrum, DiffusionSpectrum)
    np.testing.assert_allclose(spectrum.frequencies, [500.0, 1000.0])


def test_spectrum_without_normalization() -> None:
    spectrum = predicted_diffusion_spectrum(
        0.10, [500.0, 1000.0], depths=qrd_well_depths(7, 500.0), normalize=False
    )
    assert spectrum.normalized is None


# --- Plot smoke test ----------------------------------------------------------
def test_polar_response_plot_returns_polar_axes() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    surface = predict_diffuser_polar_response(
        0.10, 1000.0, depths=qrd_well_depths(7, 500.0), periods=5
    )
    ax = surface.plot()
    assert ax.name == "polar"
    ax_es = surface.plot(language="es")
    assert ax_es.name == "polar"
    plt.close("all")


# --- Input-validation guards --------------------------------------------------
def test_prime_must_be_odd_prime() -> None:
    with pytest.raises(ValueError, match="odd prime"):
        quadratic_residue_sequence(8)
    with pytest.raises(ValueError, match="prime"):
        quadratic_residue_sequence(9)


def test_requires_exactly_one_of_depths_or_reflection() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        predict_diffuser_polar_response(0.10, 1000.0)
    depths = np.zeros(7)
    reflection = np.ones(7)
    with pytest.raises(ValueError, match="exactly one"):
        predict_diffuser_polar_response(
            0.10, 1000.0, depths=depths, reflection=reflection
        )


def test_depths_need_at_least_two_wells() -> None:
    with pytest.raises(ValueError, match="at least two"):
        predict_diffuser_polar_response(0.10, 1000.0, depths=[0.05])


def test_negative_depth_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        predict_diffuser_polar_response(0.10, 1000.0, depths=[0.05, -0.01, 0.02])


def test_non_positive_geometry_rejected() -> None:
    depths = qrd_well_depths(7, 500.0)
    with pytest.raises(ValueError, match="well_width"):
        predict_diffuser_polar_response(0.0, 1000.0, depths=depths)
    with pytest.raises(ValueError, match="frequency"):
        predict_diffuser_polar_response(0.10, 0.0, depths=depths)
    with pytest.raises(ValueError, match="periods"):
        predict_diffuser_polar_response(0.10, 1000.0, depths=depths, periods=0)


def test_spectrum_requires_depths() -> None:
    """Keyword-only and required: the signature says so on its own now."""
    with pytest.raises(TypeError, match="required keyword-only argument"):
        predicted_diffusion_spectrum(0.10, [500.0])  # type: ignore[call-arg]


def test_spectrum_reserved_reflection_argument() -> None:
    depths = np.zeros(7)
    reserved = object()
    with pytest.raises(ValueError, match="reflection_of"):
        predicted_diffusion_spectrum(
            0.10, [500.0], depths=depths, reflection_of=reserved
        )
