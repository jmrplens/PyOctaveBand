#  Copyright (c) 2026. Jose Manuel Requena Plens
"""p-p sound intensity (IEC 61043:1993) and ISO 9614-1:1993 field indicators.

Physics anchors:
- Plane progressive wave: I = p_rms^2 / (rho*c), so Lp - LI =
  10*lg(rho*c/400) = 0,14 dB (IEC 61043:1993 clause 5 note).
- Finite-difference estimator bias sin(k*dr)/(k*dr) (IEC 61043:1993, 7.3;
  Table 3 nominal -10,5 dB at 6,3 kHz for 25 mm separation).
- Field indicators from ISO 9614-1:1993 Annex A, equations (A.3)-(A.9).
"""

import numpy as np
import pytest

from phonometry import emission

FS = 48000
SPACING = 0.012  # 12 mm microphone separation
RHO = 1.204
C = 343.0


def _plane_wave_pair(
    delay_s: float,
    f_lo: float = 50.0,
    f_hi: float = 2000.0,
    seconds: float = 10.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Band-limited noise observed at two points; the second is a pure
    (fractional, circular) delay of the first: p2(t) = p1(t - delay).
    """
    rng = np.random.default_rng(seed)
    n = int(FS * seconds)
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    spec = np.zeros(freqs.size, dtype=complex)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    spec[band] = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, int(band.sum())))
    p1 = np.fft.irfft(spec, n)
    p2 = np.fft.irfft(spec * np.exp(-2j * np.pi * freqs * delay_s), n)
    scale = 1.0 / np.sqrt(np.mean(p1**2))  # 1 Pa rms (94 dB)
    return p1 * scale, p2 * scale


def test_plane_progressive_wave_broadband() -> None:
    """I = p_rms^2/(rho*c) within 3 % in the valid band, F2 = 0,14 dB."""
    p1, p2 = _plane_wave_pair(delay_s=SPACING / C)
    res = emission.sound_intensity(p1, p2, FS, spacing=SPACING, rho=RHO, c=C)
    p_center = (p1 + p2) / 2.0
    expected = float(np.mean(p_center**2)) / (RHO * C)
    # Measured residual is ~0.5 % (second-order finite-difference/Welch term);
    # 0.015 keeps ~3x headroom (was 0.03).
    assert res.total_intensity == pytest.approx(expected, rel=0.015)
    assert res.total_direction == 1
    # IEC 61043:1993 clause 5: Lp - LI = 10*lg(rho*c/400) = 0,14 dB.
    assert res.total_pressure_intensity_index == pytest.approx(
        10 * np.log10(RHO * C / 400.0), abs=0.35
    )
    assert res.total_pressure_level == pytest.approx(94.0, abs=0.2)
    # The whole excitation band lies below the usable-bandwidth bound.
    assert res.max_valid_frequency == pytest.approx(0.1 * C / SPACING)
    assert res.max_valid_frequency > 2000.0


def test_bias_correct_undoes_finite_difference_underread() -> None:
    """bias_correct=True lifts each in-band bin by (k*dr)/sin(k*dr)
    (IEC 61043:1993, 7.3). For a single tone the corrected total therefore
    equals the raw total times that factor; at low frequency it is ~1.
    """
    spacing, f0 = 0.05, 500.0
    t = np.arange(int(FS * 2)) / FS
    delay = spacing / C
    p1 = np.sin(2 * np.pi * f0 * t)
    p2 = np.sin(2 * np.pi * f0 * (t - delay))
    raw = emission.sound_intensity(p1, p2, FS, spacing=spacing, rho=RHO, c=C)
    corr = emission.sound_intensity(
        p1, p2, FS, spacing=spacing, rho=RHO, c=C, bias_correct=True
    )
    k_dr = 2.0 * np.pi * f0 * spacing / C
    expected_ratio = k_dr / np.sin(k_dr)
    assert corr.total_intensity / raw.total_intensity == pytest.approx(
        expected_ratio, rel=1e-3
    )
    # At 50 Hz the correction is negligible (< 0.1 %).
    lo1 = np.sin(2 * np.pi * 50.0 * t)
    lo2 = np.sin(2 * np.pi * 50.0 * (t - delay))
    r = emission.sound_intensity(lo1, lo2, FS, spacing=spacing, rho=RHO, c=C)
    cc = emission.sound_intensity(
        lo1, lo2, FS, spacing=spacing, rho=RHO, c=C, bias_correct=True
    )
    assert cc.total_intensity == pytest.approx(r.total_intensity, rel=2e-3)


def test_bias_correct_near_null_does_not_explode() -> None:
    """A tone close to the first finite-difference null (c/(2*dr)) has a raw
    intensity that is almost nulled; the un-clamped reciprocal
    (k*dr)/sin(k*dr) would blow up by ~30x there and let a few near-null bins
    dominate the total. Clamping the correction at k*dr = pi/2 keeps the
    corrected total finite and bounded by the constant pi/2 factor.
    """
    spacing = SPACING  # 12 mm -> first null at 343/0.024 = 14.3 kHz
    f0 = 14000.0  # k*dr ~ 0.98*pi, deep in the near-null region
    t = np.arange(int(FS * 2)) / FS
    delay = spacing / C
    p1 = np.sin(2 * np.pi * f0 * t)
    p2 = np.sin(2 * np.pi * f0 * (t - delay))
    raw = emission.sound_intensity(p1, p2, FS, spacing=spacing, rho=RHO, c=C)
    corr = emission.sound_intensity(
        p1, p2, FS, spacing=spacing, rho=RHO, c=C, bias_correct=True
    )
    k_dr = 2.0 * np.pi * f0 * spacing / C
    assert k_dr > np.pi / 2.0  # the tone is past the cutoff
    # The un-clamped correction factor at this bin is huge (documented blow-up).
    assert k_dr / np.sin(k_dr) > 30.0
    assert np.isfinite(corr.total_intensity)
    # Every contributing bin sits above the cutoff, so all are held at the
    # constant pi/2 factor: the corrected total is bounded, not amplified ~30x.
    ratio = corr.total_intensity / raw.total_intensity
    assert ratio == pytest.approx(np.pi / 2.0, rel=0.02)
    assert abs(ratio) <= np.pi / 2.0 * 1.001


def test_bias_correct_below_cutoff_matches_analytic() -> None:
    """Below the pi/2 cutoff the clamped correction equals the exact
    reciprocal, so bias_correct recovers the unbiased plane-wave intensity
    A^2/(2*rho*c) (behaviour unchanged from the un-clamped correction).
    """
    spacing, f0 = SPACING, 4000.0  # k*dr = 0.88 rad < pi/2
    t = np.arange(int(FS * 5.0)) / FS
    amp = np.sqrt(2.0)  # 1 Pa rms
    phi = 2.0 * np.pi * f0 * spacing / C
    assert phi < np.pi / 2.0
    p1 = amp * np.cos(2.0 * np.pi * f0 * t)
    p2 = amp * np.cos(2.0 * np.pi * f0 * t - phi)
    corr = emission.sound_intensity(
        p1, p2, FS, spacing=spacing, rho=RHO, c=C, bias_correct=True
    )
    true_plane = amp**2 / (2.0 * RHO * C)
    assert corr.total_intensity == pytest.approx(true_plane, rel=0.02)


def test_reversing_microphones_flips_the_sign() -> None:
    p1, p2 = _plane_wave_pair(delay_s=SPACING / C)
    fwd = emission.sound_intensity(p1, p2, FS, spacing=SPACING, rho=RHO, c=C)
    rev = emission.sound_intensity(p2, p1, FS, spacing=SPACING, rho=RHO, c=C)
    assert rev.total_direction == -1
    assert rev.total_intensity == pytest.approx(-fwd.total_intensity, rel=1e-6)
    assert rev.total_intensity_level == pytest.approx(
        fwd.total_intensity_level, abs=1e-6
    )


def test_plane_wave_third_octave_bands() -> None:
    """Per-band F2 = 0 dB and positive direction inside the excited band."""
    p1, p2 = _plane_wave_pair(delay_s=SPACING / C)
    res = emission.sound_intensity(
        p1, p2, FS, spacing=SPACING, rho=RHO, c=C, fraction=3
    )
    assert res.frequency is not None
    assert res.intensity is not None
    assert res.direction is not None
    assert res.pressure_intensity_index is not None
    assert res.bias_correction is not None
    active = (res.frequency >= 80.0) & (res.frequency <= 1600.0)
    assert np.any(active)
    assert np.all(res.direction[active] == 1)
    # Free-field: per-band pressure-intensity index stays near 0,14 dB.
    assert np.all(np.abs(res.pressure_intensity_index[active]) < 1.0)
    # Correction factor grows monotonically with frequency, >= 1 in-band.
    assert np.all(res.bias_correction[active] >= 1.0)
    assert res.bias_correction[active][-1] > res.bias_correction[active][0]


def test_standing_wave_high_pressure_intensity_index() -> None:
    """Two equal opposing waves: |I| near zero while Lp is high."""
    t = np.arange(int(FS * 5.0)) / FS
    f0 = 500.0
    k = 2.0 * np.pi * f0 / C
    x1, x2 = 0.10, 0.10 + SPACING
    p1 = 2.0 * np.cos(k * x1) * np.cos(2.0 * np.pi * f0 * t)
    p2 = 2.0 * np.cos(k * x2) * np.cos(2.0 * np.pi * f0 * t)
    res = emission.sound_intensity(p1, p2, FS, spacing=SPACING, rho=RHO, c=C)
    p_center = (p1 + p2) / 2.0
    plane_equivalent = float(np.mean(p_center**2)) / (RHO * C)
    assert abs(res.total_intensity) < 1e-3 * plane_equivalent
    assert res.total_pressure_level > 90.0
    assert res.total_pressure_intensity_index > 20.0


def test_1khz_tone_exact_analytic_intensity() -> None:
    """Pure tone with exact phase lag k*dr: the cross-spectral estimator
    must return (A^2/(2*rho*c)) * sin(k*dr)/(k*dr) exactly.
    """
    t = np.arange(int(FS * 5.0)) / FS
    f0 = 1000.0
    amp = np.sqrt(2.0)  # 1 Pa rms
    phi = 2.0 * np.pi * f0 * SPACING / C  # k*dr
    p1 = amp * np.cos(2.0 * np.pi * f0 * t)
    p2 = amp * np.cos(2.0 * np.pi * f0 * t - phi)
    res = emission.sound_intensity(
        p1, p2, FS, spacing=SPACING, rho=RHO, c=C, fraction=3
    )
    true_plane = amp**2 / (2.0 * RHO * C)
    expected = true_plane * np.sin(phi) / phi
    assert res.total_intensity == pytest.approx(expected, rel=0.01)
    assert res.total_intensity_level == pytest.approx(
        10 * np.log10(expected / 1e-12), abs=0.05
    )
    # All the power falls in the 1 kHz third-octave band.
    assert res.frequency is not None
    assert res.intensity is not None
    assert res.bias_correction is not None
    idx = int(np.argmin(np.abs(res.frequency - 1000.0)))
    assert res.intensity[idx] == pytest.approx(expected, rel=0.01)
    # Applying the documented sin(k*dr)/(k*dr) correction recovers the
    # unbiased plane-wave intensity (IEC 61043:1993, 7.3).
    corrected = res.intensity[idx] * res.bias_correction[idx]
    assert corrected == pytest.approx(true_plane, rel=0.01)


def test_band_integration_consistency() -> None:
    """Sum of band intensities and pressures matches the broadband totals."""
    p1, p2 = _plane_wave_pair(delay_s=SPACING / C, f_lo=100.0, f_hi=4000.0)
    res = emission.sound_intensity(
        p1, p2, FS, spacing=SPACING, rho=RHO, c=C, fraction=3
    )
    assert res.intensity is not None
    assert res.pressure_level is not None
    assert float(np.sum(res.intensity)) == pytest.approx(res.total_intensity, rel=0.01)
    band_lp_sum = 10 * np.log10(np.sum(10 ** (0.1 * res.pressure_level)))
    assert band_lp_sum == pytest.approx(res.total_pressure_level, abs=0.05)


def test_octave_fraction_and_limits() -> None:
    p1, p2 = _plane_wave_pair(delay_s=SPACING / C)
    res = emission.sound_intensity(
        p1, p2, FS, spacing=SPACING, rho=RHO, c=C, fraction=1, limits=[63.0, 4000.0]
    )
    assert isinstance(res, emission.IntensityResult)
    assert res.frequency is not None
    assert res.frequency[0] >= 63.0 / np.sqrt(2.0)
    assert res.frequency[-1] <= 4000.0 * np.sqrt(2.0)


def test_validation_errors() -> None:
    good = np.random.default_rng(0).standard_normal(FS)
    with pytest.raises(ValueError, match="same length"):
        emission.sound_intensity(good, good[:-1], FS, spacing=SPACING)
    with pytest.raises(ValueError, match="spacing"):
        emission.sound_intensity(good, good, FS, spacing=0.0)
    with pytest.raises(ValueError, match="spacing"):
        emission.sound_intensity(good, good, FS, spacing=-0.01)
    with pytest.raises(ValueError, match="fs"):
        emission.sound_intensity(good, good, 0, spacing=SPACING)
    with pytest.raises(ValueError, match="fs"):
        emission.sound_intensity(good, good, -48000, spacing=SPACING)
    with pytest.raises(ValueError, match="rho"):
        emission.sound_intensity(good, good, FS, spacing=SPACING, rho=0.0)
    with pytest.raises(ValueError, match="'c'"):
        emission.sound_intensity(good, good, FS, spacing=SPACING, c=-1.0)
    with pytest.raises(ValueError, match="fraction"):
        emission.sound_intensity(good, good, FS, spacing=SPACING, fraction=2)
    with pytest.raises(ValueError, match="limits"):
        emission.sound_intensity(good, good, FS, spacing=SPACING, limits=[100.0])
    with pytest.raises(ValueError, match="limits"):
        emission.sound_intensity(
            good, good, FS, spacing=SPACING, limits=[1000.0, 100.0]
        )
    two_dimensional = np.zeros((2, 100))
    with pytest.raises(ValueError, match="1D"):
        emission.sound_intensity(two_dimensional, two_dimensional, FS, spacing=SPACING)
    with pytest.raises(ValueError, match="too short"):
        emission.sound_intensity(good[:8], good[:8], FS, spacing=SPACING)


def test_field_indicators_uniform_field() -> None:
    """Uniform positive intensity: F4 = 0 and F2 = F3; a plane-wave-like
    surface gives F2 = 10*lg(rho*c/400) = 0,14 dB.
    """
    i_n = np.full(8, 1.0 / (RHO * C))  # plane-wave intensity for 1 Pa^2
    lp = np.full(8, 93.98)  # 1 Pa^2 mean-square pressure
    ind = emission.field_indicators(lp, i_n)
    assert isinstance(ind, emission.FieldIndicators)
    assert ind.f4 == pytest.approx(0.0, abs=1e-12)
    assert ind.f2 == pytest.approx(ind.f3, abs=1e-12)
    assert ind.f2 == pytest.approx(10 * np.log10(RHO * C / 400.0), abs=0.02)


def test_field_indicators_negative_partial_power() -> None:
    """A negative-intensity segment raises F3 above F2 (A.6 vs A.3)."""
    i_n = np.array([2.0e-3, 1.5e-3, 1.0e-3, -0.5e-3])
    lp = np.full(4, 90.0)
    ind = emission.field_indicators(lp, i_n)
    assert ind.f3 > ind.f2
    assert ind.f4 > 0.0
    # Hand-computed anchors: mean|In| = 1,25e-3, mean In = 1,0e-3.
    lp_surf = 90.0
    assert ind.f2 == pytest.approx(lp_surf - 10 * np.log10(1.25e-3 / 1e-12), abs=1e-9)
    assert ind.f3 == pytest.approx(lp_surf - 10 * np.log10(1.0e-3 / 1e-12), abs=1e-9)


def test_field_indicators_validation() -> None:
    with pytest.raises(ValueError, match="same shape"):
        emission.field_indicators([90.0, 91.0], [1e-3])
    with pytest.raises(ValueError, match="two measurement positions"):
        emission.field_indicators([90.0], [1e-3])
    with pytest.raises(ValueError, match="not positive"):
        emission.field_indicators([90.0, 90.0], [1e-3, -2e-3])
    three_band_lp = np.full((4, 3), 90.0)
    three_band_in = np.full((4, 3), 1e-3)
    with pytest.raises(ValueError, match="one entry per band"):
        emission.field_indicators(three_band_lp, three_band_in, [125.0, 250.0])


def test_field_indicators_per_band_matches_per_column_scalars() -> None:
    """2D (positions, bands) input returns per-band arrays, one indicator
    triple per column, identical to the scalar call on that column.
    """
    rng = np.random.default_rng(9614)
    freqs = np.array([125.0, 250.0, 500.0, 1000.0])
    lp = 74.0 + rng.normal(0.0, 0.8, (8, 4))
    i_n = 1.0e-5 * (1.0 + rng.normal(0.0, 0.2, (8, 4)))
    ind = emission.field_indicators(lp, i_n, freqs)
    assert isinstance(ind.f2, np.ndarray)
    assert ind.f2.shape == (4,)
    np.testing.assert_allclose(ind.frequency, freqs)
    for b in range(4):
        one = emission.field_indicators(lp[:, b], i_n[:, b])
        assert ind.f2[b] == pytest.approx(one.f2)
        assert ind.f3[b] == pytest.approx(one.f3)
        assert ind.f4[b] == pytest.approx(one.f4)
    # A single band whose algebraic mean intensity is non-positive fails the
    # ISO 9614-1 A.2.3 test conditions and raises, exactly as the scalar path.
    bad = i_n.copy()
    bad[:, 1] = -1.0e-5
    with pytest.raises(ValueError, match="not positive"):
        emission.field_indicators(lp, bad, freqs)


def test_field_indicators_plot_draws_indicators_and_ld() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    freqs = np.array([250.0, 500.0, 1000.0, 2000.0])
    lp = 76.0 + rng.normal(0.0, 0.5, (10, 4))
    i_n = 2.0e-5 * (1.0 + rng.normal(0.0, 0.25, (10, 4)))
    ind = emission.field_indicators(lp, i_n, freqs)
    ld = emission.dynamic_capability_index(18.0)
    ax = ind.plot(dynamic_capability=ld)
    # F2 curve, F3 curve and the Ld criterion line on the main axes.
    ydata = [np.asarray(line.get_ydata(), dtype=float) for line in ax.lines]
    assert any(np.allclose(y, np.asarray(ind.f2)) for y in ydata)
    assert any(np.allclose(y, np.asarray(ind.f3)) for y in ydata)
    assert any(np.allclose(y, ld) for y in ydata)
    # F4 rides the twin axis as bars (one patch per band).
    twin = [a for a in ax.figure.axes if a is not ax]
    assert twin
    assert len(twin[0].patches) == freqs.size
    assert ax.get_title() == "ISO 9614-1 field indicators"
    plt.close("all")
    ax_es = ind.plot(language="es")
    assert ax_es.get_ylabel() == "Indicador [dB]"
    plt.close("all")
    with pytest.raises(ValueError, match="Unknown language"):
        ind.plot(language="xx")
    # The scalar (single-band) result has nothing per band to draw.
    single = emission.field_indicators(lp[:, 0], i_n[:, 0])
    with pytest.raises(ValueError, match="per-band"):
        single.plot()
    plt.close("all")


def test_dynamic_capability_index() -> None:
    """Ld = delta_pI0 - K (ISO 9614-1 equation (10)); adequate when
    Ld > F2 (criterion 1, equation (B.1)).
    """
    assert emission.dynamic_capability_index(18.0) == pytest.approx(8.0)
    assert emission.dynamic_capability_index(
        18.0, bias_error_factor=7.0
    ) == pytest.approx(11.0)
    with pytest.raises(ValueError, match="bias_error_factor"):
        emission.dynamic_capability_index(18.0, bias_error_factor=0.0)


# ---------------------------------------------------------------------------
# ISO 9614-1:1993 F1, the temporal variability indicator (equation (A.1))
#
# ISO 9614-1 publishes no numeric worked example for F1 to F4 (the real cases
# of the literature present them only as figures), so F1 is anchored on the
# closed form of equation (A.1) itself: the coefficient of variation of the M
# short-time samples of the normal intensity at one fixed position, i.e. the
# sample standard deviation (N - 1 denominator, equation (A.1)) over the
# algebraic mean (equation (A.2)). The Table B.3 threshold of 0,6 is the
# standard's own number.
# ---------------------------------------------------------------------------


def test_f1_is_zero_for_a_perfectly_steady_field() -> None:
    """A constant short-time intensity has no temporal variability."""
    assert emission.temporal_variability_indicator(
        np.full(10, 3.4e-5)
    ) == pytest.approx(0.0)


def test_f1_two_sample_closed_form() -> None:
    """For M = 2, equation (A.1) reduces to sqrt(2)*|b - a| / (a + b)."""
    a, b = 1.0e-5, 3.0e-5
    expected = np.sqrt(2.0) * abs(b - a) / (a + b)
    assert emission.temporal_variability_indicator([a, b]) == pytest.approx(expected)


def test_f1_is_the_coefficient_of_variation() -> None:
    """Equation (A.1) is the sample std (ddof = 1) over the mean of (A.2)."""
    rng = np.random.default_rng(1993)
    samples = 5.0e-5 * (1.0 + rng.normal(0.0, 0.3, 10))
    expected = float(np.std(samples, ddof=1) / np.mean(samples))
    assert emission.temporal_variability_indicator(samples) == pytest.approx(expected)


def test_f1_is_scale_invariant() -> None:
    """F1 is dimensionless: a gain change on every sample leaves it unchanged."""
    samples = np.array([1.0e-5, 1.4e-5, 0.8e-5, 1.1e-5, 1.3e-5])
    assert emission.temporal_variability_indicator(samples * 1000.0) == pytest.approx(
        emission.temporal_variability_indicator(samples)
    )


def test_f1_and_f4_share_the_closed_form() -> None:
    """(A.1) over M samples and (A.8) over N positions are the same statistic."""
    values = np.array([2.0e-5, 1.6e-5, 2.4e-5, 1.9e-5, 2.2e-5, 2.1e-5])
    surface = emission.field_indicators(np.full(values.size, 80.0), values)
    assert emission.temporal_variability_indicator(values) == pytest.approx(surface.f4)


def test_f1_per_band_matches_the_per_column_scalars() -> None:
    """2D (samples, bands) input evaluates every band of Annex A.1 at once."""
    rng = np.random.default_rng(614)
    samples = 3.0e-5 * (1.0 + rng.normal(0.0, 0.25, (10, 4)))
    per_band = emission.temporal_variability_indicator(samples)
    assert isinstance(per_band, np.ndarray)
    assert per_band.shape == (4,)
    for b in range(4):
        assert per_band[b] == pytest.approx(
            emission.temporal_variability_indicator(samples[:, b])
        )


def test_f1_validation() -> None:
    with pytest.raises(ValueError, match="two short-time samples"):
        emission.temporal_variability_indicator([1.0e-5])
    three_dimensional = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="1D .samples,. or 2D"):
        emission.temporal_variability_indicator(three_dimensional)
    with pytest.raises(ValueError, match="not positive"):
        emission.temporal_variability_indicator([1.0e-5, -3.0e-5])
    # A NaN or infinite sample would otherwise slip past the positivity test
    # and turn the indicator into a silent NaN.
    with_nan = [1.0e-5, float("nan"), 1.2e-5]
    with pytest.raises(ValueError, match="finite"):
        emission.temporal_variability_indicator(with_nan)
    with_inf = [1.0e-5, float("inf"), 1.2e-5]
    with pytest.raises(ValueError, match="finite"):
        emission.temporal_variability_indicator(with_inf)


def test_non_positive_mean_cites_a23_only_for_the_surface_indicators() -> None:
    """F4 rejects a non-positive mean *because* the standard says so; F1 does not.

    ISO 9614-1:1993 A.2.3 ends with "Si sum(Ini/I0) es negativo en alguna banda
    de frecuencia, las condiciones del ensayo no satisfacen los requerimientos
    de esta parte de la Norma ISO 9614 en esa banda", so the surface-scan
    indicators may attribute the rejection to the standard. A.2.1, which
    defines F1 over the M short-time samples at one position, states no such
    condition, so the F1 message must not claim A.2.3 backing.
    """
    with pytest.raises(
        ValueError, match="measurement surface is not positive"
    ) as surface:
        emission.field_indicators([90.0, 90.0], [1e-3, -2e-3])
    assert "A.2.3" in str(surface.value)

    with pytest.raises(
        ValueError, match="short-time normal intensity samples is not positive"
    ) as temporal:
        emission.temporal_variability_indicator([1.0e-5, -3.0e-5])
    assert "A.2.3" not in str(temporal.value)
    assert "A.1" in str(temporal.value)


def test_f1_follows_the_shape_of_its_sibling_indicators() -> None:
    """F1 is scalar where F2/F3/F4 are scalar, and an array where they are.

    A 1D per-position surface yields scalar indicators, so a one-column
    ``(M, 1)`` sample array must not leave ``f1`` as ``array([x])``: the
    Table B.3 check would then answer ``array([True])`` where its siblings are
    plain floats. A 2D one-band surface keeps per-band arrays throughout.
    """
    samples_1d = np.array([1.0e-5, 1.2e-5, 0.9e-5, 1.1e-5, 1.0e-5])
    samples_2d = samples_1d.reshape(-1, 1)

    scalar = emission.field_indicators(
        np.full(4, 80.0), np.full(4, 1.0e-5), temporal_intensity=samples_2d
    )
    assert np.ndim(scalar.f1) == 0
    assert np.ndim(scalar.f2) == 0
    assert scalar.field_is_stationary() is True

    per_band = emission.field_indicators(
        np.full((4, 1), 80.0),
        np.full((4, 1), 1.0e-5),
        [1000.0],
        temporal_intensity=samples_1d,
    )
    assert np.ndim(per_band.f1) == 1
    assert np.size(per_band.f1) == 1
    assert np.ndim(per_band.f4) == 1
    assert np.asarray(per_band.field_is_stationary()).shape == (1,)


def test_field_indicators_carries_f1_when_given_the_samples() -> None:
    """The optional temporal samples fill f1 alongside F2/F3/F4."""
    lp = np.full(6, 82.0)
    i_n = np.full(6, 1.5e-5)
    samples = np.array([1.4e-5, 1.6e-5, 1.5e-5, 1.7e-5, 1.3e-5, 1.5e-5])
    ind = emission.field_indicators(lp, i_n, temporal_intensity=samples)
    assert ind.f1 == pytest.approx(emission.temporal_variability_indicator(samples))
    assert emission.field_indicators(lp, i_n).f1 is None


def test_field_indicators_f1_column_count_must_match_the_bands() -> None:
    lp = np.full((5, 3), 82.0)
    i_n = np.full((5, 3), 1.5e-5)
    two_band_samples = np.full((10, 2), 1.5e-5)
    with pytest.raises(ValueError, match="one column per band"):
        emission.field_indicators(lp, i_n, temporal_intensity=two_band_samples)


def test_field_is_stationary_against_the_table_b3_limit() -> None:
    """ISO 9614-1 Table B.3 calls for action when F1 exceeds 0,6."""
    assert emission.TEMPORAL_VARIABILITY_LIMIT == 0.6
    lp = np.full(4, 80.0)
    i_n = np.full(4, 1.0e-5)
    steady = emission.field_indicators(lp, i_n, temporal_intensity=np.full(10, 1.0e-5))
    assert steady.field_is_stationary() is True
    # A pair chosen so equation (A.1) lands exactly on the 0,6 threshold, and
    # a wilder pair above it: sqrt(2)*(b - a)/(a + b) with a = 1, b = r.
    r = (np.sqrt(2.0) + 0.6) / (np.sqrt(2.0) - 0.6)
    on_limit = emission.field_indicators(
        lp, i_n, temporal_intensity=[1.0e-5, r * 1.0e-5]
    )
    assert float(np.asarray(on_limit.f1)) == pytest.approx(0.6)
    assert on_limit.field_is_stationary() is True  # the limit itself passes
    varying = emission.field_indicators(lp, i_n, temporal_intensity=[1.0e-5, 5.0e-5])
    assert varying.field_is_stationary() is False
    without_f1 = emission.field_indicators(lp, i_n)
    with pytest.raises(ValueError, match="no F1"):
        without_f1.field_is_stationary()


def test_field_indicators_plot_draws_f1_and_its_limit() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(43)
    freqs = np.array([250.0, 500.0, 1000.0, 2000.0])
    lp = 76.0 + rng.normal(0.0, 0.5, (10, 4))
    i_n = 2.0e-5 * (1.0 + rng.normal(0.0, 0.25, (10, 4)))
    samples = 2.0e-5 * (1.0 + rng.normal(0.0, 0.15, (10, 4)))
    ind = emission.field_indicators(lp, i_n, freqs, temporal_intensity=samples)
    ax = ind.plot()
    twin = next(a for a in ax.figure.axes if a is not ax)
    ydata = [np.asarray(line.get_ydata(), dtype=float) for line in twin.lines]
    assert any(np.allclose(y, np.asarray(ind.f1)) for y in ydata)
    assert any(np.allclose(y, emission.TEMPORAL_VARIABILITY_LIMIT) for y in ydata)
    plt.close("all")


# ---------------------------------------------------------------------------
# Per-band arrays that do not span the same bands
# ---------------------------------------------------------------------------


def _per_band_indicators() -> emission.FieldIndicators:
    """Four-band indicators with F1 measured from per-band temporal samples."""
    rng = np.random.default_rng(43)
    freqs = np.array([250.0, 500.0, 1000.0, 2000.0])
    lp = 76.0 + rng.normal(0.0, 0.5, (10, 4))
    i_n = 2.0e-5 * (1.0 + rng.normal(0.0, 0.25, (10, 4)))
    samples = 2.0e-5 * (1.0 + rng.normal(0.0, 0.6, (10, 4)))
    return emission.field_indicators(lp, i_n, freqs, temporal_intensity=samples)


@pytest.mark.parametrize("kind", ["scalar", "short", "long"])
def test_an_f1_off_the_band_axis_is_refused(kind: str) -> None:
    """F1 is pinned to the bands its siblings describe, at construction.

    Clause 8.2 measures F1 at one typical position, and
    :func:`temporal_variability_indicator` hands that single number back, so a
    scalar dropped in beside per-band F2/F3/F4 is the mistake nearest to hand.
    Nothing downstream reads it as short: plot() broadcasts F1 onto the band
    axis, so the quietest band's 0,49 is drawn as a flat line under the
    Table B.3 limit across the whole spectrum, and field_is_stationary()
    answers a single True where these bands carry [1,19 0,92 0,49 0,55] and
    two of the four exceed the limit.
    """
    import dataclasses

    result = _per_band_indicators()
    f1 = np.asarray(result.f1, dtype=float)
    wrong = {
        "scalar": float(f1.min()),
        "short": f1[:-1],
        "long": np.append(f1, f1[-1]),
    }[kind]
    with pytest.raises(ValueError, match="'f1'"):
        dataclasses.replace(result, f1=wrong)


@pytest.mark.parametrize("field_name", ["intensity", "direction", "bias_correction"])
@pytest.mark.parametrize("trim", [True, False], ids=["short", "long"])
def test_an_intensity_band_quantity_off_the_band_axis_is_refused(
    field_name: str, trim: bool
) -> None:
    """The three per-band columns no reader touches are pinned as well.

    plot() draws Lp, LI and the pressure-intensity index against ``frequency``
    and stops on a length matplotlib cannot reconcile with the band centres,
    so those four columns are caught downstream one way or another. These
    three are read nowhere in the library: off the band axis they leave the
    figure identical to the pixel and travel intact into whatever the caller
    indexes against ``frequency``.
    """
    import dataclasses

    p1, p2 = _plane_wave_pair(delay_s=SPACING / C)
    result = emission.sound_intensity(
        p1, p2, FS, spacing=SPACING, rho=RHO, c=C, fraction=3
    )
    values = np.asarray(getattr(result, field_name))
    wrong = values[:-1] if trim else np.append(values, values[-1])
    with pytest.raises(ValueError, match=f"'{field_name}'"):
        dataclasses.replace(result, **{field_name: wrong})
