#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Sound power in a reverberation test room: ISO 3741:2010 (precision, grade 1).

Physics / normative anchors (ISO 3741:2010):
- Direct method, Eq. (20):
  LW = Lp(ST) + 10 lg(A/A0) + 4,34*(A/S) + 10 lg(1 + S*c/(8*V*f)) + C1 + C2 - 6.
- A = (55,26/c)*(V/T60); c = 20,05*sqrt(273 + theta) (Sabine, clause 9.1.4).
- Waterhouse boundary term 10 lg(1 + S*c/(8*V*f)) -> 0 as f -> infinity.
- Comparison method, Eq. (21): LW = LW(RSS) + (Lp(ST) - Lp(RSS) + C2).
"""

import warnings

import numpy as np
import pytest

from phonometry import emission, room

# Reference-quantity / radiation-impedance constants (ISO 3741 clause 9.1.4).
_PS0 = 101.325  # kPa
_THETA0 = 314.0  # K (reference-quantity correction C1)
_THETA1 = 296.0  # K (radiation-impedance correction C2)


def _c1(theta: float, ps: float) -> float:
    return -10.0 * np.log10(ps / _PS0) + 5.0 * np.log10((273.15 + theta) / _THETA0)


def _c2(theta: float, ps: float) -> float:
    return -10.0 * np.log10(ps / _PS0) + 15.0 * np.log10((273.15 + theta) / _THETA1)


def _bracket(t60, volume, surface, freq, theta, ps):
    """Independent re-implementation of the Eq. (20) bracket, for inversion."""
    c = 20.05 * np.sqrt(273.0 + theta)
    a = (55.26 / c) * (volume / np.asarray(t60, dtype=float))
    waterhouse = 10.0 * np.log10(1.0 + surface * c / (8.0 * volume * np.asarray(freq)))
    return (
        10.0 * np.log10(a / 1.0)
        + 4.34 * (a / surface)
        + waterhouse
        + _c1(theta, ps)
        + _c2(theta, ps)
        - 6.0
    )


# --------------------------------------------------------------------------
# Direct method — exact inversion of Eq. (20)
# --------------------------------------------------------------------------
def test_direct_method_exact_inversion() -> None:
    """Generate Lp(ST) from a known LW with Eq. (20), recover LW digit-exact."""
    volume, surface = 200.0, 210.0
    freqs = np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0])
    t60 = np.array([2.0, 1.8, 1.5, 1.0, 0.6])
    theta, ps = 23.0, 101.325
    lw_target = np.array([80.0, 85.0, 90.0, 82.0, 75.0])

    # Invert: Lp(ST) that yields exactly lw_target (the bracket is Lp-free).
    lp = lw_target - _bracket(t60, volume, surface, freqs, theta, ps)

    res = emission.sound_power_reverberation(
        lp,
        t60,
        volume,
        surface,
        freqs,
        temperature=theta,
        static_pressure=ps,
    )
    assert isinstance(res, emission.ReverberationSoundPowerResult)
    assert np.allclose(res.sound_power_level, lw_target, atol=1e-9, rtol=0.0)


def test_direct_method_multi_position_energy_average() -> None:
    """Several microphone rows are energy-averaged (Eq. 16) before Eq. (20)."""
    volume, surface = 200.0, 210.0
    freqs = np.array([1000.0])
    t60 = np.array([1.5])
    # Two positions differing by 6 dB -> energy mean is between them.
    levels = np.array([[80.0], [86.0]])
    with warnings.catch_warnings():  # 2 positions trip the qualification advisories
        warnings.simplefilter("ignore", emission.SoundPowerWarning)
        res = emission.sound_power_reverberation(levels, t60, volume, surface, freqs)
    mean = 10.0 * np.log10((10**8.0 + 10**8.6) / 2.0)
    assert np.isclose(res.mean_pressure_level[0], mean)


# --------------------------------------------------------------------------
# Waterhouse boundary correction -> 0 at high frequency
# --------------------------------------------------------------------------
def test_waterhouse_term_vanishes_at_high_frequency() -> None:
    """10 lg(1 + S c /(8 V f)) decreases monotonically to ~0 as f grows."""
    volume, surface = 200.0, 210.0
    freqs = np.array([100.0, 1000.0, 10000.0, 1.0e7])
    t60 = np.full(freqs.shape, 1.5)
    lp = np.full(freqs.shape, 80.0)
    res = emission.sound_power_reverberation(lp, t60, volume, surface, freqs)
    w = res.waterhouse_correction
    assert np.all(np.diff(w) < 0.0)  # strictly decreasing with frequency
    assert w[0] > w[-1]
    assert w[-1] < 1e-3  # ~0 at 10 MHz
    assert w[2] < 0.1  # already small at 10 kHz


# --------------------------------------------------------------------------
# Comparison method — exact by construction (Eq. 21)
# --------------------------------------------------------------------------
def test_comparison_method_exact_by_construction() -> None:
    """LW = LW(RSS) + (Lp(ST) - Lp(RSS) + C2), recovered digit-exact."""
    theta, ps = 20.0, 100.0
    lw_ref = np.array([90.0, 92.0, 88.0])
    lp_rss = np.array([70.0, 71.0, 69.0])
    # Choose a target LW and back out Lp(ST).
    lw_target = np.array([85.0, 95.0, 80.0])
    lp_st = lw_target - lw_ref + lp_rss - _c2(theta, ps)

    res = emission.sound_power_comparison(
        lp_st,
        lp_rss,
        lw_ref,
        temperature=theta,
        static_pressure=ps,
    )
    assert isinstance(res, emission.ReverberationSoundPowerResult)
    assert res.method == "comparison"
    assert np.allclose(res.sound_power_level, lw_target, atol=1e-9, rtol=0.0)


def test_comparison_reference_conditions_c2_near_zero() -> None:
    """At theta=23 C, ps=101,325 kPa the C2 correction is ~0 (theta1=296 K)."""
    lw_ref = np.array([90.0])
    lp_rss = np.array([70.0])
    lp_st = np.array([80.0])
    res = emission.sound_power_comparison(lp_st, lp_rss, lw_ref)
    # LW ~ 90 + (80 - 70) = 100, plus a ~0,003 dB C2 term.
    assert np.isclose(res.sound_power_level[0], 100.0, atol=0.01)


def test_comparison_fewer_than_six_positions_warns() -> None:
    """2D per-position test-source levels below the 6-position minimum warn
    in the comparison method too (no V/S check -- no room geometry)."""
    levels = np.array([[80.0], [80.1], [79.9]])  # 3 positions, tight spread
    lp_rss = np.array([70.0])
    lw_ref = np.array([90.0])
    with pytest.warns(emission.SoundPowerWarning, match="microphone position"):
        emission.sound_power_comparison(levels, lp_rss, lw_ref)


def test_comparison_interposition_std_above_criterion_warns() -> None:
    """sM > 1,5 dB across the comparison test-source positions warns (Eq. 10)."""
    levels = np.array([[70.0], [80.0], [72.0], [78.0], [71.0], [79.0]])
    lp_rss = np.array([70.0])
    lw_ref = np.array([90.0])
    with pytest.warns(emission.SoundPowerWarning, match="standard deviation"):
        emission.sound_power_comparison(levels, lp_rss, lw_ref)


def test_comparison_1d_spectra_emit_no_sampling_warning() -> None:
    """Already-averaged 1D spectra carry no per-position info -> no advisory."""
    lp_st = np.array([80.0, 80.0])
    lp_rss = np.array([70.0, 70.0])
    lw_ref = np.array([90.0, 90.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", emission.SoundPowerWarning)
        emission.sound_power_comparison(lp_st, lp_rss, lw_ref)


# --------------------------------------------------------------------------
# Synergy — room_parameters T30 feeds the direct method
# --------------------------------------------------------------------------
def test_synergy_room_parameters_feeds_direct_method() -> None:
    """A T60 estimate from room_parameters composes into Eq. (20)."""
    fs = 48000
    # Synthetic exponentially decaying broadband IR (T60 ~ 1 s).
    rng = np.random.default_rng(0)
    n = fs  # 1 s
    t = np.arange(n) / fs
    t60_true = 1.0
    decay = 10.0 ** (-3.0 * t / t60_true)  # -60 dB over t60
    ir = rng.standard_normal(n) * decay
    ir[0] += 5.0  # a clear direct-sound peak

    parameters = room.room_parameters(ir, fs, limits=(500.0, 1000.0), fraction=1)
    t60_bands = parameters.t30  # decay time (s) extrapolated to 60 dB
    freqs = np.asarray(parameters.frequency, dtype=float)
    assert np.all(np.isfinite(t60_bands))

    lp = np.full(freqs.shape, 80.0)
    res = emission.sound_power_reverberation(lp, t60_bands, 200.0, 210.0, freqs)
    assert res.sound_power_level.shape == freqs.shape
    assert np.all(np.isfinite(res.sound_power_level))


# --------------------------------------------------------------------------
# Background correction (Eq. 14) and A-weighted total (Annex F)
# --------------------------------------------------------------------------
def test_background_correction_reduces_level_and_warns_below_criterion() -> None:
    """dLp = 4 dB (< 6 dB) at 100 Hz -> K1 clamped, upper-bound warning."""
    freqs = np.array([100.0])
    t60 = np.array([1.5])
    levels = np.array([[74.0]])
    background = np.array([[70.0]])  # dLp = 4 dB
    with pytest.warns(emission.SoundPowerWarning):
        res = emission.sound_power_reverberation(
            levels,
            t60,
            200.0,
            210.0,
            freqs,
            background_levels=background,
        )
    assert res.background_correction[0] > 0.0


def test_background_correction_is_per_position_eq14_to_16() -> None:
    """K1i is computed at each microphone position (Eq. 14), the position
    levels are corrected first (Eq. 15) and only then energy-averaged
    (Eq. 16) -- ISO 3741:2010, 9.1.2/9.1.3.

    Hand oracle at 1 kHz (lower criterion 10 dB): source positions
    70/68/66/72/71/69 dB over backgrounds 58/60/59/60/62/58 dB give margins
    12/8/7/12/9/11 dB, so three positions sit below the criterion and clamp
    to K1i = 0,4576 dB while the others follow Eq. (14). The corrected mean
    is 69,38872 dB (effective K1 = 0,36347 dB). A single K1 from the
    position-averaged spectra (margin 10,02 dB -> K1 = 0,45536 dB) would
    land 0,09 dB low at 69,29683 dB.
    """
    freqs = np.array([1000.0])
    t60 = np.array([1.5])
    levels = np.array([[70.0], [68.0], [66.0], [72.0], [71.0], [69.0]])
    background = np.array([[58.0], [60.0], [59.0], [60.0], [62.0], [58.0]])
    with pytest.warns(emission.SoundPowerWarning, match="9.1.2"):
        res = emission.sound_power_reverberation(
            levels,
            t60,
            200.0,
            210.0,
            freqs,
            background_levels=background,
        )
    assert res.mean_pressure_level[0] == pytest.approx(69.3887156232, abs=1e-9)
    assert res.background_correction[0] == pytest.approx(0.3634659689, abs=1e-9)
    # Regression: the position-averaged shortcut (K1 = 0,45536 dB) is not used.
    assert res.mean_pressure_level[0] != pytest.approx(69.2968263631, abs=1e-3)


def test_background_single_spectrum_broadcasts_to_positions() -> None:
    """A 1D background spectrum applies at every position (Eq. 14 per i)."""
    freqs = np.array([1000.0])
    t60 = np.array([1.5])
    levels = np.array([[70.0], [68.0], [66.0], [72.0], [71.0], [69.0]])
    with pytest.warns(emission.SoundPowerWarning):
        res_1d = emission.sound_power_reverberation(
            levels,
            t60,
            200.0,
            210.0,
            freqs,
            background_levels=np.array([60.0]),
        )
    with pytest.warns(emission.SoundPowerWarning):
        res_2d = emission.sound_power_reverberation(
            levels,
            t60,
            200.0,
            210.0,
            freqs,
            background_levels=np.full((6, 1), 60.0),
        )
    assert res_1d.mean_pressure_level[0] == pytest.approx(
        res_2d.mean_pressure_level[0], abs=1e-12
    )


def test_background_shape_mismatch_raises() -> None:
    """A 2D background that does not match the per-position levels raises."""
    freqs = np.array([1000.0])
    t60 = np.array([1.5])
    levels = np.full((6, 1), 80.0)
    mismatched_background = np.full((3, 1), 60.0)  # 3 rows against 6 positions
    with pytest.raises(ValueError, match="background_levels"):
        emission.sound_power_reverberation(
            levels,
            t60,
            200.0,
            210.0,
            freqs,
            background_levels=mismatched_background,
        )


def test_preaveraged_1d_levels_keep_single_k1_path() -> None:
    """1D pre-averaged spectra carry no per-position data: a single K1 from
    the averaged spectra applies (documented approximation of 9.1.2)."""
    freqs = np.array([1000.0])
    t60 = np.array([1.5])
    res = emission.sound_power_reverberation(
        np.array([80.0]),
        t60,
        200.0,
        210.0,
        freqs,
        background_levels=np.array([70.0]),  # margin 10 dB, no clamp
    )
    k1 = -10.0 * np.log10(1.0 - 10.0 ** (-0.1 * 10.0))
    assert res.background_correction[0] == pytest.approx(k1, abs=1e-9)
    assert res.mean_pressure_level[0] == pytest.approx(80.0 - k1, abs=1e-9)


def test_comparison_method_corrects_test_source_per_position() -> None:
    """The comparison method applies the same per-position Eq. 14/15/16 chain
    to the source under test before Eq. (21)."""
    freqs = np.array([1000.0])
    levels = np.array([[70.0], [68.0], [66.0], [72.0], [71.0], [69.0]])
    background = np.array([[58.0], [60.0], [59.0], [60.0], [62.0], [58.0]])
    lp_rss = np.array([70.0])
    lw_ref = np.array([90.0])
    with pytest.warns(emission.SoundPowerWarning):
        res = emission.sound_power_comparison(
            levels,
            lp_rss,
            lw_ref,
            frequencies=freqs,
            background_levels=background,
        )
    assert res.mean_pressure_level[0] == pytest.approx(69.3887156232, abs=1e-9)
    expected_lw = 90.0 + (69.3887156232 - 70.0 + res.c2)
    assert res.sound_power_level[0] == pytest.approx(expected_lw, abs=1e-9)


def test_a_weighted_total_from_bands() -> None:
    """LWA combines bands with Annex F A-weighting (Eq. F.2)."""
    freqs = np.array([500.0, 1000.0, 2000.0])
    t60 = np.full(freqs.shape, 1.5)
    lp = np.full(freqs.shape, 80.0)
    res = emission.sound_power_reverberation(lp, t60, 200.0, 210.0, freqs)
    assert np.isfinite(res.sound_power_level_a)
    # A-weighted total is an energy sum >= the largest single weighted band.
    assert res.sound_power_level_a > np.max(res.sound_power_level) - 10.0


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def test_invalid_volume_raises() -> None:
    lp, t60, freqs = np.array([80.0]), np.array([1.5]), np.array([1000.0])
    with pytest.raises(ValueError):
        emission.sound_power_reverberation(lp, t60, -1.0, 210.0, freqs)


def test_mismatched_shapes_raise() -> None:
    two_bands = np.array([80.0, 81.0])  # 2 levels against 1 band of T60/freq
    t60, freqs = np.array([1.5]), np.array([1000.0])
    with pytest.raises(ValueError):
        emission.sound_power_reverberation(two_bands, t60, 200.0, 210.0, freqs)


# --------------------------------------------------------------------------
# Meteorological input validation (clean ValueError before log10/sqrt)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("theta", [-273.0, -300.0, np.inf, np.nan])
def test_direct_method_invalid_temperature_raises(theta: float) -> None:
    lp, t60, freqs = np.array([80.0]), np.array([1.5]), np.array([1000.0])
    with pytest.raises(ValueError, match="temperature"):
        emission.sound_power_reverberation(
            lp,
            t60,
            200.0,
            210.0,
            freqs,
            temperature=theta,
        )


@pytest.mark.parametrize("ps", [0.0, -1.0, np.inf, np.nan])
def test_direct_method_invalid_pressure_raises(ps: float) -> None:
    lp, t60, freqs = np.array([80.0]), np.array([1.5]), np.array([1000.0])
    with pytest.raises(ValueError, match="static_pressure"):
        emission.sound_power_reverberation(
            lp,
            t60,
            200.0,
            210.0,
            freqs,
            static_pressure=ps,
        )


def test_comparison_method_invalid_temperature_raises() -> None:
    levels, levels_ref, lw_ref = (np.array([80.0]), np.array([70.0]), np.array([90.0]))
    with pytest.raises(ValueError, match="temperature"):
        emission.sound_power_comparison(levels, levels_ref, lw_ref, temperature=-300.0)


def test_comparison_method_invalid_pressure_raises() -> None:
    levels, levels_ref, lw_ref = (np.array([80.0]), np.array([70.0]), np.array([90.0]))
    with pytest.raises(ValueError, match="static_pressure"):
        emission.sound_power_comparison(levels, levels_ref, lw_ref, static_pressure=0.0)


# --------------------------------------------------------------------------
# Room-qualification advisories (ISO 3741:2010, 5.2/5.3/8.3/8.4.2.2)
# --------------------------------------------------------------------------
def test_volume_below_table1_minimum_warns() -> None:
    """Lowest band 100 Hz needs >= 200 m^3 (Table 1); 150 m^3 is advisory."""
    freqs = np.array([100.0, 500.0])
    t60 = np.array([1.5, 1.5])
    lp = np.array([80.0, 80.0])
    with pytest.warns(emission.SoundPowerWarning, match="Table 1"):
        emission.sound_power_reverberation(lp, t60, 150.0, 210.0, freqs)


def test_t60_below_v_over_s_floor_warns() -> None:
    """T60 <= V/S in a band below 6,3 kHz violates the Eq. (7) floor."""
    freqs = np.array([1000.0])
    # floor = V/S = 200/210 ~ 0,95 s; T60 = 0,5 s is below it.
    t60 = np.array([0.5])
    lp = np.array([80.0])
    with pytest.warns(emission.SoundPowerWarning, match="V/S floor"):
        emission.sound_power_reverberation(lp, t60, 200.0, 210.0, freqs)


def test_fewer_than_six_positions_warns() -> None:
    """Only 3 microphone rows -> below the 6-position minimum (8.3/8.4.1)."""
    freqs = np.array([1000.0])
    t60 = np.array([1.5])
    levels = np.array([[80.0], [80.1], [79.9]])  # 3 positions, tight spread
    with pytest.warns(emission.SoundPowerWarning, match="microphone position"):
        emission.sound_power_reverberation(levels, t60, 200.0, 210.0, freqs)


def test_interposition_std_above_criterion_warns() -> None:
    """sM > 1,5 dB across the 6 positions flags discrete tones (Eq. 10)."""
    freqs = np.array([1000.0])
    t60 = np.array([1.5])
    # Six positions with a large spread -> sM > 1,5 dB.
    levels = np.array([[70.0], [80.0], [72.0], [78.0], [71.0], [79.0]])
    with pytest.warns(emission.SoundPowerWarning, match="standard deviation"):
        emission.sound_power_reverberation(levels, t60, 200.0, 210.0, freqs)


def test_qualified_room_emits_no_warning() -> None:
    """Adequate volume, T60 above the floor, 6 tight positions -> silent."""
    freqs = np.array([500.0, 1000.0])
    t60 = np.array([1.5, 1.5])
    base = np.array([80.0, 80.0])
    levels = np.tile(base, (6, 1)) + np.array(
        [[0.0, 0.1], [-0.1, 0.0], [0.1, -0.1], [0.0, 0.1], [-0.1, 0.0], [0.05, -0.05]]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", emission.SoundPowerWarning)
        emission.sound_power_reverberation(levels, t60, 200.0, 210.0, freqs)


# --------------------------------------------------------------------------
# Validation ordering (comparison method): shape check before background block
# --------------------------------------------------------------------------
def test_comparison_wrong_frequency_length_raises_clean_error() -> None:
    """Wrong-length 'frequencies' with 'background_levels' must raise a clean
    ValueError (shape check ahead of the background broadcasting)."""
    levels = np.array([80.0, 81.0, 82.0])
    levels_ref = np.array([70.0, 71.0, 72.0])
    lw_ref = np.array([90.0, 91.0, 92.0])
    background = np.array([50.0, 51.0, 52.0])
    short_freqs = np.array([1000.0, 2000.0])  # wrong length (2 != 3)
    with pytest.raises(ValueError, match="length must match"):
        emission.sound_power_comparison(
            levels,
            levels_ref,
            lw_ref,
            frequencies=short_freqs,
            background_levels=background,
        )


def test_comparison_wrong_shape_raises_without_sampling_warning() -> None:
    """Malformed shapes raise ValueError before any position-sampling advisory
    (warn-after-validate: a fewer-than-6-positions input must not warn first)."""
    # 3 positions (< 6) x 2 bands, but levels_ref spans 3 bands -> mismatch.
    levels = np.array([[80.0, 81.0], [80.1, 81.1], [79.9, 80.9]])
    levels_ref = np.array([70.0, 71.0, 72.0])
    lw_ref = np.array([90.0, 91.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", emission.SoundPowerWarning)
        with pytest.raises(ValueError, match="span the same bands"):
            emission.sound_power_comparison(levels, levels_ref, lw_ref)


# --------------------------------------------------------------------------
# Comparison method reports the effective per-band background correction K1
# --------------------------------------------------------------------------
def test_comparison_reports_applied_background_correction() -> None:
    """The comparison method reports the ST K1 actually applied per band (Eq.
    14), and reporting the field does not alter the computed LW."""
    freqs = np.array([1000.0])
    lw_ref = np.array([90.0])
    lp_rss = np.array([70.0])
    lp_st = np.array([80.0])
    background = np.array([70.0])  # dLp = 10 dB at 1 kHz (>= lower criterion)
    # Analytic K1 with a 10 dB margin (no clamp, no zeroing).
    k1 = -10.0 * np.log10(1.0 - 10.0 ** (-0.1 * 10.0))

    res = emission.sound_power_comparison(
        lp_st,
        lp_rss,
        lw_ref,
        frequencies=freqs,
        background_levels=background,
    )
    assert res.background_correction[0] == pytest.approx(k1, abs=1e-9)
    # LW keeps the K1-corrected level: LW(RSS) + (Lp(ST) - K1 - Lp(RSS) + C2).
    c2 = _c2(23.0, _PS0)
    expected_lw = 90.0 + ((80.0 - k1) - 70.0 + c2)
    assert res.sound_power_level[0] == pytest.approx(expected_lw, abs=1e-9)


def test_comparison_background_correction_zero_without_background() -> None:
    """No background supplied -> the reported per-band K1 is zero."""
    res = emission.sound_power_comparison(
        np.array([80.0, 80.0]),
        np.array([70.0, 70.0]),
        np.array([90.0, 90.0]),
    )
    assert np.all(res.background_correction == 0.0)
