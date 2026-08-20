#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for parametric filters: Weighting (A, C), Time Weighting and Linkwitz-Riley.
"""

import numpy as np
import pytest

from phonometry import filters, metrology


def test_calibration_logic() -> None:
    """
    Verify that calibration correctly maps digital RMS to target SPL.

    **Purpose:**
    Confirm that the `sensitivity` function produces a multiplier that accurately
    scales a digital signal to a known physical Sound Pressure Level (SPL) in dB.

    **Verification:**
    - Simulate a digital recording of a calibrator tone (e.g., 94 dB).
    - Calculate the sensitivity factor.
    - Analyze the same signal using that factor.

    **Expectation:**
    - The resulting SPL must be exactly the target value (e.g., 94.0 dB).
    """
    fs = 48000
    # Create a 'recording' of a 94dB tone (RMS = 0.5 for example)
    rms_ref = 0.5
    t = np.linspace(0, 1, fs)
    ref_signal = rms_ref * np.sqrt(2) * np.sin(2 * np.pi * 1000 * t)

    # Calculate sensitivity
    factor = metrology.sensitivity(ref_signal, target_spl=94.0)

    # Analyze same signal with that factor
    spl, _ = filters.octave_filter(
        ref_signal,
        fs,
        fraction=1,
        limits=[800, 1200],
        calibration=filters.LevelCalibration(factor=factor),
    )

    # It should be exactly 94 dB
    assert abs(spl[0] - 94.0) < 0.01


def test_dbfs_logic() -> None:
    """
    Verify dBFS mode returns RMS relative to 1.0.

    **Purpose:**
    Ensure the `dbfs=True` option correctly calculates decibels relative to full scale (0 dBFS = RMS of 1.0).

    **Verification:**
    - Pass a sine wave with a peak of 1.0 (RMS = 0.707).

    **Expectation:**
    - The output should be approximately -3.01 dBFS.
    """
    fs = 48000
    # Sine wave with peak 1.0 -> RMS 0.707 -> -3.01 dBFS
    t = np.linspace(0, 1, fs)
    x = np.sin(2 * np.pi * 1000 * t)

    spl, _ = filters.octave_filter(
        x,
        fs,
        fraction=1,
        limits=[800, 1200],
        calibration=filters.LevelCalibration(dbfs=True),
    )

    assert abs(spl[0] - (-3.01)) < 0.05


def test_peak_mode_logic() -> None:
    """
    Verify that Peak mode returns the absolute maximum of the filtered band.

    **Purpose:**
    Addressing Issue #10 regarding consistency with professional software (peak-holding).

    **Verification:**
    - Create a signal with a single high peak (impulse-like).
    - Compare RMS vs Peak output.

    **Expectation:**
    - Peak value should be significantly higher than RMS for a transient signal.
    - Peak value should match 20*log10(max_abs / 2e-5) approximately.
    """
    fs = 48000
    x = np.zeros(fs)
    x[100] = 0.5  # Large peak

    spl_rms, _ = filters.octave_filter(x, fs, mode="rms", fraction=1)
    spl_peak, _ = filters.octave_filter(x, fs, mode="peak", fraction=1)

    # Peak must be greater than RMS for an impulse
    assert np.all(spl_peak > spl_rms)

    # Theoretical peak for 0.5: 20*log10(0.5 / 2e-5) = 20*log10(25000) approx 87.9 dB
    # (Minus some attenuation due to the filter bandwidth and energy spreading)
    assert np.max(spl_peak) > 75.0


def test_a_weighting_response() -> None:
    """
    Verify A-weighting frequency response.

    **Purpose:**
    Confirm that the A-weighting filter matches the standardized IEC 61672-1:2013 gains at key frequencies.

    **Verification:**
    - Measure gain at 100Hz (expected -19.1 dB).
    - Measure gain at 1000Hz (expected 0.0 dB).
    - Measure gain at 8000Hz (expected -1.1 dB).

    **Expectation:**
    - Measured gains should match standard values within 1.0 dB.
    """
    fs = 48000
    duration = 1.0
    t = np.linspace(0, duration, fs, endpoint=False)

    test_freqs = [100, 1000, 8000]
    expected_gain = [-19.1, 0.0, -1.1]

    for f, expected in zip(test_freqs, expected_gain, strict=True):
        # Generate tone
        x = np.sin(2 * np.pi * f * t)
        y = filters.weighting_filter(x, fs, curve="A")

        # Calculate gain in dB (RMS)
        gain_db = 20 * np.log10(np.std(y) / np.std(x))
        assert abs(gain_db - expected) < 1.0, (
            f"A-weighting at {f}Hz failed. Got {gain_db:.1f}dB, expected {expected}dB"
        )


def test_c_weighting_response() -> None:
    """
    Verify C-weighting frequency response.

    **Purpose:**
    Confirm that the C-weighting filter matches the standardized IEC 61672-1:2013 gains.

    **Verification:**
    - Measure gain at 31.5Hz (expected -3.0 dB).
    - Measure gain at 1000Hz (expected 0.0 dB).
    - Measure gain at 8000Hz (expected -3.0 dB).

    **Expectation:**
    - Measured gains should match standard values within 1.0 dB.
    """
    fs = 48000
    duration = 1.0
    t = np.linspace(0, duration, fs, endpoint=False)

    test_freqs = [31.5, 1000, 8000]
    expected_gain = [-3.0, 0.0, -3.0]

    for f, expected in zip(test_freqs, expected_gain, strict=True):
        x = np.sin(2 * np.pi * f * t)
        y = filters.weighting_filter(x, fs, curve="C")

        gain_db = 20 * np.log10(np.std(y) / np.std(x))
        assert abs(gain_db - expected) < 1.0, (
            f"C-weighting at {f}Hz failed. Got {gain_db:.1f}dB, expected {expected}dB"
        )


def test_weighting_filter_c_output_shape() -> None:
    """Verify C-weighting preserves signal length."""
    fs = 48000
    rng = np.random.default_rng(42)
    x = rng.standard_normal(fs)

    y = filters.weighting_filter(x, fs, curve="C")

    assert y.shape == x.shape


def test_weighting_filter_class_direct_use() -> None:
    """Verify direct WeightingFilter usage, Z bypass, and constructor validation."""
    fs = 48000
    rng = np.random.default_rng(42)
    x = rng.standard_normal((2, fs))

    wf = filters.WeightingFilter(fs, curve="A")
    y = wf.filter(x)
    assert y.shape == x.shape

    wf_z = filters.WeightingFilter(fs, curve="Z")
    y_z = wf_z.filter(x)
    np.testing.assert_allclose(y_z, x)

    with pytest.raises(ValueError):
        filters.WeightingFilter(0)
    with pytest.raises(ValueError):
        filters.WeightingFilter(fs, curve="invalid")


def test_time_weighting_fast() -> None:
    """
    Verify Fast (125ms) time weighting response to a step.

    **Purpose:**
    Validate the exponential integration constant ($\tau$) for time ballistics.

    **Verification:**
    - Apply a unit step (DC 1.0) to the fast integrator.
    - Measure the value at $t = \tau$.

    **Expectation:**
    - The value should be approximately $1 - e^{-1} \approx 0.632$ (63.2% rise).
    """
    fs = 1000
    tau = 0.125
    x = np.ones(int(fs * 2))  # Step signal

    y = filters.time_weighting(x, fs, mode="fast")

    # Check at index corresponding to tau
    idx_tau = int(fs * tau)
    # Expected: 1 - exp(-1) approx 0.632
    assert abs(y[idx_tau] - 0.632) < 0.05


def test_time_weighting_initial_state_first() -> None:
    """
    Verify that initial_state='first' starts the integrator from the first energy value.
    """
    fs = 1000
    x = np.ones(fs)

    y = filters.time_weighting(x, fs, mode="fast", initial_state="first")

    np.testing.assert_allclose(y, np.ones_like(y), rtol=1e-12, atol=1e-12)


def test_time_weighting_initial_state_custom() -> None:
    """
    Verify custom initial_state matches the recursive equation with y[-1] set.
    """
    fs = 1000
    tau = 0.125
    alpha = 1 - np.exp(-1 / (fs * tau))
    x = np.array([2.0, 0.5, -1.0, 0.0])
    initial_state = 0.25

    y = filters.time_weighting(x, fs, mode="fast", initial_state=initial_state)

    expected = np.zeros_like(x)
    current = initial_state
    for idx, value in enumerate(x**2):
        current = alpha * value + (1 - alpha) * current
        expected[idx] = current

    np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-12)


def test_time_weighting_initial_state_zero_matches_default() -> None:
    """Verify explicit zero initialization matches the default rest state."""
    fs = 1000
    x = np.array([2.0, 0.5, -1.0, 0.0])

    y_default = filters.time_weighting(x, fs, mode="fast")
    y_zero = filters.time_weighting(x, fs, mode="fast", initial_state="zero")

    np.testing.assert_allclose(y_zero, y_default, rtol=1e-12, atol=1e-12)


def test_time_weighting_initial_state_array_per_channel() -> None:
    """Verify array initial state is applied per input channel."""
    fs = 1000
    tau = 0.125
    alpha = 1 - np.exp(-1 / (fs * tau))
    x = np.vstack([np.ones(4), 2 * np.ones(4)])
    initial_state = np.array([0.25, 4.0])

    y = filters.time_weighting(x, fs, mode="fast", initial_state=initial_state)

    expected = np.empty_like(x)
    current = initial_state.copy()
    for idx in range(x.shape[-1]):
        current = alpha * x[:, idx] ** 2 + (1 - alpha) * current
        expected[:, idx] = current

    np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-12)


def test_time_weighting_initial_state_multichannel_first() -> None:
    """
    Verify initial_state='first' is applied independently per channel.
    """
    fs = 1000
    x = np.vstack([np.ones(fs), 2 * np.ones(fs)])

    y = filters.time_weighting(x, fs, mode="fast", initial_state="first")

    expected = x**2
    np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-12)


def test_time_weighting_initial_state_invalid() -> None:
    """
    Verify invalid initial_state names are rejected.
    """
    fs = 1000
    x = np.ones(fs)

    with pytest.raises(ValueError, match="initial_state"):
        filters.time_weighting(x, fs, mode="fast", initial_state="invalid")

    two_channels = np.ones((2, 10))
    mismatched_state = np.ones(3)
    with pytest.raises(ValueError, match="broadcastable"):
        filters.time_weighting(
            two_channels, fs, mode="fast", initial_state=mismatched_state
        )


def test_time_weighting_initial_state_first_rejects_empty_input() -> None:
    """Verify initial_state='first' requires at least one sample."""
    empty = np.array([])
    with pytest.raises(ValueError, match="initial_state"):
        filters.time_weighting(empty, 1000, mode="fast", initial_state="first")


@pytest.mark.parametrize("mode", ["fast", "slow", "impulse"])
def test_time_weighting_rejects_non_positive_sample_rate(mode: str) -> None:
    """Verify time weighting rejects non-positive sample rates before coefficient math."""
    x = np.ones(10)
    with pytest.raises(ValueError, match="Sample rate 'fs' must be positive"):
        filters.time_weighting(x, 0, mode=mode)


def test_time_weighting_impulse_multichannel() -> None:
    """Verify impulse time weighting supports multichannel input."""
    fs = 1000
    x = np.zeros((2, fs))
    x[:, 0] = 1.0

    y = filters.time_weighting(x, fs, mode="impulse")

    assert y.shape == x.shape
    assert y[0, 100] < y[0, 0]


def test_time_weighting_impulse_initial_state_first_1d() -> None:
    """Verify impulse time weighting can start from the first sample energy."""
    fs = 1000
    x = np.ones(fs)

    y = filters.time_weighting(x, fs, mode="impulse", initial_state="first")

    np.testing.assert_allclose(y, np.ones_like(y), rtol=1e-12, atol=1e-12)


def test_linkwitz_riley_sum() -> None:
    """
    Verify that the sum of Linkwitz-Riley bands is flat.

    **Purpose:**
    The defining characteristic of an LR4 crossover is that the combined response of the
    low-pass and high-pass bands is perfectly flat.

    **Verification:**
    - Split white noise at 1000 Hz using `linkwitz_riley`.
    - Sum the resulting bands.
    - Measure the total RMS gain.

    **Expectation:**
    - The gain of the sum should be 1.0 (0 dB) with very low error ($< 0.1$ dB).
    """
    fs = 48000
    rng = np.random.default_rng(42)
    x = rng.standard_normal(fs)

    # Split at 1000 Hz
    lp, hp = filters.linkwitz_riley(x, fs, freq=1000, order=4)

    # Sum of bands
    y_sum = lp + hp

    # Check RMS ratio
    gain_db = 20 * np.log10(np.std(y_sum) / np.std(x))
    assert abs(gain_db) < 0.1, f"Linkwitz-Riley sum not flat: {gain_db:.2f} dB"


def test_weighting_z_bypass() -> None:
    """
    Verify Z-weighting is a bypass.

    **Purpose:**
    Confirm that 'Z' (Zero weighting) does not modify the signal.

    **Verification:**
    - Compare input and output arrays.

    **Expectation:**
    - Arrays must be identical.
    """
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    y = filters.weighting_filter(x, 48000, curve="Z")
    assert np.all(x == y)


def test_time_weighting_int16_no_overflow() -> None:
    """
    Verify integer input does not overflow when squared internally.

    **Purpose:**
    int16 audio (e.g. from ``scipy.io.wavfile.read``) previously overflowed
    silently in ``time_weighting`` (x**2 on int16), producing negative
    mean-square energies.

    **Verification:**
    - Envelope of int16 input is non-negative everywhere.
    - Final envelope value matches the float64 reference within 0.1%.
    """
    fs = 48000
    t = np.arange(fs) / fs
    x_float = 0.9 * 32767 * np.sin(2 * np.pi * 1000 * t)
    x_int = x_float.astype(np.int16)

    env_int = filters.time_weighting(x_int, fs, mode="fast")
    env_float = filters.time_weighting(x_float, fs, mode="fast")

    assert (env_int >= 0).all(), "mean-square envelope can never be negative"
    np.testing.assert_allclose(env_int[-1], env_float[-1], rtol=1e-3)


def test_sensitivity_int16_matches_float() -> None:
    """
    Verify calibration works with integer reference recordings.

    **Purpose:**
    ``sensitivity`` previously squared the raw integer array,
    overflowing and returning a factor ~315x too large for int16 input.

    **Verification:**
    - Sensitivity from int16 input matches the float64 reference within 0.1%.
    """
    fs = 48000
    t = np.arange(fs) / fs
    x_float = 0.5 * 32767 * np.sin(2 * np.pi * 1000 * t)

    s_int = metrology.sensitivity(x_float.astype(np.int16))
    s_float = metrology.sensitivity(x_float)
    np.testing.assert_allclose(s_int, s_float, rtol=1e-3)


def _analytic_a_weight_db(f: float) -> float:
    """IEC 61672-1 analytic A-weighting curve."""
    f2 = float(f) ** 2
    ra = (12194**2 * f2**2) / (
        (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2)
    )
    return float(20 * np.log10(ra) + 2.0)


def _measured_gain_db(wf: filters.WeightingFilter, fs: int, f0: float) -> float:
    """End-to-end RMS gain of the weighting filter at a single frequency."""
    t = np.arange(int(fs * 0.5)) / fs
    x = np.sin(2 * np.pi * f0 * t)
    y = wf.filter(x)
    n0 = int(0.1 * fs)  # skip the filter transient
    rms_in = np.sqrt(np.mean(x[n0:] ** 2))
    rms_out = np.sqrt(np.mean(y[n0:] ** 2))
    return float(20 * np.log10(rms_out / rms_in))


@pytest.mark.parametrize("fs", [44100, 48000])
def test_a_weighting_class1_high_frequencies(fs: int) -> None:
    """
    Verify A-weighting stays within IEC 61672-1 class 1 tolerances at HF.

    **Purpose:**
    The plain bilinear design compresses the response near Nyquist: at
    fs=48000 the error at 12.5 kHz was -2.67 dB (class 1 limit: -2.5 dB).
    With high_accuracy (internal oversampling to >= 96 kHz) the error
    drops below -0.6 dB.

    **Verification:**
    - Errors vs the analytic curve stay within class 1 tolerances
      (12.5 kHz: +2.0/-2.5 dB; 16 kHz: +2.5/-16 dB, tightened here).
    """
    wf = filters.WeightingFilter(fs, "A")  # high_accuracy defaults to True
    # Tightened to lock the 144 kHz oversample-target fit (audit N1 A6): with
    # the target raised from 96 to 144 kHz both 44.1k and 48k oversample enough
    # to keep the HF error within +/-0.7 dB of the analytic curve up to 16 kHz.
    for f0, tol_lo, tol_hi in [
        (8000, -0.7, 0.7),
        (12500, -0.7, 0.7),
        (16000, -0.7, 0.7),
    ]:
        err = _measured_gain_db(wf, fs, f0) - _analytic_a_weight_db(f0)
        assert tol_lo < err < tol_hi, f"{f0} Hz: error {err:.2f} dB at fs={fs}"


def test_a_weighting_48k_hf_accuracy_locked() -> None:
    """Lock the 144 kHz oversample-target HF fit at 48 kHz (audit N1 A6).

    The former 96 kHz target oversampled 48 kHz only x2, leaving -1.11 dB @16k
    and -2.10 dB @20k vs the analytic A curve. Raising the internal target to
    144 kHz oversamples x3, halving those residuals to about -0.44 / -0.85 dB.
    """
    fs = 48000
    wf = filters.WeightingFilter(fs, "A")
    for f0, bound in [(16000, 0.6), (20000, 1.0)]:
        err = _measured_gain_db(wf, fs, f0) - _analytic_a_weight_db(f0)
        assert abs(err) < bound, f"{f0} Hz: error {err:+.2f} dB at fs={fs}"


def test_high_accuracy_incompatible_with_stateful() -> None:
    """high_accuracy resampling would break block continuity: must raise."""
    with pytest.raises(ValueError, match="high_accuracy"):
        filters.WeightingFilter(48000, "A", stateful=True, high_accuracy=True)


def test_stateful_defaults_to_legacy_mode() -> None:
    """Stateful filters keep the plain bilinear design (no oversampling)."""
    wf = filters.WeightingFilter(48000, "A", stateful=True)
    assert wf._oversample == 1


def test_high_accuracy_preserves_length_and_shape() -> None:
    """Oversample+decimate round trip must preserve the input shape."""
    x = np.random.default_rng(0).standard_normal((2, 12345))
    y = filters.WeightingFilter(48000, "A").filter(x)
    assert y.shape == x.shape


def _analytic_c_weight_db(f: float) -> float:
    """IEC 61672-1 analytic C-weighting curve."""
    f2 = float(f) ** 2
    rc = (12194**2 * f2) / ((f2 + 20.6**2) * (f2 + 12194**2))
    return float(20 * np.log10(rc) + 0.06)


@pytest.mark.parametrize("fs", [44100, 48000])
def test_c_weighting_class1_high_frequencies(fs: int) -> None:
    """
    Verify C-weighting stays within IEC 61672-1 class 1 tolerances at HF.

    **Purpose:**
    Same oversampled design path as the A curve; dedicated regression so a
    C-specific change cannot silently degrade HF accuracy.
    """
    wf = filters.WeightingFilter(fs, "C")
    # Tightened alongside the A curve for the 144 kHz oversample target (A6).
    for f0, tol_lo, tol_hi in [
        (8000, -0.7, 0.7),
        (12500, -0.7, 0.7),
        (16000, -0.7, 0.7),
    ]:
        err = _measured_gain_db(wf, fs, f0) - _analytic_c_weight_db(f0)
        assert tol_lo < err < tol_hi, f"{f0} Hz: error {err:.2f} dB at fs={fs}"


def test_weighting_filter_empty_signal_high_accuracy() -> None:
    """Empty input must pass through (resample_poly rejects zero-length input)."""
    wf = filters.WeightingFilter(48000, "A")
    y = wf.filter(np.array([]))
    assert y.shape == (0,)


def test_a_weighting_positive_gain_region() -> None:
    """
    Verify the A-curve is POSITIVE between 1.25 and 5 kHz (IEC 61672-1 Table 2).

    **Purpose:**
    The A-weighting models equal-loudness sensitivity: the ear is MORE
    sensitive than at 1 kHz around 2-4 kHz, so the curve must exceed 0 dB
    there (max +1.27 dB at ~2.5 kHz). A curve that never crosses 0 dB
    indicates a broken normalization or measurement.

    **Verification (against IEC 61672-1 Table 2 values):**
    - 1250 Hz: +0.6 | 2000 Hz: +1.2 | 2500 Hz: +1.3 | 4000 Hz: +1.0 | 5000 Hz: +0.5
    """
    fs = 48000
    wf = filters.WeightingFilter(fs, "A")
    for f0, expected in [
        (1250, 0.6),
        (2000, 1.2),
        (2500, 1.3),
        (4000, 1.0),
        (5000, 0.5),
    ]:
        gain = _measured_gain_db(wf, fs, f0)
        assert gain > 0, f"A-weighting must be positive at {f0} Hz, got {gain:.2f} dB"
        assert gain == pytest.approx(expected, abs=0.15), (
            f"{f0} Hz: {gain:.2f} dB vs Table 2 {expected}"
        )
