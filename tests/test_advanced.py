#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Advanced tests for input validation, edge cases, and robustness.
"""

import numpy as np
import pytest

from pyoctaveband import normalizedfreq, octavefilter


def test_fraction_validation():
    """
    Test the filter's behavior with standard and non-standard fractional bandwidths.

    **Purpose:**
    Verify that the `octavefilter` function can handle both standard (1/1, 1/3 octave)
    and non-standard fractions (e.g., 1/2 octave) correctly. Also ensures `normalizedfreq`
    restricts usage to defined standards.

    **Verification:**
    - Call `octavefilter` with fraction=1 and fraction=3.
    - Call `octavefilter` with a non-standard fraction=2.
    - Call `normalizedfreq` with an invalid fraction.

    **Expectation:**
    - Filtering should succeed and return frequency bands for all valid integer fractions.
    - Higher denominators (smaller bandwidths) should result in more bands (fraction=3 > fraction=1).
    - `normalizedfreq` should raise a `ValueError` for non-standard fractions,
      as it relies on a lookup table for IEC standards.
    """
    fs = 48000
    x = np.random.randn(fs)  # 1 second of noise

    # Standard fractions
    spl1, freq1 = octavefilter(x, fs, fraction=1)
    assert len(freq1) > 0
    spl3, freq3 = octavefilter(x, fs, fraction=3)
    assert len(freq3) > len(freq1)

    # Non-standard fraction (should work mathematically via _genfreqs)
    spl2, freq2 = octavefilter(x, fs, fraction=2)
    assert len(freq2) > 0

    # normalizedfreq only supports 1 and 3
    with pytest.raises(ValueError, match="Normalized frequencies only available"):
        normalizedfreq(2)


def test_invalid_inputs():
    """
    Test input validation logic for invalid parameters.

    **Purpose:**
    Ensure that the function robustly rejects invalid input configurations that could lead
    to undefined behavior, infinite loops, or crashes.

    **Verification:**
    - Provide limits where the start frequency is greater than the end frequency.
    - Provide negative or zero limits.
    - Provide an incorrect number of limit values.
    - Provide a zero or negative sampling rate.
    - Provide a zero or negative bandwidth fraction.
    - Provide a negative filter order.

    **Expectation:**
    - The function should raise a `ValueError` with a descriptive message for each invalid case.
    - This prevents the internal logic (like `_genfreqs` or filter design) from failing obscurely.
    """
    fs = 48000
    x = np.zeros(1000)

    # Invalid limits: reversed
    with pytest.raises(ValueError, match="lower limit must be less than the upper limit"):
        octavefilter(x, fs, limits=[20000, 100])

    # Invalid limits: non-positive
    with pytest.raises(ValueError, match="Limit frequencies must be positive"):
        octavefilter(x, fs, limits=[0, 1000])

    # Invalid limits: wrong length
    with pytest.raises(ValueError, match="Limits must be a list of two frequencies"):
        octavefilter(x, fs, limits=[100, 500, 1000])

    # Invalid fs
    with pytest.raises(ValueError, match="Sample rate 'fs' must be positive"):
        octavefilter(x, 0, limits=[100, 1000])

    # Invalid fraction
    with pytest.raises(ValueError, match="Bandwidth 'fraction' must be positive"):
        octavefilter(x, fs, fraction=0)

    # Invalid order
    with pytest.raises(ValueError, match="Filter 'order' must be positive"):
        octavefilter(x, fs, order=-1)

    # Invalid filter_type
    with pytest.raises(ValueError, match="Invalid filter_type"):
        octavefilter(x, fs, filter_type="invalid_type")


def test_short_signal():
    """
    Test processing of a signal shorter than the downsampling factor.

    **Purpose:**
    The library uses multirate processing (downsampling) to stabilize low-frequency filters.
    If the signal is too short, downsampling might result in an empty array or a single sample.

    **Verification:**
    - Generate a very short signal (100 samples).
    - Request filtering down to 12 Hz (which requires a large downsampling factor ~2000).

    **Expectation:**
    - The function should handle this gracefully (likely via padding or `resample_poly` handling).
    - It should return valid SPL values (not NaN) and the correct number of bands.
    """
    fs = 48000
    # Lowest frequency 12Hz requires large downsampling factor
    # approx factor = fs / (2*freq) -> 48000 / 24 = 2000
    # Let's try a very short signal
    x = np.random.randn(100) 
    
    # This might fail if resample produces empty array or 0 length
    spl, freq = octavefilter(x, fs, limits=[12, 100])
    
    assert not np.isnan(spl).any()
    assert len(spl) == len(freq)


def test_nan_handling():
    """
    Test the behavior when the input signal contains NaN values.

    **Purpose:**
    Determine if NaNs in the input propagate to the output.

    **Verification:**
    - Create a signal with a single NaN value.
    - Process it through the filter bank.

    **Expectation:**
    - Digital filters (IIR/SOS) inherently propagate NaNs because the output depends on previous inputs.
    - The output SPL array should contain NaNs, confirming standard signal processing behavior.
    """
    fs = 48000
    x = np.random.randn(4800)
    x[100] = np.nan
    
    # It will likely propagate NaNs or crash in filtering
    # We want to know what happens. 
    # Current implementation: _typesignal -> numpy array. 
    # signal.resample propogates NaNs. 
    # signal.sosfilt propagates NaNs. 
    # np.std(NaN) -> NaN.
    # 20*log10(NaN) -> NaN.
    
    spl, freq = octavefilter(x, fs)
    # Expect NaNs in SPL
    assert np.isnan(spl).any()


def test_silence():
    """
    Test the filter's output for a completely silent input signal.

    **Purpose:**
    Verify that a zero-input signal results in an appropriately low SPL (decibel) reading.

    **Verification:**
    - Pass an array of zeros.

    **Expectation:**
    - Log of zero is negative infinity. The code typically clamps this to a small epsilon to avoid math errors.
    - The resulting SPL should be a very low negative number (e.g., < -100 dB), representing the noise floor/epsilon.
    """
    fs = 48000
    x = np.zeros(fs)
    
    spl, freq = octavefilter(x, fs)
    
    # Should be very low dB (approx -inf, but code clips to eps)
    # 20 * log10(eps / 2e-5)
    # eps is approx 2.22e-16
    # 2.22e-16 / 2e-5 = 1.11e-11
    # 20 * log10(1e-11) = -220 dB
    
    assert np.all(spl < -100)


def test_nyquist_limit():
    """
    Test handling of requested frequency bands that exceed the Nyquist limit.

    **Purpose:**
    The Nyquist theorem states valid frequencies are < fs/2. Requesting bands above this
    is physically impossible for digital signals.

    **Verification:**
    - Set a low sampling rate (1000 Hz, Nyquist = 500 Hz).
    - Request bands up to 1000 Hz.

    **Expectation:**
    - The function should emit a `UserWarning`.
    - It should automatically prune the bands above 500 Hz.
    - The returned frequencies should all be less than 500 Hz.
    """
    fs = 1000 # Nyquist 500
    x = np.random.randn(fs)
    
    # Request up to 1000Hz
    # _deleteouters should warn and remove high bands
    with pytest.warns(UserWarning, match="frequencies above fs/2 removed"):
        spl, freq = octavefilter(x, fs, limits=[10, 1000])
        
    assert np.all(np.array(freq) < fs/2)


def test_high_order_stability():
    """
    Test the numerical stability of high-order filters.

    **Purpose:**
    High-order IIR filters (e.g., order > 10) are prone to numerical instability if implemented
    using Transfer Function (ba) coefficients. This library uses Second-Order Sections (SOS),
    which should remain stable even at higher orders.

    **Verification:**
    - Request filters with order 12 and 24 (very high for IIR).
    - Process random noise.

    **Expectation:**
    - The output should be valid numbers (not NaNs or Infs), confirming the SOS implementation works correctly.
    """
    fs = 48000
    x = np.random.randn(fs)
    
    # Order 12 or 24 is quite high for standard IIR, but SOS is better.
    # We just want to ensure it doesn't explode into NaNs.
    spl, freq = octavefilter(x, fs, order=12)
    assert not np.isnan(spl).any()
    
    spl2, freq2 = octavefilter(x, fs, order=24)
    assert not np.isnan(spl2).any()
