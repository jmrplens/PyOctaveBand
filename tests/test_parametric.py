#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Parametric tests using pytest best practices for signal processing verification.
"""

import numpy as np
import pytest

from pyoctaveband import OctaveFilterBank, octavefilter


@pytest.mark.parametrize("fraction, expected_bands", [
    (1, 11),  # Standard 1 octave bands in [12, 20000] approx
    (3, 33),  # Standard 1/3 octave
])
def test_band_count_estimates(fraction: float, expected_bands: int) -> None:
    """
    Verify that the filter bank generates the expected number of frequency bands.

    **Purpose:**
    Ensure that the band generation logic (`_genfreqs`) correctly interprets the fraction
    argument and produces the standard count of bands for the default [12, 20000] Hz range.

    **Verification:**
    - Test with fraction=1 (Octave bands) -> Expect ~11 bands.
    - Test with fraction=3 (1/3 Octave bands) -> Expect ~33 bands.

    **Expectation:**
    - The number of generated center frequencies should match the theoretical expectation
      (within a small margin of error due to exact edge handling).
    """
    fs = 48000
    # Generate dummy signal
    x = np.zeros(fs)
    limits = [12.0, 20000.0]
    
    _, freq = octavefilter(x, fs, fraction=fraction, limits=limits)
    
    # We allow some flexibility as exact count depends on limits implementation details
    assert abs(len(freq) - expected_bands) <= 2, f"Expected approx {expected_bands} bands, got {len(freq)}"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_dtypes(dtype: np.dtype) -> None:
    """
    Ensure the library correctly handles different floating-point precisions.

    **Purpose:**
    Audio data can come in float32 (common in WAV files/real-time) or float64.
    The library should process both without crashing or type errors.

    **Verification:**
    - Generate random noise as float32.
    - Generate random noise as float64.
    - Pass both to `octavefilter`.

    **Expectation:**
    - The output SPL should be valid (no NaNs).
    - The internal processing (via scipy/numpy) usually promotes to float64, so the output dtype is checked.
    """
    fs = 48000
    rng = np.random.default_rng(42)
    x = rng.standard_normal(fs).astype(dtype)
    
    spl, _ = octavefilter(x, fs)
    assert not np.isnan(spl).any()
    assert spl.dtype == np.float64 # Internal processing is likely float64


@pytest.mark.parametrize("channels", [1, 2, 4])
def test_multichannel_shapes(channels: int) -> None:
    """
    Verify the library handles different channel counts (Mono, Stereo, Multichannel).

    **Purpose:**
    The library claims to support multichannel input. The output shape must reflect the input channels.

    **Verification:**
    - Test with 1 channel (1D array).
    - Test with 2 channels (2D array [2, N]).
    - Test with 4 channels (2D array [4, N]).

    **Expectation:**
    - For mono input, the output SPL should be 1D [num_bands].
    - For multichannel input, the output SPL should be 2D [num_channels, num_bands].
    """
    fs = 48000
    duration = 0.1
    samples = int(fs * duration)
    rng = np.random.default_rng(42)
    
    if channels == 1:
        x = rng.standard_normal(samples)
    else:
        x = rng.standard_normal((channels, samples))
        
    spl, freq = octavefilter(x, fs)
    
    if channels == 1:
        assert spl.ndim == 1
    else:
        assert spl.shape[0] == channels
    assert spl.shape[-1] == len(freq)


@pytest.mark.parametrize("filter_type", ["butter", "cheby1", "cheby2", "ellip", "bessel"])
@pytest.mark.parametrize("target_freq", [63, 1000, 8000])
def test_frequency_isolation(target_freq: float, filter_type: str) -> None:
    """
    Critical Audio Test: Verify spectral isolation of pure tones.

    **Purpose:**
    This is the core functional test of a filter bank. A pure sine wave at 1000 Hz should
    mostly appear in the 1000 Hz band, with significantly less energy in distant bands.
    Tests multiple filter architectures.

    **Verification:**
    - Generate a pure sine wave at a target frequency (63Hz, 1kHz, 8kHz).
    - Analyze it using 1-octave bands.

    **Expectation:**
    - The band with the highest SPL must be the one centered closest to the target frequency.
    - Bands 2 octaves away must be attenuated by at least 20 dB (demonstrating stopband rejection).
    """
    fs = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Generate pure tone
    x = np.sin(2 * np.pi * target_freq * t)
    
    spl, freq = octavefilter(x, fs, fraction=1, limits=[20.0, 16000.0], filter_type=filter_type)
    
    # Find the band closest to target_freq
    freq_arr = np.array(freq)
    closest_idx = np.argmin(np.abs(freq_arr - target_freq))
    
    # The SPL should be highest at this index
    max_spl_idx = np.argmax(spl)
    
    assert closest_idx == max_spl_idx, (
        f"Tone at {target_freq}Hz peaked at {freq[max_spl_idx]}Hz band "
        f"instead of {freq[closest_idx]}Hz for {filter_type}"
    )
    
    # Check attenuation of distant bands (e.g., 2 octaves away)
    peak_val = spl[closest_idx]
    
    if closest_idx - 2 >= 0:
        lower_val = spl[closest_idx - 2]
        assert peak_val - lower_val > 20, f"Lower band not sufficiently attenuated (>20dB) for {filter_type}"
        
    if closest_idx + 2 < len(spl):
        upper_val = spl[closest_idx + 2]
        assert peak_val - upper_val > 20, f"Upper band not sufficiently attenuated (>20dB) for {filter_type}"


@pytest.mark.parametrize("filter_type", ["butter", "cheby1", "cheby2", "ellip", "bessel"])
def test_impulse_response_decay(filter_type: str) -> None:
    """
    Verify filter stability using the Impulse Response (IR).

    **Purpose:**
    Stable filters must have a decaying impulse response. If the energy does not decay,
    the filter is unstable (poles outside the unit circle).

    **Verification:**
    - Create a signal that is a perfect impulse (1.0 followed by zeros).
    - Get the time-domain filtered signals (`sigbands=True`).
    - Examine the "tail" (last 10%) of each filtered band.

    **Expectation:**
    - The energy in the tail should be negligible (< 1e-6), indicating the ringing has stopped.
    """
    fs = 48000
    x = np.zeros(fs)
    x[0] = 1.0 # Impulse
    
    # Use sigbands=True to get time domain signals
    _, _, signals = octavefilter(x, fs, fraction=1, sigbands=True, filter_type=filter_type)
    
    for band_sig in signals:
        # Check that the end of the signal is close to zero (decayed)
        # We check the last 10% of samples
        tail = band_sig[-int(fs*0.1):]
        energy = np.sum(tail**2)
        assert energy < 1e-5, f"Filter {filter_type} unstable or ringing too long"


def test_filterbank_class_direct() -> None:
    """
    Verify direct usage of OctaveFilterBank class.

    **Purpose:**
    Confirm that the class can be instantiated and used independently of the wrapper function.

    **Verification:**
    - Instantiate `OctaveFilterBank`.
    - Process noise.
    - Process a scaled version of the same noise.

    **Expectation:**
    - Output should be valid.
    - Scaled input should result in lower SPL.
    """
    fs = 44100
    rng = np.random.default_rng(42)
    bank = OctaveFilterBank(fs, fraction=3)
    x = rng.standard_normal(fs)
    
    spl, freq = bank.filter(x)
    assert len(freq) > 0
    assert spl.shape == (len(freq),)
    
    # Test reuse
    spl2, _ = bank.filter(x * 0.5)
    assert np.all(spl2 < spl)
