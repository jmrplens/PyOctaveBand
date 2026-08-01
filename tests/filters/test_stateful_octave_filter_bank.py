import numpy as np
import pytest


@pytest.mark.parametrize("block_size", [8, 256, 1024])
def test_block_processing_matches_full_signal(block_size: int):
    from phonometry import OctaveFilterBank
    """
    Ensure that block-wise processing with preserved filter state
    produces the same result as filtering the full signal at once.
    """

    rng = np.random.default_rng(42)

    fs = 48000
    duration = 0.2
    n_samples = int(fs * duration)

    # Random signal (deterministic via seed)
    signal = rng.standard_normal(n_samples)

    # --- Full signal processing (reference) ---
    full_filter = OctaveFilterBank(fs=fs, resample=False)
    _full_output_spl, _, full_output_signal = full_filter.filter(signal, sigbands=True, detrend=False)

    # --- Block-wise processing ---
    block_filter = OctaveFilterBank(fs=fs, resample=False, stateful=True, steady_ic=False)

    outputs = []

    for start in range(0, n_samples, block_size):
        block = signal[start:start + block_size]
        _block_output_spl, _, block_output_signal = block_filter.filter(block, sigbands=True, detrend=False)
        outputs.append(block_output_signal)

    block_output_signal = np.concatenate(outputs, axis=-1)
    full_output_signal = np.array(full_output_signal)

    # --- Comparison ---
    # Use tight tolerance; sosfilt should be numerically identical
    np.testing.assert_allclose(
        block_output_signal,
        full_output_signal,
        rtol=1e-10,
        atol=1e-12,
    )

def test_resample_and_stateful():
    from phonometry.filters.core import OctaveFilterBank
    with pytest.raises(ValueError):
        OctaveFilterBank(48000, resample=True, stateful=True)


def test_stateful_steady_ic_initialization():
    from phonometry.filters.core import OctaveFilterBank
    # Create a stateful filter bank with steady_ic=True
    bank = OctaveFilterBank(
        fs=48000,
        stateful=True,
        steady_ic=True,
        order=2,  # small order is enough for coverage
        fraction=1,  # 1-octave
        resample=False  # avoid the resampling branch
    )

    # With lazy initialization, zi is allocated on first filter() call.
    # Verify zi is properly initialized after processing a block.
    rng = np.random.default_rng(42)
    test_signal = rng.standard_normal(1024)
    # steady_ic implies detrending, whose block-processing advisory is
    # expected here: assert it rather than leak it to the run summary.
    from phonometry.filters.core import FilterBankWarning

    with pytest.warns(FilterBankWarning, match="Detrending"):
        bank.filter(test_signal)

    # Check that zi is a list of numpy arrays with the expected shape
    for idx, zi in enumerate(bank.zi):
        # zi should have 3 dimensions: (n_sections, n_channels, 2)
        assert isinstance(zi, np.ndarray)
        assert zi.ndim == 3
        n_sections = bank.sos[idx].shape[0]
        assert zi.shape[0] == n_sections
        assert zi.shape[1] == 1  # single-channel input
        assert zi.shape[2] == 2

def test_stateful_multichannel():
    """Test that stateful processing works with multichannel (e.g. stereo) input."""
    from phonometry.filters.core import OctaveFilterBank
    rng = np.random.default_rng(42)
    n_channels = 4
    fs = 48000
    block_size = 1024

    bank = OctaveFilterBank(
        fs=fs, stateful=True, steady_ic=True,
        order=2, fraction=1, resample=False,
    )
    stereo_block = rng.standard_normal((n_channels, block_size))
    _spl, _ = bank.filter(stereo_block, detrend=False)

    # zi must have correct multichannel shape after first call
    for idx, zi in enumerate(bank.zi):
        n_sections = bank.sos[idx].shape[0]
        assert zi.shape == (n_sections, n_channels, 2)

    # Second block should work without error (state reused)
    spl2, _ = bank.filter(rng.standard_normal((n_channels, block_size)), detrend=False)
    assert spl2 is not None
    assert spl2.shape[0] == n_channels

def test_detrend_stateful_warning():
    from phonometry.filters.core import OctaveFilterBank
    rng = np.random.default_rng(42)

    fs = 48000
    duration = 2.0
    n_samples = int(fs * duration)

    # Random signal (deterministic via seed)
    signal = rng.standard_normal(n_samples)

    bank = OctaveFilterBank(fs, stateful=True, resample=False)
    with pytest.warns(UserWarning, match="block processing"):
        bank.filter(signal, detrend=True)

