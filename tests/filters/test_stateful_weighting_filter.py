import numpy as np
import pytest


@pytest.mark.parametrize("block_size", [8, 256, 1024])
@pytest.mark.parametrize("filter_type", ["A", "C", "Z"])
def test_weighting_filter_block_processing_matches_full_signal(block_size: int, filter_type: str):
    from phonometry import WeightingFilter
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

    # high_accuracy=False: the reference must use the same plain bilinear
    # design as the stateful filter (this test verifies state continuity,
    # not absolute response accuracy).
    stateless_wf = WeightingFilter(fs, filter_type, high_accuracy=False)
    stateless_output = stateless_wf.filter(signal)

    stateful_wf = WeightingFilter(fs, filter_type, stateful=True)

    outputs = []
    for start in range(0, n_samples, block_size):
        block = signal[start:start + block_size]
        block_output = stateful_wf.filter(block)
        outputs.append(block_output)

    stateful_output = np.concatenate(outputs, axis=0)

    np.testing.assert_allclose(
        stateful_output,
        stateless_output,
        rtol=1e-10,
        atol=1e-12,
    )

def test_weighting_filter_steady_ic_initialization():
    from phonometry import WeightingFilter
    # Create a stateful weighting filter with steady_ic=True
    wf = WeightingFilter(fs=48000, stateful=True, steady_ic=True)

    # Lazy init: zi is empty until first filter() call
    assert wf.zi.size == 0

    # Trigger lazy init with 1D signal
    x = np.zeros(100)
    wf.filter(x)

    n_sections = wf.sos.shape[0]
    assert wf.zi.shape == (n_sections, 2)


def test_weighting_filter_steady_ic_initialization_multichannel():
    from phonometry import WeightingFilter

    wf = WeightingFilter(fs=48000, stateful=True, steady_ic=True)
    x = np.zeros((2, 100))

    y = wf.filter(x)

    n_sections = wf.sos.shape[0]
    assert y.shape == x.shape
    assert wf.zi.shape == (n_sections, 2, 2)


def test_weighting_filter_multichannel_to_mono_transition():
    from phonometry import WeightingFilter
    """zi must reinit when input switches from multichannel to 1D."""
    rng = np.random.default_rng(77)
    fs = 48000
    wf = WeightingFilter(fs, "A", stateful=True)

    # First call: multichannel (4 channels)
    x_multi = rng.standard_normal((4, 1200))
    wf.filter(x_multi)
    assert wf.zi.ndim == 3

    # Second call: 1D — must not crash
    x_mono = rng.standard_normal(1200)
    y = wf.filter(x_mono)
    assert wf.zi.ndim == 2
    assert y.shape == x_mono.shape


def test_weighting_filter_multichannel():
    from phonometry import WeightingFilter
    """Stateful block-wise multichannel weighting must match full-signal processing."""
    rng = np.random.default_rng(99)
    fs = 48000
    n_channels = 4
    n_samples = 4800
    block_size = 1200

    x = rng.standard_normal((n_channels, n_samples))

    # Full-signal reference (stateless, same plain bilinear design as stateful)
    ref_wf = WeightingFilter(fs, "A", high_accuracy=False)
    ref_out = ref_wf.filter(x)

    # Block-wise stateful
    stateful_wf = WeightingFilter(fs, "A", stateful=True)
    blocks = []
    for start in range(0, n_samples, block_size):
        block = x[:, start:start + block_size]
        blocks.append(stateful_wf.filter(block))

    stateful_out = np.concatenate(blocks, axis=-1)
    np.testing.assert_allclose(stateful_out, ref_out, rtol=1e-10, atol=1e-12)


def test_stateful_weighting_invalid_fs_raises():
    """Non-positive sample rates must be rejected at construction."""
    from phonometry import WeightingFilter
    with pytest.raises(ValueError, match="must be positive"):
        WeightingFilter(fs=0, stateful=True)
    with pytest.raises(ValueError, match="must be positive"):
        WeightingFilter(fs=-48000, stateful=True)


def test_stateful_weighting_high_accuracy_raises():
    """high_accuracy resampling is incompatible with block processing."""
    from phonometry import WeightingFilter
    with pytest.raises(ValueError, match="not compatible with stateful"):
        WeightingFilter(fs=48000, stateful=True, high_accuracy=True)


def test_stateful_weighting_invalid_curve_raises():
    """Unknown weighting curves must be rejected at construction."""
    from phonometry import WeightingFilter
    with pytest.raises(ValueError, match="must be 'A', 'B', 'C', 'D', 'G', 'AU' or 'Z'"):
        WeightingFilter(fs=48000, curve="E", stateful=True)

