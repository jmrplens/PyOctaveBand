#  Copyright (c) 2020. Jose Manuel Requena Plens

"""Basic test and usage example for phonometry."""

import numpy as np

import phonometry
from phonometry._version import __version__


def test_package_version_is_public() -> None:
    assert phonometry.__version__ == __version__


def test_octave_filter_basic() -> None:
    """Basic sanity check for the octave filter.

    **Purpose:**
    Ensure the function runs without errors for a simple, standard use case and returns
    output of the expected shape.

    **Verification:**
    - Generate a multi-tone signal composed of sines at various frequencies.
    - Apply a 1/3 octave filter.

    **Expectation:**
    - The function should return two arrays: SPL values and center frequencies.
    - The lengths of these arrays must match.
    - The SPL array should not contain NaNs.
    """
    # Configuration
    fs = 48000
    duration = 1.0  # Reduced for faster testing

    # Generate multi-tone signal
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    freqs = [20, 100, 500, 2000, 4000, 15000]
    y = 100 * np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)

    # 1. Filter and get only SPL spectrum
    spl, freq = phonometry.filters.octave_filter(
        y, fs=fs, fraction=3, order=6, limits=[12, 20000]
    )

    assert len(spl) == len(freq)
    assert len(freq) > 0
    assert not np.isnan(spl).any()


def test_octave_filter_sigbands() -> None:
    """Test the retrieval of time-domain filtered signals (`sigbands=True`).

    **Purpose:**
    Verify that the user can retrieve the actual time-domain waveform for each frequency band,
    not just the SPL levels. This is useful for advanced analysis or reconstruction.

    **Verification:**
    - Filter a simple sine wave.
    - Set `sigbands=True`.

    **Expectation:**
    - The function should return a third element: a list of numpy arrays `xb`.
    - `xb` should contain one array for each frequency band.
    - Each array in `xb` should have the same length as the input signal (indicating correct upsampling/restoration).
    """
    fs = 48000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    y = np.sin(2 * np.pi * 1000 * t)

    # 2. Filter and get signals in time-domain bands
    _, freq, xb = phonometry.filters.octave_filter(
        y, fs=fs, fraction=1, order=6, limits=[500, 2000], sigbands=True
    )

    assert len(xb) == len(freq)
    assert xb[0].shape == y.shape


def test_detrend_acts_on_the_input_once_not_per_band() -> None:
    """`detrend` removes the input's DC offset before filtering, not per band.

    Pins what the parameter tables promise. The DC offset never reaches a
    band signal either way, because the bandpass already rejects it; what
    `detrend` changes is the level, by keeping the offset's onset step out
    of the filters in the first place.
    """
    fs = 48000
    t = np.arange(2 * fs) / fs
    y = np.sin(2 * np.pi * 1000 * t) + 5.0  # 1 kHz tone riding a DC offset

    bank = phonometry.filters.OctaveFilterBank(fs=fs, fraction=1, limits=[12, 20000])
    spl_on, _, bands_on = bank.filter(y, sigbands=True, detrend=True)
    spl_off, _, bands_off = bank.filter(y, sigbands=True, detrend=False)

    # The lowest band swings by tens of dB: the flag is doing real work.
    assert spl_off[0] - spl_on[0] > 20.0
    # Yet no band signal carries the offset in either mode, so the operation
    # cannot be the per-band DC removal the tables used to describe.
    assert abs(float(np.mean(bands_on[0]))) < 1e-4
    assert abs(float(np.mean(bands_off[0]))) < 1e-4


def test_octave_filter_reuses_cached_bank(monkeypatch) -> None:
    """Repeated octave_filter calls with identical params must not redesign the bank."""
    from phonometry.filters.core import OctaveFilterBank

    phonometry.filters.core._cached_filter_bank.cache_clear()
    calls = {"n": 0}
    original_init = OctaveFilterBank.__init__

    def counting_init(self, *args, **kwargs):
        calls["n"] += 1
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(OctaveFilterBank, "__init__", counting_init)

    x = np.random.default_rng(0).standard_normal(4800)
    phonometry.filters.octave_filter(x, 48000, fraction=3)
    phonometry.filters.octave_filter(x, 48000, fraction=3)
    assert calls["n"] == 1

    phonometry.filters.octave_filter(
        x, 48000, fraction=1
    )  # different params -> new bank
    assert calls["n"] == 2
    phonometry.filters.core._cached_filter_bank.cache_clear()


def test_octave_filter_cached_results_identical() -> None:
    """The cached bank must return bit-identical results across calls."""
    phonometry.filters.core._cached_filter_bank.cache_clear()
    x = np.random.default_rng(1).standard_normal(4800)
    spl1, f1 = phonometry.filters.octave_filter(x, 48000, fraction=3)
    spl2, f2 = phonometry.filters.octave_filter(x, 48000, fraction=3)
    np.testing.assert_array_equal(spl1, spl2)
    assert f1 == f2


def test_octave_filter_freq_list_is_mutation_safe() -> None:
    """Mutating the returned freq list must not corrupt the cached bank."""
    phonometry.filters.core._cached_filter_bank.cache_clear()
    x = np.random.default_rng(2).standard_normal(4800)
    _, freq1 = phonometry.filters.octave_filter(x, 48000, fraction=1)
    freq1[0] = -999.0  # caller mutates the returned list
    _, freq2 = phonometry.filters.octave_filter(x, 48000, fraction=1)
    assert freq2[0] != -999.0
    phonometry.filters.core._cached_filter_bank.cache_clear()
