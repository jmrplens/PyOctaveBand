#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for error handling and edge cases across all modules."""

import numpy as np
import pytest

from phonometry import filters, metrology
from phonometry.filters.frequencies import nominal_frequencies


def test_octave_filter_bank_invalid_init() -> None:
    """Verify that OctaveFilterBank raises appropriate errors for invalid parameters.

    **Purpose:**
    Ensure that the class constructor robustly validates all input arguments to prevent
    unstable configurations or math errors during processing.

    **Verification:**
    - Pass invalid `fs`, `fraction`, `order`.
    - Pass invalid `limits` (too few elements, negative values, reversed).
    - Pass an unknown `filter_type`.

    **Expectation:**
    - Each invalid call must raise a `ValueError` with a specific error message.
    """
    with pytest.raises(ValueError, match="fs' must be positive"):
        filters.OctaveFilterBank(fs=0)

    with pytest.raises(ValueError, match="fraction' must be positive"):
        filters.OctaveFilterBank(fs=48000, fraction=-1)

    with pytest.raises(ValueError, match="order' must be positive"):
        filters.OctaveFilterBank(fs=48000, order=0)

    with pytest.raises(ValueError, match="list of two frequencies"):
        filters.OctaveFilterBank(fs=48000, limits=[1000])

    with pytest.raises(ValueError, match="Limit frequencies must be positive"):
        filters.OctaveFilterBank(fs=48000, limits=[-10, 1000])

    with pytest.raises(ValueError, match="less than the upper limit"):
        filters.OctaveFilterBank(fs=48000, limits=[2000, 1000])

    invalid_design = filters.FilterDesign(filter_type="invalid")
    with pytest.raises(ValueError, match="Invalid filter_type"):
        filters.OctaveFilterBank(fs=48000, design=invalid_design)


def test_weighting_filter_invalid() -> None:
    """Verify error handling for invalid weighting curves.

    **Purpose:**
    Ensure that only the supported curves (A, B, C, D, G, AU, Z) are accepted.

    **Verification:**
    - Call `weighting_filter` with an unsupported curve name.

    **Expectation:**
    - Raise `ValueError`.
    """
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    with pytest.raises(ValueError, match="Weighting curve must be 'A'"):
        filters.weighting_filter(x, 48000, curve="E")


def test_time_weighting_invalid() -> None:
    """Verify error handling for invalid time weighting modes.

    **Purpose:**
    Ensure that only standardized ballistic modes (Fast, Slow, Impulse) are accepted.

    **Verification:**
    - Call `time_weighting` with an unsupported mode string.

    **Expectation:**
    - Raise `ValueError`.
    """
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    with pytest.raises(ValueError, match="Invalid time weighting mode"):
        filters.time_weighting(x, 48000, mode="instant")


def test_linkwitz_riley_invalid() -> None:
    """Verify error handling for Linkwitz-Riley order.

    **Purpose:**
    Linkwitz-Riley filters require an even order to ensure correct phase alignment.

    **Verification:**
    - Call `linkwitz_riley` with an odd order.

    **Expectation:**
    - Raise `ValueError`.
    """
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    with pytest.raises(ValueError, match="order must be even"):
        filters.linkwitz_riley(x, 48000, freq=1000, order=3)


@pytest.mark.parametrize("order", [0, -2])
def test_linkwitz_riley_rejects_non_positive_order(order: int) -> None:
    """An even order of zero used to return both bands as the untouched input."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    with pytest.raises(ValueError, match="'order' must be a positive even integer"):
        filters.linkwitz_riley(x, 48000, freq=1000, order=order)


@pytest.mark.parametrize("freq", [0.0, -100.0, float("nan")])
def test_linkwitz_riley_rejects_non_positive_freq(freq: float) -> None:
    """A bad crossover frequency used to surface as a scipy message without 'freq'."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    with pytest.raises(ValueError, match="'freq' must be positive"):
        filters.linkwitz_riley(x, 48000, freq=freq)


def test_linkwitz_riley_rejects_freq_at_or_above_nyquist() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    with pytest.raises(ValueError, match="'freq' must be below the Nyquist frequency"):
        filters.linkwitz_riley(x, 48000, freq=24000.0)


def test_linkwitz_riley_rejects_non_positive_sample_rate() -> None:
    """fs=0 used to die as ZeroDivisionError; the other entry points name it."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    with pytest.raises(ValueError, match="Sample rate 'fs' must be positive"):
        filters.linkwitz_riley(x, 0, freq=1000)


def test_calculate_sensitivity_silent() -> None:
    """Verify error handling for silent reference signal.

    **Purpose:**
    Prevent division by zero during calibration when the reference signal is empty or silent.

    **Verification:**
    - Pass an array of zeros to `sensitivity`.

    **Expectation:**
    - Raise `ValueError`.
    """
    x = np.zeros(1000)
    with pytest.raises(ValueError, match="Reference signal is silent"):
        metrology.sensitivity(x)


def test_octave_filter_vs_class_consistency() -> None:
    """Verify that octave_filter function and OctaveFilterBank class yield identical results.

    **Purpose:**
    Ensure that the functional wrapper correctly proxies all arguments to the underlying
    class implementation, maintaining behavioral parity.

    **Verification:**
    - Process the same signal using both the function and the class.
    - Compare output SPL and frequencies.

    **Expectation:**
    - Arrays should be numerically identical.
    """
    fs = 44100
    rng = np.random.default_rng(42)
    x = rng.standard_normal(fs)
    fraction = 3
    order = 6
    filter_type = "butter"

    # 1. Using function
    spl_func, freq_func = filters.octave_filter(
        x,
        fs=fs,
        fraction=fraction,
        order=order,
        design=filters.FilterDesign(filter_type=filter_type),
    )

    # 2. Using class
    bank = filters.OctaveFilterBank(
        fs=fs,
        fraction=fraction,
        order=order,
        design=filters.FilterDesign(filter_type=filter_type),
    )
    spl_class, freq_class = bank.filter(x)

    assert np.allclose(spl_func, spl_class)
    assert np.allclose(freq_func, freq_class)


def test_single_sample_signal() -> None:
    """Verify handling of extremely short signals.

    **Purpose:**
    Ensure the library doesn't crash when provided with a single-sample signal,
    which might occur in edge-case stream processing.

    **Verification:**
    - Pass a single-element array to `octave_filter`.

    **Expectation:**
    - The code should return valid (though likely low) SPL values without crashing.
    """
    fs = 48000
    x = np.array([1.0])
    spl, freq = filters.octave_filter(x, fs)
    assert len(spl) == len(freq)
    assert not np.isnan(spl).any()


def test_multichannel_consistency() -> None:
    """Verify that processing 2 channels together is same as processing them separately.

    **Purpose:**
    Confirm that the multichannel implementation correctly isolates channels and does
    not introduce cross-channel artifacts.

    **Verification:**
    - Create two independent signals.
    - Process them separately.
    - Process them as a stereo pair.
    - Compare results.

    **Expectation:**
    - SPL values for each channel should match exactly.
    """
    fs = 16000
    rng = np.random.default_rng(42)
    x1 = rng.standard_normal(fs)
    x2 = rng.standard_normal(fs)
    x_stereo = np.vstack((x1, x2))

    bank = filters.OctaveFilterBank(fs, fraction=1)

    # Separate
    spl1, _ = bank.filter(x1)
    spl2, _ = bank.filter(x2)

    # Together
    spl_stereo, _ = bank.filter(x_stereo)

    assert np.allclose(spl_stereo[0], spl1)
    assert np.allclose(spl_stereo[1], spl2)


def test_octave_filter_bank_repr() -> None:
    """Verify OctaveFilterBank repr includes key configuration fields."""
    bank = filters.OctaveFilterBank(48000)
    representation = repr(bank)

    assert "OctaveFilterBank" in representation
    assert "fs=48000" in representation


def test_octavefilter_limits_none() -> None:
    """Verify None limits use package defaults and return nominal labels."""
    rng = np.random.default_rng(42)
    spl, _ = filters.octave_filter(rng.standard_normal(1000), 1000, limits=None)
    assert len(spl) > 0

    freq, freq_d, freq_u, labels = nominal_frequencies(1, limits=None)
    assert len(freq) > 0
    assert len(freq) == len(freq_d) == len(freq_u) == len(labels)
    assert all(isinstance(label, str) for label in labels)


def test_calculate_level_invalid_mode() -> None:
    """Verify invalid level calculation mode is rejected."""
    bank = filters.OctaveFilterBank(48000)

    signal = np.array([1.0])

    with pytest.raises(ValueError, match=r"Invalid mode\. Use 'rms'"):
        bank._calculate_level(signal, "invalid_mode")


def test_filter_rejects_non_string_mode() -> None:
    """A non-string mode used to die in str.lower, deep in level calculation."""
    bank = filters.OctaveFilterBank(48000)
    x = np.zeros(1000)
    with pytest.raises(TypeError, match="'mode' must be a string"):
        bank.filter(x, mode=None)  # type: ignore[arg-type]


def test_filter_rejects_unknown_mode_even_without_levels() -> None:
    """A misspelled mode was accepted in silence when no levels were computed."""
    bank = filters.OctaveFilterBank(48000)
    x = np.zeros(1000)
    with pytest.raises(ValueError, match="'mode' must be one of"):
        bank.filter(x, calculate_level=False, sigbands=True, mode="bogus")


def test_filter_rejects_three_dimensional_input() -> None:
    """A 3-D array used to fail in numpy broadcasting, or return truncated bands."""
    bank = filters.OctaveFilterBank(48000)
    cube = np.zeros((2, 2, 4800))
    with pytest.raises(ValueError, match="'x' must be a 1-D signal or a 2-D"):
        bank.filter(cube)


def test_octave_filter_rejects_empty_signal() -> None:
    """An empty signal used to reach scipy's sosfilt reshape, naming nothing."""
    empty = np.array([])
    with pytest.raises(ValueError, match="'x' must contain at least one sample"):
        filters.octave_filter(empty, 48000)


@pytest.mark.parametrize("limits", [1000, ["a", 5000]])
def test_octave_filter_rejects_malformed_limits(limits: object) -> None:
    """Malformed limits used to die in the cache key's float(), naming nothing."""
    x = np.zeros(1000)
    with pytest.raises(ValueError, match="'limits' must be a pair of frequencies"):
        filters.octave_filter(x, 48000, limits=limits)  # type: ignore[arg-type]


def test_octave_filter_rejects_malformed_limits_on_the_plotting_path() -> None:
    """The plot branch bypasses the design cache; it must refuse the same way."""
    x = np.zeros(1000)
    plot = filters.ResponsePlot(file="unused.png")
    with pytest.raises(ValueError, match="'limits' must be a pair of frequencies"):
        filters.octave_filter(
            x,
            48000,
            limits=1000,  # type: ignore[arg-type]
            response_plot=plot,
        )


def test_octave_filter_refuses_a_missing_signal_instead_of_answering_nan() -> None:
    """Verify that a lost signal is refused rather than measured.

    **Purpose:**
    A caller whose signal never arrived must be told so, not handed a reading.

    **Verification:**
    - Call the filter bank with None where the signal goes.

    **Expectation:**
    - A `ValueError` naming `'x'`. None used to convert to a one-sample NaN
      array, so this call returned eight bands of NaN and refused nothing.
    """
    with pytest.raises(ValueError, match="'x' must be a signal, not None"):
        filters.octave_filter(None, 48000)  # type: ignore[arg-type]


def test_a_refusal_names_the_parameter_of_the_entry_point_that_was_called() -> None:
    """Verify the parameter named in the message is one the caller can find.

    **Purpose:**
    The resolver sits under every entry point that consumes a recording, so a
    message hard-coded to `'x'` would name an argument that most of them do
    not have.

    **Verification:**
    - Reach the same refusal through `sensitivity`, whose signal parameter is
      called `ref_signal`.

    **Expectation:**
    - The message names `'ref_signal'`, not `'x'`.
    """
    poisoned = np.sin(2 * np.pi * 1000.0 * np.arange(4800) / 48000.0)
    poisoned[100] = np.nan
    with pytest.raises(
        ValueError, match="'ref_signal' must contain only finite samples"
    ):
        metrology.sensitivity(poisoned, fs=48000)


def test_process_bands_without_level_calculation() -> None:
    """Verify internal band processing can skip level calculation."""
    bank = filters.OctaveFilterBank(48000)
    x = np.zeros((bank.num_bands, 100))

    spl, filtered = bank._process_bands(
        x,
        num_channels=bank.num_bands,
        calculate_level=False,
        sigbands=True,
    )

    assert spl is None
    assert filtered is not None
