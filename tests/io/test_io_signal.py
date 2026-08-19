#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the Signal result object: array behaviour, metadata, plot."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from phonometry import signals
from phonometry.io import Signal, SignalOrigin

FS = 48000


def _tone(amp: float = 1.0, f0: float = 1000.0, seconds: float = 0.5) -> np.ndarray:
    t = np.arange(int(FS * seconds)) / FS
    result: np.ndarray = amp * np.sin(2 * np.pi * f0 * t)
    return result


# ---------------------------------------------------------------------------
# Array protocol: the object must stand for the array it wraps
# ---------------------------------------------------------------------------

def test_mono_signal_is_a_1d_array() -> None:
    x = _tone()
    sig = Signal(data=x, fs=FS)
    out = np.asarray(sig)
    assert out.ndim == 1
    assert out.shape == (x.size,)
    np.testing.assert_array_equal(out, x)
    # Storage is canonical (channels, samples) even for one channel.
    assert sig.data.shape == (1, x.size)


def test_multichannel_signal_is_a_2d_array() -> None:
    x = np.stack([_tone(), 0.5 * _tone()])
    sig = Signal(data=x, fs=FS)
    out = np.asarray(sig)
    assert out.shape == (2, x.shape[-1])
    np.testing.assert_array_equal(out, x)


def test_forwarding_matches_the_array_view() -> None:
    """len/shape/indexing must agree with np.asarray(sig), mono and multi."""
    mono = Signal(data=_tone(), fs=FS)
    assert len(mono) == np.asarray(mono).shape[0]
    assert mono.shape == np.asarray(mono).shape
    assert mono.ndim == 1
    assert mono[3] == np.asarray(mono)[3]
    stereo = Signal(data=np.stack([_tone(), _tone(0.5)]), fs=FS)
    assert len(stereo) == 2
    assert stereo.ndim == 2
    np.testing.assert_array_equal(stereo[1], 0.5 * _tone())
    assert stereo.dtype == np.float64
    assert stereo.size == 2 * _tone().size


def test_existing_levels_function_accepts_the_object() -> None:
    """leq(Signal) equals leq(x) with the Signal's own calibration applied.

    The overloaded signature (the signals.levels pilot) takes the factor a
    calibrated Signal carries when no explicit one is given; an explicit
    argument still wins, and an uncalibrated Signal reads in digital units
    exactly as a bare array does.
    """
    x = _tone()
    sig = Signal(data=x, fs=FS, calibration_factor=2.0)
    assert signals.leq(sig) == signals.leq(x, calibration_factor=2.0)
    assert signals.leq(sig, calibration_factor=5.0) == signals.leq(
        x, calibration_factor=5.0
    )
    assert signals.leq(Signal(data=x, fs=FS)) == signals.leq(x)


def test_signal_properties() -> None:
    sig = Signal(data=np.zeros((3, 2 * FS)), fs=FS)
    assert sig.n_channels == 3
    assert sig.n_samples == 2 * FS
    assert sig.duration == pytest.approx(2.0)


def test_metadata_fields_default_to_none() -> None:
    sig = Signal(data=_tone(), fs=FS)
    assert sig.calibration_factor is None
    assert sig.channel_labels is None
    assert sig.provenance is None
    assert sig.source is None


def test_source_metadata_travels_with_the_signal() -> None:
    source = SignalOrigin(
        path="m.wav", container="WAV", format_name="PCM",
        bit_depth=24, lossy=False,
    )
    sig = Signal(data=_tone(), fs=FS, source=source)
    assert sig.source is not None
    assert sig.source.bit_depth == 24
    assert not sig.source.lossy


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_construction_is_rejected() -> None:
    tone = _tone()
    with pytest.raises(ValueError, match="channel labels"):
        Signal(data=np.zeros((2, 10)), fs=FS, channel_labels=("FL",))
    with pytest.raises(ValueError, match="fs"):
        Signal(data=tone, fs=0)
    with pytest.raises(ValueError, match="calibration_factor"):
        Signal(data=tone, fs=FS, calibration_factor=0.0)
    with pytest.raises(ValueError, match="1-D or"):
        Signal(data=np.zeros((2, 2, 2)), fs=FS)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 0, -FS])
def test_non_finite_or_non_positive_fs_is_rejected(bad: float) -> None:
    """NaN and infinity pass a bare <= 0 check; the constructor must not."""
    tone = _tone()
    with pytest.raises(ValueError, match="fs must be a positive finite"):
        Signal(data=tone, fs=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), float("-inf"), 0.0, -0.5]
)
def test_non_finite_or_non_positive_calibration_is_rejected(bad: float) -> None:
    """A NaN calibration would flow into levels as a computed-looking wrong number."""
    tone = _tone()
    with pytest.raises(ValueError, match="calibration_factor must be a positive finite"):
        Signal(data=tone, fs=FS, calibration_factor=bad)


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

def test_plot_calibrated_axis_is_pascals() -> None:
    sig = Signal(data=_tone(), fs=FS, calibration_factor=0.5)
    ax = sig.plot()
    assert ax.get_ylabel() == "Sound pressure [Pa]"
    assert ax.get_xlabel() == "Time [s]"
    line = ax.get_lines()[0]
    # The drawn amplitude is the calibrated one.
    assert float(np.max(np.abs(line.get_ydata()))) == pytest.approx(0.5, abs=1e-3)
    plt.close("all")


def test_plot_uncalibrated_axis_is_full_scale() -> None:
    ax = Signal(data=_tone(), fs=FS).plot()
    assert ax.get_ylabel() == "Amplitude [FS]"
    assert ax.get_title() == "Waveform"
    plt.close("all")


def test_plot_multichannel_legend_uses_channel_labels() -> None:
    sig = Signal(
        data=np.stack([_tone(), _tone(0.5)]), fs=FS,
        channel_labels=("FL", "FR"),
    )
    ax = sig.plot()
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["FL", "FR"]
    plt.close("all")


def test_plot_spanish_labels() -> None:
    ax = Signal(data=_tone(), fs=FS, calibration_factor=1.0).plot(language="es")
    assert ax.get_xlabel() == "Tiempo [s]"
    assert ax.get_ylabel() == "Presión sonora [Pa]"
    assert ax.get_title() == "Forma de onda calibrada"
    plt.close("all")


def test_plot_rejects_unknown_language() -> None:
    sig = Signal(data=_tone(), fs=FS)
    with pytest.raises(ValueError, match="language"):
        sig.plot(language="fr")
