#  Copyright (c) 2026. Jose Manuel Requena Plens

"""What the loudness ``.plot()`` renderers draw (ISO 532-1 and ISO 532-2).

The two standards answer the same question on two different scales, and the
figure is where that shows: ISO 532-1 puts specific loudness on the Bark scale
from 0,1 to 24 Bark, ISO 532-2 puts it on the ERB-number (Cam) scale from 1,8 to
38,9 Cam, and a time-varying ISO 532-1 record adds a second panel with loudness
against time. These tests read the drawn artists back and check they echo the
fields of the result, axis units included; the generic plot contract (kwargs,
external axes, the soft matplotlib dependency) lives in
``tests/test_result_plots.py``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from result_factories import FS, RNG, _zwicker_stationary

import phonometry as ph


# --------------------------------------------------------------------------
# Zwicker loudness
# --------------------------------------------------------------------------
def test_zwicker_stationary_returns_single_axes_with_specific_curve() -> None:
    res = _zwicker_stationary()
    ax = res.plot()
    assert not isinstance(ax, np.ndarray)
    ydata = ax.lines[0].get_ydata()
    np.testing.assert_allclose(ydata, res.specific)
    xdata = ax.lines[0].get_xdata()
    assert xdata[0] == pytest.approx(0.1)
    assert xdata[-1] == pytest.approx(24.0)
    assert "Bark" in ax.get_xlabel()
    assert "sone/Bark" in ax.get_ylabel()
    plt.close("all")


def test_zwicker_time_varying_returns_two_panels() -> None:
    sig = RNG.standard_normal(FS) * 0.02  # 1 s of noise
    res = ph.psychoacoustics.loudness_zwicker(sig, FS, stationary=False)
    assert res.loudness_vs_time is not None
    axes = res.plot()
    assert isinstance(axes, np.ndarray)
    assert axes.size == 2
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), res.specific)
    np.testing.assert_allclose(axes[1].lines[0].get_ydata(), res.loudness_vs_time)
    plt.close("all")


# --------------------------------------------------------------------------
# Moore-Glasberg loudness (ISO 532-2)
# --------------------------------------------------------------------------
def test_moore_glasberg_returns_single_axes_with_specific_curve() -> None:
    res = ph.psychoacoustics.loudness_moore_glasberg_from_spectrum([(1000.0, 60.0)])
    ax = res.plot()
    assert not isinstance(ax, np.ndarray)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.specific)
    xdata = ax.lines[0].get_xdata()
    assert xdata[0] == pytest.approx(1.8)
    assert xdata[-1] == pytest.approx(38.9)
    assert "Cam" in ax.get_xlabel()
    assert "sone/Cam" in ax.get_ylabel()
    plt.close("all")
