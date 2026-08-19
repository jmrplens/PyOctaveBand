#  Copyright (c) 2026. Jose Manuel Requena Plens

"""What the room-acoustics ``.plot()`` renderers draw (ISO 3382-1/-2/-3).

Three results, one measurement chain: the Schroeder backward integration of an
impulse response, the decay times fitted on it band by band, and the spatial
decay of speech over an open-plan floor. The figures have to keep the chain
honest. A decay time whose evaluation range never spanned its stated dynamic
range is not a decay time, so those bars are drawn hatched and muted rather than
dropped, which is the only way a reader sees that the band was rejected; the
decay-curve figure overlays the fitted slopes on the integrated curve; and the
open-plan figure carries the ISO 3382-3 regression itself, so the line must read
``Lp,A,S,4m`` at 4 m and fall by ``D2,S`` over the doubling to 8 m, with ``rD``
and ``rP`` marked at their distances.

These are the content assertions. The generic plot contract lives in
``tests/test_result_plots.py``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from result_factories import (
    FS,
    _exp_ir,
    _open_plan,
    _room,
    _room_with_one_invalid_band,
)

import phonometry as ph
from phonometry._plot import common as _plotting


# --------------------------------------------------------------------------
# Room acoustics
# --------------------------------------------------------------------------
def test_room_acoustics_two_panels_and_bar_heights() -> None:
    res = _room(limits=[250, 2000])
    axes = res.plot()
    assert isinstance(axes, np.ndarray)
    assert axes.size == 2
    ax_times = axes[0]
    n = res.t30.size
    # 3 grouped bars (EDT/T20/T30) per band.
    assert len(ax_times.patches) == 3 * n
    assert "time" in ax_times.get_ylabel().lower()
    plt.close("all")


def test_room_acoustics_invalid_bands_are_hatched() -> None:
    # Deterministic: only the 500 Hz band is flagged invalid on all three
    # decay-time series, so exactly those three bars must be hatched/greyed.
    res = _room_with_one_invalid_band()
    invalid_idx = 1
    n = res.t30.size
    axes = res.plot()
    patches = axes[0].patches
    assert len(patches) == 3 * n  # EDT/T20/T30 grouped bars
    hatched = [p for p in patches if p.get_hatch()]
    greyed = [
        p for p in patches
        if p.get_facecolor()[:3] == plt.matplotlib.colors.to_rgb(_plotting._C_MUTED)
    ]
    # Exactly one bar per series (EDT/T20/T30) is invalid -> 3 hatched/greyed.
    assert len(hatched) == 3
    assert len(greyed) == 3
    # And they are the bars sitting over the invalid (500 Hz) band position.
    for p in hatched:
        assert round(p.get_x() + p.get_width() / 2.0) == invalid_idx
    plt.close("all")


def test_room_acoustics_single_axes_composition() -> None:
    res = _room(limits=[250, 2000])
    _fig, ax = plt.subplots()
    out = res.plot(ax=ax)
    assert out is ax
    plt.close("all")


# --------------------------------------------------------------------------
# Schroeder decay curve (backward compat + plot)
# --------------------------------------------------------------------------
def test_decay_curve_is_dataclass_and_unpacks_like_tuple() -> None:
    ir = _exp_ir()
    dc = ph.room.decay_curve(ir, FS)
    assert isinstance(dc, ph.room.DecayCurve)
    time, level = ph.room.decay_curve(ir, FS)  # backward-compatible unpacking
    np.testing.assert_array_equal(time, dc.time)
    np.testing.assert_array_equal(level, dc.level)
    assert dc.band is None


def test_decay_curve_records_band() -> None:
    dc = ph.room.decay_curve(_exp_ir(), FS, band=500.0)
    assert dc.band == 500.0


def test_decay_curve_plot_has_curve_and_fit_overlays() -> None:
    dc = ph.room.decay_curve(_exp_ir(seconds=1.0, t60=0.6), FS)
    ax = dc.plot()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), dc.level)
    labels = [ln.get_label() for ln in ax.lines]
    assert any("fit" in str(lbl) for lbl in labels)
    assert "s]" in ax.get_xlabel()
    plt.close("all")


def test_decay_curve_plot_without_fits() -> None:
    dc = ph.room.decay_curve(_exp_ir(), FS)
    ax = dc.plot(fits=False)
    labels = [str(ln.get_label()) for ln in ax.lines]
    assert not any("fit" in lbl for lbl in labels)
    plt.close("all")


# --------------------------------------------------------------------------
# Open-plan spatial decay (ISO 3382-3)
# --------------------------------------------------------------------------
def test_open_plan_plot_line_and_markers() -> None:
    res = _open_plan()
    ax = res.plot()
    # the regression line passes through Lp,A,S,4m at 4 m.
    line = ax.lines[0]
    x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    at4 = float(np.interp(4.0, x, y))
    assert at4 == pytest.approx(res.lp_as_4m, abs=0.05)
    # slope over one doubling equals -D2,S.
    at8 = float(np.interp(8.0, x, y))
    assert at4 - at8 == pytest.approx(res.d2s, abs=0.05)
    # rD / rP are marked as vertical lines at their distances.
    vlines = [
        np.asarray(ln.get_xdata())[0] for ln in ax.lines
        if np.asarray(ln.get_xdata()).size == 2
        and np.asarray(ln.get_xdata())[0] == np.asarray(ln.get_xdata())[1]
    ]
    assert any(v == pytest.approx(res.rd) for v in vlines)
    assert any(v == pytest.approx(res.rp) for v in vlines)
    plt.close("all")


def test_open_plan_plot_without_regression_raises() -> None:
    bare = ph.room.OpenPlanResult(
        d2s=float("nan"), lp_as_4m=float("nan"), rd=float("nan"), rp=float("nan")
    )
    with pytest.raises(ValueError, match="regression"):
        bare.plot()
