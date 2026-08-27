#  Copyright (c) 2026. Jose Manuel Requena Plens

"""What the emission ``.plot()`` renderers draw (ISO 3744/3741 and ISO 9614).

Sound power determined from pressure, from a reverberation room, or from
intensity is the same quantity reached three ways, and the figure has to say
which bands the method could not deliver. A band whose net intensity is negative
means power is flowing *into* the measurement surface from somewhere else, so
its bar is hatched rather than silently dropped; a reverberation band with no
decay time is drawn at zero. The companion intensity figure puts ``Lp`` and
``LI`` on the primary axis and the pressure-intensity index on a twin, whose bars
must scale their width with the band centre so a constant width does not vanish
at the top of a logarithmic frequency axis.

These are the content assertions. The generic plot contract lives in
``tests/test_result_plots.py``.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from result_factories import (
    FS,
    RNG,
    _intensity,
    _intensity_power_negative,
    _intensity_wide,
    _reverb_power,
    _sound_power,
)

import phonometry as ph


# --------------------------------------------------------------------------
# Sound power (pressure / reverberation / intensity)
# --------------------------------------------------------------------------
def test_sound_power_bars_match_lw_and_annotate_lwa() -> None:
    res = _sound_power()
    ax = res.plot()
    heights = [p.get_height() for p in ax.patches]
    np.testing.assert_allclose(heights, res.sound_power_level)
    assert f"{res.sound_power_level_a:.1f}" in ax.get_title()
    plt.close("all")


def test_reverberation_power_plot_smoke() -> None:
    res = _reverb_power()
    ax = res.plot()
    heights = [p.get_height() for p in ax.patches]
    np.testing.assert_allclose(heights, np.nan_to_num(res.sound_power_level))
    plt.close("all")


def test_intensity_power_marks_negative_band() -> None:
    res = _intensity_power_negative()
    assert bool(res.negative_band[1])
    ax = res.plot()
    hatched = [p for p in ax.patches if p.get_hatch()]
    assert len(hatched) == int(np.count_nonzero(res.negative_band))
    plt.close("all")


# --------------------------------------------------------------------------
# Sound intensity (Lp vs LI)
# --------------------------------------------------------------------------
def test_intensity_plots_lp_and_li_with_index_twin() -> None:
    res = _intensity()
    ax = res.plot()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.pressure_level)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), res.intensity_level)
    # twin axis carries the pressure-intensity index bars.
    twins = [a for a in ax.figure.axes if a is not ax]
    assert twins, "expected a twin axis for the pressure-intensity index"
    plt.close("all")


def test_intensity_bar_width_scales_with_frequency_on_log_axis() -> None:
    # On a log frequency axis a constant linear bar width vanishes at high
    # frequency; the index bars must instead scale their width with each
    # centre frequency so the drawn width/f ratio is one constant.
    res = _intensity_wide()
    ax = res.plot()
    twin = next(a for a in ax.figure.axes if a is not ax)
    assert len(twin.patches) == 3
    centers = [p.get_x() + p.get_width() / 2.0 for p in twin.patches]
    ratios = [p.get_width() / c for p, c in zip(twin.patches, centers, strict=True)]
    # width/f is the same constant at 100 Hz and at 10 kHz (and in between).
    assert min(centers) == pytest.approx(100.0)
    assert max(centers) == pytest.approx(10000.0)
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-9)
    plt.close("all")


def test_intensity_without_band_data_raises() -> None:
    p1 = RNG.standard_normal(FS)
    res = ph.emission.sound_intensity(p1, np.roll(p1, 1), FS, spacing=0.012)
    assert res.frequency is None
    with pytest.raises(ValueError, match=r"plot\(\) needs per-band intensity data"):
        res.plot()
