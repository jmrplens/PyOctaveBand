#  Copyright (c) 2026. Jose Manuel Requena Plens

"""What the Monte Carlo ``.plot()`` renderer draws (GUM Supplement 1).

The propagation of distributions of JCGM 101 answers with a sampled output
distribution, not with a single standard uncertainty, so the figure is the
histogram of those samples with the coverage interval shaded across it. Both
halves are checked here: the bars exist, and the shaded span starts and ends
exactly at the interval the result reports. The samples are kept only when the
caller asks for them (``keep_samples=True``), so the figure has to refuse
politely when they were discarded.

These are the content assertions. The generic plot contract lives in
``tests/test_result_plots.py``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from result_factories import _MC_QUANTITIES, _monte_carlo

import phonometry as ph


# --------------------------------------------------------------------------
# Monte Carlo output distribution (GUM Supplement 1)
# --------------------------------------------------------------------------
def test_monte_carlo_plot_histogram_and_interval() -> None:
    res = _monte_carlo()
    assert res.samples is not None
    assert res.samples.size == res.trials
    ax = res.plot()
    bars = [
        p for p in ax.patches
        if "coverage interval" not in str(p.get_label())
    ]
    assert bars, "expected histogram bars"
    # the coverage-interval axvspan matches the result's interval.
    spans = [
        p for p in ax.patches
        if "coverage interval" in str(p.get_label())
    ]
    assert spans, "expected the coverage-interval axvspan"
    low, high = res.interval
    assert spans[0].get_x() == pytest.approx(low)
    assert spans[0].get_x() + spans[0].get_width() == pytest.approx(high)
    plt.close("all")


def test_monte_carlo_plot_without_samples_raises() -> None:
    res = ph.metrology.monte_carlo(
        lambda a, b, c: a + b + c, _MC_QUANTITIES, trials=200, seed=7
    )
    assert res.samples is None
    with pytest.raises(ValueError, match="keep_samples"):
        res.plot()
