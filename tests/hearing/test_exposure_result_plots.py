#  Copyright (c) 2026. Jose Manuel Requena Plens

"""What the noise-exposure ``.plot()`` renderer draws (ISO 9612).

The task-based strategy builds the daily exposure out of named tasks, and the
figure exists to show which task spends the day's noise budget:
one bar per task at its own contribution to ``LEX,8h``, with the total and the
upper exposure action value drawn across them as horizontal references. The
job-based and full-day strategies produce no task breakdown at all, so the
figure has to say so instead of drawing an empty axis.

These are the content assertions. The generic plot contract lives in
``tests/test_result_plots.py``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from result_factories import _exposure

import phonometry as ph


# --------------------------------------------------------------------------
# Occupational exposure (ISO 9612)
# --------------------------------------------------------------------------
def test_exposure_plot_task_bars_and_lex_line() -> None:
    res = _exposure()
    ax = res.plot()
    heights = [p.get_height() for p in ax.patches]
    np.testing.assert_allclose(
        heights, [t.lex_8h_contribution for t in res.tasks]
    )
    hlines = [
        np.asarray(ln.get_ydata())[0] for ln in ax.lines
        if np.asarray(ln.get_ydata()).size == 2
        and np.asarray(ln.get_ydata())[0] == np.asarray(ln.get_ydata())[1]
    ]
    assert any(v == pytest.approx(res.lex_8h) for v in hlines)
    assert any(v == pytest.approx(res.upper_limit) for v in hlines)
    plt.close("all")


def test_exposure_plot_without_tasks_raises() -> None:
    levels = np.full(5, 80.0)
    res = ph.hearing.job_based_exposure(levels, 6.0)
    assert not res.tasks
    with pytest.raises(ValueError, match="per-task"):
        res.plot()
