#  Copyright (c) 2026. Jose Manuel Requena Plens

"""What the outdoor-propagation ``.plot()`` renderer draws (ISO 9613-2).

The octave-band attenuation is a sum of named terms, geometrical divergence,
atmospheric absorption, ground effect and screening, and the figure is a stacked
bar so a reader can see which term carries the band. That makes the plot
answerable to arithmetic: the signed heights of the four stacks must add up to
``A_total`` band by band, and the total line must be ``A_total`` itself. The
ground term is signed, so in a scenario with hard ground at both ends it is a
net *gain* at 63 Hz and its bar hangs below the axis, which is exactly the case
this test builds.

These are the content assertions. The generic plot contract lives in
``tests/test_result_plots.py``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from result_factories import _outdoor


# --------------------------------------------------------------------------
# Outdoor attenuation breakdown (ISO 9613-2)
# --------------------------------------------------------------------------
def test_outdoor_plot_stacks_terms_to_total() -> None:
    res = _outdoor()
    ax = res.plot()
    n = res.frequencies.size
    # four stacked terms -> 4 bars per band; signed heights sum to a_total.
    assert len(ax.patches) == 4 * n
    heights = np.array([p.get_height() for p in ax.patches]).reshape(4, n)
    np.testing.assert_allclose(heights.sum(axis=0), res.a_total, atol=1e-9)
    # the ground term is a net gain (negative) at 63 Hz in this scenario.
    assert res.a_gr[0] < 0.0
    # the total line echoes a_total.
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.a_total)
    plt.close("all")
