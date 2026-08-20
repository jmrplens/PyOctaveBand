#  Copyright (c) 2026. Jose Manuel Requena Plens

"""What the STI ``.plot()`` renderer draws (IEC 60268-16).

The speech transmission index is read from seven octave-band modulation
transfer indices, and the figure is a bar per band on the fixed 0 to 1 scale the
index lives on, with the qualification band ("good", "fair", ...) in the title.
This test reads the bars back and checks they are the ``mti`` field itself. The
generic plot contract lives in ``tests/test_result_plots.py``.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from result_factories import _sti


# --------------------------------------------------------------------------
# STI
# --------------------------------------------------------------------------
def test_sti_bars_match_mti_and_annotate_rating() -> None:
    res = _sti()
    ax = res.plot()
    heights = [patch.get_height() for patch in ax.patches]
    np.testing.assert_allclose(heights, res.mti)
    assert res.rating in ax.get_title()
    assert ax.get_ylim() == (0.0, 1.0)
    plt.close("all")
