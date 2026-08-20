#  Copyright (c) 2026. Jose Manuel Requena Plens

"""EN/ES language option of the emission ``.plot()`` renderers.

Each result exposes ``plot(language=...)``; ``"es"`` must produce Spanish
labels/titles and ``language="xx"`` must raise a clear ``ValueError``. The
English default is covered elsewhere (and must stay byte-identical).
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from phonometry.emission.intensity_compliance import (
    intensity_class_compliance,
    residual_index_limits,
)


def _labels(obj: object) -> str:
    axes = obj if isinstance(obj, np.ndarray) else [obj]
    parts = []
    for a in axes:
        parts += [a.get_xlabel(), a.get_ylabel()]
        leg = a.get_legend()
        if leg is not None:
            parts += [t.get_text() for t in leg.get_texts()]
    return " || ".join(parts)


def test_intensity_class_es() -> None:
    freqs, class1, _ = residual_index_limits("probe")
    res = intensity_class_compliance(class1 + 1.0, freqs, device="probe")
    ax = res.plot(language="es")
    assert "Tabla 2 de IEC 61043" in ax.get_title()
    assert "sonda" in ax.get_title()
    assert "Mínimo clase 1" in _labels(ax)
    assert "Región de aceptación clase 1" in _labels(ax)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")
