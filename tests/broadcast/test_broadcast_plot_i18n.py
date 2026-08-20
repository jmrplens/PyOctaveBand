#  Copyright (c) 2026. Jose Manuel Requena Plens

"""EN/ES language option of the broadcast loudness ``.plot()`` renderer."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from phonometry.broadcast import program_loudness

FS = 48000


def _sine(level_dbfs: float, duration: float) -> np.ndarray:
    t = np.arange(round(duration * FS)) / FS
    return 10.0 ** (level_dbfs / 20.0) * np.sin(2.0 * np.pi * 1000.0 * t)


def _steps() -> np.ndarray:
    mono = np.concatenate([_sine(-30.0, 4.0), _sine(-23.0, 4.0)])
    return np.vstack([mono, mono])


def test_program_loudness_es_and_bad_language() -> None:
    res = program_loudness(_steps(), FS)
    ax = res.plot(language="es")
    assert ax.get_ylabel() == "Sonoridad [LUFS]"
    assert ax.get_xlabel() == "Tiempo [s]"
    assert ax.get_title() == "Sonoridad de programa (EBU R 128)"
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(s.startswith("Integrada") for s in labels)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")
