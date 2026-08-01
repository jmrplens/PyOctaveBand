#  Copyright (c) 2026. Jose Manuel Requena Plens

"""EN/ES language option of the filters ``.plot()`` renderers.

Each result exposes ``plot(language=...)``; ``"es"`` must produce Spanish
labels/titles and ``language="xx"`` must raise a clear ``ValueError``. The
English default is covered elsewhere (and must stay byte-identical).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phonometry as ph

FS = 48000
RNG = np.random.default_rng(20260720)


def _white(n: int = 4096) -> np.ndarray:
    return RNG.standard_normal(n)


def _titles(obj: object) -> str:
    axes = obj if isinstance(obj, np.ndarray) else [obj]
    return " || ".join(a.get_title() for a in axes)


def _labels(obj: object) -> str:
    axes = obj if isinstance(obj, np.ndarray) else [obj]
    parts = []
    for a in axes:
        parts += [a.get_xlabel(), a.get_ylabel()]
        leg = a.get_legend()
        if leg is not None:
            parts += [t.get_text() for t in leg.get_texts()]
    return " || ".join(parts)


def _add4(a: float, b: float, c: float, d: float) -> float:
    return a + b + c + d


def test_filter_class_es() -> None:
    from phonometry.filters.compliance import filter_class_compliance

    bank = ph.OctaveFilterBank(fs=48000, fraction=1, order=6, limits=[500, 2000])
    res = filter_class_compliance(bank)
    ax = res.plot(language="es")
    assert "Máscara clase" in ax.get_title()
    assert ax.get_ylabel() == "Atenuación relativa [dB]"
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_parametric_eq_es_and_bad_language() -> None:
    eq = ph.ParametricEQ(FS, [
        ph.EQSection("lowshelf", 100.0, gain_db=4.0),
        ph.EQSection("peaking", 1000.0, gain_db=-6.0, q=1.5),
    ])
    res = eq.response(n_points=64)
    axes = res.plot(language="es")
    assert "Respuesta del EQ paramétrico (Audio EQ Cookbook)" in _titles(axes)
    assert "Cascada" in _labels(axes)
    assert "shelf de graves 100 Hz" in _labels(axes)
    assert "campana 1000 Hz" in _labels(axes)
    assert "Fase [grados]" in _labels(axes)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")
