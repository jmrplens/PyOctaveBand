#  Copyright (c) 2026. Jose Manuel Requena Plens

"""EN/ES language option of the metrology ``.plot()`` renderers.

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

import phonometry as ph
from phonometry.metrology import uncertainty as u

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


def test_uncertainty_budget_es() -> None:
    result = u.combine_uncertainty(_add4, [u.Quantity(0.0, 1.0) for _ in range(4)])
    ax = result.plot(language="es")
    assert ax.get_title().startswith("Presupuesto de incertidumbre (GUM)")
    assert "incertidumbre combinada" in ax.get_xlabel()
    plt.close("all")
    with pytest.raises(ValueError):
        result.plot(language="xx")


def test_monte_carlo_es() -> None:
    mc = u.monte_carlo(
        _add4,
        [u.Quantity(0.0, 1.0, "rectangular") for _ in range(4)],
        trials=20_000,
        seed=3,
        keep_samples=True,
    )
    ax = mc.plot(language="es")
    assert ax.get_ylabel() == "Densidad de probabilidad"
    assert "Distribución de Monte Carlo" in ax.get_title()
    plt.close("all")


def test_trend_test_es_and_bad_language() -> None:
    values = [
        5.2,
        6.2,
        3.7,
        6.4,
        3.9,
        4.0,
        3.9,
        5.3,
        4.0,
        4.6,
        5.9,
        6.5,
        4.3,
        5.7,
        3.1,
        5.6,
        5.2,
        3.9,
        6.2,
        5.0,
    ]
    res = ph.metrology.trend_test(values)
    ax = res.plot(language="es")
    assert "Test de tendencia" in ax.get_title()
    assert ax.get_xlabel() == "Índice de muestra"
    assert ax.get_ylabel() == "Valor de la secuencia"
    assert "Inversiones de orden" in _labels(ax)
    assert "sin tendencia" in _labels(ax)
    plt.close("all")
    runs = ph.metrology.trend_test(RNG.standard_normal(40), method="runs")
    ax = runs.plot(language="es")
    assert "Rachas $r$ =" in _labels(ax)
    assert "Mediana de la secuencia" in _labels(ax)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_stationarity_test_es_and_bad_language() -> None:
    n = 1 << 14
    x = _white(n) * np.linspace(1.0, 1.3, n)
    res = ph.metrology.stationarity_test(x, FS)
    ax = res.plot(language="es")
    assert "Test de estacionariedad" in ax.get_title()
    assert ax.get_xlabel() == "Índice de segmento"
    assert "Inversiones de orden" in _labels(ax)
    plt.close("all")
    runs = ph.metrology.stationarity_test(x, FS, method="runs")
    ax = runs.plot(language="es")
    assert "Rachas $r$ =" in _labels(ax)
    assert "Mediana de la secuencia" in _labels(ax)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_level_crossing_rate_es_and_bad_language() -> None:
    res = ph.metrology.level_crossing_rate(_white(1 << 14), FS)
    ax = res.plot(language="es")
    assert "Tasa de cruces por nivel" in ax.get_title()
    assert ax.get_ylabel() == "Cruces por segundo [1/s]"
    assert "Expectativa de Rice" in _labels(ax)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_peak_statistics_es_and_bad_language() -> None:
    res = ph.metrology.peak_statistics(_white(1 << 14), FS)
    ax = res.plot(language="es")
    assert "Distribución de alturas de pico" in ax.get_title()
    assert "Límite de Rayleigh" in _labels(ax)
    assert "Excedencia empírica de picos" in _labels(ax)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")
