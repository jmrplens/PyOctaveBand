#  Copyright (c) 2026. Jose Manuel Requena Plens
"""EN/ES internationalisation of the shared ``_plot/common.py`` renderers.

These low-level helpers (frequency/band axes, the shifted-reference rating
figure, the band-level bar chart, the facade x-axis, the ISO 717-2 500 Hz
annotation and the time axis) are reused by every ``_plot/<domain>.py``
module. Each domain module owns its own titles and legends, but the axis
labels and legend entries these *shared* helpers draw themselves must also
localise to Spanish once a domain's ``plot(language="es")`` reaches them.

English stays byte-identical to the pre-i18n renderers (``_t`` is a no-op for
``language="en"``), which is verified separately by the committed
documentation figures via ``scripts/check_figures.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from result_factories import _airborne_insulation, _impact_insulation

from phonometry._plot.common import (
    _annotate_impact_500,
    _band_axis,
    _facade_x_axis,
    _freq_axis,
    _plot_band_level_bars,
    _plot_rating,
    _time_axis,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _axes():
    _fig, ax = plt.subplots()
    return ax


def _legend_texts(ax) -> list[str]:
    legend = ax.get_legend()
    return [] if legend is None else [t.get_text() for t in legend.get_texts()]


def test_freq_axis_spanish_label() -> None:
    ax = _axes()
    _freq_axis(ax, np.array([125.0, 250.0, 500.0]), language="es")
    assert ax.get_xlabel() == "Frecuencia [Hz]"
    plt.close("all")


def test_freq_axis_english_default_unchanged() -> None:
    ax = _axes()
    _freq_axis(ax, np.array([125.0, 250.0, 500.0]))
    assert ax.get_xlabel() == "Frequency [Hz]"
    plt.close("all")


def test_band_axis_spanish_default_label() -> None:
    ax = _axes()
    _band_axis(ax, np.array([125.0, 250.0]), language="es")
    assert ax.get_xlabel() == "Frecuencia [Hz]"
    plt.close("all")


def test_band_axis_spanish_band_xlabel() -> None:
    ax = _axes()
    _band_axis(ax, ["1", "2", "3"], xlabel="Band", language="es")
    assert ax.get_xlabel() == "Banda"
    plt.close("all")


def test_band_axis_none_xlabel_left_untouched() -> None:
    ax = _axes()
    ax.set_xlabel("kept")
    _band_axis(ax, np.array([125.0, 250.0]), xlabel=None, language="es")
    assert ax.get_xlabel() == "kept"
    plt.close("all")


def test_plot_rating_spanish_labels_and_legend() -> None:
    band_centers = np.array([125.0, 250.0, 500.0, 1000.0])
    measured = np.array([30.0, 35.0, 40.0, 45.0])
    reference = np.array([32.0, 34.0, 42.0, 44.0])
    ax = _plot_rating(
        band_centers,
        measured,
        reference,
        impact=False,
        title="t",
        ylabel="y",
        ax=None,
        language="es",
    )
    assert ax.get_xlabel() == "Frecuencia [Hz]"
    legend_texts = _legend_texts(ax)
    assert "Medido" in legend_texts
    assert "Referencia desplazada" in legend_texts
    plt.close("all")


def test_plot_rating_unfavourable_deviations_localised() -> None:
    band_centers = np.array([125.0, 250.0, 500.0])
    measured = np.array([50.0, 55.0, 60.0])
    reference = np.array([40.0, 45.0, 50.0])
    ax = _plot_rating(
        band_centers,
        measured,
        reference,
        impact=True,
        title="t",
        ylabel="y",
        ax=None,
        language="es",
    )
    assert "Desviaciones desfavorables" in _legend_texts(ax)
    plt.close("all")


def test_plot_rating_english_default_unchanged() -> None:
    band_centers = np.array([125.0, 250.0, 500.0])
    measured = np.array([30.0, 35.0, 40.0])
    reference = np.array([32.0, 34.0, 42.0])
    ax = _plot_rating(
        band_centers,
        measured,
        reference,
        impact=False,
        title="t",
        ylabel="y",
        ax=None,
    )
    assert ax.get_xlabel() == "Frequency [Hz]"
    legend_texts = _legend_texts(ax)
    assert "Measured" in legend_texts
    assert "Shifted reference" in legend_texts
    plt.close("all")


def test_plot_band_level_bars_spanish_xlabel_and_comma_total() -> None:
    ax = _plot_band_level_bars(
        None,
        np.array([50.0, 55.0, 60.0]),
        np.array([125.0, 250.0, 500.0]),
        62.3,
        ylabel="y",
        title="t",
        language="es",
    )
    assert ax.get_xlabel() == "Frecuencia [Hz]"
    assert "total 62,3 dB" in _legend_texts(ax)
    plt.close("all")


def test_plot_band_level_bars_band_xlabel_when_no_frequencies() -> None:
    ax = _plot_band_level_bars(
        None,
        np.array([50.0, 55.0]),
        None,
        53.0,
        ylabel="y",
        title="t",
        language="es",
    )
    assert ax.get_xlabel() == "Banda"
    plt.close("all")


def test_plot_band_level_bars_english_default_unchanged() -> None:
    ax = _plot_band_level_bars(
        None,
        np.array([50.0, 55.0, 60.0]),
        np.array([125.0, 250.0, 500.0]),
        62.3,
        ylabel="y",
        title="t",
    )
    assert ax.get_xlabel() == "Frequency [Hz]"
    assert "total 62.3 dB" in _legend_texts(ax)
    plt.close("all")


def test_facade_x_axis_frequency_path_spanish() -> None:
    ax = _axes()
    _facade_x_axis(ax, np.array([125.0, 250.0]), 2, language="es")
    assert ax.get_xlabel() == "Frecuencia [Hz]"
    plt.close("all")


def test_facade_x_axis_band_index_path_spanish() -> None:
    ax = _axes()
    _facade_x_axis(ax, None, 3, language="es")
    assert ax.get_xlabel() == "Banda"
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert tick_labels == ["Banda 1", "Banda 2", "Banda 3"]
    plt.close("all")


def test_facade_x_axis_band_index_path_english_unchanged() -> None:
    ax = _axes()
    _facade_x_axis(ax, None, 2)
    assert ax.get_xlabel() == "Band"
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert tick_labels == ["Band 1", "Band 2"]
    plt.close("all")


@pytest.mark.parametrize("factory", [_airborne_insulation, _impact_insulation])
def test_insulation_bands_band_word_reaches_spanish_from_the_callers(
    factory: Callable[[], Any],
) -> None:
    """The ISO 16283 plotters must hand their language down to the helper.

    The band word is drawn by ``_plot_insulation_bands`` itself, so it only
    reaches Spanish if ``plot_airborne_insulation`` / ``plot_impact_insulation``
    forward their ``language``; a helper that localises but is never told the
    language ships an English axis on an otherwise Spanish figure.
    """
    ax = factory().plot(language="es")
    assert ax.get_xlabel() == "Banda"
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert tick_labels == ["Banda 1", "Banda 2", "Banda 3"]
    plt.close("all")


@pytest.mark.parametrize("factory", [_airborne_insulation, _impact_insulation])
def test_insulation_bands_english_default_unchanged(
    factory: Callable[[], Any],
) -> None:
    ax = factory().plot()
    assert ax.get_xlabel() == "Band"
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert tick_labels == ["Band 1", "Band 2", "Band 3"]
    plt.close("all")


def test_annotate_impact_500_spanish_legend_and_annotation() -> None:
    ax = _axes()
    band_centers = np.array([125.0, 250.0, 500.0, 1000.0])
    reference = np.array([30.0, 35.0, 40.0, 45.0])
    ax.plot(band_centers, reference)
    _annotate_impact_500(ax, band_centers, reference, 35, language="es")
    legend_texts = _legend_texts(ax)
    assert any("lectura 500 Hz" in t for t in legend_texts)
    annotation_texts = [child.get_text() for child in ax.texts]
    assert any("regla de octava" in t for t in annotation_texts)
    plt.close("all")


def test_annotate_impact_500_english_default_unchanged() -> None:
    ax = _axes()
    band_centers = np.array([125.0, 250.0, 500.0, 1000.0])
    reference = np.array([30.0, 35.0, 40.0, 45.0])
    ax.plot(band_centers, reference)
    _annotate_impact_500(ax, band_centers, reference, 35)
    legend_texts = _legend_texts(ax)
    assert any("500 Hz read" in t for t in legend_texts)
    annotation_texts = [child.get_text() for child in ax.texts]
    assert any("octave rule" in t for t in annotation_texts)
    plt.close("all")


def test_time_axis_spanish_labels() -> None:
    _, xlabel = _time_axis(100, 1000, language="es")
    assert xlabel == "Tiempo [s]"
    _, xlabel_no_fs = _time_axis(100, None, language="es")
    assert xlabel_no_fs == "Muestra"


def test_time_axis_english_default_unchanged() -> None:
    _, xlabel = _time_axis(100, 1000)
    assert xlabel == "Time [s]"
    _, xlabel_no_fs = _time_axis(100, None)
    assert xlabel_no_fs == "Sample"
