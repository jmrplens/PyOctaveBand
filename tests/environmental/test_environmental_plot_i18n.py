#  Copyright (c) 2026. Jose M. Requena-Plens
"""EN/ES language option for the environmental ``.plot()`` renderers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from phonometry.environmental.ground_barriers import ground_effect
from phonometry.environmental.outdoor_propagation import (
    outdoor_propagation_attenuation,
)
from phonometry.environmental.wind_turbine_noise import wind_turbine_tonality


def _result() -> object:
    df = 2.0
    freqs = np.arange(440.0, 560.0 + df, df)
    levels = np.full(freqs.size, 30.0)
    levels[int(np.argmin(np.abs(freqs - 500.0)))] = 60.0
    return wind_turbine_tonality(levels, freqs)


def test_plot_default_is_english() -> None:
    ax = _result().plot()
    assert ax.get_xlabel() == "Frequency [Hz]"
    assert ax.get_ylabel() == "Level [dB]"
    plt.close("all")


def test_plot_spanish_labels() -> None:
    ax = _result().plot(language="es")
    assert ax.get_xlabel() == "Frecuencia [Hz]"
    assert ax.get_ylabel() == "Nivel [dB]"
    plt.close("all")


def test_plot_unknown_language_raises() -> None:
    result = _result()
    with pytest.raises(ValueError, match="Unknown language"):
        result.plot(language="xx")
    plt.close("all")


def test_plot_outdoor_attenuation_spanish_xaxis_label() -> None:
    # plot_outdoor_attenuation draws its per-band terms through the shared
    # ``_band_axis`` helper and sets no xlabel of its own, so the Spanish
    # label only appears once the language is threaded into that helper.
    result = outdoor_propagation_attenuation(100.0, 2.0, 2.0)
    ax = result.plot(language="es")
    assert ax.get_xlabel() == "Frecuencia [Hz]"
    plt.close("all")
    ax = result.plot()
    assert ax.get_xlabel() == "Frequency [Hz]"
    plt.close("all")


def test_plot_spherical_ground_spanish_xaxis_label() -> None:
    # plot_spherical_ground draws its frequency axis through the shared
    # ``_freq_axis`` helper without overriding the xlabel afterwards.
    freqs = np.array([250.0, 500.0, 1000.0, 2000.0])
    result = ground_effect(freqs, 2.0, 1.5, 20.0, flow_resistivity=2.0e4)
    ax = result.plot(language="es")
    assert ax.get_xlabel() == "Frecuencia [Hz]"
    plt.close("all")
    ax = result.plot()
    assert ax.get_xlabel() == "Frequency [Hz]"
    plt.close("all")


def test_plot_cnossos_road_emission_spanish_labels() -> None:
    # The road-emission renderer localises its own title, ylabel and category
    # legend, and takes the shared ``_band_axis`` for the frequency axis.
    from phonometry.environmental.cnossos_road import (
        RoadTraffic,
        RoadVehicleCategory,
        road_source_power,
    )

    result = road_source_power(
        [
            RoadTraffic(RoadVehicleCategory.LIGHT, 1000.0, 50.0),
            RoadTraffic(RoadVehicleCategory.HEAVY, 100.0, 50.0),
        ]
    )
    ax = result.plot(language="es")
    assert ax.get_xlabel() == "Frecuencia [Hz]"
    assert ax.get_title() == "Potencia de la línea fuente viaria CNOSSOS-EU"
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "Línea total" in labels
    assert "Vehículos ligeros (1)" in labels
    assert "Vehículos pesados (3)" in labels
    plt.close("all")
    ax = result.plot()
    assert ax.get_xlabel() == "Frequency [Hz]"
    assert ax.get_title() == "CNOSSOS-EU road source line power"
    plt.close("all")
