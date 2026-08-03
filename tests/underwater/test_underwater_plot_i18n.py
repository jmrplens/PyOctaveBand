#  Copyright (c) 2026. Jose Manuel Requena Plens
"""EN/ES language option for the underwater ``.plot()`` renderers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from phonometry.underwater import seabed_reflection
from phonometry.underwater.sonar_equation import passive_sonar_equation
from phonometry.underwater.sources.ambient_noise import ocean_ambient_noise


def _result() -> object:
    return passive_sonar_equation(140.0, 80.0, 60.0, directivity_index=10.0,
                                  detection_threshold=5.0)


def test_plot_default_is_english() -> None:
    ax = _result().plot()
    assert ax.get_title() == "Sonar equation"
    assert ax.get_xlabel() == "Transmission loss [dB]"
    assert ax.get_ylabel() == "Signal excess [dB]"
    plt.close("all")


def test_plot_spanish_labels() -> None:
    ax = _result().plot(language="es")
    assert ax.get_title() == "Ecuación del sonar"
    assert ax.get_ylabel() == "Exceso de señal [dB]"
    plt.close("all")


def test_plot_unknown_language_raises() -> None:
    result = _result()
    with pytest.raises(ValueError, match="Unknown language"):
        result.plot(language="xx")
    plt.close("all")


def test_seabed_reflection_plot_labels() -> None:
    import numpy as np

    res = seabed_reflection(np.linspace(0.0, 90.0, 91), rho1=1000.0, c1=1500.0,
                            rho2=1900.0, c2=1650.0)
    ax = res.plot()
    assert ax.get_title() == "Seabed reflection coefficient"
    assert ax.get_xlabel() == "Grazing angle [°]"
    assert "$|R|$" in ax.get_ylabel()
    assert ax.get_xlim() == (0.0, 90.0)
    ax_es = res.plot(language="es")
    assert ax_es.get_title() == "Coeficiente de reflexión del fondo marino"
    assert ax_es.get_xlabel() == "Ángulo rasante [°]"
    assert any(
        "Ángulo crítico" in txt.get_text()
        for txt in ax_es.get_legend().get_texts()
    )
    with pytest.raises(ValueError, match="Unknown language"):
        res.plot(language="xx")
    plt.close("all")


def test_weston_regime_plot_labels() -> None:
    import numpy as np

    from phonometry.underwater.propagation.weston_regimes import weston_propagation_loss

    res = weston_propagation_loss(np.logspace(1.0, 5.0, 60), 250.0, 50.0, seabed="sand")
    ax = res.plot()
    assert ax.get_xlabel() == "Range [m]"
    assert ax.get_ylabel() == "Propagation loss [dB re 1 m²]"
    assert "Weston regimes" in ax.get_title()
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Composite" in labels
    ax_es = res.plot(language="es")
    labels_es = [t.get_text() for t in ax_es.get_legend().get_texts()]
    assert ax_es.get_xlabel() == "Distancia [m]"
    assert "Compuesto" in labels_es
    assert "Modo único" in labels_es
    assert not any("Cylindrical" in text for text in labels_es)
    plt.close("all")


def test_marine_mammal_plot_labels() -> None:
    import numpy as np

    from phonometry.underwater.bioacoustics.audiograms import group_audiogram
    from phonometry.underwater.bioacoustics.weighting import (
        auditory_weighting,
        weighted_exposure,
    )

    freqs = np.logspace(2.0, 5.0, 80)
    ax = group_audiogram(freqs, "VHF").plot()
    assert ax.get_title() == "Group audiogram"
    assert ax.get_ylabel() == "Hearing threshold [dB]"
    ax_es = group_audiogram(freqs, "VHF").plot(language="es")
    assert ax_es.get_title() == "Audiograma de grupo"
    assert ax_es.get_ylabel() == "Umbral de audición [dB]"

    ax_w = auditory_weighting(freqs, "LF").plot(language="es")
    assert ax_w.get_title() == "Función de ponderación auditiva"
    assert ax_w.get_ylabel() == "Ponderación W(f) [dB]"

    bands = np.array([125.0, 250.0, 500.0, 1000.0])
    res = weighted_exposure(bands, np.full(bands.size, 180.0), "LF", peak_spl=210.0)
    ax_e = res.plot(language="es")
    labels = [t.get_text() for t in ax_e.get_legend().get_texts()]
    assert ax_e.get_title() == "Exposición ponderada frente a criterios"
    assert any(text.startswith("Sin ponderar") for text in labels)
    assert not any(text.startswith("Unweighted") for text in labels)
    plt.close("all")


def test_detection_range_plot_labels() -> None:
    from phonometry.underwater.sonar_equation import detection_range

    res = detection_range(78.0, 300.0)
    ax = res.plot()
    assert ax.get_title() == "Transmission loss vs figure of merit"
    ax_es = res.plot(language="es")
    labels = [t.get_text() for t in ax_es.get_legend().get_texts()]
    assert ax_es.get_title() == "Pérdida por transmisión frente a cifra de mérito"
    assert any(text.startswith("Alcance de detección") for text in labels)
    plt.close("all")


def test_ambient_noise_legend_localized() -> None:
    # A legend-heavy renderer: the composite curve plus its wind/thermal
    # components. Guards that every legend entry is localized (the composite
    # "Total"/"wind"/"thermal" labels), not only the axis titles.
    result = ocean_ambient_noise([100.0, 1000.0, 10000.0], wind_speed_knots=15.0)
    ax = result.plot(language="es")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(text.startswith("Total") for text in labels)
    assert "Viento" in labels
    assert "Térmico" in labels
    assert not any("Wind" in text or "Thermal" in text for text in labels)
    plt.close("all")
