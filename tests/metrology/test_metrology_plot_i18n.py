#  Copyright (c) 2026. Jose M. Requena-Plens

"""EN/ES language option of the metrology ``.plot()`` renderers.

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


def test_spectral_density_es_and_bad_language() -> None:
    res = ph.power_spectral_density(_white(), FS, nperseg=1024)
    ax = res.plot(language="es")
    assert "Densidad espectral de Welch" in ax.get_title()
    assert ax.get_xlabel() == "Frecuencia [Hz]"
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_multitaper_psd_es_and_bad_language() -> None:
    res = ph.multitaper_psd(_white(), FS)
    ax = res.plot(language="es")
    assert "Densidad multitaper de Thomson" in ax.get_title()
    assert ax.get_xlabel() == "Frecuencia [Hz]"
    assert "de confianza" in _labels(ax)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_cross_spectral_density_es() -> None:
    x = _white()
    y = np.roll(x, 17) + 0.1 * _white()
    axes = ph.cross_spectral_density(x, y, FS).plot(language="es")
    assert "Densidad espectral cruzada (Bendat y Piersol)" in _titles(axes)
    assert "Fase [grados]" in _labels(axes)
    plt.close("all")


def test_coherent_output_spectrum_es() -> None:
    x = _white()
    y = np.roll(x, 17) + 0.1 * _white()
    axes = ph.coherent_output_spectrum(x, y, FS).plot(language="es")
    assert "Espectro de salida coherente" in _titles(axes)
    assert "SNR espectral [dB]" in _labels(axes)
    plt.close("all")


def test_spectrogram_es() -> None:
    res = ph.spectrogram(_white(8192), FS, nperseg=1024)
    ax = res.plot(language="es")
    assert "Espectrograma calibrado (Bendat y Piersol 12.6.4.2)" in ax.get_title()
    assert ax.get_xlabel() == "Tiempo [s]"
    assert ax.get_ylabel() == "Frecuencia [Hz]"
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_zoom_fft_es() -> None:
    res = ph.zoom_fft(_white(8192), FS, 1000.0, 2000.0)
    ax = res.plot(language="es")
    assert ax.get_title() == "FFT con zoom (Bendat y Piersol 11.5.4)"
    assert ax.get_ylabel() == "Espectro de potencia [dB]"
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_correlation_es() -> None:
    res = ph.correlation(_white(), fs=FS, max_lag=0.01)
    ax = res.plot(language="es")
    assert ax.get_title() == "Estimación de autocorrelación (Bendat y Piersol)"
    assert ax.get_xlabel() == "Retardo [s]"
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_time_delay_es() -> None:
    x = _white(8192)
    y = np.roll(x, 12)
    res = ph.time_delay(x, y, FS, nperseg=2048, signal_bandwidth=FS / 2.0)
    ax = res.plot(language="es")
    assert ax.get_title().startswith("Estimación del retardo temporal")
    plt.close("all")


def test_aligned_impulse_response_es() -> None:
    ref = np.zeros(256)
    ref[100] = 1.0
    res = ph.align_impulse_responses(np.roll(ref, 5), ref, FS)
    ax = res.plot(language="es")
    assert "Alineación de la respuesta al impulso" in ax.get_title()
    assert "RI de referencia" in _labels(ax)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_envelope_es() -> None:
    axes = ph.envelope(_white(), FS).plot(language="es")
    assert "Envolvente de Hilbert (Bendat y Piersol Cap. 13)" in _titles(axes)
    assert "Frecuencia instantánea [Hz]" in _labels(axes)
    plt.close("all")


def test_envelope_spectrum_es() -> None:
    axes = ph.envelope_spectrum(_white(), FS).plot(language="es")
    assert "Espectro de la envolvente (Bendat y Piersol 13.3)" in _titles(axes)
    assert "Amplitud de modulación" in _labels(axes)
    assert "Nivel medio" in _labels(axes)
    plt.close("all")


def test_cepstrum_es() -> None:
    res = ph.cepstrum(_white(), FS, kind="power")
    ax = res.plot(language="es")
    assert "Cepstro de potencia" in _titles(ax)
    assert "Quefrencia [ms]" in _labels(ax)
    with pytest.raises(ValueError):
        res.plot(language="xx")
    plt.close("all")


def test_echo_detection_es() -> None:
    x = np.zeros(4096)
    x[0], x[313] = 1.0, 0.4
    ax = ph.echo_detection(x, FS).plot(language="es")
    assert "Detección de ecos en el cepstro de potencia" in _titles(ax)
    assert "Banda de búsqueda" in _labels(ax)
    assert any("Eco:" in s and "," in s for s in _labels(ax).split(" || "))
    plt.close("all")


def test_lifter_es() -> None:
    axes = ph.lifter(_white(), FS, 0.002, mode="highpass").plot(language="es")
    assert "Liftering a 2 ms (paso alto)" in _titles(axes)
    assert "Lifterado (paso alto)" in _labels(axes)
    plt.close("all")


def test_phase_decomposition_es() -> None:
    resp = np.fft.rfft(np.exp(-np.arange(1024) / 50.0))
    axes = ph.phase_decomposition(resp, fs=FS).plot(language="es")
    assert "Descomposición fase mínima / pasa-todo" in _titles(axes)
    assert "Fase medida" in _labels(axes)
    plt.close("all")


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
        _add4, [u.Quantity(0.0, 1.0, "rectangular") for _ in range(4)],
        trials=20_000, seed=3, keep_samples=True,
    )
    ax = mc.plot(language="es")
    assert ax.get_ylabel() == "Densidad de probabilidad"
    assert "Distribución de Monte Carlo" in ax.get_title()
    plt.close("all")


def test_filter_class_es() -> None:
    from phonometry.metrology.compliance import filter_class_compliance

    bank = ph.OctaveFilterBank(fs=48000, fraction=1, order=6, limits=[500, 2000])
    res = filter_class_compliance(bank)
    ax = res.plot(language="es")
    assert "Máscara clase" in ax.get_title()
    assert ax.get_ylabel() == "Atenuación relativa [dB]"
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_intensity_class_es() -> None:
    from phonometry.metrology.intensity_compliance import (
        intensity_class_compliance,
        residual_index_limits,
    )

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


def test_tone_burst_es_and_bad_language() -> None:
    res = ph.tone_burst(FS, 5000.0, 25, repetitions=2, repetition_rate=10.0)
    ax = res.plot(language="es")
    assert "Salva de tono (IEC 60268-1)" in ax.get_title()
    assert ax.get_xlabel() == "Tiempo [s]"
    assert "Envolvente de conmutación" in _labels(ax)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_resampled_signal_es_and_bad_language() -> None:
    res = ph.resample_signal(
        ph.noise_signal(FS, 0.2, seed=5), FS, 32000.0
    )
    ax = res.plot(language="es")
    assert "Remuestreo polifásico" in ax.get_title()
    assert ax.get_xlabel() == "Frecuencia [Hz]"
    assert "Filtro antisolapamiento" in _labels(ax)
    assert "Borde de la banda de paso" in _labels(ax)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_window_metrics_es_and_bad_language() -> None:
    res = ph.window_metrics("hann", 1024)
    axes = res.plot(language="es")
    assert "Métricas de la ventana (Harris 1978)" in _titles(axes)
    assert "Pérdida de festoneado" in _labels(axes)
    assert "bins de la DFT" in _labels(axes)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_trend_test_es_and_bad_language() -> None:
    values = [5.2, 6.2, 3.7, 6.4, 3.9, 4.0, 3.9, 5.3, 4.0, 4.6,
              5.9, 6.5, 4.3, 5.7, 3.1, 5.6, 5.2, 3.9, 6.2, 5.0]
    res = ph.trend_test(values)
    ax = res.plot(language="es")
    assert "Test de tendencia" in ax.get_title()
    assert ax.get_xlabel() == "Índice de muestra"
    assert ax.get_ylabel() == "Valor de la secuencia"
    assert "Inversiones de orden" in _labels(ax)
    assert "sin tendencia" in _labels(ax)
    plt.close("all")
    runs = ph.trend_test(RNG.standard_normal(40), method="runs")
    ax = runs.plot(language="es")
    assert "Rachas r =" in _labels(ax)
    assert "Mediana de la secuencia" in _labels(ax)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_stationarity_test_es_and_bad_language() -> None:
    n = 1 << 14
    x = _white(n) * np.linspace(1.0, 1.3, n)
    res = ph.stationarity_test(x, FS)
    ax = res.plot(language="es")
    assert "Test de estacionariedad" in ax.get_title()
    assert ax.get_xlabel() == "Índice de segmento"
    assert "Inversiones de orden" in _labels(ax)
    plt.close("all")
    runs = ph.stationarity_test(x, FS, method="runs")
    ax = runs.plot(language="es")
    assert "Rachas r =" in _labels(ax)
    assert "Mediana de la secuencia" in _labels(ax)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_level_crossing_rate_es_and_bad_language() -> None:
    res = ph.level_crossing_rate(_white(1 << 14), FS)
    ax = res.plot(language="es")
    assert "Tasa de cruces por nivel" in ax.get_title()
    assert ax.get_ylabel() == "Cruces por segundo [1/s]"
    assert "Expectativa de Rice" in _labels(ax)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_peak_statistics_es_and_bad_language() -> None:
    res = ph.peak_statistics(_white(1 << 14), FS)
    ax = res.plot(language="es")
    assert "Distribución de alturas de pico" in ax.get_title()
    assert "Límite de Rayleigh" in _labels(ax)
    assert "Excedencia empírica de picos" in _labels(ax)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_inverse_filter_es() -> None:
    from scipy import signal as sg

    b, a = sg.butter(2, [100.0, 8000.0], btype="bandpass", fs=float(FS))
    imp = np.zeros(1024)
    imp[0] = 1.0
    res = ph.regularized_inverse_filter(
        sg.lfilter(b, a, imp), float(FS), f_range=(200.0, 4000.0)
    )
    ax = res.plot(language="es")
    assert "Inversión regularizada (Kirkeby)" in ax.get_title()
    assert "Banda ecualizada" in _labels(ax)
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
