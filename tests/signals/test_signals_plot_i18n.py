#  Copyright (c) 2026. Jose Manuel Requena Plens

"""EN/ES language option of the signal ``.plot()`` renderers.

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
    res = ph.signals.power_spectral_density(_white(), FS, nperseg=1024)
    ax = res.plot(language="es")
    assert "Densidad espectral de Welch" in ax.get_title()
    assert ax.get_xlabel() == "Frecuencia [Hz]"
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_multitaper_psd_es_and_bad_language() -> None:
    res = ph.signals.multitaper_psd(_white(), FS)
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
    axes = ph.signals.cross_spectral_density(x, y, FS).plot(language="es")
    assert "Densidad espectral cruzada (Bendat y Piersol)" in _titles(axes)
    assert "Fase [grados]" in _labels(axes)
    plt.close("all")


def test_coherent_output_spectrum_es() -> None:
    x = _white()
    y = np.roll(x, 17) + 0.1 * _white()
    axes = ph.signals.coherent_output_spectrum(x, y, FS).plot(language="es")
    assert "Espectro de salida coherente" in _titles(axes)
    assert "SNR espectral [dB]" in _labels(axes)
    plt.close("all")


def test_spectrogram_es() -> None:
    res = ph.signals.spectrogram(_white(8192), FS, nperseg=1024)
    ax = res.plot(language="es")
    assert "Espectrograma calibrado (Bendat y Piersol 12.6.4.2)" in ax.get_title()
    assert ax.get_xlabel() == "Tiempo [s]"
    assert ax.get_ylabel() == "Frecuencia [Hz]"
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_zoom_fft_es() -> None:
    res = ph.signals.zoom_fft(_white(8192), FS, f_min=1000.0, f_max=2000.0)
    ax = res.plot(language="es")
    assert ax.get_title() == "FFT con zoom (Bendat y Piersol 11.5.4)"
    assert ax.get_ylabel() == "Espectro de potencia [dB]"
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_correlation_es() -> None:
    res = ph.signals.correlation(_white(), fs=FS, max_lag=0.01)
    ax = res.plot(language="es")
    assert ax.get_title() == "Estimación de autocorrelación (Bendat y Piersol)"
    assert ax.get_xlabel() == "Retardo [s]"
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_correlation_normalization_label_es() -> None:
    """The normalization word is a table key reached through the result."""
    res = ph.signals.correlation(_white(), fs=FS, max_lag=0.01, normalization="biased")
    ax = res.plot(language="es")
    assert ax.get_ylabel() == "Correlación (sesgada)"
    plt.close("all")
    # The coefficient normalization has a label of its own, not a template.
    res = ph.signals.correlation(
        _white(), fs=FS, max_lag=0.01, normalization="coefficient"
    )
    ax = res.plot(language="es")
    assert ax.get_ylabel() == "Coeficiente de correlación"
    plt.close("all")


def test_time_delay_es() -> None:
    x = _white(8192)
    y = np.roll(x, 12)
    res = ph.signals.time_delay(x, y, FS, nperseg=2048, signal_bandwidth=FS / 2.0)
    ax = res.plot(language="es")
    assert ax.get_title().startswith("Estimación del retardo temporal")
    plt.close("all")


def test_aligned_impulse_response_es() -> None:
    ref = np.zeros(256)
    ref[100] = 1.0
    res = ph.signals.align_impulse_responses(np.roll(ref, 5), ref, FS)
    ax = res.plot(language="es")
    assert "Alineación de la respuesta al impulso" in ax.get_title()
    assert "RI de referencia" in _labels(ax)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_envelope_es() -> None:
    axes = ph.signals.envelope(_white(), FS).plot(language="es")
    assert "Envolvente de Hilbert (Bendat y Piersol Cap. 13)" in _titles(axes)
    assert "Frecuencia instantánea [Hz]" in _labels(axes)
    plt.close("all")


def test_envelope_spectrum_es() -> None:
    axes = ph.signals.envelope_spectrum(_white(), FS).plot(language="es")
    assert "Espectro de la envolvente (Bendat y Piersol 13.3)" in _titles(axes)
    assert "Amplitud de modulación" in _labels(axes)
    assert "Nivel medio" in _labels(axes)
    plt.close("all")


def test_cepstrum_es() -> None:
    res = ph.signals.cepstrum(_white(), FS, kind="power")
    ax = res.plot(language="es")
    assert "Cepstro de potencia" in _titles(ax)
    assert "Quefrencia [ms]" in _labels(ax)
    with pytest.raises(ValueError):
        res.plot(language="xx")
    plt.close("all")


def test_echo_detection_es() -> None:
    x = np.zeros(4096)
    x[0], x[313] = 1.0, 0.4
    ax = ph.signals.echo_detection(x, FS).plot(language="es")
    assert "Detección de ecos en el cepstro de potencia" in _titles(ax)
    assert "Banda de búsqueda" in _labels(ax)
    assert any("Eco:" in s and "," in s for s in _labels(ax).split(" || "))
    plt.close("all")


def test_lifter_es() -> None:
    axes = ph.signals.lifter(_white(), FS, cutoff=0.002, mode="highpass").plot(
        language="es"
    )
    assert "Liftering a 2 ms (paso alto)" in _titles(axes)
    assert "Lifterado (paso alto)" in _labels(axes)
    plt.close("all")


def test_phase_decomposition_es() -> None:
    resp = np.fft.rfft(np.exp(-np.arange(1024) / 50.0))
    axes = ph.signals.phase_decomposition(resp, fs=FS).plot(language="es")
    assert "Descomposición fase mínima / pasa-todo" in _titles(axes)
    assert "Fase medida" in _labels(axes)
    plt.close("all")


def test_tone_burst_es_and_bad_language() -> None:
    res = ph.signals.tone_burst(FS, 5000.0, 25, repetitions=2, repetition_rate=10.0)
    ax = res.plot(language="es")
    assert "Salva de tono (IEC 60268-1)" in ax.get_title()
    assert ax.get_xlabel() == "Tiempo [s]"
    assert "Envolvente de conmutación" in _labels(ax)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_resampled_signal_es_and_bad_language() -> None:
    res = ph.signals.resample_signal(
        ph.signals.noise_signal(FS, 0.2, seed=5), FS, fs_new=32000.0
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
    res = ph.signals.window_metrics("hann", 1024)
    axes = res.plot(language="es")
    assert "Métricas de la ventana (Harris 1978)" in _titles(axes)
    assert "Pérdida de festoneado" in _labels(axes)
    assert "bins de la DFT" in _labels(axes)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_inverse_filter_es() -> None:
    from scipy import signal as sg

    b, a = sg.butter(2, [100.0, 8000.0], btype="bandpass", fs=float(FS))
    imp = np.zeros(1024)
    imp[0] = 1.0
    res = ph.signals.regularized_inverse_filter(
        sg.lfilter(b, a, imp), float(FS), f_range=(200.0, 4000.0)
    )
    ax = res.plot(language="es")
    assert "Inversión regularizada (Kirkeby)" in ax.get_title()
    assert "Banda ecualizada" in _labels(ax)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")
