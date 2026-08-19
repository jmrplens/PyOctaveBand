#  Copyright (c) 2026. Jose Manuel Requena Plens

"""EN/ES language option of the electroacoustics ``.plot()`` renderers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phonometry as ph

FS = 48000
RNG = np.random.default_rng(20260720)


def _labels(obj: object) -> str:
    axes = obj if isinstance(obj, np.ndarray) else [obj]
    parts = []
    for a in axes:
        parts += [a.get_xlabel(), a.get_ylabel(), a.get_title()]
        leg = a.get_legend()
        if leg is not None:
            parts += [t.get_text() for t in leg.get_texts()]
    return " || ".join(parts)


def _tone() -> np.ndarray:
    t = np.arange(FS) / FS
    return (np.sin(2 * np.pi * 1000.0 * t)
            + 0.03 * np.sin(2 * np.pi * 2000.0 * t)
            + 0.01 * np.sin(2 * np.pi * 3000.0 * t))


def test_harmonic_distortion_es() -> None:
    res = ph.harmonic_analysis(_tone(), FS, 1000.0)
    ax = res.plot(language="es")
    assert ax.get_ylabel() == "Nivel respecto al fundamental [dB]"
    assert ax.get_xlabel().startswith("Orden del armónico")
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_frequency_response_es() -> None:
    x = RNG.standard_normal(2 ** 15)
    y = np.convolve(x, np.array([0.5, 0.3, 0.1]), mode="same")
    axes = ph.transfer_function(x, y, FS).plot(language="es")
    text = _labels(axes)
    assert "Respuesta en frecuencia" in text
    assert "coherencia" in text
    assert "Magnitud [dB]" in text
    plt.close("all")


def test_swept_sine_distortion_es() -> None:
    f1, f2, seconds = 100.0, 8000.0, 1.0
    sweep = ph.synchronized_sweep_signal(FS, f1, f2, seconds)
    rec = sweep + 0.01 * sweep ** 2
    res = ph.swept_sine_distortion(rec, FS, f1=f1, f2=f2, seconds=seconds, n_harmonics=3)
    # Two-panel figure: harmonic responses (title) + THD(f) (excitation axis).
    axes = res.plot(language="es")
    text = _labels(axes)
    assert "Respuestas en frecuencia de los armónicos" in text
    assert "Frecuencia de excitación [Hz]" in text
    plt.close("all")
    # Single-panel case (ax given) carries the swept-THD title.
    _fig, ext = plt.subplots()
    single = res.plot(ax=ext, language="es")
    assert single.get_title() == "THD de barrido sinusoidal (Farina / Novak)"
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_feedback_stability_es() -> None:
    res = ph.feedback_stability(-6.0, 74.0, 80.0)
    ax = res.plot(language="es")
    assert "Ganancia antes de la realimentación" in ax.get_title()
    assert "Ganancia [dB]" in ax.get_ylabel()
    assert "Oscilación (0 dB)" in _labels(ax)
    plt.close("all")
    unstable = ph.feedback_stability(0.0, 80.0, 80.0)
    assert "inestable" in unstable.plot(language="es").get_title()
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_sound_reinforcement_geometry_es() -> None:
    ax = ph.plot_sound_reinforcement_geometry(0.3, 4.0, 12.0, language="es")
    assert "Lazo de realimentación" in ax.get_title()
    assert "Camino de realimentación" in _labels(ax)
    plt.close("all")
    with pytest.raises(ValueError):
        ph.plot_sound_reinforcement_geometry(0.3, 4.0, 12.0, language="xx")


def test_piston_impedance_es() -> None:
    res = ph.radiating_piston(0.05, np.geomspace(50.0, 5000.0, 60))
    ax = res.plot(language="es")
    assert "pistón circular con pantalla" in ax.get_title()
    assert "Impedancia de radiación normalizada" in ax.get_ylabel()
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_piston_directivity_es() -> None:
    res = ph.piston_directivity_pattern([3.0, 8.0, 16.0])
    ax = res.plot(language="es")
    assert ax.name == "polar"
    assert ax.get_title() == "Directividad de un pistón circular con pantalla"
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def _loudspeaker() -> ph.electroacoustics.LoudspeakerCharacteristics:
    f = np.geomspace(30.0, 24000.0, 200)
    spl = 87.0 - 10 * np.log10(1 + (50.0 / f) ** 6) - 10 * np.log10(1 + (f / 16000.0) ** 7)
    fz = np.geomspace(20.0, 20000.0, 120)
    return ph.loudspeaker_characteristics(
        f, spl, 8.0, sensitivity_band=(200.0, 4000.0),
        impedance=(fz, 6.6 + 20 * np.exp(-(np.log2(fz / 52.0) ** 2) / 0.12)),
        distortion=(np.geomspace(50.0, 5000.0, 90),
                    0.4 + 2.0 * np.ones(90)),
        directivity=ph.LoudspeakerDirectivity(
            polar=(np.linspace(0.0, 90.0, 46), -np.linspace(0.0, 12.0, 46)),
            frequency=2000.0,
        ),
    )


def _microphone() -> ph.electroacoustics.MicrophoneCharacteristics:
    f = np.geomspace(20.0, 20000.0, 200)
    resp = -10 * np.log10(1 + (30.0 / f) ** 4) - 10 * np.log10(1 + (f / 19000.0) ** 8)
    ang = np.linspace(0.0, 179.0, 180)
    spl = np.linspace(100.0, 140.0, 41)
    return ph.microphone_characteristics(
        f, resp, 12.5, tolerance_db=3.0,
        noise=ph.MicrophoneNoise(
            voltage=1.25e-6,
            spectrum=(np.geomspace(20.0, 20000.0, 31), np.full(31, 10.0)),
        ),
        overload=ph.MicrophoneOverload(
            distortion=(spl, 0.5 * 10 ** ((spl - 130.0) * 0.08)), thd_percent=0.5
        ),
        directivity=ph.MicrophoneDirectivity(
            polar=(ang, 20 * np.log10((1 + np.cos(np.radians(ang))) / 2)),
            frequency=1000.0,
        ),
    )


def test_loudspeaker_characteristics_es() -> None:
    res = _loudspeaker()
    # The default quantity is the on-axis response; every quantity is one axes.
    assert res.plot(language="es").get_title() == "Respuesta en el eje"
    plt.close("all")
    titles = {
        "response": "Respuesta en el eje",
        "impedance": "Impedancia",
        "thd": "Distorsión armónica total",
        "directivity": "Respuesta direccional a 2000 Hz",
    }
    for quantity, title in titles.items():
        ax = res.plot(quantity=quantity, language="es")
        assert not isinstance(ax, np.ndarray)
        assert ax.get_title() == title
        plt.close("all")
    assert res.plot(quantity="response", language="es").get_ylabel() == (
        "Nivel de presión acústica [dB]"
    )
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(quantity="bogus")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_microphone_characteristics_es() -> None:
    res = _microphone()
    assert res.plot(language="es").get_title() == "Respuesta en campo libre"
    plt.close("all")
    titles = {
        "response": "Respuesta en campo libre",
        "directivity": "Respuesta direccional a 1000 Hz",
        "noise": "Espectro de ruido inherente",
        "distortion": "Distorsión armónica total",
    }
    for quantity, title in titles.items():
        ax = res.plot(quantity=quantity, language="es")
        assert not isinstance(ax, np.ndarray)
        assert ax.get_title() == title
        plt.close("all")
    assert res.plot(quantity="distortion", language="es").get_xlabel() == (
        "Nivel de presión acústica [dB]"
    )
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(quantity="bogus")
    with pytest.raises(ValueError):
        res.plot(language="xx")
