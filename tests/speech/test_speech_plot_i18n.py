#  Copyright (c) 2026. Jose Manuel Requena Plens
"""EN/ES language option of the speech ``.plot()`` renderers.

Each result exposes ``plot(language=...)``; ``"es"`` must produce Spanish
labels and ``language="xx"`` must raise. The renderers carry their own
translation table since the split of the hearing renderers, so this is what
holds that table to its contents: a key that landed in the wrong module
degrades a Spanish label to English silently.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phonometry as ph

FS = 48000
RNG = np.random.default_rng(20260801)


def _labels(ax: object) -> str:
    axes = ax if isinstance(ax, np.ndarray) else [ax]
    parts: list[str] = []
    for a in axes:
        parts += [a.get_xlabel(), a.get_ylabel(), a.get_title()]
        legend = a.get_legend()
        if legend is not None:
            parts += [t.get_text() for t in legend.get_texts()]
    return " || ".join(parts)


def test_standard_speech_spectrum_es_and_bad_language() -> None:
    """The four vocal efforts are table keys reached through the result."""
    res = ph.speech.standard_speech_spectra()
    ax = res.plot(language="es")
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend == ["Normal", "Elevada", "Fuerte", "Grito"]
    assert ax.get_ylabel() == "Nivel del espectro de voz [dB SPL]"
    assert ax.get_title() == "ANSI S3.5-1997 espectro de voz estándar"
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


@pytest.mark.parametrize(
    ("method", "label"),
    [
        ("critical-band", "Banda crítica (21)"),
        ("equally-contributing", "Contribución equitativa (17)"),
        ("one-third-octave", "Tercio de octava (18)"),
        ("octave", "Octava (6)"),
    ],
)
def test_sii_procedure_es(method: str, label: str) -> None:
    """All four band-importance labels: each is a key of its own."""
    ax = ph.speech.sii_procedure(method).plot(language="es")
    assert ax.get_title() == "ANSI S3.5-1997 función de importancia de banda"
    assert ax.get_ylabel() == "Importancia de banda $I_i$"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [label]
    plt.close("all")


def test_sii_es() -> None:
    res = ph.speech.speech_intelligibility_index(
        ph.speech.standard_speech_spectrum("normal"), [30.0] * 18
    )
    ax = res.plot(language="es")
    text = _labels(ax)
    assert "Audibilidad de banda" in text
    assert "Frecuencia [Hz]" in text
    # The decimal comma is the Spanish renderer's, not the table's.
    assert "ANSI S3.5 SII = 0,2" in text
    plt.close("all")


def test_sti_es() -> None:
    impulse = np.zeros(FS // 2)
    impulse[0] = 1.0
    impulse[2048] = 0.4
    res = ph.speech.sti_from_impulse_response(impulse, FS)
    ax = res.plot(language="es")
    text = _labels(ax)
    assert "Índice de transferencia de modulación MTI" in text
    assert "calificación" in text
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_stoi_es() -> None:
    clean = RNG.standard_normal(FS)
    degraded = clean + 0.2 * RNG.standard_normal(FS)
    res = ph.speech.stoi(clean, degraded, FS)
    ax = res.plot(language="es")
    text = _labels(ax)
    assert "Correlación" in text
    plt.close("all")

def test_sti_with_a_non_octave_band_set_es() -> None:
    """A result whose MTI is not the seven octave bands gets a plain axis."""
    res = ph.speech.STIResult(
        sti=0.62,
        mti=np.linspace(0.4, 0.8, 5),
        mtf=np.zeros((5, 14)),
        band_levels=np.full(5, 60.0),
        rating="good",
    )
    ax = res.plot(language="es")
    assert ax.get_xlabel() == "Banda"
    assert "calificación" in ax.get_title()
    plt.close("all")

