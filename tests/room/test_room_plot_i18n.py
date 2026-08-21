#  Copyright (c) 2026. Jose Manuel Requena Plens
"""EN/ES internationalisation of the room-domain ``.plot()`` renderers.

Each result type exposes ``plot(language=...)``: the default English output is
covered by the existing per-module plot tests, so here we assert that Spanish
renders a translated label/title and that an unsupported language raises a
clear :class:`ValueError`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from phonometry import room
from phonometry._plot.room import plot_excitation
from phonometry.room import enclosed_space_absorption as esa
from phonometry.room import noise_criteria as rn
from phonometry.room.crowd_noise import crowd_noise
from phonometry.room.image_source import image_source_rir
from phonometry.room.impulse_response import ImpulseResponseResult
from phonometry.room.modes import room_modes
from phonometry.room.open_plan import open_plan_metrics
from phonometry.room.reverberation_prediction import (
    reverberation_time_models,
)
from phonometry.room.steady_field import steady_state_field

FS = 48000


def _decaying_ir() -> np.ndarray:
    """A short decaying-noise impulse response (RT ~ 0.4 s)."""
    t = np.arange(int(0.6 * FS)) / FS
    rng = np.random.default_rng(0)
    return rng.standard_normal(t.size) * np.exp(-6.9 * t / 0.4)


def _texts(ax_or_axes: object) -> str:
    """Concatenate every axis label, title and legend text for inspection."""
    axes = np.atleast_1d(ax_or_axes).ravel()
    parts: list[str] = []
    for ax in axes:
        parts += [ax.get_xlabel(), ax.get_ylabel(), ax.get_title()]
        legend = ax.get_legend()
        if legend is not None:
            parts += [t.get_text() for t in legend.get_texts()]
    return "\n".join(parts)


def _room_acoustics():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return room.room_parameters(_decaying_ir(), FS, limits=None)


def _decay_curve():
    return room.decay_curve(_decaying_ir(), FS)


def _nc():
    return rn.noise_criterion(rn.nc_curve(40.0))


def _rc():
    return rn.room_criterion(rn.rc_curve(35.0))


def _enclosed():
    return esa.enclosed_space_reverberation(
        [(20.0, [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70])],
        50.0,
        air_condition="20C_50-70",
    )


def _models():
    return reverberation_time_models(
        (8.0, 5.0, 3.0), (0.2, 0.15, 0.3), frequencies=[125.0, 250.0]
    )


def _open_plan():
    r = np.array([2.0, 4.0, 8.0, 16.0])
    lp = 70.0 - 6.0 * np.log2(r)
    sti = 0.6 - 0.02 * r
    return open_plan_metrics(r, lp, sti)


def _impulse_response():
    return ImpulseResponseResult(ir=_decaying_ir(), fs=FS, method="spectral")


def _shaped_sweep():

    return room.shaped_sweep_signal(FS, 100.0, 5000.0, 0.4, target="pink")


def _image_source():
    return image_source_rir(
        (5.0, 4.0, 3.0), (1.2, 1.1, 1.3), (3.5, 2.6, 1.7), 0.15, fs=16000, max_order=8
    )


def _steady_field():
    return steady_state_field(90.0, 100.0, 0.2)


def _room_modes():
    return room_modes((7.0, 5.0, 3.0), max_frequency=120.0, reverberation_time=0.8)


def _crowd_noise():
    return crowd_noise([20.0, 190.0])


# (id, builder, expected Spanish substring) for every room result renderer.
_CASES = [
    ("room_acoustics", _room_acoustics, "Tiempos de caída"),
    ("decay_curve", _decay_curve, "Nivel re estado estacionario"),
    ("noise_criterion", _nc, "NPS por bandas de octava"),
    ("room_criterion", _rc, "NPS por bandas de octava"),
    ("enclosed_space", _enclosed, "Tiempo de reverberación EN 12354-6"),
    ("reverberation_models", _models, "Modelos de tiempo de reverberación"),
    ("open_plan", _open_plan, "Decaimiento espacial del habla"),
    ("impulse_response", _impulse_response, "Respuesta al impulso ISO 18233"),
    ("shaped_sweep", _shaped_sweep, "Barrido conformado"),
    ("image_source", _image_source, "Tiempo de llegada [ms]"),
    ("steady_field", _steady_field, "Distancia a la fuente [m]"),
    ("room_modes", _room_modes, "Modos de la sala"),
    ("crowd_noise", _crowd_noise, "Ruido autogenerado del público"),
]


@pytest.mark.parametrize(
    ("name", "build", "spanish"), _CASES, ids=[c[0] for c in _CASES]
)
def test_plot_spanish_labels(name: str, build, spanish: str) -> None:
    result = build()
    ax = result.plot(language="es")
    assert spanish in _texts(ax)
    plt.close("all")


@pytest.mark.parametrize(
    ("name", "build", "spanish"), _CASES, ids=[c[0] for c in _CASES]
)
def test_plot_rejects_unknown_language(name: str, build, spanish: str) -> None:
    result = build()
    with pytest.raises(ValueError, match="Unknown language"):
        result.plot(language="xx")
    plt.close("all")


def test_plot_excitation_spanish_and_rejects_unknown_language() -> None:
    signal = np.sin(2.0 * np.pi * 220.0 * np.arange(2048) / FS)
    ax = plot_excitation(signal, FS, kind="sweep", language="es")
    assert "Barrido sinusoidal exponencial ISO 18233" in _texts(ax)
    plt.close("all")
    with pytest.raises(ValueError, match="Unknown language"):
        plot_excitation(signal, FS, kind="sweep", language="xx")
    plt.close("all")
