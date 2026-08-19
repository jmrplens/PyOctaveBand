#  Copyright (c) 2026. Jose Manuel Requena Plens
"""EN/ES internationalisation of the materials-domain ``.plot()`` renderers.

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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phonometry import materials
from phonometry.materials.absorbers.airflow_resistance import (
    static_airflow_resistance,
)
from phonometry.materials.absorbers.impedance_tube import ImpedanceTubeResult
from phonometry.materials.absorbers.layered import (
    PorousLayer,
    diffuse_field_absorption,
    layered_absorber,
)
from phonometry.materials.absorbers.porous import (
    delany_bazley,
)
from phonometry.materials.absorbers.rating import weighted_absorption
from phonometry.materials.absorbers.sound_absorption import (
    measure_sound_absorption,
)
from phonometry.materials.diffusers.reverberation_room_scattering import (
    scattering_coefficient_spectrum,
)
from phonometry.materials.diffusers.scattering_diffusion import (
    directional_diffusion,
)
from phonometry.materials.resilient.dynamic_stiffness import (
    floating_floor_resonance,
)
from phonometry.materials.surfaces.road_absorption import (
    InsituAbsorptionResult,
)

_C0 = 343.0
_RHO0 = 1.205
_RC = _RHO0 * _C0


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


def _weighted():
    return weighted_absorption([0.35, 0.50, 0.65, 0.60, 0.55])


def _measurement():
    freqs = np.array([250.0, 500.0, 1000.0, 2000.0, 4000.0])
    t1 = np.array([7.8, 7.4, 6.9, 5.8, 4.6])
    t2 = np.array([6.5, 4.2, 2.85, 2.5, 2.7])
    return measure_sound_absorption(
        freqs, t1, t2, volume=200.0, area=10.8, temperature=20.0
    )


def _scattering():
    return scattering_coefficient_spectrum(
        [250.0, 500.0, 1000.0], [0.2, 0.3, 0.5], [0.1, 0.1, 0.1]
    )


def _diffusion():
    return directional_diffusion([-30.0, 0.0, 30.0], [70.0, 72.0, 69.0])


def _insitu():
    return InsituAbsorptionResult(
        frequencies=np.array([250.0, 500.0, 1000.0, 2000.0]),
        absorption=np.array([0.2, 0.4, 0.6, 0.5]),
    )


def _dynamic_stiffness():
    return floating_floor_resonance(25.0, 200.0, 100.0)


def _impedance_tube():
    f = np.array([500.0, 1000.0, 1500.0])
    reflection = np.array([0.4 - 0.3j, 0.3 - 0.2j, 0.2 - 0.1j])
    absorption = 1.0 - np.abs(reflection) ** 2
    normalized = (1.0 + reflection) / (1.0 - reflection)
    return ImpedanceTubeResult(
        frequency=f,
        reflection=reflection,
        surface_impedance=_RC * normalized,
        normalized_impedance=normalized,
        absorption=absorption,
    )


def _static_airflow():
    u = np.array([1.0e-3, 2.0e-3, 4.0e-3])
    dp = 12000.0 * u
    return static_airflow_resistance(u, dp, 0.008)


def _uncertainty():
    freqs = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
    alpha = [0.15, 0.35, 0.55, 0.70, 0.80, 0.85]
    return materials.sound_absorption_coefficient_uncertainty(alpha, freqs)


def _porous_medium():
    f = np.geomspace(100.0, 5000.0, 24)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return delany_bazley(f, 20000.0, air_density=_RHO0, speed_of_sound=_C0)


def _layered():
    f = np.geomspace(100.0, 5000.0, 24)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = delany_bazley(f, 20000.0, air_density=_RHO0, speed_of_sound=_C0)
        return layered_absorber(f, [PorousLayer(0.05, med)])


def _diffuse():
    f = np.geomspace(100.0, 5000.0, 24)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = delany_bazley(f, 20000.0, air_density=_RHO0, speed_of_sound=_C0)
        return diffuse_field_absorption(f, [PorousLayer(0.05, med)])


# (id, builder, expected Spanish substring) for every materials result renderer.
_CASES = [
    ("weighted_absorption", _weighted, "Coeficiente de absorción acústica"),
    ("sound_absorption", _measurement,
     "Absorción acústica en cámara reverberante ISO 354"),
    ("scattering", _scattering, "Coeficiente de dispersión"),
    ("diffusion_polar", _diffusion, "Coeficiente de difusión"),
    ("insitu_absorption", _insitu, "Absorción in situ de pavimentos"),
    ("dynamic_stiffness", _dynamic_stiffness, "Frecuencia natural"),
    ("impedance_tube", _impedance_tube, "Absorción a incidencia normal"),
    ("static_airflow", _static_airflow, "Velocidad lineal del aire"),
    ("absorption_uncertainty", _uncertainty, "Incertidumbre de absorción"),
    ("porous_medium", _porous_medium, "Medio poroso"),
    ("layered_absorber", _layered, "absorbente multicapa"),
    ("diffuse_field", _diffuse, "incidencia aleatoria"),
]


@pytest.mark.parametrize("name, build, spanish", _CASES, ids=[c[0] for c in _CASES])
def test_plot_spanish_labels(name: str, build, spanish: str) -> None:
    result = build()
    ax = result.plot(language="es")
    assert spanish in _texts(ax)
    plt.close("all")


@pytest.mark.parametrize("name, build, spanish", _CASES, ids=[c[0] for c in _CASES])
def test_plot_rejects_unknown_language(name: str, build, spanish: str) -> None:
    result = build()
    with pytest.raises(ValueError, match="Unknown language"):
        result.plot(language="xx")
    plt.close("all")
