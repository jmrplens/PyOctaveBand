#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Geometry drawings of the secondary set-ups (``_plot/geometry.py``).

Facade elevation, double wall, plate junctions, the in-situ absorption
set-up, the dynamic-stiffness rig, the diffusion goniometer, the baffled
plate, the open-plan line and the p-p probe: smoke, retention round-trips,
refusal paths and validation, mirroring the earlier geometry test files.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import phonometry as pm

FREQ = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_facade_result_retains_elements_and_draws() -> None:
    elements = [
        pm.FacadeElement("Masonry wall", area=6.0, r=[50.0] * 5),
        pm.FacadeElement("Window", area=1.5, r=[30.0] * 5),
    ]
    res = pm.facade_sound_reduction(
        elements, area=7.5, volume=30.0, frequencies=FREQ
    )
    assert res.elements == tuple(elements)
    assert res.plot_geometry() is not None
    with pytest.raises(ValueError, match="at least one"):
        pm.plot_facade_elements([])


def test_double_wall_result_retains_geometry_and_draws() -> None:
    res = pm.double_wall_transmission_loss(FREQ, 8.8, 8.8, 0.1)
    assert res.gap == pytest.approx(0.1)
    assert res.plot_geometry() is not None
    single = pm.single_panel_transmission_loss(
        FREQ, 10.0, critical_frequency=1200.0
    )
    with pytest.raises(ValueError, match="does not retain"):
        single.plot_geometry()
    with pytest.raises(ValueError, match="positive"):
        pm.plot_double_wall_geometry(0.0, 8.8, 0.1)


def test_junction_result_retains_thicknesses_and_draws() -> None:
    res = pm.junction_transmission(
        "T1", 0.14, 3500.0, 320.0, 0.2, 3500.0, 460.0
    )
    assert res.thickness1 == pytest.approx(0.14)
    assert res.plot_geometry() is not None
    for junction in ("L", "T1", "T2", "X"):
        assert pm.plot_junction_geometry(junction, 0.14, 0.2) is not None
    with pytest.raises(ValueError, match="junction"):
        pm.plot_junction_geometry("Y", 0.14, 0.2)


def test_insitu_setup_defaults_and_retention() -> None:
    assert pm.plot_insitu_geometry() is not None
    with pytest.raises(ValueError, match="mic_height"):
        pm.plot_insitu_geometry(source_height=0.2, mic_height=0.25)
    fs = 48000
    impulse = np.zeros(2048)
    impulse[100] = 1.0
    res = pm.insitu_absorption_spectrum(
        impulse, 0.4 * impulse, fs, source_height=1.25, mic_height=0.25
    )
    assert res.source_height == pytest.approx(1.25)
    assert res.plot_geometry() is not None


def test_dynamic_stiffness_rig_draws_and_validates() -> None:
    assert pm.plot_dynamic_stiffness_rig() is not None
    with pytest.raises(ValueError, match="positive"):
        pm.plot_dynamic_stiffness_rig(load_mass=0.0)


def test_goniometer_microphone_count() -> None:
    ax = pm.plot_goniometer_geometry()
    counts = [len(c.get_offsets()) for c in ax.collections]
    assert 37 in counts
    with pytest.raises(ValueError, match="angular_step"):
        pm.plot_goniometer_geometry(angular_step=0.0)


def test_plate_geometry_from_result() -> None:
    res = pm.radiation_efficiency(FREQ, 1.2, 0.8, 1000.0)
    assert res.plot_geometry() is not None
    with pytest.raises(ValueError, match="positive"):
        pm.plot_plate_geometry(0.0, 0.8)


def test_open_plan_result_retains_positions_and_draws() -> None:
    res = pm.open_plan_metrics(
        [2.0, 4.0, 6.0, 8.0, 12.0],
        [55.0, 51.0, 48.0, 46.0, 43.0],
        [0.9, 0.72, 0.6, 0.5, 0.35],
    )
    assert res.positions_m is not None
    assert res.plot_geometry() is not None
    with pytest.raises(ValueError, match="at least two"):
        pm.plot_open_plan_geometry([3.0])


def test_intensity_result_retains_spacing_and_draws() -> None:
    fs = 48000
    t = np.arange(fs // 4) / fs
    p1 = np.sin(2.0 * np.pi * 500.0 * t)
    p2 = np.sin(2.0 * np.pi * 500.0 * t - 0.05)
    res = pm.sound_intensity(p1, p2, fs, 0.012)
    assert res.spacing == pytest.approx(0.012)
    assert res.plot_geometry() is not None
    assert pm.plot_pp_probe_geometry() is not None
    with pytest.raises(ValueError, match="spacing"):
        pm.plot_pp_probe_geometry(0.0)


def test_secondary_geometry_spanish_labels() -> None:
    ax = pm.plot_dynamic_stiffness_rig(language="es")
    texts = " ".join(t.get_text() for t in ax.texts) + ax.get_title()
    assert "Probeta" in texts
    assert "Banco de rigidez dinámica" in ax.get_title()
    with pytest.raises(ValueError, match="Unknown language"):
        pm.plot_goniometer_geometry(language="ca")
