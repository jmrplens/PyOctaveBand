#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Geometry drawings of the secondary set-ups (``_plot/geometry/``).

Facade elevation, double wall, plate junctions, the in-situ absorption
set-up, the dynamic-stiffness rig, the diffusion goniometer, the baffled
plate, the open-plan line and the p-p probe: smoke, retention round-trips,
refusal paths and validation, mirroring the earlier geometry test files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import phonometry as pm

if TYPE_CHECKING:
    from collections.abc import Iterator

FREQ = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0])


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def test_facade_result_retains_elements_and_draws() -> None:
    elements = [
        pm.building.FacadeElement("Masonry wall", area=6.0, r=[50.0] * 5),
        pm.building.FacadeElement("Window", area=1.5, r=[30.0] * 5),
    ]
    res = pm.building.facade_sound_reduction(
        elements, area=7.5, volume=30.0, frequencies=FREQ
    )
    assert res.elements == tuple(elements)
    assert res.plot_geometry() is not None
    with pytest.raises(ValueError, match="at least one"):
        pm.building.plot_facade_elements([])


def test_a_facade_element_refuses_a_nan_area() -> None:
    """NaN fails every comparison, so a bare ``<= 0.0`` would wave it into
    the tile widths and render a titled, empty elevation.
    """
    with pytest.raises(ValueError, match="'area' must be positive"):
        pm.building.FacadeElement("Window", area=float("nan"), r=[30.0] * 5)


def test_double_wall_result_retains_geometry_and_draws() -> None:
    res = pm.building.double_wall_transmission_loss(FREQ, 8.8, 8.8, 0.1)
    assert res.gap == pytest.approx(0.1)
    assert res.plot_geometry() is not None
    single = pm.building.single_panel_transmission_loss(
        FREQ, 10.0, critical_frequency=1200.0
    )
    with pytest.raises(ValueError, match="does not retain"):
        single.plot_geometry()
    with pytest.raises(ValueError, match="'mass1' must be positive"):
        pm.building.plot_double_wall_geometry(0.0, 8.8, 0.1)


def test_double_wall_geometry_refuses_a_nan_resonance_frequency() -> None:
    """The annotation between the leaves would read 'f$_0$ = nan Hz' on an
    otherwise complete section.
    """
    with pytest.raises(ValueError, match="'resonance_frequency' must be positive"):
        pm.building.plot_double_wall_geometry(
            8.8, 8.8, 0.05, resonance_frequency=float("nan")
        )


def test_junction_result_retains_thicknesses_and_draws() -> None:
    res = pm.vibration.junction_transmission(
        "T1", 0.14, 3500.0, 320.0, 0.2, 3500.0, 460.0
    )
    assert res.thickness1 == pytest.approx(0.14)
    assert res.plot_geometry() is not None
    for junction in ("L", "T1", "T2", "X"):
        assert pm.building.plot_junction_geometry(junction, 0.14, 0.2) is not None
    with pytest.raises(ValueError, match="junction"):
        pm.building.plot_junction_geometry("Y", 0.14, 0.2)


def test_junction_geometry_rejects_nan_thickness() -> None:
    """A NaN thickness would draw collapsed plates under a 'nan mm' label."""
    with pytest.raises(ValueError, match="thickness1"):
        pm.building.plot_junction_geometry("L", float("nan"), 0.1)


def test_junction_geometry_rejects_nan_second_thickness() -> None:
    with pytest.raises(ValueError, match="thickness2"):
        pm.building.plot_junction_geometry("L", 0.14, float("nan"))


def test_insitu_setup_defaults_and_retention() -> None:
    assert pm.materials.plot_insitu_geometry() is not None
    with pytest.raises(ValueError, match="mic_height"):
        pm.materials.plot_insitu_geometry(source_height=0.2, mic_height=0.25)
    fs = 48000
    impulse = np.zeros(2048)
    impulse[100] = 1.0
    res = pm.materials.insitu_absorption_spectrum(
        impulse, 0.4 * impulse, fs, source_height=1.25, mic_height=0.25
    )
    assert res.source_height == pytest.approx(1.25)
    assert res.plot_geometry() is not None


def test_dynamic_stiffness_rig_draws_and_validates() -> None:
    assert pm.materials.plot_dynamic_stiffness_rig() is not None
    with pytest.raises(ValueError, match="'load_mass' must be positive"):
        pm.materials.plot_dynamic_stiffness_rig(load_mass=0.0)


def test_dynamic_stiffness_rig_refuses_a_nan_load_mass() -> None:
    """NaN passes a bare ``<= 0.0`` and letters 'Load plate nan kg' on an
    otherwise correct rig.
    """
    with pytest.raises(ValueError, match="'load_mass' must be positive"):
        pm.materials.plot_dynamic_stiffness_rig(load_mass=float("nan"))


def test_goniometer_microphone_count() -> None:
    ax = pm.materials.plot_goniometer_geometry()
    counts = [len(c.get_offsets()) for c in ax.collections]
    assert 37 in counts
    with pytest.raises(ValueError, match="angular_step"):
        pm.materials.plot_goniometer_geometry(angular_step=0.0)


def test_plate_geometry_from_result() -> None:
    res = pm.vibration.radiation_efficiency(FREQ, 1.2, 0.8, 1000.0)
    assert res.plot_geometry() is not None
    with pytest.raises(ValueError, match="'length_x' must be positive"):
        pm.vibration.plot_plate_geometry(0.0, 0.8)


def test_plate_geometry_rejects_nan_length() -> None:
    """A NaN length would draw no plate under a 'nan m' dimension label."""
    with pytest.raises(ValueError, match="length_x"):
        pm.vibration.plot_plate_geometry(float("nan"), 1.2)


def test_plate_geometry_rejects_nan_width() -> None:
    with pytest.raises(ValueError, match="length_y"):
        pm.vibration.plot_plate_geometry(1.2, float("nan"))


def test_open_plan_result_retains_positions_and_draws() -> None:
    res = pm.room.open_plan_metrics(
        [2.0, 4.0, 6.0, 8.0, 12.0],
        [55.0, 51.0, 48.0, 46.0, 43.0],
        [0.9, 0.72, 0.6, 0.5, 0.35],
    )
    assert res.positions_m is not None
    assert res.plot_geometry() is not None
    with pytest.raises(ValueError, match="at least two"):
        pm.room.plot_open_plan_geometry([3.0])


def test_open_plan_geometry_refuses_a_nan_microphone_position() -> None:
    """NaN passes ``pos <= 0.0`` and draws a line whose span dimension
    reads 'nan m' with the workstation blocks silently missing.
    """
    with pytest.raises(ValueError, match="'positions' must be finite"):
        pm.room.plot_open_plan_geometry([1.0, float("nan"), 3.0])


def test_open_plan_geometry_refuses_an_infinite_microphone_position() -> None:
    with pytest.raises(ValueError, match="'positions' must be finite"):
        pm.room.plot_open_plan_geometry([1.0, float("inf")])


def test_intensity_result_retains_spacing_and_draws() -> None:
    fs = 48000
    t = np.arange(fs // 4) / fs
    p1 = np.sin(2.0 * np.pi * 500.0 * t)
    p2 = np.sin(2.0 * np.pi * 500.0 * t - 0.05)
    res = pm.emission.sound_intensity(p1, p2, fs, spacing=0.012)
    assert res.spacing == pytest.approx(0.012)
    assert res.plot_geometry() is not None
    assert pm.emission.plot_pp_probe_geometry() is not None
    with pytest.raises(ValueError, match="spacing"):
        pm.emission.plot_pp_probe_geometry(0.0)


def test_secondary_geometry_spanish_labels() -> None:
    ax = pm.materials.plot_dynamic_stiffness_rig(language="es")
    texts = " ".join(t.get_text() for t in ax.texts) + ax.get_title()
    assert "Probeta" in texts
    assert "Banco de rigidez dinámica" in ax.get_title()
    with pytest.raises(ValueError, match="Unknown language"):
        pm.materials.plot_goniometer_geometry(language="ca")
