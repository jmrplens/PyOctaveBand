#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Geometry drawings of measurement and control devices (``_plot/geometry/``).

Covers the device renderers added after the materials batch: reactive
silencers, the image-source room plan, the barrier section, microphone
position arrays in 3-D, wall apertures, the baffled piston, the plenum
section and the FDTD domain preview. Smoke plus the retention round-trips,
refusal paths and input validation, mirroring the materials geometry tests.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import phonometry as pm

FREQ = np.linspace(50.0, 1600.0, 24)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Reactive silencers.
# ---------------------------------------------------------------------------
def test_silencer_results_retain_geometry_and_draw() -> None:
    cases = (
        pm.noise_control.expansion_chamber(FREQ, 0.3, 0.03, 0.005),
        pm.noise_control.extended_tube_chamber(
            FREQ, 0.3, 0.03, 0.005, inlet_extension=0.075,
            outlet_extension=0.05,
        ),
        pm.noise_control.helmholtz_resonator(FREQ, 0.01, 0.001, 0.05, 0.002),
        pm.noise_control.quarter_wave_resonator(FREQ, 0.01, 0.4, 0.003),
    )
    for result in cases:
        assert result.geometry is not None
        ax = result.plot_geometry()
        assert ax.get_aspect() == 1.0
    assert cases[0].geometry == {
        "length": 0.3, "chamber_area": 0.03, "pipe_area": 0.005,
    }


def test_silencer_hand_built_refuses() -> None:
    result = pm.noise_control.expansion_chamber(FREQ, 0.3, 0.03, 0.005)
    import dataclasses

    bare = dataclasses.replace(result, geometry=None)
    with pytest.raises(ValueError, match="does not retain"):
        bare.plot_geometry()


#: A 1 Hz grid, fine enough to resolve a branch tuning exactly and low
#: enough to stay under the 400 mm chamber's cut-on.
CHAIN_FREQ = np.linspace(20.0, 500.0, 481)


def _chain(branch_length: float = 0.686) -> pm.noise_control.SilencerChain:
    """A duct-shunt-duct-chamber-duct chain with one quarter-wave branch."""
    from phonometry.noise_control.silencers import quarter_wave_impedance

    return (
        pm.noise_control.SilencerChain(CHAIN_FREQ)
        .duct(0.15, 0.0314)
        .shunt(
            quarter_wave_impedance(CHAIN_FREQ, branch_length, 0.00785),
            label="Quarter-wave stub",
        )
        .duct(0.15, 0.0314)
        .duct(0.60, 0.1257)
        .duct(0.30, 0.0314)
    )


def _lengths_lettered(ax) -> set[str]:
    """Every measurement the drawing letters, as it letters it."""
    return {
        text.get_text() for text in ax.texts if text.get_text().endswith(" mm")
    }


def test_silencer_chain_draws_its_ducts_and_marks_its_branch() -> None:
    ax = _chain().plot_geometry()
    assert ax.get_aspect() == 1.0
    assert "element chain" in ax.get_title()
    labels = [text.get_text() for text in ax.texts]
    assert "Quarter-wave stub\nmin |Z| at 125 Hz" in labels
    # Every dimension on the page is a declared duct length, a declared bore
    # or their sum: nothing the branch would have to supply.
    assert _lengths_lettered(ax) == {
        "150 mm", "600 mm", "300 mm", "L = 1200 mm", "199.9 mm", "400.1 mm",
    }


def test_silencer_chain_draws_nothing_from_the_branch_impedance() -> None:
    # The honesty property: change the branch, and only its own callout
    # changes. Not one line, patch or dimension of the drawn duct moves,
    # because the impedance declares no geometry to move them.
    first = _chain().plot_geometry()
    second = _chain(branch_length=0.343).plot_geometry()
    assert _lengths_lettered(first) == _lengths_lettered(second)
    bounds = [patch.get_bbox().bounds for patch in first.patches]
    assert bounds == [patch.get_bbox().bounds for patch in second.patches]
    assert "min |Z| at 125 Hz" in first.texts[0].get_text()
    assert "min |Z| at 250 Hz" in second.texts[0].get_text()


def test_silencer_chain_result_and_language() -> None:
    chain = _chain()
    result = chain.result(inlet_area=0.0314, outlet_area=0.0314)
    assert "element chain" in result.plot_geometry().get_title()
    ax = chain.plot_geometry(language="es")
    assert "cadena de elementos" in ax.get_title()
    labels = [text.get_text() for text in ax.texts]
    # The user's own words for the branch stay as written; ours translate.
    assert "Quarter-wave stub\n|Z| mínima en 125 Hz" in labels
    assert any("Ramal lateral" in text for text in labels) is False
    assert any("Conductos dibujados a escala" in text for text in labels)
    assert "L = 1200 mm" in _lengths_lettered(ax)
    with pytest.raises(ValueError, match="Unknown language"):
        chain.plot_geometry(language="fr")


def test_silencer_chain_numbers_unlabelled_branches() -> None:
    ax = (
        pm.noise_control.SilencerChain(FREQ)
        .shunt(1.0e4)
        .duct(0.4, 0.02)
        .shunt(1.0e4)
        .plot_geometry()
    )
    labels = [text.get_text() for text in ax.texts]
    assert "Side branch 1" in labels
    assert "Side branch 2" in labels


def test_silencer_chain_without_a_duct_refuses_to_draw() -> None:
    chain = pm.noise_control.SilencerChain(FREQ).shunt(1.0e4)
    with pytest.raises(ValueError, match="no duct of positive length"):
        chain.plot_geometry()
    # A zero-length duct is the identity matrix: nothing acoustically, and
    # nothing to draw either.
    flat = pm.noise_control.SilencerChain(FREQ).duct(0.0, 0.02)
    with pytest.raises(ValueError, match="no duct of positive length"):
        flat.plot_geometry()


def test_silencer_free_function_validation() -> None:
    with pytest.raises(ValueError, match="Unknown silencer kind"):
        pm.noise_control.plot_silencer_geometry("muffler")
    with pytest.raises(ValueError, match="chamber_area"):
        pm.noise_control.plot_silencer_geometry(
            "expansion chamber", length=0.3, chamber_area=0.004,
            pipe_area=0.005,
        )
    with pytest.raises(ValueError, match="must not exceed"):
        pm.noise_control.plot_silencer_geometry(
            "extended-tube chamber", length=0.1, chamber_area=0.03,
            pipe_area=0.005, inlet_extension=0.08, outlet_extension=0.08,
        )
    with pytest.raises(ValueError, match="needs"):
        pm.noise_control.plot_silencer_geometry(
            "Helmholtz resonator", duct_area=0.01
        )


# ---------------------------------------------------------------------------
# Image-source plan.
# ---------------------------------------------------------------------------
def test_image_source_plan_draws_and_caps_order() -> None:
    res = pm.room.image_source_rir(
        (5.0, 4.0, 3.0), (1.5, 1.2, 1.5), (3.5, 2.8, 1.2), 0.3,
        fs=8000, max_order=4,
    )
    ax = res.plot_geometry(max_order=2)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("2" in text for text in labels)
    assert not any("3" in text for text in labels)
    with pytest.raises(ValueError, match="max_order"):
        res.plot_geometry(max_order=0)


# ---------------------------------------------------------------------------
# Barrier section.
# ---------------------------------------------------------------------------
def test_barrier_result_retains_geometry_and_draws() -> None:
    res = pm.environment.barrier_insertion_loss(FREQ, 1.5, 5.0, 3.0, 20.0, 1.5)
    assert res.barrier_height == pytest.approx(3.0)
    assert res.thickness is None
    assert res.plot_geometry() is not None
    thick = pm.environment.barrier_insertion_loss(
        FREQ, 1.5, 5.0, 3.0, 20.0, 1.5, thickness=0.4
    )
    assert thick.thickness == pytest.approx(0.4)
    assert thick.plot_geometry() is not None


def test_barrier_free_function_validation() -> None:
    with pytest.raises(ValueError, match="receiver_distance"):
        pm.environment.plot_barrier_geometry(
            source_height=1.5, barrier_distance=5.0, barrier_height=3.0,
            receiver_distance=4.0, receiver_height=1.5,
        )


# ---------------------------------------------------------------------------
# Microphone positions (3-D).
# ---------------------------------------------------------------------------
def test_microphone_positions_hemisphere_and_sphere() -> None:
    pos = pm.emission.measurement_positions("hemisphere", radius=2.0)
    ax = pm.emission.plot_microphone_positions(pos, radius=2.0)
    assert ax.name == "3d"
    sphere = pm.emission.precision_positions("sphere", radius=1.0, count=20)
    ax = pm.emission.plot_microphone_positions(sphere)
    assert ax.name == "3d"
    bad = np.zeros((3, 2))
    with pytest.raises(ValueError, match="shape"):
        pm.emission.plot_microphone_positions(bad)
    with pytest.raises(ValueError, match="radius"):
        pm.emission.plot_microphone_positions(pos, radius=-1.0)


# ---------------------------------------------------------------------------
# Wall aperture.
# ---------------------------------------------------------------------------
def test_aperture_results_retain_geometry_and_draw() -> None:
    slit = pm.building.slit_transmission_coefficient(FREQ, 0.003, 0.1)
    assert slit.width == pytest.approx(0.003)
    assert slit.depth == pytest.approx(0.1)
    assert slit.plot_geometry() is not None
    hole = pm.building.circular_aperture_transmission_coefficient(
        FREQ, 0.005, 0.1
    )
    assert hole.radius == pytest.approx(0.005)
    assert hole.plot_geometry() is not None


def test_aperture_validation() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        pm.building.plot_aperture_geometry(0.1)
    with pytest.raises(ValueError, match="exactly one"):
        pm.building.plot_aperture_geometry(0.1, width=0.003, radius=0.005)
    with pytest.raises(ValueError, match="depth"):
        pm.building.plot_aperture_geometry(0.0, width=0.003)


# ---------------------------------------------------------------------------
# Baffled piston.
# ---------------------------------------------------------------------------
def test_piston_geometry_with_and_without_lobe() -> None:
    angles = np.linspace(-np.pi / 2, np.pi / 2, 91)
    with_lobe = pm.electroacoustics.radiating_piston(
        0.1, np.array([500.0, 4000.0]), angles=angles
    )
    ax = with_lobe.plot_geometry()
    assert ax.get_legend() is not None
    without = pm.electroacoustics.radiating_piston(0.1, np.array([500.0]))
    ax = without.plot_geometry()
    assert ax.get_legend() is None
    with pytest.raises(ValueError, match="radius"):
        pm.electroacoustics.plot_piston_geometry(0.0)
    with pytest.raises(ValueError, match="together"):
        pm.electroacoustics.plot_piston_geometry(0.1, angles=angles)


# ---------------------------------------------------------------------------
# Plenum.
# ---------------------------------------------------------------------------
def test_plenum_geometry_draws_and_validates() -> None:
    ax = pm.noise_control.plot_plenum_geometry(0.09, 1.2, 6.0, angle=0.35)
    assert ax.get_aspect() == 1.0
    with pytest.raises(ValueError, match="positive"):
        pm.noise_control.plot_plenum_geometry(0.0, 1.2, 6.0)
    with pytest.raises(ValueError, match="angle"):
        pm.noise_control.plot_plenum_geometry(0.09, 1.2, 6.0, angle=2.0)


# ---------------------------------------------------------------------------
# FDTD domain preview.
# ---------------------------------------------------------------------------
def test_fdtd_domain_preview() -> None:
    from phonometry.simulation import FDTD2D, GaussianPulse

    mask = np.zeros((40, 60), dtype=bool)
    mask[15:25, 28:31] = True
    sim = FDTD2D(
        343.0, 0.05, shape=(40, 60), sponge_width=6,
        sponge_sides=("left", "right"), edge_impedance={"top": 413.0},
        obstacle_mask=mask,
    )
    sim.add_source(GaussianPulse(8, 20, width=1e-3))
    assert sim.sponge_width == 6
    assert sim.sponge_sides == ("left", "right")
    assert set(sim.edge_impedance) == {"top"}
    ax = sim.plot_geometry(probes=[(2.0, 1.0)])
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert len(labels) == 5
    bad_probes = np.zeros((2, 3))
    with pytest.raises(ValueError, match="probes"):
        sim.plot_geometry(probes=bad_probes)
    # A plain rigid box keeps the record empty and still draws.
    plain = FDTD2D(343.0, 0.05, shape=(10, 10))
    assert plain.sponge_sides == ()
    assert plain.edge_impedance == {}
    assert plain.plot_geometry() is not None


def test_device_geometry_language_validation() -> None:
    with pytest.raises(ValueError, match="Unknown language"):
        pm.noise_control.plot_plenum_geometry(0.09, 1.2, 6.0, language="fr")
    with pytest.raises(ValueError, match="Unknown language"):
        pm.environment.plot_barrier_geometry(
            source_height=1.5, barrier_distance=5.0, barrier_height=3.0,
            receiver_distance=20.0, receiver_height=1.5, language="pt",
        )


def test_device_geometry_spanish_labels() -> None:
    res = pm.environment.barrier_insertion_loss(FREQ, 1.5, 5.0, 3.0, 20.0, 1.5)
    ax = res.plot_geometry(language="es")
    texts = " ".join(t.get_text() for t in ax.texts) + " " + " ".join(
        t.get_text() for t in ax.get_legend().get_texts()
    )
    assert "Fuente" in texts
    assert "Camino difractado" in texts
