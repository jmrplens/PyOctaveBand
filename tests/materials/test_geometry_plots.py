#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Geometry drawings of materials devices (``_plot/geometry/``).

Smoke coverage plus the contracts that matter: every renderer returns the
axes it drew on with an equal aspect, results retain the geometry their
``plot_geometry()`` needs, hand-built results refuse to draw with a clear
message, and the drawings are really to scale (the sample patch of the
impedance tube spans exactly the drawn thickness).
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from phonometry import materials as m

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes

FREQ = np.linspace(200.0, 2000.0, 32)


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def _resonator() -> m.HelmholtzResonator:
    return m.HelmholtzResonator(
        neck_length=0.002,
        neck_side=0.002,
        cavity_length=0.015,
        cavity_side=0.01,
    )


def _layers() -> list:
    medium = m.miki(FREQ, 20000.0)
    return [
        m.MicroperforatedPlateLayer(0.001, 0.0002, 0.01),
        m.AirLayer(0.03),
        m.PorousLayer(0.05, medium),
    ]


# ---------------------------------------------------------------------------
# Smoke: every renderer returns an equal-aspect Axes.
# ---------------------------------------------------------------------------
def _geometry_axes() -> Iterator[Axes]:
    yield m.plot_absorber_stack(_layers())
    yield _resonator().plot()
    yield m.plot_slit_absorber_geometry(
        [_resonator()] * 3,
        slit_height=0.002,
        lattice_step=0.015,
        period=0.02,
    )
    yield m.plot_qrd_geometry(m.qrd_well_depths(7, 500.0), 0.12, periods=2)
    yield m.plot_impedance_tube_geometry(
        spacing=0.05, x1=0.15, diameter=0.1, shape="circular"
    )
    yield m.plot_transmission_tube_geometry(
        l1=0.1,
        s1=0.03,
        l2=0.15,
        s2=0.03,
        thickness=0.05,
        diameter=0.1,
        shape="square",
    )


def test_all_geometry_renderers_return_equal_aspect_axes() -> None:
    for ax in _geometry_axes():
        assert ax.get_aspect() == 1.0
        assert not ax.axison


def test_renderers_accept_external_ax_and_language() -> None:
    for language in ("en", "es"):
        _, ax = plt.subplots()
        out = m.plot_qrd_geometry(
            m.qrd_well_depths(7, 500.0), 0.12, ax=ax, language=language
        )
        assert out is ax


def test_unknown_language_rejected() -> None:
    resonator = _resonator()
    with pytest.raises(ValueError, match="Unknown language"):
        resonator.plot(language="fr")


# ---------------------------------------------------------------------------
# Geometry retention and the hand-built refusal paths.
# ---------------------------------------------------------------------------
def test_layered_result_retains_layers_and_draws() -> None:
    layers = _layers()
    res = m.layered_absorber(FREQ, layers)
    assert res.layers == tuple(layers)
    assert res.plot_geometry() is not None
    bare = m.LayeredAbsorberResult(
        frequency=res.frequency,
        angle=res.angle,
        surface_impedance=res.surface_impedance,
        normalized_impedance=res.normalized_impedance,
        reflection=res.reflection,
        absorption=res.absorption,
        transfer_matrix=res.transfer_matrix,
    )
    with pytest.raises(ValueError, match=r"does not retain its layers"):
        bare.plot_geometry()


def test_single_layer_dataclasses_plot() -> None:
    for layer in _layers():
        assert layer.plot() is not None
    assert m.MembraneLayer(0.5).plot() is not None
    assert m.PerforatedPlateLayer(0.01, 0.005, 0.2).plot() is not None


def test_slit_result_retains_geometry_and_draws() -> None:
    res = m.slit_helmholtz_absorber(
        FREQ,
        [_resonator()] * 3,
        slit_height=0.002,
        lattice_step=0.015,
        period=0.02,
    )
    assert res.resonators == (_resonator(),) * 3
    assert res.slit_height == pytest.approx(0.002)
    assert res.lattice_step == pytest.approx(0.015)
    assert res.period == pytest.approx(0.02)
    assert res.plot_geometry() is not None


def test_diffuser_response_retains_wells_and_draws() -> None:
    depths = m.qrd_well_depths(7, 500.0)
    res = m.predict_diffuser_polar_response(0.12, 1000.0, depths=depths, periods=2)
    assert res.well_width == pytest.approx(0.12)
    assert res.periods == 2
    assert res.depths is not None
    assert np.allclose(res.depths, depths)
    assert res.plot_geometry() is not None
    explicit = m.predict_diffuser_polar_response(
        0.12,
        1000.0,
        reflection=np.exp(-2j * np.pi * np.arange(7) / 7),
    )
    assert explicit.depths is None
    with pytest.raises(ValueError, match=r"does not retain its well geometry"):
        explicit.plot_geometry()


def test_impedance_tube_result_draws_from_retained_geometry() -> None:
    f = np.array([500.0, 1000.0])
    res = m.two_microphone_impedance(
        np.array([0.5 + 0.1j, 0.4 - 0.2j]),
        frequency=f,
        spacing=0.05,
        x1=0.15,
        speed_of_sound=343.2,
        characteristic_impedance=413.0,
        diameter=0.1,
    )
    assert res.plot_geometry() is not None
    bare = m.two_microphone_impedance(
        np.array([0.5 + 0.1j, 0.4 - 0.2j]),
        frequency=f,
        spacing=0.05,
        x1=0.15,
        speed_of_sound=343.2,
        characteristic_impedance=413.0,
    )
    # Without a diameter the drawing still works (nominal bore, no cut-on).
    assert bare.plot_geometry() is not None


def test_transfer_matrix_geometry_paths() -> None:
    f = np.array([500.0, 1000.0])
    k = 2.0 * np.pi * f / 343.2
    hand_built = m.air_layer_transfer_matrix(k, 0.05, 413.0)
    with pytest.raises(ValueError, match=r"does not retain its tube geometry"):
        hand_built.plot_geometry()


# ---------------------------------------------------------------------------
# To-scale checks and validation.
# ---------------------------------------------------------------------------
def test_impedance_tube_sample_patch_is_to_scale() -> None:
    thickness = 0.08
    ax = m.plot_impedance_tube_geometry(
        spacing=0.05, x1=0.15, diameter=0.1, sample_thickness=thickness
    )
    from matplotlib.patches import Rectangle

    widths = {
        round(p.get_width(), 6)
        for p in ax.patches
        if isinstance(p, Rectangle) and p.get_x() == 0.0
    }
    assert round(thickness, 6) in widths


def test_qrd_geometry_validation() -> None:
    with pytest.raises(ValueError, match=r"'depths' must be a non-empty 1-D sequence"):
        m.plot_qrd_geometry([], 0.12)
    with pytest.raises(ValueError, match=r"'depths' must be non-negative"):
        m.plot_qrd_geometry([-0.01], 0.12)
    with pytest.raises(ValueError, match=r"'well_width' must be positive"):
        m.plot_qrd_geometry([0.05], 0.0)
    with pytest.raises(ValueError, match=r"'periods' must be >="):
        m.plot_qrd_geometry([0.05], 0.12, periods=0)


def test_qrd_geometry_refuses_a_nan_well_depth() -> None:
    """NaN passes ``d < 0.0`` and poisons the drawn maximum depth into the
    well-width fallback: the profile would be drawn to a depth scale no well
    has, under a dimension reading the well width.
    """
    with pytest.raises(ValueError, match="'depths' must be finite"):
        m.plot_qrd_geometry([0.0, 0.02, float("nan"), 0.015], 0.03)


def test_transmission_tube_validation() -> None:
    with pytest.raises(
        ValueError,
        match=r"'l2' is measured from the front face and must exceed 'thickness'",
    ):
        m.plot_transmission_tube_geometry(
            l1=0.1, s1=0.03, l2=0.04, s2=0.03, thickness=0.05
        )
    with pytest.raises(ValueError, match="'s1' must be positive"):
        m.plot_transmission_tube_geometry(
            l1=0.1, s1=0.0, l2=0.15, s2=0.03, thickness=0.05
        )
    with pytest.raises(ValueError, match=r"'diameter' must be positive"):
        m.plot_transmission_tube_geometry(
            l1=0.1, s1=0.03, l2=0.15, s2=0.03, thickness=0.05, diameter=-1.0
        )


def test_impedance_tube_geometry_refuses_a_nan_spacing() -> None:
    """A NaN spacing passed the bare ``<= 0.0``, then poisoned the bore, the
    dimension line ('s = nan mm') and the working-range note
    ('Plane-wave range nan to nan Hz') of a drawing that still rendered.
    """
    with pytest.raises(ValueError, match="'spacing' must be positive"):
        m.plot_impedance_tube_geometry(spacing=float("nan"), x1=0.1)


def test_impedance_tube_geometry_refuses_a_nan_microphone_distance() -> None:
    with pytest.raises(ValueError, match="'x1' must be positive"):
        m.plot_impedance_tube_geometry(spacing=0.05, x1=float("nan"))


def test_transmission_tube_geometry_refuses_a_nan_section_length() -> None:
    with pytest.raises(ValueError, match="'l1' must be positive"):
        m.plot_transmission_tube_geometry(
            l1=float("nan"), s1=0.03, l2=0.15, s2=0.03, thickness=0.05
        )


def test_goniometer_geometry_refuses_a_nan_source_distance() -> None:
    """A NaN source distance passed the bare ``<= 0.0`` and drew the arc and
    the sample with the source star and its label silently missing.
    """
    with pytest.raises(ValueError, match="'source_distance' must be positive"):
        m.plot_goniometer_geometry(source_distance=float("nan"))


def test_insitu_geometry_refuses_a_nan_source_height() -> None:
    """``source_height <= mic_height`` is False for a NaN, so the old guard
    waved one into every path length of the set-up.
    """
    with pytest.raises(ValueError, match="'source_height' must be positive"):
        m.plot_insitu_geometry(source_height=float("nan"))


def test_metadiffuser_panel_geometry_refuses_a_nan_depth() -> None:
    with pytest.raises(ValueError, match="'depth' must be positive"):
        m.plot_metadiffuser_panel_geometry(
            [None, None], depth=float("nan"), period=0.02
        )


def test_slit_absorber_geometry_refuses_a_nan_lattice_step() -> None:
    resonator = _resonator()
    with pytest.raises(ValueError, match="'lattice_step' must be positive"):
        m.plot_slit_absorber_geometry(
            resonator,
            slit_height=0.002,
            lattice_step=float("nan"),
            period=0.02,
        )


def test_impedance_tube_validation() -> None:
    with pytest.raises(ValueError, match="'x1' must exceed 'spacing'"):
        m.plot_impedance_tube_geometry(spacing=0.05, x1=0.04)
    with pytest.raises(ValueError, match=r"'diameter' must be positive"):
        m.plot_impedance_tube_geometry(spacing=0.05, x1=0.15, diameter=-0.1)
    with pytest.raises(ValueError, match=r"'sample_thickness' must be positive"):
        m.plot_impedance_tube_geometry(spacing=0.05, x1=0.15, sample_thickness=0.0)
    with pytest.raises(ValueError, match="Unknown language"):
        m.plot_impedance_tube_geometry(spacing=0.05, x1=0.15, language="de")


def test_slit_geometry_validation() -> None:
    resonator = _resonator()
    with pytest.raises(ValueError, match="'slit_height' must be positive"):
        m.plot_slit_absorber_geometry(
            resonator, slit_height=0.0, lattice_step=0.015, period=0.02
        )
    with pytest.raises(
        ValueError, match=r"'resonators' must contain at least one resonator"
    ):
        m.plot_slit_absorber_geometry(
            [], slit_height=0.002, lattice_step=0.015, period=0.02
        )


def test_absorber_stack_validation() -> None:
    with pytest.raises(ValueError, match=r"'layers' must contain at least one layer"):
        m.plot_absorber_stack([])
    with pytest.raises(TypeError, match=r"Unsupported layer type"):
        m.plot_absorber_stack([object()])


# ---------------------------------------------------------------------------
# Spanish labels reach the artists.
# ---------------------------------------------------------------------------
def _texts(ax: Axes) -> str:
    return " ".join(t.get_text() for t in ax.texts) + " " + ax.get_title()


def test_spanish_strings_rendered() -> None:
    ax = m.plot_impedance_tube_geometry(
        spacing=0.05, x1=0.15, diameter=0.1, language="es"
    )
    joined = _texts(ax)
    assert "Altavoz" in joined
    assert "Muestra" in joined
    assert "a escala" in ax.get_title()
    ax = m.plot_absorber_stack(_layers(), language="es")
    assert "Respaldo rígido" in _texts(ax)
