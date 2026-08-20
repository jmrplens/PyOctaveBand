#  Copyright (c) 2026. Jose Manuel Requena Plens

"""What the material-characterisation ``.plot()`` renderers draw.

One subject seen from both ends of a laboratory: the impedance tube of
ISO 10534-2 and the transfer matrix of ASTM E2611 measure a sample, the porous
and layered models predict one, and ISO 9053-1 measures the airflow resistance
that feeds those models. Every one of them ends in the same pair of curves, an
absorption coefficient and a reflection magnitude, both bounded by one, which is
why the figures fix the axis at 0 to 1,05 rather than letting matplotlib choose.
The transfer-matrix figure puts the Eq. (26) transmission loss on the primary
axis in decibels and hangs the Eq. (28) hard-backed absorption on a twin, so the
two quantities keep their own scales. The airflow figure draws the fitted
Forchheimer curve in mm/s and must pass through the evaluation point the
standard reads the resistance at.

These are the content assertions. The generic plot contract lives in
``tests/test_result_plots.py``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from result_factories import (
    _impedance_tube,
    _layered_absorber,
    _porous_medium,
    _static_airflow,
    _transfer_matrix,
)


# --------------------------------------------------------------------------
# Impedance tube (ISO 10534-2)
# --------------------------------------------------------------------------
def test_impedance_tube_plot_alpha_and_reflection() -> None:
    res = _impedance_tube()
    ax = res.plot()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.absorption)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), np.abs(res.reflection))
    assert ax.get_ylim() == (0.0, 1.05)
    plt.close("all")


def test_transfer_matrix_plot_tl_and_absorption() -> None:
    tm, f, rho_c = _transfer_matrix()
    ax = tm.plot(f, rho_c)
    # Primary axis: the Eq. (26) transmission loss (0 dB for a pure air layer).
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), tm.transmission_loss(rho_c), atol=1e-12
    )
    # Twin axis: the Eq. (28) hard-backed absorption companion on a 0..1 scale.
    twins = [
        other
        for other in ax.figure.axes
        if other is not ax and other.bbox.bounds == ax.bbox.bounds
    ]
    assert len(twins) == 1
    np.testing.assert_allclose(
        twins[0].lines[0].get_ydata(), tm.absorption_hard_backed(rho_c)
    )
    assert twins[0].get_ylim() == (0.0, 1.05)
    plt.close("all")


def test_transfer_matrix_plot_forwards_kwargs_and_composes() -> None:
    tm, f, rho_c = _transfer_matrix()
    ax = tm.plot(f, rho_c, linewidth=2, color="red")
    line = ax.lines[0]
    assert line.get_linewidth() == 2.0
    assert plt.matplotlib.colors.to_rgba(line.get_color()) == (
        plt.matplotlib.colors.to_rgba("red")
    )
    plt.close("all")
    fig, external = plt.subplots()
    out = tm.plot(f, rho_c, ax=external)
    assert out is external
    plt.close(fig)


def test_transfer_matrix_plot_spanish_and_bad_language() -> None:
    tm, f, rho_c = _transfer_matrix()
    ax = tm.plot(f, rho_c, language="es")
    assert "ASTM E2611" in ax.get_title()
    assert "matriz de transferencia" in ax.get_title()
    plt.close("all")
    with pytest.raises(ValueError, match="language"):
        tm.plot(f, rho_c, language="fr")


def test_layered_absorber_plot_alpha_and_reflection() -> None:
    res = _layered_absorber()
    ax = res.plot()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.absorption)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), np.abs(res.reflection))
    assert ax.get_ylim() == (0.0, 1.05)
    plt.close("all")


def test_porous_medium_plot_normalized_components() -> None:
    res = _porous_medium()
    ax = res.plot()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.normalized_impedance.real)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), -res.normalized_impedance.imag)
    assert len(ax.lines) == 4
    plt.close("all")


# --------------------------------------------------------------------------
# Static airflow resistance (ISO 9053-1)
# --------------------------------------------------------------------------
def test_static_airflow_plot_curve_through_evaluation_point() -> None:
    res = _static_airflow()
    ax = res.plot()
    x, y = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    # x is in mm/s; the fitted curve passes through the evaluation point.
    at_eval = float(np.interp(res.evaluation_velocity * 1e3, x, y))
    assert at_eval == pytest.approx(res.pressure_drop, rel=1e-3)
    plt.close("all")
