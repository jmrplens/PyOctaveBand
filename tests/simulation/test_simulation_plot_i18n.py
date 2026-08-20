#  Copyright (c) 2026. Jose Manuel Requena Plens

"""EN/ES language option of the simulation (FDTD) ``.plot()`` renderer."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from phonometry.simulation.fdtd import GaussianPulse, fdtd_simulation


def _result() -> object:
    mask = np.zeros((30, 40), dtype=bool)
    mask[12:18, 20:22] = True
    return fdtd_simulation(
        343.0,
        9.0e-3,
        3.0e-3,
        shape=(30, 40),
        sources=[GaussianPulse(ix=7, iy=5, width=3.0e-4)],
        probes=[(10, 5), (24, 10)],
        obstacle_mask=mask,
        snapshot_every=10,
    )


def test_fdtd_probes_es_and_bad_language() -> None:
    res = _result()
    ax = res.plot(kind="probes", language="es")
    assert ax.get_title() == "Presión en las sondas FDTD"
    assert ax.get_ylabel() == "Presión [Pa]"
    assert ax.get_xlabel() == "Tiempo [ms]"
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(s.startswith("sonda") for s in labels)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")


def test_fdtd_snapshot_es() -> None:
    res = _result()
    ax = res.plot(kind="snapshot", frame=-1, language="es")
    assert ax.get_title().startswith("Campo de presión FDTD en $t$ =")
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(kind="snapshot", language="xx")
