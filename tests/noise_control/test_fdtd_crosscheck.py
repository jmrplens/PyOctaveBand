#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Independent FDTD cross-check of a single 2D expansion chamber.

The four-pole expansion-chamber transmission loss (Bies Eq. (8.111)) predicts,
for an area ratio ``m = S_exp / S_duct``, a ``TL`` peak of
``10 log10[1 + (1/4)(m - 1/m)^2]`` at ``kL = pi/2`` (``f = c / 4L``) and a
``TL = 0`` trough (fully transparent) at ``kL = pi`` (``f = c / 2L``).

This test drives an *independent* solver -- the 2D FDTD engine of
:mod:`phonometry.simulation` -- with a plane-wave duct that widens into a
chamber and narrows back, and compares the transmitted amplitude at the peak
and trough frequencies. The measured ``20 log10`` ratio of the transmitted
amplitudes (trough over peak) must reproduce the closed-form peak ``TL``. The
run is deterministic and small (a 16x200 grid, ~3000 steps, well under a
second).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry.noise_control import silencers as sl
from phonometry.simulation.fdtd import CWSource, fdtd_simulation

_C = 343.0
_DX = 0.02
_NY, _NX = 16, 200
_DUCT_ROWS = (6, 7, 8, 9)      # 4-cell duct; chamber is the full 16 rows -> m = 4
_X1, _X2 = 80, 100             # chamber columns
_L = (_X2 - _X1) * _DX         # chamber length, 0.4 m
_M = _NY / len(_DUCT_ROWS)     # area (height) ratio = 4


def _chamber_mask() -> np.ndarray:
    mask = np.ones((_NY, _NX), dtype=bool)
    for r in _DUCT_ROWS:
        mask[r, :] = False     # open the narrow inlet/outlet duct
    mask[:, _X1:_X2] = False    # open the full-height chamber
    return mask


def _transmitted_rms(frequency: float) -> float:
    mask = _chamber_mask()
    boundaries = {
        "left": "absorbing", "right": "absorbing",
        "top": "rigid", "bottom": "rigid",
    }
    src = CWSource(ix=25, iy=7, frequency=frequency, amplitude=1.0, ramp_cycles=6.0)
    steps = 3000
    dt = 0.6 * _DX / (_C * math.sqrt(2.0))
    res = fdtd_simulation(
        _C, _DX, steps * dt, sources=[src], shape=(_NY, _NX),
        probes=[(170, 7)], boundaries=boundaries,
        absorbing_layer_cells=10, obstacle_mask=mask,
    )
    p = res.pressures[0]
    tail = p[int(p.size * 0.6):]        # steady-state window
    return float(np.sqrt(np.mean(tail**2)))


def test_fdtd_matches_expansion_chamber_peak_tl() -> None:
    f_peak = _C / (4.0 * _L)    # kL = pi/2, four-pole TL maximum
    f_trough = _C / (2.0 * _L)  # kL = pi, four-pole TL = 0 (transparent)

    down_peak = _transmitted_rms(f_peak)
    down_trough = _transmitted_rms(f_trough)

    # The chamber transmits more at the trough than at the peak.
    assert down_trough > down_peak

    # The measured attenuation difference reproduces the closed-form peak TL.
    measured_delta = 20.0 * math.log10(down_trough / down_peak)
    peak_tl = 10.0 * math.log10(1.0 + 0.25 * (_M - 1.0 / _M) ** 2)
    assert peak_tl == pytest.approx(6.55, abs=0.05)
    assert measured_delta == pytest.approx(peak_tl, abs=1.2)


def test_fdtd_run_is_deterministic() -> None:
    # Two independent runs of the (RNG-free) solver must be bit-identical.
    f = _C / (4.0 * _L)
    first = _transmitted_rms(f)
    second = _transmitted_rms(f)
    assert first == second


def test_four_pole_peak_frequency_and_trough() -> None:
    # The four-pole model itself: TL peak at f = c/4L, TL = 0 at f = c/2L.
    f = np.array([_C / (4.0 * _L), _C / (2.0 * _L)])
    res = sl.expansion_chamber(f, _L, _M * 0.01, 0.01)
    assert res.transmission_loss[0] == pytest.approx(
        10.0 * math.log10(1.0 + 0.25 * (_M - 1.0 / _M) ** 2), abs=1e-6
    )
    assert res.transmission_loss[1] == pytest.approx(0.0, abs=1e-9)
