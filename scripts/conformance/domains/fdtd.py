#  Copyright (c) 2026. Jose Manuel Requena Plens
"""2D FDTD wave simulation (Attenborough & Van Renterghem 2021, Ch. 4).

The finite-difference time-domain solver checked against the three things a
wave equation solver must reproduce exactly: the eigenfrequency of a mode of a
rigid rectangular box, the arrival delay of a free-field pulse, and the
Kirchhoff-Helmholtz near-field-to-far-field transform of a known radiator.

All three are analytic, so the tolerance is the discretization error of the
grid and nothing else.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

import phonometry as ph

from ..registry import Outcome, numeric, register

_FDTD = "2D FDTD wave simulation (Attenborough & Van Renterghem 2021, Ch. 4)"


@register(
    _FDTD,
    "Rigid rectangular box eigenfrequency",
    "Mode (1,1) of a 1.0 x 0.7 m rigid box, f = (c/2)*sqrt(1/lx^2 + 1/ly^2), Hz",
)
def _chk_fdtd_box_mode() -> Outcome:
    lx, ly, dx, c = 1.0, 0.7, 0.02, 343.0
    nx, ny = round(lx / dx), round(ly / dx)
    res = ph.simulation.fdtd_simulation(
        c,
        dx,
        0.35,
        shape=(ny, nx),
        sources=[ph.simulation.GaussianPulse(ix=7, iy=5, width=2.0e-4)],
        probes=[(nx - 4, ny - 3)],
    )
    expected = 0.5 * c * math.hypot(1.0 / lx, 1.0 / ly)
    pressure = res.pressures[0]
    spec = np.abs(
        np.fft.rfft(pressure * np.hanning(pressure.size), n=8 * pressure.size)
    )
    freqs = np.fft.rfftfreq(8 * pressure.size, res.dt)
    sel = (freqs > 0.93 * expected) & (freqs < 1.07 * expected)
    measured = float(freqs[sel][np.argmax(spec[sel])])
    return numeric(expected, measured, 1.5, unit="Hz", places=2)


@register(
    _FDTD,
    "Free-field pulse arrival delay",
    "Probe-to-probe delay of a pulse over 0.6 m of air, (r2 - r1)/c, ms",
)
def _chk_fdtd_pulse_delay() -> Outcome:
    c, dx = 343.0, 0.01
    res = ph.simulation.fdtd_simulation(
        c,
        dx,
        6.5e-3,
        shape=(200, 300),
        sources=[ph.simulation.GaussianPulse(ix=40, iy=100, width=1.5e-4)],
        probes=[(100, 100), (160, 100)],
        boundaries="absorbing",
        absorbing_layer_cells=30,
    )
    t1 = res.times[int(np.argmax(res.pressures[0]))]
    t2 = res.times[int(np.argmax(res.pressures[1]))]
    expected = (160 - 100) * dx / c * 1e3
    return numeric(expected, (t2 - t1) * 1e3, 0.05, unit="ms", places=3)


_NTFF_CACHE: dict[str, Any] = {}


def _ntff_monopole() -> tuple[Any, complex, float]:
    """Far-field pattern of an enclosed CW monopole and its analytic level.

    One steady-state 2 kHz run: a contour probe around the source feeds
    the Kirchhoff-Helmholtz far-field integral, and an independent probe
    cell 0.5 m away (open air, clear of the sponge ramp) fits the
    amplitude ``A`` of the analytic line-source field ``p = A H0(2)(k r)``,
    whose exact far-field pattern is the omnidirectional
    ``A sqrt(2 / (pi k)) exp(j pi / 4)``.
    """
    if "monopole" in _NTFF_CACHE:
        out = _NTFF_CACHE["monopole"]
        return cast("tuple[Any, complex, float]", out)
    from scipy.special import hankel2

    c, dx, f = 343.0, 0.005, 2000.0
    k = 2.0 * np.pi * f / c
    sim = ph.simulation.FDTD2D(c, dx, shape=(300, 300), sponge_width=40)
    sim.add_source(ph.simulation.CWSource(ix=150, iy=150, frequency=f, ramp_cycles=4.0))
    probe = sim.add_contour_probe(90, 210, 90, 210, frequencies=[f])
    sim.run(round(4.5e-3 / sim.dt))
    probe.reset()
    probe_col = 250  # open air: the sponge ramp starts at 260
    acc = 0.0 + 0.0j
    for _ in range(round(10.0 / f / sim.dt)):
        sim.step()
        acc += sim.p[150, probe_col] * np.exp(-2j * np.pi * f * sim.n * sim.dt)
    amplitude = (2.0 * acc / probe.samples) / hankel2(0, k * (probe_col - 150) * dx)
    pattern = ph.simulation.far_field_from_contour(
        probe.phasors(f), np.arange(0.0, 360.0, 5.0), origin=(150.5 * dx, 150.5 * dx)
    )
    expected = abs(amplitude) * math.sqrt(2.0 / (math.pi * k))
    result = (pattern, complex(amplitude), float(expected))
    _NTFF_CACHE["monopole"] = result
    return result


@register(
    _FDTD,
    "2D Kirchhoff-Helmholtz NTFF: monopole directivity",
    "Far-field pattern ripple of an enclosed line source, dB",
)
def _chk_ntff_monopole_ripple() -> Outcome:
    pattern, _, _ = _ntff_monopole()
    levels = 20.0 * np.log10(np.abs(pattern))
    return numeric(0.0, float(levels.max() - levels.min()), 0.2, unit="dB", places=3)


@register(
    _FDTD,
    "2D Kirchhoff-Helmholtz NTFF: monopole level",
    "NTFF far-field level vs the 2D Green function A sqrt(2/(pi k)), dB",
)
def _chk_ntff_monopole_level() -> Outcome:
    pattern, _, expected = _ntff_monopole()
    mean_level = 20.0 * np.log10(float(np.mean(np.abs(pattern))) / expected)
    return numeric(0.0, mean_level, 0.3, unit="dB", places=3)
