#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the simulation guides: FDTD fields and numerical propagation.

The still figures of the numerically simulated wave field -- the 2D FDTD
snapshot, the room modes it resolves, the elastic half-space and Scholte
interface waves, and the meshed metadiffuser far field the analytic model is
checked against. The clips these belong with live in :mod:`figures.fields`.
Everything here is embedded by a page under ``simulation/``.
"""

from functools import cache
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .fields import _meshed_metadiffuser_ntff_levels
from .i18n import _LANG
from .materials import _qr_metadiffuser_wells
from .theme import (
    _FILENAME_SUFFIX,
    COLOR_FG,
    COLOR_GRID,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    FIELD_INK,
    LABEL_FREQ_HZ,
    save_figure,
)


def generate_metadiffuser_ntff_polar(output_dir: str) -> None:
    """The meshed metadiffuser radiates the far field the model predicts.

    Far-field polar response of the Table-1 metadiffuser at 2 kHz twice
    over: the TMM + Fraunhofer prediction of the library and a full 2D
    FDTD simulation of the meshed panel reduced through the
    Kirchhoff-Helmholtz near-to-far-field integral. One concept: the
    model chain is validated end to end by an independent full-wave
    solver, the library-side counterpart of the paper's TMM-vs-FEM
    cross-check.
    """
    print("Generating metadiffuser_ntff_polar...")
    from phonometry import metadiffuser_polar_response

    wells, depth, period = _qr_metadiffuser_wells()
    model = metadiffuser_polar_response(
        2000.0, wells, depth=depth, period=period, periods=1,
    )
    angles, levels = _meshed_metadiffuser_ntff_levels()
    _fig, ax = plt.subplots(
        figsize=(10, 6.2), subplot_kw={"projection": "polar"},
    )
    model.plot(
        ax=ax, color=COLOR_PRIMARY, marker="", linewidth=1.6,
        linestyle="--", label="TMM + Fraunhofer model", language=_LANG,
    )
    ax.plot(
        np.radians(angles), levels, color=COLOR_SECONDARY, linewidth=2.2,
        label="FDTD + NTFF, panel meshed at 0.5 mm",
    )
    ax.set_ylim(-40.0, 2.0)
    ax.set_title(
        "Far field of the meshed metadiffuser vs the model (2 kHz)",
        pad=18, fontweight="bold",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "metadiffuser_ntff_polar.svg")
    plt.close()


def generate_fdtd_domain_geometry(output_dir: str) -> None:
    """The configured FDTD domain drawn before any time stepping.

    A 4,5 x 3 m domain with sponge layers left and right, an impedance top
    edge, a rigid floor, a thin obstacle, one source and two probes: the
    setup drawing that precedes every simulation. One concept: check the
    domain before you run it.
    """
    print("Generating fdtd_domain_geometry...")
    from phonometry.simulation import FDTD2D, GaussianPulse

    mask = np.zeros((60, 90), dtype=bool)
    mask[25:35, 40:44] = True
    sim = FDTD2D(
        343.0, 0.05, shape=(60, 90), sponge_width=8,
        sponge_sides=("left", "right"), edge_impedance={"top": 413.0},
        obstacle_mask=mask,
    )
    sim.add_source(GaussianPulse(10, 30, width=1e-3))
    _fig, ax = plt.subplots(figsize=(10, 6.6))
    sim.plot_geometry(ax=ax, probes=[(3.0, 1.5), (4.0, 2.0)],
                      language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "fdtd_domain_geometry.svg")
    plt.close()


def generate_fdtd_simulation(output_dir: str) -> None:
    """2D FDTD simulation: diffraction snapshot and the probe histories."""
    print("Generating fdtd_simulation.png...")
    from phonometry import GaussianPulse, fdtd_simulation

    # A 3.0 x 2.0 m free field (absorbing edges) with a thin rigid barrier:
    # probe A sees the direct pulse plus the barrier reflection, probe B sits
    # in the shadow and only receives the wave diffracted around the edge.
    dx = 0.01
    mask = np.zeros((200, 300), dtype=bool)
    mask[60:, 150:154] = True
    res = fdtd_simulation(
        343.0, dx, 9.0e-3, shape=(200, 300),
        sources=[GaussianPulse(ix=60, iy=100, width=3.0e-4)],
        probes=[(100, 100), (240, 100)],
        obstacle_mask=mask,
        boundaries="absorbing", absorbing_layer_cells=30,
        snapshot_every=75,
    )

    _fig, (ax_f, ax_p) = plt.subplots(
        1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [1.25, 1.0]})
    res.plot(kind="snapshot", frame=7, ax=ax_f)
    res.plot(ax=ax_p)
    ax_p.set_title("FDTD probe pressure", fontweight="bold", pad=10)
    ax_f.set_title(ax_f.get_title(), fontweight="bold", pad=10)

    plt.tight_layout()
    save_figure(output_dir, "fdtd_simulation.png")
    plt.close()


def generate_elastic_halfspace_waves(output_dir: str) -> None:
    """Elastic FDTD: P, S and Rayleigh waves in an aluminium half-space."""
    print("Generating elastic_halfspace_waves...")
    from phonometry import (
        ElasticBoundaries,
        ElasticRecording,
        ForceSource,
        GaussianPulse,
        elastic_fdtd_simulation,
    )

    # A 0.6 x 0.3 m aluminium block with a free upper surface, struck by a
    # short vertical force at the middle of that surface (Lamb's problem):
    # one snapshot shows the three wave types at their own speeds, the
    # Rayleigh wave hugging the surface just behind the S front.
    c_p, c_s, rho, dx = 6320.0, 3130.0, 2700.0, 0.001
    width = 8e-6
    duration = 7.3e-5
    cfl = 0.6
    dt = cfl * dx / (c_p * np.sqrt(2.0))
    steps = round(duration / dt)
    res = elastic_fdtd_simulation(
        c_p, c_s, dx, duration, rho=rho, shape=(300, 600), cfl=cfl,
        sources=[ForceSource(ix=300, iy=0, direction="y", amplitude=1e6,
                             waveform=GaussianPulse(0, 0, width=width).value)],
        boundaries=ElasticBoundaries({"top": "free"}),
        recording=ElasticRecording(snapshot_every=steps, snapshot_field="vy"),
    )

    _fig, ax = plt.subplots(figsize=(9.5, 5.4))
    assert res.snapshots is not None and res.snapshot_times is not None
    vmax = 0.18 * float(np.abs(res.snapshots[-1]).max())
    res.plot(kind="snapshot", frame=-1, ax=ax, vmin=-vmax, vmax=vmax)

    # Wavefront radii from the pulse centre time (t0 = 4 * width).
    t_eff = float(res.snapshot_times[-1]) - 4.0 * width
    x0 = (300 + 0.5) * dx
    theta = np.linspace(0.0, np.pi, 181)
    for radius in (c_p * t_eff, c_s * t_eff):
        ax.plot(x0 + radius * np.cos(theta), radius * np.sin(theta),
                linestyle=":", linewidth=1.0, color=COLOR_FG, alpha=0.65)
    ann: dict[str, Any] = {
        "fontsize": 9, "color": COLOR_FG,
        "arrowprops": {"arrowstyle": "->", "color": COLOR_FG, "lw": 0.9}}
    r_p, r_s = c_p * t_eff, c_s * t_eff
    ax.annotate("P wave front",
                xy=(x0 + r_p * np.cos(np.radians(55)),
                    r_p * np.sin(np.radians(55))),
                xytext=(0.585, 0.15), ha="right", **ann)
    ax.annotate("S wave front",
                xy=(x0 + r_s * np.cos(np.radians(35)),
                    r_s * np.sin(np.radians(35))),
                xytext=(0.525, 0.038), ha="right", **ann)
    ax.annotate("Rayleigh wave",
                xy=(x0 - 0.92 * r_s, 0.004), xytext=(0.045, 0.06), **ann)
    ax.set_title("Rayleigh wave along a free aluminium surface "
                 "(elastic FDTD)", fontweight="bold", pad=10)

    plt.tight_layout()
    save_figure(output_dir, "elastic_halfspace_waves.png")
    plt.close()


@cache
def _scholte_interface_result() -> Any:
    """Water over a soft seabed, explosive shot near the contact (cached).

    Language/theme independent: the van Vossen (2002) benchmark media
    (water 1500/1000 over a 3500/2000/2500 solid), a 50 Hz Ricker
    explosion 10 m above the interface, run until the Scholte train has
    crawled ~330 m along the contact.
    """
    from phonometry import (
        ElasticBoundaries,
        ElasticRecording,
        ExplosionSource,
        elastic_fdtd_simulation,
    )

    ny, nx, dx = 200, 500, 1.0
    c_p = np.full((ny, nx), 1500.0)
    c_s = np.zeros((ny, nx))
    rho = np.full((ny, nx), 1000.0)
    c_p[100:], c_s[100:], rho[100:] = 3500.0, 2000.0, 2500.0
    f0, t0 = 50.0, 0.030
    duration = 0.232
    dt = 0.6 * dx / (3500.0 * np.sqrt(2.0))
    steps = round(duration / dt)

    def ricker(t: float) -> float:
        a = (np.pi * f0 * (t - t0)) ** 2
        return float((1.0 - 2.0 * a) * np.exp(-a))

    return elastic_fdtd_simulation(
        c_p, c_s, dx, duration, rho=rho,
        sources=[ExplosionSource(ix=60, iy=89, waveform=ricker,
                                 amplitude=1e3)],
        boundaries=ElasticBoundaries("absorbing", absorbing_layer_cells=20),
        recording=ElasticRecording(snapshot_every=steps, snapshot_field="vy"),
    )


def generate_scholte_interface_wave(output_dir: str) -> None:
    """Elastic FDTD: Scholte wave crawling along a water-seabed contact."""
    print("Generating scholte_interface_wave...")
    res = _scholte_interface_result()

    _fig, ax = plt.subplots(figsize=(9.5, 4.6))
    assert res.snapshots is not None
    vmax = 0.22 * float(np.abs(res.snapshots[-1]).max())
    res.plot(kind="snapshot", frame=-1, ax=ax, vmin=-vmax, vmax=vmax)
    # The light RdBu_r field stays light everywhere, so its in-axes
    # annotations keep a fixed dark ink rather than COLOR_FG; on the dark
    # theme the renderer picks the black-centred field, which takes the
    # white FIELD_INK of the animated clips.
    ink = FIELD_INK if _FILENAME_SUFFIX else "#3a3a3a"
    ax.axhline(100.0, color=ink, linestyle=":", linewidth=1.0, alpha=0.75)
    ann: dict[str, Any] = {
        "fontsize": 9, "color": ink,
        "arrowprops": {"arrowstyle": "->", "color": ink, "lw": 0.9}}
    ax.annotate("Scholte wave, evanescent on both sides",
                xy=(350.0, 108.0), xytext=(120.0, 170.0), **ann)
    ax.annotate("direct water wave", xy=(300.0, 35.0),
                xytext=(120.0, 14.0), **ann)
    txt: dict[str, Any] = {"fontsize": 9, "color": ink, "alpha": 0.9}
    ax.text(12.0, 32.0, "water 1500 m/s", **txt)
    ax.text(12.0, 192.0, "seabed 3500 / 2000 m/s", **txt)
    ax.set_title("Scholte wave along a water-sediment interface "
                 "(elastic FDTD)", fontweight="bold", pad=10)

    plt.tight_layout()
    save_figure(output_dir, "scholte_interface_wave.png")
    plt.close()


def generate_fdtd_room_modes(output_dir: str) -> None:
    """Rigid-box FDTD probe spectrum against the analytic room modes."""
    print("Generating fdtd_room_modes...")
    from phonometry import simulation

    # The fdtd-simulation guide's oracle run: a rigid 1.0 x 0.7 m box excited
    # by a Gaussian pulse; the probe spectrum peaks at the analytic rigid-room
    # mode frequencies f = (c/2) sqrt((nx/Lx)^2 + (ny/Ly)^2).
    lx, ly, dx, c = 1.0, 0.7, 0.02, 343.0
    nx, ny = round(lx / dx), round(ly / dx)
    res = simulation.fdtd_simulation(
        c, dx, 0.35, shape=(ny, nx),
        sources=[simulation.GaussianPulse(ix=7, iy=5, width=2.0e-4)],
        probes=[(nx - 4, ny - 3)],
    )
    p = res.pressures[0]
    spec = np.abs(np.fft.rfft(p * np.hanning(p.size), n=8 * p.size))
    freqs = np.fft.rfftfreq(8 * p.size, res.dt)
    sel = (freqs >= 100.0) & (freqs <= 450.0)
    level = 20.0 * np.log10(spec[sel] / np.max(spec[sel]))

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freqs[sel], level, color=COLOR_PRIMARY, linewidth=1.6,
            label="Probe pressure spectrum", zorder=3)
    modes = [(1, 0), (0, 1), (1, 1), (2, 0), (2, 1)]
    for i, (mx, my) in enumerate(modes):
        f_mode = 0.5 * c * float(np.hypot(mx / lx, my / ly))
        ax.axvline(f_mode, color=COLOR_SECONDARY, linestyle=":", linewidth=1.2,
                   zorder=2,
                   label="Analytic mode frequencies" if i == 0 else None)
        ax.annotate(f"({mx},{my})", xy=(f_mode, 1.5), ha="center", fontsize=9,
                    color=COLOR_FG)
    ax.set_title("Rigid-box FDTD probe spectrum vs analytic modes",
                 fontweight="bold", pad=14)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Probe spectrum [dB re max]")
    ax.set_xlim(100.0, 450.0)
    ax.set_ylim(-60.0, 6.0)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "fdtd_room_modes.svg")
