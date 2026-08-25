#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the simulation guides: FDTD fields and numerical propagation.

The still figures of the numerically simulated wave field -- the 2D FDTD
snapshot, the room modes it resolves, the elastic half-space and Scholte
interface waves, and the meshed metadiffuser far field the analytic model is
checked against. The clips these belong with live in :mod:`figures.fields`.
Everything here is embedded by a page under ``simulation/``.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .fields import _meshed_metadiffuser_ntff_levels
from .i18n import _LANG, _fmt_minus
from .materials import _qr_metadiffuser_wells
from .theme import (
    _FILENAME_SUFFIX,
    CMAP_FIELD,
    COLOR_FG,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_PANEL,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    FIELD_INK,
    LABEL_FREQ_HZ,
    save_figure,
)

if TYPE_CHECKING:
    from phonometry.simulation import ElasticFDTDResult


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
    from phonometry import materials

    wells, depth, period = _qr_metadiffuser_wells()
    model = materials.metadiffuser_polar_response(
        2000.0,
        wells,
        depth=depth,
        period=period,
        periods=1,
    )
    angles, levels = _meshed_metadiffuser_ntff_levels()
    _fig, ax = plt.subplots(
        figsize=(10, 6.2),
        subplot_kw={"projection": "polar"},
    )
    model.plot(
        ax=ax,
        color=COLOR_PRIMARY,
        marker="",
        linewidth=1.6,
        linestyle="--",
        label="TMM + Fraunhofer model",
        language=_LANG,
    )
    ax.plot(
        np.radians(angles),
        levels,
        color=COLOR_SECONDARY,
        linewidth=2.2,
        label="FDTD + NTFF, panel meshed at 0.5 mm",
    )
    ax.set_ylim(-40.0, 2.0)
    # The default theta formatter writes its negative angles with an ASCII
    # hyphen; restate the same grid with the typographic minus.
    polar: Any = ax
    polar.set_thetagrids(
        np.arange(-90, 91, 30), [f"{_fmt_minus(a, '.0f')}°" for a in range(-90, 91, 30)]
    )
    ax.set_title(
        "Far field of the meshed metadiffuser vs the model (2 kHz)",
        pad=18,
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
        343.0,
        0.05,
        shape=(60, 90),
        sponge_width=8,
        sponge_sides=("left", "right"),
        edge_impedance={"top": 413.0},
        obstacle_mask=mask,
    )
    sim.add_source(GaussianPulse(10, 30, width=1e-3))
    _fig, ax = plt.subplots(figsize=(10, 6.6))
    sim.plot_geometry(ax=ax, probes=[(3.0, 1.5), (4.0, 2.0)], language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "fdtd_domain_geometry.svg")
    plt.close()


def generate_fdtd_simulation(output_dir: str) -> None:
    """2D FDTD simulation: diffraction snapshot and the probe histories."""
    print("Generating fdtd_simulation.png...")
    from phonometry import simulation

    # A 3.0 x 2.0 m free field (absorbing edges) with a thin rigid barrier:
    # probe A sees the direct pulse plus the barrier reflection, probe B sits
    # in the shadow and only receives the wave diffracted around the edge.
    dx = 0.01
    mask = np.zeros((200, 300), dtype=bool)
    mask[60:, 150:154] = True
    res = simulation.fdtd_simulation(
        343.0,
        dx,
        9.0e-3,
        shape=(200, 300),
        sources=[simulation.GaussianPulse(ix=60, iy=100, width=3.0e-4)],
        probes=[(100, 100), (240, 100)],
        obstacle_mask=mask,
        boundaries="absorbing",
        absorbing_layer_cells=30,
        snapshot_every=75,
    )

    _fig, (ax_f, ax_p) = plt.subplots(
        1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )
    res.plot(kind="snapshot", frame=7, ax=ax_f)
    res.plot(ax=ax_p)
    ax_p.set_title("FDTD probe pressure", pad=10)
    ax_f.set_title(ax_f.get_title(), pad=10)

    plt.tight_layout()
    save_figure(output_dir, "fdtd_simulation.png")
    plt.close()


def generate_elastic_halfspace_waves(output_dir: str) -> None:
    """Elastic FDTD: P, S and Rayleigh waves in an aluminium half-space."""
    print("Generating elastic_halfspace_waves...")
    from phonometry import simulation

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
    res = simulation.elastic_fdtd_simulation(
        c_p,
        c_s,
        dx,
        duration,
        rho=rho,
        shape=(300, 600),
        cfl=cfl,
        sources=[
            simulation.ForceSource(
                ix=300,
                iy=0,
                direction="y",
                amplitude=1e6,
                waveform=simulation.GaussianPulse(0, 0, width=width).value,
            )
        ],
        boundaries=simulation.ElasticBoundaries({"top": "free"}),
        recording=simulation.ElasticRecording(
            snapshot_every=steps, snapshot_field="vy"
        ),
    )

    _fig, ax = plt.subplots(figsize=(9.5, 5.4))
    assert res.snapshots is not None
    assert res.snapshot_times is not None
    vmax = 0.18 * float(np.abs(res.snapshots[-1]).max())
    res.plot(kind="snapshot", frame=-1, ax=ax, vmin=-vmax, vmax=vmax)

    # Wavefront radii from the pulse centre time (t0 = 4 * width).
    t_eff = float(res.snapshot_times[-1]) - 4.0 * width
    x0 = (300 + 0.5) * dx
    theta = np.linspace(0.0, np.pi, 181)
    for radius in (c_p * t_eff, c_s * t_eff):
        ax.plot(
            x0 + radius * np.cos(theta),
            radius * np.sin(theta),
            linestyle=":",
            linewidth=1.0,
            color=COLOR_FG,
            alpha=0.65,
        )
    ann: dict[str, Any] = {
        "fontsize": 9,
        "color": COLOR_FG,
        "arrowprops": {"arrowstyle": "->", "color": COLOR_FG, "lw": 0.9},
    }
    r_p, r_s = c_p * t_eff, c_s * t_eff
    ax.annotate(
        "P wave front",
        xy=(x0 + r_p * np.cos(np.radians(55)), r_p * np.sin(np.radians(55))),
        xytext=(0.585, 0.15),
        ha="right",
        **ann,
    )
    ax.annotate(
        "S wave front",
        xy=(x0 + r_s * np.cos(np.radians(35)), r_s * np.sin(np.radians(35))),
        xytext=(0.525, 0.038),
        ha="right",
        **ann,
    )
    ax.annotate(
        "Rayleigh wave", xy=(x0 - 0.92 * r_s, 0.004), xytext=(0.045, 0.06), **ann
    )
    ax.set_title("Rayleigh wave along a free aluminium surface (elastic FDTD)", pad=10)

    plt.tight_layout()
    save_figure(output_dir, "elastic_halfspace_waves.png")
    plt.close()


@cache
def _scholte_interface_result() -> ElasticFDTDResult:
    """Water over a soft seabed, explosive shot near the contact (cached).

    Language/theme independent: the van Vossen (2002) benchmark media
    (water 1500/1000 over a 3500/2000/2500 solid), a 50 Hz Ricker
    explosion 10 m above the interface, run until the Scholte train has
    crawled ~330 m along the contact.
    """
    from phonometry import simulation

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

    return simulation.elastic_fdtd_simulation(
        c_p,
        c_s,
        dx,
        duration,
        rho=rho,
        sources=[
            simulation.ExplosionSource(ix=60, iy=89, waveform=ricker, amplitude=1e3)
        ],
        boundaries=simulation.ElasticBoundaries("absorbing", absorbing_layer_cells=20),
        recording=simulation.ElasticRecording(
            snapshot_every=steps, snapshot_field="vy"
        ),
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
        "fontsize": 9,
        "color": ink,
        "arrowprops": {"arrowstyle": "->", "color": ink, "lw": 0.9},
    }
    ax.annotate(
        "Scholte wave, evanescent on both sides",
        xy=(350.0, 108.0),
        xytext=(120.0, 170.0),
        **ann,
    )
    ax.annotate("direct water wave", xy=(300.0, 35.0), xytext=(120.0, 14.0), **ann)
    txt: dict[str, Any] = {"fontsize": 9, "color": ink, "alpha": 0.9}
    ax.text(12.0, 32.0, "water 1500 m/s", **txt)
    ax.text(12.0, 192.0, "seabed 3500 / 2000 m/s", **txt)
    ax.set_title("Scholte wave along a water-sediment interface (elastic FDTD)", pad=10)

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
        c,
        dx,
        0.35,
        shape=(ny, nx),
        sources=[simulation.GaussianPulse(ix=7, iy=5, width=2.0e-4)],
        probes=[(nx - 4, ny - 3)],
    )
    p = res.pressures[0]
    spec = np.abs(np.fft.rfft(p * np.hanning(p.size), n=8 * p.size))
    freqs = np.fft.rfftfreq(8 * p.size, res.dt)
    sel = (freqs >= 100.0) & (freqs <= 450.0)
    level = 20.0 * np.log10(spec[sel] / np.max(spec[sel]))

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        freqs[sel],
        level,
        color=COLOR_PRIMARY,
        linewidth=1.6,
        label="Probe pressure spectrum",
        zorder=3,
    )
    modes = [(1, 0), (0, 1), (1, 1), (2, 0), (2, 1)]
    for i, (mx, my) in enumerate(modes):
        f_mode = 0.5 * c * float(np.hypot(mx / lx, my / ly))
        ax.axvline(
            f_mode,
            color=COLOR_SECONDARY,
            linestyle=":",
            linewidth=1.2,
            zorder=2,
            label="Analytic mode frequencies" if i == 0 else None,
        )
        ax.annotate(
            f"({mx},{my})", xy=(f_mode, 1.5), ha="center", fontsize=9, color=COLOR_FG
        )
    ax.set_title("Rigid-box FDTD probe spectrum vs analytic modes", pad=14)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Probe spectrum [dB re max]")
    ax.set_xlim(100.0, 450.0)
    ax.set_ylim(-60.0, 6.0)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "fdtd_room_modes.svg")


def generate_fdtd_plane_wave_launch(output_dir: str) -> None:
    """A correctly configured one-way plane-wave launcher, measured.

    The FDTD guide's own 160 x 80 cell scene driven by a 1 kHz continuous
    wave on an injection line 22 cells in, with sponges above and below:
    the settled field, a transverse cut across the front, and the level of
    every row relative to the forward field. One concept: the two
    quantitative claims made for `PlaneWaveSource` -- a transversely plane
    front and a one-way launch -- are things you can measure on your own
    scene, and the second one is the sponge's doing.
    """
    print("Generating fdtd_plane_wave_launch...")
    from phonometry.simulation import FDTD2D, CWSource, PlaneWaveSource

    dx, offset, sponge = 0.01, 22, 20
    sim = FDTD2D(
        343.0, dx, shape=(160, 80), sponge_width=sponge, sponge_sides=("top", "bottom")
    )
    sim.add_source(
        PlaneWaveSource("down", CWSource(0, 0, frequency=1000.0).value, offset=offset)
    )
    sim.run(700)  # ramp, fill, then a few whole periods
    field = np.asarray(sim.p).copy()  # the instantaneous field of panel (a)
    # One more period, keeping the running peak: the envelope is what the
    # two cuts read, because a snapshot of a travelling wave is a phase.
    envelope = np.abs(sim.p)
    for _ in range(int(np.ceil(1.0e-3 / sim.dt))):
        sim.step()
        envelope = np.maximum(envelope, np.abs(sim.p))

    ny, nx = field.shape
    amp = float(np.abs(field[60:140, :]).max())
    spread = float(np.abs(np.diff(envelope[60:140, :], axis=1)).max())
    row_peak = envelope.mean(axis=1)
    forward = float(row_peak[60:140].mean())
    level = 20.0 * np.log10(np.maximum(row_peak / forward, 1e-6))
    behind = float((field[:sponge, :] ** 2).sum() / (field**2).sum())

    fig, (ax_f, ax_t, ax_l) = plt.subplots(
        1, 3, figsize=(13.6, 5.4), gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]}
    )

    # (a) the settled field, with the sponges and the injection line marked
    ax_f.imshow(
        field,
        cmap=CMAP_FIELD,
        vmin=-1.05 * amp,
        vmax=1.05 * amp,
        extent=(0.0, nx * dx, ny * dx, 0.0),
        aspect="auto",
        interpolation="nearest",
    )
    for y_band in (0.0, (ny - sponge) * dx):
        ax_f.axhspan(
            y_band,
            y_band + sponge * dx,
            facecolor="none",
            edgecolor=FIELD_INK,
            hatch="///",
            linewidth=0.8,
            alpha=0.85,
        )
    ax_f.axhline(offset * dx, color=FIELD_INK, linewidth=1.8)
    ax_f.annotate(
        "injection line",
        xy=(0.05, offset * dx),
        xytext=(0.06, (offset + 12) * dx),
        fontsize=9,
        color=FIELD_INK,
    )
    ax_f.text(0.04, 0.5 * sponge * dx, "sponge", fontsize=9, color=FIELD_INK)
    ax_f.text(0.04, (ny - 0.5 * sponge) * dx, "sponge", fontsize=9, color=FIELD_INK)
    ax_f.set(xlabel="$x$ [m]", ylabel="$y$ [m]")
    ax_f.set_title("Settled field, 1 kHz CW", pad=10)

    # (b) the transverse cut: flat to the last bit of a float64
    ax_t.plot(
        np.arange(nx) * dx,
        envelope[80, :],
        color=COLOR_PRIMARY,
        linewidth=2.2,
        label="envelope over one period",
    )
    ax_t.plot(
        np.arange(nx) * dx,
        field[80, :],
        color=COLOR_MUTED,
        linewidth=1.6,
        linestyle="--",
        label="one snapshot",
    )
    ax_t.set(xlabel="$x$ [m]", ylabel="Pressure [Pa]", ylim=(-1.45 * amp, 1.75 * amp))
    ax_t.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax_t.legend(loc="lower center", fontsize=9)
    ax_t.set_title("Cut across the front (row 80)", pad=10)
    ax_t.annotate(
        "largest difference between neighbouring columns:\n"
        f"{spread:.1e} Pa — the row is bit-identical",
        xy=(0.03, 0.88),
        xycoords="axes fraction",
        fontsize=9,
        color=COLOR_FG,
        bbox={
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
            "boxstyle": "round,pad=0.4",
        },
    )

    # (c) the longitudinal profile: what is left behind the line
    ax_l.plot(level, np.arange(ny) * dx, color=COLOR_SECONDARY, linewidth=2.0)
    ax_l.axhline(offset * dx, color=COLOR_FG, linewidth=1.2, linestyle="--")
    ax_l.axhspan(0.0, sponge * dx, color=COLOR_MUTED, alpha=0.25)
    ax_l.axhspan((ny - sponge) * dx, ny * dx, color=COLOR_MUTED, alpha=0.25)
    ax_l.invert_yaxis()
    ax_l.set(
        xlabel="Row envelope [dB re the forward field]",
        ylabel="$y$ [m]",
        xlim=(-72.0, 8.0),
    )
    ax_l.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax_l.set_title("What is left behind the line", pad=10)
    ax_l.annotate(
        f"{_fmt_minus(10 * np.log10(behind), '.1f')} dB of the "
        "field energy sits\n"
        "in the 20 sponge rows behind the line",
        xy=(0.03, 0.90),
        xycoords="axes fraction",
        fontsize=9,
        color=COLOR_FG,
        bbox={
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
            "boxstyle": "round,pad=0.4",
        },
    )
    ax_l.annotate(
        "injection line", xy=(-70.0, offset * dx - 0.03), fontsize=9, color=COLOR_FG
    )

    fig.suptitle(
        "One-way plane-wave launch: flat front, absorbed back side",
    )
    plt.tight_layout()
    save_figure(output_dir, "fdtd_plane_wave_launch.webp")
    plt.close()


def _meshed_metadiffuser_mask(dx: float = 0.0005) -> tuple[Any, dict[str, int]]:
    """The Table-1 metadiffuser panel rasterised exactly as section 4 does.

    Returns the boolean obstacle mask of the near-to-far-field scene and
    the cell indices the drawing needs (sponge width, panel corner, face
    width), so the figure and the setup diagram cannot drift apart.
    """
    from .materials import _METADIFFUSER_T1_ROWS

    pitch = 0.07
    sponge, gap, marg, front = 60, 60, 20, 40
    face = round(5 * pitch / dx)
    lat = marg + gap + sponge
    r_face = sponge + gap + front
    slab = round(0.023 / dx)
    mask = np.zeros((r_face + slab + marg + gap + sponge, face + 2 * lat), dtype=bool)
    mask[r_face : r_face + slab, lat : lat + face] = True
    for n, (h, l_n, l_c, w_n, w_c) in enumerate(_METADIFFUSER_T1_ROWS):
        x_slit = (n + 0.12) * pitch
        c0s = lat + round(x_slit / dx)
        c1s = lat + round((x_slit + h * 1e-3) / dx)
        mask[r_face : r_face + round(0.02 / dx), c0s:c1s] = False
        for m in range(2):
            y_m = (m + 0.5) * 0.01
            x_neck = x_slit + h * 1e-3
            r0 = r_face + round((y_m - 0.5e-3 * w_n) / dx)
            r1 = r_face + round((y_m + 0.5e-3 * w_n) / dx)
            mask[r0:r1, c1s : lat + round((x_neck + l_n * 1e-3) / dx)] = False
            r0 = r_face + round((y_m - 0.5e-3 * w_c) / dx)
            r1 = r_face + round((y_m + 0.5e-3 * w_c) / dx)
            mask[
                r0:r1,
                lat + round((x_neck + l_n * 1e-3) / dx) : lat
                + round((x_neck + (l_n + l_c) * 1e-3) / dx),
            ] = False
    return mask, {
        "sponge": sponge,
        "lat": lat,
        "r_face": r_face,
        "slab": slab,
        "face": face,
    }


def generate_metadiffuser_meshed_panel(output_dir: str) -> None:
    """What the transfer matrix homogenises, against what the solver sees.

    Left, the idealised unit cell the TMM works on, drawn to scale by the
    library's own `plot_metadiffuser_panel_geometry`; right, the same panel
    as the boolean obstacle mask the FDTD run steps on, with a zoom on one
    70 mm cell showing the 3.2 mm neck as six cells. One concept: the model
    prices each cell as one locally reacting reflection coefficient, and
    the solver resolves the necks that couple neighbouring cells -- which
    is where the shifted nulls of the polar comparison come from.
    """
    print("Generating metadiffuser_meshed_panel...")
    from phonometry.materials import plot_metadiffuser_panel_geometry

    from .materials import _qr_metadiffuser_wells

    wells, depth, period = _qr_metadiffuser_wells()
    dx = 0.0005
    mask, ix = _meshed_metadiffuser_mask(dx)
    r0 = ix["r_face"] - 40
    r1 = ix["r_face"] + ix["slab"] + 20
    c0 = ix["lat"] - 20
    c1 = ix["lat"] + ix["face"] + 20
    crop = mask[r0:r1, c0:c1]

    fig = plt.figure(figsize=(12.6, 7.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.35, 1.0, 1.5], hspace=0.70)
    ax_m = fig.add_subplot(gs[0])
    ax_g = fig.add_subplot(gs[1])
    ax_z = fig.add_subplot(gs[2])

    plot_metadiffuser_panel_geometry(
        wells, ax=ax_m, depth=depth, period=period, language=_LANG
    )
    ax_m.set_title(
        "What the transfer matrix homogenises: five slit "
        "resonators, each one reflection coefficient",
        pad=32,
    )

    extent = (c0 * dx, c1 * dx, r1 * dx, r0 * dx)
    ax_g.imshow(
        crop,
        cmap="Greys",
        vmin=0.0,
        vmax=1.6,
        extent=extent,
        aspect="equal",
        interpolation="nearest",
    )
    ax_g.set(xlabel="$x$ [m]", ylabel="$y$ [m]")
    ax_g.set_title(
        "What the solver steps on: the same panel as a boolean "
        r"obstacle mask at $\Delta x$ = 0.5 mm",
        pad=10,
    )
    z0 = ix["lat"] + round(4 * period / dx)
    z1 = z0 + round(period / dx)
    ax_g.add_patch(
        Rectangle(
            (z0 * dx, r0 * dx),
            (z1 - z0) * dx,
            (r1 - r0) * dx,
            facecolor="none",
            edgecolor=COLOR_SECONDARY,
            linewidth=1.8,
        )
    )

    zoom = mask[r0:r1, z0:z1]
    ax_z.imshow(
        zoom,
        cmap="Greys",
        vmin=0.0,
        vmax=1.6,
        extent=(z0 * dx, z1 * dx, r1 * dx, r0 * dx),
        aspect="equal",
        interpolation="nearest",
    )
    for spine in ax_z.spines.values():
        spine.set_color(COLOR_SECONDARY)
        spine.set_linewidth(1.8)
    ax_z.set(xlabel="$x$ [m]", ylabel="$y$ [m]")
    ax_z.set_title(
        "The fifth cell magnified: the 3.2 mm neck is six cells "
        r"wide, and it is what sets $\Delta x$",
        pad=10,
    )
    neck_y = (ix["r_face"] + round(0.005 / dx)) * dx
    neck_x = (ix["lat"] + round((4.12 * period + 20.3e-3) / dx)) * dx
    ax_z.annotate(
        "3.2 mm neck = 6 cells",
        xy=(neck_x, neck_y),
        xytext=(neck_x + 0.012, neck_y - 0.014),
        fontsize=9,
        color=COLOR_SECONDARY,
        arrowprops={"arrowstyle": "->", "color": COLOR_SECONDARY, "lw": 1.0},
    )
    # The Greys mask panel stays white on both themes, so its in-axes
    # annotations keep a fixed dark ink rather than COLOR_FG (the scholte
    # figure's rule for a field that never darkens).
    ink = "#3a3a3a"
    ax_z.annotate(
        "20.3 mm slit",
        xy=((z0 + 40) * dx, (ix["r_face"] + 22) * dx),
        xytext=((z0 + 6) * dx, (r0 + 8) * dx),
        fontsize=9,
        color=ink,
        arrowprops={"arrowstyle": "->", "color": ink, "lw": 1.0},
    )
    ax_z.annotate(
        "3 mm rigid backing",
        xy=((z0 + 60) * dx, (ix["r_face"] + ix["slab"] - 3) * dx),
        xytext=((z0 + 10) * dx, (r1 - 6) * dx),
        fontsize=9,
        color=ink,
        arrowprops={"arrowstyle": "->", "color": ink, "lw": 1.0},
    )

    save_figure(output_dir, "metadiffuser_meshed_panel.webp", bbox_inches="tight")
    plt.close()


def generate_elastic_probe_traces(output_dir: str) -> None:
    """The water column's probe history: incident pulse, then the echo.

    The elastic guide's normal-incidence run drawn the way
    ``ElasticFDTDResult.plot()`` draws it by default -- a probe pressure
    history, not a field map -- with the two arrivals timed against the
    geometry and their ratio annotated against the closed-form
    (Z2 - Z1)/(Z2 + Z1). One concept: the number the snippet prints is a
    ratio of two events on one trace, and the trace is where you see
    whether the run was clean enough to quote it.
    """
    print("Generating elastic_probe_traces...")
    from phonometry.simulation import STEEL, WATER, ElasticFDTD2D

    dx, ny = 0.005, 2400
    iy_probe, iy_interface = 900, 1200
    sim = ElasticFDTD2D.from_regions(
        (ny, 3),
        dx,
        background=WATER,
        regions=[((slice(iy_interface, None), slice(None)), STEEL)],
    )
    y = (np.arange(ny) + 0.5) * dx
    y_src, width = 3.0, 0.15
    p0 = np.exp(-(((y - y_src) / width) ** 2))[:, np.newaxis]
    sim.txx[:] = -p0
    sim.tyy[:] = -p0
    y_face = np.arange(1, ny) * dx + 0.5 * WATER.c_p * sim.dt
    sim.vy += (np.exp(-(((y_face - y_src) / width) ** 2)) / (WATER.rho * WATER.c_p))[
        :, np.newaxis
    ]

    steps = round(3.4e-3 / sim.dt)
    trace = np.empty(steps)
    for i in range(steps):
        sim.step()
        trace[i] = sim.p[iy_probe, 1]
    t_ms = (np.arange(steps) + 1) * sim.dt * 1e3

    y_probe = (iy_probe + 0.5) * dx
    y_face_m = iy_interface * dx
    t_direct = (y_probe - y_src) / WATER.c_p * 1e3
    t_echo = ((y_face_m - y_src) + (y_face_m - y_probe)) / WATER.c_p * 1e3
    incident = float(trace[t_ms < 1.8].max())
    late = np.where(t_ms > 2.6, trace, 0.0)
    j = int(np.abs(late).argmax())
    ratio = float(trace[j] / incident)
    z1, z2 = WATER.rho * WATER.c_p, STEEL.rho * STEEL.c_p
    exact = (z2 - z1) / (z2 + z1)

    _fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(
        t_ms,
        trace,
        color=COLOR_PRIMARY,
        linewidth=1.6,
        label="probe pressure, 7.5 m below the source",
    )
    ax.axhline(0.0, color=COLOR_GRID, linewidth=1.0)
    for t_mark, lab, col in (
        (t_direct, "incident", COLOR_FG),
        (t_echo, "echo off the steel", COLOR_SECONDARY),
    ):
        ax.axvline(t_mark, color=col, linestyle=":", linewidth=1.2)
        ax.annotate(
            f"{lab}\n{t_mark:.2f} ms",
            xy=(t_mark, -0.42),
            xytext=(t_mark + 0.08, -0.72),
            fontsize=9,
            color=col,
            arrowprops={"arrowstyle": "->", "color": col, "lw": 0.9},
        )
    ax.annotate(
        f"echo / incident = {ratio:.3f}\n$(Z_2 - Z_1)/(Z_2 + Z_1)$ = {exact:.3f}",
        xy=(0.02, 0.04),
        xycoords="axes fraction",
        fontsize=10,
        color=COLOR_FG,
        bbox={
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
            "boxstyle": "round,pad=0.4",
        },
    )
    ax.set_title(
        "Water over steel at normal incidence: the probe history res.plot() draws",
        pad=12,
    )
    ax.set(
        xlabel="Time [ms]",
        ylabel="Pressure [Pa]",
        xlim=(0.0, float(t_ms[-1])),
        ylim=(-1.15, 1.35),
    )
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "elastic_probe_traces.svg")
    plt.close()
