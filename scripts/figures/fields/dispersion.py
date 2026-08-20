#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Numerical dispersion: the same pulse on three grids, arriving late.

The FDTD guide states the discrete dispersion relation, its leading-order
error and the ten-cells-per-wavelength rule, and the only figure that
supports them is a mode spectrum whose own caption disowns the evidence
("the 0.15 Hz shift is at the edge of what the record can resolve").
This clip measures the error instead: one tone burst, three tubes that
differ only in ``dx``, and the exact continuous answer drawn behind each of
them so the lag is a gap on screen rather than a number in a table.
"""

from functools import lru_cache
from typing import Any

import numpy as np

from ..media import _ANIM_PILL_BOX, _anim_figure, _render_clip, _translate_str
from ..theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
)

_DISP_C = 343.0
_DISP_F0 = 500.0  # tone-burst carrier
_DISP_LAMBDA = _DISP_C / _DISP_F0  # 0.686 m
#: Cells per carrier wavelength of the three tubes: the guide's rule, one
#: grid coarser and one finer, so the second-order law is three points.
_DISP_CELLS = (5, 10, 20)
_DISP_X0 = 1.1  # launch centre [m]
_DISP_W = 0.50  # Gaussian 1/e half-width [m]
_DISP_FINISH = 6.6  # finish line the delays are read at
_DISP_LSHOW = 10.0  # visible tube [m]
_DISP_LOSSY = 3.0 * _DISP_LAMBDA  # graded absorbing tail beyond it
#: Peak decay rate of the quadratic loss ramp closing the tube: about 60 dB
#: one way over the ramp, graded so nothing measurable comes back.
_DISP_SIGMA = 3450.0
# Timeline. Flight: the launch centre at 1.1 m to the finish line at 6.6 m
# is 5.50 m, covered at the slowest speed anywhere on the path -- which is
# not c but the coarse tube's own group speed, 0.829 c = 284.2 m/s -- in
# 19.35 ms; x 1.2 = 23.2 ms captured, so every tube is seen well past its
# own arrival. The visible tube is 10 m rather than 6.6 so that the
# *fastest* wave on screen, the exact continuous one, is still inside the
# frame when the window closes (it reaches 9.06 m): sizing the tube on the
# slowest wave alone would end the clip with two of the three panels empty.
# Sampling: 40 us per frame is 50 frames per period of the 500 Hz carrier,
# above the 48-frame floor, and 576 frames cover the window; played at
# 48 fps the active part runs 12.0 s and one carrier cycle takes 42 ms of
# screen time.
_DISP_CAPTURE = 40.0e-6
_DISP_FPS = 48.0
_DISP_HOLD = round(2.0 * _DISP_FPS)
#: The lag readout divides by t, so it is revealed once the packet has
#: moved far enough for the ratio to mean something (1.5 ms, half a metre).
_DISP_REVEAL = 1.5e-3


def _disp_group_speed(cells: Any, courant: float) -> Any:
    """Exact discrete group speed / c on the grid axis.

    Differentiating ``sin(omega dt / 2) = S sin(k dx / 2)`` at fixed grid
    gives ``v_g / c = cos(theta) / sqrt(1 - S^2 sin^2 theta)`` with
    ``theta = k dx / 2 = pi / cells``, so both the three tubes and the law
    panel read the same callable in cells per wavelength.
    """
    theta = np.pi / np.asarray(cells, dtype=float)
    arg = np.clip(courant * np.sin(theta), -1.0, 1.0)
    return np.cos(theta) / np.sqrt(np.clip(1.0 - arg**2, 1e-12, None))


def _disp_phase_speed(cells: Any, courant: float) -> Any:
    """Exact discrete phase speed / c on the grid axis, in cells per
    wavelength: ``arcsin(S sin theta) / (S theta)``. This is the error the
    guide's ``(1 - S^2)(k dx)^2 / 24`` approximates.
    """
    theta = np.pi / np.asarray(cells, dtype=float)
    arg = np.clip(courant * np.sin(theta), -1.0, 1.0)
    return np.arcsin(arg) / theta / courant


def _disp_packet_group_speed(profile: Any, dx: float, courant: float) -> float:
    """Group speed of a whole packet / c, weighted by its own spectrum.

    Each wavenumber carries its energy at its own ``v_g``, so the energy
    centroid of a dispersing packet travels at the energy-weighted mean of
    ``v_g`` over the packet's spectrum -- not at ``v_g`` of the carrier,
    which a 5-cell grid is far enough into the curved part of the law for
    the difference to be measurable.
    """
    energy = np.abs(np.fft.rfft(profile)) ** 2
    spatial = np.fft.rfftfreq(profile.size, dx)
    cells = np.divide(
        1.0, spatial * dx, out=np.full(spatial.shape, np.inf), where=spatial > 0.0
    )
    return float(np.average(_disp_group_speed(cells, courant), weights=energy))


@lru_cache(maxsize=1)
def _dispersion_fields() -> tuple[Any, ...]:
    """Three plane tubes at 5, 10 and 20 cells per wavelength, cached.

    Every tube carries the same air, the same tone burst and the same
    library default Courant number; only ``dx`` differs, so what separates
    the three traces on screen is grid resolution and nothing else. Each
    runs at its own time step -- that is what a fixed ``cfl`` on a
    different ``dx`` means -- and its frames are resampled onto one common
    40 us timeline, so the panels advance together instead of the coarse
    one stuttering at a quarter of the frame rate.

    Returns, per tube: the frames over the visible tube, the cell centres,
    the running energy centroid, the measured and predicted finish-line
    crossings and the measured and predicted group speeds; plus the shared
    frame times and the per-axis Courant number.
    """
    import fdtd2d

    t_end = (
        1.2
        * (_DISP_FINISH - _DISP_X0)
        / (float(_disp_group_speed(_DISP_CELLS[0], 0.6 / np.sqrt(2.0))) * _DISP_C)
    )
    times = np.arange(0.0, t_end, _DISP_CAPTURE)
    tubes: list[dict[str, Any]] = []
    courant = 0.0
    for cells in _DISP_CELLS:
        dx = _DISP_LAMBDA / cells
        n_vis = round(_DISP_LSHOW / dx)
        n_x = n_vis + round(_DISP_LOSSY / dx)
        x = (np.arange(n_x) + 0.5) * dx
        ramp = np.clip((x - n_vis * dx) / _DISP_LOSSY, 0.0, 1.0) ** 2
        sim = fdtd2d.FDTD2D(
            _DISP_C,
            dx,
            shape=(3, n_x),
            damping=np.tile(_DISP_SIGMA * ramp, (3, 1)),
            edge_impedance={"left": 1.2 * _DISP_C, "right": 1.2 * _DISP_C},
        )
        sim.add_plane_wave(
            "right", center=_DISP_X0, width=_DISP_W, wavelength=_DISP_LAMBDA
        )
        courant = _DISP_C * sim.dt / dx
        v_pred = _disp_packet_group_speed(sim.p[1, :n_vis].copy(), dx, courant)
        raw_t: list[float] = []
        raw_p: list[Any] = []
        while sim.time <= times[-1] + sim.dt:
            raw_t.append(sim.time)
            raw_p.append(sim.p[1, :n_vis].copy())
            sim.step()
        raw = np.asarray(raw_p)
        t_raw = np.asarray(raw_t)
        # One common timeline for all three panels: each tube keeps its own
        # dt and is resampled onto the 40 us capture grid, which for the
        # coarse tube interpolates between steps 170 us apart -- 31 degrees
        # of carrier phase, so the resampling costs at most 3.5 % of the
        # local amplitude and buys a panel that moves instead of stepping.
        frames = np.empty((times.size, n_vis), dtype=np.float32)
        for j in range(n_vis):
            frames[:, j] = np.interp(times, t_raw, raw[:, j])
        # Nothing leaves the visible tube inside the window, so the energy
        # centroid of the whole visible field is an unbiased measure of
        # where the packet is, at every frame.
        x_vis = x[:n_vis]
        energy = frames.astype(np.float64) ** 2
        centroid = (energy * x_vis).sum(axis=1) / energy.sum(axis=1)
        t_cross = float(np.interp(_DISP_FINISH, centroid, times))
        distance = _DISP_FINISH - _DISP_X0
        tubes.append(
            {
                "cells": cells,
                "dx": dx,
                "x": x_vis,
                "p": frames,
                "centroid": centroid,
                "t_cross": t_cross,
                "t_pred": distance / (v_pred * _DISP_C),
                "v_meas": distance / t_cross / _DISP_C,
                "v_pred": v_pred,
            }
        )
    return tuple(tubes), times, courant


def animate_fdtd_dispersion(output_dir: str) -> None:
    """Numerical dispersion, measured (2D FDTD): one 500 Hz tone burst
    launched down three plane tubes that differ only in grid spacing, at 5,
    10 and 20 cells per carrier wavelength. The exact continuous answer
    travels behind each trace in grey, so the numerical packet visibly
    falls behind it and grows a ripple tail, and the running readout gives
    the gap in metres against the closed form. The coarse tube reaches the
    finish line milliseconds late; the fine one arrives on time. The side
    panel carries the speed-error law the three measurements are read
    against, phase and group.
    """
    T = _translate_str
    tubes, times, courant = _dispersion_fields()
    colors = (COLOR_SECONDARY, COLOR_PRIMARY, COLOR_TERTIARY)

    fig = _anim_figure()
    fig.suptitle(
        T("Numerical dispersion: one pulse, three grids (2D FDTD)"),
    )
    gs = fig.add_gridspec(3, 2, width_ratios=(3.55, 1.0), hspace=0.12, wspace=0.05)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax_law = fig.add_subplot(gs[:, 1])

    lines: list[Any] = []
    exacts: list[Any] = []
    pills: list[Any] = []
    for i, (ax, tube, color) in enumerate(zip(axes, tubes, colors, strict=True)):
        ax.grid(False)
        ax.set_xlim(0.0, _DISP_LSHOW)
        # Headroom above the trace, not around it: the packet never leaves
        # +-1, and the label and the readout need a band of their own or
        # the wave runs through them at exactly the moment they matter.
        ax.set_ylim(-1.2, 2.15)
        ax.set_yticks([])
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        if i < 2:
            ax.spines["bottom"].set_color(COLOR_GRID)
        ax.axhline(0.0, color=COLOR_GRID, lw=0.8, zorder=1)
        ax.axvline(_DISP_FINISH, color=COLOR_MUTED, lw=1.0, ls=":", zorder=2)
        if i == 0:
            ax.text(
                _DISP_FINISH,
                1.03,
                T(f"finish line, {_DISP_FINISH:.1f} m"),
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=COLOR_MUTED,
                transform=ax.get_xaxis_transform(),
            )
        exacts.append(
            ax.fill_between(
                tube["x"], 0.0, 0.0, color=COLOR_MUTED, alpha=0.38, lw=0.0, zorder=2
            )
        )
        lines.append(
            ax.plot(
                tube["x"],
                np.zeros_like(tube["x"]),
                color=color,
                lw=1.4,
                marker="o",
                ms=(3.2, 2.0, 1.1)[i],
                zorder=4,
            )[0]
        )
        ax.text(
            0.006,
            0.985,
            T(
                rf"{tube['cells']} cells per wavelength · "
                rf"$\Delta x$ = {tube['dx'] * 1000:.1f} mm"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            fontweight="bold",
            color=color,
        )
        # The readout sits at the foot of its panel: the trace occupies the
        # middle band, and the Spanish text runs long enough that a
        # top-right pill collides with the panel label.
        pills.append(
            ax.text(
                0.994,
                0.985,
                "",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color="white",
                zorder=7,
                linespacing=1.35,
                bbox={
                    "boxstyle": _ANIM_PILL_BOX,
                    "facecolor": "black",
                    "alpha": 0.55,
                    "edgecolor": "none",
                },
            )
        )
        ax.tick_params(labelsize=7.5, labelbottom=i == 2)
    axes[2].set_xlabel(T("Position along the tube [m]"), fontsize=9)

    grid = np.geomspace(3.0, 40.0, 240)
    ax_law.plot(
        grid,
        100.0 * (1.0 - _disp_group_speed(grid, courant)),
        color=COLOR_FG,
        lw=1.7,
        label=T("group (a pulse)"),
    )
    ax_law.plot(
        grid,
        100.0 * (1.0 - _disp_phase_speed(grid, courant)),
        color=COLOR_FG,
        lw=1.2,
        ls="--",
        label=T("phase (the rule)"),
    )
    for tube, color in zip(tubes, colors, strict=True):
        ax_law.plot(
            [tube["cells"]],
            [100.0 * (1.0 - _disp_group_speed(tube["cells"], courant))],
            marker="o",
            ms=6.0,
            color=color,
            zorder=5,
        )
    ax_law.axvline(10.0, color=COLOR_MUTED, lw=0.9, ls=":")
    ax_law.set(xscale="log", yscale="log", xlim=(3.0, 40.0), ylim=(0.1, 60.0))
    ax_law.set_xlabel(T("Cells per wavelength"), fontsize=8)
    ax_law.set_ylabel(T("Speed error [%]"), fontsize=8)
    # Explicit ticks on both log axes: the default locators put minor
    # labels ("3 x 10^0") on top of the four that matter.
    ax_law.set_xticks([5, 10, 20, 40])
    ax_law.set_xticklabels(["5", "10", "20", "40"])
    ax_law.set_xticks([], minor=True)
    ax_law.set_yticks([0.1, 1.0, 10.0])
    ax_law.set_yticklabels(["0.1", "1", "10"])
    ax_law.set_yticks([], minor=True)
    ax_law.tick_params(labelsize=7)
    ax_law.legend(fontsize=6.5, loc="lower left", framealpha=0.85)
    ax_law.set_title(T("The closed form"), fontsize=8.5)

    t_txt = fig.text(
        0.012,
        0.935,
        "",
        ha="left",
        va="top",
        family="monospace",
        fontsize=10,
        color=COLOR_FG,
    )
    fig.text(
        0.5,
        0.011,
        T(
            f"Air at {_DISP_C:.0f} m/s, a {_DISP_F0:.0f} Hz burst, "
            f"per-axis Courant number $S$ = {courant:.3f}; grey is the "
            f"exact continuous wave, dots are the grid cells"
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color=COLOR_FG,
        alpha=0.85,
    )
    fig.get_layout_engine().set(rect=(0.0, 0.045, 1.0, 0.94))

    def update(k: int) -> tuple[Any, ...]:
        j = min(k, times.size - 1)
        t = float(times[j])
        artists: list[Any] = [t_txt]
        for i, (tube, ax) in enumerate(zip(tubes, axes, strict=True)):
            lines[i].set_ydata(tube["p"][j])
            x = tube["x"]
            shift = x - (_DISP_X0 + _DISP_C * t)
            exact = np.exp(-((shift / _DISP_W) ** 2)) * np.sin(
                2.0 * np.pi * shift / _DISP_LAMBDA
            )
            exacts[i].remove()
            exacts[i] = ax.fill_between(
                x, 0.0, exact, color=COLOR_MUTED, alpha=0.38, lw=0.0, zorder=2
            )
            if t >= tube["t_cross"]:
                late = (tube["t_cross"] - (_DISP_FINISH - _DISP_X0) / _DISP_C) * 1e3
                pills[i].set_text(
                    T(f"crossed at {tube['t_cross'] * 1e3:.1f} ms, +{late:.1f} ms")
                    + "\n"
                    + T(
                        f"{100 * (1 - tube['v_meas']):.1f} % slow "
                        f"(theory {100 * (1 - tube['v_pred']):.1f} %)"
                    )
                )
            elif t >= _DISP_REVEAL:
                lag = _DISP_X0 + _DISP_C * t - tube["centroid"][j]
                pills[i].set_text(
                    T(
                        f"{lag:.2f} m behind (theory "
                        f"{(1 - tube['v_pred']) * _DISP_C * t:.2f} m)"
                    )
                )
            artists += [lines[i], exacts[i], pills[i]]
        t_txt.set_text(T(f"$t$ = {t * 1e3:5.2f} ms"))
        return tuple(artists)

    _render_clip(
        fig,
        update,
        output_dir,
        "anim_fdtd_dispersion",
        frames=times.size + _DISP_HOLD,
        fps=_DISP_FPS,
        gif_fps=10,
    )
