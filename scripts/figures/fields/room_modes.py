#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The room modes of a rectangular room, driven on and off resonance."""

from functools import lru_cache
from typing import Any

import numpy as np

from ..media import _ANIM_FPS, _anim_figure, _render_clip, _translate_str
from ..theme import (
    CMAP_FIELD,
    COLOR_FG,
    COLOR_TERTIARY,
    FIELD_INK,
    FIELD_STROKE,
)

# Room-modes timeline. Mesh: the highest carrier is the off-mode drive at
# 91.2 Hz (lambda = 3.76 m, lambda / 8 = 0.47 m) and the smallest geometric
# dimension is the 3.5 m room side (3.5 / 4 = 0.88 m), so the rule
# dx = min(0.88, 0.47) m allows 0.47 m; dx = 0.01 m sits 47 times finer,
# because the mode map has to be drawn, not just resolved.
# Flight: the source at (0.25, 0.25) m reaches the deepest reflector -- the
# opposite corner (5, 3.5) -- after 5.755 m and the farthest visible point
# (the near corner) 6.103 m later, so 11.859 m / 343 m/s x 1.2 = 41.5 ms.
# Sampling: 18 solver steps per frame = 222.6 us gives 49.3 frames per
# 91.2 Hz period (>= 48). The captured window keeps the committed clip's
# full 350 ms -- the flight floor is only a floor, and this clip needs
# 8.4 times it (3.5 amplitude time constants of the T60 = 0.7 s room,
# tau = T60 / 6.9077 = 101 ms) because the resonance has to be seen
# *growing* all the way into its settled mode map. 1 572 frames cover that
# window, played at 13/3 of the shared 20 fps -- exactly the factor by
# which the stride shrank from the committed clip's 78 solver steps per
# frame -- so the pacing stays the 19.30 ms of simulation per second of
# playback the committed clip had, and the clip runs 18.1 s (it ran 18 s).
_ROOM_EVERY = 18
_ROOM_FRAMES = 1572
#: 13/3 of the shared rate, matching the stride cut (see the note above).
_ROOM_FPS = _ANIM_FPS * 13.0 / 3.0


@lru_cache(maxsize=1)
def _room_mode_fields(
        n_frames: int = _ROOM_FRAMES) -> tuple[Any, Any, Any, float, float]:
    """Run the two FDTD room simulations once per process (all variants).

    Returns instantaneous-pressure frames, running-RMS frames (both float32,
    stacked ``(2, n_frames, ny, nx)``), the frame times, and the on-mode and
    off-mode drive frequencies. ``n_frames`` sets the captured window at the
    fixed ``_ROOM_EVERY`` capture stride (see the timeline note above).
    """
    import fdtd2d

    lx, ly, c0 = 5.0, 3.5, 343.0
    dx = 0.01                                    # 500 x 350 cells
    ny, nx = round(ly / dx), round(lx / dx)
    f_mode = 0.5 * c0 * float(np.hypot(2 / lx, 1 / ly))   # (2,1) ~ 84.3 Hz
    f_next = 0.5 * c0 * (2 / ly)                          # (0,2) = 98 Hz
    f_off = 0.5 * (f_mode + f_next)              # between resonances
    t60 = 0.7
    every = _ROOM_EVERY
    steps = every * n_frames
    p_all, r_all = [], []
    times = np.zeros(0)
    for f in (f_mode, f_off):
        sim = fdtd2d.FDTD2D(c0, dx, shape=(ny, nx), damping=6.9077 / t60)
        sim.add_source(fdtd2d.CWSource(ix=25, iy=25, frequency=f,
                                       ramp_cycles=2.0))
        # Running mean square with a two-period time constant: the pattern
        # (the mode map) builds up as the resonance settles.
        beta = float(np.exp(-sim.dt * f / 2.0))
        ms = np.zeros_like(sim.p)
        ps: list[Any] = []
        rs: list[Any] = []
        ts: list[float] = []
        for i in range(steps):
            sim.step()
            ms = beta * ms + (1.0 - beta) * sim.p**2
            if (i + 1) % every == 0 and len(ps) < n_frames:
                ps.append(sim.p[::2, ::2].astype(np.float32))
                rs.append(np.sqrt(ms[::2, ::2]).astype(np.float32))
                ts.append(sim.time)
        p_all.append(np.stack(ps))
        r_all.append(np.stack(rs))
        times = np.asarray(ts)
    return np.stack(p_all), np.stack(r_all), times, f_mode, f_off


def animate_fdtd_room_modes(output_dir: str) -> None:
    """On-mode vs off-mode CW drive in a rigid 5 x 3.5 m room (2D FDTD):
    on resonance the (2,1) standing-wave pattern grows until it dominates
    the field; off resonance the forced response stays weak and never
    organises into that nodal structure."""
    from matplotlib import patheffects

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    p_all, r_all, times, f_mode, f_off = _room_mode_fields()
    half = p_all.shape[1] // 2
    vmax_p = float(np.quantile(np.abs(p_all[0][half:]), 0.995))
    vmax_r = float(np.quantile(r_all[0][-1], 0.999))

    fig = _anim_figure()
    fig.suptitle(T("Room modes in a rigid 5 m × 3.5 m room (2D FDTD)"),
                 )
    gs = fig.add_gridspec(2, 2)
    titles = [T(f"On the (2,1) mode: {f_mode:.1f} Hz"),
              T(f"Off mode: {f_off:.1f} Hz")]
    ims: list[Any] = []
    for col in range(2):
        ax_p = fig.add_subplot(gs[0, col])
        ax_p.grid(False)
        im_p = ax_p.imshow(p_all[col][0], origin="lower",
                           extent=(0.0, 5.0, 0.0, 3.5), cmap=CMAP_FIELD,
                           vmin=-vmax_p, vmax=vmax_p, interpolation="bilinear")
        ax_p.set_title(titles[col], fontsize=10)
        ax_p.plot([0.25], [0.25], marker="o", ms=5, color=COLOR_TERTIARY,
                  markeredgecolor=FIELD_STROKE, markeredgewidth=0.8)
        ax_p.text(0.45, 0.22, T("source"), ha="left", va="center",
                  color=FIELD_INK, fontsize=7.5, path_effects=outline)
        ax_p.tick_params(labelsize=7, labelbottom=False)
        ax_r = fig.add_subplot(gs[1, col])
        ax_r.grid(False)
        im_r = ax_r.imshow(r_all[col][0], origin="lower",
                           extent=(0.0, 5.0, 0.0, 3.5), cmap="magma",
                           vmin=0.0, vmax=vmax_r, interpolation="bilinear")
        ax_r.tick_params(labelsize=7)
        ax_r.set_xlabel("$x$ [m]", fontsize=8)
        if col == 0:
            ax_p.set_ylabel(T("instantaneous $p(x, y)$"), fontsize=9)
            ax_r.set_ylabel(T("RMS pressure (mode map)"), fontsize=9)
            for xn in (1.25, 3.75):
                ax_r.axvline(xn, color="white", ls="--", lw=1.0, alpha=0.75)
            ax_r.axhline(1.75, color="white", ls="--", lw=1.0, alpha=0.75)
            ax_r.text(0.12, 3.3, T("nodal lines (2,1)"), color="white",
                      fontsize=8, ha="left", va="top")
        else:
            ax_p.tick_params(labelleft=False)
            ax_r.tick_params(labelleft=False)
            ax_r.text(0.12, 3.3, T("same color scale"), color="white",
                      fontsize=8, ha="left", va="top")
        ims += [im_p, im_r]
    t_txt = fig.text(0.985, 0.955, "", ha="right", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)

    def update(k: int) -> tuple[Any, ...]:
        for col in range(2):
            ims[2 * col].set_data(p_all[col][k])
            ims[2 * col + 1].set_data(r_all[col][k])
        t_txt.set_text(T(f"$t$ = {times[k] * 1000.0:3.0f} ms"))
        return (*ims, t_txt)

    # Own frame budget and rate (see the _ROOM_FRAMES timeline note):
    # 1 572 frames at 49.3 per acoustic period, played at 260/3 fps. The
    # GitHub GIF samples this long clip at a reduced rate so the
    # palette-quantized file stays well under 4 MB.
    _render_clip(fig, update, output_dir, "anim_fdtd_room_modes",
                 frames=int(p_all.shape[1]), fps=_ROOM_FPS, gif_fps=5)
