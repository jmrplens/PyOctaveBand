#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The expansion chamber: a reactive muffler passing and stopping."""

from functools import lru_cache
from typing import Any

import numpy as np

from ..media import (
    _ANIM_PILL_BOX,
    _FDTD_ANIM_FRAMES,
    _anim_figure,
    _render_clip,
    _translate_str,
)
from ..theme import CMAP_FIELD, COLOR_FG, COLOR_GRID, COLOR_PRIMARY
from ._core import _anim_speaker

_CHAMBER_L = 0.30                        # chamber length of the TL example
_CHAMBER_M = 4.0                         # expansion area ratio
_CHAMBER_PIPE_A = 0.01                   # pipe area of the TL example [m2]
# 2D pipe height. The four-pole TL depends only on m and L, but the 2D
# junction end correction grows with the bore; a 50 mm pipe (a small
# muffler with a 200 x 300 mm chamber) keeps the sudden-area-change
# phase error at kL = pi below ~0.15 dB so the simulated pass band
# actually matches the model's TL = 0 within the line width.
_CHAMBER_BORE = 0.05
# Mesh rule: dx = min(smallest scene dimension / 4, lambda/8 at the
# highest carrier) = min(50 mm / 4 = 12.5 mm, 0.6 m / 8 = 75 mm); the
# grid runs finer still (2.5 mm, 20 cells across the pipe bore) so the
# junction fields render smoothly at negligible cost.
_CHAMBER_DX = 0.0025


def _expansion_chamber_freqs() -> tuple[float, float]:
    """The two CW carriers: full transmission at ``kL = pi`` and the TL
    peak at ``kL = pi/2`` of the 0.30 m chamber (Bies Eq. (8.111))."""
    return 343.0 / (2.0 * _CHAMBER_L), 343.0 / (4.0 * _CHAMBER_L)


@lru_cache(maxsize=1)
def _expansion_chamber_fields(
    n_frames: int = _FDTD_ANIM_FRAMES,
) -> tuple[Any, Any, Any, Any]:
    """Two CW runs through the m = 4 expansion chamber, cached.

    A 1.4 m duct at the L and m of the noise-control guide's example
    (50 mm pipe, 0.30 m chamber, area ratio 4 as the 2D height ratio),
    walls built from a 10^6:1 density contrast so the sustained plane wave
    can be injected across the full left edge (an obstacle mask would
    reject the injection line). rho c edges at both ends make the source
    non-reflecting and the termination anechoic, so the outlet carries
    pure transmission. Returns the wall-masked frame stacks (pass band,
    stop band), the settled centreline |p| envelopes, the frame times and
    the library transmission losses at the two carriers.
    """
    import fdtd2d

    from phonometry import expansion_chamber

    dx = _CHAMBER_DX
    pipe_cells = round(_CHAMBER_BORE / dx)               # 20 cells
    ny = round(_CHAMBER_M * _CHAMBER_BORE / dx)          # 80 cells
    nx = round(1.4 / dx)
    c0 = round(0.55 / dx)
    c1 = c0 + round(_CHAMBER_L / dx)
    p0 = (ny - pipe_cells) // 2
    air = np.zeros((ny, nx), dtype=bool)
    air[p0:p0 + pipe_cells, :] = True
    air[:, c0:c1] = True
    rho_map = np.where(air, 1.2, 1.2e6)
    f_pass, f_peak = _expansion_chamber_freqs()
    tls = expansion_chamber(
        np.array([f_pass, f_peak]), _CHAMBER_L,
        _CHAMBER_M * _CHAMBER_PIPE_A, _CHAMBER_PIPE_A).transmission_loss
    # Clip duration per the deepest-reflector rule: d(source -> chamber
    # outlet plate, the deepest reflecting feature) = 0.85 m, plus 0.85 m
    # back to the farthest visible field point (the duct inlet at x = 0):
    # t = 1.2 * 1.7 / 343 = 5.95 ms -> every = 6 (6.68 ms captured, 94
    # frames per 572 Hz period >= 48). The window is shorter than the
    # standing wave's build-up, so both runs settle uncaptured for 30 ms
    # first and the clip shows the steady field with its exact envelope.
    every = 6
    runs = []
    times = np.zeros(0)
    for f in (f_pass, f_peak):
        sim = fdtd2d.FDTD2D(343.0, dx, shape=(ny, nx), rho=rho_map,
                            edge_impedance={"left": 1.2 * 343.0,
                                            "right": 1.2 * 343.0})
        tone = fdtd2d.CWSource(0, 0, frequency=f)
        sim.add_source(fdtd2d.PlaneWaveSource("right", tone.value,
                                              offset=2))
        mid = ny // 2
        period = round(1.0 / (f * sim.dt))
        settle = round(0.030 / sim.dt)
        env = np.zeros(sim.p[mid, ::2].shape, dtype=np.float32)
        for i in range(settle):
            sim.step()
            if i >= settle - period:
                np.maximum(env, np.abs(sim.p[mid, ::2]), out=env)
        ps: list[Any] = []
        ts: list[float] = []
        while len(ps) < n_frames:
            sim.step()
            if sim.n % every == 0:
                # The dense wall cells ring with the injection line;
                # blank them to NaN (transparent under imshow) so the
                # display and its color scale only see the air path.
                ps.append(np.where(air, sim.p, np.nan)[::2, ::2]
                          .astype(np.float32))
                ts.append(sim.time)
        runs.append((np.stack(ps), env))
        times = np.asarray(ts)
    return runs[0], runs[1], times, (float(tls[0]), float(tls[1]))


def animate_fdtd_expansion_chamber(output_dir: str) -> None:
    """The reactive expansion chamber (2D FDTD): the same silencer at its
    two characteristic frequencies. At kL = pi the chamber is a
    half-wavelength resonator and the wave crosses as if the chamber were
    not there (TL = 0); at kL = pi/2 the two area jumps reflect in phase
    (the pi round trip across the chamber returns both echoes to the
    inlet in phase) and the wave is sent back up the inlet, leaving the
    outlet at less than half amplitude (the 6.5 dB four-pole TL peak).
    No absorption anywhere: a purely reactive silencer."""
    T = _translate_str
    (p_pass, e_pass), (p_stop, e_stop), times, tls = (
        _expansion_chamber_fields())
    f_pass, f_peak = _expansion_chamber_freqs()
    dx = _CHAMBER_DX
    ny = round(_CHAMBER_M * _CHAMBER_BORE / dx)
    nx = round(1.4 / dx)
    length, height = nx * dx, ny * dx
    pipe_y = ((height - _CHAMBER_BORE) / 2.0,
              (height + _CHAMBER_BORE) / 2.0)
    env_base, env_h, env_max = 0.245, 0.105, 2.1
    x_env = (np.arange(p_pass.shape[2]) + 0.5) * 2.0 * dx
    env_from = 8                           # hide the injection-line step
    vmaxes = [float(np.nanquantile(np.abs(p[p.shape[0] // 2:]), 0.999))
              for p in (p_pass, p_stop)]

    fig = _anim_figure()
    fig.suptitle(T("Expansion-chamber silencer: pass band vs stop band "
                   "(2D FDTD)"), fontweight="bold")
    axes = fig.subplots(2, 1, sharex=True)
    titles = [T(f"Pass band: {f_pass:.0f} Hz, $kL = \\pi$"),
              T(f"Stop band peak: {f_peak:.0f} Hz, $kL = \\pi/2$")]
    verdicts = [T("the chamber is acoustically invisible"),
                T("the mismatch reflects the wave back up the pipe")]
    ims: list[Any] = []
    tl_txts: list[Any] = []
    v_txts: list[Any] = []
    for (ax, title, vmax), env in zip(
            zip(axes, titles, vmaxes, strict=True), (e_pass, e_stop),
            strict=True):
        ax.grid(False)
        im = ax.imshow(np.zeros((2, 2)), origin="lower",
                       extent=(0.0, length, 0.0, height), cmap=CMAP_FIELD,
                       vmin=-vmax, vmax=vmax, aspect="auto",
                       interpolation="bilinear", zorder=2)
        _anim_chamber_hardware(ax, length, height, pipe_y,
                               (0.55, 0.55 + _CHAMBER_L))
        ax.axhline(env_base, color=COLOR_GRID, lw=0.8, zorder=1)
        ax.text(0.005, env_base - 0.008, "$|p|$ envelope", fontsize=7.5,
                ha="left", va="top", color=COLOR_FG, alpha=0.8)
        # The settled envelope, static: the clip captures steady state.
        ax.plot(x_env[env_from:],
                env_base + env[env_from:] / env_max * env_h,
                color=COLOR_PRIMARY, lw=1.8, zorder=6)
        ax.set_xlim(-0.115, length + 0.065)
        # The verdict/TL line sits above the tallest envelope hump
        # (1.85 of 2.1 full scale), so neither text strikes the curve.
        ax.set_ylim(-0.030, env_base + env_h + 0.062)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        tl_txts.append(
            ax.text(length + 0.055, env_base + env_h + 0.050, "",
                    ha="right", va="top", fontsize=9, color="white",
                    zorder=7,
                    bbox={"boxstyle": _ANIM_PILL_BOX,
                          "facecolor": "black", "alpha": 0.55,
                          "edgecolor": "none"}))
        v_txts.append(
            ax.text(0.02, env_base + env_h + 0.046, "", ha="left",
                    va="top", fontsize=8, color=COLOR_FG, zorder=6))
        ims.append(im)
    axes[1].set_xlabel(T("Position along the duct [m]"), fontsize=9)
    t_txt = fig.text(0.988, 0.93, "", ha="right", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)
    # The captured field is already steady, so the verdict can come early.
    reveal = int(0.30 * len(times))

    def update(k: int) -> tuple[Any, ...]:
        for i, (p_all, tl, f) in enumerate(
                ((p_pass, tls[0], f_pass), (p_stop, tls[1], f_peak))):
            ims[i].set_data(p_all[k])
            tl_txts[i].set_text(
                T(f"TL = {tl:.1f} dB at {f:.0f} Hz")
                if k >= reveal else "")
            v_txts[i].set_text(verdicts[i] if k >= reveal else "")
        t_txt.set_text(T(f"$t$ = {times[k] * 1e3:5.1f} ms"))
        return (*ims, *tl_txts, *v_txts, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_expansion_chamber",
                 frames=len(times), gif_fps=8)


def _anim_chamber_hardware(ax: Any, length: float, height: float,
                           pipe_y: tuple[float, float],
                           chamber_x: tuple[float, float]) -> None:
    """Draw the silencer as hardware: inlet/outlet pipe walls, the chamber
    shell with its end plates, the drive loudspeaker and the anechoic
    termination, all to scale (metres)."""
    from matplotlib.patches import Rectangle

    wall = 0.012
    grey = "#9a9a9a"
    x0, x1 = chamber_x
    for y_pipe in pipe_y:
        y0 = y_pipe - (wall if y_pipe == pipe_y[0] else 0.0)
        for xa, xb in ((0.0, x0), (x1, length)):
            ax.add_patch(Rectangle((xa, y0), xb - xa, wall,
                                   facecolor=grey, edgecolor=COLOR_FG,
                                   linewidth=0.7, zorder=3))
    for y0 in (-wall, height):             # chamber shell
        ax.add_patch(Rectangle((x0 - wall, y0), x1 - x0 + 2 * wall, wall,
                               facecolor=grey, edgecolor=COLOR_FG,
                               linewidth=0.7, zorder=3))
    for xe in (x0 - wall, x1):             # end plates (annular in 3D)
        for ya, yb in ((0.0, pipe_y[0]), (pipe_y[1], height)):
            ax.add_patch(Rectangle((xe, ya), wall, yb - ya,
                                   facecolor=grey, edgecolor=COLOR_FG,
                                   linewidth=0.7, zorder=3))
    # Loudspeaker into the inlet.
    bore = pipe_y[1] - pipe_y[0]
    _anim_speaker(ax, 0.0, 0.5 * (pipe_y[0] + pipe_y[1]), bore,
                  tip_inset=0.003, label_y=pipe_y[0] - 0.02)
    ax.add_patch(Rectangle((length, pipe_y[0] - wall), 0.045,
                           bore + 2 * wall, facecolor=grey, hatch="////",
                           edgecolor=COLOR_FG, linewidth=0.8, zorder=3))
    ax.text(length + 0.045, pipe_y[0] - 0.02, "anechoic termination",
            ha="right", va="top", fontsize=7.5)
    ax.text(0.5 * (x0 + x1), height + wall + 0.006,
            f"$L$ = {_CHAMBER_L:.2f} m · $m$ = {_CHAMBER_M:.0f}",
            ha="center", va="bottom", fontsize=7.5, color=COLOR_FG)
