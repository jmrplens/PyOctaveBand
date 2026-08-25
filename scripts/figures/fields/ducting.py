#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The SOFAR channel: sound trapped by an ocean sound-speed minimum."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from ..media import _anim_figure, _render_clip, _translate_str
from ..schematics import _grid_axes
from ..theme import (
    CMAP_FIELD,
    COLOR_FG,
    COLOR_PRIMARY,
    COLOR_TERTIARY,
    FIELD_INK,
    FIELD_STROKE,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

_DUCT_AXIS = 400.0  # channel-axis depth [m]
_DUCT_SRC_DEPTHS = (400.0, 150.0)  # on the axis / near the surface


def _duct_profile(z: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Munk-style sound-speed profile with an exaggerated gradient.

    The canonical SOFAR shape (Munk 1974): a minimum at the channel axis,
    an exponential thermocline above and a near-linear pressure gradient
    below, ``c = c1 (1 + eps (eta + exp(-eta) - 1))``. The perturbation is
    scaled far beyond the real ocean's (eps = 0.35 vs ~0.0074, capped at
    +250 m/s) so a ray cycle fits the 2.4 km of this domain instead of
    tens of kilometres -- the mechanism is the point, not the scale.
    """
    c1, eps, b_scale = 1480.0, 0.35, 300.0
    eta = 2.0 * (np.asarray(z, dtype=np.float64) - _DUCT_AXIS) / b_scale
    c = c1 * (1.0 + eps * (eta + np.exp(-eta) - 1.0))
    return np.minimum(c, c1 + 250.0)


# SOFAR-duct timeline. Mesh: the pulse carries useful energy to ~19 Hz,
# where the slowest water on the axis (1 480 m/s) gives lambda = 77.9 m and
# lambda / 8 = 9.7 m; the geometry has no feature smaller than the 300 m
# profile scale (300 / 4 = 75 m), so the rule dx = min(75, 9.7) m allows
# 9.7 m and dx = 2 m sits 4.9 times finer, which the refracted wavefronts
# need to stay smooth over 2.4 km.
# Flight: nothing in this scene reflects (sponges all round, the turning
# points are refractive), so the flight is the source out to the farthest
# visible point, the corner (2 400, 800) m, over the slowest speed on the
# path (1 480 m/s on the channel axis). The near-surface source at
# (200, 150) m is the binding one at 2 294 m -- the on-axis source is
# 2 236 m away -- so 2 294 / 1 480 x 1.2 = 1.860 s, against the old
# 1.589 s window, which cut the wavefronts off before the frame edge.
# Sampling: every solver step is captured, 490.5 us apart, i.e. 107.3
# frames per 19 Hz period (>= 48; a two-step stride would already drop to
# 53.6, still compliant, but this grid's dt is the natural floor). 3 793
# frames cover the 1.860 s window, played at 180 fps -- nine times the
# 20 fps of the committed clip, exactly the factor by which the stride
# shrank from its 9 solver steps per frame -- so the pacing stays the
# 88.29 ms of simulation per second of playback the committed clip had,
# and the clip runs 21.1 s (the old 18 s clip covered only its shorter
# 1.589 s window at the same speed).
_DUCT_EVERY = 1
_DUCT_FRAMES = 3793
_DUCT_FPS = 180


@lru_cache(maxsize=1)
def _ducting_fields(
    n_frames: int = _DUCT_FRAMES,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Two pulse runs in the SOFAR-like channel (on/off axis), cached.

    A 2 400 m x 800 m ocean slice with the exaggerated Munk profile of
    :func:`_duct_profile` and sponges on all four sides. Each source is a
    zero-mean Gaussian doublet (two opposite-sign pulses 33 ms apart, peak
    energy near 13 Hz). Returns the pressure frame stacks, the
    time-integrated energy maps in dB (the closing verdict overlay), the
    frame times, the depth grid, the c(z) profile and the full-resolution
    energy maps for the trapping physics check.
    """
    import fdtd2d

    dx = 2.0
    ny, nx = 400, 1200  # 800 m depth x 2400 m range
    z = (np.arange(ny) + 0.5) * dx
    c_prof = _duct_profile(z)
    c_map = np.repeat(c_prof[:, np.newaxis], nx, axis=1)
    every = _DUCT_EVERY
    width, offset = 0.028, 0.033
    p_all, e_all = [], []
    times = np.zeros(0)
    for depth in _DUCT_SRC_DEPTHS:
        sim = fdtd2d.FDTD2D(c_map, dx, rho=1025.0, sponge_width=30)
        iy = round(depth / dx)
        sim.add_source(fdtd2d.GaussianPulse(ix=100, iy=iy, width=width))
        sim.add_source(
            fdtd2d.GaussianPulse(
                ix=100, iy=iy, width=width, t0=4.0 * width + offset, amplitude=-1.0
            )
        )
        energy = np.zeros_like(sim.p)
        ps: list[Any] = []
        ts: list[float] = []
        for _ in range(every * n_frames):
            sim.step()
            energy += sim.p**2
            if sim.n % every == 0 and len(ps) < n_frames:
                ps.append(sim.p[::2, ::2].astype(np.float32))
                ts.append(sim.time)
        p_all.append(np.stack(ps))
        e_all.append(energy)
        times = np.asarray(ts)
    energy_maps = np.stack(e_all)
    # Verdict overlay: the time-integrated p**2 map in dB (shared scale),
    # i.e. everywhere the pulse has carried energy over the whole run. The
    # reference comes from the far half of the domain so the near-source
    # blast saturates instead of washing out the duct-band contrast.
    ref = float(np.quantile(energy_maps[:, :, nx // 4 :], 0.999))
    with np.errstate(divide="ignore"):
        e_db = 10.0 * np.log10(energy_maps[:, ::2, ::2] / ref)
    e_db = np.clip(e_db, -20.0, 0.0).astype(np.float32)
    return np.stack(p_all), e_db, times, z, c_prof, energy_maps


def animate_fdtd_ducting(output_dir: str) -> None:
    """A low-frequency pulse in a SOFAR-like underwater sound channel
    (2D FDTD): launched on the channel axis the wavefronts refract back
    toward the sound-speed minimum and stay trapped; launched near the
    surface the energy crosses the channel and leaks away to depth. The
    closing seconds crossfade to the time-integrated energy map, so the
    verdict frame shows the whole path history.
    """
    from matplotlib import patheffects

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0, foreground=FIELD_STROKE)]
    p_all, e_db, times, z, c_prof, _ = _ducting_fields()
    half = p_all.shape[1] // 2
    vmax = float(np.quantile(np.abs(p_all[:, :half]), 0.999))

    fig = _anim_figure()
    fig.suptitle(T("SOFAR channel: sound trapped by the $c(z)$ minimum (2D FDTD)"))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.22, 1.0])
    titles = [
        T("Source on the channel axis (depth 400 m)"),
        T("Source near the surface (depth 150 m)"),
    ]
    verdicts = [
        T("trapped: wavefronts bend back to the axis"),
        T("leaks: energy escapes the channel"),
    ]
    extent = (0.0, 2400.0, 800.0, 0.0)
    ims: list[Any] = []
    ims_e: list[Any] = []
    v_txts: list[Any] = []
    ax_top: Any = None
    for row, depth in enumerate(_DUCT_SRC_DEPTHS):
        ax_c = fig.add_subplot(gs[row, 0])
        _grid_axes(ax_c)
        ax_c.plot(c_prof, z, color=COLOR_PRIMARY, lw=1.6)
        ax_c.plot(
            [float(np.interp(depth, z, c_prof))],
            [depth],
            marker="o",
            ms=5,
            color=COLOR_TERTIARY,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        ax_c.axhline(_DUCT_AXIS, color=COLOR_FG, ls="--", lw=0.9, alpha=0.6)
        ax_c.set_ylim(800.0, 0.0)
        ax_c.set_xlim(1460.0, 1750.0)
        ax_c.set_xticks([1480.0, 1730.0])
        ax_c.set_ylabel(T("Depth [m]"), fontsize=8)
        ax_c.tick_params(labelsize=6)
        if row == 1:
            ax_c.set_xlabel(T("$c(z)$ [m/s]"), fontsize=7)

        ax_f = fig.add_subplot(gs[row, 1])
        ax_f.grid(False)
        im = ax_f.imshow(
            p_all[row][0],
            origin="upper",
            extent=extent,
            cmap=CMAP_FIELD,
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
            interpolation="bilinear",
        )
        # The verdict overlay: fades in over the last seconds (and the
        # poster frame), replacing the instantaneous wavefronts with the
        # time-integrated energy paths.
        im_e = ax_f.imshow(
            e_db[row],
            origin="upper",
            extent=extent,
            cmap="magma",
            vmin=-20.0,
            vmax=0.0,
            aspect="auto",
            interpolation="bilinear",
            alpha=0.0,
            zorder=2.5,
        )
        ax_f.set_title(titles[row], fontsize=10)
        ax_f.axhline(_DUCT_AXIS, color="#888888", ls="--", lw=0.9, alpha=0.8, zorder=3)
        ax_f.plot(
            [200.0],
            [depth],
            marker="o",
            ms=5,
            color=COLOR_TERTIARY,
            markeredgecolor=FIELD_STROKE,
            markeredgewidth=0.8,
            zorder=4,
        )
        ax_f.text(
            240.0,
            depth - 25.0,
            T("source"),
            ha="left",
            va="bottom",
            color=FIELD_INK,
            fontsize=7.5,
            path_effects=outline,
            zorder=4,
        )
        # A translucent dark pill keeps this label legible once the bright
        # magma energy overlay fades in over the channel axis (a plain white
        # stroke washes out against the near-white high-energy region).
        ax_f.text(
            2360.0,
            425.0,
            T("channel axis ($c$ minimum)"),
            ha="right",
            va="top",
            color="white",
            fontsize=7,
            zorder=4,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "black",
                "alpha": 0.45,
                "edgecolor": "none",
            },
        )
        v_txt = ax_f.text(
            60.0,
            770.0,
            "",
            ha="left",
            va="bottom",
            color=FIELD_INK,
            fontsize=8,
            path_effects=outline,
            zorder=4,
        )
        ax_f.tick_params(labelsize=7, labelleft=False)
        if row == 0:
            ax_f.tick_params(labelbottom=False)
            ax_top = ax_f
        else:
            ax_f.set_xlabel(T("Range [m]"), fontsize=8)
        ims.append(im)
        ims_e.append(im_e)
        v_txts.append(v_txt)
    # Inside the upper panel, on the same dark pill as the channel-axis
    # label. Every margin of this figure is spoken for: the bottom-right
    # corner is the lower panel's "Range [m]" axis, the top-left and
    # top-right ones are a whisker from the long centred titles, and the
    # right margin between them put the readout astride the upper panel's
    # top spine -- where, once that panel crossfades to the magma energy
    # map, black text on black background disappeared. The pill carries its
    # own contrast, so the readout no longer depends on what is under it.
    t_txt = ax_top.text(
        55.0,
        25.0,
        "",
        ha="left",
        va="top",
        family="monospace",
        fontsize=10,
        color="white",
        zorder=4,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "black",
            "alpha": 0.45,
            "edgecolor": "none",
        },
    )
    reveal = int(0.83 * p_all.shape[1])  # ~17.5 s: pulse has crossed
    captions_on = int(0.38 * p_all.shape[1])  # first refocus is visible

    def update(k: int) -> tuple[Any, ...]:
        alpha = min(1.0, max(0.0, (k - reveal) / 12.0))
        for row in range(2):
            ims[row].set_data(p_all[row][k])
            ims_e[row].set_alpha(alpha)
            v_txts[row].set_text(verdicts[row] if k >= captions_on else "")
        t_txt.set_text(T(f"$t$ = {times[k]:5.2f} s"))
        return (*ims, *ims_e, *v_txts, t_txt)

    _render_clip(
        fig,
        update,
        output_dir,
        "anim_fdtd_ducting",
        frames=int(p_all.shape[1]),
        fps=_DUCT_FPS,
        gif_fps=5,
    )
