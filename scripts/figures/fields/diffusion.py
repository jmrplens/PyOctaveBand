#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The quadratic-residue diffuser: a specular slab against its wells."""

from functools import lru_cache
from typing import Any

import numpy as np

from ..media import _anim_figure, _render_clip, _translate_str
from ..theme import (
    CMAP_FIELD,
    COLOR_FG,
    COLOR_GRID,
    FIELD_INK,
    FIELD_STROKE,
)
from ._core import _fit_text_below

_QRD_DESIGN_F = 343.0 / 0.56             # ~612 Hz: lambda0 = 0.56 m
_QRD_SLAB = (2.085, 3.915, 0.55, 0.85)   # x0, x1, y0, y1 of the panel slab


def _qrd_wells() -> list[tuple[float, float, float]]:
    """The QRD well openings: (x0, x1, depth) per well, panel-relative.

    An N = 7 quadratic-residue sequence (n^2 mod 7), well depths
    ``s_n * lambda0 / (2 N)`` with the 0.56 m design wavelength, 12 cm
    wells split by 1 cm rigid fins, two periods across the 1.83 m panel.
    """
    seq = (0, 1, 4, 2, 2, 4, 1)
    unit = 0.56 / (2 * len(seq))          # = 4 cm per residue step
    well_w, fin_w = 0.12, 0.01
    wells = []
    x = _QRD_SLAB[0]
    for _ in range(2):
        for s in seq:
            wells.append((x + fin_w, x + fin_w + well_w, s * unit))
            x += fin_w + well_w
    return wells


# Diffuser timeline. Mesh: the carrier is the 612 Hz design frequency
# (lambda = 0.56 m, lambda / 8 = 70 mm) and the smallest acoustic dimension
# of the panel is the 4 cm quadratic-residue depth unit (40 / 4 = 10 mm),
# so the rule dx = min(10, 70) mm gives exactly the 10 mm mesh used here.
# (The 1 cm fins between wells are one cell wide: they are the discrete
# stand-in for the infinitely thin rigid separators of Schroeder theory,
# and one cell of 10^6:1 density contrast already makes them impermeable.
# What has to be resolved is the well -- 12 cells across, 4 to 16 deep.)
# Flight: the packet starts 3.2 m up and runs 2.51 m down to the deepest
# well bottom (0.85 - 0.16 = 0.69 m); the worst-placed of those deepest
# wells sits at x = 3.715 m (see _qrd_wells), and from there the scattered
# fan has 4.685 m to the farthest visible corner of the cropped frame
# (0.4, 4.0). So 7.195 m / 343 m/s x 1.2 = 25.17 ms, against the old
# 22.27 ms window.
# Sampling: 2 solver steps per frame = 24.74 us gives 66.0 frames per
# period of the 612 Hz carrier (>= 48; 30.0 at the 1.35 kHz edge of the
# packet spectrum, itself well over the old 12-frame floor). 1 018 frames
# cover the 25.18 ms window, played at 50 fps -- 5/2 of the 20 fps of the
# committed clip, exactly the factor by which the stride shrank from its
# 5 solver steps per frame -- so the pacing stays the 1.237 ms of
# simulation per second of playback the committed clip had, and the clip
# runs 20.4 s (the old 18 s clip covered only its shorter 22.27 ms window
# at the same speed).
_DIFF_EVERY = 2
_DIFF_FRAMES = 1018
_DIFF_FPS = 50


@lru_cache(maxsize=1)
def _diffusion_fields(
    n_frames: int = _DIFF_FRAMES,
) -> tuple[Any, Any, Any, Any, Any]:
    """Plane-wave packet onto a flat panel vs a QRD, cached (three runs).

    A 6 m x 4.4 m free-field box (sponges all around) with a floating
    1.83 m panel: solid slab vs the same slab with quadratic-residue wells
    (5 000:1 density contrast builds the rigid geometry). A downward
    plane-wave packet (carrier at the 612 Hz design frequency) is injected
    as an initial condition with matched leapfrog velocities, so a single
    clean wavefront travels down. A third, panel-free reference run gives
    the incident field, so ``scattered = total - incident`` exactly.

    Returns the two total-field frame stacks, the scattered-field envelope
    trails [dB] (a fading |total - incident| history, so the specular beam
    and the diffuse fan persist into the verdict frame), frame times, the
    arc angles [deg], and the scattered energy levels on the arc [dB] per
    panel (flat, QRD) for the diffusion-coefficient check.
    """
    import fdtd2d

    c0, dx = 343.0, 0.01
    ny, nx = 440, 600                      # 4.4 m x 6.0 m
    x0, x1, y0, y1 = _QRD_SLAB
    rho_flat = np.full((ny, nx), 1.2)
    rho_flat[round(y0 / dx):round(y1 / dx),
             round(x0 / dx):round(x1 / dx)] = 1.2e6
    rho_qrd = rho_flat.copy()
    for wx0, wx1, d in _qrd_wells():
        if d > 0.0:
            rho_qrd[round((y1 - d) / dx):round(y1 / dx),
                    round(wx0 / dx):round(wx1 / dx)] = 1.2
    rho_ref = np.full((ny, nx), 1.2)

    # Downgoing packet: carrier at the design wavelength under a Gaussian
    # envelope (~10 % spectral amplitude beyond 1.15 kHz).
    y_pkt, sig, lam0 = 3.2, 0.3, 0.56

    # Receiver arc over the panel, ISO 17497-2 goniometer style.
    theta = np.arange(15.0, 165.5, 1.0)
    rad = np.radians(theta)
    ix_arc = np.round((3.0 + 2.2 * np.cos(rad)) / dx).astype(int)
    iy_arc = np.round((y1 + 2.2 * np.sin(rad)) / dx).astype(int)

    every = _DIFF_EVERY
    sims = []
    for rho in (rho_flat, rho_qrd, rho_ref):
        sim = fdtd2d.FDTD2D(c0, dx, rho=rho, shape=(ny, nx),
                            sponge_width=40)
        # One-way carrier-under-Gaussian packet toward the panel, the
        # leapfrog-consistent initial condition the solver now provides.
        sim.add_plane_wave("up", center=y_pkt, width=sig, wavelength=lam0)
        sims.append(sim)
    # The three runs advance in lockstep so the scattered field
    # (total - incident, exact by linearity) is available at every step
    # for the arc energies and for the fading envelope trails (6 ms
    # half-life). Below the panel face the difference is just the ghost
    # of the unblocked incident wave, so the trail is masked there.
    decay = float(2.0 ** (-sims[0].dt / 0.006))
    face_row = round(y1 / dx)
    trails = [np.zeros_like(sims[0].p) for _ in range(2)]
    arc_e = np.zeros((2, theta.size))
    tot_frames: list[list[Any]] = [[], []]
    trail_frames: list[list[Any]] = [[], []]
    ts: list[float] = []
    for _ in range(every * n_frames):
        for sim in sims:
            sim.step()
        for j in range(2):
            scat = sims[j].p - sims[2].p
            scat[:face_row, :] = 0.0
            arc_e[j] += scat[iy_arc, ix_arc] ** 2
            np.maximum(trails[j] * decay, np.abs(scat), out=trails[j])
        if sims[0].n % every == 0 and len(ts) < n_frames:
            for j in range(2):
                tot_frames[j].append(
                    sims[j].p[::2, ::2].astype(np.float32))
                trail_frames[j].append(
                    trails[j][::2, ::2].astype(np.float32))
            ts.append(sims[0].time)
    times = np.asarray(ts)
    tot = np.stack([np.stack(f) for f in tot_frames])
    trail = np.stack([np.stack(f) for f in trail_frames])
    ref = float(trail[:, trail.shape[1] // 3:].max())
    with np.errstate(divide="ignore"):
        trail_db = 20.0 * np.log10(trail / ref)
    trail_db = np.clip(trail_db, -30.0, 0.0).astype(np.float32)
    # Arc levels floored 45 dB under each panel's own peak.
    levels = []
    for j in range(2):
        with np.errstate(divide="ignore"):
            lvl = 10.0 * np.log10(arc_e[j] / float(arc_e[j].max()))
        levels.append(np.maximum(lvl, -45.0))
    return tot, trail_db, times, theta, np.stack(levels)


def animate_fdtd_diffusion(output_dir: str) -> None:
    """A plane wavefront hitting a flat rigid panel vs a Schroeder QRD
    (2D FDTD, ISO 17497-2 goniometer style): the flat panel throws a
    collimated specular beam back, the diffuser's phase-step wells spray
    the same energy into a wide fan; the scattered field (total minus
    incident) and the arc diffusion coefficients make the contrast
    quantitative."""
    from matplotlib import patheffects
    from matplotlib.patches import Polygon

    from phonometry import directional_diffusion_coefficient

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    tot_all, trail_db, times, theta, levels = _diffusion_fields()
    d_coef = [directional_diffusion_coefficient(lv) for lv in levels]
    x0, x1, y0, y1 = _QRD_SLAB
    vmax = float(np.quantile(np.abs(tot_all[:, 0]), 0.999))

    fig = _anim_figure()
    fig.suptitle(T("Flat panel vs Schroeder diffuser (2D FDTD)"),
                 )
    gs = fig.add_gridspec(2, 2)
    titles = [T("Flat rigid panel"), T("Schroeder diffuser (QRD, $N$ = 7)")]
    beams = [T("specular beam"), T("scattered fan")]
    # Panel cross-sections: the flat slab, and the staircase along the QRD
    # surface (wells carved into the same slab).
    flat_poly = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
    qrd_poly: list[tuple[float, float]] = [(x0, y0), (x0, y1)]
    for wx0, wx1, d in _qrd_wells():
        qrd_poly += [(wx0, y1), (wx0, y1 - d), (wx1, y1 - d), (wx1, y1)]
    qrd_poly += [(x1, y1), (x1, y0)]
    polys = [flat_poly, qrd_poly]
    rad = np.radians(theta)
    arc_x = 3.0 + 2.2 * np.cos(rad)
    arc_y = y1 + 2.2 * np.sin(rad)

    ims: list[Any] = []
    d_txts: list[Any] = []
    for col in range(2):
        ax_t = fig.add_subplot(gs[0, col])
        ax_s = fig.add_subplot(gs[1, col])
        im_t = ax_t.imshow(tot_all[col][0], origin="lower",
                           extent=(0.0, 6.0, 0.0, 4.4), cmap=CMAP_FIELD,
                           vmin=-vmax, vmax=vmax, interpolation="bilinear")
        im_s = ax_s.imshow(trail_db[col][0], origin="lower",
                           extent=(0.0, 6.0, 0.0, 4.4), cmap="magma",
                           vmin=-30.0, vmax=0.0, interpolation="bilinear")
        ax_t.set_title(titles[col], fontsize=10)
        for ax in (ax_t, ax_s):
            ax.grid(False)
            ax.add_patch(Polygon(polys[col], closed=True,
                                 facecolor=COLOR_GRID, edgecolor=COLOR_FG,
                                 lw=1.0))
            # Crop the absorbing sponge zones out of view, so the frame
            # edge is physical field, not the boundary-layer artefacts.
            ax.set_xlim(0.4, 5.6)
            ax.set_ylim(0.0, 4.0)
            ax.tick_params(labelsize=7)
        ax_t.tick_params(labelbottom=False)
        ax_s.set_xlabel("$x$ [m]", fontsize=8)
        ax_t.text(3.0, 3.6, T("incident plane wavefront"), ha="center",
                  va="bottom", color=FIELD_INK, fontsize=7.5,
                  path_effects=outline)
        ax_t.annotate("", xy=(3.0, 3.1), xytext=(3.0, 3.55),
                      arrowprops={"arrowstyle": "-|>", "color": FIELD_INK,
                                  "lw": 1.2})
        ax_s.plot(arc_x, arc_y, ls=":", color="white", lw=0.9, alpha=0.65)
        ax_s.text(3.0, 0.15, beams[col], ha="center", va="bottom",
                  color="white", fontsize=7.5)
        # Written out here so the arc label below can be measured against
        # the annotation that will actually be drawn; update() blanks it
        # until the arc energy has settled.
        d_txt = ax_s.text(5.45, 3.82,
                          T(f"diffusion coefficient $d$ = {d_coef[col]:.2f}"),
                          ha="right", va="top", color="white",
                          fontsize=8.5, fontweight="bold")
        if col == 0:
            ax_t.set_ylabel(T("sound field $p$"), fontsize=9)
            ax_s.set_ylabel(T("scattered field (total − incident)"),
                            fontsize=8)
            arc_txt = ax_s.text(0.78, 2.05, T("receiver arc"), ha="left",
                                va="bottom", color="white", fontsize=6.5,
                                rotation=64.0)
        else:
            ax_t.tick_params(labelleft=False)
            ax_s.tick_params(labelleft=False)
            ax_t.text(3.0, 0.15, T(f"design frequency "
                                   f"{_QRD_DESIGN_F:.0f} Hz"), ha="center",
                      va="bottom", color=FIELD_INK, fontsize=6.5,
                      path_effects=outline)
        ims += [im_t, im_s]
        d_txts.append(d_txt)
    t_txt = fig.text(0.985, 0.02, "", ha="right", va="bottom",
                     family="monospace", fontsize=10, color=COLOR_FG)
    # The arc label climbs the receiver arc toward the corner the diffusion
    # coefficient is printed in, and the Spanish string is half again as
    # long as the English one: at a fixed anchor its tail ran through the
    # middle of that annotation. Push it down the arc instead, by whatever
    # its own length asks for.
    def fit_arc_label() -> None:
        _fit_text_below(fig, d_txts[0].axes, arc_txt, d_txts[0], gap=10.0)
        for d_txt in d_txts:
            d_txt.set_text("")

    reveal = int(0.8 * tot_all.shape[1])   # arc energy has settled by here

    def update(k: int) -> tuple[Any, ...]:
        for col in range(2):
            ims[2 * col].set_data(tot_all[col][k])
            ims[2 * col + 1].set_data(trail_db[col][k])
            d_txts[col].set_text(
                T(f"diffusion coefficient $d$ = {d_coef[col]:.2f}")
                if k >= reveal else "")
        t_txt.set_text(T(f"$t$ = {times[k] * 1000.0:4.1f} ms"))
        return (*ims, *d_txts, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_diffusion",
                 frames=int(tot_all.shape[1]), fps=_DIFF_FPS, gif_fps=5,
                 measure=fit_arc_label)
