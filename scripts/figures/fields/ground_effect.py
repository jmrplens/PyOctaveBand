#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The ground effect: the direct wave and its ground bounce interfering."""

from functools import lru_cache
from typing import Any

import numpy as np

from ..i18n import _LANG
from ..media import _anim_figure, _render_clip, _translate_str
from ..schematics import _grid_axes
from ..theme import (
    CMAP_FIELD,
    COLOR_FG,
    COLOR_GRID,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    FIELD_INK,
    FIELD_STROKE,
)
from ._core import _rms_to_db

_GROUND_FREQ = 400.0
_GROUND_H = 1.5
_GROUND_ARC_R = 8.0
# Ground-effect timeline. Mesh: the carrier is 400 Hz (lambda = 0.858 m,
# lambda / 8 = 107 mm) and the smallest geometric dimension is the 1.5 m
# source height (1.5 / 4 = 0.38 m), so the rule dx = min(0.38, 0.107) m
# allows 107 mm; dx = 20 mm sits 5.4 times finer, which the 8 m sampling
# arc needs to stay smooth.
# Flight: the source at (1.6, 1.5) m reaches the deepest reflector -- the
# rigid ground 1.5 m below -- and from there the farthest visible point,
# the top right corner (14, 8), is 14.757 m away, so 16.257 m / 343 m/s
# x 1.2 = 56.88 ms.
# Sampling: 2 solver steps per frame = 49.48 us gives 50.5 frames per
# 400 Hz period (>= 48). 1 150 frames cover the 56.90 ms window, played at
# 80 fps -- four times the 20 fps of the committed clip, exactly the
# factor by which the stride shrank from its 8 solver steps per frame --
# so the pacing stays the 3.958 ms of simulation per second of playback
# the committed clip had, and the clip runs 14.4 s (the old 18 s clip
# spent its extra seconds past the flight, on the 71.2 ms window trimmed
# here).
_GROUND_EVERY = 2
_GROUND_FRAMES = 1150
_GROUND_FPS = 80


@lru_cache(maxsize=1)
def _ground_effect_fields(
    n_frames: int = _GROUND_FRAMES,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """One CW run of a point source 1.5 m over rigid ground, cached.

    Returns instantaneous frames, RMS maps in dB, frame times, the arc
    angles [deg], the per-frame arc levels [dB re the final maximum], the
    two-path image-source model levels on the same arc, and the predicted
    null angles where the path difference is an odd multiple of lambda/2.
    """
    import fdtd2d

    c0, dx = 343.0, 0.02
    ny, nx = 400, 700                      # 8 m x 14 m
    sim = fdtd2d.FDTD2D(c0, dx, shape=(ny, nx), sponge_width=40,
                        sponge_sides=("left", "right", "bottom"))
    sim.add_source(fdtd2d.CWSource(ix=80, iy=75, frequency=_GROUND_FREQ,
                                   ramp_cycles=2.0))
    every = _GROUND_EVERY
    lam = c0 / _GROUND_FREQ
    theta = np.arange(1.0, 62.5, 0.5)
    rad = np.radians(theta)
    arc_x = 1.6 + _GROUND_ARC_R * np.cos(rad)
    arc_y = _GROUND_ARC_R * np.sin(rad)
    ix_arc = np.round(arc_x / dx).astype(int)
    iy_arc = np.round(arc_y / dx).astype(int)

    beta = float(np.exp(-sim.dt * _GROUND_FREQ / 2.0))
    ms = np.zeros_like(sim.p)
    ps: list[Any] = []
    rs: list[Any] = []
    ts: list[float] = []
    arc: list[Any] = []
    for _ in range(every * n_frames):
        sim.step()
        ms = beta * ms + (1.0 - beta) * sim.p**2
        if sim.n % every == 0 and len(ps) < n_frames:
            ps.append(sim.p[::2, ::2].astype(np.float32))
            rs.append(np.sqrt(ms[::2, ::2]).astype(np.float32))
            ts.append(sim.time)
            arc.append(np.sqrt(ms[iy_arc, ix_arc]))
    arc_rms = np.stack(arc)
    with np.errstate(divide="ignore"):
        arc_db = 20.0 * np.log10(arc_rms / float(arc_rms[-1].max()))
    arc_db = np.clip(arc_db, -45.0, None)

    # Two-path image-source model on the same arc (2D line source: 1/sqrt(r)
    # spreading), and the predicted nulls where r2 - r1 = (m + 1/2) lambda.
    def paths(th: Any) -> tuple[Any, Any]:
        x, y = (_GROUND_ARC_R * np.cos(np.radians(th)),
                _GROUND_ARC_R * np.sin(np.radians(th)))
        r1 = np.hypot(x, y - _GROUND_H)
        r2 = np.hypot(x, y + _GROUND_H)
        return r1, r2

    r1, r2 = paths(theta)
    k = 2.0 * np.pi / lam
    h = (np.exp(1j * k * r1) / np.sqrt(r1)
         + np.exp(1j * k * r2) / np.sqrt(r2))
    model_db = 20.0 * np.log10(np.abs(h) / float(np.abs(h).max()))
    th_fine = np.arange(0.5, 62.4, 0.02)
    r1f, r2f = paths(th_fine)
    delta = r2f - r1f
    nulls: list[float] = []
    for m in range(4):
        target = (m + 0.5) * lam
        cross = np.nonzero(np.diff(np.sign(delta - target)))[0]
        nulls.extend(float(np.interp(target, delta[c:c + 2],
                                     th_fine[c:c + 2])) for c in cross)
    return (np.stack(ps), _rms_to_db(np.stack(rs)), np.asarray(ts),
            theta, arc_db.astype(np.float32), model_db, tuple(sorted(nulls)))


def animate_fdtd_ground_effect(output_dir: str) -> None:
    """A 400 Hz point source 1.5 m above rigid ground (2D FDTD): the direct
    and ground-reflected wavefronts interfere and the lobe pattern forms;
    the ghosted image source below the ground explains the geometry, and the
    level sampled on an 8 m arc converges to the two-path model with its
    predicted nulls."""
    from matplotlib import patheffects

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    (p_frames, db_frames, times, theta, arc_db, model_db,
     nulls) = _ground_effect_fields()
    half = p_frames.shape[0] // 2
    vmax = float(np.quantile(np.abs(p_frames[half:]), 0.995))

    fig = _anim_figure()
    fig.suptitle(T("Ground effect: direct + reflected interference "
                   "(2D FDTD)"), fontweight="bold")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0])
    ax_p = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[1, 0])
    ax_l = fig.add_subplot(gs[:, 1])

    im_p = ax_p.imshow(p_frames[0], origin="lower",
                       extent=(0.0, 14.0, 0.0, 8.0), cmap=CMAP_FIELD,
                       vmin=-vmax, vmax=vmax, interpolation="bilinear")
    im_r = ax_r.imshow(db_frames[0], origin="lower",
                       extent=(0.0, 14.0, 0.0, 8.0), cmap="magma",
                       vmin=-40.0, vmax=0.0, interpolation="bilinear")
    rad = np.radians(theta)
    arc_x = 1.6 + _GROUND_ARC_R * np.cos(rad)
    arc_y = _GROUND_ARC_R * np.sin(rad)
    for ax in (ax_p, ax_r):
        ax.grid(False)
        ax.set_ylim(-1.9, 8.0)
        ax.fill_between([0.0, 14.0], -1.9, 0.0, facecolor=COLOR_GRID,
                        edgecolor=COLOR_FG, lw=0.8, hatch="///")
        ax.tick_params(labelsize=7)
    ax_p.tick_params(labelbottom=False)
    ax_r.set_xlabel("x [m]", fontsize=8)
    ax_p.set_ylabel(T("instantaneous pressure"), fontsize=9)
    ax_r.set_ylabel(T("RMS level: interference lobes"), fontsize=8)
    # Source, ghosted image source and the mirror geometry.
    ax_p.plot([1.6], [_GROUND_H], marker="o", ms=5, color=COLOR_TERTIARY,
              markeredgecolor=FIELD_STROKE, markeredgewidth=0.8)
    ax_p.text(1.95, 1.65, T("source (h = 1.5 m)"), ha="left", va="center",
              color=FIELD_INK, fontsize=7.5, path_effects=outline)
    ax_p.plot([1.6], [-_GROUND_H], marker="o", ms=5, mfc="none",
              color=COLOR_TERTIARY, alpha=0.85)
    ax_p.plot([1.6, 1.6], [_GROUND_H, -_GROUND_H], ls=":",
              color=COLOR_TERTIARY, lw=1.0, alpha=0.7)
    hatch_box = {"boxstyle": "round,pad=0.2",
                 "facecolor": fig.get_facecolor(), "edgecolor": "none"}
    ax_p.text(1.95, -1.5, T("image source (ghost)"), ha="left",
              va="center", color=COLOR_FG, fontsize=7.5, bbox=hatch_box)
    ax_p.text(13.5, -0.95, T("rigid ground"), ha="right", va="center",
              color=COLOR_FG, fontsize=6.5, bbox=hatch_box)
    ax_p.text(0.3, 7.6, f"f = {_GROUND_FREQ:.0f} Hz", ha="left", va="top",
              color=FIELD_INK, fontsize=8, path_effects=outline)
    # Sampling arc and the receiver sitting in the first-order dip.
    ax_r.plot(arc_x, arc_y, ls=":", color="white", lw=0.9, alpha=0.6)
    th_dip = nulls[1]
    dip_x = 1.6 + _GROUND_ARC_R * np.cos(np.radians(th_dip))
    dip_y = _GROUND_ARC_R * np.sin(np.radians(th_dip))
    ax_r.plot([dip_x], [dip_y], marker="o", ms=5, color="white",
              markeredgecolor="black", markeredgewidth=0.8)
    # Right-aligned to the left of the dot: the Spanish label is longer
    # and would clip at the panel edge on the other side.
    ax_r.text(dip_x - 0.35, dip_y + 0.42, T("receiver in a dip"),
              ha="right", va="center", color="white", fontsize=7.5)
    # Level vs elevation angle, converging to the image-source model.
    _grid_axes(ax_l)
    ax_l.set_title(T("Level on the 8 m arc"), fontsize=10,
                   fontweight="bold")
    ax_l.set_xlabel(T("elevation angle θ [°]"), fontsize=8)
    ax_l.set_ylabel(T("level [dB re max]"), fontsize=8)
    ax_l.set_xlim(0.0, 63.0)
    ax_l.set_ylim(-34.0, 3.0)
    ax_l.tick_params(labelsize=7)
    for i, th in enumerate(n for n in nulls if n <= 62.0):
        ax_l.axvline(th, color=COLOR_SECONDARY, ls=":", lw=1.2,
                     label=T("predicted nulls") if i == 0 else None)
    ax_l.plot(theta, model_db, ls="--", color=COLOR_FG, lw=1.3, alpha=0.75,
              label=T("image-source model"))
    (l_sim,) = ax_l.plot([], [], color=COLOR_PRIMARY, lw=2.0, label="FDTD")
    # Same white-dot styling as the field receiver, so the two views of
    # the receiver are visually the same object.
    ax_l.plot([th_dip], [float(np.interp(th_dip, theta, model_db))],
              marker="o", ms=5, color="white", markeredgecolor="black",
              markeredgewidth=0.8)
    # Bottom right: past the third null the curve climbs back to its last
    # maximum and stays in the top half of the panel, so the corner under it
    # is the one patch of the panel no trace crosses. Centre right, where
    # the legend used to sit, is exactly where the third minimum -- the one
    # the clip exists to line up with its predicted null -- comes down.
    ax_l.legend(fontsize=7, loc="lower right", framealpha=0.92)
    # The strip above 0 dB is data-free, so the closing caption fits there.
    # The longer Spanish verdict spans the whole panel at 7.5 pt, so it drops
    # a step to keep a margin on both sides. The dotted null lines run the
    # full height of the panel and through the words, so the caption carries
    # a halo to keep a clear channel around its glyphs.
    verdict_txt = ax_l.text(0.5, 0.975, "", transform=ax_l.transAxes,
                            ha="center", va="top", color=COLOR_FG,
                            fontsize=7.5 if _LANG == "en" else 6.5,
                            fontweight="bold", path_effects=[
                                patheffects.withStroke(
                                    linewidth=2.6, foreground=FIELD_STROKE)])
    # Level with the arc panel's title, over the field panels, which carry
    # no title of their own: the bottom-left corner put the readout right
    # under the first x tick of the RMS panel, and the bottom-right one
    # belongs to the arc panel's wide x-label.
    t_txt = fig.text(0.02, 0.935, "", ha="left", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)
    reveal = int(0.83 * p_frames.shape[0])

    def update(k: int) -> tuple[Any, ...]:
        im_p.set_data(p_frames[k])
        im_r.set_data(db_frames[k])
        l_sim.set_data(theta, arc_db[k])
        verdict_txt.set_text(
            T("dips land exactly on the predicted nulls")
            if k >= reveal else "")
        t_txt.set_text(T(f"t = {times[k] * 1000.0:4.1f} ms"))
        return (im_p, im_r, l_sim, verdict_txt, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_ground_effect",
                 frames=int(p_frames.shape[0]), fps=_GROUND_FPS, gif_fps=5)
