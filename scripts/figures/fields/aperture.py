#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The aperture in a wall: a sub-wavelength slit against a wide gap."""

from functools import lru_cache
from typing import Any

import numpy as np

from ..media import (
    _ANIM_FPS,
    _ANIM_PILL_BOX,
    _anim_figure,
    _render_clip,
    _translate_str,
)
from ..theme import CMAP_FIELD, COLOR_FG, FIELD_INK, FIELD_STROKE
from ._core import (
    _fdtd_cw_capture,
    _fit_text_x,
    _gain_note,
    _weak_field_gain,
)

_APERTURE_F = 686.0                      # lambda = 0.50 m exactly
_APERTURE_WIDTHS = (0.025, 0.50)         # sub-lambda slit / lambda-sized gap
_APERTURE_DEPTH = 0.10                   # wall thickness across the opening
# Mesh rule: dx = min(smallest scene dimension / 4, lambda/8 at the carrier)
# = min(25 mm / 4 = 6.25 mm, 0.5 m / 8 = 62.5 mm) -> the narrow slit governs
# (lambda/80, 4 cells across it).
_APERTURE_DX = 0.00625
_APERTURE_EVERY = 3
_APERTURE_FRAMES = 962
#: Top of the framing. The display-gain note sits under the top spine and
#: the incident-wavefront label just under the note, so the panel is framed
#: 0.1 m above the band the two of them need; the sponge layer only starts
#: at y = 4.75, so the extra strip is still physical field.
_APERTURE_YTOP = 4.4
#: 8/3 of the shared rate, matching the stride cut (see the note below).
_APERTURE_FPS = _ANIM_FPS * 8.0 / 3.0


@lru_cache(maxsize=1)
def _aperture_fields(
    n_frames: int = _APERTURE_FRAMES,
) -> tuple[Any, Any, Any, Any]:
    """Two CW runs against the slotted wall, cached.

    A 6 m x 5 m free field (sponges on the far side and above/below, kept
    out of the final framing) with a rigid 0.10 m wall at x = 2 m holding
    a centred opening: 25 mm (lambda/20) in run one, 0.50 m (= lambda) in
    run two. A sustained 686 Hz plane wave enters through the rho c left
    edge; the wall reflection travels back out through the same matched
    edge. ``dx = 6.25 mm`` keeps four cells across the narrow slit.
    Returns the frame stacks, the running-RMS maps in dB on a SHARED
    scale (both referenced to the strongest final RMS, so the two
    transmitted fields are directly comparable), the frame times and the
    library transmission of the narrow slit at the drive frequency.
    """
    import fdtd2d

    from phonometry import slit_transmission_coefficient

    dx = _APERTURE_DX
    ny, nx = 800, 960                      # 5 m x 6 m
    wall_c = (round(2.0 / dx), round((2.0 + _APERTURE_DEPTH) / dx))
    # Clip duration per the deepest-reflector rule: d(source -> far wall
    # face through the opening) = 2.1 m, plus d(slit exit -> farthest
    # visible frame corner (5.72, 0.7)) = 4.04 m: t = 1.2 * 6.14 / 343 =
    # 21.5 ms captured from cold, so the clip is the transit story itself.
    # Sampling: 3 solver steps per frame = 23.19 us gives 62.9 frames per
    # 686 Hz period (>= 48; the old 8-step stride gave 23.6 and a 4-step
    # one would stop at 47.1). 962 frames cover the 22.31 ms window,
    # played at 8/3 of the shared 20 fps -- exactly the factor by which the
    # stride shrank, so both the 18.0 s length and the 1.237 ms of
    # simulation per second of playback are the ones the committed clip
    # already had.
    every = _APERTURE_EVERY
    p_all, r_all = [], []
    times = np.zeros(0)
    for width in _APERTURE_WIDTHS:
        gap = round(width / dx)
        mask = np.zeros((ny, nx), dtype=bool)
        mask[:, wall_c[0]:wall_c[1]] = True
        mask[(ny - gap) // 2:(ny + gap) // 2, wall_c[0]:wall_c[1]] = False
        sim = fdtd2d.FDTD2D(343.0, dx, shape=(ny, nx), sponge_width=40,
                            sponge_sides=("top", "bottom", "right"),
                            obstacle_mask=mask,
                            edge_impedance={"left": 1.2 * 343.0})
        tone = fdtd2d.CWSource(0, 0, frequency=_APERTURE_F)
        sim.add_source(fdtd2d.PlaneWaveSource("right", tone.value,
                                              offset=2))
        ps, rs, times, _ = _fdtd_cw_capture(sim, _APERTURE_F, every,
                                            n_frames)
        p_all.append(ps)
        r_all.append(rs)
    rms = np.stack(r_all)
    ref = float(max(r[-1].max() for r in r_all))
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(rms / ref)
    db_all = np.clip(db, -40.0, 0.0).astype(np.float32)
    res = slit_transmission_coefficient(
        np.array([_APERTURE_F]), _APERTURE_WIDTHS[0], _APERTURE_DEPTH,
        field="normal")
    tau = float(res.transmission_coefficient[0])
    return np.stack(p_all), db_all, times, tau


def animate_fdtd_aperture_slit(output_dir: str) -> None:
    """A plane wavefront against a rigid wall with an opening (2D FDTD),
    sub-wavelength versus wavelength-sized side by side: the 25 mm slit
    re-radiates the little it lets through as a cylindrical wave into a
    nearly uniform half space, while the 0.50 m gap passes the front
    almost intact and casts sharp-edged shadows. The Gomperts model of
    the library gives the narrow slit's transmission coefficient. The
    shadow side of each instantaneous panel rides its own annotated
    display gain so the slit's re-radiation is visible next to the
    standing wave facing the wall; the RMS row keeps one shared scale."""
    from matplotlib import patheffects
    from matplotlib.patches import Rectangle

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    p_all, db_all, times, tau = _aperture_fields()
    lam = 343.0 / _APERTURE_F
    half = p_all.shape[1] // 2
    vmax = float(np.quantile(np.abs(p_all[0][half:]), 0.999))
    wall_x = (2.0, 2.0 + _APERTURE_DEPTH)
    # What the lambda/20 slit passes is ~30 dB under the standing wave that
    # faces the wall and sets the colour scale, so on the incident ramp the
    # cylindrical re-radiation this panel exists to show is a black field.
    # The shadow side of each panel therefore carries its own display gain
    # (see :func:`_weak_field_gain`), stated on the panel; the gain is per
    # panel because the wavelength-sized opening needs none, and the RMS row
    # below keeps the single shared scale that compares the two.
    past = round((wall_x[1] / 6.0) * p_all.shape[3])
    gains = [_weak_field_gain(p_all[col][half::4, :, past:], vmax)
             for col in range(2)]

    fig = _anim_figure()
    fig.suptitle(T("Sound through a wall aperture (2D FDTD)"),
                 )
    gs = fig.add_gridspec(2, 2)
    titles = [T(rf"Slit $w$ = {_APERTURE_WIDTHS[0] * 1e3:.0f} mm "
                rf"($\lambda/20$)"),
              T(rf"Opening $w$ = {_APERTURE_WIDTHS[1]:.2f} m (= $\lambda$)")]
    verdicts = [T("cylindrical re-radiation from the slit"),
                T("the front passes: sharp-edged shadow")]
    ims: list[Any] = []
    v_txts: list[Any] = []
    tau_txt: Any = None
    for col in range(2):
        gap = _APERTURE_WIDTHS[col]
        ax_p = fig.add_subplot(gs[0, col])
        ax_r = fig.add_subplot(gs[1, col])
        im_p = ax_p.imshow(p_all[col][0], origin="lower",
                           extent=(0.0, 6.0, 0.0, 5.0), cmap=CMAP_FIELD,
                           vmin=-vmax, vmax=vmax, interpolation="bilinear")
        im_r = ax_r.imshow(db_all[col][0], origin="lower",
                           extent=(0.0, 6.0, 0.0, 5.0), cmap="magma",
                           vmin=-40.0, vmax=0.0, interpolation="bilinear")
        ax_p.set_title(titles[col], fontsize=10)
        for ax in (ax_p, ax_r):
            ax.grid(False)
            for y0, y1 in ((0.0, 2.5 - gap / 2.0), (2.5 + gap / 2.0, 5.0)):
                ax.add_patch(Rectangle((wall_x[0], y0),
                                       _APERTURE_DEPTH, y1 - y0,
                                       facecolor="#707070",
                                       edgecolor="white", lw=0.5))
            # Crop the sponge layers (and the injection seam) out of view,
            # so the frame edge is physical field.
            ax.set_xlim(0.08, 5.72)
            ax.set_ylim(0.7, _APERTURE_YTOP)
            ax.tick_params(labelsize=7)
        ax_p.tick_params(labelbottom=False)
        ax_r.set_xlabel("$x$ [m]", fontsize=8)
        # The arrow and its caption ride 0.1 m lower than the free field
        # would need them to: the gain note is the other tenant of this
        # strip, and its halo was rubbing out the caption's ascenders.
        ax_p.annotate("", xy=(1.15, 3.65), xytext=(0.55, 3.65),
                      arrowprops={"arrowstyle": "-|>", "color": FIELD_INK,
                                  "lw": 1.2})
        ax_p.text(0.85, 3.75, T("incident plane wavefront"), ha="left",
                  va="bottom", color=FIELD_INK, fontsize=7.5,
                  path_effects=outline)
        ax_p.text(2.24, 1.45, T("rigid wall"), ha="left", va="center",
                  color=FIELD_INK, fontsize=7, path_effects=outline)
        if (note := _gain_note("past the wall", gains[col])):
            # The note keeps the band right under the incident-wavefront
            # label; the air over its ascenders (which used to break the top
            # spine into pieces from 3 px away) comes from the framing.
            ax_p.text(5.62, 4.28, T(note), ha="right", va="top",
                      color=FIELD_INK, fontsize=6.5, path_effects=outline)
        v_txts.append(ax_r.text(
            3.85, 0.85, verdicts[col], ha="center", va="bottom",
            color="white", fontsize=7.5, zorder=6,
            bbox={"boxstyle": _ANIM_PILL_BOX,
                  "facecolor": "black", "alpha": 0.45,
                  "edgecolor": "none"}))
        if col == 0:
            ax_p.set_ylabel(T("instantaneous $p(x, y)$"), fontsize=9)
            ax_r.set_ylabel(T("RMS level [dB]"), fontsize=9)
            ax_p.text(0.2, 0.85,
                      T(rf"$f$ = {_APERTURE_F:.0f} Hz "
                        rf"($\lambda$ = {lam:.2f} m)"),
                      ha="left", va="bottom", color=FIELD_INK, fontsize=7.5,
                      path_effects=outline)
            tau_txt = ax_r.text(5.55, _APERTURE_YTOP - 0.2, "", ha="right",
                                va="top",
                                fontsize=8.5, color="white", zorder=7,
                                bbox={"boxstyle": _ANIM_PILL_BOX,
                                      "facecolor": "black", "alpha": 0.55,
                                      "edgecolor": "none"})
        else:
            ax_p.tick_params(labelleft=False)
            ax_r.tick_params(labelleft=False)
            ax_r.text(5.55, _APERTURE_YTOP - 0.2, T("same color scale"),
                      color="white", fontsize=7, ha="right", va="top")
        ims += [im_p, im_r]
    t_txt = fig.text(0.012, 0.985, "", ha="left", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)
    # The verdict pills are centred on the aperture, near the right of each
    # RMS panel: in Spanish they carry their last letters -- and the whole
    # translucent pill, which then composes against the white page instead
    # of the field -- past the spine. Measure and slide them back in.
    def fit_verdicts() -> None:
        for v_txt in v_txts:
            _fit_text_x(fig, v_txt.axes, v_txt, *v_txt.axes.get_xlim(),
                        margin=0.12)

    reveal = int(0.5 * p_all.shape[1])

    def shadow_gained(col: int, k: int) -> Any:
        """Frame *k* of panel *col* with the shadow side amplified.

        The gain is applied frame by frame: the field stack is memoised and
        shared by the four language/theme variants, so it must not be
        rescaled in place.
        """
        frame = p_all[col][k]
        if gains[col] <= 1.0:
            return frame
        out = frame.copy()
        out[:, past:] *= gains[col]
        return out

    def update(k: int) -> tuple[Any, ...]:
        for col in range(2):
            ims[2 * col].set_data(shadow_gained(col, k))
            ims[2 * col + 1].set_data(db_all[col][k])
        tau_txt.set_text(
            T(rf"slit $\tau$ = {tau:.2f} (Gomperts)") if k >= reveal else "")
        t_txt.set_text(T(f"$t$ = {times[k] * 1e3:4.1f} ms"))
        return (*ims, tau_txt, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_aperture_slit",
                 frames=int(p_all.shape[1]), fps=_APERTURE_FPS, gif_fps=8,
                 measure=fit_verdicts)
