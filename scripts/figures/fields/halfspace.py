#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Lamb's problem: three speeds, and the surface wave the boundary makes."""

from functools import lru_cache
from typing import Any

import numpy as np

from ..media import _ANIM_PILL_BOX, _anim_figure, _render_clip, _translate_str
from ..theme import (
    CMAP_FIELD,
    COLOR_FG,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    FIELD_INK,
    FIELD_STROKE,
)

# Aluminium half-space, the material of the still figure this clip promotes
# (simulation/elastic-waves.mdx section 3) and of the solver's own Rayleigh
# test: c_P = 6320, c_S = 3130 m/s, rho = 2700 kg/m3.
_HS_CP, _HS_CS, _HS_RHO = 6320.0, 3130.0, 2700.0
# Mesh rule dx = min(smallest geometric dimension / 4, shortest wavelength
# / 8). The block has no feature smaller than itself; the source is a
# Gaussian of width 8 us, whose spectrum is 20 dB down at 60.4 kHz, and the
# slowest wave of interest is the Rayleigh train at 2921 m/s, so
# lambda / 8 = 48 mm / 8 = 6.0 mm. The grid runs at 1 mm, six times finer,
# because a free surface carries the most dispersive part of the scheme and
# the page itself asks for 15-20 cells per Rayleigh wavelength there (48 at
# this dx).
_HS_DX = 0.001
_HS_SRC_W = 8e-6                   # Gaussian source width [s]
# Domain and view. The view is the 0.6 x 0.3 m block of the still figure,
# centred on the impact; the simulated domain is 1.1 x 0.45 m so that the
# sponges, and the corner where a sponge meets the free surface, stay off
# screen. That corner is the one place this solver is weak -- the page says
# so -- because a sponge absorbs a grazing surface wave far less well than
# a body wave. Here the Rayleigh train first reaches a sponge at 148 us of
# the 163 us captured, 0.13 m outside the frame, and anything it sends back
# would need another 45 us to re-enter the view.
_HS_NY, _HS_NX = 450, 1100
_HS_SPONGE = 120
_HS_HALF = 300                     # view half-width and depth [cells]
_HS_FPS = 40


def _rayleigh_speed(c_p: float, c_s: float) -> float:
    """Root of the exact Rayleigh characteristic equation (C&H Eq. 3.149).

    ``16 kR^6 (kT^2 - kL^2) + 8 kR^4 (2 kL^2 kT^2 - 3 kT^4) + 8 kR^2 kT^6
    - kT^8 = 0`` with ``k = omega / c``, solved for the real root just below
    ``c_S``. Closed form, not a simulated quantity: it is what the clip
    measures the simulated surface train against, and the same root the
    solver's own Rayleigh test pins.
    """
    from scipy.optimize import brentq

    kt2, kl2 = 1.0 / c_s**2, 1.0 / c_p**2

    def char(c_r: float) -> float:
        kr2 = 1.0 / c_r**2
        return (16.0 * kr2**3 * (kt2 - kl2)
                + 8.0 * kr2**2 * (2.0 * kl2 * kt2 - 3.0 * kt2**2)
                + 8.0 * kr2 * kt2**3 - kt2**4)

    return float(brentq(char, 0.5 * c_s, 0.999 * c_s, xtol=1e-12))


@lru_cache(maxsize=1)
def _halfspace_fields() -> tuple[Any, Any, Any, Any, Any, float]:
    """The same vertical hit on a free and on a rigid top surface, cached.

    Two identical runs of Lamb's problem -- vertical ForceSource on the top
    row, Gaussian pulse of width 8 us, sponges left, right and bottom --
    differing only in whether the top side is declared ``free``. Free, the
    stress-imaging surface carries a Rayleigh train; rigid, the default
    shear-free clamped wall supports no surface wave at all, which is the
    guide's claim that the boundary condition and not the material is the
    cause.

    Timeline per the animation norms. Flight: the farthest visible point is
    a bottom corner of the view at sqrt(0.30^2 + 0.30^2) = 0.424 m, reached
    by the slowest body wave of interest (S at 3130 m/s) in 135 us, so
    1.2 x 135 = 163 us is captured -- 2.2 times the 73 us of the still
    figure, which stops before the surface train has separated. Sampling:
    dt = 0.6 dx / (c_P sqrt(2)) = 67.13 ns, a stride of 5 gives a frame
    every 0.336 us, i.e. 49.4 frames per period of the 60.4 kHz upper edge
    of the source spectrum (>= the 48 the clips keep), and 484 frames.

    Returns the two decimated ``vy`` stacks, the two surface-probe
    histories, the frame times and the Rayleigh speed measured off the free
    run's probes.
    """
    from phonometry import ElasticFDTD2D, ForceSource, GaussianPulse

    dx = _HS_DX
    dt = 0.6 * dx / (_HS_CP * float(np.sqrt(2.0)))
    corner = float(np.hypot(_HS_HALF * dx, _HS_HALF * dx))
    every = 5
    n_active = int(np.ceil(1.2 * corner / _HS_CS / (every * dt)))
    src_ix = _HS_NX // 2
    # Surface probes 0.15 m apart, both inside the view: the Rayleigh
    # arrival separates between them by 0.15 / c_R = 51.4 us, which is the
    # speed measurement drawn under each panel.
    probe_cols = (src_ix + 150, src_ix + 300)
    rows = slice(0, _HS_HALF, 2)
    cols = slice(src_ix - _HS_HALF, src_ix + _HS_HALF, 2)
    pulse = GaussianPulse(0, 0, width=_HS_SRC_W).value

    stacks: list[Any] = []
    traces: list[Any] = []
    times = np.zeros(0)
    for free in (True, False):
        sim = ElasticFDTD2D(_HS_CP, _HS_CS, dx, rho=_HS_RHO,
                            shape=(_HS_NY, _HS_NX),
                            sponge_width=_HS_SPONGE,
                            sponge_sides=("left", "right", "bottom"),
                            free_sides=("top",) if free else None)
        sim.add_source(ForceSource(ix=src_ix, iy=0, direction="y",
                                   amplitude=1e6, waveform=pulse))
        frames: list[Any] = []
        probe: list[Any] = []
        ts: list[float] = []
        while len(frames) < n_active:
            sim.step()
            if sim.n % every == 0:
                vy = sim.collocated("vy")
                frames.append(vy[rows, cols].astype(np.float32))
                probe.append([float(vy[0, probe_cols[0]]),
                              float(vy[0, probe_cols[1]])])
                ts.append(sim.time)
        stacks.append(np.stack(frames))
        traces.append(np.asarray(probe, dtype=np.float32))
        times = np.asarray(ts)
    # The surface train is the largest thing either probe ever sees on the
    # free run, so the speed is the peak-to-peak time of flight over the
    # 0.15 m between them.
    free_trace = traces[0]
    t1 = times[int(np.abs(free_trace[:, 0]).argmax())]
    t2 = times[int(np.abs(free_trace[:, 1]).argmax())]
    c_meas = float(150 * dx / (t2 - t1))
    scale = float(np.quantile(np.abs(stacks[0]), 0.999))
    return (stacks[0] / scale, stacks[1] / scale,
            traces[0] / scale, traces[1] / scale, times, c_meas)


def animate_elastic_halfspace_waves(output_dir: str) -> None:
    """Lamb's problem in motion, with and without a free surface.

    A vertical hit on the top of an aluminium half-space sends out three
    waves: the compressional front at 6320 m/s, the shear front at
    3130 m/s behind it, and -- only where the surface is free -- a Rayleigh
    train that stays on the surface at 2921 m/s, the root of the exact
    characteristic equation. The still figure this clip sits beside already
    measures the two body speeds, because its dotted arcs are drawn at the
    exact c_P t and c_S t radii; what one frame cannot show is the third
    wave, whose speed is not separable from S in a single picture and whose
    defining property -- it stays on the surface while the body fronts leave
    -- is a statement about persistence. The arcs are kept here and stay
    locked to the fronts as they expand. The lower panel repeats the run with the top side
    left at the solver default, a clamped rigid wall: the same material,
    the same hit, the same two body fronts, and no surface train anywhere
    -- which is what makes the boundary condition, not the aluminium, the
    reason a Rayleigh wave exists.
    """
    from matplotlib import patheffects
    from matplotlib.patches import Circle

    T = _translate_str
    vy_free, vy_rigid, tr_free, tr_rigid, times, c_meas = _halfspace_fields()
    c_r = _rayleigh_speed(_HS_CP, _HS_CS)
    n_active = int(times.size)
    half = _HS_HALF * _HS_DX
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    # Each field panel on its own colour scale, taken off the second half of
    # its own run. Two reasons, and the panels say so. The source blob is
    # orders of magnitude above the fronts it launches, so a scale set on the
    # whole run leaves the wavefronts invisible; and a vertical force *on* a
    # clamped surface nearly cancels against its own image, so the rigid run
    # radiates far less than the free one and a shared scale would bury it.
    # The probe traces below keep one shared scale, which is where the
    # comparison stays quantitative: the missing surface arrival is a missing
    # peak, not a rescaled one.
    late = slice(n_active // 2, n_active)
    vmaxes = [float(np.quantile(np.abs(d[late]), 0.995))
              for d in (vy_free, vy_rigid)]
    t_us = times * 1e6

    fig = _anim_figure()
    # Reserve the bottom strip for the two figure-level captions: fig.text
    # does not claim space from the constrained layout on its own, and
    # without this the material line lands on the lower axis label.
    fig.get_layout_engine().set(rect=(0.0, 0.055, 1.0, 0.945))
    # Short on purpose: the Spanish of a longer title overran the canvas at
    # both ends. What the boundary has to do with it is in the panel titles.
    fig.suptitle(T("Lamb's problem: P, S and the surface wave "
                   "(elastic 2D FDTD)"), fontweight="bold")
    # The field panels are 2:1 and, in a two-row canvas, height-limited: a
    # wider cell than their natural width only pads them with white. The
    # probe traces are time series and use every pixel of width they get,
    # so the split gives the leftover to them.
    grid = fig.add_gridspec(2, 2, width_ratios=(1.75, 1.0))
    titles = [T("Free top surface: a Rayleigh train rides it"),
              T("Rigid top surface: same block, no surface wave")]
    ims: list[Any] = []
    arcs: list[tuple[Any, Any]] = []
    marks: list[Any] = []
    labels: list[tuple[Any, Any]] = []
    lines: list[tuple[Any, Any]] = []
    v_txts: list[Any] = []
    r_labs: list[Any] = []
    for row, (data, trace, title, free) in enumerate(
            ((vy_free, tr_free, titles[0], True),
             (vy_rigid, tr_rigid, titles[1], False))):
        ax = fig.add_subplot(grid[row, 0])
        ax.grid(False)
        im = ax.imshow(data[0], origin="upper",
                       extent=(-half, half, half, 0.0), cmap=CMAP_FIELD,
                       vmin=-vmaxes[row], vmax=vmaxes[row], aspect="equal",
                       interpolation="bilinear")
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.plot([0.0], [0.0], marker="v", ms=6, color=COLOR_SECONDARY,
                markeredgecolor=FIELD_STROKE, markeredgewidth=0.6,
                clip_on=False, zorder=6)
        p_arc = Circle((0.0, 0.0), 0.0, fill=False, ls=":", lw=1.0,
                       edgecolor=FIELD_INK, zorder=4)
        s_arc = Circle((0.0, 0.0), 0.0, fill=False, ls=":", lw=1.0,
                       edgecolor=FIELD_INK, zorder=4)
        ax.add_patch(p_arc)
        ax.add_patch(s_arc)
        p_lab = ax.text(0.0, 0.0, "P", fontsize=8, ha="center", va="center",
                        color=FIELD_INK, path_effects=outline, zorder=5)
        s_lab = ax.text(0.0, 0.0, "S", fontsize=8, ha="center", va="center",
                        color=FIELD_INK, path_effects=outline, zorder=5)
        (mark,) = ax.plot([], [], marker="o", ms=4, ls="none",
                          color=COLOR_PRIMARY,
                          markeredgecolor=FIELD_STROKE, markeredgewidth=0.6,
                          zorder=6)
        # The Rayleigh label rides the right-hand marker instead of sitting
        # over the source, where it collided with the panel title.
        r_lab = ax.text(0.0, 0.018, "R", fontsize=8.5, ha="center",
                        va="top", color=COLOR_PRIMARY, fontweight="bold",
                        path_effects=outline, zorder=7, visible=free)
        r_labs.append(r_lab)
        for col in (0.15, 0.30):
            ax.plot([col], [0.0], marker="|", ms=8, color=COLOR_FG,
                    clip_on=False, zorder=6)
        ax.set_ylabel(T("depth [m]"), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xticks([-0.3, -0.15, 0.0, 0.15, 0.3])
        if row == 1:
            ax.set_xlabel(T("distance from the impact [m]"), fontsize=8)
        else:
            ax.set_xticklabels([])
        v_txt = ax.text(-half + 0.008, half - 0.008, "", ha="left",
                        va="bottom", color="white", fontsize=7.5, zorder=7,
                        bbox={"boxstyle": _ANIM_PILL_BOX,
                              "facecolor": "black", "alpha": 0.55,
                              "edgecolor": "none"})
        ims.append(im)
        arcs.append((p_arc, s_arc))
        labels.append((p_lab, s_lab))
        marks.append(mark)
        v_txts.append(v_txt)

        tax = fig.add_subplot(grid[row, 1])
        tax.grid(False)
        tax.set_xlim(0.0, float(t_us[-1]))
        tax.set_ylim(-1.15, 1.15)
        tax.set_yticks([])
        tax.tick_params(labelsize=7)
        tax.set_title(T("surface $v_y$ at 0.15 and 0.30 m"), fontsize=7.5)
        if row == 1:
            tax.set_xlabel(T("t [µs]"), fontsize=7.5)
        (ln0,) = tax.plot([], [], color=COLOR_PRIMARY, lw=1.0)
        (ln1,) = tax.plot([], [], color=COLOR_SECONDARY, lw=1.0)
        lines.append((ln0, ln1))
    verdicts = [
        T(f"surface train: {c_meas:.0f} m/s measured, {c_r:.0f} m/s exact"),
        T("only P and S: a clamped wall carries no surface wave"),
    ]
    # The symbols are mathtext, which is the house convention and also the
    # fix for a real defect: `_translate_str` rewrites decimal points as
    # commas for Spanish unless the string carries maths, and it had been
    # turning the equation number "3.149" into "3,149".
    speeds = fig.text(0.5, 0.030,
                      T(f"aluminium: $c_P$ = {_HS_CP:.0f}, $c_S$ = "
                        f"{_HS_CS:.0f}, $c_R$ = {c_r:.0f} m/s "
                        f"(C&H Eq. 3.149)"),
                      ha="center", va="bottom", fontsize=8, color=COLOR_FG)
    scale_txt = fig.text(0.5, 0.006,
                         T("each field panel on its own colour scale; the "
                           "probe traces share one"), ha="center",
                         va="bottom", fontsize=7, color=COLOR_FG, alpha=0.85)
    t_txt = fig.text(0.988, 0.030, "", ha="right", va="bottom",
                     family="monospace", fontsize=9, color=COLOR_FG)
    reveal = int(0.88 * n_active)

    def update(k: int) -> tuple[Any, ...]:
        kf = min(k, n_active - 1)
        t = float(times[kf])
        r_p, r_s, r_r = _HS_CP * t, _HS_CS * t, c_r * t
        for row in range(2):
            ims[row].set_data((vy_free, vy_rigid)[row][kf])
            p_arc, s_arc = arcs[row]
            p_arc.set_radius(r_p)
            s_arc.set_radius(r_s)
            for arc, lab, rad in ((p_arc, labels[row][0], r_p),
                                  (s_arc, labels[row][1], r_s)):
                arc.set_visible(rad > 0.01)
                lab.set_visible(0.02 < rad < half * 1.35)
                lab.set_position((rad * 0.72, rad * 0.72))
            if row == 0:
                inside = 0.02 < r_r < half
                marks[row].set_data(([-r_r, r_r] if inside else []),
                                    ([0.0, 0.0] if inside else []))
                r_labs[row].set_visible(inside)
                r_labs[row].set_position((r_r, 0.018))
            trace = (tr_free, tr_rigid)[row]
            for i, line in enumerate(lines[row]):
                line.set_data(t_us[:kf + 1], trace[:kf + 1, i])
            v_txts[row].set_text(verdicts[row] if k >= reveal else "")
        t_txt.set_text(T(f"$t$ = {t * 1e6:5.1f} µs"))
        return (*ims, *marks, *r_labs, *v_txts, speeds, scale_txt, t_txt)

    # The poster is grabbed mid-flight rather than at the end of the hold:
    # by the last frame every wave has left the view and the default poster
    # is an empty field. 95 us is where the Rayleigh train is still inside
    # the right edge (it leaves at 103 us) and both body fronts sit on their
    # arcs. `fields/pillar_hall._poster_ss_for` carries the same number so
    # that `make posters` agrees with a full render.
    poster_frame = 95.0e-6 / (5.0 * 0.6 * _HS_DX
                              / (_HS_CP * float(np.sqrt(2.0))))
    _render_clip(fig, update, output_dir, "anim_elastic_halfspace_waves",
                 frames=n_active + 2 * _HS_FPS, fps=_HS_FPS, gif_fps=6,
                 poster_ss=poster_frame / _HS_FPS)
