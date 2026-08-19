#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The plane-wave limit of a duct: the same run below and above cut-on."""

from functools import lru_cache
from typing import Any

import numpy as np

from ..media import _ANIM_PILL_BOX, _anim_figure, _render_clip, _translate_str
from ..theme import (
    CMAP_FIELD,
    COLOR_FG,
    COLOR_GRID,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
)

# The worked ventilation duct of the duct-path guide (Long, Table 14.9's
# supply trunk class): 0.65 x 0.40 m, whose first higher-order acoustic mode
# cuts on at 263.8 Hz. The clip simulates a 2D slice of the 0.65 m dimension,
# where the (1, 0) mode is exactly the same wave: its cut-on is c / (2 a) =
# 263.8 Hz, the number rectangular_duct_cut_on prints for the real duct, so
# the clip and the guide's cut-on ladder annotate one frequency.
_DUCT_A = 0.65                     # simulated (wider) duct dimension [m]
_DUCT_B = 0.40                     # the other one, quoted for the library call
# Mesh rule dx = min(smallest dimension / 4, shortest wavelength / 8) =
# min(0.65 / 4 = 163 mm, 343 / 400 / 8 = 107 mm). The grid runs 21 times
# finer than that bound so the cross-section profile is a smooth curve
# rather than a staircase, at a cost the domain's thinness makes trivial.
_DUCT_DX = 0.005
_DUCT_VIEW = 8.0                   # duct length shown [m]
# Simulated length. Whatever comes back off the far end has crossed the
# 0.625 m sponge twice, and it can first re-enter the shown 8 m at
# (12.5 - 0.15 + 12.5 - 8) / 343 = 49.1 ms of the 66.5 ms captured; what
# survives that is inside the 0.4 % the settled section spread is measured
# at. A rho c edge was the obvious alternative and is wrong here: it is
# anechoic only at normal incidence, and above cut-on the (1, 0) mode
# arrives at an angle (k_x / k = 0.752 at 400 Hz), which a locally reacting
# termination reflects at -17 dB.
_DUCT_LEN = 12.5
_DUCT_SPONGE = 125                 # 0.625 m, the widest this section allows
_DUCT_SRC_X = 0.15                 # source 0.15 m off the closed end
_DUCT_SRC_YF = 0.20                # and 0.20 a off the wall, so it is not
#                                    on a node of either mode
_DUCT_STATION = 5.0                # cross-section station [m]
_DUCT_FREQS = (180.0, 400.0)       # below and above the 263.8 Hz cut-on
# Source onset [s], the same in both panels rather than the same number of
# periods. A short ramp is a broadband event: with the 1.5-period default
# the 180 Hz onset splashes energy above 263.8 Hz, that content propagates
# in the (1, 0) mode at a group velocity that vanishes at cut-on, and a
# slow lumpy packet is still crawling past the station at the end of the
# window -- 6.9 % of the mean-square pressure there, and a visible tilt
# across the section whenever the plane mode passes through zero. Ramping
# over 30 ms instead keeps the onset sidebands inside the plane-wave band
# and takes that to 0.4 % (measured, same scene, same station).
_DUCT_RAMP_S = 0.030
# Vertical exaggeration of the field strips. The duct is 8 m long and
# 0.65 m across, so drawn to scale each strip would be a hairline; every
# panel says the factor out loud.
_DUCT_ASPECT = 3.0
_DUCT_EVERY = 16                   # capture stride [solver steps]
_DUCT_FPS = 50


@lru_cache(maxsize=1)
def _duct_cut_on_fields() -> tuple[Any, Any, Any, Any, Any, tuple[float, float]]:
    """Two CW runs down the same rigid duct, below and above cut-on.

    Identical scenes but for the drive frequency: a monopole 0.15 m off the
    closed end and 0.13 m off the wall, which excites the plane mode and the
    (1, 0) mode together (neither is on a node there). Below cut-on the
    transverse mode is evanescent, exp(-alpha x) with 1 / alpha = 0.28 m at
    180 Hz, i.e. 20 dB down one duct width from the source; above it the
    two modes propagate with different axial wavenumbers (k_x = 5.51 against
    k = 7.33 rad/m at 400 Hz) and beat every 3.45 m.

    Timeline per the animation norms. The capture stride is 16 solver steps
    = 99 us, i.e. 25.3 frames per period of the 400 Hz carrier (twice the 12
    the norms require; 56 at 180 Hz), which at 50 fps puts one carrier cycle
    at half a second of screen time -- fast enough to read as a wave, slow
    enough to follow. The window is the 30 ms onset ramp plus 1.2 x the
    flight from the source to the far end of the shown duct at the slowest
    speed on the path, which above cut-on is the modal group velocity
    c sqrt(1 - (f_co / f)^2) = 257.8 m/s at 400 Hz, not c:
    7.85 / 257.8 = 30.5 ms, so 30 + 36.5 = 66.5 ms and 672 frames.

    Returns the two decimated frame stacks, the two full-resolution
    cross-section histories at the station, the frame times and the
    off-plane-mode fraction measured at the station in each run.
    """
    import fdtd2d

    dx = _DUCT_DX
    ny = round(_DUCT_A / dx)
    nx = round(_DUCT_LEN / dx)
    src_ix = round(_DUCT_SRC_X / dx)
    src_iy = round(_DUCT_SRC_YF * ny)
    st_ix = round(_DUCT_STATION / dx)
    c0 = 343.0
    f_co = c0 / (2.0 * _DUCT_A)
    every = _DUCT_EVERY
    dt = 0.6 * dx / (c0 * float(np.sqrt(2.0)))
    ramp = _DUCT_RAMP_S
    c_group = c0 * float(np.sqrt(1.0 - (f_co / max(_DUCT_FREQS)) ** 2))
    span = ramp + 1.2 * (_DUCT_VIEW - _DUCT_SRC_X) / c_group
    n_active = int(np.ceil(span / (every * dt)))
    view_cols = slice(0, round(_DUCT_VIEW / dx), 4)

    stacks: list[Any] = []
    stations: list[Any] = []
    sigmas: list[float] = []
    times = np.zeros(0)
    for f in _DUCT_FREQS:
        sim = fdtd2d.FDTD2D(c0, dx, shape=(ny, nx),
                            sponge_width=_DUCT_SPONGE,
                            sponge_sides=("right",))
        sim.add_source(fdtd2d.CWSource(src_ix, src_iy, frequency=f,
                                       ramp_cycles=_DUCT_RAMP_S * f))
        frames: list[Any] = []
        column: list[Any] = []
        ts: list[float] = []
        while len(frames) < n_active:
            sim.step()
            if sim.n % every == 0:
                frames.append(sim.p[::2, view_cols].astype(np.float32))
                column.append(sim.p[:, st_ix].astype(np.float32))
                ts.append(sim.time)
        stack = np.stack(frames)
        col = np.stack(column)
        # Normalise each panel to its own plane-mode amplitude at the
        # station, measured over the settled last quarter of the window:
        # the two runs are driven by the same source strength but a duct
        # radiates them differently, and the clip compares shapes, not
        # levels. The plane-mode part is the section average, which is
        # exactly the single pressure a one-dimensional model carries.
        tail = col[3 * n_active // 4:]
        mean = tail.mean(axis=1)
        p_ref = float(np.sqrt(2.0 * np.mean(mean**2)))
        rest = tail - mean[:, None]
        sigmas.append(float(np.sqrt(np.mean(rest**2) / np.mean(tail**2))))
        stacks.append(stack / p_ref)
        stations.append(col / p_ref)
        times = np.asarray(ts)
    return (stacks[0], stacks[1], stations[0], stations[1], times,
            (sigmas[0], sigmas[1]))


def animate_fdtd_duct_cut_on(output_dir: str) -> None:
    """The plane-wave limit of a duct, on both sides of it (2D FDTD).

    One rigid 0.65 m duct, one off-axis monopole, two drive frequencies:
    180 Hz, below the 263.8 Hz cut-on of the first higher-order mode, and
    400 Hz above it. Below cut-on the transverse mode is evanescent, the
    lumpy near field dies within half a duct width and every later
    wavefront crosses the section flat -- one sound pressure really does
    describe the whole cross section, which is what every one-dimensional
    element and four-pole silencer model on the duct-path page assumes.
    Above cut-on the transverse mode propagates, the two modes beat every
    3.45 m and the profile across the section never settles. The
    cross-section panel beside each strip draws the instantaneous profile
    against the section average the 1D model would use.
    """
    p_lo, p_hi, col_lo, col_hi, times, sigmas = _duct_cut_on_fields()
    T = _translate_str
    from phonometry import noise_control

    modes = noise_control.rectangular_duct_cut_on(_DUCT_A, _DUCT_B, count=2)
    f_co = float(modes.cut_on[0])
    n_active = int(times.size)
    ny = round(_DUCT_A / _DUCT_DX)
    y_col = (np.arange(ny) + 0.5) * _DUCT_DX
    # Each strip on its own colour scale, taken off its own settled field
    # clear of the source's near field (the first metre). The two runs
    # carry very different amplitudes for the same drive: at 400 Hz the
    # off-axis monopole puts more than twice as much amplitude into the
    # (1, 0) mode as into the plane one, so a shared ramp would either
    # bleach the 180 Hz strip or saturate the 400 Hz one. The profile
    # panels beside them do carry one shared scale, and that is where the
    # comparison is quantitative.
    skip = round(0.75 / (4.0 * _DUCT_DX))
    vmaxes = [1.12 * float(np.quantile(np.abs(p[:, :, skip:]), 0.999))
              for p in (p_lo, p_hi)]
    p_lim = 1.05 * max(vmaxes)

    fig = _anim_figure()
    # Reserve the bottom strip for the two-line footer: fig.text does not
    # claim space from the constrained layout on its own.
    fig.get_layout_engine().set(rect=(0.0, 0.075, 1.0, 0.925))
    fig.suptitle(T("Duct cut-on: is one pressure enough? (2D FDTD)"),
                 )
    grid = fig.add_gridspec(2, 2, width_ratios=(6.4, 1.15))
    titles = [T(f"{_DUCT_FREQS[0]:.0f} Hz: below the {f_co:.1f} Hz cut-on"),
              T(f"{_DUCT_FREQS[1]:.0f} Hz: above it")]
    verdicts = [
        T(f"one pressure, to within {100 * sigmas[0]:.1f} %"),
        T(f"{100 * sigmas[1]:.0f} % of the section's energy is not "
          f"the plane mode"),
    ]
    ims: list[Any] = []
    profs: list[Any] = []
    means: list[Any] = []
    v_txts: list[Any] = []
    for row, (data, title, vmax) in enumerate(
            ((p_lo, titles[0], vmaxes[0]), (p_hi, titles[1], vmaxes[1]))):
        ax = fig.add_subplot(grid[row, 0])
        ax.grid(False)
        im = ax.imshow(data[0], origin="lower",
                       extent=(0.0, _DUCT_VIEW, 0.0, _DUCT_A),
                       cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
                       aspect=_DUCT_ASPECT, interpolation="bilinear")
        ax.set_title(title, fontsize=10)
        ax.axvline(_DUCT_STATION, color=COLOR_FG, lw=1.0, ls=":", zorder=4)
        ax.plot([_DUCT_SRC_X], [_DUCT_SRC_YF * _DUCT_A], marker="o", ms=4,
                color=COLOR_TERTIARY, markeredgecolor="white",
                markeredgewidth=0.6, zorder=5)
        ax.set_yticks([0.0, _DUCT_A])
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.text(_DUCT_SRC_X + 0.08, _DUCT_SRC_YF * _DUCT_A,
                    T("source"), fontsize=7, ha="left", va="center",
                    color="white", zorder=5,
                    bbox={"boxstyle": _ANIM_PILL_BOX, "facecolor": "black",
                          "alpha": 0.5, "edgecolor": "none"})
            # Above the station line, not beside its foot: the verdict
            # pill lives at the foot and the Spanish of it is long.
            ax.text(_DUCT_STATION - 0.08, _DUCT_A - 0.03,
                    T(f"section at $x$ = {_DUCT_STATION:.0f} m"),
                    fontsize=7, ha="right", va="top", color="white",
                    zorder=5,
                    bbox={"boxstyle": _ANIM_PILL_BOX, "facecolor": "black",
                          "alpha": 0.5, "edgecolor": "none"})
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(T("Distance along the duct [m]"), fontsize=9)
        v_txt = ax.text(_DUCT_VIEW - 0.06, 0.04, "", ha="right", va="bottom",
                        color="white", fontsize=8, zorder=5,
                        bbox={"boxstyle": _ANIM_PILL_BOX,
                              "facecolor": "black", "alpha": 0.55,
                              "edgecolor": "none"})
        ims.append(im)
        v_txts.append(v_txt)

        pax = fig.add_subplot(grid[row, 1])
        pax.grid(False)
        pax.set_xlim(-p_lim, p_lim)
        pax.set_ylim(0.0, _DUCT_A)
        pax.set_yticks([])
        pax.set_xticks([-4.0, -2.0, 0.0, 2.0, 4.0])
        pax.tick_params(labelsize=7)
        pax.axvline(0.0, color=COLOR_GRID, lw=0.8, zorder=1)
        mean_ln = pax.axvline(0.0, color=COLOR_SECONDARY, lw=2.6, ls="--",
                              zorder=2)
        (prof,) = pax.plot(np.zeros(ny), y_col, color=COLOR_FG, lw=1.3,
                           zorder=3)
        if row == 0:
            pax.set_title(T("$p$ across the section"), fontsize=7.5)
            pax.set_xticklabels([])
        else:
            pax.set_xlabel(T("$p$ / plane mode"), fontsize=7)
        profs.append(prof)
        means.append(mean_ln)
    fc_txt = fig.text(0.5, 0.038,
                      T(f"rectangular_duct_cut_on: first cut-on {f_co:.1f} Hz "
                        f"({_DUCT_A:.2f} × {_DUCT_B:.2f} m duct)"),
                      ha="center", va="bottom", fontsize=7.5, color=COLOR_FG)
    scale_txt = fig.text(0.5, 0.008,
                         T(f"dashed: the section average · vertical scale "
                           f"×{_DUCT_ASPECT:.0f} · each strip on its own "
                           f"colour scale"), ha="center",
                         va="bottom", fontsize=7, color=COLOR_FG, alpha=0.85)
    t_txt = fig.text(0.012, 0.982, "", ha="left", va="top",
                     family="monospace", fontsize=9, color=COLOR_FG)
    reveal = int(0.80 * n_active)

    def update(k: int) -> tuple[Any, ...]:
        kf = min(k, n_active - 1)
        for i, (field, col) in enumerate(((p_lo, col_lo), (p_hi, col_hi))):
            ims[i].set_data(field[kf])
            profs[i].set_xdata(col[kf])
            means[i].set_xdata([float(col[kf].mean())] * 2)
            v_txts[i].set_text(verdicts[i] if k >= reveal else "")
        t_txt.set_text(T(f"$t$ = {times[kf] * 1e3:5.1f} ms"))
        return (*ims, *profs, *means, *v_txts, fc_txt, scale_txt, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_duct_cut_on",
                 frames=n_active + 2 * _DUCT_FPS, fps=_DUCT_FPS, gif_fps=6)
