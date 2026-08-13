#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The 10 mm steel plate: its L-junction and its coincidence dip.

Both clips run the library's 2D P-SV solver on the same plate, so they share
the body-wave speeds, the density, the mesh and the bending stiffness
derived from them.
"""

from functools import lru_cache
from typing import Any

import numpy as np

from ..i18n import _fmt_minus
from ..media import (
    _ANIM_HOLD,
    _ANIM_PILL_BOX,
    _anim_figure,
    _render_clip,
    _translate_str,
)
from ..theme import (
    CMAP_FIELD,
    COLOR_FG,
    COLOR_GRID,
    COLOR_TERTIARY,
    FIELD_INK,
    FIELD_STROKE,
)
from ._core import _fit_text_x, _gain_note, _weak_field_gain

# --- Elastic FDTD clips: plate junction and coincidence --------------------
# Both clips run the library's 2D P-SV solver (phonometry.simulation
# .elastic_fdtd) on the same 10 mm steel plate: body-wave speeds and density
# below give the plane-strain plate modulus E' = 4 mu (lambda + mu) /
# (lambda + 2 mu) that the solver itself propagates, so every derived number
# (bending speed, coincidence frequency) matches the simulated field.
_EL_CP, _EL_CS, _EL_RHO = 5900.0, 3200.0, 7850.0
_EL_H = 0.010
# Mesh rule: dx = min(smallest geometric dimension / 4, shortest relevant
# wavelength / 8). The 10 mm plate thickness governs both clips
# (h / 4 = 2.5 mm), well below lambda_B / 8 (19.5 mm at the 4 kHz junction
# carrier, 25 mm at 2 f_c in the coincidence clip) and lambda_air / 8
# (17.8 mm at 2 f_c), and it puts the required 4 cells across the plate.
_EL_DX = _EL_H / 4.0
#: Solver time step at the default CFL 0.6: dt = cfl dx / (c_p_max sqrt(2)).
_EL_DT = 0.6 * _EL_DX / (_EL_CP * float(np.sqrt(2.0)))
_EL_C0, _EL_RHO0 = 343.0, 1.205


def _elastic_plate_bp_m2() -> tuple[float, float]:
    """(B', m'') of the 10 mm steel plate from the solver's own moduli."""
    mu = _EL_RHO * _EL_CS**2
    lam = _EL_RHO * (_EL_CP**2 - 2.0 * _EL_CS**2)
    e_apparent = 4.0 * mu * (lam + mu) / (lam + 2.0 * mu)
    return e_apparent * _EL_H**3 / 12.0, _EL_RHO * _EL_H


# Junction clip geometry [m]: a semi-infinite horizontal plate (it enters
# through the left sponge and, in the control panel, leaves through the
# right one) carrying a 4 kHz bending packet from x = 0.45 into an
# L-junction at x = 1.05 with a 0.35 m plate hanging down from it.
_EJ_F0 = 4000.0
_EJ_SRC_X, _EJ_JUNC_X = 0.45, 1.05
_EJ_PLATE_Y = 0.36
_EJ_VERT_LEN = 0.35
_EJ_SPONGE = 120                      # 0.30 m absorbing skirt on every side
_EJ_NY, _EJ_NX = 440, 680             # 1.10 m x 1.70 m
_EJ_VIEW = (0.31, 1.39, 0.31, 0.78)   # x0, x1, y0, y1 kept in frame


@lru_cache(maxsize=1)
def _plate_junction_fields() -> tuple[Any, Any, Any, Any, int]:
    """|v| frame stacks of the straight-plate and L-junction runs, cached.

    Two identical elastic runs except for the geometry past x = 1.05: the
    control plate continues into the right sponge (an infinite plate), the
    other stops against a perpendicular plate of the same 10 mm steel. A
    vertical tone-burst force on the surface launches the A0 bending
    packet (Gaussian envelope, sigma = T/2, t0 = 4 sigma). Besides the
    fields, the mid-plane velocity profiles of each plate (vy along the
    horizontal plates, vx down the vertical one) are recorded per frame,
    to be drawn as exaggerated deflection lines over the thin plates.

    Timeline per the animation norms: the frame step is 28 solver steps =
    5.03 us, i.e. 49.7 frames per 4 kHz carrier period (>= the 48 = 4 x
    12-frame visual minimum), and the active span is the source build-up
    t0 plus 1.2 x the full flight tour [source -> junction -> free end ->
    junction -> source] = 2 x (0.60 + 0.35) = 1.90 m at the bending group
    speed c_g = 2 c_B(4 kHz) = 1251 m/s: t = 0.5 ms + 1.2 x 1.52 ms =
    2.32 ms -> 462 active frames.
    """
    from phonometry import ElasticFDTD2D, ForceSource

    dx = _EL_DX
    bp, m2 = _elastic_plate_bp_m2()
    c_b = float((bp / m2) ** 0.25 * np.sqrt(2.0 * np.pi * _EJ_F0))
    c_g = 2.0 * c_b
    sigma_t = 0.5 / _EJ_F0
    t0 = 4.0 * sigma_t
    tour = 2.0 * ((_EJ_JUNC_X - _EJ_SRC_X) + _EJ_VERT_LEN)
    every = 28
    n_active = int(np.ceil((t0 + 1.2 * tour / c_g) / (every * _EL_DT)))

    r0 = round(_EJ_PLATE_Y / dx)                    # plate rows r0..r0+3
    j0 = round(_EJ_JUNC_X / dx)                     # junction columns
    r_end = r0 + round((_EL_H + _EJ_VERT_LEN) / dx)

    def burst(t: float) -> float:
        return float(np.exp(-(((t - t0) / sigma_t) ** 2))
                     * np.sin(2.0 * np.pi * _EJ_F0 * (t - t0)))

    x0, x1, y0, y1 = _EJ_VIEW
    rows = slice(round(y0 / dx), round(y1 / dx), 2)
    cols = slice(round(x0 / dx), round(x1 / dx), 2)
    runs = []
    for with_junction in (False, True):
        c_p = np.full((_EJ_NY, _EJ_NX), _EL_C0)
        c_s = np.zeros((_EJ_NY, _EJ_NX))
        rho = np.full((_EJ_NY, _EJ_NX), _EL_RHO0)
        h_cols = slice(0, j0 + 4) if with_junction else slice(0, _EJ_NX)
        c_p[r0:r0 + 4, h_cols] = _EL_CP
        c_s[r0:r0 + 4, h_cols] = _EL_CS
        rho[r0:r0 + 4, h_cols] = _EL_RHO
        if with_junction:
            c_p[r0:r_end, j0:j0 + 4] = _EL_CP
            c_s[r0:r_end, j0:j0 + 4] = _EL_CS
            rho[r0:r_end, j0:j0 + 4] = _EL_RHO
        sim = ElasticFDTD2D(c_p, c_s, dx, rho=rho,
                            sponge_width=_EJ_SPONGE)
        # Vertical force on the top surface (the vy face between the air
        # row r0 - 1 and the first plate row): Lamb-style A0 launcher.
        sim.add_source(ForceSource(ix=round(_EJ_SRC_X / dx), iy=r0 - 1,
                                   direction="y", amplitude=1e4,
                                   waveform=burst))
        frames: list[Any] = []
        hl: list[Any] = []
        vl: list[Any] = []
        ts: list[float] = []
        for _ in range(n_active):
            for _s in range(every):
                sim.step()
            v = np.hypot(sim.collocated("vx"), sim.collocated("vy"))
            frames.append(v[rows, cols].astype(np.float32))
            # Mid-plane plate velocities: the vy face row and (junction
            # run) the vx face column through the middle of each plate.
            hl.append(sim.vy[r0 + 1, :].astype(np.float32))
            vl.append(sim.vx[:, j0 + 1].astype(np.float32))
            ts.append(sim.time)
        runs.append((np.stack(frames), np.stack(hl), np.stack(vl)))
    lines = (runs[0][1], runs[1][1], runs[1][2])
    return runs[0][0], runs[1][0], lines, np.asarray(ts), n_active


def animate_elastic_plate_junction(output_dir: str) -> None:
    """A 4 kHz bending packet on a 10 mm steel plate (elastic 2D FDTD):
    on the straight control plate it just runs on; at an L-junction with an
    identical perpendicular plate it splits into a reflected and a
    transmitted bending wave plus a fast in-plane precursor, and the
    verdict pill quotes the closed-form junction_transmission coefficient
    the guide computes for this corner."""
    from matplotlib.patches import Rectangle

    from phonometry import junction_transmission

    T = _translate_str
    v_ctrl, v_junc, lines, times, n_active = _plate_junction_fields()
    hl_ctrl, hl_junc, vl_junc = lines
    bp, m2 = _elastic_plate_bp_m2()
    c_pl = float(np.sqrt(12.0 * bp / (m2 * _EL_H**2)))
    res = junction_transmission("L", _EL_H, c_pl, m2, _EL_H, c_pl, m2)
    tau0 = float(res.corner[0])
    kij = float(res.corner_reduction_index)
    x0, x1, y0, y1 = _EJ_VIEW
    vmax = float(np.quantile(v_junc, 0.9995))
    # Exaggerated plate-deflection overlays: the plate is only 10 mm
    # thick on screen, so the signed mid-plane velocity is drawn as a
    # deflected line (a fixed gain shared by all three lines).
    gain = 0.045 / float(np.quantile(np.abs(hl_junc), 0.999))
    dx = _EL_DX
    j0 = round(_EJ_JUNC_X / dx)
    r0 = round(_EJ_PLATE_Y / dx)
    r_end = r0 + round((_EL_H + _EJ_VERT_LEN) / dx)
    c0, c1 = round(x0 / dx), round(x1 / dx)
    x_h = (np.arange(c0, c1) + 0.5) * dx           # vy faces, cropped
    x_hj = (np.arange(c0, j0 + 4) + 0.5) * dx      # up to the junction
    y_v = (np.arange(r0, r_end) + 0.5) * dx        # vx faces, vertical
    y_mid = _EJ_PLATE_Y + 0.5 * _EL_H
    x_mid = _EJ_JUNC_X + 2.0 * dx

    fig = _anim_figure()
    fig.suptitle(T("Bending waves at an L-junction (elastic 2D FDTD)"),
                 fontweight="bold")
    axes = fig.subplots(2, 1, sharex=True)
    titles = [T("Straight plate: the packet just runs on"),
              T("L-junction: reflected, transmitted and mode-converted")]
    verdicts = [T("nothing comes back"),
                T(f"junction_transmission('L'): τ(0°) = {tau0:.2f}, "
                  f"K12 = {kij:.1f} dB")]
    ims: list[Any] = []
    v_txts: list[Any] = []
    ln_arts: list[Any] = []
    for ax, title, data, with_junction in (
            (axes[0], titles[0], v_ctrl, False),
            (axes[1], titles[1], v_junc, True)):
        ax.grid(False)
        im = ax.imshow(data[0], origin="upper", extent=(x0, x1, y1, y0),
                       cmap="magma", vmin=0.0, vmax=vmax,
                       aspect="equal", interpolation="bilinear")
        ax.set_title(title, fontsize=10, fontweight="bold")
        # The 10 mm plates, drawn as thin open rectangles so the field
        # inside stays visible.
        px1 = _EJ_JUNC_X + 4 * _EL_DX if with_junction else x1
        ax.add_patch(Rectangle((x0, _EJ_PLATE_Y), px1 - x0, _EL_H,
                               facecolor="none", edgecolor=COLOR_FG,
                               lw=0.7, alpha=0.8))
        if with_junction:
            y_end = _EJ_PLATE_Y + _EL_H + _EJ_VERT_LEN
            ax.add_patch(Rectangle((_EJ_JUNC_X, _EJ_PLATE_Y),
                                   4 * _EL_DX, y_end - _EJ_PLATE_Y,
                                   facecolor="none", edgecolor=COLOR_FG,
                                   lw=0.7, alpha=0.8))
            # The same dark pill the clip's other labels carry: white on
            # its own over magma, this one washed out to ~1.2:1 every time
            # the bright spot reached the free end, which is exactly when
            # the label matters.
            ax.text(_EJ_JUNC_X - 0.055, y_end - 0.005, T("free end"),
                    ha="right", va="bottom", color="white", fontsize=7,
                    zorder=5,
                    bbox={"boxstyle": _ANIM_PILL_BOX, "facecolor": "black",
                          "alpha": 0.5, "edgecolor": "none"})
        ax.plot([_EJ_SRC_X], [_EJ_PLATE_Y - 0.008], marker="v", ms=5,
                color=COLOR_TERTIARY, markeredgecolor="white",
                markeredgewidth=0.6, zorder=4)
        if not with_junction:
            ax.text(x0 + 0.015, y1 - 0.02, T("4 kHz tone burst at ▼"),
                    ha="left", va="bottom", color="white", fontsize=7.5,
                    zorder=5,
                    bbox={"boxstyle": _ANIM_PILL_BOX, "facecolor": "black",
                          "alpha": 0.5, "edgecolor": "none"})
        # The exaggerated deflection lines riding on the plates.
        if with_junction:
            (ln_h,) = ax.plot(x_hj, np.full(x_hj.size, y_mid), color="white",
                              lw=1.1, alpha=0.85, zorder=3.5)
            (ln_v,) = ax.plot(np.full(y_v.size, x_mid), y_v, color="white",
                              lw=1.1, alpha=0.85, zorder=3.5)
            ln_arts += [ln_h, ln_v]
        else:
            (ln_h,) = ax.plot(x_h, np.full(x_h.size, y_mid), color="white",
                              lw=1.1, alpha=0.85, zorder=3.5)
            ln_arts.append(ln_h)
        ax.set_ylim(y1, y0)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        # Written out here so the clamp below measures the pill that will
        # actually be drawn; update() blanks it until the reveal.
        v_txt = ax.text(x1 - 0.015, y1 - 0.02,
                        verdicts[1 if with_junction else 0], ha="right",
                        va="bottom", color="white", fontsize=8, zorder=5,
                        bbox={"boxstyle": _ANIM_PILL_BOX,
                              "facecolor": "black", "alpha": 0.5,
                              "edgecolor": "none"})
        ims.append(im)
        v_txts.append(v_txt)
    axes[1].set_xlabel("x [m]", fontsize=8)
    t_txt = fig.text(0.985, 0.965, "", ha="right", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)
    # aspect="equal" makes the axes box narrower than its gridspec cell, so
    # a pill anchored a fixed distance inside the data limits still hung its
    # fill out over the page. Slide it back against the measured box.
    for v_txt in v_txts:
        _fit_text_x(fig, v_txt.axes, v_txt, x0, x1, margin=0.010)
        v_txt.set_text("")
    reveal = int(0.62 * n_active)

    def update(k: int) -> tuple[Any, ...]:
        kf = min(k, n_active - 1)
        ims[0].set_data(v_ctrl[kf])
        ims[1].set_data(v_junc[kf])
        ln_arts[0].set_ydata(y_mid + gain * hl_ctrl[kf][c0:c1])
        ln_arts[1].set_ydata(y_mid + gain * hl_junc[kf][c0:j0 + 4])
        ln_arts[2].set_xdata(x_mid + gain * vl_junc[kf][r0:r_end])
        for row, v_txt in enumerate(v_txts):
            v_txt.set_text(verdicts[row] if k >= reveal else "")
        t_txt.set_text(T(f"t = {times[kf] * 1e3:5.2f} ms"))
        return (*ims, *ln_arts, *v_txts, t_txt)

    _render_clip(fig, update, output_dir, "anim_elastic_plate_junction",
                 frames=n_active + _ANIM_HOLD, gif_fps=6)


# Coincidence clip geometry [m]: the same 10 mm steel plate lying flat at
# y = 0.65 across the whole domain (its ends die in the side sponges: an
# infinite plate), air above and below, driven from a phase-graded source
# line at y = 0.32 that radiates a sustained plane wave 45 degrees down.
# The two panels differ only in frequency: f_c / 2 and 2 f_c, with
# sin(theta) = sqrt(f_c / f) putting the exact trace match of the second
# panel at 45 degrees.
_EC_PLATE_Y = 0.65
_EC_SRC_Y = 0.32
_EC_THETA = 45.0
_EC_SPONGE = 120
_EC_NY, _EC_NX = 564, 1036            # 1.41 m x 2.59 m
# The view starts at x = 1.06 so that even its bottom-left corner sits
# inside the steady 45-degree beam of the source aperture (the stationary
# source point of a view point (x, y) is at x - (y - y_src) tan(theta),
# which must stay right of the sponge edge at x = 0.30).
_EC_VIEW = (1.06, 2.06, 0.345, 1.02)
# Numerical-dispersion compensation of the simulated plate. The trace-match
# resonance of a steel plate in air is razor sharp (its scale is
# 2 rho0 c0 / (omega m'' cos theta) ~ 1e-3 here), while the 4-cell discrete
# plate propagates A0 a few percent slower than the Kirchhoff continuum
# (thickness shear plus grid dispersion): driven at the continuum 2 f_c the
# discrete plate would sit far off its own coincidence and transmit nothing.
# The plate's body-wave speeds are therefore raised by the factor measured
# with the cross-spectrum phase-speed rig of the flexural-dispersion test
# (same dx, same 4-cell immersed plate: c_A0 = 463.1 m/s at 2 f_c against
# the 485.1 m/s of the continuum Kirchhoff plate; c_B scales as
# sqrt(c_pl), so c_pl needs (485.1 / 463.1)**2 = 1.0971, re-verified with
# the same rig on the compensated material). This puts the discrete
# plate's bending wavelength at the 2 f_c carrier back on the continuum
# value that every displayed library number describes; the below-f_c mass
# law is untouched by it (that only sees m'').
_EC_CB_TUNE = 1.0971
# The transmitted wave stays faint on the incident colour scale even at
# coincidence: air drives steel so weakly that the resonant bending wave
# needs a radiation build-up length of c_g m'' cos(theta) / (2 rho0 c0)
# ~ 65 m, so over the ~2 m lit span the coincidence beam only climbs to
# ~13 dB above the mass law (growing visibly along x). A steady-scene scan
# of the tune factor (+-4 % around the value below, same domain and
# source) moves the under-plate level by < 1.2 dB, confirming the level is
# this aperture-limited build-up and not residual detuning. Both panels
# therefore draw the air below the plate with the display gain
# :func:`_weak_field_gain` measures off the settled field -- one gain for
# the two, since the clip's whole point is that they transmit the same
# level -- and print it.
# Colour half-range of the instantaneous panels, on the normalisation that
# puts the incident wave at unit amplitude: 1.6 leaves the standing wave
# above the plate (incident plus near-total reflection) room to breathe.
_EC_VLIM = 1.6


@lru_cache(maxsize=1)
def _coincidence_fields() -> tuple[Any, Any, Any, list[float], int]:
    """The below-f_c and above-f_c oblique plane-wave runs, cached.

    1D pre-validation (same dx, same 4-cell immersed plate, normal
    incidence): the measured transmission of a narrowband packet follows
    the library mass_law_transmission_loss within 0.07 dB at f_c / 2 and
    0.17 dB at 2 f_c (narrowband spectral amplitudes at the carrier;
    peak-amplitude estimates stay within 0.5 dB, limited by the packet's
    finite bandwidth), so the air-steel contrast is trusted at the -48 dB
    transmission this clip displays.

    Timeline per the animation norms: the frame step is 52 solver steps =
    8.52 us, i.e. 48.6 frames per period of the 2 f_c carrier (>= the 4 x
    12-frame visual minimum; four times that at f_c / 2), and the active
    span is the 0.62 ms source ramp plus 1.2 x the flight from the source
    line to the farthest visible point at the bottom of the view,
    (1.02 - 0.321) m / (c0 cos 45) = 2.88 ms -> 4.08 ms -> 479 active
    frames. Returns the two normalized frame stacks (incident amplitude
    = 1), the frame times, the measured transmitted levels [dB re
    incident] and the active frame count.
    """
    from phonometry import ElasticFDTD2D, coincidence_frequency

    dx = _EL_DX
    bp, m2 = _elastic_plate_bp_m2()
    fc = coincidence_frequency(m2, bp)
    freqs = (0.5 * fc, 2.0 * fc)
    theta = np.radians(_EC_THETA)
    r0 = round(_EC_PLATE_Y / dx)
    src_row = round(_EC_SRC_Y / dx)
    y_src = (src_row + 0.5) * dx
    # The compensated plate raises c_p_max and with it the solver step, so
    # the 48-frames-per-period pacing is derived from the actual dt.
    cp_plate = _EL_CP * _EC_CB_TUNE
    dt = 0.6 * dx / (cp_plate * float(np.sqrt(2.0)))
    every = int((1.0 / freqs[1]) / (48.0 * dt))
    ramp_t = 1.5 / freqs[1]
    flight = (_EC_VIEW[3] - y_src) / (_EL_C0 * float(np.cos(theta)))
    n_active = int(np.ceil((ramp_t + 1.2 * flight) / (every * dt)))
    steps = n_active * every

    src_cols = np.arange(_EC_SPONGE, _EC_NX - _EC_SPONGE)
    x_src = (src_cols + 0.5) * dx
    x0, x1, y0, y1 = _EC_VIEW
    rows = slice(round(y0 / dx), round(y1 / dx), 2)
    cols = slice(round(x0 / dx), round(x1 / dx), 2)
    # Incident-amplitude probe: reached by the direct wave at ~0.74 ms,
    # by the first plate reflection at ~1.98 ms.
    p_row, p_col = round(0.50 / dx), round(1.65 / dx)
    w0, w1 = 1.05e-3, 1.90e-3
    # Transmitted-level box under the plate, averaged over the last
    # quarter of the active frames (the settled state).
    b_rows = slice(round(0.75 / dx), round(0.95 / dx), 2)

    stacks = []
    trans_db: list[float] = []
    for f0 in freqs:
        c_p = np.full((_EC_NY, _EC_NX), _EL_C0)
        c_s = np.zeros((_EC_NY, _EC_NX))
        rho = np.full((_EC_NY, _EC_NX), _EL_RHO0)
        c_p[r0:r0 + 4] = _EL_CP * _EC_CB_TUNE
        c_s[r0:r0 + 4] = _EL_CS * _EC_CB_TUNE
        rho[r0:r0 + 4] = _EL_RHO
        sim = ElasticFDTD2D(c_p, c_s, dx, rho=rho, sponge_width=_EC_SPONGE)
        omega = 2.0 * np.pi * f0
        phase_x = omega * np.sin(theta) * x_src / _EL_C0
        frames: list[Any] = []
        ts: list[float] = []
        inc = 0.0
        for i in range(steps):
            sim.step()
            t = sim.time
            r = min(t / ramp_t, 1.0)
            drive = (r * r * (3.0 - 2.0 * r)) * np.sin(omega * t - phase_x)
            sim.txx[src_row, src_cols] -= drive
            sim.tyy[src_row, src_cols] -= drive
            if w0 <= t <= w1:
                inc = max(inc, abs(float(sim.p[p_row, p_col])))
            if (i + 1) % every == 0:
                frames.append(sim.p[rows, cols].astype(np.float32))
                ts.append(t)
        stack = np.stack(frames) / inc
        tail = stack[3 * n_active // 4:]
        # The stored rows start at y0; index the under-plate box within.
        rb0 = (b_rows.start - rows.start) // 2
        rb1 = (b_rows.stop - rows.start) // 2
        rms = float(np.sqrt(np.mean(tail[:, rb0:rb1, :] ** 2)))
        trans_db.append(20.0 * float(np.log10(rms * np.sqrt(2.0))))
        stacks.append(stack)
    return stacks[0], stacks[1], np.asarray(ts), trans_db, n_active


def animate_elastic_coincidence(output_dir: str) -> None:
    """A sustained oblique plane wave in air on a 10 mm steel plate
    (elastic 2D FDTD, fluid-solid coupling in the same solver): below the
    coincidence frequency the mass law rules and the far side stays dark;
    at 2 f_c and 45 degrees the acoustic trace matches the free bending
    wavelength and the plate re-radiates a 45-degree beam that grows
    along the lit span, holding the transmitted level at the f_c/2 figure
    where the mass law demanded 12 dB more. The two frequencies and the
    angle come from the library's coincidence_frequency; the verdicts
    quote the oblique mass law (Bies Eq. 7.41) each panel is judged
    against. Both panels draw the air below the plate with the same
    annotated display gain, so the transmitted field is legible next to an
    incident wave some 45 dB louder."""
    from matplotlib import patheffects
    from matplotlib.patches import Rectangle

    from phonometry import coincidence_frequency

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    p_lo, p_hi, times, trans_db, n_active = _coincidence_fields()
    bp, m2 = _elastic_plate_bp_m2()
    fc = coincidence_frequency(m2, bp)
    x0, x1, y0, y1 = _EC_VIEW
    # The air below the plate rides an annotated display gain measured off
    # the settled field of both runs at once (see the _EC_VLIM note); the
    # field stacks are cached and shared by the four variants, so the gain
    # goes on copies.
    dx = _EL_DX
    i_below = (round((_EC_PLATE_Y + _EL_H) / dx) - round(y0 / dx)) // 2
    settled = slice(3 * n_active // 4, n_active)
    gain = _weak_field_gain(
        np.concatenate([p_lo[settled, i_below:, :].ravel(),
                        p_hi[settled, i_below:, :].ravel()]), _EC_VLIM)
    p_lo = p_lo.copy()
    p_hi = p_hi.copy()
    for stack in (p_lo, p_hi):
        stack[:, i_below:, :] *= gain

    fig = _anim_figure()
    fig.suptitle(T("Coincidence: the same steel plate, below and above "
                   "f_c (elastic 2D FDTD)"), fontweight="bold")
    axes = fig.subplots(1, 2, sharey=True)
    titles = [T(f"f = f_c/2 = {fc / 2:.0f} Hz, 45° incidence"),
              T(f"f = 2 f_c = {2 * fc:.0f} Hz, 45° incidence")]
    # Each panel is judged against the oblique mass law (Bies Eq. 7.41):
    # TL(theta) = 10 log10[1 + (pi f m'' cos(theta) / rho0 c0)^2]. Below f_c
    # the measured level lands on it; above f_c the trace-matched plate
    # re-radiates and beats it by ~13 dB, so the two panels transmit the
    # same level even though the mass law demands 12 dB more at 4x the
    # frequency.
    theta = np.radians(_EC_THETA)
    ml = [10.0 * float(np.log10(1.0 + (np.pi * f * m2 * np.cos(theta)
                                       / (_EL_RHO0 * _EL_C0)) ** 2))
          for f in (0.5 * fc, 2.0 * fc)]
    verdicts = [T(f"below f_c: the mass law holds: "
                  f"{_fmt_minus(trans_db[0], '.0f')} dB "
                  f"(it predicts {_fmt_minus(-ml[0], '.0f')})"),
                T(f"above f_c: trace matches λ_B: "
                  f"{_fmt_minus(trans_db[1], '.0f')} dB, "
                  f"the mass law said {_fmt_minus(-ml[1], '.0f')}")]
    ims: list[Any] = []
    v_txts: list[Any] = []
    for col, (ax, title, data) in enumerate(
            ((axes[0], titles[0], p_lo), (axes[1], titles[1], p_hi))):
        ax.grid(False)
        im = ax.imshow(data[0], origin="upper", extent=(x0, x1, y1, y0),
                       cmap=CMAP_FIELD, vmin=-_EC_VLIM, vmax=_EC_VLIM,
                       aspect="equal", interpolation="bilinear")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.add_patch(Rectangle((x0, _EC_PLATE_Y), x1 - x0, _EL_H,
                               facecolor=COLOR_GRID, edgecolor=COLOR_FG,
                               lw=0.8, zorder=3))
        if col == 0:
            # Right-aligned at the far end of the plate it names: anchored
            # at the near end it grew, in Spanish, into the incidence
            # arrow, whose head then landed on the "0" of "10 mm" and left
            # the plate reading 1 mm thick.
            ax.text(x1 - 0.02, _EC_PLATE_Y - 0.012,
                    T("10 mm steel plate"), ha="right", va="bottom",
                    color=FIELD_INK, fontsize=7.5, path_effects=outline,
                    zorder=4)
            ax.set_ylabel("y [m]", fontsize=8)
        ax.text(x0 + 0.02, _EC_PLATE_Y + _EL_H + 0.012,
                T(_gain_note("air below the plate", gain)), ha="left",
                va="top", color=FIELD_INK, fontsize=6.5,
                path_effects=outline, zorder=4)
        # Incidence arrow at 45 degrees.
        ax.annotate("", xy=(x0 + 0.34, y0 + 0.30),
                    xytext=(x0 + 0.13, y0 + 0.09),
                    arrowprops={"arrowstyle": "-|>", "color": FIELD_INK,
                                "lw": 1.4}, zorder=4)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        v_txt = ax.text(x1 - 0.02, y1 - 0.02, "", ha="right", va="bottom",
                        color="white", fontsize=8, zorder=5,
                        bbox={"boxstyle": _ANIM_PILL_BOX,
                              "facecolor": "black", "alpha": 0.55,
                              "edgecolor": "none"})
        ims.append(im)
        v_txts.append(v_txt)
    fc_txt = fig.text(0.5, 0.035,
                      T(f"coincidence_frequency: f_c = {fc:.0f} Hz "
                        f"(10 mm steel)"), ha="center", va="bottom",
                      fontsize=9, color=COLOR_FG)
    t_txt = fig.text(0.985, 0.035, "", ha="right", va="bottom",
                     family="monospace", fontsize=10, color=COLOR_FG)
    reveal = int(0.75 * n_active)

    def update(k: int) -> tuple[Any, ...]:
        kf = min(k, n_active - 1)
        ims[0].set_data(p_lo[kf])
        ims[1].set_data(p_hi[kf])
        for col, v_txt in enumerate(v_txts):
            v_txt.set_text(verdicts[col] if k >= reveal else "")
        t_txt.set_text(T(f"t = {times[kf] * 1e3:5.2f} ms"))
        return (*ims, *v_txts, fc_txt, t_txt)

    _render_clip(fig, update, output_dir, "anim_elastic_coincidence",
                 frames=n_active + _ANIM_HOLD, gif_fps=5)
