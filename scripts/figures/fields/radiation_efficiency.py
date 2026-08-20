#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Why a plate below coincidence hardly radiates, and above it radiates fully.

The mechanical counterpart of the coincidence clip: the same 10 mm steel
plate, but driven by a force on the plate itself rather than by an airborne
wave, so what the air above it does is *radiation* and not transmission.
"""

from functools import lru_cache
from typing import Any

import numpy as np

from ..media import _ANIM_PILL_BOX, _anim_figure, _render_clip, _translate_str
from ..theme import CMAP_FIELD, COLOR_FG, COLOR_GRID, FIELD_INK, FIELD_STROKE
from .elastic import (
    _EC_CB_TUNE,
    _EL_C0,
    _EL_CP,
    _EL_CS,
    _EL_DX,
    _EL_H,
    _EL_RHO,
    _EL_RHO0,
    _elastic_plate_bp_m2,
)

# Grid. The plate, its mesh and its numerical-dispersion compensation are
# the coincidence clip's, imported rather than restated: same 10 mm steel,
# same dx = h / 4 = 2.5 mm, same _EC_CB_TUNE, so the two clips describe one
# plate with one critical frequency and the reader can put them side by side.
_RE_NY, _RE_NX = 280, 540
_RE_SPONGE = 80  # 0.20 m on all four sides
_RE_R0 = 180  # first plate row (its upper face)
_RE_PLATE_ROWS = 4  # h / dx, the 10 mm plate
_RE_VIEW_ROWS = (80, 200)  # rows shown: the sponges stay off screen
_RE_VIEW_COLS = (80, 460)  # 0.95 m of plate, sponge-free
_RE_DEC = 2  # display decimation
# Drive. A vertical line force on the mid-plate face, phase-graded with the
# panel's own k_B so the plate carries a wave running to the right, applied
# over the *whole* plate and rolled off only in the outer 0.15 m at each end,
# which sit inside the sponges. A compact driving patch was the obvious
# design and is the wrong one: any finite patch has a wavenumber spectrum
# broad enough to put a large part of itself inside the radiating circle
# |k_x| < k_0, and below coincidence that aperture radiation -- real, but not
# the mechanism this clip is about -- is the *only* thing in the air above,
# because the travelling subsonic wave radiates exactly nothing. A first cut
# drove a patch one bending wavelength wide and the subsonic panel filled
# with broad lobes that did not decay; the fitted skin depth came out at
# 0.17 m against the 0.091 m of an evanescent wave. Driving the whole plate
# instead is both the cleaner picture and the honest model of the infinite
# plate the section is about.
_RE_ROLLOFF = 0.15  # taper length at each end of the drive [m]
_RE_SPAN_X0 = 0.45  # measurement span starts here [m]; it runs
#                                    one bending wavelength of the lower
#                                    panel, exactly two of the upper one, so
#                                    both close on a whole number of cycles
#                                    in space as well as in time, and it sits
#                                    in the middle of the view, clear of both
#                                    roll-offs
_RE_PROBE_H = 0.02  # height of the air row the horizontal
#                                    wavenumber is read from [m]; close in,
#                                    where the evanescent branch of the
#                                    subsonic panel still dominates whatever
#                                    little else is in the air
_RE_RAMP_CYCLES = 1.5  # source onset, in periods of each panel
_RE_AVG_CYCLES = 2  # settled averaging, in periods of f_c / 2
_RE_EVERY = 64  # capture stride [solver steps]
_RE_FPS = 70  # 810 active frames + 2 s hold = 13.3 s
#: Colour half-range on the normalisation that divides the air pressure by
#: ``rho0 c0 v_plate``, i.e. by the pressure a piston of the same surface
#: velocity would make. 1.6 leaves the supersonic panel's beam room without
#: bleaching the subsonic panel's near field, which reaches the same order
#: at the plate and simply does not leave.
_RE_VLIM = 1.6


def _re_geometry() -> tuple[float, float, float, float]:
    """View extent ``(x0, x1, h0, h1)`` [m], x from the left of the view and
    h measured from the plate's upper face, positive upwards.
    """
    dx = _EL_DX
    x1 = (_RE_VIEW_COLS[1] - _RE_VIEW_COLS[0]) * dx
    h1 = (_RE_R0 - _RE_VIEW_ROWS[0]) * dx
    h0 = (_RE_R0 - (_RE_VIEW_ROWS[1] - _RE_DEC)) * dx
    return 0.0, x1, h0, h1


def _re_bending_wavenumber(frequency: float) -> float:
    """Free bending wavenumber of the plate [rad/m], ``(m'' w^2 / B')^(1/4)``."""
    bp, m2 = _elastic_plate_bp_m2()
    return float((m2 * (2.0 * np.pi * frequency) ** 2 / bp) ** 0.25)


def _re_decay_length(rms_by_row: Any, r0: int, dx: float) -> float:
    """e-folding height of the pressure envelope above the plate [m].

    A log-linear fit of the row-wise RMS between 1 and 4 cm above the
    plate's upper face. Close in on purpose: a plate driven over any finite
    stretch radiates a little from the ends of that stretch, and that
    propagating floor does not decay with height, so a band reaching further
    up biases the fit long (measured: 0.137 m over 1.5-7 cm against the
    0.091 m an evanescent wave must have). For a subsonic bending wave the
    evanescent branch is what the fit should see and it returns
    ``1 / sqrt(k_B^2 - k_0^2)``, a closed form the clip prints itself
    against; above coincidence nothing decays and the fit is meaningless,
    which is why only the lower panel quotes it.
    """
    lo, hi = r0 - round(0.04 / dx), r0 - round(0.010 / dx)
    h = (r0 - np.arange(lo, hi)) * dx
    y = np.log(np.maximum(rms_by_row[lo:hi], 1e-300))
    return float(-1.0 / np.polyfit(h, y, 1)[0])


@lru_cache(maxsize=1)
def _radiation_efficiency_fields() -> tuple[Any, Any, Any, Any, Any]:
    """The subsonic and supersonic runs of the same driven plate, cached.

    Two identical scenes -- 10 mm steel plate in air, sponges on all four
    sides, a phase-graded line force along the whole plate -- differing only
    in the drive frequency: ``f_c / 2``, where the free bending wave is
    slower than sound (``k_B = 1.414 k_0``) so the air above can carry only
    an evanescent field, and ``2 f_c``, where it is faster
    (``k_B = 0.707 k_0``) and the trace match launches a plane beam at
    ``arcsin(c_0 / c_B) = 45`` degrees.

    **Nothing is measured off this scene, and that is deliberate.** Two
    quantities were tried and both came out wrong in a way the picture makes
    obvious. A radiation efficiency is an artefact of the drive (see the note
    on the drive above). The horizontal wavenumber of the air, and the skin
    depth with it, are spoiled by the ends: the plate wave meets the sponge at
    each end and a damping ramp reflects a bending wave partially, so the air
    above carries a leftward beam on top of the rightward one. The
    phase-increment estimator then reads the *difference* of the two, and it
    printed k_x = 0.34 k_0 -- a beam at 20 degrees -- on a panel whose beam is
    visibly leaving at 45, while the fitted skin depth came out at 0.133 m
    against 0.091 m. Terminating a bending wave well enough to fix that needs
    a much wider domain and a graded plate absorber, which is a different
    clip. What this one shows is the mechanism; the numbers beside it are the
    closed forms the mechanism obeys, from the plate's own constants.

    Timeline per the animation norms. The capture stride is 64 solver steps
    = 10.5 us, i.e. 39.5 frames per period of the 2 f_c carrier, more than
    three times the 12 the norms require (158 at f_c / 2). The window is the
    lower panel's onset ramp (1.5 periods = 2.49 ms) plus 1.2 x the flight
    of sound across the *diagonal* of the visible field -- the beam is
    launched along the whole plate and the last thing to arrive is the
    corner, sqrt(0.95^2 + 0.25^2) = 0.98 m at 343 m/s = 2.86 ms -- plus two
    whole periods of f_c / 2 to average the measurements over. Both panels
    are captured over that same window so the two strips stay in step.

    Returns the two pressure stacks (normalised to ``rho0 c0 v_plate``, i.e.
    to the pressure a piston of the same surface velocity would make), the
    two plate-velocity histories (normalised to their own amplitude), the
    frame times, the measured ``k_x / k_0`` of the air above each panel and
    the fitted e-folding height of the pressure envelope in each.
    """
    from phonometry import simulation, vibration

    dx = _EL_DX
    bp, m2 = _elastic_plate_bp_m2()
    fc = vibration.coincidence_frequency(m2, bp)
    freqs = (0.5 * fc, 2.0 * fc)
    cp_plate = _EL_CP * _EC_CB_TUNE
    dt = 0.6 * dx / (cp_plate * float(np.sqrt(2.0)))
    lam_b_lo = 2.0 * np.pi / _re_bending_wavenumber(freqs[0])
    ramp_lo = _RE_RAMP_CYCLES / freqs[0]
    span = slice(round(_RE_SPAN_X0 / dx), round((_RE_SPAN_X0 + lam_b_lo) / dx))
    x0, x1, _h0, h1 = _re_geometry()
    diagonal = float(np.hypot(x1 - x0, h1))
    span_t = ramp_lo + 1.2 * diagonal / _EL_C0 + _RE_AVG_CYCLES / freqs[0]
    n_active = int(np.ceil(span_t / (_RE_EVERY * dt)))
    steps = n_active * _RE_EVERY

    r0 = _RE_R0
    drive_row = r0 + 1  # mid-plate y-face
    rows = slice(_RE_VIEW_ROWS[0], _RE_VIEW_ROWS[1], _RE_DEC)
    cols = slice(_RE_VIEW_COLS[0], _RE_VIEW_COLS[1], _RE_DEC)
    src_cols = np.arange(_RE_NX)
    x_src = (src_cols + 0.5) * dx
    # Full-width drive with a cosine-squared roll-off over the outer
    # _RE_ROLLOFF at each end, both of which sit inside the sponges.
    edge = np.clip(np.minimum(x_src, _RE_NX * dx - x_src) / _RE_ROLLOFF, 0.0, 1.0)
    taper = np.sin(0.5 * np.pi * edge) ** 2

    stacks: list[Any] = []
    plates: list[Any] = []
    times = np.zeros(0)
    for f0 in freqs:
        c_p = np.full((_RE_NY, _RE_NX), _EL_C0)
        c_s = np.zeros((_RE_NY, _RE_NX))
        rho = np.full((_RE_NY, _RE_NX), _EL_RHO0)
        c_p[r0 : r0 + _RE_PLATE_ROWS] = cp_plate
        c_s[r0 : r0 + _RE_PLATE_ROWS] = _EL_CS * _EC_CB_TUNE
        rho[r0 : r0 + _RE_PLATE_ROWS] = _EL_RHO
        sim = simulation.ElasticFDTD2D(c_p, c_s, dx, rho=rho, sponge_width=_RE_SPONGE)
        omega = 2.0 * np.pi * f0
        k_b = _re_bending_wavenumber(f0)
        ramp_t = _RE_RAMP_CYCLES / f0
        # Whole number of drive periods at the end of the window: the upper
        # panel gets four times as many cycles inside the same 3.32 ms.
        cycles = _RE_AVG_CYCLES * round(f0 / freqs[0])
        avg_from = steps - round(cycles / (f0 * dt))
        v2_sum = 0.0
        n_avg = 0
        frames: list[Any] = []
        plate: list[Any] = []
        ts: list[float] = []
        for i in range(steps):
            sim.step()
            t = sim.time
            r = min(t / ramp_t, 1.0)
            sim.vy[drive_row] += (
                (r * r * (3.0 - 2.0 * r)) * taper * np.sin(omega * t - k_b * x_src)
            )
            if i >= avg_from:
                v2_sum += float(np.mean(sim.vy[drive_row, span] ** 2))
                n_avg += 1
            if (i + 1) % _RE_EVERY == 0:
                p = sim.p.copy()
                # Blank exactly the plate rows: inside the steel `p` is the
                # mean normal stress, orders above any air pressure, and the
                # plate is drawn as a patch instead. One row either side
                # would blank the air the near field lives in.
                p[r0 : r0 + _RE_PLATE_ROWS] = 0.0
                frames.append(p[rows, cols].astype(np.float32))
                plate.append(sim.vy[drive_row, cols].astype(np.float32))
                ts.append(t)
        v2 = v2_sum / n_avg
        # The colour scale is the pressure a piston of the same surface
        # velocity would make, so a panel that reaches +-1 is radiating like
        # one and a panel that stays dark is not.
        p_piston = _EL_RHO0 * _EL_C0 * float(np.sqrt(2.0 * v2))
        stack = np.stack(frames) / p_piston
        prof = np.stack(plate)
        stacks.append(stack[:, ::-1, :])
        plates.append(prof / float(np.abs(prof[3 * n_active // 4 :]).max()))
        times = np.asarray(ts)
    return stacks[0], stacks[1], plates[0], plates[1], times


def animate_elastic_radiation_efficiency(output_dir: str) -> None:
    """A driven steel plate below and above coincidence (elastic 2D FDTD).

    The mechanism behind the radiation factor, which the sigma curve on the
    surface-velocity page can only state. One 10 mm steel plate in air,
    driven along its own length so a bending wave runs to the right, at
    f_c / 2 and at 2 f_c. Below coincidence the bending wavelength is
    *shorter* than the acoustic one, adjacent half-waves push and pull the
    same air in antiphase, and what the air above shows is a skin that
    clings to the plate and dies within a fraction of a wavelength: the
    pressure is there, the power is not. Above coincidence the bending wave
    outruns sound, the trace match is satisfied, and a plane beam leaves at
    arcsin(c_0 / c_B) = 45 degrees. Both panels are drawn on the pressure a
    piston of the same surface velocity would make, and each is annotated
    with the bending wavenumber measured off the simulated plate and the
    sin(theta) = k_B / k_0 it implies: greater than one below coincidence,
    where no radiating angle exists, and the sine of the beam's own angle
    above it.
    """
    from matplotlib import patheffects
    from matplotlib.patches import Rectangle

    from phonometry import vibration

    T = _translate_str
    p_lo, p_hi, plate_lo, plate_hi, times = _radiation_efficiency_fields()
    bp, m2 = _elastic_plate_bp_m2()
    fc = vibration.coincidence_frequency(m2, bp)
    freqs = (0.5 * fc, 2.0 * fc)
    # Closed forms from the plate's own constants: the bending wavelength of
    # each panel against the acoustic one, the trace-match angle above
    # coincidence and the evanescent skin depth below it.
    lam_b = [2.0 * np.pi / _re_bending_wavenumber(f) for f in freqs]
    lam_0 = [_EL_C0 / f for f in freqs]
    theta = float(np.degrees(np.arcsin(lam_0[1] / lam_b[1])))
    k_b0 = _re_bending_wavenumber(freqs[0])
    k_00 = 2.0 * np.pi * freqs[0] / _EL_C0
    skin_exact = 1.0 / float(np.sqrt(k_b0**2 - k_00**2))
    outline = [patheffects.withStroke(linewidth=2.0, foreground=FIELD_STROKE)]
    x0, x1, h0, h1 = _re_geometry()
    n_active = int(times.size)
    x_plate = np.linspace(x0, x1, plate_lo.shape[1])
    # The deflection overlay is a display device: the real transverse
    # amplitude of a driven steel plate is microns against a 0.95 m span.
    swing = 0.030

    fig = _anim_figure()
    # Reserve the bottom strip for the two figure-level captions and the
    # clock: fig.text does not claim space from the constrained layout.
    fig.get_layout_engine().set(rect=(0.0, 0.055, 1.0, 0.945))
    # Short on purpose: the Spanish of the longer title overran the canvas
    # at both ends. The solver and the plate moved to the footer.
    fig.suptitle(
        T("Radiation efficiency: a driven plate below and above $f_\\mathrm{c}$")
    )
    axes = fig.subplots(2, 1, sharex=True)
    titles = [
        T(
            f"$f = f_\\mathrm{{c}}/2$ = {freqs[0]:.0f} Hz, below coincidence: the plate "
            f"wave is slower than sound"
        ),
        T(
            f"$f = 2f_\\mathrm{{c}}$ = {freqs[1]:.0f} Hz, above coincidence: the plate "
            f"wave is faster than sound"
        ),
    ]
    # One measurement per panel, each the one its regime admits: a decay
    # where the field is evanescent, an angle where it propagates.
    verdicts = [
        T(
            f"$\\lambda_\\mathrm{{B}}$ = {lam_b[0]:.2f} m is shorter than $\\lambda$ = "
            f"{lam_0[0]:.2f} m in air\nno angle solves "
            f"$\\sin\\theta = \\lambda/\\lambda_\\mathrm{{B}}$: the skin dies in "
            f"{skin_exact:.3f} m"
        ),
        T(
            f"$\\lambda_\\mathrm{{B}}$ = {lam_b[1]:.2f} m is longer than $\\lambda$ = "
            f"{lam_0[1]:.2f} m in air\nthe trace match sends a beam out at "
            f"{theta:.0f}°"
        ),
    ]
    ims: list[Any] = []
    defl: list[Any] = []
    v_txts: list[Any] = []
    for row, (data, title) in enumerate(((p_lo, titles[0]), (p_hi, titles[1]))):
        ax = axes[row]
        ax.grid(False)
        im = ax.imshow(
            data[0],
            origin="lower",
            extent=(x0, x1, h0, h1),
            cmap=CMAP_FIELD,
            vmin=-_RE_VLIM,
            vmax=_RE_VLIM,
            aspect="equal",
            interpolation="bilinear",
        )
        ax.set_title(title, fontsize=9)
        ax.add_patch(
            Rectangle(
                (x0, -_EL_H),
                x1 - x0,
                _EL_H,
                facecolor=COLOR_GRID,
                edgecolor=COLOR_FG,
                lw=0.7,
                zorder=3,
            )
        )
        (line,) = ax.plot(
            x_plate,
            np.zeros_like(x_plate),
            lw=1.2,
            color=FIELD_INK,
            path_effects=outline,
            zorder=4,
        )
        ax.set_ylabel(T("height above the plate [m]"), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yticks([0.0, 0.1, 0.2])
        if row == 1:
            ax.set_xlabel(T("distance along the plate [m]"), fontsize=8)
            # The trace-matched beam direction, drawn once on the panel
            # where it exists. It starts well left of centre so the label
            # lands inside the panel rather than against its right edge.
            reach = 0.85 * h1
            foot_x = 0.12
            ax.annotate(
                "",
                xy=(foot_x + reach * float(np.tan(np.radians(theta))), reach),
                xytext=(foot_x, 0.0),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": FIELD_INK,
                    "lw": 1.3,
                    "ls": "--",
                },
                zorder=5,
            )
            ax.text(
                foot_x + 0.6 * reach * float(np.tan(np.radians(theta))) - 0.012,
                0.6 * reach,
                T(f"{theta:.0f}°"),
                fontsize=8,
                ha="right",
                va="center",
                color=FIELD_INK,
                path_effects=outline,
                zorder=6,
            )
        else:
            # Bottom-left, under the plate: the top-left abutted the
            # verdict pill once the Spanish widened it.
            ax.text(
                x0 + 0.012,
                h0 + 0.008,
                T("the whole plate is driven"),
                ha="left",
                va="bottom",
                fontsize=7,
                color="white",
                zorder=6,
                bbox={
                    "boxstyle": _ANIM_PILL_BOX,
                    "facecolor": "black",
                    "alpha": 0.5,
                    "edgecolor": "none",
                },
            )
        # Top-right, clear of the "driven here" pill over the aperture, of
        # the trace-match arrow that climbs to the left of it, and of the
        # plate and its deflection overlay along the bottom.
        v_txt = ax.text(
            x1 - 0.012,
            h1 - 0.010,
            "",
            ha="right",
            va="top",
            color="white",
            fontsize=8,
            zorder=7,
            bbox={
                "boxstyle": _ANIM_PILL_BOX,
                "facecolor": "black",
                "alpha": 0.6,
                "edgecolor": "none",
            },
        )
        ims.append(im)
        defl.append(line)
        v_txts.append(v_txt)
    foot = fig.text(
        0.5,
        0.030,
        T(
            "colour: air pressure / the pressure a piston of the "
            "same surface velocity would make"
        ),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=COLOR_FG,
    )
    foot2 = fig.text(
        0.5,
        0.006,
        T(
            f"elastic 2D FDTD, 10 mm steel plate, "
            f"$f_\\mathrm{{c}}$ = {fc:.0f} Hz · overlaid line: its "
            f"deflection, exaggerated"
        ),
        ha="center",
        va="bottom",
        fontsize=7,
        color=COLOR_FG,
        alpha=0.85,
    )
    t_txt = fig.text(
        0.988,
        0.030,
        "",
        ha="right",
        va="bottom",
        family="monospace",
        fontsize=9,
        color=COLOR_FG,
    )
    reveal = int(0.80 * n_active)

    def update(k: int) -> tuple[Any, ...]:
        kf = min(k, n_active - 1)
        for i, (stack, prof) in enumerate(((p_lo, plate_lo), (p_hi, plate_hi))):
            ims[i].set_data(stack[kf])
            defl[i].set_ydata(-0.5 * _EL_H + swing * prof[kf])
            v_txts[i].set_text(verdicts[i] if k >= reveal else "")
        t_txt.set_text(T(f"$t$ = {times[kf] * 1e3:5.2f} ms"))
        return (*ims, *defl, *v_txts, foot, foot2, t_txt)

    _render_clip(
        fig,
        update,
        output_dir,
        "anim_elastic_radiation_efficiency",
        frames=n_active + 2 * _RE_FPS,
        fps=_RE_FPS,
        gif_fps=5,
    )
