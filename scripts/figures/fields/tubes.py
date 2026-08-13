#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The impedance tube and the transmission tube built from the same bore.

Both FDTD clips drive the same 2.5 mm meshed 100 mm bore at 850 Hz with the
same equivalent-fluid sample and draw the same tube hardware, so they share
their sample, mesh and carrier. The standing-wave clip is the schematic view
of the same ISO 10534-2 apparatus.
"""

from functools import lru_cache
from typing import Any

import numpy as np

from ..media import (
    _ANIM_FPS,
    _ANIM_FRAMES,
    _ANIM_HOLD,
    _ANIM_PILL_BOX,
    _anim_figure,
    _render_clip,
    _translate_str,
)
from ..schematics import _draw_mic, _draw_speaker, _schematic_axes
from ..theme import (
    CMAP_FIELD,
    COLOR_FG,
    COLOR_GRID,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
)
from ._core import _anim_speaker

#: Virtual-tube sample: the equivalent fluid of the solver cross-checks
#: (slower, denser, lossy; k = (omega - j sigma)/c at the real rho c).
_VTUBE_C2, _VTUBE_RHO2, _VTUBE_SIGMA = 0.6 * 343.0, 3.6, 600.0
_VTUBE_DX = 0.0025
_VTUBE_F = 850.0                          # CW inside the 100 mm plane range


def _anim_tube_hardware(ax: Any, length: float, *, bore: float = 0.1,
                        sample: tuple[float, float] | None = None,
                        termination: str = "rigid",
                        mics: tuple[tuple[float, str], ...] = (),
                        label_speaker: bool = True) -> None:
    """Draw the tube as hardware around the FDTD bore: walls, loudspeaker,
    sample block, microphones and the termination, all to scale (metres)."""
    from matplotlib.patches import Rectangle

    wall = 0.014
    grey = "#9a9a9a"
    for y0 in (-wall, bore):
        ax.add_patch(Rectangle((0.0, y0), length, wall, facecolor=grey,
                               edgecolor=COLOR_FG, linewidth=0.7, zorder=3))
    # Loudspeaker driving the left end: magnet + cone into the bore.
    _anim_speaker(ax, 0.0, 0.5 * bore, bore,
                  label_y=-0.55 * bore if label_speaker else None)
    if sample is not None:
        ax.add_patch(Rectangle((sample[0], 0.0), sample[1] - sample[0],
                               bore, facecolor=COLOR_SECONDARY, alpha=0.30,
                               hatch="..", edgecolor=COLOR_SECONDARY,
                               linewidth=0.0, zorder=2.6))
    if termination == "rigid":
        ax.add_patch(Rectangle((length, -wall), 0.030, bore + 2 * wall,
                               facecolor="#5a5a5a", edgecolor=COLOR_FG,
                               linewidth=0.8, zorder=3))
        ax.text(length + 0.030, -0.55 * bore, "rigid plug", ha="right",
                va="top", fontsize=7.5)
    else:
        ax.add_patch(Rectangle((length, -wall), 0.045, bore + 2 * wall,
                               facecolor=grey, hatch="////",
                               edgecolor=COLOR_FG, linewidth=0.8, zorder=3))
        ax.text(length + 0.045, -0.55 * bore, "anechoic termination",
                ha="right", va="top", fontsize=7.5)
    for x_m, label in mics:
        ax.plot([x_m, x_m], [bore + wall, bore + wall + 0.35 * bore],
                color=COLOR_FG, lw=1.2, zorder=5)
        ax.plot([x_m], [bore + wall + 0.50 * bore], marker="o", ms=7,
                markerfacecolor=COLOR_PRIMARY, markeredgecolor=COLOR_FG,
                markeredgewidth=0.7, zorder=5)
        ax.text(x_m, bore + wall + 0.78 * bore, label, ha="center",
                va="bottom", fontsize=7.5, color=COLOR_PRIMARY)


# Impedance-tube timeline. Mesh: the carrier is 850 Hz, where the air
# wavelength is 0.404 m (lambda / 8 = 50 mm) and the equivalent-fluid
# sample's is 0.242 m (lambda / 8 = 30 mm); the smallest geometric
# dimension is the 0.1 m bore (0.1 / 4 = 25 mm), so the rule
# dx = min(25, 30) mm allows 25 mm and dx = 2.5 mm sits ten times finer,
# which the 10 cm sample (40 cells) and the envelope readout need.
# Flight: the wave enters at x = 0, crosses 1.1 m of air and 0.1 m of
# sample to the rigid plug and comes back to the farthest visible point
# (the left edge), i.e. 2.2 m at 343 m/s plus 0.2 m at the sample's
# 205.8 m/s = 7.39 ms, x 1.2 = 8.86 ms.
# Sampling: 7 solver steps per frame = 21.6 us gives 54.3 frames per 850 Hz
# period (>= 48), and 410 frames cover the 8.87 ms window (the old one ran
# 16.7 ms at 25 frames per period), played at 15/7 of the shared 20 fps --
# exactly the factor by which the stride shrank from the committed clip's
# 15 solver steps per frame -- so the pacing stays the 0.928 ms of
# simulation per second of playback the committed clip had, and the flight
# takes 9.6 s (the old 18 s clip ran on well past it). The clip then holds
# the settled frame for the shared 2 s (86 frames at this rate): the
# reflected front reaches the left edge on the last captured frame, so the
# standing-wave envelope -- a running maximum over the trailing period --
# is only complete right at the end, and the absorption pill is revealed
# there rather than mid-flight.
_VTUBE_EVERY = 7
_VTUBE_ACTIVE = 410
#: 15/7 of the shared rate, matching the stride cut (see the note above).
_VTUBE_FPS = _ANIM_FPS * 15.0 / 7.0
#: The shared 2 s closing hold, counted at this clip's own frame rate.
_VTUBE_HOLD = round(2.0 * _VTUBE_FPS)
_VTUBE_FRAMES = _VTUBE_ACTIVE + _VTUBE_HOLD


@lru_cache(maxsize=1)
def _impedance_tube_fields(
    n_frames: int = _VTUBE_ACTIVE,
) -> tuple[Any, Any, Any, Any]:
    """CW build-up in the virtual impedance tube, empty vs sample, cached.

    Two 1,2 m x 0,1 m tubes: a rho c edge on the left carries a sustained
    one-way 850 Hz plane wave in; the right end is rigid. The lower tube
    ends in a 10 cm equivalent-fluid sample. Returns the frame stacks, the
    running envelope stacks max|p(x)| over the trailing period, the frame
    times and the analytic absorption of the sample at the drive frequency.
    """
    import fdtd2d

    dx = _VTUBE_DX
    ny, nx = 40, 480                       # 0.1 m x 1.2 m
    every = _VTUBE_EVERY
    sample_cells = round(0.10 / dx)
    runs = []
    for with_sample in (False, True):
        c_map = np.full((ny, nx), 343.0)
        rho_map = np.full((ny, nx), 1.2)
        damping = np.zeros((ny, nx))
        if with_sample:
            c_map[:, nx - sample_cells:] = _VTUBE_C2
            rho_map[:, nx - sample_cells:] = _VTUBE_RHO2
            damping[:, nx - sample_cells:] = _VTUBE_SIGMA
        sim = fdtd2d.FDTD2D(c_map, dx, rho=rho_map, damping=damping,
                            edge_impedance={"left": 1.2 * 343.0})
        tone = fdtd2d.CWSource(0, 0, frequency=_VTUBE_F)
        sim.add_source(fdtd2d.PlaneWaveSource("right", tone.value,
                                              offset=2))
        period_frames = max(1, round(1.0 / (_VTUBE_F * every * sim.dt)))
        ps: list[Any] = []
        env: list[Any] = []
        ts: list[float] = []
        recent: list[Any] = []
        while len(ps) < n_frames:
            sim.step()
            if sim.n % every == 0:
                row = np.abs(sim.p[ny // 2, :]).astype(np.float32)
                recent.append(row)
                if len(recent) > period_frames:
                    recent.pop(0)
                ps.append(sim.p[::2, ::2].astype(np.float32))
                env.append(np.max(np.stack(recent), axis=0))
                ts.append(sim.time)
        runs.append((np.stack(ps), np.stack(env)))
    times = np.asarray(ts)
    # Analytic absorption of the sample at the drive frequency.
    omega = 2.0 * np.pi * _VTUBE_F
    k2 = (omega - 1j * _VTUBE_SIGMA) / _VTUBE_C2
    z2 = _VTUBE_RHO2 * _VTUBE_C2
    zs = -1j * z2 / np.tan(k2 * 0.10)
    r = (zs - 1.2 * 343.0) / (zs + 1.2 * 343.0)
    alpha = float(1.0 - abs(r) ** 2)
    return runs[0], runs[1], times, alpha


def animate_fdtd_impedance_tube(output_dir: str) -> None:
    """The virtual impedance tube (2D FDTD): a loudspeaker drives a
    sustained plane tone into a rigid-walled tube; against the rigid plug
    the standing wave grows deep nulls (|r| ~ 1), while a 10 cm lossy
    sample in front of the same plug leaves the minima shallow: the ISO
    10534-2 microphone pair reads exactly that envelope. One concept: the
    standing-wave ratio IS the absorption measurement."""
    T = _translate_str
    (p_e, env_e), (p_s, env_s), times, alpha = _impedance_tube_fields()
    vmax = float(np.quantile(np.abs(p_e), 0.999))
    length, bore = 1.2, 0.1
    # The envelope baseline clears the microphone index labels: at its floor
    # the trace runs along env_base, and the "1"/"2" over each microphone
    # reach 0.192 (see _anim_tube_hardware), so the gap is the clip's own,
    # not whatever the settled envelope happens to leave.
    env_base, env_h, env_max = 0.265, 0.34, 2.2
    fig = _anim_figure()
    fig.suptitle(T("The virtual impedance tube: standing waves read the "
                   "absorption (2D FDTD)"), fontweight="bold")
    axes = fig.subplots(2, 1, sharex=True)
    titles = [T(r"Rigid end: deep minima, $|r| \sim 1$"),
              T("10 cm lossy sample: shallow minima")]
    mics = ((length - 0.10 - 0.20, "1"), (length - 0.10 - 0.15, "2"))
    ims: list[Any] = []
    lines: list[Any] = []
    x_env = (np.arange(p_e.shape[2] * 2) + 0.5) * _VTUBE_DX
    env_from = 16                          # hide the injection-line step
    for ax, title, with_sample in ((axes[0], titles[0], False),
                                   (axes[1], titles[1], True)):
        ax.grid(False)
        im = ax.imshow(np.zeros((20, 240)), origin="lower",
                       extent=(0.0, length, 0.0, bore),
                       cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
                       aspect="auto", interpolation="bilinear", zorder=2)
        _anim_tube_hardware(
            ax, length, bore=bore,
            sample=(length - 0.10, length) if with_sample else None,
            termination="rigid", mics=mics,
        )
        # The standing-wave envelope, drawn above the hardware.
        ax.axhline(env_base, color=COLOR_GRID, lw=0.8, zorder=1)
        # Under the baseline, not over it: the envelope grows out of that
        # line and the caption used to sit in the band the trace climbs
        # through, so the rising curve struck its own name out. Below the
        # line the band is empty all the way down to the tube wall.
        ax.text(0.005, env_base - 0.035, "$|p|$ envelope", fontsize=7.5,
                ha="left", va="top", color=COLOR_FG, alpha=0.8)
        (line,) = ax.plot([], [], color=COLOR_PRIMARY, lw=1.9, zorder=6)
        ax.set_aspect(0.42, adjustable="box")
        ax.set_xlim(-0.125, length + 0.075)
        ax.set_ylim(-0.150, env_base + env_h + 0.045)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        ims.append(im)
        lines.append(line)
    axes[1].set_xlabel(T("Position along the tube [m]"), fontsize=9)
    a_txt = axes[1].text(0.03, env_base + env_h - 0.015, "", ha="left",
                         va="top", fontsize=9, color="white", zorder=7,
                         bbox={"boxstyle": _ANIM_PILL_BOX,
                               "facecolor": "black", "alpha": 0.55,
                               "edgecolor": "none"})
    t_txt = fig.text(0.988, 0.93, "", ha="right", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)
    # The envelope is only settled once the reflected front has run back
    # down the whole tube (one round trip plus a period of the trailing
    # maximum), so the absorption pill is revealed at the very end of the
    # flight and read during the closing hold.
    n_active = len(times)
    reveal = n_active - 1

    def update(k: int) -> tuple[Any, ...]:
        kf = min(k, n_active - 1)
        for im, line, (p_all, env) in zip(ims, lines,
                                          ((p_e, env_e), (p_s, env_s))):
            im.set_data(p_all[kf])
            env_row = np.repeat(env[kf], 2)[: x_env.size]
            line.set_data(x_env[env_from:],
                          env_base + env_row[env_from:] / env_max * env_h)
        a_txt.set_text(
            T(f"$\\alpha$ = {alpha:.2f} at {_VTUBE_F:.0f} Hz")
            if k >= reveal else ""
        )
        t_txt.set_text(T(f"$t$ = {times[kf] * 1e3:5.1f} ms"))
        return (*ims, *lines, a_txt, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_impedance_tube",
                 frames=n_active + _VTUBE_HOLD, fps=_VTUBE_FPS, gif_fps=8)


# Transmission-tube timeline. Same 2.5 mm mesh and the same 850 Hz carrier
# as the impedance tube, for the same reason (bore / 4 = 25 mm against
# lambda_sample / 8 = 30 mm).
# Flight: the packet starts at x = 0.35 m; its transmitted half crosses
# 1.15 m of air and the 0.1 m sample to the far end of the frame
# (1.15 / 343 + 0.1 / 205.8 = 3.84 ms) while its reflected half runs
# 1.25 m of air back to the left edge (3.64 ms), so 1.2 x 3.84 ms =
# 4.61 ms of flight -- the old 3.2 ms window froze while both halves were
# still travelling.
# Sampling: 7 solver steps per frame = 21.6 us gives 54.3 frames per 850 Hz
# period (>= 48; the committed clip's 4-step stride gave 95, more than the
# norm asks, so the capture is coarsened to it), and 213 frames carry the
# flight, played at 4/7 of the shared 20 fps -- exactly the inverse of the
# factor by which the stride grew -- so the pacing stays the 0.247 ms of
# simulation per second of playback the committed clip had, and the flight
# takes 18.6 s. The clip then holds the settled verdict frame for the
# shared 2 s (23 frames at this rate).
_TTUBE_EVERY = 7
_TTUBE_ACTIVE = 213
#: 4/7 of the shared rate, undoing the stride coarsening (note above).
_TTUBE_FPS = _ANIM_FPS * 4.0 / 7.0
#: The shared 2 s closing hold, counted at this clip's own frame rate.
_TTUBE_HOLD = round(2.0 * _TTUBE_FPS)
_TTUBE_FRAMES = _TTUBE_ACTIVE + _TTUBE_HOLD
#: Frame the deferred-loading poster is cut from: t = 3.2 ms, where the
#: reflected and transmitted halves are fully separated and both still
#: inside the tube. Both ends are anechoic, so by the end of the flight the
#: packets have been absorbed and the tube is quiet: a correct last frame,
#: and the one the closing hold freezes on, but not the still that tells a
#: reader what the clip is about. Only the poster is cut from here; the
#: clip's own clock never leaves the flight it is timing.
_TTUBE_VERDICT = 148


@lru_cache(maxsize=1)
def _transmission_tube_fields(
    n_frames: int = _TTUBE_FRAMES,
) -> tuple[Any, Any, Any, Any]:
    """A carrier packet crossing the virtual transmission tube, cached.

    Two 1,6 m x 0,1 m tubes with anechoic rho c ends: the lower one holds
    a 10 cm equivalent-fluid layer mid-tube, so the packet splits into a
    reflected and an attenuated transmitted part. Returns the two frame
    stacks, the frame times and the analytic transmission loss at the
    carrier frequency.
    """
    import fdtd2d

    dx = _VTUBE_DX
    ny, nx = 40, 640                       # 0.1 m x 1.6 m
    every = _TTUBE_EVERY
    face = 320
    sample_cells = round(0.10 / dx)
    runs = []
    for with_sample in (False, True):
        c_map = np.full((ny, nx), 343.0)
        rho_map = np.full((ny, nx), 1.2)
        damping = np.zeros((ny, nx))
        if with_sample:
            sl = slice(face, face + sample_cells)
            c_map[:, sl] = _VTUBE_C2
            rho_map[:, sl] = _VTUBE_RHO2
            damping[:, sl] = _VTUBE_SIGMA
        sim = fdtd2d.FDTD2D(c_map, dx, rho=rho_map, damping=damping,
                            edge_impedance={"left": 1.2 * 343.0,
                                            "right": 1.2 * 343.0})
        sim.add_plane_wave("right", center=0.35, width=0.10,
                           wavelength=343.0 / _VTUBE_F)
        ps: list[Any] = []
        ts: list[float] = []
        active = min(n_frames, _TTUBE_ACTIVE)
        while len(ps) < active:
            sim.step()
            if sim.n % every == 0:
                ps.append(sim.p[::2, ::2].astype(np.float32))
                ts.append(sim.time)
        while len(ps) < n_frames:          # end hold on the last frame
            ps.append(ps[-1])
            ts.append(ts[-1])
        runs.append(np.stack(ps))
    omega = 2.0 * np.pi * _VTUBE_F
    k2 = (omega - 1j * _VTUBE_SIGMA) / _VTUBE_C2
    kd = k2 * 0.10
    z2 = _VTUBE_RHO2 * _VTUBE_C2
    rc = 1.2 * 343.0
    combo = (np.cos(kd) + 1j * z2 * np.sin(kd) / rc
             + rc * 1j * np.sin(kd) / z2 + np.cos(kd))
    tl = float(20.0 * np.log10(abs(combo) / 2.0))
    return runs[0], runs[1], np.asarray(ts), tl


def animate_fdtd_transmission_tube(output_dir: str) -> None:
    """The virtual transmission tube (2D FDTD): the loudspeaker end fires a
    carrier packet down a rigid-walled tube with an anechoic termination.
    The empty tube passes it unchanged; a 10 cm lossy layer mid-tube splits
    it into a reflection and a weakened transmission that the four ASTM
    E2611 microphones resolve. One concept: transmission loss is what fails
    to come out the other side."""
    T = _translate_str
    p_e, p_s, times, tl = _transmission_tube_fields()
    vmax = float(np.quantile(np.abs(p_e), 0.999))
    length, bore = 1.6, 0.1
    fig = _anim_figure()
    fig.suptitle(T("The virtual transmission tube: what gets through "
                   "(2D FDTD)"), fontweight="bold")
    axes = fig.subplots(2, 1, sharex=True)
    titles = [T("Empty tube: the packet crosses unchanged"),
              T("10 cm lossy layer: reflected + attenuated transmission")]
    mics = ((0.60, "1"), (0.65, "2"), (1.05, "3"), (1.10, "4"))
    ims: list[Any] = []
    for ax, title, with_sample in ((axes[0], titles[0], False),
                                   (axes[1], titles[1], True)):
        ax.grid(False)
        im = ax.imshow(np.zeros((20, 320)), origin="lower",
                       extent=(0.0, length, 0.0, bore),
                       cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
                       aspect="auto", interpolation="bilinear", zorder=2)
        _anim_tube_hardware(
            ax, length, bore=bore,
            sample=(0.80, 0.90) if with_sample else None,
            termination="anechoic", mics=mics,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.125, length + 0.10)
        ax.set_ylim(-0.150, 0.255)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        ims.append(im)
    axes[1].set_xlabel(T("Position along the tube [m]"), fontsize=9)
    tl_txt = axes[1].text(0.02, 0.205, "", ha="left", va="top",
                          fontsize=9, color="white", zorder=7,
                          bbox={"boxstyle": _ANIM_PILL_BOX,
                                "facecolor": "black", "alpha": 0.55,
                                "edgecolor": "none"})
    t_txt = fig.text(0.988, 0.93, "", ha="right", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)
    reveal = int(0.55 * len(times))

    def update(k: int) -> tuple[Any, ...]:
        ims[0].set_data(p_e[k])
        ims[1].set_data(p_s[k])
        tl_txt.set_text(
            T(f"TL = {tl:.1f} dB at {_VTUBE_F:.0f} Hz")
            if k >= reveal else ""
        )
        t_txt.set_text(T(f"$t$ = {times[k] * 1e3:5.1f} ms"))
        return (*ims, tl_txt, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_transmission_tube",
                 frames=len(times), fps=_TTUBE_FPS, gif_fps=8,
                 poster_ss=_TTUBE_VERDICT / _TTUBE_FPS)


def animate_standing_wave_tube(output_dir: str) -> None:
    """ISO 10534-2 impedance tube: the incident and reflected waves travel
    inside a drawn tube and their sum forms the standing-wave envelope; a
    rigid termination (deep nodes) is compared with a porous sample
    (shallow nodes) sampled by the two wall microphones."""
    from matplotlib.patches import Polygon, Rectangle

    T = _translate_str
    lam = 3.5                       # display wavelength inside the tube
    k = 2.0 * np.pi / lam
    x0, xs = 1.5, 11.0              # tube mouth and sample face
    x = np.linspace(x0, xs, 400)
    amp = 0.42                      # per-wave display amplitude
    # Mic positions from the sample face (ISO 10534-2: two flush wall mics
    # near the specimen, spacing < half a wavelength).
    mic_xi = (0.90 * lam, 0.65 * lam)
    cases = [
        (T("Rigid termination"), 1.0),
        (T("Porous sample"), 0.55),
    ]

    fig = _anim_figure()
    fig.suptitle(T("Standing wave in the impedance tube (ISO 10534-2)"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 1)
    panels: list[dict[str, Any]] = []
    for row, (title, refl) in enumerate(cases):
        ax = fig.add_subplot(gs[row])
        _schematic_axes(ax, (0.0, 14.2), (-1.75, 1.85), equal=True)
        ax.text(0.1, 1.7, title, ha="left", va="top", color=COLOR_FG,
                fontsize=10, fontweight="bold")
        # Tube walls, loudspeaker and termination
        for yw in (-1.0, 1.0):
            ax.plot([x0 - 0.1, xs + (0.8 if refl < 1.0 else 0.0)], [yw, yw],
                    color=COLOR_FG, lw=1.8)
        _draw_speaker(ax, x0 - 0.05, 0.0, size=1.55)
        if refl < 1.0:
            # porous specimen slab in front of the rigid backing
            ax.add_patch(Rectangle((xs, -1.0), 0.8, 2.0,
                                   facecolor=COLOR_TERTIARY, alpha=0.45,
                                   edgecolor=COLOR_FG, lw=1.0))
            wall_x = xs + 0.8
            ax.text(xs + 0.4, -1.15, T("sample"), ha="center", va="top",
                    color=COLOR_FG, fontsize=8)
        else:
            wall_x = xs
            ax.text(xs + 0.25, -1.15, T("rigid wall"), ha="center", va="top",
                    color=COLOR_FG, fontsize=8)
        ax.add_patch(Polygon(
            [(wall_x, -1.0), (wall_x, 1.0), (wall_x + 0.35, 1.0),
             (wall_x + 0.35, -1.0)], closed=True, facecolor="none",
            edgecolor=COLOR_FG, lw=1.2, hatch="///"))
        # Two flush wall microphones pointing down into the tube
        for j, xi in enumerate(mic_xi):
            xm = xs - xi
            _draw_mic(ax, xm, 1.42, direction=1, size=0.62, angle=-90.0)
            # Clear of the microphone body: the italic p's bowl starts at
            # the very top of the glyph box, so a label parked on the body
            # edge loses the arc that closes it.
            ax.text(xm, 1.83, f"$p_{j + 1}$", ha="center", va="bottom",
                    color=COLOR_FG, fontsize=9)
        # Waves: incident, reflected (fades in), their sum and the envelope
        (l_inc,) = ax.plot([], [], color=COLOR_PRIMARY, lw=1.2, alpha=0.85)
        (l_ref,) = ax.plot([], [], color=COLOR_TERTIARY, lw=1.2, alpha=0.0)
        (l_sum,) = ax.plot([], [], color=COLOR_SECONDARY, lw=2.2)
        env = amp * np.sqrt(1.0 + refl**2
                            + 2.0 * refl * np.cos(2.0 * k * (xs - x)))
        env_lines = [ax.plot(x, s * env, color=COLOR_FG, lw=1.0, ls="--",
                             alpha=0.0)[0] for s in (1.0, -1.0)]
        (mic_dots,) = ax.plot([], [], marker="o", ms=5, ls="none",
                              color=COLOR_SECONDARY)
        # left of the shared legend row, clear of the bottom panel's strip
        note = ax.text(2.5, -1.42, "", ha="center", va="top",
                       color=COLOR_FG, fontsize=9)
        readout = ax.text(13.9, 1.7, "", ha="right", va="top",
                          color=COLOR_FG, fontsize=9, family="monospace")
        panels.append({"refl": refl, "l_inc": l_inc, "l_ref": l_ref,
                       "l_sum": l_sum, "env": env, "env_lines": env_lines,
                       "mic_dots": mic_dots, "note": note,
                       "readout": readout})

    legend_ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    legend_ax.axis("off")
    for xl, color, lab in ((0.30, COLOR_PRIMARY, T("incident")),
                           (0.44, COLOR_TERTIARY, T("reflected")),
                           (0.58, COLOR_SECONDARY, T("sum $p(x, t)$")),
                           (0.72, COLOR_FG, T("envelope $|p(x)|$"))):
        legend_ax.plot([xl, xl + 0.025], [0.028, 0.028], color=color,
                       lw=2.0, ls="--" if color == COLOR_FG else "-",
                       transform=legend_ax.transAxes)
        legend_ax.text(xl + 0.032, 0.028, lab, ha="left", va="center",
                       color=COLOR_FG, fontsize=8.5,
                       transform=legend_ax.transAxes)

    f_disp = 0.55                   # display oscillation frequency [Hz]
    sweep = _ANIM_FRAMES - _ANIM_HOLD
    t_ref, t_env = 3.5, 7.0         # phase starts [s of clip time]

    def update(kf: int) -> tuple[Any, ...]:
        tc = min(kf, sweep - 1) / _ANIM_FPS
        ph = 2.0 * np.pi * f_disp * tc
        arts: list[Any] = []
        a_ref = float(np.clip((tc - t_ref) / 1.2, 0.0, 1.0))
        a_env = float(np.clip((tc - t_env) / 1.2, 0.0, 1.0))
        for pn in panels:
            refl = pn["refl"]
            # xi is the distance to the sample face; the incident wave
            # cos(ph + k*xi) travels toward +x (into the sample) and the
            # reflected wave cos(ph - k*xi) back toward the loudspeaker.
            xi = xs - x
            p_inc = amp * np.cos(ph + k * xi)
            p_ref = refl * amp * np.cos(ph - k * xi)
            pn["l_inc"].set_data(x, p_inc)
            pn["l_ref"].set_data(x, p_ref)
            pn["l_ref"].set_alpha(0.85 * a_ref)
            pn["l_sum"].set_data(x, p_inc + a_ref * p_ref)
            for line in pn["env_lines"]:
                line.set_alpha(0.8 * a_env)
            if a_env > 0.0:
                xm = np.array([xs - m for m in mic_xi])
                pim = amp * np.cos(ph + k * (xs - xm))
                prm = refl * amp * np.cos(ph - k * (xs - xm))
                pn["mic_dots"].set_data(xm, pim + a_ref * prm)
                pn["note"].set_text(
                    T("deep nodes") if refl >= 1.0 else T("shallow nodes"))
                pn["note"].set_alpha(a_env)
                pn["readout"].set_text(
                    T(f"$|R|$ = {refl:.2f}   "
                      f"$\\alpha$ = {1.0 - refl**2:.2f}"))
                pn["readout"].set_alpha(a_env)
            else:
                pn["mic_dots"].set_data([], [])
                pn["note"].set_text("")
                pn["readout"].set_text("")
            arts += [pn["l_inc"], pn["l_ref"], pn["l_sum"], pn["mic_dots"],
                     pn["note"], pn["readout"], *pn["env_lines"]]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_standing_wave_tube")
