#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The simulated wave fields and the clips that show them moving.

Every clip here is a 2D FDTD (or elastic FDTD) run: a barrier, a ground
effect, a SOFAR duct, an impedance or transmission tube, a diffuser, a slit
absorber, a plate junction. The simulation is the expensive part and is
memoised per process behind an ``lru_cache``, so the field builder and the
clip that renders it belong together: the four language x theme variants of a
clip are rendered off one computation of its field.
"""

import os
from functools import lru_cache
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .i18n import _LANG
from .materials import _METADIFFUSER_T1_ROWS
from .media import (
    _ANIM_DPI,
    _ANIM_FPS,
    _ANIM_FRAMES,
    _ANIM_HOLD,
    _ANIM_PILL_BOX,
    _FDTD_ANIM_FRAMES,
    _anim_figure,
    _render_clip,
    _translate_str,
)
from .schematics import _draw_mic, _draw_speaker, _grid_axes, _schematic_axes
from .theme import (
    _FILENAME_SUFFIX,
    CMAP_FIELD,
    COLOR_FG,
    COLOR_GRID,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    FIELD_INK,
    FIELD_STROKE,
)


@lru_cache(maxsize=1)
def _meshed_metadiffuser_ntff_levels() -> tuple[Any, Any]:
    """Far-field polar levels of the meshed Table-1 panel at 2 kHz.

    A dedicated continuous-wave FDTD run at dx = 0.5 mm (every slit, neck
    and cavity of the real panel resolved by at least four cells, unit-cell
    reflection phases within 2 degrees of a 0.25 mm run): a plane wave from
    just inside the top sponge drives the panel to steady state, a closed
    contour probe accumulates the 2 kHz pressure and normal-velocity
    phasors on the fly, and the 2D Kirchhoff-Helmholtz integral turns them
    into the far-field pattern. The incident wave extinguishes in the
    exterior integral, so no reference run is needed. Returns the polar
    angles [deg from the panel normal] and the levels re the peak [dB].
    """
    from phonometry.simulation import (
        FDTD2D,
        CWSource,
        PlaneWaveSource,
        far_field_from_contour,
    )

    dx, sponge, f0 = 0.0005, 60, _META_F0
    gap, marg, front = (round(m / dx) for m in (0.030, 0.010, 0.020))
    face_cells = round(5 * _META_PITCH / dx)
    lat = marg + gap + sponge
    nx = face_cells + 2 * lat
    r_face = sponge + gap + front
    slab = round(0.023 / dx)
    ny = r_face + slab + marg + gap + sponge
    mask = np.zeros((ny, nx), dtype=bool)
    mask[r_face:r_face + slab, lat:lat + face_cells] = True
    for n, (h, l_n, l_c, w_n, w_c) in enumerate(_METADIFFUSER_T1_ROWS):
        x_slit = (n + 0.12) * _META_PITCH
        c0s = lat + round(x_slit / dx)
        c1s = lat + round((x_slit + h * 1e-3) / dx)
        mask[r_face:r_face + round(0.02 / dx), c0s:c1s] = False
        for m in range(2):
            y_m = (m + 0.5) * 0.01
            x_neck = x_slit + h * 1e-3
            r0 = r_face + round((y_m - 0.5e-3 * w_n) / dx)
            r1 = r_face + round((y_m + 0.5e-3 * w_n) / dx)
            mask[r0:r1, c1s:lat + round((x_neck + l_n * 1e-3) / dx)] = False
            r0 = r_face + round((y_m - 0.5e-3 * w_c) / dx)
            r1 = r_face + round((y_m + 0.5e-3 * w_c) / dx)
            mask[r0:r1, lat + round((x_neck + l_n * 1e-3) / dx):
                 lat + round((x_neck + (l_n + l_c) * 1e-3) / dx)] = False
    # cfl 0.9 trims the step count; at 340 cells per wavelength the
    # numerical dispersion is negligible at any stable Courant number.
    sim = FDTD2D(343.0, dx, shape=(ny, nx), sponge_width=sponge, cfl=0.9,
                 obstacle_mask=mask)
    sim.add_source(PlaneWaveSource("down", CWSource(0, 0, f0).value,
                                   offset=sponge))
    probe = sim.add_contour_probe(lat - marg, lat + face_cells + marg - 1,
                                  r_face - front, r_face + slab + marg - 1,
                                  frequencies=[f0])
    sim.run(round(8e-3 / sim.dt))            # transient out (ramp + ring-up)
    probe.reset()
    sim.run(round(10.0 / f0 / sim.dt))       # ten whole periods of DFT
    angles = np.arange(-90.0, 90.1, 5.0)
    pattern = far_field_from_contour(
        probe.phasors(f0), angles - 90.0,
        origin=((lat + face_cells / 2.0) * dx, r_face * dx))
    magnitude = np.abs(pattern)
    levels = 20.0 * np.log10(np.maximum(magnitude / magnitude.max(), 1e-10))
    return angles, levels


def _poster_ss_for(webm: str) -> float | None:
    """Per-clip poster-time override; None keeps the end-of-hold default.

    The pillar-hall banner poster is grabbed mid-flight (simulation time
    6 ms, the front threading the middle of the hall) instead of the
    settled last frame, which for a single travelling packet is an almost
    empty field.
    """
    if _PILLAR_STEM in os.path.basename(webm):
        import fdtd2d
        dt = fdtd2d.FDTD2D(343.0, _PILLAR_DX, shape=(3, 8)).dt
        dt_frame = _PILLAR_EVERY * dt
        return float((6.0e-3 / dt_frame - _PILLAR_WARM) / _PILLAR_FPS)
    return None


# Room-modes timeline. Mesh: the highest carrier is the off-mode drive at
# 91.2 Hz (lambda = 3.76 m, lambda / 8 = 0.47 m) and the smallest geometric
# dimension is the 3.5 m room side (3.5 / 4 = 0.88 m), so the rule
# dx = min(0.88, 0.47) m allows 0.47 m; dx = 0.01 m sits 47 times finer,
# because the mode map has to be drawn, not just resolved.
# Flight: the source at (0.25, 0.25) m reaches the deepest reflector -- the
# opposite corner (5, 3.5) -- after 5.755 m and the farthest visible point
# (the near corner) 6.103 m later, so 11.859 m / 343 m/s x 1.2 = 41.5 ms.
# Sampling: 18 solver steps per frame = 222.6 us gives 49.3 frames per
# 91.2 Hz period (>= 48). The captured window keeps the committed clip's
# full 350 ms -- the flight floor is only a floor, and this clip needs
# 8.4 times it (3.5 amplitude time constants of the T60 = 0.7 s room,
# tau = T60 / 6.9077 = 101 ms) because the resonance has to be seen
# *growing* all the way into its settled mode map. 1 572 frames cover that
# window, played at 13/3 of the shared 20 fps -- exactly the factor by
# which the stride shrank from the committed clip's 78 solver steps per
# frame -- so the pacing stays the 19.30 ms of simulation per second of
# playback the committed clip had, and the clip runs 18.1 s (it ran 18 s).
_ROOM_EVERY = 18
_ROOM_FRAMES = 1572
#: 13/3 of the shared rate, matching the stride cut (see the note above).
_ROOM_FPS = _ANIM_FPS * 13.0 / 3.0


@lru_cache(maxsize=1)
def _room_mode_fields(
        n_frames: int = _ROOM_FRAMES) -> tuple[Any, Any, Any, float, float]:
    """Run the two FDTD room simulations once per process (all variants).

    Returns instantaneous-pressure frames, running-RMS frames (both float32,
    stacked ``(2, n_frames, ny, nx)``), the frame times, and the on-mode and
    off-mode drive frequencies. ``n_frames`` sets the captured window at the
    fixed ``_ROOM_EVERY`` capture stride (see the timeline note above).
    """
    import fdtd2d

    lx, ly, c0 = 5.0, 3.5, 343.0
    dx = 0.01                                    # 500 x 350 cells
    ny, nx = round(ly / dx), round(lx / dx)
    f_mode = 0.5 * c0 * float(np.hypot(2 / lx, 1 / ly))   # (2,1) ~ 84.3 Hz
    f_next = 0.5 * c0 * (2 / ly)                          # (0,2) = 98 Hz
    f_off = 0.5 * (f_mode + f_next)              # between resonances
    t60 = 0.7
    every = _ROOM_EVERY
    steps = every * n_frames
    p_all, r_all = [], []
    times = np.zeros(0)
    for f in (f_mode, f_off):
        sim = fdtd2d.FDTD2D(c0, dx, shape=(ny, nx), damping=6.9077 / t60)
        sim.add_source(fdtd2d.CWSource(ix=25, iy=25, frequency=f,
                                       ramp_cycles=2.0))
        # Running mean square with a two-period time constant: the pattern
        # (the mode map) builds up as the resonance settles.
        beta = float(np.exp(-sim.dt * f / 2.0))
        ms = np.zeros_like(sim.p)
        ps: list[Any] = []
        rs: list[Any] = []
        ts: list[float] = []
        for i in range(steps):
            sim.step()
            ms = beta * ms + (1.0 - beta) * sim.p**2
            if (i + 1) % every == 0 and len(ps) < n_frames:
                ps.append(sim.p[::2, ::2].astype(np.float32))
                rs.append(np.sqrt(ms[::2, ::2]).astype(np.float32))
                ts.append(sim.time)
        p_all.append(np.stack(ps))
        r_all.append(np.stack(rs))
        times = np.asarray(ts)
    return np.stack(p_all), np.stack(r_all), times, f_mode, f_off


def animate_fdtd_room_modes(output_dir: str) -> None:
    """On-mode vs off-mode CW drive in a rigid 5 x 3.5 m room (2D FDTD):
    on resonance the (2,1) standing-wave pattern grows until it dominates
    the field; off resonance the forced response stays weak and never
    organises into that nodal structure."""
    from matplotlib import patheffects

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    p_all, r_all, times, f_mode, f_off = _room_mode_fields()
    half = p_all.shape[1] // 2
    vmax_p = float(np.quantile(np.abs(p_all[0][half:]), 0.995))
    vmax_r = float(np.quantile(r_all[0][-1], 0.999))

    fig = _anim_figure()
    fig.suptitle(T("Room modes in a rigid 5 m × 3.5 m room (2D FDTD)"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2)
    titles = [T(f"On the (2,1) mode: {f_mode:.1f} Hz"),
              T(f"Off mode: {f_off:.1f} Hz")]
    ims: list[Any] = []
    for col in range(2):
        ax_p = fig.add_subplot(gs[0, col])
        ax_p.grid(False)
        im_p = ax_p.imshow(p_all[col][0], origin="lower",
                           extent=(0.0, 5.0, 0.0, 3.5), cmap=CMAP_FIELD,
                           vmin=-vmax_p, vmax=vmax_p, interpolation="bilinear")
        ax_p.set_title(titles[col], fontsize=10, fontweight="bold")
        ax_p.plot([0.25], [0.25], marker="o", ms=5, color=COLOR_TERTIARY,
                  markeredgecolor=FIELD_STROKE, markeredgewidth=0.8)
        ax_p.text(0.45, 0.22, T("source"), ha="left", va="center",
                  color=FIELD_INK, fontsize=7.5, path_effects=outline)
        ax_p.tick_params(labelsize=7, labelbottom=False)
        ax_r = fig.add_subplot(gs[1, col])
        ax_r.grid(False)
        im_r = ax_r.imshow(r_all[col][0], origin="lower",
                           extent=(0.0, 5.0, 0.0, 3.5), cmap="magma",
                           vmin=0.0, vmax=vmax_r, interpolation="bilinear")
        ax_r.tick_params(labelsize=7)
        ax_r.set_xlabel("x [m]", fontsize=8)
        if col == 0:
            ax_p.set_ylabel(T("instantaneous p(x, y)"), fontsize=9)
            ax_r.set_ylabel(T("RMS pressure (mode map)"), fontsize=9)
            for xn in (1.25, 3.75):
                ax_r.axvline(xn, color="white", ls="--", lw=1.0, alpha=0.75)
            ax_r.axhline(1.75, color="white", ls="--", lw=1.0, alpha=0.75)
            ax_r.text(0.12, 3.3, T("nodal lines (2,1)"), color="white",
                      fontsize=8, ha="left", va="top")
        else:
            ax_p.tick_params(labelleft=False)
            ax_r.tick_params(labelleft=False)
            ax_r.text(0.12, 3.3, T("same color scale"), color="white",
                      fontsize=8, ha="left", va="top")
        ims += [im_p, im_r]
    t_txt = fig.text(0.985, 0.955, "", ha="right", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)

    def update(k: int) -> tuple[Any, ...]:
        for col in range(2):
            ims[2 * col].set_data(p_all[col][k])
            ims[2 * col + 1].set_data(r_all[col][k])
        t_txt.set_text(T(f"t = {times[k] * 1000.0:3.0f} ms"))
        return (*ims, t_txt)

    # Own frame budget and rate (see the _ROOM_FRAMES timeline note):
    # 1 572 frames at 49.3 per acoustic period, played at 260/3 fps. The
    # GitHub GIF samples this long clip at a reduced rate so the
    # palette-quantized file stays well under 4 MB.
    _render_clip(fig, update, output_dir, "anim_fdtd_room_modes",
                 frames=int(p_all.shape[1]), fps=_ROOM_FPS, gif_fps=5)


# ---------------------------------------------------------------------------
# FDTD wave-field clips (barrier, ground effect, SOFAR duct, diffuser).
# Every simulation runs once per process behind an lru_cache and its frames
# are re-rendered for the four language x theme variants; each clip keeps
# >= 48 captured frames per acoustic period of its highest carrier, raising
# its playback rate with the capture stride so the wavefronts read as
# continuous motion at the pacing the clip was composed for.
# ---------------------------------------------------------------------------


def _fdtd_cw_capture(
        sim: Any, frequency: float, every: int, n_frames: int,
        decimate: int = 2) -> tuple[Any, Any, Any, Any]:
    """Drive a CW simulation and capture instantaneous + running-RMS frames.

    The running mean square uses a two-period time constant, so the RMS map
    (the lobe/shadow pattern) builds up as the field settles. Returns the
    float32 frame stacks (decimated), the frame times and the final
    full-resolution RMS map for physics probes.
    """
    beta = float(np.exp(-sim.dt * frequency / 2.0))
    ms = np.zeros_like(sim.p)
    ps: list[Any] = []
    rs: list[Any] = []
    ts: list[float] = []
    for _ in range(every * n_frames):
        sim.step()
        ms = beta * ms + (1.0 - beta) * sim.p**2
        if sim.n % every == 0 and len(ps) < n_frames:
            ps.append(sim.p[::decimate, ::decimate].astype(np.float32))
            rs.append(np.sqrt(ms[::decimate, ::decimate]).astype(np.float32))
            ts.append(sim.time)
    return np.stack(ps), np.stack(rs), np.asarray(ts), np.sqrt(ms)


def _rms_to_db(rms_frames: Any, *, floor: float = -40.0) -> Any:
    """RMS frame stack -> dB re the final frame's maximum, clipped at floor."""
    ref = float(rms_frames[-1].max())
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(rms_frames / ref)
    return np.clip(db, floor, 0.0).astype(np.float32)


# --- weak-field display gain ----------------------------------------------
# A transmitted or shadowed field can sit tens of dB below the incident one
# that sets the colour scale, and a single linear diverging ramp cannot show
# both: give the quiet side half the ramp and the loud side saturates into a
# flat slab, hiding the wavefronts the clip exists to show. Compressing the
# ramp instead (signed log, asinh, gamma) has the same ceiling -- a factor of
# 250 between the two sides is a factor of 250 whatever the transfer curve --
# and it costs the loud side its shape as well.
# The treatment used across the clips is therefore a per-region display gain:
# the quiet region is drawn amplified by a fixed factor picked from the field
# itself, so each region uses the full ramp, the *shape* of both fields
# survives, and the panel states the factor (and the dB it stands for) in
# writing. The measured level annotations stay the physical ones, so nothing
# on screen is silently rescaled.
# The ladder is coarse on purpose (1-1.5-2-3-5-7 per decade): a round factor is
# readable in an annotation and stays put when the field is re-simulated. It
# starts at 5 (14 dB) so a region that already uses a fifth of the ramp is
# left alone -- the caveat such a panel would have to print costs more than
# the contrast it would buy.
_GAIN_STEPS = (5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0,
               200.0, 300.0, 500.0, 700.0, 1000.0)


def _weak_field_gain(weak: Any, vmax: float, *,
                     quantile: float = 0.999) -> float:
    """Rounded display gain that lifts a weak field onto a readable colour.

    ``weak`` is the quiet region of the field (any shape), ``vmax`` the
    colour-scale half-range the loud region set. The gain is the largest rung
    of :data:`_GAIN_STEPS` at or under the factor that makes the ``quantile``
    amplitude of ``weak`` fill the ramp, so the quiet region gets the whole
    colour scale and at most a thousandth of its samples saturate. A region
    already within 14 dB of the ramp is returned at 1.0 (no gain, and so no
    annotation).
    """
    peak = float(np.quantile(np.abs(weak), quantile))
    if not np.isfinite(peak) or peak <= 0.0:
        return 1.0
    raw = float(vmax) / peak
    return max((g for g in _GAIN_STEPS if g <= raw), default=1.0)


def _gain_note(region: str, gain: float) -> str:
    """English annotation for a region drawn with a display gain.

    Empty for a unit gain (nothing to declare). The dB equivalent rides along
    so the reader can put the compression next to the level annotations.
    """
    if gain <= 1.0:
        return ""
    return f"{region} drawn ×{gain:g} (+{20.0 * np.log10(gain):.0f} dB)"


_BARRIER_FREQS = (100.0, 500.0)
# Receiver low over the ground: there the ground bounce reinforces both
# frequencies almost equally (tiny path difference), so the insertion-loss
# contrast is pure diffraction instead of ground-interference lobes.
_BARRIER_RECEIVER = (9.0, 0.5)
# Barrier timeline. Mesh: the highest carrier is 500 Hz (lambda = 0.686 m,
# lambda / 8 = 86 mm) and the smallest resolved geometric dimension is the
# 2.5 m screen height (2.5 / 4 = 0.63 m), so the rule dx = min(0.63, 0.086)
# m allows 86 mm; dx = 20 mm sits 4.3 times finer. (The screen is three
# cells thick, but its thickness is not an acoustic dimension: a rigid
# knife edge only has to be impermeable, which the 10^6:1 density contrast
# makes it, and the diffraction the clip shows is set by the edge height.)
# Flight: the source at (2.0, 0.5) m reaches the deepest reflector -- the
# rigid ground 0.5 m below -- and from there the farthest visible point,
# the top right corner (12, 7), is 12.207 m away, so 12.707 m / 343 m/s
# x 1.2 = 44.45 ms.
# Sampling: every solver step is captured, 24.74 us apart, i.e. 80.9 frames
# per 500 Hz period (>= 48; the grid offers nothing between this and the
# 40.4 of a two-step stride). 1 800 frames cover the 44.53 ms window,
# played at 120 fps -- six times the 20 fps of the committed clip, exactly
# the factor by which the stride shrank from its 6 solver steps per frame
# -- so the pacing stays the 2.969 ms of simulation per second of playback
# the committed clip had, and the clip runs 15 s (the old 18 s clip spent
# its extra seconds past the flight, on the 53.4 ms window trimmed here).
_BARRIER_EVERY = 1
_BARRIER_FRAMES = 1800
_BARRIER_FPS = 120


@lru_cache(maxsize=1)
def _barrier_fields(
        n_frames: int = _BARRIER_FRAMES) -> tuple[Any, Any, Any, Any]:
    """Two CW barrier-diffraction runs (low/high frequency), cached.

    A 12 m x 7 m half-space over rigid ground with a thin rigid barrier
    (5 000:1 density contrast) 2.5 m tall at x = 5.5 m; absorbing sponges
    on the two sides and the sky. Each frequency also gets a barrier-free
    reference run over the same ground, so the receiver annotation is a
    true insertion loss (patch-averaged around the receiver to keep the
    ground-interference lobes out of the number). After the captured clip
    both runs keep stepping, uncaptured, into genuine steady state and the
    insertion loss is an exact RMS over the final two full periods of that
    extended window (see the settle comment below). Returns the
    instantaneous-pressure frames, RMS maps in dB re each run's own
    maximum, the frame times and the per-frequency insertion losses.
    """
    import fdtd2d

    c0, dx = 343.0, 0.02
    ny, nx = 350, 600                      # 7 m x 12 m
    rho = np.full((ny, nx), 1.2)
    bx = round(5.5 / dx)              # thin barrier: 3 cells = 6 cm
    rho[:round(2.5 / dx), bx:bx + 3] = 1.2e6
    every = _BARRIER_EVERY
    # Receiver patch: 0.6 m x 0.6 m around the shadow-zone receiver,
    # energy-averaged so residual interference fringes average out.
    rx, ry = _BARRIER_RECEIVER
    patch = (slice(int((ry - 0.3) / dx), int((ry + 0.3) / dx) + 1),
             slice(int((rx - 0.3) / dx), int((rx + 0.3) / dx) + 1))
    p_all, db_all, ils = [], [], []
    times = np.zeros(0)
    for f in _BARRIER_FREQS:
        # Row 0 is the ground (displayed at the bottom via origin="lower"),
        # so the absorbing sides are left/right and the *high* rows ("bottom"
        # in the imshow-origin naming of fdtd2d); the ground stays rigid.
        rms_patch = []
        for rho_map in (rho, None):
            sim = fdtd2d.FDTD2D(c0, dx, shape=(ny, nx),
                                rho=1.2 if rho_map is None else rho_map,
                                sponge_width=40,
                                sponge_sides=("left", "right", "bottom"))
            sim.add_source(fdtd2d.CWSource(ix=100, iy=25, frequency=f,
                                           ramp_cycles=2.0))
            if rho_map is not None:
                ps, rs, times, _ = _fdtd_cw_capture(sim, f, every, n_frames)
                p_all.append(ps)
                db_all.append(_rms_to_db(rs))
            else:
                # Barrier-free reference: same steps, no frames captured.
                for _ in range(every * n_frames):
                    sim.step()
            # The clip ends at 44.5 ms, but at 100 Hz the field behind the
            # barrier has not settled by then: after the 20 ms source ramp
            # the diffracted and ground-bounced paths over the edge keep
            # building the receiver level for several more periods. Step
            # both runs on, uncaptured, to ~113 ms -- where the measured
            # insertion loss sits within 0.05 dB of its value 30 ms later
            # -- and measure an exact RMS over the last two full periods,
            # so neither run's transient biases the published number.
            period = round(1.0 / (f * sim.dt))
            settle = round(0.113 / sim.dt) - sim.n
            acc = np.zeros_like(sim.p)
            for i in range(settle):
                sim.step()
                if i >= settle - 2 * period:
                    acc += sim.p**2
            rms = np.sqrt(acc / (2 * period))
            rms_patch.append(float(np.sqrt(np.mean(rms[patch] ** 2))))
        ils.append(20.0 * float(np.log10(rms_patch[1] / rms_patch[0])))
    return np.stack(p_all), np.stack(db_all), times, tuple(ils)


def animate_fdtd_barrier(output_dir: str) -> None:
    """A point source behind a thin rigid barrier on reflecting ground
    (2D FDTD), at 100 Hz and 500 Hz side by side: the long wavelength
    diffracts around the edge and fills the shadow zone, the short one is
    cast into a deep, clean shadow -- why barriers fail at low frequency."""
    from matplotlib import patheffects
    from matplotlib.patches import Rectangle

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    p_all, db_all, times, ils = _barrier_fields()
    half = p_all.shape[1] // 2
    lam = tuple(343.0 / f for f in _BARRIER_FREQS)
    rx, ry = _BARRIER_RECEIVER

    fig = _anim_figure()
    fig.suptitle(T("Barrier diffraction into the shadow zone (2D FDTD)"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2)
    titles = [
        T(f"Low frequency: {_BARRIER_FREQS[0]:.0f} Hz "
          f"(λ ≈ {lam[0]:.1f} m)"),
        T(f"High frequency: {_BARRIER_FREQS[1]:.0f} Hz "
          f"(λ ≈ {lam[1]:.2f} m)"),
    ]
    verdicts = [T("diffraction fills the shadow"), T("deep, clean shadow")]
    ims: list[Any] = []
    il_txts: list[Any] = []
    for col in range(2):
        vmax = float(np.quantile(np.abs(p_all[col][half:]), 0.995))
        ax_p = fig.add_subplot(gs[0, col])
        ax_r = fig.add_subplot(gs[1, col])
        im_p = ax_p.imshow(p_all[col][0], origin="lower",
                           extent=(0.0, 12.0, 0.0, 7.0), cmap=CMAP_FIELD,
                           vmin=-vmax, vmax=vmax, interpolation="bilinear")
        im_r = ax_r.imshow(db_all[col][0], origin="lower",
                           extent=(0.0, 12.0, 0.0, 7.0), cmap="magma",
                           vmin=-40.0, vmax=0.0, interpolation="bilinear")
        ax_p.set_title(titles[col], fontsize=10, fontweight="bold")
        for ax in (ax_p, ax_r):
            ax.grid(False)
            ax.set_ylim(-0.5, 7.0)
            ax.add_patch(Rectangle((0.0, -0.5), 12.0, 0.5,
                                   facecolor=COLOR_GRID, edgecolor=COLOR_FG,
                                   lw=0.8, hatch="///"))
            # Theme-independent bar: mid-gray with a white edge stays
            # visible on the near-white RdBu row and the black magma row.
            ax.add_patch(Rectangle((5.44, 0.0), 0.18, 2.5,
                                   facecolor="#707070", edgecolor="white",
                                   lw=0.6))
            ax.tick_params(labelsize=7)
        ax_p.tick_params(labelbottom=False)
        ax_r.set_xlabel("x [m]", fontsize=8)
        ax_p.plot([2.0], [0.5], marker="o", ms=5, color=COLOR_TERTIARY,
                  markeredgecolor=FIELD_STROKE, markeredgewidth=0.8)
        ax_p.text(2.25, 0.55, T("source"), ha="left", va="center",
                  color=FIELD_INK, fontsize=7.5, path_effects=outline)
        ax_p.text(5.53, 2.7, T("barrier"), ha="center", va="bottom",
                  color=FIELD_INK, fontsize=7.5, path_effects=outline)
        ax_p.text(9.2, 1.1, T("shadow zone"), ha="center", va="center",
                  color=FIELD_INK, fontsize=7.5, path_effects=outline)
        ax_r.text(11.7, 6.4, verdicts[col], ha="right", va="top",
                  color="white", fontsize=8)
        ax_r.plot([rx], [ry], marker="o", ms=5, color="white",
                  markeredgecolor="black", markeredgewidth=0.8)
        il_txts.append(
            ax_r.text(rx, ry + 0.45, "", ha="center", va="bottom",
                      color="white", fontsize=7.5))
        if col == 0:
            ax_p.set_ylabel(T("instantaneous p(x, y)"), fontsize=9)
            ax_r.set_ylabel(T("RMS level [dB re panel max]"), fontsize=8)
            ax_p.text(0.25, -0.27, T("rigid ground"), ha="left",
                      va="center", color=COLOR_FG, fontsize=6.5,
                      bbox={"boxstyle": "round,pad=0.2",
                            "facecolor": fig.get_facecolor(),
                            "edgecolor": "none"})
        else:
            ax_p.tick_params(labelleft=False)
            ax_r.tick_params(labelleft=False)
            ax_r.text(0.3, 5.45, T("each panel on its own dB scale"),
                      color="white", fontsize=7, ha="left", va="top")
        ims += [im_p, im_r]
    # Top-left margin: the field panels run their x-axis (ticks + "x [m]")
    # to the very bottom-right corner, so a bottom readout collides with the
    # tick labels; the top-left stays clear of the centred column titles.
    t_txt = fig.text(0.012, 0.985, "", ha="left", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)

    def update(k: int) -> tuple[Any, ...]:
        for col in range(2):
            ims[2 * col].set_data(p_all[col][k])
            ims[2 * col + 1].set_data(db_all[col][k])
            # The measured insertion loss appears once the field has
            # actually reached and settled at the receiver, so the number
            # never precedes its cause.
            il_txts[col].set_text(
                T(f"insertion loss {ils[col]:.0f} dB")
                if times[k] >= 0.032 else "")
        t_txt.set_text(T(f"t = {times[k] * 1000.0:4.1f} ms"))
        return (*ims, *il_txts, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_barrier",
                 frames=int(p_all.shape[1]), fps=_BARRIER_FPS, gif_fps=5)


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
    ax_p.text(13.7, -0.95, T("rigid ground"), ha="right", va="center",
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
    ax_l.legend(fontsize=7, loc="center right")
    # The strip above 0 dB is data-free, so the closing caption fits there.
    # The longer Spanish verdict spans the whole panel at 7.5 pt, so it drops
    # a step to keep a margin on both sides.
    verdict_txt = ax_l.text(0.5, 0.975, "", transform=ax_l.transAxes,
                            ha="center", va="top", color=COLOR_FG,
                            fontsize=7.5 if _LANG == "en" else 6.5,
                            fontweight="bold")
    # Bottom-left corner: the arc panel's wide x-label owns bottom-right.
    t_txt = fig.text(0.015, 0.02, "", ha="left", va="bottom",
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


_DUCT_AXIS = 400.0                       # channel-axis depth [m]
_DUCT_SRC_DEPTHS = (400.0, 150.0)        # on the axis / near the surface


def _duct_profile(z: Any) -> Any:
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
    ny, nx = 400, 1200                     # 800 m depth x 2400 m range
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
        sim.add_source(fdtd2d.GaussianPulse(ix=100, iy=iy, width=width,
                                            t0=4.0 * width + offset,
                                            amplitude=-1.0))
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
    ref = float(np.quantile(energy_maps[:, :, nx // 4:], 0.999))
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
    verdict frame shows the whole path history."""
    from matplotlib import patheffects

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    p_all, e_db, times, z, c_prof, _ = _ducting_fields()
    half = p_all.shape[1] // 2
    vmax = float(np.quantile(np.abs(p_all[:, :half]), 0.999))

    fig = _anim_figure()
    fig.suptitle(T("SOFAR channel: sound trapped by the c(z) minimum "
                   "(2D FDTD)"), fontweight="bold")
    gs = fig.add_gridspec(2, 2, width_ratios=[0.22, 1.0])
    titles = [T("Source on the channel axis (depth 400 m)"),
              T("Source near the surface (depth 150 m)")]
    verdicts = [T("trapped: wavefronts bend back to the axis"),
                T("leaks: energy escapes the channel")]
    extent = (0.0, 2400.0, 800.0, 0.0)
    ims: list[Any] = []
    ims_e: list[Any] = []
    v_txts: list[Any] = []
    for row, depth in enumerate(_DUCT_SRC_DEPTHS):
        ax_c = fig.add_subplot(gs[row, 0])
        _grid_axes(ax_c)
        ax_c.plot(c_prof, z, color=COLOR_PRIMARY, lw=1.6)
        ax_c.plot([float(np.interp(depth, z, c_prof))], [depth],
                  marker="o", ms=5, color=COLOR_TERTIARY,
                  markeredgecolor="white", markeredgewidth=0.8)
        ax_c.axhline(_DUCT_AXIS, color=COLOR_FG, ls="--", lw=0.9, alpha=0.6)
        ax_c.set_ylim(800.0, 0.0)
        ax_c.set_xlim(1460.0, 1750.0)
        ax_c.set_xticks([1480.0, 1730.0])
        ax_c.set_ylabel(T("Depth [m]"), fontsize=8)
        ax_c.tick_params(labelsize=6)
        if row == 1:
            ax_c.set_xlabel(T("c(z) [m/s]"), fontsize=7)

        ax_f = fig.add_subplot(gs[row, 1])
        ax_f.grid(False)
        im = ax_f.imshow(p_all[row][0], origin="upper", extent=extent,
                         cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
                         aspect="auto", interpolation="bilinear")
        # The verdict overlay: fades in over the last seconds (and the
        # poster frame), replacing the instantaneous wavefronts with the
        # time-integrated energy paths.
        im_e = ax_f.imshow(e_db[row], origin="upper", extent=extent,
                           cmap="magma", vmin=-20.0, vmax=0.0,
                           aspect="auto", interpolation="bilinear",
                           alpha=0.0, zorder=2.5)
        ax_f.set_title(titles[row], fontsize=10, fontweight="bold")
        ax_f.axhline(_DUCT_AXIS, color="#888888", ls="--", lw=0.9,
                     alpha=0.8, zorder=3)
        ax_f.plot([200.0], [depth], marker="o", ms=5, color=COLOR_TERTIARY,
                  markeredgecolor=FIELD_STROKE, markeredgewidth=0.8,
                  zorder=4)
        ax_f.text(240.0, depth - 25.0, T("source"), ha="left", va="bottom",
                  color=FIELD_INK, fontsize=7.5, path_effects=outline,
                  zorder=4)
        # A translucent dark pill keeps this label legible once the bright
        # magma energy overlay fades in over the channel axis (a plain white
        # stroke washes out against the near-white high-energy region).
        ax_f.text(2360.0, 425.0, T("channel axis (c minimum)"), ha="right",
                  va="top", color="white", fontsize=7, zorder=4,
                  bbox={"boxstyle": "round,pad=0.2", "facecolor": "black",
                        "alpha": 0.45, "edgecolor": "none"})
        v_txt = ax_f.text(60.0, 770.0, "", ha="left", va="bottom",
                          color=FIELD_INK, fontsize=8, path_effects=outline,
                          zorder=4)
        ax_f.tick_params(labelsize=7, labelleft=False)
        if row == 0:
            ax_f.tick_params(labelbottom=False)
        else:
            ax_f.set_xlabel(T("Range [m]"), fontsize=8)
        ims.append(im)
        ims_e.append(im_e)
        v_txts.append(v_txt)
    # Right margin, below the suptitle: the lower field panel carries the
    # "Range [m]" x-axis to the bottom-right corner, so a bottom readout
    # collides with its tick labels; this spot clears both the tick labels
    # and the (long) centred suptitle.
    t_txt = fig.text(0.988, 0.90, "", ha="right", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)
    reveal = int(0.83 * p_all.shape[1])    # ~17.5 s: pulse has crossed
    captions_on = int(0.38 * p_all.shape[1])   # first refocus is visible

    def update(k: int) -> tuple[Any, ...]:
        alpha = min(1.0, max(0.0, (k - reveal) / 12.0))
        for row in range(2):
            ims[row].set_data(p_all[row][k])
            ims_e[row].set_alpha(alpha)
            v_txts[row].set_text(verdicts[row] if k >= captions_on else "")
        t_txt.set_text(T(f"t = {times[k]:5.2f} s"))
        return (*ims, *ims_e, *v_txts, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_ducting",
                 frames=int(p_all.shape[1]), fps=_DUCT_FPS, gif_fps=5)


#: Virtual-tube sample: the equivalent fluid of the solver cross-checks
#: (slower, denser, lossy; k = (omega - j sigma)/c at the real rho c).
_VTUBE_C2, _VTUBE_RHO2, _VTUBE_SIGMA = 0.6 * 343.0, 3.6, 600.0
_VTUBE_DX = 0.0025
_VTUBE_F = 850.0                          # CW inside the 100 mm plane range


def _anim_speaker(ax: Any, x0: float, y_mid: float, bore: float, *,
                  tip_inset: float | None = None,
                  label_y: float | None = None) -> None:
    """Drive loudspeaker of the FDTD tube/duct clips: magnet block plus
    cone, the cone tip on the ``x0`` plane, centred on ``y_mid`` for the
    given ``bore``. The tip stops ``tip_inset`` short of each bore edge
    (3 % of the bore when omitted); a "loudspeaker" caption is drawn at
    ``label_y`` when given (``None`` skips it)."""
    from matplotlib.patches import Polygon, Rectangle

    if tip_inset is None:
        tip_inset = 0.03 * bore
    magnet_w, cone_w = 0.05, 0.045
    ax.add_patch(Rectangle((x0 - magnet_w - cone_w, y_mid - 0.32 * bore),
                           magnet_w, 0.64 * bore, facecolor="#9a9a9a",
                           edgecolor=COLOR_FG, linewidth=0.8, zorder=4))
    ax.add_patch(Polygon(
        [(x0 - cone_w, y_mid - 0.20 * bore),
         (x0 - cone_w, y_mid + 0.20 * bore),
         (x0, y_mid + 0.5 * bore - tip_inset),
         (x0, y_mid - 0.5 * bore + tip_inset)],
        closed=True, facecolor="#e8b98a", edgecolor=COLOR_FG,
        linewidth=0.8, zorder=4))
    if label_y is not None:
        ax.text(x0 - magnet_w - cone_w, label_y, "loudspeaker",
                ha="left", va="top", fontsize=7.5)


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
    env_base, env_h, env_max = 0.24, 0.34, 2.2
    fig = _anim_figure()
    fig.suptitle(T("The virtual impedance tube: standing waves read the "
                   "absorption (2D FDTD)"), fontweight="bold")
    axes = fig.subplots(2, 1, sharex=True)
    titles = [T("Rigid end: deep minima, |r| ~ 1"),
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
        ax.text(0.005, env_base + 0.015, "|p| envelope", fontsize=7.5,
                ha="left", va="bottom", color=COLOR_FG, alpha=0.8)
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
            T(f"alpha = {alpha:.2f} at {_VTUBE_F:.0f} Hz")
            if k >= reveal else ""
        )
        t_txt.set_text(T(f"t = {times[kf] * 1e3:5.1f} ms"))
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
#: Frame the closing hold (and with it the deferred-loading poster) freezes
#: on: t = 3.2 ms, where the reflected and transmitted halves are fully
#: separated and both still inside the tube. Both ends are anechoic, so by
#: the end of the flight the packets have been absorbed and the tube is
#: quiet -- a correct last frame, but an empty verdict. The time readout
#: follows the frame back, so nothing on screen contradicts anything else.
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
        while len(ps) < n_frames:          # end hold on the verdict frame
            ps.append(ps[_TTUBE_VERDICT])
            ts.append(ts[_TTUBE_VERDICT])
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
        t_txt.set_text(T(f"t = {times[k] * 1e3:5.1f} ms"))
        return (*ims, tl_txt, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_transmission_tube",
                 frames=len(times), fps=_TTUBE_FPS, gif_fps=8)


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
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2)
    titles = [T("Flat rigid panel"), T("Schroeder diffuser (QRD, N = 7)")]
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
        ax_t.set_title(titles[col], fontsize=10, fontweight="bold")
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
        ax_s.set_xlabel("x [m]", fontsize=8)
        ax_t.text(3.0, 3.6, T("incident plane wavefront"), ha="center",
                  va="bottom", color=FIELD_INK, fontsize=7.5,
                  path_effects=outline)
        ax_t.annotate("", xy=(3.0, 3.1), xytext=(3.0, 3.55),
                      arrowprops={"arrowstyle": "-|>", "color": FIELD_INK,
                                  "lw": 1.2})
        ax_s.plot(arc_x, arc_y, ls=":", color="white", lw=0.9, alpha=0.65)
        ax_s.text(3.0, 0.15, beams[col], ha="center", va="bottom",
                  color="white", fontsize=7.5)
        d_txt = ax_s.text(5.45, 3.82, "", ha="right", va="top",
                          color="white", fontsize=8.5, fontweight="bold")
        if col == 0:
            ax_t.set_ylabel(T("sound field p"), fontsize=9)
            ax_s.set_ylabel(T("scattered field (total − incident)"),
                            fontsize=8)
            ax_s.text(0.78, 2.05, T("receiver arc"), ha="left", va="bottom",
                      color="white", fontsize=6.5, rotation=64.0)
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
    reveal = int(0.8 * tot_all.shape[1])   # arc energy has settled by here

    def update(k: int) -> tuple[Any, ...]:
        for col in range(2):
            ims[2 * col].set_data(tot_all[col][k])
            ims[2 * col + 1].set_data(trail_db[col][k])
            d_txts[col].set_text(
                T(f"diffusion coefficient d = {d_coef[col]:.2f}")
                if k >= reveal else "")
        t_txt.set_text(T(f"t = {times[k] * 1000.0:4.1f} ms"))
        return (*ims, *d_txts, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_diffusion",
                 frames=int(tot_all.shape[1]), fps=_DIFF_FPS, gif_fps=5)


_META_DX = 0.00025
_META_NY, _META_NX = 4800, 6400
_META_XL, _META_FACE = 0.45, 0.36
_META_PITCH, _META_PERIODS = 0.07, 2
_META_F0 = 2000.0


def _metadiffuser_panel_mask(rho: Any) -> None:
    """Carve the Table-1 metadiffuser (two periods) into a dense slab.

    The slab spans the panel depth L = 2 cm plus a 3 mm back wall under
    the face line; each well is a vertical slit from the face with its two
    resonators shelved sideways into the septum, the same layout the
    to-scale drawing uses (slit at 0.12 d into the cell, lattice a = L/2).
    """
    dx, y1 = _META_DX, _META_FACE
    rows = _METADIFFUSER_T1_ROWS
    depth, back = 0.02, 0.003
    rho[round((y1 - depth - back) / dx):round(y1 / dx),
        round(_META_XL / dx):round((_META_XL + _META_PERIODS * 5
                                    * _META_PITCH) / dx)] = 1.2e6
    lattice = depth / 2
    for period in range(_META_PERIODS):
        for n, (h, ln, lc, wn, wc) in enumerate(rows):
            x0 = _META_XL + (period * 5 + n) * _META_PITCH
            x_slit = x0 + 0.12 * _META_PITCH
            c0s, c1s = round(x_slit / dx), round((x_slit + h * 1e-3) / dx)
            rho[round((y1 - depth) / dx):round(y1 / dx), c0s:c1s] = 1.2
            for m in range(2):
                y_m = y1 - depth + (2 - m - 0.5) * lattice
                x_neck = x_slit + h * 1e-3
                r0 = round((y_m - 0.5 * wn * 1e-3) / dx)
                r1 = round((y_m + 0.5 * wn * 1e-3) / dx)
                rho[r0:r1, c1s:round((x_neck + ln * 1e-3) / dx)] = 1.2
                r0 = round((y_m - 0.5 * wc * 1e-3) / dx)
                r1 = round((y_m + 0.5 * wc * 1e-3) / dx)
                rho[r0:r1, round((x_neck + ln * 1e-3) / dx):
                    round((x_neck + (ln + lc) * 1e-3) / dx)] = 1.2


def _meta_qrd_wells() -> list[tuple[float, float, float]]:
    """QRD well openings (x0, x1, depth) matched to the metadiffuser.

    The same N = 5 residue order as the Table-1 metadiffuser
    (s = 1, 4, 4, 1, 0), designed at 500 Hz: depths s lambda0 / (2 N) up
    to 27.4 cm, 65 mm wells split by 5 mm fins on the 7 cm pitch.
    """
    unit = (343.0 / 500.0) / 10.0
    wells: list[tuple[float, float, float]] = []
    for period in range(_META_PERIODS):
        for n, s_n in enumerate((1, 4, 4, 1, 0)):
            x0 = _META_XL + (period * 5 + n) * _META_PITCH
            wells.append((x0 + 0.005, x0 + _META_PITCH, s_n * unit))
    return wells


def _meta_rho(kind: str) -> Any:
    """Density map of one run: ``flat``, ``qrd``, ``meta`` or ``ref``."""
    dx, ny, nx = _META_DX, _META_NY, _META_NX
    y1 = _META_FACE
    x_r = _META_XL + _META_PERIODS * 5 * _META_PITCH
    rho = np.full((ny, nx), 1.2)
    if kind == "qrd":
        rho[round(0.05 / dx):round(y1 / dx),
            round(_META_XL / dx):round(x_r / dx)] = 1.2e6
        for wx0, wx1, d in _meta_qrd_wells():
            if d > 0.0:
                rho[round((y1 - d) / dx):round(y1 / dx),
                    round(wx0 / dx):round(wx1 / dx)] = 1.2
    elif kind == "meta":
        _metadiffuser_panel_mask(rho)
    elif kind == "flat":
        # A flat rigid slab with the metadiffuser's exact silhouette, so
        # the fans can only come from the slits, not from the outline.
        rho[round((y1 - 0.023) / dx):round(y1 / dx),
            round(_META_XL / dx):round(x_r / dx)] = 1.2e6
    return rho


def _meta_taper() -> Any:
    """Lateral cosine taper of the incident packet (free-field edges).

    The wavefront is flat over the panels and dies smoothly well before
    the lateral sponges, so the exterior boundaries only ever absorb
    outgoing scattered waves and the incident front stays plane: no edge
    arcs in the total field, and the reference run subtracts exactly.
    """
    x = (np.arange(_META_NX) + 0.5) * _META_DX
    x0, x1, x2, x3 = 0.10, 0.30, 1.30, 1.50
    w = np.zeros(_META_NX)
    rise = (x >= x0) & (x < x1)
    w[rise] = 0.5 - 0.5 * np.cos(np.pi * (x[rise] - x0) / (x1 - x0))
    w[(x >= x1) & (x <= x2)] = 1.0
    fall = (x > x2) & (x <= x3)
    w[fall] = 0.5 + 0.5 * np.cos(np.pi * (x[fall] - x2) / (x3 - x2))
    return w


_META_FRAMES = 600   # 6.13 ms / 10.2 us per frame (49 frames per period)


def _meta_pair_worker(kind: str, n_frames: int) -> tuple[Any, Any, Any]:
    """One panel run in lockstep with its own free-field reference.

    Each worker process pays for its private incident-field run, but the
    three workers advance in parallel, so the wall time of the cached
    field computation is about two simulations instead of four.
    """
    import fdtd2d

    c0, dx = 343.0, _META_DX
    y1 = _META_FACE
    lam0 = c0 / _META_F0
    # Duration rule: full flight source -> deepest well bottom -> top of
    # the visible frame, (0.714 + 1.034) m / c * 1.2 = 6.12 ms, sampled at
    # four times the 12 frames-per-period visual floor (49/period at 2 kHz).
    every = 33
    sims = []
    for rho in (_meta_rho(kind), _meta_rho("ref")):
        sim = fdtd2d.FDTD2D(c0, dx, rho=rho, shape=(_META_NY, _META_NX),
                            sponge_width=200)
        sim.add_plane_wave("up", center=0.80, width=0.05, wavelength=lam0)
        taper = _meta_taper()
        sim.p *= taper[np.newaxis, :]
        sim.vy *= taper[np.newaxis, :]
        sims.append(sim)
    decay = float(2.0 ** (-sims[0].dt / 0.0015))
    face_row = round(y1 / dx)
    trail = np.zeros_like(sims[0].p)
    tot_frames: list[Any] = []
    trail_frames: list[Any] = []
    ts: list[float] = []
    for _ in range(every * n_frames):
        for sim in sims:
            sim.step()
        scat = sims[0].p - sims[1].p
        scat[:face_row, :] = 0.0
        np.maximum(trail * decay, np.abs(scat), out=trail)
        if sims[0].n % every == 0 and len(ts) < n_frames:
            tot_frames.append(sims[0].p[::20, ::20].astype(np.float32))
            trail_frames.append(trail[::20, ::20].astype(np.float32))
            ts.append(sims[0].time)
    return np.stack(tot_frames), np.stack(trail_frames), np.asarray(ts)


@lru_cache(maxsize=1)
def _metadiffuser_fields(
    n_frames: int = _META_FRAMES,
) -> tuple[Any, Any, Any]:
    """Plane-wave packet onto a deep QRD vs the 2 cm metadiffuser, cached.

    A 1.6 m x 1.2 m free-field box at dx = 0.25 mm, fine enough to mesh the
    real millimetre slits, necks and cavities of the Table-1 metadiffuser
    (density contrast 1e6:1 builds the panels). Each panel (flat control
    with the metadiffuser's silhouette, deep QRD, metadiffuser) advances
    in lockstep with a free-field reference inside its own worker process,
    so ``total - incident`` is exact; a downgoing carrier packet at the
    2 kHz evaluation frequency plays the goniometer source. Returns the total
    frame stacks, the fading scattered-envelope trails [dB] and the frame
    times. The quantitative far-field comparison lives in the companion
    polar figure; this clip shows the near fields of the meshed panels.
    """
    import multiprocessing as mp

    tot, trail, times = None, None, None
    try:
        # The GPU host of .env turns the half-hour CPU computation into a
        # couple of minutes (the runner falls back by itself, but the CPU
        # pool below is faster than its single-process local mode).
        import fdtd_gpu_remote

        fdtd_gpu_remote.load_env()
        config = fdtd_gpu_remote.RemoteConfig.from_env()
        use_gpu = bool(config.host) and fdtd_gpu_remote.remote_available(
            config)
    except (ImportError, OSError, ValueError):
        use_gpu = False
    if use_gpu:
        import fdtd2d

        # Same duration rule as the CPU worker: 6.12 ms at 49 frames/period.
        every = 33
        stride = 20
        dt = fdtd2d.FDTD2D(343.0, _META_DX, shape=(4, 8)).dt
        sample_steps = [every * k for k in range(1, n_frames + 1)]
        lam0 = 343.0 / _META_F0
        frames: dict[str, Any] = {}
        for kind in ("flat", "qrd", "meta", "ref"):
            job = fdtd_gpu_remote.build_job(
                343.0, _META_DX, steps=every * n_frames,
                sample_steps=sample_steps, shape=(_META_NY, _META_NX),
                rho=_meta_rho(kind), sponge_width=200,
                plane_waves=[{"direction": "up", "center": 0.80,
                              "width": 0.05, "wavelength": lam0}],
                init_scale_x=_meta_taper(),
                sample_stride=stride, sample_dtype="float32",
            )
            # The four 19 800-step field jobs run ~10-12 min each on the
            # GPU host; give each submit an hour before falling back.
            frames[kind] = fdtd_gpu_remote.submit(
                job, config, timeout=3600.0)["frames"]
        face_row = round(_META_FACE / _META_DX) // stride
        decay = float(2.0 ** (-(every * dt) / 0.0015))
        tot_list, trail_list = [], []
        for kind in ("flat", "qrd", "meta"):
            scat = frames[kind] - frames["ref"]
            scat[:, :face_row, :] = 0.0
            running = np.zeros_like(scat[0])
            history = np.empty_like(scat)
            for k in range(scat.shape[0]):
                np.maximum(running * decay, np.abs(scat[k]), out=running)
                history[k] = running
            tot_list.append(frames[kind].astype(np.float32))
            trail_list.append(history)
        tot = np.stack(tot_list)
        trail = np.stack(trail_list)
        times = np.asarray(sample_steps, dtype=np.float64) * dt
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=3) as pool:
            parts = pool.starmap(
                _meta_pair_worker,
                [(kind, n_frames) for kind in ("flat", "qrd", "meta")],
            )
        tot = np.stack([part[0] for part in parts])
        trail = np.stack([part[1] for part in parts])
        times = parts[0][2]
    ref = float(trail[:, trail.shape[1] // 3:].max()) or 1.0
    with np.errstate(divide="ignore"):
        trail_db = 20.0 * np.log10(trail / ref)
    trail_db = np.clip(trail_db, -30.0, 0.0).astype(np.float32)
    return tot, trail_db, times


def animate_fdtd_metadiffuser(output_dir: str) -> None:
    """The 27 cm deep Schroeder QRD vs the 2 cm metadiffuser that mimics
    it, next to a flat control slab (2D FDTD at 0.25 mm, real slits, necks
    and cavities meshed): the same 2 kHz wavefront leaves the same kind of
    scattered fan, from a panel 13.7 times thinner."""
    from matplotlib import patheffects
    from matplotlib.patches import Polygon, Rectangle

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    tot_all, trail_db, times = _metadiffuser_fields()
    y1 = _META_FACE
    x_l = _META_XL
    x_r = x_l + _META_PERIODS * 5 * _META_PITCH
    vmax = float(np.quantile(np.abs(tot_all[:, 0]), 0.999))

    fig = _anim_figure()
    fig.suptitle(T("Schroeder diffuser vs metadiffuser (2D FDTD)"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 3)
    titles = [T("Flat rigid panel"), T("QRD, wells down to 27 cm"),
              T("Metadiffuser, 2 cm panel")]
    qrd_poly: list[tuple[float, float]] = [(x_l, 0.05), (x_l, y1)]
    for wx0, wx1, d in _meta_qrd_wells():
        qrd_poly += [(wx0, y1), (wx0, y1 - d), (wx1, y1 - d), (wx1, y1)]
    qrd_poly += [(x_r, y1), (x_r, 0.05)]
    xc = 0.5 * (x_l + x_r)

    ims: list[Any] = []
    d_txts: list[Any] = []
    for col in range(3):
        ax_t = fig.add_subplot(gs[0, col])
        ax_s = fig.add_subplot(gs[1, col])
        im_t = ax_t.imshow(tot_all[col][0], origin="lower",
                           extent=(0.0, 1.6, 0.0, 1.2), cmap=CMAP_FIELD,
                           vmin=-vmax, vmax=vmax, interpolation="bilinear")
        im_s = ax_s.imshow(trail_db[col][0], origin="lower",
                           extent=(0.0, 1.6, 0.0, 1.2), cmap="magma",
                           vmin=-30.0, vmax=0.0, interpolation="bilinear")
        ax_t.set_title(titles[col], fontsize=10, fontweight="bold")
        for ax in (ax_t, ax_s):
            ax.grid(False)
            if col == 1:
                ax.add_patch(Polygon(qrd_poly, closed=True,
                                     facecolor=COLOR_GRID,
                                     edgecolor=COLOR_FG, lw=0.8))
            else:
                ax.add_patch(Rectangle((x_l, y1 - 0.023), x_r - x_l, 0.023,
                                       facecolor=COLOR_GRID,
                                       edgecolor=COLOR_FG, lw=0.8))
            ax.set_xlim(0.06, 1.54)
            ax.set_ylim(0.0, 1.12)
            ax.tick_params(labelsize=7)
        ax_t.tick_params(labelbottom=False)
        ax_s.set_xlabel("x [m]", fontsize=8)
        if col == 1:
            ax_t.text(xc, 0.97, T("incident plane wavefront"), ha="center",
                      va="bottom", color=FIELD_INK, fontsize=7.5,
                      path_effects=outline)
            ax_t.annotate("", xy=(xc, 0.83), xytext=(xc, 0.955),
                          arrowprops={"arrowstyle": "-|>", "color": FIELD_INK,
                                      "lw": 1.2})
        d_txt = ax_s.text(xc, 1.03, "", ha="center", va="top",
                          color="white", fontsize=7.5, fontweight="bold")
        if col == 0:
            ax_t.set_ylabel(T("sound field p"), fontsize=9)
            ax_s.set_ylabel(T("scattered field (total − incident)"),
                            fontsize=8)
        else:
            ax_t.tick_params(labelleft=False)
            ax_s.tick_params(labelleft=False)
        if col == 1:
            ax_t.annotate("", xy=(x_r + 0.045, y1 - 0.274),
                          xytext=(x_r + 0.045, y1),
                          arrowprops={"arrowstyle": "-", "color": COLOR_FG,
                                      "lw": 1.6})
            ax_t.text(x_r + 0.07, y1 - 0.14, "27 cm", ha="left",
                      va="center", fontsize=7, color=COLOR_FG)
        if col == 2:
            ax_t.text(xc, 0.06, T("real slits and resonators meshed at "
                                  "0.25 mm"), ha="center", va="bottom",
                      color=FIELD_INK, fontsize=6.5, path_effects=outline)
            ax_t.annotate("", xy=(x_r + 0.045, y1 - 0.023),
                          xytext=(x_r + 0.045, y1),
                          arrowprops={"arrowstyle": "-", "color": COLOR_FG,
                                      "lw": 1.6})
            ax_t.text(x_r + 0.07, y1 - 0.012, "2 cm", ha="left",
                      va="center", fontsize=7, color=COLOR_FG)
        ims += [im_t, im_s]
        d_txts.append(d_txt)
    t_txt = fig.text(0.985, 0.02, "", ha="right", va="bottom",
                     family="monospace", fontsize=10, color=COLOR_FG)
    reveal = int(0.8 * tot_all.shape[1])

    verdicts = [T("a collimated specular beam"),
                T("a wide scattered fan"),
                T("the same fan, from 2 cm")]

    def update(k: int) -> tuple[Any, ...]:
        for col in range(3):
            ims[2 * col].set_data(tot_all[col][k])
            ims[2 * col + 1].set_data(trail_db[col][k])
            d_txts[col].set_text(verdicts[col] if k >= reveal else "")
        t_txt.set_text(T(f"t = {times[k] * 1000.0:4.2f} ms"))
        return (*ims, *d_txts, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_metadiffuser",
                 frames=int(tot_all.shape[1]), gif_fps=8)


# --- README banner: a wavefront in a hall of columns (2D FDTD) -------------
# A wide 4:1 clip made for the top of the README: an 800 Hz plane wavefront
# sweeps down a rigid-walled hall through a staggered field of rigid columns,
# diffracting around every pillar until the whole hall is filled with the
# interference of the scattered wavelets. Banner canvas: 8.0 x 2.0 in at
# 300 dpi = 2400 x 600 px; the visible window is exactly 4 m x 1 m so the
# field is shown at true physical aspect.
_PILLAR_DX = 0.0025                   # [m]
_PILLAR_LX, _PILLAR_LY = 4.4, 1.0     # domain [m]; 0.2 m sponge ends hidden
_PILLAR_NX = round(_PILLAR_LX / _PILLAR_DX)   # 1760
_PILLAR_NY = round(_PILLAR_LY / _PILLAR_DX)   # 400
_PILLAR_F0 = 800.0                    # carrier [Hz], lambda = 42.9 cm
_PILLAR_VIEW = (0.2, 4.2)             # visible x window [m] (4:1 exact)
_PILLAR_FIGSIZE = (8.0, 2.0)          # in at _ANIM_DPI -> 2400 x 600 px
# Mesh rule: the smallest geometric dimension is the 6.6 cm column-wall
# aperture (6.6 / 4 = 1.65 cm) against lambda / 8 = 5.36 cm at 800 Hz, so
# the bound is 1.65 cm; dx = 2.5 mm sits 6.6x finer (>= 44 cells per
# column diameter) for the definition the banner format demands.
# Duration rule: the packet crosses injection (x = 0.30 m) to the far
# absorbing end (4.40 m) in 12.0 ms and the norm floor is x 1.2 = 14.3 ms;
# the captured window runs 22.3 ms so the multiple-scattering coda is
# also seen decaying to silence. With dt = 0.6 dx / (c sqrt(2)) = 3.09 us
# and one stored frame every 8 steps (24.7 us), the 1.25 ms carrier
# period is sampled 50.5 times (>= 48 frames per period). The first 40
# frames (~1.0 ms of free travel before the first columns) are trimmed,
# leaving 900 frames played at 40 fps: the same 0.99 ms of simulation
# per clip second as the approved preview pacing, 22.5 s + hold.
_PILLAR_EVERY = 8
_PILLAR_FRAMES = 900
_PILLAR_FPS = 40
_PILLAR_WARM = 40                     # frames of free travel trimmed
_PILLAR_STEM = "anim_fdtd_pillar_hall"


def _pillar_layout() -> list[tuple[float, float, float]]:
    """Column centres and radii [m] of the banner hall, deterministic.

    A staggered lattice (0.30 m pitch, three rows) with seeded jitter,
    radius spread and random vacancies, so the hall reads as an irregular
    colonnade rather than a perfect crystal, porous enough that the
    wavetrain threads through and fills the right half instead of
    mirroring off a solid wall of columns. The 10-17 cm column diameters
    sit around a quarter to half of the 42.9 cm carrier wavelength, where
    a rigid cylinder scatters strongly and casts a readable shadow. The
    tightest apertures of the seeded layout are 12.1 cm between column
    surfaces and 6.6 cm between a column and a wall, so
    dx = min(smallest aperture / 4, lambda / 8) allows up to 1.6 cm; the
    2.5 mm grid is three times finer than that bound and rasterises the
    cylinders at >= 44 cells per diameter.
    """
    rng = np.random.default_rng(20260728)
    pitch = 0.30
    out: list[tuple[float, float, float]] = []
    for j, cy in enumerate((0.17, 0.50, 0.83)):
        for cx in np.arange(1.15, 3.46, pitch):
            x = float(cx) + (pitch / 2.0 if j % 2 else 0.0)
            drop = rng.random() < 0.18
            jx, jy = rng.uniform(-0.03, 0.03, 2)
            r = rng.uniform(0.05, 0.085)
            if drop:
                continue
            out.append((x + float(jx), float(cy) + float(jy), float(r)))
    return out


def _pillar_mask() -> Any:
    """Rasterise the columns into the rigid-cell mask of the banner run."""
    xs = (np.arange(_PILLAR_NX) + 0.5) * _PILLAR_DX
    ys = (np.arange(_PILLAR_NY) + 0.5) * _PILLAR_DX
    xx, yy = np.meshgrid(xs, ys)
    mask = np.zeros((_PILLAR_NY, _PILLAR_NX), dtype=bool)
    for cx, cy, r in _pillar_layout():
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return mask


@lru_cache(maxsize=1)
def _pillar_fields(n_frames: int = _PILLAR_FRAMES) -> tuple[Any, Any]:
    """Field frames of the banner run: rigid hall, absorbing ends, cached.

    The top and bottom edges stay rigid (the hall walls), the left and
    right ends absorb through 0.2 m sponges hidden outside the visible
    window, and the columns are rigid rasterised cells. The incident wave
    is a single one-way plane-wave packet (Gaussian envelope one carrier
    wavelength wide) launched just before the first columns, so one front
    sweeps the hall, each pillar sheds a visible reflection into the
    structured coda behind it, and the hall then empties through the
    absorbing ends; a mild bulk damping (25 1/s, a ~0.28 s T60 stand-in
    for air and wall absorption) keeps the decay physical. 704 k cells
    and 7520 steps, still comfortably a CPU job, so the GPU runner is
    not involved.
    """
    import fdtd2d

    sim = fdtd2d.FDTD2D(
        343.0, _PILLAR_DX, shape=(_PILLAR_NY, _PILLAR_NX),
        sponge_width=80, sponge_sides=("left", "right"),
        damping=25.0, obstacle_mask=_pillar_mask())

    lam0 = 343.0 / _PILLAR_F0
    sim.add_plane_wave("right", center=0.30, width=0.08, wavelength=lam0)
    for _ in range(_PILLAR_WARM * _PILLAR_EVERY):
        sim.step()
    frames = np.empty((n_frames, _PILLAR_NY, _PILLAR_NX), dtype=np.float32)
    ts = np.empty(n_frames)
    for k in range(n_frames):
        for _ in range(_PILLAR_EVERY):
            sim.step()
        frames[k] = sim.p
        ts[k] = sim.time
    return frames, ts


def animate_fdtd_pillar_hall(output_dir: str) -> None:
    """README banner: an 800 Hz plane wavefront sweeps through a hall of
    rigid columns (2D FDTD at 2.5 mm): every pillar diffracts the front and
    the scattered wavelets interfere until they fill the whole hall."""
    from matplotlib.patches import Circle

    T = _translate_str
    frames, ts = _pillar_fields()
    n_active = frames.shape[0]
    dark = bool(_FILENAME_SUFFIX)
    cmap = CMAP_FIELD
    # Colour scale: 55 % of the packet's global peak, so the travelling
    # front saturates boldly while each column's shed reflection sits in
    # the mid ramp and the decaying coda fades instead of clipping.
    vmax = 0.55 * float(np.max(np.abs(frames)))
    fig = plt.figure(figsize=_PILLAR_FIGSIZE, dpi=_ANIM_DPI)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    im = ax.imshow(frames[0], origin="lower",
                   extent=(0.0, _PILLAR_LX, 0.0, _PILLAR_LY), cmap=cmap,
                   vmin=-vmax, vmax=vmax, interpolation="bilinear")
    for cx, cy, r in _pillar_layout():
        ax.add_patch(Circle((cx, cy), r, facecolor=COLOR_GRID,
                            edgecolor=COLOR_FG, lw=0.8))
    ax.set_xlim(*_PILLAR_VIEW)
    ax.set_ylim(0.0, _PILLAR_LY)
    ax.set_aspect("auto")
    ax.grid(False)
    ax.axis("off")
    caption_bbox = {"facecolor": "black" if dark else "white",
                    "alpha": 0.55, "edgecolor": "none", "pad": 2.0}
    ax.text(0.012, 0.055, T("2D FDTD wavefront in a hall of columns"),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            color=COLOR_FG, bbox=caption_bbox)
    t_txt = ax.text(0.988, 0.055, "", transform=ax.transAxes, ha="right",
                    va="bottom", family="monospace", fontsize=9,
                    color=COLOR_FG, bbox=caption_bbox)

    def update(k: int) -> tuple[Any, ...]:
        kf = min(k, n_active - 1)
        im.set_data(frames[kf])
        t_txt.set_text(T(f"t = {ts[kf] * 1e3:5.2f} ms"))
        return (im, t_txt)

    # GIF budget: the full-frame wave motion compresses poorly, so the
    # README GIF decimates the 40 fps WebM 5x (gif_fps = 8) at the shared
    # 640 px width, landing at 6.2 MB (light) and 6.6 MB (dark), under
    # the ~8 MB GitHub autoplay budget.
    _render_clip(fig, update, output_dir, _PILLAR_STEM,
                 frames=n_active + _ANIM_HOLD, fps=_PILLAR_FPS, gif_fps=8,
                 poster_ss=_poster_ss_for(_PILLAR_STEM + ".webm"))


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
            ax.text(xm, 1.62, f"$p_{j + 1}$", ha="center", va="bottom",
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
                           (0.58, COLOR_SECONDARY, T("sum p(x, t)")),
                           (0.72, COLOR_FG, T("envelope |p(x)|"))):
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
                    T(f"|R| = {refl:.2f}   α = {1.0 - refl**2:.2f}"))
                pn["readout"].set_alpha(a_env)
            else:
                pn["mic_dots"].set_data([], [])
                pn["note"].set_text("")
                pn["readout"].set_text("")
            arts += [pn["l_inc"], pn["l_ref"], pn["l_sum"], pn["mic_dots"],
                     pn["note"], pn["readout"], *pn["env_lines"]]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_standing_wave_tube")


# ---------------------------------------------------------------------------
# FDTD wave-field clips, second batch (slit absorber, expansion chamber,
# wall aperture, atmospheric refraction). Same conventions as the first
# batch: each simulation runs once per process behind an lru_cache, every
# clip keeps >= 48 captured frames per period of its highest carrier, and
# every number printed on screen comes from the library models, never from
# the simulation itself.
# ---------------------------------------------------------------------------


_SLIT_ABS_F0 = 300.0                     # critical-coupling design frequency
_SLIT_ABS_TUBE = 0.60                    # air column before the panel face
_SLIT_ABS_PERIOD = 5.0e-2                # panel period d = the tube bore
_SLIT_ABS_LATTICE = 3.0e-2               # lattice step a = panel depth (N=1)
_SLIT_ABS_DETUNE = 1.7                   # wide-slit factor of the alpha figure


@lru_cache(maxsize=1)
def _slit_absorber_design() -> Any:
    """The 300 Hz critical-coupling design of the slow-sound figures."""
    from phonometry import HelmholtzResonator, critical_coupling_design

    base = HelmholtzResonator(neck_length=1.0e-3, neck_side=3.0e-3,
                              cavity_length=30.0e-3, cavity_side=27.0e-3)
    return critical_coupling_design(_SLIT_ABS_F0, base,
                                    lattice_step=_SLIT_ABS_LATTICE,
                                    period=_SLIT_ABS_PERIOD)


def _lossy_fluid_params(rho_c: complex,
                        kappa_c: complex) -> tuple[float, float, float]:
    """Real-coefficient FDTD equivalent of a complex effective fluid.

    The solver carries plane waves through a uniform lossy region as
    ``k = (omega - j sigma)/c`` at the real impedance ``rho c``, so a
    visco-thermal effective density/bulk-modulus pair maps onto the phase
    speed ``omega / Re(k)``, the density ``Re(Z)/c`` and the decay rate
    ``sigma = -Im(k) c`` at the drive frequency. Returns ``(c, rho, sigma)``.
    """
    omega = 2.0 * np.pi * _SLIT_ABS_F0
    k = omega * complex(np.sqrt(rho_c / kappa_c))
    z = complex(np.sqrt(rho_c * kappa_c))
    c_ph = omega / k.real
    return c_ph, z.real / c_ph, -k.imag * c_ph


def _slit_absorber_cell_rows(dx: float, slit_cells: int,
                             ny: int) -> dict[str, tuple[int, int]]:
    """Row/column spans of the meshed cell (slit at the top of the period,
    neck and cavity below it, mirroring ``plot_slit_absorber_geometry``)."""
    res = _slit_absorber_design().resonator
    neck_cells = round(res.neck_length / dx)
    cav_cells = round(res.cavity_length / dx)
    slit0 = ny - slit_cells
    neck0 = slit0 - neck_cells
    return {"slit": (slit0, ny), "neck": (neck0, slit0),
            "cavity": (neck0 - cav_cells, neck0)}


@lru_cache(maxsize=1)
def _slit_absorber_fields(
    n_frames: int = _FDTD_ANIM_FRAMES,
) -> tuple[Any, Any, Any, Any]:
    """CW build-up against the meshed critical-coupling cell, cached.

    Two runs of a 0.60 m plane-wave tube whose bore is one 50 mm panel
    period, driven at 300 Hz through a rho c left edge; the right end is
    the panel itself: slit, neck and cavity carved out of the rigid body
    with the obstacle mask at ``dx = h/4`` (the slit is the smallest
    feature) and filled with the library's visco-thermal effective fluids
    mapped onto real ``(c, rho, sigma)`` triplets. The neck and cavity are
    square ducts of side ``w`` in 3D, so the 2D slice scales their density
    (and with it ``kappa = rho c^2``) by the out-of-plane ratio ``a / w``,
    keeping the 3D cell's acoustic mass and compliance per unit depth. Run
    one is the critical design; run two detunes only the slit height by
    the wide-slit factor of the absorption figure. ``cfl = 0.95``: at
    lambda/4700 the numerical dispersion is negligible and the wide margin
    of the default would just double the run time. Returns per-run (tube
    frames, cell-zoom frames, the settled |p| envelope), the frame times
    and the library absorption at 300 Hz for both slit heights.
    """
    import fdtd2d

    from phonometry import slit_helmholtz_absorber
    from phonometry.materials import (
        rectangular_duct_properties,
        slit_effective_properties,
    )

    design = _slit_absorber_design()
    res = design.resonator
    h0 = design.slit_height
    # Mesh rule: dx = min(smallest scene dimension / 4, lambda/8 at the
    # carrier) = min(h/4 = 0.244 mm, 1.143 m / 8 = 143 mm) -> the slit
    # height governs and the grid is h/4 = 0.244 mm (lambda/4677).
    dx = h0 / 4.0
    ny = round(_SLIT_ABS_PERIOD / dx)
    face = round(_SLIT_ABS_TUBE / dx)
    nx = face + round(_SLIT_ABS_LATTICE / dx)
    zoom0 = round((_SLIT_ABS_TUBE - 0.015) / dx)
    x_mouth = _SLIT_ABS_TUBE + 0.5 * _SLIT_ABS_LATTICE
    neck_c = (round((x_mouth - 0.5 * res.neck_side) / dx),
              round((x_mouth + 0.5 * res.neck_side) / dx))
    cav_c = (round((x_mouth - 0.5 * res.cavity_side) / dx),
             round((x_mouth + 0.5 * res.cavity_side) / dx))
    freq = np.array([_SLIT_ABS_F0])
    neck_f = _lossy_fluid_params(
        *(p.item() for p in rectangular_duct_properties(
            freq, side=res.neck_side)))
    cav_f = _lossy_fluid_params(
        *(p.item() for p in rectangular_duct_properties(
            freq, side=res.cavity_side)))
    # 2D-slice calibration: the meshed cell's slot-shaped mouths carry
    # smaller end-correction masses than the square ducts of the 3D
    # model, and in this slow-sound cell the resonance is the slit
    # inertance against the cavity compliance, so the whole cell lands
    # ~11 % high: a broadband two-microphone probe of the meshed cell
    # puts its absorption peak at 333 Hz instead of 300. Rescaling the
    # cavity sound speed by 300/333 = 0.90 restores the design
    # resonance; the same probe then measures alpha = 0.99 at 300.0 Hz,
    # the critical coupling the annotation quotes. The rescale is fitted
    # at the critical slit height only and does not transfer exactly to
    # the detuned row (the wide-slit cell measures |R| ~ 0.95 where the
    # 3D model gives 0.81); the visual contrast and the qualitative
    # message are the correct ones, and the alpha pill quotes the
    # library model's value.
    cav_f = (0.90 * cav_f[0], cav_f[1], cav_f[2])
    scale_neck = _SLIT_ABS_LATTICE / res.neck_side
    scale_cav = _SLIT_ABS_LATTICE / res.cavity_side

    alphas = (float(design.absorption),
              float(slit_helmholtz_absorber(
                  freq, res, slit_height=_SLIT_ABS_DETUNE * h0,
                  lattice_step=_SLIT_ABS_LATTICE,
                  period=_SLIT_ABS_PERIOD).absorption[0]))
    # Clip duration per the deepest-reflector rule: d(source -> cavity
    # bottom through slit mouth and neck) = 0.60 + 0.015 + 0.001 + 0.0447
    # = 0.6607 m, the same back to the farthest visible point (the frame's
    # left edge), over the slowest medium on the path (the slit fluid,
    # 313.7 m/s): t = 1.2 * 2 * 0.6607 / 313.7 = 5.06 ms -> every = 30
    # (5.17 ms captured, 232 frames per 300 Hz period >= 48). The window
    # is far shorter than the resonator ring-up (~6 ms time constant), so
    # both runs first settle uncaptured for 20 ms and the clip shows the
    # steady field in slow motion with its exact settled envelope.
    every = 30
    runs = []
    times = np.zeros(0)
    for h in (h0, _SLIT_ABS_DETUNE * h0):
        slit_f = _lossy_fluid_params(
            *(p.item() for p in slit_effective_properties(
                freq, slit_height=h)))
        rows = _slit_absorber_cell_rows(dx, round(h / dx), ny)
        mask = np.zeros((ny, nx), dtype=bool)
        mask[:, face:] = True
        c_map = np.full((ny, nx), 343.0)
        rho_map = np.full((ny, nx), 1.2)
        sig_map = np.zeros((ny, nx))
        for name, (c_e, rho_e, sig_e), scale, cols in (
                ("slit", slit_f, 1.0, (face, nx)),
                ("neck", neck_f, scale_neck, neck_c),
                ("cavity", cav_f, scale_cav, cav_c)):
            r0, r1 = rows[name]
            sl = (slice(r0, r1), slice(cols[0], cols[1]))
            mask[sl] = False
            c_map[sl] = c_e
            rho_map[sl] = rho_e * scale
            sig_map[sl] = sig_e
        sim = fdtd2d.FDTD2D(c_map, dx, rho=rho_map, damping=sig_map,
                            cfl=0.95, obstacle_mask=mask,
                            edge_impedance={"left": 1.2 * 343.0})
        tone = fdtd2d.CWSource(0, 0, frequency=_SLIT_ABS_F0)
        sim.add_source(fdtd2d.PlaneWaveSource("right", tone.value, offset=2))
        mid = ny // 2
        # Settle to steady state, sampling the exact standing-wave
        # envelope over the final full period.
        period = round(1.0 / (_SLIT_ABS_F0 * sim.dt))
        settle = round(0.020 / sim.dt)
        env = np.zeros(sim.p[mid, :face:2].shape, dtype=np.float32)
        for i in range(settle):
            sim.step()
            if i >= settle - period:
                np.maximum(env, np.abs(sim.p[mid, :face:2]), out=env)
        tube: list[Any] = []
        cell: list[Any] = []
        ts: list[float] = []
        while len(tube) < n_frames:
            sim.step()
            if sim.n % every == 0:
                tube.append(sim.p[::4, ::4].astype(np.float32))
                cell.append(sim.p[:, zoom0:].astype(np.float32))
                ts.append(sim.time)
        runs.append((np.stack(tube), np.stack(cell), env))
        times = np.asarray(ts)
    return runs[0], runs[1], times, alphas


def animate_fdtd_slit_absorber(output_dir: str) -> None:
    """Inside the slow-sound slit absorber (2D FDTD): a 300 Hz plane tone
    meets the meshed critical-coupling cell, whose sub-millimetre slit and
    Helmholtz resonator are resolved on the grid and filled with the
    library's visco-thermal effective fluids. At the design slit height
    the loss/leakage balance swallows the wave (alpha = 1, flat envelope);
    widening the slit breaks the balance and the reflection rebuilds the
    standing wave. A zoom shows the field crawling through the slit."""
    from matplotlib.patches import Rectangle

    T = _translate_str
    (t_c, z_c, e_c), (t_w, z_w, e_w), times, alphas = _slit_absorber_fields()
    design = _slit_absorber_design()
    res = design.resonator
    h0 = design.slit_height
    dx = h0 / 4.0
    ny = round(_SLIT_ABS_PERIOD / dx)
    nx = round(_SLIT_ABS_TUBE / dx) + round(_SLIT_ABS_LATTICE / dx)
    x_end = nx * dx
    x_zoom0 = round((_SLIT_ABS_TUBE - 0.015) / dx) * dx
    bore = ny * dx
    half = t_c.shape[0] // 2
    # Color scale from the TUBE columns only: at critical coupling the
    # cavity rings at several times the incident amplitude and would
    # otherwise wash the travelling/standing waves out of the map (the
    # cell interior saturating instead is the point).
    tube_cols = round(_SLIT_ABS_TUBE / dx) // 4
    vmax = float(np.quantile(np.abs(t_c[half:, :, :tube_cols]), 0.999))
    env_base, env_h, env_max = 0.074, 0.034, 2.3
    x_env = (np.arange(e_c.shape[0]) + 0.5) * 2.0 * dx
    env_from = 10                          # hide the injection-line step

    fig = _anim_figure()
    fig.suptitle(T("Slow-sound slit absorber at critical coupling "
                   "(2D FDTD)"), fontweight="bold")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 0.52])
    titles = [T("Critically coupled slit: the wave dies inside"),
              T("Wide slit (detuned): the reflection returns")]
    heights = (h0, _SLIT_ABS_DETUNE * h0)
    envs = (e_c, e_w)
    ims: list[Any] = []
    ims_zoom: list[Any] = []
    a_txts: list[Any] = []
    for row, (title, h) in enumerate(zip(titles, heights, strict=True)):
        rows = _slit_absorber_cell_rows(dx, round(h / dx), ny)
        # Cell silhouette: the rigid part of the panel as five rectangles
        # around the slit/neck/cavity openings, so both views show the
        # field inside the openings through the gaps.
        y_slit0 = rows["slit"][0] * dx
        y_neck = tuple(r * dx for r in rows["neck"])
        y_cav = tuple(r * dx for r in rows["cavity"])
        x_face = _SLIT_ABS_TUBE
        x_mouth = _SLIT_ABS_TUBE + 0.5 * _SLIT_ABS_LATTICE
        x_neck = (x_mouth - 0.5 * res.neck_side,
                  x_mouth + 0.5 * res.neck_side)
        x_cav = (x_mouth - 0.5 * res.cavity_side,
                 x_mouth + 0.5 * res.cavity_side)
        body = [
            (x_face, 0.0, x_end - x_face, y_cav[0]),
            (x_face, y_cav[0], x_cav[0] - x_face, y_cav[1] - y_cav[0]),
            (x_cav[1], y_cav[0], x_end - x_cav[1], y_cav[1] - y_cav[0]),
            (x_face, y_neck[0], x_neck[0] - x_face, y_neck[1] - y_neck[0]),
            (x_neck[1], y_neck[0], x_end - x_neck[1],
             y_neck[1] - y_neck[0]),
        ]
        ax = fig.add_subplot(gs[row, 0])
        ax.grid(False)
        im = ax.imshow(np.zeros((2, 2)), origin="lower",
                       extent=(0.0, x_end, 0.0, bore), cmap=CMAP_FIELD,
                       vmin=-vmax, vmax=vmax, aspect="auto",
                       interpolation="bilinear", zorder=2)
        _anim_slit_tube_walls(ax, x_end, bore, speaker=row == 0)
        for rect in body:
            ax.add_patch(Rectangle(rect[:2], rect[2], rect[3],
                                   facecolor="#8b8b8b",
                                   edgecolor=COLOR_FG, lw=0.4, zorder=3))
        # Dashed magnifier frame around the cell, echoed on the zoom axes.
        ax.add_patch(Rectangle((x_zoom0, 0.0), x_end - x_zoom0, bore,
                               facecolor="none", edgecolor=COLOR_FG,
                               ls="--", lw=0.8, zorder=5))
        ax.axhline(env_base, color=COLOR_GRID, lw=0.8, zorder=1)
        ax.text(0.005, env_base + 0.004, "|p| envelope", fontsize=7.5,
                ha="left", va="bottom", color=COLOR_FG, alpha=0.8)
        # The settled standing-wave envelope, static: the clip captures
        # the steady state, where it no longer changes.
        ax.plot(x_env[env_from:],
                env_base + envs[row][env_from:] / env_max * env_h,
                color=COLOR_PRIMARY, lw=1.8, zorder=6)
        ax.set_xlim(-0.115, x_end + 0.022)
        # Headroom above the envelope trace keeps the alpha pill clear
        # of the curve in both rows.
        ax.set_ylim(-0.033, env_base + env_h + 0.026)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(T("Position along the tube [m]"), fontsize=9)
        a_txt = ax.text(0.03, env_base + env_h + 0.023, "", ha="left",
                        va="top", fontsize=9, color="white", zorder=7,
                        bbox={"boxstyle": _ANIM_PILL_BOX,
                              "facecolor": "black", "alpha": 0.55,
                              "edgecolor": "none"})
        # Zoom: the last 45 mm at full grid resolution, geometry to scale.
        ax_z = fig.add_subplot(gs[row, 1])
        ax_z.grid(False)
        im_z = ax_z.imshow(np.zeros((2, 2)), origin="lower",
                           extent=(x_zoom0, x_end, 0.0, bore),
                           cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
                           aspect="equal", interpolation="nearest",
                           zorder=2)
        for rect in body:
            ax_z.add_patch(Rectangle(rect[:2], rect[2], rect[3],
                                     facecolor="#8b8b8b",
                                     edgecolor=COLOR_FG, lw=0.5, zorder=3))
        for spine in ax_z.spines.values():
            spine.set_linestyle("--")
            spine.set_edgecolor(COLOR_FG)
        ax_z.set_xlim(x_zoom0, x_end)
        ax_z.set_ylim(0.0, bore)
        ax_z.set_xticks([])
        ax_z.set_yticks([])
        ax_z.annotate(
            T(f"slit h = {h * 1e3:.2f} mm"),
            xy=(x_zoom0 + 0.006, y_slit0 + 0.5 * h),
            xytext=(x_zoom0 + 0.0035, bore - 0.0095),
            fontsize=7, color=COLOR_FG, ha="left", va="top",
            arrowprops={"arrowstyle": "-", "color": COLOR_FG, "lw": 0.7},
            zorder=6,
            bbox={"boxstyle": "round,pad=0.2",
                  "facecolor": fig.get_facecolor(), "alpha": 0.55,
                  "edgecolor": "none"})
        ax_z.text(x_mouth, 0.5 * (y_cav[0] + y_cav[1]),
                  T("Helmholtz resonator"), ha="center", va="center",
                  fontsize=6.5, color=COLOR_FG, zorder=6,
                  bbox={"boxstyle": "round,pad=0.2",
                        "facecolor": fig.get_facecolor(), "alpha": 0.55,
                        "edgecolor": "none"})
        if row == 0:
            ax_z.set_title(T("inside the cell"), fontsize=8.5)
        ims.append(im)
        ims_zoom.append(im_z)
        a_txts.append(a_txt)
    # Bottom-right corner: the top corners belong to the suptitle and the
    # zoom-column title, and the x-label of the tube column sits centred
    # well to the left of this spot.
    t_txt = fig.text(0.988, 0.015, "", ha="right", va="bottom",
                     family="monospace", fontsize=10, color=COLOR_FG)
    # The captured field is already steady, so the verdict can come early.
    reveal = int(0.30 * len(times))

    def update(k: int) -> tuple[Any, ...]:
        for i, (t_all, z_all, alpha) in enumerate(
                ((t_c, z_c, alphas[0]), (t_w, z_w, alphas[1]))):
            ims[i].set_data(t_all[k])
            ims_zoom[i].set_data(z_all[k])
            a_txts[i].set_text(
                T(f"alpha = {alpha:.2f} at {_SLIT_ABS_F0:.0f} Hz")
                if k >= reveal else "")
        t_txt.set_text(T(f"t = {times[k] * 1e3:5.1f} ms"))
        return (*ims, *ims_zoom, *a_txts, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_slit_absorber",
                 frames=len(times), gif_fps=8)


def _anim_slit_tube_walls(ax: Any, length: float, bore: float, *,
                          speaker: bool = True) -> None:
    """Tube walls and drive loudspeaker for the slit-absorber clip (the
    shared ``_anim_tube_hardware`` labels its termination as a plug, but
    here the termination IS the meshed panel, drawn by the caller)."""
    from matplotlib.patches import Rectangle

    wall = 0.007
    grey = "#9a9a9a"
    for y0 in (-wall, bore):
        ax.add_patch(Rectangle((0.0, y0), length, wall, facecolor=grey,
                               edgecolor=COLOR_FG, linewidth=0.7, zorder=3))
    _anim_speaker(ax, 0.0, 0.5 * bore, bore,
                  label_y=-0.30 * bore if speaker else None)


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
    titles = [T(f"Pass band: {f_pass:.0f} Hz, kL = π"),
              T(f"Stop band peak: {f_peak:.0f} Hz, kL = π/2")]
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
        ax.text(0.005, env_base - 0.008, "|p| envelope", fontsize=7.5,
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
        t_txt.set_text(T(f"t = {times[k] * 1e3:5.1f} ms"))
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
            f"L = {_CHAMBER_L:.2f} m · m = {_CHAMBER_M:.0f}", ha="center",
            va="bottom", fontsize=7.5, color=COLOR_FG)


_APERTURE_F = 686.0                      # lambda = 0.50 m exactly
_APERTURE_WIDTHS = (0.025, 0.50)         # sub-lambda slit / lambda-sized gap
_APERTURE_DEPTH = 0.10                   # wall thickness across the opening
# Mesh rule: dx = min(smallest scene dimension / 4, lambda/8 at the carrier)
# = min(25 mm / 4 = 6.25 mm, 0.5 m / 8 = 62.5 mm) -> the narrow slit governs
# (lambda/80, 4 cells across it).
_APERTURE_DX = 0.00625
_APERTURE_EVERY = 3
_APERTURE_FRAMES = 962
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
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2)
    titles = [T(f"Slit w = {_APERTURE_WIDTHS[0] * 1e3:.0f} mm (λ/20)"),
              T(f"Opening w = {_APERTURE_WIDTHS[1]:.2f} m (= λ)")]
    verdicts = [T("cylindrical re-radiation from the slit"),
                T("the front passes: sharp-edged shadow")]
    ims: list[Any] = []
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
        ax_p.set_title(titles[col], fontsize=10, fontweight="bold")
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
            ax.set_ylim(0.7, 4.3)
            ax.tick_params(labelsize=7)
        ax_p.tick_params(labelbottom=False)
        ax_r.set_xlabel("x [m]", fontsize=8)
        ax_p.annotate("", xy=(1.15, 3.75), xytext=(0.55, 3.75),
                      arrowprops={"arrowstyle": "-|>", "color": FIELD_INK,
                                  "lw": 1.2})
        ax_p.text(0.85, 3.85, T("incident plane wavefront"), ha="left",
                  va="bottom", color=FIELD_INK, fontsize=7.5,
                  path_effects=outline)
        ax_p.text(2.24, 1.45, T("rigid wall"), ha="left", va="center",
                  color=FIELD_INK, fontsize=7, path_effects=outline)
        if (note := _gain_note("past the wall", gains[col])):
            ax_p.text(5.62, 4.28, T(note), ha="right", va="top",
                      color=FIELD_INK, fontsize=6.5, path_effects=outline)
        ax_r.text(3.85, 0.85, verdicts[col], ha="center", va="bottom",
                  color="white", fontsize=7.5, zorder=6,
                  bbox={"boxstyle": _ANIM_PILL_BOX,
                        "facecolor": "black", "alpha": 0.45,
                        "edgecolor": "none"})
        if col == 0:
            ax_p.set_ylabel(T("instantaneous p(x, y)"), fontsize=9)
            ax_r.set_ylabel(T("RMS level [dB]"), fontsize=9)
            ax_p.text(0.2, 0.85,
                      T(f"f = {_APERTURE_F:.0f} Hz (λ = {lam:.2f} m)"),
                      ha="left", va="bottom", color=FIELD_INK, fontsize=7.5,
                      path_effects=outline)
            tau_txt = ax_r.text(5.55, 4.1, "", ha="right", va="top",
                                fontsize=8.5, color="white", zorder=7,
                                bbox={"boxstyle": _ANIM_PILL_BOX,
                                      "facecolor": "black", "alpha": 0.55,
                                      "edgecolor": "none"})
        else:
            ax_p.tick_params(labelleft=False)
            ax_r.tick_params(labelleft=False)
            ax_r.text(5.55, 4.1, T("same color scale"), color="white",
                      fontsize=7, ha="right", va="top")
        ims += [im_p, im_r]
    t_txt = fig.text(0.012, 0.985, "", ha="left", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)
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
            T(f"slit τ = {tau:.2f} (Gomperts)") if k >= reveal else "")
        t_txt.set_text(T(f"t = {times[k] * 1e3:4.1f} ms"))
        return (*ims, tau_txt, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_aperture_slit",
                 frames=int(p_all.shape[1]), fps=_APERTURE_FPS, gif_fps=8)


_REFR_B = 1.0                            # log-profile strength [m/s]
_REFR_C0 = 340.0                         # ground-level effective speed
_REFR_SRC = (30.0, 2.0)                  # source range/height [m]
_REFR_RECV = 350.0                       # receiver distance from the source
_REFR_F = 50.0                           # CW drive: lambda ~ 6.9 m
# Mesh rule: dx = min(smallest scene dimension / 4, lambda/8 at the
# carrier) = min(2 m source height / 4 = 0.5 m, 6.86 m / 8 = 0.857 m);
# the grid runs finer still (0.3 m, lambda/23) so the long-range phase
# stays clean over the ~60-wavelength domain.
_REFR_DX = 0.3
_REFR_EVERY = 1
_REFR_FRAMES = 1440
_REFR_FPS = 80


def _refraction_profiles() -> tuple[Any, ...]:
    """The guide's realistic logarithmic surface-layer profiles: downwind
    (+1 m/s) and upwind (-1 m/s) effective sound speed."""
    from phonometry import log_linear_sound_speed_profile

    return tuple(
        log_linear_sound_speed_profile(sign * _REFR_B,
                                       ground_speed=_REFR_C0,
                                       max_height=140.0)
        for sign in (1.0, -1.0)
    )


@lru_cache(maxsize=1)
def _refraction_fields(
    n_frames: int = _REFR_FRAMES,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Two CW runs through the refracting atmosphere, cached.

    A 452 m x 132 m slice over rigid ground with the guide's logarithmic
    effective-speed profiles (downwind +1 m/s, upwind -1 m/s) written into
    the sound-speed map row by row; sponges on the sides and the sky. A
    50 Hz CW source (lambda ~ 6.9 m, short against the ~110 m ray-model
    shadow distance, so the shadow actually forms) sits 2 m over the
    ground, the guide's source height. A low-frequency PULSE cannot show
    this: within the 30-periods-per-clip frame budget its wavelength
    reaches ~20 m and diffraction refills the shadow (a measured contrast
    of barely 2 dB). Instead both runs step uncaptured to steady state
    (~1.3 s, the transit of the domain) and the clip then shows the
    settled wave field streaming through the refracting atmosphere at 13.6
    frames per period. Returns the frame stacks, the steady running-RMS
    maps in dB compensated for cylindrical spreading (the verdict
    overlay, on a shared scale), the frame times, the height grid and the
    two c(z) profiles sampled on it, and the library ray fans traced
    through the same profiles.
    """
    import fdtd2d

    from phonometry import atmospheric_ray_paths

    dx = _REFR_DX
    ny, nx = 440, 1507                     # 132 m x 452 m
    z = (np.arange(ny) + 0.5) * dx
    profiles = _refraction_profiles()
    # Clip duration per the deepest-reflector rule: d(source -> ground)
    # = 2 m plus d(ground -> farthest visible frame corner (430, 105))
    # = 413.6 m at the slowest c on the path (upwind, ~333 m/s) gives
    # t = 1.2 * 415.6 / 333 = 1.50 s of flight, which the 1.30 s of
    # uncaptured settling plus this 0.53 s window covers with margin: the
    # wave has crossed the whole domain before the first captured frame
    # and the clip then shows the settled streaming field.
    # Sampling: every solver step is captured, 366.6 us apart downwind and
    # 375.4 us apart upwind (each run's own c_max sets its dt), i.e. 54.6
    # and 53.3 frames per 50 Hz period (>= 48; the old 4-step stride gave
    # 13.6). 1 440 frames, played at 80 fps -- four times the shared
    # 20 fps, exactly the factor by which the stride shrank, so both the
    # 18.0 s length and the 29.3 ms of simulation per second of playback
    # are the ones the committed clip already had.
    every = _REFR_EVERY
    rays = [
        atmospheric_ray_paths(prof, source_height=_REFR_SRC[1],
                              launch_angles_deg=angles, max_range=430.0,
                              n_steps=900)
        for prof, angles in zip(
            profiles,
            # Downwind: a shallow fan (the log profile turns anything
            # under ~8 deg back down, so the duct is thin) plus one
            # escaping ray; upwind: the same fan bends up and away.
            ((-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0),
             (-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0)), strict=True)
    ]
    p_all, r_all = [], []
    times = np.zeros(0)
    for prof in profiles:
        c_prof = prof.speed_at(z)
        c_map = np.repeat(c_prof[:, np.newaxis], nx, axis=1)
        # Row 0 is the ground (rigid, no sponge); "bottom" is the sky in
        # the low-row-origin naming of fdtd2d, as in the barrier clip.
        sim = fdtd2d.FDTD2D(c_map, dx, sponge_width=40,
                            sponge_sides=("left", "right", "bottom"))
        ix, iy = round(_REFR_SRC[0] / dx), round(_REFR_SRC[1] / dx)
        sim.add_source(fdtd2d.CWSource(ix=ix, iy=iy, frequency=_REFR_F,
                                       ramp_cycles=2.0))
        beta = float(np.exp(-sim.dt * _REFR_F / 2.0))
        ms = np.zeros_like(sim.p)
        settle = round(1.30 / sim.dt)      # domain transit + ring-up
        for _ in range(settle):
            sim.step()
            ms = beta * ms + (1.0 - beta) * sim.p**2
        ps: list[Any] = []
        ts: list[float] = []
        while len(ps) < n_frames:
            sim.step()
            ms = beta * ms + (1.0 - beta) * sim.p**2
            if sim.n % every == 0:
                ps.append(sim.p[::3, ::3].astype(np.float32))
                ts.append(sim.time)
        p_all.append(np.stack(ps))
        r_all.append(np.sqrt(ms))
        times = np.asarray(ts)
    rms_maps = np.stack(r_all)
    # Verdict overlay: the steady RMS map times sqrt(r) to undo the 2D
    # cylindrical spreading; what remains is pure refraction (the bright
    # downwind ground duct and interference lobes, the dark upwind shadow
    # wedge) on a shared dB scale referenced away from the source blast.
    xs, zs = _REFR_SRC
    xg, zg = np.meshgrid((np.arange(nx) + 0.5) * dx, z)
    r_map = np.maximum(np.hypot(xg - xs, zg - zs), 5.0)
    comp = rms_maps * np.sqrt(r_map)
    ref = float(np.quantile(comp[:, :, nx // 4:], 0.999))
    with np.errstate(divide="ignore"):
        e_db = 20.0 * np.log10(comp[:, ::3, ::3] / ref)
    e_db = np.clip(e_db, -30.0, 0.0).astype(np.float32)
    c_profs = np.stack([prof.speed_at(z) for prof in profiles])
    return np.stack(p_all), e_db, times, z, c_profs, rays


def animate_fdtd_refraction(output_dir: str) -> None:
    """Atmospheric refraction (2D FDTD): the same steady 50 Hz source 2 m
    over rigid ground, downwind and upwind. With the effective sound
    speed growing with height the wavefronts bend back down and stream
    along the ground, keeping the 350 m receiver loud; with it falling
    they lift off the surface and an acoustic shadow opens at the ground.
    The library's ray fans, traced through the same c(z) profiles,
    overlay the fields; the closing seconds crossfade to the
    spreading-compensated RMS map."""
    from matplotlib import patheffects

    from phonometry import shadow_zone_distance

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    p_all, e_db, times, z, c_profs, rays = _refraction_fields()
    profiles = _refraction_profiles()
    half = p_all.shape[1] // 2
    vmax = float(np.quantile(np.abs(p_all[:, half // 2:half]), 0.999))
    # Shadow-boundary estimate exactly as the guide does it: the log
    # profile's linear-equivalent gradient over the first 10 m, then the
    # closed-form grazing-ray distance.
    grad = float(profiles[1].speed_at(10.0) - profiles[1].speed_at(0.0))
    x_shadow = shadow_zone_distance(grad / 10.0, _REFR_SRC[1],
                                    _REFR_SRC[1], ground_speed=_REFR_C0)
    extent = (0.0, 1507 * _REFR_DX, 0.0, 440 * _REFR_DX)
    src_x, src_h = _REFR_SRC
    recv_x = src_x + _REFR_RECV

    fig = _anim_figure()
    fig.suptitle(T("Atmospheric refraction: downwind duct, upwind shadow "
                   "(2D FDTD)"), fontweight="bold")
    gs = fig.add_gridspec(2, 2, width_ratios=[0.20, 1.0])
    titles = [T("Downwind: sound speed grows with height"),
              T("Upwind: sound speed falls with height")]
    verdicts = [T("bent down: a duct hugs the ground, the receiver "
                  "stays loud"),
                T("bent up: a shadow opens, the receiver goes quiet")]
    ims: list[Any] = []
    ims_e: list[Any] = []
    v_txts: list[Any] = []
    ray_lines: list[Any] = []
    for row in range(2):
        ax_c = fig.add_subplot(gs[row, 0])
        _grid_axes(ax_c)
        ax_c.plot(c_profs[row], z, color=COLOR_PRIMARY, lw=1.6)
        ax_c.plot([float(np.interp(src_h, z, c_profs[row]))], [src_h],
                  marker="o", ms=5, color=COLOR_TERTIARY,
                  markeredgecolor="white", markeredgewidth=0.8)
        ax_c.set_ylim(0.0, 105.0)
        ax_c.set_xlim(332.0, 348.0)
        ax_c.set_xticks([335.0, 345.0])
        ax_c.set_ylabel(T("Height [m]"), fontsize=8)
        ax_c.tick_params(labelsize=6)
        if row == 1:
            ax_c.set_xlabel(T("c_eff(z) [m/s]"), fontsize=7)

        ax_f = fig.add_subplot(gs[row, 1])
        ax_f.grid(False)
        im = ax_f.imshow(p_all[row][0], origin="lower", extent=extent,
                         cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
                         aspect="auto", interpolation="bilinear")
        im_e = ax_f.imshow(e_db[row], origin="lower", extent=extent,
                           cmap="magma", vmin=-30.0, vmax=0.0,
                           aspect="auto", interpolation="bilinear",
                           alpha=0.0, zorder=2.5)
        ax_f.set_title(titles[row], fontsize=10, fontweight="bold")
        ax_f.set_xlim(14.0, 430.0)
        ax_f.set_ylim(-7.0, 105.0)
        ax_f.fill_between([14.0, 430.0], -7.0, 0.0, facecolor=COLOR_GRID,
                          edgecolor=COLOR_FG, lw=0.8, hatch="///")
        # Library ray fan through the same profile, revealed once the
        # wavefront has drawn the geometry it explains.
        lines_row = []
        result = rays[row]
        for i in range(result.heights.shape[0]):
            (ln,) = ax_f.plot(src_x + result.ranges[i],
                              result.heights[i], color="white",
                              lw=0.8, alpha=0.0, zorder=3.4,
                              path_effects=[patheffects.withStroke(
                                  linewidth=1.5, foreground="#40404060")])
            lines_row.append(ln)
        ray_lines.append(lines_row)
        ax_f.plot([src_x], [src_h], marker="o", ms=5,
                  color=COLOR_TERTIARY, markeredgecolor=FIELD_STROKE,
                  markeredgewidth=0.8, zorder=4)
        ax_f.text(src_x + 8.0, src_h + 4.0, T("source (h = 2 m)"),
                  ha="left", va="bottom", color=FIELD_INK, fontsize=7.5,
                  path_effects=outline, zorder=4)
        ax_f.plot([recv_x], [src_h], marker="o", ms=5, color=FIELD_STROKE,
                  markeredgecolor=FIELD_INK, markeredgewidth=0.8, zorder=4)
        ax_f.text(recv_x, src_h + 6.0, T("receiver 350 m"), ha="center",
                  va="bottom", color=FIELD_INK, fontsize=7.5,
                  path_effects=outline, zorder=4)
        if row == 0:
            ax_f.text(20.0, -3.5, T("rigid ground"), ha="left",
                      va="center", color=COLOR_FG, fontsize=6.5,
                      bbox={"boxstyle": "round,pad=0.2",
                            "facecolor": fig.get_facecolor(),
                            "edgecolor": "none"})
            ax_f.text(22.0, 97.0, T(f"f = {_REFR_F:.0f} Hz"), ha="left",
                      va="top", color=FIELD_INK, fontsize=7.5,
                      path_effects=outline, zorder=4)
        else:
            # The ray-model shadow boundary of the library, where the
            # grazing ray leaves the ground for good.
            ax_f.axvline(src_x + x_shadow, color="#888888", ls="--",
                         lw=0.9, alpha=0.8, zorder=3)
            ax_f.text(src_x + x_shadow + 6.0, 84.0,
                      T(f"shadow beyond ≈ {x_shadow:.0f} m (ray model)"),
                      ha="left", va="top", color=FIELD_INK, fontsize=7,
                      path_effects=outline, zorder=4)
        v_txt = ax_f.text(424.0, 97.0, "", ha="right", va="top",
                          color="white", fontsize=8, zorder=4,
                          bbox={"boxstyle": _ANIM_PILL_BOX,
                                "facecolor": "black", "alpha": 0.45,
                                "edgecolor": "none"})
        ax_f.tick_params(labelsize=7, labelleft=False)
        if row == 0:
            ax_f.tick_params(labelbottom=False)
        else:
            ax_f.set_xlabel(T("Range [m]"), fontsize=8)
        ims.append(im)
        ims_e.append(im_e)
        v_txts.append(v_txt)
        if row == 1:
            # Inside the lower field, top-left: the Spanish suptitle and
            # row titles leave no free margin at the figure corners, and
            # this corner stays clear of the shadow-boundary annotation.
            t_txt = ax_f.text(20.0, 97.0, "", ha="left", va="top",
                              family="monospace", fontsize=9,
                              color="white", zorder=4,
                              bbox={"boxstyle": _ANIM_PILL_BOX,
                                    "facecolor": "black", "alpha": 0.45,
                                    "edgecolor": "none"})
    # The field is already steady when the clip starts, so the ray fan can
    # come in early and the RMS verdict follows once the streaming motion
    # has been established.
    rays_on = int(0.18 * p_all.shape[1])
    reveal = int(0.80 * p_all.shape[1])

    def update(k: int) -> tuple[Any, ...]:
        alpha_e = min(1.0, max(0.0, (k - reveal) / 12.0))
        alpha_r = min(0.65, max(0.0, (k - rays_on) / 20.0))
        arts: list[Any] = []
        for row in range(2):
            ims[row].set_data(p_all[row][k])
            ims_e[row].set_alpha(alpha_e)
            for ln in ray_lines[row]:
                ln.set_alpha(alpha_r)
                arts.append(ln)
            v_txts[row].set_text(verdicts[row] if k >= reveal else "")
        t_txt.set_text(T(f"t = {times[k]:5.2f} s"))
        return (*ims, *ims_e, *arts, *v_txts, t_txt)

    # gif_fps 4: the steady CW field moves everywhere in every frame, the
    # worst case for GIF palette coding; 8 fps left the fallback near 8 MB
    # where every other FDTD clip stays under 4 MB.
    _render_clip(fig, update, output_dir, "anim_fdtd_refraction",
                 frames=int(p_all.shape[1]), fps=_REFR_FPS, gif_fps=4)


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
            ax.text(_EJ_JUNC_X - 0.055, y_end - 0.005, T("free end"),
                    ha="right", va="bottom", color="white", fontsize=7)
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
        ax.tick_params(labelsize=7)
        v_txt = ax.text(x1 - 0.015, y1 - 0.02, "", ha="right", va="bottom",
                        color="white", fontsize=8, zorder=5,
                        bbox={"boxstyle": _ANIM_PILL_BOX,
                              "facecolor": "black", "alpha": 0.5,
                              "edgecolor": "none"})
        ims.append(im)
        v_txts.append(v_txt)
    axes[1].set_xlabel("x [m]", fontsize=8)
    t_txt = fig.text(0.985, 0.965, "", ha="right", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)
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
    verdicts = [T(f"below f_c: the mass law holds: {trans_db[0]:.0f} dB "
                  f"(it predicts {-ml[0]:.0f})"),
                T(f"above f_c: trace matches λ_B: {trans_db[1]:.0f} dB, "
                  f"the mass law said {-ml[1]:.0f}")]
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
            ax.text(x0 + 0.02, _EC_PLATE_Y - 0.012,
                    T("10 mm steel plate"), ha="left", va="bottom",
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
