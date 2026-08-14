#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The quarter-wave side branch: a resonator charging, on tune and off.

The silencers guide states that a side branch's notch is "only a few hertz
wide", that its interior works at a large amplitude, and that the practical
procedure is to compute a nominal length, build it long and trim it against
a measurement -- and it shows none of the three. This clip runs the guide's
own 0.30 m closed stub on its 0.01 m2 duct at two drive frequencies: on the
library tuning, where the stub *charges* -- the closed-end amplitude ratchets
up over several periods to several times the incident wave -- and off tune,
where it never does. A pulse pre-run on the same grid measures the 2D
device's own resonance, which the junction end correction has pulled below
the drilled-length tuning; the clip reports the effective length
``c / (4 f_sim)`` that falls out of it, which is the guide's "build it long
and trim it" done once on screen.

The trap the design catalogue recorded, honoured here: **no transmission
loss is annotated anywhere.** In the lossless four-pole model the TL at the
tuning frequency is infinite (the figure's 60 dB is only the frequency
grid's resolution), and the 2D junction's end correction moves the simulated
notch by more than the notch is wide, so any TL number would disagree with
the picture by tens of decibels. What is robust is the internal amplitude,
whose bandwidth is ``f/Q`` and therefore percent-wide.
"""

from functools import lru_cache
from typing import Any

import numpy as np

from ..media import (
    _ANIM_PILL_BOX,
    _anim_figure,
    _render_clip,
    _translate_str,
)
from ..theme import (
    CMAP_FIELD,
    COLOR_FG,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
)
from ._core import _anim_speaker

_SBR_C0 = 343.0
_SBR_DX = 0.0025
#: Duct bore in cells: the guide's 0.01 m2 pipe has a 112.8 mm equivalent
#: diameter, rounded to 45 cells (112.5 mm) on this grid.
_SBR_BORE_C = 45
#: Stub width in cells: the guide's 2e-3 / 0.01 m2 area ratio read as a 2D
#: height ratio, 0.2 x the bore = 22.5 mm.
_SBR_STUB_W = 9
_SBR_STUB_D = 120                       # 0.30 m closed stub, the guide's own
_SBR_NX = 800                           # 2.0 m duct
_SBR_JUNC = 480                         # stub mouth at x = 1.2 m
_SBR_WALL = 4                           # wall cells around the air path
_SBR_F_OFF = 150.0                      # the off-tune drive
# Timeline. Mesh: dx = min(smallest scene dimension / 4, lambda / 8 at the
# highest carrier) = min(22.5 mm / 4 = 5.6 mm, 1.2 m / 8 = 150 mm); the
# 2.5 mm grid sits finer still so the 9-cell stub mouth rasterises cleanly
# (it is the expansion-chamber rig with the stub in place of the chamber).
# Flight: source (left edge) -> junction at 1.2 m -> stub closed end
# (+0.3 m, the deepest well) -> back to the mouth -> back to the farthest
# visible point (the left end): 3.0 m / 343 x 1.2 = 10.5 ms. That floor is
# only a floor: the payload is the charge, which the ring-down pre-run
# measures at tau_amp ~ 9.3 ms, so the window is 36.0 ms -- the front
# arrives at the junction at 3.5 ms and the closed-end amplitude passes
# 90 % of its plateau ~6 periods later, leaving ~2 settled periods before
# the hold. Sampling: one frame every 20 solver steps (dt = 3.09 us) is
# 61.85 us, i.e. 56.6 frames per period of the 285.8 Hz carrier (>= 48).
# 582 frames at 48 fps = 12.1 s + the shared 2 s hold.
_SBR_EVERY = 20
_SBR_FRAMES = 582
_SBR_FPS = 48.0
_SBR_HOLD = round(2.0 * _SBR_FPS)
#: Trace decimation: every 4 solver steps = 12.4 us, 282 points per period.
_SBR_TRACE_EVERY = 4


def _sbr_rig() -> tuple[Any, Any, int]:
    """The duct-plus-stub rig: air mask, density-contrast walls, rho c ends.

    Walls are a 10^6 density contrast rather than an obstacle mask so the
    sustained plane wave can be injected across the full left edge, exactly
    as in the expansion-chamber clip this rig is cloned from.
    """
    import fdtd2d

    ny = _SBR_WALL + _SBR_BORE_C + _SBR_STUB_D + _SBR_WALL
    air = np.zeros((ny, _SBR_NX), dtype=bool)
    duct0 = _SBR_WALL
    air[duct0:duct0 + _SBR_BORE_C, :] = True
    air[duct0 + _SBR_BORE_C:duct0 + _SBR_BORE_C + _SBR_STUB_D,
        _SBR_JUNC:_SBR_JUNC + _SBR_STUB_W] = True
    rho_map = np.where(air, 1.2, 1.2e6)
    sim = fdtd2d.FDTD2D(_SBR_C0, _SBR_DX, shape=(ny, _SBR_NX), rho=rho_map,
                        edge_impedance={"left": 1.2 * _SBR_C0,
                                        "right": 1.2 * _SBR_C0})
    return sim, air, duct0


def _sbr_ring_down() -> float:
    """Measure the 2D device's own resonance from a pulse ring-down.

    A broadband plane pulse rings the stub; the closed-end trace after the
    pulse has left (t > 30 ms) is windowed, zero-padded and peak-refined.
    Measured on the same grid the clip runs on, because the junction end
    correction this measures is a property of the rasterised device.
    """
    import fdtd2d

    sim, _, duct0 = _sbr_rig()
    w = 0.0008

    def wave(t: float) -> float:
        return float(np.exp(-(((t - 4.0 * w) / w) ** 2)))

    sim.add_source(fdtd2d.PlaneWaveSource("right", wave, offset=2))
    iy = duct0 + _SBR_BORE_C + _SBR_STUB_D - 1
    ix = _SBR_JUNC + _SBR_STUB_W // 2
    n_steps = round(0.10 / sim.dt)
    trace = np.empty(n_steps)
    for i in range(n_steps):
        sim.step()
        trace[i] = sim.p[iy, ix]
    t = (np.arange(n_steps) + 1) * sim.dt
    tail = trace[t > 0.03]
    spec = np.abs(np.fft.rfft(tail * np.hanning(tail.size), n=2 ** 18))
    freqs = np.fft.rfftfreq(2 ** 18, sim.dt)
    band = (freqs > 150.0) & (freqs < 450.0)
    k = int(np.argmax(spec[band]))
    f_pk = float(freqs[band][k])
    y0, y1, y2 = (float(v) for v in np.log(spec[band][k - 1:k + 2]))
    return f_pk + float(freqs[1] - freqs[0]) * 0.5 * (y0 - y2) / (
        y0 - 2.0 * y1 + y2)


@lru_cache(maxsize=1)
def _side_branch_fields() -> tuple[Any, ...]:
    """The ring-down and the two CW runs, cached for the four variants.

    Returns the wall-masked frame stacks (on tune, off tune), the frame
    times, the closed-end pressure traces with their own time axis, the
    library tuning frequency, the measured 2D resonance, and the two
    closed-end plateau ratios with the on-tune charge time in periods.
    """
    import fdtd2d

    from phonometry import quarter_wave_resonator

    f_grid = np.linspace(20.0, 600.0, 4000)
    resonances = quarter_wave_resonator(f_grid, duct_area=0.01, length=0.3,
                                        branch_area=2e-3).resonances
    if resonances is None:  # never for a closed stub; narrows the type
        raise RuntimeError("quarter_wave_resonator returned no resonances")
    f0 = float(resonances[0])
    f_sim = _sbr_ring_down()

    runs: list[tuple[Any, Any]] = []
    metrics: list[tuple[float, float]] = []
    times = np.zeros(0)
    t_tr = np.zeros(0)
    for f in (f0, _SBR_F_OFF):
        sim, air, duct0 = _sbr_rig()
        tone = fdtd2d.CWSource(0, 0, frequency=f, ramp_cycles=2.0)
        sim.add_source(fdtd2d.PlaneWaveSource("right", tone.value, offset=2))
        iy = duct0 + _SBR_BORE_C + _SBR_STUB_D - 1
        ix = _SBR_JUNC + _SBR_STUB_W // 2
        n_steps = _SBR_EVERY * _SBR_FRAMES
        ps: list[Any] = []
        ts: list[float] = []
        tr: list[float] = []
        tt: list[float] = []
        for i in range(n_steps):
            sim.step()
            if (i + 1) % _SBR_TRACE_EVERY == 0:
                tr.append(float(sim.p[iy, ix]))
                tt.append(sim.time)
            if (i + 1) % _SBR_EVERY == 0:
                ps.append(np.where(air, sim.p, np.nan)[::2, ::2]
                          .astype(np.float32))
                ts.append(sim.time)
        trace = np.asarray(tr)
        # Per-period closed-end amplitude: the plateau ratio (the incident
        # wave has amplitude 1) and the time to 90 % of it, counted in
        # periods of the drive from the front's arrival at the junction.
        n_per = round(1.0 / (f * sim.dt * _SBR_TRACE_EVERY))
        n_full = (trace.size // n_per) * n_per
        amps = np.abs(trace[:n_full]).reshape(-1, n_per).max(axis=1)
        plateau = float(amps[-3:].mean())
        t_arr = (_SBR_JUNC * _SBR_DX) / _SBR_C0
        k90 = int(np.argmax(amps >= 0.9 * plateau))
        t90 = (k90 + 1) / f                # per-period bins of the drive
        periods90 = (t90 - t_arr) * f
        metrics.append((plateau, float(periods90)))
        runs.append((np.stack(ps), trace))
        times = np.asarray(ts)
        t_tr = np.asarray(tt)
    (p_on, tr_on), (p_off, tr_off) = runs
    return (p_on, p_off, times, t_tr, tr_on, tr_off, f0, f_sim,
            metrics[0], metrics[1])


def animate_fdtd_side_branch(output_dir: str) -> None:
    """The quarter-wave side branch charging (2D FDTD): the guide's 0.30 m
    closed stub on its duct, driven at the tuning frequency and off it. On
    tune the closed-end pressure ratchets up over ~6 periods to ~8 times
    the incident wave; off tune it never charges. The ring-down pre-run
    measures the 2D device's own resonance below the drilled-length tuning
    (the junction end correction), and the clip reports the effective
    length c/(4 f_sim) -- the guide's build-long-and-trim procedure done
    on screen. No model TL is annotated anywhere: the lossless notch is
    infinite at f0, while the charge bandwidth f/Q is percent-wide."""
    from matplotlib.patches import Rectangle

    T = _translate_str
    (p_on, p_off, times, t_tr, tr_on, tr_off, f0, f_sim,
     (ratio_on, per90_on), (ratio_off, _)) = _side_branch_fields()
    l_eff = _SBR_C0 / (4.0 * f_sim) * 1000.0          # [mm]
    length = _SBR_NX * _SBR_DX
    ny = _SBR_WALL + _SBR_BORE_C + _SBR_STUB_D + _SBR_WALL
    height = ny * _SBR_DX
    duct_y0 = _SBR_WALL * _SBR_DX
    duct_y1 = duct_y0 + _SBR_BORE_C * _SBR_DX
    stub_x0 = _SBR_JUNC * _SBR_DX
    stub_x1 = stub_x0 + _SBR_STUB_W * _SBR_DX
    stub_y1 = duct_y1 + _SBR_STUB_D * _SBR_DX
    # Colour scale from the duct region of the settled on-tune run, at 70 %
    # so the standing-wave antinodes saturate boldly; the stub interior
    # sits far above it and saturates hard on purpose -- the point of the
    # clip is that the inside of the branch runs several times hotter than
    # the duct that feeds it.
    duct_rows = slice(_SBR_WALL // 2, (_SBR_WALL + _SBR_BORE_C) // 2)
    vmax = 0.7 * float(np.nanquantile(
        np.abs(p_on[p_on.shape[0] // 2:, duct_rows]), 0.999))

    fig = _anim_figure()
    fig.suptitle(T("The quarter-wave side branch, on and off tune (2D FDTD)"),
                 )
    gs = fig.add_gridspec(3, 1, height_ratios=(1.0, 1.0, 0.74), hspace=0.1)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]
    ax_t = fig.add_subplot(gs[2, 0])

    titles = [T(f"On the tuning frequency: {f0:.1f} Hz"),
              T(f"Off tune: {_SBR_F_OFF:.0f} Hz")]
    ims: list[Any] = []
    pills: list[Any] = []
    trims: list[Any] = []
    for panel, (ax, title) in enumerate(zip(axes, titles, strict=True)):
        ax.grid(False)
        # The walls are NaN in the frames; a grey slab behind the imshow
        # renders them as solid hardware, and the air path draws over it.
        ax.add_patch(Rectangle((0.0, 0.0), length, height,
                               facecolor="#9a9a9a", edgecolor="none",
                               zorder=1))
        ims.append(ax.imshow(np.zeros((2, 2)), origin="lower",
                             extent=(0.0, length, 0.0, height),
                             cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
                             aspect="auto", interpolation="bilinear",
                             zorder=2))
        # Air-path outline: duct bore, stub throat and cap.
        for xa, xb, y in ((0.0, stub_x0, duct_y1), (stub_x1, length, duct_y1),
                          (0.0, length, duct_y0)):
            ax.plot([xa, xb], [y, y], color=COLOR_FG, lw=0.9, zorder=3)
        for x in (stub_x0, stub_x1):
            ax.plot([x, x], [duct_y1, stub_y1], color=COLOR_FG, lw=0.9,
                    zorder=3)
        ax.plot([stub_x0, stub_x1], [stub_y1, stub_y1], color=COLOR_FG,
                lw=0.9, zorder=3)
        _anim_speaker(ax, 0.0, 0.5 * (duct_y0 + duct_y1),
                      duct_y1 - duct_y0, tip_inset=0.003,
                      label_y=None)
        ax.add_patch(Rectangle((length, duct_y0 - 0.01), 0.045,
                               duct_y1 - duct_y0 + 0.02,
                               facecolor="#9a9a9a", hatch="////",
                               edgecolor=COLOR_FG, linewidth=0.8, zorder=3))
        ax.text(stub_x1 + 0.03, duct_y1 + 0.24,
                T("closed stub, built $\\ell$ = 300 mm"), fontsize=7.5,
                ha="left", va="center", color=COLOR_FG, zorder=4)
        if panel == 1:
            # Labelled once, on the panel without the trim verdict: the
            # rig is identical in both.
            ax.text(length - 0.06, duct_y1 + 0.035,
                    T("anechoic termination"), ha="right", va="bottom",
                    fontsize=7.0, color=COLOR_FG, zorder=4)
        pills.append(ax.text(
            0.03, height - 0.012, "", ha="left", va="top", fontsize=9,
            color="white", zorder=5,
            bbox={"boxstyle": _ANIM_PILL_BOX, "facecolor": "black",
                  "alpha": 0.55, "edgecolor": "none"}))
        # The trim verdict lives in the wall grey under the stub label,
        # where no field or curve can ever strike it; only the on-tune
        # panel's copy is filled in (see update()).
        trims.append(ax.text(
            stub_x1 + 0.03, duct_y1 + 0.185, "", ha="left", va="top",
            fontsize=7.8, color=COLOR_FG, linespacing=1.5, zorder=4))
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.115, length + 0.065)
        ax.set_ylim(-0.012, height + 0.012)
        ax.set_yticks([])
        ax.tick_params(labelsize=7, labelbottom=False)
    axes[1].tick_params(labelbottom=True)
    axes[1].set_xlabel(T("Position along the duct [m]"), fontsize=8.5,
                       labelpad=1.5)

    # The closed-end pressure of both runs: the charge is one curve
    # ratcheting up over several periods while the other stays flat.
    ax_t.set_xlim(0.0, float(times[-1]) * 1e3)
    ax_t.set_ylim(-10.4, 10.4)
    ax_t.set_xlabel(T("Time [ms]"), fontsize=8.5, labelpad=1.5)
    # Two lines: rotated on a one-inch axis, a single-line label runs off
    # the canvas bottom once the Spanish text lengthens it.
    ax_t.set_ylabel(T("closed-end\npressure [Pa]"), fontsize=8)
    ax_t.tick_params(labelsize=7)
    ax_t.grid(True, color=COLOR_GRID, lw=0.6)
    # The +-1 guides are the incident amplitude; the footer says so, which
    # keeps the axis clear of a label the red curve would run through.
    for y in (1.0, -1.0):
        ax_t.axhline(y, color=COLOR_MUTED, lw=0.9, ls="--", zorder=2)
    line_on = ax_t.plot([], [], color=COLOR_SECONDARY, lw=1.1,
                        label=T("on tune"), zorder=4)[0]
    line_off = ax_t.plot([], [], color=COLOR_PRIMARY, lw=1.1,
                         label=T("off tune"), zorder=3)[0]
    ax_t.legend(fontsize=7, loc="upper left", framealpha=0.85)
    t_txt = fig.text(0.988, 0.955, "", ha="right", va="top",
                     family="monospace", fontsize=9.5, color=COLOR_FG)
    fig.get_layout_engine().set(rect=(0.0, 0.03, 1.0, 0.965))
    fig.text(0.5, 0.005,
             T("Rigid walls, anechoic ends, incident amplitude 1; the "
               "charge bandwidth $f/Q$ is percent-wide, the lossless notch "
               "hertz-wide."),
             ha="center", va="bottom", fontsize=7.2, color=COLOR_FG,
             alpha=0.85)

    reveal = int(0.68 * times.size)
    verdicts = (T(f"×{ratio_on:.1f} the incident wave, in ≈"
                  f"{per90_on:.0f} periods"),
                T(f"×{ratio_off:.1f} at once: it never charges"))
    trim = T(f"the stub rings at {f_sim:.1f} Hz, not {f0:.1f}:\n"
             f"$\\ell_{{\\mathrm{{eff}}}} = c/4f$ = {l_eff:.0f} mm, "
             f"so trim it to tune")

    def update(k: int) -> tuple[Any, ...]:
        j = min(k, times.size - 1)
        keep = t_tr <= times[j]
        ims[0].set_data(p_on[j])
        ims[1].set_data(p_off[j])
        line_on.set_data(t_tr[keep] * 1e3, tr_on[keep])
        line_off.set_data(t_tr[keep] * 1e3, tr_off[keep])
        for pill, verdict in zip(pills, verdicts, strict=True):
            pill.set_text(verdict if k >= reveal else "")
        trims[0].set_text(trim if k >= reveal else "")
        t_txt.set_text(T(f"$t$ = {times[j] * 1e3:4.1f} ms"))
        return (*ims, line_on, line_off, *pills, trims[0], t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_side_branch",
                 frames=times.size + _SBR_HOLD, fps=_SBR_FPS, gif_fps=10)
