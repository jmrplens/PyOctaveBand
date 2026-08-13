#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The critical grazing angle, drawn as a field and measured on it.

The propagation guide states total reflection below
``psi_c = arccos(c1/c2)`` and reads the angle filtering of shallow water off
it, and its only pictures of either are two ``|R|(psi)`` curves. This clip
runs one scene twice, changing nothing but the two numbers that describe the
sediment: the guide's own fast sand, which has a critical angle, and a soft
mud, which has none. What the reader watches is the beam entering the sand
switching off as the expanding front's contact with the seabed sweeps past
the critical ray, while the mud keeps taking energy at every angle -- and a
lower axis where the *measured* energy flux through the seabed fills in
behind the contact point and lands on the closed form.
"""

from functools import lru_cache
from typing import Any

import numpy as np

from ..i18n import _fmt_minus
from ..media import _ANIM_PILL_BOX, _anim_figure, _render_clip, _translate_str
from ..theme import (
    CMAP_FIELD,
    COLOR_FG,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
)

_SB_C1, _SB_RHO1 = 1500.0, 1000.0         # water: the guide's own pair
#: The two sediments. Sand is the guide's water-over-sand example, whose
#: critical angle its two figures are drawn for; mud is slower than the
#: water, so ``arccos(c1/c2)`` has no solution and no angle is protected.
_SB_BOTTOMS = (("sand", 1650.0, 1900.0), ("mud", 1450.0, 1600.0))
_SB_WATER = 60.0                          # water shown above the seabed [m]
_SB_SED = 20.0                            # sediment shown below it [m]
_SB_ZS = 24.0                             # source depth [m]
_SB_XS = 40.0                             # source range [m]
_SB_RANGE = 450.0                         # range shown [m]
_SB_SWEEP = 280.0                         # seabed section the flight covers
_SB_F0 = 100.0                            # burst centre frequency [Hz]
_SB_TAU = 0.005                           # Gaussian envelope half-width [s]
_SB_DX = 0.8
_SB_SPONGE = 40                           # cells, all four sides
# Timeline. Mesh: the burst is 20 dB down by 168 Hz, where the slowest
# medium in the scene (the 1 450 m/s mud) has lambda = 8.63 m and
# lambda / 8 = 1.08 m; the smallest geometric dimension is the 20 m of
# sediment shown (20 / 4 = 5 m), so the rule allows 1.08 m and dx = 0.8 m
# sits finer, giving 18.75 cells per wavelength at the 100 Hz centre.
# Flight: the wave's job here is to sweep the seabed, and the contact point
# of the expanding front runs from directly under the source out to the far
# end of the measured section at 280 m, where the grazing angle is 7.3
# degrees; that is 282.3 m of travel at 1 500 m/s plus the 15 ms burst
# delay = 203.2 ms, x 1.2 = 243.8 ms captured. The frame is 450 m wide and
# not 280 for the same reason clip-mate anim_fdtd_dispersion is: over a
# window scaled to the sweep the front travels half as far again, and sizing
# the *view* on the sweep would close the clip with the front already out of
# it, which is the one thing a field clip must not do.
# Sampling: dt = 205.7 us on this grid (the sand run sets it), one solver
# step per frame, i.e. 48.6 frames per period of the 100 Hz centre (>= 48).
# 1 186 frames cover the window, played at 100 fps: 11.9 s, plus the shared
# 2 s hold.
_SB_FPS = 100.0
_SB_HOLD = round(2.0 * _SB_FPS)
#: Burst delay: the modulated Gaussian peaks three half-widths in.
_SB_T0 = 3.0 * _SB_TAU


def _sb_geometry() -> tuple[int, int, int, int, int]:
    """Visible columns and rows, the seabed row inside the visible block,
    and the source cell in absolute grid indices."""
    n_vis = round(_SB_RANGE / _SB_DX)
    rows_water = round(_SB_WATER / _SB_DX)
    n_rows = rows_water + round(_SB_SED / _SB_DX)
    ix_src = _SB_SPONGE + round(_SB_XS / _SB_DX)
    iy_src = _SB_SPONGE + round(_SB_ZS / _SB_DX)
    return n_vis, n_rows, rows_water, ix_src, iy_src


def _sb_front_angle(times: Any) -> Any:
    """Grazing angle at which the direct front meets the seabed [deg].

    The front is a circle of radius ``c1 (t - t0)`` about a source sitting
    ``h`` above the bed, so the angle between the front and the interface
    at the contact point is ``arcsin(h / R)``: 90 degrees where the front
    first touches, falling as the contact runs outward. NaN before contact.
    """
    height = _SB_WATER - _SB_ZS
    radius = _SB_C1 * (np.asarray(times) - _SB_T0)
    with np.errstate(invalid="ignore", divide="ignore"):
        angle = np.degrees(np.arcsin(np.clip(height / radius, -1.0, 1.0)))
    return np.where(radius >= height, angle, np.nan)


@lru_cache(maxsize=1)
def _seabed_fields() -> tuple[Any, ...]:
    """Two runs of one water-over-sediment scene, sand and mud, cached.

    Identical but for ``c2`` and ``rho2``. Sponges on all four sides, so
    the only interface in the model is the seabed and nothing returns from
    the edges. The source is a one-cycle 100 Hz Gaussian-modulated burst.

    The measurement is the **net energy flux through the seabed**,
    ``\\int p v_z dt`` accumulated per column on the interface face: it is
    exactly what enters the bottom, it needs no ray tracing, and an
    evanescent skin contributes nothing to it, which is what makes the sand
    run go to zero past the critical ray instead of merely dim.

    The two runs step at different rates (the sand's 1 650 m/s sets a
    shorter ``dt`` than the mud's 1 500 m/s water) and are resampled onto
    one common timeline, so the panels advance together.

    Returns the decimated frame stacks, the shared frame times, the running
    seabed flux of each run, the range axis of that flux and the two closed
    forms it is read against.
    """
    import fdtd2d

    from phonometry import underwater

    n_vis, n_rows, rows_water, ix_src, iy_src = _sb_geometry()
    n_x = n_vis + 2 * _SB_SPONGE
    n_y = n_rows + 2 * _SB_SPONGE
    i_bed = _SB_SPONGE + rows_water
    sample_rate = 8000.0
    n_samples = round(6.0 * _SB_TAU * sample_rate)
    t_sig = np.arange(n_samples) / sample_rate
    burst = np.exp(-(((t_sig - _SB_T0) / _SB_TAU) ** 2)) * np.sin(
        2.0 * np.pi * _SB_F0 * (t_sig - _SB_T0))
    height = _SB_WATER - _SB_ZS
    t_end = 1.2 * (float(np.hypot(_SB_SWEEP, height)) / _SB_C1 + _SB_T0)

    raw_frames: list[Any] = []
    raw_flux: list[Any] = []
    raw_times: list[Any] = []
    for _, c2, rho2 in _SB_BOTTOMS:
        c_map = np.full((n_y, n_x), _SB_C1)
        rho_map = np.full((n_y, n_x), _SB_RHO1)
        c_map[i_bed:, :] = c2
        rho_map[i_bed:, :] = rho2
        sim = fdtd2d.FDTD2D(c_map, _SB_DX, rho=rho_map,
                            sponge_width=_SB_SPONGE)
        sim.add_source(fdtd2d.SignalSource(ix=ix_src, iy=iy_src,
                                           samples=burst,
                                           sample_rate=sample_rate))
        flux = np.zeros(n_vis)
        ps: list[Any] = []
        fs_: list[Any] = []
        ts: list[float] = []
        cut = slice(_SB_SPONGE, _SB_SPONGE + n_vis)
        while sim.time <= t_end:
            sim.step()
            face = 0.5 * (sim.p[i_bed - 1, cut] + sim.p[i_bed, cut])
            flux += face * sim.vy[i_bed - 1, cut] * sim.dt
            ps.append(sim.p[_SB_SPONGE:_SB_SPONGE + n_rows, cut]
                      [::2, ::2].astype(np.float32))
            fs_.append(flux.astype(np.float32).copy())
            ts.append(sim.time)
        raw_frames.append(np.stack(ps))
        raw_flux.append(np.stack(fs_))
        raw_times.append(np.asarray(ts))

    # One timeline for both panels: the finer-dt run sets it, the other is
    # resampled onto it by linear interpolation between its own steps.
    lead = int(np.argmax([t.size for t in raw_times]))
    times = raw_times[lead]
    frames: list[Any] = []
    fluxes: list[Any] = []
    for k in range(len(_SB_BOTTOMS)):
        if k == lead:
            frames.append(raw_frames[k][:times.size])
            fluxes.append(raw_flux[k][:times.size])
            continue
        pos = np.clip(times / float(raw_times[k][1] - raw_times[k][0]) - 1.0,
                      0.0, raw_times[k].size - 1.0001)
        lo = pos.astype(int)
        w = (pos - lo).astype(np.float32)
        frames.append((1.0 - w)[:, None, None] * raw_frames[k][lo]
                      + w[:, None, None] * raw_frames[k][lo + 1])
        fluxes.append((1.0 - w)[:, None] * raw_flux[k][lo]
                      + w[:, None] * raw_flux[k][lo + 1])

    x_flux = (np.arange(n_vis) + 0.5) * _SB_DX
    offset = np.maximum(x_flux - _SB_XS, 1e-6)
    psi = np.degrees(np.arctan2(height, offset))
    closed: list[Any] = []
    criticals: list[Any] = []
    for _, c2, rho2 in _SB_BOTTOMS:
        res = underwater.seabed_reflection(psi, rho1=_SB_RHO1, c1=_SB_C1,
                                           rho2=rho2, c2=c2)
        # A cylindrical front spreads as 1/R in intensity and meets the bed
        # at sin(psi) to its normal, and R = h / sin(psi), so the flux per
        # metre of seabed goes as (1 - |R|^2) sin^2(psi).
        closed.append((1.0 - np.abs(np.asarray(res.magnitude)) ** 2)
                      * np.sin(np.radians(psi)) ** 2)
        criticals.append(res.critical_angle)
    # Each curve is scaled on its own maximum outside the source's near
    # field: what the panel compares is the *shape* of the two, and above
    # all where each of them stops, not an absolute level that a 2D model
    # of a 3D ocean has no claim to. The measured curves are smoothed over
    # 5.6 m of seabed, a third of a wavelength, which is far below the
    # scale of anything the curve is asked to show.
    far = offset > 8.0
    closed = [c / float(np.max(c[far])) for c in closed]
    window = np.ones(7) / 7.0
    smooth = [np.apply_along_axis(
        lambda row: np.convolve(row, window, mode="same"), 1, f)
        for f in fluxes]
    fluxes = [f / float(np.max(f[-1][far])) for f in smooth]
    return (tuple(frames), times, tuple(fluxes), x_flux, psi,
            tuple(closed), tuple(criticals))


def animate_fdtd_critical_angle(output_dir: str) -> None:
    """The critical grazing angle in a field (2D FDTD): one water-over-
    sediment scene run twice, changing only the sediment. Over the guide's
    fast sand the beam entering the bottom switches off as the front's
    contact with the seabed sweeps past the 24.62 degree critical ray, and
    everything striking more shallowly is returned whole; over a slow mud,
    which has no critical angle, sound enters the bottom at every angle
    right out to the edge of the frame. A lower axis fills in the measured
    energy flux through the seabed behind the contact point and lands on
    the closed form."""
    from matplotlib.patches import Rectangle

    from phonometry import underwater

    T = _translate_str
    frames, times, fluxes, x_flux, _, closed, criticals = _seabed_fields()
    n_vis, n_rows, _, _, _ = _sb_geometry()
    length, depth = n_vis * _SB_DX, n_rows * _SB_DX
    height = _SB_WATER - _SB_ZS
    colors = (COLOR_PRIMARY, COLOR_SECONDARY)

    angles = _sb_front_angle(times)
    finite = np.nan_to_num(angles, nan=90.0)
    refl = [np.abs(np.asarray(underwater.seabed_reflection(
        finite, rho1=_SB_RHO1, c1=_SB_C1, rho2=rho2, c2=c2).magnitude))
        for _, c2, rho2 in _SB_BOTTOMS]
    # The contact point of the front with the bed, which is also how far
    # along the seabed the measured curve below is allowed to be believed.
    contact = _SB_XS + np.sqrt(np.maximum(
        (_SB_C1 * (times - _SB_T0)) ** 2 - height**2, 0.0))
    # The verdict both panels are built to deliver: of the energy a bottom
    # takes inside the critical ray, how much more does it take beyond it?
    # For the sand that must be zero, and the measurement finds it slightly
    # negative -- past the critical range the net flux reverses, because
    # the lateral wave born at the critical point runs along the interface
    # in the sediment and radiates back up into the water.
    r_crit = _SB_XS + height / np.tan(np.radians(criticals[0]))
    inside = (x_flux > _SB_XS - 2.0) & (x_flux <= r_crit)
    beyond = x_flux > r_crit
    # (the denominator is zero until the front has reached the bed, and the
    # readout is only revealed long after that)
    ratios = [100.0 * f[:, beyond].sum(axis=1)
              / np.where(f[:, inside].sum(axis=1) > 0.0,
                         f[:, inside].sum(axis=1), np.inf)
              for f in fluxes]
    forms = [100.0 * float(c[beyond].sum() / c[inside].sum())
             for c in closed]
    # A closed form that is exactly zero must not print as "-0.0 %".
    forms = [0.0 if abs(f) < 0.05 else f for f in forms]

    fig = _anim_figure()
    fig.suptitle(T("The critical grazing angle: a fast seabed and a slow one "
                   "(2D FDTD)"), fontweight="bold")
    # The two field panels are drawn to true scale (an angle on screen is
    # the angle), so their height follows their width: the ratios below are
    # the scene's own aspect, and the flux axis takes what is left.
    gs = fig.add_gridspec(3, 1, height_ratios=(1.0, 1.0, 0.46), hspace=0.13)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]
    ax_f = fig.add_subplot(gs[2, 0])

    # Cylindrical spreading is compensated, so the colour scale is set once,
    # from the compensated field after the front has left the source's near
    # field, and what fades afterwards is loss and not geometry.
    t_ref = float(times[40])
    gain = np.sqrt(np.maximum(times, t_ref) / t_ref).astype(np.float32)
    probe = np.stack([frames[0][i] * gain[i]
                      for i in range(60, times.size, 40)])
    peak = float(np.quantile(np.abs(probe), 0.9995))

    ims: list[Any] = []
    pills: list[Any] = []
    for i, ((name, c2, rho2), ax) in enumerate(
            zip(_SB_BOTTOMS, axes, strict=True)):
        ax.grid(False)
        ims.append(ax.imshow(
            np.zeros((2, 2)), origin="upper",
            extent=(0.0, length, depth, 0.0), cmap=CMAP_FIELD,
            vmin=-peak, vmax=peak, aspect="equal",
            interpolation="bilinear", zorder=2))
        ax.axhline(_SB_WATER, color=COLOR_FG, lw=1.6, zorder=4)
        ax.plot([_SB_XS], [_SB_ZS], marker="*", ms=10,
                markerfacecolor="white", markeredgecolor=COLOR_FG,
                markeredgewidth=0.8, zorder=6)
        ax.add_patch(Rectangle(
            (0.0, _SB_WATER), length, depth - _SB_WATER, facecolor="none",
            hatch="///", edgecolor=COLOR_MUTED, alpha=0.55, lw=0.0,
            zorder=3))
        if criticals[i] is not None:
            r_c = _SB_XS + height / np.tan(np.radians(criticals[i]))
            ax.plot([_SB_XS, r_c], [_SB_ZS, _SB_WATER], color=COLOR_FG,
                    lw=1.2, ls="--", zorder=5)
            for a in (ax, ax_f):
                a.axvline(r_c, color=COLOR_MUTED, lw=1.0, ls=":", zorder=3)
            ax.text(r_c - 4.0, 3.0,
                    T(f"critical ray, ψ = {criticals[i]:.2f}°"),
                    fontsize=7.5, ha="right", va="top", color=COLOR_FG,
                    zorder=6)
        ax.text(6.0, _SB_WATER + 4.0,
                T(f"{name}: c₂ = {c2:.0f} m/s, ρ₂ = {rho2:.0f} kg/m³"),
                fontsize=8.5, ha="left", va="top", color=COLOR_FG,
                fontweight="bold", zorder=6)
        ax.set_xlim(0.0, length)
        ax.set_ylim(depth, 0.0)
        ax.set_ylabel(T("Depth [m]"), fontsize=8.5)
        ax.set_yticks([0, 30, 60])
        ax.tick_params(labelsize=7.5, labelbottom=False)
        # The contact angle is geometry and is the same in both panels, so
        # it is printed once in the header; the panels carry only what the
        # sediment does with it, which keeps both languages inside them.
        pills.append(ax.text(
            length - 6.0, 3.0, "", ha="right", va="top", fontsize=8.5,
            color="white", zorder=7, linespacing=1.35,
            bbox={"boxstyle": _ANIM_PILL_BOX, "facecolor": "black",
                  "alpha": 0.55, "edgecolor": "none"}))
        ax.text(length - 6.0, depth - 1.5,
                T("an evanescent skin: bright, and it carries nothing away")
                if criticals[i] is not None
                else T("no critical angle: every angle leaks"),
                fontsize=7.5, ha="right", va="bottom", color=COLOR_FG,
                zorder=6)

    # Logarithmic, because the whole claim lives two decades below the
    # peak: on a linear axis both bottoms look like zero past 120 m, and
    # what has to be seen is that one line stops there and the other does
    # not. A net flux that has reversed sign cannot be drawn on it, and
    # that is exactly the sand's signature past the critical ray.
    ax_f.set_xlim(0.0, length)
    ax_f.set_yscale("log")
    ax_f.set_ylim(6.0e-3, 2.2)
    ax_f.set_xlabel(T("Range [m]"), fontsize=9)
    ax_f.set_ylabel(T("Into the bed"), fontsize=8.5)
    ax_f.set_yticks([0.01, 0.1, 1.0])
    ax_f.set_yticklabels(["0.01", "0.1", "1"])
    ax_f.set_yticks([], minor=True)
    ax_f.tick_params(labelsize=7.5)
    ax_f.grid(True, color=COLOR_GRID, lw=0.6)
    f_lines: list[Any] = []
    for (name, _, _), color, form in zip(_SB_BOTTOMS, colors, closed,
                                         strict=True):
        ax_f.plot(x_flux, np.where(form > 0.0, form, np.nan), color=color,
                  lw=1.0, ls="--", alpha=0.8)
        # The two measured curves coincide until the critical angle, so the
        # first-drawn one has to stay on top or it reads as missing.
        f_lines.append(ax_f.plot([], [], color=color, lw=2.0,
                                 label=T(name),
                                 zorder=6 - len(f_lines))[0])
    ax_f.legend(fontsize=7.5, loc="lower left", ncol=2, framealpha=0.85)
    for label_x, label_psi in ((118.6, 24.6), (210.0, 11.9), (340.0, 6.8)):
        ax_f.text(label_x + 5.0, 1.7, T(f"ψ = {label_psi:.1f}°"),
                  fontsize=7.0, ha="left", va="top", color=COLOR_MUTED)


    t_txt = fig.text(0.012, 0.955, "", ha="left", va="top",
                     family="monospace", fontsize=9.5, color=COLOR_FG)
    # Kept short: at 300 dpi the canvas clips a footer much beyond ~150
    # characters, and the Spanish runs about 15 % longer again.
    fig.text(0.5, 0.008,
             T("Solid: the flux measured into the bed; dashed: "
               "(1 − |R|²) sin²ψ. Each curve on its own maximum; the "
               "field compensated for spreading."),
             ha="center", va="bottom", fontsize=7.2, color=COLOR_FG,
             alpha=0.85)
    fig.get_layout_engine().set(rect=(0.0, 0.035, 1.0, 0.95))

    def update(k: int) -> tuple[Any, ...]:
        j = min(k, times.size - 1)
        artists: list[Any] = [t_txt]
        # Only draw the measured curve where the front has already been.
        keep = x_flux <= contact[j]
        for i in range(2):
            ims[i].set_data(frames[i][j] * gain[j])
            drawn = np.where(fluxes[i][j][keep] > 0.0,
                             fluxes[i][j][keep], np.nan)
            f_lines[i].set_data(x_flux[keep], drawn)
            psi = angles[j]
            if np.isnan(psi):
                pills[i].set_text("")
            else:
                head = (T(f"|R| = {refl[i][j]:.3f}: nothing enters")
                        if criticals[i] is not None and psi <= criticals[i]
                        else T(f"|R| = {refl[i][j]:.3f}"))
                tail = ""
                if contact[j] > r_crit + 60.0:
                    tail = "\n" + T(
                        f"beyond {r_crit:.0f} m the net is "
                        f"{_fmt_minus(ratios[i][j], '+.1f')}"
                        f" % of what entered inside (theory "
                        f"{_fmt_minus(forms[i], '+.1f')} %)")
                pills[i].set_text(head + tail)
            artists += [ims[i], pills[i], f_lines[i]]
        psi = angles[j]
        sweep = ("" if np.isnan(psi)
                 else T(f" · the front meets the bed at ψ = {psi:.0f}°"))
        t_txt.set_text(T(f"t = {times[j] * 1e3:5.1f} ms") + sweep)
        return tuple(artists)

    _render_clip(fig, update, output_dir, "anim_fdtd_critical_angle",
                 frames=times.size + _SB_HOLD, fps=_SB_FPS, gif_fps=12)
