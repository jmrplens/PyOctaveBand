#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Atmospheric refraction: an upward and a downward refracting profile."""

from functools import lru_cache
from typing import Any

import numpy as np

from ..media import _ANIM_PILL_BOX, _anim_figure, _render_clip, _translate_str
from ..schematics import _grid_axes
from ..theme import (
    CMAP_FIELD,
    COLOR_FG,
    COLOR_GRID,
    COLOR_PRIMARY,
    COLOR_TERTIARY,
    FIELD_INK,
    FIELD_STROKE,
)

_REFR_B = 1.0  # log-profile strength [m/s]
_REFR_C0 = 340.0  # ground-level effective speed
_REFR_SRC = (30.0, 2.0)  # source range/height [m]
_REFR_RECV = 350.0  # receiver distance from the source
_REFR_F = 50.0  # CW drive: lambda ~ 6.9 m
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
    (+1 m/s) and upwind (-1 m/s) effective sound speed.
    """
    from phonometry import environment

    return tuple(
        environment.log_linear_sound_speed_profile(
            sign * _REFR_B, ground_speed=_REFR_C0, max_height=140.0
        )
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

    from phonometry import environment

    dx = _REFR_DX
    ny, nx = 440, 1507  # 132 m x 452 m
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
        environment.atmospheric_ray_paths(
            prof,
            source_height=_REFR_SRC[1],
            launch_angles_deg=angles,
            max_range=430.0,
            n_steps=900,
        )
        for prof, angles in zip(
            profiles,
            # Downwind: a shallow fan (the log profile turns anything
            # under ~8 deg back down, so the duct is thin) plus one
            # escaping ray; upwind: the same fan bends up and away.
            (
                (-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0),
                (-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0),
            ),
            strict=True,
        )
    ]
    p_all, r_all = [], []
    times = np.zeros(0)
    for prof in profiles:
        c_prof = prof.speed_at(z)
        c_map = np.repeat(c_prof[:, np.newaxis], nx, axis=1)
        # Row 0 is the ground (rigid, no sponge); "bottom" is the sky in
        # the low-row-origin naming of fdtd2d, as in the barrier clip.
        sim = fdtd2d.FDTD2D(
            c_map, dx, sponge_width=40, sponge_sides=("left", "right", "bottom")
        )
        ix, iy = round(_REFR_SRC[0] / dx), round(_REFR_SRC[1] / dx)
        sim.add_source(
            fdtd2d.CWSource(ix=ix, iy=iy, frequency=_REFR_F, ramp_cycles=2.0)
        )
        beta = float(np.exp(-sim.dt * _REFR_F / 2.0))
        ms = np.zeros_like(sim.p)
        settle = round(1.30 / sim.dt)  # domain transit + ring-up
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
    ref = float(np.quantile(comp[:, :, nx // 4 :], 0.999))
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
    spreading-compensated RMS map.
    """
    from matplotlib import patheffects

    from phonometry import environment

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0, foreground=FIELD_STROKE)]
    # The ray fan is drawn over the scene, and two of its rays run straight
    # through the source and receiver labels (worst on the dark page, where
    # the rays are nearly the colour of the text). A wider halo keeps a
    # clear channel around those two labels instead of a struck-through
    # word; the rays still pass, which is what a ray fan does.
    ray_outline = [patheffects.withStroke(linewidth=3.6, foreground=FIELD_STROKE)]
    p_all, e_db, times, z, c_profs, rays = _refraction_fields()
    profiles = _refraction_profiles()
    half = p_all.shape[1] // 2
    vmax = float(np.quantile(np.abs(p_all[:, half // 2 : half]), 0.999))
    # Shadow-boundary estimate exactly as the guide does it: the log
    # profile's linear-equivalent gradient over the first 10 m, then the
    # closed-form grazing-ray distance.
    grad = float(profiles[1].speed_at(10.0) - profiles[1].speed_at(0.0))
    x_shadow = environment.shadow_zone_distance(
        grad / 10.0, _REFR_SRC[1], _REFR_SRC[1], ground_speed=_REFR_C0
    )
    extent = (0.0, 1507 * _REFR_DX, 0.0, 440 * _REFR_DX)
    src_x, src_h = _REFR_SRC
    recv_x = src_x + _REFR_RECV

    fig = _anim_figure()
    fig.suptitle(T("Atmospheric refraction: downwind duct, upwind shadow (2D FDTD)"))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.20, 1.0])
    titles = [
        T("Downwind: sound speed grows with height"),
        T("Upwind: sound speed falls with height"),
    ]
    verdicts = [
        T("bent down: a duct hugs the ground, the receiver stays loud"),
        T("bent up: a shadow opens, the receiver goes quiet"),
    ]
    ims: list[Any] = []
    ims_e: list[Any] = []
    v_txts: list[Any] = []
    ray_lines: list[Any] = []
    for row in range(2):
        ax_c = fig.add_subplot(gs[row, 0])
        _grid_axes(ax_c)
        ax_c.plot(c_profs[row], z, color=COLOR_PRIMARY, lw=1.6)
        ax_c.plot(
            [float(np.interp(src_h, z, c_profs[row]))],
            [src_h],
            marker="o",
            ms=5,
            color=COLOR_TERTIARY,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        ax_c.set_ylim(0.0, 105.0)
        ax_c.set_xlim(332.0, 348.0)
        ax_c.set_xticks([335.0, 345.0])
        ax_c.set_ylabel(T("Height [m]"), fontsize=8)
        ax_c.tick_params(labelsize=6)
        if row == 1:
            ax_c.set_xlabel(T(r"$c_{\mathrm{eff}}(z)$ [m/s]"), fontsize=7)

        ax_f = fig.add_subplot(gs[row, 1])
        ax_f.grid(False)
        im = ax_f.imshow(
            p_all[row][0],
            origin="lower",
            extent=extent,
            cmap=CMAP_FIELD,
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
            interpolation="bilinear",
        )
        im_e = ax_f.imshow(
            e_db[row],
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=-30.0,
            vmax=0.0,
            aspect="auto",
            interpolation="bilinear",
            alpha=0.0,
            zorder=2.5,
        )
        ax_f.set_title(titles[row], fontsize=10)
        ax_f.set_xlim(14.0, 430.0)
        ax_f.set_ylim(-7.0, 105.0)
        ax_f.fill_between(
            [14.0, 430.0],
            -7.0,
            0.0,
            facecolor=COLOR_GRID,
            edgecolor=COLOR_FG,
            lw=0.8,
            hatch="///",
        )
        # Library ray fan through the same profile, revealed once the
        # wavefront has drawn the geometry it explains.
        lines_row = []
        result = rays[row]
        for i in range(result.heights.shape[0]):
            (ln,) = ax_f.plot(
                src_x + result.ranges[i],
                result.heights[i],
                color="white",
                lw=0.8,
                alpha=0.0,
                zorder=3.4,
                path_effects=[
                    patheffects.withStroke(linewidth=1.5, foreground="#40404060")
                ],
            )
            lines_row.append(ln)
        ray_lines.append(lines_row)
        ax_f.plot(
            [src_x],
            [src_h],
            marker="o",
            ms=5,
            color=COLOR_TERTIARY,
            markeredgecolor=FIELD_STROKE,
            markeredgewidth=0.8,
            zorder=4,
        )
        ax_f.text(
            src_x + 8.0,
            src_h + 4.0,
            T("source ($h$ = 2 m)"),
            ha="left",
            va="bottom",
            color=FIELD_INK,
            fontsize=7.5,
            path_effects=ray_outline,
            zorder=4,
        )
        ax_f.plot(
            [recv_x],
            [src_h],
            marker="o",
            ms=5,
            color=FIELD_STROKE,
            markeredgecolor=FIELD_INK,
            markeredgewidth=0.8,
            zorder=4,
        )
        ax_f.text(
            recv_x,
            src_h + 6.0,
            T("receiver 350 m"),
            ha="center",
            va="bottom",
            color=FIELD_INK,
            fontsize=7.5,
            path_effects=ray_outline,
            zorder=4,
        )
        if row == 0:
            # Halo rather than a filled box: the box was opaque and sat on
            # the hatched ground, wiping out the first 170 px of its
            # hatching so the band looked broken right after the spine.
            ax_f.text(
                20.0,
                -3.5,
                T("rigid ground"),
                ha="left",
                va="center",
                color=COLOR_FG,
                fontsize=6.5,
                path_effects=ray_outline,
                zorder=4,
            )
            ax_f.text(
                22.0,
                97.0,
                T(f"$f$ = {_REFR_F:.0f} Hz"),
                ha="left",
                va="top",
                color=FIELD_INK,
                fontsize=7.5,
                path_effects=outline,
                zorder=4,
            )
        else:
            # The ray-model shadow boundary of the library, where the
            # grazing ray leaves the ground for good.
            ax_f.axvline(
                src_x + x_shadow, color="#888888", ls="--", lw=0.9, alpha=0.8, zorder=3
            )
            ax_f.text(
                src_x + x_shadow + 6.0,
                84.0,
                T(f"shadow beyond ≈ {x_shadow:.0f} m (ray model)"),
                ha="left",
                va="top",
                color=FIELD_INK,
                fontsize=7,
                path_effects=outline,
                zorder=4,
            )
        v_txt = ax_f.text(
            424.0,
            97.0,
            "",
            ha="right",
            va="top",
            color="white",
            fontsize=8,
            zorder=4,
            bbox={
                "boxstyle": _ANIM_PILL_BOX,
                "facecolor": "black",
                "alpha": 0.45,
                "edgecolor": "none",
            },
        )
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
            t_txt = ax_f.text(
                20.0,
                97.0,
                "",
                ha="left",
                va="top",
                family="monospace",
                fontsize=9,
                color="white",
                zorder=4,
                bbox={
                    "boxstyle": _ANIM_PILL_BOX,
                    "facecolor": "black",
                    "alpha": 0.45,
                    "edgecolor": "none",
                },
            )
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
        t_txt.set_text(T(f"$t$ = {times[k]:5.2f} s"))
        return (*ims, *ims_e, *arts, *v_txts, t_txt)

    # gif_fps 4: the steady CW field moves everywhere in every frame, the
    # worst case for GIF palette coding; 8 fps left the fallback near 8 MB
    # where every other FDTD clip stays under 4 MB.
    _render_clip(
        fig,
        update,
        output_dir,
        "anim_fdtd_refraction",
        frames=int(p_all.shape[1]),
        fps=_REFR_FPS,
        gif_fps=4,
    )
