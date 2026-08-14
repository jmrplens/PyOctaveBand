#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The slit absorber: a critically coupled panel swallowing the wave."""

from functools import lru_cache
from typing import Any

import numpy as np

from ..media import (
    _ANIM_PILL_BOX,
    _FDTD_ANIM_FRAMES,
    _anim_figure,
    _render_clip,
    _translate_str,
)
from ..theme import CMAP_FIELD, COLOR_FG, COLOR_GRID, COLOR_PRIMARY
from ._core import _anim_speaker, _fit_text_x

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
    helm_txts: list[Any] = []
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
        ax.text(0.005, env_base + 0.004, "$|p|$ envelope", fontsize=7.5,
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
            T(f"slit $h$ = {h * 1e3:.2f} mm"),
            xy=(x_zoom0 + 0.006, y_slit0 + 0.5 * h),
            xytext=(x_zoom0 + 0.0035, bore - 0.0095),
            fontsize=7, color=COLOR_FG, ha="left", va="top",
            arrowprops={"arrowstyle": "-", "color": COLOR_FG, "lw": 0.7},
            zorder=6,
            bbox={"boxstyle": "round,pad=0.2",
                  "facecolor": fig.get_facecolor(), "alpha": 0.55,
                  "edgecolor": "none"})
        helm_txts.append(ax_z.text(
            x_mouth, 0.5 * (y_cav[0] + y_cav[1]),
            T("Helmholtz resonator"), ha="center", va="center",
            fontsize=6.5, color=COLOR_FG, zorder=6,
            bbox={"boxstyle": "round,pad=0.2",
                  "facecolor": fig.get_facecolor(), "alpha": 0.55,
                  "edgecolor": "none"}))
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
    # The resonator label is centred on the cell mouth, which sits close to
    # the right edge of the 45 mm zoom: the English string just fits, the
    # Spanish one crosses the dashed frame. Slide it back inside once the
    # figure is complete and the zoom axes have their final width.
    def fit_resonator_labels() -> None:
        for helm in helm_txts:
            _fit_text_x(fig, helm.axes, helm, x_zoom0, x_end,
                        margin=0.05 * (x_end - x_zoom0))

    # The captured field is already steady, so the verdict can come early.
    reveal = int(0.30 * len(times))

    def update(k: int) -> tuple[Any, ...]:
        for i, (t_all, z_all, alpha) in enumerate(
                ((t_c, z_c, alphas[0]), (t_w, z_w, alphas[1]))):
            ims[i].set_data(t_all[k])
            ims_zoom[i].set_data(z_all[k])
            a_txts[i].set_text(
                T(r"$\alpha$ = " f"{alpha:.2f} at {_SLIT_ABS_F0:.0f} Hz")
                if k >= reveal else "")
        t_txt.set_text(T(f"$t$ = {times[k] * 1e3:5.1f} ms"))
        return (*ims, *ims_zoom, *a_txts, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_slit_absorber",
                 frames=len(times), gif_fps=8, measure=fit_resonator_labels)


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
