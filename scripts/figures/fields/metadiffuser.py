#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The slit-resonator metadiffuser: a panel a twentieth of a QRD deep.

Both the far-field polar levels the prediction figure is checked against and
the clip that shows the panel scattering are runs of the same meshed
Table-1 geometry, so they share its pitch, period count and design
frequency.
"""

from functools import lru_cache
from typing import Any

import numpy as np

from ..materials import _METADIFFUSER_T1_ROWS
from ..media import _anim_figure, _render_clip, _translate_str
from ..theme import (
    CMAP_FIELD,
    COLOR_FG,
    COLOR_GRID,
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
            # No dimension bar on this one: the panel is 2 cm deep against
            # the QRD's 27, and a bar drawn to that scale is four pixels of
            # a stroke 1.6 wide -- a full stop in front of the text, not a
            # measurement. The number sits against the end of the slab it
            # measures instead, which at this thickness is the slab's own
            # thickness anyway.
            ax_t.text(x_r + 0.02, y1 - 0.012, "2 cm", ha="left",
                      va="center", fontsize=7, color=COLOR_FG)
        ims += [im_t, im_s]
        d_txts.append(d_txt)
    # Top-left margin, beside the centred suptitle: three columns of field
    # panels carry their x ticks and "x [m]" all the way into the
    # bottom-right corner, and a readout parked there merged with the third
    # column's tick row -- the stem of the "t" reading as part of 1.00, the
    # "ms" as part of 1.50.
    t_txt = fig.text(0.012, 0.985, "", ha="left", va="top",
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
