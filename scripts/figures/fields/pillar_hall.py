#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The README banner: a wavefront diffracting through a hall of columns.

The poster-time override lives here because the only clip that overrides it
is this one, and the override is computed from this clip's own mesh, stride
and trimmed warm-up.
"""

import os
from functools import lru_cache
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..media import _ANIM_DPI, _ANIM_HOLD, _render_clip, _translate_str
from ..theme import _FILENAME_SUFFIX, CMAP_FIELD, COLOR_FG, COLOR_GRID


def _poster_ss_for(webm: str) -> float | None:
    """Per-clip poster-time override; None keeps the end-of-hold default.

    The end-of-hold default is right for a clip that settles into a steady
    state, and wrong for one whose subject is a wave passing through: by the
    last frame the wave has gone and the poster is an empty field. The clips
    listed here are grabbed mid-flight instead.

    * the pillar-hall banner at simulation time 6 ms, the front threading the
      middle of the hall;
    * Lamb's problem at 95 us, where the Rayleigh train is still 22 mm inside
      the right edge (it leaves the 0.30 m view at 103 us) and both body
      fronts are drawn against their analytic arcs.
    """
    name = os.path.basename(webm)
    if _PILLAR_STEM in name:
        import fdtd2d
        dt = fdtd2d.FDTD2D(343.0, _PILLAR_DX, shape=(3, 8)).dt
        dt_frame = _PILLAR_EVERY * dt
        return float((6.0e-3 / dt_frame - _PILLAR_WARM) / _PILLAR_FPS)
    if "anim_elastic_halfspace_waves" in name:
        from .halfspace import _HS_CP, _HS_DX, _HS_FPS
        dt = 0.6 * _HS_DX / (_HS_CP * float(np.sqrt(2.0)))
        return float((95.0e-6 / (5.0 * dt)) / _HS_FPS)
    return None


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
