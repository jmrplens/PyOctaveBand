#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The noise barrier: diffraction into the shadow at two frequencies."""

from functools import lru_cache
from typing import Any

import numpy as np

from ..media import _anim_figure, _render_clip, _translate_str
from ..theme import (
    CMAP_FIELD,
    COLOR_FG,
    COLOR_GRID,
    COLOR_TERTIARY,
    FIELD_INK,
    FIELD_STROKE,
)
from ._core import (
    _fdtd_cw_capture,
    _fit_text_x,
    _rms_to_db,
    _settle,
    _text_width_x,
)

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
def _barrier_fields(n_frames: int = _BARRIER_FRAMES) -> tuple[Any, Any, Any, Any]:
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
    ny, nx = 350, 600  # 7 m x 12 m
    rho = np.full((ny, nx), 1.2)
    bx = round(5.5 / dx)  # thin barrier: 3 cells = 6 cm
    rho[: round(2.5 / dx), bx : bx + 3] = 1.2e6
    every = _BARRIER_EVERY
    # Receiver patch: 0.6 m x 0.6 m around the shadow-zone receiver,
    # energy-averaged so residual interference fringes average out.
    rx, ry = _BARRIER_RECEIVER
    patch = (
        slice(int((ry - 0.3) / dx), int((ry + 0.3) / dx) + 1),
        slice(int((rx - 0.3) / dx), int((rx + 0.3) / dx) + 1),
    )
    p_all, db_all, ils = [], [], []
    times = np.zeros(0)
    for f in _BARRIER_FREQS:
        # Row 0 is the ground (displayed at the bottom via origin="lower"),
        # so the absorbing sides are left/right and the *high* rows ("bottom"
        # in the imshow-origin naming of fdtd2d); the ground stays rigid.
        rms_patch = []
        for rho_map in (rho, None):
            sim = fdtd2d.FDTD2D(
                c0,
                dx,
                shape=(ny, nx),
                rho=1.2 if rho_map is None else rho_map,
                sponge_width=40,
                sponge_sides=("left", "right", "bottom"),
            )
            sim.add_source(fdtd2d.CWSource(ix=100, iy=25, frequency=f, ramp_cycles=2.0))
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
    outline = [patheffects.withStroke(linewidth=2.0, foreground=FIELD_STROKE)]
    p_all, db_all, times, ils = _barrier_fields()
    half = p_all.shape[1] // 2
    lam = tuple(343.0 / f for f in _BARRIER_FREQS)
    rx, ry = _BARRIER_RECEIVER

    fig = _anim_figure()
    sup = fig.suptitle(
        T("Barrier diffraction into the shadow zone (2D FDTD)"),
    )
    gs = fig.add_gridspec(2, 2)
    titles = [
        T(
            rf"Low frequency: {_BARRIER_FREQS[0]:.0f} Hz "
            rf"($\lambda$ ≈ {lam[0]:.1f} m)"
        ),
        T(
            rf"High frequency: {_BARRIER_FREQS[1]:.0f} Hz "
            rf"($\lambda$ ≈ {lam[1]:.2f} m)"
        ),
    ]
    verdicts = [T("diffraction fills the shadow"), T("deep, clean shadow")]
    ims: list[Any] = []
    il_txts: list[Any] = []
    for col in range(2):
        vmax = float(np.quantile(np.abs(p_all[col][half:]), 0.995))
        ax_p = fig.add_subplot(gs[0, col])
        ax_r = fig.add_subplot(gs[1, col])
        im_p = ax_p.imshow(
            p_all[col][0],
            origin="lower",
            extent=(0.0, 12.0, 0.0, 7.0),
            cmap=CMAP_FIELD,
            vmin=-vmax,
            vmax=vmax,
            interpolation="bilinear",
        )
        im_r = ax_r.imshow(
            db_all[col][0],
            origin="lower",
            extent=(0.0, 12.0, 0.0, 7.0),
            cmap="magma",
            vmin=-40.0,
            vmax=0.0,
            interpolation="bilinear",
        )
        ax_p.set_title(titles[col], fontsize=10)
        for ax in (ax_p, ax_r):
            ax.grid(False)
            ax.set_ylim(-0.5, 7.0)
            ax.add_patch(
                Rectangle(
                    (0.0, -0.5),
                    12.0,
                    0.5,
                    facecolor=COLOR_GRID,
                    edgecolor=COLOR_FG,
                    lw=0.8,
                    hatch="///",
                )
            )
            # Theme-independent bar: mid-gray with a white edge stays
            # visible on the near-white RdBu row and the black magma row.
            ax.add_patch(
                Rectangle(
                    (5.44, 0.0),
                    0.18,
                    2.5,
                    facecolor="#707070",
                    edgecolor="white",
                    lw=0.6,
                )
            )
            ax.tick_params(labelsize=7)
        ax_p.tick_params(labelbottom=False)
        ax_r.set_xlabel("$x$ [m]", fontsize=8)
        ax_p.plot(
            [2.0],
            [0.5],
            marker="o",
            ms=5,
            color=COLOR_TERTIARY,
            markeredgecolor=FIELD_STROKE,
            markeredgewidth=0.8,
        )
        ax_p.text(
            2.25,
            0.55,
            T("source"),
            ha="left",
            va="center",
            color=FIELD_INK,
            fontsize=7.5,
            path_effects=outline,
        )
        ax_p.text(
            5.53,
            2.7,
            T("barrier"),
            ha="center",
            va="bottom",
            color=FIELD_INK,
            fontsize=7.5,
            path_effects=outline,
        )
        ax_p.text(
            9.2,
            1.1,
            T("shadow zone"),
            ha="center",
            va="center",
            color=FIELD_INK,
            fontsize=7.5,
            path_effects=outline,
        )
        ax_r.text(
            11.7, 6.4, verdicts[col], ha="right", va="top", color="white", fontsize=8
        )
        ax_r.plot(
            [rx],
            [ry],
            marker="o",
            ms=5,
            color="white",
            markeredgecolor="black",
            markeredgewidth=0.8,
        )
        # Written out here, not in update(): the string never changes once
        # the number is revealed, and the clamp below has to measure the
        # label that will actually be drawn.
        il_txts.append(
            ax_r.text(
                rx,
                ry + 0.45,
                T(f"insertion loss {ils[col]:.0f} dB"),
                ha="center",
                va="bottom",
                color="white",
                fontsize=7.5,
            )
        )
        if col == 0:
            ax_p.set_ylabel(T("instantaneous $p(x, y)$"), fontsize=9)
            ax_r.set_ylabel(T("RMS level [dB re panel max]"), fontsize=8)
            # 0.45 m, not flush with the corner: anchored at 0.25 m the
            # rounded box behind the label started about a pixel off the
            # left spine and read as touching it. The extra 0.2 m is
            # daylight between the box and the spine, not data.
            ax_p.text(
                0.45,
                -0.27,
                T("rigid ground"),
                ha="left",
                va="center",
                color=COLOR_FG,
                fontsize=6.5,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": fig.get_facecolor(),
                    "edgecolor": "none",
                },
            )
        else:
            ax_p.tick_params(labelleft=False)
            ax_r.tick_params(labelleft=False)
            ax_r.text(
                0.3,
                5.45,
                T("each panel on its own dB scale"),
                color="white",
                fontsize=7,
                ha="left",
                va="top",
            )
        ims += [im_p, im_r]
    # Top-left margin: the field panels run their x-axis (ticks + "x [m]")
    # to the very bottom-right corner, so a bottom readout collides with the
    # tick labels; the top-left stays clear of the centred column titles.
    t_txt = fig.text(
        0.012,
        0.985,
        "",
        ha="left",
        va="top",
        family="monospace",
        fontsize=10,
        color=COLOR_FG,
    )
    # The insertion-loss labels are centred on the receiver, which sits
    # 9 m along a 12 m panel with the screen at 5.5 m: the room they have
    # is the shadow zone, and the English string fits in it while the
    # Spanish one runs past the right spine. Measure the label that will be
    # drawn, scale it to the room if it is over, then slide it back inside
    # the panel -- so it neither leaves the axes nor climbs onto the screen.
    shadow = (5.62 + 0.15, il_txts[0].axes.get_xlim()[1] - 0.35)

    # One size for both panels -- they are twins, and two sizes would read
    # as two kinds of annotation -- stepped down until the longer string
    # fits. Stepped rather than scaled by the ratio: the rasteriser rounds
    # the glyph size to whole pixels, so a 2 % reduction can come back the
    # same width it went in.
    def fit_annotations() -> None:
        while (
            max(_text_width_x(fig, il.axes, il) for il in il_txts)
            > shadow[1] - shadow[0]
            and il_txts[0].get_fontsize() > 6.0
        ):
            for il in il_txts:
                il.set_fontsize(il.get_fontsize() - 0.25)
        for il in il_txts:
            _fit_text_x(fig, il.axes, il, *shadow)
            il.set_text("")
        # The title is centred on the canvas and the clock owns the
        # top-left corner, so the room between them is what the title
        # leaves over: the Spanish title runs long enough that its first
        # letter once landed 37 px from the clock and the two read as one
        # run-on line at web size. The clock string is fixed-width (a
        # monospace 4.1f number in an unchanging frame), so draw it,
        # measure both, and step the title down -- same whole-pixel
        # stepping as the pills above -- until 0.6 in of daylight
        # separates them.
        t_txt.set_text(T(f"$t$ = {times[-1] * 1000.0:4.1f} ms"))
        _settle(fig)
        while (
            sup.get_window_extent().x0 - t_txt.get_window_extent().x1 < 0.6 * fig.dpi
            and sup.get_fontsize() > 9.0
        ):
            sup.set_fontsize(sup.get_fontsize() - 0.25)
        t_txt.set_text("")

    def update(k: int) -> tuple[Any, ...]:
        for col in range(2):
            ims[2 * col].set_data(p_all[col][k])
            ims[2 * col + 1].set_data(db_all[col][k])
            # The measured insertion loss appears once the field has
            # actually reached and settled at the receiver, so the number
            # never precedes its cause.
            il_txts[col].set_text(
                T(f"insertion loss {ils[col]:.0f} dB") if times[k] >= 0.032 else ""
            )
        t_txt.set_text(T(f"$t$ = {times[k] * 1000.0:4.1f} ms"))
        return (*ims, *il_txts, t_txt)

    _render_clip(
        fig,
        update,
        output_dir,
        "anim_fdtd_barrier",
        frames=int(p_all.shape[1]),
        fps=_BARRIER_FPS,
        gif_fps=5,
        measure=fit_annotations,
    )
