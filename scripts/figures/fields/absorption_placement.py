#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Where the absorption sits: the same total, two different decays.

The reverberation-prediction guide motivates Fitzroy and Arau-Puchades with
"when the absorption is concentrated on one axis ... a single mean alpha
misrepresents the field", closes with disproportionate rooms keeping "a
grazing sound field parallel to the hard surfaces that the absorber barely
touches", and advises quoting a *band* of predictions rather than one number
-- and its only absorption figure draws T against how *much* absorption
there is, never against where it sits. This clip runs one flat 8 x 2.5 m
section twice with the same **total statistical absorption**: spread over
all four edges, or concentrated on the floor/ceiling pair (the guide's own
carpeted-floor-plus-acoustic-ceiling case, whose z-pair carries alpha ~0.78
at mid frequency) between rigid ends. A lower axis races the two measured
energy decays through the statistical band both rooms share -- the Sabine
and Eyring lines, which use only the total absorption and therefore cannot
tell the rooms apart. The spread room decays inside the band; the
concentrated room's decay bends and its tail leaves the band entirely,
and the RMS panel shows why: what survives is a grazing field running
nearly parallel to the absorbing pair, horizontal stripes that the spread
room never develops.

The trap the design catalogue recorded, honoured here: a locally reacting
edge does not deliver its normal-incidence alpha to a room. The band is
therefore computed from the boundary's own *statistical* (cosine-weighted
oblique) absorption, the footer declares both values, and every
reverberation time printed on screen is measured from the decay the room
actually produced, never asserted from the nominal alpha.
"""

from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

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
    FIELD_STROKE,
)

_AP_C0, _AP_RHO = 343.0, 1.2
_AP_LX, _AP_LY = 8.0, 2.5  # a flat, corridor-like section [m]
_AP_DX = 0.02
_AP_F0 = 250.0  # burst centre frequency [Hz]
_AP_CYCLES = 4  # Hann-windowed burst length
_AP_SRC = (1.73, 0.97)  # off-centre, off-symmetry [m]
#: Normal-incidence absorption of the concentrated pair: the guide's own
#: carpeted-floor/acoustic-ceiling case (its z-pair carries 0.78 at 1 kHz,
#: which is this lining's statistical value on this boundary).
_AP_ALPHA_N_CONC = 0.7
# Timeline. Mesh: the 4-cycle Hann burst is essentially contained below
# ~375 Hz (lambda / 8 = 114 mm) and the smallest geometric dimension is the
# 2.5 m height (2.5 / 4 = 0.63 m), so the rule allows 114 mm; dx = 20 mm
# sits 5.7 times finer because the field map has to be drawn, not just
# resolved. Flight: source (1.73, 0.97) -> the far corner (8, 2.5), the
# deepest reflector, -> the origin corner, the farthest visible point:
# (6.45 + 8.38) m / 343 x 1.2 = 51.9 ms. That floor is only a floor: the
# payload is the decay, and the window is 185 ms (~0.9 of the shared
# Sabine time), by which time the spread room has fallen past -55 dB and
# the concentrated room sits more than 12 dB above it. Sampling: one frame
# every 3 solver steps (dt = 24.74 us) is 74.2 us, i.e. 53.9 frames per
# period of the 250 Hz centre (>= 48). 2 493 frames at 160 fps = 15.6 s +
# the shared 2 s hold, the pacing band of the committed pillar-hall and
# ducting clips.
_AP_EVERY = 3
_AP_FRAMES = 2493
_AP_FPS = 160.0
_AP_HOLD = round(2.0 * _AP_FPS)
#: dB floor of the decay axis (deep enough that the spread room's curve
#: stays on screen to the close; the gap arrow needs both endpoints).
_AP_FLOOR = -64.0
#: Span of the per-frame re-referenced RMS panels [dB].
_AP_RMS_SPAN = 25.0


def _ap_alpha_stat(zeta: float) -> float:
    """2D statistical absorption of a real locally reacting boundary.

    ``alpha(theta) = 4 zeta cos(theta) / (zeta cos(theta) + 1)^2`` averaged
    over the cosine-weighted incidence of a 2D diffuse field.
    """
    th = np.linspace(0.0, np.pi / 2, 4001)
    zc = zeta * np.cos(th)
    alpha = 4.0 * zc / (zc + 1.0) ** 2
    return float(np.trapezoid(alpha * np.cos(th), th) / np.trapezoid(np.cos(th), th))


def _ap_linings() -> tuple[float, float, float, float]:
    """The two linings: (zeta_spread, zeta_conc, astat_spread, astat_conc).

    The concentrated pair is fixed by its normal-incidence alpha (hard
    branch of |R| = sqrt(1 - alpha), Z = rho c (1+R)/(1-R)); the spread
    room's lining is solved so that the *total* statistical absorption
    matches: astat_spread x perimeter = astat_conc x 2 Lx.
    """
    r = float(np.sqrt(1.0 - _AP_ALPHA_N_CONC))
    zeta_b = (1.0 + r) / (1.0 - r)
    a_b = _ap_alpha_stat(zeta_b)
    target = a_b * 2.0 * _AP_LX / (2.0 * (_AP_LX + _AP_LY))
    lo, hi = 1.0, 400.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _ap_alpha_stat(mid) > target:
            lo = mid
        else:
            hi = mid
    zeta_a = 0.5 * (lo + hi)
    return zeta_a, zeta_b, _ap_alpha_stat(zeta_a), a_b


def _ap_fit_t60(
    t: NDArray[np.float64], level: NDArray[np.float64], lo: float, hi: float
) -> float:
    """T60 extrapolated from a straight fit of the decay between two levels,
    the T20/T30 practice of the measurement guides.
    """
    m = (level <= lo) & (level >= hi) & (t > 0.015)
    slope = float(np.polyfit(t[m], level[m], 1)[0])
    return -60.0 / slope


@lru_cache(maxsize=1)
def _absorption_placement_fields() -> tuple[Any, ...]:
    """The two decaying rooms, cached for the four variants.

    Returns the instantaneous frame stacks (float16), the running-RMS
    stacks re-referenced per frame to their own maximum (float16, dB, 0 to
    -_AP_RMS_SPAN), the frame times, the smoothed total-energy levels on
    the frame times, the shared Sabine and Eyring times, the fitted
    reverberation times (spread; concentrated early and late) and the two
    linings' statistical absorptions.
    """
    import fdtd2d

    zeta_a, zeta_b, a_a, a_b = _ap_linings()
    area, per = _AP_LX * _AP_LY, 2.0 * (_AP_LX + _AP_LY)
    # The band both rooms share: Sabine and Eyring on the one number they
    # can see, the total statistical absorption (2D mean free path pi A/P).
    mean_alpha = a_b * 2.0 * _AP_LX / per
    t_sab = 13.8155 * np.pi * area / (_AP_C0 * mean_alpha * per)
    t_eyr = 13.8155 * np.pi * area / (_AP_C0 * -np.log(1.0 - mean_alpha) * per)

    fs = 40.0 * _AP_F0
    tb = np.arange(round(_AP_CYCLES / _AP_F0 * fs)) / fs
    burst = np.sin(2.0 * np.pi * _AP_F0 * tb) * np.hanning(tb.size)
    z_a = zeta_a * _AP_RHO * _AP_C0
    z_b = zeta_b * _AP_RHO * _AP_C0
    edges = (
        {"left": z_a, "right": z_a, "top": z_a, "bottom": z_a},
        {"top": z_b, "bottom": z_b},
    )
    nx, ny = round(_AP_LX / _AP_DX), round(_AP_LY / _AP_DX)

    p_all: list[Any] = []
    db_all: list[Any] = []
    e_all: list[Any] = []
    times = np.zeros(0)
    for edge_imp in edges:
        sim = fdtd2d.FDTD2D(_AP_C0, _AP_DX, shape=(ny, nx), edge_impedance=edge_imp)
        sim.add_source(
            fdtd2d.SignalSource(
                ix=round(_AP_SRC[0] / _AP_DX),
                iy=round(_AP_SRC[1] / _AP_DX),
                samples=burst,
                sample_rate=fs,
            )
        )
        beta = float(np.exp(-sim.dt * _AP_F0 / 2.0))
        ms = np.zeros_like(sim.p)
        ps: list[Any] = []
        rs: list[Any] = []
        es: list[float] = []
        ts: list[float] = []
        for i in range(_AP_EVERY * _AP_FRAMES):
            sim.step()
            ms = beta * ms + (1.0 - beta) * sim.p**2
            if (i + 1) % _AP_EVERY == 0:
                ps.append(sim.p.astype(np.float16))
                # Re-referenced per frame: the panel shows where the
                # energy lives, the decay axis says how much is left.
                with np.errstate(divide="ignore", invalid="ignore"):
                    db = 10.0 * np.log10(ms / max(float(ms.max()), 1e-300))
                rs.append(np.clip(db, -_AP_RMS_SPAN, 0.0).astype(np.float16))
                pe = float(np.sum(sim.p**2)) / (2.0 * _AP_RHO * _AP_C0**2)
                ke = (
                    0.5
                    * _AP_RHO
                    * (float(np.sum(sim.vx**2)) + float(np.sum(sim.vy**2)))
                )
                es.append((pe + ke) * _AP_DX**2)
                ts.append(sim.time)
        p_all.append(np.stack(ps))
        db_all.append(np.stack(rs))
        e_all.append(np.asarray(es))
        times = np.asarray(ts)

    e_ref = max(float(e.max()) for e in e_all)
    k_sm = max(1, round(2.0 / _AP_F0 / (times[1] - times[0])))
    kernel = np.ones(k_sm) / k_sm
    levels = []
    for e in e_all:
        sm = np.convolve(e / e_ref, kernel, mode="same")
        levels.append(10.0 * np.log10(np.maximum(sm, 1e-12)))

    t_meas = _ap_fit_t60(times, levels[0], -5.0, -35.0)
    t_early = _ap_fit_t60(times, levels[1], -5.0, -20.0)
    t_late = _ap_fit_t60(times, levels[1], -25.0, -40.0)
    return (
        p_all[0],
        p_all[1],
        db_all[0],
        db_all[1],
        times,
        levels[0],
        levels[1],
        float(t_sab),
        float(t_eyr),
        float(t_meas),
        float(t_early),
        float(t_late),
        a_a,
        a_b,
    )


def animate_fdtd_absorption_placement(output_dir: str) -> None:
    """Absorption placement (2D FDTD): one flat 8 x 2.5 m section run twice
    with the same total statistical absorption -- spread over all four
    edges, or concentrated on the floor/ceiling pair between rigid ends. A
    lower axis races the two measured energy decays through the shared
    Sabine-Eyring band, which uses only the total absorption and so
    cannot tell the rooms apart: the spread room decays inside the band,
    the concentrated room's decay bends and its tail leaves it, and the
    per-frame RMS panel shows the surviving field running nearly parallel
    to the absorbing pair, in horizontal stripes.
    """
    from matplotlib import patheffects
    from matplotlib.patches import Rectangle

    T = _translate_str
    (
        p_a,
        p_b,
        db_a,
        db_b,
        times,
        lev_a,
        lev_b,
        t_sab,
        t_eyr,
        t_meas,
        t_early,
        t_late,
        a_a,
        a_b,
    ) = _absorption_placement_fields()
    outline = [patheffects.withStroke(linewidth=1.8, foreground="black")]
    # Colour scale of the instantaneous row: referenced to the field a
    # third of the way into the decay (60 ms), so the blast saturates
    # boldly, the row stays alive well past the reveal, and what finally
    # fades out is the decay itself.
    k0 = int(np.searchsorted(times, 0.060))
    vmax = float(np.quantile(np.abs(p_a[k0 : k0 + 40].astype(np.float32)), 0.999))

    fig = _anim_figure()
    fig.suptitle(T("Where the absorption sits: same total, two decays (2D FDTD)"))
    gs = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 1.15), hspace=0.12)
    titles = [
        T("Absorption spread over all four edges"),
        T("The same total, floor and ceiling only"),
    ]
    ims: list[Any] = []
    rms_axes: list[Any] = []
    strip = 0.1
    for col, title in enumerate(titles):
        treated_all = col == 0
        ax_p = fig.add_subplot(gs[0, col])
        ax_r = fig.add_subplot(gs[1, col])
        rms_axes.append(ax_r)
        for row, ax in enumerate((ax_p, ax_r)):
            ax.grid(False)
            kwargs: dict[str, Any] = (
                {"cmap": CMAP_FIELD, "vmin": -vmax, "vmax": vmax}
                if row == 0
                else {"cmap": "magma", "vmin": -_AP_RMS_SPAN, "vmax": 0.0}
            )
            ims.append(
                ax.imshow(
                    np.zeros((2, 2)),
                    origin="lower",
                    extent=(0.0, _AP_LX, 0.0, _AP_LY),
                    aspect="equal",
                    interpolation="bilinear",
                    zorder=2,
                    **kwargs,
                )
            )
            # The treated edges as hatched absorber strips, the rigid ends
            # as solid grey: the geometry is the whole experiment.
            for side in ("top", "bottom") + (("left", "right") if treated_all else ()):
                x0, y0, w, h = {
                    "top": (0.0, _AP_LY, _AP_LX, strip),
                    "bottom": (0.0, -strip, _AP_LX, strip),
                    "left": (-strip, 0.0, strip, _AP_LY),
                    "right": (_AP_LX, 0.0, strip, _AP_LY),
                }[side]
                ax.add_patch(
                    Rectangle(
                        (x0, y0),
                        w,
                        h,
                        facecolor="none",
                        hatch="////",
                        edgecolor=COLOR_MUTED,
                        lw=0.0,
                        zorder=3,
                    )
                )
            if not treated_all:
                for x0 in (-strip, _AP_LX):
                    ax.add_patch(
                        Rectangle(
                            (x0, 0.0),
                            strip,
                            _AP_LY,
                            facecolor="#9a9a9a",
                            edgecolor=COLOR_FG,
                            lw=0.5,
                            zorder=3,
                        )
                    )
            ax.add_patch(
                Rectangle(
                    (0.0, 0.0),
                    _AP_LX,
                    _AP_LY,
                    facecolor="none",
                    edgecolor=COLOR_FG,
                    lw=0.8,
                    zorder=4,
                )
            )
            ax.plot(
                [_AP_SRC[0]],
                [_AP_SRC[1]],
                marker="o",
                ms=4,
                color=FIELD_STROKE,
                markeredgecolor=COLOR_FG,
                markeredgewidth=0.7,
                zorder=5,
            )
            ax.set_xlim(-strip - 0.05, _AP_LX + strip + 0.05)
            ax.set_ylim(-strip - 0.06, _AP_LY + strip + 0.06)
            ax.set_xticks([])
            ax.set_yticks([])
        ax_p.set_title(title, fontsize=9.5)
        if col == 0:
            ax_p.set_ylabel(T("$p(x, y)$"), fontsize=8.5)
            ax_r.set_ylabel(T("RMS level [dB]"), fontsize=8.5)
            ax_p.text(
                0.12,
                _AP_LY - 0.16,
                "8 × 2.5 m",
                fontsize=6.8,
                ha="left",
                va="top",
                color="white",
                zorder=6,
                path_effects=outline,
            )
            ax_r.text(
                0.12,
                0.16,
                T("each frame on its own scale, 25 dB"),
                fontsize=6.8,
                ha="left",
                va="bottom",
                color="white",
                zorder=6,
                path_effects=outline,
            )
    graze_txt = rms_axes[1].text(
        _AP_LX - 0.16,
        _AP_LY - 0.22,
        "",
        ha="right",
        va="top",
        fontsize=8,
        color="white",
        zorder=6,
        path_effects=outline,
    )

    ax_d = fig.add_subplot(gs[2, :])
    t_end = float(times[-1]) * 1e3
    ax_d.set_xlim(0.0, t_end)
    ax_d.set_ylim(_AP_FLOOR - 2.0, 3.5)
    ax_d.set_xlabel(T("Time [ms]"), fontsize=8.5, labelpad=1.5)
    ax_d.set_ylabel(T("Total energy [dB]"), fontsize=8.5)
    ax_d.tick_params(labelsize=7)
    ax_d.grid(True, color=COLOR_GRID, lw=0.6)
    # The statistical band both rooms share, hung from the decay's peak:
    # Sabine above, Eyring below, and neither can tell the rooms apart.
    k_pk = int(np.argmax(lev_a))
    t_pk = float(times[k_pk]) * 1e3
    span = _AP_FLOOR + 2.0
    x_sab = t_pk + t_sab * 1e3 * -span / 60.0
    x_eyr = t_pk + t_eyr * 1e3 * -span / 60.0
    ax_d.fill(
        [t_pk, x_sab, x_eyr],
        [0.0, span, span],
        color=COLOR_MUTED,
        alpha=0.14,
        lw=0.0,
        zorder=1,
    )
    for x_line, t_line, name in ((x_sab, t_sab, "Sabine"), (x_eyr, t_eyr, "Eyring")):
        ax_d.plot(
            [t_pk, x_line], [0.0, span], color=COLOR_MUTED, lw=1.3, ls="--", zorder=3
        )
        frac = 0.52 if name == "Sabine" else 0.30
        ax_d.annotate(
            T(f"{name}, $T$ = {t_line * 1e3:.0f} ms"),
            (t_pk + t_line * 1e3 * frac, -60.0 * frac),
            xytext=(6.0 if name == "Sabine" else -6.0, 0.0),
            textcoords="offset points",
            fontsize=7.6,
            color=COLOR_MUTED,
            ha="left" if name == "Sabine" else "right",
            va="bottom" if name == "Sabine" else "top",
        )
    line_a = ax_d.plot(
        [], [], color=COLOR_PRIMARY, lw=1.8, label=T("all four edges"), zorder=5
    )[0]
    line_b = ax_d.plot(
        [], [], color=COLOR_SECONDARY, lw=1.8, label=T("floor + ceiling"), zorder=4
    )[0]
    # One row, hugging the top-right: the two-row box used to reach down to
    # ~-20 dB, where the concentrated room's verdict pill (whose Spanish
    # text runs wider) slid underneath it and covered the second entry's
    # swatch. A single row bottoms out near -11 dB, clear of the pill's top
    # (~-16 dB) in both languages, and the curves stay below -30 dB in the
    # strip it occupies.
    ax_d.legend(fontsize=7.2, loc="upper right", framealpha=0.85, ncols=2)
    pill_a = ax_d.text(
        0.055 * t_end,
        _AP_FLOOR + 4.0,
        "",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="white",
        zorder=6,
        bbox={
            "boxstyle": _ANIM_PILL_BOX,
            "facecolor": COLOR_PRIMARY,
            "alpha": 0.8,
            "edgecolor": "none",
        },
    )
    pill_b = ax_d.text(
        0.56 * t_end,
        -24.0,
        "",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="white",
        zorder=6,
        bbox={
            "boxstyle": _ANIM_PILL_BOX,
            "facecolor": COLOR_SECONDARY,
            "alpha": 0.8,
            "edgecolor": "none",
        },
    )
    t_txt = fig.text(
        0.988,
        0.955,
        "",
        ha="right",
        va="top",
        family="monospace",
        fontsize=9.5,
        color=COLOR_FG,
    )
    fig.get_layout_engine().set(rect=(0.0, 0.03, 1.0, 0.965))  # type: ignore[union-attr, call-arg]  # _anim_figure: constrained engine, never None
    fig.text(
        0.5,
        0.005,
        T(
            f"Locally reacting resistive edges; both rooms hold "
            f"{a_b * 2.0 * _AP_LX:.1f} m of statistical absorption "
            r"($\alpha_{\mathrm{st}}$ = "
            f"{a_a:.2f} on 21 m vs {a_b:.2f} on 16 m); "
            f"250 Hz burst."
        ),
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=COLOR_FG,
        alpha=0.85,
    )

    # The placement penalty, drawn as the closing gap between the twins.
    # The suptitle and footer carry "same total"; the number is enough
    # here, and short enough to clear the Eyring line and both curves.
    k_gap = int(np.searchsorted(times, 0.955 * float(times[-1])))
    gap = float(lev_b[k_gap] - lev_a[k_gap])
    x_gap = float(times[k_gap]) * 1e3
    arrow = ax_d.annotate(
        "",
        (x_gap, float(lev_b[k_gap])),
        (x_gap, float(lev_a[k_gap])),
        arrowprops={"arrowstyle": "<->", "color": COLOR_FG, "lw": 1.2},
        visible=False,
    )
    gap_txt = ax_d.text(
        x_gap - 2.5,
        0.5 * float(lev_a[k_gap] + lev_b[k_gap]),
        "",
        ha="right",
        va="center",
        fontsize=8.5,
        color=COLOR_FG,
    )

    reveal = int(0.7 * times.size)
    verdict_a = T(f"measured $T$ = {t_meas * 1e3:.0f} ms: inside the band")
    verdict_b = T(
        f"early {t_early * 1e3:.0f} ms, tail {t_late * 1e3:.0f} ms: no single $T$"
    )
    gap_note = f"{gap:.0f} dB"
    graze = T("what survives runs parallel to the absorber")

    def update(k: int) -> tuple[Any, ...]:
        j = min(k, times.size - 1)
        # ims order: col 0 rows p/RMS, then col 1 rows p/RMS.
        ims[0].set_data(p_a[j].astype(np.float32))
        ims[1].set_data(db_a[j].astype(np.float32))
        ims[2].set_data(p_b[j].astype(np.float32))
        ims[3].set_data(db_b[j].astype(np.float32))
        keep = slice(0, j + 1)
        line_a.set_data(times[keep] * 1e3, lev_a[keep])
        line_b.set_data(times[keep] * 1e3, lev_b[keep])
        pill_a.set_text(verdict_a if k >= reveal else "")
        pill_b.set_text(verdict_b if k >= reveal else "")
        graze_txt.set_text(graze if k >= reveal else "")
        # The gap arrow only exists once both its endpoints do.
        arrow.set_visible(j >= k_gap)
        gap_txt.set_text(gap_note if j >= k_gap else "")
        t_txt.set_text(T(f"$t$ = {times[j] * 1e3:5.1f} ms"))
        return (*ims, line_a, line_b, pill_a, pill_b, graze_txt, arrow, gap_txt, t_txt)

    _render_clip(
        fig,
        update,
        output_dir,
        "anim_fdtd_absorption_placement",
        frames=times.size + _AP_HOLD,
        fps=_AP_FPS,
        gif_fps=6,
    )
