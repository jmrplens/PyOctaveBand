#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The clip vocabulary, and the schematic clips drawn with it.

A schematic clip is a diagram that moves: a microphone, a loudspeaker, a fan
of wavefronts, a needle gauge, a signal-flow box that lights up as the value
reaches it. The drawing primitives are shared -- every clip that shows a
measurement chain draws the same microphone -- so they live here with the
clips that use them, ahead of the wave-field clips of :mod:`figures.fields`,
which borrow the same vocabulary to annotate a simulated field.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .i18n import _LANG
from .media import (
    _ANIM_FPS,
    _ANIM_FRAMES,
    _ANIM_HOLD,
    _anim_figure,
    _render_clip,
    _translate_str,
)
from .theme import (
    _FILENAME_SUFFIX,
    COLOR_FG,
    COLOR_GRID,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
)


def _grid_axes(ax: Any) -> None:
    """Apply the standard documentation grid to a data axes."""
    ax.grid(True, color=COLOR_GRID, linestyle="--", alpha=0.5)


def _schematic_axes(ax: Any, xlim: tuple[float, float],
                    ylim: tuple[float, float], *, equal: bool = False) -> None:
    """Turn *ax* into a bare drawing canvas (no ticks, frame or grid)."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if equal:
        ax.set_aspect("equal")
    ax.grid(False)
    ax.axis("off")


# --- shared schematic vocabulary (mics, gauges, flow boxes, arrows) --------


def _draw_mic(ax: Any, x: float, y: float, *, direction: int = 1,
              size: float = 1.0, label: str = "", angle: float = 0.0) -> None:
    """A measurement-microphone symbol: rounded body plus capsule head.

    ``direction = +1`` points the capsule toward +x, ``-1`` toward -x. Draw
    on an equal-aspect schematic axes so the capsule stays round. ``angle``
    rotates the whole symbol (degrees, counterclockwise) around ``(x, y)``
    -- e.g. ``-90`` points a ``direction = +1`` capsule straight down; the
    label stays horizontal and rides on the rotated anchor.
    """
    from matplotlib import transforms
    from matplotlib.patches import Circle, FancyBboxPatch

    tr = ax.transData
    if angle:
        tr = transforms.Affine2D().rotate_deg_around(x, y, angle) + tr
    body_w, body_h = 1.1 * size, 0.42 * size
    head_r = 0.26 * size
    body = FancyBboxPatch(
        (x - body_w / 2, y - body_h / 2), body_w, body_h,
        boxstyle="round,pad=0.03", facecolor=COLOR_GRID,
        edgecolor=COLOR_FG, lw=1.2)
    body.set_transform(tr)
    ax.add_patch(body)
    hx = x + direction * (body_w / 2 + head_r * 0.85)
    head = Circle((hx, y), head_r, facecolor="none", edgecolor=COLOR_FG,
                  lw=1.2)
    head.set_transform(tr)
    ax.add_patch(head)
    for frac in (-0.45, 0.0, 0.45):
        half = head_r * float(np.sqrt(1.0 - frac * frac)) * 0.9
        (grille,) = ax.plot([hx - half, hx + half], [y + frac * head_r] * 2,
                            color=COLOR_FG, lw=0.6)
        grille.set_transform(tr)
    if label:
        rad = np.deg2rad(angle)
        dx, dy = 0.0, body_h * 1.7
        lx = x + dx * np.cos(rad) - dy * np.sin(rad)
        ly = y + dx * np.sin(rad) + dy * np.cos(rad)
        ax.text(lx, ly, label, ha="center", va="bottom",
                color=COLOR_FG, fontsize=11)


def _draw_speaker(ax: Any, x: float, y: float, *, size: float = 1.0,
                  direction: int = 1) -> None:
    """A loudspeaker symbol: cabinet plus cone, radiating toward
    ``direction`` (+1 = +x, -1 = -x). ``(x, y)`` is the cone mouth centre."""
    from matplotlib.patches import Polygon, Rectangle

    d = float(direction)
    cone_l, cone_h = 0.42 * size, 0.55 * size
    box_l, box_h = 0.5 * size, 0.36 * size
    xb = x - d * cone_l
    ax.add_patch(Polygon(
        [(x, y - cone_h), (x, y + cone_h), (xb, y + box_h), (xb, y - box_h)],
        closed=True, facecolor=COLOR_GRID, edgecolor=COLOR_FG, lw=1.2))
    ax.add_patch(Rectangle((min(xb, xb - d * box_l), y - box_h),
                           box_l, 2 * box_h, facecolor=COLOR_GRID,
                           edgecolor=COLOR_FG, lw=1.2))


def _make_wavefronts(ax: Any, x: float, y: float, color: str, n: int = 4,
                     *, theta1: float = 0.0, theta2: float = 360.0,
                     lw: float = 1.6) -> list[Any]:
    """``n`` expanding wavefront arcs centred on ``(x, y)``, initially hidden.

    Drive them with :func:`_set_wavefronts`; ``theta1``/``theta2`` restrict
    the arc span (degrees) for sources radiating into a half or quarter space.
    """
    from matplotlib.patches import Arc

    arcs = []
    for _ in range(n):
        arc = Arc((x, y), 0.01, 0.01, theta1=theta1, theta2=theta2,
                  edgecolor=color, lw=lw)
        arc.set_visible(False)
        ax.add_patch(arc)
        arcs.append(arc)
    return arcs


def _set_wavefronts(arcs: list[Any], radii: list[float], rmax: float,
                    *, alpha: float = 0.9, color: Any = None) -> list[Any]:
    """Resize each wavefront arc; fronts fade toward ``rmax`` and hide beyond.

    ``radii`` pairs with ``arcs``; non-positive radii hide the arc. Returns
    the arcs (for blit-style artist lists).
    """
    for arc, r in zip(arcs, radii, strict=True):
        if r <= 0.0 or r > rmax:
            arc.set_visible(False)
            continue
        arc.set_visible(True)
        arc.width = 2.0 * r
        arc.height = 2.0 * r
        arc.set_alpha(alpha * max(0.0, 1.0 - r / rmax))
        if color is not None:
            arc.set_edgecolor(color)
    return arcs


def _polyline_point(pts: Any, frac: float) -> tuple[float, float]:
    """The point a fraction ``frac`` of the arc length along a polyline.

    ``pts`` is an ``(N, 2)`` array of vertices; ``frac`` is clipped to
    [0, 1]. Used to move pulses/probes along schematic paths at constant
    speed regardless of how the vertices are spaced.
    """
    p = np.asarray(pts, dtype=float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = float(np.clip(frac, 0.0, 1.0)) * cum[-1]
    i = int(np.searchsorted(cum[1:], s, side="right"))
    i = min(i, len(seg) - 1)
    t = 0.0 if seg[i] == 0.0 else (s - cum[i]) / seg[i]
    x = p[i, 0] + t * (p[i + 1, 0] - p[i, 0])
    y = p[i, 1] + t * (p[i + 1, 1] - p[i, 1])
    return float(x), float(y)


def _make_gauge(ax: Any, cx: float, cy: float, r: float, label: str,
                color: str, lo: str = "", hi: str = "") -> dict[str, Any]:
    """A semicircular meter dial; move the needle with :func:`_set_gauge`.

    ``lo``/``hi`` are optional scale-endpoint labels (left and right end of
    the arc), so a reader can anchor the needle position to numbers.
    """
    from matplotlib.patches import Arc

    ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=0.0, theta2=180.0,
                     edgecolor=COLOR_FG, lw=1.4))
    for frac in np.linspace(0.0, 1.0, 5):
        a = np.pi * (1.0 - float(frac))
        ax.plot([cx + 0.86 * r * np.cos(a), cx + r * np.cos(a)],
                [cy + 0.86 * r * np.sin(a), cy + r * np.sin(a)],
                color=COLOR_FG, lw=0.8)
    if lo:
        ax.text(cx - 1.12 * r, cy - 0.02 * r, lo, ha="center", va="top",
                color=COLOR_FG, fontsize=7)
    if hi:
        ax.text(cx + 1.12 * r, cy - 0.02 * r, hi, ha="center", va="top",
                color=COLOR_FG, fontsize=7)
    (needle,) = ax.plot([cx, cx - 0.78 * r], [cy, cy], color=color, lw=2.4,
                        solid_capstyle="round")
    ax.plot([cx], [cy], marker="o", ms=4.5, color=color)
    ax.text(cx, cy - 0.30 * r, label, ha="center", va="top", color=color,
            fontsize=11, fontweight="bold")
    value = ax.text(cx, cy - 0.72 * r, "", ha="center", va="top",
                    color=COLOR_FG, fontsize=9, family="monospace")
    return {"needle": needle, "value": value, "cx": cx, "cy": cy, "r": r}


def _set_gauge(gauge: dict[str, Any], frac: float, text: str) -> tuple[Any, Any]:
    """Point the needle at ``frac`` in [0, 1] (left to right) + set readout."""
    a = np.pi * (1.0 - float(np.clip(frac, 0.0, 1.0)))
    cx, cy, r = gauge["cx"], gauge["cy"], gauge["r"]
    gauge["needle"].set_data([cx, cx + 0.78 * r * np.cos(a)],
                             [cy, cy + 0.78 * r * np.sin(a)])
    gauge["value"].set_text(text)
    return gauge["needle"], gauge["value"]


def _flow_box(ax: Any, cx: float, cy: float, w: float, h: float,
              title: str) -> dict[str, Any]:
    """A processing-pipeline box, dimmed until :func:`_light_box` lights it."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                         boxstyle="round,pad=0.05,rounding_size=0.12",
                         facecolor="none",
                         edgecolor=to_rgba(COLOR_FG, 0.35), lw=1.6)
    ax.add_patch(box)
    title_t = ax.text(cx, cy + 0.22 * h, title, ha="center", va="center",
                      color=COLOR_FG, alpha=0.55, fontsize=9)
    value_t = ax.text(cx, cy - 0.22 * h, "", ha="center", va="center",
                      color=COLOR_FG, fontsize=9, fontweight="bold",
                      family="monospace")
    return {"box": box, "title": title_t, "value": value_t}


def _light_box(box: dict[str, Any], value: str, color: str,
               fill: bool = False) -> None:
    """Light a pipeline box up: colored edge, full-strength title, a value."""
    from matplotlib.colors import to_rgba

    box["box"].set_edgecolor(color)
    box["box"].set_linewidth(2.2)
    box["box"].set_facecolor(to_rgba(color, 0.15) if fill else "none")
    box["title"].set_alpha(1.0)
    box["value"].set_text(value)
    box["value"].set_color(color)


def _dim_box(box: dict[str, Any]) -> None:
    """Return a pipeline box to its dimmed, not-yet-evaluated state."""
    from matplotlib.colors import to_rgba

    box["box"].set_edgecolor(to_rgba(COLOR_FG, 0.35))
    box["box"].set_linewidth(1.6)
    box["box"].set_facecolor("none")
    box["title"].set_alpha(0.55)
    box["value"].set_text("")


def _make_arrow(ax: Any, color: str, scale: float = 14.0) -> Any:
    """An updatable arrow patch; reposition it with ``set_positions``."""
    from matplotlib.patches import FancyArrowPatch

    arrow = FancyArrowPatch((0.0, 0.0), (0.0, 0.0), arrowstyle="-|>",
                            mutation_scale=scale, color=color, lw=2.0,
                            shrinkA=0.0, shrinkB=0.0)
    ax.add_patch(arrow)
    return arrow


def _draw_resistor(ax: Any, x0: float, x1: float, y: float) -> None:
    """A zigzag resistor symbol on a horizontal wire from *x0* to *x1*."""
    n = 6
    xs = np.linspace(x0, x1, 2 * n + 1)
    ys = np.full(xs.size, y)
    ys[1:-1] = y + 0.22 * np.where(np.arange(1, 2 * n) % 2 == 1, 1.0, -1.0)
    ax.plot(xs, ys, color=COLOR_FG, lw=1.4)


def animate_time_weighting_ballistics(output_dir: str) -> None:
    """A tone burst through the IEC 61672-1 RC detector: the capacitor fills
    and drains while the F/S/I meter needles follow their own ballistics."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    from phonometry import time_weighting

    T = _translate_str
    fs = 8000
    t = np.linspace(0, 4.0, int(fs * 4.0), endpoint=False)
    # A steady 250 Hz tone burst (unit amplitude) from 1.0 s to 2.5 s. The
    # carrier is high enough that even the 35 ms Impulse detector smooths the
    # squared ripple, so each detector shows its own clean rise and decay
    # (a noise burst would drown the ballistics in fluctuation).
    x = np.zeros_like(t)
    on = (t >= 1.0) & (t < 2.5)
    x[on] = np.sin(2 * np.pi * 250 * t[on])
    # Each detector is normalized to its own steady reading of the burst, so
    # the clip compares pure ballistics: a real meter shows the same level on
    # F, S and I for a steady tone (the asymmetric Impulse kernel otherwise
    # rides the carrier ripple toward its peaks and would sit ~2.6 dB high).
    i_ss = int(2.45 * fs)   # inside the burst, all detectors settled
    fast = time_weighting(x, fs, mode="fast")
    slow = time_weighting(x, fs, mode="slow")
    imp = time_weighting(x, fs, mode="impulse")
    fast /= fast[i_ss]
    slow /= 0.5             # Slow has not settled by 2.45 s; use the tone MS
    imp /= imp[i_ss]
    # Slope of the Fast output for the charging/draining indicator, smoothed
    # over 50 ms so the residual 500 Hz detector ripple cannot flip its sign.
    kernel = np.ones(400) / 400.0
    dfast = np.gradient(np.convolve(fast, kernel, mode="same"), t)
    col_imp = "#7e57c2"

    fig = _anim_figure()
    fig.suptitle(T("Time-weighting ballistics (IEC 61672-1)"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35],
                          height_ratios=[1.0, 1.15])
    ax_s = fig.add_subplot(gs[:, 0])
    _schematic_axes(ax_s, (0.0, 10.0), (0.0, 10.0))
    ax_g = fig.add_subplot(gs[0, 1])
    _schematic_axes(ax_g, (0.0, 3.4), (-0.55, 0.68), equal=True)
    ax_t = fig.add_subplot(gs[1, 1])
    _grid_axes(ax_t)

    # --- schematic: input strip feeding the square-law + RC detector ------
    ax_s.text(5.0, 10.0, T("RC exponential detector"), ha="center", va="top",
              color=COLOR_FG, fontsize=11, fontweight="bold")
    # Display carrier slowed to ~4 Hz so the burst reads as a waveform, not
    # a filled block (the detectors run on the real 250 Hz burst).
    x_vis = np.where(on, np.sin(2 * np.pi * 4.0 * t), 0.0)
    ax_s.plot(0.6 + 8.8 * t[::4] / 4.0, 8.2 + 0.7 * x_vis[::4],
              color=COLOR_PRIMARY, lw=1.0, alpha=0.9)
    ax_s.text(9.4, 9.05, T("input x(t)"), ha="right", va="bottom",
              color=COLOR_FG, fontsize=9)
    (strip_cur,) = ax_s.plot([0.6, 0.6], [7.4, 9.0], color=COLOR_FG,
                             lw=1.2, alpha=0.7)
    ax_s.annotate("", xy=(0.6, 6.0), xytext=(0.6, 7.3),
                  arrowprops={"arrowstyle": "-|>", "color": COLOR_FG, "lw": 1.2})
    wire = {"color": COLOR_FG, "lw": 1.4}
    ax_s.plot([0.6, 1.3], [5.6, 5.6], **wire)
    ax_s.add_patch(FancyBboxPatch((1.3, 5.0), 1.4, 1.2,
                                  boxstyle="round,pad=0.04",
                                  facecolor="none", edgecolor=COLOR_FG, lw=1.4))
    ax_s.text(2.0, 5.6, "$x^2$", ha="center", va="center", color=COLOR_FG,
              fontsize=13)
    ax_s.text(2.0, 4.55, T("square-law rectifier"), ha="center", va="top",
              color=COLOR_FG, fontsize=8.5)
    ax_s.plot([2.7, 3.2], [5.6, 5.6], **wire)
    _draw_resistor(ax_s, 3.2, 5.0, 5.6)
    ax_s.text(4.1, 6.1, "R", ha="center", va="bottom", color=COLOR_FG,
              fontsize=11)
    ax_s.plot([5.0, 8.1], [5.6, 5.6], **wire)
    ax_s.plot([6.3], [5.6], marker="o", ms=4, color=COLOR_FG)
    ax_s.annotate("", xy=(9.0, 5.6), xytext=(8.1, 5.6),
                  arrowprops={"arrowstyle": "-|>", "color": COLOR_FG, "lw": 1.4})
    ax_s.text(8.55, 6.0, r"$10\,\log_{10}$", ha="center", va="bottom",
              color=COLOR_FG, fontsize=9)
    ax_s.text(9.15, 5.6, "dB", ha="left", va="center", color=COLOR_FG,
              fontsize=10)
    # capacitor drawn as a tank so its state of charge is visible
    ax_s.plot([6.3, 6.3], [5.6, 4.7], **wire)
    ax_s.plot([5.6, 7.0], [4.7, 4.7], color=COLOR_FG, lw=2.2)
    ax_s.plot([5.6, 7.0], [3.2, 3.2], color=COLOR_FG, lw=2.2)
    ax_s.text(7.2, 3.95, "C", ha="left", va="center", color=COLOR_FG,
              fontsize=11)
    # tank walls, animated fill and a liquid-level line so partial charge
    # reads as partial even in a still frame
    for xw in (5.6, 7.0):
        ax_s.plot([xw, xw], [3.2, 4.7], color=COLOR_GRID, lw=1.0, ls=":")
    cap_fill = Rectangle((5.62, 3.24), 1.36, 0.0, facecolor=COLOR_PRIMARY,
                         alpha=0.8 if _FILENAME_SUFFIX else 0.55,
                         edgecolor="none")
    ax_s.add_patch(cap_fill)
    (cap_level,) = ax_s.plot([5.62, 6.98], [3.24, 3.24], color=COLOR_PRIMARY,
                             lw=2.0)
    ax_s.plot([6.3, 6.3], [3.2, 2.62], **wire)
    for gw, gy in ((0.7, 2.62), (0.44, 2.48), (0.18, 2.34)):
        ax_s.plot([6.3 - gw / 2, 6.3 + gw / 2], [gy, gy], color=COLOR_FG,
                  lw=1.4)
    ax_s.text(6.3, 2.05, T("stored charge (Fast shown)"), ha="center",
              va="top", color=COLOR_FG, fontsize=8.5)
    charge_arrow = _make_arrow(ax_s, COLOR_TERTIARY, scale=12.0)
    charge_txt = ax_s.text(4.95, 3.9, "", ha="right", va="center",
                           color=COLOR_FG, fontsize=8.5)
    ax_s.text(5.0, 1.3, T("τ = RC sets attack and decay"), ha="center",
              va="center", color=COLOR_FG, fontsize=9)
    ax_s.text(5.0, 0.6, "F 125 ms · S 1000 ms · I 35/1500 ms", ha="center",
              va="center", color=COLOR_FG, fontsize=8.5, alpha=0.85)

    # --- meter gauges + response traces -----------------------------------
    # One shared 0..1.2 scale, spelled out on the first dial only (the
    # endpoint labels of adjacent dials would otherwise collide).
    gauges = [_make_gauge(ax_g, 0.6, 0.0, 0.5, "F", COLOR_PRIMARY,
                          lo="0", hi="1.2"),
              _make_gauge(ax_g, 1.7, 0.0, 0.5, "S", COLOR_SECONDARY),
              _make_gauge(ax_g, 2.8, 0.0, 0.5, "I", col_imp)]
    ax_t.set_xlim(0.5, 4.0)
    ax_t.set_ylim(0, 1.25)
    ax_t.axvspan(1.0, 2.5, color=COLOR_GRID, alpha=0.4, lw=0)
    ax_t.text(1.75, 1.19, T("tone burst"), ha="center", va="top",
              color=COLOR_FG, fontsize=8.5, alpha=0.8)
    (l_f,) = ax_t.plot([], [], color=COLOR_PRIMARY, lw=2.0,
                       label=T("Fast (125 ms)"))
    (l_s,) = ax_t.plot([], [], color=COLOR_SECONDARY, lw=2.0,
                       label=T("Slow (1000 ms)"))
    (l_i,) = ax_t.plot([], [], color=col_imp, lw=1.7, ls="-.",
                       label=T("Impulse (35 ms / 1.5 s)"))
    cursor = ax_t.axvline(0.5, color=COLOR_FG, lw=1.0, alpha=0.45)
    ax_t.set_xlabel(T("Time [s]"))
    ax_t.set_ylabel(T("Mean-square response (normalized)"), fontsize=8)
    ax_t.legend(loc="upper right", fontsize=7.5)

    tmin, tmax = 0.5, 4.0
    # Sweep the burst-and-decay, then hold the settled three-trace comparison.
    sweep = _ANIM_FRAMES - _ANIM_HOLD

    def update(k: int) -> tuple[Any, ...]:
        tc = tmin + (tmax - tmin) * min(k, sweep - 1) / (sweep - 1)
        i = max(0, min(t.size - 1, round(tc * fs)))
        xc = 0.6 + 8.8 * tc / 4.0
        strip_cur.set_data([xc, xc], [7.4, 9.0])
        level = 3.24 + 1.42 * min(float(fast[i]), 1.02)
        cap_fill.set_height(level - 3.24)
        cap_level.set_data([5.62, 6.98], [level, level])
        slope = float(dfast[i])
        if abs(slope) > 0.05:
            charging = slope > 0
            charge_arrow.set_visible(True)
            y0, y1 = (4.55, 3.35) if charging else (3.35, 4.55)
            charge_arrow.set_positions((5.2, y0), (5.2, y1))
            charge_arrow.set_color(COLOR_TERTIARY if charging
                                   else COLOR_SECONDARY)
            charge_txt.set_text(T("charging") if charging else T("draining"))
        else:
            charge_arrow.set_visible(False)
            charge_txt.set_text("")
        arts: list[Any] = [strip_cur, cap_fill, cap_level, charge_arrow,
                           charge_txt]
        for gauge, val in zip(gauges, (fast[i], slow[i], imp[i]), strict=True):
            # Normalize onto the dial's 0..1.2 scale so the needle position
            # matches the endpoint labels and the numeric readout.
            arts += _set_gauge(gauge, float(val) / 1.2, T(f"{val:.2f}"))
        m = t <= tc
        l_f.set_data(t[m], fast[m])
        l_s.set_data(t[m], slow[m])
        l_i.set_data(t[m], imp[m])
        cursor.set_xdata([tc, tc])
        arts += [l_f, l_s, l_i, cursor]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_time_weighting")


def animate_onset_detection(output_dir: str) -> None:
    """The NT ACOU 112 detector scanning L_AF with a magnifier; the
    OR -> LD -> P -> KI decision chain lights up once the onset is found."""
    from matplotlib.patches import Ellipse

    from phonometry import impulse_adjustment, predicted_prominence

    T = _translate_str
    fs = 500
    t = np.linspace(0, 3.0, int(fs * 3.0), endpoint=False)
    ls, le = 55.0, 85.0    # start and end level of the onset, dB
    t0, rise = 1.0, 0.15   # onset at 1.0 s, lasting 150 ms
    laf = np.full_like(t, ls)
    ramp = (t >= t0) & (t < t0 + rise)
    laf[ramp] = ls + (le - ls) * 0.5 * (1 - np.cos(np.pi * (t[ramp] - t0) / rise))
    after = t >= t0 + rise
    laf[after] = ls + (le - ls) * np.exp(-(t[after] - t0 - rise) / 0.6)
    grad = np.gradient(laf, t)
    onset_rate = (le - ls) / rise      # 200 dB/s
    level_diff = le - ls               # 30 dB
    prom = float(predicted_prominence(onset_rate, level_diff))
    ki = float(impulse_adjustment(prom))
    is_onset = grad > 10.0             # clauses 4.5-4.7

    fig = _anim_figure()
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 0.9])
    ax = fig.add_subplot(gs[0])
    _grid_axes(ax)
    ax.set_xlim(0.55, 3.0)
    ax.set_ylim(42, 100)
    ax.plot(t, laf, color=COLOR_PRIMARY, lw=1.8,
            label=T("L_AF (A-weighted, Fast)"))
    (hot,) = ax.plot([], [], color=COLOR_SECONDARY, lw=4.0,
                     solid_capstyle="round", label=T("onset (> 10 dB/s)"))
    rx, ry = 0.13, 6.5
    lens = Ellipse((0.7, ls), width=2 * rx, height=2 * ry, facecolor="none",
                   edgecolor=COLOR_FG, lw=2.2, zorder=5)
    ax.add_patch(lens)
    (handle,) = ax.plot([], [], color=COLOR_FG, lw=3.2,
                        solid_capstyle="round", zorder=5)
    (tangent,) = ax.plot([], [], color=COLOR_FG, lw=1.6, ls="--", zorder=6)
    lens_txt = ax.text(0.7, ls, "", ha="center", va="bottom", color=COLOR_FG,
                       fontsize=9, zorder=6)
    ax.text(0.02, 0.05, T("detector: onset when dL/dt > 10 dB/s"),
            transform=ax.transAxes, ha="left", va="bottom", color=COLOR_FG,
            fontsize=9)
    ax.set_title(T("Impulse onset detection (NT ACOU 112)"), fontweight="bold")
    ax.set_xlabel(T("Time [s]"))
    ax.set_ylabel(T("A-weighted level L_AF [dB]"), fontsize=9)
    ax.legend(loc="upper right", fontsize=8)

    ax_b = fig.add_subplot(gs[1])
    _schematic_axes(ax_b, (0.0, 10.0), (0.0, 2.3))
    titles = [T("onset rate"), T("level difference"), T("prominence"),
              T("adjustment")]
    boxes = [_flow_box(ax_b, cx, 1.35, 1.95, 1.35, title)
             for cx, title in zip((1.5, 3.9, 6.3, 8.7), titles, strict=True)]
    for xa in (2.5, 4.9, 7.3):
        ax_b.annotate("", xy=(xa + 0.42, 1.35), xytext=(xa, 1.35),
                      arrowprops={"arrowstyle": "-|>", "color": COLOR_GRID,
                                  "lw": 1.6})
    verdict = ax_b.text(5.0, 0.28, "", ha="center", va="center",
                        color=COLOR_SECONDARY, fontsize=10, fontweight="bold")
    values = (f"OR = {onset_rate:.0f} dB/s", f"LD = {level_diff:.0f} dB",
              f"P = {prom:.1f}", f"KI = {ki:.1f} dB")

    # Nonuniform sweep: slow motion while the magnifier crosses the onset.
    # The sweep ends _ANIM_HOLD frames early; np.interp clamps past the last
    # knot, so the lit OR -> LD -> P -> KI chain and the verdict hold ~2 s.
    end_k = float(_ANIM_FRAMES - 1 - _ANIM_HOLD)
    knots_k = (0.0, 0.22 * end_k, 0.52 * end_k, end_k)
    knots_t = (0.62, 0.985, 1.19, 3.0)

    def update(k: int) -> tuple[Any, ...]:
        tc = float(np.interp(k, knots_k, knots_t))
        # Park the magnifier just short of the right edge so its lens and
        # handle stay fully inside the panel during the closing hold (tc
        # reaches 3.0 s, the xlim). The decision chain below still keys off
        # the true tc, so only the glyph position is clamped.
        tc_view = min(max(tc, 0.55 + rx), 3.0 - rx * 1.9)
        i = max(0, min(t.size - 1, round(tc_view * fs)))
        y0, g = float(laf[i]), float(grad[i])
        detecting = g > 10.0
        color = COLOR_SECONDARY if detecting else COLOR_FG
        lens.set_center((tc_view, y0))
        lens.set_edgecolor(color)
        dxl = rx * 0.72
        if abs(g) * dxl > ry * 0.72:
            dxl = ry * 0.72 / abs(g)
        tangent.set_data([tc_view - dxl, tc_view + dxl],
                         [y0 - g * dxl, y0 + g * dxl])
        tangent.set_color(color)
        handle.set_data([tc_view + rx * 0.75, tc_view + rx * 1.9],
                        [y0 - ry * 0.75, y0 - ry * 1.9])
        lens_txt.set_position((tc_view, y0 + ry * 1.35))
        lens_txt.set_text(T(f"dL/dt = {g:.0f} dB/s"))
        lens_txt.set_color(color)
        hot_m = (t <= tc) & is_onset
        hot.set_data(t[hot_m], laf[hot_m])
        arts: list[Any] = [lens, handle, tangent, lens_txt, hot]
        for j, (box, val) in enumerate(zip(boxes, values, strict=True)):
            if tc >= 1.3 + 0.4 * j:
                _light_box(box, T(val),
                           COLOR_SECONDARY if j == 3 else COLOR_PRIMARY,
                           fill=j == 3)
            else:
                _dim_box(box)
            arts += [box["box"], box["title"], box["value"]]
        verdict.set_text(T("add KI to the rating level") if tc >= 2.6 else "")
        arts.append(verdict)
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_onset_detection")


def animate_instantaneous_intensity(output_dir: str) -> None:
    """A p-p probe with p/u phasors: the instantaneous intensity arrow flips
    while its running average settles to net flow (active) or zero (reactive)."""
    from matplotlib.patches import Circle

    T = _translate_str
    t = np.linspace(0, 3.0, 600)
    w = 2 * np.pi * 2.0
    # Progressive: p and u in phase -> p·u >= 0, non-zero mean (net flow).
    # Standing (at a point): p and u 90 deg out of phase -> p·u averages zero.
    cases = [
        (T("Progressive wave — active"), 0.0, T("p and u in phase")),
        (T("Standing wave — reactive"), np.pi / 2, T("p and u 90° apart")),
    ]
    fig = _anim_figure()
    fig.suptitle(T("Two-microphone p-p probe: instantaneous intensity p·u"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    dial_c, dial_r = (1.55, 3.05), 1.25
    i_axis_y, i_scale = 1.05, 2.35
    panels: list[dict[str, Any]] = []
    for col, (title, phi, caption) in enumerate(cases):
        ax_s = fig.add_subplot(gs[0, col])
        _schematic_axes(ax_s, (0.0, 10.0), (0.0, 4.9), equal=True)
        ax_s.set_title(title, fontsize=10.5, fontweight="bold")
        ax_s.add_patch(Circle(dial_c, dial_r, facecolor="none",
                              edgecolor=COLOR_GRID, lw=1.2))
        p_ph = _make_arrow(ax_s, COLOR_PRIMARY, scale=11.0)
        u_ph = _make_arrow(ax_s, COLOR_TERTIARY, scale=11.0)
        ax_s.text(dial_c[0] - dial_r - 0.15, dial_c[1] + dial_r - 0.15, "p",
                  color=COLOR_PRIMARY, fontsize=11, ha="right", va="center",
                  fontweight="bold")
        ax_s.text(dial_c[0] - dial_r - 0.15, dial_c[1] - dial_r + 0.15, "u",
                  color=COLOR_TERTIARY, fontsize=11, ha="right", va="center",
                  fontweight="bold")
        # Tucked up under the phasor dial so it clears the intensity
        # number-line below (the longer Spanish caption otherwise crowds it).
        ax_s.text(dial_c[0], 1.72, caption, ha="center", va="top",
                  color=COLOR_FG, fontsize=8.0)
        _draw_mic(ax_s, 4.6, 3.4, direction=1, size=1.05, label="$p_1$")
        _draw_mic(ax_s, 7.4, 3.4, direction=-1, size=1.05, label="$p_2$")
        ax_s.annotate("", xy=(6.85, 2.6), xytext=(5.15, 2.6),
                      arrowprops={"arrowstyle": "<->", "color": COLOR_FG,
                                  "lw": 1.0})
        ax_s.text(6.0, 2.38, T("spacer Δr"), ha="center", va="top",
                  color=COLOR_FG, fontsize=8.5)
        ax_s.plot([3.4, 8.6], [i_axis_y, i_axis_y], color=COLOR_GRID,
                  lw=1.0, ls="--")
        ax_s.plot([6.0, 6.0], [i_axis_y - 0.15, i_axis_y + 0.15],
                  color=COLOR_FG, lw=1.0, alpha=0.7)
        ax_s.text(6.0, i_axis_y + 0.22, "0", ha="center", va="bottom",
                  color=COLOR_FG, fontsize=8, alpha=0.8)
        ax_s.text(3.15, i_axis_y, "−", ha="right", va="center",
                  color=COLOR_FG, fontsize=10, alpha=0.8)
        ax_s.text(8.85, i_axis_y, "+", ha="left", va="center",
                  color=COLOR_FG, fontsize=10, alpha=0.8)
        ax_s.text(3.4, 1.55, T("I(t) = p·u"), ha="left", va="bottom",
                  color=COLOR_SECONDARY, fontsize=9)
        i_arrow = _make_arrow(ax_s, COLOR_SECONDARY, scale=16.0)
        (mean_marker,) = ax_s.plot([], [], marker="^", ms=7, color=COLOR_FG)
        mean_lab = ax_s.text(6.0, 0.42, "⟨I⟩", ha="center", va="top",
                             color=COLOR_FG, fontsize=8.5)

        ax_tr = fig.add_subplot(gs[1, col])
        _grid_axes(ax_tr)
        ax_tr.set_xlim(0, 3.0)
        ax_tr.set_ylim(-1.15, 1.15)
        p_sig = np.cos(w * t)
        u_sig = np.cos(w * t - phi)
        ax_tr.plot(t, p_sig, color=COLOR_PRIMARY, alpha=0.55, lw=1.1,
                   label=T("pressure p"))
        ax_tr.plot(t, u_sig, color=COLOR_TERTIARY, alpha=0.55, lw=1.1,
                   label=T("velocity u"))
        (iline,) = ax_tr.plot([], [], color=COLOR_SECONDARY, lw=2.0,
                              label=T("intensity p·u"))
        mline = ax_tr.axhline(0.0, color=COLOR_FG, ls="--", lw=1.1, alpha=0.7)
        txt = ax_tr.text(0.5, 0.02, "", transform=ax_tr.transAxes,
                         ha="center", va="bottom", family="monospace",
                         fontsize=10, color=COLOR_FG)
        ax_tr.set_xlabel(T("Time [s]"))
        if col == 0:
            ax_tr.set_ylabel(T("amplitude (normalized)"), fontsize=9)
        ax_tr.legend(loc="upper right", fontsize=7.5)
        panels.append({"ax": ax_tr, "phi": phi, "I": p_sig * u_sig,
                       "p_ph": p_ph, "u_ph": u_ph, "i_arrow": i_arrow,
                       "mean_marker": mean_marker, "mean_lab": mean_lab,
                       "iline": iline, "mline": mline, "txt": txt,
                       "fill": None})

    # Six carrier periods sweep by, then the settled averages (net flow vs
    # zero) hold so the active/reactive contrast can be read.
    sweep = _ANIM_FRAMES - _ANIM_HOLD

    def update(k: int) -> tuple[Any, ...]:
        tc = 3.0 * min(k, sweep - 1) / (sweep - 1)
        idx = max(1, int(np.searchsorted(t, tc)))
        # Average over whole periods once one is complete, so the reactive
        # mean pins to exactly zero instead of hovering near it.
        n_per = np.floor(tc * 2.0)
        idx_mean = idx if n_per < 1 else max(1, int(np.searchsorted(
            t, float(n_per) / 2.0)))
        ph = w * tc
        cx, cy = dial_c
        arts: list[Any] = []
        for pn in panels:
            pv = float(np.cos(ph))
            uv = float(np.cos(ph - pn["phi"]))
            pn["p_ph"].set_positions(
                (cx, cy), (cx + 1.05 * dial_r * np.cos(ph),
                           cy + 1.05 * dial_r * np.sin(ph)))
            pn["u_ph"].set_positions(
                (cx, cy), (cx + 0.88 * dial_r * np.cos(ph - pn["phi"]),
                           cy + 0.88 * dial_r * np.sin(ph - pn["phi"])))
            ival = pv * uv
            tip = 6.0 + i_scale * ival
            if abs(tip - 6.0) < 1e-3:
                tip = 6.0 + 1e-3
            pn["i_arrow"].set_positions((6.0, i_axis_y), (tip, i_axis_y))
            mean = float(np.mean(pn["I"][:idx_mean]))
            if abs(mean) < 5e-3:
                mean = 0.0     # avoid a distracting "-0.00" readout
            xm = 6.0 + i_scale * mean
            pn["mean_marker"].set_data([xm], [i_axis_y - 0.22])
            pn["mean_lab"].set_position((xm, i_axis_y - 0.5))
            pn["iline"].set_data(t[:idx], pn["I"][:idx])
            if pn["fill"] is not None:
                pn["fill"].remove()
            pn["fill"] = pn["ax"].fill_between(t[:idx], 0.0, pn["I"][:idx],
                                               color=COLOR_SECONDARY,
                                               alpha=0.22)
            pn["mline"].set_ydata([mean, mean])
            pn["txt"].set_text(T(f"⟨p·u⟩ = {mean:+.2f}"))
            arts += [pn["p_ph"], pn["u_ph"], pn["i_arrow"],
                     pn["mean_marker"], pn["mean_lab"],
                     pn["iline"], pn["fill"], pn["mline"], pn["txt"]]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_instantaneous_intensity")


def animate_schroeder(output_dir: str) -> None:
    """Backward integration of p²(t): the tail energy fills up on one axis
    while the decay curve and its T20/T30 fits emerge on a companion axis."""
    from matplotlib.patches import Patch

    from phonometry import decay_curve, room_parameters
    from phonometry.room.acoustics import _T20_RANGE, _T30_RANGE, _onset_index

    T = _translate_str
    fs, reverb_t = 48000, 1.2
    rng = np.random.default_rng(2026)
    t = np.arange(int(2.0 * fs)) / fs
    ir = (rng.standard_normal(t.size) * np.exp(-6.9077 * t / reverb_t)
          + rng.standard_normal(t.size) * 10.0 ** (-45.0 / 20.0))
    time, level = decay_curve(ir, fs)
    res = room_parameters(ir, fs, limits=None)
    t20, t30 = float(res.t20[0]), float(res.t30[0])
    p2 = ir.astype(np.float64) ** 2
    p2 = p2[_onset_index(p2):]
    t_raw = np.arange(p2.size) / fs
    raw_db = 10.0 * np.log10(np.maximum(p2, p2.max() * 1e-12) / p2.max())

    # Regression lines drawn once the sweep finishes: slope -60/T over each
    # evaluation range, extended to the -60 dB crossing at t = T.
    def _fit(rng_db: tuple[float, float]) -> tuple[float, float] | None:
        mask = (level <= -rng_db[0]) & (level >= -rng_db[1])
        if int(mask.sum()) < 2:
            return None
        slope, intercept = np.polyfit(time[mask], level[mask], 1)
        if slope >= 0.0:   # a non-decaying fit has no meaningful -60 dB crossing
            return None
        return float(slope), float(intercept)

    fits = []
    for rng_db, color, style, key in (
        (_T20_RANGE, COLOR_SECONDARY, "--", "T20 fit"),
        (_T30_RANGE, COLOR_TERTIARY, "-.", "T30 fit"),
    ):
        fit = _fit(rng_db)
        if fit is None:
            continue
        slope, intercept = fit
        fits.append((color, style, key,
                     (-intercept / slope, (-60.0 - intercept) / slope)))
    tmax = float(time.max())
    xmax = max([tmax, *(f[3][1] for f in fits)]) * 1.03
    # Tail-energy fraction E(t)/E(0) on the onset-trimmed squared IR, and a
    # display-decimated copy of the raw level for the mechanism panel.
    cum = np.cumsum(p2[::-1])[::-1]
    e_frac = cum / cum[0]
    ds = slice(None, None, 8)

    fig = _anim_figure()
    fig.suptitle(T("Schroeder backward integration (ISO 3382)"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.3])
    ax_e = fig.add_subplot(gs[0])
    _grid_axes(ax_e)
    ax_e.set_xlim(0, xmax)
    ax_e.set_ylim(-65, 3)
    ax_e.tick_params(labelbottom=False)
    sweep_max = float(t_raw[-1])   # start the front at the very end of p²
    fill_alpha = 0.5 if _FILENAME_SUFFIX else 0.35
    ax_e.plot(t_raw[ds], raw_db[ds], color="gray", alpha=0.4, lw=0.6)
    e_fill = {"art": None}
    front_e = ax_e.axvline(sweep_max, color=COLOR_FG, lw=1.3, alpha=0.55)
    # Sits well below the upper-right legend (a higher anchor ran under it).
    front_txt = ax_e.text(sweep_max, -22.0, T("← integrate from the tail"),
                          ha="left", va="top", color=COLOR_FG, fontsize=9)
    e_txt = ax_e.text(0.02, 0.06, "", transform=ax_e.transAxes, ha="left",
                      va="bottom", family="monospace", fontsize=10,
                      color=COLOR_FG)
    ax_e.set_ylabel(T("Level [dB]"), fontsize=9)
    ax_e.legend(handles=[
        Patch(facecolor="gray", alpha=0.4,
              label=T("squared impulse response p²")),
        Patch(facecolor=COLOR_SECONDARY, alpha=fill_alpha,
              label=T("tail energy") + r"  $E(t)=\int_t^{\infty}p^2\,d\tau$"),
    ], loc="upper right", fontsize=8.5)

    ax_d = fig.add_subplot(gs[1], sharex=ax_e)
    _grid_axes(ax_d)
    ax_d.set_ylim(-65, 3)
    (curve,) = ax_d.plot([], [], color=COLOR_PRIMARY, lw=2.4,
                         label=T("Schroeder decay curve"))
    # The fit lines only exist in the closing frames, so they are announced
    # by the color-matched T20/T30 readouts instead of a premature legend.
    fit_lines = []
    for color, style, _key, span in fits:
        (fl,) = ax_d.plot([], [], color=color, ls=style, lw=1.7)
        fit_lines.append((fl, span))
    front_d = ax_d.axvline(sweep_max, color=COLOR_FG, lw=1.3, alpha=0.55)
    ax_d.text(0.02, 0.08, r"$L(t) = 10\,\log_{10}\,E(t)\,/\,E(0)$",
              transform=ax_d.transAxes, ha="left", va="bottom",
              color=COLOR_FG, fontsize=10)
    ann_fits = [
        ax_d.text(0.56, 0.16, "", transform=ax_d.transAxes, ha="left",
                  va="bottom", family="monospace", fontsize=11,
                  color=COLOR_SECONDARY),
        ax_d.text(0.56, 0.05, "", transform=ax_d.transAxes, ha="left",
                  va="bottom", family="monospace", fontsize=11,
                  color=COLOR_TERTIARY),
    ]
    ax_d.set_xlabel(T("Time [s]"))
    ax_d.set_ylabel(T("Level [dB]"), fontsize=9)
    ax_d.legend(loc="upper right", fontsize=8.5)

    # Sweep for 80% of the frames; the T20/T30 fits and readouts then hold
    # for the remaining ~2.4 s so the verdict can be read.
    reveal = int(_ANIM_FRAMES * 0.8)

    def update(k: int) -> tuple[Any, ...]:
        xf = sweep_max * (1.0 - k / (reveal - 1)) if k < reveal else 0.0
        m = time >= xf
        curve.set_data(time[m], level[m])
        front_e.set_xdata([xf, xf])
        front_d.set_xdata([xf, xf])
        if e_fill["art"] is not None:
            e_fill["art"].remove()
        mr = t_raw[ds] >= xf
        e_fill["art"] = ax_e.fill_between(t_raw[ds][mr], -65.0, raw_db[ds][mr],
                                          color=COLOR_SECONDARY,
                                          alpha=fill_alpha, lw=0)
        idx = min(int(np.searchsorted(t_raw, xf)), e_frac.size - 1)
        e_txt.set_text(T(f"remaining energy: {100.0 * e_frac[idx]:.1f} %"))
        front_txt.set_position((min(xf, xmax * 0.72) + 0.012 * xmax, -22.0))
        front_txt.set_visible(k < reveal)
        arts: list[Any] = [curve, front_e, front_d, e_fill["art"], e_txt,
                           front_txt, *ann_fits]
        if k >= reveal:
            for (fl, (t_lo, t_hi)), ann, name, val in zip(
                    fit_lines, ann_fits, ("T20", "T30"), (t20, t30),
                    strict=False):
                fl.set_data([t_lo, t_hi], [0.0, -60.0])
                ann.set_text(T(f"{name} = {val:.2f} s"))
                arts.append(fl)
        else:
            for ann in ann_fits:
                ann.set_text("")
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_schroeder")


def animate_flanking_paths(output_dir: str) -> None:
    """EN 12354-1 junction schematic: energy pulses leave the source room
    over the Dd, Ff, Fd and Df paths, shrinking at the junction, and each
    path label lights up as its pulse re-radiates into the receiving room."""
    from matplotlib.patches import Circle, Rectangle

    T = _translate_str
    fig = _anim_figure()
    fig.suptitle(T("Flanking transmission paths (EN 12354-1)"),
                 fontweight="bold")
    ax = fig.add_subplot()
    _schematic_axes(ax, (0.0, 14.2), (0.0, 7.6), equal=True)

    # Cross-section: source room | separating wall | receiving room, with a
    # continuous floor slab running through the junction.
    wall_x0, wall_x1 = 6.6, 7.1
    floor_y0, floor_y1 = 1.0, 1.6
    ax.add_patch(Rectangle((wall_x0, floor_y1), wall_x1 - wall_x0, 5.0,
                           facecolor=COLOR_GRID, edgecolor=COLOR_FG, lw=1.2))
    ax.add_patch(Rectangle((0.7, floor_y0), 12.8, floor_y1 - floor_y0,
                           facecolor=COLOR_GRID, edgecolor=COLOR_FG, lw=1.2))
    for xr in (0.7, 13.5):
        ax.plot([xr, xr], [floor_y1, 6.6], color=COLOR_FG, lw=1.4)
    ax.plot([0.7, 13.5], [6.6, 6.6], color=COLOR_FG, lw=1.4)
    ax.text(3.6, 6.35, T("source room"), ha="center", va="top",
            color=COLOR_FG, fontsize=10)
    ax.text(10.3, 6.35, T("receiving room"), ha="center", va="top",
            color=COLOR_FG, fontsize=10)
    # Element letters: capital = source side, lowercase = receiving side
    for xt, yt, s in ((wall_x0 - 0.25, 4.3, "D"), (wall_x1 + 0.25, 4.3, "d"),
                      (3.6, 1.3, "F"), (10.3, 1.3, "f")):
        ax.text(xt, yt, s, ha="center", va="center", color=COLOR_FG,
                fontsize=12, fontweight="bold", fontstyle="italic")
    _draw_speaker(ax, 2.6, 3.6, size=1.35)

    src = (2.9, 3.6)
    col_df = "#7e57c2"
    # Path polylines (source -> element -> junction/wall -> radiator) with a
    # per-path arrival strength standing in for the junction attenuation Kij.
    paths: list[dict[str, Any]] = [
        {"key": "Dd", "color": COLOR_PRIMARY, "arrive": 0.62,
         "pts": [src, (wall_x0, 3.6), (wall_x1, 3.6), (8.6, 3.6)],
         "rad": (wall_x1, 3.6), "span": (-70.0, 70.0),
         "desc": T("direct, wall to wall")},
        {"key": "Ff", "color": COLOR_SECONDARY, "arrive": 0.40,
         "pts": [src, (4.4, floor_y1), (4.4, 1.3), (10.0, 1.3),
                 (10.0, floor_y1)],
         "rad": (10.0, floor_y1), "span": (20.0, 160.0),
         "desc": T("floor to floor")},
        {"key": "Fd", "color": COLOR_TERTIARY, "arrive": 0.30,
         "pts": [src, (4.9, floor_y1), (4.9, 1.3), (6.85, 1.45),
                 (6.85, 3.0), (wall_x1, 3.0), (8.4, 3.0)],
         "rad": (wall_x1, 3.0), "span": (-70.0, 70.0),
         "desc": T("floor to wall")},
        {"key": "Df", "color": col_df, "arrive": 0.30,
         "pts": [src, (wall_x0, 2.4), (6.85, 2.3), (6.85, 1.3),
                 (9.4, 1.3), (9.4, floor_y1)],
         "rad": (9.4, floor_y1), "span": (20.0, 160.0),
         "desc": T("wall to floor")},
    ]
    junction = (6.85, 1.3)
    for pn in paths:
        pts = np.asarray(pn["pts"])
        ax.plot(pts[:, 0], pts[:, 1], color=pn["color"], lw=1.0, ls=":",
                alpha=0.35)
        (trail,) = ax.plot([], [], color=pn["color"], lw=2.0, alpha=0.85)
        pulse = Circle((0.0, 0.0), 0.16, facecolor=pn["color"],
                       edgecolor="none", visible=False)
        ax.add_patch(pulse)
        pn["trail"], pn["pulse"] = trail, pulse
        rad_x, rad_y = pn["rad"]
        pn["arcs"] = _make_wavefronts(ax, rad_x, rad_y, pn["color"], n=3,
                                      theta1=pn["span"][0],
                                      theta2=pn["span"][1], lw=1.4)
        pn["pts_a"] = pts
    junc_txt = ax.text(junction[0] + 0.3, junction[1] - 0.55, "",
                       ha="left", va="top", color=COLOR_FG, fontsize=8.5)
    # Path legend column: lights up as each pulse arrives. The longer Spanish
    # descriptions start a little further left so they clear the right wall.
    labels = []
    lab_x = 10.15 if _LANG == "en" else 9.6
    for j, pn in enumerate(paths):
        yt = 5.6 - 0.55 * j
        lab = ax.text(lab_x, yt, f"{pn['key']} — {pn['desc']}", ha="left",
                      va="center", color=pn["color"], fontsize=8.5,
                      fontweight="bold", alpha=0.35)
        labels.append(lab)
    verdict = ax.text(7.05, 0.35, "", ha="center", va="center",
                      color=COLOR_FG, fontsize=10, fontweight="bold")

    travel = 2.1                    # seconds a pulse takes over its path
    starts = (0.4, 2.6, 4.8, 7.0)   # launch time of each pulse [s]
    sweep_s = (_ANIM_FRAMES - _ANIM_HOLD) / _ANIM_FPS

    def update(kf: int) -> tuple[Any, ...]:
        tc = min(kf / _ANIM_FPS, sweep_s)
        arts: list[Any] = []
        junc_txt.set_text("")
        for pn, t0, lab in zip(paths, starts, labels, strict=True):
            frac = (tc - t0) / travel
            pts = pn["pts_a"]
            if frac <= 0.0:
                pn["pulse"].set_visible(False)
                pn["trail"].set_data([], [])
                _set_wavefronts(pn["arcs"], [0.0] * 3, 1.0)
            elif frac < 1.0:
                px, py = _polyline_point(pts, frac)
                pn["pulse"].set_center((px, py))
                pn["pulse"].set_visible(True)
                # Pulse shrinks and dims along the path: the transmitted
                # energy that is left after the element and the junction.
                scale = 1.0 - (1.0 - pn["arrive"]) * frac
                pn["pulse"].set_radius(0.17 * scale)
                pn["pulse"].set_alpha(0.35 + 0.65 * scale)
                n_tr = max(2, int(frac * 60))
                tr = np.array([_polyline_point(pts, f)
                               for f in np.linspace(0.0, frac, n_tr)])
                pn["trail"].set_data(tr[:, 0], tr[:, 1])
                pn["trail"].set_alpha(0.3 + 0.5 * scale)
                _set_wavefronts(pn["arcs"], [0.0] * 3, 1.0)
                if pn["key"] != "Dd" and 0.35 < frac < 0.75:
                    junc_txt.set_text(
                        T("junction: Kij attenuates each transfer"))
            else:
                pn["pulse"].set_visible(False)
                age = (tc - t0) - travel
                radii = [0.55 * (age - 0.35 * i) for i in range(3)]
                _set_wavefronts(pn["arcs"], radii, 1.8,
                                alpha=0.4 + 0.5 * pn["arrive"])
                lab.set_alpha(1.0)
            arts += [pn["pulse"], pn["trail"], *pn["arcs"], lab]
        verdict.set_text(
            T("R'w sums all paths — always below the wall alone")
            if tc >= 9.4 else "")
        arts += [junc_txt, verdict]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_flanking_paths")


def animate_intensity_scan_power(output_dir: str) -> None:
    """ISO 9614-2 sound power: a p-p probe traces the serpentine scan over
    the top face of the measurement box while the normal-intensity arrows
    appear behind it, and the partial powers of the five faces accumulate
    into the L_W meter."""
    from matplotlib.patches import Circle, Polygon

    T = _translate_str

    # Parallel projection of the measurement box (w x d x h metres).
    bw, bd, bh = 5.2, 2.6, 3.2
    ox, oy = 0.52, 0.30              # projected offset per metre of depth
    x0, y0 = 1.0, 1.1

    def proj(u: float, v: float, w: float) -> tuple[float, float]:
        return (x0 + u + ox * v, y0 + w + oy * v)

    fig = _anim_figure()
    fig.suptitle(T("Intensity scanning over a box surface (ISO 9614-2)"),
                 fontweight="bold")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0])
    ax = fig.add_subplot(gs[0])
    _schematic_axes(ax, (0.0, 9.4), (0.0, 6.4), equal=True)

    # Box edges (hidden rear edges dashed)
    edges = [((0, 0, 0), (bw, 0, 0)), ((bw, 0, 0), (bw, bd, 0)),
             ((bw, bd, 0), (0, bd, 0)), ((0, bd, 0), (0, 0, 0)),
             ((0, 0, bh), (bw, 0, bh)), ((bw, 0, bh), (bw, bd, bh)),
             ((bw, bd, bh), (0, bd, bh)), ((0, bd, bh), (0, 0, bh)),
             ((0, 0, 0), (0, 0, bh)), ((bw, 0, 0), (bw, 0, bh)),
             ((bw, bd, 0), (bw, bd, bh)), ((0, bd, 0), (0, bd, bh))]
    for a, b in edges:
        xa, ya = proj(*a)
        xb, yb = proj(*b)
        ax.plot([xa, xb], [ya, yb], color=COLOR_FG, lw=1.1, alpha=0.65)
    # The machine under test inside the box (a plain block on the floor)
    mx, md = bw / 2, bd / 2
    m_w, m_d, m_h = 1.5, 1.0, 1.0
    base = [(mx - m_w / 2, md - m_d / 2), (mx + m_w / 2, md - m_d / 2),
            (mx + m_w / 2, md + m_d / 2), (mx - m_w / 2, md + m_d / 2)]
    top = [proj(u, v, m_h) for u, v in base]
    front = [proj(base[0][0], base[0][1], 0.0),
             proj(base[1][0], base[1][1], 0.0),
             proj(base[1][0], base[1][1], m_h),
             proj(base[0][0], base[0][1], m_h)]
    side = [proj(base[1][0], base[1][1], 0.0),
            proj(base[2][0], base[2][1], 0.0),
            proj(base[2][0], base[2][1], m_h),
            proj(base[1][0], base[1][1], m_h)]
    # Opaque theme-blended face tints so the box edges behind the machine
    # do not show through it (or through its label).
    from matplotlib.colors import to_rgb
    bg = np.asarray(to_rgb(plt.rcParams["figure.facecolor"]))
    grid_rgb = np.asarray(to_rgb(COLOR_GRID))
    for poly, al in ((front, 0.75), (side, 0.55), (top, 0.9)):
        tint = tuple(al * grid_rgb + (1.0 - al) * bg)
        ax.add_patch(Polygon(poly, closed=True, facecolor=tint,
                             edgecolor=COLOR_FG, lw=1.0))
    ax.text(*proj(mx, md, 0.45), T("source"), ha="center", va="center",
            color=COLOR_FG, fontsize=8.5)
    src_arcs = _make_wavefronts(ax, *proj(mx, md, m_h), COLOR_TERTIARY,
                                n=3, theta1=15.0, theta2=165.0, lw=1.2)

    # Serpentine scan over the TOP face: passes along u at stepped v.
    n_pass = 4
    vs = np.linspace(0.25, bd - 0.25, n_pass)
    serp: list[tuple[float, float]] = []
    for i, v in enumerate(vs):
        us = (0.3, bw - 0.3) if i % 2 == 0 else (bw - 0.3, 0.3)
        serp += [(us[0], float(v)), (us[1], float(v))]
    serp_a = np.asarray(serp)
    (scan_line,) = ax.plot([], [], color=COLOR_PRIMARY, lw=1.6, alpha=0.8)
    probe = Circle((0.0, 0.0), 0.11, facecolor=COLOR_PRIMARY,
                   edgecolor=COLOR_FG, lw=1.0, zorder=6, visible=False)
    ax.add_patch(probe)
    probe_lab = ax.text(0.0, 0.0, T("p-p probe"), ha="left", va="bottom",
                        color=COLOR_FG, fontsize=8.5, visible=False)
    # Normal-intensity arrows on the top face: longer where the source is
    # closer (I·n ~ cos(theta) / r^2), revealed as the probe passes.
    gu, gv = np.meshgrid(np.linspace(0.55, bw - 0.55, 6),
                         np.linspace(0.4, bd - 0.4, 3))
    serp_fine = np.array([_polyline_point(serp_a, fr)
                          for fr in np.linspace(0.0, 1.0, 300)])
    arrows: list[dict[str, Any]] = []
    for u, v in zip(gu.ravel(), gv.ravel(), strict=True):
        r2 = (u - mx) ** 2 + (v - md) ** 2 + (bh - m_h) ** 2
        inten = (bh - m_h) / r2 ** 1.5      # cos(theta) / r^2, unnormalised
        # the serpentine fraction at which the probe passes this grid point
        near = int(np.argmin(np.hypot(serp_fine[:, 0] - u,
                                      serp_fine[:, 1] - v)))
        arrows.append({"i": float(inten), "frac": near / 299.0,
                       "u": float(u), "v": float(v)})
    i_max = max(ad["i"] for ad in arrows)
    for ad in arrows:
        xa, ya = proj(ad["u"], ad["v"], bh)
        ln = 0.28 + 0.75 * ad["i"] / i_max
        arr = _make_arrow(ax, COLOR_SECONDARY, scale=9.0)
        arr.set_positions((xa, ya), (xa, ya + ln))
        arr.set_visible(False)
        ad["arrow"] = arr
    ax.text(0.15, 6.25, T("normal intensity I·n on the surface"),
            ha="left", va="top", color=COLOR_SECONDARY, fontsize=9)

    # Right column: per-face partial powers into the L_W meter.
    ax_m = fig.add_subplot(gs[1])
    _schematic_axes(ax_m, (0.0, 4.4), (0.0, 6.4))
    ax_m.text(2.2, 6.25, T("partial powers"), ha="center", va="top",
              color=COLOR_FG, fontsize=10, fontweight="bold")
    # Face shares of the total power for a source at the box centre floor:
    # top sees the most, the four sides split the rest (plausible shares).
    faces = [("top", T("top"), 0.34), ("front", T("front"), 0.20),
             ("back", T("back"), 0.20), ("left", T("left"), 0.13),
             ("right", T("right"), 0.13)]
    p_total = 1.6e-3                 # W -> L_W = 92.0 dB
    boxes = {}
    for j, (key, lab, _share) in enumerate(faces):
        boxes[key] = _flow_box(ax_m, 1.1, 5.35 - 0.95 * j, 1.9, 0.8, lab)
    gauge = _make_gauge(ax_m, 3.3, 1.1, 0.85, "$L_W$", COLOR_SECONDARY,
                        lo="80", hi="95")
    ax_m.text(3.3, 2.75, r"$P = \sum_i \int I{\cdot}n\ dS_i$", ha="center",
              va="bottom", color=COLOR_FG, fontsize=10)
    # verdict rides under the box drawing, where the full width is free
    verdict = ax.text(4.7, 0.35, "", ha="center", va="center",
                      color=COLOR_SECONDARY, fontsize=10,
                      fontweight="bold")

    scan_end, face_step = 6.2, 0.85  # top-face scan, then one face per step
    sweep_s = (_ANIM_FRAMES - _ANIM_HOLD) / _ANIM_FPS

    def update(kf: int) -> tuple[Any, ...]:
        tc = min(kf / _ANIM_FPS, sweep_s)
        arts: list[Any] = []
        # source breathing wavefronts above the machine
        age = (tc * 0.5) % 0.6
        _set_wavefronts(src_arcs, [0.9 * (age + 0.2 * i) for i in range(3)],
                        0.9, alpha=0.5)
        arts += src_arcs
        frac = min(tc / scan_end, 1.0)
        px, pv = _polyline_point(serp_a, frac)
        xpr, ypr = proj(px, pv, bh)
        probe.set_center((xpr, ypr + 0.06))
        probe.set_visible(tc > 0.0 and frac < 1.0)
        # name the probe during the first pass only, riding below the face
        # so the revealed intensity arrows never strike the label
        probe_lab.set_position((xpr + 0.28, ypr - 0.52))
        probe_lab.set_visible(probe.get_visible() and tc < 1.6)
        n_tr = max(2, int(frac * 90))
        tr = np.array([_polyline_point(serp_a, f)
                       for f in np.linspace(0.0, frac, n_tr)])
        trp = np.array([proj(u, v, bh) for u, v in tr])
        scan_line.set_data(trp[:, 0], trp[:, 1])
        # reveal the normal arrows the probe has already passed
        for a in arrows:
            a["arrow"].set_visible(a["frac"] <= frac)
            arts.append(a["arrow"])
        arts += [probe, probe_lab, scan_line]
        # accumulate the partial powers: top face during the scan, then the
        # remaining faces one by one
        acc = 0.0
        for j, (key, _lab, share) in enumerate(faces):
            if j == 0:
                got = share * frac
            else:
                got = share * float(np.clip(
                    (tc - scan_end - (j - 1) * face_step) / face_step,
                    0.0, 1.0))
            acc += got
            if got >= share - 1e-9:
                _light_box(boxes[key],
                           T(f"{got * p_total * 1e3:.2f} mW"),
                           COLOR_PRIMARY, fill=j == 0)
            elif got > 0.0:
                _light_box(boxes[key],
                           T(f"{got * p_total * 1e3:.2f} mW"),
                           COLOR_PRIMARY)
            else:
                _dim_box(boxes[key])
            b = boxes[key]
            arts += [b["box"], b["title"], b["value"]]
        if acc > 0.0:
            lw_now = 10.0 * np.log10(acc * p_total / 1e-12)
            arts += _set_gauge(gauge, (lw_now - 80.0) / 15.0,
                               T(f"{lw_now:.1f} dB"))
        verdict.set_text(
            T("any enclosing surface gives the same P") if acc >= 0.999
            else "")
        arts.append(verdict)
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_intensity_scan_power")


def animate_sweep_deconvolution(output_dir: str) -> None:
    """ISO 18233 swept-sine measurement: the exponential sweep crosses a
    drawn room while its spectrogram builds, then the inverse filter
    collapses the whole record into the impulse response."""
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    from scipy.signal import fftconvolve, spectrogram

    from phonometry import impulse_response, sweep_signal

    T = _translate_str
    fs = 8000
    sweep_len, f1, f2 = 2.5, 60.0, 3500.0
    sweep = sweep_signal(fs, f1, f2, sweep_len)
    # Synthetic room: direct sound, two discrete echoes far enough apart
    # (90/160 ms >> the 64 ms spectrogram window) that the recorded sweep
    # shows them as visibly separate delayed copies of the main ridge,
    # plus a diffuse tail.
    rng = np.random.default_rng(18233)
    ir_len = int(0.45 * fs)
    system = np.zeros(ir_len)
    for delay_ms, g in ((12.0, 1.0), (90.0, 0.55), (160.0, 0.35)):
        system[int(delay_ms * 1e-3 * fs)] = g
    tail_t = np.arange(ir_len) / fs
    system += (0.05 * rng.standard_normal(ir_len)
               * np.exp(-6.9077 * tail_t / 0.5) * (tail_t > 0.012))
    recorded = fftconvolve(sweep, system)
    rec_dur = recorded.size / fs
    _freqs, times, sxx = spectrogram(recorded, fs, nperseg=512,
                                    noverlap=384)
    sxx_db = 10.0 * np.log10(np.maximum(sxx, sxx.max() * 1e-8) / sxx.max())
    ir = impulse_response(recorded, sweep, fs, method="spectral",
                          length=int(0.2 * fs))
    t_ir = np.arange(ir.ir.size) / fs * 1e3
    ir_n = ir.ir / np.max(np.abs(ir.ir))

    fig = _anim_figure()
    fig.suptitle(T("Sweep measurement and deconvolution (ISO 18233)"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.15])
    ax_r = fig.add_subplot(gs[0, :])
    _schematic_axes(ax_r, (0.0, 18.0), (0.0, 3.4), equal=True)
    room_box = Rectangle((1.0, 0.3), 16.0, 2.8, facecolor="none",
                         edgecolor=COLOR_FG, lw=1.4)
    ax_r.add_patch(room_box)
    _draw_speaker(ax_r, 2.6, 1.7, size=1.1)
    _draw_mic(ax_r, 14.6, 1.7, direction=-1, size=0.85, label=T("mic"))
    # direct and reflected ray paths
    ax_r.plot([2.8, 13.9], [1.7, 1.7], color=COLOR_FG, lw=1.0, ls="--",
              alpha=0.5)
    for ry in (0.3, 3.1):
        ax_r.plot([2.8, 8.3, 13.9], [1.7, ry, 1.75], color=COLOR_FG,
                  lw=0.9, ls=":", alpha=0.45)
    ax_r.text(8.3, 0.05, T("direct + reflections"), ha="center", va="bottom",
              color=COLOR_FG, fontsize=8)
    arcs = _make_wavefronts(ax_r, 2.8, 1.7, COLOR_PRIMARY, n=4,
                            theta1=-75.0, theta2=75.0)
    for arc in arcs:            # wavefronts stay inside the drawn room
        arc.set_clip_path(room_box)
    freq_txt = ax_r.text(2.0, 2.55, "", ha="left", va="bottom",
                         color=COLOR_FG, fontsize=9, family="monospace")
    note = ax_r.text(9.0, 0.8, "", ha="center", va="center",
                     color=COLOR_FG, fontsize=9.5, fontstyle="italic",
                     visible=False,
                     bbox={"boxstyle": "round,pad=0.35",
                           "facecolor": plt.rcParams["figure.facecolor"],
                           "edgecolor": "none", "alpha": 0.85})

    ax_s = fig.add_subplot(gs[1, 0])
    ax_s.grid(False)
    disp = np.full_like(sxx_db, sxx_db.min())
    im = ax_s.imshow(disp, origin="lower", aspect="auto",
                     extent=(0.0, float(times[-1]), 0.0, fs / 2 / 1e3),
                     cmap="magma", norm=Normalize(-60.0, 0.0),
                     interpolation="bilinear")
    ax_s.set_xlabel(T("Time [s]"), fontsize=8)
    ax_s.set_ylabel(T("Frequency [kHz]"), fontsize=8)
    ax_s.tick_params(labelsize=7)
    ax_s.set_title(T("recorded sweep (spectrogram)"), fontsize=9)
    ridge_txt = ax_s.text(0.97, 0.08, "", transform=ax_s.transAxes,
                          ha="right", va="bottom", color="white", fontsize=8)

    ax_i = fig.add_subplot(gs[1, 1])
    _grid_axes(ax_i)
    ax_i.set_xlim(0.0, 200.0)
    ax_i.set_ylim(-1.05, 1.05)
    (ir_line,) = ax_i.plot([], [], color=COLOR_PRIMARY, lw=1.0)
    ax_i.set_xlabel(T("Time [ms]"), fontsize=8)
    ax_i.set_title(T("impulse response"), fontsize=9)
    ax_i.tick_params(labelsize=7)
    tap_marks = [ax_i.annotate("", xy=(ms, g * 1.0),
                               xytext=(ms + 22.0, min(g + 0.35, 0.95)),
                               fontsize=7.5, color=COLOR_FG, ha="left",
                               arrowprops={"arrowstyle": "-",
                                           "color": COLOR_FG, "lw": 0.7},
                               visible=False)
                 for ms, g in ((12.0, 1.0), (90.0, 0.55))]
    tap_marks[0].set_text(T("direct"))
    tap_marks[1].set_text(T("reflections"))
    deconv = _flow_box(ax_i, 140.0, -0.72, 85.0, 0.42,
                       T("⊛ inverse filter"))

    play_end, dec_t = 6.5, 7.6      # sweep playback, deconvolution moment
    sweep_frames = _ANIM_FRAMES - _ANIM_HOLD

    def update(kf: int) -> tuple[Any, ...]:
        tc = min(kf, sweep_frames - 1) / _ANIM_FPS
        arts: list[Any] = []
        t_sig = min(tc / play_end, 1.0) * rec_dur
        # room wavefronts coloured by the instantaneous sweep frequency
        if tc < play_end:
            f_now = f1 * (f2 / f1) ** min(t_sig / sweep_len, 1.0)
            cmap = plt.get_cmap("plasma")
            col = cmap(float(np.log(f_now / f1) / np.log(f2 / f1)))
            age = (tc * 2.2) % 1.0
            _set_wavefronts(arcs, [11.5 * ((age + 0.25 * i) % 1.0)
                                   for i in range(4)], 11.5, color=col)
            freq_txt.set_text(f"f = {f_now:5.0f} Hz")
        else:
            _set_wavefronts(arcs, [0.0] * 4, 1.0)
            freq_txt.set_text("")
        arts += [*arcs, freq_txt]
        # spectrogram builds in sync with the playback
        n_col = int(np.searchsorted(times, t_sig))
        disp[:, :n_col] = sxx_db[:, :n_col]
        im.set_data(disp)
        ridge_txt.set_text(T("delayed copies = reflections")
                           if tc >= 3.4 else "")
        arts += [im, ridge_txt]
        # deconvolution: the record collapses into the impulse response
        if tc >= dec_t:
            _light_box(deconv, "", COLOR_SECONDARY, fill=True)
            reveal = float(np.clip((tc - dec_t) / 1.2, 0.0, 1.0))
            n = max(2, int(reveal * ir_n.size))
            ir_line.set_data(t_ir[:n], ir_n[:n])
            for mk in tap_marks:
                mk.set_visible(reveal >= 1.0)
            if reveal >= 1.0:
                note.set_text(
                    T("same information, different domain: sweep ⊛ inverse"
                      " filter = impulse response"))
                note.set_visible(True)
        else:
            _dim_box(deconv)
            ir_line.set_data([], [])
            note.set_text("")
            note.set_visible(False)
        arts += [ir_line, deconv["box"], deconv["title"], deconv["value"],
                 note, *tap_marks]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_sweep_deconvolution")


def animate_specific_loudness(output_dir: str) -> None:
    """ISO 532-1 specific loudness: the N'(z) pattern of a 1 kHz narrowband
    sound builds along the Bark axis as the band level steps up, and the
    area under the pattern integrates to the total loudness in sone."""
    T = _translate_str
    from phonometry import loudness_zwicker_from_spectrum

    bands = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
             500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
             6300, 8000, 10000, 12500]
    i_1k = bands.index(1000)
    levels = np.arange(40, 86)
    patterns: dict[int, Any] = {}
    totals: dict[int, float] = {}
    for lv in levels:
        third = [-60.0] * 28
        third[i_1k] = float(lv)
        res = loudness_zwicker_from_spectrum(third, field="free")
        patterns[int(lv)] = res.specific
        totals[int(lv)] = float(res.loudness)
    z = np.arange(1, 241) * 0.1

    fig = _anim_figure()
    fig.suptitle(T("Specific loudness N'(z) and its integral (ISO 532-1)"),
                 fontweight="bold")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0])
    ax = fig.add_subplot(gs[0])
    _grid_axes(ax)
    ax.set_xlim(0.0, 24.0)
    # headroom above the 85 dB pattern (max N' ~ 5.4 sone/Bark)
    ax.set_ylim(0.0, 6.0)
    ax.set_xlabel(T("Critical-band rate z [Bark]"))
    ax.set_ylabel(T("Specific loudness N' [sone/Bark]"), fontsize=9)
    (line,) = ax.plot([], [], color=COLOR_PRIMARY, lw=2.2)
    fill = {"art": None}
    ax.axvline(8.5, color=COLOR_FG, lw=0.9, ls=":", alpha=0.6)
    ax.text(8.75, 5.85, T("1 kHz ≈ 8.5 Bark"), ha="left", va="top",
            color=COLOR_FG, fontsize=8)
    spread = ax.annotate(T("upward spread of masking"), xy=(13.0, 1.1),
                         xytext=(16.0, 3.2), fontsize=8.5, color=COLOR_FG,
                         arrowprops={"arrowstyle": "->", "color": COLOR_FG,
                                     "lw": 1.0}, visible=False)
    level_txt = ax.text(0.02, 0.965, "", transform=ax.transAxes, ha="left",
                        va="top", color=COLOR_FG, fontsize=11,
                        family="monospace")

    ax_m = fig.add_subplot(gs[1])
    _schematic_axes(ax_m, (0.0, 4.0), (0.0, 8.6))
    _draw_speaker(ax_m, 0.55, 7.7, size=0.7)
    ax_m.text(1.15, 7.7, T("1 kHz narrowband"), ha="left", va="center",
              color=COLOR_FG, fontsize=8.5)
    gauge = _make_gauge(ax_m, 2.0, 4.6, 1.05, "N", COLOR_SECONDARY,
                        lo="0", hi=T("20 sone"))
    ax_m.text(2.0, 6.5, r"$N = \int_0^{24} N'(z)\, dz$", ha="center",
              va="bottom", color=COLOR_FG, fontsize=11)
    steps = [45, 65, 85]
    step_boxes = [_flow_box(ax_m, 2.0, 2.6 - 0.95 * j, 3.4, 0.8,
                            T(f"{lv} dB"))
                  for j, lv in enumerate(steps)]

    # Level trajectory: three plateaus with 1 s ramps between them.
    knots_t = (0.0, 2.9, 3.9, 6.6, 7.6, 10.0)
    knots_l = (45.0, 45.0, 65.0, 65.0, 85.0, 85.0)
    sweep_s = (_ANIM_FRAMES - _ANIM_HOLD) / _ANIM_FPS
    # First plateau: an explicit left-to-right integration sweep.
    int_t0, int_t1 = 0.4, 2.6

    def update(kf: int) -> tuple[Any, ...]:
        tc = min(kf / _ANIM_FPS, sweep_s)
        lv = round(float(np.interp(tc, knots_t, knots_l)))
        spec = patterns[lv]
        line.set_data(z, spec)
        if tc <= int_t1:
            z_lim = float(np.interp(tc, (int_t0, int_t1), (0.0, 24.0)))
        else:
            z_lim = 24.0
        m = z <= z_lim
        if fill["art"] is not None:
            fill["art"].remove()
        fill["art"] = ax.fill_between(z[m], 0.0, spec[m],
                                      color=COLOR_PRIMARY, alpha=0.3, lw=0)
        n_part = float(np.trapezoid(spec[m], z[m])) if m.any() else 0.0
        n_show = totals[lv] if z_lim >= 24.0 else n_part
        level_txt.set_text(f"L = {lv} dB")
        spread.set_visible(lv >= 80)
        arts: list[Any] = [line, fill["art"], level_txt, spread]
        arts += _set_gauge(gauge, n_show / 20.0, T(f"{n_show:.1f} sone"))
        for j, (lv_s, box) in enumerate(zip(steps, step_boxes, strict=True)):
            done = (tc >= (int_t1, 5.2, 8.9)[j])
            if done:
                _light_box(box, T(f"N = {totals[lv_s]:.1f} sone"),
                           COLOR_SECONDARY if j == 2 else COLOR_PRIMARY,
                           fill=j == 2)
            else:
                _dim_box(box)
            arts += [box["box"], box["title"], box["value"]]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_specific_loudness")


def animate_power_two_rooms(output_dir: str) -> None:
    """The same source in an anechoic room (ISO 3745 free field) and in a
    reverberation room (ISO 3741 diffuse build-up): both microphone
    pressures differ, both routes converge to the same sound power L_W."""
    from matplotlib.patches import Circle, Polygon, Rectangle

    T = _translate_str
    lw_true = 92.0
    r_mic = 2.05                   # projected mic-ring radius (anechoic)
    lp_free = 77.5                 # L_W - 10 log10(4 pi 1.5^2)
    lp_diff = 86.0                 # L_W - 10 log10 V + 10 log10 T + 14

    fig = _anim_figure()
    fig.suptitle(T("One source, two rooms, one sound power"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2, height_ratios=[2.35, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    _schematic_axes(ax_a, (0.0, 8.0), (0.0, 6.6), equal=True)
    ax_r = fig.add_subplot(gs[0, 1])
    _schematic_axes(ax_r, (0.0, 8.0), (0.0, 6.6), equal=True)
    ax_b = fig.add_subplot(gs[1, :])
    _schematic_axes(ax_b, (0.0, 16.0), (0.0, 2.6))

    # --- anechoic room: wedges on every wall, mic ring, free field -------
    ax_a.set_title(T("Anechoic room (ISO 3745)"), fontsize=10,
                   fontweight="bold")
    ax_a.add_patch(Rectangle((0.6, 0.4), 6.8, 5.6, facecolor="none",
                             edgecolor=COLOR_FG, lw=1.4))
    wedge = 0.42
    for xw in np.arange(0.85, 7.3, 0.55):
        ax_a.add_patch(Polygon([(xw - 0.22, 0.4), (xw + 0.22, 0.4),
                                (xw, 0.4 + wedge)], closed=True,
                               facecolor=COLOR_GRID, edgecolor=COLOR_FG,
                               lw=0.6))
        ax_a.add_patch(Polygon([(xw - 0.22, 6.0), (xw + 0.22, 6.0),
                                (xw, 6.0 - wedge)], closed=True,
                               facecolor=COLOR_GRID, edgecolor=COLOR_FG,
                               lw=0.6))
    for yw in np.arange(0.85, 5.9, 0.55):
        ax_a.add_patch(Polygon([(0.6, yw - 0.22), (0.6, yw + 0.22),
                                (0.6 + wedge, yw)], closed=True,
                               facecolor=COLOR_GRID, edgecolor=COLOR_FG,
                               lw=0.6))
        ax_a.add_patch(Polygon([(7.4, yw - 0.22), (7.4, yw + 0.22),
                                (7.4 - wedge, yw)], closed=True,
                               facecolor=COLOR_GRID, edgecolor=COLOR_FG,
                               lw=0.6))
    ca = (4.0, 3.2)
    _draw_speaker(ax_a, ca[0] - 0.05, ca[1], size=0.85)
    # offset by 15 deg so no dot lands on the caption lines at the bottom
    mic_angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False) + np.pi / 12
    for ang in mic_angles:
        ax_a.plot([ca[0] + r_mic * np.cos(ang)],
                  [ca[1] + r_mic * np.sin(ang)], marker="o", ms=4,
                  color=COLOR_PRIMARY)
    ax_a.text(ca[0], ca[1] - r_mic + 0.14, T("microphone sphere, r"),
              ha="center", va="bottom", color=COLOR_PRIMARY, fontsize=8)
    arcs_a = _make_wavefronts(ax_a, *ca, COLOR_TERTIARY, n=4)
    note_a = ax_a.text(4.0, 0.92, T("direct sound only — no reflections"),
                       ha="center", va="bottom", color=COLOR_FG, fontsize=8,
                       alpha=0.0)
    lp_a = ax_a.text(6.9, 5.5, "", ha="right", va="top", color=COLOR_FG,
                     fontsize=9, family="monospace")

    # --- reverberation room: bare walls, diffuse build-up, one mic path --
    ax_r.set_title(T("Reverberation room (ISO 3741)"), fontsize=10,
                   fontweight="bold")
    # animated fill (the diffuse level building up) + a fixed outline
    room = Rectangle((0.6, 0.4), 6.8, 5.6, facecolor=COLOR_PRIMARY,
                     edgecolor="none", alpha=0.0)
    ax_r.add_patch(room)
    ax_r.add_patch(Rectangle((0.6, 0.4), 6.8, 5.6, facecolor="none",
                             edgecolor=COLOR_FG, lw=1.4))
    # a tilted diffuser panel hanging in the volume
    ax_r.plot([1.3, 2.7], [4.7, 5.3], color=COLOR_FG, lw=2.4, alpha=0.8)
    cr = (1.7, 1.3)
    _draw_speaker(ax_r, cr[0] - 0.05, cr[1], size=0.85)
    # bouncing ray trails (specular folding inside the rectangle)
    rays = [{"v": (2.9, 1.9), "trail": ax_r.plot(
        [], [], color=COLOR_TERTIARY, lw=1.1, alpha=0.75)[0]},
        {"v": (2.2, -2.6), "trail": ax_r.plot(
            [], [], color=COLOR_TERTIARY, lw=1.1, alpha=0.75)[0]}]
    mic_path_c, mic_path_r = (4.4, 3.4), 1.35
    ax_r.add_patch(Circle(mic_path_c, mic_path_r, facecolor="none",
                          edgecolor=COLOR_PRIMARY, lw=0.9, ls="--",
                          alpha=0.8))
    (mic_dot,) = ax_r.plot([], [], marker="o", ms=6, color=COLOR_PRIMARY)
    ax_r.text(mic_path_c[0], mic_path_c[1] - mic_path_r - 0.28,
              T("rotating microphone"), ha="center", va="top",
              color=COLOR_PRIMARY, fontsize=8)
    note_r = ax_r.text(4.0, 0.62, T("reflections build a diffuse field"),
                       ha="center", va="bottom", color=COLOR_FG, fontsize=8,
                       alpha=0.0)
    lp_r = ax_r.text(6.9, 5.5, "", ha="right", va="top", color=COLOR_FG,
                     fontsize=9, family="monospace")

    def _fold(p: float, lo: float, hi: float) -> float:
        span = hi - lo
        q = (p - lo) % (2.0 * span)
        return lo + (q if q <= span else 2.0 * span - q)

    # --- bottom strip: the two formulas converge on one L_W --------------
    box_a = _flow_box(ax_b, 3.4, 1.55, 6.2, 1.15,
                      r"$L_W = \bar{L}_p + 10\,\log_{10}(4\pi r^2/S_0)$")
    box_r = _flow_box(ax_b, 12.6, 1.55, 6.2, 1.15,
                      r"$L_W = \bar{L}_p + 10\,\log_{10} V - 10\,\log_{10} T - 14$")
    lw_box = _flow_box(ax_b, 8.0, 1.3, 2.6, 1.3, "$L_W$")
    arr_a = _make_arrow(ax_b, COLOR_SECONDARY, scale=13.0)
    arr_r = _make_arrow(ax_b, COLOR_SECONDARY, scale=13.0)
    for arr in (arr_a, arr_r):
        arr.set_visible(False)
    verdict = ax_b.text(8.0, 0.12, "", ha="center", va="bottom",
                        color=COLOR_SECONDARY, fontsize=10,
                        fontweight="bold")

    sweep_s = (_ANIM_FRAMES - _ANIM_HOLD) / _ANIM_FPS
    t_meter, t_form, t_conv = 4.5, 7.0, 9.4

    def update(kf: int) -> tuple[Any, ...]:
        tc = min(kf / _ANIM_FPS, sweep_s)
        arts: list[Any] = []
        # free field: wavefronts die before the wedges
        age = (tc * 0.55) % 1.0
        _set_wavefronts(arcs_a, [2.9 * ((age + 0.25 * i) % 1.0)
                                 for i in range(4)], 2.9, alpha=0.8)
        arts += arcs_a
        note_a.set_alpha(min(tc / 2.0, 1.0) * 0.9)
        # diffuse field: rays bounce, the background level rises
        build = float(np.clip(tc / 4.0, 0.0, 1.0))
        room.set_alpha(0.16 * (1.0 - np.exp(-3.0 * build)))
        for ray in rays:
            ts = np.linspace(max(0.0, tc - 0.9), tc, 24)
            xs = [_fold(cr[0] + ray["v"][0] * s, 0.7, 7.3) for s in ts]
            ys = [_fold(cr[1] + ray["v"][1] * s, 0.5, 5.9) for s in ts]
            ray["trail"].set_data(xs, ys)
            arts.append(ray["trail"])
        ang = 2.0 * np.pi * 0.12 * tc
        mic_dot.set_data([mic_path_c[0] + mic_path_r * np.cos(ang)],
                         [mic_path_c[1] + mic_path_r * np.sin(ang)])
        note_r.set_alpha(min(tc / 2.0, 1.0) * 0.9)
        arts += [room, mic_dot, note_a, note_r]
        # microphone readings appear, then each formula computes L_W
        if tc >= t_meter:
            lp_a.set_text(T(f"mean Lp = {lp_free:.1f} dB"))
            lp_r.set_text(T(f"mean Lp = {lp_diff:.1f} dB"))
        else:
            lp_a.set_text("")
            lp_r.set_text("")
        # Mathtext subscript for the sound-power symbol; localise the decimal
        # comma by hand because ``$`` disables the ``_translate_str`` comma pass.
        lw_val = f"{lw_true:.1f}" if _LANG == "en" else f"{lw_true:.1f}".replace(".", ",")
        for box, on in ((box_a, tc >= t_form), (box_r, tc >= t_form + 0.6)):
            if on:
                _light_box(box, f"$L_W$ = {lw_val} dB", COLOR_PRIMARY)
            else:
                _dim_box(box)
            arts += [box["box"], box["title"], box["value"]]
        if tc >= t_conv:
            _light_box(lw_box, T(f"{lw_true:.1f} dB"), COLOR_SECONDARY,
                       fill=True)
            arr_a.set_positions((6.6, 1.55), (7.3, 1.4))
            arr_r.set_positions((9.4, 1.55), (8.7, 1.4))
            arr_a.set_visible(True)
            arr_r.set_visible(True)
            verdict.set_text(
                T("the room changes Lp, not the source power"))
        else:
            _dim_box(lw_box)
            arr_a.set_visible(False)
            arr_r.set_visible(False)
            verdict.set_text("")
        arts += [lp_a, lp_r, lw_box["box"], lw_box["title"],
                 lw_box["value"], arr_a, arr_r, verdict]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_power_two_rooms")


def animate_comb_filtering(output_dir: str) -> None:
    """Direct sound plus one floor reflection at a microphone: as the mic
    height changes, the delayed copy shifts and the comb filter in the
    frequency response moves with it, which is why measurement position matters
    near reflecting surfaces."""
    T = _translate_str
    c0 = 343.0
    xs_, hs = 0.6, 1.5              # source position (x, height)
    xm = 4.2                        # mic x position
    refl_g = 0.8                    # floor reflection factor

    fig = _anim_figure()
    fig.suptitle(T("Comb filtering from a single reflection"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    _schematic_axes(ax, (0.0, 5.6), (-2.3, 2.6), equal=True)
    ax.axhline(0.0, color=COLOR_FG, lw=2.0)
    ax.fill_between([0.0, 5.6], -2.3, 0.0, color=COLOR_GRID, alpha=0.35,
                    lw=0)
    ax.text(5.5, -0.34, T("reflecting floor"), ha="right", va="top",
            color=COLOR_FG, fontsize=8.5)
    _draw_speaker(ax, xs_, hs, size=0.8)
    ax.plot([xs_ - 0.25, xs_ - 0.25], [0.0, hs - 0.28], color=COLOR_FG,
            lw=1.2)
    # image source below the floor
    _draw_speaker(ax, xs_, -hs, size=0.8)
    for art in ax.patches[-2:]:
        art.set_alpha(0.3)
    ax.text(xs_ + 0.4, -hs, T("image source"), ha="left", va="center",
            color=COLOR_FG, fontsize=8.5, alpha=0.75)
    (l_dir,) = ax.plot([], [], color=COLOR_PRIMARY, lw=1.8)
    (l_ref,) = ax.plot([], [], color=COLOR_SECONDARY, lw=1.6, ls="--")
    (l_img,) = ax.plot([], [], color=COLOR_SECONDARY, lw=1.0, ls=":",
                       alpha=0.5)
    (mic_stand,) = ax.plot([], [], color=COLOR_FG, lw=1.2)
    (mic_dot,) = ax.plot([], [], marker="o", ms=8, color=COLOR_GRID,
                         markeredgecolor=COLOR_FG, markeredgewidth=1.2)
    mic_lab = ax.text(xm + 0.18, 0.0, T("mic"), ha="left", va="center",
                      color=COLOR_FG, fontsize=8.5)
    delta_txt = ax.text(0.15, 2.45, "", ha="left", va="top",
                        color=COLOR_FG, fontsize=9.5, family="monospace")
    stage_txt = ax.text(3.1, -2.05, "", ha="center", va="center",
                        color=COLOR_FG, fontsize=9, fontweight="bold")

    ax_t = fig.add_subplot(gs[0, 1])
    _grid_axes(ax_t)
    ax_t.set_xlim(9.0, 25.0)
    ax_t.set_ylim(0.0, 1.15)
    ax_t.set_xlabel(T("arrival time [ms]"), fontsize=8)
    ax_t.set_ylabel(T("amplitude"), fontsize=8)
    ax_t.tick_params(labelsize=7)
    (stem_d,) = ax_t.plot([], [], color=COLOR_PRIMARY, lw=2.6,
                          solid_capstyle="butt")
    (stem_r,) = ax_t.plot([], [], color=COLOR_SECONDARY, lw=2.6,
                          solid_capstyle="butt")
    tau_ann = ax_t.annotate("", xy=(0.0, 0.9), xytext=(0.0, 0.9),
                            arrowprops={"arrowstyle": "<->",
                                        "color": COLOR_FG, "lw": 1.0})
    tau_txt = ax_t.text(0.0, 0.98, "τ", ha="center", va="bottom",
                        color=COLOR_FG, fontsize=9)
    ax_t.text(0.97, 0.92, T("direct"), transform=ax_t.transAxes, ha="right",
              va="top", color=COLOR_PRIMARY, fontsize=8)
    ax_t.text(0.97, 0.80, T("delayed copy"), transform=ax_t.transAxes,
              ha="right", va="top", color=COLOR_SECONDARY, fontsize=8)

    ax_f = fig.add_subplot(gs[1, 1])
    _grid_axes(ax_f)
    f = np.logspace(np.log10(50.0), np.log10(8000.0), 500)
    ax_f.set_xscale("log")
    ax_f.set_xlim(50.0, 8000.0)
    ax_f.set_ylim(-16.0, 8.0)
    ax_f.set_xlabel(T("Frequency [Hz]"), fontsize=8)
    ax_f.set_ylabel(T("response [dB]"), fontsize=8)
    ax_f.tick_params(labelsize=7)
    (comb,) = ax_f.plot([], [], color=COLOR_PRIMARY, lw=1.8)
    (notch_dot,) = ax_f.plot([], [], marker="v", ms=7, ls="none",
                             color=COLOR_SECONDARY)
    notch_txt = ax_f.text(0.03, 0.97, "", transform=ax_f.transAxes,
                          ha="left", va="top", color=COLOR_SECONDARY,
                          fontsize=8.5, family="monospace")

    # Mic height trajectory: three plateaus (high, mid, on the floor).
    knots_t = (0.0, 2.6, 3.8, 6.2, 7.4, 10.0)
    knots_h = (1.5, 1.5, 0.6, 0.6, 0.02, 0.02)
    stages = ((2.0, T("high mic: dense comb")),
              (5.6, T("lower: notches move up")),
              (9.0, T("on the floor: copies merge — no comb in band")))
    sweep_s = (_ANIM_FRAMES - _ANIM_HOLD) / _ANIM_FPS

    def update(kf: int) -> tuple[Any, ...]:
        tc = min(kf / _ANIM_FPS, sweep_s)
        hm = float(np.interp(tc, knots_t, knots_h))
        r1 = float(np.hypot(xm - xs_, hm - hs))
        r2 = float(np.hypot(xm - xs_, hm + hs))
        tau = (r2 - r1) / c0
        # geometry: direct ray, floor bounce and the image-source ray
        xb = xs_ + (xm - xs_) * hs / (hs + hm)
        l_dir.set_data([xs_, xm], [hs, hm])
        l_ref.set_data([xs_, xb, xm], [hs, 0.0, hm])
        l_img.set_data([xs_, xm], [-hs, hm])
        mic_stand.set_data([xm, xm], [0.0, hm])
        mic_dot.set_data([xm], [hm])
        mic_lab.set_position((xm + 0.18, hm + 0.16))
        # three decimals keep the two readouts mutually consistent on
        # screen even for the near-zero floor geometry (Δ = c·τ)
        delta_txt.set_text(
            T(f"Δ = {r2 - r1:.3f} m   τ = {tau * 1e3:.3f} ms"))
        # time domain: two arrivals separated by tau
        t1, t2 = r1 / c0 * 1e3, r2 / c0 * 1e3
        stem_d.set_data([t1, t1], [0.0, 1.0])
        stem_r.set_data([t2, t2], [0.0, refl_g])
        tau_ann.xy = (t1, 0.9)
        tau_ann.set_position((t2, 0.9))
        tau_txt.set_position(((t1 + t2) / 2.0, 0.93))
        # hide both the label and the double arrow once the two arrivals
        # merge, otherwise the collapsed arrow reads as a stray glyph
        tau_txt.set_visible(t2 - t1 > 0.6)
        tau_ann.set_visible(t2 - t1 > 0.6)
        # frequency domain: |1 + g e^{-j 2 pi f tau}|
        h = np.abs(1.0 + refl_g * np.exp(-2j * np.pi * f * tau))
        comb.set_data(f, 20.0 * np.log10(h))
        f1n = 1.0 / (2.0 * tau) if tau > 0.0 else np.inf
        if f1n <= 8000.0:
            notch_dot.set_data([f1n], [20.0 * np.log10(1.0 - refl_g) + 1.2])
            notch_txt.set_text(T(f"first notch {f1n:.0f} Hz"))
        else:
            notch_dot.set_data([], [])
            notch_txt.set_text(T("first notch above 8 kHz"))
        stage = ""
        for t_s, s in stages:
            if tc >= t_s - 1.4:
                stage = s
        stage_txt.set_text(stage)
        return (l_dir, l_ref, l_img, mic_stand, mic_dot, mic_lab, delta_txt,
                stem_d, stem_r, tau_ann, tau_txt, comb, notch_dot,
                notch_txt, stage_txt)

    _render_clip(fig, update, output_dir, "anim_comb_filtering")
