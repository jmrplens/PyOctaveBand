#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The clip vocabulary, and the schematic clips drawn with it.

A schematic clip is a diagram that moves: a microphone, a loudspeaker, a fan
of wavefronts, a needle gauge, a signal-flow box that lights up as the value
reaches it. The drawing primitives are shared -- every clip that shows a
measurement chain draws the same microphone -- so they live here with the
clips that use them, ahead of the wave-field clips of :mod:`figures.fields`,
which borrow the same vocabulary to annotate a simulated field.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from phonometry._plot.common import format_frequency_axis

from .i18n import _LANG, _fmt_minus
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
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_QUATERNARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    FIELD_STROKE,
)

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.collections import FillBetweenPolyCollection
    from matplotlib.figure import Figure
    from matplotlib.patches import Arc, FancyArrowPatch, Rectangle
    from matplotlib.typing import ColorType
    from numpy.typing import NDArray


def _halo(linewidth: float = 2.4) -> list[Any]:
    """A background-coloured outline for text a moving artist crosses.

    The clips draw over their own captions: a magnifier handle, a reflection
    ray, a phase curve. Ordering the text above the artist keeps the glyphs
    whole but still lets a line of the same ink run through the word, so the
    labels at risk carry this halo as well, in whichever colour the theme
    paints the page.
    """
    from matplotlib import patheffects

    return [patheffects.withStroke(linewidth=linewidth, foreground=FIELD_STROKE)]


def _pending_ink(color: str) -> tuple[float, float, float]:
    """The colour a label waits in until the thing it names happens.

    Dimming by alpha alone only works on the light page: 35 % of a saturated
    ink over white is a readable pastel, while the same 35 % over black is
    near-black and the entry disappears (1.4:1 measured, against 2.0:1 for
    its light-theme twin). On the dark page the ink is washed toward the page
    text instead, which keeps the waiting entry a clear step below the lit one
    and still above 3:1.
    """
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(color)
    if _FILENAME_SUFFIX:
        return (0.33 * r + 0.27, 0.33 * g + 0.27, 0.33 * b + 0.27)
    return (0.35 * r + 0.65, 0.35 * g + 0.65, 0.35 * b + 0.65)


def _half_width(fig: Figure, ax: Axes, artist: Artist) -> float:
    """Half the rendered width of *artist*, in x data units of *ax*.

    Several clips centre a readout on a moving glyph, or anchor an annotation
    at a fixed data coordinate: both overflow the axes as soon as the Spanish
    string is longer than the English one it was placed for. Measuring the
    string that is actually about to be drawn is what lets them clamp to the
    panel instead of to a number that only holds in one language.
    """
    fig.canvas.draw()
    inv = ax.transData.inverted()
    (x0, _), (x1, _) = inv.transform(
        [(0.0, 0.0), (artist.get_window_extent().width, 0.0)]
    )
    return float(abs(x1 - x0)) / 2.0


def _grid_axes(ax: Axes) -> None:
    """Apply the standard documentation grid to a data axes."""
    ax.grid(True, color=COLOR_GRID, linestyle="--", alpha=0.5)


def _schematic_axes(
    ax: Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    equal: bool = False,
) -> None:
    """Turn *ax* into a bare drawing canvas (no ticks, frame or grid)."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if equal:
        ax.set_aspect("equal")
    ax.grid(False)
    ax.axis("off")


# --- shared schematic vocabulary (mics, gauges, flow boxes, arrows) --------


def _draw_mic(
    ax: Axes,
    x: float,
    y: float,
    *,
    direction: int = 1,
    size: float = 1.0,
    label: str = "",
    angle: float = 0.0,
) -> None:
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
        (x - body_w / 2, y - body_h / 2),
        body_w,
        body_h,
        boxstyle="round,pad=0.03",
        facecolor=COLOR_GRID,
        edgecolor=COLOR_FG,
        lw=1.2,
    )
    body.set_transform(tr)
    ax.add_patch(body)
    hx = x + direction * (body_w / 2 + head_r * 0.85)
    head = Circle((hx, y), head_r, facecolor="none", edgecolor=COLOR_FG, lw=1.2)
    head.set_transform(tr)
    ax.add_patch(head)
    for frac in (-0.45, 0.0, 0.45):
        half = head_r * float(np.sqrt(1.0 - frac * frac)) * 0.9
        (grille,) = ax.plot(
            [hx - half, hx + half], [y + frac * head_r] * 2, color=COLOR_FG, lw=0.6
        )
        grille.set_transform(tr)
    if label:
        rad = np.deg2rad(angle)
        dx, dy = 0.0, body_h * 1.7
        lx = x + dx * np.cos(rad) - dy * np.sin(rad)
        ly = y + dx * np.sin(rad) + dy * np.cos(rad)
        ax.text(lx, ly, label, ha="center", va="bottom", color=COLOR_FG, fontsize=11)


def _draw_speaker(
    ax: Axes, x: float, y: float, *, size: float = 1.0, direction: int = 1
) -> None:
    """A loudspeaker symbol: cabinet plus cone, radiating toward
    ``direction`` (+1 = +x, -1 = -x). ``(x, y)`` is the cone mouth centre.
    """
    from matplotlib.patches import Polygon, Rectangle

    d = float(direction)
    cone_l, cone_h = 0.42 * size, 0.55 * size
    box_l, box_h = 0.5 * size, 0.36 * size
    xb = x - d * cone_l
    ax.add_patch(
        Polygon(
            [(x, y - cone_h), (x, y + cone_h), (xb, y + box_h), (xb, y - box_h)],
            closed=True,
            facecolor=COLOR_GRID,
            edgecolor=COLOR_FG,
            lw=1.2,
        )
    )
    ax.add_patch(
        Rectangle(
            (min(xb, xb - d * box_l), y - box_h),
            box_l,
            2 * box_h,
            facecolor=COLOR_GRID,
            edgecolor=COLOR_FG,
            lw=1.2,
        )
    )


def _make_wavefronts(
    ax: Axes,
    x: float,
    y: float,
    color: str,
    n: int = 4,
    *,
    theta1: float = 0.0,
    theta2: float = 360.0,
    lw: float = 1.6,
) -> list[Arc]:
    """``n`` expanding wavefront arcs centred on ``(x, y)``, initially hidden.

    Drive them with :func:`_set_wavefronts`; ``theta1``/``theta2`` restrict
    the arc span (degrees) for sources radiating into a half or quarter space.
    """
    from matplotlib.patches import Arc

    arcs = []
    for _ in range(n):
        arc = Arc(
            (x, y), 0.01, 0.01, theta1=theta1, theta2=theta2, edgecolor=color, lw=lw
        )
        arc.set_visible(False)
        ax.add_patch(arc)
        arcs.append(arc)
    return arcs


def _set_wavefronts(
    arcs: list[Arc],
    radii: list[float],
    rmax: float,
    *,
    alpha: float = 0.9,
    color: ColorType | None = None,
) -> list[Arc]:
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


def _polyline_point(pts: NDArray[np.float64], frac: float) -> tuple[float, float]:
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


def _make_gauge(
    ax: Axes,
    cx: float,
    cy: float,
    r: float,
    label: str,
    color: str,
    lo: str = "",
    hi: str = "",
    end_dy: float = -0.12,
    end_dx: float = 1.12,
) -> dict[str, Any]:
    """A semicircular meter dial; move the needle with :func:`_set_gauge`.

    ``lo``/``hi`` are optional scale-endpoint labels (left and right end of
    the arc), so a reader can anchor the needle position to numbers, and
    ``end_dy`` is where they hang, in radii below the arc's own baseline.
    The default drops them far enough to clear the needle at full scale,
    which is what a label wide for its dial needs; a dial whose endpoint
    labels are two or three characters wide is better served by a shallower
    drop, close enough under the arc ends to read as belonging to that arc
    rather than floating in the gap to the next dial. ``end_dx`` is the
    horizontal hang, in radii from the dial centre: the default puts the
    labels just outside the arc ends, and a dial in a row packed so tight
    that an outside label would fall into the gap to its neighbour tucks
    them inside the arc instead (``end_dx`` < 1, with a drop deep enough
    to clear the needle lying flat along the baseline).
    """
    from matplotlib.patches import Arc

    ax.add_patch(
        Arc(
            (cx, cy), 2 * r, 2 * r, theta1=0.0, theta2=180.0, edgecolor=COLOR_FG, lw=1.4
        )
    )
    for frac in np.linspace(0.0, 1.0, 5):
        a = np.pi * (1.0 - float(frac))
        ax.plot(
            [cx + 0.86 * r * np.cos(a), cx + r * np.cos(a)],
            [cy + 0.86 * r * np.sin(a), cy + r * np.sin(a)],
            color=COLOR_FG,
            lw=0.8,
        )
    # Hung a little under the arc ends: at full scale the needle lies flat
    # along cy, and an endpoint label whose text is wide for the dial (the
    # Spanish "20 sonios" against the English "20 sone") otherwise reaches
    # back under the needle tip with a couple of pixels to spare.
    if lo:
        ax.text(
            cx - end_dx * r,
            cy + end_dy * r,
            lo,
            ha="center",
            va="top",
            color=COLOR_FG,
            fontsize=7,
        )
    if hi:
        ax.text(
            cx + end_dx * r,
            cy + end_dy * r,
            hi,
            ha="center",
            va="top",
            color=COLOR_FG,
            fontsize=7,
        )
    (needle,) = ax.plot(
        [cx, cx - 0.78 * r], [cy, cy], color=color, lw=2.4, solid_capstyle="round"
    )
    ax.plot([cx], [cy], marker="o", ms=4.5, color=color)
    ax.text(
        cx,
        cy - 0.30 * r,
        label,
        ha="center",
        va="top",
        color=color,
        fontsize=11,
        fontweight="bold",
    )
    value = ax.text(
        cx,
        cy - 0.72 * r,
        "",
        ha="center",
        va="top",
        color=COLOR_FG,
        fontsize=9,
        family="monospace",
    )
    return {"needle": needle, "value": value, "cx": cx, "cy": cy, "r": r}


def _set_gauge(gauge: dict[str, Any], frac: float, text: str) -> tuple[Any, Any]:
    """Point the needle at ``frac`` in [0, 1] (left to right) + set readout."""
    a = np.pi * (1.0 - float(np.clip(frac, 0.0, 1.0)))
    cx, cy, r = gauge["cx"], gauge["cy"], gauge["r"]
    gauge["needle"].set_data(
        [cx, cx + 0.78 * r * np.cos(a)], [cy, cy + 0.78 * r * np.sin(a)]
    )
    gauge["value"].set_text(text)
    return gauge["needle"], gauge["value"]


def _flow_box(
    ax: Axes, cx: float, cy: float, w: float, h: float, title: str
) -> dict[str, Any]:
    """A processing-pipeline box, dimmed until :func:`_light_box` lights it."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor="none",
        edgecolor=to_rgba(COLOR_FG, 0.35),
        lw=1.6,
    )
    ax.add_patch(box)
    title_t = ax.text(
        cx,
        cy + 0.22 * h,
        title,
        ha="center",
        va="center",
        color=COLOR_FG,
        alpha=0.55,
        fontsize=9,
    )
    value_t = ax.text(
        cx,
        cy - 0.22 * h,
        "",
        ha="center",
        va="center",
        color=COLOR_FG,
        fontsize=9,
        fontweight="bold",
        family="monospace",
    )
    return {"box": box, "title": title_t, "value": value_t}


def _light_box(box: dict[str, Any], value: str, color: str, fill: bool = False) -> None:
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


def _make_arrow(ax: Axes, color: str, scale: float = 14.0) -> FancyArrowPatch:
    """An updatable arrow patch; reposition it with ``set_positions``."""
    from matplotlib.patches import FancyArrowPatch

    arrow = FancyArrowPatch(
        (0.0, 0.0),
        (0.0, 0.0),
        arrowstyle="-|>",
        mutation_scale=scale,
        color=color,
        lw=2.0,
        shrinkA=0.0,
        shrinkB=0.0,
    )
    ax.add_patch(arrow)
    return arrow


def _draw_resistor(ax: Axes, x0: float, x1: float, y: float) -> None:
    """A zigzag resistor symbol on a horizontal wire from *x0* to *x1*."""
    n = 6
    xs = np.linspace(x0, x1, 2 * n + 1)
    ys = np.full(xs.size, y)
    ys[1:-1] = y + 0.22 * np.where(np.arange(1, 2 * n) % 2 == 1, 1.0, -1.0)
    ax.plot(xs, ys, color=COLOR_FG, lw=1.4)


def animate_time_weighting_ballistics(output_dir: str) -> None:
    """A tone burst through the IEC 61672-1 RC detector: the capacitor fills
    and drains while the F/S/I meter needles follow their own ballistics.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle

    from phonometry import filters

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
    i_ss = int(2.45 * fs)  # inside the burst, all detectors settled
    fast = filters.time_weighting(x, fs, mode="fast")
    slow = filters.time_weighting(x, fs, mode="slow")
    imp = filters.time_weighting(x, fs, mode="impulse")
    fast /= fast[i_ss]
    slow /= 0.5  # Slow has not settled by 2.45 s; use the tone MS
    imp /= imp[i_ss]
    # Slope of the Fast output for the charging/draining indicator, smoothed
    # over 50 ms so the residual 500 Hz detector ripple cannot flip its sign.
    kernel = np.ones(400) / 400.0
    dfast = np.gradient(np.convolve(fast, kernel, mode="same"), t)
    col_imp = "#7e57c2"

    fig = _anim_figure()
    fig.suptitle(
        T("Time-weighting ballistics (IEC 61672-1)"),
    )
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[1.0, 1.15])
    ax_s = fig.add_subplot(gs[:, 0])
    _schematic_axes(ax_s, (0.0, 10.0), (0.0, 10.0))
    ax_g = fig.add_subplot(gs[0, 1])
    _schematic_axes(ax_g, (0.0, 3.4), (-0.55, 0.68), equal=True)
    ax_t = fig.add_subplot(gs[1, 1])
    _grid_axes(ax_t)

    # --- schematic: input strip feeding the square-law + RC detector ------
    ax_s.text(
        5.0,
        10.0,
        T("RC exponential detector"),
        ha="center",
        va="top",
        color=COLOR_FG,
        fontsize=11,
        fontweight="bold",
    )
    # Display carrier slowed to ~4 Hz so the burst reads as a waveform, not
    # a filled block (the detectors run on the real 250 Hz burst).
    x_vis = np.where(on, np.sin(2 * np.pi * 4.0 * t), 0.0)
    ax_s.plot(
        0.6 + 8.8 * t[::4] / 4.0,
        8.2 + 0.7 * x_vis[::4],
        color=COLOR_PRIMARY,
        lw=1.0,
        alpha=0.9,
    )
    ax_s.text(
        9.4,
        9.05,
        T("input $x(t)$"),
        ha="right",
        va="bottom",
        color=COLOR_FG,
        fontsize=9,
    )
    (strip_cur,) = ax_s.plot([0.6, 0.6], [7.4, 9.0], color=COLOR_FG, lw=1.2, alpha=0.7)
    ax_s.annotate(
        "",
        xy=(0.6, 6.0),
        xytext=(0.6, 7.3),
        arrowprops={"arrowstyle": "-|>", "color": COLOR_FG, "lw": 1.2},
    )
    wire: dict[str, Any] = {"color": COLOR_FG, "lw": 1.4}
    ax_s.plot([0.6, 1.3], [5.6, 5.6], **wire)
    ax_s.add_patch(
        FancyBboxPatch(
            (1.3, 5.0),
            1.4,
            1.2,
            boxstyle="round,pad=0.04",
            facecolor="none",
            edgecolor=COLOR_FG,
            lw=1.4,
        )
    )
    ax_s.text(2.0, 5.6, "$x^2$", ha="center", va="center", color=COLOR_FG, fontsize=13)
    ax_s.text(
        2.0,
        4.55,
        T("square-law rectifier"),
        ha="center",
        va="top",
        color=COLOR_FG,
        fontsize=8.5,
    )
    ax_s.plot([2.7, 3.2], [5.6, 5.6], **wire)
    _draw_resistor(ax_s, 3.2, 5.0, 5.6)
    ax_s.text(4.1, 6.1, "$R$", ha="center", va="bottom", color=COLOR_FG, fontsize=11)
    ax_s.plot([5.0, 8.1], [5.6, 5.6], **wire)
    ax_s.plot([6.3], [5.6], marker="o", ms=4, color=COLOR_FG)
    ax_s.annotate(
        "",
        xy=(9.0, 5.6),
        xytext=(8.1, 5.6),
        arrowprops={"arrowstyle": "-|>", "color": COLOR_FG, "lw": 1.4},
    )
    ax_s.text(
        8.55,
        6.0,
        r"$10\,\log_{10}$",
        ha="center",
        va="bottom",
        color=COLOR_FG,
        fontsize=9,
    )
    ax_s.text(9.15, 5.6, "dB", ha="left", va="center", color=COLOR_FG, fontsize=10)
    # capacitor drawn as a tank so its state of charge is visible
    ax_s.plot([6.3, 6.3], [5.6, 4.7], **wire)
    ax_s.plot([5.6, 7.0], [4.7, 4.7], color=COLOR_FG, lw=2.2)
    ax_s.plot([5.6, 7.0], [3.2, 3.2], color=COLOR_FG, lw=2.2)
    ax_s.text(7.2, 3.95, "$C$", ha="left", va="center", color=COLOR_FG, fontsize=11)
    # tank walls, animated fill and a liquid-level line so partial charge
    # reads as partial even in a still frame
    for xw in (5.6, 7.0):
        ax_s.plot([xw, xw], [3.2, 4.7], color=COLOR_GRID, lw=1.0, ls=":")
    cap_fill = Rectangle(
        (5.62, 3.24),
        1.36,
        0.0,
        facecolor=COLOR_PRIMARY,
        alpha=0.8 if _FILENAME_SUFFIX else 0.55,
        edgecolor="none",
    )
    ax_s.add_patch(cap_fill)
    (cap_level,) = ax_s.plot([5.62, 6.98], [3.24, 3.24], color=COLOR_PRIMARY, lw=2.0)
    ax_s.plot([6.3, 6.3], [3.2, 2.62], **wire)
    for gw, gy in ((0.7, 2.62), (0.44, 2.48), (0.18, 2.34)):
        ax_s.plot([6.3 - gw / 2, 6.3 + gw / 2], [gy, gy], color=COLOR_FG, lw=1.4)
    ax_s.text(
        6.3,
        2.05,
        T("stored charge (Fast shown)"),
        ha="center",
        va="top",
        color=COLOR_FG,
        fontsize=8.5,
    )
    charge_arrow = _make_arrow(ax_s, COLOR_TERTIARY, scale=12.0)
    charge_txt = ax_s.text(
        4.95, 3.9, "", ha="right", va="center", color=COLOR_FG, fontsize=8.5
    )
    ax_s.text(
        5.0,
        1.3,
        T(r"$\tau = RC$ sets attack and decay"),
        ha="center",
        va="center",
        color=COLOR_FG,
        fontsize=9,
    )
    ax_s.text(
        5.0,
        0.6,
        "F 125 ms · S 1000 ms · I 35/1500 ms",
        ha="center",
        va="center",
        color=COLOR_FG,
        fontsize=8.5,
        alpha=0.85,
    )

    # --- meter gauges + response traces -----------------------------------
    # One shared 0..1.2 scale, spelled out on the F and S dials. The dials
    # sit so close that a label hung outside an arc end lands in the gap to
    # the neighbouring dial (that is where the lone "1.2" of the
    # first-dial-only design sat, nearer the S arc than the F one, reading
    # as labelling neither), so the endpoint labels tuck inside each
    # labelled arc instead, under the horizontal end ticks; the drop is
    # deep enough that the needle lying flat along the baseline at zero
    # clears the digits under its tip.
    gauges = [
        _make_gauge(
            ax_g,
            0.6,
            0.0,
            0.5,
            "F",
            COLOR_PRIMARY,
            lo="0",
            hi="1.2",
            end_dx=0.80,
            end_dy=-0.14,
        ),
        _make_gauge(
            ax_g,
            1.7,
            0.0,
            0.5,
            "S",
            COLOR_SECONDARY,
            lo="0",
            hi="1.2",
            end_dx=0.80,
            end_dy=-0.14,
        ),
        _make_gauge(ax_g, 2.8, 0.0, 0.5, "I", col_imp),
    ]
    ax_t.set_xlim(0.5, 4.0)
    ax_t.set_ylim(0, 1.25)
    ax_t.axvspan(1.0, 2.5, color=COLOR_GRID, alpha=0.4, lw=0)
    ax_t.text(
        1.75,
        1.19,
        T("tone burst"),
        ha="center",
        va="top",
        color=COLOR_FG,
        fontsize=8.5,
        alpha=0.8,
    )
    (l_f,) = ax_t.plot([], [], color=COLOR_PRIMARY, lw=2.0, label=T("Fast (125 ms)"))
    (l_s,) = ax_t.plot([], [], color=COLOR_SECONDARY, lw=2.0, label=T("Slow (1000 ms)"))
    (l_i,) = ax_t.plot(
        [], [], color=col_imp, lw=1.7, ls="-.", label=T("Impulse (35 ms / 1.5 s)")
    )
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
            charge_arrow.set_color(COLOR_TERTIARY if charging else COLOR_SECONDARY)
            charge_txt.set_text(T("charging") if charging else T("draining"))
        else:
            charge_arrow.set_visible(False)
            charge_txt.set_text("")
        arts: list[Any] = [strip_cur, cap_fill, cap_level, charge_arrow, charge_txt]
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
    OR -> LD -> P -> KI decision chain lights up once the onset is found.
    """
    from matplotlib.patches import Ellipse

    from phonometry import environment

    T = _translate_str
    fs = 500
    t = np.linspace(0, 3.0, int(fs * 3.0), endpoint=False)
    ls, le = 55.0, 85.0  # start and end level of the onset, dB
    t0, rise = 1.0, 0.15  # onset at 1.0 s, lasting 150 ms
    laf = np.full_like(t, ls)
    ramp = (t >= t0) & (t < t0 + rise)
    laf[ramp] = ls + (le - ls) * 0.5 * (1 - np.cos(np.pi * (t[ramp] - t0) / rise))
    after = t >= t0 + rise
    laf[after] = ls + (le - ls) * np.exp(-(t[after] - t0 - rise) / 0.6)
    grad = np.gradient(laf, t)
    onset_rate = (le - ls) / rise  # 200 dB/s
    level_diff = le - ls  # 30 dB
    prom = float(environment.predicted_prominence(onset_rate, level_diff))
    ki = float(environment.impulse_adjustment(prom))
    is_onset = grad > 10.0  # clauses 4.5-4.7

    fig = _anim_figure()
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 0.9])
    ax = fig.add_subplot(gs[0])
    _grid_axes(ax)
    ax.set_xlim(0.55, 3.0)
    ax.set_ylim(42, 100)
    ax.plot(
        t,
        laf,
        color=COLOR_PRIMARY,
        lw=1.8,
        label=T(r"$L_\mathrm{AF}$ (A-weighted, Fast)"),
    )
    (hot,) = ax.plot(
        [],
        [],
        color=COLOR_SECONDARY,
        lw=4.0,
        solid_capstyle="round",
        label=T("onset (> 10 dB/s)"),
    )
    rx, ry = 0.13, 6.5
    lens = Ellipse(
        (0.7, ls),
        width=2 * rx,
        height=2 * ry,
        facecolor="none",
        edgecolor=COLOR_FG,
        lw=2.2,
        zorder=5,
    )
    ax.add_patch(lens)
    (handle,) = ax.plot(
        [], [], color=COLOR_FG, lw=3.2, solid_capstyle="round", zorder=5
    )
    (tangent,) = ax.plot([], [], color=COLOR_FG, lw=1.6, ls="--", zorder=6)
    lens_txt = ax.text(
        0.7, ls, "", ha="center", va="bottom", color=COLOR_FG, fontsize=9, zorder=6
    )
    # The magnifier sweeps the panel in data coordinates and its handle dips
    # into the bottom-left corner where this caption is anchored, so the
    # caption is drawn above it and haloed: the handle is the same ink as the
    # text, and crossing it whole reads as a struck-through word.
    ax.text(
        0.02,
        0.05,
        T("detector: onset when $dL/dt > 10$ dB/s"),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=COLOR_FG,
        fontsize=9,
        zorder=8,
        path_effects=_halo(3.0),
    )
    ax.set_title(T("Impulse onset detection (NT ACOU 112)"))
    ax.set_xlabel(T("Time [s]"))
    ax.set_ylabel(T(r"A-weighted level $L_\mathrm{AF}$ [dB]"), fontsize=9)
    ax.legend(loc="upper right", fontsize=8)

    ax_b = fig.add_subplot(gs[1])
    _schematic_axes(ax_b, (0.0, 10.0), (0.0, 2.3))
    titles = [T("onset rate"), T("level difference"), T("prominence"), T("adjustment")]
    boxes = [
        _flow_box(ax_b, cx, 1.35, 1.95, 1.35, title)
        for cx, title in zip((1.5, 3.9, 6.3, 8.7), titles, strict=True)
    ]
    for xa in (2.5, 4.9, 7.3):
        ax_b.annotate(
            "",
            xy=(xa + 0.42, 1.35),
            xytext=(xa, 1.35),
            arrowprops={"arrowstyle": "-|>", "color": COLOR_GRID, "lw": 1.6},
        )
    verdict = ax_b.text(
        5.0,
        0.28,
        "",
        ha="center",
        va="center",
        color=COLOR_SECONDARY,
        fontsize=10,
        fontweight="bold",
    )
    values = (
        f"OR = {onset_rate:.0f} dB/s",
        f"LD = {level_diff:.0f} dB",
        f"$P$ = {prom:.1f}",
        f"KI = {ki:.1f} dB",
    )

    # How far the readout reaches either side of the lens it is centred on,
    # measured on the widest gradient the sweep ever shows.
    lens_txt.set_text(T(f"$dL/dt$ = {onset_rate:.0f} dB/s"))
    txt_half = _half_width(fig, ax, lens_txt)
    lens_txt.set_text("")

    # Nonuniform sweep: slow motion while the magnifier crosses the onset.
    # The sweep ends _ANIM_HOLD frames early; np.interp clamps past the last
    # knot, so the lit OR -> LD -> P -> KI chain and the verdict hold ~2 s.
    end_k = float(_ANIM_FRAMES - 1 - _ANIM_HOLD)
    knots_k = (0.0, 0.22 * end_k, 0.52 * end_k, end_k)
    knots_t = (0.62, 0.985, 1.19, 3.0)

    def update(k: int) -> tuple[Any, ...]:
        tc = float(np.interp(k, knots_k, knots_t))
        # Park the magnifier just short of either edge so its lens and handle
        # stay fully inside the panel at both ends of the sweep (tc runs from
        # the left xlim to 3.0 s, the right one). The handle reaches
        # ``rx * 1.9`` past the centre and the lens ``rx``, so the margins are
        # a fraction wider than that. The decision chain below still keys off
        # the true tc, so only the glyph position is clamped.
        tc_view = min(max(tc, 0.55 + rx * 1.15), 3.0 - rx * 2.15)
        i = max(0, min(t.size - 1, round(tc_view * fs)))
        y0, g = float(laf[i]), float(grad[i])
        detecting = g > 10.0
        color = COLOR_SECONDARY if detecting else COLOR_FG
        lens.set_center((tc_view, y0))
        lens.set_edgecolor(color)
        dxl = rx * 0.72
        if abs(g) * dxl > ry * 0.72:
            dxl = ry * 0.72 / abs(g)
        tangent.set_data([tc_view - dxl, tc_view + dxl], [y0 - g * dxl, y0 + g * dxl])
        tangent.set_color(color)
        handle.set_data(
            [tc_view + rx * 0.75, tc_view + rx * 1.9], [y0 - ry * 0.75, y0 - ry * 1.9]
        )
        # Centred on the lens, but never far enough left to have the spine and
        # the y tick labels run through it.
        x_txt = min(max(tc_view, 0.55 + txt_half + 0.02), 3.0 - txt_half - 0.02)
        lens_txt.set_position((x_txt, y0 + ry * 1.35))
        # A gradient of -0.3 dB/s rounds to "-0", a sign in front of a zero.
        lens_txt.set_text(
            T(f"$dL/dt$ = {_fmt_minus(g if round(g) else 0.0, '.0f')} dB/s")
        )
        lens_txt.set_color(color)
        hot_m = (t <= tc) & is_onset
        hot.set_data(t[hot_m], laf[hot_m])
        arts: list[Any] = [lens, handle, tangent, lens_txt, hot]
        for j, (box, val) in enumerate(zip(boxes, values, strict=True)):
            if tc >= 1.3 + 0.4 * j:
                _light_box(
                    box,
                    T(val),
                    COLOR_SECONDARY if j == 3 else COLOR_PRIMARY,
                    fill=j == 3,
                )
            else:
                _dim_box(box)
            arts += [box["box"], box["title"], box["value"]]
        verdict.set_text(T("add KI to the rating level") if tc >= 2.6 else "")
        arts.append(verdict)
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_onset_detection")


def animate_instantaneous_intensity(output_dir: str) -> None:
    """A p-p probe with p/u phasors: the instantaneous intensity arrow flips
    while its running average settles to net flow (active) or zero (reactive).
    """
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
    fig.suptitle(
        T(r"Two-microphone p-p probe: instantaneous intensity $p\cdot u$"),
    )
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    dial_c, dial_r = (1.55, 3.05), 1.25
    i_axis_y, i_scale = 1.05, 2.35
    panels: list[dict[str, Any]] = []
    for col, (title, phi, caption) in enumerate(cases):
        ax_s = fig.add_subplot(gs[0, col])
        _schematic_axes(ax_s, (0.0, 10.0), (0.0, 4.9), equal=True)
        ax_s.set_title(title, fontsize=10.5)
        ax_s.add_patch(
            Circle(dial_c, dial_r, facecolor="none", edgecolor=COLOR_GRID, lw=1.2)
        )
        p_ph = _make_arrow(ax_s, COLOR_PRIMARY, scale=11.0)
        u_ph = _make_arrow(ax_s, COLOR_TERTIARY, scale=11.0)
        ax_s.text(
            dial_c[0] - dial_r - 0.15,
            dial_c[1] + dial_r - 0.15,
            "$p$",
            color=COLOR_PRIMARY,
            fontsize=11,
            ha="right",
            va="center",
            fontweight="bold",
        )
        ax_s.text(
            dial_c[0] - dial_r - 0.15,
            dial_c[1] - dial_r + 0.15,
            "$u$",
            color=COLOR_TERTIARY,
            fontsize=11,
            ha="right",
            va="center",
            fontweight="bold",
        )
        # Tucked up under the phasor dial so it clears the intensity
        # number-line below (the longer Spanish caption otherwise crowds it).
        ax_s.text(
            dial_c[0],
            1.72,
            caption,
            ha="center",
            va="top",
            color=COLOR_FG,
            fontsize=8.0,
        )
        _draw_mic(ax_s, 4.6, 3.4, direction=1, size=1.05, label="$p_1$")
        _draw_mic(ax_s, 7.4, 3.4, direction=-1, size=1.05, label="$p_2$")
        ax_s.annotate(
            "",
            xy=(6.85, 2.6),
            xytext=(5.15, 2.6),
            arrowprops={"arrowstyle": "<->", "color": COLOR_FG, "lw": 1.0},
        )
        ax_s.text(
            6.0,
            2.38,
            T(r"spacer $\Delta r$"),
            ha="center",
            va="top",
            color=COLOR_FG,
            fontsize=8.5,
        )
        ax_s.plot([3.4, 8.6], [i_axis_y, i_axis_y], color=COLOR_GRID, lw=1.0, ls="--")
        ax_s.plot(
            [6.0, 6.0],
            [i_axis_y - 0.15, i_axis_y + 0.15],
            color=COLOR_FG,
            lw=1.0,
            alpha=0.7,
        )
        ax_s.text(
            6.0,
            i_axis_y + 0.22,
            "0",
            ha="center",
            va="bottom",
            color=COLOR_FG,
            fontsize=8,
            alpha=0.8,
        )
        ax_s.text(
            3.15,
            i_axis_y,
            "−",
            ha="right",
            va="center",
            color=COLOR_FG,
            fontsize=10,
            alpha=0.8,
        )
        ax_s.text(
            8.85,
            i_axis_y,
            "+",
            ha="left",
            va="center",
            color=COLOR_FG,
            fontsize=10,
            alpha=0.8,
        )
        # Anchored 0.4 to the right of the number-line's left end: the
        # Spanish phase caption under the dial is wider than the English
        # one and its right edge otherwise runs to within a few pixels of
        # this label (they share a vertical band).
        ax_s.text(
            3.8,
            1.55,
            T(r"$I(t) = p\cdot u$"),
            ha="left",
            va="bottom",
            color=COLOR_SECONDARY,
            fontsize=9,
        )
        i_arrow = _make_arrow(ax_s, COLOR_SECONDARY, scale=16.0)
        (mean_marker,) = ax_s.plot([], [], marker="^", ms=7, color=COLOR_FG)
        mean_lab = ax_s.text(
            6.0,
            0.42,
            r"$\langle I\rangle$",
            ha="center",
            va="top",
            color=COLOR_FG,
            fontsize=8.5,
        )

        ax_tr = fig.add_subplot(gs[1, col])
        _grid_axes(ax_tr)
        ax_tr.set_xlim(0, 3.0)
        ax_tr.set_ylim(-1.15, 1.15)
        p_sig = np.cos(w * t)
        u_sig = np.cos(w * t - phi)
        ax_tr.plot(
            t, p_sig, color=COLOR_PRIMARY, alpha=0.55, lw=1.1, label=T("pressure $p$")
        )
        # Dashed, because in the progressive panel p and u are the same
        # normalised curve: drawn solid, the velocity covers the pressure
        # pixel for pixel and the legend announces a blue trace that is
        # nowhere on the page.
        ax_tr.plot(
            t,
            u_sig,
            color=COLOR_TERTIARY,
            alpha=0.55,
            lw=1.3,
            ls=(0, (5, 3)),
            label=T("velocity $u$"),
        )
        (iline,) = ax_tr.plot(
            [], [], color=COLOR_SECONDARY, lw=2.0, label=T(r"intensity $p\cdot u$")
        )
        mline = ax_tr.axhline(0.0, color=COLOR_FG, ls="--", lw=1.1, alpha=0.7)
        txt = ax_tr.text(
            0.5,
            0.02,
            "",
            transform=ax_tr.transAxes,
            ha="center",
            va="bottom",
            family="monospace",
            fontsize=10,
            color=COLOR_FG,
        )
        ax_tr.set_xlabel(T("Time [s]"))
        if col == 0:
            ax_tr.set_ylabel(T("amplitude (normalized)"), fontsize=9)
        # The traces reach the corner the legend sits in, and the running mean
        # draws a dashed rule straight across it: at the default frame alpha
        # both bleed through and streak the rows they cross.
        ax_tr.legend(loc="upper right", fontsize=7.5, framealpha=1.0)
        panels.append(
            {
                "ax": ax_tr,
                "phi": phi,
                "I": p_sig * u_sig,
                "p_ph": p_ph,
                "u_ph": u_ph,
                "i_arrow": i_arrow,
                "mean_marker": mean_marker,
                "mean_lab": mean_lab,
                "iline": iline,
                "mline": mline,
                "txt": txt,
                "fill": None,
            }
        )

    # Six carrier periods sweep by, then the settled averages (net flow vs
    # zero) hold so the active/reactive contrast can be read.
    sweep = _ANIM_FRAMES - _ANIM_HOLD

    def update(k: int) -> tuple[Any, ...]:
        tc = 3.0 * min(k, sweep - 1) / (sweep - 1)
        idx = max(1, int(np.searchsorted(t, tc)))
        # Average over whole periods once one is complete, so the reactive
        # mean pins to exactly zero instead of hovering near it.
        n_per = np.floor(tc * 2.0)
        idx_mean = (
            idx if n_per < 1 else max(1, int(np.searchsorted(t, float(n_per) / 2.0)))
        )
        ph = w * tc
        cx, cy = dial_c
        arts: list[Any] = []
        for pn in panels:
            pv = float(np.cos(ph))
            uv = float(np.cos(ph - pn["phi"]))
            pn["p_ph"].set_positions(
                (cx, cy),
                (cx + 1.05 * dial_r * np.cos(ph), cy + 1.05 * dial_r * np.sin(ph)),
            )
            pn["u_ph"].set_positions(
                (cx, cy),
                (
                    cx + 0.88 * dial_r * np.cos(ph - pn["phi"]),
                    cy + 0.88 * dial_r * np.sin(ph - pn["phi"]),
                ),
            )
            ival = pv * uv
            tip = 6.0 + i_scale * ival
            if abs(tip - 6.0) < 1e-3:
                tip = 6.0 + 1e-3
            pn["i_arrow"].set_positions((6.0, i_axis_y), (tip, i_axis_y))
            mean = float(np.mean(pn["I"][:idx_mean]))
            if abs(mean) < 5e-3:
                mean = 0.0  # avoid a distracting "-0.00" readout
            xm = 6.0 + i_scale * mean
            pn["mean_marker"].set_data([xm], [i_axis_y - 0.22])
            pn["mean_lab"].set_position((xm, i_axis_y - 0.5))
            pn["iline"].set_data(t[:idx], pn["I"][:idx])
            if pn["fill"] is not None:
                pn["fill"].remove()
            pn["fill"] = pn["ax"].fill_between(
                t[:idx], 0.0, pn["I"][:idx], color=COLOR_SECONDARY, alpha=0.22
            )
            pn["mline"].set_ydata([mean, mean])
            pn["txt"].set_text(
                T(rf"$\langle p\cdot u\rangle$ = {_fmt_minus(mean, '+.2f')}")
            )
            arts += [
                pn["p_ph"],
                pn["u_ph"],
                pn["i_arrow"],
                pn["mean_marker"],
                pn["mean_lab"],
                pn["iline"],
                pn["fill"],
                pn["mline"],
                pn["txt"],
            ]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_instantaneous_intensity")


def animate_schroeder(output_dir: str) -> None:
    """Backward integration of p²(t): the tail energy fills up on one axis
    while the decay curve and its T20/T30 fits emerge on a companion axis.
    """
    from matplotlib.patches import Patch

    from phonometry import room
    from phonometry.room._shared import onset_index
    from phonometry.room.acoustics import _T20_RANGE, _T30_RANGE

    T = _translate_str
    fs, reverb_t = 48000, 1.2
    rng = np.random.default_rng(2026)
    t = np.arange(int(2.0 * fs)) / fs
    ir = rng.standard_normal(t.size) * np.exp(
        -6.9077 * t / reverb_t
    ) + rng.standard_normal(t.size) * 10.0 ** (-45.0 / 20.0)
    time, level = room.decay_curve(ir, fs)
    res = room.room_parameters(ir, fs, limits=None)
    t20, t30 = float(res.t20[0]), float(res.t30[0])
    p2 = ir.astype(np.float64) ** 2
    p2 = p2[onset_index(p2) :]
    t_raw = np.arange(p2.size) / fs
    raw_db = 10.0 * np.log10(np.maximum(p2, p2.max() * 1e-12) / p2.max())

    # Regression lines drawn once the sweep finishes: slope -60/T over each
    # evaluation range, extended to the -60 dB crossing at t = T.
    def _fit(rng_db: tuple[float, float]) -> tuple[float, float] | None:
        mask = (level <= -rng_db[0]) & (level >= -rng_db[1])
        if int(mask.sum()) < 2:
            return None
        slope, intercept = np.polyfit(time[mask], level[mask], 1)
        if slope >= 0.0:  # a non-decaying fit has no meaningful -60 dB crossing
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
        fits.append(
            (color, style, key, (-intercept / slope, (-60.0 - intercept) / slope))
        )
    tmax = float(time.max())
    xmax = max([tmax, *(f[3][1] for f in fits)]) * 1.03
    # Tail-energy fraction E(t)/E(0) on the onset-trimmed squared IR, and a
    # display-decimated copy of the raw level for the mechanism panel.
    cum = np.cumsum(p2[::-1])[::-1]
    e_frac = cum / cum[0]
    ds = slice(None, None, 8)

    fig = _anim_figure()
    fig.suptitle(
        T("Schroeder backward integration (ISO 3382)"),
    )
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.3])
    ax_e = fig.add_subplot(gs[0])
    _grid_axes(ax_e)
    ax_e.set_xlim(0, xmax)
    ax_e.set_ylim(-65, 3)
    ax_e.tick_params(labelbottom=False)
    sweep_max = float(t_raw[-1])  # start the front at the very end of p²
    fill_alpha = 0.5 if _FILENAME_SUFFIX else 0.35
    ax_e.plot(t_raw[ds], raw_db[ds], color="gray", alpha=0.4, lw=0.6)
    e_fill: dict[str, FillBetweenPolyCollection | None] = {"art": None}
    front_e = ax_e.axvline(sweep_max, color=COLOR_FG, lw=1.3, alpha=0.55)
    # Sits well below the upper-right legend (a higher anchor ran under it).
    front_txt = ax_e.text(
        sweep_max,
        -22.0,
        T("← integrate from the tail"),
        ha="left",
        va="top",
        color=COLOR_FG,
        fontsize=9,
    )
    e_txt = ax_e.text(
        0.02,
        0.06,
        "",
        transform=ax_e.transAxes,
        ha="left",
        va="bottom",
        family="monospace",
        fontsize=10,
        color=COLOR_FG,
    )
    ax_e.set_ylabel(T("Level [dB]"), fontsize=9)
    ax_e.legend(
        handles=[
            Patch(
                facecolor="gray", alpha=0.4, label=T("squared impulse response $p^2$")
            ),
            Patch(
                facecolor=COLOR_SECONDARY,
                alpha=fill_alpha,
                label=T("tail energy") + r"  $E(t)=\int_t^{\infty}p^2\,d\tau$",
            ),
        ],
        loc="upper right",
        fontsize=8.5,
    )

    ax_d = fig.add_subplot(gs[1], sharex=ax_e)
    _grid_axes(ax_d)
    ax_d.set_ylim(-65, 3)
    (curve,) = ax_d.plot(
        [], [], color=COLOR_PRIMARY, lw=2.4, label=T("Schroeder decay curve")
    )
    # The fit lines only exist in the closing frames, so they are announced
    # by the color-matched T20/T30 readouts instead of a premature legend.
    fit_lines = []
    for color, style, _key, span in fits:
        (fl,) = ax_d.plot([], [], color=color, ls=style, lw=1.7)
        fit_lines.append((fl, span))
    front_d = ax_d.axvline(sweep_max, color=COLOR_FG, lw=1.3, alpha=0.55)
    ax_d.text(
        0.02,
        0.08,
        r"$L(t) = 10\,\log_{10}\,E(t)\,/\,E(0)$",
        transform=ax_d.transAxes,
        ha="left",
        va="bottom",
        color=COLOR_FG,
        fontsize=10,
    )
    ann_fits = [
        ax_d.text(
            0.56,
            0.16,
            "",
            transform=ax_d.transAxes,
            ha="left",
            va="bottom",
            family="monospace",
            fontsize=11,
            color=COLOR_SECONDARY,
        ),
        ax_d.text(
            0.56,
            0.05,
            "",
            transform=ax_d.transAxes,
            ha="left",
            va="bottom",
            family="monospace",
            fontsize=11,
            color=COLOR_TERTIARY,
        ),
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
        e_fill["art"] = ax_e.fill_between(
            t_raw[ds][mr],
            -65.0,
            raw_db[ds][mr],
            color=COLOR_SECONDARY,
            alpha=fill_alpha,
            lw=0,
        )
        idx = min(int(np.searchsorted(t_raw, xf)), e_frac.size - 1)
        e_txt.set_text(T(f"remaining energy: {100.0 * e_frac[idx]:.1f} %"))
        front_txt.set_position((min(xf, xmax * 0.72) + 0.012 * xmax, -22.0))
        front_txt.set_visible(k < reveal)
        arts: list[Any] = [
            curve,
            front_e,
            front_d,
            e_fill["art"],
            e_txt,
            front_txt,
            *ann_fits,
        ]
        if k >= reveal:
            for (fl, (t_lo, t_hi)), ann, name, val in zip(
                fit_lines, ann_fits, ("T20", "T30"), (t20, t30), strict=False
            ):
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
    path label lights up as its pulse re-radiates into the receiving room.
    """
    from matplotlib.patches import Circle, Rectangle

    T = _translate_str
    fig = _anim_figure()
    fig.suptitle(
        T("Flanking transmission paths (EN 12354-1)"),
    )
    ax = fig.add_subplot()
    _schematic_axes(ax, (0.0, 14.2), (0.0, 7.6), equal=True)

    # Cross-section: source room | separating wall | receiving room, with a
    # continuous floor slab running through the junction.
    wall_x0, wall_x1 = 6.6, 7.1
    floor_y0, floor_y1 = 1.0, 1.6
    ax.add_patch(
        Rectangle(
            (wall_x0, floor_y1),
            wall_x1 - wall_x0,
            5.0,
            facecolor=COLOR_GRID,
            edgecolor=COLOR_FG,
            lw=1.2,
        )
    )
    ax.add_patch(
        Rectangle(
            (0.7, floor_y0),
            12.8,
            floor_y1 - floor_y0,
            facecolor=COLOR_GRID,
            edgecolor=COLOR_FG,
            lw=1.2,
        )
    )
    for xr in (0.7, 13.5):
        ax.plot([xr, xr], [floor_y1, 6.6], color=COLOR_FG, lw=1.4)
    ax.plot([0.7, 13.5], [6.6, 6.6], color=COLOR_FG, lw=1.4)
    ax.text(
        3.6, 6.35, T("source room"), ha="center", va="top", color=COLOR_FG, fontsize=10
    )
    ax.text(
        10.3,
        6.35,
        T("receiving room"),
        ha="center",
        va="top",
        color=COLOR_FG,
        fontsize=10,
    )
    # Element letters: capital = source side, lowercase = receiving side
    for xt, yt, s in (
        (wall_x0 - 0.25, 4.3, "D"),
        (wall_x1 + 0.25, 4.3, "d"),
        (3.6, 1.3, "F"),
        (10.3, 1.3, "f"),
    ):
        ax.text(
            xt,
            yt,
            s,
            ha="center",
            va="center",
            color=COLOR_FG,
            fontsize=12,
            fontweight="bold",
            fontstyle="italic",
        )
    _draw_speaker(ax, 2.6, 3.6, size=1.35)

    src = (2.9, 3.6)
    col_df = "#7e57c2"
    # Path polylines (source -> element -> junction/wall -> radiator) with a
    # per-path arrival strength standing in for the junction attenuation Kij.
    paths: list[dict[str, Any]] = [
        {
            "key": "Dd",
            "color": COLOR_PRIMARY,
            "arrive": 0.62,
            "pts": [src, (wall_x0, 3.6), (wall_x1, 3.6), (8.6, 3.6)],
            "rad": (wall_x1, 3.6),
            "span": (-70.0, 70.0),
            "desc": T("direct, wall to wall"),
        },
        {
            "key": "Ff",
            "color": COLOR_SECONDARY,
            "arrive": 0.40,
            "pts": [src, (4.4, floor_y1), (4.4, 1.3), (10.0, 1.3), (10.0, floor_y1)],
            "rad": (10.0, floor_y1),
            "span": (20.0, 160.0),
            "desc": T("floor to floor"),
        },
        {
            "key": "Fd",
            "color": COLOR_TERTIARY,
            "arrive": 0.30,
            "pts": [
                src,
                (4.9, floor_y1),
                (4.9, 1.3),
                (6.85, 1.45),
                (6.85, 3.0),
                (wall_x1, 3.0),
                (8.4, 3.0),
            ],
            "rad": (wall_x1, 3.0),
            "span": (-70.0, 70.0),
            "desc": T("floor to wall"),
        },
        {
            "key": "Df",
            "color": col_df,
            "arrive": 0.30,
            "pts": [
                src,
                (wall_x0, 2.4),
                (6.85, 2.3),
                (6.85, 1.3),
                (9.4, 1.3),
                (9.4, floor_y1),
            ],
            "rad": (9.4, floor_y1),
            "span": (20.0, 160.0),
            "desc": T("wall to floor"),
        },
    ]
    junction = (6.85, 1.3)
    for pn in paths:
        pts = np.asarray(pn["pts"])
        ax.plot(pts[:, 0], pts[:, 1], color=pn["color"], lw=1.0, ls=":", alpha=0.35)
        (trail,) = ax.plot([], [], color=pn["color"], lw=2.0, alpha=0.85)
        pulse = Circle(
            (0.0, 0.0), 0.16, facecolor=pn["color"], edgecolor="none", visible=False
        )
        ax.add_patch(pulse)
        pn["trail"], pn["pulse"] = trail, pulse
        rad_x, rad_y = pn["rad"]
        pn["arcs"] = _make_wavefronts(
            ax,
            rad_x,
            rad_y,
            pn["color"],
            n=3,
            theta1=pn["span"][0],
            theta2=pn["span"][1],
            lw=1.4,
        )
        pn["pts_a"] = pts
    junc_txt = ax.text(
        junction[0] + 0.3,
        junction[1] - 0.55,
        "",
        ha="left",
        va="top",
        color=COLOR_FG,
        fontsize=8.5,
    )
    # Path legend column: lights up as each pulse arrives. The longer Spanish
    # descriptions start a little further left so they clear the right wall.
    labels = []
    lab_x = 10.15 if _LANG == "en" else 9.6
    for j, pn in enumerate(paths):
        yt = 5.6 - 0.55 * j
        lab = ax.text(
            lab_x,
            yt,
            f"{pn['key']} — {pn['desc']}",
            ha="left",
            va="center",
            color=_pending_ink(pn["color"]),
            fontsize=8.5,
            fontweight="bold",
        )
        labels.append(lab)
    verdict = ax.text(
        7.05,
        0.35,
        "",
        ha="center",
        va="center",
        color=COLOR_FG,
        fontsize=10,
        fontweight="bold",
    )

    travel = 2.1  # seconds a pulse takes over its path
    starts = (0.4, 2.6, 4.8, 7.0)  # launch time of each pulse [s]
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
                tr = np.array(
                    [_polyline_point(pts, f) for f in np.linspace(0.0, frac, n_tr)]
                )
                pn["trail"].set_data(tr[:, 0], tr[:, 1])
                pn["trail"].set_alpha(0.3 + 0.5 * scale)
                _set_wavefronts(pn["arcs"], [0.0] * 3, 1.0)
                if pn["key"] != "Dd" and 0.35 < frac < 0.75:
                    junc_txt.set_text(T(r"junction: $K_{ij}$ attenuates each transfer"))
            else:
                pn["pulse"].set_visible(False)
                age = (tc - t0) - travel
                radii = [0.55 * (age - 0.35 * i) for i in range(3)]
                _set_wavefronts(pn["arcs"], radii, 1.8, alpha=0.4 + 0.5 * pn["arrive"])
                lab.set_color(pn["color"])
            arts += [pn["pulse"], pn["trail"], *pn["arcs"], lab]
        verdict.set_text(
            T("$R\u2032_\\mathrm{w}$ sums all paths — always below the wall alone")
            if tc >= 9.4
            else ""
        )
        arts += [junc_txt, verdict]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_flanking_paths")


def animate_intensity_scan_power(output_dir: str) -> None:
    """ISO 9614-2 sound power: a p-p probe traces the serpentine scan over
    the top face of the measurement box while the normal-intensity arrows
    appear behind it, and the partial powers of the five faces accumulate
    into the L_W meter.
    """
    from matplotlib.patches import Circle, Polygon

    T = _translate_str

    # Parallel projection of the measurement box (w x d x h metres).
    bw, bd, bh = 5.2, 2.6, 3.2
    ox, oy = 0.52, 0.30  # projected offset per metre of depth
    x0, y0 = 1.0, 1.1

    def proj(u: float, v: float, w: float) -> tuple[float, float]:
        return (x0 + u + ox * v, y0 + w + oy * v)

    fig = _anim_figure()
    fig.suptitle(
        T("Intensity scanning over a box surface (ISO 9614-2)"),
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0])
    ax = fig.add_subplot(gs[0])
    _schematic_axes(ax, (0.0, 9.4), (0.0, 6.4), equal=True)

    # Box edges (hidden rear edges dashed)
    edges = [
        ((0, 0, 0), (bw, 0, 0)),
        ((bw, 0, 0), (bw, bd, 0)),
        ((bw, bd, 0), (0, bd, 0)),
        ((0, bd, 0), (0, 0, 0)),
        ((0, 0, bh), (bw, 0, bh)),
        ((bw, 0, bh), (bw, bd, bh)),
        ((bw, bd, bh), (0, bd, bh)),
        ((0, bd, bh), (0, 0, bh)),
        ((0, 0, 0), (0, 0, bh)),
        ((bw, 0, 0), (bw, 0, bh)),
        ((bw, bd, 0), (bw, bd, bh)),
        ((0, bd, 0), (0, bd, bh)),
    ]
    for a, b in edges:
        xa, ya = proj(*a)
        xb, yb = proj(*b)
        ax.plot([xa, xb], [ya, yb], color=COLOR_FG, lw=1.1, alpha=0.65)
    # The machine under test inside the box (a plain block on the floor)
    mx, md = bw / 2, bd / 2
    m_w, m_d, m_h = 1.5, 1.0, 1.0
    base = [
        (mx - m_w / 2, md - m_d / 2),
        (mx + m_w / 2, md - m_d / 2),
        (mx + m_w / 2, md + m_d / 2),
        (mx - m_w / 2, md + m_d / 2),
    ]
    top = [proj(u, v, m_h) for u, v in base]
    front = [
        proj(base[0][0], base[0][1], 0.0),
        proj(base[1][0], base[1][1], 0.0),
        proj(base[1][0], base[1][1], m_h),
        proj(base[0][0], base[0][1], m_h),
    ]
    side = [
        proj(base[1][0], base[1][1], 0.0),
        proj(base[2][0], base[2][1], 0.0),
        proj(base[2][0], base[2][1], m_h),
        proj(base[1][0], base[1][1], m_h),
    ]
    # Opaque theme-blended face tints so the box edges behind the machine do
    # not show through it (or through its label). The tint alone does not do
    # that: a patch is drawn below a line at matplotlib's default zorder, so
    # the box's rear-bottom edge ran straight through the machine and
    # underlined "source" along its baseline. Order the faces above the
    # wireframe, and the wavefronts above the faces they leave from.
    from matplotlib.colors import to_rgb

    bg = np.asarray(to_rgb(plt.rcParams["figure.facecolor"]))
    grid_rgb = np.asarray(to_rgb(COLOR_GRID))
    for poly, al in ((front, 0.75), (side, 0.55), (top, 0.9)):
        tint = tuple(al * grid_rgb + (1.0 - al) * bg)
        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor=tint,
                edgecolor=COLOR_FG,
                lw=1.0,
                zorder=2.2,
            )
        )
    ax.text(
        *proj(mx, md, 0.45),
        T("source"),
        ha="center",
        va="center",
        color=COLOR_FG,
        fontsize=8.5,
    )
    src_arcs = _make_wavefronts(
        ax, *proj(mx, md, m_h), COLOR_TERTIARY, n=3, theta1=15.0, theta2=165.0, lw=1.2
    )
    for arc in src_arcs:
        arc.set_zorder(2.4)

    # Serpentine scan over the TOP face: passes along u at stepped v.
    n_pass = 4
    vs = np.linspace(0.25, bd - 0.25, n_pass)
    serp: list[tuple[float, float]] = []
    for i, v in enumerate(vs):
        us = (0.3, bw - 0.3) if i % 2 == 0 else (bw - 0.3, 0.3)
        serp += [(us[0], float(v)), (us[1], float(v))]
    serp_a = np.asarray(serp)
    (scan_line,) = ax.plot([], [], color=COLOR_PRIMARY, lw=1.6, alpha=0.8)
    probe = Circle(
        (0.0, 0.0),
        0.11,
        facecolor=COLOR_PRIMARY,
        edgecolor=COLOR_FG,
        lw=1.0,
        zorder=6,
        visible=False,
    )
    ax.add_patch(probe)
    probe_lab = ax.text(
        0.0,
        0.0,
        T("p-p probe"),
        ha="left",
        va="bottom",
        color=COLOR_FG,
        fontsize=8.5,
        visible=False,
    )
    # Normal-intensity arrows on the top face: longer where the source is
    # closer (I·n ~ cos(theta) / r^2), revealed as the probe passes.
    gu, gv = np.meshgrid(np.linspace(0.55, bw - 0.55, 6), np.linspace(0.4, bd - 0.4, 3))
    serp_fine = np.array(
        [_polyline_point(serp_a, fr) for fr in np.linspace(0.0, 1.0, 300)]
    )
    arrows: list[dict[str, Any]] = []
    for u, v in zip(gu.ravel(), gv.ravel(), strict=True):
        r2 = (u - mx) ** 2 + (v - md) ** 2 + (bh - m_h) ** 2
        inten = (bh - m_h) / r2**1.5  # cos(theta) / r^2, unnormalised
        # the serpentine fraction at which the probe passes this grid point
        near = int(np.argmin(np.hypot(serp_fine[:, 0] - u, serp_fine[:, 1] - v)))
        arrows.append(
            {"i": float(inten), "frac": near / 299.0, "u": float(u), "v": float(v)}
        )
    i_max = max(ad["i"] for ad in arrows)
    for ad in arrows:
        xa, ya = proj(ad["u"], ad["v"], bh)
        ln = 0.28 + 0.75 * ad["i"] / i_max
        arr = _make_arrow(ax, COLOR_SECONDARY, scale=9.0)
        arr.set_positions((xa, ya), (xa, ya + ln))
        arr.set_visible(False)
        ad["arrow"] = arr
    ax.text(
        0.15,
        6.25,
        T(r"normal intensity $I\cdot n$ on the surface"),
        ha="left",
        va="top",
        color=COLOR_SECONDARY,
        fontsize=9,
    )

    # Right column: per-face partial powers into the L_W meter.
    ax_m = fig.add_subplot(gs[1])
    _schematic_axes(ax_m, (0.0, 4.4), (0.0, 6.4))
    ax_m.text(
        2.2,
        6.25,
        T("partial powers"),
        ha="center",
        va="top",
        color=COLOR_FG,
        fontsize=10,
        fontweight="bold",
    )
    # Face shares of the total power for a source at the box centre floor:
    # top sees the most, the four sides split the rest (plausible shares).
    faces = [
        ("top", T("top"), 0.34),
        ("front", T("front"), 0.20),
        ("back", T("back"), 0.20),
        ("left", T("left"), 0.13),
        ("right", T("right"), 0.13),
    ]
    p_total = 1.6e-3  # W -> L_W = 92.0 dB
    boxes = {}
    for j, (key, lab, _share) in enumerate(faces):
        boxes[key] = _flow_box(ax_m, 1.1, 5.35 - 0.95 * j, 1.9, 0.8, lab)
    # The dial starts below the level the first scanned strip already carries
    # (66 dB one frame in), so the needle is never parked against the left
    # stop while the readout under it prints a number the scale does not
    # reach; the settled 92.0 dB still lands near the top of the arc.
    gauge_lo, gauge_hi = 65.0, 95.0
    gauge = _make_gauge(
        ax_m,
        3.3,
        1.1,
        0.85,
        "$L_W$",
        COLOR_SECONDARY,
        lo=f"{gauge_lo:.0f}",
        hi=f"{gauge_hi:.0f}",
    )
    ax_m.text(
        3.3,
        2.75,
        r"$P = \sum_i \int I{\cdot}n\ dS_i$",
        ha="center",
        va="bottom",
        color=COLOR_FG,
        fontsize=10,
    )
    # verdict rides under the box drawing, where the full width is free
    verdict = ax.text(
        4.7,
        0.35,
        "",
        ha="center",
        va="center",
        color=COLOR_SECONDARY,
        fontsize=10,
        fontweight="bold",
    )

    scan_end, face_step = 6.2, 0.85  # top-face scan, then one face per step
    sweep_s = (_ANIM_FRAMES - _ANIM_HOLD) / _ANIM_FPS

    def update(kf: int) -> tuple[Any, ...]:
        tc = min(kf / _ANIM_FPS, sweep_s)
        arts: list[Any] = []
        # source breathing wavefronts above the machine
        age = (tc * 0.5) % 0.6
        _set_wavefronts(
            src_arcs, [0.9 * (age + 0.2 * i) for i in range(3)], 0.9, alpha=0.5
        )
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
        tr = np.array(
            [_polyline_point(serp_a, f) for f in np.linspace(0.0, frac, n_tr)]
        )
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
                got = share * float(
                    np.clip((tc - scan_end - (j - 1) * face_step) / face_step, 0.0, 1.0)
                )
            acc += got
            if got >= share - 1e-9:
                _light_box(
                    boxes[key],
                    T(f"{got * p_total * 1e3:.2f} mW"),
                    COLOR_PRIMARY,
                    fill=j == 0,
                )
            elif got > 0.0:
                _light_box(
                    boxes[key], T(f"{got * p_total * 1e3:.2f} mW"), COLOR_PRIMARY
                )
            else:
                _dim_box(boxes[key])
            b = boxes[key]
            arts += [b["box"], b["title"], b["value"]]
        if acc > 0.0:
            lw_now = 10.0 * np.log10(acc * p_total / 1e-12)
            arts += _set_gauge(
                gauge,
                (lw_now - gauge_lo) / (gauge_hi - gauge_lo),
                T(f"{lw_now:.1f} dB"),
            )
        verdict.set_text(
            T("any enclosing surface gives the same $P$") if acc >= 0.999 else ""
        )
        arts.append(verdict)
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_intensity_scan_power")


def animate_sweep_deconvolution(output_dir: str) -> None:
    """ISO 18233 swept-sine measurement: the exponential sweep crosses a
    drawn room while its spectrogram builds, then the inverse filter
    collapses the whole record into the impulse response.
    """
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    from scipy.signal import fftconvolve, spectrogram

    from phonometry import room

    T = _translate_str
    fs = 8000
    sweep_len, f1, f2 = 2.5, 60.0, 3500.0
    sweep = room.sweep_signal(fs, f1, f2, sweep_len)
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
    system += (
        0.05
        * rng.standard_normal(ir_len)
        * np.exp(-6.9077 * tail_t / 0.5)
        * (tail_t > 0.012)
    )
    recorded = fftconvolve(sweep, system)
    rec_dur = recorded.size / fs
    _freqs, times, sxx = spectrogram(recorded, fs, nperseg=512, noverlap=384)
    sxx_db = 10.0 * np.log10(np.maximum(sxx, sxx.max() * 1e-8) / sxx.max())
    ir = room.impulse_response(
        recorded, sweep, fs, method="spectral", length=int(0.2 * fs)
    )
    t_ir = np.arange(ir.ir.size) / fs * 1e3
    ir_n = ir.ir / np.max(np.abs(ir.ir))

    fig = _anim_figure()
    fig.suptitle(
        T("Sweep measurement and deconvolution (ISO 18233)"),
    )
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.15])
    ax_r = fig.add_subplot(gs[0, :])
    # The band below zero is the caption's: at the drawn scale an 8 pt line is
    # taller than the 0.3 units between the room's floor and the old axes
    # bottom, so the caption sat on the floor line and lost its ascenders.
    _schematic_axes(ax_r, (0.0, 18.0), (-0.35, 3.4), equal=True)
    room_box = Rectangle(
        (1.0, 0.3), 16.0, 2.8, facecolor="none", edgecolor=COLOR_FG, lw=1.4
    )
    ax_r.add_patch(room_box)
    _draw_speaker(ax_r, 2.6, 1.7, size=1.1)
    _draw_mic(ax_r, 14.6, 1.7, direction=-1, size=0.85, label=T("mic"))
    # direct and reflected ray paths
    ax_r.plot([2.8, 13.9], [1.7, 1.7], color=COLOR_FG, lw=1.0, ls="--", alpha=0.5)
    for ry in (0.3, 3.1):
        ax_r.plot(
            [2.8, 8.3, 13.9],
            [1.7, ry, 1.75],
            color=COLOR_FG,
            lw=0.9,
            ls=":",
            alpha=0.45,
        )
    ax_r.text(
        8.3,
        -0.30,
        T("direct + reflections"),
        ha="center",
        va="bottom",
        color=COLOR_FG,
        fontsize=8,
    )
    arcs = _make_wavefronts(
        ax_r, 2.8, 1.7, COLOR_PRIMARY, n=4, theta1=-75.0, theta2=75.0
    )
    for arc in arcs:  # wavefronts stay inside the drawn room
        arc.set_clip_path(room_box)
    # The wavefronts are clipped to the room and sweep every part of it, this
    # corner included, so the readout carries the page under it and the arc
    # passes behind the pill instead of through the digits.
    freq_txt = ax_r.text(
        2.0,
        2.55,
        "",
        ha="left",
        va="bottom",
        color=COLOR_FG,
        fontsize=9,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": plt.rcParams["figure.facecolor"],
            "edgecolor": "none",
            "alpha": 0.92,
        },
    )
    note = ax_r.text(
        9.0,
        0.8,
        "",
        ha="center",
        va="center",
        color=COLOR_FG,
        fontsize=9.5,
        fontstyle="italic",
        visible=False,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": plt.rcParams["figure.facecolor"],
            "edgecolor": "none",
            "alpha": 0.85,
        },
    )

    ax_s = fig.add_subplot(gs[1, 0])
    ax_s.grid(False)
    disp = np.full_like(sxx_db, sxx_db.min())
    im = ax_s.imshow(
        disp,
        origin="lower",
        aspect="auto",
        extent=(0.0, float(times[-1]), 0.0, fs / 2 / 1e3),
        cmap="magma",
        norm=Normalize(-60.0, 0.0),
        interpolation="bilinear",
    )
    ax_s.set_xlabel(T("Time [s]"), fontsize=8)
    ax_s.set_ylabel(T("Frequency [kHz]"), fontsize=8)
    ax_s.tick_params(labelsize=7)
    ax_s.set_title(T("recorded sweep (spectrogram)"), fontsize=9)
    ridge_txt = ax_s.text(
        0.97,
        0.08,
        "",
        transform=ax_s.transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=8,
    )

    ax_i = fig.add_subplot(gs[1, 1])
    _grid_axes(ax_i)
    ax_i.set_xlim(0.0, 200.0)
    ax_i.set_ylim(-1.05, 1.05)
    (ir_line,) = ax_i.plot([], [], color=COLOR_PRIMARY, lw=1.0)
    ax_i.set_xlabel(T("Time [ms]"), fontsize=8)
    ax_i.set_title(T("impulse response"), fontsize=9)
    ax_i.tick_params(labelsize=7)
    tap_marks = [
        ax_i.annotate(
            "",
            xy=(ms, g * 1.0),
            xytext=(ms + 22.0, min(g + 0.35, 0.95)),
            fontsize=7.5,
            color=COLOR_FG,
            ha="left",
            arrowprops={"arrowstyle": "-", "color": COLOR_FG, "lw": 0.7},
            visible=False,
        )
        for ms, g in ((12.0, 1.0), (90.0, 0.55))
    ]
    tap_marks[0].set_text(T("direct"))
    tap_marks[1].set_text(T("reflections"))
    deconv = _flow_box(ax_i, 140.0, -0.72, 85.0, 0.42, T("⊛ inverse filter"))

    play_end, dec_t = 6.5, 7.6  # sweep playback, deconvolution moment
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
            _set_wavefronts(
                arcs,
                [11.5 * ((age + 0.25 * i) % 1.0) for i in range(4)],
                11.5,
                color=col,
            )
            freq_txt.set_text(f"$f$ = {f_now:5.0f} Hz")
        else:
            _set_wavefronts(arcs, [0.0] * 4, 1.0)
            freq_txt.set_text("")
        arts += [*arcs, freq_txt]
        # spectrogram builds in sync with the playback
        n_col = int(np.searchsorted(times, t_sig))
        disp[:, :n_col] = sxx_db[:, :n_col]
        im.set_data(disp)
        ridge_txt.set_text(T("delayed copies = reflections") if tc >= 3.4 else "")
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
                    T(
                        "same information, different domain: sweep ⊛ inverse"
                        " filter = impulse response"
                    )
                )
                note.set_visible(True)
        else:
            _dim_box(deconv)
            ir_line.set_data([], [])
            note.set_text("")
            note.set_visible(False)
        arts += [
            ir_line,
            deconv["box"],
            deconv["title"],
            deconv["value"],
            note,
            *tap_marks,
        ]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_sweep_deconvolution")


def animate_specific_loudness(output_dir: str) -> None:
    """ISO 532-1 specific loudness: the N'(z) pattern of a 1 kHz narrowband
    sound builds along the Bark axis as the band level steps up, and the
    area under the pattern integrates to the total loudness in sone.
    """
    T = _translate_str
    from phonometry import psychoacoustics

    bands = [
        25,
        31.5,
        40,
        50,
        63,
        80,
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
        5000,
        6300,
        8000,
        10000,
        12500,
    ]
    i_1k = bands.index(1000)
    levels = np.arange(40, 86)
    patterns: dict[int, Any] = {}
    totals: dict[int, float] = {}
    for lv in levels:
        third = [-60.0] * 28
        third[i_1k] = float(lv)
        res = psychoacoustics.loudness_zwicker_from_spectrum(third, field="free")
        patterns[int(lv)] = res.specific
        totals[int(lv)] = float(res.loudness)
    z = np.arange(1, 241) * 0.1

    fig = _anim_figure()
    fig.suptitle(
        T(r"Specific loudness $N^{\prime}(z)$ and its integral (ISO 532-1)"),
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0])
    ax = fig.add_subplot(gs[0])
    _grid_axes(ax)
    ax.set_xlim(0.0, 24.0)
    # headroom above the 85 dB pattern (max N' ~ 5.4 sone/Bark)
    ax.set_ylim(0.0, 6.0)
    ax.set_xlabel(T("Critical-band rate $z$ [Bark]"))
    ax.set_ylabel(T(r"Specific loudness $N^{\prime}$ [sone/Bark]"), fontsize=9)
    (line,) = ax.plot([], [], color=COLOR_PRIMARY, lw=2.2)
    fill: dict[str, FillBetweenPolyCollection | None] = {"art": None}
    ax.axvline(8.5, color=COLOR_FG, lw=0.9, ls=":", alpha=0.6)
    ax.text(
        8.75,
        5.85,
        T("1 kHz ≈ 8.5 Bark"),
        ha="left",
        va="top",
        color=COLOR_FG,
        fontsize=8,
    )
    # Anchored by its right edge, a short step inside the right spine: placed
    # by its left edge, the Spanish string ran 380 px past the axes, and since
    # a text artist counts toward the axes' tight bounding box, the frame it
    # appeared on also shrank the whole plot column by 16 %.
    spread = ax.annotate(
        T("upward spread of masking"),
        xy=(13.0, 1.1),
        xytext=(23.6, 3.2),
        ha="right",
        fontsize=8.5,
        color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0},
        visible=False,
    )
    level_txt = ax.text(
        0.02,
        0.965,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=COLOR_FG,
        fontsize=11,
        family="monospace",
    )

    ax_m = fig.add_subplot(gs[1])
    _schematic_axes(ax_m, (0.0, 4.0), (0.0, 8.6))
    _draw_speaker(ax_m, 0.55, 7.7, size=0.7)
    ax_m.text(
        1.15,
        7.7,
        T("1 kHz narrowband"),
        ha="left",
        va="center",
        color=COLOR_FG,
        fontsize=8.5,
    )
    gauge = _make_gauge(
        ax_m, 2.0, 4.6, 1.05, "$N$", COLOR_SECONDARY, lo="0", hi=T("20 sone")
    )
    ax_m.text(
        2.0,
        6.5,
        r"$N = \int_0^{24} N'(z)\, dz$",
        ha="center",
        va="bottom",
        color=COLOR_FG,
        fontsize=11,
    )
    steps = [45, 65, 85]
    step_boxes = [
        _flow_box(ax_m, 2.0, 2.6 - 0.95 * j, 3.4, 0.8, T(f"{lv} dB"))
        for j, lv in enumerate(steps)
    ]

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
        fill["art"] = ax.fill_between(
            z[m], 0.0, spec[m], color=COLOR_PRIMARY, alpha=0.3, lw=0
        )
        n_part = float(np.trapezoid(spec[m], z[m])) if m.any() else 0.0
        n_show = totals[lv] if z_lim >= 24.0 else n_part
        level_txt.set_text(f"$L$ = {lv} dB")
        spread.set_visible(lv >= 80)
        arts: list[Any] = [line, fill["art"], level_txt, spread]
        arts += _set_gauge(gauge, n_show / 20.0, T(f"{n_show:.1f} sone"))
        for j, (lv_s, box) in enumerate(zip(steps, step_boxes, strict=True)):
            done = tc >= (int_t1, 5.2, 8.9)[j]
            if done:
                _light_box(
                    box,
                    T(f"$N$ = {totals[lv_s]:.1f} sone"),
                    COLOR_SECONDARY if j == 2 else COLOR_PRIMARY,
                    fill=j == 2,
                )
            else:
                _dim_box(box)
            arts += [box["box"], box["title"], box["value"]]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_specific_loudness")


def animate_power_two_rooms(output_dir: str) -> None:
    """The same source in an anechoic room (ISO 3745 free field) and in a
    reverberation room (ISO 3741 diffuse build-up): both microphone
    pressures differ, both routes converge to the same sound power L_W.
    """
    from matplotlib.patches import Circle, Polygon, Rectangle

    T = _translate_str
    lw_true = 92.0
    # Projected mic-ring radius (anechoic). Drawn a touch smaller than it once
    # was: at 2.05 the two lowest dots of the ring sat on the top of the
    # caption below it, and the dot at 75 deg touched the reading above it.
    r_mic = 1.95
    lp_free = 77.5  # L_W - 10 log10(4 pi 1.5^2)
    lp_diff = 86.0  # L_W - 10 log10 V + 10 log10 T + 14

    fig = _anim_figure()
    fig.suptitle(
        T("One source, two rooms, one sound power"),
    )
    gs = fig.add_gridspec(2, 2, height_ratios=[2.35, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    _schematic_axes(ax_a, (0.0, 8.0), (0.0, 6.6), equal=True)
    ax_r = fig.add_subplot(gs[0, 1])
    _schematic_axes(ax_r, (0.0, 8.0), (0.0, 6.6), equal=True)
    ax_b = fig.add_subplot(gs[1, :])
    _schematic_axes(ax_b, (0.0, 16.0), (0.0, 2.6))

    # --- anechoic room: wedges on every wall, mic ring, free field -------
    ax_a.set_title(
        T("Anechoic room (ISO 3745)"),
        fontsize=10,
    )
    ax_a.add_patch(
        Rectangle((0.6, 0.4), 6.8, 5.6, facecolor="none", edgecolor=COLOR_FG, lw=1.4)
    )
    wedge = 0.42
    for xw in np.arange(0.85, 7.3, 0.55):
        ax_a.add_patch(
            Polygon(
                [(xw - 0.22, 0.4), (xw + 0.22, 0.4), (xw, 0.4 + wedge)],
                closed=True,
                facecolor=COLOR_GRID,
                edgecolor=COLOR_FG,
                lw=0.6,
            )
        )
        ax_a.add_patch(
            Polygon(
                [(xw - 0.22, 6.0), (xw + 0.22, 6.0), (xw, 6.0 - wedge)],
                closed=True,
                facecolor=COLOR_GRID,
                edgecolor=COLOR_FG,
                lw=0.6,
            )
        )
    for yw in np.arange(0.85, 5.9, 0.55):
        ax_a.add_patch(
            Polygon(
                [(0.6, yw - 0.22), (0.6, yw + 0.22), (0.6 + wedge, yw)],
                closed=True,
                facecolor=COLOR_GRID,
                edgecolor=COLOR_FG,
                lw=0.6,
            )
        )
        ax_a.add_patch(
            Polygon(
                [(7.4, yw - 0.22), (7.4, yw + 0.22), (7.4 - wedge, yw)],
                closed=True,
                facecolor=COLOR_GRID,
                edgecolor=COLOR_FG,
                lw=0.6,
            )
        )
    ca = (4.0, 3.2)
    _draw_speaker(ax_a, ca[0] - 0.05, ca[1], size=0.85)
    # offset by 15 deg so no dot lands on the caption lines at the bottom
    mic_angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False) + np.pi / 12
    for ang in mic_angles:
        ax_a.plot(
            [ca[0] + r_mic * np.cos(ang)],
            [ca[1] + r_mic * np.sin(ang)],
            marker="o",
            ms=4,
            color=COLOR_PRIMARY,
        )
    ax_a.text(
        ca[0],
        ca[1] - r_mic + 0.14,
        T("microphone sphere, $r$"),
        ha="center",
        va="bottom",
        color=COLOR_PRIMARY,
        fontsize=8,
    )
    arcs_a = _make_wavefronts(ax_a, *ca, COLOR_TERTIARY, n=4)
    note_a = ax_a.text(
        4.0,
        0.92,
        T("direct sound only — no reflections"),
        ha="center",
        va="bottom",
        color=COLOR_FG,
        fontsize=8,
        alpha=0.0,
        path_effects=_halo(),
    )
    lp_a = ax_a.text(
        6.9,
        5.5,
        "",
        ha="right",
        va="top",
        color=COLOR_FG,
        fontsize=9,
        family="monospace",
    )

    # --- reverberation room: bare walls, diffuse build-up, one mic path --
    ax_r.set_title(
        T("Reverberation room (ISO 3741)"),
        fontsize=10,
    )
    # animated fill (the diffuse level building up) + a fixed outline
    room_patch = Rectangle(
        (0.6, 0.4), 6.8, 5.6, facecolor=COLOR_PRIMARY, edgecolor="none", alpha=0.0
    )
    ax_r.add_patch(room_patch)
    ax_r.add_patch(
        Rectangle((0.6, 0.4), 6.8, 5.6, facecolor="none", edgecolor=COLOR_FG, lw=1.4)
    )
    # a tilted diffuser panel hanging in the volume
    ax_r.plot([1.3, 2.7], [4.7, 5.3], color=COLOR_FG, lw=2.4, alpha=0.8)
    cr = (1.7, 1.3)
    _draw_speaker(ax_r, cr[0] - 0.05, cr[1], size=0.85)
    # bouncing ray trails (specular folding inside the rectangle)
    rays: list[dict[str, Any]] = [
        {
            "v": (2.9, 1.9),
            "trail": ax_r.plot([], [], color=COLOR_TERTIARY, lw=1.1, alpha=0.75)[0],
        },
        {
            "v": (2.2, -2.6),
            "trail": ax_r.plot([], [], color=COLOR_TERTIARY, lw=1.1, alpha=0.75)[0],
        },
    ]
    mic_path_c, mic_path_r = (4.4, 3.4), 1.35
    ax_r.add_patch(
        Circle(
            mic_path_c,
            mic_path_r,
            facecolor="none",
            edgecolor=COLOR_PRIMARY,
            lw=0.9,
            ls="--",
            alpha=0.8,
        )
    )
    (mic_dot,) = ax_r.plot([], [], marker="o", ms=6, color=COLOR_PRIMARY)
    # Both captions sit in the path of the bouncing rays, which fold across
    # the whole room, so they are haloed rather than merely drawn above them.
    ax_r.text(
        mic_path_c[0],
        mic_path_c[1] - mic_path_r - 0.28,
        T("rotating microphone"),
        ha="center",
        va="top",
        color=COLOR_PRIMARY,
        fontsize=8,
        path_effects=_halo(),
    )
    note_r = ax_r.text(
        4.0,
        0.62,
        T("reflections build a diffuse field"),
        ha="center",
        va="bottom",
        color=COLOR_FG,
        fontsize=8,
        alpha=0.0,
        path_effects=_halo(),
    )
    lp_r = ax_r.text(
        6.9,
        5.5,
        "",
        ha="right",
        va="top",
        color=COLOR_FG,
        fontsize=9,
        family="monospace",
    )

    def _fold(p: float, lo: float, hi: float) -> float:
        span = hi - lo
        q = (p - lo) % (2.0 * span)
        return lo + (q if q <= span else 2.0 * span - q)

    # --- bottom strip: the two formulas converge on one L_W --------------
    box_a = _flow_box(
        ax_b, 3.4, 1.55, 6.2, 1.15, r"$L_W = \bar{L}_p + 10\,\log_{10}(4\pi r^2/S_0)$"
    )
    box_r = _flow_box(
        ax_b,
        12.6,
        1.55,
        6.2,
        1.15,
        r"$L_W = \bar{L}_p + 10\,\log_{10} V - 10\,\log_{10} T - 14$",
    )
    lw_box = _flow_box(ax_b, 8.0, 1.3, 2.6, 1.3, "$L_W$")
    arr_a = _make_arrow(ax_b, COLOR_SECONDARY, scale=13.0)
    arr_r = _make_arrow(ax_b, COLOR_SECONDARY, scale=13.0)
    for arr in (arr_a, arr_r):
        arr.set_visible(False)
    verdict = ax_b.text(
        8.0,
        0.12,
        "",
        ha="center",
        va="bottom",
        color=COLOR_SECONDARY,
        fontsize=10,
        fontweight="bold",
    )

    # Centred in the room, the longer Spanish caption reaches back into the
    # loudspeaker cone; slide it right by whatever its own width demands.
    # Measured with the figure complete, so the axes is at its final size.
    note_r.set_position((max(4.0, 2.05 + _half_width(fig, ax_r, note_r)), 0.62))

    sweep_s = (_ANIM_FRAMES - _ANIM_HOLD) / _ANIM_FPS
    t_meter, t_form, t_conv = 4.5, 7.0, 9.4

    def update(kf: int) -> tuple[Any, ...]:
        tc = min(kf / _ANIM_FPS, sweep_s)
        arts: list[Any] = []
        # free field: wavefronts die before the wedges
        age = (tc * 0.55) % 1.0
        _set_wavefronts(
            arcs_a, [2.9 * ((age + 0.25 * i) % 1.0) for i in range(4)], 2.9, alpha=0.8
        )
        arts += arcs_a
        note_a.set_alpha(min(tc / 2.0, 1.0) * 0.9)
        # diffuse field: rays bounce, the background level rises
        build = float(np.clip(tc / 4.0, 0.0, 1.0))
        room_patch.set_alpha(0.16 * (1.0 - np.exp(-3.0 * build)))
        for ray in rays:
            ts = np.linspace(max(0.0, tc - 0.9), tc, 24)
            xs = [_fold(cr[0] + ray["v"][0] * s, 0.7, 7.3) for s in ts]
            ys = [_fold(cr[1] + ray["v"][1] * s, 0.5, 5.9) for s in ts]
            ray["trail"].set_data(xs, ys)
            arts.append(ray["trail"])
        ang = 2.0 * np.pi * 0.12 * tc
        mic_dot.set_data(
            [mic_path_c[0] + mic_path_r * np.cos(ang)],
            [mic_path_c[1] + mic_path_r * np.sin(ang)],
        )
        note_r.set_alpha(min(tc / 2.0, 1.0) * 0.9)
        arts += [room_patch, mic_dot, note_a, note_r]
        # microphone readings appear, then each formula computes L_W
        if tc >= t_meter:
            lp_a.set_text(T(f"mean $L_p$ = {lp_free:.1f} dB"))
            lp_r.set_text(T(f"mean $L_p$ = {lp_diff:.1f} dB"))
        else:
            lp_a.set_text("")
            lp_r.set_text("")
        # Mathtext subscript for the sound-power symbol; localise the decimal
        # comma by hand because ``$`` disables the ``_translate_str`` comma pass.
        lw_val = (
            f"{lw_true:.1f}" if _LANG == "en" else f"{lw_true:.1f}".replace(".", ",")
        )
        for box, on in ((box_a, tc >= t_form), (box_r, tc >= t_form + 0.6)):
            if on:
                _light_box(box, f"$L_W$ = {lw_val} dB", COLOR_PRIMARY)
            else:
                _dim_box(box)
            arts += [box["box"], box["title"], box["value"]]
        if tc >= t_conv:
            _light_box(lw_box, T(f"{lw_true:.1f} dB"), COLOR_SECONDARY, fill=True)
            arr_a.set_positions((6.6, 1.55), (7.3, 1.4))
            arr_r.set_positions((9.4, 1.55), (8.7, 1.4))
            arr_a.set_visible(True)
            arr_r.set_visible(True)
            verdict.set_text(T("the room changes $L_p$, not the source power"))
        else:
            _dim_box(lw_box)
            arr_a.set_visible(False)
            arr_r.set_visible(False)
            verdict.set_text("")
        arts += [
            lp_a,
            lp_r,
            lw_box["box"],
            lw_box["title"],
            lw_box["value"],
            arr_a,
            arr_r,
            verdict,
        ]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_power_two_rooms")


def animate_comb_filtering(output_dir: str) -> None:
    """Direct sound plus one floor reflection at a microphone: as the mic
    height changes, the delayed copy shifts and the comb filter in the
    frequency response moves with it, which is why measurement position matters
    near reflecting surfaces.
    """
    T = _translate_str
    c0 = 343.0
    xs_, hs = 0.6, 1.5  # source position (x, height)
    xm = 4.2  # mic x position
    refl_g = 0.8  # floor reflection factor

    fig = _anim_figure()
    fig.suptitle(
        T("Comb filtering from a single reflection"),
    )
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    _schematic_axes(ax, (0.0, 5.6), (-2.3, 2.6), equal=True)
    ax.axhline(0.0, color=COLOR_FG, lw=2.0)
    ax.fill_between([0.0, 5.6], -2.3, 0.0, color=COLOR_GRID, alpha=0.35, lw=0)
    ax.text(
        5.5,
        -0.34,
        T("reflecting floor"),
        ha="right",
        va="top",
        color=COLOR_FG,
        fontsize=8.5,
    )
    _draw_speaker(ax, xs_, hs, size=0.8)
    ax.plot([xs_ - 0.25, xs_ - 0.25], [0.0, hs - 0.28], color=COLOR_FG, lw=1.2)
    # image source below the floor
    _draw_speaker(ax, xs_, -hs, size=0.8)
    for art in ax.patches[-2:]:
        art.set_alpha(0.3)
    ax.text(
        xs_ + 0.4,
        -hs,
        T("image source"),
        ha="left",
        va="center",
        color=COLOR_FG,
        fontsize=8.5,
        alpha=0.75,
    )
    (l_dir,) = ax.plot([], [], color=COLOR_PRIMARY, lw=1.8)
    (l_ref,) = ax.plot([], [], color=COLOR_SECONDARY, lw=1.6, ls="--")
    (l_img,) = ax.plot([], [], color=COLOR_SECONDARY, lw=1.0, ls=":", alpha=0.5)
    (mic_stand,) = ax.plot([], [], color=COLOR_FG, lw=1.2)
    (mic_dot,) = ax.plot(
        [],
        [],
        marker="o",
        ms=8,
        color=COLOR_GRID,
        markeredgecolor=COLOR_FG,
        markeredgewidth=1.2,
    )
    mic_lab = ax.text(
        xm + 0.18, 0.0, T("mic"), ha="left", va="center", color=COLOR_FG, fontsize=8.5
    )
    delta_txt = ax.text(
        0.15,
        2.45,
        "",
        ha="left",
        va="top",
        color=COLOR_FG,
        fontsize=9.5,
        family="monospace",
    )
    stage_txt = ax.text(
        3.1,
        -2.05,
        "",
        ha="center",
        va="center",
        color=COLOR_FG,
        fontsize=9,
        fontweight="bold",
    )

    ax_t = fig.add_subplot(gs[0, 1])
    _grid_axes(ax_t)
    ax_t.set_xlim(9.0, 25.0)
    ax_t.set_ylim(0.0, 1.15)
    ax_t.set_xlabel(T("arrival time [ms]"), fontsize=8)
    ax_t.set_ylabel(T("amplitude"), fontsize=8)
    ax_t.tick_params(labelsize=7)
    (stem_d,) = ax_t.plot([], [], color=COLOR_PRIMARY, lw=2.6, solid_capstyle="butt")
    (stem_r,) = ax_t.plot([], [], color=COLOR_SECONDARY, lw=2.6, solid_capstyle="butt")
    tau_ann = ax_t.annotate(
        "",
        xy=(0.0, 0.9),
        xytext=(0.0, 0.9),
        arrowprops={"arrowstyle": "<->", "color": COLOR_FG, "lw": 1.0},
    )
    tau_txt = ax_t.text(
        0.0, 0.98, r"$\tau$", ha="center", va="bottom", color=COLOR_FG, fontsize=9
    )
    ax_t.text(
        0.97,
        0.92,
        T("direct"),
        transform=ax_t.transAxes,
        ha="right",
        va="top",
        color=COLOR_PRIMARY,
        fontsize=8,
    )
    ax_t.text(
        0.97,
        0.80,
        T("delayed copy"),
        transform=ax_t.transAxes,
        ha="right",
        va="top",
        color=COLOR_SECONDARY,
        fontsize=8,
    )

    ax_f = fig.add_subplot(gs[1, 1])
    _grid_axes(ax_f)
    f = np.logspace(np.log10(50.0), np.log10(8000.0), 500)
    ax_f.set_xlim(50.0, 8000.0)
    # Band centres (63, 125, ... 8k) instead of the log formatter's 10^2 /
    # 10^3, which left two labels across the whole span the notches are
    # counted in. The helper switches the axis to log itself.
    format_frequency_axis(ax_f, 50.0, 8000.0)
    ax_f.set_ylim(-16.0, 8.0)
    ax_f.set_xlabel(T("Frequency [Hz]"), fontsize=8)
    ax_f.set_ylabel(T("response [dB]"), fontsize=8)
    ax_f.tick_params(labelsize=7)
    (comb,) = ax_f.plot([], [], color=COLOR_PRIMARY, lw=1.8)
    (notch_dot,) = ax_f.plot([], [], marker="v", ms=7, ls="none", color=COLOR_SECONDARY)
    notch_txt = ax_f.text(
        0.03,
        0.97,
        "",
        transform=ax_f.transAxes,
        ha="left",
        va="top",
        color=COLOR_SECONDARY,
        fontsize=8.5,
        family="monospace",
    )

    # Mic height trajectory: three plateaus (high, mid, on the floor).
    knots_t = (0.0, 2.6, 3.8, 6.2, 7.4, 10.0)
    knots_h = (1.5, 1.5, 0.6, 0.6, 0.02, 0.02)
    stages = (
        (2.0, T("high mic: dense comb")),
        (5.6, T("lower: notches move up")),
        (9.0, T("on the floor: copies merge — no comb in band")),
    )
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
            T(rf"$\Delta$ = {r2 - r1:.3f} m   $\tau$ = {tau * 1e3:.3f} ms")
        )
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
        return (
            l_dir,
            l_ref,
            l_img,
            mic_stand,
            mic_dot,
            mic_lab,
            delta_txt,
            stem_d,
            stem_r,
            tau_ann,
            tau_txt,
            comb,
            notch_dot,
            notch_txt,
            stage_txt,
        )

    _render_clip(fig, update, output_dir, "anim_comb_filtering")


def animate_dynamic_stiffness_sweep(output_dir: str) -> None:
    """EN 29052-1 resonance sweep: the load plate on its resilient specimen
    driven through fr, with the amplitude peaking and the response flipping
    from in phase with the force to a quarter cycle behind it and on to
    antiphase, which is what the measurement actually reads.
    """
    from matplotlib.patches import Rectangle

    T = _translate_str

    # The worked determination of the guide: an 8 kg plate (m't = 200 kg/m2)
    # on a 200 mm specimen resonating at fr = 25 Hz. eta is the loss factor a
    # mineral-wool layer of this class shows on the rig.
    f_r, eta = 25.0, 0.14
    f_lo, f_hi = 8.0, 60.0
    freqs = np.linspace(f_lo, f_hi, 800)

    def response(f: np.ndarray | float) -> NDArray[np.complex128]:
        ratio = np.asarray(f, dtype=float) / f_r
        return 1.0 / (1.0 - ratio**2 + 1j * eta * ratio)  # type: ignore[return-value]  # numpy stubs miss the 1j promotion to complex128

    mag = np.abs(response(freqs))
    phase = np.degrees(np.angle(response(freqs)))

    fig = _anim_figure()
    fig.suptitle(T(r"Reading $f_\mathrm{r}$ on the EN 29052-1 rig"))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[1.0, 1.0])

    # --- left: the rig, stroboscopic at the drive frequency ---------------
    ax_r = fig.add_subplot(gs[:, 0])
    _schematic_axes(ax_r, (0.0, 4.0), (0.0, 5.4))
    ax_r.add_patch(
        Rectangle((0.35, 0.35), 3.3, 0.34, facecolor="none", edgecolor=COLOR_FG, lw=1.6)
    )
    for hx in np.linspace(0.45, 3.55, 14):
        ax_r.plot([hx, hx - 0.12], [0.35, 0.20], color=COLOR_FG, lw=0.8)
    ax_r.text(
        2.0,
        0.51,
        T("rigid base"),
        ha="center",
        va="center",
        color=COLOR_FG,
        fontsize=8.5,
    )
    spec = Rectangle(
        (0.9, 0.69),
        2.2,
        0.62,
        facecolor=COLOR_TERTIARY,
        alpha=0.28,
        edgecolor=COLOR_TERTIARY,
        lw=1.6,
    )
    ax_r.add_patch(spec)
    plate = Rectangle(
        (0.75, 1.31), 2.5, 0.42, facecolor="none", edgecolor=COLOR_PRIMARY, lw=2.4
    )
    ax_r.add_patch(plate)
    plate_lbl = ax_r.text(
        2.0,
        1.52,
        T("load plate, 8 kg"),
        ha="center",
        va="center",
        color=COLOR_FG,
        fontsize=9,
    )
    spec_lbl = ax_r.text(
        3.25, 1.00, T("specimen"), ha="left", va="center", color=COLOR_FG, fontsize=8.5
    )
    y_arrow = 2.75
    force = ax_r.annotate(
        "",
        xy=(1.2, y_arrow),
        xytext=(1.2, y_arrow),
        arrowprops={"arrowstyle": "-|>", "lw": 2.6, "color": COLOR_SECONDARY},
    )
    ax_r.text(
        0.95,
        3.58,
        T("$F(t)$"),
        ha="center",
        va="bottom",
        color=COLOR_SECONDARY,
        fontsize=10,
        family="monospace",
    )
    ax_r.plot(
        [1.2, 1.2], [y_arrow - 0.03, y_arrow + 0.03], color=COLOR_SECONDARY, lw=1.0
    )
    motion = ax_r.annotate(
        "",
        xy=(2.9, y_arrow),
        xytext=(2.9, y_arrow),
        arrowprops={"arrowstyle": "-|>", "lw": 2.6, "color": COLOR_PRIMARY},
    )
    ax_r.text(
        3.0,
        3.58,
        T("plate motion"),
        ha="center",
        va="bottom",
        color=COLOR_PRIMARY,
        fontsize=9,
    )
    state_txt = ax_r.text(
        2.0, 4.9, "", ha="center", va="top", color=COLOR_FG, fontsize=10
    )
    drive_txt = ax_r.text(
        2.0,
        4.35,
        "",
        ha="center",
        va="top",
        color=COLOR_FG,
        fontsize=11,
        family="monospace",
    )

    # --- right: magnitude and phase, with the sweep marker ----------------
    ax_m = fig.add_subplot(gs[0, 1])
    _grid_axes(ax_m)
    ax_m.plot(freqs, mag, color=COLOR_PRIMARY, lw=2.0)
    ax_m.axvline(f_r, color=COLOR_FG, lw=0.9, ls=":", alpha=0.7)
    # A margin either side of the swept span: the marker is 8 pt across and
    # the sweep starts and parks on the limits themselves, so with the axes
    # ending at f_lo / f_hi the spine cut the disc in half for the whole
    # closing hold (and in every poster taken from it).
    f_pad = 1.5
    ax_m.set_xlim(f_lo - f_pad, f_hi + f_pad)
    ax_m.set_ylim(0.0, float(mag.max()) * 1.18)
    ax_m.set_ylabel(T("Response magnitude"), fontsize=9)
    ax_m.text(
        f_r + 0.8,
        float(mag.max()) * 1.05,
        T(r"$f_\mathrm{r}$ = 25 Hz"),
        ha="left",
        va="top",
        color=COLOR_FG,
        fontsize=9,
    )
    (dot_m,) = ax_m.plot([], [], "o", color=COLOR_SECONDARY, ms=8, zorder=5)

    ax_p = fig.add_subplot(gs[1, 1], sharex=ax_m)
    _grid_axes(ax_p)
    ax_p.plot(freqs, phase, color=COLOR_PRIMARY, lw=2.0)
    ax_p.axvline(f_r, color=COLOR_FG, lw=0.9, ls=":", alpha=0.7)
    ax_p.axhline(-90.0, color=COLOR_SECONDARY, lw=1.0, ls="--", alpha=0.8)
    ax_p.set_xlim(f_lo - f_pad, f_hi + f_pad)
    ax_p.set_ylim(-190.0, 10.0)
    ax_p.set_yticks([0, -90, -180])
    ax_p.set_xlabel(T("Excitation frequency [Hz]"))
    ax_p.set_ylabel(T("Phase [deg]"), fontsize=9)
    # Haloed: the Spanish string reaches as far as the phase flank, and the
    # trace was filling the counter of its last letter.
    ax_p.text(
        f_lo + 1.0,
        -83.0,
        T("−90 deg: resonance"),
        ha="left",
        va="bottom",
        color=COLOR_SECONDARY,
        fontsize=8.5,
        path_effects=_halo(),
    )
    (dot_p,) = ax_p.plot([], [], "o", color=COLOR_SECONDARY, ms=8, zorder=5)

    sweep_s = (_ANIM_FRAMES - _ANIM_HOLD) / _ANIM_FPS
    # The plate is watched stroboscopically, one slow cycle per second of
    # clip, so what moves on screen is the phase of the response relative to
    # the force, not the 25 Hz oscillation itself.
    strobe_hz = 1.0

    def update(kf: int) -> tuple[Any, ...]:
        tc = min(kf / _ANIM_FPS, sweep_s)
        f = f_lo + (f_hi - f_lo) * tc / sweep_s
        h = complex(response(f))
        ang = 2.0 * np.pi * strobe_hz * tc
        drive = float(np.cos(ang))
        norm = float(mag.max())
        # Amplitude compressed for the drawing so that the phase stays
        # readable far from resonance; the magnitude panel carries the true
        # amplitude. Positive displacement is drawn downward, like the force.
        amp = (abs(h) / norm) ** 0.35
        disp = amp * float(np.cos(ang + np.angle(h)))

        # The plate rides its displacement; the specimen compresses with it.
        y_plate = 1.31 - 0.26 * disp
        plate.set_y(y_plate)
        plate_lbl.set_position((2.0, y_plate + 0.21))
        spec.set_height(max(0.12, y_plate - 0.69))
        spec_lbl.set_position((3.25, 0.69 + 0.5 * (y_plate - 0.69)))
        force.set_position((1.2, y_arrow))
        force.xy = (1.2, y_arrow - 0.75 * drive)
        motion.set_position((2.9, y_arrow))
        motion.xy = (2.9, y_arrow - 0.75 * disp)

        dot_m.set_data([f], [abs(h)])
        dot_p.set_data([f], [np.degrees(np.angle(h))])
        deg = np.degrees(np.angle(h))
        if f < f_r - 3.0:
            state = T(r"below $f_\mathrm{r}$: the plate follows the force")
        elif f <= f_r + 3.0:
            state = T(r"at $f_\mathrm{r}$: a quarter cycle behind, amplitude peaks")
        else:
            state = T(r"above $f_\mathrm{r}$: the plate moves against the force")
        state_txt.set_text(state)
        # Translated whole, values included: assembled from an f-string and a
        # lone T("phase"), the readout kept the English decimal point through
        # the entire Spanish variant.
        drive_txt.set_text(
            T(f"$f$ = {f:4.1f} Hz    phase = {_fmt_minus(deg, '6.1f')}\u00b0")
        )
        return (
            plate,
            plate_lbl,
            spec,
            spec_lbl,
            force,
            motion,
            dot_m,
            dot_p,
            state_txt,
            drive_txt,
        )

    _render_clip(fig, update, output_dir, "anim_dynamic_stiffness_sweep")


# --- the modulation transfer function, drawn on the envelope --------------
#
# The clip that shows what ``sti_vs_t60.svg`` and ``sti_mtf_curves.svg``
# summarise: an intensity envelope losing its depth. Everything drawn is the
# measurement itself -- the received envelope is the 100 % modulated probe
# convolved with the band-filtered h_k^2(t) the standard integrates, so the
# depth (max - min)/(max + min) a reader measures off the trace is the same
# number IEC 60268-16's Schroeder integral returns (checked to four decimals
# against ``STIResult.mtf`` at T60 = 0.3, 1.0 and 2.5 s).

#: The 14 modulation frequencies of the full method (Ed.5 A.2.2), in Hz.
_MTF_MOD_FREQS = np.array(
    [0.63, 0.80, 1.00, 1.25, 1.60, 2.00, 2.50, 3.15, 4.00, 5.00, 6.30, 8.00, 10.0, 12.5]
)
#: The 1 kHz octave band of the seven, and the syllable-rate probe drawn in
#: it (4 Hz is index 8 of the fourteen).
_MTF_BAND = 3
_MTF_PROBE_INDEX = 8
_MTF_BAND_LABELS = ("125", "250", "500", "1k", "2k", "4k", "8k")
#: The two sweeps, coarse enough to stay cheap and fine enough that neither
#: the m(F) curve nor the STI needle steps visibly at 20 fps.
_MTF_T60S = tuple(round(v, 2) for v in np.arange(0.30, 2.501, 0.05))
_MTF_SNRS = tuple(round(v, 1) for v in np.arange(25.0, -0.001, -0.5))
#: The reverberation time act 2 freezes at while the noise rises.
_MTF_ACT2_T60 = 1.0
#: Envelope drawing grid: 1 ms bins (energy preserving) over a 1.5 s window,
#: six full cycles of the 4 Hz probe.
_MTF_ENV_HZ = 1000
_MTF_WINDOW_S = 1.5


def _modulation_transfer_data() -> dict[str, Any]:
    """Envelopes, MTF rows, band MTIs and STI for both sweeps of the clip.

    Cached because the four language x theme variants render one after
    another in the same process (this clip registers no field builder), and
    the octave-bank filtering plus the 96 STI runs would otherwise be paid
    four times over.
    """
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def build() -> dict[str, Any]:
        from phonometry import filters, speech

        fs = 48000
        # One fixed noise carrier for every decay, so the sweep moves
        # smoothly instead of jittering on a fresh realisation per frame.
        rng = np.random.default_rng(60268)
        carrier = rng.standard_normal(int(2.5 * max(_MTF_T60S) * fs))
        bank = filters.OctaveFilterBank(
            fs=fs, fraction=1, order=6, limits=[125.0, 8000.0]
        )
        step = fs // _MTF_ENV_HZ
        n_win = int(_MTF_WINDOW_S * _MTF_ENV_HZ)
        f_probe = float(_MTF_MOD_FREQS[_MTF_PROBE_INDEX])

        def channel(t60: float) -> tuple[Any, Any, Any]:
            n = int(2.5 * t60 * fs)
            h = carrier[:n] * np.exp(-6.9078 * np.arange(n) / fs / t60)
            bands = np.asarray(
                bank.filter(
                    h,
                    sigbands=True,
                    detrend=False,
                    calculate_level=False,
                    zero_phase=True,
                )[2]
            )
            h2 = bands[_MTF_BAND] ** 2
            nb = h2.size // step
            h2b = h2[: nb * step].reshape(nb, step).sum(axis=1)
            h2b /= h2b.sum()
            # A 100 % modulated intensity envelope through h_k^2: the lead-in
            # is the impulse-response length, so the drawn window is in
            # steady state and keeps the true modulation delay.
            drive = 1.0 + np.cos(
                2 * np.pi * f_probe * np.arange(nb + n_win) / _MTF_ENV_HZ
            )
            rx = np.convolve(drive, h2b)[nb - 1 : nb - 1 + n_win]
            return h, drive[nb - 1 : nb - 1 + n_win], rx

        rev: list[dict[str, Any]] = []
        for t60 in _MTF_T60S:
            h, tx, rx = channel(t60)
            res = speech.sti_from_impulse_response(h, fs)
            rev.append(
                {
                    "t60": t60,
                    "rx": rx,
                    "floor": 0.0,
                    "mtf": res.mtf[_MTF_BAND],
                    "mti": res.mti,
                    "sti": float(res.sti),
                    "rating": res.rating,
                }
            )
        h, tx, rx = channel(_MTF_ACT2_T60)
        noise: list[dict[str, Any]] = []
        for snr in _MTF_SNRS:
            res = speech.sti_from_impulse_response(h, fs, snr=float(snr))
            # Signal and noise are drawn at a *constant received mean*, which
            # is the point of the act: an ordinary level meter reads the same
            # number throughout while m collapses. Splitting the unit mean as
            # (I_rx + N) / (1 + N) reproduces the standard's noise factor
            # 1/(1 + 10^(-SNR/10)) exactly, and N/(1 + N) is the floor.
            n = 10 ** (-snr / 10)
            noise.append(
                {
                    "snr": snr,
                    "rx": (rx + n) / (1.0 + n),
                    "floor": n / (1.0 + n),
                    "mtf": res.mtf[_MTF_BAND],
                    "mti": res.mti,
                    "sti": float(res.sti),
                    "rating": res.rating,
                }
            )
        return {
            "t": np.arange(n_win) / _MTF_ENV_HZ,
            "tx": tx,
            "reverb": rev,
            "noise": noise,
        }

    return build()


def animate_modulation_transfer(output_dir: str) -> None:
    """The IEC 60268-16 modulation transfer function drawn where it lives:
    on a speech-rate intensity envelope. Reverberation shrinks the envelope
    about its mean; steady noise lifts a floor under it. Both shrink m, the
    m(F) curve and the band MTIs follow, and the STI falls with them.
    """
    T = _translate_str

    data = _modulation_transfer_data()
    t_env = data["t"]
    tx = data["tx"]
    rev, noi = data["reverb"], data["noise"]
    f_probe = float(_MTF_MOD_FREQS[_MTF_PROBE_INDEX])

    fig = _anim_figure()
    fig.suptitle(T("The modulation transfer function on the envelope (IEC 60268-16)"))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0])

    # --- left: the envelope itself, the picture the page never carries ---
    ax_e = fig.add_subplot(gs[:, 0])
    _grid_axes(ax_e)
    ax_e.set_xlim(0.0, float(_MTF_WINDOW_S))
    ax_e.set_ylim(0.0, 2.6)
    ax_e.set_xlabel(T("Time [s]"), fontsize=9)
    ax_e.set_ylabel(T("Intensity envelope, received mean = 1"), fontsize=9)
    ax_e.plot(
        t_env, tx, color=COLOR_MUTED, lw=1.2, ls="--", label=T("transmitted, $m$ = 1")
    )
    (rx_line,) = ax_e.plot([], [], color=COLOR_PRIMARY, lw=2.4, label=T("received"))
    ax_e.axhline(1.0, color=COLOR_FG, lw=0.9, ls="-.", alpha=0.55)
    ax_e.text(
        float(_MTF_WINDOW_S) - 0.02,
        1.03,
        T("mean, the same in every frame"),
        ha="right",
        va="bottom",
        color=COLOR_FG,
        fontsize=7.5,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": plt.rcParams["figure.facecolor"],
            "edgecolor": "none",
            "alpha": 0.8,
        },
    )
    floor_fill: dict[str, FillBetweenPolyCollection | None] = {"art": None}
    peak_line = ax_e.axhline(0.0, color=COLOR_SECONDARY, lw=1.0, ls=":")
    dip_line = ax_e.axhline(0.0, color=COLOR_SECONDARY, lw=1.0, ls=":")
    depth = _make_arrow(ax_e, COLOR_SECONDARY, scale=9.0)
    depth.set_arrowstyle("<|-|>", head_length=0.55, head_width=0.32)
    m_txt = ax_e.text(
        0.03,
        0.975,
        "",
        transform=ax_e.transAxes,
        ha="left",
        va="top",
        color=COLOR_FG,
        fontsize=9.5,
        family="monospace",
    )
    ax_e.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

    # --- top right: m over the fourteen modulation frequencies -----------
    ax_m = fig.add_subplot(gs[0, 1])
    _grid_axes(ax_m)
    ax_m.set_xscale("log")
    ax_m.set_xlim(0.55, 14.5)
    ax_m.set_ylim(0.0, 1.05)
    ax_m.set_xticks([1.0, 10.0])
    ax_m.set_xticklabels(["1", "10"])
    ax_m.set_xlabel(T("Modulation frequency $F$ [Hz]"), fontsize=8.5)
    ax_m.set_ylabel(T("$m$"), fontsize=8.5)
    ax_m.tick_params(labelsize=8)
    (m_line,) = ax_m.plot([], [], color=COLOR_PRIMARY, lw=1.8, marker="o", ms=3.4)
    (m_dot,) = ax_m.plot([], [], color=COLOR_SECONDARY, marker="o", ms=8.0, ls="none")
    ax_m.axvline(f_probe, color=COLOR_SECONDARY, lw=0.9, ls=":", alpha=0.7)
    ax_m.text(
        0.03,
        0.06,
        T("the red point is the 4 Hz probe on the left"),
        transform=ax_m.transAxes,
        ha="left",
        va="bottom",
        color=COLOR_FG,
        fontsize=7.5,
    )

    # --- bottom right: the seven band MTIs and the index they weight into -
    ax_b = fig.add_subplot(gs[1, 1])
    _grid_axes(ax_b)
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_ylabel(T("Band MTI"), fontsize=8.5)
    ax_b.set_xlabel(T("Octave band [Hz]"), fontsize=8.5)
    ax_b.tick_params(labelsize=8)
    xs = np.arange(7)
    bars = ax_b.bar(xs, np.zeros(7), width=0.66, color=COLOR_PRIMARY, alpha=0.85)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(list(_MTF_BAND_LABELS))
    sti_txt = ax_b.set_title("", fontsize=10.5, family="monospace", color=COLOR_FG)

    # Act timing over the 10 s sweep (the last 2 s of the clip are the hold).
    sweep = _ANIM_FRAMES - _ANIM_HOLD
    a1_end = int(0.60 * sweep)  # reverberation act
    a2_start = int(0.66 * sweep)  # noise act, after a short beat

    def update(kf: int) -> tuple[Any, ...]:
        k = min(kf, sweep)
        if k < a1_end:
            frac = np.clip((k - 0.07 * sweep) / (0.43 * sweep), 0.0, 1.0)
            state = rev[round(frac * (len(rev) - 1))]
            head = T(f"T60 = {state['t60']:.2f} s") + T("  no noise")
            # Short enough that the Spanish fits too: the longer wording ran
            # 50 px off the left edge of the figure once translated. The
            # "does not move" point is made by the labelled mean line itself.
            act = T("Reverberation shrinks the envelope about a fixed mean")
        elif k < a2_start:
            state = noi[0]
            head = T(f"T60 = {_MTF_ACT2_T60:.2f} s") + T(
                f"  SNR = {state['snr']:.0f} dB"
            )
            act = T("Now hold the room and let noise take part of the level")
        else:
            frac = np.clip((k - a2_start) / (0.28 * sweep), 0.0, 1.0)
            state = noi[round(frac * (len(noi) - 1))]
            head = T(f"T60 = {_MTF_ACT2_T60:.2f} s") + T(
                f"  SNR = {state['snr']:.0f} dB"
            )
            act = T("Noise raises a floor under the same mean: $m$ falls again")
        rx = state["rx"]
        rx_line.set_data(t_env, rx)
        hi, lo = float(rx.max()), float(rx.min())
        peak_line.set_ydata([hi, hi])
        dip_line.set_ydata([lo, lo])
        depth.set_positions((1.40, lo), (1.40, hi))
        if floor_fill["art"] is not None:
            floor_fill["art"].remove()
            floor_fill["art"] = None
        if state["floor"] > 0.01:
            floor_fill["art"] = ax_e.fill_between(
                t_env, 0.0, state["floor"], color=COLOR_TERTIARY, alpha=0.30, lw=0
            )
        m_drawn = (hi - lo) / (hi + lo)
        m_txt.set_text(head + "\n" + T(f"$m$({f_probe:.0f} Hz) = {m_drawn:.2f}"))
        ax_e.set_title(act, fontsize=9.5, fontstyle="italic", color=COLOR_FG)
        m_line.set_data(_MTF_MOD_FREQS, state["mtf"])
        m_dot.set_data([f_probe], [state["mtf"][_MTF_PROBE_INDEX]])
        for bar, value in zip(bars, state["mti"], strict=True):
            bar.set_height(float(value))
        sti_txt.set_text(T(f"STI = {state['sti']:.2f}") + f"  ({state['rating']})")
        arts: list[Any] = [
            rx_line,
            peak_line,
            dip_line,
            depth,
            m_txt,
            ax_e.title,
            m_line,
            m_dot,
            sti_txt,
            *bars,
        ]
        if floor_fill["art"] is not None:
            arts.append(floor_fill["art"])
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_modulation_transfer")


# --- the two-pass EBU R 128 gate, watched while it decides ----------------
#
# The gating blocks are the momentary series at a 100 ms hop: a 400 ms window
# every 100 ms is the 75 % overlap of BS.1770-5 Formula (3), so
# ``ProgramLoudnessResult.momentary`` with ``momentary_step = 0.1`` *is* the
# block loudness l_j. Their energy mean is the gated loudness of a block set
# (the -0.691 offset of Formula (2) cancels in the mean), so the two passes
# are two energy means: over the blocks above -70 LUFS, then over those of
# them above that mean minus 10 LU. Checked against the library on the
# programme below: relative threshold -34.279 LUFS and integrated
# -23.0 LUFS, both to three decimals.

#: The section-3 programme of the page: five shaped-noise sections with a
#: 0.9 Hz and 2.83 Hz wobble, normalised to the -23.0 LUFS target.
_GATE_SECTIONS = (
    (-38.0, 8.0),
    (-23.0, 16.0),
    (-17.0, 12.0),
    (-25.0, 16.0),
    (-45.0, 8.0),
)
_GATE_ABSOLUTE = -70.0  # BS.1770-5 Formula (4), the absolute gate
_GATE_RELATIVE = -10.0  # Formula (5), the integrated relative gate
_LRA_RELATIVE = -20.0  # EBU Tech 3342, the deeper loudness-range gate


def _energy_mean_lufs(values: NDArray[np.float64]) -> float:
    """Energy (not arithmetic) mean of block loudness values, in LUFS."""
    return float(10.0 * np.log10(np.mean(10.0 ** (np.asarray(values) / 10.0))))


def _loudness_gating_data() -> dict[str, Any]:
    """The page's five-section programme, metered into gating blocks.

    Cached: the four language x theme variants render in the same process
    and the K-weighting of a minute of 48 kHz stereo is the whole cost.
    """
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def build() -> dict[str, Any]:
        from scipy import signal as sp

        from phonometry import broadcast

        fs = 48000
        rng = np.random.default_rng(1770)
        sos = sp.butter(2, 2000.0, fs=fs, output="sos")
        chunks = []
        for level, seconds in _GATE_SECTIONS:
            noise = sp.sosfilt(sos, rng.standard_normal(int(seconds * fs)))
            noise /= np.sqrt(np.mean(noise**2))
            t = np.arange(noise.size) / fs
            wobble = (
                1
                + 0.22 * np.sin(2 * np.pi * 0.9 * t)
                + 0.14 * np.sin(2 * np.pi * 2.83 * t + 1.0)
            )
            chunks.append(10 ** (level / 20) * noise * wobble)
        x = np.concatenate(chunks)
        stereo = np.vstack([x, x])
        x *= 10 ** ((-23.0 - broadcast.integrated_loudness(stereo, fs)) / 20)
        res = broadcast.program_loudness(np.vstack([x, x]), fs, momentary_step=0.1)
        blocks = np.asarray(res.momentary, dtype=float)
        finite = np.isfinite(blocks)
        return {
            "block_t": np.asarray(res.momentary_time, dtype=float)[finite],
            "block_l": blocks[finite],
            "st_t": np.asarray(res.short_term_time, dtype=float)[
                np.isfinite(res.short_term)
            ],
            "st_l": np.asarray(res.short_term, dtype=float)[
                np.isfinite(res.short_term)
            ],
            "integrated": float(res.integrated),
            "relative_threshold": float(res.relative_threshold),
            "lra": float(res.loudness_range),
            "lra_low": float(res.lra_low),
            "lra_high": float(res.lra_high),
            "duration": float(sum(s for _, s in _GATE_SECTIONS)),
        }

    return build()


def animate_loudness_gating(output_dir: str) -> None:
    """The BS.1770-5 double gate deciding, block by block: the relative
    threshold is recomputed from the survivors of the first pass, so it
    slides as the programme plays and retroactively drops blocks that were
    passing. Then the deeper -20 LU gate of the loudness range.
    """
    T = _translate_str

    data = _loudness_gating_data()
    bt, bl = data["block_t"], data["block_l"]
    st_t, st_l = data["st_t"], data["st_l"]
    duration = data["duration"]
    edges = np.arange(-72.0, -7.9, 1.0)
    centres = 0.5 * (edges[:-1] + edges[1:])
    max_count = int(np.histogram(bl, bins=edges)[0].max())

    fig = _anim_figure()
    fig.suptitle(
        T("The two passes of the EBU R 128 gate (BS.1770-5)"),
    )
    gs = fig.add_gridspec(2, 2, width_ratios=[3.05, 1.0], height_ratios=[2.5, 1.0])

    # --- top left: the programme, block by block -------------------------
    ax_t = fig.add_subplot(gs[0, 0])
    _grid_axes(ax_t)
    ax_t.set_xlim(0.0, duration)
    ax_t.set_ylim(-72.0, -8.0)
    ax_t.set_xlabel(T("Time [s]"), fontsize=9)
    ax_t.set_ylabel(T("Loudness [LUFS]"), fontsize=9)
    ax_t.tick_params(labelsize=8)
    (kept,) = ax_t.plot(
        [],
        [],
        ls="none",
        marker="s",
        ms=2.6,
        color=COLOR_PRIMARY,
        label=T("block, counted"),
    )
    (dropped,) = ax_t.plot(
        [],
        [],
        ls="none",
        marker="s",
        ms=2.6,
        markerfacecolor="none",
        markeredgewidth=0.7,
        color=COLOR_MUTED,
        label=T("block, gated out"),
    )
    (short,) = ax_t.plot(
        [], [], color=COLOR_QUATERNARY, lw=1.4, label=T("short-term (3 s)")
    )
    ax_t.axhline(_GATE_ABSOLUTE, color=COLOR_FG, lw=1.0, ls=":", alpha=0.8)
    ax_t.text(
        0.4,
        _GATE_ABSOLUTE + 0.8,
        T("absolute gate, −70 LUFS"),
        ha="left",
        va="bottom",
        color=COLOR_FG,
        fontsize=7.5,
    )
    rel_line = ax_t.axhline(-70.0, color=COLOR_SECONDARY, lw=1.6, ls="--")
    # The threshold labels ride over the block scatter, so they carry the
    # figure's own background: a block square landing behind a digit is the
    # one thing that would make the number unreadable.
    _label_bbox = {
        "boxstyle": "round,pad=0.2",
        "facecolor": plt.rcParams["figure.facecolor"],
        "edgecolor": "none",
        "alpha": 0.85,
    }
    rel_txt = ax_t.text(
        duration - 0.4,
        -70.0,
        "",
        ha="right",
        va="bottom",
        color=COLOR_SECONDARY,
        fontsize=8,
        bbox=_label_bbox,
    )
    int_line = ax_t.axhline(-70.0, color=COLOR_TERTIARY, lw=1.6)
    (head,) = ax_t.plot([], [], color=COLOR_FG, lw=1.0, alpha=0.55)
    lra_gate = ax_t.axhline(
        -70.0, color=COLOR_QUATERNARY, lw=1.2, ls="--", visible=False
    )
    lra_gate_txt = ax_t.text(
        duration - 0.4,
        -70.0,
        "",
        ha="right",
        va="bottom",
        color=COLOR_QUATERNARY,
        fontsize=8,
        visible=False,
        bbox=_label_bbox,
    )
    lra_band: dict[str, Rectangle | None] = {"art": None}
    ax_t.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    # --- top right: the distribution the second pass is computed from ----
    ax_h = fig.add_subplot(gs[0, 1], sharey=ax_t)
    _grid_axes(ax_h)
    ax_h.set_xlim(0.0, max_count * 1.12)
    ax_h.set_xlabel(T("Blocks per LU"), fontsize=8.5)
    ax_h.tick_params(labelsize=8, labelleft=False)
    hbars = ax_h.barh(
        centres, np.zeros(centres.size), height=0.92, color=COLOR_PRIMARY, alpha=0.85
    )
    ax_h.axhline(_GATE_ABSOLUTE, color=COLOR_FG, lw=1.0, ls=":", alpha=0.8)
    rel_line_h = ax_h.axhline(-70.0, color=COLOR_SECONDARY, lw=1.6, ls="--")
    p10 = ax_h.axhline(-70.0, color=COLOR_QUATERNARY, lw=1.4, ls="-.", visible=False)
    p95 = ax_h.axhline(-70.0, color=COLOR_QUATERNARY, lw=1.4, ls="-.", visible=False)
    lra_gate_h = ax_h.axhline(
        -70.0, color=COLOR_QUATERNARY, lw=1.2, ls="--", visible=False
    )

    # --- bottom: what the two passes have produced so far ----------------
    ax_v = fig.add_subplot(gs[1, :])
    _schematic_axes(ax_v, (0.0, 12.0), (0.0, 1.6))
    boxes = [
        _flow_box(ax_v, 1.6, 0.8, 2.9, 1.15, T("Integrated $I$ (gated)")),
        _flow_box(ax_v, 4.7, 0.8, 2.9, 1.15, T("Ungated energy mean")),
        _flow_box(ax_v, 7.8, 0.8, 2.9, 1.15, T("What the gate is worth")),
        _flow_box(ax_v, 10.7, 0.8, 2.5, 1.15, T("Blocks gated out")),
    ]
    # The loudness-range act replaces the third box in place, because its
    # title has to change with its value and a flow box carries a fixed one.
    # Slightly wider than the box it replaces: this is the one title that
    # carries a standard number, and at 2.9 the English had 11 px of margin
    # and the Spanish overflowed its own border by 27 px on each side.
    lra_box = _flow_box(ax_v, 7.8, 0.8, 3.05, 1.15, T("Loudness range (Tech 3342)"))
    for key in ("box", "title", "value"):
        lra_box[key].set_visible(False)

    sweep = _ANIM_FRAMES - _ANIM_HOLD
    play_end = int(0.70 * sweep)  # the programme has finished playing
    lra_start = int(0.78 * sweep)  # the loudness-range act begins

    def update(kf: int) -> tuple[Any, ...]:
        k = min(kf, sweep)
        played = duration * min(k / play_end, 1.0)
        seen = bt <= played
        l_seen = bl[seen]
        arts: list[Any] = []
        if l_seen.size == 0:
            l_seen = bl[:1]
            seen = bt <= bt[0]
        above_abs = l_seen > _GATE_ABSOLUTE
        gamma = _energy_mean_lufs(l_seen[above_abs]) + _GATE_RELATIVE
        passing = above_abs & (l_seen > gamma)
        integrated = _energy_mean_lufs(l_seen[passing])
        ungated = _energy_mean_lufs(l_seen)
        kept.set_data(bt[seen][passing], l_seen[passing])
        dropped.set_data(bt[seen][~passing], l_seen[~passing])
        st_seen = st_t <= played
        short.set_data(st_t[st_seen], st_l[st_seen])
        rel_line.set_ydata([gamma, gamma])
        rel_line_h.set_ydata([gamma, gamma])
        rel_txt.set_position((duration - 0.4, gamma + 0.6))
        rel_txt.set_text(T(f"relative gate {_fmt_minus(gamma, '.1f')} LUFS"))
        int_line.set_ydata([integrated, integrated])
        head.set_data([played, played], [-72.0, -8.0])
        counts = np.histogram(l_seen, bins=edges)[0]
        for bar, count, centre in zip(hbars, counts, centres, strict=True):
            bar.set_width(float(count))
            bar.set_color(COLOR_PRIMARY if centre > gamma else COLOR_MUTED)
        arts += [
            kept,
            dropped,
            short,
            rel_line,
            rel_line_h,
            rel_txt,
            int_line,
            head,
            *hbars,
        ]

        lra_on = k >= lra_start
        for key in ("box", "title", "value"):
            lra_box[key].set_visible(lra_on)
            boxes[2][key].set_visible(not lra_on)
        arts += [lra_box["box"], lra_box["title"], lra_box["value"]]
        if lra_on:
            frac = np.clip((k - lra_start) / (sweep - lra_start), 0.0, 1.0)
            mid = 0.5 * (data["lra_low"] + data["lra_high"])
            lo = mid + frac * (data["lra_low"] - mid)
            hi = mid + frac * (data["lra_high"] - mid)
            p10.set_ydata([lo, lo])
            p95.set_ydata([hi, hi])
            p10.set_visible(True)
            p95.set_visible(True)
            if lra_band["art"] is not None:
                lra_band["art"].remove()
            lra_band["art"] = ax_t.axhspan(
                lo, hi, color=COLOR_QUATERNARY, alpha=0.14, lw=0
            )
            st_above = st_l[st_l > _GATE_ABSOLUTE]
            gate_lra = _energy_mean_lufs(st_above) + _LRA_RELATIVE
            for line in (lra_gate, lra_gate_h):
                line.set_ydata([gate_lra, gate_lra])
                line.set_visible(True)
            lra_gate_txt.set_position((duration - 0.4, gate_lra + 0.6))
            lra_gate_txt.set_text(
                T(f"short-term gate {_fmt_minus(gate_lra, '.1f')} LUFS")
            )
            lra_gate_txt.set_visible(True)
            arts += [p10, p95, lra_band["art"], lra_gate, lra_gate_h, lra_gate_txt]
            act = T(
                "The loudness range gates 10 LU deeper and reads the "
                "10th to 95th percentile spread"
            )
        elif k >= play_end:
            act = T(
                "154 of the 597 blocks never counted: the quiet opening "
                "and the fade-out"
            )
        elif played < 9.0:
            act = T(
                "Nothing loud has played yet, so the relative gate sits "
                "low and every block counts"
            )
        else:
            act = T(
                "Louder material raises the relative gate, and blocks "
                "that were counted stop counting"
            )
        ax_t.set_title(act, fontsize=9.5, fontstyle="italic", color=COLOR_FG)

        values = [
            T(f"{_fmt_minus(integrated, '.1f')} LUFS"),
            T(f"{_fmt_minus(ungated, '.1f')} LUFS"),
            T(f"{integrated - ungated:+.2f} LU"),
            T(f"{int((~passing).sum())} of {l_seen.size}"),
        ]
        colors = [COLOR_TERTIARY, COLOR_FG, COLOR_SECONDARY, COLOR_FG]
        for box, value, color in zip(boxes, values, colors, strict=True):
            _light_box(box, value, color, fill=color is COLOR_TERTIARY)
            arts += [box["box"], box["title"], box["value"]]
        if lra_on:
            _light_box(lra_box, T(f"{data['lra']:.1f} LU"), COLOR_QUATERNARY, fill=True)
        arts.append(ax_t.title)
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_loudness_gating")


# --- EPNL, record by record ------------------------------------------------
#
# The certification metric is sequential and the page's figure is its end
# state: a spectrum arrives every half second, the slope ("encircling")
# method fits a background SPL'' to it and the tone excess F over that
# background sets C, PNLT = PNL + C rises and falls, and only once the peak
# PNLTM is known can the 10 dB-down window be located and the duration
# correction D exist. The clip runs that order. The synthetic flyover is the
# page's own (41 records at dt = 0.5 s, a Gaussian gain envelope and a
# 2500 Hz fan tone), and reproduces its printed numbers: PNLTM = 120.57
# PNdB, C = 3.97 dB at the peak, window records 16 to 24, D = -6.56 dB,
# EPNL = 114.01 EPNdB.

_EPNL_RECORDS = 41
_EPNL_DT = 0.5
_EPNL_TONE_BAND = 17  # 2500 Hz, the fan tone of the page's flyover


def _epnl_flyover_data() -> dict[str, Any]:
    """The page's synthetic flyover, with the per-record tone-correction fit."""
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def build() -> dict[str, Any]:
        from phonometry import aircraft
        from phonometry.aircraft.certification import (
            _tone_background,
            _tone_factor,
        )

        idx = np.arange(_EPNL_RECORDS)
        shape = 15.0 * np.exp(
            -((np.log10(aircraft.NOY_BANDS) - np.log10(400.0)) ** 2) / 0.5
        )
        gain = 30.0 * np.exp(-((idx - 20.0) ** 2) / (2 * 5.0**2)) - 5.0
        spectra = (55.0 + shape)[None, :] + gain[:, None]
        # The 2500 Hz fan tone that grows and fades with the pass-by.
        spectra[:, _EPNL_TONE_BAND] += 12.0 * np.exp(
            -((idx - 20.0) ** 2) / (2 * 6.0**2)
        )
        res = aircraft.effective_perceived_noise_level(spectra, _EPNL_DT)
        backgrounds = np.empty_like(spectra)
        winners = np.zeros(_EPNL_RECORDS, dtype=int)
        excess = np.zeros(_EPNL_RECORDS)
        for j in range(_EPNL_RECORDS):
            bg, exc = _tone_background(spectra[j])
            factors = [
                _tone_factor(float(exc[i]), float(aircraft.NOY_BANDS[i]))
                for i in range(24)
            ]
            backgrounds[j] = bg
            winners[j] = int(np.argmax(factors))
            excess[j] = float(exc[winners[j]])
        return {
            "bands": np.asarray(aircraft.NOY_BANDS, dtype=float),
            "spectra": spectra,
            "background": backgrounds,
            "winner": winners,
            "excess": excess,
            "times": np.asarray(res.times, dtype=float),
            "pnl": np.asarray(res.pnl, dtype=float),
            "pnlt": np.asarray(res.pnlt, dtype=float),
            "c": np.asarray(res.tone_correction, dtype=float),
            "pnltm": float(res.pnltm),
            "epnl": float(res.epnl),
            "duration_correction": float(res.duration_correction),
            "limits": tuple(int(v) for v in res.band_limits),
        }

    return build()


def animate_epnl_flyover(output_dir: str) -> None:
    """EPNL built in the order the standard builds it: a spectrum every
    half second, a background fitted under its fan tone, PNLT rising over
    PNL by that correction, and only at the end -- once the peak is known --
    the 10 dB-down window, the duration correction and the EPNL.
    """
    from matplotlib.patches import Polygon

    T = _translate_str

    d = _epnl_flyover_data()
    bands, spectra, background = d["bands"], d["spectra"], d["background"]
    times, pnl, pnlt = d["times"], d["pnl"], d["pnlt"]
    k_first, k_last = d["limits"]
    threshold = d["pnltm"] - 10.0
    duration = float(times[-1])

    fig = _anim_figure()
    fig.suptitle(
        T("EPNL, record by record (ICAO Annex 16 Appendix 2)"),
    )
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[1.45, 1.0])

    # --- left: the spectrum of the current record and its fitted background
    ax_s = fig.add_subplot(gs[:, 0])
    _grid_axes(ax_s)
    xs = np.arange(24)
    ax_s.set_xlim(-0.7, 23.7)
    # Headroom on purpose: the fan tone reaches 96.2 dB at the peak record,
    # which on a 40-105 axis lands exactly under the legend and the tone
    # readout. 115 keeps the top of the panel free for both.
    ax_s.set_ylim(40.0, 115.0)
    ax_s.set_xlabel(T("One-third-octave band [Hz]"), fontsize=8.5)
    ax_s.set_ylabel(T("Band level [dB]"), fontsize=9)
    ax_s.set_xticks([0, 5, 10, 15, 20, 23])
    ax_s.set_xticklabels(["50", "160", "500", "1.6k", "5k", "10k"])
    ax_s.tick_params(labelsize=8)
    bars = ax_s.bar(xs, np.zeros(24), width=0.78, color=COLOR_PRIMARY, alpha=0.85)
    (bg_line,) = ax_s.plot(
        [],
        [],
        color=COLOR_SECONDARY,
        lw=1.8,
        ls="--",
        label=T(r"fitted background $\mathrm{SPL}''$"),
    )
    excess_arrow = _make_arrow(ax_s, COLOR_SECONDARY, scale=9.0)
    excess_arrow.set_arrowstyle("<|-|>", head_length=0.55, head_width=0.32)
    # Anchored in axes coordinates, in the empty band above the spectrum:
    # the tone can sit anywhere from 50 Hz to 10 kHz, and a readout that
    # follows it would leave the axes as soon as the winner moved right.
    exc_txt = ax_s.text(
        0.97,
        0.88,
        "",
        transform=ax_s.transAxes,
        ha="right",
        va="top",
        color=COLOR_SECONDARY,
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": plt.rcParams["figure.facecolor"],
            "edgecolor": "none",
            "alpha": 0.85,
        },
    )
    ax_s.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    # --- top right: the two level histories being written ----------------
    ax_h = fig.add_subplot(gs[0, 1])
    _grid_axes(ax_h)
    ax_h.set_xlim(0.0, duration)
    ax_h.set_ylim(75.0, 128.0)
    ax_h.set_xlabel(T("Time [s]"), fontsize=8.5)
    ax_h.set_ylabel(T("Level [PNdB]"), fontsize=8.5)
    ax_h.tick_params(labelsize=8)
    (pnl_line,) = ax_h.plot(
        [], [], color=COLOR_MUTED, lw=1.6, marker="o", ms=2.4, label=T("PNL")
    )
    (pnlt_line,) = ax_h.plot(
        [],
        [],
        color=COLOR_PRIMARY,
        lw=2.0,
        marker="o",
        ms=2.4,
        label=T(r"$\mathrm{PNLT} = \mathrm{PNL} + C$"),
    )
    gap: dict[str, FillBetweenPolyCollection | None] = {"art": None, "win": None}
    thr_line = ax_h.axhline(
        threshold, color=COLOR_SECONDARY, lw=1.2, ls=":", visible=False
    )
    thr_txt = ax_h.text(
        0.3,
        threshold + 0.8,
        T(r"$\mathrm{PNLTM} - 10$ dB"),
        ha="left",
        va="bottom",
        color=COLOR_SECONDARY,
        fontsize=8,
        visible=False,
    )
    (peak_dot,) = ax_h.plot(
        [], [], ls="none", marker="v", ms=8.0, color=COLOR_SECONDARY
    )
    ax_h.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

    # --- bottom right: where the aircraft is, and what exists yet --------
    # The elevation lives on the left of the strip and the three results
    # stack on the right, so nothing is ever drawn over anything else.
    ax_v = fig.add_subplot(gs[1, 1])
    _schematic_axes(ax_v, (0.0, 12.0), (0.0, 4.4))
    ax_v.plot([0.15, 4.6], [0.30, 0.30], color=COLOR_FG, lw=1.2)
    ax_v.plot([0.15, 4.6], [3.55, 3.55], color=COLOR_GRID, lw=1.0, ls="--")
    ax_v.plot([2.35, 2.35], [0.30, 0.95], color=COLOR_FG, lw=1.2)
    ax_v.plot([2.35], [1.02], marker="o", ms=5.0, color=COLOR_FG, ls="none")
    ax_v.text(
        2.35,
        0.05,
        T("microphone"),
        ha="center",
        va="bottom",
        color=COLOR_FG,
        fontsize=7.5,
    )
    plane = Polygon(
        [[0, 0]], closed=True, facecolor=COLOR_PRIMARY, edgecolor=COLOR_FG, lw=0.8
    )
    ax_v.add_patch(plane)
    (slant,) = ax_v.plot([], [], color=COLOR_PRIMARY, lw=1.0, ls=":")
    boxes = [
        _flow_box(ax_v, 8.7, 3.55, 6.4, 1.10, T("Peak PNLTM")),
        _flow_box(ax_v, 8.7, 2.15, 6.4, 1.10, T("Duration correction $D$")),
        _flow_box(ax_v, 8.7, 0.75, 6.4, 1.10, T("EPNL")),
    ]

    sweep = _ANIM_FRAMES - _ANIM_HOLD
    fly_end = int(0.74 * sweep)  # the last record has arrived
    peak_at = int(0.80 * sweep)  # PNLTM is known, the threshold drawn
    win_at = int(0.86 * sweep)  # the 10 dB-down window opens
    epnl_at = int(0.96 * sweep)  # the duration correction and the EPNL

    def plane_shape(x: float, y: float) -> NDArray[np.float64]:
        s_x, s_y = 0.42, 0.30
        return np.array(
            [
                [x + 1.30 * s_x, y],
                [x + 0.10 * s_x, y + 0.28 * s_y],
                [x - 0.15 * s_x, y + 1.15 * s_y],
                [x - 0.55 * s_x, y + 1.15 * s_y],
                [x - 0.50 * s_x, y + 0.25 * s_y],
                [x - 1.35 * s_x, y + 0.20 * s_y],
                [x - 1.55 * s_x, y + 0.85 * s_y],
                [x - 1.85 * s_x, y + 0.85 * s_y],
                [x - 1.80 * s_x, y - 0.05 * s_y],
                [x - 1.55 * s_x, y - 0.85 * s_y],
                [x - 1.85 * s_x, y - 0.85 * s_y],
                [x - 1.35 * s_x, y - 0.20 * s_y],
                [x - 0.50 * s_x, y - 0.25 * s_y],
                [x - 0.15 * s_x, y - 1.15 * s_y],
                [x - 0.55 * s_x, y - 1.15 * s_y],
                [x + 0.10 * s_x, y - 0.28 * s_y],
            ]
        )

    def update(kf: int) -> tuple[Any, ...]:
        k = min(kf, sweep)
        j = min(int(k / fly_end * (_EPNL_RECORDS - 1)), _EPNL_RECORDS - 1)
        for bar, level in zip(bars, spectra[j], strict=True):
            bar.set_height(float(level))
        bg_line.set_data(xs[2:], background[j][2:])
        w = int(d["winner"][j])
        f_exc = float(d["excess"][j])
        arts: list[Any] = [*bars, bg_line, excess_arrow, exc_txt]
        if f_exc >= 1.5:
            excess_arrow.set_visible(True)
            excess_arrow.set_positions(
                (float(w), float(background[j][w])), (float(w), float(spectra[j][w]))
            )
            exc_txt.set_visible(True)
            exc_txt.set_text(
                T(f"$F$ = {f_exc:.1f} dB at {bands[w]:.0f} Hz")
                + "\n"
                + T(f"$C$ = {d['c'][j]:.2f} dB")
            )
        else:
            excess_arrow.set_visible(False)
            exc_txt.set_visible(False)

        pnl_line.set_data(times[: j + 1], pnl[: j + 1])
        pnlt_line.set_data(times[: j + 1], pnlt[: j + 1])
        if gap["art"] is not None:
            gap["art"].remove()
        gap["art"] = ax_h.fill_between(
            times[: j + 1],
            pnl[: j + 1],
            pnlt[: j + 1],
            color=COLOR_SECONDARY,
            alpha=0.22,
            lw=0,
        )
        arts += [pnl_line, pnlt_line, gap["art"]]

        # The flight path, drawn so the aircraft is overhead at the peak.
        x_plane = 0.45 + 3.90 * (j / (_EPNL_RECORDS - 1))
        plane.set_xy(plane_shape(x_plane, 3.55))
        slant.set_data([x_plane, 2.35], [3.55, 1.02])
        arts += [plane, slant]

        known_peak = k >= peak_at
        thr_line.set_visible(known_peak)
        thr_txt.set_visible(known_peak)
        peak_dot.set_data(
            [times[int(np.argmax(pnlt))]] if known_peak else [],
            [d["pnltm"] + 2.5] if known_peak else [],
        )
        arts += [thr_line, thr_txt, peak_dot]

        if k >= win_at:
            frac = np.clip((k - win_at) / max(epnl_at - win_at, 1), 0.0, 1.0)
            k_end = k_first + round(frac * (k_last - k_first))
            sel = slice(k_first, k_end + 1)
            if gap["win"] is not None:
                gap["win"].remove()
            gap["win"] = ax_h.fill_between(
                times[sel], threshold, pnlt[sel], color=COLOR_TERTIARY, alpha=0.30, lw=0
            )
            arts.append(gap["win"])

        if k < fly_end:
            # Kept short on purpose: at 9 pt the longer wording overflowed
            # the 2400 px figure by 87 px in English and 137 in Spanish.
            # "add C to PNL" is already on screen twice, in the panel legend
            # and in the C readout beside the tone.
            stage_text = T("Each record: fit the background, measure the tone excess")
        elif k < peak_at:
            stage_text = T("The pass is over; only now is the peak PNLTM known")
        elif k < win_at:
            stage_text = T("The 10 dB-down window is located from that peak")
        elif k < epnl_at:
            stage_text = T(
                f"Sum the energy inside the window, records {k_first} to {k_last}"
            )
        else:
            stage_text = T("Divide by the fixed 10 s reference: D, then EPNL")
        ax_h.set_title(stage_text, fontsize=9.0, fontstyle="italic", color=COLOR_FG)

        for box in boxes:
            _dim_box(box)
        if known_peak:
            _light_box(boxes[0], T(f"{d['pnltm']:.2f} PNdB"), COLOR_SECONDARY)
        if k >= epnl_at:
            _light_box(
                boxes[1],
                T(f"{_fmt_minus(d['duration_correction'], '.2f')} dB"),
                COLOR_TERTIARY,
            )
            _light_box(boxes[2], T(f"{d['epnl']:.2f} EPNdB"), COLOR_PRIMARY, fill=True)
        for box in boxes:
            arts += [box["box"], box["title"], box["value"]]
        arts.append(ax_h.title)
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_epnl_flyover")


# --- the image lattice being swept ----------------------------------------
#
# The reflectogram is not a decaying signal, it is a lattice read off by an
# expanding sphere: every image at distance r_n contributes one arrival at
# r_n / c. Drawn that way the reflection density stops being a quoted
# formula and becomes a rate the viewer watches. The run stops at 60 ms,
# where the counted arrivals (344) and the analytic
# N = (4 pi / 3)(c t)^3 / V (347.7) still agree to 1 %; past about 80 ms the
# order-10 cut-off truncates the lattice and the two part company, which is
# the page's own "choosing max_order" argument.

_IS_ROOM = (7.0, 5.0, 3.0)
_IS_SOURCE = (2.0, 1.6, 1.5)
_IS_RECEIVER = (5.2, 3.4, 1.7)
_IS_ABSORPTION = 0.12
_IS_MAX_ORDER = 10
_IS_RUN_MS = 60.0


def _image_source_data() -> dict[str, Any]:
    """The page's 7 x 5 x 3 m room, its images and its reflectogram."""
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def build() -> dict[str, Any]:
        from phonometry import room

        res = room.image_source_rir(
            dimensions=_IS_ROOM,
            source=_IS_SOURCE,
            receiver=_IS_RECEIVER,
            absorption=_IS_ABSORPTION,
            fs=48000,
            max_order=_IS_MAX_ORDER,
        )
        t_ms = np.asarray(res.times, dtype=float) * 1e3
        amp = np.abs(np.asarray(res.amplitudes, dtype=float))
        level = 20.0 * np.log10(amp / amp.max())
        order = np.asarray(res.orders, dtype=int)
        pos = np.asarray(res.image_positions, dtype=float)
        keep = t_ms <= _IS_RUN_MS
        # The plan draws the images that share the source's own height, the
        # set ``plot_geometry()`` draws: every other image projects onto one
        # of the same plan positions, so plotting them only overprints.
        plane = keep & np.isclose(pos[:, 2], _IS_SOURCE[2])
        return {
            "t_ms": t_ms[keep],
            "level": level[keep],
            "order": order[keep],
            "plan_xy": pos[plane][:, :2],
            "plan_t": t_ms[plane],
            "plan_order": order[plane],
            "c": float(res.speed_of_sound),
            "direct_ms": float(np.min(t_ms)),
        }

    return build()


def animate_image_source_buildup(output_dir: str) -> None:
    """A circle expanding from the receiver at c t sweeps the mirror-room
    lattice; every image it reaches writes its arrival into the
    reflectogram, and the running count follows the (4 pi / 3)(c t)^3 / V
    law whose derivative is the reflection density of the guide.
    """
    from matplotlib.patches import Circle, Rectangle

    from .theme import series_colors

    T = _translate_str

    d = _image_source_data()
    c = d["c"]
    t_ms, level, order = d["t_ms"], d["level"], d["order"]
    plan_xy, plan_t = d["plan_xy"], d["plan_t"]
    volume = float(np.prod(_IS_ROOM))
    n_orders = int(order.max()) + 1
    palette = series_colors(n_orders)

    fig = _anim_figure()
    fig.suptitle(T("The reflectogram is a lattice being swept (image-source method)"))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0], height_ratios=[1.5, 1.0])

    # --- left: the mirror-room plan the circle sweeps --------------------
    ax_p = fig.add_subplot(gs[:, 0])
    _schematic_axes(ax_p, (-17.0, 27.0), (-18.0, 25.0), equal=True)
    lx, ly = _IS_ROOM[0], _IS_ROOM[1]
    for i in range(-3, 5):
        ax_p.plot([i * lx, i * lx], [-18.0, 25.0], color=COLOR_GRID, lw=0.7, ls="--")
    for j in range(-4, 6):
        ax_p.plot([-17.0, 27.0], [j * ly, j * ly], color=COLOR_GRID, lw=0.7, ls="--")
    ax_p.add_patch(
        Rectangle((0.0, 0.0), lx, ly, facecolor="none", edgecolor=COLOR_FG, lw=2.0)
    )
    ax_p.plot(
        [_IS_SOURCE[0]],
        [_IS_SOURCE[1]],
        marker="*",
        ms=13,
        color=COLOR_SECONDARY,
        ls="none",
    )
    ax_p.plot(
        [_IS_RECEIVER[0]],
        [_IS_RECEIVER[1]],
        marker="v",
        ms=9,
        color=COLOR_PRIMARY,
        ls="none",
    )
    ax_p.text(
        _IS_SOURCE[0],
        _IS_SOURCE[1] - 1.6,
        T("source"),
        ha="center",
        va="top",
        color=COLOR_FG,
        fontsize=8,
    )
    ax_p.text(
        _IS_RECEIVER[0] + 1.0,
        _IS_RECEIVER[1] + 0.6,
        T("receiver"),
        ha="left",
        va="bottom",
        color=COLOR_FG,
        fontsize=8,
    )
    ax_p.plot(
        plan_xy[:, 0],
        plan_xy[:, 1],
        ls="none",
        marker="o",
        ms=2.6,
        color=COLOR_MUTED,
        alpha=0.55,
    )
    lit = ax_p.scatter([], [], s=30.0, zorder=4)
    rays = LineCollection([], colors=COLOR_PRIMARY, linewidths=0.5, alpha=0.45)
    ax_p.add_collection(rays)
    front = Circle(
        (_IS_RECEIVER[0], _IS_RECEIVER[1]),
        0.01,
        facecolor="none",
        edgecolor=COLOR_SECONDARY,
        lw=2.0,
        zorder=5,
    )
    ax_p.add_patch(front)
    # Two lines, not one: on a single line this caption started 303 px off
    # the left edge of the 2400 px figure, and the Spanish is longer still.
    ax_p.set_title(
        T(
            "the plan draws only the images at the source's own "
            "height;\nthe floor and ceiling families arrive between "
            "them"
        ),
        fontsize=8,
        color=COLOR_FG,
    )
    radius_txt = ax_p.text(
        -16.0,
        -17.0,
        "",
        ha="left",
        va="bottom",
        color=COLOR_SECONDARY,
        fontsize=9,
        family="monospace",
    )

    # --- top right: the reflectogram being written -----------------------
    ax_r = fig.add_subplot(gs[0, 1])
    _grid_axes(ax_r)
    ax_r.set_xlim(0.0, _IS_RUN_MS)
    ax_r.set_ylim(-26.0, 3.0)
    ax_r.set_xlabel(T("Arrival time [ms]"), fontsize=8.5)
    ax_r.set_ylabel(T("Level re direct [dB]"), fontsize=8.5)
    ax_r.tick_params(labelsize=8)
    stems = LineCollection([], colors=COLOR_GRID, linewidths=0.7)
    ax_r.add_collection(stems)
    dots = ax_r.scatter([], [], s=9.0, zorder=3)
    t_env = np.linspace(d["direct_ms"], _IS_RUN_MS, 200)
    ax_r.plot(
        t_env,
        20.0 * np.log10(d["direct_ms"] / t_env),
        color=COLOR_SECONDARY,
        lw=1.2,
        ls="--",
        label=T("$1/r$ spreading"),
    )
    ax_r.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

    # --- bottom right: the count, against the analytic law ---------------
    ax_n = fig.add_subplot(gs[1, 1])
    _grid_axes(ax_n)
    ax_n.set_xlim(0.0, _IS_RUN_MS)
    ax_n.set_ylim(0.0, 400.0)
    ax_n.set_xlabel(T("Time [ms]"), fontsize=8.5)
    ax_n.set_ylabel(T("Arrivals so far"), fontsize=8.5)
    ax_n.tick_params(labelsize=8)
    t_law = np.linspace(0.0, _IS_RUN_MS, 200)
    ax_n.plot(
        t_law,
        4.0 * np.pi / 3.0 * (c * t_law / 1e3) ** 3 / volume,
        color=COLOR_SECONDARY,
        lw=1.4,
        ls="--",
        label=T(r"$(4\pi/3)(ct)^3/V$"),
    )
    (count_line,) = ax_n.plot([], [], color=COLOR_PRIMARY, lw=2.0, label=T("counted"))
    count_txt = ax_n.text(
        0.03,
        0.95,
        "",
        transform=ax_n.transAxes,
        ha="left",
        va="top",
        color=COLOR_FG,
        fontsize=9,
        family="monospace",
    )
    ax_n.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    sweep = _ANIM_FRAMES - _ANIM_HOLD
    lead = int(0.05 * sweep)
    grid_t = np.linspace(0.0, _IS_RUN_MS, 240)

    def update(kf: int) -> tuple[Any, ...]:
        k = min(kf, sweep)
        now = _IS_RUN_MS * np.clip((k - lead) / (sweep - lead), 0.0, 1.0)
        radius = c * now / 1e3
        front.set_radius(max(radius, 0.01))
        radius_txt.set_text(
            T(f"$t$ = {now:4.1f} ms") + "\n" + T(f"$ct$ = {radius:5.1f} m")
        )

        reached = plan_t <= now
        pts = plan_xy[reached]
        lit.set_offsets(pts if pts.size else np.empty((0, 2)))
        lit.set_facecolors([palette[o] for o in d["plan_order"][reached]])  # type: ignore[attr-defined]  # runtime alias of set_facecolor, absent from the stubs
        rays.set_segments(
            [
                [(float(x), float(y)), (float(_IS_RECEIVER[0]), float(_IS_RECEIVER[1]))]
                for x, y in pts
            ]
        )

        arrived = t_ms <= now
        n = int(arrived.sum())
        xy = np.column_stack([t_ms[arrived], level[arrived]])
        dots.set_offsets(xy if n else np.empty((0, 2)))
        dots.set_facecolors([palette[o] for o in order[arrived]])  # type: ignore[attr-defined]  # runtime alias of set_facecolor, absent from the stubs
        stems.set_segments([[(float(a), -26.0), (float(a), float(b))] for a, b in xy])

        shown = grid_t <= now
        counts = np.array([int((t_ms <= g).sum()) for g in grid_t[shown]])
        count_line.set_data(grid_t[shown], counts)
        law = 4.0 * np.pi / 3.0 * (radius**3) / volume
        count_txt.set_text(T(f"counted {n}") + "\n" + T(f"law {law:.0f}"))
        return (front, radius_txt, lit, rays, dots, stems, count_line, count_txt)

    _render_clip(fig, update, output_dir, "anim_image_source_buildup")


# --- ISO 717 reference-curve fit ------------------------------------------
#
# Every static figure of the ratings guide shows the curve where it came to
# rest. The rule that put it there is iterative: shift in 1 dB steps toward
# the measurement until the sum of unfavourable deviations is as large as it
# can be without passing 32.0 dB (ISO 717-1:2020, 4.4; ISO 717-2:2020,
# 4.3.1). The clip walks the steps, including the one that is rejected for
# overshooting and the one past the answer that is legal but not maximal.

#: Airborne example: the 16 one-third-octave R values of the ratings guide.
_R717_AIRBORNE = (
    20.4,
    16.3,
    17.7,
    22.6,
    22.4,
    22.7,
    24.8,
    26.6,
    28.0,
    30.5,
    31.8,
    32.5,
    33.4,
    33.0,
    31.0,
    25.5,
)
#: Impact example: the ISO 717-2 Annex C L'nT spectrum of the same guide.
_R717_IMPACT = (
    62.1,
    63.2,
    63.5,
    66.2,
    68.5,
    70.0,
    71.7,
    73.1,
    73.8,
    73.5,
    73.8,
    73.3,
    73.1,
    73.0,
    72.4,
    71.2,
)
_R717_BANDS = (
    100,
    125,
    160,
    200,
    250,
    315,
    400,
    500,
    630,
    800,
    1000,
    1250,
    1600,
    2000,
    2500,
    3150,
)
#: The cap is one rule for both engines: 2.0 dB per band over 16 bands.
_R717_CAP = 32.0


def _iso717_track(
    measured: tuple[float, ...], *, impact: bool, first: int, last: int
) -> dict[str, Any]:
    """Every curve position the fit visits, with its unfavourable sum.

    ``first``/``last`` are readings of the reference curve at 500 Hz, walked
    in the direction the standard shifts (down toward an airborne
    measurement, up toward an impact one). The accepted position is the one
    with the largest sum not exceeding 32.0 dB, which is what the library
    returns; the walk is built so the frame before it is a rejection.
    """
    from phonometry import building

    y = np.asarray(measured, dtype=float)
    res: Any = (
        building.weighted_impact_rating(y) if impact else building.weighted_rating(y)
    )
    rating = int(res.rating)
    shifted = np.asarray(res.shifted_reference, dtype=float)
    # The unshifted reference is the returned curve moved back by the shift
    # that was applied; its 500 Hz value is the standard's Table 3 anchor.
    base = shifted - (shifted[7] - float(rating))
    step = 1 if impact else -1
    readings = list(range(first, last + step, step))
    frames = []
    for read in readings:
        curve = base + (read - float(rating))
        dev = (y - curve) if impact else (curve - y)
        dev = np.clip(dev, 0.0, None)
        total = float(np.round(dev.sum(), 1))
        frames.append(
            {"read": read, "curve": curve, "sum": total, "ok": total <= _R717_CAP}
        )
    accepted = next(i for i, f in enumerate(frames) if f["read"] == rating)
    return {
        "measured": y,
        "frames": frames,
        "accepted": accepted,
        "rating": rating,
        "result": res,
    }


def animate_iso717_shift(output_dir: str) -> None:
    """The reference curve stepping toward the measurement until the sum of
    unfavourable deviations is as large as it can be without passing 32.0 dB
    -- the iterative fit behind every single number on the ratings page,
    run once for the airborne engine and once for the impact engine, where
    the same rule holds with the deviation's sign reversed.
    """
    from matplotlib.patches import Patch

    T = _translate_str

    # Each walk starts well clear of the measurement and stops one step past
    # the answer, so the frame after the accepted one shows what "as large as
    # possible" rules out: a legal fit whose sum is needlessly small.
    air = _iso717_track(_R717_AIRBORNE, impact=False, first=36, last=29)
    imp = _iso717_track(_R717_IMPACT, impact=True, first=75, last=80)
    x = np.arange(16)

    fig = _anim_figure()
    fig.suptitle(
        T("Fitting the ISO 717 reference curve, one step at a time"),
    )
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0])

    # --- left: the spectrum, the curve and the shaded deviations ---------
    ax_s = fig.add_subplot(gs[:, 0])
    _grid_axes(ax_s)
    ax_s.set_xlim(-0.6, 15.6)
    ax_s.set_xticks(x)
    # Nominal band numbers, not the 1k/2k axis convention: the rating is a
    # band-by-band comparison and the reader counts bands here.
    ax_s.set_xticklabels([f"{b:g}" for b in _R717_BANDS], rotation=45, fontsize=7.5)
    ax_s.tick_params(labelsize=8)
    ax_s.set_xlabel(T("One-third-octave band [Hz]"), fontsize=9)
    (meas_line,) = ax_s.plot(
        [],
        [],
        color=COLOR_PRIMARY,
        lw=2.2,
        marker="o",
        ms=4.0,
        zorder=4,
        label=T("measured spectrum"),
    )
    (ref_line,) = ax_s.plot(
        [],
        [],
        color=COLOR_FG,
        lw=1.8,
        ls="--",
        marker="s",
        ms=3.4,
        zorder=3,
        label=T("reference curve, as shifted"),
    )
    shade: dict[str, FillBetweenPolyCollection | None] = {"art": None}
    (read_dot,) = ax_s.plot(
        [], [], color=COLOR_TERTIARY, marker="o", ms=11.0, ls="none", zorder=5
    )
    read_txt = ax_s.text(
        0.0,
        0.0,
        "",
        ha="left",
        va="center",
        color=COLOR_TERTIARY,
        fontsize=9.5,
        family="monospace",
        zorder=6,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": plt.rcParams["figure.facecolor"],
            "edgecolor": "none",
            "alpha": 0.85,
        },
    )
    act_txt = ax_s.set_title("", fontsize=9.0, color=COLOR_FG)

    # --- top right: the sum against curve position, with the 32.0 dB cap -
    ax_c = fig.add_subplot(gs[0, 1])
    _grid_axes(ax_c)
    ax_c.set_ylabel(T("Sum of unfavourable\ndeviations [dB]"), fontsize=8.5)
    ax_c.set_xlabel(T("Reference curve read at 500 Hz [dB]"), fontsize=8.5)
    ax_c.tick_params(labelsize=8)
    ax_c.axhline(
        _R717_CAP,
        color=COLOR_FG,
        lw=1.6,
        ls="--",
        zorder=3,
        label=T("cap 32.0 dB = 2.0 dB per band"),
    )
    n_bars = max(len(air["frames"]), len(imp["frames"]))
    cap_bars = ax_c.bar(
        np.arange(n_bars), np.zeros(n_bars), width=0.66, color=COLOR_MUTED, alpha=0.9
    )
    ax_c.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

    # --- bottom right: what the fit has decided so far -------------------
    ax_v = fig.add_subplot(gs[1, 1])
    _schematic_axes(ax_v, (0.0, 10.0), (0.0, 10.0))
    # The key lives here rather than over the spectrum: the two curves swap
    # places between the airborne and impact acts, so any in-panel legend
    # position that is clear in one is covered in the other.
    ax_v.legend(
        handles=[
            meas_line,
            ref_line,
            Patch(
                facecolor=COLOR_SECONDARY,
                alpha=0.30,
                label=T("unfavourable deviations"),
            ),
        ],
        loc="upper left",
        fontsize=8.0,
        framealpha=0.0,
        handlelength=2.4,
        borderpad=0.0,
    )
    verdict_txt = ax_v.text(
        0.2,
        6.2,
        "",
        ha="left",
        va="top",
        color=COLOR_FG,
        fontsize=10.0,
        family="monospace",
        linespacing=1.7,
    )

    sweep = _ANIM_FRAMES - _ANIM_HOLD
    a1_end = int(0.52 * sweep)
    a2_start = int(0.58 * sweep)

    def schedule(track: dict[str, Any], n_frames: int) -> np.ndarray:
        """Frame -> step index, dwelling on the steps that decide it.

        The last rejected step and the accepted one carry the whole
        argument, so they get several times the screen time of a step that
        is merely on the way. The walk then takes one step past the answer,
        to show what "as large as possible" rules out, and **returns to the
        accepted position**, so the act settles on the fit rather than on
        the counter-example.
        """
        acc = track["accepted"]
        n = len(track["frames"])
        visits = list(range(n))
        weights = [1.0] * n
        weights[max(acc - 1, 0)] = 3.0
        weights[acc] = 3.5
        if acc + 1 < n:
            weights[acc + 1] = 2.5
            visits.append(acc)
            weights.append(3.0)
        w = np.asarray(weights, dtype=float)
        edges = np.cumsum(w) / w.sum()
        pos = np.searchsorted(
            edges, np.linspace(0.0, 1.0, n_frames, endpoint=False), side="right"
        ).clip(0, len(visits) - 1)
        return np.asarray([visits[int(p)] for p in pos], dtype=int)

    air_sched = schedule(air, a1_end)
    imp_sched = schedule(imp, sweep - a2_start)
    # Bars stay on screen once a position has been tried, so stepping back
    # to the accepted fit does not erase the counter-example beside it.
    air_shown = np.maximum.accumulate(air_sched)
    imp_shown = np.maximum.accumulate(imp_sched)

    def draw(
        track: dict[str, Any], idx: int, shown: int, *, impact: bool, act: str
    ) -> None:
        y = track["measured"]
        st = track["frames"][idx]
        curve = st["curve"]
        meas_line.set_data(x, y)
        ref_line.set_data(x, curve)
        if shade["art"] is not None:
            shade["art"].remove()
        hi, lo = (y, curve) if impact else (curve, y)
        # One colour throughout: the shading means "unfavourable", not
        # "rejected". Whether the position is accepted is the bar chart's
        # job, and mixing the two would make the left panel's key move.
        shade["art"] = ax_s.fill_between(
            x,
            lo,
            hi,
            where=hi > lo,
            interpolate=True,
            zorder=2,
            color=COLOR_SECONDARY,
            alpha=0.30,
        )
        read_dot.set_data([7.0], [curve[7]])
        read_txt.set_position((7.6, curve[7]))
        read_txt.set_text(T(f"{st['read']:.0f} dB at 500 Hz"))
        lo_y, hi_y = float(min(y.min(), curve.min())), float(max(y.max(), curve.max()))
        pad = 0.18 * (hi_y - lo_y)
        ax_s.set_ylim(lo_y - pad, hi_y + pad)
        ax_s.set_ylabel(
            T("Normalized impact level $L\u2032_{\\mathrm{n}T}$ [dB]")
            if impact
            else T("Sound reduction index $R$ [dB]"),
            fontsize=9,
        )
        act_txt.set_text(act)

        reads = [f["read"] for f in track["frames"]]
        ax_c.set_xlim(-0.7, len(reads) - 0.3)
        ax_c.set_xticks(np.arange(len(reads)))
        ax_c.set_xticklabels([f"{r:d}" for r in reads], fontsize=7.5)
        ax_c.set_ylim(0.0, max(f["sum"] for f in track["frames"]) * 1.15)
        for j, bar in enumerate(cap_bars):
            if j > shown or j >= len(track["frames"]):
                bar.set_height(0.0)
                continue
            f = track["frames"][j]
            bar.set_height(f["sum"])
            if j == track["accepted"] and shown >= track["accepted"]:
                bar.set_color(COLOR_TERTIARY)
            elif f["ok"]:
                bar.set_color(COLOR_MUTED)
            else:
                bar.set_color(COLOR_SECONDARY)

    def verdict(track: dict[str, Any], idx: int, *, impact: bool) -> None:
        st = track["frames"][idx]
        res = track["result"]
        lines = [
            T(f"step {idx + 1} of {len(track['frames'])}"),
            T(f"sum = {st['sum']:.1f} dB"),
        ]
        if not st["ok"]:
            lines.append(T("over the cap: shift again"))
        elif idx == track["accepted"]:
            name = (
                T("$L\u2032_{\\mathrm{n}T,\\mathrm{w}}$")
                if impact
                else T(r"$R_\mathrm{w}$")
            )
            lines.append(T("largest sum still under the cap"))
            lines.append(f"{name} = {track['rating']:d} dB")
            if impact:
                lines.append(
                    T(rf"$C_\mathrm{{I}}$ = {_fmt_minus(int(res.ci), '+d')} dB")
                )
            else:
                lines.append(
                    T(
                        f"$C$ = {_fmt_minus(int(res.c), '+d')} dB, "
                        rf"$C_\mathrm{{tr}}$ = {_fmt_minus(int(res.ctr), '+d')} dB"
                    )
                )
        else:
            lines.append(T("legal, but the sum is smaller:"))
            lines.append(T("this is one step too far"))
        verdict_txt.set_text("\n".join(lines))

    def update(kf: int) -> tuple[Any, ...]:
        k = min(kf, sweep)
        # Two lines, measured rather than eyeballed: on one line the Spanish
        # ran 143 px off the left edge of the 2400 px canvas and the English
        # 43 px off the right.
        act1 = T(
            "ISO 717-1: the curve steps down toward the measurement;\n"
            "an unfavourable deviation is a band that falls below it"
        )
        if k < a1_end:
            idx, shown = int(air_sched[k]), int(air_shown[k])
            draw(air, idx, shown, impact=False, act=act1)
            verdict(air, idx, impact=False)
        elif k < a2_start:
            draw(air, air["accepted"], len(air["frames"]) - 1, impact=False, act=act1)
            verdict(air, air["accepted"], impact=False)
        else:
            j = min(k - a2_start, len(imp_sched) - 1)
            idx, shown = int(imp_sched[j]), int(imp_shown[j])
            draw(
                imp,
                idx,
                shown,
                impact=True,
                act=T(
                    "ISO 717-2: the same rule with the sign reversed;\n"
                    "the curve steps up, and a band above it is "
                    "unfavourable"
                ),
            )
            verdict(imp, idx, impact=True)
        return (meas_line, ref_line, read_dot, read_txt, verdict_txt)

    _render_clip(fig, update, output_dir, "anim_iso717_shift")


# --- block integrator against exponential detector -------------------------
#
# The case study of start/why-phonometry.mdx turns on one sentence: a block
# integrator "can only be right where the burst happens to fill a block". The
# page shows the exponential detector landing on the IEC 61672-1 Table 4
# values and nothing about the alignment the other detector depends on. That
# dependence is a one-parameter sweep over where the burst falls in the block
# grid, so it is the clip and not the figure.

_BVE_FS = 48_000
_BVE_FREQ = 4_000.0  # the Table 4 excitation
_BVE_BLOCK = 0.125  # the block grid, numerically tau_F
_BVE_SPAN = 0.75  # seconds of record drawn (six blocks)
_BVE_BURST_T0 = 0.25  # earliest burst start, on a block boundary
_BVE_OFFSETS = 64  # alignments sampled across one block period
#: (burst duration, IEC 61672-1 Table 4 target, class 1 tolerance).
_BVE_ACTS = ((0.050, -4.8, 1.0), (0.200, -1.0, 0.5))


def _block_vs_exponential_data() -> dict[str, Any]:
    """Both detectors' readings at every alignment, for both bursts.

    The reference is the steady Fast level of the same continuous 4 kHz
    tone, averaged over its last half second, exactly as the guide's
    procedure section defines it -- so the readings below are the
    standard's L_AFmax - L_A and are directly comparable with Table 4.
    """
    from phonometry import filters

    fs = _BVE_FS
    n_span = round(_BVE_SPAN * fs)
    n_block = round(_BVE_BLOCK * fs)
    t_steady = np.arange(3 * fs) / fs
    steady = np.sin(2.0 * np.pi * _BVE_FREQ * t_steady)
    ref = float(
        np.mean(filters.time_weighting(steady, fs, mode="fast")[int(2.5 * fs) :])
    )

    t = np.arange(n_span) / fs
    offsets = np.linspace(0.0, _BVE_BLOCK, _BVE_OFFSETS, endpoint=False)
    acts = []
    for seconds, target, limit in _BVE_ACTS:
        n_burst = round(seconds * fs)
        env_db, blk_db, exp_db, top_db = [], [], [], []
        for off in offsets:
            start = round((_BVE_BURST_T0 + off) * fs)
            x = np.zeros(n_span)
            x[start : start + n_burst] = np.sin(
                2.0 * np.pi * _BVE_FREQ * t[start : start + n_burst]
            )
            env = filters.time_weighting(x, fs, mode="fast")
            env_db.append(10.0 * np.log10(np.maximum(env, 1e-12) / ref))
            n_full = n_span // n_block
            per = (x[: n_full * n_block].reshape(n_full, n_block) ** 2).mean(axis=1)
            blk_db.append(10.0 * np.log10(np.maximum(per, 1e-12) / ref))
            exp_db.append(float(env_db[-1].max()))
            top_db.append(float(blk_db[-1].max()))
        acts.append(
            {
                "seconds": seconds,
                "target": target,
                "limit": limit,
                "n_burst": n_burst,
                "env": np.asarray(env_db),
                "blocks": np.asarray(blk_db),
                "exp": np.asarray(exp_db),
                "blk": np.asarray(top_db),
            }
        )
    return {
        "t": t,
        "offsets": offsets,
        "n_block": n_block,
        "acts": acts,
        "n_span": n_span,
    }


@lru_cache(maxsize=1)
def _bve_cached() -> dict[str, Any]:
    """One data build per process: 128 Fast-envelope runs are paid once,
    not once per language x theme variant.
    """
    return _block_vs_exponential_data()


def animate_block_vs_exponential(output_dir: str) -> None:
    """The same tone burst slid across a 125 ms block boundary. The
    exponential Fast detector answers the same number at every alignment
    because the burst duration is in its differential equation; the block
    integrator's answer swings by nearly 3 dB, because the alignment is all
    it has. The case study's central claim, measured.
    """
    T = _translate_str

    d = _bve_cached()
    t, offsets = d["t"], d["offsets"]
    n_span = d["n_span"]

    fig = _anim_figure()
    fig.suptitle(
        T("One burst, two detectors, and the block grid underneath"),
    )
    gs = fig.add_gridspec(2, 2, width_ratios=[1.28, 1.0], height_ratios=[1.0, 1.15])

    # --- top left: the record, the burst and the 125 ms block grid -------
    ax_r = fig.add_subplot(gs[0, 0])
    _grid_axes(ax_r)
    ax_r.set_xlim(0.0, _BVE_SPAN)
    ax_r.set_ylim(-1.35, 1.35)
    ax_r.set_ylabel(T("Sound pressure"), fontsize=8.5)
    ax_r.tick_params(labelsize=8, labelbottom=False)
    for k in range(int(_BVE_SPAN / _BVE_BLOCK) + 1):
        ax_r.axvline(k * _BVE_BLOCK, color=COLOR_MUTED, lw=1.0, ls=":")
    for k in range(0, int(_BVE_SPAN / _BVE_BLOCK), 2):
        ax_r.axvspan(
            k * _BVE_BLOCK, (k + 1) * _BVE_BLOCK, color=COLOR_GRID, alpha=0.45, zorder=0
        )
    (wave,) = ax_r.plot([], [], color=COLOR_PRIMARY, lw=0.7)
    ax_r.set_title(
        T("the shaded strips are the 125 ms blocks"), fontsize=8.5, color=COLOR_FG
    )
    burst_txt = ax_r.text(
        0.0, 1.16, "", ha="center", va="bottom", color=COLOR_FG, fontsize=8.5
    )
    burst_arrow = _make_arrow(ax_r, COLOR_FG, scale=8.0)
    burst_arrow.set_arrowstyle("<|-|>", head_length=0.5, head_width=0.28)

    # --- bottom left: what each detector makes of it ---------------------
    ax_d = fig.add_subplot(gs[1, 0])
    _grid_axes(ax_d)
    ax_d.set_xlim(0.0, _BVE_SPAN)
    ax_d.set_xlabel(T("Time [s]"), fontsize=8.5)
    ax_d.set_ylabel(T("Level re steady tone [dB]"), fontsize=8.5)
    ax_d.tick_params(labelsize=8)
    (env_line,) = ax_d.plot(
        [], [], color=COLOR_PRIMARY, lw=2.0, label=T("exponential Fast envelope")
    )
    (blk_line,) = ax_d.plot(
        [],
        [],
        color=COLOR_TERTIARY,
        lw=2.0,
        drawstyle="steps-post",
        label=T("block Leq, 125 ms slices"),
    )
    target_line = ax_d.axhline(
        0.0, color=COLOR_FG, lw=1.3, ls="--", label=T("IEC 61672-1 Table 4 target")
    )
    (env_peak,) = ax_d.plot(
        [], [], color=COLOR_PRIMARY, marker="o", ms=7.0, ls="none", zorder=5
    )
    (blk_peak,) = ax_d.plot(
        [], [], color=COLOR_TERTIARY, marker="s", ms=7.0, ls="none", zorder=5
    )
    # Lower left: nothing is drawn before the burst starts at 0.25 s in
    # either act, and lower right is where the exponential tail decays.
    ax_d.legend(loc="lower left", fontsize=7.5, framealpha=0.9)

    # --- top right: the reading against alignment, built up --------------
    ax_a = fig.add_subplot(gs[0, 1])
    _grid_axes(ax_a)
    ax_a.set_xlim(0.0, _BVE_BLOCK * 1e3)
    ax_a.set_xticks([0.0, 25.0, 50.0, 75.0, 100.0, 125.0])
    ax_a.set_xlabel(T("Burst start within the block [ms]"), fontsize=8.5)
    ax_a.set_ylabel(T("Reading [dB]"), fontsize=8.5)
    ax_a.tick_params(labelsize=8)
    corridor: dict[str, Rectangle | None] = {"art": None}
    (exp_trace,) = ax_a.plot([], [], color=COLOR_PRIMARY, lw=2.4)
    (blk_trace,) = ax_a.plot([], [], color=COLOR_TERTIARY, lw=2.4)
    (now_dot,) = ax_a.plot(
        [], [], color=COLOR_SECONDARY, marker="o", ms=7.0, ls="none", zorder=5
    )

    # --- bottom right: the two answers, and the spread so far ------------
    ax_v = fig.add_subplot(gs[1, 1])
    _schematic_axes(ax_v, (0.0, 10.0), (0.0, 10.0))
    # 9.2 pt, measured: at 10 pt the English readout ran 13 px and the
    # Spanish 29 px off the right edge of the 2400 px canvas.
    verdict_txt = ax_v.text(
        0.1,
        9.6,
        "",
        ha="left",
        va="top",
        color=COLOR_FG,
        fontsize=9.2,
        family="monospace",
        linespacing=1.65,
    )

    sweep = _ANIM_FRAMES - _ANIM_HOLD
    a1_end = int(0.54 * sweep)

    def update(kf: int) -> tuple[Any, ...]:
        k = min(kf, sweep)
        if k < a1_end:
            act, frac = d["acts"][0], k / max(a1_end - 1.0, 1.0)
        else:
            act = d["acts"][1]
            frac = (k - a1_end) / max(sweep - a1_end - 1.0, 1.0)
        i = int(np.clip(round(frac * (len(offsets) - 1)), 0, len(offsets) - 1))
        target, limit = act["target"], act["limit"]

        start = round(float(_BVE_BURST_T0 + offsets[i]) * _BVE_FS)
        x = np.zeros(n_span)
        x[start : start + act["n_burst"]] = np.sin(
            2.0 * np.pi * _BVE_FREQ * t[start : start + act["n_burst"]]
        )
        wave.set_data(t, x)
        t0 = _BVE_BURST_T0 + float(offsets[i])
        t1 = t0 + act["seconds"]
        burst_arrow.set_positions((t0, 1.12), (t1, 1.12))
        burst_txt.set_position(((t0 + t1) / 2.0, 1.16))
        burst_txt.set_text(T(f"4 kHz burst, {act['seconds'] * 1e3:.0f} ms"))

        env_line.set_data(t, act["env"][i])
        edges = np.arange(len(act["blocks"][i]) + 1) * _BVE_BLOCK
        blk_line.set_data(edges, np.append(act["blocks"][i], act["blocks"][i][-1]))
        j_env = int(np.argmax(act["env"][i]))
        env_peak.set_data([t[j_env]], [act["env"][i][j_env]])
        j_blk = int(np.argmax(act["blocks"][i]))
        blk_peak.set_data([(j_blk + 0.5) * _BVE_BLOCK], [act["blocks"][i][j_blk]])
        target_line.set_ydata([target, target])
        ax_d.set_ylim(target - 14.0, 4.0)

        ax_a.set_ylim(target - 3.6, target + 2.2)
        if corridor["art"] is not None:
            corridor["art"].remove()
        corridor["art"] = ax_a.axhspan(
            target - limit, target + limit, color=COLOR_MUTED, alpha=0.28, zorder=0
        )
        ms = offsets[: i + 1] * 1e3
        exp_trace.set_data(ms, act["exp"][: i + 1])
        blk_trace.set_data(ms, act["blk"][: i + 1])
        now_dot.set_data([offsets[i] * 1e3], [act["blk"][i]])
        ax_a.set_title(
            T(
                f"class 1 is {_fmt_minus(limit, '.1f')} dB "
                f"about {_fmt_minus(target, '.1f')} dB"
            ),
            fontsize=8.5,
            color=COLOR_FG,
        )

        seen_b = act["blk"][: i + 1]
        seen_e = act["exp"][: i + 1]
        lines = [
            T(
                f"burst {act['seconds'] * 1e3:.0f} ms, "
                f"IEC target {_fmt_minus(target, '.1f')} dB"
            ),
            T(
                f"exponential  {_fmt_minus(act['exp'][i], '6.2f')} dB "
                f"({_fmt_minus(act['exp'][i] - target, '+.2f')})"
            ),
            T(
                f"block Leq    {_fmt_minus(act['blk'][i], '6.2f')} dB "
                f"({_fmt_minus(act['blk'][i] - target, '+.2f')})"
            ),
            T(
                f"spread so far, exponential: "
                f"{float(seen_e.max() - seen_e.min()):.2f} dB"
            ),
            T(
                f"spread so far, block Leq:   "
                f"{float(seen_b.max() - seen_b.min()):.2f} dB"
            ),
        ]
        if abs(act["blk"][i] - target) > limit:
            lines.append(T("the block reading leaves the corridor"))
        verdict_txt.set_text("\n".join(lines))
        return (
            wave,
            burst_txt,
            burst_arrow,
            env_line,
            blk_line,
            env_peak,
            blk_peak,
            exp_trace,
            blk_trace,
            now_dot,
            verdict_txt,
        )

    _render_clip(fig, update, output_dir, "anim_block_vs_exponential")


# --- gain before feedback, round trip by round trip ------------------------
#
# The criterion Z_S + G_S = 0 (Long, Architectural Acoustics 2e, Eq. (18.16))
# is a convergence condition on a geometric series, and the page's figures are
# a level-bookkeeping bar stack. What no static figure can show is the series
# itself: whether successive round trips shrink or hold their size. Three
# cases, all of them the page's own auditorium.

_FBH_ROUND_TRIPS = 14
#: (open-loop gain Z_S, open microphones, caption key). L(H-M) = 76 dB and
#: L(H-L) = 80 dB throughout, which is the guide's worked auditorium.
_FBH_CASES = ((-6.0, 1), (-6.0, 4), (-2.0, 4))
_FBH_LHM, _FBH_LHL = 76.0, 80.0


def _feedback_howl_data() -> list[dict[str, Any]]:
    """Copy amplitudes and running totals for the three loop gains."""
    from phonometry import electroacoustics as ea

    n = np.arange(_FBH_ROUND_TRIPS + 1)
    out = []
    for zs, mics in _FBH_CASES:
        res = ea.feedback_stability(zs, _FBH_LHM, _FBH_LHL, open_microphones=mics)
        loop = float(res.loop_gain)
        g = 10.0 ** (loop / 20.0)
        copies = g**n
        totals = np.cumsum(copies)
        out.append(
            {
                "zs": zs,
                "mics": mics,
                "loop": loop,
                "g": g,
                "copies_db": 20.0 * np.log10(copies),
                "totals_db": 20.0 * np.log10(totals),
                "limit": (1.0 / (1.0 - g)) if g < 1.0 else float("inf"),
                "stable": bool(res.is_stable),
                "headroom": float(res.headroom),
            }
        )
    return out


@lru_cache(maxsize=1)
def _fbh_cached() -> list[dict[str, Any]]:
    """One data build per process."""
    return _feedback_howl_data()


def animate_feedback_howl(output_dir: str) -> None:
    """One burst goes round the reinforcement loop again and again. With
    Long's 10 dB margin each copy is a third of the last and the sum settles
    3.3 dB above the direct sound; four open microphones cost 6 dB and the
    same loop takes far longer to settle 8.7 dB up; 4 dB more system gain
    puts the loop gain at unity and the copies stop shrinking, which is the
    howl. Gain before feedback is a convergence condition, not a level.
    """
    from matplotlib.patches import Circle

    T = _translate_str

    cases = _fbh_cached()
    n_rt = _FBH_ROUND_TRIPS
    idx = np.arange(n_rt + 1)

    fig = _anim_figure()
    fig.suptitle(
        T("Gain before feedback is a convergence condition"),
    )
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 1.0])

    # --- left: the loop, with one copy travelling the feedback path ------
    ax_g = fig.add_subplot(gs[:, 0])
    # The limits span the same 12.2 units on both axes and the cell is very
    # nearly square, so the capsule stays round without an explicit equal
    # aspect -- which fought constrained_layout and squeezed the right
    # column until its legend ran off the canvas.
    _schematic_axes(ax_g, (-1.4, 10.8), (-7.4, 4.8))
    talker, mic = (0.5, 0.0), (2.6, 0.0)
    spk, listener = (5.6, 3.0), (9.8, 0.0)
    _draw_mic(ax_g, mic[0], mic[1], direction=-1, size=0.85)
    _draw_speaker(ax_g, spk[0], spk[1], size=1.4, direction=1)
    ax_g.plot(
        [talker[0]], [talker[1]], marker="o", ms=13, color=COLOR_SECONDARY, ls="none"
    )
    ax_g.plot(
        [listener[0]], [listener[1]], marker="o", ms=13, color=COLOR_TERTIARY, ls="none"
    )
    # Nothing is labelled above the microphone: the feedback path arrives
    # there from the upper right and any label in that wedge crosses it.
    ax_g.text(
        talker[0],
        talker[1] - 0.5,
        T("talker,\n0.3 m from the microphone"),
        ha="center",
        va="top",
        color=COLOR_FG,
        fontsize=9,
    )
    ax_g.text(
        mic[0] + 0.55,
        mic[1] - 0.55,
        T("microphone"),
        ha="left",
        va="top",
        color=COLOR_FG,
        fontsize=9,
    )
    ax_g.text(
        listener[0],
        listener[1] - 0.5,
        T("listener,\n12 m away"),
        ha="center",
        va="top",
        color=COLOR_FG,
        fontsize=9,
    )
    ax_g.text(
        spk[0] + 0.9,
        spk[1] + 0.1,
        T("loudspeaker"),
        ha="left",
        va="center",
        color=COLOR_FG,
        fontsize=9,
    )
    # The electrical leg is a cable, drawn round the outside, so it does not
    # lie on top of the acoustic feedback path it closes.
    ax_g.plot(
        [mic[0], mic[0], spk[0], spk[0]],
        [mic[1] - 0.45, -2.0, -2.0, spk[1] - 0.55],
        color=COLOR_MUTED,
        lw=1.4,
    )
    ax_g.text(
        (mic[0] + spk[0]) / 2.0 + 0.3,
        -2.2,
        T("microphone, mixer, amplifier"),
        ha="center",
        va="top",
        color=COLOR_FG,
        fontsize=9,
    )
    ax_g.plot(
        [spk[0] + 0.5, listener[0]],
        [spk[1] - 0.3, listener[1] + 0.4],
        color=COLOR_MUTED,
        lw=1.4,
    )
    fb_a = (spk[0] - 0.3, spk[1] - 0.6)
    fb_b = (mic[0] + 0.25, mic[1] + 0.35)
    ax_g.plot(
        [fb_a[0], fb_b[0]], [fb_a[1], fb_b[1]], color=COLOR_PRIMARY, lw=1.6, ls="--"
    )
    ang = float(np.degrees(np.arctan2(fb_a[1] - fb_b[1], fb_a[0] - fb_b[0])))
    # rotation_mode="anchor" so the offset below the line is exact: without
    # it matplotlib aligns before rotating and the label swings across the
    # cable underneath.
    # Short, because the path is short: the longer wording ran onto the
    # microphone at one end and the loudspeaker at the other, and the
    # sentence it came from belongs in the prose beside the clip.
    perp = np.deg2rad(ang)
    ax_g.text(
        (fb_a[0] + fb_b[0]) / 2.0 + 0.34 * float(np.sin(perp)),
        (fb_a[1] + fb_b[1]) / 2.0 - 0.34 * float(np.cos(perp)),
        T("the feedback path"),
        ha="center",
        va="top",
        rotation_mode="anchor",
        color=COLOR_PRIMARY,
        fontsize=9,
        rotation=ang,
    )
    copy_dot = Circle(
        spk, 0.18, facecolor=COLOR_PRIMARY, edgecolor="none", alpha=0.85, zorder=6
    )
    ax_g.add_patch(copy_dot)
    case_txt = ax_g.text(
        -1.2,
        -3.4,
        "",
        ha="left",
        va="top",
        color=COLOR_FG,
        fontsize=10.0,
        family="monospace",
        linespacing=1.7,
    )
    # The act caption lives inside this panel, not as its title: equal aspect
    # shrinks the axes box to a square, and a title centred on it ran 311 px
    # off the left edge of the figure.
    act_txt = ax_g.text(
        -1.3,
        4.7,
        "",
        ha="left",
        va="top",
        color=COLOR_FG,
        fontsize=9.5,
        fontstyle="italic",
    )

    # --- top right: the copies, one stem per round trip ------------------
    ax_c = fig.add_subplot(gs[0, 1])
    _grid_axes(ax_c)
    ax_c.set_xlim(-0.7, n_rt + 0.7)
    ax_c.set_xticks(np.arange(0, n_rt + 1, 2))
    ax_c.set_ylim(-46.0, 6.0)
    ax_c.set_xlabel(T("Round trip number"), fontsize=8.5)
    ax_c.set_ylabel(T("Copy level re direct [dB]"), fontsize=8.5)
    ax_c.tick_params(labelsize=8)
    stems = LineCollection([], colors=COLOR_PRIMARY, linewidths=2.4)
    ax_c.add_collection(stems)
    heads = ax_c.scatter([], [], s=22.0, color=COLOR_PRIMARY, zorder=4)
    ax_c.axhline(0.0, color=COLOR_FG, lw=1.0, ls="-.", alpha=0.6)

    # --- bottom right: the sum, with the earlier cases left on screen ----
    ax_t = fig.add_subplot(gs[1, 1])
    _grid_axes(ax_t)
    ax_t.set_xlim(-0.7, n_rt + 0.7)
    ax_t.set_xticks(np.arange(0, n_rt + 1, 2))
    ax_t.set_ylim(-1.0, 26.0)
    ax_t.set_xlabel(T("Round trips summed"), fontsize=8.5)
    ax_t.set_ylabel(T("Total re direct [dB]"), fontsize=8.5)
    ax_t.tick_params(labelsize=8)
    colors = (COLOR_TERTIARY, COLOR_QUATERNARY, COLOR_SECONDARY)
    labels = (T("10 dB margin"), T("four microphones"), T("loop at unity"))
    total_lines = []
    for c, lab in zip(colors, labels, strict=True):
        (ln,) = ax_t.plot([], [], color=c, lw=2.2, marker="o", ms=3.2, label=lab)
        total_lines.append(ln)
    # One column: the three-across form did not fit the Spanish, and the
    # upper left of this panel is empty in all three cases.
    ax_t.legend(loc="upper left", fontsize=7.5, framealpha=0.9, handlelength=1.4)
    limit_lines = []
    for c in colors[:2]:
        # The code that draws the clips is hashed into
        # scripts/animation_fingerprints.txt, so rewriting this asks for a
        # re-render that would draw the very same frames.
        limit_lines.append(ax_t.axhline(0.0, color=c, lw=1.1, ls=":", visible=False))  # noqa: PERF401

    sweep = _ANIM_FRAMES - _ANIM_HOLD
    per_act = sweep // 3

    def update(kf: int) -> tuple[Any, ...]:
        k = min(kf, sweep - 1)
        a = min(k // per_act, 2)
        within = (k - a * per_act) / max(per_act - 1.0, 1.0)
        case = cases[a]
        pos = float(np.clip(within, 0.0, 1.0)) * n_rt
        n_done = int(np.floor(pos))
        frac = pos - n_done

        # one copy in flight along the dashed feedback path
        amp = case["g"] ** max(n_done, 0)
        # The copy travels the middle of the path only, and stays small: at
        # unity the amplitude never decays, so a marker sized for the whole
        # path sat on the microphone symbol for a third of the act.
        span = 0.10 + 0.78 * frac
        px = fb_a[0] + (fb_b[0] - fb_a[0]) * span
        py = fb_a[1] + (fb_b[1] - fb_a[1]) * span
        copy_dot.set_center((px, py))
        copy_dot.set_radius(0.07 + 0.20 * float(np.clip(amp, 0.0, 1.4)))

        seen = idx[: n_done + 1]
        cdb = np.maximum(case["copies_db"][: n_done + 1], -46.0)
        stems.set_segments(
            [
                [(float(i), -46.0), (float(i), float(v))]
                for i, v in zip(seen, cdb, strict=True)
            ]
        )
        heads.set_offsets(np.column_stack([seen, cdb]))
        for j, ln in enumerate(total_lines):
            if j < a:
                ln.set_data(idx, case_totals[j])
                ln.set_alpha(0.45)
            elif j == a:
                ln.set_data(seen, case["totals_db"][: n_done + 1])
                ln.set_alpha(1.0)
            else:
                ln.set_data([], [])
        for j, ll in enumerate(limit_lines):
            if j <= a and np.isfinite(cases[j]["limit"]):
                ll.set_ydata([20.0 * np.log10(cases[j]["limit"])] * 2)
                ll.set_visible(True)

        mics = case["mics"]
        act = (
            T("Long's 10 dB margin: each copy is a third of the last")
            if a == 0
            else T(r"Four open microphones add $10\,\mathrm{lg}\,4$ = 6 dB to the loop")
            if a == 1
            else T("Four more decibels of system gain: the loop reaches unity")
        )
        act_txt.set_text(act)
        lines = [
            T(
                rf"$Z_\mathrm{{S}}$ = {_fmt_minus(case['zs'], '.0f')} dB, "
                f"{mics:d} open microphone(s)"
            ),
            T(
                r"loop gain $Z_\mathrm{S} + G_\mathrm{S}$ = "
                f"{_fmt_minus(case['loop'], '+.1f')} dB"
            ),
            T(f"each round trip is x {case['g']:.3f}"),
        ]
        if np.isfinite(case["limit"]):
            lines.append(
                T(f"sum converges to {20.0 * np.log10(case['limit']):+.2f} dB")
            )
        else:
            lines.append(T("the sum does not converge: this is the howl"))
        case_txt.set_text("\n".join(lines))
        return (copy_dot, stems, heads, case_txt, *total_lines)

    case_totals = [c["totals_db"] for c in cases]
    _render_clip(fig, update, output_dir, "anim_feedback_howl")
