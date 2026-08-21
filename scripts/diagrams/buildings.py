#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Diagrams of the buildings guides: insulation, rooms and design prediction.

One subject: sound between and inside the rooms of a building. The
insulation diagrams draw the field and laboratory measurements that grade a
separating element, the room diagrams draw what is measured or predicted
inside one enclosure, and the design diagrams draw the prediction models that
put the two together before anything is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .parts import _accel, _accel_wall, _rot_arrow, _spring_v

if TYPE_CHECKING:
    from .canvas import SVG, Theme

#: Width, in px, a stage-box title has to leave inside the box at the larger
#: size to keep it. It is the threshold the size decision is taken on, not a
#: promise: a title wider than the box even at the smaller size is a
#: composition to rewrite, and the fit gate is what catches one.
_BOX_PAD = 24.0

# ---------------------------------------------------------------------------
# d8 - Airborne sound insulation setup (ISO 16283-1)
# ---------------------------------------------------------------------------


def _d_insulation_setup(s: SVG, th: Theme) -> None:
    top, bot = 90.0, 470.0

    # Two rooms in plan view separated by the test partition.
    s.rect(70, top, 375, bot - top, th.panel, th.fg, rx=6, sw=3)
    s.rect(465, top, 365, bot - top, th.panel, th.fg, rx=6, sw=3)
    s.rect(445, top, 20, bot - top, th.secondary, th.fg, sw=2)  # partition (S)
    s.text(455, 80, "Test partition", 17, th.secondary, bold=True)

    s.text(90, top + 32, "Source room", 19, th.fg, bold=True, anchor="start")
    s.text(90, top + 58, "$L_1$", 17, th.muted, anchor="start")
    s.text(486, top + 32, "Receiving room", 19, th.fg, bold=True, anchor="start")
    s.text(486, top + 58, "$L_2$ , $T$", 17, th.muted, anchor="start")

    # Loudspeaker in a corner of the source room (bottom-left).
    lsx, lsy = 150.0, 405.0
    for r in (40, 66, 92):
        s.path(
            f"M {lsx + r * 0.22:.1f} {lsy - r:.1f} "
            f"A {r} {r} 0 0 1 {lsx + r:.1f} {lsy - r * 0.22:.1f}",
            stroke=th.accent,
            sw=1.6,
        )
    s.rect(lsx - 26, lsy - 30, 52, 60, th.panel, th.primary, rx=6, sw=2)
    s.circle(lsx, lsy - 10, 12, th.primary)
    s.circle(lsx, lsy - 10, 5, th.bg)
    s.circle(lsx, lsy + 16, 7, th.primary)
    s.text(lsx, lsy + 52, "Loudspeaker", 17, th.fg, bold=True)

    # Microphone positions (five per room, in the central zone).
    src_mics = [(150, 315), (255, 250), (360, 300), (300, 360), (390, 205)]
    rec_mics = [(590, 160), (653, 160), (560, 290), (690, 380), (785, 300)]
    for mics in (src_mics, rec_mics):
        for mx, my in mics:
            s.circle(mx, my, 8, th.fg)
            s.circle(mx, my, 3, th.bg)
    s.text(268, 172, "microphone positions", 15, th.muted)
    s.text(636, 430, "microphone positions", 15, th.muted)

    # Normative minimum separations (ISO 16283-1, 7.6 and 7.2.2).
    s.dim(150, 395, 150, 317, "≥ 1.0 m", offset=-42, size=17)  # 7.6c
    s.dim(178, 405, 443, 405, "≥ 1.0 m", offset=0, size=17)  # 7.2.2
    s.dim(590, 160, 653, 160, "≥ 0.7 m", offset=42, size=17)  # 7.6a
    s.dim(785, 300, 830, 300, "≥ 0.5 m", offset=-46, size=17)  # 7.6b

    # Clause legend.
    for y, txt in (
        (505, "7.6 a) ≥ 0.7 m between microphone positions"),
        (531, "7.6 b) ≥ 0.5 m to room boundaries"),
        (557, "7.6 c) ≥ 1.0 m to the loudspeaker"),
        (583, "7.2.2 ≥ 1.0 m loudspeaker to separating partition"),
    ):
        s.text(80, y, txt, 15, th.fg, anchor="start")


# ---------------------------------------------------------------------------
# d9 - ISO 18233 indirect impulse-response measurement chain
# ---------------------------------------------------------------------------


def _d_ir_measurement(s: SVG, th: Theme) -> None:
    bw, bh = 200.0, 96.0
    xs = (120.0, 350.0, 580.0)
    y1, y2 = 110.0, 300.0

    def box(x: float, y: float, title: str, subs: list[str], color: str) -> None:
        s.rect(x, y, bw, bh, th.panel, color, rx=12, sw=2)
        # Drop a step when the title will not clear the box at the larger
        # size. Counting characters decided this on the English string and
        # on the assumption that every glyph is as wide as every other, so a
        # Spanish twin that is longer on the page but not in characters kept
        # a size it overflows at; the pen advance is what the box has to
        # hold.
        t_size = 19 if s.text_width(title, 19, bold=True) <= bw - _BOX_PAD else 17
        if subs:
            s.text(x + bw / 2, y + 38, title, t_size, th.fg, bold=True)
            if len(subs) == 1:
                s.text(x + bw / 2, y + 66, subs[0], 15, color)
            else:
                s.text(x + bw / 2, y + 62, subs[0], 15, color)
                s.text(x + bw / 2, y + 82, subs[1], 15, color)
        else:
            s.text(x + bw / 2, y + bh / 2 + 7, title, t_size, th.fg, bold=True)

    # Row 1 (left to right): the physical excitation path.
    box(xs[0], y1, "Excitation", ["ESS sweep / MLS"], th.primary)
    box(xs[1], y1, "Loudspeaker", [], th.fg)
    box(xs[2], y1, "Room", ["$h(t)$"], th.secondary)
    s.arrow(xs[0] + bw, y1 + bh / 2, xs[1] - 2, y1 + bh / 2, th.fg, 2)
    s.arrow(xs[1] + bw, y1 + bh / 2, xs[2] - 2, y1 + bh / 2, th.fg, 2)

    # Serpentine connector: the acoustic field couples Room -> Microphone.
    cx = xs[2] + bw / 2
    s.arrow(cx, y1 + bh, cx, y2 - 2, th.muted, 2)
    s.text(
        cx - 12,
        (y1 + bh + y2) / 2 + 5,
        "acoustic path",
        15,
        th.muted,
        anchor="end",
        italic=True,
    )

    # Row 2 (right to left): recover the impulse response by deconvolution.
    box(xs[2], y2, "Microphone", [], th.primary)
    box(xs[1], y2, "Deconvolution", ["correlation /", "inverse filter"], th.accent)
    box(xs[0], y2, "IR", ["$ĥ(t)$"], th.accent)
    s.arrow(xs[2], y2 + bh / 2, xs[1] + bw + 2, y2 + bh / 2, th.fg, 2)
    s.arrow(xs[1], y2 + bh / 2, xs[0] + bw + 2, y2 + bh / 2, th.fg, 2)

    s.text(
        450,
        425,
        "The room response $h(t)$ is recovered by deconvolving the microphone signal.",
        15,
        th.fg,
    )


def _d_sweep_budget(s: SVG, th: Theme) -> None:
    """Dimensioning the excitation for a room with T = 1.2 s (ISO 18233).

    Three time lanes on one page: the sweep with its appended silence
    (B.3.1), the MLS period against the reverberation time (6.2.2.2,
    Eq. (10)) with the discarded warm-up, and the deconvolved record where
    the harmonic products land at negative arrival times (B.5).
    """
    reverberation = 1.2
    x0, x1 = 118.0, 838.0
    span = 6.0  # seconds across the lanes
    ppm = (x1 - x0) / span  # pixels per second

    def tick_axis(
        y: float, first: float, last: float, step: float, origin: float = 0.0
    ) -> None:
        s.line(x0, y, x1, y, th.fg, 2.0)
        value = first
        while value <= last + 1e-9:
            xx = x0 + (value - origin) * ppm
            s.line(xx, y, xx, y + 7, th.fg, 1.4)
            s.text(xx, y + 25, f"{value:g}", 12, th.muted, "middle")
            value += step

    # --- Lane 1: the sweep and the silence that follows it -----------------
    y1 = 150.0
    s.text(
        x0,
        80,
        "1  What you play, and how long you keep recording",
        15,
        th.fg,
        "start",
        bold=True,
    )
    sweep_end = 4.0
    record_end = sweep_end + reverberation
    s.rect(x0, y1 - 34, sweep_end * ppm, 34, th.primary, th.fg, rx=4, sw=1.6)
    s.text(
        x0 + sweep_end * ppm / 2,
        y1 - 12,
        "sweep, 4.0 s = 3.3 × $T$",
        14,
        th.bg,
        "middle",
        bold=True,
    )
    s.rect(
        x0 + sweep_end * ppm,
        y1 - 34,
        reverberation * ppm,
        34,
        th.panel,
        th.fg,
        rx=4,
        sw=1.6,
        dash="5,4",
    )
    s.text(
        x0 + (sweep_end + reverberation / 2) * ppm,
        y1 - 12,
        "silence $≈ T$",
        13,
        th.fg,
        "middle",
    )
    s.dim(
        x0,
        y1 - 46,
        x0 + record_end * ppm,
        y1 - 46,
        "record window 5.2 s",
        offset=0,
        size=13,
    )
    tick_axis(y1, 0.0, 6.0, 1.0)
    s.text(
        x1,
        y1 + 46,
        "B.3.1: sweep 2–4 × $T$, silent gap $≈ T$   |   "
        "B.6: +3 dB effective SNR per doubling",
        12,
        th.muted,
        "end",
    )

    # --- Lane 2: the MLS period against the reverberation time -------------
    y2 = 296.0
    s.text(
        x0,
        228,
        "2  If the excitation repeats, the period must exceed $T$ (6.2.2.2)",
        15,
        th.fg,
        "start",
        bold=True,
    )
    period = (2**17 - 1) / 48000.0  # order 17 at 48 kHz = 2.73 s
    for index in range(2):
        left = x0 + index * period * ppm
        good = index > 0
        s.rect(
            left,
            y2 - 30,
            period * ppm,
            30,
            th.panel if good else th.bg,
            th.fg,
            rx=3,
            sw=1.5,
            dash="" if good else "4,4",
        )
        s.text(
            left + period * ppm / 2,
            y2 - 10,
            "period 2, kept" if good else "period 1, warm-up: discarded",
            13,
            th.fg if good else th.muted,
            "middle",
            bold=good,
        )
    s.dim(
        x0,
        y2 - 46,
        x0 + period * ppm,
        y2 - 46,
        "order 17 → 2.73 s $≥ T$",
        offset=0,
        size=13,
    )
    # The failure: an order too low folds the tail onto the head.
    short = (2**15 - 1) / 48000.0  # order 15 = 0.68 s < T
    s.rect(x0, y2 + 40, short * ppm, 26, th.bg, th.secondary, rx=3, sw=1.6)
    s.text(
        x0 + short * ppm / 2,
        y2 + 58,
        f"{short:.2f} s",
        11,
        th.secondary,
        "middle",
        bold=True,
    )
    s.text(
        x0 + short * ppm + 14,
        y2 + 58,
        "order 15 is shorter than $T$: the tail folds onto the head and "
        "$T$ comes out short",
        13,
        th.secondary,
        "start",
    )
    tick_axis(y2, 0.0, 6.0, 1.0)

    # --- Lane 3: what the deconvolution returns ----------------------------
    y3 = 470.0
    origin = 1.2  # the axis starts at -1.2 s
    s.text(
        x0, 386, "3  After linear deconvolution (B.5)", 15, th.fg, "start", bold=True
    )
    zero = x0 + origin * ppm
    # Harmonic packets at t = -T_sweep ln N / ln(f2/f1), f2/f1 = 1000.
    import math

    for order, height in ((2, 46.0), (3, 34.0), (4, 24.0)):
        advance = sweep_end * math.log(order) / math.log(1000.0)
        xx = zero - advance * ppm
        s.line(xx, y3, xx, y3 - height, th.secondary, 2.4)
        s.text(
            xx, y3 - height - 8, f"$H_{order:d}$", 12, th.secondary, "middle", bold=True
        )
    # The linear impulse response and its decaying tail.
    s.line(zero, y3, zero, y3 - 92, th.primary, 3.0)
    tail = "M " + " L ".join(
        f"{zero + t * ppm:.1f} {y3 - 92 * math.exp(-3.0 * t):.1f}"
        for t in [i * 0.05 for i in range(1, 25)]
    )
    s.path(tail, stroke=th.primary, sw=1.6)
    s.line(zero, y3 - 118, zero, y3 + 16, th.fg, 1.6, dash="6,4")
    s.text(
        zero + 10,
        y3 + 34,
        "kept by default: the linear impulse response and its tail",
        13,
        th.primary,
        "start",
        bold=True,
    )
    s.text(
        zero - 14, y3 + 34, "discarded, or read as distortion", 13, th.secondary, "end"
    )
    # The deconvolution's own noise tail, decaying and low-passed.
    late = "M " + " L ".join(
        f"{zero + t * ppm:.1f} {y3 - 30 * math.exp(-0.9 * (t - 1.2)):.1f}"
        for t in [1.2 + i * 0.1 for i in range(1, 30)]
    )
    s.path(late, stroke=th.muted, sw=1.4, dash="5,4")
    s.text(
        x1,
        y3 - 44,
        "the linear deconvolution's own decaying noise tail — not the room",
        12,
        th.muted,
        "end",
    )
    tick_axis(y3, -1.0, 4.0, 1.0, origin=-origin)
    s.text(
        x1,
        y3 + 58,
        "Arrival time relative to the linear impulse response [s]",
        13,
        th.fg,
        "end",
    )


# ---------------------------------------------------------------------------
# d11 - ISO 16283-2 impact sound insulation setup
# ---------------------------------------------------------------------------


def _d_impact(s: SVG, th: Theme) -> None:
    bx0, bx1 = 90.0, 620.0  # building left / right walls
    top = 82.0
    floor_top, floor_bot = 292.0, 316.0  # separating floor slab
    bot = 512.0  # receiving-room floor

    # Building shell and the two stacked rooms.
    s.rect(bx0, top, bx1 - bx0, floor_top - top, th.panel, th.fg, sw=2.5)
    s.rect(bx0, floor_bot, bx1 - bx0, bot - floor_bot, th.panel, th.fg, sw=2.5)
    s.rect(
        bx0, floor_top, bx1 - bx0, floor_bot - floor_top, th.secondary, th.fg, sw=2
    )  # separating floor / ceiling
    s.text(
        bx0 + 16, top + 30, "Source room (upper)", 18, th.fg, bold=True, anchor="start"
    )
    s.text(
        bx0 + 16,
        bot - 16,
        "Receiving room (lower)",
        18,
        th.fg,
        bold=True,
        anchor="start",
    )
    s.text(
        bx1 - 12,
        floor_top - 8,
        "Separating floor",
        15,
        th.secondary,
        bold=True,
        anchor="end",
    )

    # Tapping machine standing on the separating floor (five hammers).
    mx = bx0 + 165.0
    body_y = floor_top - 40.0
    s.rect(mx - 60, body_y, 120, 28, th.primary, th.fg, rx=5, sw=2)
    for hx in range(-40, 41, 20):
        s.line(mx + hx, body_y + 28, mx + hx, floor_top - 2, th.fg, 2.4)
        s.circle(mx + hx, floor_top - 2, 4.2, th.fg)
    s.line(mx - 54, body_y + 28, mx - 54, floor_top, th.fg, 2)  # legs
    s.line(mx + 54, body_y + 28, mx + 54, floor_top, th.fg, 2)
    s.text(mx, body_y - 12, "Tapping machine", 16, th.fg, bold=True)

    # Structure-borne path through the slab, radiated into the room below.
    s.arrow(mx, floor_bot + 2, mx, floor_bot + 42, th.secondary, 2.2)
    s.text(
        mx - 12,
        floor_bot + 30,
        "structure-borne impact",
        13,
        th.secondary,
        anchor="end",
        italic=True,
    )
    for r in (46, 74, 102):
        s.path(
            f"M {mx - r * 0.72:.1f} {floor_bot + 44 + r * 0.5:.1f} "
            f"A {r} {r} 0 0 0 {mx + r * 0.72:.1f} {floor_bot + 44 + r * 0.5:.1f}",
            stroke=th.accent,
            sw=1.6,
        )
    s.text(mx, bot - 44, "radiated impact sound", 13, th.accent, italic=True)

    # Microphone positions on the receiving-room floor.
    for off in (300, 400, 500):
        s.mic(bx0 + off, bot - 120, bot, 0.95)
    s.text(bx0 + 400, floor_bot + 42, "Microphone positions", 14, th.muted)

    # Normative relations (right column); no invented spacing dimensions.
    lx = 648.0
    s.text(lx, 118, "Impact sound insulation", 15, th.fg, bold=True, anchor="start")
    box_items = [
        (160, "$L′_{nT} = L_i − 10 log_{10}(T/T_0)$", th.primary),
        (192, "$L′_n = L_i + 10 log_{10}(A/A_0)$", th.primary),
        (224, "$A = 0.16 V/T$  (Sabine)", th.muted),
        (256, "$T_0$ = 0.5 s , $A_0$ = 10 m²", th.accent),
    ]
    for y, txt, col in box_items:
        s.text(lx, y, txt, 12, col, anchor="start", bold=(col != th.muted))
    s.rect(lx - 10, 292, 236, 100, "none", th.muted, rx=10, dash="6,5")
    s.text(lx, 320, "$L_i$ = energy-averaged", 13, th.fg, anchor="start")
    s.text(lx, 342, "band level (Formula 10)", 13, th.fg, anchor="start")
    s.text(
        lx,
        374,
        "ISO 717-2 → $L_{n,w}$ , $C_I$",
        14,
        th.secondary,
        anchor="start",
        bold=True,
    )


# ---------------------------------------------------------------------------
# d13 - EN 12354 direct + flanking transmission paths across a junction
# ---------------------------------------------------------------------------


def _d_flanking(s: SVG, th: Theme) -> None:
    dark = bool(th.suffix)
    # Four legible path colours (green / blue / red / orange), independent of
    # the neutral structural fills so every path stands out in both themes.
    c_dd = th.accent
    c_ff = th.primary
    c_fd = th.secondary
    c_df = "#f0a94e" if dark else "#d9820e"

    room_top, room_bot = 96.0, 372.0
    slab_top, slab_bot = 372.0, 402.0
    slab_cy = (slab_top + slab_bot) / 2.0
    wall_l, wall_r, wx = 434.0, 466.0, 450.0
    wall_bot = 430.0  # wall runs on past the slab (cross)
    bl, br = 70.0, 830.0
    jx, jy = wx, slab_cy  # junction node

    # --- structural shell: two rooms, separating wall, flanking slab --------
    s.rect(bl, room_top, wall_l - bl, room_bot - room_top, th.panel, th.fg, sw=2.5)
    s.rect(wall_r, room_top, br - wall_r, room_bot - room_top, th.panel, th.fg, sw=2.5)
    # Flanking element (continuous slab through the junction).
    s.rect(bl, slab_top, br - bl, slab_bot - slab_top, th.panel, th.fg, sw=2)
    for hx in range(int(bl) + 16, int(br), 34):
        s.line(hx, slab_top, hx - 12, slab_bot, th.muted, 0.9)
    # Separating element (vertical wall, drawn on top -> rigid cross junction).
    s.rect(
        wall_l,
        room_top,
        wall_r - wall_l,
        wall_bot - room_top,
        th.secondary,
        th.fg,
        sw=2,
    )

    s.text(bl + 16, room_top + 34, "Source room", 19, th.fg, bold=True, anchor="start")
    s.text(bl + 16, room_top + 60, "$L_1$", 17, th.muted, anchor="start")
    s.text(
        wall_r + 16,
        room_top + 34,
        "Receiving room",
        19,
        th.fg,
        bold=True,
        anchor="start",
    )
    s.text(wall_r + 16, room_top + 60, "$L_2$ , $T$", 17, th.muted, anchor="start")
    s.text(wx, room_top - 8, "Separating element (D, d)", 15, th.secondary, bold=True)
    s.text(
        bl + 16,
        slab_bot + 22,
        "Flanking element (F, f)",
        15,
        th.fg,
        bold=True,
        anchor="start",
    )

    # Loudspeaker (airborne excitation) in the source room, mic in receiving.
    lsx, lsy = 140.0, 300.0
    for r in (30, 50, 70):
        s.path(
            f"M {lsx + r * 0.22:.1f} {lsy - r:.1f} "
            f"A {r} {r} 0 0 1 {lsx + r:.1f} {lsy - r * 0.22:.1f}",
            stroke=th.muted,
            sw=1.4,
        )
    s.rect(lsx - 22, lsy - 26, 44, 52, th.panel, th.fg, rx=5, sw=2)
    s.circle(lsx, lsy - 8, 10, th.fg)
    s.circle(lsx, lsy - 8, 4, th.bg)
    s.circle(lsx, lsy + 14, 6, th.fg)
    s.text(lsx, lsy + 50, "Loudspeaker", 15, th.fg, bold=True)
    s.mic(786.0, 236.0, room_bot, 0.9)
    s.text(786.0, 220.0, "Microphone", 15, th.fg, bold=True)

    # --- transmission paths -------------------------------------------------
    # Dd: straight through the separating element, well above the slab.
    ddy = 172.0
    s.arrow(250.0, ddy, 648.0, ddy, c_dd, 3.0)
    s.text(300.0, ddy - 12, "Dd", 21, c_dd, bold=True)

    # Ff: down onto the flanking slab, along it through the junction, up again.
    s.line(250.0, 284.0, 250.0, slab_cy, c_ff, 2.8)
    s.line(250.0, slab_cy, 650.0, slab_cy, c_ff, 2.8)
    s.arrow(650.0, slab_cy, 650.0, 288.0, c_ff, 2.8)
    s.text(662.0, 300.0, "Ff", 21, c_ff, bold=True, anchor="start")

    # Fd: flanking element (source) -> junction -> radiates from the wall.
    s.line(330.0, 320.0, 330.0, slab_cy, c_fd, 2.8)
    s.line(330.0, slab_cy, 444.0, slab_cy, c_fd, 2.8)
    s.line(444.0, slab_cy, 444.0, 296.0, c_fd, 2.8)
    s.arrow(444.0, 296.0, 556.0, 236.0, c_fd, 2.8)
    s.text(560.0, 230.0, "Fd", 21, c_fd, bold=True, anchor="start")

    # Df: separating wall (source) -> junction -> radiates from the slab.
    s.line(392.0, 236.0, 456.0, 296.0, c_df, 2.8)
    s.line(456.0, 296.0, 456.0, slab_cy, c_df, 2.8)
    s.line(456.0, slab_cy, 614.0, slab_cy, c_df, 2.8)
    s.arrow(614.0, slab_cy, 614.0, 316.0, c_df, 2.8)
    s.text(626.0, 322.0, "Df", 21, c_df, bold=True, anchor="start")

    # Junction node on top of everything.
    s.circle(jx, jy, 6.5, th.bg, th.fg, 2.2)
    s.text(360.0, slab_bot + 22, "junction", 14, th.muted, italic=True)
    s.line(392.0, slab_bot + 17, jx - 7, jy + 3, th.muted, 0.9, dash="3,3")

    # --- legend + master formula (Formula 26) -------------------------------
    rows = [
        (c_dd, "Dd — direct path: separating element both sides"),
        (c_ff, "Ff — flanking–flanking: flanking element both sides"),
        (c_fd, "Fd — flanking (source) → separating (receiving)"),
        (c_df, "Df — separating (source) → flanking (receiving)"),
    ]
    ly = 452.0
    for col, txt in rows:
        s.line(bl + 4, ly - 6, bl + 44, ly - 6, col, 4.0)
        s.text(bl + 58, ly, txt, 16, th.fg, anchor="start")
        ly += 32
    s.text(
        450.0,
        ly + 12,
        "R'w = −10 log10 Σ 10^(−Rij,w /10) dB   (EN 12354-1, Formula 26)",
        16,
        th.muted,
        bold=True,
    )


def _d_room_measurement(s: SVG, th: Theme) -> None:
    """Room-acoustics measurement layout (ISO 3382-1 positions, ISO 3382-2 grades).

    A top-view room plan with two source positions and six microphone
    positions plus the ISO 3382-1 spacing rules, and a table of the
    ISO 3382-2:2008 Table 1 minimum position counts for the three grades.
    """
    # --- Room plan (top view) ------------------------------------------------
    # A 10.0 x 6.0 m room drawn at 50 px per metre; with a 3.5 m ceiling that
    # is V = 210 m3, so an expected T of 0.6 s gives d_min = 2.0 m exactly.
    rx, ry, rw, rh = 60.0, 96.0, 500.0, 300.0
    d_min = 2.0 * 50.0  # 2.0 m at 50 px per metre
    s.rect(rx, ry, rw, rh, th.panel, th.fg, rx=6, sw=2.4)
    s.text(
        rx + 10,
        ry - 12,
        "Room plan (top view) — 10.0 × 6.0 m, 3.5 m high",
        17,
        th.fg,
        "start",
        bold=True,
    )

    # The two symmetry axes: positions on them sample mirror-image fields.
    s.line(rx + rw / 2, ry, rx + rw / 2, ry + rh, th.muted, 1.2, dash="9,4,2,4")
    s.line(rx, ry + rh / 2, rx + rw, ry + rh / 2, th.muted, 1.2, dash="9,4,2,4")

    # The d_min exclusion zones, clipped to the room so the plan stays a plan.
    s1 = (rx + 70, ry + 70)
    s2 = (rx + 430, ry + 224)
    s.add(
        f'<defs><clipPath id="room-plan-clip"><rect x="{rx}" y="{ry}" '
        f'width="{rw}" height="{rh}"/></clipPath></defs>'
        f'<g clip-path="url(#room-plan-clip)">'
    )
    for cx, cy in (s1, s2):
        s.circle(cx, cy, d_min, "none", th.secondary, 1.6)
    s.add("</g>")
    s.text(
        s2[0] - d_min - 6, s2[1] - 12, "$d_{min}$", 14, th.secondary, "end", bold=True
    )

    # Two loudspeaker source positions (ISO 3382-1: at least two).
    def _speaker(x: float, y: float, label: str) -> None:
        s.rect(x - 13, y - 11, 26, 22, th.primary, th.fg, rx=3, sw=1.6)
        s.circle(x, y, 5, th.bg, th.fg, 1.2)
        s.text(x, y - 18, label, 15, th.primary, "middle", bold=True)

    _speaker(*s1, "S1")
    _speaker(*s2, "S2")

    # Six microphone positions: >= 2 m apart, >= 1 m from every surface,
    # outside both d_min circles and off both symmetry axes.
    mics = [
        (250.0, 150.0, "M1"),
        (390.0, 150.0, "M2"),
        (510.0, 190.0, "M3"),
        (160.0, 290.0, "M4"),
        (265.0, 330.0, "M5"),
        (395.0, 285.0, "M6"),
    ]
    for mx, my, label in mics:
        s.circle(mx, my, 7, th.secondary, th.fg, 1.4)
        s.text(mx + 12, my + 6, label, 15, th.fg, "start", bold=True)

    # The position the standard tells you not to take: on a symmetry axis.
    gx, gy = rx + rw / 2, ry + 104.0
    s.circle(gx, gy, 7, "none", th.muted, 1.4)
    s.line(gx - 10, gy - 10, gx + 10, gy + 10, th.muted, 1.8)
    s.line(gx - 10, gy + 10, gx + 10, gy - 10, th.muted, 1.8)
    s.text(gx + 16, gy - 8, "avoid symmetry lines", 13, th.muted, "start")

    # Spacing annotations.
    s.line(250.0, 150.0, 390.0, 150.0, th.accent, 1.6, dash="5,4")
    s.text(320.0, 142.0, "≥ 2 m", 15, th.accent, "middle", bold=True)
    s.arrow(160.0, 299.0, 160.0, ry + rh, th.muted, 1.4)
    s.text(152.0, 350.0, "≥ 1 m", 14, th.fg, "end")
    # The source-receiver distance, dimensioned outside the exclusion circle.
    s.line(s1[0], s1[1], 250.0, 150.0, th.primary, 1.3, dash="4,4")
    s.text(212.0, 182.0, "2.4 m $> d_{min}$", 13, th.primary, "middle")

    # Legend + ISO 3382-1 rules, to the right of the plan.
    # 32 px off the room, not 24: the source-clearance circle bulges 30 px
    # past the wall it is centred on, and the legend column has to start
    # outside it.
    lx = rx + rw + 32
    s.circle(lx + 8, ry + 16, 7, th.secondary, th.fg, 1.4)
    s.text(lx + 24, ry + 22, "Microphone position", 15, th.fg, "start")
    s.rect(lx, ry + 40, 16, 14, th.primary, th.fg, rx=2, sw=1.4)
    s.text(lx + 24, ry + 52, "Loudspeaker source", 15, th.fg, "start")
    for i, line in enumerate(
        (
            "ISO 3382-1 (positions):",
            "• ≥ 2 source positions",
            "• mics ≥ 2 m apart",
            "• ≥ 1 m from surfaces",
            "• mic height 1.2 m",
            "• source height 1.5 m",
            "• off the symmetry axes",
            "ISO 3382-2 (source clearance):",
            "$d_{min} = 2√(V/(c·T̂))$ = 2.0 m",
            "for $V$ = 210 m³, $T̂$ = 0.6 s",
        )
    ):
        bold = line.endswith(":")
        color = th.secondary if i >= 7 else th.fg
        # 24 px of leading, so the last line of the column clears the
        # table heading that runs under it: the Spanish heading is 600 px
        # long and passes right below this column.
        s.text(lx, ry + 84 + i * 24, line, 15, color, "start", bold=bold)

    # --- ISO 3382-2 Table 1: minimum measurement positions per grade ---------
    ty = ry + rh + 46.0
    s.text(
        60,
        ty - 14,
        "ISO 3382-2 — reverberation-time measurement grades",
        17,
        th.fg,
        "start",
        bold=True,
    )
    cols = [
        (70.0, "Method", "start"),
        (330.0, "Source pos.", "middle"),
        (470.0, "Mic pos.", "middle"),
        (630.0, "Source–mic comb.", "middle"),
        (820.0, "Decays / comb.", "middle"),
    ]
    rows = [
        ("Survey", "≥ 1", "≥ 2", "2", "1"),
        ("Engineering", "≥ 2", "≥ 2", "6", "2"),
        ("Precision", "≥ 2", "≥ 3", "12", "3"),
    ]
    tw, th_row = 840.0, 40.0
    s.rect(60, ty, tw, th_row * (len(rows) + 1), "none", th.fg, rx=6, sw=1.8)
    s.rect(60, ty, tw, th_row, th.panel, th.fg, rx=6, sw=1.8)
    for cx, label, anchor in cols:
        s.text(cx, ty + 26, label, 15, th.fg, anchor, bold=True)
    for r, row in enumerate(rows):
        yy = ty + th_row * (r + 1)
        if r < len(rows) - 1:
            s.line(60, yy + th_row, 60 + tw, yy + th_row, th.muted, 1.0)
        for (cx, _, anchor), value in zip(cols, row, strict=True):
            col = th.primary if cx == 70.0 else th.fg
            s.text(cx, yy + 26, value, 15, col, anchor, bold=(cx == 70.0))


def _d_open_plan_setup(s: SVG, th: Theme) -> None:
    """Where the ISO 3382-3 line goes (clauses 5.1 and 5.2).

    Panel (a): plan of a 30 x 12 m open-plan floor with two ceiling zones,
    the non-straight measurement path, both source positions and every
    clearance the standard names. Panel (b): the section that fixes both
    heights at 1.2 m.
    """
    ppm = 24.0  # 30.0 m over 720 px
    x0, y0 = 96.0, 92.0
    x1, y1 = x0 + 30.0 * ppm, y0 + 12.0 * ppm  # 30 x 12 m floor

    # --- Panel (a): the floor in plan ---------------------------------------
    s.text(
        x0,
        y0 - 14,
        "(a) Plan — 30 × 12 m floor, two ceiling zones",
        15,
        th.fg,
        "start",
        bold=True,
    )
    zone = x0 + 18.0 * ppm
    s.rect(x0, y0, zone - x0, y1 - y0, th.panel, th.fg, sw=2.4)
    s.rect(zone, y0, x1 - zone, y1 - y0, th.bg, th.fg, sw=2.4)
    s.text(x0 + 20, y0 + 22, "absorbent raft ceiling", 12, th.muted, "start")
    s.text(x1 - 20, y0 + 22, "plain plaster ceiling", 12, th.muted, "end")
    s.line(zone, y0, zone, y1, th.fg, 1.6, dash="9,5")
    s.text(
        zone, y1 + 34, "zones measured and reported separately", 12, th.muted, "middle"
    )

    # The 2.0 m keep-out band along every wall (5.2.2).
    band = 2.0 * ppm
    s.rect(
        x0 + band,
        y0 + band,
        (x1 - x0) - 2 * band,
        (y1 - y0) - 2 * band,
        "none",
        th.accent,
        sw=1.4,
        dash="6,4",
    )
    s.text(
        x0 + band + 8,
        y0 + band + 18,
        "≥ 2.0 m from walls and other reflecting surfaces",
        12,
        th.accent,
        "start",
    )

    # Desk clusters with 1.2 m screens between them.
    for col in range(6):
        for row in (0, 1):
            dx = x0 + (4.0 + col * 4.2) * ppm
            dy = y0 + (3.6 + row * 4.6) * ppm
            s.rect(dx - 30, dy - 14, 60, 28, th.panel, th.muted, rx=2, sw=1.2)
            s.line(dx - 30, dy - 14, dx + 30, dy - 14, th.secondary, 2.4)

    # The measurement path: not a straight line (Figure 1, path A). Distances
    # are metres from S1, which itself keeps 2.6 m off the end wall.
    src = (x0 + 2.6 * ppm, y0 + 6.0 * ppm)
    path = [
        (2.0, -0.8),
        (3.4, 0.6),
        (5.5, -0.9),
        (7.8, 0.7),
        (10.5, -0.7),
        (13.0, 0.8),
        (16.0, -0.6),
        (19.5, 0.5),
    ]
    pts = [(src[0] + dx * ppm, src[1] + dy * ppm) for dx, dy in path]
    s.path(
        "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in [src, *pts]),
        stroke=th.primary,
        sw=1.6,
        dash="6,4",
    )
    for index, (px, py) in enumerate(pts, start=1):
        s.circle(px, py, 6, th.secondary, th.fg, 1.3)
        s.text(px, py - 12, f"P{index:d}", 11, th.fg, "middle", bold=True)
    s.text(x0 + band + 8, y1 - band - 22, "1.2 m screens", 12, th.secondary, "start")
    s.text(
        x0 + band + 8,
        y1 - band - 2,
        "P1 at the nearest workstation; the path need not be straight",
        12,
        th.primary,
        "start",
    )

    # Both source positions: S2 fires back along the same line (5.2.2).
    for (sx, sy), label in ((src, "S1"), ((x1 - 3.0 * ppm, src[1] + 0.4 * ppm), "S2")):
        s.rect(sx - 12, sy - 10, 24, 20, th.primary, th.fg, rx=3, sw=1.5)
        s.circle(sx, sy, 4.5, th.bg, th.fg, 1.1)
        s.text(sx, sy - 16, label, 14, th.primary, "middle", bold=True)

    # The 2 m to 16 m regression window (6.2).
    w_lo, w_hi = src[0] + 2.0 * ppm, src[0] + 16.0 * ppm
    s.dim(
        w_lo,
        y1 - 22,
        w_hi,
        y1 - 22,
        "only 2 m to 16 m enter $D_{2,S}$",
        offset=0,
        size=13,
    )
    for xx in (w_lo, w_hi):
        s.line(xx, y1 - 36, xx, y1 - 12, th.fg, 1.2, dash="4,3")
    # One 0.5 m clearance called out at a desk, and the screens named once.
    s.dim(
        pts[1][0],
        pts[1][1] + 6,
        pts[1][0],
        pts[1][1] + 0.5 * ppm + 6,
        "≥ 0.5 m from tables",
        offset=0,
        size=12,
        label_side="right",
    )

    # --- Panel (b): the section that fixes the heights ----------------------
    sy0 = y1 + 78.0
    floor_y = sy0 + 118.0
    s.text(
        x0,
        sy0 - 12,
        "(b) Section — both heights are 1.2 m (5.2.2)",
        15,
        th.fg,
        "start",
        bold=True,
    )
    s.ground(floor_y, x0, x0 + 460)
    hpm = 74.0  # taller scale for one person
    s.person(x0 + 300, floor_y, h=1.2 * hpm, seated=True)
    # The omnidirectional source at head height, radiating pink noise.
    src_x, src_y = x0 + 90, floor_y - 1.2 * hpm
    s.circle(src_x, src_y, 17, th.panel, th.primary, 2.0)
    for radius in (30.0, 46.0):
        s.path(
            f"M {src_x + radius * 0.28:.1f} {src_y - radius:.1f} "
            f"A {radius} {radius} 0 0 1 {src_x + radius:.1f} "
            f"{src_y - radius * 0.28:.1f}",
            stroke=th.primary,
            sw=1.2,
            dash="4,4",
        )
    s.line(src_x, src_y + 17, src_x, floor_y, th.fg, 2.2)
    s.line(src_x - 16, floor_y, src_x + 16, floor_y, th.fg, 2.2)
    s.text(
        src_x,
        floor_y + 34,
        "omnidirectional, pink noise",
        12,
        th.primary,
        "middle",
        bold=True,
    )
    s.dim(
        src_x - 40,
        floor_y,
        src_x - 40,
        src_y,
        "1.2 m",
        offset=0,
        size=13,
        label_side="left",
    )
    mic_x = x0 + 210
    s.mic(mic_x, floor_y - 1.2 * hpm, floor_y, scale=0.8)
    s.dim(
        mic_x + 40,
        floor_y,
        mic_x + 40,
        floor_y - 1.2 * hpm,
        "1.2 m",
        offset=0,
        size=13,
        label_side="right",
    )
    s.text(x0 + 300, floor_y + 34, "seated head position", 12, th.muted, "middle")

    # --- The clause list, beside the section --------------------------------
    lx = x0 + 486
    for i, line in enumerate(
        (
            "Source (5.1.1):",
            "omnidirectional, pink noise, ISO 3382-1",
            "directivity; a pink-spectrum sweep or",
            "MLS may be used instead",
            "Receiver (5.1.2):",
            "class 1 to IEC 61672-1, IEC 61260 octave",
            "filters, omnidirectional capsule,",
            "≥ 10 s integration",
            "Room (5.2.1):",
            "furnished, nobody present but the",
            "operators, HVAC and any masking system",
            "at working-day power",
            "Line (5.2.2):",
            "6 to 10 positions preferred, 4 the minimum;",
            "≥ 2 source positions, or the line walked",
            "in both directions",
        )
    ):
        s.text(
            lx,
            sy0 - 14 + i * 20,
            line,
            12,
            th.primary if line.endswith(":") else th.fg,
            "start",
            bold=line.endswith(":"),
        )


def _d_room_measurement_section(s: SVG, th: Theme) -> None:
    """The measuring chain in section (ISO 3382-1 clauses 4.2, 4.3).

    The same 10.0 x 6.0 x 3.5 m room as the plan, cut along its length: the
    dodecahedron with its acoustic centre at 1.5 m, microphones at 1.2 m, the
    quarter-wavelength clearances including the floor, and the Table 1
    directivity tolerances the source has to meet.
    """
    ppm = 58.0  # 10.0 m over 580 px
    x0, x1 = 70.0, 70.0 + 10.0 * ppm
    floor_y = 336.0
    ceil_y = floor_y - 3.5 * ppm  # 3.5 m high

    s.rect(x0, ceil_y, x1 - x0, floor_y - ceil_y, th.panel, th.fg, sw=2.6)
    s.ground(floor_y, x0 - 20, x1 + 20)
    s.text(
        x0,
        ceil_y - 16,
        "Section through the same 10.0 × 6.0 × 3.5 m room",
        16,
        th.fg,
        "start",
        bold=True,
    )

    # --- The dodecahedron on its stand -------------------------------------
    src_x = x0 + 2.0 * ppm
    src_y = floor_y - 1.5 * ppm  # acoustic centre at 1.5 m
    s.line(src_x, src_y + 22, src_x, floor_y, th.fg, 2.4)
    s.line(src_x - 20, floor_y, src_x + 20, floor_y, th.fg, 2.4)
    s.path(
        f"M {src_x - 26} {src_y} L {src_x - 13} {src_y - 24} "
        f"L {src_x + 13} {src_y - 24} L {src_x + 26} {src_y} "
        f"L {src_x + 13} {src_y + 24} L {src_x - 13} {src_y + 24} Z",
        fill=th.panel,
        stroke=th.primary,
        sw=2.2,
    )
    s.line(src_x - 13, src_y - 24, src_x - 13, src_y + 24, th.primary, 1.0)
    s.line(src_x + 13, src_y - 24, src_x + 13, src_y + 24, th.primary, 1.0)
    s.circle(src_x, src_y, 4, th.secondary)
    s.text(src_x, src_y - 34, "dodecahedron", 13, th.primary, "middle", bold=True)
    s.dim(
        src_x - 40,
        floor_y,
        src_x - 40,
        src_y,
        "1.5 m",
        offset=0,
        size=13,
        label_side="left",
    )
    s.text(src_x + 32, src_y + 5, "acoustic centre", 12, th.secondary, "start")

    # --- The d_min exclusion zone, clipped to the room ---------------------
    d_min = 2.0 * ppm
    s.add(
        f'<defs><clipPath id="room-section-clip"><rect x="{x0}" y="{ceil_y}" '
        f'width="{x1 - x0}" height="{floor_y - ceil_y}"/></clipPath></defs>'
        f'<g clip-path="url(#room-section-clip)">'
    )
    s.circle(src_x, src_y, d_min, "none", th.secondary, 1.6)
    s.add("</g>")
    s.text(
        src_x + d_min - 6,
        floor_y - 16,
        "$d_{min}$ = 2.0 m",
        13,
        th.secondary,
        "end",
        bold=True,
    )

    # --- Two microphones at seated-ear height ------------------------------
    mic_y = floor_y - 1.2 * ppm
    m1_x, m2_x = x0 + 5.0 * ppm, x0 + 8.2 * ppm
    for mx in (m1_x, m2_x):
        s.mic(mx, mic_y, floor_y, scale=0.85)
    s.dim(m1_x, mic_y - 30, m2_x, mic_y - 30, "≥ 2 m", offset=0, size=14)
    s.dim(
        m2_x + 40,
        floor_y,
        m2_x + 40,
        mic_y,
        "1.2 m",
        offset=0,
        size=13,
        label_side="right",
    )
    # Quarter-wavelength clearances: the floor counts as a surface too.
    s.dim(
        m1_x - 34,
        mic_y,
        m1_x - 34,
        ceil_y,
        "≥ 1 m",
        offset=0,
        size=13,
        label_side="left",
    )
    s.arrow(m2_x, mic_y + 8, m2_x, floor_y - 6, th.muted, 1.3)
    s.arrow(m2_x + 8, mic_y, x1 - 4, mic_y, th.muted, 1.3)
    s.text((m2_x + x1) / 2, mic_y - 8, "≥ 1 m", 13, th.fg, "middle")
    s.text(m1_x, floor_y + 34, "M1", 15, th.fg, "middle", bold=True)
    s.text(m2_x, floor_y + 34, "M2", 15, th.fg, "middle", bold=True)

    # --- The equipment clause, as a band under the section -----------------
    ty = 424.0
    s.text(
        70,
        ty - 12,
        "ISO 3382-1 Table 1 — omnidirectionality over gliding 30° arcs",
        15,
        th.fg,
        "start",
        bold=True,
    )
    cols = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
    tol = ["± 1", "± 1", "± 1", "± 3", "± 5", "± 6"]
    tw, row_h = 516.0, 36.0
    s.rect(70, ty, tw, row_h * 2, "none", th.fg, rx=6, sw=1.8)
    s.rect(70, ty, tw, row_h, th.panel, th.fg, rx=6, sw=1.8)
    for i, (freq, value) in enumerate(zip(cols, tol, strict=True)):
        cx = 70 + tw * (i + 0.5) / len(cols)
        s.text(cx, ty + 24, f"{freq:g}", 14, th.fg, "middle", bold=True)
        s.text(cx, ty + 24 + row_h, value, 14, th.primary, "middle")
    s.text(
        70,
        ty + row_h * 2 + 24,
        "Hz / dB, measured at ≥ 1.5 m — in practice a dodecahedron, not a monitor",
        13,
        th.muted,
        "start",
    )

    lx = 606.0
    for i, line in enumerate(
        (
            "Level (4.2.1):",
            "≥ 45 dB over the background",
            "per band for $T_{30}$, ≥ 35 dB for $T_{20}$",
            "Receiving chain (4.2.2.2):",
            "class 1 to IEC 61672-1,",
            "IEC 61260 filters, omnidirectional",
            "capsule, ≤ 13 mm preferred",
        )
    ):
        s.text(
            lx,
            ty - 34 + i * 26,
            line,
            13,
            th.fg if not line.endswith(":") else th.primary,
            "start",
            bold=line.endswith(":"),
        )


def _d_room_noise(s: SVG, th: Theme) -> None:
    """Room-noise rating methods (ANSI/ASA S12.2-2019): NC and RC Mark II.

    From a single octave-band spectrum, two parallel lanes: the NC tangency
    method (Table 1) and the RC Mark II rating and spectral tag (Annex D).
    """
    # --- Shared input spectrum ----------------------------------------------
    cx = 450.0
    iw, ih = 540.0, 62.0
    s.rect(cx - iw / 2, 56, iw, ih, th.panel, th.fg, rx=10, sw=2)
    s.text(
        cx,
        84,
        "Octave-band sound pressure levels  $L(f)$",
        17,
        th.fg,
        "middle",
        bold=True,
    )
    s.text(cx, 106, "16 Hz – 8000 Hz", 13, th.muted, "middle")

    lxc, rxc = 232.0, 668.0
    s.arrow(cx, 118, lxc, 158, th.fg, 1.8)
    s.arrow(cx, 118, rxc, 158, th.fg, 1.8)

    bw, bh = 372.0, 62.0

    def _step(cxx: float, y: float, l1: str, l2: str, color: str) -> None:
        s.rect(cxx - bw / 2, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cxx, y + 27, l1, 15, th.fg, "middle", bold=True)
        if l2:
            s.text(cxx, y + 48, l2, 12, th.muted, "middle")

    # --- Left lane: NC tangency method (Table 1) ----------------------------
    _step(lxc, 158, "NC — tangency method", "Table 1 curves", th.primary)
    _step(lxc, 256, "NC value in each band", "curve level = $L(f)$ at that $f$", th.fg)
    _step(lxc, 354, "NC = highest curve touched", "note the governing band", th.fg)
    s.arrow(lxc, 220, lxc, 256, th.fg, 1.8)
    s.arrow(lxc, 318, lxc, 354, th.fg, 1.8)
    s.arrow(lxc, 416, lxc, 470, th.fg, 1.8)
    s.rect(lxc - bw / 2, 470, bw, 58, "none", th.primary, rx=10, sw=2.4)
    s.text(lxc, 505, "NC-NN (band)", 20, th.fg, "middle", bold=True)

    # --- Right lane: RC Mark II rating and tag (Annex D) ---------------------
    _step(rxc, 158, "RC Mark II  (Annex D)", "−5 dB/octave curves", th.secondary)
    _step(
        rxc,
        256,
        "$L_{MF} = (L_{500} + L_{1000} + L_{2000}) / 3$",
        "RC = round($L_{MF}$)   (clause D.4)",
        th.fg,
    )
    s.arrow(rxc, 220, rxc, 256, th.fg, 1.8)
    s.arrow(rxc, 318, rxc, 354, th.fg, 1.8)
    # Spectral-tag rule box (clause D.3).
    s.rect(rxc - bw / 2, 354, bw, 116, th.panel, th.fg, rx=10, sw=2)
    s.text(rxc, 379, "Spectral tag  (clause D.3)", 15, th.fg, "middle", bold=True)
    for i, line in enumerate(
        (
            "R  rumble: a band ≤ 500 Hz exceeds RC by > 5 dB",
            "H  hiss: a band ≥ 1000 Hz exceeds RC by > 3 dB",
            "N  neutral: within both tolerances",
        )
    ):
        s.text(rxc - bw / 2 + 18, 403 + i * 22, line, 12, th.fg, "start")
    s.arrow(rxc, 470, rxc, 490, th.fg, 1.8)
    s.rect(rxc - bw / 2, 490, bw, 58, "none", th.secondary, rx=10, sw=2.4)
    s.text(rxc, 525, "RC-NN(A)", 20, th.fg, "middle", bold=True)


def _d_room_noise_setup(s: SVG, th: Theme) -> None:
    """Where the rated spectrum is measured (ANSI/ASA S12.2-2019, 5.2.5).

    Section through a 6.0 x 2.7 m office served by a ceiling diffuser, with
    the ear-height rule, the three standoff distances, the alternative slow
    room scan, and the clause 5.3.2 screen that decides whether a single
    spectrum may be rated at all.
    """
    # 72 px per metre: a 6.0 m x 2.7 m section, floor at y = 408.
    ppm = 72.0
    x0, x1 = 118.0, 118.0 + 6.0 * ppm  # 6.0 m span
    floor_y, ceil_y = 408.0, 408.0 - 2.7 * ppm

    # --- Room shell, plenum and the air-handling plant ----------------------
    s.rect(x0, ceil_y, x1 - x0, floor_y - ceil_y, th.panel, th.fg, sw=2.6)
    s.rect(x0, ceil_y - 42, x1 - x0, 42, th.bg, th.muted, sw=1.6)
    s.text(x1 - 10, ceil_y - 15, "ceiling plenum", 12, th.muted, "end")
    s.ground(floor_y, x0 - 22, x1 + 22)

    # Duct run in the plenum, branching into one diffuser.
    s.rect(x0 + 18, ceil_y - 36, 206, 24, th.panel, th.secondary, rx=4, sw=1.8)
    s.text(x0 + 121, ceil_y - 19, "supply duct", 12, th.secondary, "middle")
    dif_x = x0 + 300
    s.line(x0 + 224, ceil_y - 24, dif_x, ceil_y - 24, th.secondary, 1.8)
    s.line(dif_x, ceil_y - 24, dif_x, ceil_y - 10, th.secondary, 1.8)
    s.path(
        f"M {dif_x - 24} {ceil_y - 10} L {dif_x + 24} {ceil_y - 10} "
        f"L {dif_x + 13} {ceil_y + 6} L {dif_x - 13} {ceil_y + 6} Z",
        fill=th.panel,
        stroke=th.secondary,
        sw=1.8,
    )
    s.text(dif_x + 38, ceil_y + 28, "diffuser", 12, th.secondary, "start")
    for r in (30.0, 52.0):
        s.path(
            f"M {dif_x - r * 0.75:.1f} {ceil_y + 6 + r * 0.66:.1f} "
            f"A {r} {r} 0 0 1 {dif_x + r * 0.75:.1f} {ceil_y + 6 + r * 0.66:.1f}",
            stroke=th.secondary,
            sw=1.2,
            dash="4,4",
        )
    # Air handler beyond the wall, at its design operating condition.
    s.rect(x0 - 74, ceil_y - 30, 54, 72, th.panel, th.secondary, rx=5, sw=2)
    s.circle(x0 - 47, ceil_y + 6, 15, "none", th.secondary, 1.8)
    s.line(x0 - 58, ceil_y - 5, x0 - 36, ceil_y + 17, th.secondary, 1.6)
    s.line(x0 - 58, ceil_y + 17, x0 - 36, ceil_y - 5, th.secondary, 1.6)
    s.text(x0 - 47, ceil_y + 62, "air handler", 12, th.secondary, "middle")
    s.text(x0 - 47, ceil_y + 80, "design condition", 10, th.muted, "middle")

    # --- Standoff exclusion zones (clause 5.2.5) ----------------------------
    # 0.6 m from any single reflecting surface: a band under the ceiling.
    s.rect(x0, ceil_y, x1 - x0, 0.6 * ppm, "none", th.accent, sw=1.2, dash="5,4")
    s.text(x0 + 130, ceil_y + 29, "0.6 m", 13, th.accent, "middle", bold=True)
    # 1.2 m from a two-surface intersection: the right wall-floor edge.
    r2 = 1.2 * ppm
    s.path(
        f"M {x1 - r2} {floor_y} A {r2} {r2} 0 0 0 {x1} {floor_y - r2} Z",
        fill="none",
        stroke=th.accent,
        sw=1.4,
        dash="5,4",
    )
    s.text(x1 - 40, floor_y - 34, "1.2 m", 13, th.accent, "middle", bold=True)
    # 2.4 m from a trihedral corner: the left wall meeting floor and end wall.
    r3 = 2.4 * ppm
    s.path(
        f"M {x0 + r3} {floor_y} A {r3} {r3} 0 0 1 {x0} {floor_y - r3} Z",
        fill="none",
        stroke=th.accent,
        sw=1.4,
        dash="5,4",
    )
    s.text(x0 + 132, floor_y - 112, "2.4 m", 13, th.accent, "start", bold=True)

    # --- Alternative: the slow scan of the whole space ----------------------
    scan_y = floor_y - 1.05 * ppm
    s.path(
        f"M {x0 + 150} {scan_y + 16} C {x0 + 210} {scan_y - 18}, "
        f"{x0 + 275} {scan_y + 18}, {x0 + 340} {scan_y - 16} S "
        f"{x0 + 400} {scan_y + 16}, {x1 - 30} {scan_y - 8}",
        stroke=th.primary,
        sw=1.8,
        dash="7,5",
    )

    # --- Seated occupant and the microphone at ear height -------------------
    s.person(x0 + 70, floor_y, h=1.2 * ppm, seated=True)
    mic_x = x0 + 330
    s.mic(mic_x, floor_y - 1.2 * ppm, floor_y, scale=0.85)
    s.dim(
        mic_x - 46,
        floor_y,
        mic_x - 46,
        floor_y - 1.2 * ppm,
        "1.2 m",
        offset=0,
        size=15,
        label_side="left",
    )
    # The ghosted standing ear height on the same stand.
    s.circle(mic_x, floor_y - 1.6 * ppm, 6, "none", th.muted, 1.4)
    s.text(mic_x + 14, floor_y - 1.6 * ppm + 5, "1.6 m", 12, th.muted, "start")

    # --- Reading the section ------------------------------------------------
    # Below the right-hand column's last line rather than beside it: in
    # Spanish this caption runs to x = 610 and the column starts at 616.
    s.text(
        x0 - 74,
        462,
        "$L_{EQ}$ at the named position — or scan the whole "
        "space at ≤ 0.5 m/s for ≥ 20 s",
        13,
        th.primary,
        "start",
        bold=True,
    )
    s.text(
        x0 - 74,
        480,
        "green dashed: microphone exclusion zones (5.2.5)",
        12,
        th.muted,
        "start",
    )

    # --- Right column: heights, standoffs and the meter ---------------------
    cx = 616.0
    s.text(cx, 88, "Microphone height (5.2.5)", 15, th.fg, "start", bold=True)
    for i, (who, height) in enumerate(
        (
            ("Adult, standing", "1.6 m"),
            ("Adult, seated", "1.2 m"),
            ("Child, standing", "1.1 m"),
            ("Child, seated", "0.75 m"),
        )
    ):
        s.text(cx, 114 + i * 24, who, 13, th.fg, "start")
        s.text(878, 114 + i * 24, height, 13, th.primary, "end", bold=True)
    s.line(cx, 206, 878, 206, th.muted, 1.0)

    s.text(cx, 232, "Standoff (5.2.5)", 15, th.fg, "start", bold=True)
    for i, (what, dist) in enumerate(
        (
            ("One reflecting surface", "≥ 0.6 m"),
            ("Two surfaces meeting", "≥ 1.2 m"),
            ("Three surfaces meeting", "≥ 2.4 m"),
        )
    ):
        s.text(cx, 258 + i * 24, what, 13, th.fg, "start")
        s.text(878, 258 + i * 24, dist, 13, th.accent, "end", bold=True)
    s.line(cx, 326, 878, 326, th.muted, 1.0)

    s.text(cx, 352, "Instrument and condition", 15, th.fg, "start", bold=True)
    for i, line in enumerate(
        (
            "Integrating-averaging, $L_{EQ}$",
            "Class 2 minimum (5.1.1)",
            "Octave bands 16 Hz – 8 kHz",
            "Room unoccupied, plant running",
        )
    ):
        s.text(cx, 378 + i * 22, line, 12, th.fg, "start")

    # --- The screen that decides whether a single spectrum may be rated -----
    s.rect(44, 488, 812, 96, th.panel, th.secondary, rx=10, sw=2.2)
    s.text(
        64,
        514,
        "Before rating (5.3.2): is the noise steady?",
        15,
        th.secondary,
        "start",
        bold=True,
    )
    s.text(
        64,
        538,
        "screen 16, 31.5 and 63 Hz aurally and on a fast, "
        "Z-weighted meter, then check $L_{MAX} − L_{EQ}$ and "
        "$L_{10} − L_{EQ}$",
        13,
        th.fg,
        "start",
    )
    s.text(
        64,
        560,
        "against Table 3 — a field that fails belongs to RNC "
        "(clause 5.3), not to NC or RC",
        13,
        th.fg,
        "start",
    )


def _d_enclosed_space_absorption(s: SVG, th: Theme) -> None:
    """Absorption area and reverberation time of a room (EN 12354-6:2003)."""
    cx = 450.0
    bw, bh = 660.0, 58.0
    x0 = cx - bw / 2

    # --- Inputs (two feeder boxes) -----------------------------------------
    iw = 320.0
    s.rect(cx - bw / 2, 48, iw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(
        cx - bw / 2 + iw / 2,
        72,
        "Surfaces  ($S_i$, $α_{s,i}$)",
        15,
        th.fg,
        "middle",
        bold=True,
    )
    s.text(
        cx - bw / 2 + iw / 2, 92, "area and absorption per band", 11, th.muted, "middle"
    )
    s.rect(cx + bw / 2 - iw, 48, iw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(
        cx + bw / 2 - iw / 2, 72, "Objects  ($V_{obj}$)", 15, th.fg, "middle", bold=True
    )
    s.text(
        cx + bw / 2 - iw / 2,
        92,
        "$A_{obj} = V_{obj}^{2/3}$  (Formula 4)",
        11,
        th.muted,
        "middle",
    )
    s.arrow(cx - bw / 2 + iw / 2, 106, cx - 60, 150, th.fg, 1.8)
    s.arrow(cx + bw / 2 - iw / 2, 106, cx + 60, 150, th.fg, 1.8)

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 25, l1, 15, th.fg, "middle", bold=True)
        s.text(cx, y + 45, l2, 11, th.muted, "middle")

    _step(
        150,
        "Equivalent absorption area  $A$  (clause 4.3, Formula 1)",
        "$A = Σ α_{s,i}·S_i + Σ A_{obj} + A_{air}$;   "
        "$A_{air} = 4·m·V·(1 − ψ)$  (Formula 2)",
        th.primary,
    )
    _step(
        238,
        "Object fraction  $ψ = Σ V_{obj} / V$   (Formula 3)",
        "air absorption negligible below 1 kHz for $V < 200$ m³",
        th.fg,
    )
    s.arrow(cx, 210, cx, 238, th.fg, 1.8)
    s.arrow(cx, 296, cx, 324, th.fg, 1.8)

    # --- Output -------------------------------------------------------------
    s.rect(x0, 324, bw, 58, "none", th.primary, rx=10, sw=2.4)
    s.text(
        cx,
        349,
        "Reverberation time  $T = 55.3/c_0 · V·(1 − ψ) / A$  (Formula 5)",
        15,
        th.fg,
        "middle",
        bold=True,
    )
    s.text(
        cx,
        369,
        "$c_0$ = 345.6 m/s so $55.3/c_0$ = 0.16  (clause 4.4)",
        11,
        th.muted,
        "middle",
    )


def _d_open_plan(s: SVG, th: Theme) -> None:
    """ISO 3382-3 open-plan measurement line and its single-number quantities."""
    ly = 150.0
    lx0, lx1 = 120.0, 812.0
    # Talker/source near the origin.
    s.person(lx0, ly, h=70)
    s.text(lx0, ly + 22, "source", 11, th.muted, "middle")
    s.text(lx0, ly + 40, "($r_0$ = 1 m)", 11, th.muted, "middle")
    # Measurement line with workstations and positions.
    s.line(lx0 + 26, ly - 30, lx1, ly - 30, th.fg, 1.8, dash="6,5")
    dists = [
        (0.18, "2 m"),
        (0.36, "4 m"),
        (0.56, "8 m"),
        (0.78, "12 m"),
        (0.98, "16 m"),
    ]
    for frac, lab in dists:
        px = lx0 + 26 + frac * (lx1 - lx0 - 26)
        s.rect(px - 22, ly + 4, 44, 26, th.panel, th.muted, rx=4, sw=1.3)  # desk
        s.circle(px, ly - 30, 5, th.primary)  # measurement position
        s.text(px, ly - 42, lab, 11, th.fg, "middle")
    # Evaluation-range bracket (2 m to 16 m).
    bx0 = lx0 + 26 + 0.18 * (lx1 - lx0 - 26)
    bx1 = lx0 + 26 + 0.98 * (lx1 - lx0 - 26)
    s.line(bx0, ly + 52, bx1, ly + 52, th.accent, 1.6)
    s.line(bx0, ly + 46, bx0, ly + 58, th.accent, 1.6)
    s.line(bx1, ly + 46, bx1, ly + 58, th.accent, 1.6)
    s.text(
        (bx0 + bx1) / 2,
        ly + 74,
        "spatial-decay fit range (2 m to 16 m)",
        12,
        th.accent,
        "middle",
    )

    chips = [
        ("$D_{2,S}$", "spatial decay rate", "dB per doubling · Cl. 6.2", th.primary),
        # The 4 m of the level's subscript is a value with its unit, which
        # the composer has no roman run for yet; the chip stays plain until
        # that case is adjudicated.
        ("Lp,A,S,4m", "speech level at 4 m", "A-weighted · Cl. 3.3", th.primary),
        ("$r_D$", "distraction distance", "fitted STI = 0.50 · Cl. 3.6", th.secondary),
        ("$r_P$", "privacy distance", "fitted STI = 0.20 · Cl. 3.7", th.secondary),
    ]
    cw, cgap = 190.0, 14.0
    cx = (900 - (len(chips) * cw + (len(chips) - 1) * cgap)) / 2
    s.text(450, 306, "what open_plan_metrics returns", 12, th.muted, "middle")
    for sym, name, note, color in chips:
        s.rect(cx, 320, cw, 118, th.panel, color, rx=10, sw=2)
        s.text(cx + cw / 2, 356, sym, 19, th.fg, "middle", bold=True)
        s.text(cx + cw / 2, 384, name, 13, color, "middle", bold=True)
        s.text(cx + cw / 2, 412, note, 10, th.muted, "middle")
        cx += cw + cgap
    s.text(
        450,
        464,
        "Clause 4 also requires the average A-weighted background "
        "noise  $L_{p,A,B}$  (Cl. 6.4)",
        11,
        th.muted,
        "middle",
    )


def _d_iso12999(s: SVG, th: Theme) -> None:
    """ISO 12999-1 uncertainty: from tabulated reproducibility to the expanded U."""
    cx = 450.0
    bw, bh = 664.0, 60.0
    x0 = cx - bw / 2

    s.rect(x0, 48, bw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(
        cx,
        72,
        "Standard uncertainty  $u$  — reproducibility read from the tables",
        15,
        th.fg,
        "middle",
        bold=True,
    )
    s.text(
        cx,
        92,
        "bands: Tables 2/4 · ratings: Tables 3/5 · situation A ($σ_R$) / "
        "B ($σ_{situ}$) / C ($σ_r$)",
        11,
        th.muted,
        "middle",
    )
    s.arrow(cx, 108, cx, 138, th.fg, 1.8)

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 25, l1, 15, th.fg, "middle", bold=True)
        s.text(cx, y + 45, l2, 11, th.muted, "middle")

    _step(
        138,
        "Reduce by  $m$  independent measurements   $u/√m$   (Formula A.7)",
        "and combine model with reality per Annex A when predicting",
        th.fg,
    )
    _step(
        226,
        "Combine uncorrelated contributions   $u_c = √(Σ u_i^2)$   (Formula C.2)",
        "single-number combination of Annex B uses Formula B.2",
        th.primary,
    )
    _step(
        314,
        "Expand   $U = k·u$   (Formula 2),   $k$ from Table 8   ($k ≥ 1$)",
        "the coverage factor depends on the reported quantity and situation",
        th.secondary,
    )
    for y0, y1 in ((198, 226), (286, 314)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)
    s.arrow(cx, 374, cx, 404, th.fg, 1.8)

    # Two-sided reporting vs one-sided conformity.
    hw = 320.0
    s.rect(x0, 404, hw, 66, "none", th.primary, rx=10, sw=2.2)
    s.text(
        x0 + hw / 2,
        430,
        "Report   $Y = y ± U$   (Formula 3)",
        14,
        th.fg,
        "middle",
        bold=True,
    )
    s.text(x0 + hw / 2, 452, "two-sided coverage factor", 11, th.muted, "middle")
    s.rect(cx + bw / 2 - hw, 404, hw, 66, "none", th.secondary, rx=10, sw=2.2)
    s.text(
        cx + bw / 2 - hw / 2,
        430,
        "Declare conformity   (Formulae 4/5)",
        14,
        th.fg,
        "middle",
        bold=True,
    )
    s.text(
        cx + bw / 2 - hw / 2, 452, "one-sided coverage factor", 11, th.muted, "middle"
    )


# ---------------------------------------------------------------------------
# Reception plate (EN 15657)
# ---------------------------------------------------------------------------


def _d_reception_plate(s: SVG, th: Theme) -> None:
    """EN 15657 reception plate: source machine on a resiliently supported
    plate, averaged plate velocity, plate power balance.
    """
    # ===== Source machine standing on the plate =====
    mx = 260.0
    s.text(mx, 150, "Source under test (pump, fan, boiler …)", 17, th.fg, bold=True)
    s.rect(mx - 75, 218, 150, 70, th.panel, th.fg, rx=8, sw=2.2)
    s.rect(mx - 48, 190, 58, 28, th.panel, th.muted, rx=6, sw=1.8)
    s.circle(mx + 40, 240, 12, th.bg, th.muted, 1.8)
    for fx in (mx - 52.0, mx + 52.0):
        s.rect(fx - 8, 288, 16, 14, th.fg, rx=2)
        s.arrow(fx, 306, fx, 326, th.secondary, 2.4)
    s.text(310, 356, "injected structure-borne power", 13, th.secondary, italic=True)

    # ===== Reception plate on resilient supports =====
    s.rect(100, 302, 460, 32, th.panel, th.primary, rx=3, sw=2.4)
    s.text(430, 324, "Reception plate  ($m$, $S$, $η$)", 14, th.fg, bold=True)
    for ax_ in (140.0, 190.0, 400.0, 500.0):
        _accel(s, ax_, 302)
    s.text(560, 272, "velocity positions → $L_v$", 14, th.secondary, anchor="end")
    for sx in (150.0, 510.0):
        _spring_v(s, sx, 334, 430, th.accent, coils=3)
    s.ground(430, 80, 580)
    s.text(330, 404, "resilient supports", 12, th.muted)

    # ===== Right column: the power balance and the source quantities =====
    s.text(735, 150, "Plate power balance", 17, th.fg, bold=True)
    s.rect(590, 172, 292, 148, "none", th.muted, rx=10, dash="6,5")
    s.text(735, 206, "$P = ω·η·(m·S)·⟨v^2⟩$", 16, th.primary, bold=True)
    s.text(735, 238, "$η = 2.2 / (f·T_s)$   (Formula 13)", 13, th.fg)
    # Longest line of the panel: a smaller face keeps it inside the dashed box.
    s.text(735, 270, "$L_{Ws} = 10 log_{10}(2πf·η·m·S / f_0 m_0 S_0)$", 11, th.fg)
    s.text(735, 296, "$+ L_v − 60$   (Formula 14)", 12, th.fg)
    s.text(735, 366, "→ source quantities (Formulae 15–19):", 13, th.fg, bold=True)
    s.text(735, 394, "equivalent blocked force $L_{Fb,eq}$ ,", 13, th.muted)
    s.text(735, 418, "$L_{Wsn}$ consumed by EN 12354-5", 13, th.muted)

    # Footer: the spatial velocity average.
    s.text(
        450,
        516,
        "spatial average:  Lv = 10 log10[(1/N)·Σ 10^(Lv,i/10)]   (Formula 12)",
        15,
        th.fg,
        mono=True,
    )


# ---------------------------------------------------------------------------
# Installed structure-borne sound paths (EN 12354-5)
# ---------------------------------------------------------------------------


def _d_installed_paths(s: SVG, th: Theme) -> None:
    """EN 12354-5: service equipment on a floor slab, structure-borne paths
    into the receiving room below, and the prediction cascade.
    """
    bx0, bx1 = 80.0, 590.0
    top, slab_top, slab_bot, bot = 92.0, 296.0, 324.0, 528.0

    # Rooms, continuous floor slab and flanking wall (drawn over the slab).
    s.rect(bx0, top, bx1 - bx0, slab_top - top, th.panel, th.fg, sw=2.5)
    s.rect(bx0, slab_bot, bx1 - bx0, bot - slab_bot, th.panel, th.fg, sw=2.5)
    s.rect(bx0, slab_top, bx1 - bx0 + 26, slab_bot - slab_top, th.panel, th.fg, sw=2)
    for hx in range(int(bx0) + 16, int(bx1) + 26, 34):
        s.line(hx, slab_top, hx - 12, slab_bot, th.muted, 0.9)
    s.rect(bx1, top, 26, bot - top, th.panel, th.fg, sw=2)
    s.text(bx0 + 16, top + 32, "Source room", 18, th.fg, bold=True, anchor="start")
    s.text(bx0 + 16, bot - 18, "Receiving room", 18, th.fg, bold=True, anchor="start")

    # Service equipment on resilient mounts on the slab.
    mx = 210.0
    s.rect(mx - 55, 238, 110, 42, th.panel, th.fg, rx=7, sw=2.2)
    s.rect(mx - 34, 216, 40, 22, th.panel, th.muted, rx=5, sw=1.6)
    for fx in (mx - 36.0, mx + 36.0):
        _spring_v(s, fx, 280, slab_top, th.accent, coils=2, width=6.0, sw=1.6)
    s.text(mx, 200, "Service equipment (pump)", 16, th.fg, bold=True)
    s.text(
        mx + 78, 268, "coupling $D_C$   (Formula 19b)", 13, th.secondary, anchor="start"
    )

    # Path i = j: the excited slab radiates into the room below.
    s.arrow(mx, slab_bot + 2, mx, slab_bot + 40, th.secondary, 2.4)
    for r in (40, 66, 92):
        s.path(
            f"M {mx - r * 0.72:.1f} {slab_bot + 42 + r * 0.5:.1f} "
            f"A {r} {r} 0 0 0 {mx + r * 0.72:.1f} {slab_bot + 42 + r * 0.5:.1f}",
            stroke=th.accent,
            sw=1.6,
        )
    s.text(mx, 484, "excited floor radiates (path $i = j$)", 13, th.secondary)

    # Path i -> j: along the slab, through the junction, down the wall.
    s.line(mx + 40, 310, 596, 310, th.primary, 2.6)
    s.line(603, 310, 603, 420, th.primary, 2.6)
    s.arrow(603, 420, 574, 420, th.primary, 2.6)
    s.circle(603, 310, 5, th.bg, th.fg, 2)
    for r in (30, 52):
        s.path(
            f"M {588 - r * 0.5:.1f} {420 - r * 0.72:.1f} "
            f"A {r} {r} 0 0 0 {588 - r * 0.5:.1f} {420 + r * 0.72:.1f}",
            stroke=th.accent,
            sw=1.6,
        )
    s.text(
        584,
        288,
        "path along the slab into the wall  ($i$ → $j$)",
        13,
        th.primary,
        anchor="end",
    )

    # ===== Right column: the prediction cascade =====
    s.text(760, 120, "Prediction cascade", 17, th.fg, bold=True)
    # The energetic sum is far longer than the other terms, so it carries its
    # own face to stay inside the column instead of running off the canvas.
    steps = [
        ("$L_{Ws,c}$", "characteristic power (EN 15657)", th.fg, 16),
        ("$− D_C$", "coupling at the contacts (19b)", th.secondary, 16),
        ("$L_{Ws,inst}$", "installed power (18b)", th.fg, 16),
        ("$− D_{sa} − R_{ij,ref}$", "per transmission path (18a)", th.primary, 16),
        # The energetic sum sets a subscripted level inside an exponent,
        # which the composer's single script level cannot carry yet; the
        # term stays plain until that case is adjudicated.
        ("10 log10 Σ 10^(L_n,s,ij/10)", "energetic sum $L_{n,s}$ (17)", th.accent, 13),
    ]
    y = 164.0
    for k, (term, caption, col, size) in enumerate(steps):
        s.text(760, y, term, size, col, bold=True, mono="$" not in term)
        s.text(760, y + 22, caption, 12, th.muted)
        if k < len(steps) - 1:
            s.arrow(760, y + 34, 760, y + 56, th.muted, 1.6)
        y += 84

    # Footer: what a path is.
    s.text(
        335,
        574,
        "each path $i$ → $j$: excited element $i$, radiating element $j$ "
        "in the receiving room",
        14,
        th.muted,
    )


# ---------------------------------------------------------------------------
# Laboratory sound insulation suite (ISO 10140)
# ---------------------------------------------------------------------------


def _d_insulation_lab(s: SVG, th: Theme) -> None:
    """ISO 10140 laboratory transmission suite in plan view: two
    structurally decoupled reverberant rooms, the test element mounted in
    the ~10 m2 test opening, a corner loudspeaker in the source room and a
    continuously moving (rotating) microphone in each room.
    """
    top = 92.0
    sc = 72.0  # px per metre
    src_bot = top + 4.4 * sc  # source room 5.0 m x 4.4 m
    rec_bot = top + 4.1 * sc  # receiving room 4.6 m x 4.1 m
    rec_r = 470.0 + 4.6 * sc

    # Room shells (separate structures).
    s.rect(70, top, 360, src_bot - top, th.panel, th.fg, rx=4, sw=3)
    s.rect(470, top, rec_r - 470, rec_bot - top, th.panel, th.fg, rx=4, sw=3)
    s.text(90, top + 30, "Source room", 18, th.fg, bold=True, anchor="start")
    s.text(90, top + 56, "$V_1 ≈ 59$ m³", 15, th.muted, anchor="start")
    s.text(486, top + 30, "Receiving room", 18, th.fg, bold=True, anchor="start")
    s.text(486, top + 56, "$V_2 ≈ 51$ m³", 15, th.muted, anchor="start")

    # Test opening (3.75 m in plan) with the specimen mounted; filler stubs
    # from each shell with an air gap between them (the structural break).
    op_t, op_b = 110.0, 380.0
    s.rect(430, top, 14, op_t - top, th.panel, th.fg, sw=1.6)
    s.rect(430, op_b, 14, src_bot - op_b, th.panel, th.fg, sw=1.6)
    s.rect(456, top, 14, op_t - top, th.panel, th.fg, sw=1.6)
    s.rect(456, op_b, 14, rec_bot - op_b, th.panel, th.fg, sw=1.6)
    s.rect(438, op_t, 24, op_b - op_t, th.panel, th.secondary, sw=2)
    for hy in range(int(op_t) + 12, int(op_b), 16):
        s.line(440, hy + 10, 460, hy - 4, th.secondary, 1.0)
    s.text(450, 66, "structural break", 13, th.muted, italic=True)
    s.line(450, 72, 450, 102, th.muted, 1.0, dash="3,3")
    s.text(450, 452, "Test element in the test opening", 15, th.secondary, bold=True)
    s.line(450, op_b + 4, 450, 436, th.muted, 1.0, dash="3,3")

    # Loudspeaker in a corner of the source room.
    lsx, lsy = 150.0, 350.0
    for r in (36, 60, 84):
        s.path(
            f"M {lsx + r * 0.22:.1f} {lsy - r:.1f} "
            f"A {r} {r} 0 0 1 {lsx + r:.1f} {lsy - r * 0.22:.1f}",
            stroke=th.accent,
            sw=1.5,
        )
    s.rect(lsx - 24, lsy - 27, 48, 54, th.panel, th.primary, rx=6, sw=2)
    s.circle(lsx, lsy - 9, 11, th.primary)
    s.circle(lsx, lsy - 9, 4.5, th.bg)
    s.circle(lsx, lsy + 15, 6, th.primary)
    s.text(lsx + 4, lsy + 48, "Loudspeaker", 15, th.fg, bold=True)

    # Continuously moving (rotating) microphone in each room: the sweep
    # circle, the boom and the microphone on its tip.
    for mcx, mcy, a_mic in ((285.0, 200.0, 40.0), (640.0, 215.0, 150.0)):
        import math

        s.ellipse(mcx, mcy, sc, sc, "none", th.muted, 1.3, dash="6,5")
        pxm = mcx + sc * math.cos(math.radians(a_mic))
        pym = mcy - sc * math.sin(math.radians(a_mic))
        s.line(mcx, mcy, pxm, pym, th.fg, 2.0)
        s.circle(mcx, mcy, 4, th.fg)
        s.circle(pxm, pym, 7.5, th.secondary)
        s.circle(pxm, pym, 2.6, th.bg)
        _rot_arrow(s, mcx, mcy, sc + 12, -78, -8, th.accent, 1.8)
    s.text(285, 298, "moving microphone", 14, th.fg)
    s.text(285, 320, "sweep radius ≥ 1 m", 13, th.muted)
    s.text(640, 313, "moving microphone", 14, th.fg)

    # Dimensions (72 px per metre).
    s.dim(70, src_bot, 430, src_bot, "5.0 m", offset=30, size=15)
    s.dim(470, rec_bot, rec_r, rec_bot, "4.6 m", offset=30 + src_bot - rec_bot, size=15)
    s.dim(rec_r, top, rec_r, rec_bot, "4.1 m", offset=32, size=15, label_side="right")
    s.dim(430, op_t, 430, op_b, "3.75 m", offset=-24, size=15)

    # Normative facility limits.
    for y, txt in (
        (508.0, "Test opening ≈ 10 m² (3.75 m × 2.7 m); shorter edge ≥ 2.3 m"),
        (536.0, "Room volumes ≥ 50 m³, differing by at least 10 %"),
        (564.0, "Continuously moving microphone: sweep radius ≥ 1 m, traverse ≥ 15 s"),
    ):
        s.text(80, y, txt, 15, th.fg, anchor="start")


# ---------------------------------------------------------------------------
# Reverberation-time prediction (Sabine / Eyring)
# ---------------------------------------------------------------------------


def _d_reverberation_prediction(s: SVG, th: Theme) -> None:
    """The guide's 10 x 7 x 3.5 m room through the Sabine and Eyring
    absorption terms into the per-band T60 table the library returns,
    with the diffuse-field validity note.
    """
    # Room data
    s.rect(170, 52, 560, 78, th.panel, th.fg, rx=10, sw=2)
    s.text(
        450,
        77,
        "Room 10 × 7 × 3.5 m — $V$ = 245 m³, $S$ = 259 m²",
        14,
        th.fg,
        bold=True,
    )
    s.text(
        450,
        97,
        "hard end walls, lightly treated side walls, carpet and acoustic ceiling",
        10,
        th.muted,
    )
    s.text(
        450,
        117,
        "mean absorption $ᾱ$ runs from 0.21 at 125 Hz to 0.51 at 4 kHz",
        10,
        th.muted,
    )

    # The two models
    s.rect(60, 178, 380, 96, th.panel, th.primary, rx=10, sw=2)
    s.text(250, 203, "Sabine", 13, th.fg, bold=True)
    s.text(250, 224, "$T = 0.161·V / (Σ S_i·α_i + 4·m·V)$", 11, th.primary)
    s.text(250, 245, "low, even absorption ($ᾱ$ up to ≈ 0.2);", 10, th.muted)
    s.text(250, 262, "stays finite even at $α$ = 1", 10, th.muted)
    s.rect(460, 178, 380, 96, th.panel, th.secondary, rx=10, sw=2)
    s.text(650, 203, "Eyring", 13, th.fg, bold=True)
    s.text(650, 224, "$T = 0.161·V / (−S·ln(1 − ᾱ) + 4·m·V)$", 11, th.secondary)
    s.text(650, 245, "strong, even absorption;", 10, th.muted)
    s.text(650, 262, "reaches $T$ = 0 at total absorption", 10, th.muted)
    s.arrow(350, 130, 265, 174, th.fg, 1.8)
    s.arrow(550, 130, 635, 174, th.fg, 1.8)

    # Per-band table
    s.rect(100, 318, 700, 150, th.panel, th.fg, rx=10, sw=1.8)
    s.text(450, 344, "Predicted $T_{60}$ per octave band", 12, th.fg, bold=True)
    freqs = ("125 Hz", "250", "500", "1k", "2k", "4k")
    sab = ("0.74", "0.47", "0.37", "0.31", "0.30", "0.30")
    eyr = ("0.66", "0.39", "0.29", "0.23", "0.21", "0.22")
    xc = [292.0 + 94.0 * i for i in range(6)]
    for x, f in zip(xc, freqs, strict=True):
        s.text(x, 372, f, 10, th.muted, bold=True)
    s.line(130, 382, 770, 382, th.muted, 1.0)
    s.text(130, 404, "Sabine [s]", 10, th.primary, bold=True, anchor="start")
    for x, v in zip(xc, sab, strict=True):
        s.text(x, 404, v, 11, th.fg)
    s.text(130, 432, "Eyring [s]", 10, th.secondary, bold=True, anchor="start")
    for x, v in zip(xc, eyr, strict=True):
        s.text(x, 432, v, 11, th.fg)
    s.text(
        450,
        456,
        "Eyring runs 11 to 29 % shorter here: $ᾱ$ is past Sabine's comfort zone",
        10,
        th.muted,
    )
    s.arrow(250, 274, 320, 314, th.fg, 1.8)
    s.arrow(650, 274, 580, 314, th.fg, 1.8)

    # Validity note
    s.rect(130, 500, 640, 68, "none", th.accent, rx=10, sw=1.6, dash="6,5")
    s.text(
        450,
        527,
        "Domain of validity: a diffuse field that stays diffuse while it decays",
        12,
        th.accent,
        bold=True,
    )
    s.text(
        450,
        551,
        "below the Schroeder frequency, in coupled volumes "
        "and in corridor-like rooms no single $T_{60}$ exists",
        10,
        th.fg,
    )


# ---------------------------------------------------------------------------
# Panel between rooms: mass law and the coincidence dip
# ---------------------------------------------------------------------------


def _d_panel_insulation(s: SVG, th: Theme) -> None:
    """A single 12.5 mm plasterboard leaf (m'' = 8.75 kg/m2) mounted in its
    test opening under diffuse incidence, with the predicted R(f) of
    ``single_panel_transmission_loss`` inset: the mass-law rise and the
    coincidence dip at the fc = 2619 Hz of this leaf (Rw = 27 dB).
    """
    # --- test opening: heavy filler above and below, the leaf between ------
    px_l, px_r = 380.0, 396.0
    op_t, op_b = 108.0, 332.0
    s.rect(348, 62, 80, op_t - 62, th.panel, th.fg, sw=2)
    s.rect(348, op_b, 80, 46, th.panel, th.fg, sw=2)
    s.rect(px_l, op_t, px_r - px_l, op_b - op_t, th.panel, th.secondary, sw=2)
    for hy in range(int(op_t) + 10, int(op_b) - 2, 14):
        s.line(px_l + 1, hy + 8, px_r - 1, hy - 4, th.secondary, 1.0)
    s.text(388, 52, "Panel under test: 12.5 mm plasterboard", 16, th.fg, bold=True)

    # Thickness callout (witness lines up, arrows pointing inward).
    s.line(px_l, op_t, px_l, 88, th.muted, 0.9, dash="3,3")
    s.line(px_r, op_t, px_r, 88, th.muted, 0.9, dash="3,3")
    s.arrow(352.0, 92.0, px_l - 2, 92.0, th.muted, 1.2)
    s.arrow(424.0, 92.0, px_r + 2, 92.0, th.muted, 1.2)
    s.text(434, 97, "12.5 mm", 13, th.fg, anchor="start")

    # --- diffuse incidence on the left, weaker transmitted field right -----
    s.text(180, 116, "Source room", 17, th.fg, bold=True)
    s.text(180, 140, "diffuse incidence", 14, th.muted, italic=True)
    s.arrow(258.0, 152.0, px_l - 6, 196.0, th.accent, 2.2)
    s.arrow(218.0, 244.0, px_l - 6, 246.0, th.accent, 2.2)
    s.arrow(252.0, 330.0, px_l - 6, 292.0, th.accent, 2.2)
    s.text(560, 116, "Receiving room", 17, th.fg, bold=True)
    s.text(510, 226, "transmitted", 14, th.muted, italic=True)
    s.arrow(px_r + 4, 196.0, 500.0, 172.0, th.primary, 1.5)
    s.arrow(px_r + 4, 246.0, 508.0, 246.0, th.primary, 1.5)
    s.arrow(px_r + 4, 292.0, 498.0, 318.0, th.primary, 1.5)

    # Bending wave travelling along the leaf (the coincidence mechanism).
    d = f"M 388 {op_t + 8:.0f}"
    y = op_t + 8
    sign = 1
    while y + 24 <= op_b - 8:
        d += f" Q {388 + sign * 9} {y + 12:.0f} 388 {y + 24:.0f}"
        y += 24
        sign = -sign
    s.path(d, stroke=th.accent, sw=2.0)
    s.text(300, 366, "bending wave at $f_c$", 13, th.accent, anchor="end")
    s.line(305.0, 360.0, 382.0, 326.0, th.muted, 1.0)
    s.text(388, 404, "$m″$ = 8.8 kg/m²", 14, th.fg)

    # --- inset: predicted R(f) with the coincidence dip --------------------
    ix0, iy0 = 572.0, 390.0  # axes origin (bottom-left)
    s.line(ix0, iy0, 850.0, iy0, th.muted, 1.3)
    s.arrow(ix0, iy0, ix0, 128.0, th.muted, 1.3)
    s.text(ix0 - 8, 140, "$R$", 12, th.muted, anchor="end")
    s.text(854, iy0 + 16, "$f$", 12, th.muted, anchor="end")
    s.text(645, 198, "predicted $R(f)$", 12, th.primary)
    import math

    def fx(f: float) -> float:
        return ix0 + math.log10(f / 50.0) * 135.0

    def ry(r: float) -> float:
        return 386.0 - r * 240.0 / 35.0

    for f_t, lab in ((100.0, "100"), (1000.0, "1k")):
        s.line(fx(f_t), iy0, fx(f_t), iy0 + 5, th.muted, 1.2)
        s.text(fx(f_t), iy0 + 20, lab, 10, th.muted)
    # single_panel_transmission_loss(bands, 8.75, fc=2619.3, eta=0.01), dB.
    curve = [
        (50, 5.3),
        (63, 7.2),
        (80, 9.2),
        (100, 11.1),
        (125, 13.0),
        (160, 15.1),
        (200, 17.0),
        (250, 18.9),
        (315, 20.9),
        (400, 23.0),
        (500, 24.9),
        (630, 26.9),
        (800, 29.0),
        (1000, 31.0),
        (1250, 32.9),
        (1600, 30.3),
        (2000, 26.9),
        (2500, 23.6),
        (3150, 25.3),
        (4000, 28.4),
        (5000, 31.3),
    ]
    d = ""
    for i, (f_c, r_c) in enumerate(curve):
        d += f"{'M' if i == 0 else ' L'} {fx(f_c):.1f} {ry(r_c):.1f}"
    s.path(d, stroke=th.primary, sw=2.4)
    fcx = fx(2619.3)
    s.line(fcx, iy0, fcx, 150.0, th.secondary, 1.3, dash="5,4")
    s.text(fcx, 142, "$f_c$ = 2.6 kHz", 12, th.secondary, bold=True)
    s.circle(fx(2500.0), ry(23.6), 4.0, th.secondary)
    s.text(690, 330, "+6 dB/octave", 12, th.primary, italic=True)
    s.text(628, 170, "$R_w$ = 27 dB", 14, th.fg, bold=True)

    # --- captions ----------------------------------------------------------
    s.text(
        80,
        452,
        "Diffuse-field mass law: $R$ rises 6 dB per octave and 6 dB per doubling of $m″$",
        15,
        th.fg,
        anchor="start",
    )
    s.text(
        80,
        480,
        "At $f_c = (c_0^2/2π) √(m″/B′)$ = 2619 Hz the free bending wave matches the trace wavelength",
        15,
        th.fg,
        anchor="start",
    )
    s.text(
        80,
        508,
        "Sharp's prediction rates $R_w$ = 27 dB; the dip takes the unfavourable deviations",
        15,
        th.primary,
        anchor="start",
        bold=True,
    )


# ---------------------------------------------------------------------------
# Image-source lattice in plan (first reflections of a shoebox room)
# ---------------------------------------------------------------------------


def _d_room_image_sources(s: SVG, th: Theme) -> None:
    """Plan of the guide's 7 x 5 x 3 m room with the source at (2, 1.6),
    the receiver at (5.2, 3.4) and the in-plane images of order 1 and 2 on
    the mirror-room grid, each labelled with its image_source_rir arrival
    time (direct 10.7 ms, first reflections 17.3 to 21.6 ms).
    """
    sc = 32.0  # px per metre

    def x(mx: float) -> float:
        return 98.0 + (mx + 7.5) * sc

    def y(my: float) -> float:
        return 88.0 + (10.4 - my) * sc

    # Mirror-room grid (3 x 3) around the bold real room.
    for gx in (-7.0, 0.0, 7.0):
        for gy_ in (-5.0, 0.0, 5.0):
            if gx == 0.0 and gy_ == 0.0:
                continue
            s.rect(
                x(gx),
                y(gy_ + 5.0),
                7.0 * sc,
                5.0 * sc,
                "none",
                th.muted,
                sw=1.1,
                dash="6,5",
            )
    s.rect(x(0.0), y(5.0), 7.0 * sc, 5.0 * sc, th.panel, th.fg, sw=2.6)
    s.text(
        x(3.5),
        y(0.4),
        "plan at the source plane $z$ = 1.5 m",
        10 if s.lang == "es" else 11,
        th.muted,
    )

    # Room dimensions on the real room's walls.
    s.dim(x(0.0), y(0.0), x(7.0), y(0.0), "7.0 m", offset=26, size=13)
    s.dim(
        x(7.0), y(5.0), x(7.0), y(0.0), "5.0 m", offset=28, size=13, label_side="right"
    )

    # Source, receiver and the direct sound.
    sx_, sy_ = x(2.0), y(1.6)
    rx_, ry_ = x(5.2), y(3.4)
    s.line(sx_, sy_, rx_, ry_, th.fg, 1.5)
    s.text((sx_ + rx_) / 2 + 4, (sy_ + ry_) / 2 + 18, "10.7 ms", 12, th.fg, mono=True)
    s.circle(sx_, sy_, 7.0, th.secondary)
    s.text(sx_ - 12, sy_ + 5, "$S$", 15, th.secondary, bold=True, anchor="end")
    s.path(
        f"M {rx_:.1f} {ry_ - 9:.1f} L {rx_ - 8:.1f} {ry_ + 7:.1f} "
        f"L {rx_ + 8:.1f} {ry_ + 7:.1f} Z",
        fill=th.primary,
    )
    s.text(rx_ + 13, ry_ + 5, "$R$", 15, th.primary, bold=True, anchor="start")

    # Example first reflection off the y = 5 wall: real specular path and
    # the equivalent straight line from the image.
    bx, by = x(4.176), y(5.0)
    s.line(sx_, sy_, bx, by, th.accent, 2.0)
    s.arrow(bx, by, rx_, ry_ - 6, th.accent, 2.0)
    s.line(x(2.0), y(8.4), rx_, ry_ - 6, th.accent, 1.3, dash="5,4")
    s.text(502, 185, "the image sees", 11, th.accent, italic=True)
    s.text(502, 204, "a straight path", 11, th.accent, italic=True)

    # Images of order 1 (secondary) and order 2 (accent), with their
    # image_source_rir arrival times.
    order1 = [
        (2.0, -1.6, "17.3 ms", 14, 18),
        (2.0, 8.4, "17.3 ms", 14, -12),
        (12.0, 1.6, "20.5 ms", 14, 18),
        (-2.0, 1.6, "21.6 ms", 14, 18),
    ]
    for mx, my, lab, _sz, dy_ in order1:
        s.circle(x(mx), y(my), 6.0, th.secondary)
        s.text(x(mx), y(my) + dy_ + (4 if dy_ < 0 else 0), lab, 11, th.fg, mono=True)
    order2 = [
        (-2.0, -1.6, "25.6 ms"),
        (-2.0, 8.4, "25.6 ms"),
        (12.0, -1.6, "24.6 ms"),
        (12.0, 8.4, "24.6 ms"),
    ]
    for mx, my, lab in order2:
        s.circle(x(mx), y(my), 6.0, "none", th.accent, 2.2)
        s.text(x(mx), y(my) + 20, lab, 11, th.muted, mono=True)

    # Legend inside the top-left mirror room.
    s.circle(x(-6.6), y(9.55), 5.5, th.secondary)
    s.text(x(-6.3), y(9.4), "1st order", 12, th.fg, anchor="start")
    s.circle(x(-6.6), y(8.75), 5.5, "none", th.accent, 2.2)
    s.text(x(-6.3), y(8.6), "2nd order", 12, th.fg, anchor="start")

    # --- captions ----------------------------------------------------------
    s.text(
        80,
        612,
        "every reflection is the free-field arrival of a mirror image: $t "
        "= r/c$, $√(1−α)$ per bounce, $1/(4πr)$",
        15,
        th.fg,
        anchor="start",
    )
    s.text(
        80,
        638,
        "in-plane images up to order 2 shown; the full lattice adds floor, ceiling and outer mirror rooms",
        15,
        th.muted,
        anchor="start",
    )


# ---------------------------------------------------------------------------
# EN 15657 reception plates: the low- and high-mobility rigs, and the bench
# ---------------------------------------------------------------------------


def _d_reception_plate_rigs(s: SVG, th: Theme) -> None:
    """The two plates EN 15657 clauses 7.2.2 and 7.3.2 specify, and the
    three-plate bench of its Figure 2, drawn with conforming dimensions.
    """
    top = 74.0

    # ===== Panel 1: the low-mobility plate in plan (clause 7.2.2) =====
    # 3,15 m x 2,23 m = 7,0 m2, drawn at 71,7 px per metre so the 0,5 m
    # position spacing of clause 7.1 is the length the dimension states.
    px0, py0, pw, ph = 48.0, top + 52.0, 226.0, 160.0
    cx1 = px0 + pw / 2
    # 18 px, not 19: the outer two panels are centred close enough to the
    # sheet edge that their Spanish headings run off it at 19.
    s.text(cx1, top + 22, "Low-mobility plate (7.2.2)", 15, th.fg, bold=True)
    s.rect(px0, py0, pw, ph, th.panel, th.fg, sw=2.4)
    for cx, cy in (
        (px0 + 13, py0 + 13),
        (px0 + pw - 13, py0 + 13),
        (px0 + 13, py0 + ph - 13),
        (px0 + pw - 13, py0 + ph - 13),
    ):
        s.rect(cx - 8, cy - 8, 16, 16, th.accent, th.fg, rx=3, sw=1.4)

    # Source footprint (0,9 m x 0,6 m) near the centre, contacts at its corners.
    fw, fh = 65.0, 43.0
    fx, fy = px0 + (pw - fw) / 2, py0 + (ph - fh) / 2
    s.rect(fx, fy, fw, fh, th.panel, th.primary, rx=5, sw=2.0, dash="7,4")
    for cx in (fx + 9, fx + fw - 9):
        for cy in (fy + 8, fy + fh - 8):
            s.circle(cx, cy, 4.0, th.primary)
    s.text(fx + fw / 2, fy - 9, "source", 13, th.primary, italic=True)

    # Six velocity positions in two columns, 0,5 m apart within a column.
    for col in (px0 + 36, px0 + pw - 36):
        for row in (py0 + 30, py0 + 66, py0 + 102):
            s.circle(col, row, 6.0, th.secondary, th.fg, 1.3)
    s.dim(
        px0 + 36,
        py0 + 30,
        px0 + 36,
        py0 + 66,
        "0,5 m",
        offset=0,
        size=13,
        label_side="right",
    )
    s.dim(
        px0,
        py0 + ph + 18,
        px0 + pw,
        py0 + ph + 18,
        "3,15 m x 2,23 m",
        offset=0,
        size=14,
    )

    for i, txt in enumerate(
        (
            "100 mm concrete, 2 300 ± 200 kg/m³",
            "$S$ = 7,0 m² (≥ 5 m²), sides ≈ $√2$ : 1",
            "$η ≥ 0,08$ over 50 Hz to 100 Hz",
            "≥ 6 velocity positions, ≈ 0,5 m apart",
            "and ≥ 0,1 m from any contact point",
            "elastic pads ≤ 100 × 100 mm",
        )
    ):
        s.text(cx1, py0 + ph + 58 + 24 * i, txt, 13, th.fg)

    # ===== Panel 2: the high-mobility plate (clause 7.3.2) =====
    cx2 = 450.0
    s.text(cx2, top + 22, "High-mobility plate (7.3.2)", 15, th.fg, bold=True)
    frame_y = top + 128.0
    s.rect(cx2 - 118, frame_y - 18, 236, 112, "none", th.muted, rx=5, sw=3.0)
    s.rect(cx2 - 104, frame_y, 208, 22, th.panel, th.fg, sw=2.2)
    for hx in range(int(cx2) - 94, int(cx2) + 100, 14):
        s.circle(hx, frame_y + 11, 3.2, th.bg, th.muted, 1.0)
    # Source bolted rigidly to the sheet (no springs: clause 7.3.3).
    s.rect(cx2 - 37, frame_y - 56, 74, 56, th.panel, th.primary, rx=6, sw=2.2)
    for bx in (cx2 - 23, cx2 + 23):
        s.line(bx, frame_y - 2, bx, frame_y + 22, th.fg, 2.6)
    s.text(cx2, frame_y - 66, "source bolted rigidly", 12, th.primary, italic=True)
    _accel(s, cx2 - 76, frame_y + 50)
    _accel(s, cx2 + 76, frame_y + 50)
    s.text(cx2, frame_y + 112, "support frame", 12, th.muted, italic=True)

    for i, txt in enumerate(
        (
            "1 mm steel or 1,5 mm aluminium",
            "$|Y| ≥ 10^{−2}$ m/(N·s)",
            "≈ 50 % perforated, ⌀ ≈ 6 mm holes,",
            "so the source's own airborne sound",
            "cannot drive the sheet",
            "$T_s$ and $Y$ measured with the",
            "source fitted (7.1)",
        )
    ):
        s.text(cx2, frame_y + 142 + 22 * i, txt, 13, th.fg)

    # ===== Panel 3: the Figure 2 three-plate bench =====
    cx3 = 748.0
    s.text(cx3, top + 22, "Three-plate bench (Fig. 2)", 15, th.fg, bold=True)
    bench_y = top + 138.0
    plate_x = (cx3 - 118, cx3 - 34, cx3 + 50)
    for bx in plate_x:
        s.rect(bx, bench_y, 68.0, 20, th.panel, th.fg, sw=2.2)
        _spring_v(
            s,
            bx + 34.0,
            bench_y + 20,
            bench_y + 58,
            th.accent,
            coils=2,
            width=7.0,
            sw=1.6,
        )
    s.ground(bench_y + 58, cx3 - 130, cx3 + 130)
    # A whirlpool bath bridging all three plates.
    s.path(
        f"M {cx3 - 104} {bench_y - 12} L {cx3 + 104} {bench_y - 12} "
        f"L {cx3 + 86} {bench_y - 58} L {cx3 - 86} {bench_y - 58} Z",
        fill=th.panel,
        stroke=th.primary,
        sw=2.2,
    )
    s.text(cx3, bench_y - 70, "whirlpool bath", 13, th.primary, italic=True)
    for bx in plate_x:
        s.line(bx + 34.0, bench_y - 12, bx + 34.0, bench_y, th.fg, 2.4)
    s.text(cx3, bench_y + 86, "> 10 dB between plates", 14, th.secondary, bold=True)

    for i, txt in enumerate(
        (
            "up to three isolated plates,",
            "for a source that touches",
            "several building elements",
            "the velocity level difference",
            "is measured per EN ISO 10848-1",
            "in every band, with the",
            "equipment removed",
        )
    ):
        s.text(cx3, bench_y + 122 + 22 * i, txt, 13, th.fg)

    # ===== Footer: which plate feeds which formula =====
    s.rect(48, 502, 804, 72, "none", th.muted, rx=10, dash="6,5")
    s.text(
        450,
        528,
        "low-mobility plate -> blocked force (15) -> "
        "characteristic power $L_{Wsn}$ (17)",
        13,
        th.fg,
    )
    s.text(
        450,
        556,
        "high-mobility plate -> free velocity (18) -> "
        "source mobility $|Y_{S,eq}|$ (19)",
        13,
        th.fg,
    )


# ---------------------------------------------------------------------------
# ISO 16251-1 small floor mock-up for a floor covering
# ---------------------------------------------------------------------------


def _d_iso16251_mockup(s: SVG, th: Theme) -> None:
    """The Annex A rig: the 1 200 x 800 x 200 mm slab on four elastic pads,
    the covering specimen, the tapping machine and the accelerometers.
    """
    top = 74.0

    # ===== Left: a section through the rig =====
    sx0, sw_ = 48.0, 372.0
    s.text(sx0 + sw_ / 2, top + 22, "Section", 16, th.fg, bold=True)
    slab_top, slab_h = top + 172.0, 62.0  # 200 mm at 310 px per metre
    s.rect(sx0, slab_top, sw_, slab_h, th.panel, th.fg, sw=2.4)
    s.text(
        sx0 + sw_ - 10,
        slab_top + slab_h - 14,
        "concrete slab, 200 ± 10 mm",
        13,
        th.fg,
        anchor="end",
    )
    # Covering specimen on top, big enough to carry the whole machine.
    s.rect(sx0 + 10, slab_top - 13, sw_ - 20, 13, th.accent, th.fg, sw=1.6)
    s.text(
        sx0 + 12,
        slab_top - 22,
        "covering specimen",
        13,
        th.accent,
        anchor="start",
        italic=True,
    )
    # Four elastic supports (two visible in section) and the laboratory floor.
    for bx in (sx0 + 40, sx0 + sw_ - 40):
        s.rect(bx - 18, slab_top + slab_h, 36, 22, th.panel, th.accent, rx=3, sw=2.0)
    s.ground(slab_top + slab_h + 22, sx0 - 10, sx0 + sw_ + 10)

    # Tapping machine standing wholly on the specimen (five hammers).
    mx = sx0 + 186.0
    body_y = slab_top - 78.0
    s.rect(mx - 66, body_y, 132, 30, th.primary, th.fg, rx=5, sw=2)
    for hx in range(-44, 45, 22):
        s.line(mx + hx, body_y + 30, mx + hx, slab_top - 15, th.fg, 2.4)
        s.circle(mx + hx, slab_top - 15, 4.2, th.fg)
    s.text(mx, body_y - 14, "tapping machine (ISO 10140-5)", 14, th.fg, bold=True)
    s.text(mx, body_y - 34, "5 hammers, 0,5 kg from 40 mm, 10 s⁻¹", 12, th.muted)

    # Accelerometer glued under the slab, with its cable route.
    _accel_wall(s, sx0 + 232, slab_top + slab_h + 11)
    s.line(
        sx0 + 240, slab_top + slab_h + 11, sx0 + 316, slab_top + slab_h + 11, th.fg, 1.3
    )
    # One line each: the Spanish pair is 375 px of type against the 372 px
    # of section they would have to share.
    s.text(
        sx0 + sw_ - 10,
        slab_top + slab_h + 66,
        "accelerometer screwed or glued underneath",
        12,
        th.secondary,
        anchor="end",
    )
    s.text(
        sx0 + 40, slab_top + slab_h + 46, "elastic pads", 12, th.accent, anchor="middle"
    )

    for i, txt in enumerate(
        (
            "four elastic pads at the corners, each ≤ 100 × 100 mm",
            "vertical resonance of the slab on its pads < 20 Hz",
            "top flat to ± 1 mm in a line edge to edge",
        )
    ):
        s.text(
            sx0 + sw_ / 2,
            slab_top + slab_h + 96 + 24 * i,
            txt,
            13,
            th.secondary if i == 1 else th.fg,
            bold=(i == 1),
        )

    # ===== Right: the plan of the slab =====
    qx0, qy0, qw, qh = 480.0, top + 66.0, 342.0, 220.0  # 1 200 x 800 mm
    s.text(qx0 + qw / 2, top + 22, "Plan", 16, th.fg, bold=True)
    s.text(
        qx0 + qw / 2,
        top + 46,
        "machine positions above, accelerometers below",
        13,
        th.muted,
    )
    s.rect(qx0, qy0, qw, qh, th.panel, th.fg, sw=2.4)
    # The 100 mm edge keep-out band for the accelerometer positions.
    s.rect(qx0 + 28, qy0 + 28, qw - 56, qh - 56, "none", th.muted, sw=1.4, dash="6,4")

    # Two tapping-machine footprints, skew to the edges, ≥ 300 mm apart.
    for fx, fy, tilt in ((qx0 + 82, qy0 + 62, 11.0), (qx0 + 232, qy0 + 158, -9.0)):
        pts = []
        for dx, dy in ((-42, -25), (42, -25), (42, 25), (-42, 25)):
            pts.append((fx + dx - tilt * dy / 25.0, fy + dy + tilt * dx / 42.0))
        s.path(
            "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts) + " Z",
            fill="none",
            stroke=th.primary,
            sw=2.0,
            dash="7,4",
        )
        for dx in (-26, -13, 0, 13, 26):
            s.circle(fx + dx, fy + tilt * dx / 42.0, 3.4, th.primary)
    s.line(qx0 + 82, qy0 + 62, qx0 + 232, qy0 + 158, th.muted, 1.0, dash="4,4")
    s.text(qx0 + 160, qy0 + 104, "≥ 300 mm", 13, th.fg)

    # Four accelerometer positions on the underside: random, off the axes.
    for ax_, ay_ in (
        (qx0 + 50, qy0 + 190),
        (qx0 + 158, qy0 + 200),
        (qx0 + 268, qy0 + 54),
        (qx0 + 300, qy0 + 148),
    ):
        s.circle(ax_, ay_, 7.0, th.secondary, th.fg, 1.4)
    s.dim(
        qx0,
        qy0 + qh + 20,
        qx0 + qw,
        qy0 + qh + 20,
        "1 200 × 800 mm (± 50 mm)",
        offset=0,
        size=14,
    )

    for i, txt in enumerate(
        (
            "≥ 2 machine positions, skew to the edges,",
            "no hammer within 100 mm of an edge, all feet on the specimen",
            "≥ 4 accelerometer positions, uniform but random,",
            "off the symmetry lines and ≥ 100 mm from every edge",
        )
    ):
        s.text(
            qx0 + qw / 2,
            qy0 + qh + 48 + 22 * i,
            txt,
            13,
            th.primary if i < 2 else th.secondary,
        )

    # ===== Footer: the three measurement cycles =====
    s.rect(48, 496, 804, 84, "none", th.muted, rx=10, dash="6,5")
    s.text(
        450,
        522,
        "three cycles: with specimen  |  without specimen (hammers "
        "repeated within ± 20 mm)  |  background",
        13,
        th.fg,
    )
    s.text(
        450,
        546,
        "≥ 20 s per level; background rule: unchanged ≥ 15 dB, energy "
        "subtraction 6-15 dB, −1,3 dB below 6 dB",
        13,
        th.fg,
    )
    s.text(
        450,
        570,
        "$L_a = 10 lg(⟨a^2⟩/a_0^2)$,  $a_0$ = 10⁻⁶ m/s²   (Formula 1)",
        13,
        th.primary,
    )


def _d_en12354_6_takeoff(s: SVG, th: Theme) -> None:
    """The EN 12354-6 Annex E room turned into the three input lists."""
    # Annex E: 4.54 x 2.73 x 2.40 m, V = 29.75 m3, 1 kHz octave band.
    ox, oy = 70.0, 300.0  # near-bottom-left corner of the floor
    px, py = 152.0, 0.0  # +x (length, 4.54 m)
    qx, qy = 74.0, -46.0  # +y (depth, 2.73 m), receding
    hx, hy = 0.0, -132.0  # +z (height, 2.40 m)

    def pt(u: float, v: float, w: float) -> tuple[float, float]:
        return (ox + u * px + v * qx + w * hx, oy + u * py + v * qy + w * hy)

    floor = [pt(0, 0, 0), pt(1, 0, 0), pt(1, 1, 0), pt(0, 1, 0)]
    back = [pt(0, 1, 0), pt(1, 1, 0), pt(1, 1, 1), pt(0, 1, 1)]
    side = [pt(0, 0, 0), pt(0, 1, 0), pt(0, 1, 1), pt(0, 0, 1)]
    ceiling = [pt(0, 0, 1), pt(1, 0, 1), pt(1, 1, 1), pt(0, 1, 1)]

    def poly(
        points: list[tuple[float, float]],
        fill: str,
        stroke: str,
        sw: float = 1.6,
        dash: str = "",
    ) -> None:
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points) + " Z"
        s.path(d, fill=fill, stroke=stroke, sw=sw, dash=dash)

    poly(floor, th.panel, th.fg, 1.8)
    poly(back, th.panel, th.fg, 1.8)
    poly(side, th.panel, th.fg, 1.8)
    poly(ceiling, "none", th.muted, 1.2, dash="6,5")
    s.line(*pt(1, 0, 0), *pt(1, 0, 1), th.muted, 1.2)
    s.line(*pt(1, 0, 1), *pt(1, 1, 1), th.muted, 1.2)

    # The glass facade is the far long wall; hatch it so it is identifiable.
    for k in range(1, 9):
        a = pt(k / 9.0, 1, 0)
        b = pt(k / 9.0, 1, 1)
        s.line(a[0], a[1], b[0], b[1], th.primary, 0.9, dash="4,4")

    s.dim(*pt(0, 0, 0), *pt(1, 0, 0), "4.54 m", offset=40, size=13)
    s.dim(*pt(0, 0, 0), *pt(0, 0, 1), "2.40 m", offset=-40, size=12, label_side="right")
    s.dim(*pt(1, 0, 0), *pt(1, 1, 0), "2.73 m", offset=34, size=13)
    s.text(183, 94, "$V$ = 29.75 m³", 15, th.fg, "middle", bold=True)
    s.text(183, 112, "1000 Hz octave band", 10, th.muted, "middle")

    # Tags outside the body, each on a leader to the surface it names.
    tags = (
        ((330.0, 76.0), "start", pt(0.55, 0.55, 1.0), "ceiling  12.39 m²  $α_s$ 0.02"),
        (
            (330.0, 100.0),
            "start",
            pt(0.60, 1.0, 0.72),
            "glass facade  10.90 m²  $α_s$ 0.04",
        ),
        ((330.0, 124.0), "start", pt(0.42, 0.16, 0.0), "floor  12.39 m²  $α_s$ 0.05"),
        (
            (330.0, 148.0),
            "start",
            pt(0.0, 0.55, 0.55),
            "short wall  6.55 m²  $α_s$ 0.04  (x2)",
        ),
        (
            (330.0, 172.0),
            "start",
            pt(0.86, 0.62, 0.30),
            "long wall (brick)  10.90 m²  $α_s$ 0.04",
        ),
    )
    for (tx, ty), anchor, target, label in tags:
        s.text(tx, ty, label, 11, th.primary, anchor)
        s.line(tx - 6, ty - 4, target[0], target[1], th.muted, 0.9, dash="3,3")

    # The furniture of Annex E case 2.
    boxes = (
        (0.14, 0.62, 0.16, 0.10, "0.65"),
        (0.34, 0.60, 0.14, 0.09, "0.65"),
        (0.55, 0.30, 0.15, 0.07, "0.60"),
        (0.74, 0.34, 0.10, 0.05, "0.15"),
        (0.30, 0.22, 0.07, 0.04, "0.05"),
        (0.46, 0.14, 0.07, 0.04, "0.05"),
    )
    for u, v, du, dh, vol in boxes:
        base = pt(u, v, 0.0)
        top = pt(u, v, dh * 3.0)
        far = pt(u + du, v, dh * 3.0)
        near = pt(u + du, v, 0.0)
        poly([base, near, far, top], th.accent, th.fg, 1.2)
        _ = vol
    s.text(
        ox + 170,
        oy + 72,
        "objects: 0.15, 0.60, 2 × 0.05, 2 × 0.65 m³",
        11,
        th.accent,
        "middle",
    )

    # The lists the room becomes.
    lx, lw = 545.0, 320.0
    s.rect(lx, 60, lw, 214, th.panel, th.primary, rx=10, sw=2)
    s.text(lx + 14, 86, "surfaces = [", 13, th.fg, "start", mono=True)
    rows = (
        "    (12.39, 0.05),   # floor",
        "    (12.39, 0.02),   # ceiling",
        "    (10.90, 0.04),   # long wall",
        "    (10.90, 0.04),   # facade",
        "    (6.55, 0.04),    # short wall",
        "    (6.55, 0.04),    # short wall",
        "]",
    )
    for k, row in enumerate(rows):
        s.text(lx + 14, 110 + 21 * k, row, 12, th.fg, "start", mono=True)
    s.text(
        lx + lw / 2,
        264,
        "$A$ = 2.26 m²   (Formula 1)",
        13,
        th.primary,
        "middle",
        bold=True,
    )

    s.rect(lx, 292, lw, 96, th.panel, th.accent, rx=10, sw=2)
    s.text(
        lx + 14,
        316,
        "objects = hard_object_absorption(volumes)",
        11,
        th.fg,
        "start",
        mono=True,
    )
    s.text(
        lx + 14,
        337,
        "psi = object_fraction(volumes, 29.75)",
        11,
        th.fg,
        "start",
        mono=True,
    )
    s.text(
        lx + lw / 2,
        362,
        "$A_{obj}$ = 2.77 m²    $ψ$ = 0.072",
        13,
        th.accent,
        "middle",
        bold=True,
    )
    s.text(lx + lw / 2, 380, "(Formula 4, then Formula 3)", 10, th.muted, "middle")

    # The inset: one wall split by a window.
    ix, iy, iw, ih = 70.0, 400.0, 400.0, 132.0
    s.rect(ix, iy, iw, ih, "none", th.muted, rx=10, sw=1.4, dash="6,5")
    s.text(ix + 12, iy + 24, "One wall, two rows", 13, th.fg, "start", bold=True)
    wx, wy, ww, wh = ix + 20, iy + 40, 190.0, 74.0
    s.rect(wx, wy, ww, wh, th.panel, th.fg, sw=1.6)
    s.rect(wx + 28, wy + 16, 78, 42, "none", th.primary, sw=1.8)
    s.text(wx + 67, wy + 42, "window", 10, th.primary, "middle")
    s.text(wx + ww / 2, wy + wh + 15, "one wall on the drawing", 10, th.muted, "middle")
    s.arrow(wx + ww + 8, wy + wh / 2, wx + ww + 44, wy + wh / 2, th.fg, 1.8)
    s.text(
        wx + ww + 52, wy + 26, "($S_{wall} − S_{win}$, $α_{wall}$)", 11, th.fg, "start"
    )
    s.text(wx + ww + 52, wy + 48, "($S_{win}$, $α_{win}$)", 11, th.fg, "start")
    s.text(wx + ww + 52, wy + 72, "areas sum to the wall", 10, th.muted, "start")

    s.text(
        450,
        552,
        "Never average a lining into its wall by hand: the areas "
        "are weighted inside the formula.",
        12,
        th.muted,
        "middle",
    )


def _d_directivity_factor(s: SVG, th: Theme) -> None:
    """Q as four mountings, with the critical distance each one produces."""
    import math

    cells = (
        (1, "$4π$", "free space", "on a stand", 1.11),
        (2, "$2π$", "hard floor", "on the slab", 1.57),
        (4, "$π$", "floor-wall edge", "against a wall on the slab", 2.22),
        (8, "$π/2$", "trihedral corner", "in the corner of the workshop", 3.14),
    )
    cw = 210.0
    for k, (q, solid, place, mounting, rc) in enumerate(cells):
        x0 = 15.0 + cw * k
        cx = x0 + cw / 2
        s.rect(x0 + 8, 72, cw - 16, 246, th.panel, th.muted, rx=10, sw=1.4)

        gy = 268.0  # the slab surface inside the cell
        wx = cx - 62.0  # the wall face
        colour = (th.primary, th.accent, th.fg, th.secondary)[k]

        if q == 1:
            sx, sy = cx, 208.0
            start_a, end_a = -180.0, 180.0
        elif q == 2:
            sx, sy = cx, gy
            start_a, end_a = -180.0, 0.0
            s.rect(cx - 74, gy, 148, 11, th.muted, th.fg, sw=1.2)
        elif q == 4:
            sx, sy = wx, gy
            start_a, end_a = -90.0, 0.0
            s.rect(wx - 11, gy, 159, 11, th.muted, th.fg, sw=1.2)
            s.rect(wx - 11, gy - 92, 11, 92, th.muted, th.fg, sw=1.2)
        else:
            sx, sy = wx, gy
            start_a, end_a = -90.0, 0.0
            s.rect(wx - 11, gy, 159, 11, th.muted, th.fg, sw=1.2)
            s.rect(wx - 11, gy - 92, 11, 92, th.muted, th.fg, sw=1.2)
            s.rect(wx - 11, gy - 103, 148, 11, th.muted, th.fg, sw=1.2)

        steps = 16
        for i in range(steps + 1):
            ang = math.radians(start_a + (end_a - start_a) * i / steps)
            s.line(
                sx, sy, sx + 62 * math.cos(ang), sy + 62 * math.sin(ang), colour, 1.1
            )
        s.circle(sx, sy, 8, colour, th.fg, 1.4)
        s.text(sx, sy + 4, "$S$", 10, th.bg, "middle", bold=True)

        s.text(cx, 102, f"$Q$ = {q}", 19, colour, "middle", bold=True)
        s.text(cx, 126, f"radiates into {solid} sr", 12, th.fg, "middle")
        s.text(cx, 148, place, 11, th.muted, "middle")
        s.text(cx, 300, mounting, 10, th.fg, "middle")
        s.text(cx, 336, f"$r_c$ = {rc:.2f} m", 14, colour, "middle", bold=True)

    s.text(
        450,
        56,
        "The same compact source, four mountings (workshop with $R$ = 62 m²)",
        12,
        th.muted,
        "middle",
    )
    s.text(
        450,
        366,
        "$Q$ multiplies the direct term only: the reverberant plateau does not move.",
        13,
        th.fg,
        "middle",
        bold=True,
    )
    s.text(
        450,
        390,
        "$r_c = √(Q·R/16π)$, so two steps of mounting move the "
        "crossover by a factor of 2.",
        11,
        th.muted,
        "middle",
    )


def _d_decay_range(s: SVG, th: Theme) -> None:
    """The level budget of one band: INR, the truncation point and the three
    ISO 3382 evaluation windows with their 15 dB margins.
    """
    x0, x1 = 118.0, 588.0  # time axis
    y0, y1 = 84.0, 384.0  # 0 dB to -70 dB
    per_db = (y1 - y0) / 70.0

    def y_of(level: float) -> float:
        return y0 - level * per_db

    # ===== Axes =====
    s.line(x0, y0 - 12, x0, y1 + 8, th.fg, 2.0)
    s.line(x0 - 8, y1, x1 + 46, y1, th.fg, 2.0)
    s.text(x1 + 46, y1 + 24, "time", 14, th.fg, anchor="end")
    for level in range(0, -71, -10):
        yy = y_of(float(level))
        s.line(x0 - 6, yy, x0, yy, th.muted, 1.2)
        s.text(x0 - 12, yy + 5, f"{level}", 12, th.muted, anchor="end")
    s.text(x0 - 12, y0 - 24, "Level [dB]", 14, th.fg, anchor="end")

    # ===== The band-filtered squared impulse response =====
    noise = -55.0
    x_peak, x_cross = x0 + 26.0, x0 + 322.0
    s.circle(x_peak, y_of(0.0), 5.0, th.primary)
    s.text(x_peak + 10, y_of(0.0) - 8, "peak", 13, th.primary, anchor="start")
    s.line(x_peak, y_of(0.0), x_cross, y_of(noise), th.primary, 2.6)
    s.line(
        x_cross, y_of(noise), x_cross + 92.0, y_of(-70.0), th.primary, 1.6, dash="6,4"
    )
    s.line(x0, y_of(noise), x1, y_of(noise), th.secondary, 2.2, dash="9,5")
    s.text(
        x0 + 8, y_of(noise) - 10, "background noise", 13, th.secondary, anchor="start"
    )

    # The compensated tail: everything past the crossing.
    s.rect(
        x_cross,
        y_of(noise),
        x1 - x_cross,
        y1 - y_of(noise),
        th.panel,
        th.muted,
        sw=1.2,
        dash="4,4",
    )
    s.circle(x_cross, y_of(noise), 5.5, th.bg, th.fg, 2.0)
    s.text(
        x_cross - 12,
        y_of(-64.0),
        "integration truncated here ($t_1$)",
        13,
        th.fg,
        anchor="end",
    )
    s.text(
        x_cross + (x1 - x_cross) / 2,
        y_of(noise) + 26,
        "tail compensated as",
        12,
        th.muted,
    )
    s.text(
        x_cross + (x1 - x_cross) / 2,
        y_of(noise) + 46,
        "an exponential decay (C)",
        12,
        th.muted,
    )

    # The INR bracket between the peak and the noise floor.
    s.dim(
        x_peak - 12,
        y_of(0.0),
        x_peak - 12,
        y_of(noise),
        "INR = 55 dB",
        offset=0,
        size=14,
        label_side="right",
    )

    # ===== The three evaluation windows, to the same dB scale =====
    wx = 604.0
    s.text(wx + 100, y0 - 24, "Evaluation windows", 15, th.fg, bold=True)
    # The library's tightened flags first, so the window boxes overlay them.
    for flag in (-46.0, -54.0):
        fy = y_of(flag)
        s.line(wx, fy, wx + 196, fy, th.accent, 1.8, dash="7,4")
        s.text(wx + 202, fy + 5, f"{-flag:.0f} dB", 12, th.accent, anchor="start")
    windows = (
        ("EDT", 0.0, -10.0),
        ("$T_{20}$", -5.0, -25.0),
        ("$T_{30}$", -5.0, -35.0),
    )
    for i, (name, hi, lo) in enumerate(windows):
        bx = wx + 16 + 64 * i
        s.rect(bx, y_of(hi), 38, (hi - lo) * per_db, th.panel, th.primary, sw=2.0)
        s.text(bx + 19, y_of(hi) - 8, name, 14, th.primary, bold=True)
        # The 15 dB margin ISO 3382-1 asks for beyond the window.
        top = y_of(lo)
        s.rect(bx, top, 38, 15.0 * per_db, th.bg, th.muted, sw=1.2, dash="5,4")
        for k in range(4):
            yy = top + 5 + k * (15.0 * per_db - 10) / 3.0
            s.line(bx + 3, yy, bx + 35, yy + 5, th.muted, 0.9)

    # ===== Footer: what the windows need, and the remedy order =====
    s.rect(48, 416, 804, 88, "none", th.muted, rx=10, dash="6,5")
    s.text(
        450,
        442,
        "hatched: the 15 dB margin ISO 3382-1 asks for beyond each window "
        "— EDT needs 25 dB, $T_{20}$ 35 dB, $T_{30}$ 45 dB",
        12,
        th.fg,
    )
    s.text(
        450,
        466,
        "the library flags at 46 dB and 54 dB instead, where the fit's "
        "positive bias crosses 5 %",
        12,
        th.accent,
    )
    s.text(
        450,
        490,
        "short of range? $T_{20}$ instead of $T_{30}$ -> a longer sweep "
        "or more averages -> EDT; never a fit into the noise",
        12,
        th.secondary,
    )


# ---------------------------------------------------------------------------
# EN/ISO 12354-1 Annex E junction catalogue
# ---------------------------------------------------------------------------


def _d_junction_catalogue(s: SVG, th: Theme) -> None:
    """The Annex E junction types, the three path branches and the mass ratio,
    with the argument pair each drawing maps onto.
    """
    dark = bool(th.suffix)
    c_through = th.primary  # K13, the 'through' branch
    c_corner = "#f0a94e" if dark else "#d9820e"  # K12 = K23, the 'corner' branch
    c_leaf = th.secondary  # K24, the double-leaf branch
    t_el = 13.0  # drawn element thickness
    arm = 58.0

    def element(x: float, y: float, w: float, h: float, tag: str) -> None:
        s.rect(x, y, w, h, th.panel, th.fg, sw=1.8)
        s.text(x + w / 2, y + h / 2 + 6, tag, 13, th.fg, bold=True)

    def title(cx: float, cy: float, name: str) -> None:
        s.text(cx, cy - arm - 36, name, 15, th.fg, bold=True)

    def branch(
        x1_: float,
        y1_: float,
        x2_: float,
        y2_: float,
        colour: str,
        label: str,
        lx: float,
        ly: float,
        anchor: str = "middle",
    ) -> None:
        s.arrow(x1_, y1_, x2_, y2_, colour, 2.6)
        s.text(lx, ly, label, 14, colour, bold=True, anchor=anchor)

    col = (152.0, 450.0, 748.0)
    row = (182.0, 378.0)

    # (1) Rigid cross, carrying the full annotation set.
    cx, cy = col[0], row[0]
    title(cx, cy, "rigid cross")
    element(cx - arm, cy - t_el / 2, arm - t_el / 2, t_el, "1")
    element(cx + t_el / 2, cy - t_el / 2, arm - t_el / 2, t_el, "3")
    element(cx - t_el / 2, cy - arm, t_el, arm - t_el / 2, "2")
    element(cx - t_el / 2, cy + t_el / 2, t_el, arm - t_el / 2, "4")
    branch(cx - 48, cy + 34, cx + 48, cy + 34, c_through, "$K_{13}$", cx + 30, cy + 54)
    branch(
        cx - 40,
        cy - 16,
        cx - 12,
        cy - 40,
        c_corner,
        "$K_{12}$",
        cx - 46,
        cy - 38,
        "end",
    )
    s.circle(cx, cy, 5.0, th.accent)
    s.text(cx + arm + 8, cy + 5, "$ℓ_f$", 14, th.accent, anchor="start")

    # (2) Rigid T.
    cx, cy = col[1], row[0]
    title(cx, cy, "rigid T")
    element(cx - arm, cy - t_el / 2, arm - t_el / 2, t_el, "1")
    element(cx + t_el / 2, cy - t_el / 2, arm - t_el / 2, t_el, "3")
    element(cx - t_el / 2, cy + t_el / 2, t_el, arm - t_el / 2, "2")
    branch(cx - 48, cy - 30, cx + 48, cy - 30, c_through, "$K_{13}$", cx, cy - 40)
    branch(
        cx - 40,
        cy + 16,
        cx - 12,
        cy + 42,
        c_corner,
        "$K_{12}$",
        cx - 46,
        cy + 42,
        "end",
    )

    # (3) T with a flexible interlayer.
    cx, cy = col[2], row[0]
    title(cx, cy, "T with a flexible interlayer")
    element(cx - arm, cy - t_el / 2, arm - t_el / 2, t_el, "1")
    element(cx + t_el / 2, cy - t_el / 2, arm - t_el / 2, t_el, "3")
    element(cx - t_el / 2, cy + t_el / 2 + 8, t_el, arm - t_el / 2 - 8, "2")
    s.rect(cx - t_el / 2 - 4, cy + t_el / 2, t_el + 8, 8, th.accent, th.fg, sw=1.2)
    branch(cx - 48, cy - 30, cx + 48, cy - 30, c_through, "$K_{13}$", cx, cy - 40)
    s.text(cx + 22, cy + 30, "elastic layer", 12, th.accent, anchor="start")

    # (4) Corner.
    cx, cy = col[0], row[1]
    title(cx, cy, "corner")
    element(cx - arm, cy - t_el / 2, arm + t_el / 2, t_el, "1")
    element(cx - t_el / 2, cy + t_el / 2, t_el, arm - t_el / 2, "2")
    branch(
        cx - 46,
        cy - 24,
        cx - 16,
        cy + 40,
        c_corner,
        "$K_{12}$",
        cx - 52,
        cy - 24,
        "end",
    )

    # (5) Thickness change: one line, two thicknesses.
    cx, cy = col[1], row[1]
    title(cx, cy, "thickness change")
    element(cx - arm, cy - t_el / 2, arm, t_el, "1")
    element(cx, cy - t_el, arm, 2 * t_el, "2")
    branch(cx - 48, cy - 34, cx + 48, cy - 34, c_through, "$K_{12}$", cx, cy - 44)

    # (6) Lightweight double leaf meeting a homogeneous floor: the K24 branch.
    cx, cy = col[2], row[1]
    title(cx, cy, "lightweight double leaf")
    element(cx - arm, cy - t_el / 2, arm - 18, t_el, "1")
    element(cx + 18, cy - t_el / 2, arm - 18, t_el, "3")
    element(cx - 18, cy - arm, 9, arm - t_el / 2, "2")
    element(cx + 9, cy - arm, 9, arm - t_el / 2, "4")
    for hy in range(int(cy - arm) + 8, int(cy) - 12, 12):
        s.line(cx - 9, hy, cx + 9, hy + 5, th.muted, 0.8)
    branch(cx - 48, cy + 30, cx + 48, cy + 30, c_through, "$K_{13}$", cx, cy + 48)
    branch(
        cx - 14,
        cy - arm + 14,
        cx + 14,
        cy - arm + 14,
        c_leaf,
        "$K_{24}$",
        cx + 24,
        cy - arm + 18,
        "start",
    )

    # ===== Legend: the argument pair each drawing maps onto =====
    s.rect(48, 442, 804, 116, "none", th.muted, rx=10, dash="6,5")
    pairs = (
        (70.0, 468.0, "rigid cross", "'rigid_cross', 'through' / 'corner'"),
        (70.0, 492.0, "rigid T", "'rigid_t', 'through' / 'corner'"),
        (70.0, 516.0, "flexible T", "'flexible_t', 'through' / 'corner'"),
        (532.0, 468.0, "corner", "'corner', 'corner'"),
        (532.0, 492.0, "thickness change", "'thickness_change', 'through'"),
    )
    for xx, yy, name, args in pairs:
        s.text(xx, yy, f"{name}:", 12, th.fg, anchor="start", bold=True)
        s.text(xx + 140, yy, args, 11, th.muted, anchor="start", mono=True)
    s.text(70.0, 540, "lightweight double leaf:", 12, th.fg, anchor="start", bold=True)
    s.text(
        70.0 + 190,
        540,
        "'lightweight_double_homogeneous', 'double_leaf'",
        11,
        th.muted,
        anchor="start",
        mono=True,
    )

    # ===== The mass ratio, worked on the Annex H.3 floor =====
    s.rect(48, 568, 804, 96, "none", th.muted, rx=10, dash="6,5")
    s.text(
        70,
        594,
        "$M = lg(m′_{perp,i} / m′_i)$: $m′_i$ is the element carrying the "
        "path, so the ratio is per path, not per junction.",
        13,
        th.fg,
        anchor="start",
    )
    s.text(
        70,
        620,
        "The functions take the RATIO, not $M$. Floor H.3, ratio 1,61: "
        "'through' -> $K_{13}$ = 12,5 dB, 'corner' -> $K_{12}$ = 8,9 dB",
        13,
        th.fg,
        anchor="start",
    )
    s.text(
        70,
        646,
        "$ℓ_f$ is the coupling length along the junction line, surface to "
        "surface. Annex E values are read at 500 Hz, ± 3 dB.",
        13,
        th.accent,
        anchor="start",
    )


# ---------------------------------------------------------------------------
# Facade sound insulation setup (ISO 16283-3)
# ---------------------------------------------------------------------------


def _d_facade_setup(s: SVG, th: Theme) -> None:
    """Section through a dwelling facade: the loudspeaker and both methods."""
    gy = 470.0  # ground line
    fx = 640.0  # outer face of the facade
    ftop = 150.0
    floor_y = 400.0  # receiving-room floor
    cx, cy = fx, 300.0  # centre of the test specimen

    s.ground(gy, 50, 880)

    # -- the building ----------------------------------------------------
    s.rect(fx, ftop, 230, gy - ftop, th.panel, th.fg, sw=2.5)
    s.rect(fx, ftop, 20, gy - ftop, th.secondary, th.fg, sw=2)  # facade leaf
    s.line(fx + 20, floor_y, 868, floor_y, th.fg, 2.2)  # room floor
    s.text(768, ftop + 42, "Receiving room", 17, th.fg, bold=True)
    s.text(768, ftop + 68, "$L_2$ , $T$ , $V$", 16, th.muted)
    s.text(650, ftop - 12, "$S$ = 11.5 m²", 15, th.secondary, bold=True, anchor="start")

    # -- loudspeaker on the ground, 45 degrees to the specimen centre ----
    lx, ly = 130.0, gy - 34.0
    s.rect(lx - 28, ly - 32, 56, 64, th.panel, th.primary, rx=6, sw=2)
    s.circle(lx, ly - 12, 12, th.primary)
    s.circle(lx, ly - 12, 4, th.bg)
    s.circle(lx, ly + 16, 7, th.primary)
    s.text(lx, gy + 34, "Loudspeaker", 16, th.fg, bold=True)
    s.text(lx, gy + 56, "(on the ground)", 15, th.muted)

    s.arrow(lx + 26, ly - 22, cx - 6, cy + 4, th.accent, 2.6)
    s.line(lx + 26, ly - 22, 560, ly - 22, th.muted, 1.0, dash="4,4")
    s.text(300, ly - 34, "45° ± 5°", 18, th.accent, bold=True)
    s.text(430, 250, "$r ≥ 5$ m element / ≥ 7 m global", 16, th.accent, anchor="middle")

    # D, the perpendicular distance from the facade plane.
    s.line(lx, gy, lx, gy + 110, th.muted, 0.9, dash="3,3")
    s.line(cx, gy, cx, gy + 110, th.muted, 0.9, dash="3,3")
    s.dim(
        lx,
        gy + 106,
        cx,
        gy + 106,
        "$D > 3.5$ m (element) / > 5 m (global)",
        offset=0,
        size=16,
    )

    # -- element method: microphone flush on the test specimen -----------
    s.circle(fx - 8, cy, 8, th.fg)
    s.path(
        f"M {fx - 26:.0f} {cy:.0f} A 18 18 0 0 1 {fx + 10:.0f} {cy:.0f}",
        stroke=th.primary,
        sw=2.2,
    )
    s.line(fx - 26, cy, 560, 150, th.muted, 1.0)
    # Sized on "de 3 a 10 posiciones, nunca en rejilla", 279 px against the
    # 240 of its English twin.
    s.rect(272, 92, 308, 76, th.panel, th.primary, rx=8, sw=1.6)
    s.text(
        282, 116, "$L_{1,s}$  element method", 16, th.primary, bold=True, anchor="start"
    )
    s.text(282, 138, "≤ 10 mm parallel / ≤ 3 mm normal", 15, th.muted, anchor="start")
    s.text(282, 158, "3 to 10 positions, never gridded", 15, th.muted, anchor="start")

    # -- global method: microphone 2 m out, 1.5 m above the room floor ---
    gmx = 470.0
    gmy = 322.0
    s.mic(gmx, gmy, gy, scale=1.1)
    s.text(
        gmx - 6,
        gmy - 16,
        "L₁,2m  global method",
        16,
        th.primary,
        bold=True,
        anchor="end",
    )
    s.line(gmx, gmy + 4, gmx, 292, th.muted, 0.9, dash="3,3")
    s.line(fx, cy, fx, 292, th.muted, 0.9, dash="3,3")
    s.dim(gmx, 288, fx, 288, "(2.0 ± 0.2) m", offset=0, size=16)
    s.line(gmx + 12, gmy + 4, 566, gmy + 4, th.muted, 0.9, dash="3,3")
    s.line(fx, floor_y, 566, floor_y, th.muted, 0.9, dash="3,3")
    s.dim(562, gmy + 4, 562, floor_y, "1.5 m", offset=0, size=16, label_side="left")
    s.text(556, 424, "above the", 14, th.muted, anchor="end")
    s.text(556, 444, "receiving-room floor", 14, th.muted, anchor="end")

    for y, txt in (
        (
            600,
            (
                "Element method → $R′_{45°}$ or $R′_{tr,s}$: one component, "
                "comparable with a laboratory $R$."
            ),
        ),
        # The 2 m of the quantity's subscript is a value with its unit,
        # which the composer has no roman run for yet; that one symbol
        # stays plain until the case is adjudicated (as does the L1,2m
        # label above), while the R beside it is set like its twin.
        (
            626,
            (
                "Global method → D2m,nT: the whole facade as built, "
                "not comparable with a laboratory $R$."
            ),
        ),
        (
            652,
            (
                "Road traffic replaces the loudspeaker at all angles: inside "
                "and outside at once, ≥ 50 pass-bys."
            ),
        ),
        (
            678,
            (
                "Clauses 9.4, 9.5.1, 9.6.1 and 10.2. None of it is checked "
                "by the functions."
            ),
        ),
    ):
        s.text(50, y, txt, 15, th.fg, anchor="start")


# ---------------------------------------------------------------------------
# Heavy and soft impact sources (ISO 16283-2 Annex A, JIS A 1418-2)
# ---------------------------------------------------------------------------


def _d_heavy_impact_sources(s: SVG, th: Theme) -> None:
    """Three sources over one slab, with the drop heights dimensioned."""
    slab_top, slab_bot = 430.0, 470.0
    s.rect(50, slab_top, 800, slab_bot - slab_top, th.panel, th.fg, sw=2.4)
    s.text(450, slab_bot + 26, "Floor under test (source room)", 16, th.muted)
    s.rect(50, slab_bot, 800, 4, th.muted)

    panels = (86.0, 350.0, 630.0)

    # (a) ISO tapping machine: five hammers, 40 mm drop.
    ax = panels[0] + 110.0
    s.text(ax, 96, "(a) tapping machine", 17, th.primary, bold=True)
    s.text(ax, 120, "ISO 10140-5 Annex E", 15, th.muted)
    body_y = 356.0
    s.rect(ax - 92, body_y, 184, 44, th.panel, th.primary, rx=5, sw=2)
    for i in range(5):
        hx = ax - 72 + i * 36
        s.rect(hx - 7, body_y + 44, 14, 18, th.primary, th.fg, sw=1.2)
        s.line(hx, body_y + 62, hx, slab_top - 12, th.muted, 1.0, dash="3,3")
    s.line(ax - 100, slab_top, ax - 100, body_y + 44, th.muted, 0.9, dash="3,3")
    s.dim(ax - 112, body_y + 62, ax - 112, slab_top, "40 mm", offset=0, size=15)
    s.text(ax, 300, "5 hammers, 500 g each", 15, th.fg)
    s.text(ax, 322, "(100 ± 20) ms apart", 15, th.fg)

    # (b) rubber ball: 180 mm, 30 mm wall, drop measured from the BOTTOM.
    bx = panels[1] + 105.0
    s.text(bx, 96, "(b) rubber ball", 17, th.accent, bold=True)
    s.text(bx, 120, "ISO 16283-2 Annex A / ISO 10140-5 Annex F", 13, th.muted)
    ball_r = 46.0
    ball_cy = 210.0
    s.circle(bx, ball_cy, ball_r, "none", th.accent, sw=3)
    s.ellipse(
        bx, ball_cy, ball_r - 15, ball_r - 15, "none", th.accent, sw=1.6, dash="5,4"
    )
    s.dim(
        bx - ball_r,
        ball_cy - 62,
        bx + ball_r,
        ball_cy - 62,
        "180 mm",
        offset=0,
        size=15,
    )
    s.line(bx - ball_r, ball_cy, bx - ball_r, ball_cy - 62, th.muted, 0.9, dash="3,3")
    s.line(bx + ball_r, ball_cy, bx + ball_r, ball_cy - 62, th.muted, 0.9, dash="3,3")
    s.text(bx + ball_r + 16, ball_cy - 6, "30 mm wall", 15, th.muted, anchor="start")
    s.text(
        bx + ball_r + 16,
        ball_cy + 16,
        "$m_{eff}$ = (2.5 ± 0.1) kg",
        15,
        th.muted,
        anchor="start",
    )
    s.text(
        bx + ball_r + 16, ball_cy + 38, "$e$ = 0.8 ± 0.1", 15, th.muted, anchor="start"
    )
    s.arrow(bx, ball_cy + ball_r + 8, bx, slab_top - 8, th.accent, 2.2)
    s.line(bx - 100, ball_cy + ball_r, bx, ball_cy + ball_r, th.secondary, 2.0)
    s.dim(
        bx - 88, ball_cy + ball_r, bx - 88, slab_top, "(100 ± 1) cm", offset=0, size=15
    )
    s.text(
        bx - 100,
        ball_cy + ball_r + 22,
        "from the ball's BOTTOM",
        14,
        th.secondary,
        bold=True,
        anchor="middle",
    )

    # (c) bang machine: a car tyre dropped 85 cm.
    cx = panels[2] + 110.0
    s.text(cx, 96, "(c) bang machine", 17, th.secondary, bold=True)
    s.text(cx, 120, "JIS A 1418-2 only", 15, th.muted)
    tyre_cy = 240.0
    s.ellipse(cx, tyre_cy, 62, 34, "none", th.secondary, sw=3.4)
    s.ellipse(cx, tyre_cy, 30, 15, "none", th.secondary, sw=2.0)
    s.text(cx, tyre_cy - 52, "(2.4 ± 0.2)·10⁵ Pa", 15, th.muted)
    s.text(cx, tyre_cy + 62, "$m_{eff}$ = (7.3 ± 0.2) kg", 15, th.muted)
    s.arrow(cx, tyre_cy + 76, cx, slab_top - 8, th.secondary, 2.2)
    s.dim(
        cx + 92,
        tyre_cy + 34,
        cx + 92,
        slab_top,
        "85 cm",
        offset=0,
        size=15,
        label_side="right",
    )
    s.line(cx + 62, tyre_cy + 34, cx + 92, tyre_cy + 34, th.muted, 0.9, dash="3,3")

    # -- the calibration chain, where the filter position decides the answer
    s.rect(60, 508, 780, 60, th.panel, th.muted, rx=8, sw=1.4)
    stages = (
        ("source", th.fg),
        ("rigid floor +\nforce plate", th.fg),
        ("octave filter", th.primary),
        ("analyser → $L_{FE}$", th.fg),
    )
    # Each connector spans the gap its two stages leave, measured: at a
    # fixed length the arrowhead lands inside "filtro de octava", which is
    # 129 px against the 101 of "octave filter".
    half = [
        max(
            s.text_width(part, 15, bold=(colour == th.primary))
            for part in label.split("\n")
        )
        / 2
        for label, colour in stages
    ]
    for i, (label, colour) in enumerate(stages):
        px = 150.0 + i * 200.0
        if "\n" in label:
            a, b = label.split("\n")
            s.text(px, 532, a, 15, colour, bold=(colour == th.primary))
            s.text(px, 552, b, 15, colour)
        else:
            s.text(px, 543, label, 15, colour, bold=(colour == th.primary))
        if i < len(stages) - 1:
            s.arrow(
                px + half[i] + 14,
                538,
                px + 200.0 - half[i + 1] - 14,
                538,
                th.muted,
                1.8,
            )
    s.text(
        450,
        592,
        "JIS A 1418-2 Annex C: the filter goes BEFORE the analyser,",
        15,
        th.secondary,
        bold=True,
    )
    s.text(
        450, 613, "so $L_{FE}$ is evaluated once per band", 15, th.secondary, bold=True
    )
    s.text(
        450,
        638,
        "The dimensions above are the standards' informative construction examples;",
        15,
        th.muted,
        italic=True,
    )
    s.text(
        450,
        658,
        "the specification is the force spectrum, not the shape.",
        15,
        th.muted,
        italic=True,
    )


# ---------------------------------------------------------------------------
# ISO 10052 survey sweep
# ---------------------------------------------------------------------------


def _d_survey_sweep(s: SVG, th: Theme) -> None:
    """The survey method is a body posture and a sweep path, in plan."""
    x0, y0, x1, y1 = 60.0, 90.0, 500.0, 430.0
    s.rect(x0, y0, x1 - x0, y1 - y0, th.panel, th.fg, rx=6, sw=2.6)
    s.text((x0 + x1) / 2, y0 - 14, "Plan of the room", 16, th.muted)

    # Separating element on the right wall of the plan.
    s.rect(x1 - 12, y0, 12, y1 - y0, th.secondary, th.fg, sw=2)
    s.text(
        x1 - 22,
        y0 + 30,
        "separating element",
        15,
        th.secondary,
        anchor="end",
        bold=True,
    )

    # Loudspeaker in the far corner, facing into it, >= 0.5 m off the walls.
    lx, ly = x0 + 74.0, y0 + 74.0
    s.rect(lx - 26, ly - 26, 52, 52, th.panel, th.primary, rx=6, sw=2)
    s.circle(lx, ly, 13, th.primary)
    s.circle(lx, ly, 5, th.bg)
    s.arrow(lx - 14, ly - 14, x0 + 16, y0 + 16, th.primary, 2.0)
    s.dim(x0, ly + 34, lx - 26, ly + 34, "≥ 0.5 m", offset=0, size=15)
    s.dim(
        lx + 34, y0, lx + 34, ly - 26, "≥ 0.5 m", offset=0, size=15, label_side="right"
    )
    s.text(
        x0 + 14, ly + 96, "corner opposite the element,", 15, th.muted, anchor="start"
    )
    s.text(x0 + 14, ly + 116, "facing into the corner", 15, th.muted, anchor="start")

    # Operator near the centre, facing away from the loudspeaker.
    ox, oy = (x0 + x1) / 2 + 52.0, (y0 + y1) / 2 + 46.0
    s.circle(ox, oy, 15, th.muted)
    s.arrow(ox + 12, oy + 12, ox + 52, oy + 52, th.muted, 1.6)
    s.text(ox + 58, oy + 70, "facing away", 15, th.muted, anchor="start")

    # The 180-degree arm sweep, four times, at arm's length.
    r = 82.0
    s.path(
        f"M {ox - r:.0f} {oy:.0f} A {r} {r} 0 0 1 {ox + r:.0f} {oy:.0f}",
        stroke=th.accent,
        sw=3.2,
    )
    s.arrow(ox + r - 14, oy - 12, ox + r, oy, th.accent, 2.6)
    s.line(ox, oy, ox - r, oy, th.fg, 2.0)
    s.circle(ox - r, oy, 8, th.fg)
    s.circle(ox - r, oy, 3, th.bg)
    s.dim(ox, oy + 30, ox - r, oy + 30, "arm's length", offset=0, size=15)
    s.text(ox + 16, oy - r - 18, "180° × 4 traverses,", 16, th.accent, bold=True)
    s.text(ox + 16, oy + r + 32, "≈ 30 s in total", 16, th.accent, bold=True)

    # Elevation beside it: the vertical component of the traverse.
    ex0, ey_top, ey_bot = 560.0, 120.0, 430.0
    s.rect(ex0, ey_top, 300, ey_bot - ey_top, th.panel, th.fg, rx=6, sw=2.2)
    s.text(ex0 + 150, ey_top - 14, "Elevation: the same sweep", 16, th.muted)
    s.person(ex0 + 70, ey_bot - 20, h=150)
    s.path(
        f"M {ex0 + 96:.0f} {ey_bot - 112:.0f} "
        f"C {ex0 + 160:.0f} {ey_bot - 172:.0f} "
        f"{ex0 + 208:.0f} {ey_bot - 74:.0f} "
        f"{ex0 + 258:.0f} {ey_bot - 134:.0f}",
        stroke=th.accent,
        sw=3.0,
    )
    s.circle(ex0 + 258, ey_bot - 134, 8, th.fg)
    s.circle(ex0 + 258, ey_bot - 134, 3, th.bg)
    s.text(ex0 + 150, ey_bot - 232, "raise and lower the arm", 15, th.accent)
    s.text(ex0 + 150, ey_bot - 210, "during each traverse", 15, th.accent)

    for y, txt in (
        (
            470,
            (
                "Alternative (6.3.1): a rotating microphone on a stand, "
                "≥ 10° to the horizontal, radius ≥ 1 m."
            ),
        ),
        (
            496,
            (
                "Without a real-time octave analyser, repeat the sweep per "
                "band and read each 30 s $L_{eq}$."
            ),
        ),
        (
            522,
            (
                "Tapping machine (6.2.3): floor centre, on the diagonal; "
                "three positions at 45° to the ribs."
            ),
        ),
    ):
        s.text(60, y, txt, 15, th.fg, anchor="start")


# ---------------------------------------------------------------------------
# ISO 12354-1 Annex L / ISO 12354-2 Annex G worked building
# ---------------------------------------------------------------------------


def _d_iso12354_annexl(s: SVG, th: Theme) -> None:
    """The worked building both parts share: two stacked dwellings, the
    separating floor and its four junctions, and where the thirteen paths run.
    """
    top = 74.0

    # ===== Left: a section through the two dwellings =====
    ax0, aw = 40.0, 400.0
    s.text(
        ax0 + aw / 2, top + 22, "Section: two stacked dwellings", 16, th.fg, bold=True
    )
    wall_t, floor_t = 24.0, 20.0
    room_h = 92.0  # 2,75 m storey height
    y_top = top + 56.0
    y_floor = y_top + room_h
    y_bot = y_floor + floor_t + room_h
    lx, rx = ax0 + 46.0, ax0 + aw - 46.0
    ix = ax0 + aw * 0.56  # internal wall

    for wx in (lx, rx - wall_t, ix):
        s.rect(wx, y_top - 18, wall_t, y_bot - y_top + 36, th.panel, th.fg, sw=2.0)
    s.rect(lx + wall_t, y_floor, rx - lx - 2 * wall_t, floor_t, th.panel, th.fg, sw=2.2)
    s.rect(lx + wall_t, y_floor - 12, rx - lx - 2 * wall_t, 5, th.accent, th.fg, sw=1.0)
    s.rect(lx + wall_t, y_floor - 7, rx - lx - 2 * wall_t, 7, th.panel, th.fg, sw=1.0)

    s.text(ax0 + aw / 2, y_top + 34, "source dwelling", 15, th.fg)
    s.text(ax0 + aw / 2, y_bot - 24, "receiving dwelling", 15, th.fg)

    # The junction nodes, tagged with the letter the key explains.
    for nx, tag in (
        (lx + wall_t / 2, "T"),
        (rx - wall_t / 2, "T"),
        (ix + wall_t / 2, "X"),
    ):
        ny = y_floor + floor_t / 2
        s.circle(nx, ny, 11.0, th.bg, th.secondary, 2.4)
        s.text(nx, ny + 6, tag, 14, th.secondary, bold=True)

    # ===== Key under the section =====
    s.text(
        ax0 + 6,
        y_bot + 34,
        "T  rigid T (floor to external wall): $K_{ij}$ = 6,4 / 11,2 dB",
        13,
        th.secondary,
        anchor="start",
    )
    s.text(
        ax0 + 6,
        y_bot + 56,
        "X  rigid cross (floor to internal wall): $K_{ij}$ = 8,8 / 11,0 dB",
        13,
        th.secondary,
        anchor="start",
    )
    s.text(
        ax0 + 6,
        y_bot + 84,
        "separating floor  220 mm concrete, 484 kg/m², $f_c$ = 76,8 Hz",
        12,
        th.fg,
        anchor="start",
    )
    s.text(
        ax0 + 6,
        y_bot + 106,
        "on it  35 mm screed, 73,5 kg/m², on $s′$ = 8 MN/m³",
        12,
        th.accent,
        anchor="start",
    )
    s.text(
        ax0 + 6,
        y_bot + 128,
        "external walls  365 mm AAC, 219 kg/m², $f_c$ = 92,6 Hz",
        12,
        th.muted,
        anchor="start",
    )
    s.text(
        ax0 + 6,
        y_bot + 150,
        "internal walls  200 mm calcium silicate, 360 kg/m², $f_c$ = 128,4 Hz",
        12,
        th.muted,
        anchor="start",
    )

    # ===== Right: the plan of the separating floor =====
    bx0, by0, bw, bh = 546.0, top + 76.0, 268.0, 214.0  # 5,00 m x 4,00 m
    s.text(bx0 + bw / 2, top + 22, "Plan: the separating floor", 16, th.fg, bold=True)
    s.rect(bx0, by0, bw, bh, th.panel, th.fg, sw=2.4)
    # The four junction lines, each shared with one flanking element.
    for x, y, w, h in (
        (bx0, by0 - 7, bw, 7),
        (bx0, by0 + bh, bw, 7),
        (bx0 - 7, by0, 7, bh),
        (bx0 + bw, by0, 7, bh),
    ):
        s.rect(x, y, w, h, th.secondary, th.fg, sw=1.0)
    s.text(bx0 + bw / 2, by0 + bh / 2 - 4, "$S$ = 20 m²", 16, th.fg, bold=True)
    s.text(bx0 + bw / 2, by0 + bh / 2 + 22, "5,00 m × 4,00 m", 13, th.muted)
    s.dim(bx0, by0 + bh + 26, bx0 + bw, by0 + bh + 26, "5,00 m", offset=0, size=14)
    s.dim(bx0 - 26, by0, bx0 - 26, by0 + bh, "4,00 m", offset=0, size=14)
    s.text(bx0 + bw / 2, by0 - 18, "external wall (T)", 12, th.secondary)
    s.text(bx0 + bw / 2, by0 + bh + 62, "internal wall (X)", 12, th.secondary)
    for i, txt in enumerate(
        (
            "two external and two internal walls meet the floor, with",
            "5,00 m of junction along each long edge and 4,00 m along",
            "each short one: perimeter 9 m external + 9 m internal",
        )
    ):
        s.text(
            bx0 + bw / 2,
            by0 + bh + 92 + 22 * i,
            txt,
            13,
            th.accent if i == 2 else th.fg,
        )

    # ===== Footer: the path count =====
    s.rect(40, 508, 820, 84, "none", th.muted, rx=10, dash="6,5")
    s.text(
        450,
        536,
        "13 airborne paths = 1 direct (Dd) + 4 flanking elements × "
        "3 branches (Ff, Df, Fd)",
        14,
        th.fg,
    )
    s.text(
        450,
        564,
        "5 impact paths = 1 direct + 4 Df: only the floor is excited, so "
        "there is no Ff or Fd",
        14,
        th.primary,
    )


# ---------------------------------------------------------------------------
# The three resilient build-ups a prediction is chosen by
# ---------------------------------------------------------------------------


def _d_resilient_buildups(s: SVG, th: Theme) -> None:
    """Floating floor, discrete mounts and a wall lining in section, each with
    the formula its construction detail selects.
    """
    top = 74.0
    col = (48.0, 330.0, 612.0)
    cw = 240.0

    # ===== (a) Floating floor on a continuous resilient layer =====
    ax = col[0]
    s.text(ax + cw / 2, top + 22, "(a) floating floor", 15, th.fg, bold=True)
    slab_y = top + 158.0
    s.rect(ax, slab_y, cw, 46, th.panel, th.fg, sw=2.2)
    s.text(ax + cw / 2, slab_y + 30, "220 mm structural slab", 12, th.fg)
    s.rect(ax, slab_y - 14, cw, 14, th.accent, th.fg, sw=1.4)
    s.rect(ax + 10, slab_y - 40, cw - 20, 26, th.panel, th.fg, sw=2.0)
    s.text(ax + cw / 2, slab_y - 22, "35 mm screed, 73,5 kg/m²", 12, th.fg)
    # The edge strip: the resilient layer turned up at the wall.
    for wx in (ax, ax + cw - 12):
        s.rect(wx, slab_y - 92, 12, 78, th.panel, th.fg, sw=2.0)
        s.rect(
            wx + (12 if wx == ax else -6), slab_y - 44, 6, 30, th.accent, th.fg, sw=1.0
        )
    s.text(ax + cw / 2, slab_y - 60, "edge strip, both sides", 12, th.accent)
    s.arrow(ax + 24, slab_y - 100, ax + 16, slab_y - 34, th.secondary, 2.0)
    s.text(
        ax + 30, slab_y - 106, "any rigid bridge here", 12, th.secondary, anchor="start"
    )
    s.text(
        ax + 30,
        slab_y - 88,
        "short-circuits the spring",
        12,
        th.secondary,
        anchor="start",
    )
    s.text(
        ax + cw / 2,
        slab_y + 74,
        "$s′$ = 8 MN/m³  →  $f_0$ = 52,8 Hz",
        13,
        th.accent,
        bold=True,
    )
    s.text(ax + cw / 2, slab_y + 98, "$ΔL = 30 lg(f/f_0)$ or $40 lg(f/f_0)$", 13, th.fg)
    s.text(ax + cw / 2, slab_y + 120, "(ISO 12354-2 C.1 / C.3)", 12, th.muted)

    # ===== (b) A walking surface on discrete mounts =====
    bx = col[1]
    s.text(bx + cw / 2, top + 22, "(b) discrete mounts", 15, th.fg, bold=True)
    s.rect(bx, slab_y, cw, 46, th.panel, th.fg, sw=2.2)
    s.text(bx + cw / 2, slab_y + 30, "structural slab", 12, th.fg)
    s.rect(bx + 10, slab_y - 54, cw - 20, 22, th.panel, th.fg, sw=2.0)
    s.text(bx + cw / 2, slab_y - 40, "50 mm surface, 115 kg/m²", 12, th.fg)
    for k in range(4):
        mx = bx + 34 + k * (cw - 68) / 3.0
        _spring_v(s, mx, slab_y - 32, slab_y, th.accent, coils=2, width=7.0, sw=1.8)
    # The reverberant bending field the surface carries between the mounts.
    s.path(
        f"M {bx + 16} {slab_y - 66} Q {bx + 60} {slab_y - 80} "
        f"{bx + 104} {slab_y - 66} T {bx + 192} {slab_y - 66} "
        f"T {bx + 224} {slab_y - 66}",
        stroke=th.primary,
        sw=1.8,
    )
    s.text(
        bx + cw / 2,
        slab_y - 86,
        "reverberant bending field",
        12,
        th.primary,
        italic=True,
    )
    s.text(
        bx + cw / 2, slab_y + 74, "4 mounts per m² of 2 MN/m", 13, th.accent, bold=True
    )
    s.text(bx + cw / 2, slab_y + 98, "30 dB per decade, not 40", 13, th.fg, mono=True)
    s.text(bx + cw / 2, slab_y + 120, "(Vér's two-subsystem SEA model)", 12, th.muted)

    # ===== (c) A wall lining, twice: bonded and on studs =====
    cx = col[2]
    s.text(cx + cw / 2, top + 22, "(c) wall lining: two fixings", 15, th.fg, bold=True)
    for k, (label, formula, f_0, rating) in enumerate(
        (
            ("adhesive dabs", "D.1", "542 Hz", "−9,0 dB"),
            ("studs + cavity", "D.2", "70,8 Hz", "+13,8 dB"),
        )
    ):
        wx = cx + k * 136.0
        s.rect(wx, slab_y - 96, 26, 150, th.panel, th.fg, sw=2.0)
        s.text(wx + 13, slab_y + 70, "masonry", 11, th.muted)
        if k == 0:
            for j in range(4):
                s.rect(wx + 26, slab_y - 84 + j * 36, 12, 14, th.accent, th.fg, sw=1.0)
        else:
            s.rect(
                wx + 26, slab_y - 96, 40, 150, th.panel, th.muted, sw=1.2, dash="4,3"
            )
            for j in range(2):
                s.rect(wx + 30, slab_y - 90 + j * 96, 32, 8, th.accent, th.fg, sw=1.0)
        board_x = wx + (38 if k == 0 else 66)
        s.rect(board_x, slab_y - 96, 12, 150, th.panel, th.primary, sw=2.0)
        s.text(wx + 44, slab_y - 108, label, 11, th.fg, bold=True)
        s.text(wx + 44, slab_y + 92, f"({formula})  $f_0$ = {f_0}", 12, th.accent)
        s.text(
            wx + 44,
            slab_y + 114,
            rating,
            13,
            th.secondary if k == 0 else th.primary,
            bold=True,
        )

    # ===== Footer =====
    s.rect(48, 402, 804, 82, "none", th.muted, rx=10, dash="6,5")
    s.text(
        450,
        430,
        "the same board, two fixings: nearly 23 dB between them, and no "
        "formula here can see which one was built",
        13,
        th.fg,
    )
    s.text(
        450,
        458,
        "$s′$ is the EN 29052-1 value measured WITHOUT pre-load, and the "
        "series law (C.6) holds only for an uncut layer",
        13,
        th.accent,
    )
