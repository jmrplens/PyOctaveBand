#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Diagrams of the buildings guides: insulation, rooms and design prediction.

One subject: sound between and inside the rooms of a building. The
insulation diagrams draw the field and laboratory measurements that grade a
separating element, the room diagrams draw what is measured or predicted
inside one enclosure, and the design diagrams draw the prediction models that
put the two together before anything is built.
"""

from __future__ import annotations

from .canvas import SVG, Theme
from .parts import _accel, _rot_arrow, _spring_v

# ---------------------------------------------------------------------------
# d8 - Airborne sound insulation setup (ISO 16283-1)
# ---------------------------------------------------------------------------

def _d_insulation_setup(s: SVG, th: Theme) -> None:
    top, bot = 90.0, 470.0

    # Two rooms in plan view separated by the test partition.
    s.rect(70, top, 375, bot - top, th.panel, th.fg, rx=6, sw=3)
    s.rect(465, top, 365, bot - top, th.panel, th.fg, rx=6, sw=3)
    s.rect(445, top, 20, bot - top, th.secondary, th.fg, sw=2)  # partition (S)
    s.text(455, 80, "Test partition", 20, th.secondary, bold=True)

    s.text(90, top + 32, "Source room", 22, th.fg, bold=True, anchor="start")
    s.text(90, top + 58, "L₁", 20, th.muted, anchor="start")
    s.text(486, top + 32, "Receiving room", 22, th.fg, bold=True, anchor="start")
    s.text(486, top + 58, "L₂ , T", 20, th.muted, anchor="start")

    # Loudspeaker in a corner of the source room (bottom-left).
    lsx, lsy = 150.0, 405.0
    for r in (40, 66, 92):
        s.path(f"M {lsx + r * 0.22:.1f} {lsy - r:.1f} "
               f"A {r} {r} 0 0 1 {lsx + r:.1f} {lsy - r * 0.22:.1f}",
               stroke=th.accent, sw=1.6)
    s.rect(lsx - 26, lsy - 30, 52, 60, th.panel, th.primary, rx=6, sw=2)
    s.circle(lsx, lsy - 10, 12, th.primary)
    s.circle(lsx, lsy - 10, 5, th.bg)
    s.circle(lsx, lsy + 16, 7, th.primary)
    s.text(lsx, lsy + 52, "Loudspeaker", 20, th.fg, bold=True)

    # Microphone positions (five per room, in the central zone).
    src_mics = [(150, 315), (255, 250), (360, 300), (300, 360), (390, 205)]
    rec_mics = [(590, 160), (653, 160), (560, 290), (690, 380), (785, 300)]
    for mics in (src_mics, rec_mics):
        for mx, my in mics:
            s.circle(mx, my, 8, th.fg)
            s.circle(mx, my, 3, th.bg)
    s.text(268, 172, "microphone positions", 18, th.muted)
    s.text(636, 430, "microphone positions", 18, th.muted)

    # Normative minimum separations (ISO 16283-1, 7.6 and 7.2.2).
    s.dim(150, 395, 150, 317, "≥ 1.0 m", offset=-42, size=20)          # 7.6c
    s.dim(178, 405, 443, 405, "≥ 1.0 m", offset=0, size=20)            # 7.2.2
    s.dim(590, 160, 653, 160, "≥ 0.7 m", offset=42, size=20)           # 7.6a
    s.dim(785, 300, 830, 300, "≥ 0.5 m", offset=-46, size=20)          # 7.6b

    # Clause legend.
    for y, txt in (
        (505, "7.6 a) ≥ 0.7 m between microphone positions"),
        (531, "7.6 b) ≥ 0.5 m to room boundaries"),
        (557, "7.6 c) ≥ 1.0 m to the loudspeaker"),
        (583, "7.2.2 ≥ 1.0 m loudspeaker to separating partition"),
    ):
        s.text(80, y, txt, 18, th.fg, anchor="start")


# ---------------------------------------------------------------------------
# d9 - ISO 18233 indirect impulse-response measurement chain
# ---------------------------------------------------------------------------

def _d_ir_measurement(s: SVG, th: Theme) -> None:
    bw, bh = 200.0, 96.0
    xs = (120.0, 350.0, 580.0)
    y1, y2 = 110.0, 300.0

    def box(x: float, y: float, title: str, subs: list[str], color: str,
            mono: bool) -> None:
        s.rect(x, y, bw, bh, th.panel, color, rx=12, sw=2)
        t_size = 20 if len(title) > 11 else 22
        if subs:
            s.text(x + bw / 2, y + 38, title, t_size, th.fg, bold=True)
            if len(subs) == 1:
                s.text(x + bw / 2, y + 66, subs[0], 18, color,
                       mono=mono, italic=mono)
            else:
                s.text(x + bw / 2, y + 62, subs[0], 18, color)
                s.text(x + bw / 2, y + 82, subs[1], 18, color)
        else:
            s.text(x + bw / 2, y + bh / 2 + 7, title, t_size, th.fg, bold=True)

    # Row 1 (left to right): the physical excitation path.
    box(xs[0], y1, "Excitation", ["ESS sweep / MLS"], th.primary, False)
    box(xs[1], y1, "Loudspeaker", [], th.fg, False)
    box(xs[2], y1, "Room", ["h(t)"], th.secondary, True)
    s.arrow(xs[0] + bw, y1 + bh / 2, xs[1] - 2, y1 + bh / 2, th.fg, 2)
    s.arrow(xs[1] + bw, y1 + bh / 2, xs[2] - 2, y1 + bh / 2, th.fg, 2)

    # Serpentine connector: the acoustic field couples Room -> Microphone.
    cx = xs[2] + bw / 2
    s.arrow(cx, y1 + bh, cx, y2 - 2, th.muted, 2)
    s.text(cx - 12, (y1 + bh + y2) / 2 + 5, "acoustic path", 18, th.muted,
           anchor="end", italic=True)

    # Row 2 (right to left): recover the impulse response by deconvolution.
    box(xs[2], y2, "Microphone", [], th.primary, False)
    box(xs[1], y2, "Deconvolution", ["correlation /", "inverse filter"],
        th.accent, False)
    box(xs[0], y2, "IR", ["ĥ(t)"], th.accent, True)
    s.arrow(xs[2], y2 + bh / 2, xs[1] + bw + 2, y2 + bh / 2, th.fg, 2)
    s.arrow(xs[1], y2 + bh / 2, xs[0] + bw + 2, y2 + bh / 2, th.fg, 2)

    s.text(450, 425,
           "The room response h(t) is recovered by deconvolving the "
           "microphone signal.", 18, th.fg)


# ---------------------------------------------------------------------------
# d11 - ISO 16283-2 impact sound insulation setup
# ---------------------------------------------------------------------------

def _d_impact(s: SVG, th: Theme) -> None:
    bx0, bx1 = 90.0, 620.0          # building left / right walls
    top = 82.0
    floor_top, floor_bot = 292.0, 316.0  # separating floor slab
    bot = 512.0                     # receiving-room floor

    # Building shell and the two stacked rooms.
    s.rect(bx0, top, bx1 - bx0, floor_top - top, th.panel, th.fg, sw=2.5)
    s.rect(bx0, floor_bot, bx1 - bx0, bot - floor_bot, th.panel, th.fg, sw=2.5)
    s.rect(bx0, floor_top, bx1 - bx0, floor_bot - floor_top, th.secondary,
           th.fg, sw=2)  # separating floor / ceiling
    s.text(bx0 + 16, top + 30, "Source room (upper)", 21, th.fg, bold=True,
           anchor="start")
    s.text(bx0 + 16, bot - 16, "Receiving room (lower)", 21, th.fg, bold=True,
           anchor="start")
    s.text(bx1 - 12, floor_top - 8, "Separating floor", 17, th.secondary,
           bold=True, anchor="end")

    # Tapping machine standing on the separating floor (five hammers).
    mx = bx0 + 165.0
    body_y = floor_top - 40.0
    s.rect(mx - 60, body_y, 120, 28, th.primary, th.fg, rx=5, sw=2)
    for hx in range(-40, 41, 20):
        s.line(mx + hx, body_y + 28, mx + hx, floor_top - 2, th.fg, 2.4)
        s.circle(mx + hx, floor_top - 2, 4.2, th.fg)
    s.line(mx - 54, body_y + 28, mx - 54, floor_top, th.fg, 2)   # legs
    s.line(mx + 54, body_y + 28, mx + 54, floor_top, th.fg, 2)
    s.text(mx, body_y - 12, "Tapping machine", 19, th.fg, bold=True)

    # Structure-borne path through the slab, radiated into the room below.
    s.arrow(mx, floor_bot + 2, mx, floor_bot + 42, th.secondary, 2.2)
    s.text(mx - 12, floor_bot + 30, "structure-borne impact", 15, th.secondary,
           anchor="end", italic=True)
    for r in (46, 74, 102):
        s.path(f"M {mx - r * 0.72:.1f} {floor_bot + 44 + r * 0.5:.1f} "
               f"A {r} {r} 0 0 0 {mx + r * 0.72:.1f} {floor_bot + 44 + r * 0.5:.1f}",
               stroke=th.accent, sw=1.6)
    s.text(mx, bot - 44, "radiated impact sound", 15, th.accent, italic=True)

    # Microphone positions on the receiving-room floor.
    for off in (300, 400, 500):
        s.mic(bx0 + off, bot - 120, bot, 0.95)
    s.text(bx0 + 400, floor_bot + 42, "Microphone positions", 16, th.muted)

    # Normative relations (right column); no invented spacing dimensions.
    lx = 648.0
    s.text(lx, 118, "Impact sound insulation", 18, th.fg, bold=True,
           anchor="start")
    box_items = [
        (160, "L′nT = Li − 10 log10(T/T₀)", th.primary),
        (192, "L′n = Li + 10 log10(A/A₀)", th.primary),
        (224, "A = 0.16 V/T  (Sabine)", th.muted),
        (256, "T₀ = 0.5 s , A₀ = 10 m²", th.accent),
    ]
    for y, txt, col in box_items:
        # The face is the one that keeps the two spelt-out logarithms flush
        # with the panel below them.
        s.text(lx, y, txt, 14, col, anchor="start", mono=True,
               bold=(col != th.muted))
    s.rect(lx - 10, 292, 236, 100, "none", th.muted, rx=10, dash="6,5")
    s.text(lx, 320, "Li = energy-averaged", 15, th.fg, anchor="start")
    s.text(lx, 342, "band level (Formula 10)", 15, th.fg, anchor="start")
    s.text(lx, 374, "ISO 717-2 → Ln,w , CI", 16, th.secondary, anchor="start",
           bold=True)


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
    wall_bot = 430.0                       # wall runs on past the slab (cross)
    bl, br = 70.0, 830.0
    jx, jy = wx, slab_cy                   # junction node

    # --- structural shell: two rooms, separating wall, flanking slab --------
    s.rect(bl, room_top, wall_l - bl, room_bot - room_top, th.panel, th.fg, sw=2.5)
    s.rect(wall_r, room_top, br - wall_r, room_bot - room_top, th.panel, th.fg, sw=2.5)
    # Flanking element (continuous slab through the junction).
    s.rect(bl, slab_top, br - bl, slab_bot - slab_top, th.panel, th.fg, sw=2)
    for hx in range(int(bl) + 16, int(br), 34):
        s.line(hx, slab_top, hx - 12, slab_bot, th.muted, 0.9)
    # Separating element (vertical wall, drawn on top -> rigid cross junction).
    s.rect(wall_l, room_top, wall_r - wall_l, wall_bot - room_top, th.secondary,
           th.fg, sw=2)

    s.text(bl + 16, room_top + 34, "Source room", 22, th.fg, bold=True, anchor="start")
    s.text(bl + 16, room_top + 60, "L₁", 20, th.muted, anchor="start")
    s.text(wall_r + 16, room_top + 34, "Receiving room", 22, th.fg, bold=True, anchor="start")
    s.text(wall_r + 16, room_top + 60, "L₂ , T", 20, th.muted, anchor="start")
    s.text(wx, room_top - 8, "Separating element (D, d)", 18, th.secondary, bold=True)
    s.text(bl + 16, slab_bot + 22, "Flanking element (F, f)", 18, th.fg, bold=True, anchor="start")

    # Loudspeaker (airborne excitation) in the source room, mic in receiving.
    lsx, lsy = 140.0, 300.0
    for r in (30, 50, 70):
        s.path(f"M {lsx + r * 0.22:.1f} {lsy - r:.1f} "
               f"A {r} {r} 0 0 1 {lsx + r:.1f} {lsy - r * 0.22:.1f}",
               stroke=th.muted, sw=1.4)
    s.rect(lsx - 22, lsy - 26, 44, 52, th.panel, th.fg, rx=5, sw=2)
    s.circle(lsx, lsy - 8, 10, th.fg)
    s.circle(lsx, lsy - 8, 4, th.bg)
    s.circle(lsx, lsy + 14, 6, th.fg)
    s.text(lsx, lsy + 50, "Loudspeaker", 18, th.fg, bold=True)
    s.mic(786.0, 236.0, room_bot, 0.9)
    s.text(786.0, 220.0, "Microphone", 18, th.fg, bold=True)

    # --- transmission paths -------------------------------------------------
    # Dd: straight through the separating element, well above the slab.
    ddy = 172.0
    s.arrow(250.0, ddy, 648.0, ddy, c_dd, 3.0)
    s.text(300.0, ddy - 12, "Dd", 24, c_dd, bold=True)

    # Ff: down onto the flanking slab, along it through the junction, up again.
    s.line(250.0, 284.0, 250.0, slab_cy, c_ff, 2.8)
    s.line(250.0, slab_cy, 650.0, slab_cy, c_ff, 2.8)
    s.arrow(650.0, slab_cy, 650.0, 288.0, c_ff, 2.8)
    s.text(662.0, 300.0, "Ff", 24, c_ff, bold=True, anchor="start")

    # Fd: flanking element (source) -> junction -> radiates from the wall.
    s.line(330.0, 320.0, 330.0, slab_cy, c_fd, 2.8)
    s.line(330.0, slab_cy, 444.0, slab_cy, c_fd, 2.8)
    s.line(444.0, slab_cy, 444.0, 296.0, c_fd, 2.8)
    s.arrow(444.0, 296.0, 556.0, 236.0, c_fd, 2.8)
    s.text(560.0, 230.0, "Fd", 24, c_fd, bold=True, anchor="start")

    # Df: separating wall (source) -> junction -> radiates from the slab.
    s.line(392.0, 236.0, 456.0, 296.0, c_df, 2.8)
    s.line(456.0, 296.0, 456.0, slab_cy, c_df, 2.8)
    s.line(456.0, slab_cy, 614.0, slab_cy, c_df, 2.8)
    s.arrow(614.0, slab_cy, 614.0, 316.0, c_df, 2.8)
    s.text(626.0, 322.0, "Df", 24, c_df, bold=True, anchor="start")

    # Junction node on top of everything.
    s.circle(jx, jy, 6.5, th.bg, th.fg, 2.2)
    s.text(360.0, slab_bot + 22, "junction", 16, th.muted, italic=True)
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
        s.text(bl + 58, ly, txt, 19, th.fg, anchor="start")
        ly += 32
    s.text(450.0, ly + 12,
           "R'w = −10 log10 Σ 10^(−Rij,w /10) dB   (EN 12354-1, Formula 26)",
           19, th.muted, bold=True)


def _d_room_measurement(s: SVG, th: Theme) -> None:
    """Room-acoustics measurement layout (ISO 3382-1 positions, ISO 3382-2 grades).

    A top-view room plan with two source positions and six microphone
    positions plus the ISO 3382-1 spacing rules, and a table of the
    ISO 3382-2:2008 Table 1 minimum position counts for the three grades.
    """
    # --- Room plan (top view) ------------------------------------------------
    rx, ry, rw, rh = 60.0, 96.0, 500.0, 300.0
    s.rect(rx, ry, rw, rh, th.panel, th.fg, rx=6, sw=2.4)
    s.text(rx + 10, ry - 12, "Room plan (top view)", 20, th.fg, "start", bold=True)

    # Two loudspeaker source positions (ISO 3382-1: at least two).
    def _speaker(x: float, y: float, label: str) -> None:
        s.rect(x - 13, y - 11, 26, 22, th.primary, th.fg, rx=3, sw=1.6)
        s.circle(x, y, 5, th.bg, th.fg, 1.2)
        s.text(x, y - 18, label, 18, th.primary, "middle", bold=True)

    _speaker(rx + 70, ry + 70, "S1")
    _speaker(rx + rw - 80, ry + rh - 70, "S2")

    # Six microphone positions, asymmetric (ISO 3382-1: >= 2 m apart,
    # >= 1 m from surfaces; >= 3 receivers per source in ISO 3382-2 precision).
    mics = [
        (rx + 180, ry + 90, "M1"),
        (rx + 300, ry + 55, "M2"),
        (rx + 420, ry + 130, "M3"),
        (rx + 250, ry + 220, "M4"),
        (rx + 380, ry + 250, "M5"),
        (rx + 130, ry + 210, "M6"),
    ]
    for mx, my, label in mics:
        s.circle(mx, my, 7, th.secondary, th.fg, 1.4)
        s.text(mx + 12, my + 6, label, 17, th.fg, "start", bold=True)

    # Spacing annotations.
    m1 = (rx + 180, ry + 90)
    m2 = (rx + 300, ry + 55)
    s.line(m1[0], m1[1], m2[0], m2[1], th.accent, 1.6, dash="5,4")
    s.text((m1[0] + m2[0]) / 2, (m1[1] + m2[1]) / 2 - 8,
           "≥ 2 m", 17, th.accent, "middle", bold=True)
    m6 = (rx + 130, ry + 210)
    s.arrow(m6[0], m6[1] + 9, m6[0], ry + rh, th.muted, 1.4)
    s.text(m6[0] - 8, (m6[1] + ry + rh) / 2 + 6, "≥ 1 m", 16, th.fg, "end")
    # Minimum source-receiver distance guideline.
    s.line(rx + 70, ry + 70, m1[0], m1[1], th.primary, 1.3, dash="4,4")

    # Legend + ISO 3382-1 rules, to the right of the plan.
    lx = rx + rw + 24
    s.circle(lx + 8, ry + 16, 7, th.secondary, th.fg, 1.4)
    s.text(lx + 24, ry + 22, "Microphone position", 17, th.fg, "start")
    s.rect(lx, ry + 40, 16, 14, th.primary, th.fg, rx=2, sw=1.4)
    s.text(lx + 24, ry + 52, "Loudspeaker source", 17, th.fg, "start")
    for i, line in enumerate((
        "ISO 3382-1 (positions):",
        "• ≥ 2 source positions",
        "• mics ≥ 2 m apart",
        "• ≥ 1 m from surfaces",
        "• mic height 1.2 m",
        "d_min = 2√(V/cT)",
    )):
        bold = i == 0 or line.startswith("d_min")
        s.text(lx, ry + 88 + i * 30, line, 17, th.fg, "start", bold=bold)

    # --- ISO 3382-2 Table 1: minimum measurement positions per grade ---------
    ty = ry + rh + 46.0
    s.text(60, ty - 14, "ISO 3382-2 — reverberation-time measurement grades",
           20, th.fg, "start", bold=True)
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
        s.text(cx, ty + 26, label, 17, th.fg, anchor, bold=True)
    for r, row in enumerate(rows):
        yy = ty + th_row * (r + 1)
        if r < len(rows) - 1:
            s.line(60, yy + th_row, 60 + tw, yy + th_row, th.muted, 1.0)
        for (cx, _, anchor), value in zip(cols, row):
            col = th.primary if cx == 70.0 else th.fg
            s.text(cx, yy + 26, value, 17, col, anchor, bold=(cx == 70.0))


def _d_room_noise(s: SVG, th: Theme) -> None:
    """Room-noise rating methods (ANSI/ASA S12.2-2019): NC and RC Mark II.

    From a single octave-band spectrum, two parallel lanes: the NC tangency
    method (Table 1) and the RC Mark II rating and spectral tag (Annex D).
    """
    # --- Shared input spectrum ----------------------------------------------
    cx = 450.0
    iw, ih = 540.0, 62.0
    s.rect(cx - iw / 2, 56, iw, ih, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 84, "Octave-band sound pressure levels  L(f)", 20, th.fg,
           "middle", bold=True)
    s.text(cx, 106, "16 Hz – 8000 Hz", 15, th.muted, "middle")

    lxc, rxc = 232.0, 668.0
    s.arrow(cx, 118, lxc, 158, th.fg, 1.8)
    s.arrow(cx, 118, rxc, 158, th.fg, 1.8)

    bw, bh = 372.0, 62.0

    def _step(cxx: float, y: float, l1: str, l2: str, color: str) -> None:
        s.rect(cxx - bw / 2, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cxx, y + 27, l1, 18, th.fg, "middle", bold=True)
        if l2:
            s.text(cxx, y + 48, l2, 14, th.muted, "middle")

    # --- Left lane: NC tangency method (Table 1) ----------------------------
    _step(lxc, 158, "NC — tangency method", "Table 1 curves", th.primary)
    _step(lxc, 256, "NC value in each band", "curve level = L(f) at that f", th.fg)
    _step(lxc, 354, "NC = highest curve touched", "note the governing band", th.fg)
    s.arrow(lxc, 220, lxc, 256, th.fg, 1.8)
    s.arrow(lxc, 318, lxc, 354, th.fg, 1.8)
    s.arrow(lxc, 416, lxc, 470, th.fg, 1.8)
    s.rect(lxc - bw / 2, 470, bw, 58, "none", th.primary, rx=10, sw=2.4)
    s.text(lxc, 505, "NC-NN (band)", 23, th.fg, "middle", bold=True)

    # --- Right lane: RC Mark II rating and tag (Annex D) ---------------------
    _step(rxc, 158, "RC Mark II  (Annex D)", "−5 dB/octave curves", th.secondary)
    _step(rxc, 256, "LMF = (L500 + L1000 + L2000) / 3", "RC = round(LMF)   (clause D.4)",
          th.fg)
    s.arrow(rxc, 220, rxc, 256, th.fg, 1.8)
    s.arrow(rxc, 318, rxc, 354, th.fg, 1.8)
    # Spectral-tag rule box (clause D.3).
    s.rect(rxc - bw / 2, 354, bw, 116, th.panel, th.fg, rx=10, sw=2)
    s.text(rxc, 379, "Spectral tag  (clause D.3)", 18, th.fg, "middle", bold=True)
    for i, line in enumerate((
        "R  rumble: a band ≤ 500 Hz exceeds RC by > 5 dB",
        "H  hiss: a band ≥ 1000 Hz exceeds RC by > 3 dB",
        "N  neutral: within both tolerances",
    )):
        s.text(rxc - bw / 2 + 18, 403 + i * 22, line, 14, th.fg, "start")
    s.arrow(rxc, 470, rxc, 490, th.fg, 1.8)
    s.rect(rxc - bw / 2, 490, bw, 58, "none", th.secondary, rx=10, sw=2.4)
    s.text(rxc, 525, "RC-NN(A)", 23, th.fg, "middle", bold=True)


def _d_enclosed_space_absorption(s: SVG, th: Theme) -> None:
    """Absorption area and reverberation time of a room (EN 12354-6:2003)."""
    cx = 450.0
    bw, bh = 660.0, 58.0
    x0 = cx - bw / 2

    # --- Inputs (two feeder boxes) -----------------------------------------
    iw = 320.0
    s.rect(cx - bw / 2, 48, iw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(cx - bw / 2 + iw / 2, 72, "Surfaces  (Si, αs,i)", 17, th.fg, "middle",
           bold=True)
    s.text(cx - bw / 2 + iw / 2, 92, "area and absorption per band", 13, th.muted,
           "middle")
    s.rect(cx + bw / 2 - iw, 48, iw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(cx + bw / 2 - iw / 2, 72, "Objects  (Vobj)", 17, th.fg, "middle",
           bold=True)
    s.text(cx + bw / 2 - iw / 2, 92, "Aobj = Vobj^(2/3)  (Formula 4)", 13,
           th.muted, "middle")
    s.arrow(cx - bw / 2 + iw / 2, 106, cx - 60, 150, th.fg, 1.8)
    s.arrow(cx + bw / 2 - iw / 2, 106, cx + 60, 150, th.fg, 1.8)

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 25, l1, 17, th.fg, "middle", bold=True)
        s.text(cx, y + 45, l2, 13, th.muted, "middle")

    _step(150, "Equivalent absorption area  A  (clause 4.3, Formula 1)",
          "A = Σ αs,i·Si + Σ Aobj + Aair;   Aair = 4·m·V·(1 − ψ)  (Formula 2)",
          th.primary)
    _step(238, "Object fraction  ψ = Σ Vobj / V   (Formula 3)",
          "air absorption negligible below 1 kHz for V < 200 m³", th.fg)
    s.arrow(cx, 210, cx, 238, th.fg, 1.8)
    s.arrow(cx, 296, cx, 324, th.fg, 1.8)

    # --- Output -------------------------------------------------------------
    s.rect(x0, 324, bw, 58, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, 349, "Reverberation time  T = 55.3/c₀ · V·(1 − ψ) / A  (Formula 5)",
           17, th.fg, "middle", bold=True)
    s.text(cx, 369, "c₀ = 345.6 m/s so 55.3/c₀ = 0.16  (clause 4.4)", 13,
           th.muted, "middle")


def _d_open_plan(s: SVG, th: Theme) -> None:
    """ISO 3382-3 open-plan measurement line and its single-number quantities."""
    ly = 150.0
    lx0, lx1 = 120.0, 812.0
    # Talker/source near the origin.
    s.person(lx0, ly, h=70)
    s.text(lx0, ly + 22, "source", 13, th.muted, "middle")
    s.text(lx0, ly + 40, "(r₀ = 1 m)", 13, th.muted, "middle")
    # Measurement line with workstations and positions.
    s.line(lx0 + 26, ly - 30, lx1, ly - 30, th.fg, 1.8, dash="6,5")
    dists = [(0.18, "2 m"), (0.36, "4 m"), (0.56, "8 m"), (0.78, "12 m"), (0.98, "16 m")]
    for frac, lab in dists:
        px = lx0 + 26 + frac * (lx1 - lx0 - 26)
        s.rect(px - 22, ly + 4, 44, 26, th.panel, th.muted, rx=4, sw=1.3)  # desk
        s.circle(px, ly - 30, 5, th.primary)  # measurement position
        s.text(px, ly - 42, lab, 13, th.fg, "middle")
    # Evaluation-range bracket (2 m to 16 m).
    bx0 = lx0 + 26 + 0.18 * (lx1 - lx0 - 26)
    bx1 = lx0 + 26 + 0.98 * (lx1 - lx0 - 26)
    s.line(bx0, ly + 52, bx1, ly + 52, th.accent, 1.6)
    s.line(bx0, ly + 46, bx0, ly + 58, th.accent, 1.6)
    s.line(bx1, ly + 46, bx1, ly + 58, th.accent, 1.6)
    s.text((bx0 + bx1) / 2, ly + 74, "spatial-decay fit range (2 m to 16 m)", 14,
           th.accent, "middle")

    chips = [
        ("D₂,S", "spatial decay rate", "dB per doubling · Cl. 6.2", th.primary),
        ("Lp,A,S,4m", "speech level at 4 m", "A-weighted · Cl. 3.3", th.primary),
        ("rD", "distraction distance", "fitted STI = 0.50 · Cl. 3.6", th.secondary),
        ("rP", "privacy distance", "fitted STI = 0.20 · Cl. 3.7", th.secondary),
    ]
    cw, cgap = 190.0, 14.0
    cx = (900 - (len(chips) * cw + (len(chips) - 1) * cgap)) / 2
    for sym, name, note, color in chips:
        s.rect(cx, 320, cw, 118, th.panel, color, rx=10, sw=2)
        s.text(cx + cw / 2, 356, sym, 22, th.fg, "middle", bold=True)
        s.text(cx + cw / 2, 384, name, 15, color, "middle", bold=True)
        s.text(cx + cw / 2, 412, note, 12, th.muted, "middle")
        cx += cw + cgap


def _d_iso12999(s: SVG, th: Theme) -> None:
    """ISO 12999-1 uncertainty: from tabulated reproducibility to the expanded U."""
    cx = 450.0
    bw, bh = 664.0, 60.0
    x0 = cx - bw / 2

    s.rect(x0, 48, bw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 72, "Standard uncertainty  u  — reproducibility read from the tables",
           18, th.fg, "middle", bold=True)
    s.text(cx, 92, "bands: Tables 2/4 · ratings: Tables 3/5 · situation A (σR) / "
           "B (σsitu) / C (σr)", 13, th.muted, "middle")
    s.arrow(cx, 108, cx, 138, th.fg, 1.8)

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 25, l1, 17, th.fg, "middle", bold=True)
        s.text(cx, y + 45, l2, 13, th.muted, "middle")

    _step(138, "Reduce by  m  independent measurements   u/√m   (Formula A.7)",
          "and combine model with reality per Annex A when predicting", th.fg)
    _step(226, "Combine uncorrelated contributions   uc = √(Σ u_i²)   (Formula C.2)",
          "single-number combination of Annex B uses Formula B.2", th.primary)
    _step(314, "Expand   U = k·u   (Formula 2),   k from Table 8   (k ≥ 1)",
          "the coverage factor depends on the reported quantity and situation",
          th.secondary)
    for y0, y1 in ((198, 226), (286, 314)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)
    s.arrow(cx, 374, cx, 404, th.fg, 1.8)

    # Two-sided reporting vs one-sided conformity.
    hw = 320.0
    s.rect(x0, 404, hw, 66, "none", th.primary, rx=10, sw=2.2)
    s.text(x0 + hw / 2, 430, "Report   Y = y ± U   (Formula 3)", 16, th.fg,
           "middle", bold=True)
    s.text(x0 + hw / 2, 452, "two-sided coverage factor", 13, th.muted, "middle")
    s.rect(cx + bw / 2 - hw, 404, hw, 66, "none", th.secondary, rx=10, sw=2.2)
    s.text(cx + bw / 2 - hw / 2, 430, "Declare conformity   (Formulae 4/5)", 16,
           th.fg, "middle", bold=True)
    s.text(cx + bw / 2 - hw / 2, 452, "one-sided coverage factor", 13, th.muted, "middle")


# ---------------------------------------------------------------------------
# Reception plate (EN 15657)
# ---------------------------------------------------------------------------

def _d_reception_plate(s: SVG, th: Theme) -> None:
    """EN 15657 reception plate: source machine on a resiliently supported
    plate, averaged plate velocity, plate power balance."""
    # ===== Source machine standing on the plate =====
    mx = 260.0
    s.text(mx, 150, "Source under test (pump, fan, boiler …)", 20, th.fg,
           bold=True)
    s.rect(mx - 75, 218, 150, 70, th.panel, th.fg, rx=8, sw=2.2)
    s.rect(mx - 48, 190, 58, 28, th.panel, th.muted, rx=6, sw=1.8)
    s.circle(mx + 40, 240, 12, th.bg, th.muted, 1.8)
    for fx in (mx - 52.0, mx + 52.0):
        s.rect(fx - 8, 288, 16, 14, th.fg, rx=2)
        s.arrow(fx, 306, fx, 326, th.secondary, 2.4)
    s.text(310, 356, "injected structure-borne power", 15, th.secondary,
           italic=True)

    # ===== Reception plate on resilient supports =====
    s.rect(100, 302, 460, 32, th.panel, th.primary, rx=3, sw=2.4)
    s.text(455, 324, "Reception plate  (m, S, η)", 16, th.fg, bold=True)
    for ax_ in (140.0, 190.0, 400.0, 500.0):
        _accel(s, ax_, 302)
    s.text(560, 272, "velocity positions → Lv", 16, th.secondary,
           anchor="end")
    for sx in (150.0, 510.0):
        _spring_v(s, sx, 334, 430, th.accent, coils=3)
    s.ground(430, 80, 580)
    s.text(330, 404, "resilient supports", 14, th.muted)

    # ===== Right column: the power balance and the source quantities =====
    s.text(735, 150, "Plate power balance", 20, th.fg, bold=True)
    s.rect(590, 172, 292, 148, "none", th.muted, rx=10, dash="6,5")
    s.text(735, 206, "P = ω·η·(m·S)·⟨v²⟩", 19, th.primary, bold=True,
           mono=True)
    s.text(735, 238, "η = 2.2 / (f·Ts)   (Formula 13)", 15, th.fg, mono=True)
    # Longest line of the panel: a smaller face keeps it inside the dashed box.
    s.text(735, 270, "L_Ws = 10 log10(2πf·η·m·S / f₀m₀S₀)", 13, th.fg, mono=True)
    s.text(735, 296, "+ Lv − 60   (Formula 14)", 14, th.fg, mono=True)
    s.text(735, 366, "→ source quantities (Formulae 15–19):", 15, th.fg,
           bold=True)
    s.text(735, 394, "equivalent blocked force L_Fb,eq ,", 15, th.muted)
    s.text(735, 418, "L_Wsn consumed by EN 12354-5", 15, th.muted)

    # Footer: the spatial velocity average.
    s.text(450, 516,
           "spatial average:  Lv = 10 log10[(1/N)·Σ 10^(Lv,i/10)]   (Formula 12)",
           17, th.fg, mono=True)


# ---------------------------------------------------------------------------
# Installed structure-borne sound paths (EN 12354-5)
# ---------------------------------------------------------------------------

def _d_installed_paths(s: SVG, th: Theme) -> None:
    """EN 12354-5: service equipment on a floor slab, structure-borne paths
    into the receiving room below, and the prediction cascade."""
    bx0, bx1 = 80.0, 590.0
    top, slab_top, slab_bot, bot = 92.0, 296.0, 324.0, 528.0

    # Rooms, continuous floor slab and flanking wall (drawn over the slab).
    s.rect(bx0, top, bx1 - bx0, slab_top - top, th.panel, th.fg, sw=2.5)
    s.rect(bx0, slab_bot, bx1 - bx0, bot - slab_bot, th.panel, th.fg, sw=2.5)
    s.rect(bx0, slab_top, bx1 - bx0 + 26, slab_bot - slab_top, th.panel,
           th.fg, sw=2)
    for hx in range(int(bx0) + 16, int(bx1) + 26, 34):
        s.line(hx, slab_top, hx - 12, slab_bot, th.muted, 0.9)
    s.rect(bx1, top, 26, bot - top, th.panel, th.fg, sw=2)
    s.text(bx0 + 16, top + 32, "Source room", 21, th.fg, bold=True,
           anchor="start")
    s.text(bx0 + 16, bot - 18, "Receiving room", 21, th.fg, bold=True,
           anchor="start")

    # Service equipment on resilient mounts on the slab.
    mx = 210.0
    s.rect(mx - 55, 238, 110, 42, th.panel, th.fg, rx=7, sw=2.2)
    s.rect(mx - 34, 216, 40, 22, th.panel, th.muted, rx=5, sw=1.6)
    for fx in (mx - 36.0, mx + 36.0):
        _spring_v(s, fx, 280, slab_top, th.accent, coils=2, width=6.0, sw=1.6)
    s.text(mx, 200, "Service equipment (pump)", 19, th.fg, bold=True)
    s.text(mx + 78, 268, "coupling D_C   (Formula 19b)", 15, th.secondary,
           anchor="start")

    # Path i = j: the excited slab radiates into the room below.
    s.arrow(mx, slab_bot + 2, mx, slab_bot + 40, th.secondary, 2.4)
    for r in (40, 66, 92):
        s.path(f"M {mx - r * 0.72:.1f} {slab_bot + 42 + r * 0.5:.1f} "
               f"A {r} {r} 0 0 0 {mx + r * 0.72:.1f} {slab_bot + 42 + r * 0.5:.1f}",
               stroke=th.accent, sw=1.6)
    s.text(mx, 484, "excited floor radiates (path i = j)", 15, th.secondary,
           italic=True)

    # Path i -> j: along the slab, through the junction, down the wall.
    s.line(mx + 40, 310, 596, 310, th.primary, 2.6)
    s.line(603, 310, 603, 420, th.primary, 2.6)
    s.arrow(603, 420, 574, 420, th.primary, 2.6)
    s.circle(603, 310, 5, th.bg, th.fg, 2)
    for r in (30, 52):
        s.path(f"M {588 - r * 0.5:.1f} {420 - r * 0.72:.1f} "
               f"A {r} {r} 0 0 0 {588 - r * 0.5:.1f} {420 + r * 0.72:.1f}",
               stroke=th.accent, sw=1.6)
    s.text(584, 288, "path along the slab into the wall  (i → j)", 15,
           th.primary, anchor="end")

    # ===== Right column: the prediction cascade =====
    s.text(760, 120, "Prediction cascade", 20, th.fg, bold=True)
    # The energetic sum is far longer than the other terms, so it carries its
    # own face to stay inside the column instead of running off the canvas.
    steps = [
        ("L_Ws,c", "characteristic power (EN 15657)", th.fg, 19),
        ("− D_C", "coupling at the contacts (19b)", th.secondary, 19),
        ("L_Ws,inst", "installed power (18b)", th.fg, 19),
        ("− D_sa − R_ij,ref", "per transmission path (18a)", th.primary, 19),
        ("10 log10 Σ 10^(L_n,s,ij/10)", "energetic sum L_n,s (17)", th.accent, 15),
    ]
    y = 164.0
    for k, (term, caption, col, size) in enumerate(steps):
        s.text(760, y, term, size, col, bold=True, mono=True)
        s.text(760, y + 22, caption, 14, th.muted)
        if k < len(steps) - 1:
            s.arrow(760, y + 34, 760, y + 56, th.muted, 1.6)
        y += 84

    # Footer: what a path is.
    s.text(335, 574,
           "each path i → j: excited element i, radiating element j in the receiving room",
           16, th.muted)


# ---------------------------------------------------------------------------
# Laboratory sound insulation suite (ISO 10140)
# ---------------------------------------------------------------------------

def _d_insulation_lab(s: SVG, th: Theme) -> None:
    """ISO 10140 laboratory transmission suite in plan view: two
    structurally decoupled reverberant rooms, the test element mounted in
    the ~10 m2 test opening, a corner loudspeaker in the source room and a
    continuously moving (rotating) microphone in each room."""
    top = 92.0
    sc = 72.0                       # px per metre
    src_bot = top + 4.4 * sc        # source room 5.0 m x 4.4 m
    rec_bot = top + 4.1 * sc        # receiving room 4.6 m x 4.1 m
    rec_r = 470.0 + 4.6 * sc

    # Room shells (separate structures).
    s.rect(70, top, 360, src_bot - top, th.panel, th.fg, rx=4, sw=3)
    s.rect(470, top, rec_r - 470, rec_bot - top, th.panel, th.fg, rx=4, sw=3)
    s.text(90, top + 30, "Source room", 21, th.fg, bold=True, anchor="start")
    s.text(90, top + 56, "V₁ ≈ 59 m³", 17, th.muted, anchor="start", mono=True)
    s.text(486, top + 30, "Receiving room", 21, th.fg, bold=True, anchor="start")
    s.text(486, top + 56, "V₂ ≈ 51 m³", 17, th.muted, anchor="start", mono=True)

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
    s.text(450, 66, "structural break", 15, th.muted, italic=True)
    s.line(450, 72, 450, 102, th.muted, 1.0, dash="3,3")
    s.text(450, 452, "Test element in the test opening", 17, th.secondary,
           bold=True)
    s.line(450, op_b + 4, 450, 436, th.muted, 1.0, dash="3,3")

    # Loudspeaker in a corner of the source room.
    lsx, lsy = 150.0, 350.0
    for r in (36, 60, 84):
        s.path(f"M {lsx + r * 0.22:.1f} {lsy - r:.1f} "
               f"A {r} {r} 0 0 1 {lsx + r:.1f} {lsy - r * 0.22:.1f}",
               stroke=th.accent, sw=1.5)
    s.rect(lsx - 24, lsy - 27, 48, 54, th.panel, th.primary, rx=6, sw=2)
    s.circle(lsx, lsy - 9, 11, th.primary)
    s.circle(lsx, lsy - 9, 4.5, th.bg)
    s.circle(lsx, lsy + 15, 6, th.primary)
    s.text(lsx + 4, lsy + 48, "Loudspeaker", 17, th.fg, bold=True)

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
    s.text(285, 298, "moving microphone", 16, th.fg)
    s.text(285, 320, "sweep radius ≥ 1 m", 15, th.muted)
    s.text(640, 313, "moving microphone", 16, th.fg)

    # Dimensions (72 px per metre).
    s.dim(70, src_bot, 430, src_bot, "5.0 m", offset=30, size=18)
    s.dim(470, rec_bot, rec_r, rec_bot, "4.6 m", offset=30 + src_bot - rec_bot,
          size=18)
    s.dim(rec_r, top, rec_r, rec_bot, "4.1 m", offset=32, size=18,
          label_side="right")
    s.dim(430, op_t, 430, op_b, "3.75 m", offset=-24, size=17)

    # Normative facility limits.
    for y, txt in (
        (508.0, "Test opening ≈ 10 m² (3.75 m × 2.7 m); shorter edge ≥ 2.3 m"),
        (536.0, "Room volumes ≥ 50 m³, differing by at least 10 %"),
        (564.0, "Continuously moving microphone: sweep radius ≥ 1 m, traverse ≥ 15 s"),
    ):
        s.text(80, y, txt, 18, th.fg, anchor="start")


# ---------------------------------------------------------------------------
# Reverberation-time prediction (Sabine / Eyring)
# ---------------------------------------------------------------------------

def _d_reverberation_prediction(s: SVG, th: Theme) -> None:
    """The guide's 10 x 7 x 3.5 m room through the Sabine and Eyring
    absorption terms into the per-band T60 table the library returns,
    with the diffuse-field validity note."""
    # Room data
    s.rect(170, 52, 560, 78, th.panel, th.fg, rx=10, sw=2)
    s.text(450, 77, "Room 10 × 7 × 3.5 m — V = 245 m³, S = 259 m²", 16,
           th.fg, bold=True)
    s.text(450, 97, "hard end walls, lightly treated side walls, carpet "
           "and acoustic ceiling", 12, th.muted)
    s.text(450, 117, "mean absorption ᾱ runs from 0.21 at 125 Hz to 0.51 "
           "at 4 kHz", 12, th.muted)

    # The two models
    s.rect(60, 178, 380, 96, th.panel, th.primary, rx=10, sw=2)
    s.text(250, 203, "Sabine", 15, th.fg, bold=True)
    s.text(250, 224, "T = 0.161·V / (Σ Si·αi + 4mV)", 13, th.primary,
           mono=True)
    s.text(250, 245, "low, even absorption (ᾱ up to ≈ 0.2);", 11,
           th.muted)
    s.text(250, 262, "stays finite even at α = 1", 11, th.muted)
    s.rect(460, 178, 380, 96, th.panel, th.secondary, rx=10, sw=2)
    s.text(650, 203, "Eyring", 15, th.fg, bold=True)
    s.text(650, 224, "T = 0.161·V / (−S·ln(1 − ᾱ) + 4mV)", 13,
           th.secondary, mono=True)
    s.text(650, 245, "strong, even absorption;", 11, th.muted)
    s.text(650, 262, "reaches T = 0 at total absorption", 11, th.muted)
    s.arrow(350, 130, 265, 174, th.fg, 1.8)
    s.arrow(550, 130, 635, 174, th.fg, 1.8)

    # Per-band table
    s.rect(100, 318, 700, 150, th.panel, th.fg, rx=10, sw=1.8)
    s.text(450, 344, "Predicted T60 per octave band", 14, th.fg, bold=True)
    freqs = ("125 Hz", "250", "500", "1k", "2k", "4k")
    sab = ("0.74", "0.47", "0.37", "0.31", "0.30", "0.30")
    eyr = ("0.66", "0.39", "0.29", "0.23", "0.21", "0.22")
    xc = [292.0 + 94.0 * i for i in range(6)]
    for x, f in zip(xc, freqs):
        s.text(x, 372, f, 12, th.muted, bold=True)
    s.line(130, 382, 770, 382, th.muted, 1.0)
    s.text(130, 404, "Sabine [s]", 12, th.primary, bold=True,
           anchor="start")
    for x, v in zip(xc, sab):
        s.text(x, 404, v, 13, th.fg)
    s.text(130, 432, "Eyring [s]", 12, th.secondary, bold=True,
           anchor="start")
    for x, v in zip(xc, eyr):
        s.text(x, 432, v, 13, th.fg)
    s.text(450, 456, "Eyring runs 11 to 29 % shorter here: ᾱ is past "
           "Sabine's comfort zone", 12, th.muted, italic=True)
    s.arrow(250, 274, 320, 314, th.fg, 1.8)
    s.arrow(650, 274, 580, 314, th.fg, 1.8)

    # Validity note
    s.rect(130, 500, 640, 68, "none", th.accent, rx=10, sw=1.6, dash="6,5")
    s.text(450, 527, "Domain of validity: a diffuse field that stays "
           "diffuse while it decays", 14, th.accent, bold=True)
    s.text(450, 551, "below the Schroeder frequency, in coupled volumes "
           "and in corridor-like rooms no single T60 exists", 12, th.fg)


# ---------------------------------------------------------------------------
# Panel between rooms: mass law and the coincidence dip
# ---------------------------------------------------------------------------

def _d_panel_insulation(s: SVG, th: Theme) -> None:
    """A single 12.5 mm plasterboard leaf (m'' = 8.75 kg/m2) mounted in its
    test opening under diffuse incidence, with the predicted R(f) of
    ``single_panel_transmission_loss`` inset: the mass-law rise and the
    coincidence dip at the fc = 2619 Hz of this leaf (Rw = 27 dB)."""
    # --- test opening: heavy filler above and below, the leaf between ------
    px_l, px_r = 380.0, 396.0
    op_t, op_b = 108.0, 332.0
    s.rect(348, 62, 80, op_t - 62, th.panel, th.fg, sw=2)
    s.rect(348, op_b, 80, 46, th.panel, th.fg, sw=2)
    s.rect(px_l, op_t, px_r - px_l, op_b - op_t, th.panel, th.secondary, sw=2)
    for hy in range(int(op_t) + 10, int(op_b) - 2, 14):
        s.line(px_l + 1, hy + 8, px_r - 1, hy - 4, th.secondary, 1.0)
    s.text(388, 52, "Panel under test: 12.5 mm plasterboard", 19, th.fg,
           bold=True)

    # Thickness callout (witness lines up, arrows pointing inward).
    s.line(px_l, op_t, px_l, 88, th.muted, 0.9, dash="3,3")
    s.line(px_r, op_t, px_r, 88, th.muted, 0.9, dash="3,3")
    s.arrow(352.0, 92.0, px_l - 2, 92.0, th.muted, 1.2)
    s.arrow(424.0, 92.0, px_r + 2, 92.0, th.muted, 1.2)
    s.text(434, 97, "12.5 mm", 15, th.fg, anchor="start")

    # --- diffuse incidence on the left, weaker transmitted field right -----
    s.text(180, 116, "Source room", 20, th.fg, bold=True)
    s.text(180, 140, "diffuse incidence", 16, th.muted, italic=True)
    s.arrow(258.0, 152.0, px_l - 6, 196.0, th.accent, 2.2)
    s.arrow(218.0, 244.0, px_l - 6, 246.0, th.accent, 2.2)
    s.arrow(252.0, 330.0, px_l - 6, 292.0, th.accent, 2.2)
    s.text(560, 116, "Receiving room", 20, th.fg, bold=True)
    s.text(510, 226, "transmitted", 16, th.muted, italic=True)
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
    s.text(300, 366, "bending wave at fc", 15, th.accent, anchor="end")
    s.line(305.0, 360.0, 382.0, 326.0, th.muted, 1.0)
    s.text(388, 404, "m″ = 8.8 kg/m²", 16, th.fg, mono=True)

    # --- inset: predicted R(f) with the coincidence dip --------------------
    ix0, iy0 = 572.0, 390.0            # axes origin (bottom-left)
    s.line(ix0, iy0, 850.0, iy0, th.muted, 1.3)
    s.arrow(ix0, iy0, ix0, 128.0, th.muted, 1.3)
    s.text(ix0 - 8, 140, "R", 14, th.muted, italic=True, anchor="end")
    s.text(854, iy0 + 16, "f", 14, th.muted, italic=True, anchor="end")
    s.text(645, 198, "predicted R(f)", 14, th.primary, italic=True)
    import math

    def fx(f: float) -> float:
        return ix0 + math.log10(f / 50.0) * 135.0

    def ry(r: float) -> float:
        return 386.0 - r * 240.0 / 35.0

    for f_t, lab in ((100.0, "100"), (1000.0, "1k")):
        s.line(fx(f_t), iy0, fx(f_t), iy0 + 5, th.muted, 1.2)
        s.text(fx(f_t), iy0 + 20, lab, 12, th.muted)
    # single_panel_transmission_loss(bands, 8.75, fc=2619.3, eta=0.01), dB.
    curve = [(50, 5.3), (63, 7.2), (80, 9.2), (100, 11.1), (125, 13.0),
             (160, 15.1), (200, 17.0), (250, 18.9), (315, 20.9), (400, 23.0),
             (500, 24.9), (630, 26.9), (800, 29.0), (1000, 31.0),
             (1250, 32.9), (1600, 30.3), (2000, 26.9), (2500, 23.6),
             (3150, 25.3), (4000, 28.4), (5000, 31.3)]
    d = ""
    for i, (f_c, r_c) in enumerate(curve):
        d += f"{'M' if i == 0 else ' L'} {fx(f_c):.1f} {ry(r_c):.1f}"
    s.path(d, stroke=th.primary, sw=2.4)
    fcx = fx(2619.3)
    s.line(fcx, iy0, fcx, 150.0, th.secondary, 1.3, dash="5,4")
    s.text(fcx, 142, "fc = 2.6 kHz", 14, th.secondary, bold=True)
    s.circle(fx(2500.0), ry(23.6), 4.0, th.secondary)
    s.text(690, 330, "+6 dB/octave", 14, th.primary, italic=True)
    s.text(628, 170, "Rw = 27 dB", 16, th.fg, bold=True, mono=True)

    # --- captions ----------------------------------------------------------
    s.text(80, 452,
           "Diffuse-field mass law: R rises 6 dB per octave and 6 dB per doubling of m″",
           18, th.fg, anchor="start")
    s.text(80, 480,
           "At fc = (c₀²/2π) √(m″/B′) = 2619 Hz the free bending wave matches the trace wavelength",
           18, th.fg, anchor="start")
    s.text(80, 508,
           "Sharp's prediction rates at Rw = 27 dB; the dip collects the unfavourable deviations",
           18, th.primary, anchor="start", bold=True)


# ---------------------------------------------------------------------------
# Image-source lattice in plan (first reflections of a shoebox room)
# ---------------------------------------------------------------------------

def _d_room_image_sources(s: SVG, th: Theme) -> None:
    """Plan of the guide's 7 x 5 x 3 m room with the source at (2, 1.6),
    the receiver at (5.2, 3.4) and the in-plane images of order 1 and 2 on
    the mirror-room grid, each labelled with its image_source_rir arrival
    time (direct 10.7 ms, first reflections 17.3 to 21.6 ms)."""
    sc = 32.0                            # px per metre

    def x(mx: float) -> float:
        return 98.0 + (mx + 7.5) * sc

    def y(my: float) -> float:
        return 88.0 + (10.4 - my) * sc

    # Mirror-room grid (3 x 3) around the bold real room.
    for gx in (-7.0, 0.0, 7.0):
        for gy_ in (-5.0, 0.0, 5.0):
            if gx == 0.0 and gy_ == 0.0:
                continue
            s.rect(x(gx), y(gy_ + 5.0), 7.0 * sc, 5.0 * sc, "none", th.muted,
                   sw=1.1, dash="6,5")
    s.rect(x(0.0), y(5.0), 7.0 * sc, 5.0 * sc, th.panel, th.fg, sw=2.6)
    s.text(x(3.5), y(0.4), "plan at the source plane z = 1.5 m",
           11 if s.lang == "es" else 13, th.muted)

    # Room dimensions on the real room's walls.
    s.dim(x(0.0), y(0.0), x(7.0), y(0.0), "7.0 m", offset=26, size=15)
    s.dim(x(7.0), y(5.0), x(7.0), y(0.0), "5.0 m", offset=28, size=15,
          label_side="right")

    # Source, receiver and the direct sound.
    sx_, sy_ = x(2.0), y(1.6)
    rx_, ry_ = x(5.2), y(3.4)
    s.line(sx_, sy_, rx_, ry_, th.fg, 1.5)
    s.text((sx_ + rx_) / 2 + 4, (sy_ + ry_) / 2 + 18, "10.7 ms", 14, th.fg,
           mono=True)
    s.circle(sx_, sy_, 7.0, th.secondary)
    s.text(sx_ - 12, sy_ + 5, "S", 17, th.secondary, bold=True, anchor="end")
    s.path(f"M {rx_:.1f} {ry_ - 9:.1f} L {rx_ - 8:.1f} {ry_ + 7:.1f} "
           f"L {rx_ + 8:.1f} {ry_ + 7:.1f} Z", fill=th.primary)
    s.text(rx_ + 13, ry_ + 5, "R", 17, th.primary, bold=True, anchor="start")

    # Example first reflection off the y = 5 wall: real specular path and
    # the equivalent straight line from the image.
    bx, by = x(4.176), y(5.0)
    s.line(sx_, sy_, bx, by, th.accent, 2.0)
    s.arrow(bx, by, rx_, ry_ - 6, th.accent, 2.0)
    s.line(x(2.0), y(8.4), rx_, ry_ - 6, th.accent, 1.3, dash="5,4")
    s.text(502, 185, "the image sees", 13, th.accent, italic=True)
    s.text(502, 204, "a straight path", 13, th.accent, italic=True)

    # Images of order 1 (secondary) and order 2 (accent), with their
    # image_source_rir arrival times.
    order1 = [(2.0, -1.6, "17.3 ms", 14, 18), (2.0, 8.4, "17.3 ms", 14, -12),
              (12.0, 1.6, "20.5 ms", 14, 18), (-2.0, 1.6, "21.6 ms", 14, 18)]
    for mx, my, lab, _sz, dy_ in order1:
        s.circle(x(mx), y(my), 6.0, th.secondary)
        s.text(x(mx), y(my) + dy_ + (4 if dy_ < 0 else 0), lab, 13, th.fg,
               mono=True)
    order2 = [(-2.0, -1.6, "25.6 ms"), (-2.0, 8.4, "25.6 ms"),
              (12.0, -1.6, "24.6 ms"), (12.0, 8.4, "24.6 ms")]
    for mx, my, lab in order2:
        s.circle(x(mx), y(my), 6.0, "none", th.accent, 2.2)
        s.text(x(mx), y(my) + 20, lab, 13, th.muted, mono=True)

    # Legend inside the top-left mirror room.
    s.circle(x(-6.6), y(9.55), 5.5, th.secondary)
    s.text(x(-6.3), y(9.4), "1st order", 14, th.fg, anchor="start")
    s.circle(x(-6.6), y(8.75), 5.5, "none", th.accent, 2.2)
    s.text(x(-6.3), y(8.6), "2nd order", 14, th.fg, anchor="start")

    # --- captions ----------------------------------------------------------
    s.text(80, 612,
           "every reflection is the free-field arrival of a mirror image: t = r/c, √(1−α) per bounce, 1/(4πr) spreading",
           17, th.fg, anchor="start")
    s.text(80, 638,
           "in-plane images up to order 2 shown; the full lattice adds floor, ceiling and outer mirror rooms",
           17, th.muted, anchor="start")
