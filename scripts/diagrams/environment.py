#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Diagrams of the environment guides: sources, propagation and assessment.

One subject: outdoor sound, from what emits it to what the law makes of it at
the receiver. The propagation diagrams draw the geometry the models work in
(barrier and ground, image source, refracted rays), the source diagram draws
a measurement geometry fixed by its own standard, and the assessment diagrams
draw what happens to the level once it has arrived.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .canvas import SVG, Theme

# ---------------------------------------------------------------------------
# d2 - Environmental noise microphone positions (ISO 1996-2)
# ---------------------------------------------------------------------------


def _d_env_positions(s: SVG, th: Theme) -> None:
    gy = 470.0
    s.ground(gy, 40, 860)

    # Building facade (right)
    fx = 700.0
    s.rect(fx, 120, 160, gy - 120, th.panel, th.fg, sw=2)
    for wy in range(158, int(gy) - 50, 78):
        s.rect(fx + 24, wy, 38, 46, th.bg, th.muted, rx=3, sw=1.2)
        s.rect(fx + 96, wy, 38, 46, th.bg, th.muted, rx=3, sw=1.2)
    s.text(fx + 80, 104, "Building façade", 19, th.fg, bold=True)

    # Source (left): car on a road
    s.rect(60, gy - 9, 140, 9, th.muted)
    s.path(
        f"M 88 {gy - 30} L 106 {gy - 48} L 146 {gy - 48} L 164 {gy - 30} Z",
        fill=th.secondary,
    )
    s.rect(80, gy - 32, 96, 14, th.secondary, rx=5)
    s.circle(102, gy - 13, 9, th.fg)
    s.circle(156, gy - 13, 9, th.fg)
    for r in (44, 76, 108):
        s.path(
            f"M {168 + r * 0.5} {gy - 34 - r * 0.55} "
            f"A {r} {r} 0 0 1 {168 + r * 0.87} {gy - 34 + r * 0.1}",
            stroke=th.accent,
            sw=1.6,
        )

    # Position A: free field, capsule 4 m above ground
    ax = 330.0
    a_cap = gy - 230.0
    s.mic(ax, a_cap, gy, 1.15)
    s.dim(ax, gy, ax, a_cap, "4.0 ± 0.2 m", offset=-60, size=17)
    s.text(ax - 20, a_cap - 58, "A — free field", 19, th.fg, bold=True)
    s.text(ax - 20, a_cap - 30, "0 dB", 19, th.accent, bold=True, mono=True)

    # Position B: 2 m in front of the facade, dimension at capsule height
    bx = fx - 108.0
    b_cap = gy - 230.0
    s.mic(bx, b_cap, gy, 1.15)
    s.dim(bx, b_cap + 6, fx, b_cap + 6, "2 m", offset=-14, size=17)
    s.text(bx - 30, b_cap - 58, "B — 2 m from façade", 19, th.fg, bold=True)
    s.text(bx - 30, b_cap - 30, "−3 dB", 19, th.secondary, bold=True, mono=True)

    # Position C: flush-mounted on the facade, below B's dimension zone
    cy = gy - 120.0
    s.circle(fx + 3, cy, 7, th.fg)
    # The leader crosses mic B's mast (plain line crossing, standard
    # drafting); the label itself sits in the clear zone between masts.
    s.line(fx - 2, cy + 5, 470, cy + 60, th.muted, 1.4)
    s.text(462, cy + 84, "C — flush-mounted", 19, th.fg, bold=True)
    s.text(462, cy + 110, "−6 dB", 19, th.secondary, bold=True, mono=True)


def _d_outdoor(s: SVG, th: Theme) -> None:
    c_diff = th.accent  # diffracted (over-the-top) ray
    c_direct = th.muted  # blocked direct ray
    gy = 430.0  # ground line
    s.ground(gy, 60.0, 840.0)
    s.text(
        66.0, gy + 26.0, "Ground ($G_s$, $G_m$, $G_r$)", 15, th.muted, anchor="start"
    )

    # --- source (loudspeaker) on the left, acoustic centre at (sx, sy) -------
    sx, sy = 150.0, 300.0
    for r in (26, 44, 62):
        s.path(
            f"M {sx + r * 0.22:.1f} {sy - r:.1f} "
            f"A {r} {r} 0 0 1 {sx + r:.1f} {sy - r * 0.22:.1f}",
            stroke=th.muted,
            sw=1.3,
        )
    s.rect(sx - 20, sy - 24, 40, 48, th.panel, th.fg, rx=5, sw=2)
    s.circle(sx, sy - 6, 9, th.fg)
    s.circle(sx, sy - 6, 3.5, th.bg)
    s.circle(sx, sy + 14, 6, th.fg)
    s.line(sx, sy + 24, sx, gy, th.fg, 2.0)  # mast to the ground
    s.text(sx, sy - 74, "Source", 17, th.fg, bold=True)

    # --- barrier in the middle, top edge at (ex, ey) -------------------------
    ex, ey = 450.0, 150.0
    bw = 16.0
    s.rect(ex - bw / 2, ey, bw, gy - ey, th.secondary, th.fg, sw=2)
    s.text(
        ex + 16.0,
        (ey + gy) / 2 + 6.0,
        "Barrier",
        17,
        th.secondary,
        bold=True,
        anchor="start",
    )
    s.circle(ex, ey, 5.5, th.bg, th.fg, 2.0)  # diffraction edge node

    # --- receiver (microphone) on the right, capsule at (rx, ry) -------------
    rx, ry = 770.0, 288.0
    s.mic(rx, ry, gy, 1.0)
    s.text(rx, ry - 18.0, "Receiver", 17, th.fg, bold=True)

    # --- rays ---------------------------------------------------------------
    # Direct (blocked) ray straight through the barrier.
    s.line(sx + 14, sy - 6, rx, ry + 6, c_direct, 1.8, dash="7,6")
    s.text(
        285.0,
        sy + 40.0,
        "direct path (blocked)",
        14,
        c_direct,
        anchor="middle",
        italic=True,
    )
    # Diffracted ray up to the top edge, then down to the receiver.
    # dss/dsr stay uncomposed for now: ISO 9613-2:1996 Eq. (16) prints both
    # subscripts upright, but _ROMAN_SCRIPTS holds "ss" and not "sr", so
    # $d_{ss}$/$d_{sr}$ would set the pair in two different styles.
    s.line(sx + 12, sy - 12, ex, ey, c_diff, 3.0)
    s.arrow(ex, ey, rx, ry + 2, c_diff, 3.0)
    s.text(300.0, 208.0, "dss", 15, c_diff, anchor="middle")
    s.text(610.0, 200.0, "dsr", 15, c_diff, anchor="middle")
    s.text(ex, ey - 22.0, "diffracted path", 15, c_diff, bold=True)

    # --- heights (witness dimensions) ---------------------------------------
    s.dim(sx - 44, gy, sx - 44, sy - 6, "$h_s$", offset=0, label_side="left")
    s.line(sx - 44, gy, sx, gy, th.muted, 0.9, dash="3,3")
    s.line(sx - 44, sy - 6, sx, sy - 6, th.muted, 0.9, dash="3,3")
    s.dim(rx + 40, gy, rx + 40, ry + 6, "$h_r$", offset=0, label_side="right")
    s.line(rx, gy, rx + 40, gy, th.muted, 0.9, dash="3,3")
    s.line(rx, ry + 6, rx + 40, ry + 6, th.muted, 0.9, dash="3,3")

    # --- master relations ---------------------------------------------------
    # Both stay uncomposed for now: the first carries the dss/dsr pair noted
    # above, and Eq. (14) prints "met" upright (ISO 9613-2:1996, p. 9) while
    # _ROMAN_SCRIPTS has no "met", so $K_{met}$ would set it as an index.
    s.text(
        450.0, gy + 58.0, "z = dss + dsr − d   (path difference)", 16, th.fg, bold=True
    )
    s.text(
        450.0,
        gy + 84.0,
        "Dz = 10 log10[ 3 + (C₂/λ) C₃ z Kmet ]   (Eq. 14)",
        15,
        th.muted,
    )


def _d_impulse_prominence(s: SVG, th: Theme) -> None:
    """Impulsive-sound prominence and the LAeq adjustment (NT ACOU 112:2002)."""
    cx = 450.0
    bw, bh = 640.0, 60.0
    x0 = cx - bw / 2

    # --- Input --------------------------------------------------------------
    s.rect(x0, 56, bw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(
        cx,
        82,
        "A-weighted level history  $L_{pAF}$  (time weighting F)",
        16,
        th.fg,
        "middle",
        bold=True,
    )
    s.text(
        cx,
        103,
        "an onset = a stretch where the gradient exceeds 10 dB/s (clauses 4.5-4.7)",
        11,
        th.muted,
        "middle",
    )
    s.arrow(cx, 116, cx, 150, th.fg, 1.8)

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 26, l1, 15, th.fg, "middle", bold=True)
        s.text(cx, y + 47, l2, 11, th.muted, "middle")

    _step(
        150,
        "Per impulse: onset rate OR and level difference LD",
        "OR = onset slope [dB/s],   $LD = L_e − L_s$ [dB]",
        th.primary,
    )
    _step(
        242,
        "Predicted prominence  $P$   (clause 7, Formula 1)",
        "$P = 3·log_{10}(OR) + 2·log_{10}(LD)$;   highest $P$ over 30 min governs",
        th.fg,
    )
    _step(
        334,
        "Adjustment  $K_I$   (clause 8, Formula 2)",
        "$K_I = 1.8·(P − 5)$ dB for $P > 5$, else 0",
        th.secondary,
    )
    s.arrow(cx, 210, cx, 242, th.fg, 1.8)
    s.arrow(cx, 302, cx, 334, th.fg, 1.8)
    s.arrow(cx, 394, cx, 426, th.fg, 1.8)

    # --- Output -------------------------------------------------------------
    # The rating-level formula stays uncomposed for now: the exponent
    # carries the subscripts of LAeq and KI, and the composer sets a single
    # script level (same 10^(L/10) family as the energy sums elsewhere).
    s.rect(x0, 426, bw, 60, "none", th.primary, rx=10, sw=2.4)
    s.text(
        cx,
        452,
        "Rating level  LAr,T = 10·log10( (1/T) Σ Δt·10^((LAeq+KI)/10) )",
        15,
        th.fg,
        "middle",
        bold=True,
    )
    s.text(
        cx,
        473,
        "impulse-adjusted level over the reference time  (Note 1)",
        11,
        th.muted,
        "middle",
    )


# ---------------------------------------------------------------------------
# Wind-turbine noise measurement geometry (IEC 61400-11)
# ---------------------------------------------------------------------------


def _d_wind_turbine(s: SVG, th: Theme) -> None:
    """IEC 61400-11 apparent-sound-power geometry: downwind ground-board
    microphone at R0 = H + D/2, slant distance R1 to the rotor centre and
    the Figure 3 plan-view position pattern.
    """
    import math

    gy = 470.0
    s.ground(gy, 40.0, 668.0)

    # Wind arrow on the upwind side.
    s.arrow(52.0, 108.0, 148.0, 108.0, th.accent, 2.6)
    s.text(100.0, 88.0, "Wind", 15, th.accent, bold=True)

    # --- met mast: anemometer cups + wind vane -----------------------------
    mmx = 108.0
    s.line(mmx, gy, mmx, gy - 96.0, th.fg, 2.2)
    s.line(mmx - 12, gy - 96.0, mmx + 12, gy - 96.0, th.fg, 1.8)
    s.circle(mmx - 12, gy - 102.0, 4.5, th.panel, th.fg, 1.4)  # cup
    s.circle(mmx + 12, gy - 102.0, 4.5, th.panel, th.fg, 1.4)  # cup
    s.line(mmx, gy - 82.0, mmx + 20, gy - 82.0, th.fg, 1.6)  # vane arm
    s.path(
        f"M {mmx + 20:.0f} {gy - 87:.0f} L {mmx + 34:.0f} {gy - 82:.0f} "
        f"L {mmx + 20:.0f} {gy - 77:.0f} Z",
        fill=th.secondary,
    )
    s.text(mmx, gy - 118.0, "Met mast", 14, th.fg, bold=True)
    s.text(mmx, gy + 30.0, "wind speed + direction", 12, th.muted)

    # --- turbine: tower, nacelle and the rotor edge-on ----------------------
    tx = 262.0  # tower vertical centreline
    hub_y = 168.0  # rotor centre => H = 302 px
    rr = 104.0  # rotor radius (D = 208 px)
    s.path(
        f"M {tx - 10:.0f} {gy:.0f} L {tx - 5:.0f} {hub_y + 12:.0f} "
        f"L {tx + 5:.0f} {hub_y + 12:.0f} L {tx + 10:.0f} {gy:.0f} Z",
        fill=th.panel,
        stroke=th.fg,
        sw=1.8,
    )
    s.rect(tx - 28, hub_y - 12, 56, 24, th.panel, th.fg, rx=6, sw=1.8)
    rx_ = tx - 34.0  # rotor plane (upwind of the tower)
    s.ellipse(rx_, hub_y, 9.0, rr, stroke=th.muted, sw=1.3, dash="6,5")
    s.line(rx_, hub_y - 8, rx_ - 4, hub_y - rr + 4, th.fg, 3.2)  # blade up
    s.line(rx_, hub_y + 8, rx_ + 3, hub_y + rr - 4, th.fg, 3.2)  # blade down
    s.circle(rx_, hub_y, 6.5, th.fg)
    s.text(tx + 36, hub_y - 26, "rotor centre", 13, th.fg, anchor="start")
    s.line(rx_ + 8, hub_y - 6, tx + 33, hub_y - 21, th.muted, 0.9)

    # Rotor diameter D across the swept ellipse.
    dx_ = rx_ - 58.0
    s.dim(dx_, hub_y - rr, dx_, hub_y + rr, "$D$", offset=0, label_side="left")
    s.line(rx_ - 8, hub_y - rr, dx_, hub_y - rr, th.muted, 0.9, dash="3,3")
    s.line(rx_ - 8, hub_y + rr, dx_, hub_y + rr, th.muted, 0.9, dash="3,3")

    # Hub height H, downwind of the tower.
    hx_ = tx + 56.0
    s.dim(hx_, gy, hx_, hub_y, "$H$", offset=0, label_side="right")
    s.line(tx + 10, gy, hx_, gy, th.muted, 0.9, dash="3,3")
    s.line(tx + 28, hub_y, hx_, hub_y, th.muted, 0.9, dash="3,3")

    # --- downwind microphone on a ground board ------------------------------
    mx = 640.0
    s.ellipse(mx, gy - 3.0, 36.0, 7.0, fill=th.panel, stroke=th.fg, sw=1.6)
    s.rect(mx - 16, gy - 10.0, 20, 6, th.fg, rx=2.5)  # capsule flat
    s.rect(mx + 4, gy - 11.0, 10, 8, th.primary, rx=2)  # body
    # Under the ground line and to the right of the R0 dimension: set above
    # the board, the label runs back across the hub-height dimension and
    # into the tower itself, and it is the longer of the two languages that
    # gets there first.
    s.text(740.0, gy + 62.0, "Microphone on a ground board", 14, th.fg, bold=True)

    # Slant distance R1 from the rotor centre to the microphone.
    s.line(rx_, hub_y, mx - 12, gy - 8.0, th.primary, 2.2, dash="9,6")
    s.text(430.0, 296.0, "$R_1$", 16, th.primary, bold=True)
    # Board-to-R1 inclination angle (25°..40°).
    ang = math.atan2(gy - 8.0 - hub_y, mx - 12 - rx_)  # slope of R1
    r_arc = 52.0
    axp = mx - 12 - r_arc * math.cos(ang)
    ayp = gy - 8.0 - r_arc * math.sin(ang)
    s.path(
        f"M {mx - 12 - r_arc:.1f} {gy - 8:.1f} "
        f"A {r_arc:.0f} {r_arc:.0f} 0 0 1 {axp:.1f} {ayp:.1f}",
        stroke=th.muted,
        sw=1.3,
    )
    s.text(mx - 74, gy - 22.0, "$φ$", 15, th.muted)

    # Horizontal reference distance R0.
    s.dim(tx, gy, mx, gy, "$R_0 = H + D/2$", offset=40)

    # --- plan-view inset: the Figure 3 position pattern ---------------------
    pcx, pcy, pr = 794.0, 218.0, 76.0
    s.text(pcx, 104.0, "Plan view (Figure 3)", 15, th.fg, bold=True)
    s.arrow(pcx - 60.0, 118.0, pcx - 60.0, 150.0, th.accent, 2.2)  # wind, from top
    s.circle(pcx, pcy, pr, "none", th.muted, 1.2)
    s.line(pcx - pr - 8, pcy, pcx + pr + 8, pcy, th.muted, 1.0, dash="4,4")
    s.line(pcx, pcy - pr - 8, pcx, pcy + pr + 8, th.muted, 1.0, dash="4,4")
    s.line(pcx - 16, pcy, pcx + 16, pcy, th.fg, 3.0)  # rotor, plan
    s.circle(pcx, pcy, 4.0, th.fg)
    # Reference position 1, downwind (diamond).
    p1x, p1y = pcx, pcy + pr
    s.path(
        f"M {p1x:.0f} {p1y - 7:.0f} L {p1x + 7:.0f} {p1y:.0f} "
        f"L {p1x:.0f} {p1y + 7:.0f} L {p1x - 7:.0f} {p1y:.0f} Z",
        fill=th.secondary,
    )
    s.text(p1x + 13, p1y + 5, "1", 13, th.secondary, anchor="start", bold=True)
    # Optional positions 2 and 4 at ±60° from downwind, 3 upwind.
    for lbl, adeg, lx, ly, anch in (
        ("2", 150.0, -12.0, 4.0, "end"),
        ("3", 270.0, 12.0, 4.0, "start"),
        ("4", 30.0, 12.0, 4.0, "start"),
    ):
        pxx = pcx + pr * math.cos(math.radians(adeg))
        pyy = pcy + pr * math.sin(math.radians(adeg))
        s.circle(pxx, pyy, 5.5, th.bg, th.fg, 1.6)
        s.text(pxx + lx, pyy + ly, lbl, 13, th.muted, anchor=anch)
        s.line(pcx, pcy, pxx, pyy, th.muted, 0.9, dash="3,4")
    s.line(pcx, pcy, p1x, p1y, th.muted, 0.9, dash="3,4")
    s.text(pcx - 34, pcy + 52.0, "60°", 11, th.muted)
    s.text(pcx + 34, pcy + 52.0, "60°", 11, th.muted)
    s.text(pcx, pcy + pr + 30.0, "reference 1 (downwind)", 12, th.secondary)
    s.text(pcx, pcy + pr + 50.0, "optional positions 2–4", 12, th.muted)

    # --- governing relations -------------------------------------------------
    s.text(
        450.0,
        560.0,
        "$R_1 = √(H^2 + R_0^2)$   slant distance, rotor centre → microphone",
        16,
        th.fg,
        bold=True,
    )
    s.text(
        450.0,
        588.0,
        "$L_{WA,i} = L_{p,i} − 6 + 10 log_{10}(4π R_1^2 / S_0)$   (Formula 26, $S_0$ = 1 m²)",
        16,
        th.primary,
        bold=True,
    )
    s.text(
        450.0,
        614.0,
        "the −6 dB removes the board's pressure doubling; board-to-$R_1$ angle $φ$ = 25°–40°",
        14,
        th.muted,
    )


# ---------------------------------------------------------------------------
# Ground reflection: direct ray, image source, path difference
# ---------------------------------------------------------------------------


def _d_ground_reflection(s: SVG, th: Theme) -> None:
    """Two-path ground interference: source, receiver, direct ray, the
    specular reflection unfolded through the image source, and the path
    difference that sets the interference phase (ISO 9613-2 ground effect,
    Chien-Soroka geometry).
    """
    gy = 372.0
    sx, sy = 170.0, 232.0  # source (hs = 140 px)
    rx, ry = 700.0, 282.0  # receiver capsule tip (hr = 90 px)
    ix, iy = sx, gy + (gy - sy)  # image source, mirrored below the ground
    # Specular point: equal angles, found by unfolding through the image.
    bx = sx + (rx - sx) * (gy - sy) / ((gy - sy) + (gy - ry))

    s.ground(gy, 60.0, 840.0)

    # Source: point with radiating arcs.
    for r in (22, 38, 54):
        s.path(
            f"M {sx + r * 0.30:.1f} {sy - r * 0.95:.1f} "
            f"A {r} {r} 0 0 1 {sx + r * 0.95:.1f} {sy - r * 0.30:.1f}",
            stroke=th.muted,
            sw=1.3,
        )
    s.circle(sx, sy, 8.0, th.fg)
    s.text(sx, sy - 66.0, "Source", 17, th.fg, bold=True)
    s.text(sx - 14, sy + 24, "$S$", 13, th.fg, anchor="end")
    s.line(sx, sy + 8, sx, gy, th.fg, 2.0)

    # Receiver: measurement microphone.
    s.mic(rx, ry, gy, 1.0)
    s.text(rx, ry - 18.0, "Receiver", 17, th.fg, bold=True)
    s.text(rx - 18, ry + 10.0, "$R$", 13, th.fg, anchor="end")

    # Direct ray r1.
    s.arrow(sx + 10, sy, rx - 8, ry - 2, th.primary, 2.6)
    s.text(430.0, 236.0, "direct ray  $r_1$", 15, th.primary, bold=True)

    # Reflected ray via the specular point (equal angles).
    s.line(sx + 6, sy + 7, bx, gy, th.accent, 2.6)
    s.arrow(bx, gy, rx - 6, ry + 4, th.accent, 2.6)
    s.text(330.0, gy - 34.0, "reflected ray", 15, th.accent, bold=True)
    # Equal grazing angles at the bounce.
    s.path(
        f"M {bx - 34:.1f} {gy:.1f} A 34 34 0 0 1 {bx - 26:.1f} {gy - 21:.1f}",
        stroke=th.muted,
        sw=1.2,
    )
    s.path(
        f"M {bx + 34:.1f} {gy:.1f} A 34 34 0 0 0 {bx + 26:.1f} {gy - 21:.1f}",
        stroke=th.muted,
        sw=1.2,
    )
    s.text(bx, gy - 40.0, "equal angles", 12, th.muted)

    # Image source: ghosted mirror of the source below the ground.
    s.circle(ix, iy, 8.0, "none", th.secondary, 1.8)
    s.line(sx, gy, ix, iy - 8, th.secondary, 1.2, dash="4,4")
    s.text(ix + 18, iy + 5, "image source", 14, th.secondary, anchor="start")
    s.text(ix - 16, iy + 5, "$S′$", 13, th.secondary, anchor="end")
    # The unfolded path S' -> R is straight through the bounce point: r2.
    s.line(ix, iy - 6, bx, gy, th.secondary, 1.6, dash="7,5")
    s.line(bx, gy, rx - 6, ry + 4, th.secondary, 1.2, dash="2,6")
    s.text(380.0, iy - 62.0, "$r_2 = |S′R|$", 14, th.secondary)

    # Heights.
    s.dim(sx - 46, gy, sx - 46, sy, "$h_s$", offset=0, label_side="left")
    s.line(sx - 46, gy, sx - 8, gy, th.muted, 0.9, dash="3,3")
    s.line(sx - 46, sy, sx - 8, sy, th.muted, 0.9, dash="3,3")
    s.dim(rx + 42, gy, rx + 42, ry, "$h_r$", offset=0, label_side="right")
    s.line(rx + 8, gy, rx + 42, gy, th.muted, 0.9, dash="3,3")
    s.line(rx + 8, ry, rx + 42, ry, th.muted, 0.9, dash="3,3")

    # Governing relations (top block, clear of the geometry).
    s.text(560.0, 88.0, "path difference  $δ = r_2 − r_1$", 17, th.fg, bold=True)
    s.text(560.0, 114.0, "phase difference  $Δφ = 2π δ / λ$  (+ $arg Q$)", 15, th.fg)
    # The pressure sum stays uncomposed for now: each exponent carries
    # the subscripts of r1 and r2, and the composer sets a single script
    # level (nested-script family).
    s.text(
        560.0,
        142.0,
        "p ∝ e^(jkr1)/r1 + Q · e^(jkr2)/r2   (Q = ground reflection coefficient)",
        14,
        th.muted,
    )
    s.text(
        560.0,
        168.0,
        "in phase ($δ ≈ nλ$): up to +6 dB    ·    out of phase ($δ ≈ λ/2$ on hard ground): a deep dip",
        13,
        th.muted,
    )


# Atmospheric refraction: downwind multipath and the upwind shadow zone
# ---------------------------------------------------------------------------


def _d_atmospheric_refraction(s: SVG, th: Theme) -> None:
    """Refracting surface layer (Salomons 2001; Attenborough & Van
    Renterghem 2021, Ch. 11): wind profile arrows, an effective-sound-speed
    inset, downward-curved rays with a ground bounce on the downwind side
    and upward-curved rays opening an acoustic shadow on the upwind side.
    Horizontal scale about 1 px per metre; heights exaggerated.
    """
    gy = 452.0
    sx, sy = 450.0, gy - 56.0  # source, hs = 2 m (schematic)
    mlx, mrx = 90.0, 795.0  # receivers, 350 m to each side

    # --- upwind side (left): rays curve upward -----------------------------
    # Limiting ray: grazes the ground at ~220 m upwind, then climbs; the
    # region under it is the acoustic shadow.
    s.path(
        f"M {sx:.0f} {sy:.0f} C 380 430 300 452 232 452", stroke=th.secondary, sw=1.8
    )
    shadow = "M 232 452 C 150 452 90 420 60 340 L 60 452 Z"
    s.path(shadow, fill=th.panel)
    s.path("M 232 452 C 150 452 90 420 60 340", stroke=th.secondary, sw=1.4, dash="6,5")
    # Fan of upwind rays, all refracted upward.
    s.path(
        f"M {sx:.0f} {sy:.0f} C 380 390 310 372 250 340", stroke=th.secondary, sw=1.8
    )
    s.arrow(250.0, 340.0, 214.0, 316.0, th.secondary, 1.8)
    s.path(
        f"M {sx:.0f} {sy:.0f} C 400 375 340 330 290 270", stroke=th.secondary, sw=1.8
    )
    s.arrow(290.0, 270.0, 264.0, 234.0, th.secondary, 1.8)
    s.path(
        f"M {sx:.0f} {sy:.0f} C 415 345 385 270 355 195", stroke=th.secondary, sw=1.8
    )
    s.arrow(355.0, 195.0, 340.0, 156.0, th.secondary, 1.8)
    s.text(
        188.0, 414.0, "acoustic shadow", 13, th.secondary, italic=True, anchor="start"
    )
    s.line(184.0, 410.0, 105.0, 395.0, th.muted, 1.0)
    # Shadow-boundary marker at the grazing point.
    s.line(232.0, 452.0, 232.0, 386.0, th.muted, 1.1, dash="4,4")
    s.text(232.0, 370.0, "≈ 220 m", 12, th.muted)

    # --- downwind side (right): rays curve down, ground bounce -------------
    s.path(f"M {sx:.0f} {sy:.0f} Q 620 366 786 402", stroke=th.primary, sw=2.0)
    s.arrow(770.0, 399.0, 788.0, 403.0, th.primary, 2.0)
    s.path(f"M {sx:.0f} {sy:.0f} Q 590 300 726 452", stroke=th.accent, sw=2.0)
    s.path("M 726 452 Q 758 420 782 408", stroke=th.accent, sw=2.0)
    s.arrow(766.0, 415.0, 784.0, 407.0, th.accent, 2.0)

    # --- scene: ground, source, receivers ----------------------------------
    s.ground(gy, 40.0, 860.0)
    for r in (18, 30, 42):
        s.path(
            f"M {sx - r:.0f} {sy:.0f} A {r} {r} 0 0 1 {sx + r:.0f} {sy:.0f}",
            stroke=th.muted,
            sw=1.2,
        )
    s.circle(sx, sy, 7.0, th.fg)
    s.line(sx, sy + 7, sx, gy, th.fg, 2.0)
    s.text(sx, sy - 54.0, "Source", 15, th.fg, bold=True)
    s.dim(sx + 34, gy, sx + 34, sy, "2 m", offset=0, size=13, label_side="right")
    s.line(sx + 7, sy, sx + 34, sy, th.muted, 0.9, dash="3,3")
    s.mic(mlx, gy - 42.0, gy, 0.85)
    s.mic(mrx, gy - 46.0, gy, 1.0)
    s.text(mrx, gy - 66.0, "Receiver", 15, th.fg, bold=True)
    s.dim(
        mrx + 38,
        gy,
        mrx + 38,
        gy - 46.0,
        "1.5 m",
        offset=0,
        size=12,
        label_side="right",
    )
    s.line(mrx + 8, gy - 46.0, mrx + 38, gy - 46.0, th.muted, 0.9, dash="3,3")
    s.dim(mlx, gy, sx, gy, "350 m", offset=36, size=14)
    s.dim(sx, gy, mrx, gy, "350 m", offset=36, size=14)

    # --- wind profile arrows (blowing left to right) -----------------------
    for wy, wl in ((84.0, 116.0), (114.0, 82.0), (144.0, 52.0)):
        s.arrow(500.0, wy, 500.0 + wl, wy, th.accent, 2.2)
    s.text(548.0, 66.0, "wind $u(z)$", 14, th.accent, bold=True)

    # --- inset: effective-sound-speed profiles -----------------------------
    s.text(143.0, 52.0, "$c_{eff}(z) = c(z) + u(z)$", 13, th.fg, bold=True)
    s.rect(58, 64, 170, 170, th.panel, th.fg, rx=8, sw=1.5)
    s.arrow(76.0, 214.0, 76.0, 88.0, th.muted, 1.3)
    s.text(76.0, 80.0, "$z$", 11, th.muted)
    s.line(76.0, 214.0, 214.0, 214.0, th.muted, 1.3)
    s.path("M 143 214 Q 146 150 192 96", stroke=th.primary, sw=2.2)
    s.path("M 143 214 Q 140 150 100 96", stroke=th.secondary, sw=2.2)
    s.text(197.0, 92.0, "$+u$", 11, th.primary, anchor="start")
    s.text(96.0, 92.0, "$−u$", 11, th.secondary, anchor="end")
    s.text(143.0, 229.0, "340 m/s", 10, th.muted, mono=True)

    # --- physics captions --------------------------------------------------
    s.text(
        80.0,
        540.0,
        "Upwind: rays bend up; beyond ≈ 220 m a ground shadow opens and "
        "the level collapses over 20 dB",
        14,
        th.fg,
        anchor="start",
    )
    s.text(
        80.0,
        566.0,
        "Downwind: rays bend down; the receiver hears the direct and the ground-bounced arrival (multipath)",
        14,
        th.fg,
        anchor="start",
    )
    s.text(
        80.0,
        592.0,
        "a ±0.1 (m/s)/m gradient curves rays with $R_c = c_0/|g|$ ≈ "
        "3.4 km; source $h_s$ = 2 m, receiver $h_r$ = 1.5 m",
        14,
        th.muted,
        anchor="start",
    )


# ---------------------------------------------------------------------------
# Barrier diffraction over ground (Fresnel number, Kurze-Anderson)
# ---------------------------------------------------------------------------


def _d_ground_barrier(s: SVG, th: Theme) -> None:
    """The guide's barrier geometry: a 1 m source, a 4 m thin screen at
    50 m and a 1.5 m receiver at 100 m. The diffracted segments A and B,
    the blocked direct path d and the barrier_insertion_loss values
    (N = 0.44 and 10.0 dB at 500 Hz, 15.5 dB at 2 kHz).
    """
    gy = 340.0  # ground; 7 px/m horizontal, 60 px/m vertical
    sx, sy = 110.0, gy - 60.0  # source, hs = 1 m
    ex, ey = 460.0, gy - 240.0  # barrier top edge, 4 m
    rx, ry = 810.0, gy - 90.0  # receiver, hr = 1.5 m

    s.ground(gy, 40.0, 860.0)

    # Source loudspeaker on its mast.
    for r in (22, 38, 54):
        s.path(
            f"M {sx + r * 0.22:.1f} {sy - r:.1f} "
            f"A {r} {r} 0 0 1 {sx + r:.1f} {sy - r * 0.22:.1f}",
            stroke=th.muted,
            sw=1.2,
        )
    s.rect(sx - 17, sy - 20, 34, 40, th.panel, th.fg, rx=5, sw=2)
    s.circle(sx, sy - 5, 8, th.fg)
    s.circle(sx, sy - 5, 3, th.bg)
    s.circle(sx, sy + 12, 5, th.fg)
    s.line(sx, sy + 20, sx, gy, th.fg, 2.0)
    s.text(sx, sy - 66, "Source", 16, th.fg, bold=True)

    # Thin screen with its diffraction edge.
    s.rect(ex - 8, ey, 16, gy - ey, th.secondary, th.fg, sw=2)
    s.text(ex + 18, ey + 74, "Barrier", 16, th.secondary, bold=True, anchor="start")

    # Receiver microphone.
    s.mic(rx, ry, gy, 1.0)
    s.text(rx, ry - 20, "Receiver", 16, th.fg, bold=True)

    # Blocked direct path and the diffracted path A + B.
    s.line(sx + 12, sy - 4, rx - 4, ry + 4, th.muted, 1.6, dash="7,6")
    s.text(268, 308, "direct $d$ = 100.00 m (blocked)", 13, th.muted)
    s.line(sx + 10, sy - 10, ex, ey, th.accent, 2.6)
    s.arrow(ex, ey, rx - 4, ry - 2, th.accent, 2.6)
    s.circle(ex, ey, 5.5, th.bg, th.fg, 2.0)
    s.text(258, 162, "$A$ = 50.09 m", 15, th.accent, bold=True)
    s.text(660, 152, "$B$ = 50.06 m", 15, th.accent, bold=True)

    # Height and distance dimensions.
    s.dim(64, gy, 64, sy - 4, "1.0 m", offset=0, size=13, label_side="left")
    s.line(64, sy - 4, sx - 17, sy - 4, th.muted, 0.9, dash="3,3")
    s.dim(430, gy, 430, ey, "4.0 m", offset=0, size=13, label_side="left")
    s.line(430, ey, ex - 8, ey, th.muted, 0.9, dash="3,3")
    s.dim(846, gy, 846, ry, "1.5 m", offset=0, size=12, label_side="right")
    s.line(rx + 6, ry, 846, ry, th.muted, 0.9, dash="3,3")
    s.dim(sx, gy, ex, gy, "50 m", offset=34, size=14)
    s.dim(ex, gy, rx, gy, "50 m", offset=34, size=14)

    # --- captions ----------------------------------------------------------
    s.text(
        80,
        420,
        "path difference $δ = A + B − d$ = 0.15 m; Fresnel number $N = 2δ/λ$ = 0.44 at 500 Hz",
        15,
        th.fg,
        anchor="start",
    )
    # The Kurze-Anderson line stays uncomposed for now: ISO 9613-2:1996
    # prints the "bar" subscript upright (Eq. 12, p. 9) and
    # _ROMAN_SCRIPTS has no "bar", so $Δ_{bar}$ would set it as an index.
    s.text(
        80,
        448,
        "Kurze–Anderson: Δbar = 5 + 20 log10( √(2πN) / tanh √(2πN) ) = 10.0 dB, 500 Hz",
        15,
        th.primary,
        anchor="start",
        bold=True,
    )
    s.text(
        80,
        476,
        "$N$ grows with frequency: the same screen gives 15.5 dB at 2 kHz (vertical scale exaggerated)",
        15,
        th.muted,
        anchor="start",
    )


# ---------------------------------------------------------------------------
# ISO 9613-2 ground regions and the ground factor G (7.3.1, Table 3 note 2)
# ---------------------------------------------------------------------------


def _d_ground_regions(s: SVG, th: Theme) -> None:
    gy = 250.0
    x0, x1 = 70.0, 830.0  # 0 m and 200 m of the path
    scale = (x1 - x0) / 200.0  # px per metre

    # The three regions, shaded along the ground.
    src_end = x0 + 45.0 * scale
    rec_start = x1 - 45.0 * scale
    band = 34.0
    s.rect(x0, gy - band, src_end - x0, band, th.panel, th.primary, sw=2.0)
    s.rect(src_end, gy - band, rec_start - src_end, band, th.panel, th.accent, sw=2.0)
    s.rect(rec_start, gy - band, x1 - rec_start, band, th.panel, th.primary, sw=2.0)

    # Mixed cover inside the middle region: 70 m grass then 40 m asphalt.
    grass_end = src_end + 70.0 * scale
    s.rect(
        grass_end, gy - band, rec_start - grass_end, band, th.muted, th.accent, sw=2.0
    )
    s.text((src_end + grass_end) / 2, gy - band - 12, "grass, $G = 1$", 15, th.fg)
    s.text((grass_end + rec_start) / 2, gy - band - 12, "asphalt, $G = 0$", 15, th.fg)

    s.ground(gy, 50.0, 850.0)

    # Source and receiver, both 1,5 m high (drawn out of scale for legibility).
    hs = 86.0
    s.circle(x0, gy - hs, 9, th.secondary)
    s.line(x0, gy - hs, x0, gy, th.fg, 2.0)
    s.text(x0 + 12, gy - hs - 40, "Source", 17, th.fg, anchor="start", bold=True)
    s.text(x0 + 12, gy - hs - 18, "$h_s$ = 1,5 m", 15, th.muted, anchor="start")
    s.mic(x1, gy - hs, gy, 1.0)
    s.text(x1 - 12, gy - hs - 40, "Receiver", 17, th.fg, anchor="end", bold=True)
    s.text(x1 - 12, gy - hs - 18, "$h_r$ = 1,5 m", 15, th.muted, anchor="end")
    s.line(x0, gy - hs, x1, gy - hs, th.muted, 1.4, dash="7,5")

    # Region dimensions along the ground.
    s.dim(x0, gy + 44, src_end, gy + 44, "source region  $30 h_s$ = 45 m", 0, 15)
    s.dim(src_end, gy + 80, rec_start, gy + 80, "middle region  110 m", 0, 15)
    s.dim(rec_start, gy + 44, x1, gy + 44, "receiver region  $30 h_r$ = 45 m", 0, 15)
    for xv in (x0, src_end, rec_start, x1):
        s.line(xv, gy + 28, xv, gy + 92, th.muted, 0.9, dash="3,3")
    s.dim(x0, gy + 118, x1, gy + 118, "$d_p$ = 200 m", 0, 16)

    # Inset: the same path at 60 m, regions overlapping, no middle region.
    iy = gy + 190.0
    ix0, ix1 = 260.0, 640.0
    s.rect(ix0, iy - 26, ix1 - ix0, 26, th.panel, th.primary, sw=2.0)
    s.line(ix0 - 40, iy, ix1 + 40, iy, th.fg, 1.8)
    s.circle(ix0, iy - 46, 6, th.secondary)
    s.circle(ix1, iy - 46, 6, th.primary)
    s.dim(
        ix0,
        iy + 48,
        ix1,
        iy + 48,
        "$d_p$ = 60 m: the regions overlap, so there is no middle region",
        0,
        15,
    )

    # The two reading boxes, on one row under the drawing.
    by = 500.0
    s.rect(70.0, by, 380, 100, th.panel, th.muted, rx=8, sw=1.2)
    s.text(
        84.0,
        by + 28,
        "$G$ is the porous fraction of its region",
        16,
        th.fg,
        anchor="start",
        bold=True,
    )
    s.text(84.0, by + 54, "$G_s = 25/45$ = 0,55", 15, th.fg, anchor="start")
    s.text(
        84.0, by + 78, "$G_m = 70/110$ = 0,64   $G_r$ = 1,00", 15, th.fg, anchor="start"
    )

    s.rect(470.0, by, 380, 100, th.panel, th.muted, rx=8, sw=1.2)
    s.text(
        484.0, by + 28, "$q = 1 − 30(h_s + h_r)/d_p$ = 0,55", 16, th.fg, anchor="start"
    )
    s.text(
        484.0,
        by + 54,
        "below $30(h_s + h_r)$ = 90 m: $q = 0$, $A_m = 0$",
        15,
        th.secondary,
        anchor="start",
    )
    s.text(
        484.0,
        by + 78,
        "and ground_middle is ignored entirely",
        15,
        th.fg,
        anchor="start",
    )


# ---------------------------------------------------------------------------
# The four diffracted paths of a barrier on finite-impedance ground
# ---------------------------------------------------------------------------


def _d_barrier_four_paths(s: SVG, th: Theme) -> None:
    gy = 260.0
    x0, x1 = 110.0, 800.0
    sx, sy = x0, gy - 60.0  # source, 1 m high (out of scale)
    rx, ry = x1, gy - 90.0  # receiver, 1,5 m high
    ex = x0 + (x1 - x0) * 0.5  # barrier at 50 m of 100 m
    ey = gy - 200.0  # 4 m screen (out of scale)

    s.ground(gy, 60.0, 850.0)
    # Mirror images below the ground plane.
    sy_i = gy + (gy - sy)
    ry_i = gy + (gy - ry)

    # The screen.
    s.rect(ex - 7, ey, 14, gy - ey, th.panel, th.fg, sw=2)
    s.text(ex + 14, gy - 16, "4 m screen", 16, th.fg, anchor="start", bold=True)

    # Four routes, each one a source (or its image) over the edge to the
    # receiver (or its image).
    routes = (
        (sy, ry, th.primary, "0 bounces", "none"),
        (sy_i, ry, th.secondary, "1 bounce, source side", "6,4"),
        (sy, ry_i, th.accent, "1 bounce, receiver side", "2,4"),
        (sy_i, ry_i, "#9467bd", "2 bounces", "10,5"),
    )
    # Fanned by a few pixels at the edge so four coincident legs stay legible.
    for i, (y_src, y_rec, color, _label, dash) in enumerate(routes):
        eyy = ey + (i - 1.5) * 5.0
        s.path(
            f"M {sx:.1f} {y_src:.1f} L {ex:.1f} {eyy:.1f} L {rx:.1f} {y_rec:.1f}",
            stroke=color,
            sw=2.0,
            dash="" if dash == "none" else dash,
        )

    # Q at each ground reflection (the specular points of the image paths).
    for qx in (sx + (ex - sx) * 0.5, ex + (rx - ex) * 0.5):
        s.circle(qx, gy, 7, th.bg, th.fg, sw=1.8)
        s.text(qx, gy - 12, "$Q$", 15, th.fg, bold=True)

    # Source, receiver and their images.
    s.circle(sx, sy, 9, th.secondary)
    s.text(sx, sy - 20, "Source", 16, th.fg, bold=True)
    s.mic(rx, ry, gy, 0.9)
    s.text(rx, ry - 20, "Receiver", 16, th.fg, bold=True)
    s.circle(sx, sy_i, 8, th.secondary, th.bg, sw=1.4)
    s.text(sx, sy_i + 26, "image source", 15, th.muted)
    s.circle(rx, ry_i, 8, th.primary, th.bg, sw=1.4)
    s.text(rx, ry_i + 26, "image receiver", 15, th.muted)
    s.line(sx, sy, sx, sy_i, th.muted, 1.0, dash="3,3")
    s.line(rx, ry, rx, ry_i, th.muted, 1.0, dash="3,3")

    # Legend of the four routes.
    lx, ly = 96.0, 430.0
    s.rect(lx, ly, 430, 128, th.panel, th.muted, rx=8, sw=1.2)
    s.text(
        lx + 14,
        ly + 26,
        "Four coherent routes over one edge",
        16,
        th.fg,
        anchor="start",
        bold=True,
    )
    for i, (_, _, color, label, dash) in enumerate(routes):
        yy = ly + 52 + i * 24
        s.line(
            lx + 16,
            yy - 5,
            lx + 56,
            yy - 5,
            color,
            2.4,
            dash="" if dash == "none" else dash,
        )
        s.text(lx + 66, yy, label, 15, th.fg, anchor="start")

    # Inset: the two-edge (thick) case.
    tx, ty = 580.0, 430.0
    s.rect(tx, ty, 250, 128, th.panel, th.muted, rx=8, sw=1.2)
    s.text(
        tx + 14,
        ty + 26,
        "Thick barrier: two edges",
        16,
        th.fg,
        anchor="start",
        bold=True,
    )
    bx0, bx1, by0 = tx + 60, tx + 130, ty + 96
    s.rect(bx0, ty + 56, bx1 - bx0, by0 - ty - 56, th.bg, th.fg, sw=1.6)
    s.path(
        f"M {tx + 20} {ty + 84} L {bx0} {ty + 56} L {bx1} {ty + 56} "
        f"L {tx + 226} {ty + 88}",
        stroke=th.primary,
        sw=2.0,
    )
    s.dim(bx0, ty + 46, bx1, ty + 46, "$e$", 0, 15)


# ---------------------------------------------------------------------------
# CNOSSOS-EU road source line geometry (2.2)
# ---------------------------------------------------------------------------


def _d_cnossos_road(s: SVG, th: Theme) -> None:
    # --- plan ------------------------------------------------------------
    py = 150.0
    rx0, rx1 = 70.0, 700.0
    lane = 44.0
    # The heading takes the band right under the plate title and the
    # 60 m dimension takes the one under it: the Spanish heading runs to
    # x = 409, past the source the dimension starts from, so the two
    # cannot share a line.
    s.text(
        rx0,
        56.0,
        "Plan — two-lane urban arterial",
        17,
        th.fg,
        anchor="start",
        bold=True,
    )
    s.rect(rx0, py - lane, rx1 - rx0, 2 * lane, th.panel, th.muted, sw=1.2)
    s.line(rx0, py, rx1, py, th.muted, 1.6, dash="14,10")

    # A source line on each lane centre.
    for cy in (py - lane / 2, py + lane / 2):
        s.line(rx0, cy, rx1, cy, th.secondary, 2.2, dash="9,6")
    s.text(
        rx1 + 10, py - lane / 2 + 6, "source line,", 15, th.secondary, anchor="start"
    )
    s.text(
        rx1 + 10,
        py + lane / 2 + 6,
        "one per lane centre",
        15,
        th.secondary,
        anchor="start",
    )

    # dL = 20 m segments on the near lane, one with its point source.
    seg = (rx1 - rx0) / 12.0
    cy = py + lane / 2
    for i in range(13):
        s.line(rx0 + i * seg, cy - 9, rx0 + i * seg, cy + 9, th.muted, 1.2)
    hx = rx0 + 5.5 * seg
    s.circle(hx, cy, 8, th.secondary)
    s.dim(rx0 + 5 * seg, cy + 30, rx0 + 6 * seg, cy + 30, "dL = 20 m", 0, 15)
    # The per-segment level stays uncomposed for now: Directive (EU)
    # 2015/996 Eq. (2.2.1) prints the whole subscript chain (W',eq,line,
    # i,m) in italic while _ROMAN_SCRIPTS pins "eq" upright, so the
    # composer can reproduce neither the source nor the house style.
    s.text(
        rx0,
        cy + 62,
        "each segment carries L'W,eq,line,i + 10 lg(dL)",
        15,
        th.fg,
        anchor="start",
        mono=False,
    )

    # The junction and its taper.
    jx = rx0 + 9.0 * seg
    s.line(jx, py - lane - 20, jx, py + lane + 20, th.primary, 2.4)
    s.text(
        jx + 8,
        py - lane - 26,
        "signal-controlled junction",
        15,
        th.primary,
        anchor="start",
    )
    s.dim(hx, py - lane - 23, jx, py - lane - 23, "$x$ = 60 m", 0, 15)
    s.line(hx, py - lane - 29, hx, py - lane - 4, th.muted, 0.9, dash="3,3")

    # A small inline graph of the taper, clear of the road.
    gx0, gx1, gy0 = 720.0, 850.0, 300.0
    s.rect(gx0 - 16, gy0 - 74, (gx1 - gx0) + 46, 106, th.panel, th.muted, rx=6, sw=1.1)
    s.line(gx0, gy0, gx1, gy0, th.muted, 1.4)
    s.line(gx0, gy0, gx0, gy0 - 48, th.muted, 1.4)
    s.path(f"M {gx0} {gy0 - 48} L {gx1} {gy0}", stroke=th.primary, sw=2.0)
    s.text(gx0 - 12, gy0 - 56, "$max(1 − |x|/100, 0)$", 13, th.fg, anchor="start")
    s.text(gx1, gy0 + 22, "100 m", 13, th.muted)

    # The receiving facade, in plan.
    fy = py + lane + 118.0
    s.rect(220.0, fy, 300.0, 30, th.panel, th.fg, sw=1.8)
    s.text(370.0, fy + 22, "dwelling façade", 15, th.fg)
    s.dim(560.0, cy, 560.0, fy, "25 m", 0, 15, label_side="right")
    s.line(520.0, fy + 15, 560.0, fy + 15, th.muted, 0.9, dash="3,3")
    s.line(rx0 + 6 * seg, cy, 560.0, cy, th.muted, 0.9, dash="3,3")

    # --- section ---------------------------------------------------------
    sy = 500.0
    s.text(70.0, 392.0, "Section", 17, th.fg, anchor="start", bold=True)
    s.ground(sy, 60.0, 860.0)
    s.rect(120.0, sy - 8, 300.0, 8, th.muted)
    s.text(240.0, sy + 34, "road surface", 15, th.muted)
    s.circle(300.0, sy - 22, 8, th.secondary)
    s.dim(300.0, sy, 300.0, sy - 22, "0,05 m", 40, 15, label_side="right")
    s.text(300.0, sy - 44, "equivalent point source", 15, th.fg, bold=True)

    # The gradient, with the two halves of a bidirectional flow.
    s.path(f"M 120 {sy - 6} L 420 {sy - 40}", stroke=th.primary, sw=2.2)
    s.arrow(150.0, sy - 66, 250.0, sy - 78, th.primary, 1.8)
    s.arrow(390.0, sy - 90, 290.0, sy - 78, th.secondary, 1.8)
    s.text(430.0, sy - 118, "gradient $s$: the flow is split", 15, th.primary)
    s.text(430.0, sy - 98, "and corrected uphill and downhill", 15, th.primary)

    rxr = 700.0
    s.rect(rxr - 40, sy - 150, 120, 150, th.panel, th.fg, sw=1.8)
    s.mic(rxr - 90, sy - 120, sy, 0.9)
    s.dim(rxr - 90, sy, rxr - 90, sy - 120, "4 m", -46, 15)
    s.text(rxr - 46, sy - 162, "receiver point", 15, th.fg, anchor="start", bold=True)


# ---------------------------------------------------------------------------
# CNOSSOS-EU railway source lines and directivity angles (2.3)
# ---------------------------------------------------------------------------


def _d_cnossos_rail(s: SVG, th: Theme) -> None:
    import math

    # --- section ---------------------------------------------------------
    gy = 400.0
    s.text(70.0, 74.0, "Section", 17, th.fg, anchor="start", bold=True)
    s.ground(gy, 190.0, 470.0)
    cx = 330.0  # track centre
    gauge = 92.0  # 1,435 m, out of scale
    s.path(
        f"M {cx - 130} {gy} L {cx - 96} {gy - 30} L {cx + 96} {gy - 30} "
        f"L {cx + 130} {gy} Z",
        fill=th.panel,
        stroke=th.muted,
        sw=1.4,
    )
    s.rect(cx - 86, gy - 42, 172, 12, th.muted)
    for dx in (-gauge / 2, gauge / 2):
        s.rect(cx + dx - 7, gy - 62, 14, 20, th.fg)
    datum = gy - 62.0
    s.line(cx - 120, datum, cx + 150, datum, th.secondary, 2.0, dash="9,5")
    # Both captions of the section end on the same right edge, clear of
    # the receiver's stand at x = 800: set from the left they run into it
    # in either language.
    s.text(
        782,
        datum + 26,
        "datum: the plane tangent to the two rail heads",
        15,
        th.secondary,
        anchor="end",
    )
    s.dim(cx - gauge / 2, gy - 76, cx + gauge / 2, gy - 76, "1,435 m", 0, 15)

    # Source A and source B on the track centre.
    ay = datum - 60.0
    by = datum - 250.0
    s.circle(cx, ay, 9, th.primary)
    s.circle(cx, by, 9, th.accent)
    s.dim(cx, datum, cx, ay, "0,5 m", 60, 15, label_side="right")
    s.dim(cx, datum, cx, by, "4,0 m", 120, 15, label_side="right")
    s.text(cx - 26, ay - 6, "A — rolling, impact, squeal,", 14, th.fg, anchor="end")
    s.text(cx - 26, ay + 16, "bridge, low traction", 14, th.fg, anchor="end")
    s.text(cx - 26, by + 16, "B — exhausts, roof apparatus,", 14, th.fg, anchor="end")
    s.text(cx - 26, by + 38, "pantograph recess", 14, th.fg, anchor="end")

    # The receiver, with the line of sight to each source and psi.
    rx = 800.0
    ry = gy - 150.0
    s.mic(rx, ry, gy, 0.9)
    s.line(cx, ay, rx, ry, th.primary, 1.8)
    s.line(cx, by, rx, ry, th.accent, 1.8)
    s.line(cx, ay, rx + 10, ay, th.muted, 1.2, dash="5,4")
    s.text(cx + 250, ay - 14, "$ψ > 0$", 16, th.primary)
    s.text(
        782,
        ay + 34,
        "$ψ ≤ 0$: the vertical correction of A is zero",
        15,
        th.secondary,
        anchor="end",
    )
    s.text(rx, ry - 24, "receiver, 4 m", 15, th.fg, bold=True)

    s.circle(cx - 12, ay, 0.1, th.primary)

    # --- plan ------------------------------------------------------------
    py = 610.0
    s.text(70.0, 512.0, "Plan", 17, th.fg, anchor="start", bold=True)
    s.line(90.0, py, 520.0, py, th.muted, 2.0, dash="12,8")
    s.text(96.0, py - 14, "track axis", 15, th.muted, anchor="start")
    ox = 300.0
    s.circle(ox, py, 7, th.fg)
    pts = []
    for deg in range(0, 361, 4):
        a = math.radians(deg)
        lvl = 10.0 * math.log10(0.01 + 0.99 * math.sin(a) ** 2)
        r = max(76.0 * (1.0 + lvl / 24.0), 4.0)
        pts.append(f"{ox + r * math.cos(a):.1f} {py - r * math.sin(a):.1f}")
    s.path("M " + " L ".join(pts) + " Z", stroke=th.accent, sw=2.0)
    s.text(ox, py - 92, "0 dB broadside", 15, th.accent)
    s.text(ox - 96, py + 26, "−20 dB along the track", 15, th.accent, anchor="end")
    s.arrow(ox, py, ox + 140, py - 78, th.primary, 1.8)
    s.text(ox + 148, py - 82, "receiver bearing", 15, th.primary, anchor="start")
    s.text(ox + 56, py - 18, "$φ$", 17, th.primary, bold=True)
    s.rect(600.0, py - 140, 280, 112, th.panel, th.muted, rx=8, sw=1.2)
    s.text(614.0, py - 110, "Impact noise applies from 50 m", 14, th.fg, anchor="start")
    s.text(614.0, py - 90, "before a joint to 50 m after it", 14, th.fg, anchor="start")
    s.text(614.0, py - 60, "Curve squeal needs ≥ 50 m", 14, th.fg, anchor="start")
    s.text(614.0, py - 40, "of continuous curve", 14, th.fg, anchor="start")


# ---------------------------------------------------------------------------
# IEC 61400-11 ground-board microphone mounting (6.1.5, 7.1)
# ---------------------------------------------------------------------------


def _d_wind_turbine_board(s: SVG, th: Theme) -> None:
    # --- plan ------------------------------------------------------------
    cx, cy, r = 250.0, 210.0, 130.0
    s.circle(cx, cy, r, th.panel, th.fg, sw=2.2)
    s.dim(cx - r, cy + r + 34, cx + r, cy + r + 34, "diameter ≥ 1,0 m", 0, 15)
    s.circle(cx, cy, 9, th.fg)
    s.arrow(cx, cy, cx + r + 46, cy, th.secondary, 2.0)
    s.text(cx + r + 54, cy + 6, "to the turbine", 15, th.secondary, anchor="start")
    # The optional split, off the centre line and parallel to the axis.
    s.line(cx - r * 0.86, cy - 46, cx + r * 0.86, cy - 46, th.primary, 2.0, dash="8,5")
    s.text(
        cx, cy - 58, "split (if any): off centre, parallel, gap < 1 mm", 14, th.primary
    )
    s.text(cx, cy - r - 24, "Plan", 17, th.fg, bold=True)

    # --- section ---------------------------------------------------------
    gy = 470.0
    bx0, bx1 = 470.0, 830.0
    s.ground(gy, 440.0, 860.0)
    s.rect(bx0, gy - 12, bx1 - bx0, 12, th.panel, th.fg, sw=2.0)
    s.text((bx0 + bx1) / 2, gy + 86, "plywood ≥ 12,0 mm  ·  metal ≥ 2,5 mm", 15, th.fg)
    # Soil fillet at the edges.
    s.path(f"M {bx0 - 34} {gy} L {bx0} {gy - 12} L {bx0} {gy} Z", fill=th.muted)
    s.path(f"M {bx1 + 34} {gy} L {bx1} {gy - 12} L {bx1} {gy} Z", fill=th.muted)
    s.text(bx0 + 6, gy + 30, "soil fillet", 14, th.muted, anchor="start")

    # Capsule with its diaphragm in the plane of the board.
    mcx = (bx0 + bx1) / 2
    s.rect(mcx - 22, gy - 12, 44, 10, th.fg, rx=3)
    s.text(mcx, gy + 58, "capsule diaphragm in the board plane, ≤ 13 mm", 14, th.fg)
    # Primary windscreen: a foam half-sphere ~90 mm across.
    s.path(
        f"M {mcx - 68} {gy - 12} A 68 68 0 0 1 {mcx + 68} {gy - 12} Z",
        fill=th.accent,
        stroke=th.fg,
        sw=1.6,
    )
    s.text(mcx, gy - 96, "primary windscreen ≈ 90 mm", 15, th.fg)
    # Ghosted secondary windscreen.
    s.path(
        f"M {mcx - 116} {gy - 12} A 116 116 0 0 1 {mcx + 116} {gy - 12}",
        stroke=th.muted,
        sw=1.8,
        dash="8,6",
    )
    s.text(mcx, gy - 186, "secondary: high wind only —", 14, th.muted)
    s.text(mcx, gy - 166, "document and correct its insertion loss", 14, th.muted)
    s.text(bx0 - 20, 330.0, "Section", 17, th.fg, anchor="start", bold=True)

    # Tolerance box, sized on its longest line in either language and set
    # against the sheet's own left margin, so it still stops short of the
    # ground the section stands on.
    s.rect(40.0, 400.0, 392.0, 156.0, th.panel, th.muted, rx=8, sw=1.2)
    s.text(56.0, 430.0, "Position (clause 7.1)", 16, th.fg, anchor="start", bold=True)
    for i, line in enumerate(
        (
            "within ±15° of downwind",
            "$R_0$ to ±20 %, max ±30 m, measured to ±2 %",
            "board inclination $φ$ between 25° and 40°",
            "reflections from structures < 0,2 dB",
        )
    ):
        s.text(56.0, 458.0 + i * 24.0, line, 15, th.fg, anchor="start")


# ---------------------------------------------------------------------------
# RD 1367/2007: phase to period to year, and the three criteria
# ---------------------------------------------------------------------------


def _d_rd1367_chain(s: SVG, th: Theme) -> None:
    # Stage 1: the 24 h clock.
    y0 = 90.0
    x0, x1 = 70.0, 830.0
    w = x1 - x0
    s.text(
        x0,
        y0 - 12,
        "1 — the day, split into evaluation periods (Annex I A.1)",
        16,
        th.fg,
        anchor="start",
        bold=True,
    )
    bounds = (
        (0.0, 7.0, th.primary, "night 23-07"),
        (7.0, 19.0, th.accent, "day 07-19"),
        (19.0, 23.0, "#e8a838", "evening 19-23"),
        (23.0, 24.0, th.primary, ""),
    )
    for a, b, color, label in bounds:
        s.rect(x0 + w * a / 24.0, y0, w * (b - a) / 24.0, 42, th.panel, color, sw=2.0)
        if label:
            s.text(x0 + w * (a + b) / 48.0, y0 + 28, label, 15, th.fg)
    # The day, broken into the worked example's three phases.
    py = y0 + 62.0
    phases = (
        (7.0, 9.0, "2 h shut"),
        (9.0, 15.0, "6 h, machine"),
        (15.0, 19.0, "4 h, rest"),
    )
    for a, b, label in phases:
        s.rect(
            x0 + w * a / 24.0, py, w * (b - a) / 24.0, 34, th.panel, th.muted, sw=1.2
        )
        s.text(x0 + w * (a + b) / 48.0, py + 23, label, 14, th.fg)
    s.text(
        x0,
        py + 58,
        "noise phases $T_i$ of uniformly perceived level",
        15,
        th.muted,
        anchor="start",
    )

    # Stage 2: the phase corrections.
    y1 = 250.0
    s.rect(x0, y1, 360, 108, th.panel, th.muted, rx=8, sw=1.2)
    s.text(
        x0 + 14,
        y1 + 28,
        "2 — each phase, corrected",
        16,
        th.fg,
        anchor="start",
        bold=True,
    )
    # The LKeq strings of this builder stay uncomposed for now: the
    # tokenizer reads "Keq" as one run, so it cannot set it as the roman
    # "Aeq" of _ROMAN_SCRIPTS beside it, and the period level also nests
    # the LKeq,Ti subscript inside the 10^(L/10) exponent.
    s.text(
        x0 + 14,
        y1 + 56,
        "LKeq,Ti = LAeq,Ti + Kt + Kf + Ki",
        15,
        th.fg,
        anchor="start",
        mono=True,
    )
    s.text(
        x0 + 14,
        y1 + 82,
        "$K_t + K_f + K_i ≤ 9$ dB (Annex IV A.3.3)",
        15,
        th.secondary,
        anchor="start",
    )

    # Stage 3: the duration-weighted mean.
    s.arrow(x0 + 372, y1 + 54, x0 + 424, y1 + 54, th.muted, 1.8)
    s.rect(x0 + 436, y1, 324, 124, th.panel, th.muted, rx=8, sw=1.2)
    s.text(
        x0 + 450, y1 + 28, "3 — the period level", 16, th.fg, anchor="start", bold=True
    )
    s.text(
        x0 + 450,
        y1 + 54,
        "LKeq,x = 10 lg[ (1/T) Σ Ti",
        14,
        th.fg,
        anchor="start",
        mono=True,
    )
    s.text(
        x0 + 450,
        y1 + 74,
        "                10^(LKeq,Ti/10) ]",
        14,
        th.fg,
        anchor="start",
        mono=True,
    )
    s.text(
        x0 + 450,
        y1 + 98,
        "round_reported_level → 57 dB",
        15,
        th.accent,
        anchor="start",
        mono=True,
    )

    # Stage 4: the annual mean.
    y2 = 400.0
    s.rect(x0, y2, 360, 96, th.panel, th.muted, rx=8, sw=1.2)
    s.text(
        x0 + 14, y2 + 28, "4 — the annual value", 16, th.fg, anchor="start", bold=True
    )
    s.text(
        x0 + 14, y2 + 56, "$L_{K,x}$ over the operating days", 15, th.fg, anchor="start"
    )
    s.text(
        x0 + 14,
        y2 + 82,
        "303 open / 62 closed → 56 dB",
        15,
        th.accent,
        anchor="start",
        mono=True,
    )
    s.arrow(x0 + 180, y1 + 118, x0 + 180, y2 - 8, th.muted, 1.8)

    # The three criteria.
    s.arrow(x0 + 372, y2 + 46, x0 + 424, y2 + 46, th.muted, 1.8)
    cx = x0 + 436
    for i, (label, value) in enumerate(
        (
            ("worst phase ≤ limit + 5 dB", "59 ≤ 60 ✓"),
            ("daily LKeq,x ≤ limit + 3 dB", "57 ≤ 58 ✓"),
            ("annual $L_{K,x}$ ≤ limit", "56 > 55 ✗"),
        )
    ):
        yy = y2 + i * 40
        color = th.secondary if i == 2 else th.accent
        s.rect(cx, yy, 324, 32, th.bg, color, rx=6, sw=1.8)
        s.text(cx + 12, yy + 22, label, 14, th.fg, anchor="start")
        s.text(cx + 312, yy + 22, value, 14, color, anchor="end", mono=True)
    s.text(
        x0,
        y2 + 146,
        "Article 25.2 drops the third criterion for an activity already in operation",
        14,
        th.muted,
        anchor="start",
    )
