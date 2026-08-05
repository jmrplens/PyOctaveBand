#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Diagrams of the environment guides: sources, propagation and assessment.

One subject: outdoor sound, from what emits it to what the law makes of it at
the receiver. The propagation diagrams draw the geometry the models work in
(barrier and ground, image source, refracted rays), the source diagram draws
a measurement geometry fixed by its own standard, and the assessment diagrams
draw what happens to the level once it has arrived.
"""

from __future__ import annotations

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
    s.text(fx + 80, 104, "Building façade", 22, th.fg, bold=True)

    # Source (left): car on a road
    s.rect(60, gy - 9, 140, 9, th.muted)
    s.path(f"M 88 {gy - 30} L 106 {gy - 48} L 146 {gy - 48} L 164 {gy - 30} Z", fill=th.secondary)
    s.rect(80, gy - 32, 96, 14, th.secondary, rx=5)
    s.circle(102, gy - 13, 9, th.fg)
    s.circle(156, gy - 13, 9, th.fg)
    for r in (44, 76, 108):
        s.path(f"M {168 + r * 0.5} {gy - 34 - r * 0.55} "
               f"A {r} {r} 0 0 1 {168 + r * 0.87} {gy - 34 + r * 0.1}",
               stroke=th.accent, sw=1.6)

    # Position A: free field, capsule 4 m above ground
    ax = 330.0
    a_cap = gy - 230.0
    s.mic(ax, a_cap, gy, 1.15)
    s.dim(ax, gy, ax, a_cap, "4.0 ± 0.2 m", offset=-60, size=20)
    s.text(ax - 20, a_cap - 58, "A — free field", 22, th.fg, bold=True)
    s.text(ax - 20, a_cap - 30, "0 dB", 22, th.accent, bold=True, mono=True)

    # Position B: 2 m in front of the facade, dimension at capsule height
    bx = fx - 108.0
    b_cap = gy - 230.0
    s.mic(bx, b_cap, gy, 1.15)
    s.dim(bx, b_cap + 6, fx, b_cap + 6, "2 m", offset=-14, size=20)
    s.text(bx - 30, b_cap - 58, "B — 2 m from façade", 22, th.fg, bold=True)
    s.text(bx - 30, b_cap - 30, "−3 dB", 22, th.secondary, bold=True, mono=True)

    # Position C: flush-mounted on the facade, below B's dimension zone
    cy = gy - 120.0
    s.circle(fx + 3, cy, 7, th.fg)
    # The leader crosses mic B's mast (plain line crossing, standard
    # drafting); the label itself sits in the clear zone between masts.
    s.line(fx - 2, cy + 5, 470, cy + 60, th.muted, 1.4)
    s.text(462, cy + 84, "C — flush-mounted", 22, th.fg, bold=True)
    s.text(462, cy + 110, "−6 dB", 22, th.secondary, bold=True, mono=True)


def _d_outdoor(s: SVG, th: Theme) -> None:
    c_diff = th.accent          # diffracted (over-the-top) ray
    c_direct = th.muted         # blocked direct ray
    gy = 430.0                  # ground line
    s.ground(gy, 60.0, 840.0)
    s.text(66.0, gy + 26.0, "Ground (Gs, Gm, Gr)", 18, th.muted, anchor="start")

    # --- source (loudspeaker) on the left, acoustic centre at (sx, sy) -------
    sx, sy = 150.0, 300.0
    for r in (26, 44, 62):
        s.path(f"M {sx + r * 0.22:.1f} {sy - r:.1f} "
               f"A {r} {r} 0 0 1 {sx + r:.1f} {sy - r * 0.22:.1f}",
               stroke=th.muted, sw=1.3)
    s.rect(sx - 20, sy - 24, 40, 48, th.panel, th.fg, rx=5, sw=2)
    s.circle(sx, sy - 6, 9, th.fg)
    s.circle(sx, sy - 6, 3.5, th.bg)
    s.circle(sx, sy + 14, 6, th.fg)
    s.line(sx, sy + 24, sx, gy, th.fg, 2.0)          # mast to the ground
    s.text(sx, sy - 74, "Source", 20, th.fg, bold=True)

    # --- barrier in the middle, top edge at (ex, ey) -------------------------
    ex, ey = 450.0, 150.0
    bw = 16.0
    s.rect(ex - bw / 2, ey, bw, gy - ey, th.secondary, th.fg, sw=2)
    s.text(ex + 16.0, (ey + gy) / 2 + 6.0, "Barrier", 20, th.secondary,
           bold=True, anchor="start")
    s.circle(ex, ey, 5.5, th.bg, th.fg, 2.0)          # diffraction edge node

    # --- receiver (microphone) on the right, capsule at (rx, ry) -------------
    rx, ry = 770.0, 288.0
    s.mic(rx, ry, gy, 1.0)
    s.text(rx, ry - 18.0, "Receiver", 20, th.fg, bold=True)

    # --- rays ---------------------------------------------------------------
    # Direct (blocked) ray straight through the barrier.
    s.line(sx + 14, sy - 6, rx, ry + 6, c_direct, 1.8, dash="7,6")
    s.text(285.0, sy + 40.0, "direct path (blocked)", 16, c_direct,
           anchor="middle", italic=True)
    # Diffracted ray up to the top edge, then down to the receiver.
    s.line(sx + 12, sy - 12, ex, ey, c_diff, 3.0)
    s.arrow(ex, ey, rx, ry + 2, c_diff, 3.0)
    s.text(300.0, 208.0, "dss", 18, c_diff, anchor="middle")
    s.text(610.0, 200.0, "dsr", 18, c_diff, anchor="middle")
    s.text(ex, ey - 22.0, "diffracted path", 17, c_diff, bold=True)

    # --- heights (witness dimensions) ---------------------------------------
    s.dim(sx - 44, gy, sx - 44, sy - 6, "hs", offset=0, label_side="left")
    s.line(sx - 44, gy, sx, gy, th.muted, 0.9, dash="3,3")
    s.line(sx - 44, sy - 6, sx, sy - 6, th.muted, 0.9, dash="3,3")
    s.dim(rx + 40, gy, rx + 40, ry + 6, "hr", offset=0, label_side="right")
    s.line(rx, gy, rx + 40, gy, th.muted, 0.9, dash="3,3")
    s.line(rx, ry + 6, rx + 40, ry + 6, th.muted, 0.9, dash="3,3")

    # --- master relations ---------------------------------------------------
    s.text(450.0, gy + 58.0, "z = dss + dsr − d   (path difference)", 19,
           th.fg, bold=True)
    s.text(450.0, gy + 84.0,
           "Dz = 10 log10[ 3 + (C₂/λ) C₃ z Kmet ]   (Eq. 14)", 18, th.muted)


def _d_impulse_prominence(s: SVG, th: Theme) -> None:
    """Impulsive-sound prominence and the LAeq adjustment (NT ACOU 112:2002)."""
    cx = 450.0
    bw, bh = 640.0, 60.0
    x0 = cx - bw / 2

    # --- Input --------------------------------------------------------------
    s.rect(x0, 56, bw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 82, "A-weighted level history  L_pAF  (time weighting F)", 19,
           th.fg, "middle", bold=True)
    s.text(cx, 103, "an onset = a stretch where the gradient exceeds 10 dB/s "
           "(clauses 4.5-4.7)", 13, th.muted, "middle")
    s.arrow(cx, 116, cx, 150, th.fg, 1.8)

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 26, l1, 18, th.fg, "middle", bold=True)
        s.text(cx, y + 47, l2, 13, th.muted, "middle")

    _step(150, "Per impulse: onset rate OR and level difference LD",
          "OR = onset slope [dB/s],   LD = Le − Ls [dB]", th.primary)
    _step(242, "Predicted prominence  P   (clause 7, Formula 1)",
          "P = 3·log10(OR) + 2·log10(LD);   highest P over 30 min governs", th.fg)
    _step(334, "Adjustment  KI   (clause 8, Formula 2)",
          "KI = 1.8·(P − 5) dB for P > 5, else 0", th.secondary)
    s.arrow(cx, 210, cx, 242, th.fg, 1.8)
    s.arrow(cx, 302, cx, 334, th.fg, 1.8)
    s.arrow(cx, 394, cx, 426, th.fg, 1.8)

    # --- Output -------------------------------------------------------------
    s.rect(x0, 426, bw, 60, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, 452, "Rating level  LAr,T = 10·log10( (1/T) Σ Δt·10^((LAeq+KI)/10) )",
           18, th.fg, "middle", bold=True)
    s.text(cx, 473, "impulse-adjusted level over the reference time  (Note 1)",
           13, th.muted, "middle")


# ---------------------------------------------------------------------------
# Wind-turbine noise measurement geometry (IEC 61400-11)
# ---------------------------------------------------------------------------

def _d_wind_turbine(s: SVG, th: Theme) -> None:
    """IEC 61400-11 apparent-sound-power geometry: downwind ground-board
    microphone at R0 = H + D/2, slant distance R1 to the rotor centre and
    the Figure 3 plan-view position pattern."""
    import math
    gy = 470.0
    s.ground(gy, 40.0, 668.0)

    # Wind arrow on the upwind side.
    s.arrow(52.0, 108.0, 148.0, 108.0, th.accent, 2.6)
    s.text(100.0, 88.0, "Wind", 18, th.accent, bold=True)

    # --- met mast: anemometer cups + wind vane -----------------------------
    mmx = 108.0
    s.line(mmx, gy, mmx, gy - 96.0, th.fg, 2.2)
    s.line(mmx - 12, gy - 96.0, mmx + 12, gy - 96.0, th.fg, 1.8)
    s.circle(mmx - 12, gy - 102.0, 4.5, th.panel, th.fg, 1.4)   # cup
    s.circle(mmx + 12, gy - 102.0, 4.5, th.panel, th.fg, 1.4)   # cup
    s.line(mmx, gy - 82.0, mmx + 20, gy - 82.0, th.fg, 1.6)     # vane arm
    s.path(f"M {mmx + 20:.0f} {gy - 87:.0f} L {mmx + 34:.0f} {gy - 82:.0f} "
           f"L {mmx + 20:.0f} {gy - 77:.0f} Z", fill=th.secondary)
    s.text(mmx, gy - 118.0, "Met mast", 16, th.fg, bold=True)
    s.text(mmx, gy + 30.0, "wind speed + direction", 14, th.muted)

    # --- turbine: tower, nacelle and the rotor edge-on ----------------------
    tx = 262.0                    # tower vertical centreline
    hub_y = 168.0                 # rotor centre => H = 302 px
    rr = 104.0                    # rotor radius (D = 208 px)
    s.path(f"M {tx - 10:.0f} {gy:.0f} L {tx - 5:.0f} {hub_y + 12:.0f} "
           f"L {tx + 5:.0f} {hub_y + 12:.0f} L {tx + 10:.0f} {gy:.0f} Z",
           fill=th.panel, stroke=th.fg, sw=1.8)
    s.rect(tx - 28, hub_y - 12, 56, 24, th.panel, th.fg, rx=6, sw=1.8)
    rx_ = tx - 34.0               # rotor plane (upwind of the tower)
    s.ellipse(rx_, hub_y, 9.0, rr, stroke=th.muted, sw=1.3, dash="6,5")
    s.line(rx_, hub_y - 8, rx_ - 4, hub_y - rr + 4, th.fg, 3.2)   # blade up
    s.line(rx_, hub_y + 8, rx_ + 3, hub_y + rr - 4, th.fg, 3.2)   # blade down
    s.circle(rx_, hub_y, 6.5, th.fg)
    s.text(tx + 36, hub_y - 26, "rotor centre", 15, th.fg, anchor="start")
    s.line(rx_ + 8, hub_y - 6, tx + 33, hub_y - 21, th.muted, 0.9)

    # Rotor diameter D across the swept ellipse.
    dx_ = rx_ - 58.0
    s.dim(dx_, hub_y - rr, dx_, hub_y + rr, "D", offset=0, label_side="left")
    s.line(rx_ - 8, hub_y - rr, dx_, hub_y - rr, th.muted, 0.9, dash="3,3")
    s.line(rx_ - 8, hub_y + rr, dx_, hub_y + rr, th.muted, 0.9, dash="3,3")

    # Hub height H, downwind of the tower.
    hx_ = tx + 56.0
    s.dim(hx_, gy, hx_, hub_y, "H", offset=0, label_side="right")
    s.line(tx + 10, gy, hx_, gy, th.muted, 0.9, dash="3,3")
    s.line(tx + 28, hub_y, hx_, hub_y, th.muted, 0.9, dash="3,3")

    # --- downwind microphone on a ground board ------------------------------
    mx = 640.0
    s.ellipse(mx, gy - 3.0, 36.0, 7.0, fill=th.panel, stroke=th.fg, sw=1.6)
    s.rect(mx - 16, gy - 10.0, 20, 6, th.fg, rx=2.5)              # capsule flat
    s.rect(mx + 4, gy - 11.0, 10, 8, th.primary, rx=2)            # body
    s.text(mx - 84, gy - 42.0, "Microphone on a ground board", 16, th.fg,
           bold=True, anchor="end")

    # Slant distance R1 from the rotor centre to the microphone.
    s.line(rx_, hub_y, mx - 12, gy - 8.0, th.primary, 2.2, dash="9,6")
    s.text(430.0, 296.0, "R1", 19, th.primary, bold=True)
    # Board-to-R1 inclination angle (25°..40°).
    ang = math.atan2(gy - 8.0 - hub_y, mx - 12 - rx_)   # slope of R1
    r_arc = 52.0
    axp = mx - 12 - r_arc * math.cos(ang)
    ayp = gy - 8.0 - r_arc * math.sin(ang)
    s.path(f"M {mx - 12 - r_arc:.1f} {gy - 8:.1f} "
           f"A {r_arc:.0f} {r_arc:.0f} 0 0 1 {axp:.1f} {ayp:.1f}",
           stroke=th.muted, sw=1.3)
    s.text(mx - 74, gy - 22.0, "φ", 17, th.muted)

    # Horizontal reference distance R0.
    s.dim(tx, gy, mx, gy, "R0 = H + D/2", offset=40)

    # --- plan-view inset: the Figure 3 position pattern ---------------------
    pcx, pcy, pr = 794.0, 218.0, 76.0
    s.text(pcx, 104.0, "Plan view (Figure 3)", 17, th.fg, bold=True)
    s.arrow(pcx - 60.0, 118.0, pcx - 60.0, 150.0, th.accent, 2.2)  # wind, from top
    s.circle(pcx, pcy, pr, "none", th.muted, 1.2)
    s.line(pcx - pr - 8, pcy, pcx + pr + 8, pcy, th.muted, 1.0, dash="4,4")
    s.line(pcx, pcy - pr - 8, pcx, pcy + pr + 8, th.muted, 1.0, dash="4,4")
    s.line(pcx - 16, pcy, pcx + 16, pcy, th.fg, 3.0)              # rotor, plan
    s.circle(pcx, pcy, 4.0, th.fg)
    # Reference position 1, downwind (diamond).
    p1x, p1y = pcx, pcy + pr
    s.path(f"M {p1x:.0f} {p1y - 7:.0f} L {p1x + 7:.0f} {p1y:.0f} "
           f"L {p1x:.0f} {p1y + 7:.0f} L {p1x - 7:.0f} {p1y:.0f} Z",
           fill=th.secondary)
    s.text(p1x + 13, p1y + 5, "1", 15, th.secondary, anchor="start", bold=True)
    # Optional positions 2 and 4 at ±60° from downwind, 3 upwind.
    for lbl, adeg, lx, ly, anch in (
        ("2", 150.0, -12.0, 4.0, "end"),
        ("3", 270.0, 12.0, 4.0, "start"),
        ("4", 30.0, 12.0, 4.0, "start"),
    ):
        pxx = pcx + pr * math.cos(math.radians(adeg))
        pyy = pcy + pr * math.sin(math.radians(adeg))
        s.circle(pxx, pyy, 5.5, th.bg, th.fg, 1.6)
        s.text(pxx + lx, pyy + ly, lbl, 15, th.muted, anchor=anch)
        s.line(pcx, pcy, pxx, pyy, th.muted, 0.9, dash="3,4")
    s.line(pcx, pcy, p1x, p1y, th.muted, 0.9, dash="3,4")
    s.text(pcx - 34, pcy + 52.0, "60°", 13, th.muted)
    s.text(pcx + 34, pcy + 52.0, "60°", 13, th.muted)
    s.text(pcx, pcy + pr + 30.0, "reference position 1 (downwind)", 14,
           th.secondary)
    s.text(pcx, pcy + pr + 50.0, "optional positions 2–4", 14, th.muted)

    # --- governing relations -------------------------------------------------
    s.text(450.0, 560.0,
           "R1 = √(H² + R0²)   slant distance, rotor centre → microphone",
           19, th.fg, bold=True)
    s.text(450.0, 588.0,
           "LWA,i = Lp,i − 6 + 10 log10(4π R1² / S0)   (Formula 26, S0 = 1 m²)",
           19, th.primary, bold=True)
    s.text(450.0, 614.0,
           "the −6 dB removes the board's pressure doubling; board-to-R1 angle φ = 25°–40°",
           16, th.muted)


# ---------------------------------------------------------------------------
# Ground reflection: direct ray, image source, path difference
# ---------------------------------------------------------------------------

def _d_ground_reflection(s: SVG, th: Theme) -> None:
    """Two-path ground interference: source, receiver, direct ray, the
    specular reflection unfolded through the image source, and the path
    difference that sets the interference phase (ISO 9613-2 ground effect,
    Chien-Soroka geometry)."""
    gy = 372.0
    sx, sy = 170.0, 232.0          # source (hs = 140 px)
    rx, ry = 700.0, 282.0          # receiver capsule tip (hr = 90 px)
    ix, iy = sx, gy + (gy - sy)    # image source, mirrored below the ground
    # Specular point: equal angles, found by unfolding through the image.
    bx = sx + (rx - sx) * (gy - sy) / ((gy - sy) + (gy - ry))

    s.ground(gy, 60.0, 840.0)

    # Source: point with radiating arcs.
    for r in (22, 38, 54):
        s.path(f"M {sx + r * 0.30:.1f} {sy - r * 0.95:.1f} "
               f"A {r} {r} 0 0 1 {sx + r * 0.95:.1f} {sy - r * 0.30:.1f}",
               stroke=th.muted, sw=1.3)
    s.circle(sx, sy, 8.0, th.fg)
    s.text(sx, sy - 66.0, "Source", 20, th.fg, bold=True)
    s.text(sx - 14, sy + 24, "S", 15, th.fg, anchor="end", mono=True)
    s.line(sx, sy + 8, sx, gy, th.fg, 2.0)

    # Receiver: measurement microphone.
    s.mic(rx, ry, gy, 1.0)
    s.text(rx, ry - 18.0, "Receiver", 20, th.fg, bold=True)
    s.text(rx - 18, ry + 10.0, "R", 15, th.fg, anchor="end", mono=True)

    # Direct ray r1.
    s.arrow(sx + 10, sy, rx - 8, ry - 2, th.primary, 2.6)
    s.text(430.0, 236.0, "direct ray  r1", 17, th.primary, bold=True)

    # Reflected ray via the specular point (equal angles).
    s.line(sx + 6, sy + 7, bx, gy, th.accent, 2.6)
    s.arrow(bx, gy, rx - 6, ry + 4, th.accent, 2.6)
    s.text(330.0, gy - 34.0, "reflected ray", 17, th.accent, bold=True)
    # Equal grazing angles at the bounce.
    s.path(f"M {bx - 34:.1f} {gy:.1f} A 34 34 0 0 1 {bx - 26:.1f} {gy - 21:.1f}",
           stroke=th.muted, sw=1.2)
    s.path(f"M {bx + 34:.1f} {gy:.1f} A 34 34 0 0 0 {bx + 26:.1f} {gy - 21:.1f}",
           stroke=th.muted, sw=1.2)
    s.text(bx, gy - 40.0, "equal angles", 14, th.muted)

    # Image source: ghosted mirror of the source below the ground.
    s.circle(ix, iy, 8.0, "none", th.secondary, 1.8)
    s.line(sx, gy, ix, iy - 8, th.secondary, 1.2, dash="4,4")
    s.text(ix + 18, iy + 5, "image source", 16, th.secondary, anchor="start")
    s.text(ix - 16, iy + 5, "S′", 15, th.secondary, anchor="end", mono=True)
    # The unfolded path S' -> R is straight through the bounce point: r2.
    s.line(ix, iy - 6, bx, gy, th.secondary, 1.6, dash="7,5")
    s.line(bx, gy, rx - 6, ry + 4, th.secondary, 1.2, dash="2,6")
    s.text(380.0, iy - 62.0, "r2 = |S′R|", 16, th.secondary, mono=True)

    # Heights.
    s.dim(sx - 46, gy, sx - 46, sy, "hs", offset=0, label_side="left")
    s.line(sx - 46, gy, sx - 8, gy, th.muted, 0.9, dash="3,3")
    s.line(sx - 46, sy, sx - 8, sy, th.muted, 0.9, dash="3,3")
    s.dim(rx + 42, gy, rx + 42, ry, "hr", offset=0, label_side="right")
    s.line(rx + 8, gy, rx + 42, gy, th.muted, 0.9, dash="3,3")
    s.line(rx + 8, ry, rx + 42, ry, th.muted, 0.9, dash="3,3")

    # Governing relations (top block, clear of the geometry).
    s.text(560.0, 88.0, "path difference  δ = r2 − r1", 20, th.fg, bold=True)
    s.text(560.0, 114.0, "phase difference  Δφ = 2π δ / λ  (+ arg Q)", 18,
           th.fg)
    s.text(560.0, 142.0,
           "p ∝ e^(jkr1)/r1 + Q · e^(jkr2)/r2   (Q = ground reflection coefficient)",
           16, th.muted)
    s.text(560.0, 168.0,
           "in phase (δ ≈ nλ): up to +6 dB    ·    out of phase (δ ≈ λ/2 on hard ground): a deep dip",
           15, th.muted)


# Atmospheric refraction: downwind multipath and the upwind shadow zone
# ---------------------------------------------------------------------------

def _d_atmospheric_refraction(s: SVG, th: Theme) -> None:
    """Refracting surface layer (Salomons 2001; Attenborough & Van
    Renterghem 2021, Ch. 11): wind profile arrows, an effective-sound-speed
    inset, downward-curved rays with a ground bounce on the downwind side
    and upward-curved rays opening an acoustic shadow on the upwind side.
    Horizontal scale about 1 px per metre; heights exaggerated."""
    gy = 452.0
    sx, sy = 450.0, gy - 56.0            # source, hs = 2 m (schematic)
    mlx, mrx = 90.0, 795.0               # receivers, 350 m to each side

    # --- upwind side (left): rays curve upward -----------------------------
    # Limiting ray: grazes the ground at ~220 m upwind, then climbs; the
    # region under it is the acoustic shadow.
    s.path(f"M {sx:.0f} {sy:.0f} C 380 430 300 452 232 452", stroke=th.secondary, sw=1.8)
    shadow = "M 232 452 C 150 452 90 420 60 340 L 60 452 Z"
    s.path(shadow, fill=th.panel)
    s.path("M 232 452 C 150 452 90 420 60 340", stroke=th.secondary,
           sw=1.4, dash="6,5")
    # Fan of upwind rays, all refracted upward.
    s.path(f"M {sx:.0f} {sy:.0f} C 380 390 310 372 250 340", stroke=th.secondary, sw=1.8)
    s.arrow(250.0, 340.0, 214.0, 316.0, th.secondary, 1.8)
    s.path(f"M {sx:.0f} {sy:.0f} C 400 375 340 330 290 270", stroke=th.secondary, sw=1.8)
    s.arrow(290.0, 270.0, 264.0, 234.0, th.secondary, 1.8)
    s.path(f"M {sx:.0f} {sy:.0f} C 415 345 385 270 355 195", stroke=th.secondary, sw=1.8)
    s.arrow(355.0, 195.0, 340.0, 156.0, th.secondary, 1.8)
    s.text(188.0, 414.0, "acoustic shadow", 15, th.secondary, italic=True,
           anchor="start")
    s.line(184.0, 410.0, 105.0, 395.0, th.muted, 1.0)
    # Shadow-boundary marker at the grazing point.
    s.line(232.0, 452.0, 232.0, 386.0, th.muted, 1.1, dash="4,4")
    s.text(232.0, 370.0, "≈ 220 m", 14, th.muted)

    # --- downwind side (right): rays curve down, ground bounce -------------
    s.path(f"M {sx:.0f} {sy:.0f} Q 620 366 786 402", stroke=th.primary, sw=2.0)
    s.arrow(770.0, 399.0, 788.0, 403.0, th.primary, 2.0)
    s.path(f"M {sx:.0f} {sy:.0f} Q 590 300 726 452", stroke=th.accent, sw=2.0)
    s.path("M 726 452 Q 758 420 782 408", stroke=th.accent, sw=2.0)
    s.arrow(766.0, 415.0, 784.0, 407.0, th.accent, 2.0)

    # --- scene: ground, source, receivers ----------------------------------
    s.ground(gy, 40.0, 860.0)
    for r in (18, 30, 42):
        s.path(f"M {sx - r:.0f} {sy:.0f} A {r} {r} 0 0 1 {sx + r:.0f} {sy:.0f}",
               stroke=th.muted, sw=1.2)
    s.circle(sx, sy, 7.0, th.fg)
    s.line(sx, sy + 7, sx, gy, th.fg, 2.0)
    s.text(sx, sy - 54.0, "Source", 18, th.fg, bold=True)
    s.dim(sx + 34, gy, sx + 34, sy, "2 m", offset=0, size=15,
          label_side="right")
    s.line(sx + 7, sy, sx + 34, sy, th.muted, 0.9, dash="3,3")
    s.mic(mlx, gy - 42.0, gy, 0.85)
    s.mic(mrx, gy - 46.0, gy, 1.0)
    s.text(mrx, gy - 66.0, "Receiver", 17, th.fg, bold=True)
    s.dim(mrx + 38, gy, mrx + 38, gy - 46.0, "1.5 m", offset=0, size=14,
          label_side="right")
    s.line(mrx + 8, gy - 46.0, mrx + 38, gy - 46.0, th.muted, 0.9, dash="3,3")
    s.dim(mlx, gy, sx, gy, "350 m", offset=36, size=16)
    s.dim(sx, gy, mrx, gy, "350 m", offset=36, size=16)

    # --- wind profile arrows (blowing left to right) -----------------------
    for wy, wl in ((84.0, 116.0), (114.0, 82.0), (144.0, 52.0)):
        s.arrow(500.0, wy, 500.0 + wl, wy, th.accent, 2.2)
    s.text(548.0, 66.0, "wind u(z)", 16, th.accent, bold=True)

    # --- inset: effective-sound-speed profiles -----------------------------
    s.text(143.0, 52.0, "c_eff(z) = c(z) + u(z)", 15, th.fg, bold=True)
    s.rect(58, 64, 170, 170, th.panel, th.fg, rx=8, sw=1.5)
    s.arrow(76.0, 214.0, 76.0, 88.0, th.muted, 1.3)
    s.text(76.0, 80.0, "z", 13, th.muted, italic=True)
    s.line(76.0, 214.0, 214.0, 214.0, th.muted, 1.3)
    s.path("M 143 214 Q 146 150 192 96", stroke=th.primary, sw=2.2)
    s.path("M 143 214 Q 140 150 100 96", stroke=th.secondary, sw=2.2)
    s.text(197.0, 92.0, "+u", 13, th.primary, anchor="start")
    s.text(96.0, 92.0, "−u", 13, th.secondary, anchor="end")
    s.text(143.0, 229.0, "340 m/s", 12, th.muted, mono=True)

    # --- physics captions --------------------------------------------------
    s.text(80.0, 540.0,
           "Upwind: rays bend up; beyond ≈ 220 m a ground shadow opens and the level collapses by over 20 dB",
           16, th.fg, anchor="start")
    s.text(80.0, 566.0,
           "Downwind: rays bend down; the receiver hears the direct and the ground-bounced arrival (multipath)",
           16, th.fg, anchor="start")
    s.text(80.0, 592.0,
           "a ±0.1 (m/s)/m gradient curves rays with radius Rc = c0/|g| ≈ 3.4 km; source hs = 2 m, receiver hr = 1.5 m",
           16, th.muted, anchor="start")


# ---------------------------------------------------------------------------
# Barrier diffraction over ground (Fresnel number, Kurze-Anderson)
# ---------------------------------------------------------------------------

def _d_ground_barrier(s: SVG, th: Theme) -> None:
    """The guide's barrier geometry: a 1 m source, a 4 m thin screen at
    50 m and a 1.5 m receiver at 100 m. The diffracted segments A and B,
    the blocked direct path d and the barrier_insertion_loss values
    (N = 0.44 and 10.0 dB at 500 Hz, 15.5 dB at 2 kHz)."""
    gy = 340.0                          # ground; 7 px/m horizontal, 60 px/m vertical
    sx, sy = 110.0, gy - 60.0           # source, hs = 1 m
    ex, ey = 460.0, gy - 240.0          # barrier top edge, 4 m
    rx, ry = 810.0, gy - 90.0           # receiver, hr = 1.5 m

    s.ground(gy, 40.0, 860.0)

    # Source loudspeaker on its mast.
    for r in (22, 38, 54):
        s.path(f"M {sx + r * 0.22:.1f} {sy - r:.1f} "
               f"A {r} {r} 0 0 1 {sx + r:.1f} {sy - r * 0.22:.1f}",
               stroke=th.muted, sw=1.2)
    s.rect(sx - 17, sy - 20, 34, 40, th.panel, th.fg, rx=5, sw=2)
    s.circle(sx, sy - 5, 8, th.fg)
    s.circle(sx, sy - 5, 3, th.bg)
    s.circle(sx, sy + 12, 5, th.fg)
    s.line(sx, sy + 20, sx, gy, th.fg, 2.0)
    s.text(sx, sy - 66, "Source", 19, th.fg, bold=True)

    # Thin screen with its diffraction edge.
    s.rect(ex - 8, ey, 16, gy - ey, th.secondary, th.fg, sw=2)
    s.text(ex + 18, ey + 74, "Barrier", 19, th.secondary, bold=True,
           anchor="start")

    # Receiver microphone.
    s.mic(rx, ry, gy, 1.0)
    s.text(rx, ry - 20, "Receiver", 19, th.fg, bold=True)

    # Blocked direct path and the diffracted path A + B.
    s.line(sx + 12, sy - 4, rx - 4, ry + 4, th.muted, 1.6, dash="7,6")
    s.text(268, 308, "direct d = 100.00 m (blocked)", 15, th.muted,
           italic=True)
    s.line(sx + 10, sy - 10, ex, ey, th.accent, 2.6)
    s.arrow(ex, ey, rx - 4, ry - 2, th.accent, 2.6)
    s.circle(ex, ey, 5.5, th.bg, th.fg, 2.0)
    s.text(258, 162, "A = 50.09 m", 17, th.accent, bold=True)
    s.text(660, 152, "B = 50.06 m", 17, th.accent, bold=True)

    # Height and distance dimensions.
    s.dim(64, gy, 64, sy - 4, "1.0 m", offset=0, size=15, label_side="left")
    s.line(64, sy - 4, sx - 17, sy - 4, th.muted, 0.9, dash="3,3")
    s.dim(430, gy, 430, ey, "4.0 m", offset=0, size=15, label_side="left")
    s.line(430, ey, ex - 8, ey, th.muted, 0.9, dash="3,3")
    s.dim(846, gy, 846, ry, "1.5 m", offset=0, size=14, label_side="right")
    s.line(rx + 6, ry, 846, ry, th.muted, 0.9, dash="3,3")
    s.dim(sx, gy, ex, gy, "50 m", offset=34, size=16)
    s.dim(ex, gy, rx, gy, "50 m", offset=34, size=16)

    # --- captions ----------------------------------------------------------
    s.text(80, 420,
           "path difference δ = A + B − d = 0.15 m; Fresnel number N = 2δ/λ = 0.44 at 500 Hz",
           18, th.fg, anchor="start")
    s.text(80, 448,
           "Kurze–Anderson: Δbar = 5 + 20 log10( √(2πN) / tanh √(2πN) ) = 10.0 dB at 500 Hz",
           18, th.primary, anchor="start", bold=True)
    s.text(80, 476,
           "N grows with frequency: the same screen gives 15.5 dB at 2 kHz (vertical scale exaggerated)",
           17, th.muted, anchor="start")
