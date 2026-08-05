#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Diagrams of the underwater guides: radiated noise and ocean propagation.

One subject: sound in water, where the medium itself is the instrument. The
deployment diagram draws the geometry that a radiated-noise measurement is
only valid in, and the SOFAR diagram draws the sound-speed profile that turns
the deep ocean into a waveguide.
"""

from __future__ import annotations

from .canvas import SVG, Theme

# ---------------------------------------------------------------------------
# Ship radiated-noise measurement geometry (ISO 17208-1)
# ---------------------------------------------------------------------------

def _d_hydrophone_deployment(s: SVG, th: Theme) -> None:
    """ISO 17208-1 deep-water geometry: ship transiting past a buoy-suspended
    vertical array of three hydrophones at 15/30/45 degree depression angles,
    lateral CPA distance of at least 100 m, plus the plan-view data window."""
    import math
    surf = 150.0
    sc = 2.6                              # px per metre
    shx = 190.0                           # ship reference point at the CPA
    bx = shx + 100 * sc                   # buoy: dCPA = 100 m away

    # Sea surface as a gentle wave.
    dsur = f"M 50 {surf}"
    x = 50.0
    while x < 590:
        dsur += f" Q {x + 8:.0f} {surf - 5:.0f} {x + 16:.0f} {surf:.0f}"
        dsur += f" Q {x + 24:.0f} {surf + 5:.0f} {x + 32:.0f} {surf:.0f}"
        x += 32
    s.path(dsur, stroke=th.primary, sw=1.8)

    # Ship (side profile) at the closest point of approach.
    s.path(f"M 108 132 L 262 132 L 276 {surf} L 254 166 L 130 166 "
           f"L 108 {surf} Z", fill=th.panel, stroke=th.fg, sw=2)
    s.rect(122, 104, 44, 28, th.panel, th.fg, rx=3, sw=1.6)
    s.text(212, 88, "Ship under test", 18, th.fg, bold=True, anchor="end")
    s.circle(shx, surf, 3.5, th.fg)

    # Surface buoy with the suspended array and its ballast.
    s.circle(bx, 148, 10, th.panel, th.fg, 2)
    s.line(bx, 138, bx, 120, th.fg, 1.6)
    s.path(f"M {bx:.0f} 120 L {bx - 18:.0f} 125 L {bx:.0f} 130 Z",
           fill=th.secondary)
    s.text(bx + 18, 122, "Surface buoy", 16, th.fg, anchor="start", bold=True)
    s.line(bx, 158, bx, 448, th.fg, 2)
    s.rect(bx - 8, 448, 16, 22, th.fg, rx=2)
    s.text(bx, 490, "ballast", 14, th.muted)

    # Hydrophones at the depths set by the three depression angles.
    hyd = [(15, "≈ 27 m"), (30, "≈ 58 m"), (45, "= 100 m")]
    for ang, dlab in hyd:
        dy = 100 * math.tan(math.radians(ang)) * sc
        hy = surf + dy
        s.line(shx, surf, bx, hy, th.muted, 1.1, dash="5,4")
        s.circle(bx, hy, 7, th.secondary)
        s.circle(bx, hy, 2.5, th.bg)
        s.text(bx + 16, hy + 5, dlab, 15, th.fg, anchor="start", mono=True)
        lx_ = 305.0
        ly_ = surf + (lx_ - shx) * math.tan(math.radians(ang))
        s.text(lx_, ly_ - 7, f"{ang}°", 15, th.muted)
    s.text(bx + 16, surf + 100 * math.tan(math.radians(30)) * sc + 32,
           "vertical array of 3 hydrophones", 15, th.muted, anchor="start")

    # Lateral distance at the CPA and the water depth.
    s.dim(shx, 100, bx, 100, "dCPA ≥ 100 m (or 1·L)", offset=0, size=17)
    s.line(shx, 130, shx, 106, th.muted, 0.9, dash="3,3")
    s.line(bx, 116, bx, 106, th.muted, 0.9, dash="3,3")
    s.ground(540, 50, 600)
    s.text(90, 570, "sea floor", 14, th.muted, anchor="start")
    s.dim(70, surf, 70, 540, "water depth ≥ 150 m (or 1.5·L)", offset=0,
          size=16, label_side="right")

    # Plan view: course, CPA and the +/-30 degree data window.
    s.text(750, 130, "Plan view", 17, th.fg, bold=True)
    s.arrow(640, 170, 860, 170, th.fg, 2.0)
    s.text(852, 156, "course", 14, th.muted, anchor="end")
    s.rect(676, 162, 28, 14, th.panel, th.fg, rx=3, sw=1.4)
    s.circle(750, 170, 3.5, th.fg)
    # dCPA line drawn in two runs so it does not cross the label below.
    s.line(750, 170, 750, 184, th.muted, 1.1, dash="5,4")
    s.line(750, 210, 750, 330, th.muted, 1.1, dash="5,4")
    s.text(758, 256, "dCPA", 14, th.fg, anchor="start", mono=True)
    s.circle(750, 330, 6, th.secondary)
    s.circle(750, 330, 2.2, th.bg)
    win = 160 * math.tan(math.radians(30))
    s.line(750, 330, 750 - win, 170, th.muted, 1.0, dash="3,4")
    s.line(750, 330, 750 + win, 170, th.muted, 1.0, dash="3,4")
    s.text(750, 296, "±30°", 14, th.muted)
    s.line(750 - win, 178, 750 + win, 178, th.accent, 3.0)
    s.text(750, 200, "data window", 15, th.accent)

    # Normative context.
    s.text(80, 594, "Four runs, two per side; levels averaged while the ship crosses the data window",
           17, th.fg, anchor="start")
    s.text(80, 620, "Hydrophone depths from the 15°, 30° and 45° depression angles at r = dCPA; L = ship length",
           17, th.fg, anchor="start")


# ---------------------------------------------------------------------------
# SOFAR channel (deep sound channel)
# ---------------------------------------------------------------------------

def _d_sofar_channel(s: SVG, th: Theme) -> None:
    """The deep sound channel: measured North Atlantic values (sound speed
    1524 m/s at the surface, minimum near 1492 m/s at the 1200 m axis,
    1527 m/s at 4800 m) and rays oscillating about the channel axis."""
    import math
    surf, bot = 100.0, 520.0
    ax_y = surf + 1200.0 / 4800.0 * (bot - surf)      # channel axis, 1200 m

    # Ocean frame: surface, seabed and the left depth axis.
    s.line(60, surf, 850, surf, th.fg, 2.2)
    s.text(845, 88, "sea surface", 14, th.muted, anchor="end")
    s.ground(bot, 60, 850)
    s.line(90, surf, 90, bot, th.muted, 1.4)
    for dy_, ly_, dlab in ((surf, surf - 8, "0 m"), (ax_y, ax_y + 5, "1200 m"),
                           (bot, bot - 8, "4800 m")):
        s.line(84, dy_, 90, dy_, th.muted, 1.4)
        s.text(78, ly_, dlab, 14, th.fg, anchor="end", mono=True)

    # Channel axis (the sound-speed minimum).
    s.line(90, ax_y, 850, ax_y, th.muted, 1.2, dash="7,5")

    # --- Left: the sound-speed profile c(z) --------------------------------
    s.text(195, 76, "Sound-speed profile c(z)", 18, th.fg, bold=True)
    def cx_of(c: float) -> float:                     # 1480..1540 m/s
        return 90.0 + (c - 1480.0) / 60.0 * 180.0
    x_s, x_m, x_b = cx_of(1524), cx_of(1492), cx_of(1527)
    s.path(f"M {x_s:.1f} {surf:.1f} Q {x_m + 24:.1f} {surf + 52:.1f} "
           f"{x_m:.1f} {ax_y:.1f} Q {x_m + 14:.1f} {ax_y + 160:.1f} "
           f"{x_b:.1f} {bot:.1f}", stroke=th.primary, sw=2.6)
    s.circle(x_s, surf, 3.5, th.primary)
    s.circle(x_m, ax_y, 3.5, th.primary)
    s.circle(x_b, bot, 3.5, th.primary)
    s.text(x_s + 10, surf + 22, "1524 m/s", 14, th.fg, anchor="start", mono=True)
    s.text(x_m + 10, ax_y + 24, "≈ 1492 m/s", 14, th.fg, anchor="start", mono=True)
    s.text(x_b + 10, bot - 12, "1527 m/s", 14, th.fg, anchor="start", mono=True)

    # --- Right: rays trapped about the axis --------------------------------
    s.text(600, 76, "Ray paths near the axis", 18, th.fg, bold=True)
    sx = 315.0
    s.circle(sx, ax_y, 6, th.secondary)
    s.circle(sx, ax_y, 2.2, th.bg)
    s.text(310, 130, "source on the channel axis", 15, th.fg, anchor="start")
    s.line(322, 136, sx + 1, ax_y - 8, th.muted, 1.0)
    for amp, lam, col in ((45.0, 260.0, th.accent), (68.0, 310.0, th.primary),
                          (90.0, 360.0, th.secondary)):
        d = f"M {sx:.1f} {ax_y:.1f}"
        xr = sx
        while xr < 833:
            xr += 7
            yr = ax_y + amp * math.sin(2 * math.pi * (xr - sx) / lam)
            d += f" L {xr:.1f} {yr:.1f}"
        s.path(d, stroke=col, sw=1.8)
        y_end = ax_y + amp * math.sin(2 * math.pi * (840 - sx) / lam)
        y_prev = ax_y + amp * math.sin(2 * math.pi * (833 - sx) / lam)
        s.arrow(833.0, y_prev, 841.0, y_end, col, 1.8)
    s.text(575, 420, "rays that stay in the channel meet no surface or bottom loss",
           16, th.muted, italic=True)

    # Physics of the channel.
    s.text(80, 560, "c rises toward the surface (temperature) and toward the bottom (pressure); the minimum traps sound",
           17, th.fg, anchor="start")
    s.text(80, 588, "rays launched within about ±12° of the axis stay trapped and can cross entire oceans",
           17, th.fg, anchor="start")
