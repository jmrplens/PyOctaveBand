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
    s.text(212, 88, "Ship under test", 15, th.fg, bold=True, anchor="end")
    s.circle(shx, surf, 3.5, th.fg)

    # Surface buoy with the suspended array and its ballast.
    s.circle(bx, 148, 10, th.panel, th.fg, 2)
    s.line(bx, 138, bx, 120, th.fg, 1.6)
    s.path(f"M {bx:.0f} 120 L {bx - 18:.0f} 125 L {bx:.0f} 130 Z",
           fill=th.secondary)
    s.text(bx + 18, 122, "Surface buoy", 14, th.fg, anchor="start", bold=True)
    s.line(bx, 158, bx, 448, th.fg, 2)
    s.rect(bx - 8, 448, 16, 22, th.fg, rx=2)
    s.text(bx, 490, "ballast", 12, th.muted)

    # Hydrophones at the depths set by the three depression angles.
    hyd = [(15, "≈ 27 m"), (30, "≈ 58 m"), (45, "= 100 m")]
    for ang, dlab in hyd:
        dy = 100 * math.tan(math.radians(ang)) * sc
        hy = surf + dy
        s.line(shx, surf, bx, hy, th.muted, 1.1, dash="5,4")
        s.circle(bx, hy, 7, th.secondary)
        s.circle(bx, hy, 2.5, th.bg)
        s.text(bx + 16, hy + 5, dlab, 13, th.fg, anchor="start", mono=True)
        lx_ = 305.0
        ly_ = surf + (lx_ - shx) * math.tan(math.radians(ang))
        s.text(lx_, ly_ - 7, f"{ang}°", 13, th.muted)
    s.text(bx + 16, surf + 100 * math.tan(math.radians(30)) * sc + 32,
           "vertical array of 3 hydrophones", 13, th.muted, anchor="start")

    # Lateral distance at the CPA and the water depth. The d_CPA and l_DW
    # labels stay plain for now: ISO 17208-1 prints the CPA and DW
    # subscripts upright (3.3 Note 1, 3.6) and the composer's curated roman
    # list does not carry them, so $d_{CPA}$ would set them as italic
    # indices; the sibling strings with L ride along so the plate keeps one
    # style.
    s.dim(shx, 100, bx, 100, "dCPA ≥ 100 m (or 1·L)", offset=0, size=15)
    s.line(shx, 130, shx, 106, th.muted, 0.9, dash="3,3")
    s.line(bx, 116, bx, 106, th.muted, 0.9, dash="3,3")
    s.ground(540, 50, 600)
    s.text(90, 570, "sea floor", 12, th.muted, anchor="start")
    s.dim(70, surf, 70, 540, "water depth ≥ 150 m (or 1.5·L)", offset=0,
          size=14, label_side="right")

    # Plan view: course, CPA and the +/-30 degree data window.
    s.text(750, 130, "Plan view", 15, th.fg, bold=True)
    s.arrow(640, 170, 860, 170, th.fg, 2.0)
    s.text(852, 156, "course", 12, th.muted, anchor="end")
    s.rect(676, 162, 28, 14, th.panel, th.fg, rx=3, sw=1.4)
    s.circle(750, 170, 3.5, th.fg)
    # dCPA line drawn in two runs so it does not cross the label below.
    s.line(750, 170, 750, 184, th.muted, 1.1, dash="5,4")
    s.line(750, 234, 750, 330, th.muted, 1.1, dash="5,4")
    s.text(758, 256, "dCPA", 12, th.fg, anchor="start", mono=True)
    s.circle(750, 330, 6, th.secondary)
    s.circle(750, 330, 2.2, th.bg)
    win = 160 * math.tan(math.radians(30))
    s.line(750, 330, 750 - win, 170, th.muted, 1.0, dash="3,4")
    s.line(750, 330, 750 + win, 170, th.muted, 1.0, dash="3,4")
    s.text(750, 296, "±30°", 12, th.muted)
    s.line(750 - win, 178, 750 + win, 178, th.accent, 3.0)
    s.text(750, 200, "data window", 13, th.accent)
    s.text(750, 220, "lDW = 2 dCPA tan 30°", 12, th.fg, mono=True)

    # Run schedule: what the data window sits inside (Clause 5.5-5.6).
    s.text(752, 384, "Run schedule (not to scale)", 15, th.fg, bold=True)
    s.line(630, 430, 878, 430, th.fg, 1.8)
    for tx, tlab in ((640.0, "COMEX"), (866.0, "FINEX")):
        s.line(tx, 420, tx, 440, th.fg, 1.6)
        s.text(tx, 412, tlab, 12, th.fg)
    s.line(722, 430, 784, 430, th.accent, 4.0)
    s.text(753, 458, "data window", 12, th.accent)
    s.dim(640, 444, 722, 444, "≥ 2 × DWL", offset=0, size=11)
    s.dim(784, 444, 866, 444, "≥ 2 × DWL", offset=0, size=11)
    s.path("M 866 430 Q 890 452 866 470 L 640 470", stroke=th.muted, sw=1.4,
           dash="6,4")
    s.arrow(660, 470, 640, 470, th.muted, 1.4)
    s.text(752, 490, "reverse course; 4 runs, 2 per side", 12, th.fg)
    s.text(752, 512, "background: stopped, ≥ 2 km, ≥ 30 s,", 12, th.muted)
    s.text(752, 530, "at the start and end of each test period", 12, th.muted)

    # Normative context.
    s.text(80, 594, "Four runs, two per side; levels averaged while the ship crosses the data window",
           15, th.fg, anchor="start")
    s.text(80, 620, "Hydrophone depths from the 15°, 30° and 45° depression angles at r = dCPA; L = ship length",
           15, th.fg, anchor="start")


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
    s.text(845, 88, "sea surface", 12, th.muted, anchor="end")
    s.ground(bot, 60, 850)
    s.line(90, surf, 90, bot, th.muted, 1.4)
    for dy_, ly_, dlab in ((surf, surf - 8, "0 m"), (ax_y, ax_y + 5, "1200 m"),
                           (bot, bot - 8, "4800 m")):
        s.line(84, dy_, 90, dy_, th.muted, 1.4)
        s.text(78, ly_, dlab, 12, th.fg, anchor="end", mono=True)

    # Channel axis (the sound-speed minimum).
    s.line(90, ax_y, 850, ax_y, th.muted, 1.2, dash="7,5")

    # --- Left: the sound-speed profile c(z) --------------------------------
    s.text(195, 76, "Sound-speed profile $c(z)$", 15, th.fg, bold=True)
    def cx_of(c: float) -> float:                     # 1480..1540 m/s
        return 90.0 + (c - 1480.0) / 60.0 * 180.0
    x_s, x_m, x_b = cx_of(1524), cx_of(1492), cx_of(1527)
    s.path(f"M {x_s:.1f} {surf:.1f} Q {x_m + 24:.1f} {surf + 52:.1f} "
           f"{x_m:.1f} {ax_y:.1f} Q {x_m + 14:.1f} {ax_y + 160:.1f} "
           f"{x_b:.1f} {bot:.1f}", stroke=th.primary, sw=2.6)
    s.circle(x_s, surf, 3.5, th.primary)
    s.circle(x_m, ax_y, 3.5, th.primary)
    s.circle(x_b, bot, 3.5, th.primary)
    s.text(x_s + 10, surf + 22, "1524 m/s", 12, th.fg, anchor="start", mono=True)
    s.text(x_m + 10, ax_y + 24, "≈ 1492 m/s", 12, th.fg, anchor="start", mono=True)
    s.text(x_b + 10, bot - 12, "1527 m/s", 12, th.fg, anchor="start", mono=True)

    # --- Right: rays trapped about the axis --------------------------------
    s.text(600, 76, "Ray paths near the axis", 15, th.fg, bold=True)
    sx = 315.0
    s.circle(sx, ax_y, 6, th.secondary)
    s.circle(sx, ax_y, 2.2, th.bg)
    s.text(310, 130, "source on the channel axis", 13, th.fg, anchor="start")
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
           14, th.muted, italic=True)

    # Physics of the channel.
    s.text(80, 560, "$c$ rises toward the surface (temperature) and the "
                    "bottom (pressure); the minimum traps sound",
           15, th.fg, anchor="start")
    s.text(80, 588, "rays launched within about ±12° of the axis stay trapped and can cross entire oceans",
           15, th.fg, anchor="start")


# ---------------------------------------------------------------------------
# Percussive pile-driving survey geometry (ISO 18406)
# ---------------------------------------------------------------------------

def _d_pile_driving_deployment(s: SVG, th: Theme) -> None:
    """ISO 18406 piling survey: the monopile and hammer, the 750 m measurement
    position, the allowed hydrophone depth band in the lower half of a 30 m
    water column, and the additional positions at three times the water depth."""
    import math
    surf, bed = 170.0, 440.0                 # 30 m of water over 270 px
    ppm = (bed - surf) / 30.0                # px per metre of depth
    px_, rx_ = 110.0, 490.0                  # pile and recorder abscissae
    left, right = 60.0, 570.0

    # --- Section view -------------------------------------------------------
    s.text(left + 4, 64, "Section: the minimum campaign", 16, th.fg, bold=True,
           anchor="start")

    dsur = f"M {left:.0f} {surf}"
    x = left
    while x < right:
        dsur += f" Q {x + 8:.0f} {surf - 5:.0f} {x + 16:.0f} {surf:.0f}"
        dsur += f" Q {x + 24:.0f} {surf + 5:.0f} {x + 32:.0f} {surf:.0f}"
        x += 32
    s.path(dsur, stroke=th.primary, sw=1.8)
    s.ground(bed, left, right)
    s.text(300, bed + 30, "seabed", 12, th.muted)

    # Water depth, inside the 4 m to 100 m scope of Clause 1.
    s.dim(left + 22, surf, left + 22, bed, "30 m", offset=0, size=14)

    # Monopile with its hydraulic hammer, driven below the bed.
    s.rect(px_ - 11, 118, 22, bed - 118 + 44, th.panel, th.fg, sw=2)
    s.rect(px_ - 25, 84, 50, 34, th.panel, th.fg, rx=3, sw=2)
    s.arrow(px_, 66, px_, 82, th.secondary, 3.0)
    s.text(px_ + 34, 100, "impact hammer", 13, th.fg, anchor="start")
    s.text(176, 214, "monopile", 13, th.fg, anchor="start")
    s.line(172, 209, px_ + 13, 209, th.muted, 0.9)
    s.line(px_ + 11, bed + 44, px_ + 60, bed + 44, th.muted, 0.9, dash="3,3")
    s.dim(px_ + 48, bed, px_ + 48, bed + 44, "penetration", offset=0, size=12,
          label_side="right")

    # Bubble curtain, ghosted: mitigation is optional and out of the model.
    for dx in (-32.0, 32.0):
        d = f"M {px_ + dx:.0f} {bed:.0f}"
        yb = bed
        while yb > surf:
            yb -= 18
            d += f" Q {px_ + dx * 1.4:.0f} {yb + 9:.0f} {px_ + dx:.0f} {yb:.0f}"
        s.path(d, stroke=th.muted, sw=1.2, dash="4,5")
    s.text(px_ + 52, 330, "bubble curtain, if used", 12, th.muted, anchor="start")
    s.line(px_ + 48, 326, px_ + 34, 312, th.muted, 0.9)

    # Range break: 750 m does not fit the depth scale, so the surface is cut.
    for bx in (286.0, 306.0):
        s.path(f"M {bx:.0f} {surf - 20:.0f} L {bx + 10:.0f} {surf - 2:.0f} "
               f"L {bx:.0f} {surf + 16:.0f}", stroke=th.bg, sw=7)
        s.path(f"M {bx:.0f} {surf - 20:.0f} L {bx + 10:.0f} {surf - 2:.0f} "
               f"L {bx:.0f} {surf + 16:.0f}", stroke=th.muted, sw=1.4)
    s.dim(px_, 134, rx_, 134, "as close as possible to 750 m", offset=0, size=15)
    s.line(px_, 120, px_, 140, th.muted, 0.9, dash="3,3")
    s.line(rx_, surf + 4, rx_, 140, th.muted, 0.9, dash="3,3")

    # The allowed hydrophone depth band: 2 m above the bed to half the depth.
    y_half, y_2m = surf + 15.0 * ppm, bed - 2.0 * ppm
    s.rect(rx_ - 40, y_half, right - rx_ + 40, y_2m - y_half, th.panel,
           th.muted, sw=1.0, dash="5,4")
    s.text(right - 2, y_half - 30, "lower half of the water column:", 12,
           th.fg, anchor="end")
    s.text(right - 2, y_half - 12, "2 m above the bed to ½ depth", 12, th.fg,
           anchor="end")

    # Bottom-mounted recorder with two hydrophones at 1/2 and 3/4 depth.
    s.rect(rx_ - 20, bed - 18, 40, 18, th.panel, th.fg, rx=2, sw=1.8)
    s.line(rx_, bed - 18, rx_, y_half, th.fg, 1.8)
    for frac, lab in ((0.5, "½ depth"), (0.75, "¾ depth")):
        hy = surf + 30.0 * frac * ppm
        s.circle(rx_, hy, 7, th.secondary)
        s.circle(rx_, hy, 2.5, th.bg)
        s.text(rx_ + 14, hy + 5, lab, 12, th.fg, anchor="start", mono=True)
    s.text(rx_ + 2, bed + 34, "bottom-mounted recorder", 12, th.muted)

    # Survey vessel and the noise the deployment itself makes.
    s.path("M 336 156 L 394 156 L 388 170 L 342 170 Z", fill=th.panel,
           stroke=th.fg, sw=1.6)
    s.rect(354, 142, 18, 14, th.panel, th.fg, rx=2, sw=1.4)
    s.arrow(248, 348, 352, 176, th.muted, 1.2)
    s.text(176, 372, "engines, generator and echo-sounder off;", 12,
           th.muted, anchor="start")
    s.text(176, 390, "flow noise, cable strum and surface heave", 12,
           th.muted, anchor="start")
    s.text(176, 408, "all read as signal", 12, th.muted, anchor="start")

    # --- Plan view ----------------------------------------------------------
    s.text(742, 64, "Plan: where else to measure", 16, th.fg, bold=True)
    cx_, cy_ = 742.0, 300.0
    s.circle(cx_, cy_, 34, th.panel, th.muted, 1.2)
    s.circle(cx_, cy_, 100, "none", th.secondary, 1.6)
    s.ellipse(cx_, cy_, 142, 142, "none", th.muted, 1.2, dash="6,5")
    s.circle(cx_, cy_, 9, th.panel, th.fg, 2)
    s.circle(cx_, cy_, 3, th.fg)
    s.text(cx_, cy_ - 18, "pile", 12, th.fg)
    for ang, rr, col in ((-58.0, 100.0, th.secondary), (28.0, 142.0, th.accent),
                         (170.0, 142.0, th.accent)):
        hx = cx_ + rr * math.cos(math.radians(ang))
        hy = cy_ + rr * math.sin(math.radians(ang))
        s.line(cx_, cy_, hx, hy, th.muted, 1.0, dash="4,4")
        s.circle(hx, hy, 6, col)
        s.circle(hx, hy, 2.2, th.bg)
    s.text(cx_, cy_ + 58, "3 × water depth = 90 m:", 11, th.muted)
    s.text(cx_, cy_ + 74, "nothing inside", 11, th.muted)
    s.text(cx_ + 64, cy_ - 92, "750 m", 13, th.secondary, anchor="start",
           mono=True)
    s.text(cx_, cy_ + 176, "further positions on a transect,", 13, th.fg)
    s.text(cx_, cy_ + 196, "clear of banks and trenches", 13, th.fg)
    s.text(cx_, cy_ + 216, "(radii not to scale)", 11, th.muted)

    # Normative context.
    s.text(70, 552, "Percussive driving only, in 4 m to 100 m of water: vibro- and sheet-piling are out of scope",
           15, th.fg, anchor="start")
    s.text(70, 580, "750 m is a comparability convention, not a regulatory limit; the actual range is reported, to 5 %",
           15, th.fg, anchor="start")
    s.text(70, 608, "The station records the entire driving sequence, soft start included, at one fixed range",
           15, th.fg, anchor="start")
    s.text(70, 636, "Record hydrophone depth, GPS, water depth and tide, "
                    "seabed class and energy per blow",
           15, th.fg, anchor="start")


# ---------------------------------------------------------------------------
# Sonar equation geometry (underwater/underwater-propagation)
# ---------------------------------------------------------------------------

def _d_sonar_equation(s: SVG, th: Theme) -> None:
    """The two sonar geometries the equations encode: a passive one-way path
    against ambient noise, and an active monostatic two-way path against a
    target strength and reverberation, with the reference conventions."""

    def _noise_field(x1: float, x2: float, y1: float, y2: float) -> None:
        """Isotropic ambient field, drawn as a sparse stipple."""
        n = 0
        yy = y1
        while yy < y2:
            xx = x1 + (28.0 if (n % 2) else 0.0)
            while xx < x2:
                s.line(xx, yy, xx + 7, yy - 5, th.muted, 0.8)
                xx += 56
            yy += 26
            n += 1

    # --- Passive ------------------------------------------------------------
    s.text(60, 70, "Passive: the target radiates, one way", 16, th.fg,
           bold=True, anchor="start")
    s.rect(60, 84, 780, 214, th.bg, th.muted, rx=4, sw=1.0)
    _noise_field(74, 700, 106, 250)
    s.text(300, 118, "ambient noise field NL", 13, th.muted)

    # Radiating target on the left.
    s.path("M 96 186 L 168 186 L 158 206 L 106 206 Z", fill=th.panel,
           stroke=th.fg, sw=1.8)
    s.rect(120, 172, 20, 14, th.panel, th.fg, rx=2, sw=1.4)
    for r_ in (18.0, 28.0, 38.0):
        s.path(f"M {132 - r_:.0f} 196 A {r_:.0f} {r_:.0f} 0 0 0 132 {196 + r_:.0f}",
               stroke=th.secondary, sw=1.2)
    s.text(132, 152, "SL", 15, th.secondary, bold=True)
    s.text(132, 252, "source level", 12, th.fg)

    # One-way path.
    s.arrow(184, 196, 596, 196, th.primary, 2.2)
    s.text(390, 182, "PL", 15, th.primary, bold=True)
    s.text(390, 224, "one-way propagation loss", 12, th.fg)

    # Receiving array with its beam.
    for k in range(5):
        s.circle(636, 160 + 18 * k, 5, th.fg)
    s.line(636, 152, 636, 244, th.fg, 1.6)
    s.path("M 636 196 L 566 168 L 566 224 Z", fill="none", stroke=th.accent,
           sw=1.6, dash="6,4")
    s.text(600, 268, "DI", 15, th.accent, bold=True)
    s.text(600, 288, "array gain", 12, th.fg)

    # Detector.
    s.rect(700, 168, 128, 56, th.panel, th.fg, rx=4, sw=1.6)
    s.text(764, 192, "detector", 13, th.fg)
    s.text(764, 212, "DT", 15, th.fg, bold=True, mono=True)
    s.arrow(668, 196, 698, 196, th.fg, 1.6)
    s.text(388, 282, "$SE = SL − PL − (NL − DI) − DT$", 14, th.fg)

    # --- Active monostatic --------------------------------------------------
    s.text(60, 350, "Active, monostatic: out and back", 16, th.fg, bold=True,
           anchor="start")
    s.rect(60, 364, 780, 224, th.bg, th.muted, rx=4, sw=1.0)
    s.line(74, 386, 826, 386, th.primary, 1.6)
    s.text(818, 380, "sea surface", 11, th.muted, anchor="end")
    s.ground(566, 74, 826, hatch=30)

    # Transmitter / receiver at one point.
    s.rect(96, 452, 34, 48, th.panel, th.fg, rx=3, sw=1.8)
    s.text(113, 434, "SL, DI, DT", 12, th.fg)
    s.text(113, 522, "transmit and", 12, th.fg)
    s.text(113, 540, "receive here", 12, th.fg)

    # Outbound and return legs.
    s.arrow(136, 464, 520, 464, th.primary, 2.0)
    s.text(330, 456, "PL out", 14, th.primary, bold=True)
    s.arrow(520, 490, 136, 490, th.primary, 2.0)
    s.text(330, 508, "PL back", 14, th.primary, bold=True)

    # Target with its backscattered lobe.
    s.ellipse(566, 477, 40, 15, th.panel, th.fg, 1.8)
    s.text(566, 440, "TS", 15, th.secondary, bold=True)
    s.text(566, 420, "target strength", 12, th.fg)
    for r_ in (20.0, 30.0):
        s.path(f"M {526 - r_ * 0.2:.0f} {477 - r_:.0f} A {r_:.0f} {r_:.0f} 0 0 0 "
               f"{526 - r_ * 0.2:.0f} {477 + r_:.0f}", stroke=th.secondary, sw=1.2)

    # Reverberation: surface, volume and bottom scattering all return.
    for sx, sy, ey in ((236.0, 388.0, 470.0), (274.0, 432.0, 474.0),
                       (192.0, 560.0, 500.0)):
        s.circle(sx, sy, 3, th.accent)
        s.arrow(sx, sy, 150, ey, th.accent, 1.2)
    s.text(316, 536, "$SE = SL − 2 PL + TS − (NL − DI) − DT$", 14, th.fg)
    s.text(660, 524, "RL: surface, volume and bottom scattering", 12,
           th.accent, anchor="middle")
    s.text(660, 548, "replaces $NL − DI$ when reverberation-limited", 12,
           th.fg, anchor="middle")

    # --- Reference conventions ---------------------------------------------
    s.text(70, 620, "Field levels are re 1 µPa; a source level carries the "
                    "squared metre of its range, re 1 µPa²m²",
           15, th.fg, anchor="start")
    s.text(70, 648, "Every term is in the same bandwidth; the figure of merit "
                    "is the PL that drives SE to zero, re m²",
           15, th.fg, anchor="start")


# ---------------------------------------------------------------------------
# The waveguide the four numerical solvers share (underwater/underwater-solvers)
# ---------------------------------------------------------------------------

def _d_underwater_waveguide(s: SVG, th: Theme) -> None:
    """Range-depth section with the coordinate frame, the two boundary
    conditions, the c(z) profile, the source and receiver depths, one turning
    ray, and what each of the four solvers computes in that same frame."""
    import math
    surf, bot = 120.0, 400.0
    x0, x1 = 250.0, 850.0

    # --- c(z) profile, in its own narrow panel on the left ------------------
    s.text(150, 96, "$c(z)$", 15, th.fg, bold=True)
    s.line(90, surf, 90, bot, th.muted, 1.4)
    s.line(90, bot, 210, bot, th.muted, 1.4)
    s.text(212, bot + 18, "$c$", 13, th.muted, anchor="start")
    axis_y = surf + 0.42 * (bot - surf)
    s.path(f"M 186 {surf:.0f} Q 120 {surf + 60:.0f} 116 {axis_y:.0f} "
           f"Q 132 {bot - 60:.0f} 196 {bot:.0f}", stroke=th.primary, sw=2.4)
    s.line(90, axis_y, 210, axis_y, th.muted, 1.0, dash="6,4")
    s.text(208, axis_y - 8, "channel axis", 11, th.muted, anchor="end")

    # --- The waveguide ------------------------------------------------------
    s.line(x0, surf, x1, surf, th.primary, 2.4)
    s.text(x0 + 8, surf - 12, "sea surface: pressure release, $p = 0$", 13, th.fg,
           anchor="start")
    s.ground(bot, x0, x1, hatch=30)
    # The bottom condition stays plain for now: in $dΨ/dz$ the composer
    # would set the differential d italic (single-letter default before a
    # Greek letter) while the dz run composes upright, and there is no
    # mechanism for the roman d of ISO 80000-2 next to Ψ.
    s.text(x0 + 8, bot + 30, "bottom: Ψ(D) = 0 (pressure release) or dΨ/dz = 0 (rigid)",
           13, th.fg, anchor="start")

    # Coordinate frame: r to the right, z down.
    s.arrow(x0, surf, x0 + 74, surf, th.fg, 1.8)
    s.text(x0 + 80, surf + 18, "$r$", 15, th.fg, anchor="start")
    s.arrow(x0, surf, x0, surf + 66, th.fg, 1.8)
    s.text(x0 - 12, surf + 66, "$z$", 15, th.fg, anchor="end")
    s.dim(228, surf, 228, bot, "$D$", offset=0, size=15)

    # Source, receiver and one turning ray.
    zs = surf + 0.30 * (bot - surf)
    zr = surf + 0.66 * (bot - surf)
    rr = 700.0
    s.circle(310, zs, 7, th.secondary)
    s.circle(310, zs, 2.4, th.bg)
    s.text(306, zs - 14, "source, $z_s$", 12, th.fg, anchor="middle")
    s.line(rr, surf, rr, bot, th.muted, 1.0, dash="5,4")
    s.circle(rr, zr, 7, th.accent)
    s.circle(rr, zr, 2.4, th.bg)
    s.text(rr + 14, zr + 24, "receiver, $z$", 12, th.fg, anchor="start")

    d = f"M 310 {zs:.1f}"
    zt = surf + 0.86 * (bot - surf)
    for i in range(1, 61):
        t = i / 60.0
        xr = 310 + t * 520
        zz = zs + (zt - zs) * math.sin(math.pi * t)
        d += f" L {xr:.1f} {zz:.1f}"
    s.path(d, stroke=th.secondary, sw=1.8)
    s.circle(570, zt, 4, th.secondary)
    s.text(520, zt + 24, "turning depth $z_t$:  $c(z_t) = c(z_s)/cos θ_0$", 12, th.fg)

    # --- What each solver returns in that frame -----------------------------
    s.text(96, 452, "modes:", 14, th.fg, bold=True, anchor="start")
    s.text(176, 452, "standing waves in $z$, travelling as $exp(i k_{rm} r)$", 14,
           th.fg, anchor="start")
    s.text(96, 478, "rays:", 14, th.fg, bold=True, anchor="start")
    s.text(176, 478, "trajectories, travel times, convergence zones", 14, th.fg,
           anchor="start")
    s.text(96, 504, "beams:", 14, th.fg, bold=True, anchor="start")
    s.text(176, 504, "those rays widened, summed into PL($z$, $r$)", 14, th.fg,
           anchor="start")
    s.text(96, 530, "PE:", 14, th.fg, bold=True, anchor="start")
    s.text(176, 530, "the envelope marched in $r$, one step $Δr$ at a time", 14,
           th.fg, anchor="start")

    s.text(96, 570, "All four take the same range-independent $c(z)$: no sediment attenuation, no bathymetry",
           15, th.fg, anchor="start")


# ---------------------------------------------------------------------------
# Marine-mammal exposure assessment (underwater/marine-mammal-exposure)
# ---------------------------------------------------------------------------

def _d_marine_mammal_exposure(s: SVG, th: Theme) -> None:
    """Where the spectrum is measured against where the criterion applies: the
    hydrophone at the ISO 18406 range, the animal further out, the three steps
    between them, and the two isopleths of the dual metric."""
    surf, bed = 150.0, 366.0
    px_, hx_, ax_ = 92.0, 250.0, 430.0

    # --- Section ------------------------------------------------------------
    s.text(60, 74, "Section: pile, hydrophone, animal", 16, th.fg, bold=True,
           anchor="start")
    dsur = f"M 60 {surf}"
    x = 60.0
    while x < 520:
        dsur += f" Q {x + 8:.0f} {surf - 5:.0f} {x + 16:.0f} {surf:.0f}"
        dsur += f" Q {x + 24:.0f} {surf + 5:.0f} {x + 32:.0f} {surf:.0f}"
        x += 32
    s.path(dsur, stroke=th.primary, sw=1.8)
    s.ground(bed, 60, 520, hatch=30)

    # Pile and hammer.
    s.rect(px_ - 9, 118, 18, bed - 118 + 26, th.panel, th.fg, sw=1.8)
    s.rect(px_ - 19, 92, 38, 26, th.panel, th.fg, rx=3, sw=1.8)
    s.arrow(px_, 76, px_, 90, th.secondary, 2.6)
    s.text(px_ + 22, 250, "pile", 12, th.fg, anchor="start")

    # Calibrated measurement hydrophone.
    s.line(hx_, surf, hx_, 300, th.fg, 1.6)
    s.circle(hx_, 300, 7, th.secondary)
    s.circle(hx_, 300, 2.4, th.bg)
    s.text(hx_, 326, "calibrated hydrophone", 12, th.fg)
    s.text(hx_, 344, "band SEL measured here", 12, th.secondary)
    s.dim(px_, 128, hx_, 128, "750 m", offset=0, size=13)

    # The receiving animal.
    s.ellipse(ax_, 262, 40, 15, th.panel, th.fg, 1.8)
    s.path(f"M {ax_ + 4:.0f} 249 L {ax_ + 16:.0f} 232 L {ax_ + 20:.0f} 251 Z",
           fill=th.panel, stroke=th.fg, sw=1.6)
    s.path(f"M {ax_ + 36:.0f} 262 L {ax_ + 62:.0f} 254 L {ax_ + 62:.0f} 270 Z",
           fill=th.panel, stroke=th.fg, sw=1.6)
    s.circle(ax_ - 26, 258, 2.4, th.fg)
    s.text(ax_ + 30, 214, "the criterion applies here", 12, th.accent)
    s.arrow(hx_ + 16, 274, ax_ - 46, 262, th.primary, 1.8)
    s.text(ax_ - 74, 296, "$PL(f, R)$", 13, th.primary, bold=True, anchor="middle")
    s.dim(hx_, 128, ax_, 128, "range $R$", offset=0, size=13)

    # --- The three steps between the two ------------------------------------
    s.rect(60, 400, 460, 132, th.panel, th.muted, rx=4, sw=1.0)
    boxes = (("per-band SEL", 120.0), ("$× W(f)$", 260.0),
             ("$+ 10 lg N$", 400.0))
    for label, bx in boxes:
        s.rect(bx - 58, 418, 116, 40, th.bg, th.fg, rx=4, sw=1.6)
        s.text(bx, 443, label, 13, th.fg)
    for bx in (180.0, 320.0):
        s.arrow(bx, 438, bx + 20, 438, th.fg, 1.6)
    s.text(290, 486, "weighted cumulative SEL  vs  AUD INJ / TTS", 13, th.fg)
    s.text(290, 510, "unweighted peak SPL  vs  the flat criterion", 13, th.fg)

    # --- Plan: the two isopleths -------------------------------------------
    s.text(700, 74, "Plan: the two isopleths", 16, th.fg, bold=True)
    cx_, cy_ = 700.0, 260.0
    s.circle(cx_, cy_, 66, "none", th.secondary, 1.8)
    s.circle(cx_, cy_, 116, "none", th.accent, 2.0)
    s.circle(cx_, cy_, 8, th.panel, th.fg, 2)
    s.circle(cx_, cy_, 3, th.fg)
    s.text(cx_, cy_ - 18, "pile", 12, th.fg)
    s.text(cx_, cy_ + 88, "peak SPL isopleth", 11, th.secondary)
    s.text(cx_, cy_ + 138, "weighted cumulative SEL isopleth", 11, th.accent)
    s.text(cx_, cy_ + 168, "the larger isopleth governs", 14, th.fg, bold=True)
    s.text(cx_, cy_ + 196, "VHF, impulsive, NMFS 2024:", 12, th.fg)
    s.text(cx_, cy_ + 216, "144 / 159 dB re 1 µPa²·s weighted", 12, th.fg)
    s.text(cx_, cy_ + 236, "196 / 202 dB re 1 µPa flat", 12, th.fg)

    # Normative context.
    s.text(70, 566, "An isopleth is the contour where a criterion is exactly "
                    "met: the answer is a radius, not a verdict",
           15, th.fg, anchor="start")
    s.text(70, 594, "A cumulative level means nothing without its range and "
                    "the window the strikes were counted in",
           15, th.fg, anchor="start")
