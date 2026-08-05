#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Diagrams of the aircraft guides: fixed-wing and rotorcraft certification.

One subject: the certification geometry of ICAO Annex 16. Both diagrams draw
the same thing, a flight path over reference points on the ground, and they
share the side-view aircraft glyph that marks the aircraft along it.
"""

from __future__ import annotations

from .canvas import SVG, Theme

# ---------------------------------------------------------------------------
# Aircraft noise certification points (ICAO Annex 16 Vol. I, Chapter 3)
# ---------------------------------------------------------------------------

def _plane_glyph(s: SVG, x: float, y: float, deg: float,
                 size: float = 1.0) -> None:
    """Small side-view jet silhouette pointing along +x, rotated ``deg``."""
    th = s.th
    s.add(f'<g transform="translate({x:.1f} {y:.1f}) rotate({deg:.1f}) '
          f'scale({size:.2f})">'
          f'<path d="M -24 0 Q -24 -4 -18 -4 L 16 -4 Q 26 -2 26 0 Q 26 2 '
          f'16 3 L -18 3 Q -24 3 -24 0 Z" fill="{th.fg}"/>'
          f'<path d="M -22 -3 L -13 -14 L -7 -3 Z" fill="{th.fg}"/>'
          f'<path d="M 2 0 L -10 9 L -3 0 Z" fill="{th.fg}"/></g>')


def _d_aircraft_certification(s: SVG, th: Theme) -> None:
    """The three ICAO Annex 16 Vol. I Chapter 3 reference points around the
    runway: lateral (450 m line), flyover (6.5 km from start of roll) and
    approach (2 000 m from the threshold, 120 m under the 3-degree path).
    Plan and side views share the same x mapping (0.062 px per metre)."""
    yc = 185.0                            # plan-view runway centre line
    x_sor = 330.0                         # start of roll / threshold
    x_fly = x_sor + 6500.0 * 0.062       # flyover point, 6.5 km
    x_app = x_sor - 2000.0 * 0.062       # approach point, 2 km out

    # --- plan view ---------------------------------------------------------
    s.text(75.0, 80.0, "Plan view", 17, th.fg, bold=True, anchor="start")
    s.line(70.0, yc, x_sor, yc, th.muted, 1.1, dash="7,5")
    s.line(516.0, yc, 850.0, yc, th.muted, 1.1, dash="7,5")
    s.rect(x_sor, yc - 9, 186, 18, th.panel, th.fg, sw=1.8)
    _plane_glyph(s, 366.0, yc, 0.0, 0.75)
    s.arrow(535.0, 163.0, 605.0, 163.0, th.accent, 2.0)
    s.text(570.0, 150.0, "take-off", 14, th.accent)
    s.text(338.0, 168.0, "start of roll", 14, th.muted, anchor="start")

    # Reference distances along the extended centre line.
    s.dim(x_app, yc, x_sor, yc, "2 000 m", offset=-48, size=16)
    s.dim(x_sor, yc, x_fly, yc, "6 500 m", offset=-48, size=16)

    # Flyover and approach points.
    for px_ in (x_fly, x_app):
        s.circle(px_, yc, 6.0, th.secondary)
        s.circle(px_, yc, 2.2, th.bg)
    s.text(x_fly, 218.0, "Flyover reference point", 16, th.fg, bold=True)
    s.text(x_app, 218.0, "Approach reference point", 16, th.fg, bold=True)

    # Lateral line 450 m from the runway centre line, and its mirror.
    s.line(340.0, 278.0, 740.0, 278.0, th.muted, 1.3, dash="6,5")
    for lxp, filled in ((460.0, False), (560.0, True), (660.0, False)):
        if filled:
            s.circle(lxp, 278.0, 6.0, th.secondary)
            s.circle(lxp, 278.0, 2.2, th.bg)
        else:
            s.circle(lxp, 278.0, 5.0, th.bg, th.fg, 1.5)
    s.text(540.0, 306.0, "Lateral reference line", 16, th.fg, bold=True)
    s.text(540.0, 326.0, "where take-off noise is greatest", 14, th.muted)
    s.dim(620.0, yc, 620.0, 278.0, "450 m", offset=0, size=15,
          label_side="right")
    s.line(340.0, 92.0, 740.0, 92.0, th.muted, 1.0, dash="3,5")
    s.circle(560.0, 92.0, 5.0, th.bg, th.fg, 1.5)
    s.text(540.0, 76.0, "symmetric lateral point (measured on both sides)",
           14, th.muted)

    # --- side view (heights exaggerated; distances to scale) ---------------
    gy = 488.0
    s.text(75.0, 434.0, "Side view", 17, th.fg, bold=True, anchor="start")
    s.ground(gy, 60.0, 850.0)
    s.rect(x_sor, gy - 4, 186, 5, th.muted)
    # Approach: 3-degree glide path meeting the ground 300 m past the
    # threshold; the reference point is 120 m below it.
    xg = x_sor + 300.0 * 0.062
    s.line(100.0, gy - (xg - 100.0) * 0.465, xg, gy, th.secondary, 2.2)
    _plane_glyph(s, 150.0, gy - (xg - 150.0) * 0.465 - 9.0, 25.0, 0.9)
    s.text(150.0, 348.0, "approach", 15, th.secondary, bold=True)
    s.path(f"M {xg - 40:.1f} {gy:.1f} A 40 40 0 0 1 "
           f"{xg - 40 * 0.906:.1f} {gy - 40 * 0.423:.1f}",
           stroke=th.muted, sw=1.2)
    s.text(xg - 52.0, gy - 14.0, "3°", 13, th.muted)
    s.mic(x_app, gy - 24.0, gy, 0.7)
    s.dim(x_app + 22, gy, x_app + 22, gy - (xg - x_app) * 0.465, "120 m",
          offset=0, size=14, label_side="right")
    s.line(x_app, gy - (xg - x_app) * 0.465, x_app + 22,
           gy - (xg - x_app) * 0.465, th.muted, 0.9, dash="3,3")
    # Take-off: ground roll, then climb; the flyover microphone sits under
    # the climb-out at 6.5 km.
    s.line(410.0, gy, 850.0, gy - 132.0, th.accent, 2.2)
    _plane_glyph(s, 700.0, gy - 87.0 - 9.0, -16.7, 0.9)
    s.text(700.0, 352.0, "take-off", 15, th.accent, bold=True)
    s.mic(x_fly, gy - 24.0, gy, 0.7)
    s.line(x_fly, gy - 30.0, x_fly, gy - (x_fly - 410.0) * 0.30 + 6.0,
           th.muted, 1.0, dash="4,4")

    # --- normative context -------------------------------------------------
    s.text(80.0, 552.0,
           "Microphones 1.2 m above the ground; the certification metric at the three points is EPNL, in EPNdB",
           17, th.fg, anchor="start")
    s.text(80.0, 580.0,
           "Lateral: full take-off power · Flyover: 6.5 km from brake release · Approach: 3° ± 0.5° glide path",
           16, th.fg, anchor="start")
    s.text(80.0, 608.0,
           "the approach point lies 120 m below the 3° path, which meets the ground 300 m beyond the threshold",
           16, th.muted, anchor="start")


# ---------------------------------------------------------------------------
# Helicopter overflight certification (ICAO Annex 16 Vol. I, Chapter 8)
# ---------------------------------------------------------------------------

def _d_rotorcraft_certification(s: SVG, th: Theme) -> None:
    """Chapter 8 overflight: level flight at 150 m over the central
    microphone with two sideline microphones 150 m to each side (plan
    inset). Side view to scale at about 0.47 px per metre vertically."""
    gy = 470.0
    hx, hy = 300.0, 150.0                # helicopter on the flight path

    # --- side view ---------------------------------------------------------
    s.ground(gy, 40.0, 560.0)
    s.line(70.0, hy, 530.0, hy, th.muted, 1.3, dash="8,6")
    s.arrow(530.0, hy, 556.0, hy, th.fg, 2.0)
    s.text(72.0, 112.0, "level flight at 0.9 VH", 16, th.fg, anchor="start")

    # Helicopter silhouette (flying to the right).
    s.line(240.0, 126.0, 360.0, 126.0, th.fg, 3.0)          # main rotor
    s.line(hx, 126.0, hx, 140.0, th.fg, 2.2)                # mast
    s.ellipse(hx, 152.0, 27.0, 14.0, th.panel, th.fg, 2.0)  # cabin
    s.line(274.0, 149.0, 218.0, 143.0, th.fg, 2.6)          # tail boom
    s.line(218.0, 132.0, 218.0, 152.0, th.fg, 2.0)          # tail rotor
    s.line(286.0, 166.0, 286.0, 174.0, th.fg, 1.8)          # skid struts
    s.line(316.0, 166.0, 316.0, 174.0, th.fg, 1.8)
    s.line(276.0, 174.0, 328.0, 174.0, th.fg, 2.2)          # skid
    for r in (30, 52, 74):
        s.path(f"M {hx - r * 0.95:.1f} {176 + r * 0.30:.1f} A {r} {r} 0 0 0 "
               f"{hx - r * 0.30:.1f} {176 + r * 0.95:.1f}",
               stroke=th.muted, sw=1.2)
        s.path(f"M {hx + r * 0.30:.1f} {176 + r * 0.95:.1f} A {r} {r} 0 0 0 "
               f"{hx + r * 0.95:.1f} {176 + r * 0.30:.1f}",
               stroke=th.muted, sw=1.2)

    # Height above the central microphone.
    s.dim(390.0, gy, 390.0, hy, "150 m (492 ft)", offset=0, size=16,
          label_side="right")
    s.mic(hx, gy - 26.0, gy, 0.8)
    s.text(hx, 508.0, "centre microphone", 15, th.fg)

    # --- plan inset: the three-microphone line -----------------------------
    s.text(735.0, 96.0, "Plan view", 17, th.fg, bold=True)
    s.arrow(620.0, 190.0, 860.0, 190.0, th.fg, 2.0)
    s.text(852.0, 176.0, "track", 14, th.muted, anchor="end")
    s.ellipse(680.0, 190.0, 15.0, 15.0, "none", th.muted, 1.2)
    s.circle(680.0, 190.0, 4.5, th.fg)
    s.line(735.0, 120.0, 735.0, 260.0, th.muted, 1.1, dash="5,4")
    for my_ in (120.0, 190.0, 260.0):
        s.circle(735.0, my_, 6.0, th.secondary)
        s.circle(735.0, my_, 2.2, th.bg)
    s.dim(772.0, 120.0, 772.0, 190.0, "150 m", offset=0, size=14,
          label_side="right")
    s.dim(772.0, 190.0, 772.0, 260.0, "150 m", offset=0, size=14,
          label_side="right")
    for wy_ in (120.0, 190.0, 260.0):
        s.line(741.0, wy_, 772.0, wy_, th.muted, 0.9, dash="3,3")
    s.text(725.0, 296.0, "3 microphones on a line perpendicular to the track",
           14, th.muted)

    # --- normative context -------------------------------------------------
    s.text(80.0, 540.0,
           "Speed: the least of 0.9 VH, 0.9 VNE, 0.45 VH + 120 km/h and 0.45 VNE + 120 km/h",
           16, th.fg, anchor="start")
    s.text(80.0, 566.0,
           "EPNL in EPNdB at the three points; at least six overflights, headwind and tailwind in equal number",
           16, th.fg, anchor="start")
    s.text(80.0, 592.0,
           "microphones 1.2 m above ground; the sideline pair sees the overhead helicopter at 45° (slant ≈ 212 m)",
           16, th.muted, anchor="start")
