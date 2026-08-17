#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Diagrams of the aircraft guides: certification, segments and hemispheres.

One subject: where an aircraft is when it is measured or modelled. The
certification plates draw the reference points of ICAO Annex 16 and the
station a technician builds under them; the modelling plates draw the two
source geometries the contour methods rest on, the ECAC Doc 29 flight-path
segment and the ECAC Doc 32 noise hemisphere. They share the side-view
aircraft glyph that marks the aircraft along a path.
"""

from __future__ import annotations

import math

from .canvas import SVG, Theme


def _arc(s: SVG, cx: float, cy: float, r: float, a0: float, a1: float,
         stroke: str, sw: float = 1.4, dash: str = "") -> None:
    """Circular arc from ``a0`` to ``a1``, degrees anticlockwise from +x.

    Screen y grows downward, so the angles are the ones a reader measures on
    the drawing: 0 is to the right and 90 is straight up.
    """
    x0, y0 = cx + r * math.cos(math.radians(a0)), cy - r * math.sin(math.radians(a0))
    x1, y1 = cx + r * math.cos(math.radians(a1)), cy - r * math.sin(math.radians(a1))
    large = 1 if abs(a1 - a0) > 180.0 else 0
    sweep = 0 if a1 > a0 else 1
    s.path(f"M {x0:.1f} {y0:.1f} A {r:.1f} {r:.1f} 0 {large} {sweep} "
           f"{x1:.1f} {y1:.1f}", stroke=stroke, sw=sw, dash=dash)


def _polar(cx: float, cy: float, r: float, a_deg: float) -> tuple[float, float]:
    """Point at radius ``r`` and angle ``a_deg`` (anticlockwise from +x)."""
    return cx + r * math.cos(math.radians(a_deg)), cy - r * math.sin(math.radians(a_deg))

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
    s.text(75.0, 80.0, "Plan view", 15, th.fg, bold=True, anchor="start")
    s.line(70.0, yc, x_sor, yc, th.muted, 1.1, dash="7,5")
    s.line(516.0, yc, 850.0, yc, th.muted, 1.1, dash="7,5")
    s.rect(x_sor, yc - 9, 186, 18, th.panel, th.fg, sw=1.8)
    _plane_glyph(s, 366.0, yc, 0.0, 0.75)
    s.arrow(535.0, 163.0, 605.0, 163.0, th.accent, 2.0)
    s.text(570.0, 150.0, "take-off", 12, th.accent)
    s.text(338.0, 168.0, "start of roll", 12, th.muted, anchor="start")

    # Reference distances along the extended centre line.
    s.dim(x_app, yc, x_sor, yc, "2 000 m", offset=-48, size=14)
    s.dim(x_sor, yc, x_fly, yc, "6 500 m", offset=-48, size=14)

    # Flyover and approach points.
    for px_ in (x_fly, x_app):
        s.circle(px_, yc, 6.0, th.secondary)
        s.circle(px_, yc, 2.2, th.bg)
    s.text(x_fly, 218.0, "Flyover reference point", 14, th.fg, bold=True)
    s.text(x_app, 218.0, "Approach reference point", 14, th.fg, bold=True)

    # Lateral line 450 m from the runway centre line, and its mirror.
    s.line(340.0, 278.0, 740.0, 278.0, th.muted, 1.3, dash="6,5")
    for lxp, filled in ((460.0, False), (560.0, True), (660.0, False)):
        if filled:
            s.circle(lxp, 278.0, 6.0, th.secondary)
            s.circle(lxp, 278.0, 2.2, th.bg)
        else:
            s.circle(lxp, 278.0, 5.0, th.bg, th.fg, 1.5)
    s.text(540.0, 306.0, "Lateral reference line", 14, th.fg, bold=True)
    s.text(540.0, 326.0, "where take-off noise is greatest", 12, th.muted)
    s.dim(620.0, yc, 620.0, 278.0, "450 m", offset=0, size=13,
          label_side="right")
    s.line(340.0, 92.0, 740.0, 92.0, th.muted, 1.0, dash="3,5")
    s.circle(560.0, 92.0, 5.0, th.bg, th.fg, 1.5)
    s.text(540.0, 76.0, "symmetric lateral point (measured on both sides)",
           12, th.muted)

    # --- side view (heights exaggerated; distances to scale) ---------------
    gy = 488.0
    s.text(75.0, 434.0, "Side view", 15, th.fg, bold=True, anchor="start")
    s.ground(gy, 60.0, 850.0)
    s.rect(x_sor, gy - 4, 186, 5, th.muted)
    # Approach: 3-degree glide path meeting the ground 300 m past the
    # threshold; the reference point is 120 m below it.
    xg = x_sor + 300.0 * 0.062
    s.line(100.0, gy - (xg - 100.0) * 0.465, xg, gy, th.secondary, 2.2)
    _plane_glyph(s, 150.0, gy - (xg - 150.0) * 0.465 - 9.0, 25.0, 0.9)
    s.text(150.0, 348.0, "approach", 13, th.secondary, bold=True)
    s.path(f"M {xg - 40:.1f} {gy:.1f} A 40 40 0 0 1 "
           f"{xg - 40 * 0.906:.1f} {gy - 40 * 0.423:.1f}",
           stroke=th.muted, sw=1.2)
    s.text(xg - 52.0, gy - 14.0, "3°", 11, th.muted)
    s.mic(x_app, gy - 24.0, gy, 0.7)
    s.dim(x_app + 22, gy, x_app + 22, gy - (xg - x_app) * 0.465, "120 m",
          offset=0, size=12, label_side="right")
    s.line(x_app, gy - (xg - x_app) * 0.465, x_app + 22,
           gy - (xg - x_app) * 0.465, th.muted, 0.9, dash="3,3")
    # Take-off: ground roll, then climb; the flyover microphone sits under
    # the climb-out at 6.5 km.
    s.line(410.0, gy, 850.0, gy - 132.0, th.accent, 2.2)
    _plane_glyph(s, 700.0, gy - 87.0 - 9.0, -16.7, 0.9)
    s.text(700.0, 352.0, "take-off", 13, th.accent, bold=True)
    s.mic(x_fly, gy - 24.0, gy, 0.7)
    s.line(x_fly, gy - 30.0, x_fly, gy - (x_fly - 410.0) * 0.30 + 6.0,
           th.muted, 1.0, dash="4,4")

    # --- normative context -------------------------------------------------
    s.text(80.0, 552.0,
           "Microphones 1.2 m above ground; the certification metric at the three points is EPNL, in EPNdB",
           15, th.fg, anchor="start")
    s.text(80.0, 580.0,
           "Lateral: full take-off power · Flyover: 6.5 km from brake release · Approach: 3° ± 0.5° glide path",
           14, th.fg, anchor="start")
    s.text(80.0, 608.0,
           "the approach point lies 120 m below the 3° path, which meets the ground 300 m past the threshold",
           14, th.muted, anchor="start")


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
    s.text(72.0, 112.0, "level flight at 0.9 $V_H$", 14, th.fg, anchor="start")

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
    s.dim(390.0, gy, 390.0, hy, "150 m (492 ft)", offset=0, size=14,
          label_side="right")
    s.mic(hx, gy - 26.0, gy, 0.8)
    s.text(hx, 508.0, "centre microphone", 13, th.fg)

    # --- plan inset: the three-microphone line -----------------------------
    s.text(735.0, 96.0, "Plan view", 15, th.fg, bold=True)
    s.arrow(620.0, 190.0, 860.0, 190.0, th.fg, 2.0)
    s.text(852.0, 176.0, "track", 12, th.muted, anchor="end")
    s.ellipse(680.0, 190.0, 15.0, 15.0, "none", th.muted, 1.2)
    s.circle(680.0, 190.0, 4.5, th.fg)
    s.line(735.0, 120.0, 735.0, 260.0, th.muted, 1.1, dash="5,4")
    for my_ in (120.0, 190.0, 260.0):
        s.circle(735.0, my_, 6.0, th.secondary)
        s.circle(735.0, my_, 2.2, th.bg)
    s.dim(772.0, 120.0, 772.0, 190.0, "150 m", offset=0, size=12,
          label_side="right")
    s.dim(772.0, 190.0, 772.0, 260.0, "150 m", offset=0, size=12,
          label_side="right")
    for wy_ in (120.0, 190.0, 260.0):
        s.line(741.0, wy_, 772.0, wy_, th.muted, 0.9, dash="3,3")
    s.text(725.0, 296.0, "3 microphones on a line across the track",
           12, th.muted)

    # --- normative context -------------------------------------------------
    s.text(80.0, 540.0,
           "Speed: the least of 0.9 $V_H$, 0.9 $V_{NE}$, 0.45 $V_H$ + 120 km/h and 0.45 $V_{NE}$ + 120 km/h",
           14, th.fg, anchor="start")
    s.text(80.0, 566.0,
           "EPNL in EPNdB at the three points; at least six overflights, headwind and tailwind in equal number",
           14, th.fg, anchor="start")
    s.text(80.0, 592.0,
           "microphones 1.2 m above ground; the sideline pair sees the "
           "helicopter overhead at 45° (≈ 212 m)",
           14, th.muted, anchor="start")


# ---------------------------------------------------------------------------
# Doc 29 flight-path segment geometry (ECAC Doc 29 Vol. 2, Figs. 4-2, 4-3, 4-8)
# ---------------------------------------------------------------------------

def _plan_plane_glyph(s: SVG, x: float, y: float, size: float = 1.0) -> None:
    """Small plan-view jet silhouette with the nose pointing along +x."""
    th = s.th
    s.add(f'<g transform="translate({x:.1f} {y:.1f}) scale({size:.2f})">'
          f'<path d="M -26 0 Q -26 -3 -20 -3 L 20 -3 Q 28 -1.5 28 0 Q 28 1.5 '
          f'20 3 L -20 3 Q -26 3 -26 0 Z" fill="{th.fg}"/>'
          f'<path d="M 2 -2 L -16 -22 L -6 -22 L 8 -2 Z" fill="{th.fg}"/>'
          f'<path d="M 2 2 L -16 22 L -6 22 L 8 2 Z" fill="{th.fg}"/>'
          f'<path d="M -22 -2 L -30 -11 L -24 -11 L -18 -2 Z" fill="{th.fg}"/>'
          f'<path d="M -22 2 L -30 11 L -24 11 L -18 2 Z" fill="{th.fg}"/></g>')


def _segment_panel(s: SVG, th: Theme, x_s1: float, y_path: float, q_px: float,
                   dp_px: float, length_px: float) -> tuple[float, float]:
    """One flight-path segment with its observer; returns the observer point.

    ``q_px`` is signed exactly as Doc 29 signs ``q``: measured from S1 along
    the flight direction to the foot of the perpendicular, negative when the
    observer lies behind the segment.
    """
    x_s2 = x_s1 + length_px
    x_sp = x_s1 + q_px
    ox, oy = x_sp, y_path + dp_px
    # The infinite flight path the NPD data describe, and the finite segment
    # cut out of it.
    s.line(min(x_s1, x_sp) - 46.0, y_path, x_s2 + 52.0, y_path, th.muted, 1.1,
           dash="4,5")
    s.arrow(x_s2 + 30.0, y_path, x_s2 + 56.0, y_path, th.muted, 1.4)
    s.line(x_s1, y_path, x_s2, y_path, th.accent, 3.4)
    for xp, lab in ((x_s1, "$S_1$"), (x_s2, "$S_2$")):
        s.circle(xp, y_path, 5.0, th.accent)
        s.text(xp, y_path - 12.0, lab, 13, th.fg, bold=True)
    s.circle(x_sp, y_path, 4.0, th.bg, th.fg, 1.6)
    s.text(x_sp, y_path - 12.0, "$S_p$", 13, th.muted)
    s.circle(ox, oy, 6.0, th.secondary)
    s.text(ox + 14.0, oy + 6.0, "$O$", 14, th.fg, bold=True, anchor="start")
    return ox, oy


def _d_doc29_segment_geometry(s: SVG, th: Theme) -> None:
    """The geometry every ECAC Doc 29 per-segment correction is a function of.

    Four panels on the worked departure of the airport-noise guide: the
    observer alongside a segment and behind it (Vol. 2 Fig. 4-2b/4-2a, drawn
    to scale at 0.30 px per metre in the plane containing the segment and the
    observer), the plane normal to the flight path where the elevation, bank
    and depression angles live (Fig. 4-3), and the start-of-roll azimuth in
    plan (Fig. 4-8)."""
    sc = 0.30                                   # px per metre, panels (a)/(b)

    # --- (a) observer alongside the segment (Fig. 4-2b) --------------------
    s.text(60.0, 70.0, "(a) Observer alongside the segment", 15, th.fg,
           bold=True, anchor="start")
    ya = 156.0
    ox, oy = _segment_panel(s, th, 118.0, ya, 214.0 * sc, 526.0 * sc, 464.0 * sc)
    s.line(118.0, ya, ox, oy, th.muted, 1.2, dash="5,4")          # d1
    s.line(118.0 + 464.0 * sc, ya, ox, oy, th.muted, 1.2, dash="5,4")   # d2
    s.line(ox, ya, ox, oy, th.primary, 2.0)                        # dp
    s.text(ox + 20.0, (ya + oy) / 2 + 5.0, "$d_p$ = 526 m", 13, th.primary,
           anchor="start")
    s.text(150.0, (ya + oy) / 2 + 26.0, "$d_1$ = 568 m", 12, th.muted, anchor="end")
    s.text(300.0, (ya + oy) / 2 + 26.0, "$d_2$ = 582 m", 12, th.muted, anchor="start")
    s.dim(118.0, ya - 32.0, ox, ya - 32.0, "$q$ = 214 m", offset=0, size=12)
    s.dim(118.0, ya - 60.0, 118.0 + 464.0 * sc, ya - 60.0, "$λ$ = 464 m",
          offset=0, size=12)
    s.text(60.0, 350.0, "$0 ≤ q ≤ λ$, so $d_s = d_p$ and an exposure level",
           12, th.muted, anchor="start")
    s.text(60.0, 370.0, "reads the NPD table at $d_p$ (§4.4.1)", 12, th.muted,
           anchor="start")

    # --- (b) observer behind the segment (Fig. 4-2a) ------------------------
    s.text(478.0, 70.0, "(b) Observer behind the segment", 15, th.fg,
           bold=True, anchor="start")
    x_s1b = 626.0
    obx, oby = _segment_panel(s, th, x_s1b, ya, -300.0 * sc, 520.0 * sc,
                              464.0 * sc)
    s.line(x_s1b, ya, obx, oby, th.secondary, 2.0)                 # ds = d1
    s.text(obx + 22.0, (ya + oby) / 2 + 30.0, "$d_s = d_1$ = 600 m", 13,
           th.secondary, anchor="start")
    s.line(obx, ya, obx, oby, th.primary, 1.4, dash="4,4")         # dp
    s.text(obx - 10.0, (ya + oby) / 2 - 22.0, "$d_p$ = 520 m", 12, th.primary,
           anchor="end")
    s.dim(obx, ya - 32.0, x_s1b, ya - 32.0, "$q$ = −300 m", offset=0, size=12)
    s.text(478.0, 350.0, "$q < 0$, so the exposure level reads the table",
           12, th.muted, anchor="start")
    s.text(478.0, 370.0, "at $d_s$ behind a take-off roll and at $d_p$ elsewhere",
           12, th.muted, anchor="start")

    # --- (c) the plane normal to the flight path (Fig. 4-3) -----------------
    s.text(60.0, 412.0, "(c) In the plane normal to the flight path", 15, th.fg,
           bold=True, anchor="start")
    gy = 578.0
    s.ground(gy, 66.0, 436.0)
    obs_x = 406.0
    ac_x, ac_y = 106.0, gy - 300.0 * math.tan(math.radians(18.0))
    s.line(ac_x, ac_y, obs_x, gy, th.primary, 2.2)                 # propagation path
    s.line(ac_x - 62.0 * math.cos(math.radians(15.0)),
           ac_y + 62.0 * math.sin(math.radians(15.0)),
           ac_x + 62.0 * math.cos(math.radians(15.0)),
           ac_y - 62.0 * math.sin(math.radians(15.0)), th.accent, 3.0)
    s.line(ac_x - 66.0, ac_y, ac_x + 78.0, ac_y, th.muted, 1.0, dash="4,4")
    s.ellipse(ac_x, ac_y, 13.0, 9.0, th.panel, th.fg, 2.0)
    _arc(s, ac_x, ac_y, 44.0, 0.0, 15.0, th.accent, 1.4)
    s.text(ac_x + 66.0, ac_y - 22.0, "$ε$ = 15°", 12, th.accent, anchor="start")
    _arc(s, ac_x, ac_y, 84.0, 15.0, -18.0, th.fg, 1.4)
    s.text(ac_x + 96.0, ac_y + 14.0, "$φ = β + ε$ = 33°", 13, th.fg, anchor="start")
    _arc(s, obs_x, gy, 62.0, 180.0, 162.0, th.primary, 1.4)
    s.text(obs_x - 86.0, gy - 10.0, "$β$ = 18°", 13, th.primary, anchor="end")
    s.mic(obs_x, gy - 22.0, gy, 0.6)
    s.text(obs_x + 22.0, gy + 20.0, "receiver, 1.2 m", 12, th.muted,
           anchor="middle")
    s.text(62.0, 448.0, "wing plane", 12, th.accent, anchor="start")

    # --- (d) start-of-roll azimuth in plan (Fig. 4-8) -----------------------
    s.text(478.0, 412.0, "(d) Behind the take-off roll, in plan", 15, th.fg,
           bold=True, anchor="start")
    cx, cy = 700.0, 522.0
    s.rect(cx, cy - 9.0, 138.0, 18.0, th.panel, th.fg, sw=1.6)
    s.text(cx + 70.0, cy - 18.0, "runway", 12, th.muted)
    _plan_plane_glyph(s, cx + 26.0, cy, 0.62)
    s.line(cx, cy, cx + 150.0, cy, th.muted, 1.1, dash="4,4")
    for r in (56.0, 96.0):
        _arc(s, cx, cy, r, 92.0, 268.0, th.muted, 1.0, dash="3,4")
    rx, ry = _polar(cx, cy, 96.0, 120.0)
    s.line(cx, cy, rx, ry, th.secondary, 2.0)
    s.circle(rx, ry, 6.0, th.secondary)
    s.text(rx - 4.0, ry - 12.0, "$O$", 13, th.fg, bold=True, anchor="end")
    _arc(s, cx, cy, 36.0, 0.0, 120.0, th.fg, 1.4)
    s.text(cx + 18.0, cy - 46.0, "$ψ$ = 120°", 13, th.fg, anchor="start")
    s.text(cx - 66.0, cy - 26.0, "$d_{SOR}$", 12, th.secondary, anchor="end")
    s.line(cx, cy, cx - 110.0, cy, th.muted, 1.0, dash="3,4")
    s.text(cx - 118.0, cy + 5.0, "180°", 11, th.muted, anchor="end")
    s.text(cx + 150.0, cy + 30.0, "0° nose", 11, th.muted, anchor="middle")

    # --- what the angles are for --------------------------------------------
    s.text(60.0, 636.0,
           "$β$ elevation of the path over the ground line · $ε$ bank, positive with the starboard wing up · $φ = β + ε$ to",
           13, th.muted, anchor="start")
    s.text(60.0, 658.0,
           "starboard and $β − ε$ to port · $ψ = arccos(q/d_{SOR})$, 90° abeam to 180° astern, the jet lobe peaking near 120°",
           13, th.muted, anchor="start")
    s.text(60.0, 680.0,
           "NPD lookup: $d_p$ for exposure, $d_s$ for maxima, floored at "
           "30 m; $Δ_{SOR}$ scaled by 762 m/$d_{SOR}$ beyond 762 m",
           13, th.muted, anchor="start")


# ---------------------------------------------------------------------------
# One certification measurement station (ICAO Annex 16 Vol. I, Appendix 2)
# ---------------------------------------------------------------------------

def _d_aircraft_noise_station(s: SVG, th: Theme) -> None:
    """The station a technician builds under one of the reference points.

    A site elevation with the obstruction-free cone of 80° half-angle, the
    10 m meteorological mast and the independent tracking sensor; below it,
    the microphone itself at a larger scale and the plan that explains why
    its capsule points across the track. The Appendix 2 test window and the
    sample-size rule are in the footer."""
    gy = 320.0                                   # site ground line
    mx = 300.0                                   # microphone station

    # --- site elevation -----------------------------------------------------
    s.text(56.0, 70.0, "Site", 15, th.fg, bold=True, anchor="start")
    s.line(60.0, 104.0, 592.0, 104.0, th.muted, 1.2, dash="8,6")
    s.arrow(560.0, 104.0, 600.0, 104.0, th.fg, 1.8)
    _plane_glyph(s, 380.0, 104.0, 0.0, 0.8)
    s.text(380.0, 84.0, "reference flight path", 13, th.fg)
    s.ground(gy, 60.0, 866.0)
    s.line(378.0, 112.0, mx + 4.0, gy - 30.0, th.primary, 2.2)
    s.text(360.0, 232.0, "slant path", 12, th.primary, anchor="start")

    # The obstruction-free cone has an 80° half-angle about the vertical, so
    # its rays run 10° above the horizon.
    for ex in (68.0, 866.0):
        s.line(mx, gy, ex, gy - abs(ex - mx) * math.tan(math.radians(10.0)),
               th.accent, 1.4, dash="6,5")
    s.text(mx + 140.0, gy - 38.0, "80° half-angle about the vertical", 12,
           th.accent, anchor="start")
    # A tree clear of the cone, and one that breaks into it.
    for tx, ok in ((150.0, True), (612.0, False)):
        top = gy - (18.0 if ok else 90.0)
        s.line(tx, gy, tx, top + 10.0, th.muted, 3.0)
        s.ellipse(tx, top, 14.0, 13.0, th.panel, th.muted, 1.6)
    s.text(612.0, gy - 112.0, "inside the cone: site rejected", 11,
           th.secondary)

    # The station itself, small at this scale, and the tracking sensor.
    s.mic(mx, gy - 26.0, gy, 0.7)
    s.text(mx, gy + 22.0, "microphone", 12, th.muted)
    s.rect(84.0, gy - 34.0, 46.0, 34.0, th.panel, th.fg, rx=4.0, sw=1.6)
    s.line(107.0, gy - 34.0, 134.0, gy - 62.0, th.fg, 2.0)
    s.line(134.0, gy - 62.0, 366.0, 116.0, th.muted, 1.0, dash="4,4")
    s.text(84.0, gy + 44.0, "tracking, independent of the cockpit", 12,
           th.muted, anchor="start")

    # 10 m meteorological mast, at 12 px per metre.
    mast, mast_h = 782.0, 120.0
    s.line(mast, gy, mast, gy - mast_h, th.fg, 2.6)
    s.line(mast - 20.0, gy - mast_h, mast + 20.0, gy - mast_h, th.fg, 2.0)
    for cup in (-20.0, 0.0, 20.0):
        s.circle(mast + cup, gy - mast_h - 7.0, 5.0, th.primary)
    s.rect(mast + 8.0, gy - mast_h + 34.0, 32.0, 18.0, th.panel, th.fg,
           rx=3.0, sw=1.4)
    s.text(mast + 46.0, gy - mast_h + 48.0, "$T$, RH", 11, th.muted,
           anchor="start")
    s.dim(mast - 36.0, gy, mast - 36.0, gy - mast_h, "10 m", offset=0, size=13)
    s.text(mast, gy + 22.0, "met mast, within 2 000 m", 12, th.muted)

    # --- the microphone, at a larger scale ----------------------------------
    s.text(56.0, 386.0, "The microphone, at 60 px per metre", 15, th.fg,
           bold=True, anchor="start")
    dgy = 540.0
    dx = 210.0
    cap = dgy - 72.0                              # 1.2 m at 60 px per metre
    s.ground(dgy, 84.0, 400.0)
    s.line(dx, dgy, dx, cap, th.fg, 3.0)
    s.line(dx - 22.0, dgy, dx + 22.0, dgy, th.fg, 2.4)
    s.circle(dx, cap, 26.0, "none", th.muted, 1.8)
    s.circle(dx, cap, 8.0, th.fg)
    s.arrow(dx - 74.0, cap - 74.0, dx - 22.0, cap - 22.0, th.primary, 2.0)
    s.text(dx - 84.0, cap - 42.0, "arriving ray", 12, th.primary, anchor="end")
    s.text(dx + 40.0, cap - 26.0, "windscreen: insertion loss", 12, th.muted,
           anchor="start")
    s.text(dx + 40.0, cap - 8.0, "within ±1.5 dB, and corrected for", 12,
           th.muted, anchor="start")
    s.dim(dx - 54.0, dgy, dx - 54.0, cap, "1.2 m", offset=0, size=13)
    s.line(dx - 26.0, cap, dx - 54.0, cap, th.muted, 0.9, dash="3,3")
    s.text(dx, dgy + 24.0, "sensing element 1.2 m above local ground", 12,
           th.muted)

    # --- plan: why the capsule points across the track ----------------------
    s.text(470.0, 386.0, "Plan: the capsule axis", 15, th.fg, bold=True,
           anchor="start")
    s.line(482.0, 452.0, 862.0, 452.0, th.muted, 1.2, dash="7,5")
    s.arrow(836.0, 452.0, 866.0, 452.0, th.muted, 1.5)
    s.text(858.0, 440.0, "track", 11, th.muted, anchor="end")
    s.line(672.0, 424.0, 672.0, 506.0, th.primary, 2.6)
    s.circle(672.0, 486.0, 8.0, th.fg)
    s.text(672.0, 530.0, "the capsule axis is perpendicular to the plane of the",
           11, th.muted)
    s.text(672.0, 548.0, "flight path, so every ray arrives at 90°, grazing",
           11, th.muted)
    s.rect(478.0, 566.0, 388.0, 40.0, th.panel, th.fg, rx=4.0, sw=1.6)
    s.text(672.0, 582.0, "24 one-third-octave bands, 50 Hz to 10 kHz", 12, th.fg)
    s.text(672.0, 600.0, "one sample every 500 ms ± 5 ms", 11, th.muted)

    # --- normative context --------------------------------------------------
    s.text(70.0, 646.0,
           "Test window (aeroplanes): no precipitation; −10 to 35 °C and 20 to 95 % RH over the path above 10 m;",
           13, th.fg, anchor="start")
    s.text(70.0, 670.0,
           "8 kHz attenuation ≤ 12 dB/100 m; wind ≤ 6.2 m/s average and 7.7 m/s peak, crosswind ≤ 3.6 and 5.1 m/s",
           13, th.fg, anchor="start")
    s.text(70.0, 694.0,
           "Helicopters: average wind ≤ 5.1 m/s and crosswind ≤ 2.6 m/s, "
           "temperature and humidity limits at 10 m only",
           13, th.muted, anchor="start")
    s.text(70.0, 718.0,
           "At least six valid runs per measurement point, with a 90 % confidence limit not exceeding ±1.5 EPNdB",
           13, th.muted, anchor="start")


# ---------------------------------------------------------------------------
# The rotorcraft noise hemisphere and its angles (ECAC Doc 32 Eq. 3, §4.2)
# ---------------------------------------------------------------------------

def _heli_glyph(s: SVG, x: float, y: float, side: bool = True,
                scale: float = 1.0) -> None:
    """Small helicopter: side view flying right, or rear view seen from astern."""
    th, k = s.th, scale
    if side:
        s.line(x - 34 * k, y - 16 * k, x + 34 * k, y - 16 * k, th.fg, 2.8)
        s.line(x, y - 16 * k, x, y - 6 * k, th.fg, 2.0)
        s.ellipse(x, y, 20 * k, 10 * k, th.panel, th.fg, 2.0)
        s.line(x - 16 * k, y - 3 * k, x - 44 * k, y - 7 * k, th.fg, 2.4)
        s.line(x - 44 * k, y - 15 * k, x - 44 * k, y + 1 * k, th.fg, 1.8)
    else:
        s.line(x - 36 * k, y - 16 * k, x + 36 * k, y - 16 * k, th.fg, 2.8)
        s.line(x, y - 16 * k, x, y - 8 * k, th.fg, 2.0)
        s.ellipse(x, y, 13 * k, 11 * k, th.panel, th.fg, 2.0)
        s.line(x - 15 * k, y + 11 * k, x + 15 * k, y + 11 * k, th.fg, 2.2)


def _d_rotorcraft_hemisphere(s: SVG, th: Theme) -> None:
    """What the two hemisphere angles mean, and what a lookup returns.

    ECAC Doc 32 Eq. 3 in two sections: the polar angle θ measured from the
    nose in the vertical centre plane, and the azimuth φ measured about the
    nose axis from straight down. The third panel places the 60 m sphere on a
    track, so the source level and the three propagation adjustments read as
    one chain."""
    r = 112.0

    # --- (a) the polar angle, in the vertical centre plane (φ = 0) ---------
    ax, ay = 236.0, 162.0
    s.text(52.0, 72.0, "(a) Polar angle $θ$, centre plane $φ$ = 0", 15, th.fg,
           bold=True, anchor="start")
    _arc(s, ax, ay, r, 0.0, -180.0, th.muted, 1.4)
    for ang in (0.0, -90.0, -180.0):
        s.line(ax, ay, *_polar(ax, ay, r, ang), th.muted, 1.0, dash="4,4")
    s.text(ax + r + 10.0, ay + 6.0, "$θ$ = 0 nose", 12, th.fg, anchor="start")
    s.text(ax - r - 10.0, ay + 6.0, "$θ$ = 180 tail", 12, th.fg, anchor="end")
    s.text(ax, ay + r + 26.0, "$θ$ = 90 beneath", 12, th.fg)
    _arc(s, ax, ay, r - 20.0, -40.0, -140.0, th.accent, 5.0)
    s.line(*_polar(ax, ay, r * 0.30, -52.0), *_polar(ax, ay, r, -52.0),
           th.primary, 1.4)
    s.text(*_polar(ax, ay, r * 0.68, -52.0), "$r_h$ = 60 m", 12, th.primary,
           anchor="start")
    s.text(ax, ay + r + 50.0, "measured polar band $θ_{t1}$ … $θ_{t2}$,", 11, th.accent)
    s.text(ax, ay + r + 68.0, "the two 10 dB-down instants", 11, th.muted)
    _heli_glyph(s, ax, ay, True, 1.0)

    # --- (b) the azimuth, in the plane across the aircraft (θ = 90) --------
    bx, by = 664.0, 162.0
    s.text(470.0, 72.0, "(b) Azimuth $φ$, seen from astern", 15, th.fg, bold=True,
           anchor="start")
    _arc(s, bx, by, r, 0.0, -180.0, th.muted, 1.4)
    _arc(s, bx, by, r - 20.0, -30.0, -150.0, th.accent, 5.0)
    for ang in (0.0, -90.0, -180.0, -30.0, -150.0):
        s.line(bx, by, *_polar(bx, by, r, ang), th.muted, 1.0, dash="4,4")
    for ang in (-30.0, -150.0):
        px, py = _polar(bx, by, r + 20.0, ang)
        s.text(px, py + 6.0, "60°", 11, th.accent)
    s.text(bx + r, by + 26.0, "$φ$ = +90 starboard", 12, th.fg)
    s.text(bx - r, by + 26.0, "$φ$ = −90 port", 12, th.fg)
    s.text(bx, by + r + 26.0, "$φ$ = 0 beneath", 12, th.fg)
    s.text(bx, by + r + 50.0, "measured lateral band $−60° ≤ φ ≤ 60°$,", 11,
           th.accent)
    s.text(bx, by + r + 68.0, "outside it the bins are gap-filled", 11, th.muted)
    _heli_glyph(s, bx, by, False, 1.0)

    # --- (c) the sphere on a track ----------------------------------------
    gy = 532.0
    s.text(52.0, 386.0, "(c) The 60 m sphere on a flight path", 15, th.fg,
           bold=True, anchor="start")
    s.ground(gy, 60.0, 862.0)
    hx, hy = 280.0, 434.0
    s.line(70.0, hy, 520.0, hy, th.muted, 1.2, dash="8,6")
    s.arrow(486.0, hy, 528.0, hy, th.fg, 1.8)
    s.circle(hx, hy, 38.0, "none", th.primary, 1.3)
    _heli_glyph(s, hx, hy, True, 0.8)
    s.text(hx - 46.0, hy - 26.0, "60 m", 11, th.primary, anchor="end")
    s.mic(742.0, gy - 22.0, gy, 0.6)
    s.line(hx + 26.0, hy + 26.0, 740.0, gy - 26.0, th.primary, 2.2)
    s.text(540.0, 468.0, "slant range $r$", 12, th.primary)
    s.text(742.0, gy + 24.0, "receiver, 1.2 m", 12, th.muted)
    s.line(hx - 32.0, hy + 8.0, hx + 32.0, hy - 8.0, th.accent, 2.0, dash="5,4")
    s.text(hx + 44.0, hy - 22.0, "banked by $Φ$ in turns", 11, th.accent,
           anchor="start")

    # --- the array contract -------------------------------------------------
    s.text(60.0, 584.0,
           "$ΔL_s = −20 lg(r/60)$ · $ΔL_a = −α(f)(r − 60)$ · $ΔL_g$ from the two-ray model over the ground",
           13, th.fg, anchor="start")
    s.text(60.0, 608.0,
           "levels[a, p, f] in dB at 60 m under the ICAO reference atmosphere (25 °C, 70 % RH, 101.325 kPa):",
           13, th.fg, anchor="start")
    s.text(60.0, 632.0,
           "19 azimuths × 19 polar angles at 10°, 31 one-third-octave bands from 10 Hz to 10 kHz",
           13, th.muted, anchor="start")
    s.text(60.0, 656.0,
           "unmeasured bins are NaN, never 0 dB; mirrored-rotor class members read the same data at $−φ$",
           13, th.muted, anchor="start")
