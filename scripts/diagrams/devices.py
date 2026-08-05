#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Diagrams of the devices guides: emission, electroacoustics, broadcast and noise control.

One subject: a device under test and the setup that measures it. The
emission diagrams draw the sound power and intensity methods that grade a
machine, the electroacoustics diagrams draw the transducer measurements, the
broadcast diagram draws the programme loudness chain, and the noise control
diagram draws what is done to a device once it has been graded.
"""

from __future__ import annotations

import itertools

from .canvas import SVG, Theme
from .parts import _accel, _box_solid, _box_wire, _motion_arrows

# ---------------------------------------------------------------------------
# d6 - Two-microphone (p-p) intensity probe (IEC 61043)
# ---------------------------------------------------------------------------

def _d_pp_probe(s: SVG, th: Theme) -> None:
    ay = 232.0  # probe axis height

    # Measurement axis / intensity direction (drawn first, under the probe)
    s.line(70, ay, 820, ay, th.accent, 1.4, dash="10,4,2,4")
    s.arrow(820, ay, 852, ay, th.accent, 1.8)
    s.text(450, 305, "measurement axis / intensity direction", 18, th.accent)

    # Two opposed capsules facing each other with a spacer between the tips
    for side in (-1, 1):
        # bodies: 180..320 and 580..720; capsules: 320..400 / 500..580;
        # tips (grilles): 400..414 / 486..500; gap 414..486 = Δr
        bx = 180.0 if side < 0 else 580.0
        cx = 320.0 if side < 0 else 500.0
        tx = 400.0 if side < 0 else 486.0
        s.rect(bx, ay - 28, 140, 56, th.panel, th.primary, rx=10, sw=2)
        s.rect(cx, ay - 20, 80, 40, th.fg, rx=4)
        s.rect(tx, ay - 16, 14, 32, th.muted, rx=2)
    s.rect(414, ay - 6, 72, 12, th.panel, th.muted, rx=4, sw=1.2)  # spacer

    s.text(360, ay - 38, "p1", 20, th.fg, mono=True, bold=True)
    s.text(540, ay - 38, "p2", 20, th.fg, mono=True, bold=True)

    # Δr dimension between the capsule tips, drafting style
    s.dim(414, ay - 16, 486, ay - 16, "Δr = 12 mm", offset=-66, size=18)

    # p-p estimator notes near the capsules
    s.text(280, 365, "u from the p2−p1 gradient", 19, th.muted, mono=True)
    s.text(620, 365, "p = (p1+p2)/2", 19, th.muted, mono=True)


# ---------------------------------------------------------------------------
# d10 - ISO 3744/3746 sound power measurement surfaces
# ---------------------------------------------------------------------------

def _d_surfaces(s: SVG, th: Theme) -> None:
    # ===== Left panel: hemispherical surface over a reflecting plane =====
    cx, gy, R = 235.0, 420.0, 150.0
    s.text(cx, 74, "Hemispherical surface", 22, th.fg, bold=True)

    # Reflecting plane (hatched line through the equator / footprint centre).
    s.ground(gy, 55, 430)
    s.text(70, gy + 34, "Reflecting plane", 17, th.muted, anchor="start")

    # Hemisphere: dashed footprint ellipse + solid dome silhouette.
    ky = 0.30
    s.ellipse(cx, gy, R, R * ky, "none", th.muted, 1.3, dash="5,4")
    s.path(f"M {cx - R} {gy} A {R} {R} 0 0 1 {cx + R} {gy}",
           stroke=th.primary, sw=2.4)

    # Source box at the centre O.
    _box_solid(s, th, cx, gy, 30, 24, 34)
    s.circle(cx, gy, 3.4, th.fg)

    # Ten key microphone positions (ISO 3744 Table B.1), oblique-projected.
    b1 = [(0.16, -0.96, 0.22), (0.78, -0.60, 0.20), (0.78, 0.55, 0.31),
          (0.16, 0.90, 0.41), (-0.83, 0.32, 0.45), (-0.83, -0.40, 0.38),
          (-0.26, -0.65, 0.71), (0.74, -0.07, 0.67), (-0.26, 0.50, 0.83),
          (0.10, -0.10, 0.99)]
    labelled = {1, 8, 10}
    pts = []
    for x, y, z in b1:
        px = cx + R * x + 42 * y
        py = gy - 34 * y - R * z
        pts.append((px, py))
    # radius r drawn to position 8 (a mid-height point on the surface).
    r8 = pts[7]
    s.line(cx, gy, r8[0], r8[1], th.accent, 1.6, dash="6,4")
    s.text((cx + r8[0]) / 2 + 10, (gy + r8[1]) / 2 + 4, "radius r ≥ 2 d₀",
           17, th.accent, anchor="start")
    for i, (px, py) in enumerate(pts, start=1):
        s.circle(px, py, 6.5, th.secondary)
        s.circle(px, py, 2.2, th.bg)
        if i in labelled:
            s.text(px, py - 12, str(i), 16, th.fg, bold=True)
    s.text(cx, gy + 62, "10 key positions (Table B.1)", 17, th.muted)
    s.text(cx, gy + 86, "one plane · S = 2πr²", 18, th.primary, bold=True, mono=True)

    # ===== Right panel: parallelepiped measurement surface =====
    bx2, gy2 = 675.0, 420.0
    s.text(bx2, 74, "Parallelepiped surface", 22, th.fg, bold=True)
    s.ground(gy2, 500, 872)

    # Source box (solid) enclosed by the measurement box (dashed wireframe).
    _box_solid(s, th, bx2, gy2, 46, 40, 58)
    _box_wire(s, th, bx2, gy2, 96, 90, 108, th.accent)
    s.text(bx2, gy2 + 40, "Measurement surface", 17, th.muted)
    s.text(bx2, gy2 + 64, "one plane · S = 4(ab+bc+ca)", 18, th.accent,
           bold=True, mono=True)

    # Measurement distance d: vertical clearance between the source top face
    # and the enveloping measurement surface (labelled arrow + caption above).
    s.text(bx2, 208, "measurement distance d", 18, th.secondary, bold=True)
    s.dim(bx2, gy2 - 108, bx2, gy2 - 58, "d", offset=0, size=20,
          label_side="right")


# ---------------------------------------------------------------------------
# d12 - Sound power methods comparison infographic
# ---------------------------------------------------------------------------

def _d_methods(s: SVG, th: Theme) -> None:
    cols = [
        ("ISO 3744 / 3746", "Free field over a reflecting plane",
         "Grade 2 / 3 (engineering / survey)",
         "Sound pressure · enveloping surface",
         "LW = L̄p + 10 log10(S/S₀) − K1 − K2",
         "K2A ≤ 4 dB (3744) / ≤ 7 dB (3746)", th.primary, "hemi"),
        ("ISO 3741", "Reverberation test room",
         "Grade 1 (precision)",
         "Sound pressure · diffuse field",
         "LW ← L̄p , T , V",
         "V ≥ 200 m³ , qualified room", th.accent, "reverb"),
        ("ISO 9614-2", "In situ — any environment",
         "Grade 2 / 3 (engineering / survey)",
         "Sound intensity · scanning",
         "LW = 10 log10 |Σ IᵢSᵢ| / W₀",
         "no negative-power bands", th.secondary, "probe"),
    ]
    cw, gap = 270.0, 15.0
    x0 = (900 - (3 * cw + 2 * gap)) / 2
    ctop, cbot = 66.0, 540.0
    for i, (name, env, grade, method, formula, note, col, pic) in enumerate(cols):
        x = x0 + i * (cw + gap)
        cxc = x + cw / 2
        s.rect(x, ctop, cw, cbot - ctop, th.panel, col, rx=14, sw=2.4)
        s.rect(x, ctop, cw, 44, col, col, rx=14, sw=0)
        s.rect(x, ctop + 22, cw, 22, col, "none")  # square off header bottom
        s.text(cxc, ctop + 30, name, 22, th.bg, bold=True)

        # Mini-pictogram band.
        py = ctop + 120.0
        if pic == "hemi":
            R = 58.0
            s.ellipse(cxc, py + 30, R, R * 0.3, "none", th.muted, 1.2, dash="4,3")
            s.path(f"M {cxc - R} {py + 30} A {R} {R} 0 0 1 {cxc + R} {py + 30}",
                   stroke=col, sw=2.2)
            s.line(cxc - R, py + 30, cxc + R, py + 30, th.muted, 1.4)
            _box_solid(s, th, cxc, py + 30, 12, 10, 16, stroke=col)
            for ang in (35, 90, 145):
                import math
                a = math.radians(ang)
                s.circle(cxc + R * math.cos(a), py + 30 - R * math.sin(a), 4.5,
                         th.secondary)
        elif pic == "reverb":
            s.rect(cxc - 58, py - 26, 116, 84, "none", col, rx=6, sw=2.2)
            for k in range(3):
                yy = py - 12 + k * 22
                s.path(f"M {cxc - 44} {yy} q 12 -12 24 0 q 12 12 24 0 q 12 -12 24 0",
                       stroke=th.muted, sw=1.6)
            s.circle(cxc - 40, py + 44, 6, th.secondary)   # RSS / source
        else:  # probe scanning a surface
            s.rect(cxc - 56, py - 30, 112, 92, "none", col, rx=6, sw=2.0, )
            # serpentine scan path
            s.path(f"M {cxc - 44} {py - 16} L {cxc + 40} {py - 16} "
                   f"L {cxc + 40} {py + 4} L {cxc - 44} {py + 4} "
                   f"L {cxc - 44} {py + 24} L {cxc + 40} {py + 24}",
                   stroke=th.accent, sw=1.7)
            s.circle(cxc + 40, py + 24, 5, th.secondary)
            s.text(cxc, py + 54, "I⊥", 17, col, bold=True, mono=True)

        # Attribute rows.
        rows = [(py + 96, env, th.fg, False),
                (py + 128, grade, col, True),
                (py + 160, method, th.muted, False)]
        for yy, txt, cc, bold in rows:
            s.text(cxc, yy, txt, 14, cc, bold=bold)

        # Headline formula in a boxed footer. The ISO 3744 expression is the
        # longest of the three, so the shared face is the one that keeps it
        # inside its box with a margin left over.
        s.rect(x + 6, cbot - 96, cw - 12, 46, "none", col, rx=8, dash="5,4")
        s.text(cxc, cbot - 67, formula, 12, th.fg, bold=True, mono=True)
        s.text(cxc, cbot - 26, note, 14, th.muted)


# ---------------------------------------------------------------------------
# d19 - ISO 3745 precision sound power (anechoic / hemi-anechoic room)
# ---------------------------------------------------------------------------

def _d_precision_anechoic(s: SVG, th: Theme) -> None:
    """ISO 3745 precision sound power on a (hemi-)spherical array."""
    x0, y0, x1, gy = 60.0, 70.0, 840.0, 470.0
    s.rect(x0, y0, x1 - x0, gy - y0, th.bg, th.fg, sw=3)

    # Anechoic wedges lining the ceiling and the two side walls.
    for wx in range(int(x0) + 4, int(x1) - 36, 40):
        s.path(f"M {wx} {y0} L {wx + 40} {y0} L {wx + 20} {y0 + 28} Z",
               fill=th.panel, stroke=th.muted, sw=1.0)
    for wy in range(int(y0) + 30, int(gy) - 36, 40):
        s.path(f"M {x0} {wy} L {x0} {wy + 40} L {x0 + 28} {wy + 20} Z",
               fill=th.panel, stroke=th.muted, sw=1.0)
        s.path(f"M {x1} {wy} L {x1} {wy + 40} L {x1 - 28} {wy + 20} Z",
               fill=th.panel, stroke=th.muted, sw=1.0)
    s.text(200, 120, "Anechoic wedges", 15, th.muted, anchor="start")

    # Reflecting floor (hemi-anechoic room).
    s.ground(gy, x0, x1)
    s.text(70, gy - 8, "Reflecting plane (hemi-anechoic)", 15, th.muted,
           anchor="start")

    # Source (DUT) at the centre of the reflecting plane.
    cx, R = 450.0, 200.0
    _box_solid(s, th, cx, gy, 34, 26, 40)
    s.circle(cx, gy, 3.4, th.fg)
    s.text(cx + 52, gy - 14, "Source (DUT)", 17, th.fg, bold=True,
           anchor="start")

    # Hemispherical measurement surface of radius r.
    s.ellipse(cx, gy, R, R * 0.16, "none", th.muted, 1.3, dash="5,4")
    s.path(f"M {cx - R} {gy} A {R} {R} 0 0 1 {cx + R} {gy}",
           stroke=th.primary, sw=2.4)

    # Ten normative microphone positions (ISO 3744/3745 Annex B), projected.
    b1 = [(0.16, -0.96, 0.22), (0.78, -0.60, 0.20), (0.78, 0.55, 0.31),
          (0.16, 0.90, 0.41), (-0.83, 0.32, 0.45), (-0.83, -0.40, 0.38),
          (-0.26, -0.65, 0.71), (0.74, -0.07, 0.67), (-0.26, 0.50, 0.83),
          (0.10, -0.10, 0.99)]
    pts = [(cx + R * x + 46 * y, gy - 30 * y - R * z) for x, y, z in b1]
    r8 = pts[7]
    s.line(cx, gy, r8[0], r8[1], th.accent, 1.6, dash="6,4")
    s.text((cx + r8[0]) / 2 + 8, (gy + r8[1]) / 2 + 2, "radius r", 16,
           th.accent, anchor="start")
    for px, py in pts:
        s.circle(px, py, 6.5, th.secondary)
        s.circle(px, py, 2.2, th.bg)
    s.text(688, 300, "20 / 40 mic positions", 16, th.muted, anchor="start")

    # Governing relations.
    for y, txt, col, bold in (
        (514, "LW = ⟨Lp⟩ + 10 log10(S/S0) + C1 + C2 + C3", th.fg, True),
        (540, "S = 2πr² (hemi-anechoic) · 4πr² (anechoic)", th.primary, True),
        (564, "K1: per-position background correction", th.muted, False),
        (587, "C1, C2, C3: meteorological corrections (ps, θ, a(f))",
         th.muted, False),
    ):
        s.text(450, y, txt, 19 if bold else 18, col, bold=bold)


# ---------------------------------------------------------------------------
# d20 - ISO 9614-3 precision sound intensity scanning
# ---------------------------------------------------------------------------

def _d_intensity_scan(s: SVG, th: Theme) -> None:
    """ISO 9614-3 precision sound power by intensity scanning."""
    gy, bx = 470.0, 360.0

    # Measurement surface (dashed wireframe) enclosing the source.
    _box_wire(s, th, bx, gy, 150, 120, 240, th.primary)
    _box_solid(s, th, bx, gy, 45, 34, 70)
    s.text(bx, gy - 82, "Source", 18, th.fg, bold=True)
    s.text(bx, 214, "Measurement surface (segments S_i)", 17, th.primary,
           bold=True)

    # Segment grid on the front face (3 x 3 segments Sᵢ).
    fl, fr, ft, fb = bx - 150, bx + 150, gy - 240, gy
    for gx in (fl + 100, fl + 200):
        s.line(gx, ft, gx, fb, th.muted, 1.2, dash="4,4")
    for gyy in (ft + 80, ft + 160):
        s.line(fl, gyy, fr, gyy, th.muted, 1.2, dash="4,4")
    s.text(fl + 50, ft + 46, "S_i", 18, th.fg, bold=True)

    # Serpentine scan path across the segment-row centres.
    ys = (ft + 40, ft + 120, ft + 200)
    px = [(fl + 30, ys[0]), (fr - 30, ys[0]), (fr - 30, ys[1]),
          (fl + 30, ys[1]), (fl + 30, ys[2]), (fr - 30, ys[2])]
    for (ax, ay), (bxx, byy) in itertools.pairwise(px):
        s.line(ax, ay, bxx, byy, th.accent, 2.0, dash="2,3")
    s.arrow(px[-2][0] + 60, px[-1][1], px[-1][0], px[-1][1], th.accent, 2.0)
    s.text(fr + 8, ys[2] + 6, "serpentine scan", 15, th.accent, anchor="start")

    # A p-p intensity probe on the scan path.
    ppx, ppy = bx, ys[1]
    s.line(ppx, ppy, ppx + 46, ppy - 26, th.fg, 2.2)
    s.circle(ppx, ppy - 6, 5, th.fg)
    s.circle(ppx, ppy + 6, 5, th.fg)
    s.text(ppx + 52, ppy - 30, "p-p probe", 15, th.fg, anchor="start")

    # Normal-intensity arrows exiting the left column of segments.
    for yy in ys:
        s.arrow(fl, yy, fl - 34, yy + 8, th.secondary, 2.0)
    s.text(fl - 40, ys[1] + 30, "I_n (normal intensity)", 15, th.secondary,
           anchor="end")

    # Governing relations.
    for y, txt, col, bold in (
        (505, "P = Σ I_n,i · S_i   (partial powers per segment)", th.fg, True),
        (533, "LW = 10 log10(P/P0),  P0 = 1 pW", th.accent, True),
        (559, "Field indicators: F_pIn , FT , FS", th.primary, True),
        (583, "Five acceptance criteria (Annex C); band invalid if P < 0",
         th.muted, False),
    ):
        s.text(450, y, txt, 19 if bold else 18, col, bold=bold)


def _d_loudspeaker_freefield(s: SVG, th: Theme) -> None:
    """IEC 60268-5 loudspeaker sensitivity on the reference axis (free field)."""
    x0, y0, x1, gy = 60.0, 70.0, 840.0, 470.0
    s.rect(x0, y0, x1 - x0, gy - y0, th.bg, th.fg, sw=3)

    # Anechoic wedges on all four boundaries (full free field: no floor).
    for wx in range(int(x0) + 4, int(x1) - 36, 40):
        s.path(f"M {wx} {y0} L {wx + 40} {y0} L {wx + 20} {y0 + 28} Z",
               fill=th.panel, stroke=th.muted, sw=1.0)
        s.path(f"M {wx} {gy} L {wx + 40} {gy} L {wx + 20} {gy - 28} Z",
               fill=th.panel, stroke=th.muted, sw=1.0)
    for wy in range(int(y0) + 30, int(gy) - 64, 40):
        s.path(f"M {x0} {wy} L {x0} {wy + 40} L {x0 + 28} {wy + 20} Z",
               fill=th.panel, stroke=th.muted, sw=1.0)
        s.path(f"M {x1} {wy} L {x1} {wy + 40} L {x1 - 28} {wy + 20} Z",
               fill=th.panel, stroke=th.muted, sw=1.0)
    s.text(210, 122, "Anechoic wedges", 15, th.muted, anchor="start")

    # Loudspeaker cabinet on a stand, reference point on the front baffle.
    ax_y, fx = 275.0, 250.0
    s.line(219, ax_y + 70, 219, 462, th.fg, 2.2)
    s.line(199, 462, 239, 462, th.fg, 2.2)
    s.rect(fx - 62, ax_y - 70, 62, 140, th.panel, th.primary, rx=6, sw=2)
    s.circle(fx - 18, ax_y, 14, th.primary)
    s.circle(fx - 18, ax_y, 5.5, th.bg)
    s.text(219, ax_y - 84, "Loudspeaker", 18, th.fg, bold=True)
    for r in (26, 44, 62):
        s.path(f"M {fx + r * 0.34:.1f} {ax_y - r * 0.94:.1f} "
               f"A {r} {r} 0 0 1 {fx + r * 0.34:.1f} {ax_y + r * 0.94:.1f}",
               stroke=th.accent, sw=1.5)

    # Reference axis through the reference point, out to the right.
    s.circle(fx, ax_y, 3.4, th.fg)
    s.line(fx, ax_y, 782, ax_y, th.muted, 1.4, dash="7,5")
    s.arrow(760, ax_y, 792, ax_y, th.muted, 1.4)
    s.text(724, ax_y + 24, "Reference axis", 15, th.muted)

    # Measurement microphone on axis, capsule facing the loudspeaker.
    mx = 620.0
    s.line(mx + 23, ax_y + 6, mx + 23, 462, th.fg, 2.2)
    s.line(mx + 7, 462, mx + 39, 462, th.fg, 2.2)
    s.rect(mx, ax_y - 6, 46, 12, th.primary, rx=4)
    s.rect(mx - 12, ax_y - 4, 12, 8, th.fg, rx=2.5)
    s.text(mx + 24, ax_y - 24, "Measurement microphone", 17, th.fg, bold=True)

    # Reference distance, drafting style, between baffle and capsule tip.
    s.dim(fx, ax_y, mx - 12, ax_y, "r = 1 m", offset=92)

    # Drive: amplifier delivering 1 W into the rated impedance.
    s.rect(85, 383, 140, 54, th.panel, th.primary, rx=8, sw=2)
    s.text(155, 405, "Amplifier", 17, th.fg, bold=True)
    s.text(155, 427, "2.83 V (8 Ω)", 15, th.secondary, mono=True)
    s.line(155, 383, 155, 345, th.fg, 1.6)
    s.line(155, 345, fx - 62, 345, th.fg, 1.6)

    # Governing relations.
    for y, txt, col, bold in (
        (508, "Characteristic sensitivity: Lp at 1 m for 1 W into the rated impedance",
         th.fg, True),
        (534, "Up = √(R · 1 W): 2.83 V is 1 W into 8 Ω but 2 W into 4 Ω (+3 dB)",
         th.secondary, True),
        (559, "Lp(1 m) = Lp(r) + 20 log10(r / 1 m)   (far field, inverse-distance law)",
         th.primary, True),
        (583, "Microphone (IEC 60268-4): M in mV/Pa, or LM = 20 log10(M / 1 V/Pa) dB",
         th.muted, False),
    ):
        s.text(450, y, txt, 19 if bold else 18, col, bold=bold)


# ---------------------------------------------------------------------------
# Sound power from surface vibration (ISO/TS 7849)
# ---------------------------------------------------------------------------

def _d_vibration_sound_power(s: SVG, th: Theme) -> None:
    """ISO/TS 7849 surface-velocity method: the machine's radiating surface
    divided into N equal cells, one accelerometer per cell centre, and the
    survey sound power from the mean velocity level over the area S."""
    gy = 470.0
    s.ground(gy, 50.0, 560.0)

    # Machine body with the vibrating measurement surface on top.
    bx, hw, dp, ht = 270.0, 140.0, 115.0, 170.0
    _box_solid(s, th, bx, gy, hw, dp, ht)
    fx0, fx1 = bx - hw, bx + hw          # top-face front edge
    fy = gy - ht
    dxo, dyo = dp * 0.72, dp * 0.55

    # Measurement grid: 5 x 4 cells on the top face (the Table 1 initial
    # count for a 1-10 m2 surface), a dot per cell centre.
    for i in range(1, 5):
        gx = fx0 + i * (2 * hw) / 5
        s.line(gx, fy, gx + dxo, fy - dyo, th.muted, 1.0)
    for f_row in (0.25, 0.5, 0.75):
        s.line(fx0 + dxo * f_row, fy - dyo * f_row,
               fx1 + dxo * f_row, fy - dyo * f_row, th.muted, 1.0)
    pts = []
    for r_ in (0.125, 0.375, 0.625, 0.875):
        for i in range(5):
            u = (i + 0.5) / 5
            pts.append((fx0 + u * 2 * hw + r_ * dxo, fy - r_ * dyo))
    for px_, py_ in pts:
        s.circle(px_, py_, 4, th.secondary)
        s.circle(px_, py_, 1.5, th.bg)
    # One accelerometer drawn explicitly, with its vibratory motion.
    _accel(s, pts[15][0], pts[15][1] - 4)
    _motion_arrows(s, pts[15][0], pts[15][1] - 46, 16, th.secondary)
    s.text(250, 150, "Vibrating measurement surface S", 19, th.fg, bold=True)
    s.line(310, 160, 340, 228, th.muted, 1.0)

    # Radiated sound from the surface.
    for r in (36, 60, 84):
        s.path(f"M {475 + r * 0.30:.1f} {370 - r:.1f} "
               f"A {r} {r} 0 0 1 {475 + r:.1f} {370 - r * 0.30:.1f}",
               stroke=th.accent, sw=1.6)
    s.text(672, 432, "radiated airborne sound", 16, th.accent)
    s.line(618, 424, 570, 372, th.muted, 1.0)

    # Dimensions of the surface (2.5 m x 1.6 m -> S = 4 m2).
    s.dim(fx0, gy, fx1, gy, "2.5 m", offset=32, size=18)
    s.arrow(fx1 + 20, gy + 18, fx1 + dxo + 14, gy - dyo + 18, th.muted, 1.2)
    s.arrow(fx1 + dxo + 14, gy - dyo + 18, fx1 + 20, gy + 18, th.muted, 1.2)
    s.text(fx1 + dxo - 4, gy - dyo + 46, "1.6 m", 18, th.fg, anchor="start")
    s.text(bx, 540, "Machine under test", 18, th.fg, bold=True)

    # Number of measurement positions and the survey relation.
    lx = 575.0
    s.text(lx, 110, "Initial number of positions N", 19, th.fg, bold=True,
           anchor="start")
    for y, txt in ((140, "S < 1 m²   →   10"),
                   (166, "1 m² ≤ S ≤ 10 m²  →  20"),
                   (192, "S > 10 m²  →  2 S / S₀")):
        s.text(lx, y, txt, 16, th.fg, anchor="start", mono=True)
    s.text(lx, 220, "one accelerometer per cell of area S/N", 15, th.muted,
           anchor="start")
    s.text(lx, 284, "Survey sound power", 19, th.fg, bold=True, anchor="start")
    # Two logarithms in one line: the smaller face keeps it inside the column.
    s.text(lx, 314, "LWA = LvA + 10 log10(S/S₀) + 10 log10 ε", 13, th.primary,
           anchor="start", bold=True, mono=True)
    s.text(lx, 342, "ε = 1 assumed → upper limit LWA,max", 15, th.muted,
           anchor="start")
    s.text(lx, 368, "normal surface velocity, A-weighted r.m.s.", 15,
           th.muted, anchor="start")


# ---------------------------------------------------------------------------
# Swept-sine distortion: deconvolution and the harmonic pre-arrivals
# ---------------------------------------------------------------------------

def _d_swept_sine(s: SVG, th: Theme) -> None:
    """Farina's exponential-sweep method: sweep through the weakly nonlinear
    DUT, deconvolve with the inverse filter, and the order-n distortion
    products compress into impulse responses L*ln(n) ahead of the linear
    one (L = 0.701 s for 20 Hz to 6 kHz in 4 s; 260 px per second)."""
    def box(x0: float, x1: float, y0: float, l1: str, l2: str,
            color: str) -> None:
        s.rect(x0, y0, x1 - x0, 76.0, th.panel, color, rx=10, sw=2)
        s.text((x0 + x1) / 2, y0 + 32.0, l1, 18, th.fg, bold=True)
        s.text((x0 + x1) / 2, y0 + 56.0, l2, 14, th.muted)

    box(60, 300, 64, "Exponential sweep x(t)", "20 Hz → 6 kHz in T = 4 s",
        th.fg)
    box(340, 560, 64, "Device under test", "weakly nonlinear: gain + harmonics",
        th.primary)
    box(600, 840, 64, "Recording y(t)", "sweep + distortion products", th.fg)
    s.arrow(300.0, 102.0, 336.0, 102.0, th.fg, 2.0)
    s.arrow(560.0, 102.0, 596.0, 102.0, th.fg, 2.0)
    box(520, 840, 180, "Deconvolve with the inverse filter",
        "time-reversed sweep with a +6 dB/octave tilt", th.secondary)
    s.arrow(720.0, 140.0, 720.0, 176.0, th.fg, 2.0)
    s.arrow(660.0, 256.0, 648.0, 298.0, th.fg, 2.0)

    # --- impulse-response timeline -----------------------------------------
    ax_y = 430.0
    s.line(80.0, ax_y, 830.0, ax_y, th.fg, 1.8)
    s.arrow(830.0, ax_y, 850.0, ax_y, th.fg, 1.8)
    s.text(845.0, 452.0, "time", 14, th.muted, anchor="end")
    s.line(640.0, ax_y - 5, 640.0, ax_y + 6, th.fg, 1.8)

    def ir(x0: float, amp: float, color: str) -> None:
        d = (f"M {x0:.0f} {ax_y:.0f} L {x0:.0f} {ax_y - amp:.0f} "
             f"L {x0 + 4:.0f} {ax_y:.0f} L {x0 + 10:.0f} {ax_y - amp * 0.45:.0f} "
             f"L {x0 + 16:.0f} {ax_y:.0f} L {x0 + 22:.0f} {ax_y - amp * 0.2:.0f} "
             f"L {x0 + 28:.0f} {ax_y:.0f} L {x0 + 36:.0f} {ax_y - amp * 0.08:.0f} "
             f"L {x0 + 44:.0f} {ax_y:.0f}")
        s.path(d, stroke=color, sw=2.0)

    ir(640.0, 94.0, th.primary)
    ir(514.0, 60.0, th.secondary)
    ir(440.0, 38.0, th.accent)
    ir(387.0, 22.0, th.muted)
    s.rect(630, 322, 66, 108, "none", th.primary, rx=8, sw=1.2, dash="5,4")
    s.rect(505, 358, 62, 72, "none", th.secondary, rx=8, sw=1.2, dash="5,4")
    s.rect(432, 382, 60, 48, "none", th.accent, rx=8, sw=1.2, dash="5,4")
    s.rect(380, 402, 54, 28, "none", th.muted, rx=6, sw=1.0, dash="5,4")
    s.text(663.0, 310.0, "h1 (linear), t = 0", 15, th.primary, bold=True)
    s.text(536.0, 346.0, "h2", 15, th.secondary, bold=True)
    s.text(462.0, 370.0, "h3", 15, th.accent, bold=True)
    s.text(398.0, 396.0, "h4", 13, th.muted)
    s.text(210.0, 344.0, "harmonic orders arrive early,", 15, th.muted,
           italic=True)
    s.text(210.0, 366.0, "each in its own window", 15, th.muted, italic=True)

    # Pre-arrival advances (260 px per second).
    s.dim(514.0, ax_y, 640.0, ax_y, "L·ln 2 = 0.49 s", offset=42, size=15)
    s.dim(440.0, ax_y, 640.0, ax_y, "L·ln 3 = 0.77 s", offset=80, size=15)

    s.text(450.0, 562.0,
           "L = T / ln(f2/f1) = 0.70 s here; the order-n products compress L·ln n ahead of the linear response",
           17, th.fg, bold=True)
    s.text(450.0, 590.0,
           "window each arrival  →  H1(f), H2(f), H3(f), …  →  THD(f) = √( Σ |Hn(nf)|² ) / |H1(f)|",
           16, th.primary)


# ---------------------------------------------------------------------------
# Programme loudness (ITU-R BS.1770 / EBU R 128)
# ---------------------------------------------------------------------------

def _d_program_loudness(s: SVG, th: Theme) -> None:
    """K-weighting, 400 ms blocks and the two gates into the integrated
    loudness of the guide's example (I = -23.1 LUFS, relative threshold
    -39.0 LUFS), with the LRA and true-peak branches beside the chain."""
    cx, bw = 450.0, 560.0
    x0 = cx - bw / 2

    def step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, 58, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 25, l1, 15, th.fg, bold=True)
        s.text(cx, y + 45, l2, 12, th.muted)

    step(52, "Programme x — channel weights Gi: 1.0 front, 1.41 surround",
         "anchor: a 0 dB FS 997 Hz sine on one front channel reads "
         "−3.01 LKFS", th.fg)
    step(138, "K-weighting: +4 dB spherical-head shelf + RLB high-pass",
         "LK = −0.691 + 10·log10 Σ Gi·zi;  LKFS ≡ LUFS, 1 LU = 1 dB",
         th.primary)
    step(224, "Mean square in 400 ms blocks, 75 % overlap",
         "absolute gate: blocks below −70 LUFS are dropped", th.primary)
    step(310, "Relative gate: −10 LU below the survivors",
         "example: 10 s at −23 dBFS + 30 s of quiet → threshold "
         "−39.0 LUFS", th.primary)
    s.rect(x0, 396, bw, 60, "none", th.accent, rx=10, sw=2.4)
    s.text(cx, 421, "Integrated loudness I = −23.1 LUFS: the tail is "
           "gated out", 16, th.fg, bold=True)
    s.text(cx, 443, "EBU R 128 target −23.0 LUFS; tolerance ±0.2 LU in "
           "QC, ±1.0 LU live", 12, th.muted)
    for y0, y1 in ((110, 134), (196, 220), (282, 306), (368, 392)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)

    # Side rails: LRA taps the K-weighted signal, true peak the raw one.
    s.line(170, 167, 120, 167, th.muted, 1.4)
    s.line(120, 167, 120, 484, th.muted, 1.4)
    s.arrow(120, 484, 120, 488, th.muted, 1.4)
    s.text(120, 157, "K-weighted", 11, th.muted)
    s.line(730, 81, 780, 81, th.muted, 1.4)
    s.line(780, 81, 780, 484, th.muted, 1.4)
    s.arrow(780, 484, 780, 488, th.muted, 1.4)
    s.text(780, 71, "raw signal", 11, th.muted)

    s.rect(70, 492, 360, 82, th.panel, th.secondary, rx=10, sw=2)
    s.text(250, 517, "Loudness range LRA = P95 − P10", 14, th.fg, bold=True)
    s.text(250, 538, "short-term 3 s windows, deeper −20 LU gate", 11,
           th.muted)
    s.text(250, 558, "10.0 LU on the Tech 3342 two-step case", 11,
           th.secondary, bold=True)
    s.rect(470, 492, 360, 82, th.panel, th.secondary, rx=10, sw=2)
    s.text(650, 517, "True peak: 4× oversampling, in dBTP", 14, th.fg,
           bold=True)
    s.text(650, 538, "the fs/4 tone: sample peak −3.01 dB, true peak "
           "+0.12 dBTP", 11, th.muted)
    s.text(650, 558, "R 128 production ceiling −1 dBTP", 11,
           th.secondary, bold=True)

    s.text(450, 618, "the gates keep quiet passages from dragging the "
           "foreground down", 14, th.fg)
    s.text(450, 642, "ungated, the same 40 s example would read near "
           "−29 LUFS", 13, th.muted)


# ---------------------------------------------------------------------------
# Noise control at the source, along the path and at the receiver
# ---------------------------------------------------------------------------

def _d_noise_control(s: SVG, th: Theme) -> None:
    """The source-path-receiver triad with the guide's numbers: a lined
    machine enclosure (IL = R - C = 25 dB at 500 Hz), a duct run with the
    m = 4 expansion chamber (TL peak 6.5 dB at 286 Hz), a lined elbow
    (6 dB at 1 kHz), the open-end reflection (18 dB at 63 Hz) and an
    operator cabin rated by the same IL = R - C (31 dB at 1 kHz)."""
    gy = 440.0
    s.ground(gy, 40.0, 860.0)
    for zx in (340.0, 660.0):
        s.line(zx, 80.0, zx, gy - 4, th.muted, 1.0, dash="7,7")
    s.text(185, 70, "1 · At the source", 19, th.fg, bold=True)
    s.text(480, 70, "2 · Along the path", 19, th.fg, bold=True)
    s.text(770, 70, "3 · At the receiver", 19, th.fg, bold=True)

    # --- source: machine inside a lined enclosure --------------------------
    s.rect(80, 306, 150, gy - 306, "none", th.primary, rx=6, sw=2.6)
    s.rect(90, 316, 130, gy - 320, "none", th.accent, rx=4, sw=1.3, dash="5,4")
    s.rect(100, 356, 110, 84, th.panel, th.fg, rx=6, sw=2)
    s.circle(130, 396, 14, th.primary)
    s.circle(130, 396, 5, th.bg)
    for r in (26, 42):
        s.path(f"M {130 + r * 0.3:.0f} {396 - r:.0f} "
               f"A {r} {r} 0 0 1 {130 + r:.0f} {396 - r * 0.3:.0f}",
               stroke=th.muted, sw=1.2)
    s.text(155, 296, "Enclosure", 17, th.primary, bold=True)
    s.text(178, 428, "Machine", 15, th.fg)
    s.text(185, 482, "enclosure IL = R − C", 15, th.primary, bold=True)
    s.text(185, 504, "25 dB at 500 Hz", 14, th.fg)

    # --- path: duct with expansion chamber, lined elbow and open end -------
    dt, db = 350.0, 374.0                # duct walls (24 px = 113 mm bore)
    ch_l, ch_r, ct, cb = 390.0, 480.0, 338.0, 386.0   # 0.30 m chamber
    s.line(230.0, dt, ch_l, dt, th.fg, 2.0)
    s.line(230.0, db, ch_l, db, th.fg, 2.0)
    s.rect(ch_l, ct, ch_r - ch_l, cb - ct, th.panel, th.primary, sw=2)
    s.line(ch_r, dt, 590.0, dt, th.fg, 2.0)
    s.line(ch_r, db, 614.0, db, th.fg, 2.0)
    s.line(590.0, dt, 590.0, 224.0, th.fg, 2.0)          # elbow, inner wall
    s.line(614.0, db, 614.0, 224.0, th.fg, 2.0)          # elbow, outer wall
    s.line(592.5, 348.0, 592.5, 240.0, th.accent, 2.0, dash="4,4")  # lining
    s.line(611.5, 360.0, 611.5, 240.0, th.accent, 2.0, dash="4,4")
    for r in (16, 28, 40):
        s.path(f"M {602 - r:.0f} {220:.0f} A {r} {r} 0 0 1 {602 + r:.0f} {220:.0f}",
               stroke=th.muted, sw=1.3)
    s.text(300, 338, "Ø 113 mm", 13, th.muted, mono=True)
    s.text(435, 326, "expansion chamber", 15, th.fg, bold=True)
    s.text(435, 424, "Ø 226 mm", 13, th.muted, mono=True)
    s.dim(ch_l, cb, ch_r, cb, "0.30 m", offset=18, size=14)
    s.text(560, 292, "lined elbow", 14, th.accent, anchor="end")
    s.line(566.0, 288.0, 590.0, 272.0, th.muted, 1.0)
    s.text(548, 170, "open end", 14, th.fg, anchor="end")
    s.line(554.0, 176.0, 572.0, 190.0, th.muted, 1.0)
    s.text(480, 482, "silencer TL peak 6.5 dB at 286 Hz (m = 4)", 14,
           th.primary, bold=True)
    s.text(480, 504, "lined elbow 6 dB at 1 kHz; open end 18 dB at 63 Hz",
           13, th.fg)

    # --- receiver: operator cabin ------------------------------------------
    s.rect(700, 300, 150, gy - 300, th.panel, th.fg, rx=4, sw=2.4)
    s.rect(716, 320, 54, 44, th.bg, th.muted, sw=1.5)
    s.person(806, gy, 92)
    s.text(775, 290, "Operator cabin", 17, th.fg, bold=True)
    s.text(770, 482, "cabin IL = R − C", 15, th.primary, bold=True)
    s.text(770, 504, "31 dB at 1 kHz", 14, th.fg)

    # --- captions ----------------------------------------------------------
    s.text(80, 540,
           "the classic ranking: quiet the source first, treat the path next, shield the receiver last",
           17, th.fg, anchor="start")
    # Both caption lines run the full width of the canvas; the smaller face is
    # what keeps the longest of them (the silencer one) off the right edge.
    s.text(80, 568,
           "enclosure and cabin share IL = R − C, with C = 10 log10(0.3 + S_E/R_i) = 4.9 dB for a lined interior (ᾱ = 0.3)",
           15, th.fg, anchor="start")
    s.text(80, 596,
           "reactive silencer: TL = 10 log10[1 + ¼ (m − 1/m)² sin²(kL)], peaking where the 0.3 m chamber is a quarter wavelength",
           15, th.muted, anchor="start")
