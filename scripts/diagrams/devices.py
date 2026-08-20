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
from .parts import _accel, _box_solid, _box_wire, _motion_arrows, _rot_arrow

# ---------------------------------------------------------------------------
# d6 - Two-microphone (p-p) intensity probe (IEC 61043)
# ---------------------------------------------------------------------------


def _d_pp_probe(s: SVG, th: Theme) -> None:
    ay = 232.0  # probe axis height

    # Measurement axis / intensity direction (drawn first, under the probe)
    s.line(70, ay, 820, ay, th.accent, 1.4, dash="10,4,2,4")
    s.arrow(820, ay, 852, ay, th.accent, 1.8)
    s.text(450, 305, "measurement axis / intensity direction", 15, th.accent)

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

    s.text(360, ay - 38, "$p_1$", 17, th.fg, bold=True)
    s.text(540, ay - 38, "$p_2$", 17, th.fg, bold=True)

    # Δr dimension between the capsule tips, drafting style
    s.dim(414, ay - 16, 486, ay - 16, "$Δr$ = 12 mm", offset=-66, size=15)

    # p-p estimator notes near the capsules
    s.text(280, 365, "$u$ from the $p_2−p_1$ gradient", 16, th.muted)
    s.text(620, 365, "$p = (p_1+p_2)/2$", 16, th.muted)


# ---------------------------------------------------------------------------
# d10 - ISO 3744/3746 sound power measurement surfaces
# ---------------------------------------------------------------------------


def _d_surfaces(s: SVG, th: Theme) -> None:
    # ===== Left panel: hemispherical surface over a reflecting plane =====
    cx, gy, R = 235.0, 420.0, 150.0
    s.text(cx, 74, "Hemispherical surface", 19, th.fg, bold=True)

    # Reflecting plane (hatched line through the equator / footprint centre).
    s.ground(gy, 55, 430)
    s.text(70, gy + 34, "Reflecting plane", 15, th.muted, anchor="start")

    # Hemisphere: dashed footprint ellipse + solid dome silhouette.
    ky = 0.30
    s.ellipse(cx, gy, R, R * ky, "none", th.muted, 1.3, dash="5,4")
    s.path(f"M {cx - R} {gy} A {R} {R} 0 0 1 {cx + R} {gy}", stroke=th.primary, sw=2.4)

    # Source box at the centre O.
    _box_solid(s, th, cx, gy, 30, 24, 34)
    s.circle(cx, gy, 3.4, th.fg)

    # Ten key microphone positions (ISO 3744 Table B.1), oblique-projected.
    b1 = [
        (0.16, -0.96, 0.22),
        (0.78, -0.60, 0.20),
        (0.78, 0.55, 0.31),
        (0.16, 0.90, 0.41),
        (-0.83, 0.32, 0.45),
        (-0.83, -0.40, 0.38),
        (-0.26, -0.65, 0.71),
        (0.74, -0.07, 0.67),
        (-0.26, 0.50, 0.83),
        (0.10, -0.10, 0.99),
    ]
    labelled = {1, 8, 10}
    pts = []
    for x, y, z in b1:
        px = cx + R * x + 42 * y
        py = gy - 34 * y - R * z
        pts.append((px, py))
    # radius r drawn to position 8 (a mid-height point on the surface).
    r8 = pts[7]
    s.line(cx, gy, r8[0], r8[1], th.accent, 1.6, dash="6,4")
    s.text(
        (cx + r8[0]) / 2 + 10,
        (gy + r8[1]) / 2 + 4,
        "radius $r ≥ 2 d_0$",
        15,
        th.accent,
        anchor="start",
    )
    for i, (px, py) in enumerate(pts, start=1):
        s.circle(px, py, 6.5, th.secondary)
        s.circle(px, py, 2.2, th.bg)
        if i in labelled:
            s.text(px, py - 12, str(i), 14, th.fg, bold=True)
    s.text(cx, gy + 62, "10 key positions (Table B.1)", 15, th.muted)
    s.text(cx, gy + 86, "one plane · $S = 2πr^2$", 15, th.primary, bold=True)

    # ===== Right panel: parallelepiped measurement surface =====
    bx2, gy2 = 675.0, 420.0
    s.text(bx2, 74, "Parallelepiped surface", 19, th.fg, bold=True)
    s.ground(gy2, 500, 872)

    # Source box (solid) enclosed by the measurement box (dashed wireframe).
    _box_solid(s, th, bx2, gy2, 46, 40, 58)
    _box_wire(s, th, bx2, gy2, 96, 90, 108, th.accent)
    s.text(bx2, gy2 + 40, "Measurement surface", 15, th.muted)
    s.text(bx2, gy2 + 64, "one plane · $S = 4(a·b+b·c+c·a)$", 15, th.accent, bold=True)

    # Measurement distance d: vertical clearance between the source top face
    # and the enveloping measurement surface (labelled arrow + caption above).
    s.text(bx2, 208, "measurement distance $d$", 15, th.secondary, bold=True)
    s.dim(bx2, gy2 - 108, bx2, gy2 - 58, "$d$", offset=0, size=17, label_side="right")


# ---------------------------------------------------------------------------
# d12 - Sound power methods comparison infographic
# ---------------------------------------------------------------------------


def _d_methods(s: SVG, th: Theme) -> None:
    """All six determination routes, two rows of three.

    Every cell carries the same five attributes -- environment, grade, measured
    quantity, headline relation and the limit that binds it -- so the columns
    can be read across. The sixth is ISO/TS 7849, the only route that takes no
    acoustic measurement at all.
    """
    cols = [
        (
            "ISO 3744 / 3746",
            "Free field over a reflecting plane",
            "Grade 2 / 3 (engineering / survey)",
            "Sound pressure · enveloping surface",
            "$L_W = L̄′_p + 10 log_{10}(S/S_0) − K_1 − K_2$",
            "$K_{2A}$ ≤ 4 dB (3744) / ≤ 7 dB (3746)",
            th.primary,
            "hemi",
        ),
        (
            "ISO 3745",
            "Qualified anechoic / hemi-anechoic room",
            "Grade 1 (precision)",
            "Sound pressure · fixed 20 / 40 array",
            "$L_W = L̄_p + 10 log_{10}(S/S_0) + C_1+C_2+C_3$",
            "$r ≥ 2 d_0$ , qualified free field",
            th.primary,
            "anech",
        ),
        (
            "ISO 3741",
            "Reverberation test room",
            "Grade 1 (precision)",
            "Sound pressure · diffuse field",
            "$L_W ← L̄_p , A , V , S , f$",
            "$V$ ≥ 200 m³ , source ≤ 2 % of $V$",
            th.accent,
            "reverb",
        ),
        (
            "ISO 9614-2",
            "In situ — any environment",
            "Grade 2 / 3 (engineering / survey)",
            "Sound intensity · scanning",
            "$L_W = 10 log_{10} |Σ I_i·S_i| / W_0$",
            "no non-positive bands · $F_{pI} < L_d$",
            th.secondary,
            "probe",
        ),
        (
            "ISO 9614-3",
            "In situ — any environment",
            "Grade 1 (precision)",
            "Sound intensity · scanning, tighter",
            "$L_W = 10 log_{10} |Σ I_i·S_i| / W_0$",
            "five Annex C criteria per band",
            th.secondary,
            "probe",
        ),
        (
            "ISO/TS 7849-1 / -2",
            "Any — no acoustic measurement",
            "Upper limit ($ε = 1$) / engineering",
            "Surface velocity · accelerometers",
            "$L_{WA} = L_{vA} + 10 lg(S/S_0) + 10 lg ε$",
            "$ε$ assumed (-1) or measured (-2)",
            th.fg,
            "accel",
        ),
    ]
    cw, ch, gap = 286.0, 300.0, 12.0
    x0 = (900 - (3 * cw + 2 * gap)) / 2
    for i, (name, env, grade, method, formula, note, col, pic) in enumerate(cols):
        x = x0 + (i % 3) * (cw + gap)
        ctop = 56.0 + (i // 3) * (ch + 20.0)
        cxc, cbot = x + cw / 2, ctop + ch
        s.rect(x, ctop, cw, ch, th.panel, col, rx=12, sw=2.2)
        s.rect(x, ctop, cw, 38, col, col, rx=12, sw=0)
        s.rect(x, ctop + 19, cw, 19, col, "none")  # square off header bottom
        s.text(cxc, ctop + 27, name, 16, th.bg, bold=True)

        # Mini-pictogram band, centred on py.
        py = ctop + 92.0
        if pic == "hemi":
            R = 44.0
            s.ellipse(cxc, py + 22, R, R * 0.3, "none", th.muted, 1.2, dash="4,3")
            s.path(
                f"M {cxc - R} {py + 22} A {R} {R} 0 0 1 {cxc + R} {py + 22}",
                stroke=col,
                sw=2.2,
            )
            s.line(cxc - R, py + 22, cxc + R, py + 22, th.muted, 1.4)
            _box_solid(s, th, cxc, py + 22, 10, 8, 13, stroke=col)
            for ang in (35, 90, 145):
                import math

                a = math.radians(ang)
                s.circle(
                    cxc + R * math.cos(a), py + 22 - R * math.sin(a), 4.0, th.secondary
                )
        elif pic == "anech":
            s.rect(cxc - 56, py - 24, 112, 68, th.bg, th.fg, rx=4, sw=1.8)
            for k in range(6):
                wx = cxc - 56 + k * 19
                s.path(
                    f"M {wx} {py - 24} L {wx + 19} {py - 24} L {wx + 9.5} {py - 12} Z",
                    fill=th.muted,
                    stroke="none",
                )
            s.line(cxc - 56, py + 44, cxc + 56, py + 44, th.fg, 2.0)
            _box_solid(s, th, cxc, py + 44, 10, 8, 13, stroke=col)
            for ang in (30, 90, 150):
                import math

                a = math.radians(ang)
                s.circle(
                    cxc + 42 * math.cos(a),
                    py + 44 - 42 * math.sin(a),
                    4.0,
                    th.secondary,
                )
        elif pic == "reverb":
            s.rect(cxc - 50, py - 22, 100, 68, "none", col, rx=6, sw=2.2)
            for k in range(3):
                yy = py - 10 + k * 18
                s.path(
                    f"M {cxc - 38} {yy} q 10 -10 20 0 q 10 10 20 0 q 10 -10 20 0",
                    stroke=th.muted,
                    sw=1.5,
                )
            s.circle(cxc - 34, py + 36, 5, th.secondary)
        elif pic == "accel":
            s.rect(cxc - 50, py + 6, 100, 34, th.panel, col, rx=3, sw=2.0)
            _accel(s, cxc + 18, py + 6)
            _motion_arrows(s, cxc - 20, py - 4, 12, th.secondary)
            for r in (18, 30):
                s.path(
                    f"M {cxc + 44 + r * 0.3:.1f} {py + 6 - r:.1f} "
                    f"A {r} {r} 0 0 1 {cxc + 44 + r:.1f} {py + 6 - r * 0.3:.1f}",
                    stroke=th.accent,
                    sw=1.5,
                )
        else:  # probe scanning a surface
            s.rect(cxc - 46, py - 26, 92, 76, "none", col, rx=6, sw=2.0)
            s.path(
                f"M {cxc - 36} {py - 14} L {cxc + 32} {py - 14} "
                f"L {cxc + 32} {py + 2} L {cxc - 36} {py + 2} "
                f"L {cxc - 36} {py + 18} L {cxc + 32} {py + 18}",
                stroke=th.accent,
                sw=1.7,
            )
            s.circle(cxc + 32, py + 18, 5, th.secondary)
            s.text(cxc, py + 44, "$I⊥$", 14, col, bold=True)

        # Attribute rows.
        for yy, txt, cc, bold in (
            (py + 78, env, th.fg, False),
            (py + 100, grade, col, True),
            (py + 122, method, th.muted, False),
        ):
            s.text(cxc, yy, txt, 11, cc, bold=bold)

        # Headline relation in a boxed footer, then the binding limit.
        s.rect(x + 6, cbot - 66, cw - 12, 36, "none", col, rx=8, dash="5,4")
        s.text(cxc, cbot - 43, formula, 10, th.fg, bold=True)
        s.text(cxc, cbot - 12, note, 11, th.muted)


# ---------------------------------------------------------------------------
# d19 - ISO 3745 precision sound power (anechoic / hemi-anechoic room)
# ---------------------------------------------------------------------------


def _d_precision_anechoic(s: SVG, th: Theme) -> None:
    """ISO 3745 precision sound power on a (hemi-)spherical array."""
    x0, y0, x1, gy = 60.0, 70.0, 840.0, 470.0
    s.rect(x0, y0, x1 - x0, gy - y0, th.bg, th.fg, sw=3)

    # Anechoic wedges lining the ceiling and the two side walls.
    for wx in range(int(x0) + 4, int(x1) - 36, 40):
        s.path(
            f"M {wx} {y0} L {wx + 40} {y0} L {wx + 20} {y0 + 28} Z",
            fill=th.panel,
            stroke=th.muted,
            sw=1.0,
        )
    for wy in range(int(y0) + 30, int(gy) - 36, 40):
        s.path(
            f"M {x0} {wy} L {x0} {wy + 40} L {x0 + 28} {wy + 20} Z",
            fill=th.panel,
            stroke=th.muted,
            sw=1.0,
        )
        s.path(
            f"M {x1} {wy} L {x1} {wy + 40} L {x1 - 28} {wy + 20} Z",
            fill=th.panel,
            stroke=th.muted,
            sw=1.0,
        )
    s.text(200, 120, "Anechoic wedges", 13, th.muted, anchor="start")

    # Reflecting floor (hemi-anechoic room).
    s.ground(gy, x0, x1)
    s.text(70, gy - 8, "Reflecting plane (hemi-anechoic)", 13, th.muted, anchor="start")

    # Source (DUT) at the centre of the reflecting plane.
    cx, R = 450.0, 200.0
    _box_solid(s, th, cx, gy, 34, 26, 40)
    s.circle(cx, gy, 3.4, th.fg)
    s.text(cx + 52, gy - 14, "Source (DUT)", 15, th.fg, bold=True, anchor="start")

    # Hemispherical measurement surface of radius r.
    s.ellipse(cx, gy, R, R * 0.16, "none", th.muted, 1.3, dash="5,4")
    s.path(f"M {cx - R} {gy} A {R} {R} 0 0 1 {cx + R} {gy}", stroke=th.primary, sw=2.4)

    # Ten normative microphone positions (ISO 3744/3745 Annex B), projected.
    b1 = [
        (0.16, -0.96, 0.22),
        (0.78, -0.60, 0.20),
        (0.78, 0.55, 0.31),
        (0.16, 0.90, 0.41),
        (-0.83, 0.32, 0.45),
        (-0.83, -0.40, 0.38),
        (-0.26, -0.65, 0.71),
        (0.74, -0.07, 0.67),
        (-0.26, 0.50, 0.83),
        (0.10, -0.10, 0.99),
    ]
    pts = [(cx + R * x + 46 * y, gy - 30 * y - R * z) for x, y, z in b1]
    r8 = pts[7]
    s.line(cx, gy, r8[0], r8[1], th.accent, 1.6, dash="6,4")
    s.text(
        (cx + r8[0]) / 2 + 8,
        (gy + r8[1]) / 2 + 2,
        "radius $r$",
        14,
        th.accent,
        anchor="start",
    )
    for px, py in pts:
        s.circle(px, py, 6.5, th.secondary)
        s.circle(px, py, 2.2, th.bg)
    s.text(688, 300, "20 / 40 mic positions", 14, th.muted, anchor="start")

    # Governing relations.
    for y, txt, col, bold in (
        (514, "$L_W = ⟨L_p⟩ + 10 log_{10}(S/S_0) + C_1 + C_2 + C_3$", th.fg, True),
        (540, "$S = 2πr^2$ (hemi-anechoic) · $4πr^2$ (anechoic)", th.primary, True),
        (564, "$K_1$: per-position background correction", th.muted, False),
        (
            587,
            ("$C_1$, $C_2$, $C_3$: meteorological corrections ($p_s$, $θ$, $a(f)$)"),
            th.muted,
            False,
        ),
    ):
        s.text(450, y, txt, 16 if bold else 15, col, bold=bold)


# ---------------------------------------------------------------------------
# d20 - ISO 9614-3 precision sound intensity scanning
# ---------------------------------------------------------------------------


def _d_intensity_scan(s: SVG, th: Theme) -> None:
    """ISO 9614-3 precision sound power by intensity scanning."""
    gy, bx = 470.0, 360.0

    # Measurement surface (dashed wireframe) enclosing the source.
    _box_wire(s, th, bx, gy, 150, 120, 240, th.primary)
    _box_solid(s, th, bx, gy, 45, 34, 70)
    s.text(bx, gy - 82, "Source", 15, th.fg, bold=True)
    # Above the wireframe's topmost edge rather than across its back
    # corners: the caption is 325 px in Spanish and 305 in English, and
    # both reach the box's slanted edges where it used to sit.
    s.text(bx, 150, "Measurement surface (segments $S_i$)", 15, th.primary, bold=True)

    # Segment grid on the front face (3 x 3 segments Sᵢ).
    fl, fr, ft, fb = bx - 150, bx + 150, gy - 240, gy
    for gx in (fl + 100, fl + 200):
        s.line(gx, ft, gx, fb, th.muted, 1.2, dash="4,4")
    for gyy in (ft + 80, ft + 160):
        s.line(fl, gyy, fr, gyy, th.muted, 1.2, dash="4,4")
    s.text(fl + 50, ft + 46, "$S_i$", 15, th.fg, bold=True)

    # Serpentine scan path across the segment-row centres.
    ys = (ft + 40, ft + 120, ft + 200)
    px = [
        (fl + 30, ys[0]),
        (fr - 30, ys[0]),
        (fr - 30, ys[1]),
        (fl + 30, ys[1]),
        (fl + 30, ys[2]),
        (fr - 30, ys[2]),
    ]
    for (ax, ay), (bxx, byy) in itertools.pairwise(px):
        s.line(ax, ay, bxx, byy, th.accent, 2.0, dash="2,3")
    s.arrow(px[-2][0] + 60, px[-1][1], px[-1][0], px[-1][1], th.accent, 2.0)
    s.text(fr + 8, ys[2] + 6, "serpentine scan", 13, th.accent, anchor="start")

    # A p-p intensity probe on the scan path.
    ppx, ppy = bx, ys[1]
    s.line(ppx, ppy, ppx + 46, ppy - 26, th.fg, 2.2)
    s.circle(ppx, ppy - 6, 5, th.fg)
    s.circle(ppx, ppy + 6, 5, th.fg)
    s.text(ppx + 52, ppy - 30, "p-p probe", 13, th.fg, anchor="start")

    # Normal-intensity arrows exiting the left column of segments.
    for yy in ys:
        s.arrow(fl, yy, fl - 34, yy + 8, th.secondary, 2.0)
    s.text(
        fl - 40, ys[1] + 30, "$I_n$ (normal intensity)", 13, th.secondary, anchor="end"
    )

    # Governing relations.
    for y, txt, col, bold in (
        (505, "$P = Σ I_{n,i} · S_i$   (partial powers per segment)", th.fg, True),
        (533, "$L_W = 10 log_{10}(P/P_0)$,  $P_0$ = 1 pW", th.accent, True),
        (559, "Field indicators: $F_{pIn}$ , $F_T$ , $F_S$", th.primary, True),
        (
            583,
            "Five acceptance criteria (Annex C); band invalid if $P < 0$",
            th.muted,
            False,
        ),
    ):
        s.text(450, y, txt, 16 if bold else 15, col, bold=bold)


# ---------------------------------------------------------------------------
# ISO 3741 reverberation test room: the direct and the comparison methods
# ---------------------------------------------------------------------------

_REV_MICS = (
    (3.6, 1.4),
    (6.0, 1.8),
    (6.4, 4.2),
    (4.8, 5.0),
    (2.8, 4.6),
    (1.2, 3.6),
)  #: six positions, in metres on the room floor plan
_REV_SOURCE = (1.7, 1.7)  #: source under test, asymmetric and 1,7 m off both walls


def _d_reverberation_power(s: SVG, th: Theme) -> None:
    """ISO 3741 reverberation test room drawn in plan, both methods.

    Left, the direct method (Eq. 20): the source on the floor at least 1,5 m
    from every wall and six microphone positions carrying the three clause 8.3
    clearances. Right, the comparison method (Eq. 21): the reference sound
    source run at the *same* six positions with the source under test left in
    the room. The room is the guide's own example, V = 200 m3, S = 220 m2,
    T60 = 2,0 s, so d_min = 0,08 sqrt(200/2) = 0,8 m (1,6 m recommended below
    5 kHz) and half a wavelength at 100 Hz is 1,7 m.
    """
    sc, base = 45.0, 352.0  # 45 px per metre; y of the plan origin

    def mp(x0: float, x: float, y: float) -> tuple[float, float]:
        return x0 + x * sc, base - y * sc

    def shell(x0: float, header: str) -> None:
        s.text(x0 + 180, 58, header, 17, th.fg, bold=True)
        # Hard walls, deliberately not parallel (the room's own skew).
        s.path(
            f"M {x0 - 6} {base + 10} L {x0 + 364} {base + 2} "
            f"L {x0 + 372} {base - 276} L {x0 + 2} {base - 268} Z",
            fill=th.panel,
            stroke=th.fg,
            sw=3.0,
        )

    def mic(px: float, py: float, n: str = "") -> None:
        s.circle(px, py, 6.5, th.secondary)
        s.circle(px, py, 2.2, th.bg)
        if n:
            s.text(px, py - 12, n, 13, th.fg, bold=True)

    # ===== Left panel: the direct method =====
    ax = 60.0
    shell(ax, "Direct method (Eq. 20)")

    # One hung diffuser, well clear of every fixed position.
    dpx, dpy = mp(ax, 0.9, 5.2)
    s.path(
        f"M {dpx - 24} {dpy - 12} L {dpx + 18} {dpy - 22} "
        f"L {dpx + 24} {dpy + 10} L {dpx - 18} {dpy + 20} Z",
        fill=th.bg,
        stroke=th.muted,
        sw=1.6,
    )
    s.text(dpx, dpy + 42, "diffuser", 12, th.muted)

    # Alternative continuous traverse, drawn under the fixed positions.
    s.ellipse(ax + 176, base - 140, 118.0, 86.0, "none", th.accent, 1.6, dash="7,5")
    s.arrow(ax + 292, base - 152, ax + 290, base - 128, th.accent, 1.6)
    s.text(ax + 176, base - 136, "or one continuous traverse", 12, th.accent)

    sx, sy = mp(ax, *_REV_SOURCE)
    for i, (mx, my) in enumerate(_REV_MICS, start=1):
        mic(*mp(ax, mx, my), str(i))
    # The three clearances, each measured on the position that binds it.
    m4x, m4y = mp(ax, *_REV_MICS[3])
    s.line(m4x, m4y, m4x, base - 272, th.muted, 1.0, dash="4,3")
    s.text(m4x + 8, (m4y + base - 272) / 2, "> 1,0 m", 13, th.fg, anchor="start")
    m1x, m1y = mp(ax, *_REV_MICS[0])
    s.line(sx, sy, m1x, m1y, th.primary, 1.4, dash="6,4")
    s.text((sx + m1x) / 2, sy - 10, "$> d_{min}$", 13, th.primary, bold=True)
    m5x, m5y = mp(ax, *_REV_MICS[4])
    m6x, m6y = mp(ax, *_REV_MICS[5])
    s.line(m5x, m5y, m6x, m6y, th.secondary, 1.4, dash="6,4")
    lx_, ly_ = (m5x + m6x) / 2, (m5y + m6y) / 2
    s.line(lx_, ly_, lx_, ly_ - 34, th.secondary, 1.0)
    s.text(lx_, ly_ - 40, "$≥ λ/2$", 13, th.secondary, bold=True)

    s.rect(sx - 16, sy - 13, 32, 26, th.fg, rx=3)
    s.text(sx - 22, sy + 34, "Source under test", 13, th.fg, bold=True, anchor="start")

    # ===== Right panel: the comparison method =====
    bx = 480.0
    shell(bx, "Comparison method (Eq. 21)")
    sx2, sy2 = mp(bx, *_REV_SOURCE)
    s.rect(sx2 - 16, sy2 - 13, 32, 26, th.panel, th.muted, rx=3, sw=1.6)
    s.text(
        sx2 - 22,
        sy2 + 34,
        "source under test, left in place",
        12,
        th.muted,
        anchor="start",
    )
    for mx, my in _REV_MICS:
        mic(*mp(bx, mx, my))
    rx_, ry_ = mp(bx, 4.2, 3.2)
    s.circle(rx_, ry_, 15.0, th.accent)
    s.circle(rx_, ry_, 6.0, th.bg)
    s.text(
        rx_ + 24, ry_ - 6, "Reference sound", 13, th.accent, bold=True, anchor="start"
    )
    s.text(
        rx_ + 24,
        ry_ + 12,
        "source (ISO 6926)",
        13,
        th.accent,
        bold=True,
        anchor="start",
    )
    s.line(sx2, sy2, rx_, ry_, th.muted, 1.2, dash="5,4")
    s.text((sx2 + rx_) / 2 + 6, (sy2 + ry_) / 2 + 22, "≥ 1,5 m", 13, th.fg)

    # ===== Footer: the rules the drawing cannot dimension =====
    s.line(50, 386, 850, 386, th.muted, 1.0)
    for k, (txt, col, bold) in enumerate(
        (
            (
                (
                    "$d_{min} = D_1 √(V / T_{60})$ ,  $D_1$ = 0,08  "
                    "(0,16 recommended below 5 kHz)"
                ),
                th.primary,
                True,
            ),
            (
                (
                    "$V$ = 200 m³ · $T_{60}$ = 2,0 s  →  $d_{min}$ = 0,8 m,  or 1,6 m "
                    "at the recommended $D_1$"
                ),
                th.primary,
                True,
            ),
            (
                (
                    "six positions: > 1,0 m from every room surface · $> d_{min}$ from "
                    "the source · spacing $≥ λ/2$ (1,7 m at 100 Hz)"
                ),
                th.fg,
                False,
            ),
            (
                (
                    "traverse instead: $≥ d_{min}$ from the source · ≥ 1,0 m from any "
                    "surface · ≥ 0,5 m from a diffuser"
                ),
                th.fg,
                False,
            ),
            (
                (
                    "· not within 10° of a room surface · length $≥ 3λ$ or 10,3 m, "
                    "whichever is smaller"
                ),
                th.fg,
                False,
            ),
            (
                (
                    "comparison method: the same six positions, and Eq. 21 without "
                    "$A$, $V$, $S$, Waterhouse or $C_1$"
                ),
                th.accent,
                True,
            ),
            (
                (
                    "averaging ≥ 30 s to 160 Hz, ≥ 10 s from 200 Hz · background at "
                    "those positions, before or after"
                ),
                th.secondary,
                True,
            ),
            (
                (
                    "hard walls, $α < 0,06$ within one wavelength of the source · "
                    "$T_{60}$ per ISO 3382-2, first 10 dB or 15 dB only"
                ),
                th.muted,
                False,
            ),
        )
    ):
        s.text(58, 412 + k * 23, txt, 13, col, anchor="start", bold=bold)


# ---------------------------------------------------------------------------
# ISO 3744 parallelepiped measurement surface and its microphone array
# ---------------------------------------------------------------------------


def _d_box_array(s: SVG, th: Theme) -> None:
    """ISO 3744 Annex C array on a box surface, in two orthographic views.

    The reference box is the guide's own 1,4 x 0,9 x 1,1 m machine and the
    measurement distance is d = 1 m, so a = 1,7 m, b = 1,45 m, c = 2,1 m and
    S = 4(ab + bc + ca) = 36,3 m2. The 3,4 m long faces exceed the clause C.1
    limit of 3d on a partial-area side, so they split in two and the array
    grows past the nine-position minimum.
    """
    sc = 70.0  # pixels per metre, shared by both views

    def key(px: float, py: float, normal: tuple[float, float] | None) -> None:
        if normal is not None:
            s.arrow(px, py, px + normal[0], py + normal[1], th.secondary, 1.4)
        s.circle(px, py, 6.0, th.secondary)
        s.circle(px, py, 2.0, th.bg)

    # ===== Left: top view =====
    tx, ty = 68.0, 92.0  # top-left corner of the measurement box
    tw, td = 3.4 * sc, 2.9 * sc  # 2a x 2b
    s.text(tx + tw / 2, 68, "Top view", 17, th.fg, bold=True)
    s.rect(tx, ty, tw, td, th.bg, th.accent, sw=2.2, dash="7,5")
    rw, rd = 1.4 * sc, 0.9 * sc  # reference box, in plan
    s.rect(tx + (tw - rw) / 2, ty + (td - rd) / 2, rw, rd, th.panel, th.fg, sw=2.0)
    s.text(tx + tw / 2, ty + (td + rd) / 2 + 20, "reference box", 13, th.fg)
    # The partial-area split of the top face and its key positions.
    s.line(tx + tw / 2, ty, tx + tw / 2, ty + td, th.muted, 1.4, dash="4,4")
    for px, py in (
        (tx, ty),
        (tx + tw / 2, ty),
        (tx + tw, ty),
        (tx, ty + td),
        (tx + tw / 2, ty + td),
        (tx + tw, ty + td),
        (tx + tw / 4, ty + td / 2),
        (tx + 3 * tw / 4, ty + td / 2),
    ):
        key(px, py, None)
    s.dim(tx + tw / 2, ty + td + 30, tx + tw, ty + td + 30, "$≤ 3d$", offset=0, size=14)
    s.dim(tx, ty + 34, tx + (tw - rw) / 2, ty + 34, "$d$ = 1 m", offset=0, size=14)
    s.text(
        tx + tw / 2,
        ty + td + 74,
        "$2a = l_1 + 2d$ = 3,4 m   ·   $2b$ = 2,9 m",
        14,
        th.accent,
    )

    # ===== Right: side view =====
    ex, gy = 512.0, 336.0  # left edge and the reflecting plane
    ew, eh = 3.4 * sc, 2.1 * sc  # 2a x c
    s.text(ex + ew / 2, 68, "Side view", 17, th.fg, bold=True)
    s.ground(gy, ex - 44, ex + ew + 44)
    s.text(ex - 40, gy + 30, "Reflecting plane", 13, th.muted, anchor="start")
    s.rect(ex, gy - eh, ew, eh, th.bg, th.accent, sw=2.2, dash="7,5")
    rwe, rhe = 1.4 * sc, 1.1 * sc  # reference box, in elevation
    s.rect(ex + (ew - rwe) / 2, gy - rhe, rwe, rhe, th.panel, th.fg, sw=2.0)
    s.text(ex + ew / 2, gy - rhe / 2 + 6, "source", 13, th.fg)
    s.line(ex + ew / 2, gy - eh, ex + ew / 2, gy - rhe, th.muted, 1.4, dash="4,4")
    ox, oy = ex + ew / 2, gy
    for px, py in (
        (ex + ew / 4, gy - eh / 2),
        (ex + 3 * ew / 4, gy - eh / 2),
        (ex + ew, gy - eh),
    ):
        key(px, py, None)
    # One position of each kind carries its reference direction: the top-face
    # centre normal to its face, the corner aimed at the origin O.
    key(ex + ew / 2, gy - eh, (0.0, 34.0))
    s.text(
        ex + ew / 2 + 10,
        gy - eh + 24,
        "normal to the face",
        12,
        th.muted,
        anchor="start",
    )
    dx_, dy_ = ox - ex, oy - (gy - eh)
    n = (dx_**2 + dy_**2) ** 0.5
    key(ex, gy - eh, (44 * dx_ / n, 44 * dy_ / n))
    s.text(
        ex + 6, gy - eh - 12, "at a corner: aimed at $O$", 12, th.muted, anchor="start"
    )
    s.circle(ox, oy, 3.8, th.fg)
    s.text(ox + 10, oy - 8, "$O$", 14, th.fg, bold=True, anchor="start")
    s.dim(
        ex + ew,
        gy,
        ex + ew,
        gy - eh,
        "$c$ = 2,1 m",
        offset=36,
        size=14,
        label_side="right",
    )

    # ===== Footer =====
    s.line(50, 418, 850, 418, th.muted, 1.0)
    for k, (txt, col, bold) in enumerate(
        (
            (
                (
                    "$S = 4(a·b + b·c + c·a)$ = 36,3 m²   for $l_1 × l_2 × l_3$ = "
                    "1,4 × 0,9 × 1,1 m at $d$ = 1 m"
                ),
                th.accent,
                True,
            ),
            (
                (
                    "each of the five planes is split on its own into equal partial "
                    "areas of side $≤ 3d$ (clause C.1)"
                ),
                th.fg,
                False,
            ),
            (
                (
                    "key positions: the centre of every partial area, plus its corners "
                    "except those in the reflecting plane"
                ),
                th.fg,
                False,
            ),
            (
                (
                    "nine is the minimum, for one partial area per plane; here "
                    "$2a > 3d$, so the long faces split and the array grows"
                ),
                th.muted,
                False,
            ),
            (
                "the survey method (ISO 3746) keeps only the partial-area centres",
                th.muted,
                False,
            ),
        )
    ):
        s.text(60, 446 + k * 25, txt, 13, col, anchor="start", bold=bold)


# ---------------------------------------------------------------------------
# ISO/TS 7849-2 determination of the radiation factor
# ---------------------------------------------------------------------------


def _d_radiation_factor(s: SVG, th: Theme) -> None:
    """The second, paired measurement Part 2 needs: an ISO 9614 intensity scan
    over a surface offset from the casing, taken on the same machine in the
    same operating mode as the velocity survey that runs on the casing."""
    gy = 400.0
    s.ground(gy, 60.0, 560.0)

    bx, hw, dp, ht = 280.0, 140.0, 108.0, 150.0
    _box_solid(s, th, bx, gy, hw, dp, ht)
    fx0, fx1, fy = bx - hw, bx + hw, gy - ht
    dxo, dyo = dp * 0.72, dp * 0.55

    # The Table 1 grid of the velocity survey: 5 x 2 cells on the 4 m2 face.
    for i in range(1, 5):
        gx = fx0 + i * (2 * hw) / 5
        s.line(gx, fy, gx + dxo, fy - dyo, th.muted, 1.0)
    s.line(
        fx0 + dxo * 0.5, fy - dyo * 0.5, fx1 + dxo * 0.5, fy - dyo * 0.5, th.muted, 1.0
    )
    cells = []
    for r_ in (0.25, 0.75):
        for i in range(5):
            u = (i + 0.5) / 5
            cells.append((fx0 + u * 2 * hw + r_ * dxo, fy - r_ * dyo))
    for cxp, cyp in cells:
        s.circle(cxp, cyp, 4.0, th.secondary)
        s.circle(cxp, cyp, 1.5, th.bg)
    _accel(s, cells[5][0], cells[5][1] - 4)
    _motion_arrows(s, cells[5][0], cells[5][1] - 46, 14, th.secondary)
    s.text(bx - 40, 120, "$⟨v_j^2⟩$ on the casing", 15, th.secondary, bold=True)
    s.line(bx - 10, 130, bx - 44, 196, th.muted, 1.0)

    # The ISO 9614 measurement surface, offset 0,25 m from the casing.
    off = 44.0
    _box_wire(s, th, bx, gy, hw + off, dp + off * 0.8, ht + off, th.primary)
    s.dim(fx1, fy - 26, fx1 + off, fy - 26, "0,25 m", offset=-30, size=14)
    s.text(bx, 84, "ISO 9614 measurement surface", 15, th.primary, bold=True)

    # A serpentine sweep over the front segment of that surface, with a probe.
    sl, sr = fx0 - off + 22, fx1 + off - 22
    for k, yy in enumerate((fy + 26, fy + 76, fy + 126)):
        s.line(sl, yy, sr, yy, th.accent, 1.8, dash="2,3")
        if k < 2:
            xx = sr if k % 2 == 0 else sl
            s.line(xx, yy, xx, yy + 50, th.accent, 1.8, dash="2,3")
    s.arrow(sr - 60, fy + 126, sr, fy + 126, th.accent, 1.8)
    ppx, ppy = bx + 10, fy + 76
    s.line(ppx, ppy, ppx + 42, ppy - 24, th.fg, 2.2)
    s.circle(ppx, ppy - 6, 5, th.fg)
    s.circle(ppx, ppy + 6, 5, th.fg)
    s.text(ppx + 48, ppy - 28, "p-p probe", 13, th.fg, anchor="start")

    # Radiated power leaving the surface.
    for yy in (fy + 26, fy + 76, fy + 126):
        s.arrow(fx0 - off, yy, fx0 - off - 34, yy + 8, th.primary, 2.0)
    s.text(fx0 - off - 44, fy + 82, "$P_j$", 15, th.primary, bold=True, anchor="end")

    # Relation strip.
    lx = 590.0
    s.text(lx, 120, "Determining $ε_j$ (Part 2)", 16, th.fg, bold=True, anchor="start")
    for y, txt, col, bold in (
        (154, "$ε_j = P_j / (Z_{c,n} ⟨v_j^2⟩ S)$", th.primary, True),
        (186, "$P_j$ : ISO 9614 band power", th.muted, False),
        (212, "$⟨v_j^2⟩$ : surface-averaged", th.muted, False),
        (238, "normal velocity, same bands", th.muted, False),
    ):
        s.text(lx, y, txt, 14 if bold else 13, col, anchor="start", bold=bold)
    s.rect(lx - 10, 268, 260, 118, th.panel, th.secondary, rx=10, sw=2.0)
    for k, txt in enumerate(
        (
            "one machine, one run:",
            "the same operating mode,",
            "the same mounting, the same bands.",
            "$ε_j$ is a property of the structure",
            "and its excitation together.",
        )
    ):
        s.text(lx, 292 + k * 22, txt, 13, th.fg, anchor="start", bold=(k == 0))
    s.text(
        450,
        450,
        "determined once, then the velocity survey alone serves the rest of the family",
        15,
        th.fg,
        bold=True,
    )
    s.text(
        450,
        478,
        "mean segment-to-source distance ≥ 200 mm (ISO 9614-2, clause 8.2)",
        14,
        th.muted,
    )
    s.text(
        450,
        506,
        "Part 1 skips this measurement and sets $ε = 1$, which "
        "is why it returns an upper limit",
        14,
        th.muted,
    )


# ---------------------------------------------------------------------------
# The residual-intensity test and the before-use probe check
# ---------------------------------------------------------------------------


def _d_residual_intensity_check(s: SVG, th: Theme) -> None:
    """What δpI0 is measured with, and the two checks that precede a series.

    Panel 1 is the IEC 61043 clause 10.2 residual-intensity testing device;
    panel 2 the clause 14 b) pressure calibration of both channels; panel 3
    the ISO 9614-2 clause 6.2.2 probe-reversal test in situ.
    """
    tops, w = (30.0, 315.0, 600.0), 270.0
    heads = (
        "1 · Residual-intensity test",
        "2 · Pressure check",
        "3 · Probe reversal, in situ",
    )
    # One size for the row, chosen on its longest heading: "1 · Ensayo de
    # intensidad residual" is 281 px against the 233 of its English twin,
    # and a panel that holds one does not hold the other.
    hsize = s.fit_size(heads, (15, 13), w - 24, bold=True)
    for x0, head in zip(tops, heads, strict=True):
        s.rect(x0, 62, w, 300, th.panel, th.muted, rx=12, sw=1.6)
        s.text(x0 + w / 2, 90, head, hsize, th.fg, bold=True)

    def capsules(cx: float, cy: float, col: str) -> None:
        for dx in (-30.0, 30.0):
            s.rect(cx + dx - 13, cy - 24, 26, 48, th.panel, col, rx=6, sw=2.0)
            s.rect(cx + dx - 8, cy - 30, 16, 8, th.fg, rx=2)
        s.rect(cx - 12, cy - 5, 24, 10, th.panel, th.muted, rx=3, sw=1.2)

    # --- 1: the coupler, both capsules in one identical pressure field ------
    x0 = tops[0]
    cx, cy = x0 + w / 2, 186.0
    s.rect(cx - 78, cy - 62, 156, 124, th.bg, th.fg, rx=10, sw=2.4)
    for k in range(3):
        s.path(
            f"M {cx - 60} {cy - 40 + k * 34} q 20 -14 40 0 q 20 14 40 0",
            stroke=th.primary,
            sw=1.6,
        )
    capsules(cx, cy, th.secondary)
    s.text(cx, cy + 84, "pink or white noise, 45 Hz to 7,1 kHz", 12, th.muted)
    s.text(cx, cy + 106, "both capsules within ± 0,1 dB", 12, th.muted)
    s.text(cx, cy + 134, "$δ_{pI0} = L_p − L_{I0}$", 15, th.primary, bold=True)

    # --- 2: the sound calibrator on one capsule at a time -------------------
    x1 = tops[1]
    cx, cy = x1 + w / 2, 186.0
    s.rect(cx - 74, cy - 40, 62, 80, th.bg, th.accent, rx=8, sw=2.2)
    s.text(cx - 43, cy + 4, "94,0", 13, th.accent, bold=True, mono=True)
    s.text(cx - 43, cy + 22, "dB", 12, th.accent, mono=True)
    s.rect(cx - 12, cy - 14, 22, 28, th.panel, th.secondary, rx=5, sw=2.0)
    s.rect(cx + 10, cy - 8, 44, 16, th.panel, th.muted, rx=4, sw=1.4)
    s.rect(cx + 54, cy - 14, 22, 28, th.panel, th.secondary, rx=5, sw=2.0)
    s.text(cx, cy + 66, "IEC 60942 calibrator, class 0 or 1", 12, th.muted)
    s.text(cx, cy + 88, "on each microphone in turn", 12, th.muted)
    s.text(cx, cy + 120, "adjust to ± 0,1 dB", 15, th.accent, bold=True)
    s.text(cx, cy + 144, "in both channels", 13, th.muted)

    # --- 3: the probe reversal on the measurement surface -------------------
    x2 = tops[2]
    cx, cy = x2 + w / 2, 196.0
    sxl = cx - 100
    s.line(sxl, cy - 78, sxl, cy + 78, th.fg, 2.6)
    for k in range(7):
        s.line(sxl, cy - 76 + k * 24, sxl - 9, cy - 67 + k * 24, th.muted, 1.1)
    s.text(cx, cy + 100, "measurement surface", 12, th.muted)
    s.arrow(sxl + 12, cy - 74, sxl + 92, cy - 74, th.muted, 2.0)
    s.text(sxl + 52, cy - 82, "energy leaving the source", 11, th.muted)
    for yy, col, lab, sgn in (
        (cy - 36, th.secondary, "$+ I_n$", 1.0),
        (cy + 36, th.primary, "$− I_n$", -1.0),
    ):
        s.line(cx - 34, yy, cx + 34, yy, th.fg, 2.0)
        s.circle(cx - 14, yy, 5.5, th.fg)
        s.circle(cx + 14, yy, 5.5, th.fg)
        s.arrow(cx, yy - 20, cx + sgn * 38, yy - 20, col, 2.0)
        s.text(cx + 50, yy + 5, lab, 13, col, bold=True, anchor="start")
    _rot_arrow(s, cx, cy, 46.0, 250.0, 470.0, th.muted, 1.8)
    s.text(cx - 52, cy + 5, "180°", 13, th.muted, anchor="end")
    s.text(cx, cy + 124, "acoustic centre held in place", 12, th.muted)

    # --- verdict strip ------------------------------------------------------
    s.rect(30, 390, 840, 62, th.panel, th.secondary, rx=10, sw=2.0)
    s.text(
        450,
        416,
        "accepted when the two readings have opposite signs and "
        "differ by less than 1,5 dB",
        15,
        th.fg,
        bold=True,
    )
    s.text(
        450,
        440,
        "in the band of maximum level (ISO 9614-2, clause 6.2.2)",
        13,
        th.muted,
    )
    for k, txt in enumerate(
        (
            "same signs → the two channels are swapped, or one is inverted",
            (
                "more than 1,5 dB apart → the probe disturbs its own field, or the "
                "channels are not matched"
            ),
            (
                "$δ_{pI0}$ belongs to the probe, its spacer and the analyser "
                "together — not to the microphones"
            ),
        )
    ):
        s.text(44, 482 + k * 24, txt, 13, th.muted, anchor="start")


def _d_loudspeaker_freefield(s: SVG, th: Theme) -> None:
    """IEC 60268-5 loudspeaker sensitivity on the reference axis (free field)."""
    x0, y0, x1, gy = 60.0, 70.0, 840.0, 470.0
    s.rect(x0, y0, x1 - x0, gy - y0, th.bg, th.fg, sw=3)

    # Anechoic wedges on all four boundaries (full free field: no floor).
    for wx in range(int(x0) + 4, int(x1) - 36, 40):
        s.path(
            f"M {wx} {y0} L {wx + 40} {y0} L {wx + 20} {y0 + 28} Z",
            fill=th.panel,
            stroke=th.muted,
            sw=1.0,
        )
        s.path(
            f"M {wx} {gy} L {wx + 40} {gy} L {wx + 20} {gy - 28} Z",
            fill=th.panel,
            stroke=th.muted,
            sw=1.0,
        )
    for wy in range(int(y0) + 30, int(gy) - 64, 40):
        s.path(
            f"M {x0} {wy} L {x0} {wy + 40} L {x0 + 28} {wy + 20} Z",
            fill=th.panel,
            stroke=th.muted,
            sw=1.0,
        )
        s.path(
            f"M {x1} {wy} L {x1} {wy + 40} L {x1 - 28} {wy + 20} Z",
            fill=th.panel,
            stroke=th.muted,
            sw=1.0,
        )
    s.text(210, 122, "Anechoic wedges", 13, th.muted, anchor="start")

    # Loudspeaker cabinet on a stand, reference point on the front baffle.
    ax_y, fx = 275.0, 250.0
    s.line(219, ax_y + 70, 219, 462, th.fg, 2.2)
    s.line(199, 462, 239, 462, th.fg, 2.2)
    s.rect(fx - 62, ax_y - 70, 62, 140, th.panel, th.primary, rx=6, sw=2)
    s.circle(fx - 18, ax_y, 14, th.primary)
    s.circle(fx - 18, ax_y, 5.5, th.bg)
    s.text(219, ax_y - 84, "Loudspeaker", 15, th.fg, bold=True)
    for r in (26, 44, 62):
        s.path(
            f"M {fx + r * 0.34:.1f} {ax_y - r * 0.94:.1f} "
            f"A {r} {r} 0 0 1 {fx + r * 0.34:.1f} {ax_y + r * 0.94:.1f}",
            stroke=th.accent,
            sw=1.5,
        )

    # Reference axis through the reference point, out to the right.
    s.circle(fx, ax_y, 3.4, th.fg)
    s.line(fx, ax_y, 782, ax_y, th.muted, 1.4, dash="7,5")
    s.arrow(760, ax_y, 792, ax_y, th.muted, 1.4)
    s.text(724, ax_y + 24, "Reference axis", 13, th.muted)

    # Measurement microphone on axis, capsule facing the loudspeaker.
    mx = 620.0
    s.line(mx + 23, ax_y + 6, mx + 23, 462, th.fg, 2.2)
    s.line(mx + 7, 462, mx + 39, 462, th.fg, 2.2)
    s.rect(mx, ax_y - 6, 46, 12, th.primary, rx=4)
    s.rect(mx - 12, ax_y - 4, 12, 8, th.fg, rx=2.5)
    s.text(mx + 24, ax_y - 24, "Measurement microphone", 15, th.fg, bold=True)

    # Reference distance, drafting style, between baffle and capsule tip.
    s.dim(fx, ax_y, mx - 12, ax_y, "$r$ = 1 m", offset=92)

    # Drive: amplifier delivering 1 W into the rated impedance.
    s.rect(85, 383, 140, 54, th.panel, th.primary, rx=8, sw=2)
    s.text(155, 405, "Amplifier", 15, th.fg, bold=True)
    s.text(155, 427, "2.83 V (8 Ω)", 13, th.secondary, mono=True)
    s.line(155, 383, 155, 345, th.fg, 1.6)
    s.line(155, 345, fx - 62, 345, th.fg, 1.6)

    # Governing relations. The Up, Lp(1 m) and LM relations stay plain: each
    # carries a unit as a factor or an argument (1 W, 1 m, 1 V/Pa), and the
    # composer has no upright run for a single-letter unit inside math.
    for y, txt, col, bold in (
        (
            508,
            (
                "Characteristic sensitivity: $L_p$ at 1 m for 1 W into the "
                "rated impedance"
            ),
            th.fg,
            True,
        ),
        (
            534,
            "Up = √(R · 1 W): 2.83 V is 1 W into 8 Ω but 2 W into 4 Ω (+3 dB)",
            th.secondary,
            True,
        ),
        (
            559,
            "Lp(1 m) = Lp(r) + 20 log10(r / 1 m)   (far field, inverse-distance law)",
            th.primary,
            True,
        ),
        (
            583,
            "Microphone (IEC 60268-4): M in mV/Pa, or LM = 20 log10(M / 1 V/Pa) dB",
            th.muted,
            False,
        ),
    ):
        s.text(450, y, txt, 16 if bold else 15, col, bold=bold)


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
    fx0, fx1 = bx - hw, bx + hw  # top-face front edge
    fy = gy - ht
    dxo, dyo = dp * 0.72, dp * 0.55

    # Measurement grid: 5 x 2 cells on the top face (the Table 1 initial
    # count N = 10 for a 1-10 m2 surface), a dot per cell centre.
    for i in range(1, 5):
        gx = fx0 + i * (2 * hw) / 5
        s.line(gx, fy, gx + dxo, fy - dyo, th.muted, 1.0)
    for f_row in (0.5,):
        s.line(
            fx0 + dxo * f_row,
            fy - dyo * f_row,
            fx1 + dxo * f_row,
            fy - dyo * f_row,
            th.muted,
            1.0,
        )
    pts = []
    for r_ in (0.25, 0.75):
        for i in range(5):
            u = (i + 0.5) / 5
            pts.append((fx0 + u * 2 * hw + r_ * dxo, fy - r_ * dyo))
    for px_, py_ in pts:
        s.circle(px_, py_, 4, th.secondary)
        s.circle(px_, py_, 1.5, th.bg)
    # One accelerometer drawn explicitly, with its vibratory motion.
    _accel(s, pts[5][0], pts[5][1] - 4)
    _motion_arrows(s, pts[5][0], pts[5][1] - 46, 16, th.secondary)
    s.text(250, 150, "Vibrating measurement surface $S$", 16, th.fg, bold=True)
    s.line(310, 160, 340, 228, th.muted, 1.0)

    # Radiated sound from the surface.
    for r in (36, 60, 84):
        s.path(
            f"M {475 + r * 0.30:.1f} {370 - r:.1f} "
            f"A {r} {r} 0 0 1 {475 + r:.1f} {370 - r * 0.30:.1f}",
            stroke=th.accent,
            sw=1.6,
        )
    s.text(672, 432, "radiated airborne sound", 14, th.accent)
    s.line(618, 424, 570, 372, th.muted, 1.0)

    # Dimensions of the surface (2.5 m x 1.6 m -> S = 4 m2).
    s.dim(fx0, gy, fx1, gy, "2.5 m", offset=32, size=15)
    s.arrow(fx1 + 20, gy + 18, fx1 + dxo + 14, gy - dyo + 18, th.muted, 1.2)
    s.arrow(fx1 + dxo + 14, gy - dyo + 18, fx1 + 20, gy + 18, th.muted, 1.2)
    s.text(fx1 + dxo - 4, gy - dyo + 46, "1.6 m", 15, th.fg, anchor="start")
    s.text(bx, 540, "Machine under test", 15, th.fg, bold=True)

    # Number of measurement positions and the survey relation.
    lx = 575.0
    s.text(
        lx, 110, "Initial number of positions $N$", 16, th.fg, bold=True, anchor="start"
    )
    for y, txt in (
        (140, "$S$ < 1 m²   →   5"),
        (166, "1 m² ≤ $S$ ≤ 10 m²  →  10"),
        (192, "$S$ > 10 m²  →  $S / S_0$"),
    ):
        s.text(lx, y, txt, 14, th.fg, anchor="start")
    s.text(
        lx,
        220,
        "one accelerometer per cell of area $S/N$",
        13,
        th.muted,
        anchor="start",
    )
    s.text(lx, 284, "Survey sound power", 16, th.fg, bold=True, anchor="start")
    # Two logarithms in one line: the smaller face keeps it inside the column.
    s.text(
        lx,
        314,
        "$L_{WA} = L_{vA} + 10 log_{10}(S/S_0) + 10 log_{10} ε$",
        11,
        th.primary,
        anchor="start",
        bold=True,
    )
    s.text(
        lx,
        342,
        "$ε = 1$ assumed → upper limit $L_{WA,max}$",
        13,
        th.muted,
        anchor="start",
    )
    s.text(
        lx,
        368,
        "normal surface velocity, A-weighted r.m.s.",
        13,
        th.muted,
        anchor="start",
    )


# ---------------------------------------------------------------------------
# Swept-sine distortion: deconvolution and the harmonic pre-arrivals
# ---------------------------------------------------------------------------


def _d_swept_sine(s: SVG, th: Theme) -> None:
    """Farina's exponential-sweep method: sweep through the weakly nonlinear
    DUT, deconvolve with the inverse filter, and the order-n distortion
    products compress into impulse responses L*ln(n) ahead of the linear
    one (L = 0.701 s for 20 Hz to 6 kHz in 4 s; 260 px per second)."""

    def box(x0: float, x1: float, y0: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y0, x1 - x0, 76.0, th.panel, color, rx=10, sw=2)
        s.text((x0 + x1) / 2, y0 + 32.0, l1, 15, th.fg, bold=True)
        s.text((x0 + x1) / 2, y0 + 56.0, l2, 12, th.muted)

    box(60, 300, 64, "Exponential sweep $x(t)$", "20 Hz → 6 kHz in $T$ = 4 s", th.fg)
    box(
        340,
        560,
        64,
        "Device under test",
        "weakly nonlinear: gain + harmonics",
        th.primary,
    )
    box(600, 840, 64, "Recording $y(t)$", "sweep + distortion products", th.fg)
    s.arrow(300.0, 102.0, 336.0, 102.0, th.fg, 2.0)
    s.arrow(560.0, 102.0, 596.0, 102.0, th.fg, 2.0)
    box(
        520,
        840,
        180,
        "Deconvolve with the inverse filter",
        "time-reversed sweep with a +6 dB/octave tilt",
        th.secondary,
    )
    s.arrow(720.0, 140.0, 720.0, 176.0, th.fg, 2.0)
    s.arrow(660.0, 256.0, 648.0, 298.0, th.fg, 2.0)

    # --- impulse-response timeline -----------------------------------------
    ax_y = 430.0
    s.line(80.0, ax_y, 830.0, ax_y, th.fg, 1.8)
    s.arrow(830.0, ax_y, 850.0, ax_y, th.fg, 1.8)
    s.text(845.0, 452.0, "time", 12, th.muted, anchor="end")
    s.line(640.0, ax_y - 5, 640.0, ax_y + 6, th.fg, 1.8)

    def ir(x0: float, amp: float, color: str) -> None:
        d = (
            f"M {x0:.0f} {ax_y:.0f} L {x0:.0f} {ax_y - amp:.0f} "
            f"L {x0 + 4:.0f} {ax_y:.0f} L {x0 + 10:.0f} {ax_y - amp * 0.45:.0f} "
            f"L {x0 + 16:.0f} {ax_y:.0f} L {x0 + 22:.0f} {ax_y - amp * 0.2:.0f} "
            f"L {x0 + 28:.0f} {ax_y:.0f} L {x0 + 36:.0f} {ax_y - amp * 0.08:.0f} "
            f"L {x0 + 44:.0f} {ax_y:.0f}"
        )
        s.path(d, stroke=color, sw=2.0)

    ir(640.0, 94.0, th.primary)
    ir(514.0, 60.0, th.secondary)
    ir(440.0, 38.0, th.accent)
    ir(387.0, 22.0, th.muted)
    s.rect(630, 322, 66, 108, "none", th.primary, rx=8, sw=1.2, dash="5,4")
    s.rect(505, 358, 62, 72, "none", th.secondary, rx=8, sw=1.2, dash="5,4")
    s.rect(432, 382, 60, 48, "none", th.accent, rx=8, sw=1.2, dash="5,4")
    s.rect(380, 402, 54, 28, "none", th.muted, rx=6, sw=1.0, dash="5,4")
    s.text(663.0, 310.0, "$h_1$ (linear), $t = 0$", 13, th.primary, bold=True)
    s.text(536.0, 346.0, "$h_2$", 13, th.secondary, bold=True)
    s.text(462.0, 370.0, "$h_3$", 13, th.accent, bold=True)
    s.text(398.0, 396.0, "$h_4$", 11, th.muted)
    s.text(210.0, 344.0, "harmonic orders arrive early,", 13, th.muted, italic=True)
    s.text(210.0, 366.0, "each in its own window", 13, th.muted, italic=True)

    # Pre-arrival advances (260 px per second).
    s.dim(514.0, ax_y, 640.0, ax_y, "$L·ln 2$ = 0.49 s", offset=42, size=13)
    s.dim(440.0, ax_y, 640.0, ax_y, "$L·ln 3$ = 0.77 s", offset=80, size=13)

    s.text(
        450.0,
        562.0,
        "$L = T / ln(f_2/f_1)$ = 0.70 s; the order-$n$ products compress "
        "$L·ln n$ ahead of the linear response",
        15,
        th.fg,
        bold=True,
    )
    s.text(
        450.0,
        590.0,
        "window each arrival  →  $H_{1}(f)$, $H_{2}(f)$, $H_{3}(f)$, …  →  "
        "$THD(f) = √( Σ |H_{n}(n f)|^2 ) / |H_{1}(f)|$",
        14,
        th.primary,
    )


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
        s.text(cx, y + 25, l1, 13, th.fg, bold=True)
        s.text(cx, y + 45, l2, 10, th.muted)

    step(
        52,
        "Programme $x$ — channel weights $G_i$: 1.0 front, 1.41 surround",
        "anchor: a 0 dB FS 997 Hz sine on one front channel reads −3.01 LKFS",
        th.fg,
    )
    step(
        138,
        "K-weighting: +4 dB spherical-head shelf + RLB high-pass",
        "$L_K = −0.691 + 10·log_{10} Σ G_i·z_i$;  LKFS ≡ LUFS, 1 LU = 1 dB",
        th.primary,
    )
    step(
        224,
        "Mean square in 400 ms blocks, 75 % overlap",
        "absolute gate: blocks below −70 LUFS are dropped",
        th.primary,
    )
    step(
        310,
        "Relative gate: −10 LU below the survivors",
        "example: 10 s at −23 dBFS + 30 s of quiet → threshold −39.0 LUFS",
        th.primary,
    )
    s.rect(x0, 396, bw, 60, "none", th.accent, rx=10, sw=2.4)
    s.text(
        cx,
        421,
        "Integrated loudness $I$ = −23.1 LUFS: the tail is gated out",
        14,
        th.fg,
        bold=True,
    )
    s.text(
        cx,
        443,
        "EBU R 128 target −23.0 LUFS; tolerance ±0.2 LU in QC, ±1.0 LU live",
        10,
        th.muted,
    )
    for y0, y1 in ((110, 134), (196, 220), (282, 306), (368, 392)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)

    # Side rails: LRA taps the K-weighted signal, true peak the raw one.
    s.line(170, 167, 120, 167, th.muted, 1.4)
    s.line(120, 167, 120, 484, th.muted, 1.4)
    s.arrow(120, 484, 120, 488, th.muted, 1.4)
    s.text(120, 157, "K-weighted", 10, th.muted)
    s.line(730, 81, 780, 81, th.muted, 1.4)
    s.line(780, 81, 780, 484, th.muted, 1.4)
    s.arrow(780, 484, 780, 488, th.muted, 1.4)
    s.text(780, 71, "raw signal", 10, th.muted)

    s.rect(70, 492, 360, 82, th.panel, th.secondary, rx=10, sw=2)
    s.text(250, 517, "Loudness range $LRA = P_{95} − P_{10}$", 12, th.fg, bold=True)
    s.text(250, 538, "short-term 3 s windows, deeper −20 LU gate", 10, th.muted)
    s.text(
        250, 558, "10.0 LU on the Tech 3342 two-step case", 10, th.secondary, bold=True
    )
    s.rect(470, 492, 360, 82, th.panel, th.secondary, rx=10, sw=2)
    s.text(650, 517, "True peak: 4× oversampling, in dBTP", 12, th.fg, bold=True)
    s.text(
        650,
        538,
        "the $f_s/4$ tone: sample peak −3.01 dB, true peak +0.12 dBTP",
        10,
        th.muted,
    )
    s.text(650, 558, "R 128 production ceiling −1 dBTP", 10, th.secondary, bold=True)

    s.text(
        450,
        618,
        "the gates keep quiet passages from dragging the foreground down",
        12,
        th.fg,
    )
    s.text(
        450,
        642,
        "ungated, the same 40 s example would read near −29 LUFS",
        11,
        th.muted,
    )


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
    s.text(185, 70, "1 · At the source", 16, th.fg, bold=True)
    s.text(480, 70, "2 · Along the path", 16, th.fg, bold=True)
    s.text(770, 70, "3 · At the receiver", 16, th.fg, bold=True)

    # --- source: machine inside a lined enclosure --------------------------
    s.rect(80, 306, 150, gy - 306, "none", th.primary, rx=6, sw=2.6)
    s.rect(90, 316, 130, gy - 320, "none", th.accent, rx=4, sw=1.3, dash="5,4")
    s.rect(100, 356, 110, 84, th.panel, th.fg, rx=6, sw=2)
    s.circle(130, 396, 14, th.primary)
    s.circle(130, 396, 5, th.bg)
    for r in (26, 42):
        s.path(
            f"M {130 + r * 0.3:.0f} {396 - r:.0f} "
            f"A {r} {r} 0 0 1 {130 + r:.0f} {396 - r * 0.3:.0f}",
            stroke=th.muted,
            sw=1.2,
        )
    s.text(155, 296, "Enclosure", 15, th.primary, bold=True)
    s.text(178, 428, "Machine", 13, th.fg)
    s.text(185, 482, "enclosure $IL = R − C$", 13, th.primary, bold=True)
    s.text(185, 504, "25 dB at 500 Hz", 12, th.fg)

    # --- path: duct with expansion chamber, lined elbow and open end -------
    dt, db = 350.0, 374.0  # duct walls (24 px = 113 mm bore)
    ch_l, ch_r, ct, cb = 390.0, 480.0, 338.0, 386.0  # 0.30 m chamber
    s.line(230.0, dt, ch_l, dt, th.fg, 2.0)
    s.line(230.0, db, ch_l, db, th.fg, 2.0)
    s.rect(ch_l, ct, ch_r - ch_l, cb - ct, th.panel, th.primary, sw=2)
    s.line(ch_r, dt, 590.0, dt, th.fg, 2.0)
    s.line(ch_r, db, 614.0, db, th.fg, 2.0)
    s.line(590.0, dt, 590.0, 224.0, th.fg, 2.0)  # elbow, inner wall
    s.line(614.0, db, 614.0, 224.0, th.fg, 2.0)  # elbow, outer wall
    s.line(592.5, 348.0, 592.5, 240.0, th.accent, 2.0, dash="4,4")  # lining
    s.line(611.5, 360.0, 611.5, 240.0, th.accent, 2.0, dash="4,4")
    for r in (16, 28, 40):
        s.path(
            f"M {602 - r:.0f} {220:.0f} A {r} {r} 0 0 1 {602 + r:.0f} {220:.0f}",
            stroke=th.muted,
            sw=1.3,
        )
    s.text(300, 338, "Ø 113 mm", 11, th.muted, mono=True)
    s.text(435, 326, "expansion chamber", 13, th.fg, bold=True)
    s.text(435, 424, "Ø 226 mm", 11, th.muted, mono=True)
    s.dim(ch_l, cb, ch_r, cb, "0.30 m", offset=18, size=12)
    s.text(560, 292, "lined elbow", 12, th.accent, anchor="end")
    s.line(566.0, 288.0, 590.0, 272.0, th.muted, 1.0)
    s.text(548, 170, "open end", 12, th.fg, anchor="end")
    s.line(554.0, 176.0, 572.0, 190.0, th.muted, 1.0)
    s.text(
        480,
        482,
        "silencer TL peak 6.5 dB at 286 Hz ($m = 4$)",
        12,
        th.primary,
        bold=True,
    )
    s.text(480, 504, "lined elbow 6 dB at 1 kHz; open end 18 dB at 63 Hz", 11, th.fg)

    # --- receiver: operator cabin ------------------------------------------
    s.rect(700, 300, 150, gy - 300, th.panel, th.fg, rx=4, sw=2.4)
    s.rect(716, 320, 54, 44, th.bg, th.muted, sw=1.5)
    s.person(806, gy, 92)
    s.text(775, 290, "Operator cabin", 15, th.fg, bold=True)
    s.text(770, 482, "cabin $IL = R − C$", 13, th.primary, bold=True)
    s.text(770, 504, "31 dB at 1 kHz", 12, th.fg)

    # --- captions ----------------------------------------------------------
    s.text(
        80,
        540,
        "the classic ranking: quiet the source first, treat the path next, shield the receiver last",
        15,
        th.fg,
        anchor="start",
    )
    # Both caption lines run the full width of the canvas; the smaller face is
    # what keeps the longest of them (the silencer one) off the right edge.
    s.text(
        80,
        568,
        "enclosure and cabin share $IL = R − C$, with "
        "$C = 10 log_{10}(0.3 + S_E/R_i)$ = 4.9 dB for a lined interior "
        "($ᾱ = 0.3$)",
        13,
        th.fg,
        anchor="start",
    )
    s.text(
        80,
        596,
        "reactive silencer: $TL = 10 log_{10}[1 + ¼ (m − 1/m)^2 "
        "sin^{2}(k·L)]$, peaking where the 0.3 m chamber is $λ/4$",
        13,
        th.muted,
        anchor="start",
    )


# ---------------------------------------------------------------------------
# IEC 60268-3 electrical distortion bench
# ---------------------------------------------------------------------------


def _d_duct_path(s: SVG, th: Theme) -> None:
    """Long's Table 14.9 installation as a place, not as a thirteen-row table.

    Every box carries the sheet code the snippet stamps on its
    ``DuctElement``, so the two paths of ``duct_path_cascade.svg`` become
    readable as positions in a building. The stamps under each box are the
    element's own two spectra at 63 Hz: what it takes out, and what its
    airflow puts back.
    """
    s.add('<g transform="translate(0,-28)">')
    plant_x, room_x = 60.0, 596.0
    ceil_y, floor_y = 300.0, 470.0

    # --- the two spaces ----------------------------------------------------
    s.ground(floor_y, 40.0, 860.0)
    s.rect(plant_x, 150.0, 250.0, floor_y - 150.0, th.panel, th.fg, sw=2.4)
    s.rect(room_x, ceil_y, 244.0, floor_y - ceil_y, th.panel, th.fg, sw=2.4)
    s.text(plant_x + 10, 172, "Plant room", 14, th.fg, bold=True, anchor="start")
    s.text(
        room_x + 12, 322, "Office, 20 × 20 × 8 ft", 14, th.fg, bold=True, anchor="start"
    )
    s.text(room_x + 12, 342, "drywall and carpet, NC 30", 11, th.muted, anchor="start")
    s.line(room_x, ceil_y, room_x + 244.0, ceil_y, th.fg, 2.4)

    # --- the fan, radiating both ways --------------------------------------
    s.rect(plant_x + 66, 356, 116, 94, th.bg, th.fg, rx=6, sw=2.4)
    s.circle(plant_x + 124, 403, 26, "none", th.primary, sw=2.6)
    s.circle(plant_x + 124, 403, 7, th.primary)
    s.text(
        plant_x + 124, 342, "1 · Fan, 5000 cfm, 2 in w.g.", 12, th.primary, bold=True
    )
    s.text(plant_x + 124, 466, "same $L_W$ into both runs", 11, th.muted)

    # --- the supply run, left to right, above the office ceiling -----------
    supply = (
        ("2", "elbow", "36×24 in", 246.0, 62.0),
        ("3", "silencer", "3 ft", 320.0, 66.0),
        ("4", "lined duct", "5 ft", 396.0, 70.0),
        ("5", "branch", "25 %", 478.0, 56.0),
        ("6", "lined duct", "18×12 in", 546.0, 74.0),
        ("7", "flex duct", "12 in, 6 ft", 632.0, 72.0),
    )
    y = 224.0
    s.line(plant_x + 182, 240.0, 246.0, 240.0, th.fg, 2.2)
    for code, name, size, x, w in supply:
        s.rect(x, y, w, 34.0, th.bg, th.primary, rx=3, sw=2.0)
        s.text(x + w / 2, y + 23, code, 13, th.primary, bold=True)
        s.text(x + w / 2, y - 8, name, 10, th.fg)
        s.text(x + w / 2, y + 50, size, 10, th.muted, mono=True)
    for a, b in (
        (308.0, 320.0),
        (386.0, 396.0),
        (466.0, 478.0),
        (534.0, 546.0),
        (620.0, 632.0),
    ):
        s.line(a, y + 17, b, y + 17, th.fg, 2.2)
    s.line(704.0, y + 17, 780.0, y + 17, th.fg, 2.2)
    s.line(780.0, y + 17, 780.0, ceil_y - 8, th.fg, 2.2)
    s.rect(760.0, ceil_y - 8, 40.0, 12.0, th.bg, th.primary, sw=2.0)
    s.text(752.0, 296.0, "8 · diffuser 24 × 24 in", 11, th.primary, anchor="end")

    # --- the return run, below the supply, back to the plant room ----------
    ret = (
        ("2", "elbow", 246.0, 62.0),
        ("3", "silencer", 320.0, 74.0),
        ("4", "elbow, lined", 404.0, 74.0),
        ("5", "plenum", 488.0, 82.0),
    )
    yr = 140.0
    for code, name, x, w in ret:
        s.rect(x, yr, w, 30.0, th.bg, th.secondary, rx=3, sw=2.0)
        s.text(x + w / 2, yr + 21, code, 12, th.secondary, bold=True)
        s.text(x + w / 2, yr - 8, name, 10, th.fg)
    for a, b in ((308.0, 320.0), (394.0, 404.0), (478.0, 488.0)):
        s.line(a, yr + 15, b, yr + 15, th.fg, 2.2)
    s.line(570.0, yr + 15, 806.0, yr + 15, th.fg, 2.2)
    s.line(806.0, yr + 15, 806.0, 196.0, th.fg, 2.2)
    s.text(824.0, yr + 20, "6 · grille", 11, th.secondary, anchor="start")
    s.line(plant_x + 182, yr + 15, 246.0, yr + 15, th.fg, 2.2)
    s.arrow(246.0, yr + 15, plant_x + 190, yr + 15, th.secondary, 2.0)

    # --- the receiver ------------------------------------------------------
    s.person(room_x + 214, floor_y, 82, seated=True)
    # Clear of the room's caption block, which in Spanish reaches x = 782.
    s.dim(788.0, ceil_y + 6, 788.0, floor_y - 68, "$r$ = 1.83 m", size=11)
    s.text(
        room_x + 12, 392, "$Q = 2$, flush in the ceiling", 11, th.muted, anchor="start"
    )

    # --- the legend --------------------------------------------------------
    s.text(
        60,
        520,
        "blue: the supply path — red: the return path — each box "
        "is one row of the sheet, stamped with its code",
        13,
        th.fg,
        anchor="start",
    )
    s.text(
        60,
        546,
        "attenuates only: 4, 5, 6 (supply) and 4, 5 (return) — "
        "attenuates and regenerates: 2, 3 — self-noise only: "
        "8 and the grille",
        12,
        th.muted,
        anchor="start",
    )
    s.text(
        60,
        574,
        "the return wins above 1 kHz: its silencer floors the room "
        "near 25 dB, and no amount of supply attenuation moves that",
        12,
        th.muted,
        anchor="start",
    )
    s.add("</g>")


def _d_silencer_iso7235(s: SVG, th: Theme) -> None:
    """The ISO 7235 substitution measurement, both series on one duct axis.

    Series I carries the test object, series II the substitution duct, and
    the insertion loss is the difference of the two spatially energy-averaged
    receiving-side levels. The annotations are the standard's own
    qualification numbers: the modal filter's >= 3 dB / >= 5 dB longitudinal
    attenuation (5.2.2.3), r <= 0.3 at both qualification planes (5.2.2.5,
    5.2.4.3), the 5 % linear tolerance on the substitution duct (5.2.3), the
    >= 6 dB and preferably >= 10 dB signal-to-background rule (5.2.2.2), and
    the three microphone positions of 6.2.1.
    """
    x_box, w_box = 60.0, 105.0
    x_mf, w_mf = 190.0, 80.0
    x_tr, w_tr = 285.0, 70.0
    x_ob, w_ob = 375.0, 150.0
    x_rd, w_rd = 545.0, 285.0
    duct_h = 56.0

    for row, (y, title, obj, colour) in enumerate(
        (
            (150.0, "Series I — test object installed", "test object", th.primary),
            (330.0, "Series II — substitution duct", "substitution duct", th.secondary),
        )
    ):
        ax = y + duct_h / 2
        s.text(60, y - 24, title, 15, colour, bold=True, anchor="start")

        s.rect(x_box, y - 16, w_box, duct_h + 32, th.panel, th.fg, rx=5, sw=2.2)
        s.rect(
            x_box + 6,
            y - 10,
            w_box - 12,
            duct_h + 20,
            "none",
            th.accent,
            rx=3,
            sw=1.2,
            dash="4,3",
        )
        s.circle(x_box + w_box / 2, ax, 15, th.fg)
        s.circle(x_box + w_box / 2, ax, 6, th.bg)

        s.rect(x_mf, y, w_mf, duct_h, th.panel, th.fg, sw=2.0)
        for k in range(3):
            s.line(
                x_mf + 14 + 26 * k,
                y + 4,
                x_mf + 14 + 26 * k,
                y + duct_h - 4,
                th.accent,
                3.0,
            )
        s.path(
            f"M {x_tr} {y - 6} L {x_tr + w_tr} {y + 6} "
            f"L {x_tr + w_tr} {y + duct_h - 6} L {x_tr} {y + duct_h + 6} Z",
            fill=th.panel,
            stroke=th.fg,
            sw=2.0,
        )
        s.rect(x_ob, y - 10, w_ob, duct_h + 20, th.panel, colour, rx=4, sw=2.6)
        if row == 0:
            for k in range(4):
                s.line(
                    x_ob + 18 + 32 * k,
                    y - 4,
                    x_ob + 18 + 32 * k,
                    y + duct_h + 4,
                    colour,
                    2.4,
                )
        s.text(x_ob + w_ob / 2, y + duct_h + 40, obj, 12, colour, bold=True)

        s.rect(x_rd, y, w_rd, duct_h, th.panel, th.fg, sw=2.0)
        for k in range(3):
            s.path(
                f"M {x_rd + w_rd - 10 - 24 * k} {y + 2} "
                f"L {x_rd + w_rd - 34 - 24 * k} {ax} "
                f"L {x_rd + w_rd - 10 - 24 * k} {y + duct_h - 2} Z",
                fill="none",
                stroke=th.muted,
                sw=1.3,
            )
        s.line(x_rd + 44, ax - 21, x_rd + 128, ax + 17, th.muted, 1.1, dash="4,3")
        for k in range(3):
            s.circle(x_rd + 50 + 36 * k, ax - 18 + 17 * k, 6, th.fg)

        s.arrow(x_box + w_box, ax, x_mf, ax, th.fg, 2.0)
        s.arrow(x_mf + w_mf, ax, x_tr, ax, th.fg, 2.0)
        s.arrow(x_ob + w_ob, ax, x_rd, ax, th.fg, 2.0)
        s.line(x_ob - 12, y - 24, x_ob - 12, y + duct_h + 24, th.muted, 1.2, dash="3,3")
        s.line(x_rd - 8, y - 24, x_rd - 8, y + duct_h + 24, th.muted, 1.2, dash="3,3")
        s.text(
            x_rd + w_rd,
            y - 10,
            "$L_{pI}$" if row == 0 else "$L_{pII}$",
            13,
            colour,
            anchor="end",
        )

    # --- what each element is ----------------------------------------------
    for x, label in (
        (x_box + w_box / 2, "sealed, lined loudspeaker box"),
        (x_mf + w_mf / 2, "modal filter"),
        (x_tr + w_tr / 2, "transition"),
        (x_rd + w_rd / 2, "test duct, anechoic termination"),
    ):
        s.text(x, 468, label, 10, th.muted)
    s.text(
        x_rd + w_rd / 2,
        488,
        "three positions on a line inclined to the axis, at mid-length",
        10,
        th.muted,
    )
    s.text(x_ob + w_ob / 2, 488, "$r ≤ 0.3$ planes", 10, th.muted)

    # --- the quantity and its qualification rules ---------------------------
    s.text(
        60,
        532,
        "$D_i = L_{pI} − L_{pII}$, one third octave at a time",
        15,
        th.fg,
        anchor="start",
    )
    left = (
        "modal filter: ≥ 3 dB on the fundamental at the low-frequency end,",
        "≥ 5 dB above the cut-on of higher-order modes (5.2.2.3)",
        "substitution duct: the empty housing where possible, otherwise",
        "matched within 5 % in every linear dimension (5.2.3)",
    )
    right = (
        "reflection coefficient $r ≤ 0.3$ at the source and receiving",
        "qualification planes (5.2.2.5, 5.2.4.3)",
        "signal ≥ 6 dB and preferably ≥ 10 dB above the background",
        "(5.2.2.2); IEC 61260 third octaves, class 1 chain (5.2.4.6)",
    )
    for k, line in enumerate(left):
        s.text(60, 562 + 20 * k, line, 10, th.muted, anchor="start")
    for k, line in enumerate(right):
        s.text(468, 562 + 20 * k, line, 10, th.muted, anchor="start")

    s.text(
        60,
        668,
        "the reported figure is an insertion loss against a "
        "substitution duct, not a transmission loss",
        14,
        th.fg,
        anchor="start",
    )
    s.text(
        60,
        692,
        "and the facility's own limiting insertion loss — flanking "
        "along the duct walls — caps what it can report at all",
        13,
        th.muted,
        anchor="start",
    )


def _d_room_to_room(s: SVG, th: Theme) -> None:
    """Section through Norton problem 4.18, drawn at 50 px per metre.

    Every symbol of Equation (4.101) is a piece of this drawing: the blower
    in the floor-wall intersection that makes Q = 4 worth 6.0 dB, the
    reverberant L_p1 that drives the whole partition, the 5 x 3 m separating
    wall, the receiving room whose absorption area overtakes that wall
    between 250 and 500 Hz, and the receiver against NC 45. The annotations
    sit outside the rooms because 3 m of room is 150 px of drawing.
    """
    cl, fl = 262.0, 412.0  # ceiling and floor: 3 m at 50 px/m
    xa, xb = 130.0, 530.0  # plant room, 8 m
    xc, xd = 542.0, 792.0  # operator room, 5 m

    # The scene is laid out on its own grid and lifted clear of the title.
    s.add('<g transform="translate(0,-44)">')
    s.ground(fl, 90.0, 830.0)
    for x0, x1 in ((xa, xb), (xc, xd)):
        s.rect(x0, cl, x1 - x0, fl - cl, th.panel, th.fg, sw=2.4)
        s.line(x0 + 4, cl + 7, x1 - 4, cl + 7, th.accent, 2.4, dash="6,4")
    s.rect(xb, cl, xc - xb, fl - cl, th.secondary, th.fg, sw=2.0)
    s.line(xc + 4, fl - 5, xd - 4, fl - 5, th.accent, 2.4, dash="2,4")

    s.text(
        xa + 10, 288, "Plant room 8 × 10 × 3 m", 14, th.fg, bold=True, anchor="start"
    )
    s.text(xa + 10, 306, "bare floor, absorbent ceiling", 11, th.muted, anchor="start")
    s.text(
        xd - 10, 288, "Operator room 5 × 5 × 3 m", 14, th.fg, bold=True, anchor="end"
    )
    s.text(xd - 10, 306, "carpet, same ceiling", 11, th.muted, anchor="end")
    s.text(xb + 6, 250, "$S_w$", 13, th.secondary, bold=True)

    # --- the blower in the floor-wall intersection: Q = 4 ------------------
    s.path(
        f"M {xa} {fl} L {xa + 104} {fl} A 104 104 0 0 0 {xa} {fl - 104} Z",
        fill=th.panel,
        stroke=th.primary,
        sw=1.4,
    )
    s.rect(xa + 2, fl - 42, 58, 42, th.bg, th.fg, rx=4, sw=2.2)
    s.circle(xa + 31, fl - 21, 11, th.primary)
    s.circle(xa + 31, fl - 21, 4, th.bg)
    for r in (58, 84):
        s.path(
            f"M {xa + 31 + r * 0.30:.0f} {fl - 21 - r:.0f} "
            f"A {r} {r} 0 0 1 {xa + 31 + r:.0f} {fl - 21 - r * 0.30:.0f}",
            stroke=th.muted,
            sw=1.1,
        )
    s.text(xa + 118, fl - 84, "$Q = 4$", 13, th.primary, anchor="start", bold=True)

    # --- the path across the partition -------------------------------------
    s.arrow(xa + 250, 350, xb - 6, 350, th.primary, 2.2)
    s.text(xa + 250, 340, "$L_{p1}$", 12, th.primary, anchor="start")
    s.arrow(xc + 6, 350, xd - 78, 350, th.primary, 2.2)
    s.text(xd - 86, 340, "$L_{p2}$", 12, th.primary, anchor="end")
    s.arrow(xc + 6, 380, xb - 6, 380, th.muted, 1.0)
    s.text(xc + 14, 376, "$τ S_w$", 10, th.muted, anchor="start")
    s.person(xd - 40, fl, 78)

    # --- the flanking route the equation does not price --------------------
    s.path(
        f"M {xb - 120} {cl - 4} Q {xb + 6} {186} {xc + 120} {cl - 4}",
        stroke=th.secondary,
        sw=2.0,
        dash="7,5",
    )
    s.text(
        xb + 6,
        172,
        "flanking over the ceiling void: an allowance, not a model",
        11,
        th.secondary,
    )

    # --- dimensions ---------------------------------------------------------
    s.dim(xa, fl + 30, xb, fl + 30, "8 m", size=13)
    s.dim(xc, fl + 30, xd, fl + 30, "5 m", size=13)
    s.dim(xa - 46, cl, xa - 46, fl, "3 m", size=13)

    # --- the numbers, outside the rooms -------------------------------------
    s.text(
        50,
        480,
        "source side — blower on the floor at a wall mid-point: "
        "$L_W$ = 105 dB at 125 Hz, $Q = 4$ adds 6.0 dB, "
        "$L_{p1}$ = 107.0 dB",
        12,
        th.fg,
        anchor="start",
    )
    s.text(
        50,
        502,
        "partition — 5 m × 3 m = 15 m², TL = 39 dB at 125 Hz; "
        "the $τ S_w$ returned to the source room is off by default",
        12,
        th.fg,
        anchor="start",
    )
    s.text(
        50,
        524,
        "receiving side — $S_{2}α_{2}$ = 5.5 m² at 125 Hz rising to "
        "39.2 m² at 4 kHz; $L_{p2}$ = 72.4 dB against 60 dB for "
        "NC 45",
        12,
        th.fg,
        anchor="start",
    )

    # --- the equation and where it turns over -------------------------------
    s.text(
        50,
        558,
        "$NR = TL − 10 log_{10}[S_w / (S_{2}α_{2} + τ S_w)]$",
        15,
        th.fg,
        anchor="start",
    )
    s.text(
        50,
        584,
        "$S_{2}α_{2}$ passes the 15 m² of the wall between 250 and "
        "500 Hz: below it the wall delivers less than its TL",
        13,
        th.primary,
        anchor="start",
    )
    s.text(
        50,
        608,
        "both levels are reverberant-field spatial averages; the "
        "balance says nothing below 163 Hz (Schroeder, 75 m³)",
        13,
        th.muted,
        anchor="start",
    )
    s.add("</g>")


def _d_machine_enclosure(s: SVG, th: Theme) -> None:
    """Section through the close-fitting enclosure of this page's own fiche.

    The fiche case is S_E = 24 m2 of exposed shell (a 3.0 x 2.0 x 1.8 m box,
    five faces), S_i = 30 m2 of interior surface and a mean interior
    absorption of 0.30, so R_i = 12.9 m2 and C = 3.4 dB. Every element the
    insertion loss actually depends on is drawn: the three wall layers, the
    isolated machine, the sealed door, the sleeved penetration, the lined
    cooling ducts, and the one unsealed gap that caps the result.
    """
    gy, yt = 470.0, 200.0
    x0, x1 = 230.0, 680.0  # 3.0 m of shell at 150 px/m
    t = 12.0  # drawn wall thickness

    s.ground(gy, 40.0, 700.0)
    s.text(
        455,
        92,
        "$S_E$ = 24 m² of exposed shell — a 3.0 × 2.0 × 1.8 m box, five faces",
        14,
        th.primary,
        bold=True,
    )

    # --- roof: the lined cooling outlet and the sleeved service entry ------
    s.rect(520, 128, 56, yt - 128, "none", th.fg, sw=2.2)
    s.line(530, 128, 530, yt, th.accent, 2.0, dash="5,4")
    s.line(566, 128, 566, yt, th.accent, 2.0, dash="5,4")
    s.arrow(548, yt - 10, 548, 136, th.primary, 2.0)
    s.text(592, 156, "lined cooling outlet", 12, th.fg, anchor="start")
    s.text(
        592, 176, "a short lined duct, never a bare hole", 10, th.muted, anchor="start"
    )
    s.line(310, yt, 310, 156, th.fg, 3.0)
    s.circle(310, yt - 6, 9, "none", th.accent, sw=2.0)
    s.text(310, 142, "cable and pipe entry, sealed sleeve", 11, th.fg)

    # --- the shell, drawn as its three layers ------------------------------
    s.rect(x0, yt, x1 - x0, gy - yt, "none", th.fg, sw=2.6)
    s.rect(
        x0 + t * 0.5,
        yt + t * 0.5,
        x1 - x0 - t,
        gy - yt - t * 0.5,
        "none",
        th.accent,
        sw=2.0,
        dash="6,4",
    )
    s.rect(
        x0 + t,
        yt + t,
        x1 - x0 - 2 * t,
        gy - yt - t,
        "none",
        th.muted,
        sw=1.4,
        dash="2,4",
    )
    s.text(455, 240, "$S_i$ = 30 m², $ᾱ_i = 0.30$  →  $R_i$ = 12.9 m²", 13, th.fg)
    s.text(
        455,
        262,
        "wall build-up: sheet-steel mass, absorbent lining, perforated facing",
        11,
        th.muted,
    )

    # --- the machine, isolated from the shell and from the slab ------------
    s.rect(330, 356, 260, 88, th.panel, th.fg, rx=6, sw=2.2)
    s.circle(460, 400, 17, th.primary)
    s.circle(460, 400, 6, th.bg)
    for mx in (352, 412, 508, 568):
        s.path(f"M {mx} 444 q 8 7 0 13 q -8 7 0 13", stroke=th.secondary, sw=2.0)
    s.text(460, 342, "machine on vibration isolators", 13, th.fg, bold=True)
    s.text(
        455, 496, "no rigid contact with the shell or with its slab", 12, th.secondary
    )

    # --- access door with its seal, and the gap at its foot ----------------
    s.rect(218, 262, 24, 190, th.panel, th.fg, sw=2.0)
    for dy in (268, 446):
        s.line(218, dy, 242, dy, th.accent, 2.6)
    s.rect(218, 452, 24, gy - 452, th.secondary, th.secondary, sw=1.0)
    s.text(206, 252, "access door", 13, th.fg, anchor="end")
    s.text(202, 272, "1.28 m², $R$ = 15 dB", 11, th.muted, anchor="end")
    s.text(206, 312, "compression seal", 11, th.accent, anchor="end")
    s.line(208, 308, 216, 274, th.muted, 1.0)
    s.text(206, 420, "unsealed gap at the foot", 11, th.secondary, anchor="end")
    s.text(206, 440, "0.24 m² = 1 % of $S_E$", 11, th.secondary, anchor="end")
    s.line(208, 446, 216, 460, th.muted, 1.0)

    # --- the lined cooling inlet -------------------------------------------
    s.rect(x1, 384, 96, 56, "none", th.fg, sw=2.2)
    s.line(x1, 394, x1 + 96, 394, th.accent, 2.0, dash="5,4")
    s.line(x1, 430, x1 + 96, 430, th.accent, 2.0, dash="5,4")
    s.arrow(x1 + 88, 412, x1 + 12, 412, th.primary, 2.0)
    s.text(700, 300, "lined cooling inlet", 12, th.fg, anchor="start")
    s.text(700, 320, "cooling air needs a path", 10, th.muted, anchor="start")

    # --- dimensions --------------------------------------------------------
    s.dim(x0, gy + 58, x1, gy + 58, "3.0 m", size=13)
    s.dim(858, yt, 858, gy, "1.8 m", size=13)

    # --- the equation and the two results ----------------------------------
    s.text(
        40,
        560,
        "$IL = R − C$,   $C = 10 log_{10}(0.3 + S_E/R_i)$ = 3.4 dB",
        15,
        th.fg,
        anchor="start",
    )
    s.text(
        40,
        586,
        "sealed shell (mean $R$ = 32.3 dB): mean IL = 28.9 dB",
        13,
        th.primary,
        anchor="start",
    )
    s.text(
        40,
        610,
        "with the door: 21.4 dB — with the 1 % gap as well: "
        "15.1 dB, against the $10 log_{10}(S_E/S_a)$ = 20 dB cap",
        13,
        th.secondary,
        anchor="start",
    )
    s.text(
        40,
        634,
        "an enclosure delivers its worst element, not its panels",
        13,
        th.muted,
        anchor="start",
    )


def _d_distortion_bench(s: SVG, th: Theme) -> None:
    """The bench every IEC 60268-3 distortion figure is defined on.

    Clause 3.1.2 fixes the chain (source e.m.f. through the rated source
    impedance, output terminals on the rated load impedance) and 3.1.3 the
    operating point (rated conditions with the drive dropped 10 dB). The
    lower band carries the three acceptance rules of 14.12.3.2 and 14.12.4.1,
    which decide whether a reading may be believed at all.
    """
    ay = 196.0  # signal axis

    def box(x: float, w: float, head: str, sub: str, col: str) -> None:
        s.rect(x, ay - 44, w, 88, th.panel, col, rx=10, sw=2.2)
        s.text(x + w / 2, ay - 8, head, 15, th.fg, bold=True)
        s.text(x + w / 2, ay + 18, sub, 13, th.muted, mono=True)

    # Generator -> rated source impedance -> amplifier under test.
    box(46, 186, "Signal generator", "1 kHz sine", th.primary)
    s.line(232, ay, 268, ay, th.fg, 2.0)
    s.rect(268, ay - 15, 66, 30, th.bg, th.fg, rx=3, sw=2.0)
    s.text(301, 154, "rated source", 12, th.muted)
    s.text(301, 170, "impedance", 12, th.muted)
    s.line(334, ay, 372, ay, th.fg, 2.0)
    box(372, 206, "Amplifier under test", "Class A / B / D", th.secondary)

    # Output terminals across the rated load, drawn to ground.
    s.line(578, ay, 640, ay, th.fg, 2.0)
    s.circle(640, ay, 5.0, th.bg, th.fg, 2.0)
    s.line(640, ay, 640, ay + 74, th.fg, 2.0)
    s.rect(618, ay + 74, 44, 62, th.bg, th.accent, rx=3, sw=2.2)
    s.line(640, ay + 136, 640, ay + 162, th.fg, 2.0)
    for k, hw in enumerate((22.0, 14.0, 7.0)):
        s.line(640 - hw, ay + 162 + k * 7, 640 + hw, ay + 162 + k * 7, th.fg, 2.0)
    s.text(
        676, ay + 96, "Rated load impedance", 13, th.accent, anchor="start", bold=True
    )
    s.text(676, ay + 116, "8 Ω non-inductive resistor,", 12, th.muted, anchor="start")
    s.text(676, ay + 134, "never a loudspeaker", 12, th.muted, anchor="start")

    # Class D branch: the IEC 61606-1 analogue low-pass before the analyser.
    s.line(640, ay, 700, ay, th.fg, 2.0)
    s.rect(700, ay - 34, 152, 68, th.bg, th.muted, rx=8, sw=1.8, dash="7,5")
    s.text(776, ay - 10, "Class D only:", 12, th.muted)
    s.text(776, ay + 10, "IEC 61606-1", 12, th.muted, mono=True)
    s.text(776, ay + 28, "analogue low-pass", 11, th.muted)

    # Analyser tap, above the load and after the optional Class D filter.
    s.line(776, ay - 34, 776, 116, th.fg, 2.0)
    s.rect(596, 46, 256, 70, th.panel, th.primary, rx=10, sw=2.2)
    s.text(724, 74, "Analyser / ADC", 15, th.fg, bold=True)
    s.text(724, 98, "≥ 10 dB headroom", 12, th.muted)

    # The operating point, called out on the generator side.
    s.rect(46, 46, 400, 62, th.panel, th.secondary, rx=10, sw=2.0)
    s.text(246, 70, "Standard measuring conditions (3.1.3)", 15, th.fg, bold=True)
    s.text(
        246, 94, "rated conditions (3.1.2) with the source e.m.f. −10 dB", 13, th.muted
    )
    s.line(246, 108, 246, ay - 44, th.muted, 1.4, dash="6,5")

    # Pre-conditioning and the reportable set.
    s.text(
        46,
        402,
        "Clause 9: hold the amplifier at that operating point for "
        "1 h before the first reading.",
        14,
        th.fg,
        anchor="start",
    )

    # The three acceptance rules the clauses make part of the method.
    s.rect(46, 424, 806, 132, th.panel, th.accent, rx=12, sw=2.0)
    s.text(
        66,
        452,
        "Three checks, from the method itself",
        15,
        th.fg,
        anchor="start",
        bold=True,
    )
    checks = (
        ("source THD ≥ 10 dB below the lowest distortion to be measured (14.12.3.2 a)"),
        (
            "generator muted: residual < 1/3 of the distortion voltage, or the "
            "result is discarded (14.12.3.2 d)"
        ),
        (
            "highest significant harmonic inside the band: $f_1 ≤ f_{limit} / n$ "
            "(14.12.4.1 — 30 kHz and $n = 5$ give 6 kHz)"
        ),
    )
    for k, txt in enumerate(checks):
        s.circle(76, 476 + k * 24, 4.0, th.accent)
        s.text(92, 481 + k * 24, txt, 13, th.muted, anchor="start")
    s.text(
        852, 452, "AES17 band 20 Hz – 20 kHz", 13, th.primary, anchor="end", mono=True
    )


# ---------------------------------------------------------------------------
# Swept-sine distortion: the two benches and the time budget
# ---------------------------------------------------------------------------


def _d_sweep_bench(s: SVG, th: Theme) -> None:
    """What plays and records a sweep, electrically and acoustically.

    Panel (a) is the electrical case, where a loopback channel fixes t = 0;
    panel (b) the acoustic one, where the reflection-free time bounds what
    may be called a free-field response. The strip below is the same time
    axis the harmonic pre-arrivals live on, drawn for the page's own sweep.
    """
    # ---- panel (a): electrical, with the loopback reference ---------------
    s.rect(40, 56, 386, 214, th.panel, th.muted, rx=12, sw=1.6)
    s.text(233, 84, "(a) Electrical device under test", 15, th.fg, bold=True)

    s.rect(64, 112, 118, 60, th.bg, th.primary, rx=8, sw=2.0)
    s.text(123, 138, "Audio", 14, th.fg, bold=True)
    s.text(123, 158, "interface", 14, th.fg, bold=True)
    s.rect(268, 112, 132, 60, th.bg, th.secondary, rx=8, sw=2.0)
    s.text(334, 138, "Device", 14, th.fg, bold=True)
    s.text(334, 158, "under test", 14, th.fg, bold=True)
    s.arrow(182, 128, 268, 128, th.fg, 1.8)
    s.text(225, 120, "out", 11, th.muted)
    s.arrow(400, 156, 400, 200, th.fg, 1.8)
    s.line(400, 200, 123, 200, th.fg, 1.8)
    s.arrow(160, 200, 123, 200, th.fg, 1.8)
    s.text(262, 194, "in — channel 1", 11, th.muted)
    s.line(182, 148, 216, 148, th.accent, 1.8, dash="7,4")
    s.line(216, 148, 216, 232, th.accent, 1.8, dash="7,4")
    s.line(216, 232, 123, 232, th.accent, 1.8, dash="7,4")
    s.arrow(160, 232, 123, 232, th.accent, 1.8)
    s.text(233, 252, "loopback — channel 2 fixes $t = 0$", 12, th.accent)

    # ---- panel (b): acoustic, with the first reflection --------------------
    s.rect(444, 56, 416, 214, th.panel, th.muted, rx=12, sw=1.6)
    s.text(652, 82, "(b) Loudspeaker in a room", 15, th.fg, bold=True)
    s.text(652, 104, "source and microphone at 1.20 m over a hard floor", 12, th.muted)
    gy = 250.0
    s.ground(gy, 474.0, 834.0, hatch=18)

    sx, mx, hy = 546.0, 762.0, 178.0
    s.line(sx, hy + 24, sx, gy, th.fg, 2.0)
    s.rect(sx - 24, hy - 28, 48, 52, th.bg, th.primary, rx=5, sw=2.0)
    s.circle(sx, hy, 10, th.primary)
    s.line(mx, hy + 6, mx, gy, th.fg, 2.0)
    s.rect(mx - 20, hy - 6, 32, 12, th.primary, rx=4)
    s.rect(mx + 12, hy - 4, 10, 8, th.fg, rx=2)

    s.line(sx + 24, hy, mx - 20, hy, th.fg, 2.0)
    s.dim(sx, hy, mx, hy, "$d$ = 1.00 m", offset=-38, size=13)
    fx = (sx + mx) / 2
    s.line(sx, hy, fx, gy, th.secondary, 1.8, dash="8,5")
    s.line(fx, gy, mx, hy, th.secondary, 1.8, dash="8,5")
    s.text(fx, gy - 10, "reflected path 2.60 m", 12, th.secondary)

    # ---- what the room costs, under both panels ---------------------------
    s.text(
        450,
        296,
        "Reflection-free time $t_g$ = (2.60 − 1.00) m / (343 "
        "m/s) = 4.7 ms: past that the record is the room, not "
        "the box.",
        14,
        th.fg,
    )

    # ---- the shared time budget of the harmonic arrivals -------------------
    x0, x1, ty = 90.0, 830.0, 412.0
    s.text(
        450,
        336,
        "Recording on the sweep's time axis: $f_1$ = 20 Hz, "
        "$f_2$ = 6 kHz, $T$ = 4 s, $L = T / ln(f_2/f_1)$ = "
        "0.70 s",
        15,
        th.fg,
        bold=True,
    )
    s.line(x0, ty, x1, ty, th.fg, 2.0)
    s.arrow(x1 - 30, ty, x1, ty, th.fg, 2.0)

    def tx(t: float) -> float:
        """Map an arrival time in seconds onto the strip (−1.0 s … +0.3 s)."""
        return x0 + (t + 1.0) / 1.3 * (x1 - 40 - x0)

    for t, lab, col, h in (
        (0.0, "$h_1$ (linear)", th.primary, 46.0),
        (-0.486, "$h_2$  −0.49 s", th.secondary, 34.0),
        (-0.770, "$h_3$  −0.77 s", th.accent, 26.0),
    ):
        s.line(tx(t), ty, tx(t), ty - h, col, 3.0)
        s.text(tx(t), ty - h - 9, lab, 13, col, bold=True)
    s.text(tx(0.0), ty + 22, "$t = 0$", 12, th.muted)

    # The per-order window against the closest spacing.
    s.rect(
        tx(-0.486),
        ty + 14,
        tx(-0.202) - tx(-0.486),
        26,
        th.panel,
        th.muted,
        rx=5,
        sw=1.4,
    )
    s.text(
        (tx(-0.486) + tx(-0.202)) / 2,
        ty + 62,
        "per-order window: 8192 samples = 0.17 s at 48 kHz",
        12,
        th.muted,
    )
    s.dim(
        tx(-0.770),
        ty + 104,
        tx(-0.486),
        ty + 104,
        "$L ln(3/2)$ = 0.28 s: the closest pair of arrivals",
        offset=0,
        size=13,
    )

    s.text(
        40,
        ty + 144,
        "Record until the decay has died, or the "
        "pre-arrivals wrap round the circular deconvolution "
        "into $h_1$.",
        14,
        th.fg,
        anchor="start",
    )
    s.text(
        40,
        ty + 168,
        "State the drive amplitude and the fade with the "
        "result, and pass the same fade to the analysis.",
        14,
        th.muted,
        anchor="start",
    )


# ---------------------------------------------------------------------------
# IEC 60268-5 polar measurement
# ---------------------------------------------------------------------------


def _d_loudspeaker_polar(s: SVG, th: Theme) -> None:
    """How a polar cut is taken (IEC 60268-5 23.1.2), in plan.

    The loudspeaker turns and the microphone does not: the reference point
    sits on the rotation axis so the measuring distance never changes, and
    the drive is retrimmed per band so that the on-axis pressure is held
    constant. The inset is the same cut on the IEC 60263 reference circle.
    """
    import math

    cx, cy, r = 292.0, 300.0, 176.0

    # Anechoic boundary.
    s.rect(40, 62, 512, 438, th.bg, th.fg, sw=2.4)
    for wx in range(44, 532, 40):
        s.path(
            f"M {wx} {62} L {wx + 40} {62} L {wx + 20} {88} Z",
            fill=th.panel,
            stroke=th.muted,
            sw=1.0,
        )
        s.path(
            f"M {wx} {500} L {wx + 40} {500} L {wx + 20} {474} Z",
            fill=th.panel,
            stroke=th.muted,
            sw=1.0,
        )
    for wy in range(90, 466, 40):
        s.path(
            f"M {40} {wy} L {40} {wy + 40} L {66} {wy + 20} Z",
            fill=th.panel,
            stroke=th.muted,
            sw=1.0,
        )
        s.path(
            f"M {552} {wy} L {552} {wy + 40} L {526} {wy + 20} Z",
            fill=th.panel,
            stroke=th.muted,
            sw=1.0,
        )
    s.text(58, 110, "Anechoic room, plan view", 13, th.muted, anchor="start")

    # The measuring arc the microphone is stepped along.
    s.path(
        f"M {cx + r:.1f} {cy:.1f} A {r} {r} 0 0 0 {cx - r:.1f} {cy:.1f}",
        stroke=th.accent,
        sw=1.8,
        dash="6,6",
    )
    for ang in (30, 60, 90, 120, 150):
        a = math.radians(ang)
        s.circle(cx + r * math.cos(a), cy - r * math.sin(a), 3.6, th.accent)
    s.text(cx - r + 4, cy - 14, "180°", 12, th.accent)
    s.text(cx, cy - r - 14, "90°", 12, th.accent)
    s.text(cx + 122, cy - 116, "$θ$ stepped by 10° or 15°", 12, th.accent)

    # Turntable, with the reference point over the rotation axis.
    s.ellipse(cx, cy, 70, 28, th.panel, th.muted, sw=1.8)
    s.circle(cx, cy, 4.0, th.fg)

    # Cabinet, seen from above, radiating along the reference axis.
    s.rect(cx - 32, cy - 24, 28, 48, th.panel, th.primary, rx=4, sw=2.0)
    s.line(cx - 4, cy - 24, cx - 4, cy + 24, th.primary, 2.4)

    # Reference axis at 0 degrees, out to the fixed microphone.
    s.line(cx, cy, cx + r + 40, cy, th.muted, 1.6, dash="8,5")
    s.text(cx + 96, cy - 14, "reference axis  0°", 13, th.muted)
    mx = cx + r
    s.rect(mx - 2, cy - 8, 38, 16, th.primary, rx=5)
    s.rect(mx - 16, cy - 5, 14, 10, th.fg, rx=3)
    s.text(mx + 16, cy + 32, "measuring microphone", 12, th.fg)

    # The two facts the geometry exists to guarantee.
    s.line(cx - 30, cy + 22, cx - 116, cy + 78, th.muted, 1.2, dash="4,4")
    s.text(
        60,
        cy + 116,
        "reference point on the rotation axis:",
        12,
        th.muted,
        anchor="start",
    )
    s.text(
        60, cy + 134, "$r$ never changes as $θ$ is swept", 12, th.muted, anchor="start"
    )
    s.dim(cx, cy, mx - 16, cy, "$r$ = 2 m", offset=150, size=14)

    # The drive rule that makes the cut a pattern rather than a response.
    s.rect(596, 62, 264, 128, th.panel, th.secondary, rx=10, sw=2.0)
    s.text(728, 90, "Drive condition (23.1.2.3)", 14, th.fg, bold=True)
    s.text(728, 114, "input voltage retrimmed at each", 12, th.muted)
    s.text(728, 132, "frequency or band so that $L_p$ on", 12, th.muted)
    s.text(728, 150, "the reference axis stays constant", 12, th.muted)
    s.text(728, 176, "500 Hz · 1 k · 2 k · 4 k · 8 k", 13, th.secondary)

    # Inset: the same cut on the IEC 60263 25 dB reference circle.
    ix, iy, ir = 728.0, 344.0, 108.0
    s.text(728, 216, "The cut, on the IEC 60263 circle", 14, th.fg, bold=True)
    for k, frac in enumerate((1.0, 0.8, 0.6, 0.4, 0.2)):
        s.circle(ix, iy, ir * frac, "none", th.muted, 1.0)
        if k:
            s.text(
                ix + 5, iy - ir * frac + 13, f"−{k * 5}", 10, th.muted, anchor="start"
            )
    s.line(ix - ir - 12, iy, ix + ir + 12, iy, th.muted, 1.0)
    s.line(ix, iy - ir - 12, ix, iy + ir + 12, th.muted, 1.0)
    pts = []
    for ang in range(-180, 181, 5):
        a = math.radians(ang)
        # A first-order cut, drawn on the clause-3 convention: the full
        # radius is 25 dB and the reference-axis level is the outer ring.
        mag = max(abs((1.0 + math.cos(a)) / 2.0), 1e-3)
        lev = max(-25.0, 20.0 * math.log10(mag))
        rad = ir * (25.0 + lev) / 25.0
        pts.append(f"{ix + rad * math.sin(a):.1f} {iy - rad * math.cos(a):.1f}")
    s.path("M " + " L ".join(pts) + " Z", stroke=th.primary, sw=2.2)
    s.text(728, iy + ir + 38, "outer ring = the reference-axis level,", 12, th.muted)
    s.text(728, iy + ir + 56, "full radius = 25 dB (clause 3)", 12, th.muted)

    s.text(
        450,
        542,
        "Directivity: free field on axis against a "
        "reverberation room, $D_i = L_{ax} − L_p + 10 lg(T/T_0) − "
        "10 lg(V/V_0)$ + 25 dB",
        14,
        th.fg,
    )
    s.text(
        450,
        566,
        "(23.3.2.1), or by integrating these polar curves over the sphere (23.3.2.2).",
        14,
        th.muted,
    )


# ---------------------------------------------------------------------------
# IEC 60268-4: the three fields a microphone sensitivity can be defined in
# ---------------------------------------------------------------------------


def _d_microphone_references(s: SVG, th: Theme) -> None:
    """Free field, diffuse field and pressure, on one capsule.

    The same capsule in three sound fields defines three different
    sensitivities (11.2.1, 11.2.2, 11.2.4). They agree while the capsule is
    small against the wavelength and separate once it is not, which is the
    whole reason the *type* has to be quoted with the number.
    """
    import math

    tops, w = (34.0, 320.0, 606.0), 260.0
    heads = ("Free field (11.2.1)", "Diffuse field (11.2.2)", "Pressure (11.2.4)")
    for x0, head in zip(tops, heads, strict=True):
        s.rect(x0, 54, w, 306, th.panel, th.muted, rx=12, sw=1.6)
        s.text(x0 + w / 2, 80, head, 15, th.fg, bold=True)

    def capsule(cx: float, cy: float) -> None:
        """One capsule glyph, diaphragm facing left along its reference axis."""
        s.rect(cx - 4, cy - 15, 54, 30, th.panel, th.primary, rx=6, sw=2.0)
        s.rect(cx - 18, cy - 12, 14, 24, th.fg, rx=3)
        s.line(cx - 18, cy - 12, cx - 18, cy + 12, th.secondary, 2.6)

    # --- 1: free field, plane wave on the reference axis --------------------
    x0 = tops[0]
    cx, cy = x0 + 168.0, 176.0
    for k in range(4):
        s.path(
            f"M {x0 + 40 + k * 22} {cy - 52} q 10 52 0 104", stroke=th.primary, sw=1.6
        )
    s.arrow(x0 + 118, cy, x0 + 140, cy, th.primary, 1.8)
    capsule(cx, cy)
    s.line(cx - 18, cy, x0 + 246, cy, th.muted, 1.2, dash="7,5")
    s.text(x0 + w / 2, cy + 78, "one source on the reference axis,", 12, th.muted)
    s.text(x0 + w / 2, cy + 96, "far enough that $r ≥ d$,", 12, th.muted)
    s.text(x0 + w / 2, cy + 114, "$r ≥ d^2/λ$ and $r$ ≥ 3 × the source", 12, th.muted)
    s.text(
        x0 + w / 2, cy + 144, "$M_{ff}$ : the undisturbed", 14, th.primary, bold=True
    )
    s.text(x0 + w / 2, cy + 164, "pressure of the plane wave", 13, th.muted)

    # --- 2: diffuse field, rays from every direction ------------------------
    x1 = tops[1]
    cx, cy = x1 + 130.0, 176.0
    for ang in range(0, 360, 30):
        a = math.radians(ang)
        s.arrow(
            cx + 96 * math.cos(a),
            cy + 96 * math.sin(a),
            cx + 40 * math.cos(a),
            cy + 40 * math.sin(a),
            th.accent,
            1.4,
        )
    capsule(cx - 22, cy)
    s.text(x1 + w / 2, cy + 78, "sound from every direction,", 12, th.muted)
    s.text(x1 + w / 2, cy + 96, "with equal probability", 12, th.muted)
    s.text(x1 + w / 2, cy + 114, "(a reverberation room)", 12, th.muted)
    s.text(
        x1 + w / 2,
        cy + 144,
        "$M_{diff}$ : the r.m.s. of $M(θ)$",
        14,
        th.accent,
        bold=True,
    )
    s.text(x1 + w / 2, cy + 164, "$D = 20 lg(M_0 / M_{diff})$", 13, th.muted)

    # --- 3: pressure, the capsule closed into a coupler ---------------------
    x2 = tops[2]
    cx, cy = x2 + 150.0, 176.0
    s.rect(cx - 118, cy - 48, 122, 96, th.bg, th.secondary, rx=8, sw=2.2)
    s.circle(cx - 92, cy, 16, th.panel, th.muted, 1.6)
    s.arrow(cx - 92, cy + 30, cx - 92, cy - 30, th.secondary, 1.6)
    capsule(cx, cy)
    s.text(cx - 56, cy - 62, "cavity small against $λ$", 12, th.muted)
    s.text(x2 + w / 2, cy + 78, "a coupler or a calibrator:", 12, th.muted)
    s.text(x2 + w / 2, cy + 96, "the pressure the capsule", 12, th.muted)
    s.text(x2 + w / 2, cy + 114, "itself replaces", 12, th.muted)
    s.text(x2 + w / 2, cy + 144, "$M_p$ : pressure at", 14, th.secondary, bold=True)
    s.text(x2 + w / 2, cy + 164, "the acoustic entry", 13, th.muted)

    # --- the bench the first two are realised on ----------------------------
    s.rect(34, 380, 832, 116, th.panel, th.primary, rx=12, sw=2.0)
    s.text(
        54,
        408,
        "The bench (clauses 5.5.2, 5.6.2, 5.7)",
        15,
        th.fg,
        anchor="start",
        bold=True,
    )
    rules = (
        (
            "anechoic room; the spherical wave counts as plane at least $λ/2$ "
            "from the centre of curvature at the lowest frequency"
        ),
        (
            "substitution: the microphone under test and a calibrated reference "
            "at the same point, in turn (highest accuracy)"
        ),
        (
            "simultaneous comparison at two nearby points only after showing it "
            "agrees with substitution within ± 1 dB"
        ),
    )
    for k, txt in enumerate(rules):
        s.circle(64, 432 + k * 22, 4.0, th.primary)
        s.text(80, 437 + k * 22, txt, 12, th.muted, anchor="start")
    s.text(
        846,
        408,
        "overall accuracy ± 2 dB or better",
        13,
        th.primary,
        anchor="end",
        mono=True,
    )

    s.text(
        450,
        528,
        "Polar cuts (13.1.2 a): distance, sound pressure and "
        "frequency held constant while $θ$ is stepped, by 10° or "
        "15°,",
        14,
        th.fg,
    )
    s.text(
        450,
        552,
        "at the octave centres 125 Hz to 16 kHz, with the "
        "reference axis as 0° of the polar diagram.",
        14,
        th.muted,
    )
