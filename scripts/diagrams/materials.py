#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Diagrams of the materials guides: absorbers, diffusers, surfaces and resilient layers.

One subject because every one of these is a specimen on a test rig: an
impedance tube, an airflow resistance bench, a reverberation room or
goniometer for scattering, an in-situ rig on a road surface, or a load plate
on a resilient layer. What is being characterised is the material, not the
building it will end up in.
"""

from __future__ import annotations

from .canvas import SVG, Theme
from .parts import _accel, _exciter, _motion_arrows, _rot_arrow, _spring_v


def _d_impedance_tube(s: SVG, th: Theme) -> None:
    """ISO 10534-2 two-microphone impedance tube (side view)."""
    tube_top, tube_bot, mid = 215.0, 335.0, 275.0
    tube_l, tube_r = 165.0, 778.0
    back_w, spec_w = 20.0, 48.0
    spec_l = tube_r - back_w - spec_w

    # Tube body.
    s.rect(tube_l, tube_top, tube_r - tube_l, tube_bot - tube_top, th.bg, th.fg, sw=3)

    # Loudspeaker sealed to the left end, cone opening into the tube.
    s.rect(72, mid - 46, 70, 92, th.panel, th.primary, rx=6, sw=2)
    s.path(f"M 142 {mid - 18} L 142 {mid + 18} L {tube_l} {tube_bot} "
           f"L {tube_l} {tube_top} Z", fill=th.panel, stroke=th.primary, sw=2)
    s.circle(120, mid, 11, th.primary)
    s.text(118, tube_bot + 42, "Loudspeaker", 20, th.fg, bold=True)

    # Test specimen and rigid backing at the right end.
    s.rect(tube_r - back_w, tube_top, back_w, tube_bot - tube_top, th.fg)
    s.rect(spec_l, tube_top, spec_w, tube_bot - tube_top, th.panel, th.secondary, sw=2)
    for hx in range(int(spec_l) + 8, int(spec_l + spec_w), 11):
        s.line(hx, tube_bot - 4, hx - 16, tube_top + 4, th.secondary, 1.0)
    s.text(spec_l + spec_w / 2, tube_top - 14, "Test specimen", 19, th.secondary, bold=True)
    s.text(tube_r - back_w / 2, tube_bot + 42, "Rigid backing", 18, th.muted)

    # Two microphones flush in the top wall (mic 1 = farther from specimen).
    m1x, m2x = 460.0, 555.0
    for mx, lab in ((m1x, "Mic 1"), (m2x, "Mic 2")):
        s.rect(mx - 7, tube_top - 20, 14, 20, th.fg, rx=3)
        s.circle(mx, tube_top, 5, th.primary)
        s.text(mx, tube_top - 28, lab, 18, th.fg, bold=True)

    # Plane-wave arrows inside the tube.
    s.arrow(tube_l + 30, mid - 18, spec_l - 16, mid - 18, th.accent, 2.2)
    s.text((tube_l + spec_l) / 2 - 40, mid - 26, "incident", 17, th.accent)
    s.arrow(spec_l - 16, mid + 20, tube_l + 30, mid + 20, th.secondary, 2.2)
    s.text((tube_l + spec_l) / 2 - 40, mid + 38, "reflected", 17, th.secondary)

    # Dimensions: x1 (specimen face -> far mic) above, spacing s below.
    s.dim(spec_l, tube_top, m1x, tube_top, "x₁", offset=-58, size=19)
    s.dim(m1x, tube_bot, m2x, tube_bot, "s", offset=70, size=19)

    # Governing relations and range.
    for y, txt, col in (
        (438, ("H₁₂ → reflection factor r (Eq. 17), "
              "absorption α = 1 − |r|² (Eq. 18), "
              "Z/ρc₀ = (1+r)/(1−r) (Eq. 19)"), th.fg),
        (466, ("Working range f_l < f < f_u set by the microphone spacing s "
              "and the tube diameter (Clause 6.1)"), th.muted),
        (492, ("ASTM E2611: two further microphones behind the specimen also "
              "give the transmission loss"), th.muted),
    ):
        s.text(450, y, txt, 18, col)


def _d_astm_tube(s: SVG, th: Theme) -> None:
    """ASTM E2611 four-microphone transmission-loss tube (side view)."""
    tube_top, tube_bot, mid = 225.0, 345.0, 285.0
    tube_l, tube_r = 140.0, 825.0
    spec_l, spec_r = 453.0, 497.0
    m1x, m2x, m3x, m4x = 250.0, 360.0, 590.0, 700.0

    # Tube body.
    s.rect(tube_l, tube_top, tube_r - tube_l, tube_bot - tube_top, th.bg, th.fg, sw=3)

    # Loudspeaker sealed to the left end.
    s.rect(56, mid - 42, 62, 84, th.panel, th.primary, rx=6, sw=2)
    s.path(f"M 118 {mid - 16} L 118 {mid + 16} L {tube_l} {tube_bot} "
           f"L {tube_l} {tube_top} Z", fill=th.panel, stroke=th.primary, sw=2)
    s.circle(96, mid, 10, th.primary)
    s.text(96, tube_bot + 40, "Source", 19, th.fg, bold=True)

    # Adjustable termination (two loads) at the right end.
    s.rect(tube_r - 20, tube_top, 20, tube_bot - tube_top, th.fg)
    s.text(tube_r - 10, tube_bot + 40, "Termination", 17, th.muted)
    s.text(tube_r - 10, tube_bot + 60, "(2 loads)", 17, th.muted)

    # Test specimen at the centre.
    s.rect(spec_l, tube_top, spec_r - spec_l, tube_bot - tube_top, th.panel,
           th.secondary, sw=2)
    for hx in range(int(spec_l) + 7, int(spec_r), 10):
        s.line(hx, tube_bot - 4, hx - 14, tube_top + 4, th.secondary, 1.0)
    s.text((spec_l + spec_r) / 2, tube_bot + 40, "Test specimen", 18,
           th.secondary, bold=True)

    # Four microphones flush in the top wall (1,2 upstream; 3,4 downstream).
    for mx, lab in ((m1x, "Mic 1"), (m2x, "Mic 2"), (m3x, "Mic 3"), (m4x, "Mic 4")):
        s.rect(mx - 6, tube_top - 18, 12, 18, th.fg, rx=3)
        s.circle(mx, tube_top, 5, th.primary)
        s.text(mx, tube_top - 26, lab, 16, th.fg, bold=True)

    # Up- and downstream travelling waves.
    s.arrow(tube_l + 26, mid - 16, spec_l - 8, mid - 16, th.accent, 2.0)
    s.arrow(spec_l - 8, mid + 18, tube_l + 26, mid + 18, th.secondary, 2.0)
    s.arrow(spec_r + 8, mid - 16, tube_r - 26, mid - 16, th.accent, 2.0)
    s.arrow(tube_r - 26, mid + 18, spec_r + 8, mid + 18, th.secondary, 2.0)
    s.text(tube_l + 40, mid - 22, "A", 17, th.accent, bold=True)
    s.text(tube_l + 40, mid + 34, "B", 17, th.secondary, bold=True)
    s.text(tube_r - 40, mid - 22, "C", 17, th.accent, bold=True)
    s.text(tube_r - 40, mid + 34, "D", 17, th.secondary, bold=True)

    # Dimensions: spacings s1/s2 below; specimen offsets l1/l2 and thickness d above.
    s.dim(m1x, tube_bot, m2x, tube_bot, "s₁", offset=62, size=18)
    s.dim(m3x, tube_bot, m4x, tube_bot, "s₂", offset=62, size=18)
    # l1, l2 are both measured from the specimen FRONT face (x = 0), matching
    # wave_decomposition/transfer_matrix_two_load; l2 therefore spans the specimen.
    s.dim(m2x, tube_top, spec_l, tube_top, "l₁", offset=-42, size=18)
    s.dim(spec_l, tube_top, m3x, tube_top, "l₂", offset=-58, size=18)
    s.dim(spec_l, tube_top - 78, spec_r, tube_top - 78, "d", offset=0, size=17)
    s.line(spec_l, tube_top, spec_l, tube_top - 78, th.muted, 0.9, dash="3,3")
    s.line(spec_r, tube_top, spec_r, tube_top - 78, th.muted, 0.9, dash="3,3")

    # Governing relations.
    for y, txt, col in (
        (452, ("Decompose A, B (upstream) and C, D (downstream) → "
              "transfer matrix T (Eq. 22)"), th.fg),
        (480, "TL = 20 log₁₀ |(T₁₁ + T₁₂/ρc + ρc·T₂₁ + T₂₂) / 2|   (Eq. 26)",
         th.muted),
        (506, ("Two-load method: repeat with two terminations; the one-load "
              "variant uses a single anechoic end"), th.muted),
    ):
        s.text(450, y, txt, 17, col)


def _d_airflow(s: SVG, th: Theme) -> None:
    """ISO 9053-1 static and ISO 9053-2 alternating airflow-resistance rigs."""
    # --- Left panel: static (DC) method -----------------------------------
    s.rect(55, 70, 385, 430, th.panel, th.fg, rx=8, sw=2)
    s.text(247, 100, "Static method (ISO 9053-1)", 21, th.fg, bold=True)

    cx = 200.0
    holder_l, holder_r = cx - 45, cx + 45
    top_y, bot_y = 170.0, 430.0
    # Vertical specimen holder (tube).
    s.line(holder_l, top_y, holder_l, bot_y, th.fg, 2.5)
    s.line(holder_r, top_y, holder_r, bot_y, th.fg, 2.5)
    # Specimen (hatched disc) in the middle.
    spec_y, spec_h = 285.0, 46.0
    s.rect(holder_l, spec_y, 90, spec_h, th.bg, th.secondary, sw=2)
    for hy in range(int(spec_y) + 8, int(spec_y + spec_h), 10):
        s.line(holder_l + 4, hy, holder_r - 4, hy - 8, th.secondary, 1.0)
    s.text(cx, spec_y + spec_h + 22, "specimen (A, d)", 17, th.secondary, bold=True)
    # Steady laminar flow up through the holder.
    s.arrow(cx, bot_y - 6, cx, spec_y + spec_h + 34, th.accent, 2.4)
    s.arrow(cx, spec_y - 12, cx, top_y + 8, th.accent, 2.4)
    s.text(cx, bot_y + 22, "laminar flow  q_v", 18, th.accent, bold=True)
    # Differential manometer across the specimen (pressure taps).
    tap_x = holder_r + 8
    s.line(holder_r, spec_y - 4, tap_x + 40, spec_y - 4, th.primary, 1.6)
    s.line(holder_r, spec_y + spec_h + 4, tap_x + 40, spec_y + spec_h + 4, th.primary, 1.6)
    s.rect(tap_x + 40, spec_y - 26, 74, spec_h + 44, th.bg, th.primary, rx=8, sw=2)
    s.text(tap_x + 77, spec_y + 8, "Δp", 22, th.primary, bold=True, mono=True)
    s.text(tap_x + 77, spec_y + 34, "manom.", 15, th.muted)
    s.text(247, 478, "R = Δp / q_v   (through-origin fit at 0.5 mm/s)",
           16, th.fg, bold=True)

    # --- Right panel: alternating (AC) method -----------------------------
    s.rect(460, 70, 385, 430, th.panel, th.fg, rx=8, sw=2)
    s.text(652, 100, "Alternating method (ISO 9053-2)", 21, th.fg, bold=True)

    cav_l, cav_r = 590.0, 715.0
    cav_top, cav_bot = 160.0, 360.0
    # Cavity walls.
    s.rect(cav_l, cav_top, cav_r - cav_l, cav_bot - cav_top, th.bg, th.fg, sw=2.5)
    s.text((cav_l + cav_r) / 2, (cav_top + cav_bot) / 2 - 6, "cavity", 18, th.fg)
    s.text((cav_l + cav_r) / 2, (cav_top + cav_bot) / 2 + 18, "V", 20, th.fg,
           bold=True, italic=True)
    # Specimen / airtight termination on top.
    s.rect(cav_l, cav_top - 26, cav_r - cav_l, 26, th.bg, th.secondary, sw=2)
    for hx in range(int(cav_l) + 8, int(cav_r), 11):
        s.line(hx, cav_top - 4, hx - 14, cav_top - 22, th.secondary, 1.0)
    s.text((cav_l + cav_r) / 2, cav_top - 36, "specimen / airtight", 16,
           th.secondary, bold=True)
    # Piston at the bottom, oscillating.
    s.rect(cav_l, cav_bot, cav_r - cav_l, 26, th.panel, th.primary, sw=2)
    s.arrow((cav_l + cav_r) / 2, cav_bot + 58, (cav_l + cav_r) / 2, cav_bot + 30,
            th.primary, 2.2)
    s.arrow((cav_l + cav_r) / 2, cav_bot + 30, (cav_l + cav_r) / 2, cav_bot + 58,
            th.primary, 2.2)
    s.text((cav_l + cav_r) / 2, cav_bot + 80, "piston  f = 1–4 Hz", 18,
           th.primary, bold=True)
    # Microphone in the cavity wall.
    s.circle(cav_r + 2, (cav_top + cav_bot) / 2, 6, th.fg)
    s.line(cav_r + 2, (cav_top + cav_bot) / 2, cav_r + 60,
           (cav_top + cav_bot) / 2, th.muted, 1.4)
    s.text(cav_r + 66, (cav_top + cav_bot) / 2 + 6, "L_p", 20, th.fg,
           bold=True, mono=True, anchor="start")
    s.text(652, 478, "R from L_p,s − L_p,t   (κ′ per Annex A)",
           16, th.fg, bold=True)


# ---------------------------------------------------------------------------
# d15 - ISO 17497-1 random-incidence scattering (reverberation room)
# ---------------------------------------------------------------------------

def _d_scattering_reverb(s: SVG, th: Theme) -> None:
    """ISO 17497-1 scattering coefficient in a reverberation room."""
    gy = 400.0
    # Reverberation room with non-parallel walls (skew quadrilateral).
    s.path("M 60 80 L 782 66 L 796 400 L 72 400 Z", fill=th.panel,
           stroke=th.fg, sw=3)
    s.text(80, 106, "Reverberation room", 20, th.fg, bold=True, anchor="start")

    # --- Turntable carrying the test sample (left, in perspective) --------
    tx, tyc = 285.0, 366.0
    s.ellipse(tx, tyc, 150, 26, th.panel, th.primary, 2.2)      # turntable
    s.ellipse(tx, tyc - 12, 82, 15, th.bg, th.secondary, 2.2)   # test sample
    for hx in range(int(tx) - 60, int(tx) + 60, 12):            # sample hatch
        s.line(hx, tyc - 10, hx + 10, tyc - 18, th.secondary, 1.0)
    s.text(tx, gy + 22, "Turntable and base plate", 17, th.fg, bold=True)
    _rot_arrow(s, tx, tyc, 150, 205, 340, th.accent, 2.2, ry=26)
    s.text(tx, tyc - 70, "the only thing that moves", 15, th.accent)
    s.text(tx, tyc - 46, "sample on the plate for T2 and T4", 15, th.muted)
    # Wall clearance of the turntable rim.
    s.line(78, 386, tx - 150, 386, th.muted, 1.4)
    s.line(78, 380, 78, 392, th.muted, 1.4)
    s.line(tx - 150, 380, tx - 150, 392, th.muted, 1.4)
    s.text(106, 376, "≥ 1.0 m", 14, th.muted)

    # --- Two fixed loudspeaker positions (right) --------------------------
    for sx, sy, lab in ((648.0, 262.0, "S1"), (752.0, 296.0, "S2")):
        s.rect(sx - 20, sy - 26, 40, 52, th.panel, th.primary, rx=6, sw=2)
        s.circle(sx, sy, 11, th.primary)
        s.circle(sx, sy, 4, th.bg)
        s.line(sx, sy + 26, sx, gy, th.fg, 2.2)
        s.line(sx - 14, gy, sx + 14, gy, th.fg, 2.2)
        s.text(sx, sy - 34, lab, 17, th.fg, bold=True)
    s.text(700, 196, "fixed sources (≥ 2)", 16, th.muted)

    # --- Three fixed microphone positions ---------------------------------
    for mx, my, lab in ((452.0, 262.0, "M1"), (520.0, 286.0, "M2"),
                        (586.0, 310.0, "M3")):
        s.mic(mx, my, gy, 1.0)
        s.text(mx, my - 12, lab, 16, th.fg, bold=True)
    s.text(470, 226, "fixed microphones (≥ 3)", 16, th.muted)

    # --- Governing relations ----------------------------------------------
    for y, txt, col, bold in (
        (448, ("T1 base plate, static  ·  T2 sample, static  →  α_s (Eq. 1)"),
         th.fg, True),
        (474, ("T3 base plate, rotating  ·  T4 sample, rotating  →  "
               "α_spec (Eq. 4)"), th.fg, True),
        (502, "s = (α_spec − α_s) / (1 − α_s)   (Eq. 5)", th.accent, True),
        (528, ("α from 55.3·(V/S)·(1/cT) − 4(V/S)m  ·  the base plate must "
               "pass the Table 1 ceiling"), th.muted, False),
    ):
        s.text(450, y, txt, 19 if bold else 17, col, bold=bold)


# ---------------------------------------------------------------------------
# d16 - ISO 17497-2 free-field diffusion goniometer
# ---------------------------------------------------------------------------

def _d_diffusion_goniometer(s: SVG, th: Theme) -> None:
    """ISO 17497-2 directional diffusion coefficient (goniometer)."""
    import math
    gy, cx, R = 430.0, 450.0, 300.0
    s.ground(gy, 90, 810)

    # Semicircular receiver arc (0 deg right .. 180 deg left, zenith at top).
    s.path(f"M {cx - R} {gy} A {R} {R} 0 0 1 {cx + R} {gy}",
           stroke=th.muted, sw=1.8)
    ends = {0, 90, 180}
    for ang in range(0, 181, 15):
        a = math.radians(ang)
        px, py = cx + R * math.cos(a), gy - R * math.sin(a)
        s.circle(px, py, 6.5, th.primary)
        s.circle(px, py, 2.2, th.bg)
    # Label the two horizon receivers and the zenith one.
    s.text(cx + R + 4, gy - 4, "L_n", 17, th.fg, anchor="start")
    s.text(cx - R - 4, gy - 4, "L_1", 17, th.fg, anchor="end")
    s.text(cx, gy - R - 14, "L_i", 17, th.fg)
    s.text(cx + 150, gy - 250, "receiver arc (5° steps)", 16, th.muted)
    _ = ends

    # Polar (scattered) response lobe about the sample centre.
    pts = []
    for ang in range(0, 181, 6):
        a = math.radians(ang)
        rr = 92.0 + 42.0 * abs(math.sin(3.0 * a))
        pts.append((cx + rr * math.cos(a), gy - rr * math.sin(a)))
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    s.path(d, stroke=th.accent, sw=2.0)
    s.text(cx + 96, gy - 150, "polar response L_i", 16, th.accent)

    # Fixed source, off to the upper left, illuminating the sample.
    sa = math.radians(155.0)
    sxx, syy = cx + (R + 44) * math.cos(sa), gy - (R + 44) * math.sin(sa)
    s.rect(sxx - 26, syy - 22, 52, 44, th.panel, th.primary, rx=6, sw=2)
    s.circle(sxx + 20, syy, 10, th.primary)
    s.circle(sxx + 20, syy, 4, th.bg)
    s.text(sxx, syy - 32, "Fixed source", 17, th.fg, bold=True)
    s.arrow(sxx + 26, syy + 6, cx - 74, gy - 12, th.accent, 2.0)

    # Test sample on the turntable at the arc centre.
    s.rect(cx - 72, gy - 13, 144, 13, th.bg, th.secondary, sw=2)
    for hx in range(int(cx) - 64, int(cx) + 64, 12):
        s.line(hx, gy - 3, hx + 9, gy - 11, th.secondary, 1.0)
    s.text(cx, gy - 20, "Test sample", 16, th.secondary, bold=True)
    s.ellipse(cx, gy + 8, 88, 12, "none", th.primary, 1.8)
    _rot_arrow(s, cx, gy + 8, 88, 200, 340, th.primary, 1.8, ry=12)
    s.text(cx + 150, gy + 12, "Turntable", 16, th.fg, bold=True, anchor="start")

    # Governing relations.
    s.text(450, 476,
           "d = [(Σ10^(L_i/10))² − Σ(10^(L_i/10))²] / "
           "[(n−1)·Σ(10^(L_i/10))²]   (Formula 5)", 17, th.fg, bold=True)
    s.text(450, 506, "d_n = (d − d_ref) / (1 − d_ref)   (Formula 7)", 18,
           th.accent, bold=True)
    s.text(450, 534,
           "5° receiver steps · turntable rotates the sample · source fixed",
           17, th.muted)


# ---------------------------------------------------------------------------
# d17 - ISO 13472-1 in-situ road absorption, subtraction technique
# ---------------------------------------------------------------------------

def _d_insitu_subtraction(s: SVG, th: Theme) -> None:
    """ISO 13472-1 extended-surface (subtraction) in-situ absorption."""
    gy = 415.0
    # Road surface (the reference plane) under the main measurement.
    s.ground(gy, 55, 590)
    s.text(66, gy + 30, "Road surface", 16, th.muted, anchor="start")

    sx = 250.0
    src_y, mic_y = gy - 235.0, gy - 47.0        # ds : dm = 1.25 : 0.25 m
    s.line(sx, src_y, sx, gy, th.muted, 1.0, dash="4,4")   # normal axis

    # Loudspeaker (source) at ds above the surface.
    s.rect(sx - 30, src_y - 30, 60, 60, th.panel, th.primary, rx=6, sw=2)
    s.circle(sx, src_y, 12, th.primary)
    s.circle(sx, src_y, 5, th.bg)
    s.text(sx, src_y - 42, "Loudspeaker", 18, th.fg, bold=True)

    # Microphone at dm above the surface.
    s.rect(sx - 6, mic_y - 9, 12, 18, th.fg, rx=3)
    s.circle(sx, mic_y - 9, 5, th.primary)
    s.text(sx + 16, mic_y + 5, "Microphone", 15, th.fg, anchor="start")

    # Direct ray (source -> mic), drawn offset to the left of the axis.
    s.arrow(sx - 7, src_y + 22, sx - 7, mic_y - 12, th.accent, 2.0)
    s.text(sx - 60, (src_y + mic_y) / 2, "direct  ds−dm", 15, th.accent,
           anchor="end")
    # Road-reflected ray: source -> surface point -> mic (shallow V, offset).
    gpx = sx + 74.0
    s.line(sx + 8, src_y + 24, gpx, gy, th.secondary, 2.0)
    s.arrow(gpx, gy, sx + 8, mic_y + 6, th.secondary, 2.0)
    s.text(gpx + 8, gy - 96, "reflected  ds+dm", 15, th.secondary,
           anchor="start")
    # Dashed continuation toward the image source below the plane.
    s.line(gpx, gy, sx + 34, gy + 66, th.muted, 1.2, dash="5,4")
    s.text(sx + 40, gy + 60, "to image source (ds below)", 14, th.muted,
           anchor="start")

    # Height dimensions ds and dm.
    s.dim(sx - 72, gy, sx - 72, src_y, "ds = 1.25 m", offset=0,
          label_side="left", size=17)
    s.line(sx - 72, gy, sx, gy, th.muted, 0.9, dash="3,3")
    s.line(sx - 72, src_y, sx - 30, src_y, th.muted, 0.9, dash="3,3")
    s.dim(sx + 122, gy, sx + 122, mic_y, "dm = 0.25 m", offset=0,
          label_side="right", size=17)
    s.line(sx, mic_y, sx + 122, mic_y, th.muted, 0.9, dash="3,3")

    # --- Free-field reference (right): source + mic high, no ground -------
    s.line(615, 90, 615, gy + 40, th.muted, 1.2, dash="6,5")
    fx = 730.0
    fs_y, fm_y = 150.0, 292.0
    s.rect(fx - 28, fs_y - 26, 56, 52, th.panel, th.primary, rx=6, sw=2)
    s.circle(fx, fs_y, 11, th.primary)
    s.circle(fx, fs_y, 4, th.bg)
    s.rect(fx - 6, fm_y - 9, 12, 18, th.fg, rx=3)
    s.circle(fx, fm_y - 9, 5, th.primary)
    s.arrow(fx, fs_y + 28, fx, fm_y - 14, th.accent, 2.0)
    s.text(fx, fs_y - 40, "Free-field reference", 17, th.fg, bold=True)
    s.text(fx, fm_y + 34, "Hi: no ground reflection in the window", 14,
           th.muted)

    # Governing relations.
    s.text(450, 502, "Kr = (ds − dm)/(ds + dm) = 2/3   (Clause 4.1)", 18,
           th.fg, bold=True)
    s.text(450, 528, "α(f) = 1 − (1/Kr²)·|Hr/Hi|²   ·   Δτ = 2 dm / c", 18,
           th.accent, bold=True)
    s.text(450, 552, "Adrienne time window isolates the reflected response Hr",
           16, th.muted)


# ---------------------------------------------------------------------------
# d18 - ISO 13472-2 in-situ road absorption, spot method
# ---------------------------------------------------------------------------

def _d_spot_tube(s: SVG, th: Theme) -> None:
    """ISO 13472-2 spot method: short tube sealed onto the road surface."""
    gy = 430.0
    cx, hw, y_top = 235.0, 72.0, 120.0

    # Road surface (the test sample) with the tube sealed onto it.
    s.ground(gy, 60, 430)
    s.text(72, gy + 30, "Road surface (test sample)", 15, th.muted,
           anchor="start")

    # Tube walls.
    s.line(cx - hw, y_top, cx - hw, gy, th.fg, 3)
    s.line(cx + hw, y_top, cx + hw, gy, th.fg, 3)
    # Sealing rings where the tube meets the road.
    s.rect(cx - hw - 7, gy - 9, 14, 18, th.muted, rx=2)
    s.rect(cx + hw - 7, gy - 9, 14, 18, th.muted, rx=2)

    # Loudspeaker cap at the top.
    s.rect(cx - hw, y_top - 40, 2 * hw, 40, th.panel, th.primary, sw=2)
    s.circle(cx, y_top - 20, 12, th.primary)
    s.circle(cx, y_top - 20, 5, th.bg)
    s.text(cx, y_top - 52, "Loudspeaker", 18, th.fg, bold=True)

    # Two microphones flush in the right wall, spacing s.
    m1y, m2y = gy - 158.0, gy - 82.0
    for my, lab in ((m1y, "Mic 1"), (m2y, "Mic 2")):
        s.rect(cx + hw - 4, my - 7, 12, 14, th.fg, rx=3)
        s.circle(cx + hw, my, 4, th.primary)
        s.text(cx + hw + 16, my + 5, lab, 15, th.fg, anchor="start")

    # Plane-wave travel down and reflection back up.
    s.arrow(cx - 34, y_top + 16, cx - 34, gy - 26, th.accent, 2.0)
    s.arrow(cx - 8, gy - 26, cx - 8, y_top + 16, th.secondary, 2.0)

    # Dimensions: tube diameter d (across) and mic spacing s (down).
    s.dim(cx - hw, y_top + 18, cx + hw, y_top + 18, "d", offset=0, size=18)
    s.dim(cx + hw + 62, m1y, cx + hw + 62, m2y, "s", offset=0,
          label_side="right", size=18)
    s.line(cx + hw + 10, m1y, cx + hw + 62, m1y, th.muted, 0.9, dash="3,3")
    s.line(cx + hw + 10, m2y, cx + hw + 62, m2y, th.muted, 0.9, dash="3,3")

    # Right panel: usable frequency range and DSP method.
    s.rect(430, 118, 430, 300, "none", th.muted, rx=12, dash="6,5")
    s.text(645, 152, "Spot method (ISO 13472-2)", 20, th.fg, bold=True)
    for y, txt, col in (
        (196, "f_u = 0.58 c₀ / d   (Clause 5.4.1)", th.accent),
        (232, "0.05 c₀/f_min < s < 0.45 c₀/f_max   (Clause 5.4.2)", th.accent),
        (268, "Working range: 250–1600 Hz (1/3-octave)", th.fg),
        (312, "Two-microphone transfer function H₁₂", th.fg),
        (344, "→ ISO 10534-2 decomposition → α(f)", th.primary),
    ):
        s.text(645, y, txt, 18, col, bold=(col is th.primary))
    s.text(645, 396, "Tube sealed onto the road; plane waves only below f_u",
           15, th.muted)


def _d_iso11654(s: SVG, th: Theme) -> None:
    """ISO 11654 single-number absorption rating: from αs to the absorption class."""
    cx = 450.0
    bw, bh = 664.0, 54.0
    x0 = cx - bw / 2

    s.rect(x0, 46, bw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 68, "Measured  αs  at one-third octaves, 200 Hz to 5000 Hz", 18,
           th.fg, "middle", bold=True)
    s.text(cx, 88, "from a reverberation room (ISO 354)", 13, th.muted, "middle")
    s.arrow(cx, 100, cx, 128, th.fg, 1.8)

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 23, l1, 17, th.fg, "middle", bold=True)
        s.text(cx, y + 42, l2, 13, th.muted, "middle")

    _step(128, "Practical  αp  per octave band, 250 Hz to 4000 Hz  (Clause 4.1)",
          "mean of the three one-third octaves, rounded to 0.05", th.primary)
    _step(206, "Shift the reference curve in 0.05 steps to best fit  (Clause 4.2)",
          "sum of unfavourable deviations kept ≤ 0.10", th.fg)
    _step(284, "Weighted coefficient  αw = shifted reference at 500 Hz", "", th.fg)
    _step(362, "Shape indicators (L, M, H) where  αp − reference ≥ 0.25", "", th.secondary)
    for y0, y1 in ((100, 128), (182, 206), (260, 284), (338, 362)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)
    s.arrow(cx, 416, cx, 444, th.fg, 1.8)

    s.rect(x0, 444, bw, 58, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, 469, "Sound absorption class  A to E   (Table B.1, Annex B)", 17,
           th.fg, "middle", bold=True)
    s.text(cx, 489, "or “Not classified” when αw falls below the class-E band",
           13, th.muted, "middle")


# ---------------------------------------------------------------------------
# Dynamic-stiffness resonance rig (ISO 9052-1 / EN 29052-1)
# ---------------------------------------------------------------------------

def _d_dynamic_stiffness_rig(s: SVG, th: Theme) -> None:
    """ISO 9052-1 rig: exciter and accelerometer on the load plate over the
    resilient specimen, read as a mass-spring resonance."""
    # ===== Left: rig cross-section =====
    s.text(240, 74, "Resonance rig", 22, th.fg, bold=True)
    gy = 466.0
    s.ground(gy, 50, 430)
    s.text(56, gy + 34, "Rigid foundation", 17, th.muted, anchor="start")

    x0, x1 = 150.0, 330.0
    spec_top, plate_h = 400.0, 26.0
    plate_top = spec_top - plate_h
    # Resilient specimen (soft diagonal hatching).
    s.rect(x0, spec_top, x1 - x0, gy - spec_top, th.panel, th.accent, sw=2)
    for hx in range(int(x0) + 14, int(x1) + 1, 22):
        s.line(hx, spec_top, hx - 12, gy, th.accent, 0.9)
    # Load plate on top of the specimen.
    s.rect(x0 - 12, plate_top, x1 - x0 + 24, plate_h, th.panel, th.primary,
           rx=3, sw=2.2)
    s.text(x1 + 26, plate_top + 19, "Load plate", 18, th.fg, anchor="start",
           bold=True)
    s.text(x1 + 26, plate_top + 43, "m′t = 200 kg/m²", 15, th.muted,
           anchor="start")
    s.text(x1 + 26, spec_top + 40, "Resilient specimen", 17, th.fg,
           anchor="start")
    s.text(x1 + 26, spec_top + 62, "200 mm × 200 mm", 15, th.muted,
           anchor="start")
    s.dim(x0, spec_top, x0, gy, "d", offset=-30, size=18)
    # Exciter, drive force and accelerometer on the plate.
    _exciter(s, 205.0, plate_top)
    s.text(205, plate_top - 100, "Exciter", 18, th.fg, bold=True)
    _motion_arrows(s, 256.0, plate_top - 36, 24.0, th.secondary)
    s.text(268, plate_top - 30, "F(t)", 16, th.secondary, anchor="start",
           mono=True)
    _accel(s, 300.0, plate_top)
    s.text(x1 + 26, plate_top - 14, "Accelerometer", 16, th.fg, anchor="start")
    s.line(309, plate_top - 8, x1 + 20, plate_top - 18, th.muted, 1.1,
           dash="3,3")

    # ===== Right: the mass-spring reading =====
    s.text(680, 74, "Mass-spring model", 22, th.fg, bold=True)
    mx = 680.0
    s.rect(mx - 60, 120, 120, 62, th.panel, th.primary, rx=8, sw=2.2)
    s.text(mx, 158, "m′t", 22, th.fg, mono=True, bold=True)
    _spring_v(s, mx, 182, 288, th.accent, coils=4)
    s.text(mx + 26, 240, "s′t", 20, th.accent, anchor="start", mono=True,
           bold=True)
    s.ground(288, mx - 70, mx + 70)
    _motion_arrows(s, mx - 92, 151, 26, th.secondary)

    # Response curve with the resonance read at its peak.
    ax0, ax1, base = 540.0, 850.0, 420.0
    s.line(ax0, base, ax1, base, th.muted, 1.4)
    s.line(ax0, base, ax0, 330.0, th.muted, 1.4)
    pk = 660.0
    s.path(f"M {ax0 + 6} {base - 12} C {pk - 60} {base - 16} {pk - 34} 336 "
           f"{pk} 334 C {pk + 34} 336 {pk + 70} {base - 8} {ax1 - 6} {base - 4}",
           stroke=th.primary, sw=2.4)
    s.line(pk, base, pk, 336, th.muted, 1.2, dash="4,3")
    s.text(pk, base + 22, "fr", 18, th.secondary, mono=True, bold=True)
    s.text((ax0 + ax1) / 2, base + 48, "resonance read from the response peak",
           15, th.muted, italic=True)

    # Headline relations.
    s.text(450, 524, "s′t = 4π² m′t fr²   (Formula 4)", 21, th.primary,
           bold=True, mono=True)
    s.text(450, 550,
           "then f₀ = (1/2π)·√(s′/m′) for the installed floating floor   (Formula 2)",
           16, th.muted, mono=True)


# ---------------------------------------------------------------------------
# Porous absorber on a rigid wall (equivalent fluid, JCA parameters)
# ---------------------------------------------------------------------------

def _d_porous_layer(s: SVG, th: Theme) -> None:
    """Section of a 50 mm mineral-wool layer on a rigid backing under a
    normal-incidence plane wave, with a magnified microstructure detail and
    the JCA parameter set of the guide's material (sigma = 20 kPa.s/m2,
    phi = 0.98, alpha_inf = 1, Lambda = Lambda' = 87 um); the layered
    absorber solves alpha = 0.91 at 1 kHz."""
    import math
    lay_l, lay_r = 560.0, 700.0        # 140 px for 50 mm
    top, bot = 100.0, 430.0

    # Rigid backing and the porous layer with a deterministic fibre texture.
    s.rect(lay_r, top, 34, bot - top, th.fg)
    s.text(784, 458, "Rigid backing", 16, th.muted)
    s.rect(lay_l, top, lay_r - lay_l, bot - top, th.panel, th.secondary, sw=2)
    for i in range(80):
        h1 = math.sin(i * 12.9898) * 43758.5453
        h1 -= math.floor(h1)
        h2 = math.sin(i * 78.233) * 24634.6345
        h2 -= math.floor(h2)
        h3 = math.sin(i * 39.425) * 11369.535
        h3 -= math.floor(h3)
        cx = lay_l + 8 + h1 * (lay_r - lay_l - 16)
        cy = top + 10 + h2 * (bot - top - 20)
        ang = h3 * math.pi
        dx, dy = 7.0 * math.cos(ang), 7.0 * math.sin(ang)
        s.line(cx - dx, cy - dy, cx + dx, cy + dy, th.muted, 1.0)
    s.text(630, 88, "Porous layer (mineral wool)", 18, th.fg, bold=True)
    s.dim(lay_l, bot, lay_r, bot, "d = 50 mm", offset=30, size=17)

    # Incident and reflected waves, and the decaying wave inside the layer.
    s.arrow(300.0, 240.0, lay_l - 8, 240.0, th.accent, 2.4)
    s.text(420, 268, "plane wave, normal incidence", 16, th.accent)
    s.arrow(lay_l - 8, 300.0, 445.0, 300.0, th.secondary, 1.6)
    s.text(438, 348, "reflected: |R|² = 1 − α = 0.09", 15, th.secondary)
    d = f"M {lay_l + 2:.0f} 240"
    for i in range(1, 35):
        x = lay_l + 2 + i * 4.0
        amp = 26.0 * math.exp(-i / 12.0)
        d += f" L {x:.1f} {240 - amp * math.sin(i * 0.9):.1f}"
    s.path(d, stroke=th.primary, sw=1.8)

    # Magnified microstructure: sampled spot on the layer, blown-up circle.
    s.circle(610.0, 160.0, 16.0, "none", th.fg, 1.6)
    s.circle(170.0, 185.0, 92.0, th.panel, th.fg, 2.0)
    s.line(597.0, 150.0, 253.0, 148.0, th.muted, 1.0, dash="4,4")
    s.line(600.0, 173.0, 246.0, 232.0, th.muted, 1.0, dash="4,4")
    for i in range(11):
        h1 = math.sin(i * 21.9898) * 43758.5453
        h1 -= math.floor(h1)
        h2 = math.sin(i * 57.233) * 24634.6345
        h2 -= math.floor(h2)
        h3 = math.sin(i * 93.719) * 11369.535
        h3 -= math.floor(h3)
        ang0 = h3 * math.pi
        r0 = 12.0 + h2 * 62.0
        cx = 170.0 + (h1 - 0.5) * 2 * r0 * math.cos(ang0)
        cy = 185.0 + (h1 - 0.5) * 2 * r0 * math.sin(ang0)
        dx, dy = 34.0 * math.cos(ang0 + 1.1), 34.0 * math.sin(ang0 + 1.1)
        # Clip fibre ends into the circle by shortening long excursions.
        s.line(cx - dx, cy - dy, cx + dx, cy + dy, th.secondary, 3.0)
    s.text(170, 80, "microstructure (zoom)", 16, th.fg, bold=True)
    s.text(96, 300, "fibre frame", 15, th.secondary, anchor="start")
    s.line(120.0, 292.0, 140.0, 252.0, th.muted, 1.0)
    s.text(190, 322, "air in the pores: φ = 0.98", 15, th.fg, anchor="start")
    s.line(214.0, 314.0, 200.0, 262.0, th.muted, 1.0)

    # JCA parameter block (the guide's material).
    for yy, txt in (
        (368.0, "σ = 20 kPa·s/m²  (flow resistivity)"),
        (392.0, "φ = 0.98  (porosity)"),
        (416.0, "α∞ = 1.0  (tortuosity)"),
        (440.0, "Λ = Λ′ = 87 µm  (viscous / thermal lengths)"),
    ):
        s.text(60, yy, txt, 16, th.fg, anchor="start", mono=True)

    # --- captions ----------------------------------------------------------
    s.text(80, 500,
           "JCA equivalent fluid: the five parameters give Zc and k; a hard-backed layer has Zs = −j Zc cot(kd)",
           17, th.fg, anchor="start")
    s.text(80, 528, "α = 1 − |R|² = 0.91 at 1 kHz for this 50 mm layer",
           18, th.primary, anchor="start", bold=True)
    s.text(80, 556,
           "viscous friction in the pores and heat exchange with the frame dissipate the sound energy",
           17, th.muted, anchor="start")
