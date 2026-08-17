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
    s.text(118, tube_bot + 42, "Loudspeaker", 17, th.fg, bold=True)

    # Test specimen and rigid backing at the right end.
    s.rect(tube_r - back_w, tube_top, back_w, tube_bot - tube_top, th.fg)
    s.rect(spec_l, tube_top, spec_w, tube_bot - tube_top, th.panel, th.secondary, sw=2)
    for hx in range(int(spec_l) + 8, int(spec_l + spec_w), 11):
        s.line(hx, tube_bot - 4, hx - 16, tube_top + 4, th.secondary, 1.0)
    s.text(spec_l + spec_w / 2, tube_top - 14, "Test specimen", 16, th.secondary, bold=True)
    s.text(tube_r - back_w / 2, tube_bot + 42, "Rigid backing", 15, th.muted)

    # Two microphones flush in the top wall (mic 1 = farther from specimen).
    m1x, m2x = 460.0, 555.0
    for mx, lab in ((m1x, "Mic 1"), (m2x, "Mic 2")):
        s.rect(mx - 7, tube_top - 20, 14, 20, th.fg, rx=3)
        s.circle(mx, tube_top, 5, th.primary)
        s.text(mx, tube_top - 28, lab, 15, th.fg, bold=True)

    # Plane-wave arrows inside the tube.
    s.arrow(tube_l + 30, mid - 18, spec_l - 16, mid - 18, th.accent, 2.2)
    s.text((tube_l + spec_l) / 2 - 40, mid - 26, "incident", 15, th.accent)
    s.arrow(spec_l - 16, mid + 20, tube_l + 30, mid + 20, th.secondary, 2.2)
    s.text((tube_l + spec_l) / 2 - 40, mid + 38, "reflected", 15, th.secondary)

    # Dimensions: x1 (specimen face -> far mic) above, spacing s below.
    s.dim(spec_l, tube_top, m1x, tube_top, "$x_1$", offset=-58, size=16)
    s.dim(m1x, tube_bot, m2x, tube_bot, "$s$", offset=70, size=16)

    # Governing relations and range.
    for y, txt, col in (
        (438, ("$H_{12}$ → reflection factor $r$ (Eq. 17), "
              "absorption $α = 1 − |r|^2$ (Eq. 18), "
              "$Z/ρc_0 = (1+r)/(1−r)$ (Eq. 19)"), th.fg),
        (466, ("Working range $f_l < f < f_u$ set by the microphone spacing "
              "$s$ and the tube diameter (Clause 6.1)"), th.muted),
        (492, ("ASTM E2611: two further microphones behind the specimen also "
              "give the transmission loss"), th.muted),
    ):
        s.text(450, y, txt, 15, col)


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
    s.text(96, tube_bot + 40, "Source", 16, th.fg, bold=True)

    # Adjustable termination (two loads) at the right end.
    s.rect(tube_r - 20, tube_top, 20, tube_bot - tube_top, th.fg)
    s.text(tube_r - 10, tube_bot + 40, "Termination", 15, th.muted)
    s.text(tube_r - 10, tube_bot + 60, "(2 loads)", 15, th.muted)

    # Test specimen at the centre.
    s.rect(spec_l, tube_top, spec_r - spec_l, tube_bot - tube_top, th.panel,
           th.secondary, sw=2)
    for hx in range(int(spec_l) + 7, int(spec_r), 10):
        s.line(hx, tube_bot - 4, hx - 14, tube_top + 4, th.secondary, 1.0)
    s.text((spec_l + spec_r) / 2, tube_bot + 40, "Test specimen", 15,
           th.secondary, bold=True)

    # Four microphones flush in the top wall (1,2 upstream; 3,4 downstream).
    for mx, lab in ((m1x, "Mic 1"), (m2x, "Mic 2"), (m3x, "Mic 3"), (m4x, "Mic 4")):
        s.rect(mx - 6, tube_top - 18, 12, 18, th.fg, rx=3)
        s.circle(mx, tube_top, 5, th.primary)
        s.text(mx, tube_top - 26, lab, 14, th.fg, bold=True)

    # Up- and downstream travelling waves.
    s.arrow(tube_l + 26, mid - 16, spec_l - 8, mid - 16, th.accent, 2.0)
    s.arrow(spec_l - 8, mid + 18, tube_l + 26, mid + 18, th.secondary, 2.0)
    s.arrow(spec_r + 8, mid - 16, tube_r - 26, mid - 16, th.accent, 2.0)
    s.arrow(tube_r - 26, mid + 18, spec_r + 8, mid + 18, th.secondary, 2.0)
    s.text(tube_l + 40, mid - 22, "$A$", 15, th.accent, bold=True)
    s.text(tube_l + 40, mid + 34, "$B$", 15, th.secondary, bold=True)
    s.text(tube_r - 40, mid - 22, "$C$", 15, th.accent, bold=True)
    s.text(tube_r - 40, mid + 34, "$D$", 15, th.secondary, bold=True)

    # Dimensions: spacings s1/s2 below; specimen offsets l1/l2 and thickness d above.
    s.dim(m1x, tube_bot, m2x, tube_bot, "$s_1$", offset=62, size=15)
    s.dim(m3x, tube_bot, m4x, tube_bot, "$s_2$", offset=62, size=15)
    # l1, l2 are both measured from the specimen FRONT face (x = 0), matching
    # wave_decomposition/transfer_matrix_two_load; l2 therefore spans the specimen.
    s.dim(m2x, tube_top, spec_l, tube_top, "$l_1$", offset=-42, size=15)
    s.dim(spec_l, tube_top, m3x, tube_top, "$l_2$", offset=-58, size=15)
    s.dim(spec_l, tube_top - 78, spec_r, tube_top - 78, "$d$", offset=0, size=15)
    s.line(spec_l, tube_top, spec_l, tube_top - 78, th.muted, 0.9, dash="3,3")
    s.line(spec_r, tube_top, spec_r, tube_top - 78, th.muted, 0.9, dash="3,3")

    # Governing relations.
    for y, txt, col in (
        (452, ("Decompose $A$, $B$ (upstream) and $C$, $D$ (downstream) → "
              "transfer matrix $T$ (Eq. 22)"), th.fg),
        (480, "$TL = 20 log_{10} |(T_{11} + T_{12}/ρc + ρc·T_{21} + T_{22}) / 2|$   (Eq. 26)",
         th.muted),
        (506, ("Two-load method: repeat with two terminations; the one-load "
              "variant uses a single anechoic end"), th.muted),
    ):
        s.text(450, y, txt, 15, col)


def _d_airflow(s: SVG, th: Theme) -> None:
    """ISO 9053-1 static and ISO 9053-2 alternating airflow-resistance rigs."""
    # --- Left panel: static (DC) method -----------------------------------
    s.rect(38, 70, 400, 560, th.panel, th.fg, rx=8, sw=2)
    s.text(238, 100, "Static method (ISO 9053-1)", 18, th.fg, bold=True)

    cx = 150.0
    holder_l, holder_r = cx - 45, cx + 45
    top_y, bot_y = 150.0, 440.0
    # Vertical measurement cell (Clause 5.2).
    s.line(holder_l, top_y, holder_l, bot_y, th.fg, 2.5)
    s.line(holder_r, top_y, holder_r, bot_y, th.fg, 2.5)
    # Specimen (hatched disc) with its edge seal.
    spec_y, spec_h = 262.0, 40.0
    s.rect(holder_l, spec_y, 90, spec_h, th.bg, th.secondary, sw=2)
    for hy in range(int(spec_y) + 10, int(spec_y + spec_h) + 8, 10):
        s.line(holder_l + 4, min(hy, spec_y + spec_h - 2),
               holder_r - 4, max(hy - 10, spec_y + 2), th.secondary, 1.0)
    s.rect(holder_l - 6, spec_y, 8, spec_h, th.accent)
    s.rect(holder_r - 2, spec_y, 8, spec_h, th.accent)
    s.text(holder_l - 12, spec_y + 26, "seal", 13, th.accent, bold=True,
           anchor="end")
    s.text(cx - 6, spec_y - 30, "specimen  $A$, $d$", 14, th.secondary, bold=True)
    # Perforated support under the specimen.
    for gx in range(int(holder_l) + 8, int(holder_r) - 2, 12):
        s.line(gx, spec_y + spec_h + 22, gx, spec_y + spec_h + 34, th.fg, 2.0)
    s.line(holder_l, spec_y + spec_h + 22, holder_r, spec_y + spec_h + 22,
           th.fg, 1.6)
    s.text(holder_r + 10, spec_y + spec_h + 34, "grid", 13, th.muted,
           anchor="start")
    # Steady laminar flow up through the holder, from a controlled source.
    s.arrow(cx, bot_y - 6, cx, spec_y + spec_h + 46, th.accent, 2.4)
    s.arrow(cx, spec_y - 46, cx, top_y + 26, th.accent, 2.4)
    s.rect(cx - 44, bot_y + 8, 88, 36, th.bg, th.accent, rx=8, sw=2)
    s.text(cx, bot_y + 32, "$q_v$", 15, th.accent, bold=True)
    s.rect(cx - 44, bot_y + 56, 88, 34, th.bg, th.muted, rx=8, sw=1.6)
    s.text(cx, bot_y + 78, "flow source", 13, th.muted)
    # Differential manometer across the specimen (pressure taps).
    tap_x = holder_r + 34
    s.line(holder_r, spec_y + 2, tap_x, spec_y + 2, th.primary, 1.6)
    s.line(holder_r, spec_y + spec_h - 2, tap_x, spec_y + spec_h - 2,
           th.primary, 1.6)
    s.rect(tap_x, spec_y - 18, 82, spec_h + 34, th.bg, th.primary, rx=8, sw=2)
    s.text(tap_x + 41, spec_y + 24, "$Δp$", 19, th.primary, bold=True)
    # Thickness gauge resting on the specimen, in position (Clause 7.3).
    s.circle(cx + 62, top_y + 42, 14, th.bg, th.muted, 1.8)
    s.text(cx + 62, top_y + 48, "$d$", 14, th.muted, bold=True)
    s.line(cx + 62, top_y + 56, cx + 62, spec_y - 2, th.muted, 1.6, dash="4,3")
    s.line(cx + 30, spec_y - 2, cx + 70, spec_y - 2, th.muted, 1.8)
    # The free space Clause 5.2 asks for ahead of the specimen.
    s.dim(holder_l - 26, spec_y, holder_l - 26, top_y, "≥ 1 bore", offset=0,
          size=12, label_side="left")

    for yy, txt in (
        (546, "cell ≥ 29 mm bore, ≥ 1 bore of free space above"),
        (568, "$q_v$ and $Δp$ each to ±5 %, $Δp$ readable to 0.1 Pa"),
        (590, "grid ≥ 50 % open, $R < 1$ %; $d$ measured in position"),
    ):
        s.text(238, yy, txt, 12, th.muted)
    s.text(238, 616, "$R = Δp / q_v$   (through-origin fit at 0.5 mm/s)",
           14, th.fg, bold=True)

    # --- Right panel: alternating (AC) method -----------------------------
    s.rect(460, 70, 400, 560, th.panel, th.fg, rx=8, sw=2)
    s.text(660, 100, "Alternating method (ISO 9053-2)", 18, th.fg, bold=True)

    cav_l, cav_r = 590.0, 715.0
    cav_top, cav_bot = 210.0, 410.0
    # Cavity walls.
    s.rect(cav_l, cav_top, cav_r - cav_l, cav_bot - cav_top, th.bg, th.fg, sw=2.5)
    s.text((cav_l + cav_r) / 2, (cav_top + cav_bot) / 2 - 6, "cavity", 15, th.fg)
    s.text((cav_l + cav_r) / 2, (cav_top + cav_bot) / 2 + 18, "$V$", 17, th.fg,
           bold=True)
    # Specimen cell, or the airtight termination that replaces it.
    s.rect(cav_l, cav_top - 26, cav_r - cav_l, 26, th.bg, th.secondary, sw=2)
    for hx in range(int(cav_l) + 8, int(cav_r), 11):
        s.line(hx, cav_top - 4, hx - 14, cav_top - 22, th.secondary, 1.0)
    s.text((cav_l + cav_r) / 2, cav_top - 36, "measurement cell → $L_{p,s}$ ($h_s$)",
           13, th.secondary, bold=True)
    s.text((cav_l + cav_r) / 2, cav_top - 58,
           "airtight termination → $L_{p,t}$ ($h_t$)", 13, th.muted)
    # Piston at the bottom, oscillating.
    s.rect(cav_l, cav_bot, cav_r - cav_l, 26, th.panel, th.primary, sw=2)
    s.arrow((cav_l + cav_r) / 2, cav_bot + 62, (cav_l + cav_r) / 2, cav_bot + 34,
            th.primary, 2.2)
    s.arrow((cav_l + cav_r) / 2, cav_bot + 34, (cav_l + cav_r) / 2, cav_bot + 62,
            th.primary, 2.2)
    s.text((cav_l + cav_r) / 2, cav_bot + 84, "piston  $f$ = 1–4 Hz", 15,
           th.primary, bold=True)
    s.text((cav_l + cav_r) / 2, cav_bot + 106, "$q_v = 2π f h A_P$", 13,
           th.muted)
    # Microphone in the cavity wall.
    s.circle(cav_r + 2, (cav_top + cav_bot) / 2, 6, th.fg)
    s.line(cav_r + 2, (cav_top + cav_bot) / 2, cav_r + 60,
           (cav_top + cav_bot) / 2, th.muted, 1.4)
    s.text(cav_r + 66, (cav_top + cav_bot) / 2 + 6, "$L_p$", 17, th.fg,
           bold=True, anchor="start")
    s.text(660, 616, "$R$ from $L_{p,s} − L_{p,t}$   ($κ′$ per Annex A)",
           14, th.fg, bold=True)


# ---------------------------------------------------------------------------
# d15 - ISO 17497-1 random-incidence scattering (reverberation room)
# ---------------------------------------------------------------------------

def _d_scattering_reverb(s: SVG, th: Theme) -> None:
    """ISO 17497-1 scattering coefficient in a reverberation room."""
    gy = 400.0
    # Reverberation room with non-parallel walls (skew quadrilateral).
    s.path("M 60 80 L 782 66 L 796 400 L 72 400 Z", fill=th.panel,
           stroke=th.fg, sw=3)
    s.text(80, 106, "Reverberation room", 17, th.fg, bold=True, anchor="start")

    # --- Turntable carrying the test sample (left, in perspective) --------
    tx, tyc = 285.0, 366.0
    s.ellipse(tx, tyc, 150, 26, th.panel, th.primary, 2.2)      # turntable
    s.ellipse(tx, tyc - 12, 82, 15, th.bg, th.secondary, 2.2)   # test sample
    for hx in range(int(tx) - 60, int(tx) + 60, 12):            # sample hatch
        s.line(hx, tyc - 10, hx + 10, tyc - 18, th.secondary, 1.0)
    s.text(tx, gy + 22, "Turntable and base plate", 15, th.fg, bold=True)
    _rot_arrow(s, tx, tyc, 150, 205, 340, th.accent, 2.2, ry=26)
    s.text(tx, tyc - 70, "the only thing that moves", 13, th.accent)
    s.text(tx, tyc - 46, "sample on the plate for $T_2$ and $T_4$", 13, th.muted)
    # Wall clearance of the turntable rim.
    s.line(78, 386, tx - 150, 386, th.muted, 1.4)
    s.line(78, 380, 78, 392, th.muted, 1.4)
    s.line(tx - 150, 380, tx - 150, 392, th.muted, 1.4)
    s.text(106, 376, "≥ 1.0 m", 12, th.muted)

    # --- Two fixed loudspeaker positions (right) --------------------------
    for sx, sy, lab in ((648.0, 262.0, "S1"), (752.0, 296.0, "S2")):
        s.rect(sx - 20, sy - 26, 40, 52, th.panel, th.primary, rx=6, sw=2)
        s.circle(sx, sy, 11, th.primary)
        s.circle(sx, sy, 4, th.bg)
        s.line(sx, sy + 26, sx, gy, th.fg, 2.2)
        s.line(sx - 14, gy, sx + 14, gy, th.fg, 2.2)
        s.text(sx, sy - 34, lab, 15, th.fg, bold=True)
    s.text(700, 196, "fixed sources (≥ 2)", 14, th.muted)

    # --- Three fixed microphone positions ---------------------------------
    for mx, my, lab in ((452.0, 262.0, "M1"), (520.0, 286.0, "M2"),
                        (586.0, 310.0, "M3")):
        s.mic(mx, my, gy, 1.0)
        s.text(mx, my - 12, lab, 14, th.fg, bold=True)
    s.text(470, 226, "fixed microphones (≥ 3)", 14, th.muted)

    # --- Governing relations ----------------------------------------------
    for y, txt, col, bold in (
        (448, ("$T_1$ base plate, static  ·  $T_2$ sample, static  →  "
               "$α_s$ (Eq. 1)"), th.fg, True),
        (474, ("$T_3$ base plate, rotating  ·  $T_4$ sample, rotating  →  "
               "$α_{spec}$ (Eq. 4)"), th.fg, True),
        (502, "$s = (α_{spec} − α_s) / (1 − α_s)$   (Eq. 5)", th.accent, True),
        (528, ("$α$ from $55.3·(V/S)·(1/(c T)) − 4(V/S)m$  ·  the base plate "
               "must pass the Table 1 ceiling"), th.muted, False),
    ):
        s.text(450, y, txt, 16 if bold else 15, col, bold=bold)


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
    s.text(cx + R + 4, gy - 4, "$L_n$", 15, th.fg, anchor="start")
    s.text(cx - R - 4, gy - 4, "$L_1$", 15, th.fg, anchor="end")
    s.text(cx, gy - R - 14, "$L_i$", 15, th.fg)
    s.text(cx + 150, gy - 250, "receiver arc (5° steps)", 14, th.muted)
    _ = ends

    # Polar (scattered) response lobe about the sample centre.
    pts = []
    for ang in range(0, 181, 6):
        a = math.radians(ang)
        rr = 92.0 + 42.0 * abs(math.sin(3.0 * a))
        pts.append((cx + rr * math.cos(a), gy - rr * math.sin(a)))
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    s.path(d, stroke=th.accent, sw=2.0)
    s.text(cx + 96, gy - 150, "polar response $L_i$", 14, th.accent)

    # Fixed source, off to the upper left, illuminating the sample.
    sa = math.radians(155.0)
    sxx, syy = cx + (R + 44) * math.cos(sa), gy - (R + 44) * math.sin(sa)
    s.rect(sxx - 26, syy - 22, 52, 44, th.panel, th.primary, rx=6, sw=2)
    s.circle(sxx + 20, syy, 10, th.primary)
    s.circle(sxx + 20, syy, 4, th.bg)
    s.text(sxx, syy - 32, "Fixed source", 15, th.fg, bold=True)
    s.arrow(sxx + 26, syy + 6, cx - 74, gy - 12, th.accent, 2.0)

    # Test sample on the turntable at the arc centre.
    s.rect(cx - 72, gy - 13, 144, 13, th.bg, th.secondary, sw=2)
    for hx in range(int(cx) - 64, int(cx) + 64, 12):
        s.line(hx, gy - 3, hx + 9, gy - 11, th.secondary, 1.0)
    s.text(cx, gy - 20, "Test sample", 14, th.secondary, bold=True)
    s.ellipse(cx, gy + 8, 88, 12, "none", th.primary, 1.8)
    _rot_arrow(s, cx, gy + 8, 88, 200, 340, th.primary, 1.8, ry=12)
    s.text(cx + 150, gy + 12, "Turntable", 14, th.fg, bold=True, anchor="start")

    # Governing relations. Formula 5 stays plain for now: its 10^(L_i/10)
    # terms put a subscript inside the exponent, one script level more than
    # the composer sets (the same energy-sum family the flanking and
    # reception-plate sums parked).
    s.text(450, 476,
           "d = [(Σ10^(L_i/10))² − Σ(10^(L_i/10))²] / "
           "[(n−1)·Σ(10^(L_i/10))²]   (Formula 5)", 15, th.fg, bold=True)
    s.text(450, 506, "$d_n = (d − d_{ref}) / (1 − d_{ref})$   (Formula 7)", 15,
           th.accent, bold=True)
    s.text(450, 534,
           "5° receiver steps · turntable rotates the sample · source fixed",
           15, th.muted)


# ---------------------------------------------------------------------------
# d17 - ISO 13472-1 in-situ road absorption, subtraction technique
# ---------------------------------------------------------------------------

def _d_insitu_subtraction(s: SVG, th: Theme) -> None:
    """ISO 13472-1 extended-surface (subtraction) in-situ absorption."""
    gy = 415.0
    # Road surface (the reference plane) under the main measurement.
    s.ground(gy, 55, 590)
    s.text(66, gy + 30, "Road surface", 14, th.muted, anchor="start")

    sx = 250.0
    src_y, mic_y = gy - 235.0, gy - 47.0        # ds : dm = 1.25 : 0.25 m
    s.line(sx, src_y, sx, gy, th.muted, 1.0, dash="4,4")   # normal axis

    # Loudspeaker (source) at ds above the surface.
    s.rect(sx - 30, src_y - 30, 60, 60, th.panel, th.primary, rx=6, sw=2)
    s.circle(sx, src_y, 12, th.primary)
    s.circle(sx, src_y, 5, th.bg)
    s.text(sx, src_y - 42, "Loudspeaker", 15, th.fg, bold=True)

    # Microphone at dm above the surface.
    s.rect(sx - 6, mic_y - 9, 12, 18, th.fg, rx=3)
    s.circle(sx, mic_y - 9, 5, th.primary)
    s.text(sx + 16, mic_y + 5, "Microphone", 13, th.fg, anchor="start")

    # Direct ray (source -> mic), drawn offset to the left of the axis.
    s.arrow(sx - 7, src_y + 22, sx - 7, mic_y - 12, th.accent, 2.0)
    s.text(sx - 60, (src_y + mic_y) / 2, "direct  $d_s−d_m$", 13, th.accent,
           anchor="end")
    # Road-reflected ray: source -> surface point -> mic (shallow V, offset).
    gpx = sx + 74.0
    s.line(sx + 8, src_y + 24, gpx, gy, th.secondary, 2.0)
    s.arrow(gpx, gy, sx + 8, mic_y + 6, th.secondary, 2.0)
    s.text(gpx + 8, gy - 96, "reflected  $d_s+d_m$", 13, th.secondary,
           anchor="start")
    # Dashed continuation toward the image source below the plane.
    s.line(gpx, gy, sx + 34, gy + 66, th.muted, 1.2, dash="5,4")
    s.text(sx + 40, gy + 60, "to image source ($d_s$ below)", 12, th.muted,
           anchor="start")

    # Height dimensions ds and dm.
    s.dim(sx - 72, gy, sx - 72, src_y, "$d_s$ = 1.25 m", offset=0,
          label_side="left", size=15)
    s.line(sx - 72, gy, sx, gy, th.muted, 0.9, dash="3,3")
    s.line(sx - 72, src_y, sx - 30, src_y, th.muted, 0.9, dash="3,3")
    s.dim(sx + 122, gy, sx + 122, mic_y, "$d_m$ = 0.25 m", offset=0,
          label_side="right", size=15)
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
    s.text(fx, fs_y - 40, "Free-field reference", 15, th.fg, bold=True)
    s.text(fx, fm_y + 34, "$H_i$: no ground reflection in the window", 12,
           th.muted)

    # Governing relations.
    s.text(450, 502, "$K_r = (d_s − d_m)/(d_s + d_m) = 2/3$   (Clause 4.1)", 15,
           th.fg, bold=True)
    s.text(450, 528, "$α(f) = 1 − (1/K_r^2)·|H_r/H_i|^2$   ·   $Δτ = 2 d_m / c$", 15,
           th.accent, bold=True)
    s.text(450, 552, "Adrienne time window isolates the reflected response $H_r$",
           14, th.muted)


# ---------------------------------------------------------------------------
# d18 - ISO 13472-2 in-situ road absorption, spot method
# ---------------------------------------------------------------------------

def _d_spot_tube(s: SVG, th: Theme) -> None:
    """ISO 13472-2 spot method: short tube sealed onto the road surface."""
    gy = 430.0
    cx, hw, y_top = 235.0, 72.0, 120.0

    # Road surface (the test sample) with the tube sealed onto it.
    s.ground(gy, 60, 430)
    s.text(72, gy + 30, "Road surface (test sample)", 13, th.muted,
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
    s.text(cx, y_top - 52, "Loudspeaker", 15, th.fg, bold=True)

    # Two microphones flush in the right wall, spacing s.
    m1y, m2y = gy - 158.0, gy - 82.0
    for my, lab in ((m1y, "Mic 1"), (m2y, "Mic 2")):
        s.rect(cx + hw - 4, my - 7, 12, 14, th.fg, rx=3)
        s.circle(cx + hw, my, 4, th.primary)
        s.text(cx + hw + 16, my + 5, lab, 13, th.fg, anchor="start")

    # Plane-wave travel down and reflection back up.
    s.arrow(cx - 34, y_top + 16, cx - 34, gy - 26, th.accent, 2.0)
    s.arrow(cx - 8, gy - 26, cx - 8, y_top + 16, th.secondary, 2.0)

    # Dimensions: tube diameter d (across) and mic spacing s (down).
    s.dim(cx - hw, y_top + 18, cx + hw, y_top + 18, "$d$", offset=0, size=15)
    s.dim(cx + hw + 62, m1y, cx + hw + 62, m2y, "$s$", offset=0,
          label_side="right", size=15)
    s.line(cx + hw + 10, m1y, cx + hw + 62, m1y, th.muted, 0.9, dash="3,3")
    s.line(cx + hw + 10, m2y, cx + hw + 62, m2y, th.muted, 0.9, dash="3,3")

    # Right panel: usable frequency range and DSP method.
    s.rect(430, 118, 430, 300, "none", th.muted, rx=12, dash="6,5")
    s.text(645, 152, "Spot method (ISO 13472-2)", 17, th.fg, bold=True)
    for y, txt, col in (
        (196, "$f_u = 0.58 c_0 / d$   (Clause 5.4.1)", th.accent),
        (232, "$0.05 c_0/f_{min} < s < 0.45 c_0/f_{max}$   (Clause 5.4.2)", th.accent),
        (268, "Working range: 250–1600 Hz (1/3-octave)", th.fg),
        (312, "Two-microphone transfer function $H_{12}$", th.fg),
        (344, "→ ISO 10534-2 decomposition → $α(f)$", th.primary),
    ):
        s.text(645, y, txt, 15, col, bold=(col is th.primary))
    s.text(645, 396, "Tube sealed onto the road; plane waves only below $f_u$",
           13, th.muted)


def _d_iso11654(s: SVG, th: Theme) -> None:
    """ISO 11654 single-number absorption rating: from αs to the absorption class."""
    cx = 450.0
    bw, bh = 664.0, 54.0
    x0 = cx - bw / 2

    s.rect(x0, 46, bw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 68, "Measured  $α_s$  at one-third octaves, 200 Hz to 5000 Hz", 15,
           th.fg, "middle", bold=True)
    s.text(cx, 88, "from a reverberation room (ISO 354)", 11, th.muted, "middle")
    s.arrow(cx, 100, cx, 128, th.fg, 1.8)

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 23, l1, 15, th.fg, "middle", bold=True)
        s.text(cx, y + 42, l2, 11, th.muted, "middle")

    _step(128, "Practical  $α_p$  per octave band, 250 Hz to 4000 Hz  (Clause 4.1)",
          "mean of the three one-third octaves, rounded to 0.05", th.primary)
    _step(206, "Shift the reference curve in 0.05 steps to best fit  (Clause 4.2)",
          "sum of unfavourable deviations kept ≤ 0.10", th.fg)
    _step(284, "Weighted coefficient  $α_w$ = shifted reference at 500 Hz",
          "read off the shifted curve, always a multiple of 0.05", th.fg)
    _step(362, "Shape indicators (L, M, H) where  $α_p$ − reference ≥ 0.25",
          "appended to $α_w$ in parentheses, e.g. 0.60(M)", th.secondary)
    for y0, y1 in ((100, 128), (182, 206), (260, 284), (338, 362)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)
    s.arrow(cx, 416, cx, 444, th.fg, 1.8)

    s.rect(x0, 444, bw, 58, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, 469, "Sound absorption class  A to E   (Table B.1, Annex B)", 15,
           th.fg, "middle", bold=True)
    s.text(cx, 489, "or “Not classified” when $α_w$ falls below the class-E band",
           11, th.muted, "middle")


# ---------------------------------------------------------------------------
# Dynamic-stiffness resonance rig (ISO 9052-1 / EN 29052-1)
# ---------------------------------------------------------------------------

def _dsr_arrangement(s: SVG, th: Theme, cx: float, base_y: float, *,
                     rigid: bool, drive_plate: bool, both: bool,
                     title: str, note: str) -> None:
    """One EN 29052-1 excitation arrangement, drawn to a common baseline.

    ``rigid`` draws the hatched foundation of the first arrangement; the
    other two stand on a baseplate of at least 100 kg carried on soft
    mounts. ``drive_plate`` puts the exciter on the load plate rather than
    under the baseplate, and ``both`` adds the second accelerometer.
    """
    half = 72.0
    x0, x1 = cx - half, cx + half
    spec_h, plate_h = 38.0, 20.0
    spec_top = base_y - spec_h
    plate_top = spec_top - plate_h

    s.text(cx, 96, title, 14, th.fg, bold=True)
    s.text(cx, 116, note, 12, th.muted)

    if rigid:
        s.ground(base_y, x0 - 40, x1 + 40)
        s.text(cx, base_y + 34, "Rigid foundation", 12, th.muted)
    else:
        s.rect(x0 - 40, base_y, 2 * half + 80, 24, th.panel, th.fg, sw=1.8)
        s.text(cx, base_y + 17, "Baseplate ≥ 100 kg", 11, th.muted)
        for sx in (x0 - 16, x1 + 16):
            _spring_v(s, sx, base_y + 24, base_y + 96, th.muted, coils=3,
                      width=9.0, sw=1.6)
        s.line(x0 - 40, base_y + 96, x1 + 40, base_y + 96, th.muted, 1.6)

    # Resilient specimen (soft diagonal hatching) and the load plate on it.
    s.rect(x0, spec_top, x1 - x0, spec_h, th.panel, th.accent, sw=1.8)
    for hx in range(int(x0) + 12, int(x1) + 1, 20):
        s.line(hx, spec_top, hx - 10, base_y, th.accent, 0.9)
    s.rect(x0 - 8, plate_top, x1 - x0 + 16, plate_h, th.panel, th.primary,
           rx=3, sw=2.0)

    # Excitation: on the load plate, or under the baseplate from below.
    if drive_plate:
        _exciter(s, cx - 30.0, plate_top, stinger=16.0, w=52.0, h=32.0)
        s.arrow(cx - 30.0, plate_top - 13, cx - 30.0, plate_top - 1,
                th.secondary, 2.0)
        s.text(cx - 30.0, plate_top - 58, "$F$", 14, th.secondary,
               bold=True)
    else:
        _exciter(s, cx, base_y + 24, stinger=18.0, w=52.0, h=32.0, up=True)
        s.arrow(cx, base_y + 38, cx, base_y + 26, th.secondary, 2.0)
        s.text(cx - 40, base_y + 52, "$F$", 14, th.secondary, anchor="end",
               bold=True)

    _accel(s, cx + 38.0, plate_top)
    if both:
        _accel(s, x1 + 24.0, base_y)


def _d_dynamic_stiffness_rig(s: SVG, th: Theme) -> None:
    """EN 29052-1 rig: the three excitation arrangements of Figures 1 to 3,
    the specimen preparation of Clauses 5 and 6, and the Formula 4 reading."""
    s.text(450, 56, "The three excitation arrangements (Figures 1 to 3)", 18,
           th.fg, bold=True)

    base_y = 300.0
    _dsr_arrangement(s, th, 158.0, base_y, rigid=True, drive_plate=True,
                     both=False,
                     title="Rigid base",
                     note="load plate measured")
    _dsr_arrangement(s, th, 450.0, base_y, rigid=False, drive_plate=True,
                     both=True,
                     title="Isolated baseplate",
                     note="load plate driven, both measured")
    _dsr_arrangement(s, th, 742.0, base_y, rigid=False, drive_plate=False,
                     both=True,
                     title="Isolated baseplate",
                     note="baseplate driven, both measured")
    s.text(450, 440,
           "all three are equivalent; sinusoidal excitation is the reference "
           "method in case of dispute (7.1)", 13, th.muted, italic=True)

    # ===== The specimen under the plate, and what the standard fixes =====
    s.text(215, 502, "Specimen and load (Clauses 5 and 6)", 16, th.fg,
           bold=True)
    gy = 690.0
    s.ground(gy, 60, 320)
    x0, x1 = 110.0, 300.0
    spec_top, plate_h, bed_h = 626.0, 24.0, 9.0
    plate_top = spec_top - bed_h - plate_h
    s.rect(x0, spec_top, x1 - x0, gy - spec_top, th.panel, th.accent, sw=2)
    for hx in range(int(x0) + 14, int(x1) + 1, 22):
        s.line(hx, spec_top, hx - 12, gy, th.accent, 0.9)
    # Plaster bed on its foil, between the specimen and the load plate.
    s.rect(x0, spec_top - bed_h, x1 - x0, bed_h, th.panel, th.secondary,
           sw=1.4)
    s.rect(x0 - 10, plate_top, x1 - x0 + 20, plate_h, th.panel, th.primary,
           rx=3, sw=2.2)
    s.line(x1, spec_top - bed_h / 2, 312, 566, th.secondary, 1.1, dash="3,3")
    s.text(318, 562, "plaster of Paris ≥ 5 mm on 0.02 mm foil", 11,
           th.secondary, anchor="start")
    s.text(318, 592, "Load plate, steel", 13, th.fg, anchor="start", bold=True)
    s.text(318, 611, "(200 ± 3) mm square, flat to 0.5 mm", 11, th.muted,
           anchor="start")
    s.text(318, 629, "8 kg ± 0.5 kg with every device on it", 11, th.muted,
           anchor="start")
    s.text(318, 657, "Resilient specimen, 200 mm × 200 mm", 13, th.fg,
           anchor="start", bold=True)
    s.text(318, 676, "three of them; irregularities < 3 mm", 11, th.muted,
           anchor="start")
    s.dim(x0, spec_top, x0, gy, "$d$", offset=-30, size=15)
    # Petroleum-jelly fillet, closed-cell materials only.
    s.path(f"M {x0} {gy} L {x0} {gy - 13} Q {x0 - 15} {gy - 5} {x0 - 17} {gy} Z",
           fill=th.secondary, stroke=th.secondary, sw=1.0)
    s.line(x0 - 12, gy - 4, 96, 722, th.secondary, 1.1, dash="3,3")
    s.text(100, 726, "petroleum-jelly fillet (closed-cell materials)", 11,
           th.secondary, anchor="start")

    # Headline relations, under the specimen half.
    s.text(258, 776, "$s′_t = 4π^2 m′_t f_r^2$   (Formula 4)", 17, th.primary,
           bold=True)
    s.text(258, 806, "$f_0 = (1/2π)·√(s′/m′)$   (Formula 2)", 14, th.muted)

    # ===== Right: the mass-spring reading and the response peak =====
    s.text(700, 502, "Mass-spring model", 16, th.fg, bold=True)
    mx = 700.0
    s.rect(mx - 52, 534, 104, 46, th.panel, th.primary, rx=8, sw=2.2)
    s.text(mx, 563, "$m′_t$", 17, th.fg, bold=True)
    _spring_v(s, mx, 580, 640, th.accent, coils=4)
    s.text(mx + 24, 618, "$s′_t$", 16, th.accent, anchor="start",
           bold=True)
    s.ground(640, mx - 62, mx + 62)
    _motion_arrows(s, mx - 78, 557, 20, th.secondary)

    ax0, ax1, base = 590.0, 862.0, 800.0
    s.line(ax0, base, ax1, base, th.muted, 1.4)
    s.line(ax0, base, ax0, 700.0, th.muted, 1.4)
    pk = 700.0
    s.path(f"M {ax0 + 6} {base - 10} C {pk - 56} {base - 14} {pk - 30} 712 "
           f"{pk} 710 C {pk + 30} 712 {pk + 64} {base - 7} {ax1 - 6} {base - 3}",
           stroke=th.primary, sw=2.4)
    s.line(pk, base, pk, 712, th.muted, 1.2, dash="4,3")
    s.text(pk, base + 20, "$f_r$", 15, th.secondary, bold=True)
    s.text((ax0 + ax1) / 2, base + 44,
           "read at the peak, extrapolated to zero force", 12, th.muted,
           italic=True)


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
    s.text(784, 458, "Rigid backing", 14, th.muted)
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
    s.text(630, 88, "Porous layer (mineral wool)", 15, th.fg, bold=True)
    s.dim(lay_l, bot, lay_r, bot, "$d$ = 50 mm", offset=30, size=15)

    # Incident and reflected waves, and the decaying wave inside the layer.
    s.arrow(300.0, 240.0, lay_l - 8, 240.0, th.accent, 2.4)
    s.text(420, 268, "plane wave, normal incidence", 14, th.accent)
    s.arrow(lay_l - 8, 300.0, 445.0, 300.0, th.secondary, 1.6)
    s.text(438, 348, "reflected: $|R|^2 = 1 − α$ = 0.09", 13, th.secondary)
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
    s.text(170, 80, "microstructure (zoom)", 14, th.fg, bold=True)
    s.text(96, 300, "fibre frame", 13, th.secondary, anchor="start")
    s.line(120.0, 292.0, 140.0, 252.0, th.muted, 1.0)
    s.text(190, 322, "air in the pores: $φ$ = 0.98", 13, th.fg, anchor="start")
    s.line(214.0, 314.0, 200.0, 262.0, th.muted, 1.0)

    # JCA parameter block (the guide's material).
    for yy, txt in (
        (368.0, "$σ$ = 20 kPa·s/m²  (flow resistivity)"),
        (392.0, "$φ$ = 0.98  (porosity)"),
        (416.0, "$α_∞$ = 1.0  (tortuosity)"),
        (440.0, "$Λ = Λ′$ = 87 µm  (viscous / thermal lengths)"),
    ):
        s.text(60, yy, txt, 14, th.fg, anchor="start")

    # --- captions ----------------------------------------------------------
    s.text(80, 500,
           "JCA equivalent fluid: five parameters give $Z_c$ and $k$; a "
           "hard-backed layer has $Z_s = −j Z_c cot(k·d)$",
           15, th.fg, anchor="start")
    s.text(80, 528, "$α = 1 − |R|^2$ = 0.91 at 1 kHz for this 50 mm layer",
           15, th.primary, anchor="start", bold=True)
    s.text(80, 556,
           "viscous friction in the pores and heat exchange with the frame dissipate the sound energy",
           15, th.muted, anchor="start")


# ---------------------------------------------------------------------------
# d24 - ISO 354 reverberation-room sound absorption
# ---------------------------------------------------------------------------

def _d_iso354_room(s: SVG, th: Theme) -> None:
    """ISO 354 reverberation-room absorption measurement (plan + two states)."""
    # --- The room in plan, with non-parallel walls (Clause 6.1.2) ----------
    s.path("M 46 100 L 596 82 L 610 424 L 60 410 Z", fill=th.panel,
           stroke=th.fg, sw=3)
    s.text(60, 76, "Reverberation room · plan", 17, th.fg, bold=True,
           anchor="start")
    s.text(596, 76, "$V$ = 200 m³ (≥ 150 m³)", 15, th.muted, anchor="end")

    # Suspended diffusers near the ceiling (Annex A.1).
    for dx, dy, tilt in ((132.0, 164.0, 14.0), (232.0, 150.0, -20.0),
                         (334.0, 166.0, 12.0), (436.0, 152.0, -14.0)):
        s.path(f"M {dx - 32} {dy + tilt} Q {dx} {dy - 12} {dx + 32} {dy - tilt}",
               stroke=th.muted, sw=3.0)
    s.text(284, 120, "diffusers  0.8–3 m² each, ≈ 5 kg/m² (Annex A)", 13,
           th.muted)

    # --- Test specimen on the floor, edges deliberately non-parallel ------
    ax_, ay = 122.0, 296.0        # corners, clockwise from top left
    bx, by = 316.0, 272.0
    cx_, cy = 330.0, 372.0
    dx_, dy = 136.0, 396.0
    s.path(f"M {ax_} {ay} L {bx} {by} L {cx_} {cy} L {dx_} {dy} Z",
           fill=th.bg, stroke=th.secondary, sw=2.4)
    for i in range(1, 14):        # hatch parallel to the short edges
        t = i / 14.0
        s.line(ax_ + t * (bx - ax_), ay + t * (by - ay),
               dx_ + t * (cx_ - dx_), dy + t * (cy - dy), th.secondary, 1.0)
    s.text(226, 262, "Test specimen  $S$ = 10.8 m²", 15, th.secondary, bold=True)
    s.text(226, 452, "10–12 m², width/length 0.7–1, no edge parallel to the "
                     "room",
           13, th.muted)
    # Clearance from the nearest room boundary (Clause 6.2.1.2).
    s.dim(66.0, 340.0, 122.0, 340.0, "≥ 0.75 m", offset=0, size=13)

    # --- Two source and three microphone positions (Clause 7.1) -----------
    for sx, sy, lab in ((470.0, 200.0, "S1"), (548.0, 344.0, "S2")):
        s.rect(sx - 19, sy - 25, 38, 50, th.panel, th.primary, rx=6, sw=2)
        s.circle(sx, sy, 10, th.primary)
        s.circle(sx, sy, 4, th.bg)
        s.text(sx + 26, sy + 5, lab, 14, th.fg, bold=True, anchor="start")
    s.line(470, 200, 548, 344, th.muted, 1.1, dash="4,4")
    s.text(478, 288, "≥ 3 m", 13, th.muted, anchor="end")

    for mx, my, lab in ((392.0, 186.0, "M1"), (392.0, 296.0, "M2"),
                        (446.0, 386.0, "M3")):
        s.circle(mx, my, 7, th.fg)
        s.circle(mx, my, 2.6, th.bg)
        s.text(mx - 12, my + 5, lab, 14, th.fg, bold=True, anchor="end")
    s.line(392, 186, 392, 296, th.muted, 1.1, dash="4,4")
    s.text(400, 246, "≥ 1.5 m", 13, th.muted, anchor="start")
    s.text(400, 480,
           "microphones ≥ 1.5 m apart, ≥ 2 m from a source, ≥ 1 m from any "
           "surface and from the specimen", 13, th.muted)

    # --- Right column: the two states the whole method rests on -----------
    s.rect(628, 82, 244, 342, th.bg, th.muted, rx=8, sw=1.6)
    s.text(750, 114, "The measurement is a difference", 14, th.fg, bold=True)
    for top, title, note, col in (
        (146.0, "1 · empty room", "$T_1$  →  $A_1$", th.primary),
        (274.0, "2 · specimen installed", "$T_2$  →  $A_2$", th.secondary),
    ):
        s.text(750, top, title, 14, col, bold=True)
        s.path(f"M 656 {top + 14} L 842 {top + 8} L 848 {top + 78} "
               f"L 660 {top + 84} Z", fill=th.panel, stroke=th.fg, sw=1.8)
        if top > 200:
            s.rect(690, top + 34, 92, 26, th.bg, th.secondary, sw=1.8)
            for hx in range(696, 782, 12):
                s.line(hx, top + 57, hx + 14, top + 37, th.secondary, 0.9)
        s.text(750, top + 106, note, 16, col, bold=True)
    s.text(750, 412, "$α_s = (A_2 − A_1) / S$", 16, th.accent, bold=True)

    # --- Mounting strip: Type A on the floor, and a Type E air space ------
    s.text(60, 524, "Annex B mounting (part of the result)", 15, th.fg,
           bold=True, anchor="start")
    for x0, lab, gap in ((80.0, "Type A: directly on the rigid floor", 0.0),
                         (470.0, "Type E-400: 400 mm face to floor", 30.0)):
        base = 590.0
        s.line(x0 - 14, base, x0 + 224, base, th.fg, 3.0)
        s.rect(x0 + 30, base - 16 - gap, 150, 16, th.bg, th.secondary, sw=1.8)
        for hx in range(int(x0) + 36, int(x0) + 176, 12):
            s.line(hx, base - 2 - gap, hx + 10, base - 15 - gap, th.secondary, 0.9)
        if gap:
            s.dim(x0 + 196, base, x0 + 196, base - 16 - gap, "400 mm",
                  offset=26, size=12, label_side="right")
        else:
            s.rect(x0 + 16, base - 16, 14, 16, th.fg)
            s.rect(x0 + 180, base - 16, 14, 16, th.fg)
            s.text(x0 + 105, base - 28, "perimeter frame, flush", 12, th.muted)
        s.text(x0 + 105, base + 24, lab, 13, th.fg)

    # --- Governing relations and acceptance checks ------------------------
    for y, txt, col, bold in (
        (646, "$A = 55.3 V/(c T) − 4 V m$   ·   $c = 331 + 0.6 t$  (15–30 °C)",
         th.fg, True),
        (672, ("≥ 12 spatially independent decays = ≥ 3 microphones × ≥ 2 sources "
               "· $T_{20}$ read from −5 dB over 20 dB"), th.muted, False),
        (696, ("the empty-room $A_1$ must clear the Table 1 ceiling, and "
               "$T_1$ is measured without the specimen frame"), th.muted,
         False),
    ):
        s.text(450, y, txt, 15 if bold else 13, col, bold=bold)


# ---------------------------------------------------------------------------
# d25 - ISO 10534-1 standing-wave-ratio apparatus
# ---------------------------------------------------------------------------

def _d_standing_wave_tube(s: SVG, th: Theme) -> None:
    """ISO 10534-1 standing-wave apparatus: probe carriage and the minima."""
    import math

    tube_top, tube_bot, mid = 216.0, 346.0, 281.0
    tube_l, tube_r = 156.0, 838.0
    back_w, spec_w = 22.0, 46.0
    face = tube_r - back_w - spec_w              # the specimen face: x = 0

    # --- Tube, source and specimen ---------------------------------------
    s.rect(tube_l, tube_top, tube_r - tube_l, tube_bot - tube_top, th.bg,
           th.fg, sw=3)
    s.rect(58, mid - 44, 66, 88, th.panel, th.primary, rx=6, sw=2)
    s.path(f"M 124 {mid - 17} L 124 {mid + 17} L {tube_l} {tube_bot} "
           f"L {tube_l} {tube_top} Z", fill=th.panel, stroke=th.primary, sw=2)
    s.circle(90, mid, 11, th.primary)
    s.text(92, 180, "Loudspeaker", 16, th.fg, bold=True)
    s.text(92, 202, "one pure tone at a time", 13, th.muted)

    s.rect(tube_r - back_w, tube_top, back_w, tube_bot - tube_top, th.fg)
    s.rect(face, tube_top, spec_w, tube_bot - tube_top, th.panel, th.secondary,
           sw=2)
    for hx in range(int(face) + 8, int(face + spec_w), 11):
        s.line(hx, tube_bot - 4, hx - 15, tube_top + 4, th.secondary, 1.0)
    s.text(tube_r, 202, "Test specimen on the rigid backing", 15,
           th.secondary, bold=True, anchor="end")
    s.line(face, 208, face, tube_bot + 14, th.accent, 1.6, dash="5,4")
    s.text(face + 6, tube_bot + 26, "$x = 0$", 14, th.accent, bold=True,
           anchor="start")

    # --- Graduated rail and the probe carriage ----------------------------
    rail_y, car_x = 150.0, 430.0
    s.line(tube_l + 24, rail_y, face, rail_y, th.fg, 2.4)
    x_tick = face
    while x_tick > tube_l + 24:
        s.line(x_tick, rail_y, x_tick, rail_y - 8, th.muted, 1.0)
        x_tick -= 26.0
    s.rect(car_x - 34, rail_y - 2, 68, 24, th.panel, th.primary, rx=5, sw=2)
    s.line(car_x, rail_y + 22, car_x, mid, th.fg, 2.4)
    s.circle(car_x, mid, 5.5, th.fg)
    s.text(car_x, rail_y - 20, "probe microphone on a graduated carriage", 14,
           th.fg, bold=True)
    s.arrow(car_x + 42, rail_y + 10, car_x + 116, rail_y + 10, th.accent, 2.0)
    s.arrow(car_x - 42, rail_y + 10, car_x - 116, rail_y + 10, th.accent, 2.0)

    # --- The standing-wave envelope inside the tube, filling in leftwards -
    span = (tube_bot - tube_top) / 2.0 - 8.0
    wavelength = 214.0                       # px per acoustic wavelength
    phi = math.radians(-54.1)

    def envelope(x: float) -> float:
        d = face - x
        r_eff = 0.5 * math.exp(-0.0014 * d)   # wall losses, exaggerated
        return math.sqrt(1.0 + r_eff**2
                         + 2.0 * r_eff * math.cos(2.0 * math.pi * d / wavelength
                                                  - phi))

    def y_of(env: float) -> float:
        return mid - (env - 1.0) * span / 0.62

    pts = [(x, y_of(envelope(x)))
           for x in [face - 3.0 * i for i in range(int((face - tube_l - 8) / 3))]]
    s.path("M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts),
           stroke=th.primary, sw=2.4)
    s.text(250, tube_top - 12, "$|p(x)|$ envelope", 14, th.primary)

    # The adjacent maximum and minimum the operator reads.
    x_min1 = face - wavelength * (phi + math.pi) / (2 * math.pi)
    x_max1 = x_min1 - wavelength / 2.0
    for px, lab in ((x_max1, "$L_{max}$"), (x_min1, "$L_{min}$")):
        py = y_of(envelope(px))
        s.circle(px, py, 5.5, th.secondary)
        s.text(px, py - 14, lab, 13, th.secondary, bold=True)
    s.dim(x_max1 - 46, y_of(envelope(x_max1)), x_max1 - 46, y_of(envelope(x_min1)),
          "$ΔL$ = 9.54 dB", offset=0, size=14, label_side="left")
    s.line(x_max1 - 52, y_of(envelope(x_max1)), x_max1, y_of(envelope(x_max1)),
           th.muted, 0.9, dash="3,3")
    s.line(x_max1 - 52, y_of(envelope(x_min1)), x_min1, y_of(envelope(x_min1)),
           th.muted, 0.9, dash="3,3")
    s.dim(x_min1, tube_bot + 14, face, tube_bot + 14, "$x_{min,1}$ = 12 cm",
          offset=46, size=14)
    s.text(340, tube_bot + 104,
           "minima far from the specimen fill in (wall losses, exaggerated "
           "here): read the nearest one", 13, th.muted)

    # --- The reduction chain, verbatim from the guide ---------------------
    for i, txt in enumerate((
        "$s = 10^{ΔL/20}$ = 3",
        "$|r| = (s − 1)/(s + 1)$ = 0.5",
        "$α = 1 − |r|^2$ = 0.75",
        "$Φ = 4π x_{min,1}/λ − π$ = −54.1°",
        "$Z/ρc_0 = (1 + r)/(1 − r)$ = 1.13 − 1.22j",
    )):
        s.text(450, 490 + 24 * i, txt, 15, th.fg)
    s.text(450, 622,
           "one channel: the microphone sensitivity cancels and there is no "
           "inter-channel phase error", 15, th.accent, bold=True)
    s.text(450, 648,
           "magnitude from the ratio, phase from the position — which is why "
           "Part 1 is the arbitration method", 13, th.muted)
