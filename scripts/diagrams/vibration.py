#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Diagrams of the vibration guides: structural paths and human exposure.

One subject: motion rather than pressure. The structural diagrams draw the
rigs that quantify how vibration enters, crosses and leaves a structure
(mobility, transfer stiffness, junctions), and the human diagrams draw the
measurement chains that judge the same motion as something a person is
exposed to.
"""

from __future__ import annotations

from .canvas import SVG, Theme
from .parts import (
    _accel,
    _accel_wall,
    _exciter,
    _motion_arrows,
    _plate_top,
    _plate_up,
    _spring_v,
)


def _d_human_vibration(s: SVG, th: Theme) -> None:
    """Whole-body vibration measurement chain (ISO 2631-1 / ISO 8041-1)."""
    gy = 510.0
    # --- Left: a seated person on a vibrating seat, triaxial accelerometer ---
    s.ground(gy, 40, 350)
    # Seat: cushion, backrest and support leg.
    s.rect(118, 424, 132, 18, th.panel, th.fg, rx=4, sw=2)      # cushion
    s.rect(118, 336, 16, 90, th.panel, th.fg, rx=3, sw=2)       # backrest
    s.line(184, 442, 184, gy, th.fg, 2.4)                       # pedestal
    # A wavy "vibration" arrow rising into the seat base.
    s.arrow(184, gy - 4, 184, 452, th.secondary, 2.4)
    s.text(184, gy - 12, "vibration input", 17, th.secondary, "middle", italic=True)
    s.person(178, gy, 176, seated=True)
    # Triaxial accelerometer at the seat/body interface with its x, y, z axes.
    ox, oy = 176.0, 420.0
    s.rect(ox - 9, oy - 8, 18, 16, th.secondary, th.fg, rx=2, sw=1.5)
    s.arrow(ox, oy - 8, ox, oy - 58, th.accent, 2.0)            # z (vertical)
    s.text(ox + 8, oy - 54, "z", 18, th.accent, "start", bold=True)
    s.arrow(ox + 9, oy, ox + 62, oy, th.accent, 2.0)            # x (fore-aft)
    s.text(ox + 66, oy + 5, "x", 18, th.accent, "start", bold=True)
    s.arrow(ox - 7, oy + 6, ox - 44, oy + 34, th.accent, 2.0)   # y (lateral)
    s.text(ox - 52, oy + 44, "y", 18, th.accent, "end", bold=True)
    s.text(150, gy + 34, "Seat/body interface", 18, th.fg, "middle")

    # --- Right: the vertical signal-processing chain ---
    cx, bw, bh = 650.0, 320.0, 72.0
    x0 = cx - bw / 2
    chain = [
        (96.0, "Triaxial accelerometer", "a_x , a_y , a_z  (m/s²)"),
        (206.0, "Band limiting + Wk / Wd", "weighting (ISO 8041-1)"),
        (316.0, "Weighted r.m.s. a_w  &  VDV", "(ISO 2631-1)"),
    ]
    for by, l1, l2 in chain:
        s.rect(x0, by, bw, bh, th.panel, th.primary, rx=12, sw=2)
        s.text(cx, by + 31, l1, 21, th.fg, "middle", bold=True)
        s.text(cx, by + 56, l2, 18, th.muted, "middle")
    s.arrow(cx, 168, cx, 206, th.fg, 2.0)
    s.arrow(cx, 278, cx, 316, th.fg, 2.0)
    # Feed the setup into the chain.
    s.arrow(252, oy, x0 - 6, 132, th.fg, 2.0)

    # --- Bottom: dominant axis, daily exposure and the Directive assessment ---
    # The Directive's whole-body A(8) is based on the HIGHEST frequency-
    # weighted axis value (1,4 a_wx, 1,4 a_wy, a_wz), Annex Part B point 1 -
    # not on the ISO 2631-1 Eq. (10) vector total a_v.
    s.arrow(cx, 388, cx, 424, th.fg, 2.0)
    s.rect(400, 424, 470, 78, "none", th.secondary, rx=12, sw=2, dash="6,5")
    s.text(635, 452, "A(8) = max(1.4·a_wx , 1.4·a_wy , a_wz)·√(T/T₀)",
           20, th.fg, "middle", bold=True)
    s.text(635, 480, "assessed vs EAV / ELV (Directive 2002/44/EC)",
           18, th.secondary, "middle")


def _d_multiple_shock(s: SVG, th: Theme) -> None:
    """Multiple-shock spinal-response dose and injury risk (ISO 2631-5:2018)."""
    cx = 450.0
    bw, bh = 660.0, 58.0
    x0 = cx - bw / 2

    # --- Input --------------------------------------------------------------
    s.rect(x0, 48, bw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 72, "Vertical seat acceleration  az(t)", 19, th.fg, "middle",
           bold=True)
    s.text(cx, 92,
           "conditioned per 5.1.3:  HP 0.01 Hz (2nd order) / LP 80 Hz "
           "(4th order)", 13, th.muted, "middle")
    s.arrow(cx, 106, cx, 136, th.fg, 1.8)
    s.text(cx - 26, 128, "not the ISO 2631-1 0.4 Hz / 100 Hz filters", 12,
           th.secondary, "end")

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 25, l1, 17, th.fg, "middle", bold=True)
        s.text(cx, y + 45, l2, 13, th.muted, "middle")

    _step(136, "Spinal response  Az(t)  (clause 5.2, Formula 1/2)",
          "seat-to-spine transfer function H(f): 1 zero, 6 poles", th.primary)
    _step(224, "Acceleration dose  Dz = 1.07·(Σ Az,i^6)^(1/6)  (Formula 3)",
          "Az,i = positive peaks;   daily dose Dzd = Dz·(td/tm)^(1/6)", th.fg)
    _step(312, "Compressive stress  Sd = mz·Dzd  (Annex C, Formula C.1)",
          "mz = 0.029 (male) / 0.025 (female) MPa per m/s²", th.fg)
    _step(400, "Stress variable  R = [Σ (Sd·N^(1/6) / (Su − Sstat))^6]^(1/6)",
          "Su = 6.75 − Sage·(b+i) MPa, cumulated over exposure years (C.3/C.4)",
          th.secondary)
    for y0, y1 in ((196, 224), (284, 312), (372, 400), (460, 488)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)

    # --- Output -------------------------------------------------------------
    s.rect(x0, 488, bw, 58, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, 513, "Injury probability  P(R) = 1 − exp(−(R/α)^β)  (Formula C.5)",
           17, th.fg, "middle", bold=True)
    s.text(cx, 533, "Weibull risk of lumbar injury, by sex (Table C.1/C.2)", 13,
           th.muted, "middle")


# ---------------------------------------------------------------------------
# Mechanical-mobility rig (ISO 7626)
# ---------------------------------------------------------------------------

def _d_mobility_rig(s: SVG, th: Theme) -> None:
    """ISO 7626 rig: free-free beam, exciter + impedance head at the driving
    point, accelerometer at a transfer point, impact-hammer variant."""
    cy_top, beam_top, beam_h = 116.0, 286.0, 26.0
    beam_bot = beam_top + beam_h
    # Ceiling with soft suspension.
    s.line(150, cy_top, 730, cy_top, th.fg, 2.2)
    for hx in range(162, 730, 26):
        s.line(hx, cy_top, hx - 9, cy_top - 9, th.muted, 1.1)
    for sx in (168.0, 712.0):
        _spring_v(s, sx, cy_top, beam_top, th.muted, coils=3, width=8.0,
                  sw=1.6)
    s.text(196, 142, "soft elastic suspension", 15, th.muted, anchor="start")

    # Beam under test.
    s.rect(150, beam_top, 580, beam_h, th.panel, th.fg, sw=2.2)
    s.text(470, beam_bot + 32, "Structure under test (free-free beam)", 19,
           th.fg, bold=True)

    # Driving point: exciter below the beam through an impedance head.
    dx = 248.0
    s.rect(dx - 14, beam_bot, 28, 16, th.secondary, th.fg, rx=3, sw=1.6)
    s.line(dx, beam_bot + 16, dx, 412, th.fg, 2.2)
    s.rect(dx - 37, 412, 74, 48, th.panel, th.primary, rx=9, sw=2)
    s.text(dx, 486, "Exciter", 18, th.fg, bold=True)
    s.text(60, 380, "Impedance head", 16, th.fg, anchor="start", bold=True)
    s.text(60, 402, "F and a at the drive point", 14, th.muted,
           anchor="start")
    s.line(dx - 16, beam_bot + 8, 154, 362, th.muted, 1.1, dash="3,3")
    s.arrow(dx + 18, 396, dx + 18, 340, th.secondary, 2.2)
    s.text(dx + 28, 372, "Fi", 16, th.secondary, anchor="start", mono=True)
    s.arrow(dx, beam_top - 4, dx, beam_top - 46, th.accent, 2.2)
    s.text(dx - 14, beam_top - 34, "vi", 16, th.accent, anchor="end",
           mono=True)
    s.text(210, 218, "driving point:  Yii = vi / Fi", 16, th.fg,
           anchor="start", mono=True)

    # Transfer point: accelerometer further along the beam.
    tx = 430.0
    _accel(s, tx, beam_top)
    s.arrow(tx, beam_top - 28, tx, beam_top - 56, th.accent, 2.2)
    s.text(tx + 12, beam_top - 40, "vj", 16, th.accent, anchor="start",
           mono=True)
    s.text(tx + 60, 192, "transfer:  Yji = vj / Fi", 16, th.fg, mono=True)

    # Impact-hammer variant striking the beam.
    hx2 = 600.0
    s.line(hx2 + 60, 172, hx2 + 6, 244, th.fg, 2.4)
    s.rect(hx2 - 16, 238, 32, 20, th.panel, th.fg, rx=4, sw=2)
    s.arrow(hx2, 262, hx2, beam_top - 6, th.secondary, 1.8)

    # FRF family footer.
    s.text(450, 520,
           "Y(f) = v/F  [m/(N·s)] · attached exciter (Part 2) · impact hammer (Part 5)",
           17, th.fg, mono=True)
    s.text(450, 546,
           "same measurement, three FRFs: x/F receptance · v/F mobility · a/F accelerance",
           15, th.muted)


# ---------------------------------------------------------------------------
# Dynamic transfer stiffness of resilient elements (ISO 10846)
# ---------------------------------------------------------------------------

def _d_transfer_stiffness_rig(s: SVG, th: Theme) -> None:
    """ISO 10846: isolator between the driven input mass and a blocked output
    (direct, force transducer) or a blocking mass (indirect)."""
    for cx, head in ((250.0, "Direct method (Part 2)"),
                     (650.0, "Indirect method (Part 3)")):
        s.text(cx, 78, head, 22, th.fg, bold=True)
        _exciter(s, cx, 158.0, stinger=18.0)
        # Driven input mass with its input displacement u1.
        s.rect(cx - 80, 158, 160, 44, th.panel, th.fg, rx=6, sw=2.2)
        s.text(cx, 186, "excitation mass", 16, th.fg)
        _motion_arrows(s, cx - 96, 180, 24, th.secondary)
        s.text(cx - 110, 186, "u₁", 18, th.secondary, anchor="end", mono=True)
        # Isolator under test.
        _spring_v(s, cx, 202, 310, th.accent, coils=4)
        s.text(cx + 28, 260, "isolator under test", 16, th.accent,
               anchor="start")

    # ===== Direct output: blocked, force transducer on a rigid foundation ===
    cx = 250.0
    s.rect(cx - 30, 310, 60, 18, th.secondary, th.fg, rx=3, sw=1.6)
    s.text(cx + 40, 322, "force transducer", 15, th.secondary, anchor="start")
    s.rect(cx - 105, 328, 210, 26, th.panel, th.fg, sw=2)
    s.ground(354, cx - 125, cx + 125)
    s.text(cx, 388, "Rigid foundation", 15, th.muted)
    s.text(cx, 470, "output blocked:  u₂ ≈ 0 → measure F₂,b", 16, th.fg,
           mono=True)
    s.text(cx, 500, "k₂,₁ = F₂,b / u₁", 20, th.primary, bold=True, mono=True)

    # ===== Indirect output: blocking mass on soft supports ==================
    cx = 650.0
    s.rect(cx - 85, 310, 170, 60, th.panel, th.fg, rx=6, sw=2.4)
    s.text(cx, 346, "blocking mass m₂", 17, th.fg)
    _accel(s, cx + 55, 310)
    s.text(cx + 72, 296, "a₂", 15, th.secondary, anchor="start", mono=True)
    for sx in (cx - 50.0, cx + 50.0):
        _spring_v(s, sx, 370, 430, th.muted, coils=3, width=8.0, sw=1.6)
    s.ground(430, cx - 115, cx + 115)
    s.text(cx + 70, 408, "soft support", 14, th.muted, anchor="start")
    s.text(cx, 470, "measure T = u₂ / u₁  (small)", 16, th.fg, mono=True)
    s.text(cx, 500, "k₂,₁ = −(2πf)²·(m₂+mf)·T", 20, th.primary, bold=True,
           mono=True)

    # Validity footer (Part 3 clause 6, Part 1 Eq. 7).
    s.text(450, 556,
           "valid where ΔL₁,₂ = La₁ − La₂ ≥ 20 dB, i.e. |T| ≤ 0.1   (Part 3, Inequality 2)",
           17, th.muted)
    s.text(450, 582,
           "the blocking force approximates the force delivered to a stiff receiver (Part 1, Eq. 7)",
           15, th.muted, italic=True)


# ---------------------------------------------------------------------------
# Junction vibration measurement, L- and T-junctions (ISO 10848)
# ---------------------------------------------------------------------------

def _d_junction_rig(s: SVG, th: Theme) -> None:
    """ISO 10848 junction rig: an L- and a T-junction of concrete plates,
    structure-borne excitation on element i, accelerometers on i and j and
    the junction length l_ij along the corner line."""
    gy = 430.0
    dp = 170.0
    dxo, dyo = dp * 0.72, dp * 0.55

    # ===== Left: L-junction (wall on the left end of the floor plate) =====
    s.text(280, 86, "L-junction", 21, th.fg, bold=True)
    _plate_top(s, th, 140, gy, 230, dp, 16)
    _plate_up(s, th, 140, gy, 16, 180, dp)
    # Junction line along the corner, highlighted, with its length label.
    s.line(156, gy, 156 + dxo, gy - dyo, th.accent, 2.6)
    s.text(58, 474, "lij ≥ 2.3 m", 17, th.fg, anchor="start")
    s.line(126, 466, 152, 438, th.muted, 1.0)

    # Exciter on the floor (element i), accelerometers on i and j.
    _exciter(s, 330, 396)
    _accel(s, 250, 410)
    _accel(s, 380, 380)
    _accel_wall(s, 205, 300)
    _accel_wall(s, 236, 262)
    s.text(196, 420, "i", 22, th.primary, bold=True, italic=True)
    s.text(178, 200, "j", 22, th.secondary, bold=True, italic=True)
    # Transmission path across the corner.
    s.path("M 300 402 Q 214 400 208 330", stroke=th.accent, sw=2.0)
    s.arrow(209.0, 344.0, 208.0, 322.0, th.accent, 2.0)
    s.text(194, 356, "Dv,ij", 16, th.accent, anchor="end", mono=True)

    # ===== Right: T-junction (wall standing mid-way on the floor) =========
    s.text(690, 86, "T-junction", 21, th.fg, bold=True)
    _plate_top(s, th, 520, gy, 220, dp, 16)
    _plate_up(s, th, 620, gy, 16, 180, dp)
    s.line(636, gy, 636 + dxo, gy - dyo, th.accent, 2.6)
    _exciter(s, 566, 404)
    _accel(s, 588, 422)
    _accel_wall(s, 685, 290)
    _accel(s, 762, 384)
    s.text(533, 423, "i", 22, th.primary, bold=True, italic=True)
    s.text(658, 200, "j", 22, th.secondary, bold=True, italic=True)
    s.text(806, 400, "j", 22, th.secondary, bold=True, italic=True)
    s.path("M 612 418 Q 690 434 756 400", stroke=th.accent, sw=2.0)
    s.arrow(742.0, 407.0, 760.0, 398.0, th.accent, 2.0)
    s.path("M 606 406 Q 646 394 654 330", stroke=th.accent, sw=2.0)
    s.arrow(655.0, 344.0, 654.0, 322.0, th.accent, 2.0)

    # Exciter label shared by both panels.
    s.text(450, 250, "Shaker or hammer on element i", 17, th.fg)
    s.line(376, 258, 348, 322, th.muted, 1.0)
    s.line(524, 258, 556, 348, th.muted, 1.0)

    # Plate thickness leader (the lines stop above the caption text).
    s.text(450, 496, "concrete plates 140 mm to 200 mm thick", 17, th.muted)
    s.line(322, 477, 300, 442, th.muted, 1.0)
    s.line(578, 477, 600, 442, th.muted, 1.0)

    # Normative relations.
    s.text(80, 536, "lij ≥ 2.3 m along the junction; element sizes 3.0 m ≤ li < 6.0 m",
           18, th.fg, anchor="start")
    s.text(80, 564, "≥ 4 excitation positions on i; accelerometers ≥ 0.25 m from edges, ≥ 0.5 m apart",
           18, th.fg, anchor="start")
    s.text(80, 596, "Kij = D̄v,ij + 10 log10( lij / √(ai·aj) ),   ai = equivalent absorption length",
           17, th.primary, anchor="start", bold=True, mono=True)
