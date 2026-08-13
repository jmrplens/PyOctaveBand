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
    _rot_arrow,
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
    s.text(ox + 8, oy - 54, "$z$", 18, th.accent, "start", bold=True)
    s.arrow(ox + 9, oy, ox + 62, oy, th.accent, 2.0)            # x (fore-aft)
    s.text(ox + 66, oy + 5, "$x$", 18, th.accent, "start", bold=True)
    s.arrow(ox - 7, oy + 6, ox - 44, oy + 34, th.accent, 2.0)   # y (lateral)
    s.text(ox - 52, oy + 44, "$y$", 18, th.accent, "end", bold=True)
    s.text(150, gy + 34, "Seat/body interface", 18, th.fg, "middle")

    # --- Right: the vertical signal-processing chain ---
    cx, bw, bh = 650.0, 320.0, 72.0
    x0 = cx - bw / 2
    chain = [
        (96.0, "Triaxial accelerometer", "$a_x , a_y , a_z$  (m/s²)"),
        (206.0, "Band limiting + Wk / Wd", "weighting (ISO 8041-1)"),
        (316.0, "Weighted r.m.s. $a_w$  &  VDV", "(ISO 2631-1)"),
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
    s.text(635, 452, "$A(8) = max(1.4·a_{wx} , 1.4·a_{wy} , a_{wz})·√(T/T_0)$",
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
    s.text(cx, 72, "Vertical seat acceleration  $a_{z}(t)$", 19, th.fg,
           "middle", bold=True)
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

    _step(136, "Spinal response  $A_{z}(t)$  (clause 5.2, Formula 1/2)",
          "seat-to-spine transfer function $H(f)$: 1 zero, 6 poles",
          th.primary)
    _step(224, "Acceleration dose  $D_z = 1.07·(Σ A_{z,i}^6)^{1/6}$  "
          "(Formula 3)",
          "$A_{z,i}$ = positive peaks;   daily dose "
          "$D_{zd} = D_z·(t_d/t_m)^{1/6}$", th.fg)
    _step(312, "Compressive stress  $S_d = m_z·D_{zd}$  (Annex C, "
          "Formula C.1)",
          "$m_z$ = 0.029 (male) / 0.025 (female) MPa per m/s²", th.fg)
    # PARKED (controller adjudication): ISO 2631-5:2018(E) prints the
    # descriptive subscripts of S_stat and S_age in roman (PDF page 24,
    # folio 18: S_stat,i and S_age beside italic indices), but "stat" and
    # "age" are not in _ROMAN_SCRIPTS and the sibling descriptors u/d of
    # S_u/S_d are single letters the list cannot take; composing part of
    # the pair would split one formula into two subscript styles.
    _step(400, "Stress variable  R = [Σ (Sd·N^(1/6) / (Su − Sstat))^6]^(1/6)",
          "Su = 6.75 − Sage·(b+i) MPa, cumulated over exposure years (C.3/C.4)",
          th.secondary)
    for y0, y1 in ((196, 224), (284, 312), (372, 400), (460, 488)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)

    # --- Output -------------------------------------------------------------
    s.rect(x0, 488, bw, 58, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, 513, "Injury probability  $P(R) = 1 − exp(−(R/α)^β)$  "
           "(Formula C.5)", 17, th.fg, "middle", bold=True)
    s.text(cx, 533, "Weibull risk of lumbar injury, by sex (Table C.1/C.2)", 13,
           th.muted, "middle")


def _d_hand_arm_vibration(s: SVG, th: Theme) -> None:
    """Where the accelerometer goes on a tool, and what the reading becomes.

    ISO 5349-2 clause 6.1.3 (location), 6.1.4/6.1.5 (mounting and mass),
    6.2.3 (cable) and clause 8 (the daily exposure), on the chain-saw front
    handle the guide's worked example is measured at.
    """
    # --- Left: the tool handle, the hand and the transducer positions -------
    s.text(258, 74, "On the tool (ISO 5349-2, 6.1.3)", 18, th.fg, "middle",
           bold=True)
    bar_y, bar_h = 200.0, 26.0                  # a Ø 30 mm handle tube
    bar_top, bar_bot = bar_y - bar_h / 2, bar_y + bar_h / 2
    s.rect(60, bar_top, 400, bar_h, th.panel, th.fg, rx=13, sw=2)

    # The hand: palm above the tube, four fingers curling under it.
    hand_x0, hand_x1 = 230.0, 330.0
    s.rect(hand_x0, 148, hand_x1 - hand_x0, bar_y - 148, th.muted, th.fg,
           rx=14, sw=1.6)
    for fx in (236.0, 259.0, 282.0, 305.0):
        s.rect(fx, bar_bot + 4, 19, 34, th.muted, th.fg, rx=9, sw=1.4)
    s.line(254, 156, 188, 92, th.muted, 15)
    s.dim(hand_x0, 272, hand_x1, 272, "gripping zone ≈ 100 mm", size=14)

    # The transducer positions of 6.1.3, in the order the clause prefers them.
    def _pos(x: float, under: bool, tag: str, cy: float) -> None:
        top = bar_bot if under else bar_top - 19
        s.rect(x - 12, top, 24, 19, th.secondary, th.fg, rx=3, sw=1.5)
        s.circle(x, cy, 12, th.secondary, th.bg, 2)
        s.text(x, cy + 5, tag, 16, th.bg, "middle", bold=True)

    _pos(280, True, "1", 174)
    _pos(196, False, "2", 148)
    _pos(372, False, "2", 148)
    _pos(372, True, "3", 253)

    s.text(520, 100, "chain-saw front handle,", 15, th.muted, "end")
    s.text(520, 118, "Ø 30 mm tube", 15, th.muted, "end")
    s.line(455, 126, 455, bar_top - 4, th.muted, 1.2, dash="4,4")

    # The cable, taped to the vibrating surface near the transducer (6.2.3).
    s.path(f"M 378 {bar_bot + 19} L 432 246 L 432 306 L 110 306",
           stroke=th.primary, sw=2.0)
    for tx in (170.0, 250.0):
        s.rect(tx, 300, 12, 12, th.primary, th.fg, rx=2, sw=1.0)
    s.text(110, 330, "cable taped to the handle near the transducer (6.2.3)",
           14, th.primary, "start")

    # Grip force: one of the two applied forces clause 9 asks the report for.
    s.arrow(306, 110, 306, 146, th.accent, 2.2)
    s.arrow(210, 250, 210, 218, th.accent, 2.2)
    s.text(306, 102, "grip force", 15, th.accent, "middle")

    s.text(45, 366, "1  middle of the gripping zone, under the hand",
           15, th.fg, "start", bold=True)
    s.text(62, 385, "the most representative location; needs an adaptor",
           15, th.muted, "start")
    s.text(45, 410, "2  either side of the hand — usual practice", 15, th.fg,
           "start", bold=True)
    s.text(62, 429, "on a side handle, average the two positions", 15,
           th.muted, "start")
    s.text(45, 454, "3  underside of the handle, next to the hand", 15, th.fg,
           "start", bold=True)
    s.text(45, 480, "grip and push force change the reading: report the",
           14, th.muted, "start")
    s.text(45, 497, "posture and the applied forces (7.1, clause 9 g))", 14,
           th.muted, "start")

    # The basicentric frame the three axes are reported in.
    # PARKED (controller adjudication): ISO 5349-1:2001(E) prints the h
    # subscript of the basicentric axes in roman (Figure 1 and its NOTE,
    # PDF page 10, folio 4: italic x/y/z with roman h, like the roman
    # hw/hv the curated list already carries), but a lone "h" cannot enter
    # _ROMAN_SCRIPTS without setting every legitimate h index upright, and
    # the italic default would put the same letter in two styles beside
    # the composed a_hv/a_hwx chain of this very diagram.
    ox, oy = 110.0, 578.0
    s.arrow(ox, oy, ox + 74, oy, th.primary, 2.2)
    s.text(ox + 80, oy + 5, "y_h", 17, th.primary, "start", bold=True)
    s.arrow(ox, oy, ox, oy - 46, th.primary, 2.2)
    s.text(ox, oy - 54, "z_h", 17, th.primary, "middle", bold=True)
    s.arrow(ox, oy, ox - 40, oy + 28, th.primary, 2.2)
    s.text(ox - 46, oy + 40, "x_h", 17, th.primary, "end", bold=True)
    s.text(236, 534, "basicentric frame (ISO 5349-1 Fig. 1):",
           14, th.fg, "start")
    s.text(236, 552, "rotated so that y_h lies along the",
           14, th.fg, "start")
    s.text(236, 570, "handle axis. All three axes are",
           14, th.fg, "start")
    s.text(236, 588, "measured, and every $k = 1$.", 14, th.fg, "start")

    # --- Right: what the three axis magnitudes become -----------------------
    cx, bw, bh = 705.0, 320.0, 66.0
    x0 = cx - bw / 2
    chain = (
        (90.0, "$a_{hwx} , a_{hwy} , a_{hwz}$",
         "Wh-weighted, one per axis (ISO 5349-1 A.1)", th.primary),
        (176.0, "$a_{hv} = √(a_{hwx}^2 + a_{hwy}^2 + a_{hwz}^2)$",
         "vibration total value (Eq. (1))", th.fg),
        (262.0, "$A_{i}(8) = a_{hv,i} · √(T_i / T_0)$",
         "$T_i$ is total contact time per day (5.5)", th.fg),
        (348.0, "$A(8) = √(Σ A_{i}(8)^2)$",
         "one per hand, two significant figures (clause 8)", th.secondary),
    )
    for by, l1, l2, colour in chain:
        s.rect(x0, by, bw, bh, th.panel, colour, rx=12, sw=2)
        s.text(cx, by + 28, l1, 18, th.fg, "middle", bold=True)
        s.text(cx, by + 50, l2, 14, th.muted, "middle")
    for y0, y1 in ((156, 176), (242, 262), (328, 348), (414, 438)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)
    s.rect(x0, 438, bw, 62, "none", th.secondary, rx=12, sw=2.4, dash="6,5")
    s.text(cx, 464, "EAV 2.5 m/s²   ·   ELV 5 m/s²", 19, th.fg, "middle",
           bold=True)
    s.text(cx, 486, "Directive 2002/44/EC, Article 3", 14, th.muted, "middle")

    # --- Bottom: the four acquisition rules that decide the reading ---------
    s.rect(524, 518, 336, 130, th.panel, th.muted, rx=12, sw=1.6)
    rules = (
        "transducer and mount below 5 % of the",
        "mass they are fixed to (6.1.5)",
        "linear averaging over complete work",
        "cycles (6.1.11)",
        "three samples per operation, a minute",
        "of record, none under 8 s (5.4.1)",
        "the lowest input range that does not",
        "overload, found by trial (6.1.10)",
    )
    for i, rule in enumerate(rules):
        if i % 2 == 0:
            s.circle(542, 540 + 14 * i - 5, 3.4, th.secondary)
        s.text(556, 540 + 14 * i, rule, 14, th.fg, "start")


def _d_iso2631_5_setup(s: SVG, th: Theme) -> None:
    """Getting the record ISO 2631-5 works on (clauses 5.1.2 and 5.1.4)."""
    import math

    # --- Left: the seat pan in section, with the ISO 10326-1 disc on it -----
    s.text(250, 78, "The seat pan, in section", 18, th.fg, "middle", bold=True)
    s.rect(200, 105, 92, 76, th.muted, th.fg, rx=26, sw=1.6)      # pelvis
    s.ellipse(224, 188, 24, 13, th.muted, th.fg, 1.6)             # tuberosity
    s.ellipse(268, 188, 24, 13, th.muted, th.fg, 1.6)             # tuberosity
    s.rect(150, 200, 220, 40, th.panel, th.fg, rx=9, sw=2)        # cushion
    s.rect(150, 240, 220, 12, th.panel, th.fg, sw=2)              # seat pan
    s.rect(150, 116, 16, 84, th.panel, th.fg, rx=4, sw=2)         # backrest
    _spring_v(s, 262, 252, 300, th.fg)
    s.ground(300, 130, 396)
    s.text(262, 322, "suspension travel", 14, th.muted, "middle")

    # The ISO 10326-1 mounting disc, edge on, taped to the cushion.
    s.rect(206, 191, 80, 9, th.secondary, th.fg, rx=4, sw=1.5)
    s.rect(234, 192.5, 24, 6, th.panel, th.fg, rx=1.5, sw=1.0)
    s.arrow(300, 190, 300, 128, th.accent, 2.4)
    s.text(308, 138, "$z$ +", 17, th.accent, "start", bold=True)
    s.rect(186, 190, 15, 11, th.primary, th.fg, rx=2, sw=1.2)
    s.line(186, 196, 128, 196, th.primary, 1.8)
    s.circle(124, 196, 4, th.primary)

    notes = (
        (True, "semi-rigid mounting disc Ø 250 ± 50 mm, height ≤ 12 mm,"),
        (False, "80-90 durometer (A), carrying a Ø 75 ± 5 mm × 1.5 mm"),
        (False, "metal disc for the accelerometers (ISO 10326-1, 5.2.3)"),
        (True, "taped to the cushion so the accelerometers sit midway"),
        (False, "between the ischial tuberosities (5.1.2)"),
        (True, "$z$ is positive to cranial: the method is about"),
        (False, "compressive spinal loading (5.1.3, first step)"),
        (True, "a contact switch or video detects loss of contact,"),
        (False, "which is reported and excluded from the exposure"),
    )
    for i, (bullet, line) in enumerate(notes):
        y = 356 + 17 * i
        if bullet:
            s.circle(50, y - 5, 3.4, th.secondary)
        s.text(64, y, line, 14, th.fg, "start")

    # --- Right: what the record then looks like -----------------------------
    ax0, ax1, base = 500.0, 862.0, 306.0
    s.text(681, 78, "The record, split where contact is lost", 18, th.fg,
           "middle", bold=True)
    for i, line in enumerate((
            "accelerations recorded while contact is lost",
            "shall not be counted as exposure, and the",
            "landing impact after a free fall shall be (5.1.2)")):
        s.text(681, 110 + 18 * i, line, 14, th.secondary, "middle")
    gaps = ((0.30, 0.40), (0.66, 0.72))
    for lo, hi in gaps:
        s.rect(ax0 + lo * (ax1 - ax0), base - 92, (hi - lo) * (ax1 - ax0),
               184, th.panel, th.secondary, sw=1.4, dash="5,4")
    pts = []
    for i in range(0, 363, 2):
        u = i / 362.0
        v = 16.0 * math.sin(2.0 * math.pi * 5.0 * u) \
            + 6.0 * math.sin(2.0 * math.pi * 13.0 * u + 1.0)
        for centre, amp in ((0.17, 62.0), (0.52, 44.0), (0.88, 78.0)):
            d = u - centre
            if 0.0 <= d < 0.2:
                v += amp * math.exp(-46.0 * d) * math.sin(2.0 * math.pi * 26.0 * d)
        for lo, hi in gaps:
            if lo <= u < hi:
                v = -34.0
        pts.append(f"{ax0 + u * (ax1 - ax0):.1f} {base - v:.1f}")
    s.path("M " + " L ".join(pts), stroke=th.primary, sw=1.8)
    s.line(ax0, base, ax1, base, th.fg, 1.4)
    s.text(ax0 - 8, base + 5, "0", 15, th.muted, "end")
    for lo, hi in gaps:
        s.text(ax0 + 0.5 * (lo + hi) * (ax1 - ax0), base - 100,
               "no contact", 14, th.secondary, "middle")
    s.text(ax0, base + 118, "$a_{z}(t)$, conditioned per 5.1.3", 14, th.muted,
           "start")
    segments = ((0.0, 0.30, "segment 1"), (0.40, 0.66, "segment 2"),
                (0.72, 1.0, "segment 3"))
    for lo, hi, label in segments:
        xa, xb = ax0 + lo * (ax1 - ax0), ax0 + hi * (ax1 - ax0)
        s.line(xa, base + 138, xb, base + 138, th.accent, 3.0)
        s.text(0.5 * (xa + xb), base + 160, label, 14, th.accent, "middle")
    s.text(681, base + 186,
           "each segment is conditioned separately (5.1.3, second step)", 14,
           th.fg, "middle")

    # --- Bottom: the hardware and duration requirements ---------------------
    s.rect(40, 518, 820, 74, th.panel, th.muted, rx=12, sw=1.6)
    rules = (
        ("flat acceleration response from 0.01 Hz to at least 80 Hz, and "
         "256 samples per second or more (5.1.2)"),
        ("equipment adequate for the highest amplitude anticipated; "
         "equipment and calibration method reported (5.1.2)"),
        ("long enough to be representative: a complete work cycle for a "
         "repeatable task, longer where the terrain varies (5.1.4)"),
    )
    for i, rule in enumerate(rules):
        s.circle(62, 542 + 21 * i - 5, 3.4, th.secondary)
        s.text(76, 542 + 21 * i, rule, 14, th.fg, "start")


# ---------------------------------------------------------------------------
# Machine fault kinematics and the condition-monitoring rig (Norton 8.4)
# ---------------------------------------------------------------------------

def _d_fault_kinematics(s: SVG, th: Theme) -> None:
    """Where the fault frequencies come from: the bearing in end section and
    in axial half-section (where the contact angle lives), the gear pair and
    the ducted fan's rotating lobe pattern."""
    import math

    # ===== Panel 1: rolling-contact bearing, end section ====================
    cx, cy = 200.0, 262.0
    mm = 4.4                                    # 1 mm of the real bearing
    r_pitch, r_ball = 17.0 * mm, 3.0 * mm       # D = 34 mm, d = 6 mm
    r_out_i, r_out_o = r_pitch + r_ball, r_pitch + r_ball + 17.0
    r_in_o, r_in_bore = r_pitch - r_ball, r_pitch - r_ball - 26.0
    s.text(cx, 74, "1. Bearing: end section", 21, th.fg, bold=True)
    s.circle(cx, cy, r_out_o, th.panel, th.fg, 2.0)
    s.circle(cx, cy, r_out_i, "none", th.fg, 2.0)
    s.circle(cx, cy, r_in_o, th.panel, th.fg, 2.0)
    s.circle(cx, cy, r_in_bore, th.bg, th.fg, 2.0)
    s.circle(cx, cy, r_pitch, "none", th.muted, 1.1)
    for k in range(15):                          # Z = 15 rolling elements
        a = math.radians(k * 24.0 - 90.0)
        s.circle(cx + r_pitch * math.cos(a), cy + r_pitch * math.sin(a),
                 r_ball, th.bg, th.primary, 1.8)
    # The pitch diameter, dimensioned between two opposite element centres.
    s.arrow(cx - 6, cy, cx - r_pitch, cy, th.muted, 1.3)
    s.arrow(cx + 6, cy, cx + r_pitch, cy, th.muted, 1.3)
    s.rect(cx - 74, cy - 25, 148, 21, th.panel)
    s.text(cx, cy - 9, "$D$ = 34 mm (pitch)", 15, th.fg)
    # One element dimensioned, with a leader out of the drawing.
    bax = cx + r_pitch * math.cos(math.radians(38.0))
    bay = cy + r_pitch * math.sin(math.radians(38.0))
    s.line(bax + r_ball * 0.7, bay + r_ball * 0.7, 352, 352, th.muted, 1.0,
           dash="3,3")
    s.text(358, 358, "$d$ = 6 mm", 16, th.fg, anchor="start")
    # The spall on the stationary outer race, at the top of the load zone.
    for da in (-26.0, -13.0, 0.0, 13.0, 26.0):
        a = math.radians(-90.0 + da)
        s.line(cx + r_out_i * math.cos(a), cy + r_out_i * math.sin(a),
               cx + (r_out_i - 14.0) * math.cos(a),
               cy + (r_out_i - 14.0) * math.sin(a), th.secondary, 2.2)
    s.text(cx, 116, "spall on the outer race", 16, th.secondary)
    s.text(cx, 136, "BPFO = 207.0 Hz: one impact per pass", 14, th.secondary)
    # Which race turns, and the cage rate that follows from it.
    _rot_arrow(s, cx, cy, r_in_bore - 9.0, 20.0, 160.0, th.accent, 2.0)
    s.text(cx, cy + r_out_o + 34, "inner race turns at $f_s$ = 33.33 Hz,", 15,
           th.fg)
    s.text(cx, cy + r_out_o + 56, "outer race stationary", 15, th.fg)
    s.text(cx, cy + r_out_o + 82, "cage FTF = 13.8 Hz = 0.41 $f_s$", 15,
           th.accent)

    # ===== Panel 2: axial half-section, where the contact angle lives =======
    ax0, axy = 560.0, 250.0
    s.text(690, 74, "2. Bearing: axial half-section", 21, th.fg, bold=True)
    s.line(ax0 - 40, axy + 128, 880, axy + 128, th.muted, 1.4, dash="14,5,3,5")
    s.text(ax0 - 36, axy + 148, "bearing axis", 14, th.muted, anchor="start")
    # Outer ring, ball and inner ring in section, stacked off the axis.
    s.rect(ax0 + 40, axy - 92, 220, 44, th.panel, th.fg, sw=2.0)
    s.text(ax0 + 92, axy - 64, "outer ring", 15, th.fg)
    s.rect(ax0 + 40, axy + 48, 220, 44, th.panel, th.fg, sw=2.0)
    s.text(ax0 + 92, axy + 76, "inner ring", 15, th.fg)
    bx, by = ax0 + 168.0, axy
    s.circle(bx, by, 36, th.bg, th.primary, 2.0)
    # The radial plane (perpendicular to the axis) and the contact line.
    s.line(bx, by - 76, bx, by + 76, th.muted, 1.2, dash="5,4")
    s.text(bx - 10, by + 108, "radial plane", 14, th.muted, anchor="end")
    phi = math.radians(12.96)
    dx, dy = 74.0 * math.sin(phi), 74.0 * math.cos(phi)
    s.line(bx - dx, by + dy, bx + dx, by - dy, th.secondary, 2.4)
    s.text(690, 136, "contact line,  $φ$ = 12.96°", 16, th.secondary)
    s.path(f"M {bx:.1f} {by - 50:.1f} A 50 50 0 0 0 "
           f"{bx + 50 * math.sin(phi):.1f} {by - 50 * math.cos(phi):.1f}",
           stroke=th.secondary, sw=1.6)
    s.text(690, axy + 186,
           "the contact angle exists only in this view: it is measured",
           15, th.muted)
    s.text(690, axy + 206,
           "from the radial plane, so $φ = 0$ for a deep-groove ball bearing",
           15, th.muted)
    s.text(690, axy + 226,
           "and $φ > 0$ for angular-contact and tapered-roller types", 15,
           th.muted)

    # ===== Panel 3: gear pair in elevation ==================================
    gy = 596.0
    s.text(200, 500, "3. Gear pair", 21, th.fg, bold=True)
    for gcx, gr, teeth in ((110.0, 52.0, 28), (240.0, 78.0, 42)):
        s.circle(gcx, gy, gr, th.panel, th.fg, 2.0)
        s.circle(gcx, gy, gr * 0.22, th.bg, th.fg, 1.6)
        for k in range(teeth):
            a = 2.0 * math.pi * k / teeth
            s.line(gcx + gr * math.cos(a), gy + gr * math.sin(a),
                   gcx + (gr + 8) * math.cos(a), gy + (gr + 8) * math.sin(a),
                   th.fg, 1.6)
    # The chipped tooth, on the pinion, at the mesh line.
    s.circle(162.0, gy, 9.0, "none", th.secondary, 2.4)
    s.line(160, gy + 9, 108, 672, th.muted, 1.0, dash="3,3")
    s.text(64, 690, "chipped tooth", 15, th.secondary, anchor="start")
    s.text(336, 574, "28-tooth pinion on a", 15, th.fg, anchor="start")
    s.text(336, 594, "1500 r/min shaft:", 15, th.fg, anchor="start")
    s.text(336, 616, "$f_s$ = 25 Hz", 15, th.muted, anchor="start")

    # ===== Panel 4: ducted axial fan seen along the duct axis ===============
    fcx, fcy, fr = 690.0, 596.0, 80.0
    s.text(690, 500, "4. Ducted axial fan", 21, th.fg, bold=True)
    s.circle(fcx, fcy, fr, "none", th.fg, 2.4)
    s.circle(fcx, fcy, 20, th.panel, th.fg, 2.0)
    for k in range(4):                            # V = 4 stator vanes, behind
        a = math.radians(45.0 + k * 90.0)
        s.line(fcx + 22 * math.cos(a), fcy + 22 * math.sin(a),
               fcx + fr * math.cos(a), fcy + fr * math.sin(a),
               th.muted, 3.0, dash="7,5")
    for k in range(6):                            # N = 6 rotor blades, solid
        a = math.radians(k * 60.0)
        s.line(fcx + 20 * math.cos(a), fcy + 20 * math.sin(a),
               fcx + (fr - 6) * math.cos(a), fcy + (fr - 6) * math.sin(a),
               th.primary, 4.4)
    _rot_arrow(s, fcx, fcy, fr + 15.0, -62.0, 28.0, th.accent, 2.0)
    s.text(fcx + fr + 24, fcy + 6, "$N f_s$", 15, th.accent, anchor="start")
    s.text(690, fcy + 104, "6 blades (solid), 4 vanes (dashed)", 15, th.fg)

    # ===== What the two lower panels are for ================================
    s.text(450, 730, "$GMF = N f_s = 28 × 25$ = 700 Hz, and a chipped tooth "
           "modulates it once per revolution: sidebands at $± f_s$", 16,
           th.fg)
    s.text(450, 758, "$m_L = n·N ± k·V = 6 ± 4$ → 2 or 10 lobes, turning at "
           "$n·N·f_s/m_L$ = 175 or 35 Hz: the faster pattern radiates far "
           "more strongly", 16, th.fg)


def _d_machine_diagnostics(s: SVG, th: Theme) -> None:
    """Condition-monitoring measurement on a motor-gearbox train: where the
    accelerometers go, how they are mounted, and what the analyser records."""
    gy = 262.0                                   # top of the baseplate
    shaft_y = 222.0

    # --- The machine train on its baseplate ---------------------------------
    s.rect(110, gy, 460, 18, th.panel, th.fg, sw=2.0)
    s.ground(gy + 18, 110, 570)
    s.rect(130, 182, 136, 80, th.panel, th.fg, rx=6, sw=2.2)
    s.text(198, 228, "motor", 19, th.fg, bold=True)
    s.rect(390, 172, 148, 90, th.panel, th.fg, rx=6, sw=2.2)
    s.text(464, 224, "gearbox", 19, th.fg, bold=True)
    s.line(60, shaft_y, 130, shaft_y, th.fg, 3.0)          # non-drive end
    s.line(266, shaft_y, 390, shaft_y, th.fg, 3.0)         # drive shaft
    s.rect(316, shaft_y - 12, 26, 24, th.panel, th.fg, rx=3, sw=1.8)
    s.text(329, shaft_y + 34, "coupling", 14, th.muted)
    for hx in (282.0, 374.0):                              # bearing housings
        s.rect(hx - 16, shaft_y - 18, 32, 36, th.bg, th.fg, rx=4, sw=1.8)

    # --- Where the accelerometers go ---------------------------------------
    for hx in (282.0, 374.0):
        _accel(s, hx, shaft_y - 18)
    s.rect(538, 196, 15, 16, th.secondary, th.fg, rx=2.5, sw=1.3)
    s.line(553, 204, 566, 204, th.fg, 1.3)
    s.text(572, 209, "axial", 15, th.secondary, anchor="start")
    s.text(328, 322, "radial, in the load zone, on the housing itself: no "
           "joint between the bearing and the sensor", 15, th.muted)
    s.line(282, 300, 276, 262, th.muted, 1.0, dash="3,3")
    s.line(374, 300, 380, 262, th.muted, 1.0, dash="3,3")

    # --- The once-per-revolution pickup on the free shaft end ---------------
    s.rect(74, 146, 40, 28, th.panel, th.accent, rx=5, sw=2.0)
    s.arrow(94, 174, 94, 210, th.accent, 1.8)
    s.rect(86, shaft_y - 7, 16, 14, th.accent, th.fg, sw=1.0)
    s.text(122, 152, "once-per-revolution", 14, th.accent, anchor="start")
    s.text(122, 170, "pickup on a mark", 14, th.accent, anchor="start")

    # --- The cable bundle into the analyser --------------------------------
    s.line(94, 146, 94, 96, th.accent, 1.6)
    s.line(282, shaft_y - 40, 282, 96, th.secondary, 1.6)
    s.line(374, shaft_y - 40, 374, 96, th.secondary, 1.6)
    s.line(553, 204, 553, 96, th.secondary, 1.6)
    s.line(94, 96, 640, 96, th.fg, 1.8)
    s.arrow(640, 96, 690, 152, th.fg, 1.8)
    s.rect(628, 158, 246, 118, th.panel, th.primary, rx=12, sw=2.2)
    s.text(751, 190, "Analyser", 20, th.fg, bold=True)
    s.text(751, 218, "ch 1: acceleration", 16, th.fg)
    s.text(751, 242, "ch 2: tacho pulse", 16, th.fg)
    s.text(751, 264, "band-pass → envelope → spectrum", 13, th.muted)

    # --- Mounting: the usable upper frequency -------------------------------
    s.rect(48, 356, 396, 244, "none", th.fg, rx=10, sw=1.6)
    s.text(70, 386, "Mounting sets the usable upper frequency", 18, th.fg,
           anchor="start", bold=True)
    rows = (
        (420.0, "stud into a prepared flat", "tens of kHz"),
        (450.0, "adhesive / thin cyanoacrylate", "~10 kHz"),
        (480.0, "magnet base", "a few kHz"),
        (510.0, "hand-held probe", "~1 kHz"),
    )
    for ry, what, limit in rows:
        s.text(70, ry, what, 16, th.fg, anchor="start")
        s.text(422, ry, limit, 16, th.muted, anchor="end")
    s.text(70, 552, "this page band-passes 2-4 kHz, so a magnet base is",
           14, th.secondary, anchor="start")
    s.text(70, 572, "marginal there and a hand-held probe is useless",
           14, th.secondary, anchor="start")

    # --- Acquisition: why 20 kHz and why 2 s --------------------------------
    s.rect(470, 356, 382, 244, "none", th.primary, rx=10, sw=1.6)
    s.text(492, 386, "Acquisition", 18, th.fg, anchor="start", bold=True)
    s.text(492, 420, "$f_s$ = 20 kHz clears the 3 kHz housing resonance", 16,
           th.fg, anchor="start")
    s.text(492, 450, "$T$ = 2 s at 2000 r/min = 67 revolutions", 16, th.fg,
           anchor="start")
    s.text(492, 480, "$Δf = 1/T$ = 0.5 Hz, against $f_s$ = 33.3 Hz", 16,
           th.fg, anchor="start")
    s.text(492, 522, "enough to resolve the $± f_s$ sidebands, which are", 14,
           th.muted, anchor="start")
    s.text(492, 542, "what separates an inner-race defect from an", 14,
           th.muted, anchor="start")
    s.text(492, 562, "outer-race one", 14, th.muted, anchor="start")


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

    # Driving point: exciter below the beam through an impedance head and a
    # drive rod (axially stiff, flexible in every other direction).
    dx = 248.0
    s.rect(dx - 14, beam_bot, 28, 16, th.secondary, th.fg, rx=3, sw=1.6)
    s.line(dx, beam_bot + 16, dx, 412, th.fg, 2.2)
    s.line(dx - 5, 352, dx + 5, 352, th.accent, 3.0)
    s.line(dx - 5, 392, dx + 5, 392, th.accent, 3.0)
    s.text(56, 440, "drive rod: stiff axially,", 14, th.accent,
           anchor="start")
    s.text(56, 458, "flexible in every other", 14, th.accent, anchor="start")
    s.text(56, 476, "direction (6.4.4)", 14, th.accent, anchor="start")
    s.line(186, 448, dx - 8, 372, th.muted, 1.0, dash="3,3")
    s.rect(dx - 37, 412, 74, 48, th.panel, th.primary, rx=9, sw=2)
    s.text(dx, 486, "Exciter", 18, th.fg, bold=True)
    s.text(60, 380, "Impedance head", 16, th.fg, anchor="start", bold=True)
    s.text(60, 402, "$F$ and $a$ at the drive point", 14, th.muted,
           anchor="start")
    s.line(dx - 16, beam_bot + 8, 154, 362, th.muted, 1.1, dash="3,3")
    s.arrow(dx + 18, 396, dx + 18, 340, th.secondary, 2.2)
    s.text(dx + 28, 372, "$F_i$", 16, th.secondary, anchor="start")
    s.arrow(dx, beam_top - 4, dx, beam_top - 46, th.accent, 2.2)
    s.text(dx - 14, beam_top - 34, "$v_i$", 16, th.accent, anchor="end")
    s.text(210, 218, "driving point:  $Y_{ii} = v_i / F_i$", 16, th.fg,
           anchor="start")

    # Transfer point: accelerometer further along the beam.
    tx = 430.0
    _accel(s, tx, beam_top)
    s.arrow(tx, beam_top - 28, tx, beam_top - 56, th.accent, 2.2)
    s.text(tx + 12, beam_top - 40, "$v_j$", 16, th.accent, anchor="start")
    s.text(tx + 60, 192, "transfer:  $Y_{ji} = v_j / F_i$", 16, th.fg)

    # Impact-hammer variant striking the beam.
    hx2 = 600.0
    s.line(hx2 + 60, 172, hx2 + 6, 244, th.fg, 2.4)
    s.rect(hx2 - 16, 238, 32, 20, th.panel, th.fg, rx=4, sw=2)
    s.arrow(hx2, 262, hx2, beam_top - 6, th.secondary, 1.8)

    # FRF family footer.
    s.text(450, 520,
           "$Y(f) = v/F$  [m/(N·s)] · attached exciter (Part 2) · "
           "impact hammer (Part 5)", 17, th.fg)
    s.text(450, 546,
           "same measurement, three FRFs: $x/F$ receptance · $v/F$ mobility "
           "· $a/F$ accelerance", 15, th.muted)

    # ----- Where the two transducers go: the ISO 7626-2 Figure 4 decision ---
    s.text(450, 600, "Where the accelerometer and the force transducer go "
           "(clause 6.4.4)", 19, th.fg, bold=True)
    verdicts = (
        (72.0, "a)", "accelerometer through the rod", "INVALID", th.secondary),
        (356.0, "b)", "force transducer at the structure", "VALID", th.accent),
        (640.0, "c)", "force transducer at the exciter", "WITH CAUTION",
         th.muted),
    )
    for px, tag, what, verdict, colour in verdicts:
        s.rect(px, 626, 224, 158, "none", colour, rx=10, sw=1.8)
        s.text(px + 14, 652, tag, 17, th.fg, anchor="start", bold=True)
        s.rect(px + 40, 668, 148, 12, th.panel, th.fg, sw=1.6)   # structure
        s.line(px + 114, 680, px + 114, 748, th.fg, 2.0)         # drive rod
        s.line(px + 108, 700, px + 120, 700, th.accent, 2.6)
        s.line(px + 108, 730, px + 120, 730, th.accent, 2.6)
        if tag == "a)":                       # accelerometer on the rod
            s.rect(px + 122, 706, 16, 14, th.secondary, th.fg, rx=2, sw=1.2)
            s.rect(px + 100, 748, 28, 12, th.primary, th.fg, rx=2, sw=1.2)
        elif tag == "b)":                     # both at the structure
            s.rect(px + 130, 656, 16, 14, th.secondary, th.fg, rx=2, sw=1.2)
            s.rect(px + 100, 680, 28, 12, th.primary, th.fg, rx=2, sw=1.2)
        else:                                 # force transducer at the exciter
            s.rect(px + 130, 656, 16, 14, th.secondary, th.fg, rx=2, sw=1.2)
            s.rect(px + 100, 736, 28, 12, th.primary, th.fg, rx=2, sw=1.2)
        s.text(px + 112, 776, what, 14, th.fg)
        s.text(px + 112, 646, verdict, 16, colour, bold=True)
    s.text(450, 806, "the exciter attachment must be at least 10× more "
           "mobile, laterally and rotationally, than the structure (6.4.4)",
           16, th.fg)
    s.text(450, 830, "and the suspension at least 10× more mobile than the "
           "structure at each attachment point (5.3)", 15, th.muted)


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
        s.text(cx - 110, 186, "$u_1$", 18, th.secondary, anchor="end")
        # The unidirectionality check: an accelerometer at the edge of the
        # excitation mass, in the plane of the input flange, sensing across
        # the excitation direction (Part 2, Inequality 3).
        s.rect(cx + 62, 190, 16, 14, th.secondary, th.fg, rx=2.5, sw=1.3)
        s.arrow(cx + 78, 197, cx + 104, 197, th.secondary, 1.6)
        # Isolator under test.
        _spring_v(s, cx, 202, 310, th.accent, coils=4)
        s.text(cx + 28, 260, "isolator under test", 16, th.accent,
               anchor="start")
    s.text(48, 296, "$a′_1$: unwanted transverse input,", 14, th.secondary,
           anchor="start")
    s.text(48, 314, "≥ 15 dB below $a_1$ (Inequality 3)", 14, th.secondary,
           anchor="start")
    s.line(150, 286, 310, 202, th.muted, 1.0, dash="3,3")

    # ===== Direct output: blocked, force transducer on a rigid foundation ===
    cx = 250.0
    s.rect(cx - 30, 310, 60, 18, th.secondary, th.fg, rx=3, sw=1.6)
    s.text(cx + 52, 326, "force transducer", 15, th.secondary, anchor="start")
    s.rect(cx - 105, 328, 210, 26, th.panel, th.fg, sw=2)
    s.ground(354, cx - 125, cx + 125)
    s.text(cx, 388, "Rigid foundation", 15, th.muted)
    s.text(cx, 470, "output blocked:  $u_2 ≈ 0$ → measure $F_{2,b}$", 16,
           th.fg)
    s.text(cx, 500, "$k_{2,1} = F_{2,b} / u_1$", 20, th.primary, bold=True)

    # ===== Indirect output: blocking mass on soft supports ==================
    cx = 650.0
    s.rect(cx - 85, 310, 170, 60, th.panel, th.fg, rx=6, sw=2.4)
    s.text(cx, 346, "blocking mass $m_2$", 17, th.fg)
    _accel(s, cx + 55, 310)
    s.text(cx + 72, 296, "$a_2$", 15, th.secondary, anchor="start")
    for sx in (cx - 50.0, cx + 50.0):
        _spring_v(s, sx, 370, 430, th.muted, coils=3, width=8.0, sw=1.6)
    s.ground(430, cx - 115, cx + 115)
    s.text(cx + 70, 408, "soft support", 14, th.muted, anchor="start")
    s.text(cx, 470, "measure $T = u_2 / u_1$  (small)", 16, th.fg)
    s.text(cx, 500, "$k_{2,1} = −(2πf)^2·(m_2+m_f)·T$", 20, th.primary,
           bold=True)

    # Validity footer (Part 3 clause 6, Part 1 Eq. 7).
    s.text(450, 556,
           "valid where $ΔL_{1,2} = L_{a1} − L_{a2} ≥ 20$ dB, i.e. "
           "$|T| ≤ 0.1$   (Part 3, Inequality 2)", 17, th.muted)
    s.text(450, 582,
           "the blocking force approximates the force delivered to a stiff receiver (Part 1, Eq. 7)",
           15, th.muted, italic=True)

    # ===== How the static preload is applied (Part 1, clause 6.3.3.1) =======
    s.text(450, 640, "The two ways the static preload is applied "
           "(ISO 10846-1, 6.3.3.1)", 21, th.fg, bold=True)
    # a) gravity loading
    s.rect(48, 664, 388, 196, "none", th.fg, rx=10, sw=1.6)
    s.text(70, 692, "a) gravity: the output-side mass is the preload", 17,
           th.fg, anchor="start", bold=True)
    s.rect(150, 712, 130, 30, th.panel, th.fg, rx=4, sw=2.0)
    _spring_v(s, 215, 742, 786, th.accent, coils=3, width=10.0, sw=2.0)
    s.rect(140, 786, 150, 34, th.panel, th.fg, rx=4, sw=2.2)
    s.text(215, 808, "load mass", 15, th.fg)
    s.ground(820, 120, 310)
    s.arrow(330, 760, 330, 800, th.secondary, 2.2)
    s.text(340, 784, "$W$ = load", 15, th.secondary, anchor="start")
    s.text(70, 848, "simple, but unstable for large isolators at high loads",
           14, th.muted, anchor="start")
    # b) frame + actuator + decoupling springs
    s.rect(464, 664, 388, 196, "none", th.primary, rx=10, sw=1.6)
    s.text(486, 692, "b) frame and actuator, with decoupling springs", 17,
           th.fg, anchor="start", bold=True)
    s.line(500, 706, 820, 706, th.fg, 2.6)                    # frame traverse
    s.line(500, 706, 500, 822, th.fg, 2.2)
    s.line(820, 706, 820, 822, th.fg, 2.2)
    s.rect(628, 712, 64, 34, th.panel, th.secondary, rx=4, sw=2.0)
    s.arrow(660, 748, 660, 768, th.secondary, 2.2)
    s.text(676, 734, "actuator: 100 % of the", 14, th.secondary,
           anchor="start")
    s.text(676, 752, "permissible static load", 14, th.secondary,
           anchor="start")
    _spring_v(s, 660, 768, 800, th.accent, coils=3, width=10.0, sw=2.0)
    s.rect(600, 800, 120, 26, th.panel, th.fg, rx=4, sw=2.2)
    s.text(660, 818, "$m_2$", 15, th.fg)
    for sx in (556.0, 764.0):
        _spring_v(s, sx, 800, 840, th.muted, coils=2, width=7.0, sw=1.5)
    s.text(500, 856, "auxiliary springs decouple $m_2$ from the frame", 14,
           th.muted, anchor="start")

    # ===== Transverse translations (Part 2, clause 5.2) =====================
    s.text(450, 900, "Transverse translations are standardised too "
           "(ISO 10846-2, 5.2)", 21, th.fg, bold=True)
    s.rect(48, 924, 804, 132, "none", th.muted, rx=10, sw=1.4)
    s.rect(300, 950, 190, 30, th.panel, th.fg, rx=4, sw=2.2)
    s.text(395, 971, "force-distribution plate", 14, th.fg)
    _motion_arrows(s, 268, 965, 22, th.secondary)
    s.text(255, 940, "$a_{1x}$", 16, th.secondary, anchor="end")
    for rx in (330.0, 460.0):                      # guiding roller bearings
        s.circle(rx, 940, 9, th.bg, th.fg, 1.6)
    s.text(508, 942, "roller bearings, or two symmetrical", 14, th.muted,
           anchor="start")
    s.text(508, 962, "elements, suppress the unwanted input", 14, th.muted,
           anchor="start")
    s.rect(320, 980, 150, 26, th.accent, th.fg, rx=4, sw=1.8)
    s.text(395, 999, "test element in shear", 13, th.bg)
    for fx in (348.0, 442.0):
        s.rect(fx - 22, 1006, 44, 16, th.panel, th.secondary, rx=3, sw=1.6)
    s.text(508, 1000, "output shear force summed from two", 14, th.fg,
           anchor="start")
    s.text(508, 1020, "transducers,  $F_2 = F_2′ + F_2″$", 14, th.fg,
           anchor="start")
    s.ground(1022, 300, 490)
    s.text(450, 1082, "a mount is loaded in shear as well as in compression, "
           "and the transverse stiffness is usually the smaller of the two",
           15, th.muted)


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
    s.text(58, 474, "$l_{ij} ≥ 2.3$ m", 17, th.fg, anchor="start")
    s.line(126, 466, 152, 438, th.muted, 1.0)

    # Exciter on the floor (element i), accelerometers on i and j.
    _exciter(s, 330, 396)
    _accel(s, 250, 410)
    _accel(s, 380, 380)
    _accel_wall(s, 205, 300)
    _accel_wall(s, 236, 262)
    s.text(196, 420, "$i$", 22, th.primary, bold=True)
    s.text(178, 200, "$j$", 22, th.secondary, bold=True)
    # Transmission path across the corner.
    s.path("M 300 402 Q 214 400 208 330", stroke=th.accent, sw=2.0)
    s.arrow(209.0, 344.0, 208.0, 322.0, th.accent, 2.0)
    s.text(194, 356, "$D_{v,ij}$", 16, th.accent, anchor="end")

    # ===== Right: T-junction (wall standing mid-way on the floor) =========
    s.text(690, 86, "T-junction", 21, th.fg, bold=True)
    _plate_top(s, th, 520, gy, 220, dp, 16)
    _plate_up(s, th, 620, gy, 16, 180, dp)
    s.line(636, gy, 636 + dxo, gy - dyo, th.accent, 2.6)
    _exciter(s, 566, 404)
    _accel(s, 588, 422)
    _accel_wall(s, 685, 290)
    _accel(s, 762, 384)
    s.text(533, 423, "$i$", 22, th.primary, bold=True)
    s.text(658, 200, "$j$", 22, th.secondary, bold=True)
    s.text(806, 400, "$j$", 22, th.secondary, bold=True)
    s.path("M 612 418 Q 690 434 756 400", stroke=th.accent, sw=2.0)
    s.arrow(742.0, 407.0, 760.0, 398.0, th.accent, 2.0)
    s.path("M 606 406 Q 646 394 654 330", stroke=th.accent, sw=2.0)
    s.arrow(655.0, 344.0, 654.0, 322.0, th.accent, 2.0)

    # Exciter label shared by both panels.
    s.text(450, 250, "Shaker or hammer on element $i$", 17, th.fg)
    s.line(376, 258, 348, 322, th.muted, 1.0)
    s.line(524, 258, 556, 348, th.muted, 1.0)

    # Plate thickness leader (the lines stop above the caption text).
    s.text(450, 496, "concrete plates 140 mm to 200 mm thick", 17, th.muted)
    s.line(322, 477, 300, 442, th.muted, 1.0)
    s.line(578, 477, 600, 442, th.muted, 1.0)

    # Normative relations.
    s.text(80, 536, "$l_{ij} ≥ 2.3$ m along the junction; element sizes "
           "3.0 m $≤ l_i <$ 6.0 m", 18, th.fg, anchor="start")
    s.text(80, 564, "≥ 4 excitation positions on $i$; accelerometers "
           "≥ 0.25 m from edges, ≥ 0.5 m apart", 18, th.fg, anchor="start")
    s.text(80, 596, "$K_{ij} = D̄_{v,ij} + 10 log_{10}( l_{ij} / √(a_i·a_j) "
           ")$,   $a_i$ = equivalent absorption length",
           17, th.primary, anchor="start", bold=True)


# ---------------------------------------------------------------------------
# Power injection: coupling loss factors from measured energies (Norton 6.6.4)
# ---------------------------------------------------------------------------

def _d_power_injection_rig(s: SVG, th: Theme) -> None:
    """The experimental-SEA rig: two plates joined along an edge, a shaker
    through an impedance head on one of them, accelerometers distributed over
    both, and the drive moved for the second run of the two-drive scheme."""
    gy = 380.0
    dp = 116.0
    dxo, dyo = dp * 0.72, dp * 0.55

    def _pair(x0: float, driven: int, head: str) -> None:
        """One state of the experiment: plate 1 horizontal, plate 2 upright."""
        s.text(x0 + 200, 80, head, 21, th.fg, bold=True)
        _plate_top(s, th, x0 + 40, gy, 240, dp, 12)
        _plate_up(s, th, x0 + 40, gy, 12, 168, dp)
        s.line(x0 + 52, gy, x0 + 52 + dxo, gy - dyo, th.accent, 2.6)
        s.line(x0 + 100, 250, x0 + 156, 156, th.muted, 1.0, dash="3,3")
        s.text(x0 + 160, 150, "subsystem 2", 17, th.secondary,
               anchor="start")
        s.text(x0 + 248, gy + 44, "subsystem 1", 17, th.primary)
        # Accelerometer positions: several per subsystem, off the edges and
        # off the drive point, because the band energy uses a space average.
        for ax in (108.0, 152.0, 216.0, 252.0):
            _accel(s, x0 + ax, gy)
        _accel(s, x0 + 178, gy - 34)
        for ay, axx in ((240.0, 20.0), (288.0, 48.0), (330.0, 26.0)):
            _accel_wall(s, x0 + 52 + axx, ay)
        # The drive: a shaker through an impedance head, on one subsystem.
        if driven == 1:
            s.rect(x0 + 120, gy + 2, 26, 13, th.secondary, th.fg, rx=3, sw=1.4)
            s.line(x0 + 133, gy + 15, x0 + 133, gy + 40, th.fg, 2.2)
            s.rect(x0 + 96, gy + 40, 74, 42, th.panel, th.primary, rx=9, sw=2)
            s.text(x0 + 176, gy + 20, "impedance head", 15, th.secondary,
                   anchor="start")
            s.text(x0 + 180, gy + 68, "shaker", 16, th.fg, anchor="start")
        else:
            s.rect(x0 + 58, 196, 26, 13, th.secondary, th.fg, rx=3, sw=1.4)
            s.line(x0 + 84, 202, x0 + 176, 202, th.fg, 2.2)
            s.rect(x0 + 176, 179, 74, 46, th.panel, th.primary, rx=9, sw=2)
            s.text(x0 + 213, 248, "shaker", 16, th.fg)
        s.text(x0 + 40, gy + 96,
               f"$Π_{'1' if driven == 1 else '2'}$ measured,  "
               f"$Π_{'2' if driven == 1 else '1'} = 0$", 16, th.fg,
               anchor="start")
        s.text(x0 + 40, gy + 122, "$E_1 = M_1⟨v_1^2⟩,   E_2 = M_2⟨v_2^2⟩$",
               16, th.fg, anchor="start")

    _pair(10.0, 1, "Run 1: drive subsystem 1")
    _pair(460.0, 2, "Run 2: drive subsystem 2")
    s.line(450, 70, 450, 528, th.muted, 1.0, dash="6,6")

    # What is measured, and what each run buys.
    s.text(450, 566, "$Π_{in} = ½ Re{F v*}$ at the drive point, from an "
           "impedance head — not the amplifier setting", 17, th.fg)
    s.text(450, 592, "$⟨v^2⟩$ space-averaged over several positions per "
           "subsystem, away from the edges and from the drive point", 17,
           th.fg)
    s.text(450, 618, "one run inverts to $η_{12}$ only if $η_1$ and $η_2$ "
           "are known from a decay measurement; two runs solve all four", 17,
           th.primary, bold=True)
    s.text(450, 644, "bands wide enough to hold several modes of each "
           "subsystem: the modal densities decide how wide", 15, th.muted)
