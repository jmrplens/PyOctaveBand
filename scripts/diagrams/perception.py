#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Diagrams of the perception guides: hearing, speech and psychoacoustics.

One subject: the listener. The hearing diagrams draw the threshold, its
shift with age and noise, and the exposure that causes the shift; the speech
diagrams draw the intelligibility chains; the psychoacoustics diagrams draw
the sensations that a level alone does not carry, from loudness to tonality
and annoyance.
"""

from __future__ import annotations

from .canvas import SVG, Theme

# ---------------------------------------------------------------------------
# d3 - Operator / bystander microphone positions (ECMA-74, clause 8.6)
# ---------------------------------------------------------------------------

def _d_emission_positions(s: SVG, th: Theme) -> None:
    gy = 470.0
    s.ground(gy, 40, 860)

    # --- Left: seated operator at table-top equipment (side view) ---------
    s.text(240, 72, "Operator — seated (P2)", 24, th.fg, bold=True)
    tx = 80.0
    table_y = gy - 150.0
    s.line(tx + 18, gy, tx + 18, table_y, th.fg, 3)
    s.line(tx + 232, gy, tx + 232, table_y, th.fg, 3)
    s.line(tx, table_y, tx + 250, table_y, th.fg, 4)
    s.rect(tx + 16, table_y - 76, 118, 76, th.panel, th.primary, rx=8, sw=2)
    s.text(tx + 75, table_y - 32, "EUT", 22, th.primary, bold=True)
    eut_front = tx + 134.0

    # microphone: capsule tip at 1.20 m, 0.25 m from the EUT front face
    mx = eut_front + 76.0
    cap = gy - 268.0
    s.mic(mx, cap, table_y, 1.1)
    s.line(mx - 18, table_y, mx + 18, table_y, th.fg, 2.2)
    s.dim(eut_front, table_y - 76, mx, cap, "0.25 m", offset=-36, size=20)
    s.dim(mx + 210, gy, mx + 210, cap, "1.20 m", offset=0, size=20, label_side="right")
    s.line(mx + 10, cap, mx + 210, cap, th.muted, 0.9, dash="3,3")  # witness to capsule

    # seated operator on a chair, clear of both dimensions
    px = mx + 120.0
    seat_y = gy - 115.0
    s.line(px - 28, seat_y, px + 32, seat_y, th.muted, 3)
    s.line(px - 24, seat_y, px - 24, gy, th.muted, 2.6)
    s.line(px + 28, seat_y, px + 28, gy, th.muted, 2.6)
    s.line(px + 32, seat_y, px + 32, seat_y - 86, th.muted, 2.6)
    s.circle(px, gy - 240, 15, th.muted)
    s.line(px, gy - 225, px + 6, seat_y, th.muted, 3.4)
    s.line(px + 6, seat_y, px - 34, seat_y - 2, th.muted, 2.8)
    s.line(px - 34, seat_y - 2, px - 34, gy, th.muted, 2.8)
    s.line(px - 1, gy - 205, px - 38, gy - 178, th.muted, 2.6)

    # --- Right: bystander positions (top view), equal face distances ------
    cx, cyv = 700.0, 270.0
    s.text(cx, 72, "Bystanders — top view", 24, th.fg, bold=True)
    s.text(cx, 100, "height 1.50 m", 20, th.muted)
    s.rect(cx - 52, cyv - 40, 104, 80, th.panel, th.primary, rx=8, sw=2)
    s.text(cx, cyv + 8, "EUT", 22, th.primary, bold=True)
    g = 92.0  # face-to-microphone distance, equal on all four sides
    for pxx, pyy in [(cx, cyv - 40 - g), (cx, cyv + 40 + g),
                     (cx - 52 - g, cyv), (cx + 52 + g, cyv)]:
        s.circle(pxx, pyy, 8, th.secondary)
        s.circle(pxx, pyy, 2.8, th.bg)
    s.dim(cx + 52, cyv - 20, cx + 52 + g, cyv, "1.00 m", offset=-44, size=20)


# ---------------------------------------------------------------------------
# d7 - STI measurement chain (IEC 60268-16)
# ---------------------------------------------------------------------------

def _d_sti_chain(s: SVG, th: Theme) -> None:
    stages = [
        ("Source", "STIPA signal", th.fg),
        ("Room", "reverberation + noise", th.secondary),
        ("Microphone", "", th.primary),
        ("Analysis", "MTF → TI → STI", th.accent),
    ]
    bw, bh, gap = 192.0, 96.0, 20.0
    total = len(stages) * bw + (len(stages) - 1) * gap
    x = (900 - total) / 2
    y = 150.0
    for i, (title, sub, color) in enumerate(stages):
        s.rect(x, y, bw, bh, th.panel, color, rx=12, sw=2)
        if sub:
            s.text(x + bw / 2, y + 42, title, 22, th.fg, bold=True)
            if "→" in sub:
                s.text(x + bw / 2, y + 70, sub, 18, color, mono=True)
            else:
                s.text(x + bw / 2, y + 70, sub, 18, color)
        else:
            s.text(x + bw / 2, y + bh / 2 + 7, title, 22, th.fg, bold=True)
        if i == 1:  # the room degrades the modulation transfer function
            cx = x + bw / 2
            s.line(cx, y + bh, cx, y + bh + 18, th.muted, 1.2, dash="3,3")
            s.text(cx, y + bh + 40, "$m(F)$ drops", 18, th.muted)
        if i < len(stages) - 1:
            s.arrow(x + bw + 1, y + bh / 2, x + bw + gap - 2, y + bh / 2, th.fg, 2)
        x += bw + gap


def _d_speech_intelligibility(s: SVG, th: Theme) -> None:
    """SII computation flow (ANSI S3.5-1997, one-third-octave method)."""
    # --- Top: three equivalent-spectrum-level inputs (per 1/3-octave band) ---
    inputs = [
        (150.0, "Speech  $E′_i$", th.primary),
        (450.0, "Noise  $N′_i$", th.secondary),
        (750.0, "Threshold  $T′_i$", th.accent),
    ]
    iw, ih, iy = 220.0, 66.0, 40.0
    for cx, label, col in inputs:
        s.rect(cx - iw / 2, iy, iw, ih, th.panel, col, rx=10, sw=2)
        s.text(cx, iy + 28, label, 21, th.fg, "middle", bold=True)
        s.text(cx, iy + 51, "spectrum level (dB)", 16, th.muted, "middle")
        s.arrow(cx, iy + ih, cx, 150, th.fg, 1.8)

    # --- Vertical processing chain (ANSI S3.5-1997 clause 5) ---
    cx, bw, bh = 450.0, 470.0, 70.0
    x0 = cx - bw / 2
    chain = [
        (150.0, "Self-masking + spread of masking", "$Z_i$   (clause 5.4)"),
        (264.0, "Equivalent disturbance $D_i$", "max(masking, internal noise) (5.6)"),
        (378.0, "Band audibility $A_i = (E′_i − D_i + 15)/30$", "clipped to [0, 1]   (clause 5.8)"),
    ]
    for by, l1, l2 in chain:
        s.rect(x0, by, bw, bh, th.panel, th.fg, rx=12, sw=2)
        s.text(cx, by + 30, l1, 20, th.fg, "middle", bold=True)
        s.text(cx, by + 54, l2, 17, th.muted, "middle")
    s.arrow(cx, 220, cx, 264, th.fg, 2.0)
    s.arrow(cx, 334, cx, 378, th.fg, 2.0)

    # --- Band-importance weighting and the final index ---
    s.arrow(cx, 448, cx, 486, th.fg, 2.0)
    s.rect(x0, 486, bw, 74, "none", th.primary, rx=12, sw=2.4)
    s.text(cx, 516, "$SII = Σ I_i A_i$", 26, th.fg, "middle", bold=True)
    s.text(cx, 542, "band importance $I_i$ (Table 3)  ·  index in [0, 1]  (clause 6)",
           16, th.primary, "middle")


def _d_hearing_threshold(s: SVG, th: Theme) -> None:
    """Hearing-threshold model: ISO 7029 age distribution + ISO 389-7 zero."""
    cx = 450.0
    # --- Inputs --------------------------------------------------------------
    iw, ih = 540.0, 62.0
    s.rect(cx - iw / 2, 56, iw, ih, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 84, "Age $Y$,  sex,  population fractile $Q$", 20, th.fg,
           "middle", bold=True)
    s.text(cx, 106, "audiometric frequencies 125 Hz – 8000 Hz", 15, th.muted,
           "middle")
    s.arrow(cx, 118, cx, 152, th.fg, 1.8)

    bw, bh = 620.0, 60.0
    x0 = cx - bw / 2

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 26, l1, 18, th.fg, "middle", bold=True)
        s.text(cx, y + 47, l2, 14, th.muted, "middle")

    # --- ISO 7029 chain ------------------------------------------------------
    # The dHmd/dHQ formulas and the su/sl spreads stay out of $...$ markup:
    # ISO 7029:2017 (4.2, Formulae (4)/(5)) prints the md, u and l
    # subscripts upright, and the curated roman list cannot carry them (md
    # is missing; a bare u or l would set every such index upright).
    _step(152, "Median deviation from age 18   (ISO 7029, 4.2)",
          "dHmd = a · (Y − 18) ^ b   (Table 1, by sex)", th.primary)
    _step(244, "Spread su / sl   (ISO 7029, 4.3)",
          "degree-5 polynomials in $(Y − 18)$   (Tables 2–5)", th.fg)
    _step(336, "Fractile threshold   (ISO 7029, 4.4)",
          "dHQ = dHmd + z(Q) * s   (su if Q >= 0.5, else sl)", th.fg)
    s.arrow(cx, 212, cx, 244, th.fg, 1.8)
    s.arrow(cx, 304, cx, 336, th.fg, 1.8)
    s.arrow(cx, 396, cx, 430, th.fg, 1.8)

    # --- Output + ISO 389-7 reference ---------------------------------------
    s.rect(x0, 430, bw, 58, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, 456, "Expected hearing threshold level (dB HL)", 19, th.fg,
           "middle", bold=True)
    s.text(cx, 476, "referenced to the audiometric zero", 14, th.primary,
           "middle")
    s.rect(x0, 506, bw, 52, th.panel, th.secondary, rx=10, sw=2)
    s.text(cx, 530, "Audiometric zero = ISO 389-7 reference threshold",
           17, th.fg, "middle", bold=True)
    s.text(cx, 549, "free-field / diffuse-field (Table 1) — the dB HL / dB SPL zero",
           14, th.muted, "middle")


def _d_nihl(s: SVG, th: Theme) -> None:
    """Noise-induced hearing loss (ISO 1999:2013): NIPTS and HTLAN.

    Two converging lanes (the age component H (HTLA, database A = ISO 7029)
    and the noise component N (NIPTS, Formulae 2-7)) combine into the hearing
    threshold associated with age and noise (HTLAN, Formula 1).
    """
    cx = 450.0
    lxc, rxc = 232.0, 668.0
    bw, bh = 372.0, 62.0

    def _step(cxx: float, y: float, l1: str, l2: str, color: str) -> None:
        s.rect(cxx - bw / 2, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cxx, y + 27, l1, 18, th.fg, "middle", bold=True)
        if l2:
            s.text(cxx, y + 48, l2, 13, th.muted, "middle")

    # --- Inputs -------------------------------------------------------------
    # L_EX,8h stays plain: ISO 1999:2013 prints the whole EX,8h subscript
    # upright, and the composer has no upright run for the "8h" unit inside
    # a script, so $L_{EX,8h}$ would set the h as an italic index.
    _step(lxc, 56, "Age $Y$,  sex,  fractile $Q$", "database A = ISO 7029", th.fg)
    _step(rxc, 56, "Exposure L_EX,8h,  t years",
          "normalized to 8 h / 5 days", th.fg)

    # --- Left lane: age component H (HTLA) ----------------------------------
    s.arrow(lxc, 118, lxc, 150, th.fg, 1.8)
    _step(lxc, 150, "Age threshold  $H$  (HTLA)",
          "ISO 7029 fractile, dB", th.primary)

    # --- Right lane: noise component N (NIPTS) ------------------------------
    s.arrow(rxc, 118, rxc, 150, th.fg, 1.8)
    _step(rxc, 150, "Median NIPTS  $N_{50}$  (6.3.1)",
          "$N_{50} = [u + v·log_{10}(t/t_0)]·(L − L_0)^2$", th.secondary)
    s.arrow(rxc, 212, rxc, 244, th.fg, 1.8)
    # The du/dl fractile arms stay plain: ISO 1999:2013, Formulae (4) and
    # (5), prints d with an upright u/l subscript, which the roman list
    # cannot carry letter by letter.
    _step(rxc, 244, "Fractile NIPTS  $N$  (6.3.2)",
          "N = N50 + z·(du if z ≥ 0 else dl)", th.fg)

    # --- Converge into HTLAN ------------------------------------------------
    box_y = 372.0
    s.arrow(lxc, 212, cx - 118.0, box_y, th.fg, 1.8)
    s.arrow(rxc, 306, cx + 118.0, box_y, th.fg, 1.8)
    s.rect(cx - bw / 2, box_y, bw, 66, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, box_y + 29, "HTLAN   $H′ = H + N − H·N / 120$", 20, th.fg,
           "middle", bold=True)
    s.text(cx, box_y + 51, "threshold from age and noise  (Formula 1, 6.1)",
           13, th.muted, "middle")


def _d_zwicker(s: SVG, th: Theme) -> None:
    """ISO 532-1 Zwicker loudness: from band levels to N (sone) and LN (phon)."""
    cx = 450.0
    bw, bh = 668.0, 58.0
    x0 = cx - bw / 2

    s.rect(x0, 46, bw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 70, "28 one-third-octave band levels, 25 Hz to 12.5 kHz", 18,
           th.fg, "middle", bold=True)
    s.text(cx, 90, "from a spectrum, or from a calibrated signal via the Annex A "
           "filterbank", 13, th.muted, "middle")
    s.arrow(cx, 104, cx, 132, th.fg, 1.8)

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 25, l1, 17, th.fg, "middle", bold=True)
        s.text(cx, y + 45, l2, 13, th.muted, "middle")

    _step(132, "Equal-loudness correction and lower critical bands  "
          "(Clause 5.4, Table A.3)",
          "the 11 lowest bands grouped into 3 critical bands, 25-250 Hz",
          th.primary)
    # The corrections line stays plain: ISO 532-1:2017 prints ΔL_DF and
    # L_TQ (Tables A.5/A.6) with upright DF/TQ subscripts the roman list
    # does not carry, and composing only the a₀ beside them would split one
    # enumeration into two styles.
    _step(218, "Core loudness of the 20 critical bands  (Tables A.4-A.7)",
          "a₀ transmission (A.4), diffuse-field DDF (A.5), threshold in quiet "
          "LTQ (A.6)", th.fg)
    _step(304, "Specific loudness  $N′(z)$  over 0.1-Bark steps to 24 Bark",
          "upper masking slopes added band to band (Table A.9)", th.secondary)
    for y0, y1 in ((190, 218), (276, 304)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)
    s.arrow(cx, 362, cx, 392, th.fg, 1.8)

    s.rect(x0, 392, bw, 60, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, 417, "Total loudness  $N = ∫ N′(z) dz$  [sone]", 17, th.fg, "middle",
           bold=True)
    s.text(cx, 438, "loudness level  $L_N = 40 + 10·log_2 N$  [phon]", 14, th.muted,
           "middle")


def _d_loudness_capture(s: SVG, th: Theme) -> None:
    """Where the microphone goes for an ISO 532-1 loudness measurement.

    Definition 3.19 fixes the measurement point (the centre of the absent
    listener's head, which is what makes the calculated loudness diotic) and
    clause 4 makes the field type part of the result: NF for free field,
    ND for diffuse. Annex D adds the head-and-torso route and its
    equalization decision. The geometry drawn is the ECMA-74 / ISO 11201
    bystander position, 1.00 m from the reference box at 1.50 m height,
    since ISO 532-1 itself prescribes no distance.
    """
    # --- Panel A: free field ------------------------------------------------
    # NF, ND and the NL/NR pair stay plain in this diagram: ISO 532-1:2017
    # (3.19 Note 1, Annex D) prints them as N with upright F/D/L/R
    # descriptor subscripts, which the composer's roman list cannot carry
    # letter by letter.
    ax0, pw = 26.0, 418.0
    s.rect(ax0, 48, pw, 330, th.panel, th.muted, rx=12, sw=1.6)
    s.text(ax0 + pw / 2, 76, "A — Free field  (NF)", 20, th.fg, bold=True)
    s.text(ax0 + pw / 2, 98, "hemi-anechoic room, one frontal source", 15,
           th.muted)

    gy = 336.0
    for k in range(11):
        wx = ax0 + 20 + k * 36
        s.path(f"M {wx} 110 L {wx + 30} 110 L {wx + 15} 134 Z",
               fill=th.panel, stroke=th.muted, sw=1.2)
    s.ground(gy, ax0 + 16, ax0 + pw - 16)

    # Equipment under test inside its reference box.
    s.rect(ax0 + 40, gy - 70, 78, 70, th.panel, th.primary, rx=6, sw=2)
    s.text(ax0 + 79, gy - 30, "EUT", 19, th.primary, bold=True)
    s.rect(ax0 + 32, gy - 80, 94, 80, "none", th.muted, rx=4, sw=1.4,
           dash="5,4")
    s.text(ax0 + 79, gy - 92, "reference box", 14, th.muted)

    # Microphone at the point the listener's head would occupy.
    mx = ax0 + 236.0
    cap = gy - 165.0
    s.circle(mx, cap + 6, 30, "none", th.secondary, 1.4)
    s.mic(mx, cap, gy, 1.0)
    s.text(mx + 44, cap - 26, "listener absent:", 14, th.secondary, "start")
    s.text(mx + 44, cap - 8, "the result is diotic", 14, th.secondary, "start")
    s.arrow(ax0 + 132, gy - 56, mx - 30, cap + 26, th.primary, 1.8)
    s.text(ax0 + 36, gy - 118, "frontal incidence, 0°", 14, th.primary,
           "start")

    s.dim(ax0 + 126, gy - 80, mx, cap, "1.00 m", offset=64, size=17)
    s.dim(mx + 122, gy, mx + 122, cap, "1.50 m", offset=0, size=17,
          label_side="right")
    s.line(mx + 6, cap, mx + 122, cap, th.muted, 0.9, dash="3,3")
    s.text(ax0 + pw / 2, 366, 'field="free"  →  quote N as NF', 16,
           th.primary, mono=True)

    # --- Panel B: diffuse field --------------------------------------------
    bx0 = 456.0
    s.rect(bx0, 48, pw, 330, th.panel, th.muted, rx=12, sw=1.6)
    s.text(bx0 + pw / 2, 76, "B — Diffuse field  (ND)", 20, th.fg, bold=True)
    s.text(bx0 + pw / 2, 98, "reverberant or in-situ room", 15, th.muted)

    rx0, ry0, rw, rh = bx0 + 26, 118.0, pw - 52, 196.0
    s.rect(rx0, ry0, rw, rh, "none", th.fg, rx=4, sw=2.4)
    s.rect(bx0 + 60, ry0 + rh - 58, 70, 58, th.panel, th.primary, rx=6, sw=2)
    s.text(bx0 + 95, ry0 + rh - 22, "EUT", 18, th.primary, bold=True)
    mxb = bx0 + 268.0
    capb = ry0 + rh - 132.0
    s.mic(mxb, capb, ry0 + rh, 1.0)
    src = (bx0 + 130.0, ry0 + rh - 36.0)
    for tx, ty in ((rx0 + 40, ry0 + 4), (rx0 + rw - 4, ry0 + 56),
                   (rx0 + 120, ry0 + 4), (rx0 + rw - 4, ry0 + 128)):
        s.line(src[0], src[1], tx, ty, th.accent, 1.2)
        s.line(tx, ty, mxb - 8, capb + 16, th.accent, 1.2)
    s.arrow(src[0], src[1], mxb - 14, capb + 30, th.secondary, 1.6)
    s.text(bx0 + pw / 2, ry0 + rh + 26,
           "direct sound plus the reflected field, from every direction", 14,
           th.muted)
    s.text(bx0 + pw / 2, 366, 'field="diffuse"  →  quote N as ND', 16,
           th.primary, mono=True)

    # --- Panel C: head-and-torso simulator (Annex D) ------------------------
    s.rect(26, 396, 848, 228, th.panel, th.muted, rx=12, sw=1.6)
    s.text(60, 424, "C — Head-and-torso simulator (Annex D)", 20, th.fg,
           "start", bold=True)

    hx, hy = 130.0, 486.0
    s.path(f"M {hx - 44} {hy + 62} L {hx - 30} {hy + 14} "
           f"L {hx + 30} {hy + 14} L {hx + 44} {hy + 62} Z",
           fill=th.panel, stroke=th.fg, sw=2)
    s.circle(hx, hy - 16, 30, th.panel, th.fg, 2)
    s.ellipse(hx - 30, hy - 12, 6, 11, th.bg, th.secondary, 1.8)
    s.ellipse(hx + 30, hy - 12, 6, 11, th.bg, th.secondary, 1.8)
    s.text(hx - 52, hy - 30, "L", 16, th.secondary, "end", bold=True)
    s.text(hx + 52, hy - 30, "R", 16, th.secondary, "start", bold=True)
    s.text(hx, hy + 82, "at the listening position", 14, th.muted)

    s.arrow(hx + 50, hy - 4, 240, hy - 4, th.fg, 1.8)
    s.rect(240, hy - 44, 250, 80, th.bg, th.primary, rx=8, sw=2)
    s.text(365, hy - 18, "Equalization matched", 15, th.fg, bold=True)
    s.text(365, hy + 2, "to the room:", 15, th.fg)
    s.text(365, hy + 24, "free-field / diffuse-field / ID", 14, th.primary,
           mono=True)
    for k, ear in enumerate(("left channel", "right channel")):
        by = hy - 44 + k * 44
        s.rect(552, by, 186, 36, th.bg, th.secondary, rx=8, sw=1.8)
        s.text(645, by + 24, ear, 15, th.fg)
        s.arrow(494, hy - 4, 548, by + 18, th.fg, 1.6)
        s.arrow(742, by + 18, 784, hy - 4, th.fg, 1.6)
    s.text(818, hy - 10, "NL, NR", 17, th.fg, bold=True)
    s.text(818, hy + 12, "both reported", 14, th.muted)

    s.text(450, 588, "free-field equalization only for one frontal source "
           "beyond 1.5 m; diffuse-field in reflective rooms; ID in vehicles",
           14, th.muted)
    s.text(450, 610, "each channel is analysed separately: report NL and NR, "
           "and quote the maximum or the mean as the single value", 14,
           th.muted)


def _d_mg_capture_routes(s: SVG, th: Theme) -> None:
    """The ISO 532-2 clause 7.2 capture routes and the arguments they imply.

    Four listening situations, each with its own transfer function to the
    tympanic membrane: a single microphone where the listener's head would be
    (free or diffuse field, Table 1), a probe microphone in the ear canal (no
    transfer function at all), and a head-and-torso simulator (none either,
    but only if it is an accurate model of an average adult).
    """
    import math

    x0, w = 34.0, 832.0
    rows = (76.0, 216.0, 356.0, 496.0)
    for y in rows:
        s.rect(x0, y, w, 128, th.panel, th.muted, rx=10, sw=1.6)
        s.line(500, y + 10, 500, y + 118, th.muted, 1.0, dash="4,4")

    def result(y: float, line1: str, line2: str, note: str) -> None:
        s.arrow(456, y + 64, 496, y + 64, th.fg, 1.8)
        s.text(520, y + 42, line1, 17, th.primary, "start", mono=True)
        s.text(520, y + 66, line2, 15, th.muted, "start")
        s.text(520, y + 92, note, 14, th.muted, "start")

    # --- 1: single microphone at the absent head centre, one frontal source -
    y = rows[0]
    s.text(x0 + 16, y + 26, "1 — single microphone where the head would be, "
           "one frontal source", 17, th.fg, "start", bold=True)
    s.rect(70, y + 60, 34, 44, th.panel, th.secondary, rx=5, sw=2)
    for k in range(3):
        s.path(f"M {124 + k * 20} {y + 52} q 9 30 0 60", stroke=th.secondary,
               sw=1.4)
    s.circle(268, y + 80, 28, "none", th.muted, 1.2)
    s.mic(268, y + 56, y + 114, 0.8)
    s.text(316, y + 76, "listener absent", 14, th.muted, "start")
    s.text(316, y + 96, "(diotic)", 14, th.muted, "start")
    result(y, 'field="free"', "Table 1 free-field transfer",
           "frontal incidence; the default")

    # --- 2: the same microphone in a reverberant field ----------------------
    y = rows[1]
    s.text(x0 + 16, y + 26, "2 — the same microphone, reverberant or in-situ "
           "field", 17, th.fg, "start", bold=True)
    for ang in range(0, 360, 45):
        a = math.radians(ang)
        s.arrow(268 + 78 * math.cos(a), y + 80 + 34 * math.sin(a),
                268 + 34 * math.cos(a), y + 80 + 15 * math.sin(a), th.accent,
                1.3)
    s.mic(268, y + 56, y + 114, 0.8)
    result(y, 'field="diffuse"', "Table 1 diffuse-field transfer",
           "also for diffuse-field earphones")

    # --- 3: probe microphone in the ear canal -------------------------------
    y = rows[2]
    s.text(x0 + 16, y + 26, "3 — probe microphone in the ear canal", 17,
           th.fg, "start", bold=True)
    hx, hy = 168.0, y + 76.0
    s.circle(hx, hy, 32, th.panel, th.fg, 2)
    s.path(f"M {hx + 30} {hy - 9} L {hx + 96} {hy - 17} L {hx + 96} "
           f"{hy + 17} L {hx + 30} {hy + 9} Z", fill=th.bg, stroke=th.fg,
           sw=1.6)
    s.line(hx + 30, hy - 9, hx + 30, hy + 9, th.secondary, 3.4)
    s.line(hx + 30, hy - 9, hx + 42, hy - 34, th.secondary, 1.2)
    s.text(hx + 46, hy - 38, "tympanic membrane", 13, th.secondary, "start")
    s.circle(hx + 58, hy, 5, th.primary)
    s.line(hx + 58, hy, hx + 132, hy + 12, th.primary, 1.6)
    s.text(hx + 138, hy + 16, "probe microphone", 13, th.primary, "start")
    s.dim(hx + 30, hy + 17, hx + 58, hy + 17,
          "10 mm — 5 mm above 3 kHz", offset=32, size=14)
    result(y, 'field="eardrum"', "no transfer function applied",
           "the ear transfer is already in the signal")

    # --- 4: head and torso simulator ----------------------------------------
    y = rows[3]
    s.text(x0 + 16, y + 26, "4 — head-and-torso simulator", 17, th.fg,
           "start", bold=True)
    hx, hy = 108.0, y + 82.0
    s.path(f"M {hx - 32} {hy + 32} L {hx - 22} {hy + 4} L {hx + 22} {hy + 4} "
           f"L {hx + 32} {hy + 32} Z", fill=th.panel, stroke=th.fg, sw=1.8)
    s.circle(hx, hy - 16, 22, th.panel, th.fg, 1.8)
    s.ellipse(hx - 22, hy - 14, 5, 9, th.bg, th.secondary, 1.6)
    s.ellipse(hx + 22, hy - 14, 5, 9, th.bg, th.secondary, 1.6)
    s.arrow(hx + 36, hy - 12, 152, hy - 12, th.fg, 1.6)
    s.path(f"M 240 {hy - 42} L 326 {hy - 12} L 240 {hy + 18} L 154 "
           f"{hy - 12} Z", fill=th.bg, stroke=th.primary, sw=1.8)
    s.text(240, hy - 18, "accurate model of", 13, th.fg)
    s.text(240, hy - 2, "an average adult?", 13, th.fg)
    s.text(336, hy - 22, "yes: no correction", 13, th.accent, "start")
    s.text(336, hy + 4, "no: correction file —", 13, th.secondary, "start")
    s.text(336, hy + 22, "not implemented", 13, th.secondary, "start")
    result(y, 'field="eardrum"  or  equalize', "clause 7.2.5",
           "equalize to the free or diffuse field first")

    s.text(450, 646, "presentation: monaural is one ear alone, diotic the "
           "same signal at both ears, binaural two independent ear signals",
           14, th.muted)
    s.text(450, 666, "a diotic sound is about 1.5 times as loud as the same "
           "sound at one ear (clause 8.1)", 14, th.muted)


def _d_tone_audibility_acquisition(s: SVG, th: Theme) -> None:
    """The two-level time structure ISO/PAS 20065 assessments are built on.

    Clause 4.3 merges the analyser's sub-second basic spectra line by line
    into spectra of about 3 s; clause 5.1 asks for at least 12 of those,
    time-staggered so that every alternating operating state is covered, and
    each yields one decisive audibility that Formula (20) energy-averages.
    """
    x0, x1 = 50.0, 850.0
    span = x1 - x0
    slots = 12
    slot = span / slots

    # --- 1: operating states over the record --------------------------------
    s.text(x0, 66, "1 — the source runs through its operating states "
           "(clause 5.1: all of them must be covered)", 16, th.fg, "start",
           bold=True)
    states = ((0.0, 0.25, "idle", th.primary), (0.25, 0.75, "full load",
              th.secondary), (0.75, 1.0, "idle", th.primary))
    for a, b, name, color in states:
        s.rect(x0 + a * span, 80, (b - a) * span, 42, th.panel, color, rx=6,
               sw=2)
        s.text(x0 + (a + b) / 2 * span, 107, name, 16, th.fg)

    s.line(x0, 140, x1, 140, th.muted, 1.4)
    for k in range(7):
        tx = x0 + span * k / 6.0
        s.line(tx, 136, tx, 144, th.muted, 1.4)
        s.text(tx, 162, f"{6 * k} s", 13, th.muted, mono=True)

    # --- 2: basic spectra merged into ~3 s averaged spectra -----------------
    s.text(x0, 196, "2 — the analyser's basic spectra (under 1 s each) are "
           "merged line by line into 3 s spectra (clause 4.3)", 16, th.fg,
           "start", bold=True)
    for j in range(slots):
        base = x0 + j * slot
        for k in range(4):
            tx = base + slot * (0.16 + 0.22 * k)
            s.line(tx, 208, tx, 232, th.muted, 2.6)
        s.rect(base + 3, 246, slot - 6, 46, th.panel, th.accent, rx=6, sw=1.8)
        s.text(base + slot / 2, 268, f"{j + 1}", 15, th.fg, bold=True)
        s.text(base + slot / 2, 285, "3 s", 12, th.muted)
        s.line(base + slot / 2, 232, base + slot / 2, 244, th.muted, 1.2)

    s.text(x0, 310, "3 — each merged spectrum gives one decisive audibility "
           "$ΔL_j$ (clause 5.3.8)", 16, th.fg, "start", bold=True)

    # --- 3: the energy mean of the decisive audibilities --------------------
    s.path(f"M {x0} 326 L {x1} 326 L {x1 - 170} 358 L {x0 + 170} 358 Z",
           fill=th.panel, stroke=th.muted, sw=1.4)
    s.rect(x0 + 170, 356, span - 340, 52, "none", th.primary, rx=8, sw=2.2)
    s.text(450, 379, "Energy mean of the $J$ decisive audibilities", 16, th.fg,
           bold=True)
    s.text(450, 399, "Formula (20); an empty spectrum counts as −10 dB "
           "(Formula 21)", 13, th.muted)
    s.arrow(450, 408, 450, 432, th.fg, 1.8)
    s.rect(x0 + 190, 432, span - 380, 50, "none", th.accent, rx=8, sw=2.4)
    s.text(450, 453, "mean audibility $ΔL$  →  tonal adjustment $K_t$", 17, th.fg,
           bold=True)
    s.text(450, 473, "ISO 1996-2:2017 Annex J, Table J.1", 13, th.muted)

    # --- The settings and acceptance rules ----------------------------------
    s.rect(x0, 502, span, 92, th.panel, th.muted, rx=10, sw=1.6)
    left = (
        "class 1 chain (IEC 61672-1), lower limit ≤ 20 Hz",
        "line spacing $Δf$ between 1.9 Hz and 4.0 Hz",
        "Hanning window, mandatory",
    )
    right = (
        "amplitude resolution ≥ 0.1 dB, anti-aliasing filter",
        "A-weighted spectrum (clause 5.3.2)",
        "$U ≤ 1.5$ dB: below 12 spectra, $U$ must be reported",
    )
    for k, txt in enumerate(left):
        s.circle(x0 + 22, 528 + k * 22, 3.6, th.primary)
        s.text(x0 + 36, 533 + k * 22, txt, 14, th.muted, "start")
    for k, txt in enumerate(right):
        s.circle(x0 + 424, 528 + k * 22, 3.6, th.primary)
        s.text(x0 + 438, 533 + k * 22, txt, 14, th.muted, "start")


def _d_dosimeter(s: SVG, th: Theme) -> None:
    """ISO 9612 occupational exposure: worn-dosimeter microphone position
    (Clause 12.3) and the three measurement strategies (Clauses 9-11)."""
    # --- Left: worker with a shoulder-mounted personal exposimeter ---------
    s.text(195, 84, "Worn instrument (Clause 12.3)", 21, th.fg, bold=True)
    gy = 560.0
    s.ground(gy, 40, 330)
    px = 150.0
    s.person(px, gy, 300)
    head_y = gy - 300 + 30.0            # head-circle centre
    sh_y = gy - 300 * 0.75              # shoulder joint (arm attachment)

    # Microphone capsule ~0.04 m above the shoulder, on the most-exposed side.
    mx = px + 46.0
    cap_y = sh_y - 30.0
    s.line(px + 6, sh_y - 6, mx + 12, sh_y + 6, th.muted, 2.4)  # shoulder slope
    s.rect(mx - 5, cap_y, 10, 14, th.fg, rx=3)                  # capsule
    s.line(mx, cap_y + 14, mx, sh_y, th.primary, 2.2)           # stub mount
    # Cable from the capsule mount to the body-worn meter.
    s.path(f"M {mx:.0f} {sh_y:.0f} C {mx + 26:.0f} {sh_y + 56:.0f} "
           f"{px + 40:.0f} {gy - 130:.0f} {px + 26:.0f} {gy - 116:.0f}",
           stroke=th.muted, sw=1.6)
    s.rect(px + 12, gy - 118, 30, 44, th.panel, th.primary, rx=5, sw=2)
    s.circle(px + 27, gy - 104, 3.5, th.primary)
    s.text(185, gy + 44, "Personal sound exposure meter", 19, th.fg)
    s.text(185, gy + 68, "(IEC 61252)", 17, th.muted)

    # Dimension: capsule height above the shoulder.
    s.dim(mx + 44, sh_y, mx + 44, cap_y, "≈ 0.04 m", offset=0, size=18,
          label_side="right")
    s.line(mx + 5, cap_y, mx + 44, cap_y, th.muted, 0.9, dash="3,3")
    s.line(mx + 12, sh_y + 2, mx + 44, sh_y, th.muted, 0.9, dash="3,3")
    s.text(mx + 53, sh_y + 22, "above the shoulder", 15, th.muted, "start")
    # Distance to the ear-canal entrance.
    s.line(px + 24, head_y + 8, mx - 4, cap_y + 4, th.secondary, 1.4,
           dash="5,4")
    s.text(px, head_y - 82, "≥ 0.1 m from the ear canal,", 17,
           th.secondary)
    s.text(px, head_y - 62, "most-exposed side", 17, th.secondary)

    # --- Right: the three sampling strategies as day timelines -------------
    s.text(620, 84, "Measurement strategies (Clauses 9–11)", 22, th.fg,
           bold=True)
    x0, x1 = 390.0, 850.0
    bw = x1 - x0
    ax_y = 132.0
    s.line(x0, ax_y, x1, ax_y, th.muted, 1.4)
    for hh in range(0, 9, 2):
        tx = x0 + bw * hh / 8.0
        s.line(tx, ax_y - 4, tx, ax_y + 4, th.muted, 1.4)
        s.text(tx, ax_y + 22, f"{hh} h", 15, th.muted, mono=True)
    s.text(620, ax_y - 12, "Working day", 17, th.muted)

    def strip(y: float, title: str, caption: str) -> None:
        s.text(x0, y - 10, title, 19, th.fg, "start", bold=True)
        s.text(x0, y + 68, caption, 16, th.muted, "start", italic=True)

    # Strategy 1: task-based; the day split into tasks, >= 3 samples each.
    y1 = 190.0
    strip(y1, "Task-based (Clause 9)",
          "split the day into tasks — ≥ 3 samples (│) per task, plus each duration")
    edges = [0.0, 0.1875, 0.8125, 1.0]      # the Annex D welder: 1.5 h / 5 h / 1.5 h
    cols = [th.accent, th.primary, th.secondary]
    for k in range(3):
        xa, xb = x0 + bw * edges[k], x0 + bw * edges[k + 1]
        s.rect(xa, y1, xb - xa, 44, th.panel, cols[k], rx=6, sw=2)
        s.text((xa + xb) / 2, y1 + 27, f"Task {k + 1}", 17, th.fg)
        for frac in (0.25, 0.5, 0.75):
            sx = xa + (xb - xa) * frac
            s.line(sx, y1 + 34, sx, y1 + 42, cols[k], 2.2)

    # Strategy 2: job-based; random samples over the homogeneous group.
    y2 = 300.0
    strip(y2, "Job-based (Clause 10)",
          "$N ≥ 5$ random samples over the homogeneous exposure group")
    s.rect(x0, y2, bw, 44, "none", th.muted, rx=6, sw=1.6, dash="5,4")
    for frac in (0.05, 0.24, 0.46, 0.65, 0.86):
        s.rect(x0 + bw * frac, y2 + 6, bw * 0.06, 32, th.panel, th.primary,
               rx=4, sw=2)

    # Strategy 3: full-day; the whole shift, repeated on several days.
    y3 = 410.0
    strip(y3, "Full-day (Clause 11)",
          "the whole shift, at least 3 times (5 if the days differ by > 3 dB)")
    s.rect(x0, y3, bw, 24, th.panel, th.primary, rx=6, sw=2)
    s.text(x0 + bw / 2, y3 + 17, "day 1", 14, th.fg)
    s.rect(x0 + 8, y3 + 30, bw - 16, 7, th.panel, th.primary, rx=3, sw=1.2)
    s.rect(x0 + 16, y3 + 43, bw - 32, 7, th.panel, th.primary, rx=3, sw=1.2)

    # All three land in the same deliverable. LEX,8h stays plain: same
    # upright EX,8h subscript as on the ISO 1999 plate, not composable
    # while the roman list has no run for the "8h" unit inside a script.
    s.text(620, 520, "choose by work pattern (Table B.1)  →  LEX,8h + Annex C uncertainty",
           17, th.fg)


# ---------------------------------------------------------------------------
# Sound-quality metric family (DIN 45692 + ECMA-418-2)
# ---------------------------------------------------------------------------

def _d_sound_quality(s: SVG, th: Theme) -> None:
    """One calibrated signal into the two auditory front ends and the four
    sound-quality metrics of the guide, each with its reference sound and
    the value the library returns for it (1.00 acum, 1.000 tu_HMS,
    0.9999 asper, 0.9957 vacil_HMS)."""
    # Input signal
    s.rect(230, 52, 440, 56, th.panel, th.fg, rx=10, sw=2)
    s.text(450, 76, "Calibrated signal $x(t)$ in pascals", 16, th.fg, bold=True)
    s.text(450, 97, "any sample rate: each metric resamples to 48 kHz "
           "internally", 12, th.muted)

    # Two auditory front ends
    s.rect(60, 148, 270, 56, th.panel, th.primary, rx=10, sw=2)
    s.text(195, 172, "Specific loudness $N′(z)$", 15, th.fg, bold=True)
    s.text(195, 192, "Zwicker pattern over 24 Bark", 12, th.muted)
    s.rect(390, 148, 450, 56, th.panel, th.primary, rx=10, sw=2)
    s.text(615, 172, "Sottek Hearing Model front end (ECMA-418-2)", 15,
           th.fg, bold=True)
    # The Bark_HMS, tu_HMS and vacil_HMS units keep their plain spelling in
    # this diagram: ECMA-418-2 prints the HMS subscript upright, and the
    # curated roman list does not carry HMS yet.
    s.text(615, 192, "outer/middle-ear filter + 53 auditory bands "
           "(Bark_HMS)", 12, th.muted)
    s.arrow(350, 108, 210, 144, th.fg, 1.8)
    s.arrow(550, 108, 605, 144, th.fg, 1.8)

    # The four metric boxes
    metrics = (
        (42.0, "Sharpness $S$", "DIN 45692",
         "$g(z)$-weighted first moment", "of $N′(z)$, with $k$ = 0.108",
         ("critical-band-wide noise", "at 1 kHz, 60 dB"),
         "→ $S$ = 1.00 acum"),
        (262.0, "Tonality $T$", "ECMA-418-2 clause 6",
         "band autocorrelation finds", "periodic components",
         ("1 kHz tone at 40 dB",), "→ $T$ = 1.000 tu_HMS (999 Hz)"),
        (482.0, "Roughness $R$", "ECMA-418-2 clause 7",
         "fast envelope modulation,", "band-pass peaking near 70 Hz",
         ("1 kHz, 100 % AM at 70 Hz, 60 dB",), "→ $R$ = 0.9999 asper"),
        (702.0, "Fluctuation strength $F$", "ECMA-418-2 clause 9 (HSA)",
         "slow envelope modulation,", "band-pass peaking near 4 Hz",
         ("1 kHz, 100 % AM at 4 Hz, 60 dB",), "→ $F$ = 0.9957 vacil_HMS"),
    )
    for x0, name, std, m1, m2, refs, val in metrics:
        cx = x0 + 98.0
        s.rect(x0, 248, 196, 128, th.panel, th.secondary, rx=10, sw=2)
        s.text(cx, 271, name, 13, th.fg, bold=True)
        s.text(cx, 289, std, 11, th.muted)
        s.text(cx, 308, m1, 11, th.muted)
        if len(refs) == 1:
            s.text(cx, 324, m2, 11, th.muted)
            s.text(cx, 345, refs[0], 11, th.fg)
        else:
            s.text(cx, 322, m2, 11, th.muted)
            s.text(cx, 337, refs[0], 11, th.fg)
            s.text(cx, 351, refs[1], 11, th.fg)
        s.text(cx, 363, val, 12, th.secondary, bold=True)
    s.arrow(195, 204, 141, 244, th.fg, 1.8)
    for xt in (360.0, 580.0, 800.0):
        s.arrow(615, 204, xt, 244, th.fg, 1.8)

    # Downstream combination note
    s.rect(130, 412, 640, 68, "none", th.accent, rx=10, sw=1.6, dash="6,5")
    s.text(450, 439, "Downstream, the sensations combine into annoyance",
           15, th.accent, bold=True)
    s.text(450, 463, "$N_5$, $S$, $R$ and $F$ feed the Fastl and Zwicker "
           "psychoacoustic annoyance $PA = N_5·(1 + √(w_S^2 + w_{FR}^2))$", 12,
           th.fg)


# ---------------------------------------------------------------------------
# Tone audibility (ISO/PAS 20065 -> ISO 1996-2 Annex J)
# ---------------------------------------------------------------------------

def _d_tone_audibility(s: SVG, th: Theme) -> None:
    """The engineering-method chain on the Annex E combustion-engine
    spectrum: critical band, LS/LT, masking threshold and the 5.01 dB
    decisive audibility, closing on the Kt = 4 dB tonal adjustment."""
    cx, bw = 450.0, 620.0
    x0 = cx - bw / 2

    def step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, 58, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 25, l1, 16, th.fg, bold=True)
        s.text(cx, y + 45, l2, 12, th.muted)

    step(52, "Narrow-band FFT spectrum — line spacing $Δf$ = 2.7 Hz",
         "Annex E engine spectrum; peak detected at $f_T$ = 137.3 Hz (not on "
         "a slope)", th.fg)
    step(138, "Critical band about the tone — $Δf_c$ = 101.36 Hz",
         "geometric placement: corners 95.67 and 197.04 Hz, $√(f_1·f_2) = f_T$",
         th.primary)
    step(224, "Levels from the spectrum lines in the band",
         "masking noise $L_S$ = 49.22 dB (iterative mean); tone $L_T$ = 67.96 dB "
         "(energy sum)", th.primary)
    step(310, "Masking threshold seen by the ear",
         "$L_G = L_S + 10·log_{10}(Δf_c/Δf)$ = 64.97 dB;  masking index "
         "$a_v$ = −2.02 dB", th.primary)
    s.rect(x0, 396, bw, 60, "none", th.accent, rx=10, sw=2.4)
    s.text(cx, 421, "Audibility $ΔL = L_T − L_G − a_v$ = 5.01 dB", 17, th.fg,
           bold=True)
    s.text(cx, 443, "the largest $ΔL$ of the nine tones: the decisive "
           "audibility of this spectrum", 12, th.muted)
    for y0, y1 in ((110, 134), (196, 220), (282, 306), (368, 392),
                   (456, 484)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)

    s.rect(130, 488, 640, 68, "none", th.secondary, rx=10, sw=1.6,
           dash="6,5")
    s.text(cx, 515, "From audibility to penalty (ISO 1996-2:2017 Annex J)",
           15, th.secondary, bold=True)
    s.text(cx, 539, "energy mean of the five spectra $ΔL$ = 6.98 dB → tonal "
           "adjustment $K_t$ = 4 dB (Table J.1)", 13, th.fg)


# ---------------------------------------------------------------------------
# Psychoacoustic annoyance (Fastl & Zwicker Eqs 16.2-16.4)
# ---------------------------------------------------------------------------

def _d_psychoacoustic_annoyance(s: SVG, th: Theme) -> None:
    """The four sensations of the guide's worked example (N5 = 30 sone,
    S = 2.0 acum, F = 0.5 vacil, R = 0.3 asper) through the two weightings
    (wS = 0.1001, wFR = 0.2125) into PA = 37.05."""
    inputs = (
        (42.0, "$S$ = 2.0 acum", "sharpness (DIN 45692)",
         "counts only above 1.75 acum"),
        (262.0, "$N_5$ = 30 sone", "percentile loudness (ISO 532-1)",
         "exceeded 5 % of the time"),
        (482.0, "$F$ = 0.5 vacil", "fluctuation strength",
         "slow modulation, ≈ 4 Hz"),
        (702.0, "$R$ = 0.3 asper", "roughness", "fast modulation, ≈ 70 Hz"),
    )
    for x0, name, s1, s2 in inputs:
        cx = x0 + 98.0
        s.rect(x0, 60, 196, 72, th.panel, th.primary, rx=10, sw=2)
        s.text(cx, 84, name, 14, th.fg, bold=True)
        s.text(cx, 103, s1, 11, th.muted)
        s.text(cx, 120, s2, 11, th.muted)

    # Weighting boxes: wS takes S and N5; wFR takes N5, F and R.
    s.rect(90, 204, 340, 86, th.panel, th.fg, rx=10, sw=2)
    s.text(260, 230, "Sharpness weighting $w_S$ = 0.1001", 15, th.fg,
           bold=True)
    s.text(260, 252, "$w_S = (S − 1.75) · 0.25 · log_{10}(N_5 + 10)$", 13,
           th.primary)
    s.text(260, 274, "zero for $S ≤ 1.75$ acum", 12, th.muted)
    s.rect(470, 204, 340, 86, th.panel, th.fg, rx=10, sw=2)
    s.text(640, 230, "Roughness and fluctuation $w_{FR}$ = 0.2125", 15, th.fg,
           bold=True)
    s.text(640, 252, "$w_{FR} = 2.18 / N_5^{0.4} · (0.4·F + 0.6·R)$", 13,
           th.primary)
    s.text(640, 274, "roughness weighs more: 0.6 against 0.4", 12,
           th.muted)
    s.arrow(141, 132, 200, 200, th.fg, 1.8)
    s.arrow(330, 132, 285, 200, th.fg, 1.8)
    s.arrow(395, 132, 520, 200, th.fg, 1.8)
    s.arrow(581, 132, 610, 200, th.fg, 1.8)
    s.arrow(800, 132, 690, 200, th.fg, 1.8)

    # Combination
    s.rect(200, 344, 500, 72, "none", th.accent, rx=10, sw=2.4)
    s.text(450, 374, "$PA = N_5 · (1 + √(w_S^2 + w_{FR}^2))$ = 37.05", 18, th.fg,
           bold=True)
    s.text(450, 399, "Fastl and Zwicker Eq. 16.2 (origin Widmann 1992)",
           12, th.muted)
    s.arrow(260, 290, 380, 340, th.fg, 1.8)
    s.arrow(640, 290, 520, 340, th.fg, 1.8)

    s.text(450, 464, "a neutral sound ($S ≤ 1.75$ acum, $F = R = 0$) sits on "
           "the baseline $PA = N_5$", 14, th.fg)
    s.text(450, 488, "sharpness, roughness and fluctuation only ever lift "
           "the annoyance above the loudness", 13, th.muted)


# ---------------------------------------------------------------------------
# Objective intelligibility (STOI / ESTOI)
# ---------------------------------------------------------------------------

def _d_objective_intelligibility(s: SVG, th: Theme) -> None:
    """The shared STOI/ESTOI front end, the split into the two intermediate
    correlations, and the guide's measured example: STOI = 0.727 for
    speech-like material in a flat masker at 0 dB SNR."""
    cx, bw = 450.0, 600.0
    x0 = cx - bw / 2

    def step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, 58, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 25, l1, 16, th.fg, bold=True)
        s.text(cx, y + 45, l2, 12, th.muted)

    step(52, "Clean reference $x(t)$ and degraded version $y(t)$",
         "the guide's example: speech-like material in a flat masker at "
         "0 dB SNR", th.fg)
    step(138, "Resample to 10 kHz and drop the silent frames",
         "frames 40 dB below the loudest clean frame carry no "
         "intelligibility", th.primary)
    step(224, "Short-time DFT: 256-sample Hann frames, 50 % overlap",
         "magnitudes grouped into 15 one-third-octave bands from 150 Hz",
         th.primary)
    step(310, "384 ms segments — 30 frames, the unit of comparison",
         "long enough to hold the slow modulations that carry speech",
         th.primary)
    for y0, y1 in ((110, 134), (196, 220), (282, 306)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)

    # The two intermediate measures
    s.rect(150, 404, 285, 76, th.panel, th.secondary, rx=10, sw=2)
    s.text(292, 428, "STOI: envelope correlation", 13, th.fg, bold=True)
    s.text(292, 447, "per band and segment; normalise,", 11, th.muted)
    s.text(292, 464, "clip at −15 dB, then average", 11, th.muted)
    s.rect(465, 404, 285, 76, th.panel, th.secondary, rx=10, sw=2)
    s.text(607, 428, "ESTOI: spectral correlation", 13, th.fg, bold=True)
    s.text(607, 447, "row- and column-normalised segments;", 11, th.muted)
    s.text(607, 464, "credits glimpses in modulated maskers", 11, th.muted)
    s.arrow(450, 368, 300, 400, th.fg, 1.8)
    s.arrow(450, 368, 600, 400, th.fg, 1.8)

    s.rect(x0, 516, bw, 60, "none", th.accent, rx=10, sw=2.4)
    s.text(cx, 541, "STOI = 0.727 for the example", 17, th.fg, bold=True)
    s.text(cx, 563, "the lowest band keeps 0.27 of the correlation; above "
           "1.9 kHz it reaches 0.90", 12, th.muted)
    s.arrow(292, 480, 390, 512, th.fg, 1.8)
    s.arrow(607, 480, 510, 512, th.fg, 1.8)


# ---------------------------------------------------------------------------
# Sound-field audiometry: what the ISO 389-7 reference zero is referenced to
# (ISO 389-7:2005 clause 1 + ISO 8253-2:2009 clauses 3.3, 5.2, 5.3)
# ---------------------------------------------------------------------------

def _d_soundfield_audiometry(s: SVG, th: Theme) -> None:
    """The three listening conditions behind the audiometric zero.

    Panels A and B are the two ISO 389-7 columns, drawn to the ISO 8253-2
    geometry that qualifies them; panel C is the earphone condition the
    values are *not* referenced to, which is the mistake the guide warns
    about.
    """
    gy = 400.0                       # common floor line
    pw = 282.0                       # panel width
    px0 = (24.0, 309.0, 594.0)       # panel left edges
    scale = 112.0                    # px per metre
    note_y = (424.0, 442.0, 460.0)   # the three note lines under each panel

    def room(x: float, title: str, sub: str, color: str) -> None:
        s.rect(x, 112, pw, gy - 112, "none", th.muted, rx=6, sw=1.4)
        s.text(x + pw / 2, 66, title, 21, color, bold=True)
        s.text(x + pw / 2, 90, sub, 14, th.muted)

    def notes(x: float, color: str, *rows: str) -> None:
        for y, row in zip(note_y, rows):
            s.text(x + pw / 2, y, row, 13, color)

    def ghost_subject(x: float) -> float:
        """The listener who will sit there, drawn absent (5.2 b / 5.3).

        Everything about the qualification is measured with the subject and
        the chair out of the room, so both are dashed and the microphone that
        replaces them is solid.
        """
        ref_y = gy - 190.0 + 22.0     # head centre = ear-canal midpoint
        seat_y = gy - 78.0
        d = "6,4"
        s.line(x - 28, seat_y, x + 62, seat_y, th.muted, 2.0, dash=d)
        s.line(x - 24, seat_y, x - 24, gy, th.muted, 2.0, dash=d)
        s.line(x + 58, seat_y, x + 58, gy, th.muted, 2.0, dash=d)
        s.line(x - 28, seat_y, x - 28, ref_y + 16, th.muted, 2.0, dash=d)
        s.ellipse(x, ref_y, 22, 22, "none", th.muted, 2.0, dash=d)
        s.line(x, ref_y + 22, x + 6, seat_y, th.muted, 2.0, dash=d)
        s.line(x + 6, seat_y, x + 52, seat_y, th.muted, 2.0, dash=d)
        s.line(x + 52, seat_y, x + 52, gy, th.muted, 2.0, dash=d)
        q = 0.15 * scale
        s.rect(x - q, ref_y - q, 2 * q, 2 * q, "none", th.accent, rx=3,
               sw=1.4, dash="5,4")
        s.circle(x, ref_y, 4.0, th.secondary)
        return ref_y

    def meas_mic(x: float, y: float) -> None:
        """Boom-mounted measurement microphone, capsule tip at (x, y)."""
        s.line(x - 64, y, x - 42, y, th.primary, 2.4)      # boom arm
        s.rect(x - 42, y - 6, 36, 12, th.fg, rx=4)         # body
        s.circle(x, y, 5.0, th.primary)                    # capsule tip
        s.line(x - 64, y, x - 64, gy, th.fg, 2.2)          # stand
        s.line(x - 78, gy, x - 50, gy, th.fg, 2.2)         # base

    # ---------------- A - free sound field (ISO 8253-2, 5.2) --------------
    xa = px0[0]
    room(xa, "A · Free field", "pure tone · frontal · binaural", th.primary)
    s.ground(gy, xa, xa + pw)
    sx = xa + 96.0
    ref_y = ghost_subject(sx)
    lx = sx + scale                    # loudspeaker reference point
    s.rect(lx - 20, ref_y - 42, 40, 84, th.panel, th.primary, rx=5, sw=2)
    s.circle(lx, ref_y, 14, "none", th.primary, 2)
    s.circle(lx, ref_y, 5, th.primary)
    s.line(lx, ref_y + 42, lx, gy, th.fg, 2.2)
    s.line(lx - 18, gy, lx + 18, gy, th.fg, 2.2)
    s.line(sx + 36, ref_y, lx - 24, ref_y, th.primary, 1.2, dash="7,4")
    s.dim(sx, ref_y - 40, lx, ref_y - 40, "≥ 1 m", offset=-52, size=17)
    meas_mic(sx, ref_y)
    s.text(xa + 104, 148, "level measured here,", 13, th.primary)
    s.text(xa + 104, 166, "subject and chair absent", 13, th.primary)
    notes(xa, th.accent, "on the reference axis, 0° azimuth",
          "and elevation, ≥ 1 m  (5.2 a)",
          "± 0.15 m: within ± 1 dB to 4 kHz  (5.2 b)")

    # ---------------- B - diffuse sound field (ISO 8253-2, 5.3) -----------
    xb = px0[1]
    room(xb, "B · Diffuse field", "third-octave noise band · binaural",
         th.accent)
    s.ground(gy, xb, xb + pw)
    sx = xb + pw / 2 + 20.0
    ref_y = ghost_subject(sx)
    for dx, dy in ((-112.0, -96.0), (108.0, -100.0), (104.0, 62.0),
                   (-108.0, 58.0)):
        bx, by = sx + dx, ref_y + dy
        s.rect(bx - 16, by - 12, 32, 24, th.panel, th.accent, rx=4, sw=1.8)
        s.arrow(bx - 0.16 * dx, by - 0.16 * dy,
                sx - 0.34 * dx, ref_y - 0.34 * dy, th.accent, 1.2)
    meas_mic(sx, ref_y)
    s.text(xb + pw / 2, 148, "the same reference point,", 13, th.primary)
    s.text(xb + pw / 2, 166, "the same absent subject", 13, th.primary)
    notes(xb, th.accent, "several loudspeakers, non-coherent feeds",
          "≥ 500 Hz: loudest and quietest directions",
          "within 5 dB  (Table 1)")

    # ---------------- C - the earphone condition, for contrast ------------
    xc = px0[2]
    room(xc, "C · Earphone — not this standard",
         "supra-aural or insert · monaural", th.secondary)
    s.ground(gy, xc, xc + pw)
    sx = xc + 74.0
    s.person(sx, gy, 190.0, seated=True)
    ref_y = gy - 190.0 + 19.0
    s.path(f"M {sx - 20:.0f} {ref_y - 8:.0f} Q {sx:.0f} {ref_y - 42:.0f} "
           f"{sx + 20:.0f} {ref_y - 8:.0f}", stroke=th.secondary, sw=2.6)
    s.rect(sx + 15, ref_y - 13, 13, 26, th.panel, th.secondary, rx=3, sw=2)
    cx0 = sx + 96.0
    s.rect(cx0, ref_y - 36, 104, 72, th.panel, th.secondary, rx=8, sw=2)
    s.text(cx0 + 52, ref_y - 12, "IEC 60318-1", 14, th.fg, bold=True)
    s.text(cx0 + 52, ref_y + 8, "coupler /", 13, th.muted)
    s.text(cx0 + 52, ref_y + 26, "ear simulator", 13, th.muted)
    s.arrow(sx + 32, ref_y, cx0 - 6, ref_y, th.secondary, 1.6)
    s.text(xc + pw / 2, 148, "the level lives in the coupler,", 13,
           th.secondary)
    s.text(xc + pw / 2, 166, "not at a point in a room", 13, th.secondary)
    notes(xc, th.secondary, "0 dB HL here is the RETSPL of the earphone",
          "fitted (ISO 389-1 / -2 / -8), referred to a",
          "coupler — never an ISO 389-7 value")

    # ---------------- shared caption strip --------------------------------
    s.rect(24, 486, 852, 68, "none", th.fg, rx=10, sw=1.4, dash="6,5")
    s.text(450, 514, "Reference point: the midpoint of the line joining the "
           "listener's ear-canal openings", 16, th.fg, bold=True)
    s.text(450, 540, "the listener in the listening position; in A and B the "
           "level is measured there with the subject and chair absent", 14,
           th.muted)
# ---------------------------------------------------------------------------
# ISO 9612 Clause 12.4: the sound level meter geometry at a workstation
# ---------------------------------------------------------------------------

def _d_slm_workstation(s: SVG, th: Theme) -> None:
    """The other half of the ISO 9612 microphone geometry.

    ``diagram_dosimeter_iso9612`` draws the worn instrument of Clause 12.3;
    the guide also offers class 1 and class 2 sound level meters, whose
    placement is Clause 12.4 and is a different drawing entirely: worker
    absent on the left, worker present on the right.
    """
    gy = 400.0
    m = 130.0                        # px per metre
    pw = 418.0
    xa, xb = 24.0, 458.0
    note_y = (424.0, 442.0, 460.0)

    def panel(x: float, title: str, sub: str, color: str) -> None:
        s.rect(x, 104, pw, gy - 104, "none", th.muted, rx=6, sw=1.4)
        s.text(x + pw / 2, 66, title, 22, color, bold=True)
        s.text(x + pw / 2, 90, sub, 15, th.muted)

    def notes(x: float, color: str, *rows: str) -> None:
        for y, row in zip(note_y, rows):
            s.text(x + pw / 2, y, row, 14, color)

    def machine(x: float) -> None:
        s.rect(x, gy - 118, 88, 118, th.panel, th.fg, rx=6, sw=2)
        s.text(x + 44, gy - 54, "machine", 15, th.fg)
        for k in range(3):
            s.path(f"M {x + 96 + k * 15:.0f} {gy - 100:.0f} "
                   f"q 8 11 0 22 q -8 11 0 22", stroke=th.secondary, sw=1.6)

    # ---------------- Left: worker absent ---------------------------------
    panel(xa, "Worker absent", "the preferred Clause 12.4 placement",
          th.primary)
    s.ground(gy, xa, xa + pw)
    machine(xa + 20)
    mx = xa + 276.0
    ref = gy - 1.55 * m              # 1.55 m: the standing head height
    s.ellipse(mx, ref, 22, 26, "none", th.muted, 1.8, dash="6,4")
    s.line(mx, ref + 26, mx, gy, th.muted, 1.8, dash="6,4")
    s.mic(mx, ref, gy, 1.0)
    s.arrow(mx - 26, ref, mx - 104, ref, th.primary, 2.0)
    s.text(mx - 64, ref + 24, "axis ∥ line of sight", 14, th.primary)
    s.dim(mx + 68, gy, mx + 68, ref, "1.55 m", offset=0, size=17,
          label_side="right")
    s.line(mx + 10, ref, mx + 68, ref, th.muted, 0.9, dash="3,3")
    # Plan inset: the constant-speed sweep along an infinity-shaped path.
    ix, iy = xa + 22.0, 126.0
    s.rect(ix, iy, 148, 84, th.panel, th.accent, rx=8, sw=1.6)
    s.text(ix + 74, iy + 20, "or sweep in plan:", 14, th.accent)
    s.path(f"M {ix + 32:.0f} {iy + 54:.0f} c 10 -26 40 -26 50 0 "
           f"c 10 26 40 26 50 0 c -10 -26 -40 -26 -50 0 "
           f"c -10 26 -40 26 -50 0 z", stroke=th.accent, sw=2.2)
    s.text(ix + 74, iy + 78, "at constant speed", 12, th.muted)
    notes(xa, th.primary,
          "capsule at the head position, on the eye line",
          "standing 1.55 m ± 0.075 m; seated 0.80 m ± 0.05 m",
          "above the middle of the seat plane")

    # ---------------- Right: worker present -------------------------------
    panel(xb, "Worker present", "hand-held meter, most-exposed ear",
          th.secondary)
    s.ground(gy, xb, xb + pw)
    machine(xb + 18)
    px = xb + 242.0
    ph = 1.75 * m
    s.person(px, gy, ph)
    ear_x, ear_y = px + 0.10 * ph, gy - ph + 0.10 * ph
    hx = ear_x + 0.25 * m            # 0.25 m: the middle of the allowed band
    s.circle(hx, ear_y, 15, "none", th.secondary, 2.2)   # 60 mm windscreen
    s.rect(hx - 5, ear_y + 13, 10, 28, th.panel, th.secondary, rx=3, sw=2)
    s.line(hx, ear_y + 41, px + 0.18 * ph, gy - 0.50 * ph, th.secondary, 2.4)
    s.dim(ear_x, ear_y - 34, hx, ear_y - 34, "0.1 m to 0.4 m", offset=-16,
          size=16)
    s.line(ear_x, ear_y - 12, ear_x, ear_y - 50, th.muted, 0.9, dash="3,3")
    s.line(hx, ear_y - 17, hx, ear_y - 50, th.muted, 0.9, dash="3,3")
    s.line(hx + 15, ear_y, xb + pw - 24, ear_y + 62, th.secondary, 1.0,
           dash="4,3")
    s.text(xb + pw - 20, ear_y + 68, "60 mm", 14, th.secondary, "end")
    s.text(xb + pw - 20, ear_y + 86, "windscreen", 14, th.secondary,
           "end")
    notes(xb, th.secondary,
          "held 0.1 m to 0.4 m from the ear-canal entrance",
          "on the most exposed side, windscreen ≥ 60 mm (13.3)",
          "beyond 0.4 m, use the worn instrument (12.3)")

    # ---------------- footer ---------------------------------------------
    s.rect(24, 486, 852, 66, "none", th.fg, rx=10, sw=1.4, dash="6,5")
    s.text(450, 514, "A fixed microphone position under-reads a hand-held "
           "tool close to the ear (13.1)", 16, th.fg, bold=True)
    s.text(450, 540, "that is exactly when the worn personal exposure meter "
           "of Clause 12.3 is the right instrument", 14, th.muted)
# ---------------------------------------------------------------------------
# IEC 60268-16 clause 7: the physical STI measurement (source, level, receiver)
# ---------------------------------------------------------------------------

def _d_sti_setup(s: SVG, th: Theme) -> None:
    """What produces the signal the STI is computed from.

    ``diagram_sti_chain`` draws the analysis; this draws the room. Panel A is
    an unamplified talker (acoustical input, clause 7.2), panel B a sound
    system driven electrically (clause 7.4), and both end on the clause 7.6.4
    reduction rule for a set of positions.
    """
    gy = 400.0
    m = 90.0                         # px per metre
    pw = 418.0
    xa, xb = 24.0, 458.0
    note_y = (424.0, 442.0, 460.0)

    def room(x: float, title: str, sub: str, color: str) -> None:
        s.rect(x, gy - 3.0 * m, pw, 3.0 * m, "none", th.muted, rx=6, sw=1.4)
        s.text(x + pw / 2, 66, title, 22, color, bold=True)
        s.text(x + pw / 2, 90, sub, 15, th.muted)
        s.ground(gy, x, x + pw)

    def notes(x: float, color: str, *rows: str) -> None:
        for y, row in zip(note_y, rows):
            s.text(x + pw / 2, y, row, 14, color)

    def receiver(x: float, *, ghost: bool = False) -> None:
        """Measurement microphone at 1.2 m seated ear height."""
        cap = gy - 1.2 * m
        if ghost:
            s.ellipse(x, cap + 24, 9, 26, "none", th.muted, 1.4, dash="5,4")
            s.line(x, cap + 50, x, gy, th.muted, 1.4, dash="5,4")
        else:
            s.mic(x, cap, gy, 1.0)

    # ---------------- A - unamplified talker (clause 7.2) -----------------
    room(xa, "A · Unamplified talker", "acoustical input, clause 7.2",
         th.primary)
    tx = xa + 74.0
    mouth = gy - 1.5 * m             # 1.5 m: a standing talker's mouth
    # Artificial mouth on a stand, aimed along the speaking direction.
    s.rect(tx - 16, mouth - 20, 32, 40, th.panel, th.primary, rx=6, sw=2)
    s.path(f"M {tx + 16:.0f} {mouth - 14:.0f} L {tx + 40:.0f} {mouth - 24:.0f} "
           f"L {tx + 40:.0f} {mouth + 24:.0f} L {tx + 16:.0f} {mouth + 14:.0f} Z",
           fill=th.panel, stroke=th.primary, sw=2)
    s.line(tx, mouth + 20, tx, gy, th.fg, 2.2)
    s.line(tx - 16, gy, tx + 16, gy, th.fg, 2.2)
    s.text(tx + 34, mouth - 52, "artificial mouth", 14, th.primary)
    s.text(tx + 34, mouth - 36, "(ITU-T P.51 directivity)", 12, th.muted)
    s.dim(tx - 44, gy, tx - 44, mouth, "1.5 m", offset=0, size=15)
    s.line(tx - 18, mouth, tx - 44, mouth, th.muted, 0.9, dash="3,3")
    # The 1 m reference point where the fallback level is defined.
    s.line(tx + m, mouth - 34, tx + m, mouth + 34, th.secondary, 1.2,
           dash="5,4")
    s.dim(tx, mouth - 78, tx + m, mouth - 78, "1 m", offset=0, size=15)
    s.line(tx + m, mouth - 34, tx + m, mouth - 78, th.muted, 0.9, dash="3,3")
    s.text(tx + m + 8, mouth - 118, "60 dB(A) here, or the", 13, th.secondary,
           "start")
    s.text(tx + m + 8, mouth - 102, "Annex J operational level", 13,
           th.secondary, "start")
    # Receiver at the listener position, plus a second position further back.
    rx1 = tx + 2.0 * m
    receiver(rx1)
    receiver(tx + 3.3 * m, ghost=True)
    s.dim(tx, gy - 18, rx1, gy - 18, "2 m", offset=0, size=15)
    s.dim(rx1 + 26, gy, rx1 + 26, gy - 1.2 * m, "1.2 m", offset=0, size=15,
          label_side="right")
    s.line(rx1 + 6, gy - 1.2 * m, rx1 + 26, gy - 1.2 * m, th.muted, 0.9,
           dash="3,3")
    notes(xa, th.primary,
          "source response flat within ± 1 dB in a free field (7.2 b)",
          "receiver: omnidirectional, diffuse-field, calibrated (7.3)",
          "ambient noise measured at the same point, source off")

    # ---------------- B - sound system (clause 7.4) -----------------------
    room(xb, "B · Sound system", "electrical input, clause 7.4", th.accent)
    ceil = gy - 3.0 * m
    for k in range(3):
        lx = xb + 110.0 + k * 106.0
        s.rect(lx - 18, ceil + 4, 36, 20, th.panel, th.accent, rx=4, sw=2)
        s.path(f"M {lx - 26:.0f} {ceil + 56:.0f} L {lx:.0f} {ceil + 24:.0f} "
               f"L {lx + 26:.0f} {ceil + 56:.0f}", stroke=th.accent, sw=1.2,
               dash="5,4")
    s.text(xb + pw / 2, ceil + 42, "ceiling loudspeaker line", 14, th.accent)
    # Electrical injection into the system input, down at the rack.
    rack_y = gy - 104.0
    s.rect(xb + 18, rack_y, 62, 48, th.panel, th.fg, rx=6, sw=2)
    s.text(xb + 49, rack_y + 30, "amp", 14, th.fg)
    s.arrow(xb + 2, rack_y + 24, xb + 16, rack_y + 24, th.secondary, 2.0)
    s.text(xb + 6, rack_y + 68, "test signal in,", 12, th.secondary, "start")
    s.text(xb + 6, rack_y + 84, "at the Annex J level", 12, th.secondary,
           "start")
    s.line(xb + 32, rack_y, xb + 32, ceil + 14, th.fg, 1.4, dash="5,4")
    s.line(xb + 32, ceil + 14, xb + 92, ceil + 14, th.fg, 1.4, dash="5,4")
    # Two coverage zones, plus the ambient-noise position.
    for k, lx in enumerate((xb + 150.0, xb + 336.0)):
        receiver(lx)
        s.text(lx, gy - 1.2 * m - 30, f"zone {k + 1}", 13, th.accent)
    amb = xb + 246.0
    receiver(amb, ghost=True)
    s.text(amb, gy - 1.2 * m - 30, "system off:", 13, th.muted)
    s.text(amb, gy - 1.2 * m - 14, "ambient", 13, th.muted)
    notes(xb, th.accent,
          "injected near the normal input, so the whole chain is in (7.4)",
          "one position per coverage zone, at listening height",
          "spread over the served area, worst corners included")

    # ---------------- footer ---------------------------------------------
    s.rect(24, 486, 852, 66, "none", th.fg, rx=10, sw=1.4, dash="6,5")
    s.text(450, 514, "The rating of the space is the mean of the positions "
           "minus one standard deviation (7.6.4)", 16, th.fg, bold=True)
    s.text(450, 540, "a plain mean over the positions overstates coverage; "
           "better still, plot the whole distribution", 14, th.muted)
# ---------------------------------------------------------------------------
# STOI / ESTOI on a physical device: where the degraded signal comes from
# ---------------------------------------------------------------------------

def _d_stoi_bench(s: SVG, th: Theme) -> None:
    """The bench behind an acoustically captured degraded signal.

    ``diagram_objective_intelligibility`` draws the algorithm; this draws the
    room and the cables, because once the pair is captured rather than
    synthesised, four things become the caller's responsibility.
    """
    fy = 118.0                       # reference (upper) lane
    ly = 288.0                       # degraded (lower) lane

    def box(x: float, y: float, w: float, h: float, title: str, sub: str,
            color: str) -> None:
        s.rect(x, y, w, h, th.panel, color, rx=8, sw=2)
        if sub:
            s.text(x + w / 2, y + h / 2 - 4, title, 16, th.fg, bold=True)
            s.text(x + w / 2, y + h / 2 + 17, sub, 13, th.muted)
        else:
            s.text(x + w / 2, y + h / 2 + 6, title, 16, th.fg, bold=True)

    # --- the clean file, forking -----------------------------------------
    s.rect(26, ly - 46, 96, 92, th.panel, th.fg, rx=8, sw=2)
    s.text(74, ly - 8, "clean", 17, th.fg, bold=True)
    s.text(74, ly + 14, "speech file", 15, th.muted)
    s.path(f"M 122 {ly - 14:.0f} C 168 {ly - 14:.0f} 168 {fy + 26:.0f} "
           f"212 {fy + 26:.0f}", stroke=th.primary, sw=2.2)
    s.arrow(200, fy + 26, 214, fy + 26, th.primary, 2.2)
    s.arrow(122, ly, 214, ly, th.secondary, 2.2)

    # --- upper lane: the reference goes straight to the comparison --------
    box(214, fy, 300, 56, "reference path",
        "the original file, never a re-recording", th.primary)
    s.arrow(514, fy + 28, 596, fy + 28, th.primary, 2.2)

    # --- lower lane: playback, device under test, capture ------------------
    box(214, ly - 26, 118, 52, "playback", "amp + loudspeaker", th.secondary)
    s.arrow(332, ly, 356, ly, th.secondary, 2.2)
    s.rect(356, ly - 74, 172, 148, "none", th.muted, rx=8, sw=1.6, dash="6,5")
    s.text(442, ly - 84, "test box", 14, th.muted)
    box(370, ly - 52, 144, 46, "device under test", "", th.secondary)
    s.text(442, ly + 20, "hearing aid on an artificial ear,", 13, th.muted)
    s.text(442, ly + 40, "or a headset on a torso simulator", 13, th.muted)
    s.arrow(528, ly, 552, ly, th.secondary, 2.2)
    box(552, ly - 26, 116, 52, "capture", "mic + preamp", th.secondary)
    s.path(f"M 668 {ly:.0f} C 696 {ly:.0f} 696 {ly - 40:.0f} 696 "
           f"{ly - 62:.0f}", stroke=th.secondary, sw=2.2)
    s.arrow(696, ly - 50, 696, fy + 88, th.secondary, 2.2)

    # --- the alignment gate both lanes pass through -----------------------
    s.rect(596, fy - 6, 200, 88, th.panel, th.accent, rx=10, sw=2.4)
    s.text(696, fy + 22, "align and trim", 18, th.fg, bold=True)
    s.text(696, fy + 46, "cross-correlate the envelopes", 13, th.accent)
    s.text(696, fy + 66, "equal length · one clock", 13, th.accent)
    s.arrow(796, fy + 38, 814, fy + 38, th.accent, 2.2)
    s.rect(816, fy - 6, 60, 88, th.panel, th.fg, rx=8, sw=2)
    s.text(846, fy + 32, "STOI /", 15, th.fg, bold=True)
    s.text(846, fy + 52, "ESTOI", 15, th.fg, bold=True)

    # --- the bypass measurement ------------------------------------------
    s.rect(214, 392, 454, 56, "none", th.primary, rx=10, sw=1.6, dash="6,5")
    s.text(441, 416, "run it once with the device bypassed", 15, th.primary,
           bold=True)
    s.text(441, 438, "the loudspeaker, the box noise and the microphone "
           "are scored as degradation too", 13, th.muted)
    s.line(268, 392, 268, ly + 30, th.primary, 1.2, dash="5,4")
    s.line(610, 392, 610, ly + 30, th.primary, 1.2, dash="5,4")

    # --- footer ----------------------------------------------------------
    s.rect(26, 468, 850, 62, "none", th.fg, rx=10, sw=1.4, dash="6,5")
    s.text(450, 494, "Play at the device's operating level, and repeat the "
           "capture", 16, th.fg, bold=True)
    s.text(450, 518, "the index is invariant to level, the device is not; a "
           "single capture carries the acoustic path's run-to-run spread", 13,
           th.muted)
