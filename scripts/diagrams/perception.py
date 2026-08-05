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
            s.text(cx, y + bh + 40, "m(F) drops", 18, th.muted, italic=True)
        if i < len(stages) - 1:
            s.arrow(x + bw + 1, y + bh / 2, x + bw + gap - 2, y + bh / 2, th.fg, 2)
        x += bw + gap


def _d_speech_intelligibility(s: SVG, th: Theme) -> None:
    """SII computation flow (ANSI S3.5-1997, one-third-octave method)."""
    # --- Top: three equivalent-spectrum-level inputs (per 1/3-octave band) ---
    inputs = [
        (150.0, "Speech  Ei'", th.primary),
        (450.0, "Noise  Ni'", th.secondary),
        (750.0, "Threshold  Ti'", th.accent),
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
        (150.0, "Self-masking + spread of masking", "Zi   (clause 5.4)"),
        (264.0, "Equivalent disturbance Di", "max(masking, internal noise) (5.6)"),
        (378.0, "Band audibility Ai = (Ei' − Di + 15)/30", "clipped to [0, 1]   (clause 5.8)"),
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
    s.text(cx, 516, "SII = Σ I_i A_i", 26, th.fg, "middle", bold=True)
    s.text(cx, 542, "band importance I_i (Table 3)  ·  index in [0, 1]  (clause 6)",
           16, th.primary, "middle")


def _d_hearing_threshold(s: SVG, th: Theme) -> None:
    """Hearing-threshold model: ISO 7029 age distribution + ISO 389-7 zero."""
    cx = 450.0
    # --- Inputs --------------------------------------------------------------
    iw, ih = 540.0, 62.0
    s.rect(cx - iw / 2, 56, iw, ih, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 84, "Age Y,  sex,  population fractile Q", 20, th.fg,
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
    _step(152, "Median deviation from age 18   (ISO 7029, 4.2)",
          "dHmd = a · (Y − 18) ^ b   (Table 1, by sex)", th.primary)
    _step(244, "Spread su / sl   (ISO 7029, 4.3)",
          "degree-5 polynomials in (Y − 18)   (Tables 2–5)", th.fg)
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
    _step(lxc, 56, "Age Y,  sex,  fractile Q", "database A = ISO 7029", th.fg)
    _step(rxc, 56, "Exposure L_EX,8h,  t years",
          "normalized to 8 h / 5 days", th.fg)

    # --- Left lane: age component H (HTLA) ----------------------------------
    s.arrow(lxc, 118, lxc, 150, th.fg, 1.8)
    _step(lxc, 150, "Age threshold  H  (HTLA)",
          "ISO 7029 fractile, dB", th.primary)

    # --- Right lane: noise component N (NIPTS) ------------------------------
    s.arrow(rxc, 118, rxc, 150, th.fg, 1.8)
    _step(rxc, 150, "Median NIPTS  N50  (6.3.1)",
          "N50 = [u + v·log10(t/t0)]·(L − L0)²", th.secondary)
    s.arrow(rxc, 212, rxc, 244, th.fg, 1.8)
    _step(rxc, 244, "Fractile NIPTS  N  (6.3.2)",
          "N = N50 + z·(du if z ≥ 0 else dl)", th.fg)

    # --- Converge into HTLAN ------------------------------------------------
    box_y = 372.0
    s.arrow(lxc, 212, cx - 118.0, box_y, th.fg, 1.8)
    s.arrow(rxc, 306, cx + 118.0, box_y, th.fg, 1.8)
    s.rect(cx - bw / 2, box_y, bw, 66, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, box_y + 29, "HTLAN   H' = H + N − H·N / 120", 20, th.fg,
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
    _step(218, "Core loudness of the 20 critical bands  (Tables A.4-A.7)",
          "a₀ transmission (A.4), diffuse-field DDF (A.5), threshold in quiet "
          "LTQ (A.6)", th.fg)
    _step(304, "Specific loudness  N′(z)  over 0.1-Bark steps to 24 Bark",
          "upper masking slopes added band to band (Table A.9)", th.secondary)
    for y0, y1 in ((190, 218), (276, 304)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)
    s.arrow(cx, 362, cx, 392, th.fg, 1.8)

    s.rect(x0, 392, bw, 60, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, 417, "Total loudness  N = ∫ N′(z) dz  [sone]", 17, th.fg, "middle",
           bold=True)
    s.text(cx, 438, "loudness level  LN = 40 + 10·log₂ N  [phon]", 14, th.muted,
           "middle")


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
          "N ≥ 5 random samples over the homogeneous exposure group")
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

    # All three land in the same deliverable.
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
    s.text(450, 76, "Calibrated signal x(t) in pascals", 16, th.fg, bold=True)
    s.text(450, 97, "any sample rate: each metric resamples to 48 kHz "
           "internally", 12, th.muted)

    # Two auditory front ends
    s.rect(60, 148, 270, 56, th.panel, th.primary, rx=10, sw=2)
    s.text(195, 172, "Specific loudness N'(z)", 15, th.fg, bold=True)
    s.text(195, 192, "Zwicker pattern over 24 Bark", 12, th.muted)
    s.rect(390, 148, 450, 56, th.panel, th.primary, rx=10, sw=2)
    s.text(615, 172, "Sottek Hearing Model front end (ECMA-418-2)", 15,
           th.fg, bold=True)
    s.text(615, 192, "outer/middle-ear filter + 53 auditory bands "
           "(Bark_HMS)", 12, th.muted)
    s.arrow(350, 108, 210, 144, th.fg, 1.8)
    s.arrow(550, 108, 605, 144, th.fg, 1.8)

    # The four metric boxes
    metrics = (
        (42.0, "Sharpness S", "DIN 45692",
         "g(z)-weighted first moment", "of N'(z), with k = 0.108",
         ("critical-band-wide noise", "at 1 kHz, 60 dB"),
         "→ S = 1.00 acum"),
        (262.0, "Tonality T", "ECMA-418-2 clause 6",
         "band autocorrelation finds", "periodic components",
         ("1 kHz tone at 40 dB",), "→ T = 1.000 tu_HMS (999 Hz)"),
        (482.0, "Roughness R", "ECMA-418-2 clause 7",
         "fast envelope modulation,", "band-pass peaking near 70 Hz",
         ("1 kHz, 100 % AM at 70 Hz, 60 dB",), "→ R = 0.9999 asper"),
        (702.0, "Fluctuation strength F", "ECMA-418-2 clause 9 (HSA)",
         "slow envelope modulation,", "band-pass peaking near 4 Hz",
         ("1 kHz, 100 % AM at 4 Hz, 60 dB",), "→ F = 0.9957 vacil_HMS"),
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
    s.text(450, 463, "N5, S, R and F feed the Fastl and Zwicker "
           "psychoacoustic annoyance PA = N5·(1 + √(wS² + wFR²))", 12,
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

    step(52, "Narrow-band FFT spectrum — line spacing Δf = 2.7 Hz",
         "Annex E engine spectrum; peak detected at fT = 137.3 Hz (not on "
         "a slope)", th.fg)
    step(138, "Critical band about the tone — Δfc = 101.36 Hz",
         "geometric placement: corners 95.67 and 197.04 Hz, √(f1·f2) = fT",
         th.primary)
    step(224, "Levels from the spectrum lines in the band",
         "masking noise LS = 49.22 dB (iterative mean); tone LT = 67.96 dB "
         "(energy sum)", th.primary)
    step(310, "Masking threshold seen by the ear",
         "LG = LS + 10·log10(Δfc/Δf) = 64.97 dB;  masking index av = −2.02 dB",
         th.primary)
    s.rect(x0, 396, bw, 60, "none", th.accent, rx=10, sw=2.4)
    s.text(cx, 421, "Audibility ΔL = LT − LG − av = 5.01 dB", 17, th.fg,
           bold=True)
    s.text(cx, 443, "the largest ΔL of the nine tones: the decisive "
           "audibility of this spectrum", 12, th.muted)
    for y0, y1 in ((110, 134), (196, 220), (282, 306), (368, 392),
                   (456, 484)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)

    s.rect(130, 488, 640, 68, "none", th.secondary, rx=10, sw=1.6,
           dash="6,5")
    s.text(cx, 515, "From audibility to penalty (ISO 1996-2:2017 Annex J)",
           15, th.secondary, bold=True)
    s.text(cx, 539, "energy mean of the five spectra ΔL = 6.98 dB → tonal "
           "adjustment Kt = 4 dB (Table J.1)", 13, th.fg)


# ---------------------------------------------------------------------------
# Psychoacoustic annoyance (Fastl & Zwicker Eqs 16.2-16.4)
# ---------------------------------------------------------------------------

def _d_psychoacoustic_annoyance(s: SVG, th: Theme) -> None:
    """The four sensations of the guide's worked example (N5 = 30 sone,
    S = 2.0 acum, F = 0.5 vacil, R = 0.3 asper) through the two weightings
    (wS = 0.1001, wFR = 0.2125) into PA = 37.05."""
    inputs = (
        (42.0, "S = 2.0 acum", "sharpness (DIN 45692)",
         "counts only above 1.75 acum"),
        (262.0, "N5 = 30 sone", "percentile loudness (ISO 532-1)",
         "exceeded 5 % of the time"),
        (482.0, "F = 0.5 vacil", "fluctuation strength",
         "slow modulation, ≈ 4 Hz"),
        (702.0, "R = 0.3 asper", "roughness", "fast modulation, ≈ 70 Hz"),
    )
    for x0, name, s1, s2 in inputs:
        cx = x0 + 98.0
        s.rect(x0, 60, 196, 72, th.panel, th.primary, rx=10, sw=2)
        s.text(cx, 84, name, 14, th.fg, bold=True)
        s.text(cx, 103, s1, 11, th.muted)
        s.text(cx, 120, s2, 11, th.muted)

    # Weighting boxes: wS takes S and N5; wFR takes N5, F and R.
    s.rect(90, 204, 340, 86, th.panel, th.fg, rx=10, sw=2)
    s.text(260, 230, "Sharpness weighting wS = 0.1001", 15, th.fg,
           bold=True)
    s.text(260, 252, "wS = (S − 1.75) · 0.25 · log10(N5 + 10)", 13,
           th.primary, mono=True)
    s.text(260, 274, "zero for S ≤ 1.75 acum", 12, th.muted)
    s.rect(470, 204, 340, 86, th.panel, th.fg, rx=10, sw=2)
    s.text(640, 230, "Roughness and fluctuation wFR = 0.2125", 15, th.fg,
           bold=True)
    s.text(640, 252, "wFR = 2.18 / N5^0.4 · (0.4·F + 0.6·R)", 13,
           th.primary, mono=True)
    s.text(640, 274, "roughness weighs more: 0.6 against 0.4", 12,
           th.muted)
    s.arrow(141, 132, 200, 200, th.fg, 1.8)
    s.arrow(330, 132, 285, 200, th.fg, 1.8)
    s.arrow(395, 132, 520, 200, th.fg, 1.8)
    s.arrow(581, 132, 610, 200, th.fg, 1.8)
    s.arrow(800, 132, 690, 200, th.fg, 1.8)

    # Combination
    s.rect(200, 344, 500, 72, "none", th.accent, rx=10, sw=2.4)
    s.text(450, 374, "PA = N5 · (1 + √(wS² + wFR²)) = 37.05", 18, th.fg,
           bold=True)
    s.text(450, 399, "Fastl and Zwicker Eq. 16.2 (origin Widmann 1992)",
           12, th.muted)
    s.arrow(260, 290, 380, 340, th.fg, 1.8)
    s.arrow(640, 290, 520, 340, th.fg, 1.8)

    s.text(450, 464, "a neutral sound (S ≤ 1.75 acum, F = R = 0) sits on "
           "the baseline PA = N5", 14, th.fg)
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

    step(52, "Clean reference x(t) and degraded version y(t)",
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
