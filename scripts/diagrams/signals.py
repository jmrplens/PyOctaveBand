#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Diagrams of the signals guides: levels, filters, spectra and metrology.

One module because these guides all draw the same thing from different
angles: a signal on its way through the library. The level and filter
diagrams follow it stage by stage (weighting, time weighting, the multirate
bank, block state, channels), the spectra diagrams open the estimators that
read it (Welch, coherence, cepstrum, synchronous averaging, correlation) and
the metrology diagrams close the loop back to physical units, uncertainty and
whether the record was fit to analyse at all.
"""

from __future__ import annotations

from collections.abc import Callable

from .canvas import SVG, Theme

# ---------------------------------------------------------------------------
# d1 - Calibration chain (IEC 60942)
# ---------------------------------------------------------------------------

def _d_calibration_chain(s: SVG, th: Theme) -> None:
    gy = 470.0
    s.ground(gy, 40, 860)

    # Calibrator on top of the microphone (left column)
    mx = 150.0
    cal_y = 110.0
    s.text(mx, cal_y - 22, "Sound calibrator", 22, th.fg, bold=True)
    s.rect(mx - 62, cal_y, 124, 86, th.panel, th.fg, rx=10, sw=2)
    s.text(mx, cal_y + 38, "94.0 dB", 26, th.secondary, bold=True, mono=True)
    s.text(mx, cal_y + 66, "1 kHz", 20, th.muted, mono=True)
    s.rect(mx - 15, cal_y + 86, 30, 12, th.fg, rx=3)   # coupler cavity
    s.mic(mx, cal_y + 98, gy, 1.3)

    # Signal chain
    boxes = [(400, "Microphone +", "preamplifier"), (650, "Audio interface", "(ADC)")]
    by, bw, bh = 176.0, 210.0, 78.0
    prev_x = mx + 62
    for bx, l1, l2 in boxes:
        s.rect(bx - bw / 2, by, bw, bh, th.panel, th.primary, rx=12, sw=2)
        s.text(bx, by + 33, l1, 22, th.fg, bold=True)
        s.text(bx, by + 60, l2, 22, th.fg, bold=True)
        s.arrow(prev_x, by + bh / 2, bx - bw / 2 - 6, by + bh / 2, th.fg, 2)
        prev_x = bx + bw / 2 + 6
    s.arrow(prev_x, by + bh / 2, 862, by + bh / 2, th.accent, 2.4)
    s.text(796, by + bh / 2 + 34, "Pa per", 20, th.accent, mono=True)
    s.text(796, by + bh / 2 + 58, "digital unit", 20, th.accent, mono=True)

    # Stability annotation, clearly separated below the chain
    s.rect(250, 340, 560, 96, "none", th.secondary, rx=12, dash="6,5")
    s.text(530, 376, "Stability: |max − mean| and |min − mean| ≤ 0.07 dB", 22, th.secondary, bold=True)
    s.text(530, 408, "(IEC 60942:2017 Table 2, class 1) — else CalibrationWarning", 20, th.fg)


# ---------------------------------------------------------------------------
# d1b - Coupling the calibrator on the capsule (IEC 60942 half-section)
# ---------------------------------------------------------------------------

def _d_calibration_coupling(s: SVG, th: Theme) -> None:
    """Half-section through the coupler with the capsule inserted.

    Drawn to the real proportions of a 1/2 inch measurement microphone
    (12.7 mm nominal capsule diameter, 6.35 mm for the 1/4 inch adaptor
    inset); the scale below is 14 drawing units per millimetre.
    """
    scale = 11.0                 # drawing units per millimetre
    cx = 352.0                   # axis of the assembly
    half = 12.7 / 2 * scale      # 1/2 inch capsule radius: 69.9 units
    wall = 24.0                  # coupler wall thickness, drawn
    ref_y = 300.0                # reference plane: contact microphone/coupler
    dia_y = 236.0                # diaphragm, inside the cavity
    cav_top = 148.0              # cavity roof, under the driver

    # -- Coupler body, sectioned: two walls and a roof around the cavity ----
    for x0 in (cx - half - wall, cx + half):
        s.rect(x0, cav_top - 30, wall, ref_y - cav_top + 30, th.panel, th.fg,
               sw=2)
        y = cav_top - 24
        while y < ref_y - 8:
            s.line(x0 + 2, y + 12, x0 + wall - 2, y, th.muted, 0.9)
            y += 12
    s.rect(cx - half - wall, cav_top - 56, 2 * (half + wall), 26, th.panel,
           th.fg, sw=2)
    x = cx - half - wall + 6
    while x < cx + half + wall - 6:
        s.line(x, cav_top - 32, x + 12, cav_top - 54, th.muted, 0.9)
        x += 12
    s.text(cx, cav_top - 70, "Sound calibrator (class 1)", 20, th.fg, bold=True)

    # Driver radiating down into the cavity.
    s.rect(cx - 40, cav_top - 28, 80, 14, th.primary, th.fg, sw=1.4)
    for r in (22, 34, 46):
        s.path(f"M {cx - r:.0f} {cav_top + 4:.0f} Q {cx:.0f} "
               f"{cav_top + 4 + r * 0.42:.0f} {cx + r:.0f} {cav_top + 4:.0f}",
               stroke=th.accent, sw=1.4)
    s.text(cx, cav_top + 56, "94.0 dB", 21, th.secondary, bold=True, mono=True)
    s.text(cx, cav_top + 80, "1 kHz", 18, th.muted, mono=True)

    # -- Effective load volume: cavity floor is the diaphragm ---------------
    s.add(f'<rect x="{cx - half}" y="{dia_y}" width="{2 * half}" '
          f'height="{ref_y - dia_y}" fill="{th.accent}" opacity="0.22"/>')
    s.rect(cx - half, dia_y, 2 * half, ref_y - dia_y, "none", th.accent,
           sw=1.6, dash="5,4")

    # -- Microphone, inserted to the reference plane ------------------------
    s.rect(cx - half, dia_y, 2 * half, 7, th.fg, rx=2)          # diaphragm
    s.rect(cx - half, ref_y, 2 * half, 132, th.panel, th.primary, rx=6, sw=2)
    s.line(cx - half - wall, ref_y, cx + half + wall, ref_y, th.secondary, 2.6)
    s.text(cx, ref_y + 52, "1/2 in capsule", 19, th.fg, bold=True)
    s.text(cx, ref_y + 76, "+ preamplifier", 19, th.muted)
    s.line(cx, ref_y + 132, cx, 470, th.fg, 2.2)
    s.dim(cx - half, ref_y + 104, cx + half, ref_y + 104, "12.7 mm", size=17)

    # Annotations on the left.
    s.line(cx - half - wall - 6, ref_y, 214, ref_y, th.secondary, 1.3,
           dash="7,4,2,4")
    s.text(210, ref_y - 28, "reference plane", 17, th.secondary, bold=True,
           anchor="end")
    s.text(210, ref_y - 8, "(3.12)", 17, th.secondary, anchor="end")
    s.arrow(216, dia_y + 26, cx - half - 6, dia_y + 22, th.accent, 1.6)
    s.text(212, dia_y + 10, "effective load", 17, th.accent, bold=True,
           anchor="end")
    s.text(212, dia_y + 32, "volume (3.13)", 17, th.accent, anchor="end")
    s.text(212, ref_y + 30, "the generated level shifts", 16, th.muted,
           anchor="end")
    s.text(212, ref_y + 50, "with that volume (6.3 k)", 16, th.muted,
           anchor="end")

    # -- Adaptor inset ------------------------------------------------------
    s.rect(468, 104, 160, 214, "none", th.fg, rx=10, sw=1.6, dash="6,5")
    s.text(548, 132, "1/4 in capsule", 18, th.fg, bold=True)
    ax0, aq = 548.0, 6.35 / 2 * scale
    s.rect(ax0 - 30, 150, 60, 24, th.panel, th.fg, sw=1.6)     # adaptor sleeve
    s.rect(ax0 - aq, 174, 2 * aq, 66, th.panel, th.primary, rx=3, sw=1.8)
    s.line(ax0 - 30, 150, ax0 + 30, 150, th.secondary, 2.2)
    s.text(548, 266, "the adaptor is part of", 15, th.fg)
    s.text(548, 288, "the calibrator (5.1.1)", 15, th.fg)

    # -- Windscreen, off to one side ---------------------------------------
    s.circle(548, 386, 28, th.panel, th.muted, sw=2)
    for k in range(-2, 3):
        s.line(524, 386 + k * 10, 572, 386 + k * 10, th.muted, 0.9)
    s.text(548, 436, "windscreen: off for the", 15, th.muted)
    s.text(548, 456, "check, back on to measure", 15, th.muted)

    # -- Right panel: the background check ---------------------------------
    s.rect(650, 104, 224, 340, "none", th.primary, rx=12, sw=1.8)
    s.text(762, 134, "Before switching it on", 19, th.primary, bold=True)
    s.rect(736, 158, 56, 52, th.panel, th.fg, rx=6, sw=1.8)
    s.rect(750, 210, 28, 62, th.panel, th.primary, rx=4, sw=1.8)
    s.text(764, 190, "OFF", 17, th.muted, bold=True, mono=True)
    for r in (26, 40):
        s.path(f"M {682 + r * 0.2:.0f} {224 - r * 0.5:.0f} A {r} {r} 0 0 1 "
               f"{682 + r * 0.5:.0f} {224 + r * 0.3:.0f}", stroke=th.secondary,
               sw=1.5)
    s.arrow(700, 232, 730, 240, th.secondary, 1.6)
    s.text(694, 282, "source in use", 14, th.secondary)
    s.text(762, 324, "the coupled capsule must", 16, th.fg)
    s.text(762, 346, "read ≥ 30 dB below the", 16, th.fg, bold=True)
    s.text(762, 368, "calibrator level: under 64 dB", 16, th.fg)
    s.text(762, 390, "for a 94 dB calibrator", 16, th.fg)
    s.text(762, 418, "(B.4.2; 40 dB in A.5.3)", 16, th.muted)

    s.text(450, 528, "The specified level is the level at the diaphragm of "
                     "the inserted microphone (5.3.1.2), and holds for", 18, th.fg)
    s.text(450, 552, "the microphone models and configurations listed in the "
           "manual (5.3.1.3, 6.3 a) — IEC 60942:2017", 18, th.muted)


# ---------------------------------------------------------------------------
# d4 - Library signal chain
# ---------------------------------------------------------------------------

def _d_signal_chain(s: SVG, th: Theme) -> None:
    stages = [
        ("Signal", "$x$, $f_s$", th.fg),
        ("Calibrate", "→ Pa", th.primary),
        ("Weighting", "A/C/G/Z", th.primary),
        ("Octave", "bands $1/b$", th.primary),
        ("Ballistics", "F / S / I", th.primary),
        ("Metrics", "$L_{eq}$, $L_N$…", th.accent),
    ]
    bw, bh, gap = 136.0, 92.0, 12.0
    total = len(stages) * bw + (len(stages) - 1) * gap
    x = (900 - total) / 2
    y = 170.0
    for i, (title, sub, color) in enumerate(stages):
        s.rect(x, y, bw, bh, th.panel, color, rx=12, sw=2)
        s.text(x + bw / 2, y + 40, title, 22, th.fg, bold=True)
        # Mono is the code voice: it styles the literal stage tags, while a
        # sub carrying $...$ mathematics composes in the text face (mono
        # plus markup is refused by the canvas).
        s.text(x + bw / 2, y + 68, sub, 19, color, mono="$" not in sub)
        if i < len(stages) - 1:
            s.arrow(x + bw + 1, y + bh / 2, x + bw + gap - 2, y + bh / 2, th.fg, 2)
        x += bw + gap


# ---------------------------------------------------------------------------
# d5 - Multirate decimation inside the filter bank
# ---------------------------------------------------------------------------

def _d_multirate(s: SVG, th: Theme) -> None:
    # Input on the left
    s.rect(36, 150, 136, 70, th.panel, th.fg, rx=10, sw=2)
    s.text(104, 180, "Signal", 22, th.fg, bold=True)
    s.text(104, 205, "$f_s$ = 48 kHz", 18, th.muted)

    rows = [
        (120.0, "16 kHz band", "$f_s$", "no decimation", th.secondary),
        (230.0, "1 kHz band", "$f_s / 8$", "6 kHz", th.primary),
        (340.0, "63 Hz band", "$f_s / 64$", "750 Hz", th.accent),
    ]
    for y, band, rate, eff, color in rows:
        bx = 455.0
        if "no" not in eff:
            s.arrow(172, 185, 240, y + 35, th.fg, 1.6)
            s.rect(250, y, 150, 70, th.panel, th.muted, rx=10, sw=1.6)
            s.text(325, y + 30, "Anti-alias", 20, th.fg)
            s.text(325, y + 54, "LPF + \u2193$M$", 18, th.muted)
            s.arrow(400, y + 35, 448, y + 35, th.fg, 1.6)
        else:
            s.arrow(172, 185, 448, y + 35, th.fg, 1.6)
        s.rect(bx, y, 190, 70, th.panel, color, rx=10, sw=2)
        s.text(bx + 95, y + 30, band, 20, th.fg, bold=True)
        s.text(bx + 95, y + 54, f"SOS @ {rate}", 18, color)
        s.text(660, y + 40, eff, 18, th.muted, "start", mono=True)

    s.text(450, 480, "Low bands are filtered at a decimated rate: the relative", 20, th.fg)
    s.text(450, 508, "bandwidth stays wide, so the SOS stays numerically healthy.", 20, th.fg)


def _d_uncertainty(s: SVG, th: Theme) -> None:
    """Two routes to measurement uncertainty (ISO/IEC Guide 98-3 and Suppl. 1).

    From a shared measurement model and its input estimates, two parallel
    lanes: the GUM law of propagation of uncertainty (clause 5) and the
    Monte Carlo propagation of distributions (Supplement 1, clause 7).
    """
    # --- Shared measurement model + inputs ----------------------------------
    cx = 450.0
    iw, ih = 560.0, 62.0
    s.rect(cx - iw / 2, 56, iw, ih, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 84, "Measurement model  $y = f(x_1, …, x_N)$", 20, th.fg,
           "middle", bold=True)
    s.text(cx, 106, "input estimates $x_i$ with standard uncertainties $u(x_i)$",
           15, th.muted, "middle")

    lxc, rxc = 232.0, 668.0
    s.arrow(cx, 118, lxc, 158, th.fg, 1.8)
    s.arrow(cx, 118, rxc, 158, th.fg, 1.8)

    bw, bh = 372.0, 62.0

    def _step(cxx: float, y: float, l1: str, l2: str, color: str) -> None:
        s.rect(cxx - bw / 2, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cxx, y + 27, l1, 18, th.fg, "middle", bold=True)
        if l2:
            s.text(cxx, y + 48, l2, 14, th.muted, "middle")

    # --- Left lane: GUM law of propagation (clause 5) -----------------------
    _step(lxc, 158, "Law of propagation  (GUM 5)",
          "sensitivity $c_i = ∂f / ∂x_i$", th.primary)
    _step(lxc, 250, "Combine in quadrature",
          "$u_c^2 = Σ c_i^2 u^{2}(x_i)$ + correlation", th.fg)
    _step(lxc, 342, "Effective dof  (Annex G.4)",
          "$v_{eff}$ — Welch–Satterthwaite", th.fg)
    s.arrow(lxc, 220, lxc, 250, th.fg, 1.8)
    s.arrow(lxc, 312, lxc, 342, th.fg, 1.8)
    s.arrow(lxc, 404, lxc, 434, th.fg, 1.8)
    s.rect(lxc - bw / 2, 434, bw, 58, "none", th.primary, rx=10, sw=2.4)
    s.text(lxc, 462, "$U = k · u_c$", 22, th.fg, "middle", bold=True)
    s.text(lxc, 482, "$k = t_{p}(v_{eff})$   (clause 6)", 14, th.muted, "middle")

    # --- Right lane: Monte Carlo (Supplement 1, clause 7) -------------------
    _step(rxc, 158, "Monte Carlo  (Suppl. 1, 7)",
          "draw $x_i$ from its PDF $g(x_i)$", th.secondary)
    _step(rxc, 250, "Propagate $M$ trials",
          "$y_r = f(x_{1r}, …, x_{Nr})$", th.fg)
    _step(rxc, 342, "Sort ${y_r}$, take fractiles",
          "prob.-symmetric 95 % interval", th.fg)
    s.arrow(rxc, 220, rxc, 250, th.fg, 1.8)
    s.arrow(rxc, 312, rxc, 342, th.fg, 1.8)
    s.arrow(rxc, 404, rxc, 434, th.fg, 1.8)
    s.rect(rxc - bw / 2, 434, bw, 58, "none", th.secondary, rx=10, sw=2.4)
    s.text(rxc, 462, "coverage interval", 22, th.fg, "middle", bold=True)
    s.text(rxc, 482, "$[y_{low}, y_{high}]$   (clause 7.7)", 14, th.muted, "middle")


def _d_uncertainty_sources(s: SVG, th: Theme) -> None:
    """Where each row of an acoustic budget comes from physically.

    One outdoor A-weighted level measurement drawn to scale at 60 units per
    metre (microphone 1.5 m above the ground and 4 m from the facade, three
    alternative positions 2 m apart), with a leader from each element to the
    budget row it feeds.
    """
    gy = 442.0
    s.ground(gy, 34, 520)
    scale = 60.0                       # drawing units per metre

    # -- Facade at the right of the scene ----------------------------------
    s.rect(496, 168, 26, gy - 168, th.panel, th.fg, sw=2)
    s.text(490, 200, "facade", 15, th.muted, anchor="end")

    # -- Source ------------------------------------------------------------
    s.rect(44, 372, 42, 52, th.panel, th.primary, rx=6, sw=2)
    s.circle(65, 391, 9, th.primary)
    s.circle(65, 391, 3.5, th.bg)
    s.circle(65, 411, 5.5, th.primary)
    s.text(65, 362, "source", 15, th.fg, bold=True)
    for r in (20, 32, 44):
        s.path(f"M {90 + r * 0.30:.0f} {398 - r * 0.55:.0f} "
               f"A {r} {r} 0 0 1 {90 + r * 0.55:.0f} {398 + r * 0.30:.0f}",
               stroke=th.accent, sw=1.3)

    # -- Meter on a tripod, microphone 1.5 m up and 4 m from the facade ----
    mic_x, mic_y = 256.0, gy - 1.5 * scale
    s.mic(mic_x, mic_y, gy, 1.1)
    s.rect(mic_x + 16, 372, 48, 64, th.panel, th.primary, rx=6, sw=1.8)
    s.text(mic_x + 40, 410, "SLM", 15, th.fg, bold=True, mono=True)
    s.dim(mic_x - 40, mic_y, mic_x - 40, gy, "1.5 m", size=15)
    s.line(mic_x - 40, mic_y, mic_x - 6, mic_y, th.muted, 0.9, dash="3,3")
    s.dim(mic_x, gy + 40, 496, gy + 40, "4 m", size=15)
    s.line(mic_x, gy, mic_x, gy + 44, th.muted, 0.9, dash="3,3")
    s.line(496, gy, 496, gy + 44, th.muted, 0.9, dash="3,3")

    # Two alternative microphone positions, 2 m either side.
    for dx in (-2.0 * scale, 2.0 * scale):
        s.circle(mic_x + dx, mic_y, 6.5, "none", th.secondary, 2.0)
        s.line(mic_x + dx, mic_y + 7, mic_x + dx, gy, th.secondary, 1.1,
               dash="5,4")
    s.dim(mic_x - 2 * scale, mic_y - 36, mic_x + 2 * scale, mic_y - 36,
          "3 positions, 2 m apart", size=15)

    # -- Calibrator lying beside the meter ---------------------------------
    s.rect(340, gy - 52, 22, 48, th.panel, th.secondary, rx=5, sw=1.8)
    s.text(351, 380, "calibrator", 15, th.secondary)

    # -- Weather station, high on the facade side --------------------------
    s.rect(300, 104, 154, 74, th.panel, th.fg, rx=8, sw=1.6)
    s.text(377, 128, "weather", 16, th.fg, bold=True)
    s.text(377, 150, "3 m/s  12 °C", 14, th.muted, mono=True)
    s.text(377, 170, "68 % RH", 14, th.muted, mono=True)

    # -- Budget table on the right, rows in the order the leaders run ------
    bx, by, bw = 552.0, 96.0, 322.0
    rows = (
        ("Meteorology and ground", "Type B - from the propagation clause",
         th.fg, (454.0, 140.0)),
        ("Position scatter", "Type A - $s/√n$, $v = n − 1$", th.secondary,
         (mic_x + 2 * scale + 8, mic_y)),
        ("Instrument class tolerance", "Type B - rectangular, $a$ = 0.3 dB",
         th.primary, (mic_x + 64, 400.0)),
        ("Calibrator class tolerance", "Type B - rectangular, $a$ = 0.4 dB",
         th.secondary, (362.0, gy - 30)),
    )
    s.rect(bx, by, bw, 252, "none", th.fg, rx=10, sw=1.8)
    s.text(bx + bw / 2, by - 14, "The budget it feeds", 19, th.fg, bold=True)
    for i, (name, kind, color, (from_x, from_y)) in enumerate(rows):
        ry = by + 26 + i * 60
        if i:
            s.line(bx + 12, ry - 18, bx + bw - 12, ry - 18, th.muted, 0.8)
        s.circle(bx + 20, ry + 4, 5, color)
        s.text(bx + 34, ry + 10, name, 17, th.fg, anchor="start", bold=True)
        s.text(bx + 34, ry + 32, kind, 14, th.muted, anchor="start")
        s.line(from_x, from_y, bx - 6, ry + 4, color, 1.0, dash="4,4")

    s.text(bx + bw / 2, by + 292, "one calibrator for two channels makes their",
           15, th.secondary)
    s.text(bx + bw / 2, by + 312, "calibration terms correlated, not two rows",
           15, th.secondary)

    s.text(450, 552, "Budgets fail by omission, not arithmetic: every row is "
                     "a piece of hardware or a decision about geometry", 17,
           th.muted)


def _d_time_weighting(s: SVG, th: Theme) -> None:
    """Exponential-detector chain of the sound-level time weightings (IEC 61672-1)."""
    # The decibel stage spells out the logarithm and is the widest title of the
    # chain, so it takes a smaller face to keep the padding inside its box.
    stages = [
        ("$p(t)$", "band signal", th.fg, 21),
        ("$( · )^2$", "square", th.primary, 21),
        ("one-pole RC", "time constant $τ$", th.primary, 21),
        ("$10·log_{10}(·/p_0^2)$", "to decibels", th.accent, 18),
        ("$L_{τ}(t)$", "time-weighted level", th.secondary, 21),
    ]
    bw, bh, gap = 150.0, 90.0, 12.0
    total = len(stages) * bw + (len(stages) - 1) * gap
    x = (900 - total) / 2
    y = 108.0
    last = len(stages) - 1
    for i, (title, sub, color, size) in enumerate(stages):
        fill = "none" if i in (0, last) else th.panel
        s.rect(x, y, bw, bh, fill, color, rx=12, sw=2.2)
        s.text(x + bw / 2, y + 38, title, size, th.fg, "middle", bold=True)
        s.text(x + bw / 2, y + 64, sub, 14, color, "middle")
        if i < last:
            s.arrow(x + bw + 1, y + bh / 2, x + bw + gap - 2, y + bh / 2, th.fg, 2)
        x += bw + gap

    # Discrete realization of the detector. Parked: the exponent of
    # α = 1 − e^(−1/(fs·τ)) carries a subscript inside the superscript,
    # one script level more than the composer sets, so the whole line
    # stays plain until that family is adjudicated.
    s.rect(130, 246, 640, 70, th.panel, th.muted, rx=10, sw=1.6)
    s.text(450, 275, "y[n] = α·x²[n] + (1 − α)·y[n−1],   α = 1 − e^(−1/(fs·τ))",
           18, th.fg, "middle", bold=True, mono=True)
    s.text(450, 299, "a first-order low-pass on the squared signal → the mean-square "
           "envelope", 14, th.muted, "middle")

    # The three standardized time constants.
    chips = [
        ("Fast (F)", "$τ$ = 125 ms", th.primary),
        ("Slow (S)", "$τ$ = 1000 ms", th.accent),
        ("Impulse (I)", "35 ms rise · 1500 ms fall", th.secondary),
    ]
    cw, cgap = 210.0, 15.0
    cx = (900 - (len(chips) * cw + (len(chips) - 1) * cgap)) / 2
    for title, sub, color in chips:
        s.rect(cx, 350, cw, 74, "none", color, rx=10, sw=2.2)
        s.text(cx + cw / 2, 380, title, 18, th.fg, "middle", bold=True)
        s.text(cx + cw / 2, 404, sub, 14, th.muted, "middle")
        cx += cw + cgap


def _d_block_processing(s: SVG, th: Theme) -> None:
    """Streaming block processing: carrying the filter state versus resetting it."""
    import math

    x0, blk_w, nblk, amp = 150.0, 190.0, 3, 66.0

    def _lane(gy: float, reset: bool, color: str) -> None:
        s.line(x0, gy, x0 + nblk * blk_w, gy, th.muted, 1.4)
        for k in range(nblk + 1):
            bx = x0 + k * blk_w
            s.line(bx, gy - amp - 16, bx, gy + 12, th.muted, 1.0, dash="3,4")
        for k in range(nblk):
            pts = []
            for j in range(31):
                frac = j / 30.0
                t = frac if reset else (k + frac)
                v = 1.0 - math.exp(-t / 0.9)
                pts.append((x0 + (k + frac) * blk_w, gy - amp * v))
            d = "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts)
            s.path(d, stroke=color, sw=2.6)
            s.text(x0 + (k + 0.5) * blk_w, gy + 30, f"block {k + 1}", 13,
                   th.muted, "middle")
        if reset:
            # Mark the discontinuity where each block restarts from rest.
            v_end = 1.0 - math.exp(-1.0 / 0.9)
            for k in range(1, nblk):
                bx = x0 + k * blk_w
                s.line(bx, gy - amp * v_end, bx, gy, th.secondary, 1.6, dash="2,3")
        else:
            # A small tag shows the carried state seeding the next block.
            for k in range(1, nblk):
                bx = x0 + k * blk_w
                s.rect(bx - 27, gy - amp - 40, 54, 22, th.bg, color, rx=6, sw=1.4)
                s.text(bx, gy - amp - 25, "y[-1]", 12, th.fg, "middle", mono=True)

    s.text(450, 62, "State carried across blocks — TimeWeighting.process()", 19,
           th.fg, "middle", bold=True)
    s.text(450, 84, "y[-1] (or the sosfilt zi vector) seeds the next block → identical "
           "to one continuous call", 13, th.muted, "middle")
    _lane(200.0, reset=False, color=th.primary)

    s.text(450, 300, "State reset each block — reset() or a fresh call", 19,
           th.fg, "middle", bold=True)
    s.text(450, 322, "every block restarts from rest → spurious discontinuities at "
           "the seams", 13, th.muted, "middle")
    _lane(430.0, reset=True, color=th.secondary)


def _d_multichannel(s: SVG, th: Theme) -> None:
    """How array shapes flow through a per-channel operation (time axis last)."""
    cell = 22.0

    def _grid(gx: float, gy: float, rows: int, cols: int, color: str) -> None:
        for r in range(rows):
            for c in range(cols):
                s.rect(gx + c * cell, gy + r * cell, cell, cell, th.panel, color, sw=1.3)

    # 1-D lane.
    _grid(64, 120, 1, 8, th.primary)
    s.text(64 + 4 * cell, 108, "1-D:  (samples,)", 15, th.fg, "middle", bold=True)
    s.rect(610, 120, cell, cell, "none", th.accent, sw=2)
    s.text(610 + cell / 2, 108, "scalar", 15, th.fg, "middle", bold=True)

    # 2-D lane.
    _grid(64, 250, 3, 8, th.primary)
    s.text(64 + 4 * cell, 238, "2-D:  (channels, samples)", 15, th.fg, "middle",
           bold=True)
    for r in range(3):
        s.rect(610, 250 + r * cell, cell, cell, "none", th.accent, sw=2)
    s.text(610 + cell / 2, 238, "(channels,)", 15, th.fg, "middle", bold=True)

    # Shared processing box.
    s.rect(360, 96, 190, 200, th.panel, th.fg, rx=12, sw=2)
    s.text(455, 178, "reduce along", 17, th.fg, "middle", bold=True)
    s.text(455, 202, "axis = −1  (time)", 17, th.primary, "middle", bold=True, mono=True)
    s.text(455, 236, "the channel axis 0", 14, th.muted, "middle")
    s.text(455, 256, "rides through untouched", 14, th.muted, "middle")

    s.arrow(64 + 8 * cell + 4, 131, 358, 150, th.fg, 1.6)
    s.arrow(64 + 8 * cell + 4, 283, 358, 244, th.fg, 1.6)
    s.arrow(552, 150, 606, 131, th.fg, 1.6)
    s.arrow(552, 244, 606, 283, th.fg, 1.6)

    s.text(450, 350, "A mono call returns a scalar; a $C$-channel call returns $C$ results.",
           15, th.fg, "middle")
    s.text(450, 374, "Band metrics widen the reduced axis instead: (…, bands).",
           14, th.muted, "middle")


# ISO 226:2023 Table 1 (p. 4): frequency Hz -> (alpha_f, L_U dB, T_f dB), the
# same parameters the library's ``equal_loudness_contour`` implements.
_ISO226_TABLE1: tuple[tuple[float, float, float, float], ...] = (
    (20.0, 0.635, -31.5, 78.1), (25.0, 0.602, -27.2, 68.7),
    (31.5, 0.569, -23.1, 59.5), (40.0, 0.537, -19.3, 51.1),
    (50.0, 0.509, -16.1, 44.0), (63.0, 0.482, -13.1, 37.5),
    (80.0, 0.456, -10.4, 31.5), (100.0, 0.433, -8.2, 26.5),
    (125.0, 0.412, -6.3, 22.1), (160.0, 0.391, -4.6, 17.9),
    (200.0, 0.373, -3.2, 14.4), (250.0, 0.357, -2.1, 11.4),
    (315.0, 0.343, -1.2, 8.6), (400.0, 0.330, -0.5, 6.2),
    (500.0, 0.320, 0.0, 4.4), (630.0, 0.311, 0.4, 3.0),
    (800.0, 0.303, 0.5, 2.2), (1000.0, 0.300, 0.0, 2.4),
    (1250.0, 0.295, -2.7, 3.5), (1600.0, 0.292, -4.2, 1.7),
    (2000.0, 0.290, -1.2, -1.3), (2500.0, 0.290, 1.4, -4.2),
    (3150.0, 0.289, 2.3, -6.0), (4000.0, 0.289, 1.0, -5.4),
    (5000.0, 0.289, -2.3, -1.5), (6300.0, 0.293, -7.2, 6.0),
    (8000.0, 0.303, -11.2, 12.6), (10000.0, 0.323, -10.9, 13.9),
    (12500.0, 0.354, -3.5, 12.3),
)


def _iso226_spl(alpha_f: float, l_u: float, t_f: float, phon: float) -> float:
    """ISO 226:2023 Formula (1): SPL of a pure tone at loudness level ``phon``."""
    import math
    term = (4.0e-10) ** (0.3 - alpha_f) * (10 ** (0.03 * phon) - 10 ** 0.072) \
        + 10 ** (alpha_f * (t_f + l_u) / 10)
    return 10 / alpha_f * math.log10(term) - l_u


def _a_weight_db(f: float) -> float:
    """IEC 61672-1 Annex E analytic A-weighting, normalized to 0 dB at 1 kHz."""
    import math

    def gain(x: float) -> float:
        f1, f2, f3, f4 = 20.599, 107.653, 737.862, 12194.217
        return (f4 ** 2 * x ** 4) / ((x ** 2 + f1 ** 2)
                                     * math.sqrt((x ** 2 + f2 ** 2) * (x ** 2 + f3 ** 2))
                                     * (x ** 2 + f4 ** 2))

    return 20 * math.log10(gain(f) / gain(1000.0))


def _d_equal_loudness_weighting(s: SVG, th: Theme) -> None:
    """Equal-loudness contours (ISO 226) inverted into the A-curve (IEC 61672-1)."""
    import math

    f_lo, f_hi = 20.0, 12500.0

    def make_fx(px0: float, px1: float) -> Callable[[float], float]:
        span = math.log10(f_hi) - math.log10(f_lo)

        def fx(f: float) -> float:
            return px0 + (math.log10(f) - math.log10(f_lo)) / span * (px1 - px0)

        return fx

    def axes(bx0: float, by0: float, bx1: float, by1: float,
             fx: Callable[[float], float]) -> None:
        s.rect(bx0, by0, bx1 - bx0, by1 - by0, "none", th.muted, sw=1.2)
        for f, lab in ((20.0, "20"), (100.0, "100"), (1000.0, "1k"), (10000.0, "10k")):
            x = fx(f)
            s.line(x, by1, x, by1 + 5, th.muted, 1.2)
            s.text(x, by1 + 21, lab, 13, th.muted, "middle")
        s.text((bx0 + bx1) / 2, by1 + 42, "Frequency [Hz]", 14, th.fg, "middle")

    # --- Left panel: the equal-loudness contours ----------------------------
    lx0, lx1, ly0, ly1 = 70.0, 390.0, 110.0, 430.0
    l_fx = make_fx(lx0, lx1)

    def l_fy(db: float) -> float:  # 0 dB at the bottom, 120 dB at the top
        return ly1 - db / 120.0 * (ly1 - ly0)

    s.text((lx0 + lx1) / 2, 92, "Equal-loudness contours (ISO 226)", 17, th.fg,
           "middle", bold=True)
    axes(lx0, ly0, lx1, ly1, l_fx)
    for db in (0, 40, 80, 120):
        y = l_fy(db)
        s.line(lx0 - 5, y, lx0, y, th.muted, 1.2)
        s.text(lx0 - 9, y + 5, str(db), 13, th.muted, "end")
    s.text(lx0 - 20, ly0 - 12, "dB SPL", 12, th.muted, "middle")

    for phon in (20, 40, 60, 80):
        main = phon == 40
        color = th.primary if main else th.muted
        pts = [(l_fx(f), l_fy(_iso226_spl(a, lu, tf, phon)))
               for f, a, lu, tf in _ISO226_TABLE1]
        d = "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts)
        s.path(d, stroke=color, sw=2.8 if main else 1.5)
        # Label each contour above its 160 Hz point, where the curves spread.
        yl = l_fy(_iso226_spl(0.391, -4.6, 17.9, phon)) - 10
        if main:
            s.text(l_fx(160.0), yl, "40 phon", 14, th.primary, "middle", bold=True)
        else:
            s.text(l_fx(160.0), yl, str(phon), 12, th.muted, "middle")

    # --- Middle: the inversion step ------------------------------------------
    s.text(462, 226, "invert", 16, th.fg, "middle", bold=True)
    s.text(462, 248, "0 dB at 1 kHz", 13, th.muted, "middle")
    s.arrow(400, 272, 546, 272, th.fg, 2.2)

    # --- Right panel: inverted contour vs the A-curve ------------------------
    rx0, rx1 = 558.0, 872.0
    r_fx = make_fx(rx0, rx1)

    def r_fy(db: float) -> float:  # +10 dB at the top, -70 dB at the bottom
        return ly0 + (10.0 - db) / 80.0 * (ly1 - ly0)

    s.text((rx0 + rx1) / 2, 92, "A-weighting (IEC 61672-1)", 17, th.fg,
           "middle", bold=True)
    axes(rx0, ly0, rx1, ly1, r_fx)
    for db in (0, -20, -40, -60):
        y = r_fy(db)
        s.line(rx0 - 5, y, rx0, y, th.muted, 1.2)
        s.text(rx0 - 9, y + 5, str(db), 13, th.muted, "end")
    s.text(rx0 - 20, ly0 - 12, "dB", 12, th.muted, "middle")
    s.line(rx0, r_fy(0.0), rx1, r_fy(0.0), th.muted, 0.9, dash="3,4")

    # Inverted 40-phon contour, relative to its 1 kHz value (dashed reference).
    inv = [(r_fx(f), r_fy(-(_iso226_spl(a, lu, tf, 40.0) - 40.0)))
           for f, a, lu, tf in _ISO226_TABLE1]
    s.path("M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in inv),
           stroke=th.secondary, sw=2.0, dash="6,5")
    # The A-curve itself, sampled densely on the log axis.
    n = 60
    aw = [(r_fx(f), r_fy(_a_weight_db(f)))
          for f in (f_lo * (f_hi / f_lo) ** (i / (n - 1)) for i in range(n))]
    s.path("M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in aw),
           stroke=th.primary, sw=2.8)

    # Legend, bottom right of the panel (the curves live top left).
    s.line(636, 386, 668, 386, th.primary, 2.8)
    s.text(676, 391, "A-weighting (IEC 61672-1)", 13, th.fg, "start")
    s.line(636, 408, 668, 408, th.secondary, 2.0, dash="6,5")
    s.text(676, 413, "inverted 40-phon contour", 13, th.fg, "start")

    # --- Footer ---------------------------------------------------------------
    s.text(450, 507, "A is the 40-phon contour flipped into a realizable filter: "
           "quiet sounds, where the ear discards bass hardest.", 14, th.fg, "middle")
    s.text(450, 529, "The match is deliberately loose (a 1930s convention, not a "
           "loudness model); C mirrors the flatter ~100-phon contour.", 13,
           th.muted, "middle")


# ---------------------------------------------------------------------------
# Sound level meter chain (IEC 61672-1)
# ---------------------------------------------------------------------------

def _d_slm_chain(s: SVG, th: Theme) -> None:
    """IEC 61672-1 sound level meter: the acoustic front end on the left
    (windscreen, microphone, coupled class 1 calibrator) feeding the
    four-stage level chain on the right."""
    gy = 508.0
    s.ground(gy, 40.0, 330.0)

    # --- Acoustic calibrator, coupled onto the capsule for the level check --
    mx = 165.0
    s.text(mx, 68.0, "Sound calibrator (class 1)", 20, th.fg, bold=True)
    s.rect(mx - 62, 82.0, 124, 84, th.panel, th.fg, rx=10, sw=2)
    s.text(mx, 118.0, "94.0 dB", 26, th.secondary, bold=True, mono=True)
    s.text(mx, 146.0, "1 kHz", 20, th.muted, mono=True)
    s.rect(mx - 15, 166.0, 30, 12, th.fg, rx=3)            # coupler cavity
    s.arrow(mx, 184.0, mx, 236.0, th.secondary, 2.0)
    s.text(mx - 16, 202.0, "coupled for", 15, th.muted, anchor="end",
           italic=True)
    s.text(mx - 16, 222.0, "the level check", 15, th.muted, anchor="end",
           italic=True)

    # --- Microphone on a stand; the windscreen is fitted for measurement ----
    cap_top = 248.0
    s.mic(mx, cap_top, gy, 1.25)
    s.ellipse(mx, cap_top + 18, 42, 42, "none", th.muted, 1.6, dash="5,4")
    s.line(mx - 30, cap_top + 49, mx - 65, cap_top + 76, th.muted, 1.0)
    s.text(mx - 79, cap_top + 94, "Windscreen", 17, th.fg)

    # --- The four-stage level chain (vertical) ------------------------------
    cx, bw, bh = 610.0, 400.0, 78.0
    x0 = cx - bw / 2
    chain = [
        (96.0, "Microphone + preamplifier",
         "free-field capsule, high-impedance stage"),
        (208.0, "Frequency weighting  A / C / Z",
         "all three are 0 dB at 1 kHz; class 1: ±0.7 dB"),
        (320.0, "Squaring + time weighting  F / S",
         "exponential detector: $τ_F$ = 125 ms, $τ_S$ = 1 s"),
    ]
    for by, l1, l2 in chain:
        s.rect(x0, by, bw, bh, th.panel, th.primary, rx=12, sw=2)
        s.text(cx, by + 33, l1, 21, th.fg, bold=True)
        s.text(cx, by + 59, l2, 17, th.muted)
    s.rect(x0, 432.0, bw, bh, "none", th.accent, rx=12, sw=2.4)
    s.text(cx, 465.0, "Display", 21, th.fg, bold=True)
    s.text(cx, 491.0, "$L_{AF}(t)$, $L_{AS}(t)$ in dB re 20 µPa", 17, th.accent)
    for y0 in (174.0, 286.0, 398.0):
        s.arrow(cx, y0, cx, y0 + 32, th.fg, 2.0)
    # Sound pressure into the front end.
    s.arrow(226.0, 268.0, x0 - 8, 135.0, th.fg, 2.0)


# ---------------------------------------------------------------------------
# Two-channel FRF measurement: H1 estimator and coherence
# ---------------------------------------------------------------------------

def _d_system_measurement(s: SVG, th: Theme) -> None:
    """Classic dual-channel frequency-response measurement: generator into
    amplifier and loudspeaker, microphone back in, the electrical reference
    on channel 1, and Welch cross-spectra feeding H1 and coherence."""
    def box(x0: float, x1: float, y0: float, h: float, l1: str, l2: str,
            color: str) -> None:
        s.rect(x0, y0, x1 - x0, h, th.panel, color, rx=10, sw=2)
        s.text((x0 + x1) / 2, y0 + 30.0, l1, 18, th.fg, bold=True)
        s.text((x0 + x1) / 2, y0 + 54.0, l2, 14, th.muted)

    box(60, 250, 68, 72.0, "Signal generator", "broadband noise or a sweep",
        th.fg)
    box(310, 460, 68, 72.0, "Power amplifier",
        "its gain is in H1", th.fg)
    s.arrow(250.0, 104.0, 306.0, 104.0, th.fg, 2.0)
    s.arrow(460.0, 104.0, 496.0, 104.0, th.fg, 2.0)

    # Loudspeaker under test and the measurement microphone.
    s.rect(500, 76, 44, 56, th.panel, th.primary, rx=6, sw=2)
    s.circle(522.0, 96.0, 10.0, th.primary)
    s.circle(522.0, 96.0, 4.0, th.bg)
    s.circle(522.0, 119.0, 5.5, th.primary)
    s.text(522.0, 60.0, "Loudspeaker under test", 15, th.fg, bold=True)
    for r in (22, 38, 54):
        s.path(f"M {548 + r * 0.30:.0f} {104 - r * 0.55:.0f} "
               f"A {r} {r} 0 0 1 {548 + r * 0.55:.0f} {104 + r * 0.30:.0f}",
               stroke=th.accent, sw=1.5)
    s.rect(640, 99, 12, 10, th.fg, rx=2.5)                 # capsule
    s.rect(652, 96, 30, 16, th.primary, rx=5)              # mic body
    s.text(680.0, 76.0, "measurement microphone", 15, th.fg, bold=True)
    s.line(682.0, 104.0, 706.0, 104.0, th.fg, 1.8)
    s.line(706.0, 104.0, 706.0, 176.0, th.fg, 1.8)

    # Reference tap after the generator.
    s.circle(278.0, 104.0, 3.5, th.fg)
    s.line(278.0, 104.0, 278.0, 176.0, th.fg, 1.8)

    box(150, 420, 208, 60.0, "Channel 1: reference $x(t)$",
        "the electrical drive signal", th.primary)
    box(470, 750, 208, 60.0, "Channel 2: response $y(t)$",
        "acoustic output at the microphone", th.secondary)
    s.arrow(278.0, 176.0, 278.0, 204.0, th.fg, 1.8)
    s.arrow(706.0, 176.0, 706.0, 204.0, th.fg, 1.8)

    box(150, 750, 300, 72.0, "Dual-channel FFT analysis (Welch)",
        "Hann segments, 50 % overlap  →  $G_{xx}(f)$, $G_{yy}(f)$, $G_{xy}(f)$", th.fg)
    s.arrow(285.0, 268.0, 285.0, 296.0, th.fg, 1.8)
    s.arrow(610.0, 268.0, 610.0, 296.0, th.fg, 1.8)

    box(60, 440, 404, 72.0, "$H_{1}(f) = G_{xy} / G_{xx}$",
        "unbiased with output noise; $H_2 = G_{yy}/G_{yx}$ for input noise",
        th.primary)
    box(460, 840, 404, 72.0, "$γ^{2}(f) = |G_{xy}|^2 / (G_{xx}·G_{yy})$",
        "1 for a noiseless linear path; less with output noise",
        th.secondary)
    s.arrow(280.0, 372.0, 264.0, 400.0, th.fg, 1.8)
    s.arrow(620.0, 372.0, 636.0, 400.0, th.fg, 1.8)

    s.text(450.0, 528.0,
           "trust $|H_1|$ only where $γ^2$ is near 1: coherence dips flag "
           "noise, distortion or unresolved delay",
           17, th.fg, bold=True)


# ---------------------------------------------------------------------------
# Test-signal family panel
# ---------------------------------------------------------------------------

def _d_test_signals(s: SVG, th: Theme) -> None:
    """Labelled miniature of each stimulus: white and pink noise with their
    PSD slopes, an MLS chip stream, linear versus exponential sweeps on a
    time-frequency sketch and an IEC 60268-1 tone burst."""
    import math

    def tile(x: float, y: float, w: float, title: str) -> None:
        s.rect(x, y, w, 240.0, th.panel, th.fg, rx=10, sw=1.8)
        s.text(x + w / 2, y + 26.0, title, 17, th.fg, bold=True)

    def spectrum_axes(x: float, y: float) -> None:
        s.line(x, y, x + 190.0, y, th.muted, 1.2)
        s.line(x, y, x, y - 58.0, th.muted, 1.2)
        s.text(x + 190.0, y + 15.0, "$log_{10} f$", 12, th.muted, anchor="end")

    # --- white noise -------------------------------------------------------
    tile(55, 62, 250, "White noise")
    d = "M 75 140"
    for i in range(1, 36):
        r = math.sin(i * 12.9898) * 43758.5453
        r -= math.floor(r)
        d += f" L {75 + i * 6:.0f} {140 - (r - 0.5) * 62:.1f}"
    s.path(d, stroke=th.primary, sw=1.3)
    spectrum_axes(85.0, 268.0)
    s.line(90.0, 226.0, 270.0, 226.0, th.accent, 2.2)
    s.text(180.0, 250.0, "flat PSD: 0 dB/octave", 14, th.fg)
    s.text(180.0, 294.0, "equal power per hertz", 13, th.muted)

    # --- pink noise --------------------------------------------------------
    tile(325, 62, 250, "Pink noise")
    d = "M 345 146"
    for i in range(1, 36):
        v = (20.0 * math.sin(0.31 * i) + 10.0 * math.sin(0.83 * i + 1.7)
             + 6.0 * math.sin(2.2 * i + 0.5) + 3.0 * math.sin(5.1 * i))
        d += f" L {345 + i * 6:.0f} {140 - v:.1f}"
    s.path(d, stroke=th.primary, sw=1.3)
    spectrum_axes(355.0, 268.0)
    s.line(360.0, 214.0, 540.0, 248.0, th.accent, 2.2)
    s.text(422.0, 204.0, "−3 dB/octave PSD", 14, th.fg)
    s.text(450.0, 294.0, "equal power per octave", 13, th.muted)

    # --- MLS ---------------------------------------------------------------
    tile(595, 62, 250, "MLS")
    bits = [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1]
    d = ""
    for i, b in enumerate(bits):
        xa, xb = 620 + i * 12, 632 + i * 12
        yl = 140 - 26 if b else 140 + 26
        d += f"{'M' if i == 0 else 'L'} {xa} {yl} L {xb} {yl} "
    s.path(d, stroke=th.primary, sw=1.6)
    spectrum_axes(625.0, 268.0)
    s.line(630.0, 226.0, 810.0, 226.0, th.accent, 2.2)
    s.text(720.0, 250.0, "flat, line spectrum", 14, th.fg)
    s.text(720.0, 294.0, "binary ±1, period $2^m − 1$ samples", 13, th.muted)

    # --- sweeps: linear vs exponential (wide tile) -------------------------
    tile(55, 318, 520, "Sweeps: linear vs exponential")
    s.line(95.0, 520.0, 545.0, 520.0, th.muted, 1.2)
    s.arrow(95.0, 520.0, 95.0, 372.0, th.muted, 1.2)
    s.text(82.0, 372.0, "$f$", 13, th.muted)
    s.text(548.0, 534.0, "$t$", 13, th.muted)
    s.line(95.0, 516.0, 540.0, 380.0, th.primary, 2.2)
    s.text(268.0, 428.0, "linear", 14, th.primary)
    pts = [(95, 519), (206, 517), (295, 513), (362, 505), (410, 494),
           (451, 475), (473, 460), (495, 441), (518, 415), (540, 380)]
    d = "M 95 519"
    for px_, py_ in pts[1:]:
        d += f" L {px_} {py_}"
    s.path(d, stroke=th.secondary, sw=2.2)
    s.text(438.0, 508.0, "exponential", 14, th.secondary)
    s.text(315.0, 548.0,
           "exponential: equal time (and energy) per octave; linear: equal time per hertz",
           13, th.muted)

    # --- tone burst --------------------------------------------------------
    tile(595, 318, 250, "Tone burst")
    s.line(615.0, 440.0, 825.0, 440.0, th.muted, 1.2)
    d = "M 665 440"
    for i in range(1, 111):
        d += f" L {665 + i:.0f} {440 - 42 * math.sin(2 * math.pi * i / 22):.1f}"
    s.path(d, stroke=th.primary, sw=1.6)
    s.rect(663, 394, 114, 92, "none", th.secondary, rx=4, sw=1.2, dash="5,4")
    s.text(720.0, 508.0, "whole periods, starting at", 13, th.muted)
    s.text(720.0, 526.0, "a zero crossing (IEC 60268-1)", 13, th.muted)
    s.text(720.0, 548.0, "25 periods of 5 kHz = 5 ms", 12, th.fg, mono=True)

    # --- captions ----------------------------------------------------------
    s.text(80.0, 590.0,
           "every stimulus is deterministic and repeatable; synchronous "
           "averaging lowers uncorrelated noise",
           16, th.fg, anchor="start")
    s.text(80.0, 616.0,
           "sweeps separate harmonic distortion, MLS smears it across the period, bursts probe dynamics",
           16, th.muted, anchor="start")


# ---------------------------------------------------------------------------
# Welch PSD pipeline (Bendat & Piersol)
# ---------------------------------------------------------------------------

def _d_spectral_analysis(s: SVG, th: Theme) -> None:
    """The Welch estimator as a chain, with the numbers of the guide's own
    example (fs = 48 kHz, 20 s of pink noise, nperseg = 4096): 467 raw
    segments, 442 effective averages, eps_r = 4.8 %."""
    cx, bw = 450.0, 680.0
    x0 = cx - bw / 2

    def step(y: float, l1: str, l2: str, color: str, h: float = 58.0) -> None:
        s.rect(x0, y, bw, h, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 25, l1, 17, th.fg, bold=True)
        s.text(cx, y + 45, l2, 13, th.muted)

    step(52, "Record $x(t)$ — $f_s$ = 48 kHz, 20 s of pink noise",
         "960 000 samples, calibrated end to end: pascals in, Pa²/Hz out",
         th.fg)
    step(138, "Split into 50 %-overlapped segments — nperseg = 4096",
         "467 segments of 85.3 ms; bin spacing $Δf = f_s/4096$ = 11.7 Hz",
         th.primary)
    step(224, "Hann taper on every segment",
         "ENBW = 1.5 bins → resolution bandwidth $B_e = 1.5·Δf$ = 17.6 Hz",
         th.primary)
    step(310, "One-sided $|FFT|^2$ periodogram of each segment, then average",
         "overlap correlation (Welch 1967): 467 segments → $n_d$ = 442 "
         "effective averages", th.fg)
    s.rect(x0, 396, bw, 60, "none", th.accent, rx=10, sw=2.4)
    s.text(cx, 421, "$G_{xx}(f)$ with its chi-square confidence interval", 17,
           th.fg, bold=True)
    s.text(cx, 443, "random error $ε_r = 1/√n_d$ = 4.8 %;  $2·n_d ≈ 885$ degrees "
           "of freedom", 13, th.muted)
    for y0, y1 in ((110, 134), (196, 220), (282, 306), (368, 392),
                   (456, 484)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)

    s.rect(130, 488, 640, 72, "none", th.secondary, rx=10, sw=1.6, dash="6,5")
    s.text(cx, 517, "The trade-off: segment length buys resolution or "
           "stability, never both", 16, th.secondary, bold=True)
    s.text(cx, 543, "longer segments → finer $B_e$ but fewer averages (larger "
           "$ε_r$); shorter → the reverse", 13, th.fg)


# ---------------------------------------------------------------------------
# MISO coherence conditioning (Bendat & Piersol Chapter 7)
# ---------------------------------------------------------------------------

def _d_miso_coherence(s: SVG, th: Theme) -> None:
    """Two correlated inputs through their paths into one output, then the
    Welch cross-spectral matrix, the conditioning and the per-source split,
    with the guide's measured numbers (ordinary 0.32 vs partial 0.00)."""
    def box(x0: float, x1: float, y0: float, h: float, l1: str, l2: str,
            color: str, s1: int = 16, s2: int = 12) -> None:
        s.rect(x0, y0, x1 - x0, h, th.panel, color, rx=10, sw=2)
        s.text((x0 + x1) / 2, y0 + 24.0, l1, s1, th.fg, bold=True)
        s.text((x0 + x1) / 2, y0 + 44.0, l2, s2, th.muted)

    box(60, 280, 64, 56, "Input $x_1$", "white noise", th.primary)
    box(60, 280, 168, 56, "Input $x_2 = 0.7·x_1$ + noise", "correlated with $x_1$",
        th.primary, s1=14)
    s.line(110, 120, 110, 168, th.muted, 1.4, dash="5,4")

    box(340, 540, 64, 56, "Path $H_{1}(f)$", "low-pass, 400 Hz", th.fg)
    box(340, 540, 168, 56, "Path $H_{2}(f)$", "high-pass, 1.5 kHz", th.fg)
    s.arrow(280, 92, 336, 92, th.fg, 1.8)
    s.arrow(280, 196, 336, 196, th.fg, 1.8)

    s.circle(600, 144, 16, th.panel, th.fg, 2)
    s.text(600, 151, "+", 22, th.fg, bold=True)
    s.arrow(540, 92, 588, 134, th.fg, 1.8)
    s.arrow(540, 196, 588, 154, th.fg, 1.8)
    s.text(600, 74, "noise $n(t)$", 13, th.muted)
    s.arrow(600, 82, 600, 124, th.muted, 1.4)

    box(660, 850, 116, 56, "Output $y(t)$", "$G_{yy}(f)$", th.secondary)
    s.arrow(616, 144, 656, 144, th.fg, 1.8)

    s.arrow(755, 172, 755, 236, th.fg, 1.8)
    s.rect(90, 240, 720, 64, th.panel, th.fg, rx=10, sw=2)
    s.text(450, 265, "Welch cross-spectral matrix — $G_{xx}$ (2×2) and $G_{xy}$, "
           "nperseg = 2048", 16, th.fg, bold=True)
    s.text(450, 287, "conditioning: Schur steps $G_{ij·r!}$ (Eq. 7.94), inputs "
           "ordered by descending ordinary coherence", 13, th.muted)

    s.arrow(270, 304, 270, 340, th.fg, 1.8)
    s.arrow(630, 304, 630, 340, th.fg, 1.8)

    s.rect(70, 344, 370, 88, th.panel, th.primary, rx=10, sw=2)
    s.text(255, 370, "Multiple and partial coherence", 15, th.fg, bold=True)
    s.text(255, 392, "input 2 in the 100-300 Hz band: ordinary 0.32 → "
           "partial 0.00", 12, th.muted)
    s.text(255, 412, "multiple $γ^2_{y:x} = 1 − G_{nn}/G_{yy}$ ≈ 1.00 (100-300 Hz)", 12,
           th.muted)

    s.rect(460, 344, 370, 88, th.panel, th.accent, rx=10, sw=2)
    s.text(645, 370, "Contribution of each source", 15, th.fg, bold=True)
    s.text(645, 392, "$G_{vi} = γ^2_{iy·(i−1)!}·G_{yy}$ per input", 12, th.muted)
    s.text(645, 412, "$ΣG_{vi} + G_{nn} = G_{yy}$, band by band", 12, th.muted)

    s.text(450, 482, "each conditioning step spends one average: the $i$-th "
           "ordered input carries $n_d − (i − 1)$; here $n_d$ = 242", 14, th.fg)
    s.text(450, 506, "average generously before reading a small partial "
           "coherence as zero", 13, th.muted)


# ---------------------------------------------------------------------------
# Time-frequency tiling trade-off (Bendat & Piersol 12.6.4.2)
# ---------------------------------------------------------------------------

def _d_time_frequency(s: SVG, th: Theme) -> None:
    """The same record tiled by a short and a long STFT window at
    fs = 16 kHz: nperseg = 256 (16 ms x 62.5 Hz cells) against 1024
    (64 ms x 15.6 Hz), with a tone and a click smeared to cell size."""
    panels = (
        (100.0, 24.0, 70.0, "Short window — nperseg = 256",
         "$T_B$ = 16 ms,  $B_e ≈ 1/T_B$ = 62.5 Hz", "sharp click, smeared tone",
         180.0, 70.0, 244.0, 24.0),
        (512.0, 96.0, 17.5, "Long window — nperseg = 1024",
         "$T_B$ = 64 ms,  $B_e$ ≈ 15.6 Hz", "sharp tone, smeared click",
         215.0, 17.5, 608.0, 96.0),
    )
    top, bot, w = 112.0, 392.0, 288.0
    for x0, cw, rh, header, res, verdict, ty, tth, cxx, cw2 in panels:
        s.text(x0 + w / 2, 74, header, 17, th.fg, bold=True)
        # tone band (frequency stripe) and click band (time stripe)
        s.rect(x0, ty, w, tth, th.primary)
        s.rect(cxx, top, cw2, bot - top, th.secondary)
        # grid over the highlighted cells
        x = x0
        while x <= x0 + w + 0.1:
            s.line(x, top, x, bot, th.muted, 0.7)
            x += cw
        y = top
        while y <= bot + 0.1:
            s.line(x0, y, x0 + w, y, th.muted, 0.7)
            y += rh
        # axes
        s.arrow(x0, bot, x0 + w + 24, bot, th.fg, 1.6)
        s.arrow(x0, bot, x0, top - 16, th.fg, 1.6)
        s.text(x0 + w + 30, bot + 16, "$t$", 14, th.muted)
        s.text(x0 - 14, top - 8, "$f$", 14, th.muted)
        s.text(x0 - 8, ty + tth / 2 + 5, "tone", 12, th.fg, anchor="end")
        s.text(cxx + cw2 / 2, 104, "click", 12, th.fg)
        s.text(x0 + w / 2, 426, res, 14, th.fg)
        s.text(x0 + w / 2, 450, verdict, 13, th.muted, italic=True)

    s.text(450, 498, "each cell is one unaveraged estimate: $B_e·T_B ≈ 1$ and "
           "$ε_r = 1$ ($n_d = 1$)", 15, th.fg, bold=True)
    s.text(450, 524, "the record fixes the product; nperseg only chooses how "
           "to spend it ($f_s$ = 16 kHz here)", 14, th.muted)


# ---------------------------------------------------------------------------
# Cepstrum chain: echo to quefrency spike (Havelock Ch. 27)
# ---------------------------------------------------------------------------

def _d_cepstrum_echoes(s: SVG, th: Theme) -> None:
    """Signal with an 8 ms echo, rippled spectrum, log, inverse FFT, and the
    quefrency axis with the rahmonic spikes and the lifter split, using the
    guide's exact numbers (a = 0.5, 1/t0 = 125 Hz, +3.5/−6.0 dB)."""
    def box(x0: float, x1: float, l1: str, l2: str, l3: str,
            color: str) -> None:
        s.rect(x0, 64, x1 - x0, 86, th.panel, color, rx=10, sw=2)
        s.text((x0 + x1) / 2, 90, l1, 15, th.fg, bold=True)
        s.text((x0 + x1) / 2, 112, l2, 12, th.muted)
        s.text((x0 + x1) / 2, 132, l3, 12, th.muted)

    box(48, 238, "Signal with one echo", "$x = s(t) + a·s(t − t_0)$",
        "$a$ = 0.5,  $t_0$ = 8 ms", th.fg)
    box(262, 442, "Ripply spectrum $|X(f)|$", "cosine ripple of period",
        "$1/t_0$ = 125 Hz", th.primary)
    box(466, 646, "Take the log: $ln |X|^2$", "the multiplicative echo",
        "becomes an additive ripple", th.primary)
    box(670, 860, "Inverse FFT", "quefrency axis, in seconds",
        "the cepstrum", th.secondary)
    for xa in (238.0, 442.0, 646.0):
        s.arrow(xa + 2, 107, xa + 22, 107, th.fg, 1.8)
    s.arrow(765, 152, 765, 196, th.fg, 1.8)

    # Quefrency panel: source envelope, rahmonics, lifter split.
    s.rect(70, 200, 760, 240, th.panel, th.fg, rx=10, sw=1.8)
    base = 370.0
    px_ms = 34.0  # horizontal scale
    x_of = 110.0
    s.line(x_of, base, 790, base, th.fg, 1.6)
    s.arrow(790, base, 806, base, th.fg, 1.6)
    s.text(800, 390, "quefrency", 13, th.muted, anchor="end")
    # source wavelet envelope below 2 ms
    s.path(f"M {x_of:.0f} {base - 76:.0f} "
           f"Q {x_of + 22:.0f} {base - 10:.0f} {x_of + 68:.0f} {base:.0f}",
           stroke=th.muted, sw=2.0)
    s.text(170, 244, "source wavelet,", 12, th.muted)
    s.text(170, 262, "below 2 ms", 12, th.muted)
    # first rahmonic at t0 = 8 ms, height a = 0.5 (scale 220 px per unit)
    x1 = x_of + 8 * px_ms
    s.line(x1, base, x1, base - 110, th.primary, 2.6)
    s.circle(x1, base - 110, 3.5, th.primary)
    s.text(x1, 246, "$a$ = 0.5 at $t_0$ = 8 ms", 14, th.primary, bold=True)
    # second rahmonic at 2 t0, height -a^2/2 = -0.125
    x2 = x_of + 16 * px_ms
    s.line(x2, base, x2, base + 27, th.secondary, 2.2)
    s.circle(x2, base + 27, 3.0, th.secondary)
    s.text(x2 + 14, base + 34, "$−a^2/2$ = −0.125", 12, th.secondary,
           anchor="start")
    # lifter cutoff
    xc = x_of + 4 * px_ms
    s.line(xc, 224, xc, 414, th.accent, 1.6, dash="6,5")
    s.text(xc, 432, "lifter cutoff 4 ms", 12, th.accent)
    s.text(176, 218, "lowpass: envelope", 12, th.fg)
    s.text(450, 218, "highpass: the echo ripple alone", 12, th.fg)
    # the 16 ms label sits lower to clear the downward second rahmonic
    for ms, lbl, dy in ((0.0, "0", 22.0), (8.0, "8 ms", 22.0),
                        (16.0, "16 ms", 50.0)):
        xt = x_of + ms * px_ms
        s.line(xt, base, xt, base + 6, th.fg, 1.4)
        s.text(xt, base + dy, lbl, 12, th.muted)

    s.text(450, 478, "rahmonics at $n·t_0$ with heights $a$, $−a^2/2$, $a^3/3$, …, "
           "whatever the source spectrum does", 15, th.fg, bold=True)
    s.text(450, 504, "the highpass ripple swings between $20·log_{10}(1 ± a)$ = +3.5 "
           "and −6.0 dB; echo_detection reads $t_0$ and $a$ off the peak", 13,
           th.muted)


# ---------------------------------------------------------------------------
# Time synchronous averaging (McFadden 1987)
# ---------------------------------------------------------------------------

def _d_echo_geometry(s: SVG, th: Theme) -> None:
    """Where the quefrency of an echo comes from, drawn to scale.

    Elevation at 150 units per metre: source and microphone 1.00 m apart at
    1.20 m over a hard floor, the floor-reflected path built through the image
    source, and a second panel for the side wall that gives the page's own
    8 ms example.
    """
    scale = 110.0
    c = 343.0

    # ---- Panel A: the floor reflection --------------------------------
    gy = 330.0
    s.ground(gy, 50, 470)
    sx = 130.0
    mx = sx + 1.0 * scale
    top = gy - 1.2 * scale
    s.rect(sx - 20, top - 22, 40, 44, th.panel, th.primary, rx=5, sw=2)
    s.circle(sx, top, 8, th.primary)
    s.text(sx, top - 34, "source", 17, th.fg, bold=True)
    s.mic(mx, top - 8, gy, 1.0)
    s.text(mx + 30, top - 6, "microphone", 17, th.fg, bold=True, anchor="start")
    s.line(sx + 22, top, mx - 8, top, th.primary, 2.4)
    s.text((sx + mx) / 2 + 28, top - 14, "$r_d$ = 1.00 m", 16, th.primary,
           bold=True)

    # Image source under the floor, and the reflected path through it.
    iy = gy + 1.2 * scale
    s.circle(sx, iy, 8, "none", th.muted, 1.6)
    s.text(sx, iy + 24, "image source", 16, th.muted)
    s.line(sx, top, sx, iy, th.muted, 1.0, dash="4,4")
    bounce_x = sx + (mx - sx) / 2.0
    s.line(sx, top, bounce_x, gy, th.secondary, 2.2)
    s.line(bounce_x, gy, mx, top - 8, th.secondary, 2.2)
    s.line(sx, iy, mx, top - 8, th.muted, 1.0, dash="4,4")
    s.circle(bounce_x, gy, 4, th.secondary)
    s.text(bounce_x - 70, gy - 46, "$r_r$ = 2.60 m", 16, th.secondary,
           bold=True)
    s.dim(mx + 46, top, mx + 46, gy, "1.20 m", size=16, label_side="right")
    s.line(mx, top - 8, mx + 46, top, th.muted, 0.9, dash="3,3")
    s.dim(sx, gy + 44, mx, gy + 44, "1.00 m", size=16)

    box_x = 500.0
    s.rect(box_x, 120, 356, 150, "none", th.fg, rx=10, sw=1.6)
    s.text(box_x + 178, 150, "Floor reflection", 20, th.fg, bold=True)
    dd = 2.6 - 1.0
    s.text(box_x + 178, 182, f"$Δd = r_r − r_d$ = {dd:.2f} m", 18, th.fg)
    s.text(box_x + 178, 210, f"$t_0 = Δd / c$ = {1000 * dd / c:.1f} ms",
           18, th.secondary, bold=True)
    s.text(box_x + 178, 240, "$a = R · r_d / r_r = 0.38 R$", 18, th.fg)

    # ---- Panel B: the side wall that gives 8 ms ------------------------
    s.rect(box_x, 292, 356, 168, "none", th.secondary, rx=10, sw=1.6)
    s.text(box_x + 178, 322, "The 8 ms example of this page", 20, th.secondary,
           bold=True)
    s.text(box_x + 178, 352, "$Δd = c · 8 ms$ = 2.74 m", 18, th.fg)
    s.text(box_x + 178, 380, "a side wall 1.37 m from the direct path", 17,
           th.fg)
    s.text(box_x + 178, 408, "$R = a · r_r / r_d = 3.74 a ≤ 1$", 18, th.fg)
    s.text(box_x + 178, 436, "so $a > 0.27$ is not one specular reflection", 16,
           th.muted)

    s.text(450, 512, "The reflection has to arrive before the record ends and "
           "at least 10 dB above its noise floor;", 17, th.muted)
    s.text(450, 534, "$c$ moves about 0.6 m/s per kelvin, so convert the delay "
           "with the temperature you measured", 17, th.muted)


def _d_synchronous_averaging(s: SVG, th: Theme) -> None:
    """Trigger train, sliced recording, coherent average and residual, with
    the guide's numbers: T = 1/32 s at 8192 Hz, N = 40 averages, 16 dB of
    noise reduction, and McFadden's 32.05-order node example."""
    import math

    s.text(450, 64, "Tachometer: one trigger pulse per revolution", 16,
           th.fg, bold=True)
    s.line(80, 112, 840, 112, th.muted, 1.4)
    pulses = [120.0, 280.0, 440.0, 600.0, 760.0]
    for px_ in pulses:
        s.rect(px_ - 3, 86, 6, 26, th.accent, rx=1.5)
    s.dim(280, 133, 440, 133, "$T$ = 1/32 s = 256 samples", size=14)

    s.text(450, 152, "Recording $y(t)$ at $f_s$ = 8192 Hz: the synchronous "
           "signature buried in noise", 14, th.fg)
    d = "M 80 196"
    for i in range(1, 191):
        x = 80 + i * 4
        v = (14.0 * math.sin(2 * math.pi * (x - 120.0) / 160.0)
             + 5.0 * math.sin(1.7 * i) + 3.5 * math.sin(4.3 * i + 1.2))
        d += f" L {x:.0f} {196 - v:.1f}"
    s.path(d, stroke=th.primary, sw=1.2)
    for px_ in pulses:
        s.line(px_, 168, px_, 224, th.fg, 1.1, dash="4,4")
    s.text(450, 246, "slice at every trigger", 13, th.muted)

    # Stack of aligned one-period blocks
    for i in (2, 1, 0):
        s.rect(100 + 10 * i, 280 + 10 * i, 190, 54, th.panel, th.muted,
               rx=8, sw=1.4)
    s.text(205, 302, "$N$ aligned blocks", 14, th.fg, bold=True)
    s.text(205, 322, "one period $T$ each", 12, th.muted)
    s.arrow(190, 254, 195, 276, th.fg, 1.8)

    s.rect(360, 288, 220, 78, th.panel, th.primary, rx=10, sw=2)
    s.text(470, 312, "Coherent average", 16, th.fg, bold=True)
    s.text(470, 334, "$a(t) = (1/N) Σ y(t + n T)$", 13, th.primary)
    s.text(470, 354, "$N$ = 40 here", 12, th.muted)
    s.arrow(310, 322, 356, 322, th.fg, 1.8)

    s.rect(640, 288, 220, 78, th.panel, th.accent, rx=10, sw=2)
    s.text(750, 312, "The periodic part survives", 14, th.fg, bold=True)
    s.text(750, 334, "comb teeth of unit gain", 12, th.muted)
    s.text(750, 354, "at every order $k/T$", 12, th.muted)
    s.arrow(580, 327, 636, 327, th.fg, 1.8)

    s.rect(100, 420, 460, 64, "none", th.accent, rx=10, sw=1.6, dash="6,5")
    s.text(330, 446, "Asynchronous noise falls as $1/√N$", 15, th.accent,
           bold=True)
    s.text(330, 470, "power $−10·log_{10} N$ = −16 dB for $N$ = 40;  amplitude gain "
           "$√N$ = 6.3", 13, th.fg)
    s.arrow(470, 366, 470, 416, th.fg, 1.6)

    s.rect(640, 420, 220, 64, th.panel, th.secondary, rx=10, sw=2)
    s.text(750, 442, "Residual", 14, th.fg, bold=True)
    s.text(750, 460, "record − tiled average:", 12, th.muted)
    s.text(750, 476, "everything not synchronous", 12, th.muted)
    s.arrow(750, 366, 750, 416, th.fg, 1.6)

    s.text(450, 526, "a tone on a non-integer order is only attenuated: "
           "choose $N$ so a comb node lands on it", 14, th.fg)
    s.text(450, 550, "McFadden's example: $N$ = 20 nulls the 32.05-order tone "
           "(20·32.05 = 641); the habitual $N$ = 32 does not", 13, th.muted)


# ---------------------------------------------------------------------------
# Correlation-based time-delay estimation (Knapp & Carter)
# ---------------------------------------------------------------------------

def _d_miso_setup(s: SVG, th: Theme) -> None:
    """Instrumenting a two-source MISO measurement, plan view.

    A plant room drawn to scale at 55 units per metre: a fan and a compressor
    3 m apart, one reference sensor each, the receiver microphone 4 m from
    the fan, the airborne leakage that correlates the two references, and the
    single simultaneously sampling front end every channel has to share.
    """
    import math

    scale = 55.0                       # drawing units per metre
    x0, y0 = 46.0, 92.0
    room_w, room_h = 9.2 * scale, 5.2 * scale
    s.rect(x0, y0, room_w, room_h, th.panel, th.fg, rx=4, sw=2.4)
    s.text(x0 + 12, y0 + 24, "Plant room — plan", 17, th.muted, anchor="start")

    # -- Machine A: a fan on resilient mounts ------------------------------
    ax, ay = x0 + 1.6 * scale, y0 + 3.4 * scale
    s.rect(ax - 40, ay - 40, 80, 80, th.bg, th.primary, rx=6, sw=2.2)
    s.circle(ax, ay, 24, "none", th.primary, 1.6)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        s.line(ax, ay, ax + 22 * math.cos(a), ay + 22 * math.sin(a),
               th.primary, 1.6)
    s.text(ax, ay - 52, "A — fan", 18, th.fg, bold=True)
    for k in (-1, 1):
        s.rect(ax + k * 26 - 8, ay + 40, 16, 9, th.muted, rx=2)
    s.rect(ax + 26 - 7, ay + 26, 14, 14, th.secondary, th.fg, rx=2, sw=1.2)
    s.text(ax + 6, ay + 74, "ref 1", 16, th.secondary, bold=True)

    # -- Machine B: a compressor -------------------------------------------
    bx, by = ax + 3.0 * scale, ay
    s.rect(bx - 36, by - 36, 72, 72, th.bg, th.primary, rx=6, sw=2.2)
    s.rect(bx - 20, by - 16, 40, 32, th.panel, th.primary, rx=3, sw=1.4)
    s.text(bx, by - 48, "B — compressor", 18, th.fg, bold=True)
    mic2_x = bx + 36 + 0.3 * scale
    s.circle(mic2_x, by, 7, th.bg, th.secondary, 2.0)
    s.dim(bx + 36, by + 52, mic2_x, by + 52, "0.3 m", size=15)
    s.line(mic2_x, by + 10, mic2_x, by + 50, th.muted, 0.9, dash="3,3")
    s.text(mic2_x + 20, by + 6, "ref 2", 16, th.secondary, bold=True,
           anchor="start")
    s.dim(ax, y0 + room_h - 22, bx, y0 + room_h - 22, "3.0 m", size=16)

    # -- Receiver microphone at the operator position ----------------------
    rx_, ry = x0 + 7.6 * scale, y0 + 1.0 * scale
    s.circle(rx_, ry, 8, th.bg, th.accent, 2.4)
    s.text(rx_, ry - 18, "receiver", 17, th.accent, bold=True)
    s.text(rx_ - 16, ry + 6, "1.5 m high", 15, th.muted, anchor="end")
    s.line(ax, ay, rx_, ry, th.accent, 1.2, dash="6,5")
    s.line(bx, by, rx_, ry, th.accent, 1.2, dash="6,5")
    s.dim(ax, ay - 74, rx_, ry - 40, "4.0 m", size=15)

    # -- The leakage that correlates the two references --------------------
    s.arrow(ax + 44, ay + 16, mic2_x - 11, by + 6, th.secondary, 1.8)
    s.text((ax + mic2_x) / 2 + 10, ay + 40, "leakage", 15, th.secondary,
           bold=True)

    # -- Legend under the room ---------------------------------------------
    ly = y0 + room_h + 26
    s.circle(x0 + 8, ly - 5, 5, th.secondary)
    s.text(x0 + 22, ly, "ref 1: accelerometer stud-mounted on the fan foot",
           16, th.fg, anchor="start")
    s.circle(x0 + 8, ly + 21, 5, th.secondary)
    s.text(x0 + 22, ly + 26, "ref 2: microphone 0.3 m from the casing", 16,
           th.fg, anchor="start")
    s.circle(x0 + 8, ly + 47, 5, th.secondary)
    s.text(x0 + 22, ly + 52, "the leakage is what correlates the two inputs",
           16, th.secondary, anchor="start")

    # -- Acquisition column -------------------------------------------------
    cx = x0 + room_w + 88
    s.rect(cx - 78, 100, 156, 172, "none", th.primary, rx=10, sw=1.8)
    s.text(cx, 126, "One front end", 18, th.primary, bold=True)
    for i, name in enumerate(("ref 1  $x_1$", "ref 2  $x_2$",
                              "receiver  $y$")):
        s.rect(cx - 62, 140 + i * 32, 124, 24, th.panel, th.fg, rx=4, sw=1.2)
        s.text(cx, 157 + i * 32, name, 14, th.fg)
    s.text(cx, 258, "one clock, fixed gains", 14, th.muted)

    s.rect(cx - 78, 292, 156, 128, "none", th.secondary, rx=10, sw=1.8)
    s.text(cx, 318, "Before reading", 17, th.secondary, bold=True)
    s.text(cx, 338, "the split", 17, th.secondary, bold=True)
    s.text(cx, 364, "coherence between", 14, th.fg)
    s.text(cx, 382, "$x_1$ and $x_2$ > 0.9", 14, th.fg, bold=True)
    s.text(cx, 402, "⇒ do not attribute", 14, th.fg)

    s.text(450, 552, "Conditioning separates only what the references "
                     "separate: one sensor per source, all sampled together", 17,
           th.muted)


def _d_tsa_setup(s: SVG, th: Theme) -> None:
    """The two channels a time synchronous average needs, on a gearbox.

    A single-stage gearbox in elevation: the input shaft at 1800 r/min with
    one strip of reflective tape and an optical tacho head 20 mm away, a
    37-tooth pinion driving an 89-tooth wheel, and a stud-mounted
    accelerometer on the pedestal bearing of the input shaft.
    """
    import math

    gy = 400.0
    s.ground(gy, 40, 470)
    s.rect(70, 140, 380, gy - 140, th.panel, th.fg, rx=6, sw=2.2)
    s.text(80, 164, "Gearbox — elevation", 17, th.muted, anchor="start")

    shaft_y = 252.0
    px, pr = 214.0, 38.0
    wx, wr = 330.0, 78.0
    for cx, r, label, dy in ((px, pr, "pinion, 37 teeth", pr + 26),
                             (wx, wr, "wheel, 89 teeth", wr + 26)):
        s.circle(cx, shaft_y, r, th.bg, th.primary, 2.2)
        s.circle(cx, shaft_y, 6, th.fg)
        for k in range(24):
            a = 2 * math.pi * k / 24
            s.line(cx + r * math.cos(a), shaft_y + r * math.sin(a),
                   cx + (r + 7) * math.cos(a), shaft_y + (r + 7) * math.sin(a),
                   th.primary, 1.4)
        s.text(cx, shaft_y + dy, label, 16, th.fg, bold=True)

    # Input shaft, the reflective tape and the optical tacho head.
    s.line(78, shaft_y, px, shaft_y, th.fg, 4.0)
    s.rect(150, shaft_y - 9, 12, 18, th.secondary, rx=2)
    s.text(156, shaft_y - 18, "tape", 15, th.secondary, bold=True)
    s.rect(130, shaft_y - 86, 52, 32, th.panel, th.secondary, rx=5, sw=1.8)
    s.text(156, shaft_y - 64, "tacho", 15, th.secondary, bold=True)
    s.arrow(156, shaft_y - 52, 156, shaft_y - 16, th.secondary, 1.6)
    s.dim(196, shaft_y - 52, 196, shaft_y - 14, "20 mm", size=14,
          label_side="right")

    # Pedestal bearing on the input shaft, with the accelerometer on its face.
    s.rect(96, shaft_y + 10, 36, gy - shaft_y - 10, th.bg, th.fg, rx=3, sw=1.8)
    s.text(114, gy - 12, "bearing", 14, th.muted)
    s.rect(74, shaft_y + 46, 22, 26, th.primary, th.fg, rx=3, sw=1.4)
    s.arrow(72, shaft_y + 59, 44, shaft_y + 59, th.primary, 1.8)
    s.text(42, shaft_y + 96, "accel.", 15, th.fg, bold=True, anchor="start")
    s.text(42, shaft_y + 114, "(stud)", 14, th.muted, anchor="start")
    s.text(44, shaft_y + 44, "load direction", 15, th.primary, anchor="start")

    s.text(260, 442, "1800 r/min  →  $T$ = 33.3 ms per revolution", 17,
           th.fg, bold=True)
    s.text(260, 468, "mesh frequency 37 × 30 = 1110 Hz; five mesh harmonics",
           16, th.muted)
    s.text(260, 490, "need $f_s ≥ 2.56 × 5550$ = 14.2 kHz", 16, th.muted)

    # -- Right column: the acquisition and the slicing --------------------
    cx = 700.0
    s.rect(cx - 160, 130, 320, 116, "none", th.primary, rx=10, sw=1.8)
    s.text(cx, 158, "One front end, one clock", 18, th.primary, bold=True)
    s.rect(cx - 140, 174, 130, 26, th.panel, th.fg, rx=4, sw=1.2)
    s.text(cx - 75, 192, "tacho", 15, th.fg, mono=True)
    s.rect(cx + 10, 174, 130, 26, th.panel, th.fg, rx=4, sw=1.2)
    s.text(cx + 75, 192, "vibration", 15, th.fg, mono=True)
    s.text(cx, 226, "$f_s$ = 25.6 kHz, gains fixed", 15, th.muted)

    base = 330.0
    s.line(cx - 160, base, cx + 160, base, th.fg, 1.4)
    for k in range(5):
        x = cx - 150 + k * 75
        s.line(x, base, x, base - 34, th.secondary, 2.2)
        s.line(x, base - 34, x + 8, base - 34, th.secondary, 2.2)
        s.line(x + 8, base - 34, x + 8, base, th.secondary, 2.2)
        if k < 4:
            s.line(x, base + 12, x, base + 96, th.muted, 1.0, dash="5,4")
    s.text(cx, base - 50, "one pulse per revolution", 16, th.secondary,
           bold=True)
    wave = "M " + " L ".join(
        f"{cx - 150 + i * 3:.0f} "
        f"{base + 56 + 22 * math.sin(i * 0.9) * math.cos(i * 0.21):.0f}"
        for i in range(101))
    s.path(wave, stroke=th.primary, sw=1.2)
    s.text(cx, base + 122, "the pulse is the block boundary", 16, th.fg)
    s.text(cx, base + 146, "record $N + 1$ revolutions", 16, th.muted)

    s.text(450, 552, "One tacho pulse per turn, an accelerometer in the load "
                     "direction: the period is measured, not assumed", 17,
           th.muted)


def _d_correlation_delay(s: SVG, th: Theme) -> None:
    """Two microphones, the extra path c*tau, and the correlogram where the
    direct correlator smears while GCC-PHAT spikes at the guide's delay of
    20 samples at 8192 Hz (2.44 ms, 0.84 m at 343 m/s)."""
    gy = 300.0
    s.ground(gy, 60, 840)

    # Source loudspeaker, top left
    s.rect(80, 64, 36, 44, th.panel, th.primary, rx=5, sw=2)
    s.circle(98, 80, 8, th.primary)
    s.circle(98, 80, 3, th.bg)
    s.circle(98, 98, 4.5, th.primary)
    s.text(98, 52, "source", 13, th.fg, bold=True)
    for r in (18, 30):
        s.path(f"M {120 + r * 0.30:.0f} {86 - r * 0.55:.0f} "
               f"A {r} {r} 0 0 1 {120 + r * 0.55:.0f} {86 + r * 0.30:.0f}",
               stroke=th.accent, sw=1.5)

    # Rays to the two microphones (drawn first, mics overlay them)
    s.line(120, 90, 330, 188, th.muted, 1.2, dash="6,5")
    s.line(120, 90, 570, 188, th.muted, 1.2, dash="6,5")
    # Wavefront arc through mic 1 crossing the second ray at P
    s.path("M 330 188 A 232 232 0 0 0 346 139", stroke=th.fg, sw=1.6)
    s.line(346.4, 139.3, 570, 188, th.secondary, 2.6)
    s.text(470, 120, "$Δr = c·τ_0$ ≈ 0.84 m  ($c$ = 343 m/s)", 14, th.secondary,
           bold=True)

    s.mic(330, 190, gy, 1.0)
    s.mic(570, 190, gy, 1.0)
    s.text(314, 258, "mic 1 — $x(t)$", 13, th.fg, bold=True, anchor="end")
    s.text(586, 258, "mic 2 — $y(t)$", 13, th.fg, bold=True, anchor="start")
    s.dim(330, 272, 570, 272, "spacing $d$", size=14)
    s.text(450, 242, "$sin θ = c·τ_0 / d$", 14, th.fg)

    # Correlogram panel
    s.rect(70, 340, 760, 210, th.panel, th.fg, rx=10, sw=1.8)
    s.text(450, 366, "cross-correlation against lag — $y(t) = α·x(t − τ_0) + "
           "n(t)$", 15, th.fg, bold=True)
    base = 505.0
    s.line(110, base, 770, base, th.fg, 1.6)
    s.arrow(770, base, 790, base, th.fg, 1.6)
    x_tau = 518.0
    s.path(f"M 388 {base:.0f} Q {x_tau:.0f} 415 648 {base:.0f}",
           stroke=th.muted, sw=2.0)
    s.text(688, 432, "direct correlator: broad peak", 13, th.muted)
    s.line(x_tau, base, x_tau, 400, th.primary, 2.6)
    s.circle(x_tau, 400, 3.5, th.primary)
    s.text(x_tau, 388, "GCC-PHAT: sharp spike", 13, th.primary, bold=True)
    s.text(150, 400, "$ψ(f) = 1/|G_{xy}|$", 13, th.fg, anchor="start")
    s.line(250, base - 5, 250, base + 5, th.fg, 1.4)
    s.text(250, base + 21, "0", 12, th.muted)
    s.text(x_tau, base + 21, "$τ_0$ = 20 samples / 8192 Hz = 2.44 ms", 13,
           th.fg)

    s.text(450, 580, "parabolic peak interpolation + ×16 local upsampling → "
           "error below 0.002 samples", 14, th.fg)
    s.text(450, 604, "the 'phase' route reads the same $τ_0$ from the slope of "
           "the unwrapped cross-spectrum phase", 13, th.muted)


# ---------------------------------------------------------------------------
# Data qualification decision flow (Bendat & Piersol 10.3)
# ---------------------------------------------------------------------------

def _d_data_qualification(s: SVG, th: Theme) -> None:
    """Record to segment mean squares to the reverse arrangement count and
    the Table A.6 verdict, with the guide's numbers: N = 20 segments,
    acceptance (64, 125), A = 91 accepted and A = 7 rejected."""
    cx = 450.0
    x0, bw = 170.0, 560.0

    def step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, 54, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 23, l1, 16, th.fg, bold=True)
        s.text(cx, y + 43, l2, 12, th.muted)

    step(52, "Time record $x(t)$",
         "before trusting any PSD, Leq or GUM average", th.fg)
    step(134, "Mean square per interval — $N$ = 20 equal segments",
         "each interval long against the record's lowest frequencies; also "
         "rms, mean or variance", th.primary)
    step(216, "Reverse arrangement count $A$",
         "pairs $i < j$ with $x_i > x_j$; trend-free mean $μ_A = N(N−1)/4$ = 95",
         th.primary)
    for y0, y1 in ((106, 130), (188, 212), (270, 294)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)

    # Decision diamond
    s.path(f"M 240 340 L {cx:.0f} 296 L 660 340 L {cx:.0f} 384 Z",
           fill=th.panel, stroke=th.fg, sw=2)
    s.text(cx, 336, "$64 < A ≤ 125$ ?", 17, th.fg, bold=True)
    s.text(cx, 358, "(Table A.6, $α = 0.05$)", 12, th.muted)

    s.arrow(390, 380, 250, 426, th.secondary, 1.8)
    s.text(298, 392, "no", 14, th.secondary, bold=True)
    s.arrow(510, 380, 650, 426, th.accent, 1.8)
    s.text(602, 392, "yes", 14, th.accent, bold=True)

    s.rect(60, 430, 360, 96, th.panel, th.secondary, rx=10, sw=2.2)
    s.text(240, 458, "Nonstationary: do not average", 15, th.fg, bold=True)
    s.text(240, 482, "+20 % gain ramp: $A = 7$ → rejected", 13, th.secondary)
    s.text(240, 504, "split at the change, or go short-time (spectrogram)",
           12, th.muted)

    s.rect(480, 430, 360, 96, th.panel, th.accent, rx=10, sw=2.2)
    s.text(660, 458, "Stationary: analyse", 15, th.fg, bold=True)
    s.text(660, 482, "steady noise: $A = 91$ → accepted", 13, th.accent)
    s.text(660, 504, "the chi-square CIs and error formulas hold", 12,
           th.muted)

    s.text(cx, 566, "the runs test (method=\"runs\") is the two-sided "
           "companion: too many runs is as suspect as too few", 14, th.fg)
    s.text(cx, 590, "a frequency glide can hide from the mean square: test "
           "statistic=\"mean\" or band-filtered copies too", 13, th.muted)


# ---------------------------------------------------------------------------
# Sound level meter pipeline: the library functions behind each IEC 61672-1
# stage, as assembled by the "Build a sound level meter" guide
# ---------------------------------------------------------------------------

def _d_slm_pipeline(s: SVG, th: Theme) -> None:
    """The guide's own pipeline: the two recordings that go in, the single
    sensitivity factor that makes them physical, and the three readout
    branches (display statistics, integrated levels, band spectrum) that a
    class 1 meter reports, closed by the class verifiers."""
    # --- The two recordings the meter needs, from the same input chain -----
    for x0, l1, l2 in ((40.0, "Calibrator tone", "94 dB at 1 kHz  (IEC 60942)"),
                       (480.0, "Measurement recording", "same microphone, same gain")):
        s.rect(x0, 54, 380, 68, th.panel, th.primary, rx=12, sw=2)
        s.text(x0 + 190, 84, l1, 21, th.fg, bold=True)
        s.text(x0 + 190, 108, l2, 16, th.muted)

    # --- The sensitivity factor, derived from the calibrator tone ----------
    s.arrow(230, 122, 230, 150, th.fg, 2.0)
    s.rect(40, 150, 380, 78, th.panel, th.primary, rx=12, sw=2)
    s.text(230, 182, "sensitivity(calibrator, target_spl=94.0, fs=fs)", 13,
           th.fg, bold=True, mono=True)
    s.text(230, 208, "the factor $S$ in pascals per digital unit", 15, th.muted)

    # --- Calibrated pressure: where both inputs meet -----------------------
    s.arrow(230, 228, 230, 266, th.fg, 2.0)
    s.arrow(670, 122, 670, 266, th.fg, 2.0)
    s.rect(120, 266, 660, 64, "none", th.accent, rx=12, sw=2.4)
    s.text(450, 296, "Calibrated pressure   $p(t) = S · x(t)$   in pascals", 21,
           th.fg, bold=True)
    s.text(450, 320, "every level function takes $S$ as calibration_factor=",
           16, th.accent)

    # --- Three readout branches, one guide section each --------------------
    branches = [
        (40.0, "Display and statistics", "weighting_filter(curve='A')",
         "time_weighting(mode='fast')",
         "exponential detector, $τ_F$ = 125 ms",
         "$L_{AF}(t)$   $L_{10} / L_{50} / L_{90}$"),
        (320.0, "Integrated levels", "laeq · sel · lc_peak",
         "", "energy average, no ballistics",
         "$L_{Aeq}$   $L_{AE}$   $L_{Cpeak}$"),
        (600.0, "Band spectrum", "octave_filter(fraction=3)",
         "OctaveFilterBank", "IEC 61260-1 band edges",
         "one-third-octave band levels"),
    ]
    for x0, head, code1, code2, note, out in branches:
        cx = x0 + 130
        s.arrow(cx, 330, cx, 370, th.fg, 2.0)
        s.rect(x0, 370, 260, 104, th.panel, th.primary, rx=12, sw=2)
        s.text(cx, 400, head, 19, th.fg, bold=True)
        # A branch with a single call keeps its one line centred in the gap the
        # two-line branches use, rather than leaving a hole under the heading.
        s.text(cx, 426 if code2 else 437, code1, 14, th.fg, mono=True)
        if code2:
            s.text(cx, 448, code2, 14, th.fg, mono=True)
        s.text(cx, 468, note, 14, th.muted)
        s.arrow(cx, 474, cx, 510, th.fg, 2.0)
        s.rect(x0, 510, 260, 62, "none", th.accent, rx=12, sw=2.2)
        s.text(cx, 538, out, 15, th.accent, bold=True)
        s.text(cx, 560, "dB re 20 µPa", 14, th.muted)

    # --- Class verification closes the guide -------------------------------
    for cx in (170.0, 730.0):
        s.line(cx, 572, cx, 588, th.muted, 1.4, dash="5,4")
        s.arrow(cx, 588, cx, 596, th.muted, 1.4)
    s.rect(40, 596, 820, 62, "none", th.secondary, rx=12, sw=2, dash="7,5")
    s.text(450, 624, "Class verification against the acceptance limits", 18,
           th.secondary, bold=True)
    s.text(450, 648,
           "verify_weighting_class (IEC 61672-1 Table 3)  ·  "
           "verify_filter_class (IEC 61260-1 Table 1)", 14, th.muted)


# ---------------------------------------------------------------------------
# Calibration data flow: from the two recordings to levels in dB SPL
# ---------------------------------------------------------------------------

def _d_calibration_dataflow(s: SVG, th: Theme) -> None:
    """The data flow of the calibration guide: the calibrator recording
    yields one factor, the measurement recording carries the samples, and
    every level function takes the factor as ``calibration_factor``. The
    dBFS reference frame is the branch taken when no factor exists."""
    # --- The two recordings, which must come from the same untouched chain --
    for x0, l1, l2 in ((40.0, "Calibrator recording", "1 kHz tone through the chain"),
                       (500.0, "Measurement recording", "the same chain, untouched")):
        s.rect(x0, 54, 360, 72, th.panel, th.primary, rx=12, sw=2)
        s.text(x0 + 180, 84, l1, 21, th.fg, bold=True)
        s.text(x0 + 180, 108, l2, 16, th.muted)
    s.line(400, 90, 500, 90, th.muted, 1.4, dash="6,5")
    s.text(450, 152, "nothing in the chain may change between the two", 15,
           th.muted, italic=True)

    # --- sensitivity(): the equation of the guide, plus its stability check -
    s.arrow(220, 126, 220, 166, th.fg, 2.0)
    s.rect(40, 166, 360, 88, th.panel, th.primary, rx=12, sw=2)
    s.text(220, 196, "sensitivity(calibrator, target_spl=94.0, fs=fs)", 12,
           th.fg, bold=True, mono=True)
    # Parked: the 10^(L_cal / 20) exponent carries a subscript inside
    # the superscript, one script level more than the composer sets, so
    # the formula stays plain until that family is adjudicated.
    s.text(220, 222, "S = p_ref · 10^(L_cal / 20) / x̃_ref", 16, th.fg)
    s.text(220, 244, "fs enables the IEC 60942 stability check", 14, th.muted)

    # --- The factor itself --------------------------------------------------
    s.arrow(220, 254, 220, 290, th.fg, 2.0)
    s.rect(40, 290, 360, 72, "none", th.accent, rx=12, sw=2.4)
    s.text(220, 320, "calibration_factor  S", 18, th.fg, bold=True, mono=True)
    s.text(220, 344, "pascals per digital unit", 16, th.accent)

    # --- Where the factor and the samples meet ------------------------------
    s.arrow(240, 362, 320, 400, th.fg, 2.0)
    s.arrow(680, 126, 680, 400, th.fg, 2.0)
    s.rect(60, 400, 780, 86, th.panel, th.fg, rx=12, sw=2)
    s.text(450, 430,
           "octave_filter · leq · laeq · sel · ln_levels · lc_peak · OctaveFilterBank",
           14, th.fg, mono=True)
    s.text(450, 456, "every level function accepts calibration_factor=", 17,
           th.fg, bold=True)
    s.text(450, 478, "one factor for the whole library", 14, th.muted)

    # --- The physical result, and the dBFS branch beside it -----------------
    s.arrow(450, 486, 450, 522, th.fg, 2.0)
    s.rect(270, 522, 360, 64, "none", th.accent, rx=12, sw=2.4)
    s.text(450, 550, "Levels in dB SPL", 21, th.fg, bold=True)
    s.text(450, 574, "re 20 µPa", 16, th.accent, mono=True)
    s.rect(650, 514, 210, 80, "none", th.secondary, rx=12, sw=1.8, dash="7,5")
    s.text(755, 542, "No calibrator?", 16, th.secondary, bold=True)
    s.text(755, 564, "$S = 1$, samples read as Pa", 13, th.muted)
    s.text(755, 584, "use dbfs=True for dBFS", 13, th.fg)


# ---------------------------------------------------------------------------
# Filter bank data flow: the decimation decision and the band outputs
# ---------------------------------------------------------------------------

def _d_bank_dataflow(s: SVG, th: Theme) -> None:
    """The two numerical-stability strategies of the filter bank as one
    path: every band is a biquad cascade, and a low band takes the decimated
    branch first. Both branches end in the band level; ``sigbands=True``
    also brings the band signal back to the input rate."""
    # --- Input and the per-band decision ------------------------------------
    s.rect(290, 54, 320, 64, th.panel, th.fg, rx=12, sw=2)
    s.text(450, 84, "Input signal  $x(t)$", 21, th.fg, bold=True)
    s.text(450, 108, "sample rate $f_s$", 16, th.muted)
    s.arrow(450, 118, 450, 132, th.fg, 2.0)
    s.path("M 450 132 L 610 184 L 450 236 L 290 184 Z", th.panel, th.primary,
           sw=2)
    s.text(450, 180, "Room to decimate?", 20, th.fg, bold=True)
    s.text(450, 204, "$f_s / 2 ≥ 1.25 · f_{upper}$", 14, th.muted)

    s.line(290, 184, 190, 184, th.fg, 2.0)
    s.arrow(190, 184, 190, 244, th.fg, 2.0)
    s.text(240, 172, "yes", 15, th.accent, bold=True)
    s.line(610, 184, 710, 184, th.fg, 2.0)
    s.arrow(710, 184, 710, 360, th.fg, 2.0)
    s.text(660, 172, "no", 15, th.secondary, bold=True)

    # --- The decimated branch ----------------------------------------------
    s.rect(50, 244, 280, 80, th.panel, th.primary, rx=12, sw=2)
    s.text(190, 274, "resample_poly(1, M)", 15, th.fg, bold=True, mono=True)
    s.text(190, 298, "$M = floor[(f_s / 2) / (1.25 · f_{upper})]$", 13, th.muted)
    s.text(190, 318, "poles stay clear of $z = 1$", 14, th.muted)
    s.arrow(190, 324, 190, 360, th.fg, 2.0)

    # --- Both branches are the same biquad cascade at a different rate ------
    for x0, head in ((50.0, "SOS band filter at $f_s / M$"),
                     (570.0, "SOS band filter at $f_s$")):
        s.rect(x0, 360, 280, 84, th.panel, th.primary, rx=12, sw=2)
        s.text(x0 + 140, 390, head, 18, th.fg, bold=True)
        s.text(x0 + 140, 414, "cascaded biquads", 15, th.muted)
        s.text(x0 + 140, 434, "designed on the IEC 61260-1 band edges", 12,
               th.muted)
    s.rect(340, 260, 220, 96, "none", th.secondary, rx=12, sw=1.8, dash="7,5")
    s.text(450, 288, "Every band filter", 14, th.secondary, bold=True)
    s.text(450, 308, "is a biquad cascade", 14, th.secondary, bold=True)
    s.text(450, 332, "not one high-order", 14, th.muted)
    s.text(450, 350, "(b, a) pair", 14, th.muted)

    # --- The band level, and the optional band signal -----------------------
    s.arrow(190, 444, 330, 496, th.fg, 2.0)
    s.arrow(710, 444, 570, 496, th.fg, 2.0)
    s.rect(270, 500, 360, 76, "none", th.accent, rx=12, sw=2.4)
    s.text(450, 530, "Band level", 21, th.fg, bold=True)
    s.text(450, 554, "RMS or peak, in dB re 20 µPa", 16, th.accent)
    s.rect(50, 604, 800, 62, "none", th.secondary, rx=12, sw=1.8, dash="7,5")
    s.text(450, 632, "sigbands=True also returns the band signal at $f_s$", 17,
           th.secondary, bold=True)
    s.text(450, 654,
           "the decimated branch is interpolated back with resample_poly(M, 1)",
           14, th.muted)


# ---------------------------------------------------------------------------
# d27 - The infrasound measurement chain (ISO 7196 Annex A)
# ---------------------------------------------------------------------------

_G_POLES_HZ: tuple[tuple[float, float], ...] = (
    (-0.707, 0.707), (-0.707, -0.707), (-19.27, 5.16), (-19.27, -5.16),
    (-14.11, 14.11), (-14.11, -14.11), (-5.16, 19.27), (-5.16, -19.27),
)


def _g_weight_db(f: float) -> float:
    """ISO 7196:1995 G weighting from its Table 1 poles, 0 dB at 10 Hz."""
    import math

    def gain(x: float) -> float:
        jw = 2j * math.pi * x
        num = jw ** 4                       # four zeros at the origin
        den = 1.0 + 0j
        for re, im in _G_POLES_HZ:
            den *= jw - 2 * math.pi * complex(re, im)
        return abs(num / den)

    return 20 * math.log10(gain(f) / gain(10.0))


def _d_infrasound_chain(s: SVG, th: Theme) -> None:
    """The chain that has to deliver 0,25 Hz, and where it usually does not.

    Left: section through the ground board with the capsule at its centre,
    the static-pressure equalisation vent called out, and the two windscreens.
    Middle: the three electrical stages. Right: the G curve against the
    chain's own high-pass corner, so the usable band is the overlap.
    Geometry after ISO 7196:1995 Annex A (A.2 microphone, A.5 integrator).
    """
    import math

    # -- Left: the capsule on the ground board ------------------------------
    gy = 372.0
    cx = 168.0
    s.ground(gy, 34, 300)
    s.rect(cx - 118, gy - 12, 236, 12, th.panel, th.fg, sw=2)   # hard board
    s.text(cx, gy + 40, "hard board on the ground", 18, th.fg, bold=True)
    s.text(cx, gy + 62, "capsule flush at its centre", 16, th.muted)

    # Capsule, drawn flush with the board, plus its equalisation vent.
    s.rect(cx - 26, gy - 40, 52, 28, th.panel, th.primary, rx=4, sw=2)
    s.rect(cx - 26, gy - 44, 52, 6, th.fg, rx=2)                # diaphragm
    s.circle(cx + 20, gy - 26, 4.5, th.secondary)               # vent
    s.arrow(120, 244, cx + 20, gy - 30, th.secondary, 1.4)
    s.text(38, 190, "static-pressure equalisation", 17, th.secondary,
           bold=True, anchor="start")
    s.text(38, 212, "vent: a first-order high-pass,", 17, th.secondary,
           anchor="start")
    s.text(38, 234, "and the chain's real corner", 17, th.secondary,
           anchor="start")

    # Primary and secondary windscreens over it.
    s.path(f"M {cx - 62} {gy - 12} A 62 62 0 0 1 {cx + 62} {gy - 12}",
           stroke=th.muted, sw=2.2)
    s.path(f"M {cx - 104} {gy - 12} A 104 104 0 0 1 {cx + 104} {gy - 12}",
           stroke=th.muted, sw=2.2, dash="8,5")
    s.text(cx, gy + 88, "primary foam screen (solid)", 16, th.muted)
    s.text(cx, gy + 108, "secondary in wind (dashed),", 16, th.muted)
    s.text(cx, gy + 128, "with its loss corrected", 16, th.muted)
    s.text(cx, 92, "Below 20 Hz the wind is louder", 19, th.fg, bold=True)
    s.text(cx, 114, "than the source", 19, th.fg, bold=True)

    # -- Middle: the electrical chain ---------------------------------------
    bx, bw, bh = 336.0, 226.0, 92.0
    stages = (
        ("Preamplifier", "corner << 0,25 Hz", th.primary),
        ("Recorder", "low-cut switch OFF", th.primary),
        ("G weighting + integrator", "$T ≥ 10$ s (A.5)", th.accent),
    )
    y = 132.0
    for title, detail, color in stages:
        s.rect(bx, y, bw, bh, th.panel, color, rx=12, sw=2)
        s.text(bx + bw / 2, y + 38, title, 20, th.fg, bold=True)
        # Mono is the code voice of the literal switch settings; the
        # averaging-time detail carries $...$ mathematics and composes in
        # the text face (mono plus markup is refused by the canvas).
        s.text(bx + bw / 2, y + 66, detail, 18, th.muted, mono="$" not in detail)
        if y > 140:
            s.arrow(bx + bw / 2, y - 34, bx + bw / 2, y - 6, th.fg, 2)
        y += bh + 34
    s.arrow(cx + 120, gy - 6, bx - 8, 200, th.fg, 2)
    s.text(bx + bw / 2, y + 4, "report $L_{pG}$ with the chain corner,", 17, th.fg)
    s.text(bx + bw / 2, y + 26, "the screens and the averaging time", 17, th.fg)

    # -- Right: the G curve against the chain's own corner ------------------
    px, py, pw, ph = 604.0, 132.0, 268.0, 236.0
    s.rect(px, py, pw, ph, "none", th.fg, rx=10, sw=1.6)
    f_lo, f_hi, db_lo, db_hi = 0.1, 1000.0, -60.0, 10.0

    def fx(f: float) -> float:
        return px + pw * (math.log10(f) - math.log10(f_lo)) / (
            math.log10(f_hi) - math.log10(f_lo))

    def fy(db: float) -> float:
        return py + ph * (db_hi - db) / (db_hi - db_lo)

    # The 0,25 Hz - 315 Hz span ISO 7196 Annex A asks the microphone to cover.
    s.add(f'<rect x="{fx(0.25):.1f}" y="{py + 2:.1f}" '
          f'width="{fx(315.0) - fx(0.25):.1f}" height="{ph - 4:.1f}" '
          f'fill="{th.accent}" opacity="0.14"/>')
    # The G curve itself.
    pts = []
    f = f_lo
    while f <= f_hi:
        pts.append(f"{fx(f):.1f},{fy(max(db_lo, _g_weight_db(f))):.1f}")
        f *= 1.06
    s.add(f'<polyline points="{" ".join(pts)}" fill="none" '
          f'stroke="{th.accent}" stroke-width="2.4"/>')
    # A chain whose vent corner sits at 2 Hz: first-order high-pass.
    corner = 2.0
    pts = []
    f = f_lo
    while f <= f_hi:
        db = 20 * math.log10(f / math.sqrt(f * f + corner * corner))
        pts.append(f"{fx(f):.1f},{fy(max(db_lo, db)):.1f}")
        f *= 1.06
    s.add(f'<polyline points="{" ".join(pts)}" fill="none" '
          f'stroke="{th.secondary}" stroke-width="2.2" stroke-dasharray="7,5"/>')
    s.line(fx(corner), py + 2, fx(corner), py + ph - 2, th.secondary, 1.2,
           dash="3,4")
    s.text(px + pw / 2, py - 14, "What the chain lets through", 18, th.fg,
           bold=True)
    for f_tick, label in ((0.1, "0,1"), (1.0, "1"), (10.0, "10"),
                          (100.0, "100"), (1000.0, "1k")):
        s.text(fx(f_tick), py + ph + 22, label, 15, th.muted, mono=True)
    s.text(px + pw / 2, py + ph + 46, "Frequency [Hz]", 16, th.muted)
    s.text(fx(0.27), fy(-55.0), "A.2: 0,25 - 315 Hz", 15, th.accent,
           bold=True, anchor="start")
    s.arrow(fx(0.45), fy(-33.0), fx(0.45), fy(-16.0), th.secondary, 1.5)
    s.text(fx(0.12), fy(-40.0), "lost", 15, th.secondary, bold=True,
           anchor="start")
    s.text(px + pw / 2, py + ph + 70, "green: the G weighting", 15, th.accent)
    s.text(px + pw / 2, py + ph + 90,
           "dashed: a chain with a 2 Hz vent corner", 15, th.secondary)
    s.text(px + pw / 2, py + ph + 116, "usable band = the overlap",
           17, th.fg, bold=True)
    s.text(px + pw / 2, py + ph + 138, "ISO 7196:1995, Annex A", 16, th.muted)


# ---------------------------------------------------------------------------
# d28 - Capturing an array: one clock, locked gains, a written row map
# ---------------------------------------------------------------------------

def _d_multichannel_capture(s: SVG, th: Theme) -> None:
    """Four microphones, one converter, and the map from row to position.

    Room-survey spacing with the capsules at 1,2 m, the calibrator drawn
    coupled onto one capsule and moved along the row, and the single-clock
    requirement stated where it is decided: at the interface.
    """
    gy = 372.0
    s.ground(gy, 30, 312)
    s.text(172, 92, "Each position, its sensitivity", 19, th.fg, bold=True)

    xs = (66.0, 132.0, 198.0, 264.0)
    sens = ("11,8", "12,3", "11,9", "12,1")
    cap_y = gy - 120.0                       # capsule height, 1,2 m to scale
    for i, (x, mv) in enumerate(zip(xs, sens)):
        s.mic(x, cap_y, gy, 0.95)
        s.text(x, gy + 24, f"P{i + 1}", 17, th.fg, bold=True)
        s.text(x, gy + 44, mv, 14, th.muted, mono=True)
    s.text(172, gy + 66, "mV/Pa, one per capsule", 14, th.muted)
    s.dim(40, cap_y, 40, gy, "1,2 m", size=15, label_side="right")

    # Calibrator coupled onto P2 and moved along the row.
    s.rect(xs[1] - 22, cap_y - 56, 44, 56, th.panel, th.secondary, rx=6, sw=2)
    s.text(xs[1], cap_y - 30, "94,0", 15, th.secondary, bold=True, mono=True)
    s.text(xs[1], cap_y - 12, "dB", 13, th.muted, mono=True)
    s.path(f"M {xs[0]:.0f} {cap_y - 78:.0f} Q {xs[1] + 34:.0f} "
           f"{cap_y - 116:.0f} {xs[3]:.0f} {cap_y - 78:.0f}",
           stroke=th.secondary, sw=1.6, dash="7,5")
    s.text(172, cap_y - 128, "one capsule at a time,", 15, th.secondary)
    s.text(172, cap_y - 110, "gains locked throughout", 15, th.secondary)

    # -- Middle: one preamplifier, one interface, one clock -----------------
    bx, bw = 352.0, 214.0
    s.rect(bx, 150, bw, 82, th.panel, th.primary, rx=10, sw=2)
    s.text(bx + bw / 2, 196, "4-channel preamplifier", 19, th.fg, bold=True)
    for i, x in enumerate(xs):
        s.arrow(x + 10, gy - 46, bx - 6, 168 + i * 16, th.muted, 1.2)

    s.rect(bx, 268, bw, 96, th.panel, th.primary, rx=10, sw=2)
    s.text(bx + bw / 2, 298, "audio interface", 19, th.fg, bold=True)
    s.rect(bx + 14, 312, bw - 28, 34, th.panel, th.accent, rx=6, sw=2)
    s.text(bx + bw / 2, 328, "single sample clock", 16, th.accent, bold=True)
    s.text(bx + bw / 2, 344, "$f_s$ = 48 kHz", 15, th.accent)
    s.arrow(bx + bw / 2, 236, bx + bw / 2, 262, th.fg, 2)

    s.rect(bx + 24, 400, bw - 48, 60, th.panel, th.muted, rx=10, sw=2,
           dash="6,5")
    s.text(bx + bw / 2, 428, "a second interface", 15, th.muted)
    s.text(bx + bw / 2, 448, "= two clocks, not one array", 15, th.muted)
    s.line(bx + 24, 400, bx + bw - 24, 460, th.secondary, 2.4)
    s.line(bx + 24, 460, bx + bw - 24, 400, th.secondary, 2.4)

    # -- Right: the array, row by row --------------------------------------
    ax0, row_h = 620.0, 44.0
    s.text(748, 122, "$x$, the array you analyse", 19, th.fg, bold=True)
    for i in range(4):
        y = 150 + i * row_h
        s.rect(ax0, y, 232, row_h - 8, th.panel, th.primary, rx=6, sw=1.8)
        s.text(ax0 + 116, y + 24, f"ch{i} = P{i + 1}", 18, th.fg, mono=True)
    s.text(748, 150 + 4 * row_h + 12, "shape (4, N)", 18, th.accent, bold=True,
           mono=True)
    s.arrow(bx + bw + 6, 300, ax0 - 8, 240, th.fg, 2)

    s.rect(600, 372, 272, 96, "none", th.fg, rx=10, sw=1.8)
    s.text(736, 398, "Write down, with the file:", 16, th.fg, bold=True)
    s.text(736, 420, "one clock  |  locked gains", 15, th.fg)
    s.text(736, 442, "the row-to-position map", 15, th.fg)
    s.text(450, 520, "A swapped pair gives perfectly valid levels attributed "
           "to the wrong positions,", 17, th.fg)
    s.text(450, 542, "and no later check can detect it", 17, th.muted)
