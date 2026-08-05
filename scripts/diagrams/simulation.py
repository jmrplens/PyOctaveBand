#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Diagrams of the simulation guides: the FDTD wave solver.

The subject is the numerical scheme itself rather than any measurement: the
staggered grid of the finite-difference time-domain solver, and the leapfrog
that alternates pressure and velocity updates on it.
"""

from __future__ import annotations

from .canvas import SVG, Theme


def _d_fdtd(s: SVG, th: Theme) -> None:
    """2D acoustic FDTD pipeline (Attenborough & Van Renterghem 2021, Ch. 4)."""
    cx = 450.0
    bw, bh = 660.0, 58.0
    x0 = cx - bw / 2

    # --- Inputs (two feeder boxes) -----------------------------------------
    iw = 320.0
    s.rect(x0, 48, iw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(x0 + iw / 2, 72, "Domain  c(x, y), ρ(x, y), dx", 17, th.fg,
           "middle", bold=True)
    s.text(x0 + iw / 2, 92, "square cells; dt from the Courant number",
           13, th.muted, "middle")
    s.rect(x0 + bw - iw, 48, iw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(x0 + bw - iw / 2, 72, "Geometry and boundaries", 17, th.fg,
           "middle", bold=True)
    s.text(x0 + bw - iw / 2, 92,
           "rigid, impedance or absorbing edges; obstacles", 13,
           th.muted, "middle")
    s.arrow(x0 + iw / 2, 106, cx - 60, 150, th.fg, 1.8)
    s.arrow(x0 + bw - iw / 2, 106, cx + 60, 150, th.fg, 1.8)

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 25, l1, 17, th.fg, "middle", bold=True)
        s.text(cx, y + 45, l2, 13, th.muted, "middle")

    _step(150, "Sources  s(t) injected at cells  (Eq. 4.11-4.12 grid)",
          "Gaussian pulse, ramped tone or arbitrary sampled signal", th.fg)
    _step(238, "Staggered-grid leapfrog update  (Eqs. 4.11-4.12)",
          "v ← v − (dt/ρ·dx)·grad p,  then  p ← p − (ρc²·dt/dx)·div v",
          th.primary)
    _step(326, "stable while  CN = c·dt·√2/dx ≤ 1  (Eqs. 4.13-4.14)",
          "resolve ≥ 10 cells per wavelength to keep dispersion low",
          th.secondary)
    for y0, y1 in ((208, 238), (296, 326), (384, 414)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)

    # --- Output -------------------------------------------------------------
    s.rect(x0, 414, bw, bh, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, 439, "FDTDResult:  probe histories p(t), field snapshots, .plot()",
           17, th.fg, "middle", bold=True)
    s.text(cx, 459, "deterministic: same inputs, bit-identical outputs", 13,
           th.muted, "middle")
