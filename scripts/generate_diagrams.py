#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Deterministic SVG generator for the experimental-setup diagrams used in the
documentation. Every diagram is emitted in a light and a dark variant
(``*_dark.svg``) with the same palette as the matplotlib figures, so the
docs can theme-switch them exactly like the raster WebP plots.

The diagrams themselves live in the ``diagrams`` package beside this file,
one module per documentation domain; this is the command that runs them.

Run directly or via ``make graphs``.
"""

from __future__ import annotations

import os
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from diagrams import DIAGRAMS, generate_all
from diagrams.canvas import DARK, LIGHT, SVG, Theme

# The drawing vocabulary is re-exported because it was this module's own: a
# reader who reaches for `generate_diagrams.SVG` is reaching for the class
# this file used to define, not for an incidental import.
__all__ = ["DARK", "DIAGRAMS", "LIGHT", "SVG", "Theme", "generate_all"]

if __name__ == "__main__":
    generate_all()
