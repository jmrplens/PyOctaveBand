#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The experimental-setup diagrams of the documentation, drawn as SVG.

The package holds what one script used to hold: the canvas the diagrams
are drawn on (:mod:`diagrams.canvas`), the Spanish string table
(:mod:`diagrams.i18n`), the parts shared between diagrams
(:mod:`diagrams.parts`), one module per documentation domain holding the
builders of the guides in it, and the catalogue that names them all
(:mod:`diagrams.registry`). ``scripts/generate_diagrams.py`` is the
command that runs it.
"""

from __future__ import annotations

import os

# Deterministic output: pin numerical thread pools to a single thread before any
# numeric backend initializes (see generate_graphs.py for the rationale).
for _threads_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_threads_var, "1")

from .registry import DIAGRAMS, generate_all

__all__ = ["DIAGRAMS", "generate_all"]
