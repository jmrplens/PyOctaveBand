#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Shared pieces of the tolerance-aware staleness checks.

Two committed directories are regenerated from the library and must not drift
from it: ``.github/images`` (:mod:`scripts.check_figures`) and
``.github/reports`` (:mod:`scripts.check_reports`). Both answer the same
question the same way -- regenerate into the working tree, then compare
against ``git HEAD`` -- and neither can do it with a byte diff, because
GitHub's runner fleet is hardware-heterogeneous and the same pinned stack
computes a few plotted coordinates ~1 ULP apart depending on which CPU
microarchitecture the run lands on.

What they share lives here: reading the committed bytes, listing what is
tracked, and comparing two rasters within a tolerance. What differs stays in
each checker, because the tolerances are not the same: a documentation figure
is a raster of a plot, a fiche preview is a rasterized document page whose
rendering runs through a pinned binary rasterizer and is markedly quieter.
"""

from __future__ import annotations

import io
import subprocess
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RasterTolerance:
    """How far two rasters of the same asset may drift and still match.

    The two criteria are complementary, and both must hold. ``max_sig_pixels``
    catches a *localised* change (a moved line, a relabelled axis) that a
    global root-mean-square would dilute across a large image; ``rms`` catches
    a *broad* change (a recoloured fill, a restyled background) that
    few-but-everywhere pixels would slip past the count. Cross-CPU sub-ULP
    coordinate drift moves neither meaningfully.

    :param level: Per-channel difference (0..255) above which a pixel counts
        as meaningfully changed rather than anti-aliasing noise.
    :param max_sig_pixels: How many such pixels are tolerated.
    :param rms: Largest tolerated per-channel root-mean-square difference.
    """

    level: float
    max_sig_pixels: int
    rms: float


def committed_bytes(path: str) -> bytes | None:
    """Return the bytes of ``path`` as committed at HEAD, or ``None``."""
    result = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def tracked_files(directory: str) -> set[str]:
    """Return the set of paths under ``directory`` tracked at HEAD."""
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", directory],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line for line in result.stdout.splitlines() if line}


def raster_problem(old: bytes, new: bytes, tol: RasterTolerance) -> str | None:
    """Describe how two rasters differ beyond ``tol``, or ``None`` if within it."""
    a = np.asarray(Image.open(io.BytesIO(old)).convert("RGBA"), dtype=np.float64)
    b = np.asarray(Image.open(io.BytesIO(new)).convert("RGBA"), dtype=np.float64)
    if a.shape != b.shape:
        return f"dimensions changed {a.shape} != {b.shape}"
    diff = np.abs(a - b)
    sig_pixels = int(np.count_nonzero(diff.max(axis=-1) > tol.level))
    if sig_pixels > tol.max_sig_pixels:
        return f"{sig_pixels} pixels changed by >{tol.level:g} (> {tol.max_sig_pixels})"
    rms = float(np.sqrt(np.mean(diff**2)))
    if rms > tol.rms:
        return f"RMS {rms:.3f} > {tol.rms}"
    return None
