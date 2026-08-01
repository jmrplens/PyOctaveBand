#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tolerance-aware staleness check for the committed documentation figures.

The ``Documentation figures up to date`` CI job regenerates ``.github/images``
with ``make graphs`` and must confirm the result still matches what is
committed. A byte-exact ``git diff`` cannot do this reliably: GitHub's runner
fleet is hardware-heterogeneous, so the same pinned software stack computes a
handful of plotted path coordinates ~1 ULP apart depending on which CPU
microarchitecture the run lands on (a different SIMD kernel in the numpy/BLAS
stack). That sub-pixel drift is numerically and visually irrelevant, yet a
byte diff flags the figure as stale, which made the job intermittently fail.

This script compares the freshly regenerated working-tree figures against the
committed versions (``git HEAD``) within a tolerance instead:

* **SVG** -- the non-numeric structure (elements, text, colours, ordering)
  must be identical, and every numeric token must agree within an absolute or
  relative tolerance. A moved element, changed label or new path fails; a
  last-bit coordinate wobble passes.
* **Raster (WebP/PNG)** -- identical dimensions, at most
  :data:`RASTER_TOL`\\ ``.max_sig_pixels`` meaningfully changed pixels and a
  bounded root-mean-square difference (see
  :class:`~generated_assets.RasterTolerance`).
* **Anything else** -- exact byte compare.

Added or removed files always fail: a new figure must be committed, and a
figure that is no longer generated must be removed from the tree.

The check is intentionally strict about *structure* and lenient only about the
*last digits of numbers*, so it still catches every real figure change while
being immune to cross-CPU floating-point non-determinism.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generated_assets import (
    RasterTolerance,
    committed_bytes,
    raster_problem,
    tracked_files,
)

IMG_DIR = ".github/images"

# Numeric tokens differing by less than this (absolute, in SVG user units, or
# relative for large magnitudes) are treated as equal. A 1-ULP coordinate
# wobble is ~1e-6; a real edit moves a coordinate by whole units.
SVG_ABS_TOL = 1e-2
SVG_REL_TOL = 1e-4

# A pixel counts as "meaningfully changed" if any channel differs by more than
# ``level`` (0..255). Cross-CPU drift perturbs a coordinate by ~1e-6 units,
# i.e. a ~1e-5-pixel geometric shift, whose anti-aliasing effect rounds to at
# most a level or two on a few edge pixels -- far below this threshold. A real
# edit moves a plotted line or glyph, changing hundreds to thousands of edge
# pixels, and a restyled fill moves the whole image past the RMS bound.
RASTER_TOL = RasterTolerance(level=12, max_sig_pixels=100, rms=2.0)

# Integers and decimals, with optional sign and exponent. ``split``/``findall``
# with this pattern partition a file into fixed text and numeric values.
_TOKEN = re.compile(rb"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

# matplotlib derives clip-path / marker / collection ids by hashing the literal
# ``str()`` of the underlying float geometry (see backend_svg._make_id). A
# cross-CPU 1-ULP wobble in that geometry avalanches the whole SHA256, so the
# id *text* would change even though the figure is numerically identical -- the
# exact drift this check exists to tolerate. Canonicalising every id and its
# ``url(#...)`` / ``href="#..."`` references to placeholders numbered by first
# appearance removes the hash text from the structural comparison while still
# catching a genuine change (an id added, removed, reordered, or a reference
# repointed changes the placeholder sequence).
_IDREF = re.compile(rb'(\bid="|url\(#|xlink:href="#|\bhref="#)([A-Za-z_][\w.:\-]*)')


def _canonicalize_ids(data: bytes) -> bytes:
    """Rewrite hash-derived ids/references to first-appearance placeholders."""
    mapping: dict[bytes, bytes] = {}

    def repl(match: re.Match[bytes]) -> bytes:
        token = match.group(2)
        placeholder = mapping.setdefault(token, b"ID%d" % len(mapping))
        return match.group(1) + placeholder

    return _IDREF.sub(repl, data)


def _svg_within_tolerance(old: bytes, new: bytes) -> bool:
    """True if two SVGs match structurally and numerically within tolerance."""
    old = _canonicalize_ids(old)
    new = _canonicalize_ids(new)
    if _TOKEN.split(old) != _TOKEN.split(new):
        return False  # non-numeric structure (elements/text/colours) differs
    old_nums = _TOKEN.findall(old)
    new_nums = _TOKEN.findall(new)
    if len(old_nums) != len(new_nums):
        return False
    for o_tok, n_tok in zip(old_nums, new_nums):
        o_val = float(o_tok)
        n_val = float(n_tok)
        limit = max(SVG_ABS_TOL, SVG_REL_TOL * max(abs(o_val), abs(n_val)))
        if abs(o_val - n_val) > limit:
            return False
    return True


def main() -> int:
    disk = {str(p) for p in Path(IMG_DIR).rglob("*") if p.is_file()}
    tracked = tracked_files(IMG_DIR)
    problems: list[str] = []

    for path in sorted(disk - tracked):
        problems.append(f"new figure not committed: {path}")
    for path in sorted(tracked - disk):
        problems.append(f"committed figure no longer generated: {path}")

    for path in sorted(disk & tracked):
        new = Path(path).read_bytes()
        old = committed_bytes(path)
        if old is None:
            problems.append(f"cannot read committed {path}")
            continue
        if old == new:
            continue  # byte-identical: fast path, no tolerance needed
        if path.endswith(".svg"):
            if not _svg_within_tolerance(old, new):
                problems.append(f"SVG changed beyond tolerance: {path}")
        elif path.endswith((".webp", ".png")):
            reason = raster_problem(old, new, RASTER_TOL)
            if reason is not None:
                problems.append(f"raster changed beyond tolerance ({reason}): {path}")
        else:
            problems.append(f"asset changed: {path}")

    if problems:
        print(
            "::error::.github/images is out of date - "
            "run 'make graphs' and commit the result."
        )
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"All {len(disk & tracked)} committed figures match within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
