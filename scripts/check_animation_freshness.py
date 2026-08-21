#!/usr/bin/env python3
#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Fail when a committed clip is older than the code that draws it.

The figures are regenerated in CI and compared; the clips cannot be, which is
why ``make animations`` is a target of its own. Rendering the forty-two clips
is an FDTD simulation plus four video encodes each, and the encoders are not
bit-reproducible across machines even when they are. So a clip is committed
once and then keeps whatever it was drawn with, while the code that draws it
moves on underneath -- and nothing said so. The Spanish minus-sign repair
reached only the clips that happened to be re-rendered after it, and twelve
clips kept publishing an ASCII hyphen in their Spanish tick labels for months
with every gate green.

This is the missing half of ``scripts/check_figures.py``, and it answers the
same question without rendering anything: every clip carries a fingerprint of
the code that drew it (:mod:`animation_fingerprint`), written by the renderer
as the clip is written and committed next to it. Here the fingerprints are
recomputed from the current sources and compared. A clip whose fingerprint
moved is stale: the code that draws it has changed since it was last
rendered, so re-render it and commit the result::

    set -a; . .env; set +a          # the AV1 encoder settings
    python scripts/generate_graphs.py --animations --anim <clip>

A render writes more than the four WebM variants, and the extra files are the
ones a half-finished render loses quietly: the poster still each embed shows
before the video loads, and the two English GIFs the GitHub pages fall back
to. They come off the same render as the WebM, so the fingerprint already
speaks for how old they are; what is checked here is that they are there at
all (:func:`outputs`).

The check is cheap (it parses the figure package, it does not import or run
it) and needs no rendering stack, so it rides in any job.
"""

from __future__ import annotations

import pathlib
import sys

_SCRIPTS = pathlib.Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import animation_fingerprint as fp

#: Where the clips are committed, used only to say which ones exist on disk.
IMAGES = _SCRIPTS.parent / ".github" / "images"

#: The language x theme variants every clip is rendered in.
VARIANTS = ("", "_dark", "_es", "_es_dark")


def outputs(clip: str) -> list[str]:
    """Every file a four-variant render of *clip* leaves in :data:`IMAGES`.

    The WebM of each variant, the poster still next to it (the site defers
    the video behind it, so a missing poster is a blank box until the reader
    presses play) and the two English GIFs the GitHub documentation embeds.
    ``figures/media.py`` writes all three for every clip: the poster always,
    the GIF for English only, in both themes.
    """
    return (
        [f"{clip}{suffix}.webm" for suffix in VARIANTS]
        + [f"{clip}{suffix}_poster.jpg" for suffix in VARIANTS]
        + [f"{clip}{suffix}.gif" for suffix in ("", "_dark")]
    )


def main() -> int:
    current = fp.fingerprints(_SCRIPTS.parent)
    stamped = fp.read_manifest()
    problems: list[str] = []
    problems.extend(
        f"{clip}: stamped in {fp.MANIFEST.name} but no longer a registered "
        "clip; delete the line if the clip is gone"
        for clip in sorted(set(stamped) - set(current))
    )

    for clip in sorted(current):
        expected = outputs(clip)
        missing = [name for name in expected if not (IMAGES / name).exists()]
        if len(missing) == len(expected):
            problems.append(
                f"{clip}: registered but not committed at all (none of its "
                f"{len(expected)} files are there); render it"
            )
        elif missing:
            problems.append(
                f"{clip}: committed half-rendered, {len(missing)} of its "
                f"{len(expected)} files are missing "
                f"({', '.join(missing)})"
            )
        if clip not in stamped:
            problems.append(
                f"{clip}: no fingerprint recorded; re-render the clip to stamp it"
            )
        elif stamped[clip] != current[clip]:
            problems.append(
                f"{clip}: drawn by code that has changed since "
                f"(recorded {stamped[clip]}, current {current[clip]})"
            )

    if problems:
        print(
            "::error::a committed clip is incomplete or older than the code "
            "that draws it - re-render it with "
            "'python scripts/generate_graphs.py --animations --anim <clip>'"
        )
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(
        f"All {len(current)} committed clips are complete and match the "
        "code that draws them."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
