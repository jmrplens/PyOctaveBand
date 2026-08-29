#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Entry point for the conformance report's committed visual indicators.

Run by ``make conformance`` after the artefact is written, so the numbers on
the banner are always the tree's own. The drawing, the palette and the reasons
behind both live in :mod:`conformance.badges`; this file is the entry point and
the facade over it, in the same relationship ``conformance_report.py`` has with
:mod:`conformance.render`.

Usage::

    python scripts/conformance_badges.py
    python scripts/conformance_badges.py --out /tmp/badges   # for a diff
"""

from __future__ import annotations

from conformance.badges import (
    BADGE_DIR,
    BANNER,
    BANNER_DARK,
    DARK,
    LIGHT,
    MARKS,
    RAW_BASE,
    RAW_PATH,
    Mark,
    Palette,
    asset_names,
    assets,
    banner_alt,
    banner_name,
    main,
    render_banner,
    render_marks,
    write,
)

__all__ = [
    "BADGE_DIR",
    "BANNER",
    "BANNER_DARK",
    "DARK",
    "LIGHT",
    "MARKS",
    "RAW_BASE",
    "RAW_PATH",
    "Mark",
    "Palette",
    "asset_names",
    "assets",
    "banner_alt",
    "banner_name",
    "main",
    "render_banner",
    "render_marks",
    "write",
]


if __name__ == "__main__":
    raise SystemExit(main())
