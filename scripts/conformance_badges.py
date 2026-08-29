#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Entry point for the conformance report's committed visual indicators.

Run by ``make conformance`` after the artefact is written, so the numbers on
the banner are always the tree's own. The work is split across two modules and
the split is load-bearing: :mod:`conformance.marks` is the vocabulary - what
each file is called, the URL it is served from and the words it says - on the
standard library alone, because the pull-request comment job imports it and
installs nothing; :mod:`conformance.badges` is the drawing, which needs
matplotlib to turn a committed font into outlines. This file is the entry
point and the facade over both, in the same relationship
``conformance_report.py`` has with :mod:`conformance.render`.

Usage::

    python scripts/conformance_badges.py
    python scripts/conformance_badges.py --out /tmp/badges   # for a diff
"""

from __future__ import annotations

from conformance.badges import (
    BADGE_DIR,
    DARK,
    LIGHT,
    Palette,
    asset_names,
    assets,
    banner_name,
    main,
    render_banner,
    render_marks,
    write,
)
from conformance.marks import (
    BANNER,
    BANNER_DARK,
    MARK_OF,
    MARKS,
    RAW_BASE,
    RAW_PATH,
    Mark,
    asset_url,
    banner_alt,
    banner_picture,
    dark_variant,
    mark_definitions,
    mark_html,
    mark_image,
    mark_reference,
    outcome_mark,
)

__all__ = [
    "BADGE_DIR",
    "BANNER",
    "BANNER_DARK",
    "DARK",
    "LIGHT",
    "MARKS",
    "MARK_OF",
    "RAW_BASE",
    "RAW_PATH",
    "Mark",
    "Palette",
    "asset_names",
    "asset_url",
    "assets",
    "banner_alt",
    "banner_name",
    "banner_picture",
    "dark_variant",
    "main",
    "mark_definitions",
    "mark_html",
    "mark_image",
    "mark_reference",
    "outcome_mark",
    "render_banner",
    "render_marks",
    "write",
]


if __name__ == "__main__":
    raise SystemExit(main())
