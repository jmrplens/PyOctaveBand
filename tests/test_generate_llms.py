#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The index of ``llms.txt`` must not file a guide under the wrong shard.

``llms.txt`` groups its guides under a ``##`` heading per area and, where an
area is split, a ``###`` heading per carved-out shard. Each heading names the
full-text shard the run under it belongs to, so a client that wants one part
fetches one file. That promise is only kept if the run under a heading is
exactly the shard's own membership.

It was not. The index used to emit a heading the first time it met a route
from a different shard and never switch back, so any guide of the parent shard
that came after the carve-out in the folder's reading order was listed under
the carved-out heading. In ``devices/emission`` that put the surface-velocity
route, which measures no sound at all, under a heading about sound intensity,
and pointed a reader at a shard that does not contain it. Nothing caught it:
the artefact regenerates deterministically, so the freshness job compared a
wrong file against itself and passed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_llms as llms  # noqa: E402


def _runs_of(text: str) -> dict[str, list[str]]:
    """The routes listed under each ``###`` heading, keyed by its shard slug.

    The shard is read from the ``full text`` link the heading carries rather
    than from the heading's words, because the link is what a client follows.
    """
    runs: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = None
        elif line.startswith("### "):
            current = ""
        elif current == "" and "/llms/llms-" in line:
            current = line.split("/llms/llms-", 1)[1].split(".txt", 1)[0]
            runs.setdefault(current, [])
        elif current and (m := re.search(r"\]\(\S+?/phonometry/(\S+?)/\)", line)):
            runs[current].append(m.group(1))
    return runs


def test_every_carved_out_heading_lists_only_its_own_shard() -> None:
    members = llms._shard_members()
    runs = _runs_of((ROOT / "llms.txt").read_text(encoding="utf-8"))
    assert runs, "the index carries no carved-out shard headings to check"
    for slug, listed in runs.items():
        expected = members.get(slug)
        assert expected is not None, (
            f"the index names a shard that does not exist: {slug}"
        )
        assert listed == expected, (
            f"the guides listed under the {slug} heading are not that shard's "
            f"own: listed {listed}, shard holds {expected}"
        )


@pytest.mark.parametrize("slug", ["devices-emission-intensity"])
def test_the_intensity_shard_holds_only_the_intensity_routes(slug: str) -> None:
    """The split of ``devices/emission`` is by the quantity measured.

    Named rather than left to the general check above, because this is the
    one the defect was found in: the surface-velocity route belongs with the
    pressure routes of the parent shard, not with the two that read intensity.
    """
    members = llms._shard_members()
    assert members[slug] == [
        "devices/emission/sound-power-intensity",
        "devices/emission/intensity",
    ]
    assert "devices/emission/vibration-sound-power" in members["devices-emission"]
