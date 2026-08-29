#!/usr/bin/env python3
r"""Fail on an annotation the reader cannot read off the page.

Two defects, and they are not the same defect.

**A curve drawn behind the letters.** The strokes of the letters and the
stroke of the curve are the same weight, and on the dark page close to the
same lightness. The answer the corpus uses, at a hundred and forty-seven
call sites under ``scripts/figures``, is an opaque chip::

    bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
          "edgecolor": COLOR_GRID}

which is *not* the translucent one a text box in an empty corner takes
(``{"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6}``): over a line
that one loses its contrast on the dark page. ``pad`` follows the neighbours --
0.5 at seventy-two of them, 0.4 at fifty-three, 0.3 for a small label pinned
near a point -- and the chip is given a ``zorder`` above the curves.

A chip is not free, and where it can be avoided it should be. It is opaque, so
whatever it lands on is gone: a curve or a filled band passing behind it reads
as passing behind it, but a plotted point, a marker or the end of a range bar
under it is a datum the reader no longer has. Move the label first, and reach
for the chip when there is nowhere for it to go.

**Something drawn over the letters.** Here a chip is no answer at all, because
the chip is being painted over with them: the label needs a ``zorder`` above
whatever covers it. A chip *and* a zorder is the whole convention, and a chip
with no zorder is the case the first rule passes and the reader fails.

Where the number comes from
---------------------------

:mod:`figure_annotation_audit` measures both during the generation run, by
rendering the figure with pieces of it taken away and counting the pixels that
move. This script reads that recording and applies the rule; the reason it is
pixels rather than geometry, and the reason the measurement has to happen
while the artists are alive, are both written down there.

The two measures answer different questions and were calibrated separately.
Each threshold is where the classes actually part, on labels classified by eye
at 3x and 8x dpi in both themes, and neither number is borrowed from the other.

**Behind**, against 94 labels (57 struck, 37 clean), and the boundary is one
pixel wide:

* at **45 px** and above, every one of the 50 labels that fire is a real
  defect -- precision 1.00, recall 0.88. The largest label judged clean scores
  44, the smallest judged struck above it scores 45, and the reported example
  (``g_weighting_response``, "0 dB @ 10 Hz") scores 49, which is what bounds the
  threshold from above.
* between **32 and 44 px** the two classes interleave completely, so those
  print as an advisory and do not fail. They are real contacts, but which of
  them a reader minds is a coin toss, and a gate that tells someone to change
  something that looks fine is a gate nobody believes.

Zero false alarms in 37 clean examples bounds the false-alarm rate near 9 % with
95 % confidence, not at zero. The costs are lopsided, which is why it gates
anyway: a false positive adds a chip that was not strictly needed -- invisible,
and nobody argues about it -- while a false negative ships a label a reader
cannot read.

A one-pixel boundary invites the question of whether the count drifts, since
the corpus is known not to be bit-identical across machines and
``check_figures.py`` compares it by tolerance for that reason. Measured, it
does not: displacing every plotted vertex of four figures by one ULP up, one
down, and randomly per coordinate moved none of thirteen counts, the two
sitting at 44 px included. Floating-point drift changes where a curve is by a
fraction of a pixel; a struck-pixel count changes when it moves by one.

**Covered** gates lower, at **20 px**, and has no advisory band, because it
separates where the other one interleaves. Exactly 40 labels in the corpus lose
any letter ink at all; the 19 at or above 21 px are all real, the largest clean
one scores 18, and nothing lands in between. The two ends of that boundary,
looked at twice: ``two_tone_separation_es`` at 21 px prints a black note
straight over the red "Hz" of "212 Hz" and destroys it, while
``detailed_impact_paths`` at 18 px is a grey dotted rule clipping the "t" of
"paths", which reads. Twenty is the round number inside the gap, with two
pixels of margin over the largest clean example -- wider than behind's one.

It gates lower because it is a different measurement, not a stricter mood:
covered pixels are ink the reader never receives, so they destroy a character
outright, while a struck pixel is a character still fully drawn and merely
competing with a curve. It also counts only letters, which the audit explains:
counting the chip as well puts a clean label at 384 px and no threshold
survives that.

:data:`REQUIRED_DPI` is part of the rule, not a detail: the count scales with
the square of the resolution, so a recording taken at any other dpi is not
comparable and is refused rather than rescaled.

Running it
----------

The measurement comes from a generation run, so the check needs one::

    make graphs             # records into build/figure-annotations
    make figure-annotations # reads it

``make graphs`` empties the directory first. Reading a directory some other run
filled would be answering about the wrong tree, so the check requires the
recording to cover every committed figure *in both languages* -- Spanish prose
is longer than English and grows into curves the English strings clear, so the
two drawings are measured separately and recorded under separate keys
(``foo`` and ``foo_es``). ``--partial`` is for a targeted re-render of one
figure: it checks what that run produced and reports no exemption as stale,
because a run that did not draw a figure cannot say anything about it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

_SCRIPTS = pathlib.Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import figure_annotation_audit as audit

#: Struck pixels at :data:`REQUIRED_DPI` that fail the build, per defect. The
#: two measures count different things -- a curve painted under a character
#: that is nonetheless fully drawn, against ink the reader never receives at
#: all -- so each carries the threshold its own classes part at.
GATE_PX = {audit.BEHIND: 45, audit.COVERED: 20}

#: Where the advisory band under a gate begins. Only the behind measure has
#: one: between 32 and 44 px an unreadable label and a clean one interleave
#: completely, so those are reported for a person to judge and do not fail.
#: The covered measure has no such band -- its classes separate, 18 px clean
#: against 21 px unreadable -- so under its gate there is nothing to say.
ADVISORY_PX = {audit.BEHIND: 32}

#: The resolution the thresholds are pinned to (``figure.dpi`` in the figure
#: theme). The count scales with its square, so a recording taken anywhere else
#: means a different rule.
REQUIRED_DPI = audit.REFERENCE_DPI

#: Figures allowed to keep an annotation the rules fire on, each with the
#: reason. Checked in both directions: an entry whose label no longer fires is
#: deleted, so the file cannot silently cover a figure that changed.
EXEMPTIONS = _SCRIPTS / "figure_annotation_exemptions.txt"

#: The committed figures, which say what a full generation run produces.
IMAGES = _SCRIPTS.parent / ".github" / "images"

#: What to do about each kind, printed under the labels that fire it. The two
#: defects need opposite fixes, which is why they are counted apart: a chip
#: answers a curve behind the letters and does nothing at all for a label
#: something is painted over.
ADVICE = {
    audit.BEHIND: (
        'give each one bbox={"boxstyle": "round,pad=0.5", "facecolor": '
        'COLOR_PANEL, "edgecolor": COLOR_GRID} with a zorder above the curves '
        "(pad as its neighbours use)"
    ),
    audit.COVERED: (
        "raise each one above what is drawn over it, with a zorder higher than "
        "that artist's (a chip does not help here: it is being painted over "
        "too)"
    ),
}

#: How each kind reads in the failure report.
HEADLINE = {
    audit.BEHIND: "struck by a curve and carry no chip",
    audit.COVERED: "painted over by something drawn after them",
}


def committed_figures() -> set[str]:
    """The drawings a full run measures, read off the committed images.

    Read off the images rather than off the registry, because a generator's
    name is not its output name: ``generate_filter_responses`` writes ten
    files, none of them called ``filter_responses``, and the recording is keyed
    by what is written. Both languages count and only the light pass does: a
    figure ships as four files and the two dark ones are the same drawing in
    other colours, so ``foo`` and ``foo_es`` are the two keys a full run owes.
    The hand-drawn plates are then subtracted -- they are written by
    ``scripts/generate_diagrams.py`` from an SVG canvas, never pass through the
    figure saver, and have no matplotlib artist to measure -- and so are the
    clips, which come from the separate ``make animations``.
    """
    from diagrams.registry import DIAGRAMS

    found = set()
    for path in IMAGES.iterdir():
        if path.suffix not in (".svg", ".webp") or path.stem.endswith("_dark"):
            continue
        stem = path.stem.removesuffix("_es")
        if stem.startswith("anim_") or stem in DIAGRAMS:
            continue
        found.add(path.stem)
    return found


def read_exemptions(path: pathlib.Path) -> dict[tuple[str, str], str]:
    """Parse the committed exemptions into ``(key, label) -> reason``."""
    entries: dict[tuple[str, str], str] = {}
    if not path.exists():
        return entries
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        entries.update([_parse_exemption(path, number, line)])
    return entries


def _parse_exemption(
    path: pathlib.Path, number: int, line: str
) -> tuple[tuple[str, str], str]:
    """One ``key: "label": reason`` line, or a :class:`SystemExit` saying so."""
    stem, _, rest = line.partition(": ")
    label, reason = "", ""
    if rest.startswith('"'):
        label, end = json.JSONDecoder().raw_decode(rest)
        reason = rest[end:].removeprefix(":").strip()
    if not stem or not label or not reason:
        msg = f"{path}:{number}: expected 'stem: \"label\": reason'"
        raise SystemExit(msg)
    return (stem, label), reason


def kind_of(hit: dict[str, Any]) -> str:
    """Which defect *hit* is. An old recording predates the distinction."""
    return str(hit.get("kind", audit.BEHIND))


def struck(
    recorded: dict[str, list[dict[str, Any]]], floors: dict[str, int]
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Every recorded label at or above its own kind's floor, worst first.

    A kind *floors* says nothing about is silent, which is what gives the
    covered measure an empty advisory band without a special case.
    """
    ranked = sorted(
        ((key, hit) for key, hits in recorded.items() for hit in hits),
        key=lambda found: -found[1]["pixels"],
    )
    for key, hit in ranked:
        floor = floors.get(kind_of(hit))
        if floor is not None and hit["pixels"] >= floor:
            yield key, hit


def describe(key: str, hit: dict[str, Any]) -> str:
    """One report line: the drawing, the label, the count and what does it."""
    text = json.dumps(hit["text"], ensure_ascii=False)
    verb = "struck by" if kind_of(hit) == audit.BEHIND else "under"
    by = ", ".join(f"{name} ({pixels} px)" for name, pixels in hit["struck_by"])
    return f"  {key}: {text} - {hit['pixels']} px" + (f", {verb} {by}" if by else "")


def wrong_dpi(recorded: dict[str, list[dict[str, Any]]]) -> set[float]:
    """The resolutions in *recorded* that the thresholds do not describe.

    Read off the measurements, so a figure that recorded none says nothing
    about its own resolution. That is the right silence: a figure with no ink
    on any letter has no count to compare with a threshold, and a contact real
    enough to matter leaves pixels at any resolution, which then carries the
    dpi that produced them.
    """
    return {
        hit["dpi"]
        for hits in recorded.values()
        for hit in hits
        if hit["dpi"] != REQUIRED_DPI
    }


def report(
    recorded: dict[str, list[dict[str, Any]]],
    exemptions: dict[tuple[str, str], str],
    *,
    partial: bool,
) -> int:
    """Fail on an un-exempt unreadable label, and on an exemption that stopped being true."""
    failures = [
        (key, hit)
        for key, hit in struck(recorded, GATE_PX)
        if (key, hit["text"]) not in exemptions
    ]
    problems = _failure_report(failures) if failures else []
    if not partial:
        problems.extend(_stale_report(recorded, exemptions))
    if problems:
        print("::error::an annotation on a figure cannot be read")
        print("\n".join(problems))
        return 1
    _print_advisory(recorded)
    print(
        f"No unreadable annotation at or above {_gates()} "
        f"({len(recorded)} drawing(s) measured, {len(exemptions)} exempt)."
    )
    return 0


def _gates() -> str:
    """The thresholds as the summary line says them, one clause per defect."""
    return " or ".join(
        f"{pixels} px {kind}" for kind, pixels in sorted(GATE_PX.items())
    )


def _failure_report(failures: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """What to print for the labels that fail the gate, one block per kind."""
    lines: list[str] = []
    for kind in (audit.BEHIND, audit.COVERED):
        of_kind = [(key, hit) for key, hit in failures if kind_of(hit) == kind]
        if not of_kind:
            continue
        figures = len({key for key, _ in of_kind})
        lines += [
            f"{len(of_kind)} annotation(s) across {figures} drawing(s) are "
            f"{HEADLINE[kind]}:",
            *(describe(key, hit) for key, hit in of_kind),
            f"  -> {ADVICE[kind]}, or, if it must stay as it is, add it to "
            f"{EXEMPTIONS.name} with the reason.",
        ]
    return lines


def _stale_report(
    recorded: dict[str, list[dict[str, Any]]],
    exemptions: dict[tuple[str, str], str],
) -> list[str]:
    """What to print for exemptions that no longer describe anything.

    An exemption is a decision about a label that *is* unreadable. Once the
    label is fixed, renamed or gone, the line covers nothing and has to go, or
    the file rots into a list of figures nobody is allowed to regress on.

    Only reached on a full run, which by then is known to cover every
    committed figure in both languages -- so a key missing from the recording
    is a drawing that is no longer made, and its exemption is as stale as a
    fixed one.
    """
    firing = {(key, hit["text"]) for key, hit in struck(recorded, GATE_PX)}
    stale = [(key, label) for key, label in exemptions if (key, label) not in firing]
    if not stale:
        return []
    return [
        f"{len(stale)} exemption(s) are no longer true (fixed, moved or gone):",
        *(f"  {key}: {json.dumps(label, ensure_ascii=False)}" for key, label in stale),
        f"  -> delete them from {EXEMPTIONS.name}.",
    ]


def _print_advisory(recorded: dict[str, list[dict[str, Any]]]) -> None:
    """List the contacts below the gate, which are for a person to judge."""
    band = [
        (key, hit)
        for key, hit in struck(recorded, ADVISORY_PX)
        if hit["pixels"] < GATE_PX[kind_of(hit)]
    ]
    if not band:
        return
    spans = ", ".join(
        f"{floor} and {GATE_PX[kind] - 1} px {kind}"
        for kind, floor in sorted(ADVISORY_PX.items())
    )
    print(
        f"{len(band)} annotation(s) score between {spans}. Advisory: in that "
        "band an unreadable label and a clean one are indistinguishable by "
        "count, so this does not fail."
    )
    for key, hit in band:
        print(describe(key, hit))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--audit",
        default=audit.DEFAULT_DIR,
        metavar="DIR",
        help=f"where the generation run recorded (default: {audit.DEFAULT_DIR})",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="the run generated only some of the figures: check what it drew, "
        "and neither require full coverage nor report an exemption as stale",
    )
    args = parser.parse_args(argv)

    directory = pathlib.Path(args.audit)
    recorded = audit.load(str(directory)) if directory.is_dir() else {}
    if not recorded:
        print(
            f"::error::no figure-annotation recording in {directory}. It is "
            "written by the generation run itself: run `make graphs` with "
            f"{audit.AUDIT_ENV} set, which is what the graphs target does, "
            "and check again."
        )
        return 1

    unexpected = wrong_dpi(recorded)
    if unexpected:
        print(
            f"::error::the recording was taken at {sorted(unexpected)} dpi and "
            f"the thresholds are pinned to {REQUIRED_DPI}. A struck-pixel count "
            "scales with the square of the resolution, so the two are not "
            "comparable; regenerate with the figure theme's rcParams."
        )
        return 1

    if not args.partial:
        missing = committed_figures() - set(recorded)
        if missing:
            print(
                f"::error::the recording in {directory} covers {len(recorded)} "
                f"drawing(s) and misses {len(missing)} that `make graphs` "
                f"produces (every figure in both languages, the `_es` twin "
                f"included), so it is not a full run: "
                f"{', '.join(sorted(missing)[:5])}..."
            )
            print("  -> run `make graphs` (it empties the directory first).")
            return 1

    return report(recorded, read_exemptions(EXEMPTIONS), partial=args.partial)


if __name__ == "__main__":
    sys.exit(main())
