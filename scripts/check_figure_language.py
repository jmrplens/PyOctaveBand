#!/usr/bin/env python3
"""Find English text inside the Spanish edition of a figure.

Every figure ships in four variants (``X.svg``, ``X_dark.svg``, ``X_es.svg``,
``X_es_dark.svg``), and the Spanish ones are not drawn in Spanish. The figure
is drawn once, in English, and translated at save time: ``figures/i18n.py``
walks the finished figure's ``Text`` artists and its tick formatters and looks
every string up, by exact match or by pattern; the plates do the same one
label at a time through ``diagrams/canvas.py``'s ``SVG.tr``. A string nobody
added to the table is not an error. The lookup misses, the English text is
written into ``X_es.svg``, and nothing downstream can see it: the page is
Spanish, the i18n parity gate sees a translated page, the site builds, the
accessibility audit passes, and the figure inside the page is in English. It
is visible only by opening the image and reading it, which is how roughly two
hundred figures reached the tree in that state.

Two ways of looking for it do not work:

* **Diffing the text nodes of ``X.svg`` and ``X_es.svg``.** A label with any
  mathtext in it is not a string in the file: matplotlib lays it out glyph by
  glyph, so ``$|\\mu|$ = fluid / frame displacement`` is written as thirty
  positioned ``<tspan>``\\ s with the mu pushed to the end (biot_waves_es.svg,
  as committed). There is nothing there to look up in a table or to name in a
  report. Nor does that method reach a label a tick formatter builds at draw
  time: it is on no ``Text`` artist beforehand, which is why
  ``_translate_figure`` has a branch for the string-category axis that rebuilds
  its labels on every draw. Both hid real misses.
* **Counting the misses per figure.** A count lets a figure swap one
  untranslated string for another and stay green, so the baseline below holds
  the strings themselves.

So the check is made where the translation happens: the lookup reports what
it could not translate (:mod:`figure_language_audit`), during the generation
run that CI already pays for, and this script reads the result.

What it refuses
---------------

A **ratchet**, failing in both directions:

* an untranslated string that :data:`BASELINE` does not already record fails,
  which is what stops the debt growing;
* a baseline entry whose string now translates fails too, asking for the line
  to be deleted. Without that half the baseline rots: a figure that was fixed
  can silently break again while its stale entry covers it.

:data:`ENGLISH_BY_DESIGN` is the escape hatch, and it is deliberately not
shaped like the baseline: an entry is a string *and the reason it stays in
English*, because it is a permanent decision rather than debt to be paid. A
string may be in one or the other, never both -- the check refuses that too,
as it means nobody decided which it is.

Running it
----------

The strings come from a generation run, so the check needs one::

    make graphs          # records into build/figure-language
    make figure-language # reads it

``make graphs`` empties the directory first. Reading a directory some other
run filled would be answering about the wrong tree, so the check requires the
recording to cover every committed figure and plate, and says so instead of
passing when it does not. ``--partial`` is for a targeted re-render of one
figure: it checks what that run produced and reports no baseline line as
stale, because a run that did not draw a figure cannot say anything about it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

_SCRIPTS = pathlib.Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import figure_language_audit as audit

#: The committed debt: every figure that ships an untranslated string today,
#: and the strings it ships. One ``stem: "string"`` per line, JSON-quoted so a
#: multi-line label stays on one line, and sorted, so a diff of this file
#: reads as "this figure gained/lost this string".
BASELINE = _SCRIPTS / "figure_language_baseline.txt"

#: The committed figures, which say what a full generation run produces.
IMAGES = _SCRIPTS.parent / ".github" / "images"

#: Strings that stay in English on purpose, each with the reason it does.
#: Unlike the baseline, nothing here is ever expected to be paid off: a string
#: belongs here only if translating it would make the figure wrong, which is
#: why an entry is a decision with a reason attached rather than a bare line.
#:
#: A string that merely happens to read the same in both languages does not
#: belong here -- write it into the translation table as its own translation
#: (``"Error [dB]": "Error [dB]"``, which is what ``figures/i18n.py`` does),
#: because that says "looked at, and it is the same", which is a different
#: statement from "do not translate this".
ENGLISH_BY_DESIGN: dict[str, str] = {
    'PlaneWaveSource("down", offset = 60)': "the API call the figure documents, quoted as the reader would type it",
    "narrowband=True (Goertzel)": "a keyword argument of the API the figure documents, and the name of "
    "the algorithm behind it",
    "shape (4, N)": "a numpy attribute and its value, as the reader sees it printed",
    "OctaveFilterBank": "a class name; translating it would name a class that does not exist",
    "francois-garrison": "the model= literal of seawater_absorption, quoted as the reader "
    "would type it",
    "ainslie-mccolm": "the model= literal of seawater_absorption, quoted as the reader "
    "would type it",
    "thorp": "the model= literal of seawater_absorption, quoted as the reader "
    "would type it",
    "unesco": "the model= literal of sound_speed_profile, quoted as the reader "
    "would type it",
    "mackenzie": "the model= literal of sound_speed_profile, quoted as the reader "
    "would type it",
    "medwin": "the model= literal of sound_speed_profile, quoted as the reader "
    "would type it",
    "surfaces = [": "the opening line of the surfaces input list the EN 12354-6 take-off "
    "plate prints, code the reader types",
    "objects = hard_object_absorption(volumes)": "the API call the EN 12354-6 take-off plate documents, quoted as the "
    "reader would type it",
    "psi = object_fraction(volumes, 29.75)": "the API call the EN 12354-6 take-off plate documents, quoted as the "
    "reader would type it",
    "'rigid_cross', 'through' / 'corner'": "the junction_type= and path= literals of junction_vibration_reduction, "
    "quoted as the reader would type them",
    "'rigid_t', 'through' / 'corner'": "the junction_type= and path= literals of junction_vibration_reduction, "
    "quoted as the reader would type them",
    "'flexible_t', 'through' / 'corner'": "the junction_type= and path= literals of junction_vibration_reduction, "
    "quoted as the reader would type them",
    "'corner', 'corner'": "the junction_type= and path= literals of junction_vibration_reduction, "
    "quoted as the reader would type them",
    "'thickness_change', 'through'": "the junction_type= and path= literals of junction_vibration_reduction, "
    "quoted as the reader would type them",
    # The machine-fault line names drawn by FaultFrequencyResult.plot() are
    # the keys of the result the figure documents (res["2xGMF+1x"],
    # res["lobe n=1 m=2"]...), and the library draws line.name verbatim even
    # for language="es": the label is what the reader indexes, in either
    # language, so translating one would name a key that does not exist.
    "2xGMF": "a FaultFrequencyResult key, drawn as the reader indexes it",
    "2xGMF+1x": "a FaultFrequencyResult key, drawn as the reader indexes it",
    "2xGMF+2x": "a FaultFrequencyResult key, drawn as the reader indexes it",
    "2xGMF-1x": "a FaultFrequencyResult key, drawn as the reader indexes it",
    "2xGMF-2x": "a FaultFrequencyResult key, drawn as the reader indexes it",
    "3xGMF": "a FaultFrequencyResult key, drawn as the reader indexes it",
    "3xGMF+1x": "a FaultFrequencyResult key, drawn as the reader indexes it",
    "3xGMF+2x": "a FaultFrequencyResult key, drawn as the reader indexes it",
    "3xGMF-1x": "a FaultFrequencyResult key, drawn as the reader indexes it",
    "3xGMF-2x": "a FaultFrequencyResult key, drawn as the reader indexes it",
    "fsh": "the rotor-slot key of induction_motor_frequencies, drawn as the "
    "reader indexes it",
    "lobe n=1 m=2": "a lobe-pattern key of blade_pass_frequencies, drawn as the reader "
    "indexes it",
    "lobe n=1 m=10": "a lobe-pattern key of blade_pass_frequencies, drawn as the reader "
    "indexes it",
    "butter": "the filter_type= literal of FilterDesign, quoted as the reader "
    "would type it (architecture_tradeoff tick labels)",
    "cheby1": "the filter_type= literal of FilterDesign, quoted as the reader "
    "would type it (architecture_tradeoff tick labels)",
    "cheby2": "the filter_type= literal of FilterDesign, quoted as the reader "
    "would type it (architecture_tradeoff tick labels)",
    "ellip": "the filter_type= literal of FilterDesign, quoted as the reader "
    "would type it (architecture_tradeoff tick labels)",
    "bessel": "the filter_type= literal of FilterDesign, quoted as the reader "
    "would type it (architecture_tradeoff tick labels)",
    "octave_filter · leq · laeq · sel · ln_levels · lc_peak · "
    "OctaveFilterBank": "the level functions and the bank class of the API, listed as the "
    "reader would type them (calibration data-flow plate)",
    "laeq · sel · lc_peak": "the level functions of the API, listed as the reader would type "
    "them (sound-level-meter pipeline plate)",
}

# A run of at least three letters, in any alphabet: shorter runs are unit and
# axis tokens ("Hz", "dB", "kg") that carry no language.
_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)

# Mathtext control words. "$\mathdefault{2^{11}}$" is a matplotlib-generated
# tick label and "$\eta$" is a symbol; neither is text anybody translates, and
# both would otherwise read as words.
_MATHTEXT = re.compile(r"\\[A-Za-z]+")

# A whitespace-delimited token that carries code punctuation: an assignment, an
# underscore identifier, a quoted literal, a call. "resample_poly(1, M)",
# 'field="free"' and "10 log10(S/S0)" are drawn as they are typed or written,
# in either language, so the words inside them are not prose. The rest of the
# string is still read: only the token is dropped, so "s't, frame (Formula 4)"
# keeps "frame".
_CODE_TOKEN = re.compile(r"\S*(?:[_=\"]|\w\()\S*")


def words(text: str) -> list[str]:
    """The words of *text* that would have to be translated, if any.

    All-uppercase runs are left out: an acronym or a standard designation
    (``ISO``, ``MLS``, ``EDT``, ``BPFO``) is the same in both languages.
    """
    stripped = _CODE_TOKEN.sub(" ", _MATHTEXT.sub(" ", text))
    return [word for word in _WORD.findall(stripped) if not word.isupper()]


def is_prose(text: str) -> bool:
    """Whether *text* is language-bearing rather than numbers and symbols."""
    return bool(words(text))


def read_baseline(path: pathlib.Path) -> dict[str, set[str]]:
    """Parse the committed baseline into ``stem -> strings``."""
    entries: dict[str, set[str]] = {}
    if not path.exists():
        return entries
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stem, _, raw = line.partition(": ")
        if not raw:
            raise SystemExit(f"{path}:{number}: expected 'stem: \"string\"'")
        entries.setdefault(stem, set()).add(json.loads(raw))
    return entries


def write_baseline(path: pathlib.Path, found: dict[str, set[str]]) -> None:
    """Rewrite the baseline from *found* (``--write-baseline``)."""
    lines = [
        "# Figures whose Spanish variant still ships English text, and the",
        "# strings it ships. Written by scripts/check_figure_language.py, which",
        "# fails on anything not listed here AND on a line that is no longer",
        "# true: an entry is paid off by translating the string in",
        "# scripts/figures/i18n.py (or scripts/diagrams/i18n.py) and deleting",
        "# the line. Nothing is ever added by hand -- a string that must stay",
        "# in English belongs in ENGLISH_BY_DESIGN, with its reason.",
        "",
    ]
    for stem in sorted(found):
        for text in sorted(found[stem]):
            lines.append(f"{stem}: {json.dumps(text, ensure_ascii=False)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def untranslated(recorded: dict[str, dict[str, set[str]]]) -> dict[str, set[str]]:
    """The English text each asset ships inside its Spanish variant.

    A string counts when the Spanish pass could not translate it *and* the
    English pass drew it too. The second half is what keeps out the labels the
    library renderer translates on its own (``phonometry._plot`` has its own
    tables, and the generators pass ``language=``): those reach the Spanish
    pass already in Spanish, unknown to these tables and rightly so. It also
    means a string the library fails to translate is caught here, because it
    then reaches both passes identically.
    """
    english, spanish = recorded["english"], recorded["spanish"]
    found: dict[str, set[str]] = {}
    for stem, strings in spanish.items():
        shipped = {
            text
            for text in strings & english.get(stem, set())
            if is_prose(text) and text not in ENGLISH_BY_DESIGN
        }
        found[stem] = shipped
    return found


def committed_assets() -> tuple[set[str], set[str]]:
    """``(required, optional)``: what a full run must produce, and what it may.

    Read off the committed images rather than off the registries, because a
    generator's name is not its output name: ``generate_filter_responses``
    writes ten files, none of them called ``filter_responses``, and the
    recording is keyed by what is written. Every committed ``*_es.svg`` /
    ``*_es.webp`` is a figure or plate a full ``make graphs`` regenerates, so
    a recording that misses one is not a full run.

    The clips (``anim_*_es.webm``) are optional: they come from ``make
    animations``, deliberately outside CI because the encodes are slow and
    video is not byte-reproducible. A baseline line for a clip is therefore
    not stale merely because this run did not render it.
    """
    required: set[str] = set()
    optional: set[str] = set()
    for path in IMAGES.glob("*_es.*"):
        if path.suffix in (".svg", ".webp"):
            required.add(path.stem.removesuffix("_es"))
        elif path.suffix == ".webm":
            optional.add(path.stem.removesuffix("_es"))
    return required, optional


def report(
    found: dict[str, set[str]],
    baseline: dict[str, set[str]],
    optional: set[str],
    *,
    partial: bool,
) -> int:
    """Compare the run against the baseline in both directions."""
    problems: list[str] = []

    new: list[tuple[str, str]] = []
    for stem in sorted(found):
        for text in sorted(found[stem] - baseline.get(stem, set())):
            new.append((stem, text))
    if new:
        problems.append(
            f"{len({stem for stem, _ in new})} figure(s) ship untranslated "
            f"English text that the baseline does not record:"
        )
        for stem, text in new:
            problems.append(f"  {stem}: {json.dumps(text, ensure_ascii=False)}")
        problems.append(
            "  -> translate them in scripts/figures/i18n.py (plates: "
            "scripts/diagrams/i18n.py), or, if the string must stay in "
            "English, add it to ENGLISH_BY_DESIGN with the reason."
        )

    if not partial:
        stale: list[tuple[str, str]] = []
        for stem in sorted(baseline):
            if stem not in found:
                # A clip the run did not render says nothing either way; a
                # stem nothing renders any more is a line to delete.
                if stem not in optional:
                    stale.extend((stem, text) for text in sorted(baseline[stem]))
                continue
            for text in sorted(baseline[stem] - found[stem]):
                stale.append((stem, text))
        if stale:
            problems.append(
                f"{len(stale)} baseline line(s) are no longer true "
                f"(translated, allowlisted or gone):"
            )
            for stem, text in stale:
                problems.append(f"  {stem}: {json.dumps(text, ensure_ascii=False)}")
            problems.append(
                f"  -> delete them from {BASELINE.name}. The baseline only "
                "ratchets down if a paid-off line is removed."
            )

    if problems:
        print("::error::a figure ships the wrong language - see below")
        print("\n".join(problems))
        return 1

    figures = sum(1 for strings in baseline.values() if strings)
    strings = sum(len(strings) for strings in baseline.values())
    print(
        f"No new untranslated figure text. Baseline unchanged: {strings} "
        f"string(s) across {figures} figure(s) still to translate."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--audit",
        default=audit.DEFAULT_DIR,
        metavar="DIR",
        help=f"where the generation run recorded (default: {audit.DEFAULT_DIR})",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="rewrite the baseline from this run instead of checking it",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="the run generated only some of the figures: check the new "
        "strings only, and neither require full coverage nor report a "
        "baseline line as stale",
    )
    args = parser.parse_args(argv)
    if args.write_baseline and args.partial:
        parser.error(
            "--write-baseline needs a full run: written from a partial one it "
            "would drop every figure that run did not draw"
        )

    directory = pathlib.Path(args.audit)
    recorded = audit.load(str(directory)) if directory.is_dir() else None
    if not recorded or not recorded["spanish"]:
        print(
            f"::error::no figure-language recording in {directory}. It is "
            "written by the generation run itself: run `make graphs` (or "
            f"`make animations`) with {audit.AUDIT_ENV} set, which is what "
            "the graphs target does, and check again."
        )
        return 1

    found = untranslated(recorded)
    required, optional = committed_assets()
    if not args.partial:
        missing = required - set(found)
        if missing:
            print(
                f"::error::the recording in {directory} covers "
                f"{len(found)} asset(s) and misses {len(missing)} that "
                "`make graphs` produces, so it is not a full run: "
                f"{', '.join(sorted(missing)[:5])}..."
            )
            print("  -> run `make graphs` (it empties the directory first).")
            return 1

    if args.write_baseline:
        write_baseline(BASELINE, {k: v for k, v in found.items() if v})
        strings = sum(len(v) for v in found.values())
        print(
            f"Wrote {BASELINE}: {strings} string(s) across "
            f"{sum(1 for v in found.values() if v)} figure(s)."
        )
        return 0

    baseline = read_baseline(BASELINE)
    both = {
        text
        for strings in baseline.values()
        for text in strings
        if text in ENGLISH_BY_DESIGN
    }
    if both:
        print(
            "::error::these strings are in the baseline AND in "
            "ENGLISH_BY_DESIGN, so nobody decided which they are:"
        )
        for text in sorted(both):
            print(f"  {json.dumps(text, ensure_ascii=False)}")
        return 1
    return report(found, baseline, optional, partial=args.partial)


if __name__ == "__main__":
    sys.exit(main())
