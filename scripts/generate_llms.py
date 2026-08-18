#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Generate llms.txt, the per-area shards and llms-full.txt.

llms.txt is the structured, LLM-oriented summary of the project: what it is,
how to install and use it, and where every documentation page lives.

The page list used to be a literal table maintained by hand. It was written
when the project was a fractional octave filter bank and it was never extended
as the library grew, so it ended up naming 42 of 221 English pages: whole
domains the front page advertises (aircraft noise, underwater propagation, FDTD
simulation, the entire API reference) were invisible to anything reading it,
and the summary still described a filter bank. The list is derived from the
documentation tree now, and :func:`_routes` raises if a page cannot be placed,
so the same drift cannot happen quietly again.

Three kinds of artifact are produced:

``llms.txt``
    The index. Every page, grouped by area, plus the evidence pages, the
    Spanish tree and the API reference as an Optional section.
``llms/llms-<area>.txt``
    One shard per documented area, each small enough to survive a single fetch.
    ``llms-full.txt`` is roughly 291k tokens, so a client that truncates at a
    few hundred kilobytes (most of them do) reads the first few metrology
    guides and nothing else, which is the worst possible sample: it looks like
    the old narrow library.
``llms-full.txt``
    The complete concatenation, kept for clients that can take it.

The two index-level artifacts live at the repo root and are published to the
site root by site/scripts/copy-llms.mjs (prebuild). The shards are written
straight into site/public/llms/, which Astro serves verbatim under
``/llms/``, so the site root holds exactly the two files the llms.txt
convention names. Deterministic: regenerate with ``make llms``.
"""

from __future__ import annotations

import functools
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


@functools.cache
def _ignored_paths() -> frozenset[str]:
    """Everything under docs/ that git ignores, as repo-relative paths."""
    result = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--", "docs"],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    return frozenset(result.stdout.split())


def _git_ignored(path: pathlib.Path) -> bool:
    """Is this file one git ignores (a local plan, a scratch note)?"""
    return path.relative_to(ROOT).as_posix() in _ignored_paths()


CONTENT = ROOT / "site" / "src" / "content" / "docs"
#: Where the per-area shards live. Everything under site/public is served
#: verbatim at the site root, so these publish at /llms/llms-<area>.txt and
#: the site root keeps exactly llms.txt and llms-full.txt.
SHARDS_DIR = ROOT / "site" / "public" / "llms"
SITE_URL = "https://jmrplens.github.io/phonometry"
REPO_URL = "https://github.com/jmrplens/phonometry"
DOI = "10.5281/zenodo.21215280"

#: Shard budget. Comfortably under the truncation limit of the AI fetch tools
#: that cap response size, with room for a page to grow. Enforced: main() exits
#: non-zero if a shard is over it (see :func:`_check_shard_budget`). A shard
#: exactly at the limit passes; only strictly over fails, both here and in the
#: message it raises.
SHARD_LIMIT_BYTES = 200_000

#: Markdown files under docs/ that are not documentation pages.
NOT_PAGES = frozenset({"README", "CONFORMANCE", "ERRATA"})

#: The documented topics, in the order the landing page lists them. Each is a
#: folder of `content/docs/` whose `index` page links the guides inside it, so
#: the grouping follows the content rather than a second hand-kept list.
AREAS: tuple[tuple[str, str], ...] = (
    ("signals", "Signal analysis"),
    ("io", "Audio files"),
    ("perception", "Hearing and perception"),
    ("buildings", "Rooms and buildings"),
    ("materials", "Materials and surfaces"),
    ("vibration", "Vibration and structure-borne sound"),
    ("environment", "Environment and transport"),
    ("aircraft", "Aircraft noise"),
    ("underwater", "Underwater acoustics"),
    ("devices", "Sources and devices"),
    ("simulation", "Wave simulation"),
)

#: Pages that open the documentation rather than belonging to a topic. About
#: joined them once it gained a mirror edition; before that it was linked by
#: absolute URL from the provenance section instead. It used to be listed
#: without a mirror and silently dropped, which is why an unresolvable entry is
#: fatal now.
START_ROUTES = ("start/getting-started", "start/why-phonometry", "start/about")


def _check_areas_cover_the_tree() -> None:
    """AREAS keeps the order and the labels; the tree decides membership.

    An unlisted topic folder used to vanish quietly: the page counter rose, the
    text landed in llms-full.txt, and the page was named nowhere in the index
    and carried by no shard, with exit 0. The docstring promised that drift
    could not happen quietly, and for this one list it could.
    """
    listed = {slug for slug, _ in AREAS}
    on_disk = {
        path.parent.name
        for path in CONTENT.glob("*/index.md*")
        if path.parent.name not in ("es", "start", "reference")
    }
    missing = sorted(on_disk - listed)
    ghosts = sorted(listed - on_disk)
    if missing or ghosts:
        parts = []
        if missing:
            parts.append(
                "topic folder(s) with no AREAS entry: " + ", ".join(missing)
            )
        if ghosts:
            parts.append("AREAS entr(ies) with no topic folder: " + ", ".join(ghosts))
        raise SystemExit("generate_llms.py: " + "; ".join(parts))


def _version() -> str:
    return (ROOT / "VERSION").read_text(encoding="utf-8").strip()


def _conformance_counts() -> tuple[int, int, int]:
    """Checks, domains and standards, from the generated report's headline."""
    source = (DOCS / "CONFORMANCE.md").read_text(encoding="utf-8")
    match = re.search(
        r"\*\*\d+\s*/\s*(\d+)\s+conformance checks pass\*\*\s+across\s+(\d+)\s+"
        r"domains\s+and\s+(\d+)\s+standards",
        source,
    )
    if match is None:
        raise SystemExit(
            "docs/CONFORMANCE.md: headline not found. Regenerate it with "
            "`make conformance`."
        )
    return int(match[1]), int(match[2]), int(match[3])


@functools.cache
def _routes() -> list[tuple[str, str]]:
    """Every docs/ page as ``(path under docs/, site route)``.

    The mirror is laid out like the site, so the route is the path: the page
    at ``docs/signals/filters/filter-banks.md`` is published at
    ``/signals/filters/filter-banks/``. Raises if a mirror page has no page on
    the site, which would otherwise vanish from llms.txt silently, exactly how
    the hand-kept list rotted.
    """
    pages: list[tuple[str, str]] = []
    unmapped: list[str] = []
    for path in sorted(DOCS.rglob("*.md")):
        if path.parent == DOCS and path.stem in NOT_PAGES:
            continue
        # Files git ignores are somebody's local working notes, not the mirror:
        # they have no site page by design, and CI never sees them. Without
        # this the target cannot run at all in a checkout that has any.
        if _git_ignored(path):
            continue
        route = path.relative_to(DOCS).with_suffix("").as_posix()
        # A folder's own page is `<folder>/index.md`, as on the site.
        route = route.removesuffix("/index")
        published = any(
            (CONTENT / f"{route}{ext}").exists() for ext in (".md", ".mdx")
        ) or any(
            (CONTENT / route / f"index{ext}").exists() for ext in (".md", ".mdx")
        )
        if not published:
            unmapped.append(route)
            continue
        pages.append((path.relative_to(DOCS).as_posix(), route))
    if unmapped:
        raise SystemExit(
            "generate_llms.py: no site page for "
            + ", ".join(f"docs/{route}.md" for route in unmapped)
            + ". The mirror mirrors the site: move the file to the route its "
            "page has, or delete it if the page is gone."
        )
    return pages


def _api_routes() -> list[str]:
    """Every generated API reference page route, English tree."""
    api = CONTENT / "reference" / "api"
    if not api.is_dir():
        return []
    return sorted(
        path.relative_to(CONTENT).with_suffix("").as_posix()
        for path in api.rglob("*.md*")
        if path.stem != "index"
    )


def _guide_routes() -> set[str]:
    """Every guide route: the pages of a topic folder, overviews excluded."""
    topics = {slug for slug, _ in AREAS}
    return {
        path.relative_to(CONTENT).with_suffix("").as_posix()
        for topic in topics
        for path in (CONTENT / topic).rglob("*.md*")
        if path.stem != "index"
    }


def _overview(folder: pathlib.Path) -> pathlib.Path:
    """The overview page of a topic or subgroup folder, whatever it is called."""
    for name in ("index.md", "index.mdx"):
        if (folder / name).exists():
            return folder / name
    return folder / "index.md"


def _linked_guides(
    overview: pathlib.Path, guides: set[str], folder: str
) -> list[str]:
    """Guide routes an overview links, in order, restricted to its own folder.

    The prose gives the reading order, which is the point of reading it, but an
    overview also cross-references guides that live elsewhere. Membership is
    structural: a page belongs to the folder it is in, and only its own
    overview can order it.
    """
    text = overview.read_text(encoding="utf-8") if overview.exists() else ""
    return [
        route
        for route in dict.fromkeys(
            re.findall(r"/phonometry/([a-z0-9-]+(?:/[a-z0-9-]+)*)/", text)
        )
        if route in guides and route.rsplit("/", 1)[0] == folder
    ]


def _folder_order(folder: str, guides: set[str]) -> list[str]:
    """The guides of one folder, in the order its own overview teaches them."""
    ordered = _linked_guides(_overview(CONTENT / folder), guides, folder)
    ordered += [
        route
        for route in sorted(guides)
        if route.rsplit("/", 1)[0] == folder and route not in ordered
    ]
    return ordered


def _subgroups(topic: str) -> list[str]:
    """Subgroup folders of a topic, in the order the topic overview links them."""
    folders = sorted(
        path.parent.relative_to(CONTENT).as_posix()
        for path in (CONTENT / topic).rglob("index.md*")
        if path.parent != CONTENT / topic
    )
    text = _overview(CONTENT / topic).read_text(encoding="utf-8")
    mentioned = [
        folder
        for folder in dict.fromkeys(
            re.findall(r"/phonometry/([a-z0-9-]+(?:/[a-z0-9-]+)*)/", text)
        )
        if folder in folders
    ]
    return mentioned + [folder for folder in folders if folder not in mentioned]


def _area_members() -> dict[str, list[str]]:
    """Topic slug -> guide routes, in the order the overviews teach them.

    A page belongs to the folder it is in, which is the taxonomy the tree
    already states; the prose of each overview only decides the order, since
    that is what a reader is being led through. Anything an overview does not
    mention still appears, alphabetically, after what it does.
    """
    guides = _guide_routes()
    members: dict[str, list[str]] = {}
    claimed: set[str] = set()
    for slug, _label in AREAS:
        routes = _folder_order(slug, guides)
        for folder in _subgroups(slug):
            routes += _folder_order(folder, guides)
        found = [route for route in routes if route not in claimed]
        claimed.update(found)
        members[slug] = found
    leftover = sorted(guides - claimed)
    if leftover:
        members["other"] = leftover
    return members


#: Manual carve-outs from a folder's own shard, for a folder whose guides are
#: still over the fetch budget after subgroup splitting. Today that is only
#: ``buildings/insulation``: ten guides and no further folder to divide by,
#: since it has no subgroups of its own (unlike ``buildings/rooms`` or
#: ``buildings/design``, which already get a shard each above). The split
#: follows how the folder's own overview already groups its pages by hand
#: (``docs/buildings/insulation/index.md``: "Laboratory.", "Field.", "Ratings
#: and the envelope."); the leaf names are its guide filenames and the label
#: mirrors that heading. The carved-out shard has no page of its own, so its
#: Overview link still resolves to the parent folder (see _shard_folders and
#: _shard_label below).
MANUAL_SPLITS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "buildings/insulation": (
        "buildings-insulation-ratings",
        "Insulation ratings and the envelope",
        ("insulation-ratings", "facade-insulation", "spanish-building-code"),
    ),
}


@functools.cache
def _shard_folders() -> dict[str, str]:
    """Shard slug -> the content folder it names.

    Undoing ``folder.replace("/", "-")`` with one string replace was lossy for
    a folder nested two levels down: ``devices-emission-vehicles`` mapped back
    to ``devices/emission-vehicles``, whose overview does not exist, and every
    consequence was silent. The mapping is recorded at the same walk that
    creates the slugs, never reconstructed from the slug.
    """
    folders = {topic: topic for topic, _ in AREAS}
    for topic, _ in AREAS:
        for path in (CONTENT / topic).rglob("index.md*"):
            if path.parent != CONTENT / topic:
                folder = path.parent.relative_to(CONTENT).as_posix()
                folders[folder.replace("/", "-")] = folder
    for folder, (slug, _label, _leaves) in MANUAL_SPLITS.items():
        folders[slug] = folder
    return folders


@functools.cache
def _shard_members() -> dict[str, list[str]]:
    """Shard slug -> guide routes, at the finest grouping the tree offers.

    The index groups by the ten topics a reader recognises, but several of
    those are large enough that a whole-topic shard blows the fetch budget on
    its own. Most topics are already subdivided in the tree (octave filtering,
    levels and weighting, and so on), each subgroup with its own overview page,
    so sharding follows those subdivisions where they exist and falls back to
    the topic where they do not. The shard slug is the folder path with slashes
    turned into dashes, so ``signal/filters`` publishes as
    ``llms-signal-filters.txt``.
    """
    guides = _guide_routes()
    shards: dict[str, list[str]] = {}
    claimed: set[str] = set()

    def slug_of(folder: str) -> str:
        return folder.replace("/", "-")

    # Subgroups first: they are the finer grain, and claiming there keeps the
    # parent topic shard down to whatever it holds directly.
    subgroups = sorted(
        path.parent.relative_to(CONTENT).as_posix()
        for topic, _ in AREAS
        for path in (CONTENT / topic).rglob("index.md*")
        if path.parent != CONTENT / topic
    )
    for folder in subgroups:
        found = [
            route
            for route in _linked_guides(_overview(CONTENT / folder), guides, folder)
            if route not in claimed
        ]
        found += [
            route
            for route in sorted(guides)
            if route.rsplit("/", 1)[0] == folder
            and route not in claimed
            and route not in found
        ]
        if not found:
            continue
        claimed.update(found)
        split = MANUAL_SPLITS.get(folder)
        if split is not None:
            split_slug, _label, leaves = split
            split_routes = {f"{folder}/{leaf}" for leaf in leaves}
            missing = sorted(split_routes - set(found))
            if missing:
                raise SystemExit(
                    f"generate_llms.py: MANUAL_SPLITS for {folder} names "
                    "route(s) not among its guides: " + ", ".join(missing)
                )
            # Order follows found (the overview's own reading order), not the
            # tuple above: MANUAL_SPLITS just names which guides move, not the
            # order to read them in.
            shards[split_slug] = [route for route in found if route in split_routes]
            found = [route for route in found if route not in split_routes]
        shards[slug_of(folder)] = found
    for topic, _label in AREAS:
        found = [
            route
            for route in sorted(guides)
            if route.split("/")[0] == topic and route not in claimed
        ]
        # Unconditional: a topic whose guides are all claimed by its subgroups
        # still gets a shard, carrying its own overview. Without it, the five
        # fully subdivided topics had their overview text in no shard at all.
        claimed.update(found)
        shards[slug_of(topic)] = found
    leftover = sorted(guides - claimed)
    if leftover:
        shards["other"] = leftover
    return shards


def _page_title(path: pathlib.Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def _shard_label(slug: str) -> str:
    """The human name of a shard: the H1 of the overview it opens with.

    ``slug.capitalize()`` named none of the sections the site shows: it said
    "Signal metrology" where the site says "Calibration and uncertainty" and
    "Buildings rooms" where it says "Room acoustics". The overview's own title
    is the name a reader of the site would recognise, and the mirror carries it
    now.
    """
    labels = dict(AREAS)
    if slug in labels:
        return labels[slug]
    split_labels = {
        split_slug: label for _folder, (split_slug, label, _leaves) in MANUAL_SPLITS.items()
    }
    if slug in split_labels:
        return split_labels[slug]
    folder = _shard_folders().get(slug, slug)
    index = DOCS / folder / "index.md"
    if index.exists():
        return _page_title(index)
    return slug.replace("-", " ").capitalize()


def _summary(version: str) -> list[str]:
    """The header every artifact opens with."""
    checks, domains, standards = _conformance_counts()
    return [
        "# phonometry",
        "",
        (
            "> Acoustic measurement, analysis and prediction in Python. Sound level "
            "metrology, psychoacoustics, room and building acoustics, materials, "
            "vibration, environmental and transport noise, underwater acoustics, "
            "electroacoustics and wave simulation, with every metric implemented "
            f"from the text of its governing standard and {checks} numerical "
            f"conformance checks against {standards} standards enforced in CI."
        ),
        "",
        (
            f"phonometry v{version} is a pure-Python library built on NumPy/SciPy "
            "(Python >= 3.13). Each result is a typed, frozen dataclass that carries "
            "the inputs it was computed from, draws its own figure with a one-line "
            "`.plot()` in English or Spanish, and, where a standard defines a "
            "reporting format, renders that format as a PDF with `.report()`. The "
            f"conformance report pins each of the {checks} checks to a standard, a "
            "clause or table, the normative expected value and the value the library "
            f"computes, across {domains} domains."
        ),
        "",
        (
            "The project was published as **PyOctaveBand** until version 3.0.0. "
            "Third-party references and dependency pins still using that name refer "
            "to this library."
        ),
        "",
        "Install:",
        "",
        "```bash",
        "pip install phonometry            # core (NumPy + SciPy only)",
        "pip install phonometry[plot]      # + matplotlib figures and .plot()",
        "pip install phonometry[perf]      # + numba-jitted impulse kernel",
        "pip install phonometry[report]    # + reportlab and svglib, for .report() PDF fiches",
        "pip install phonometry[full]      # all of the above (recommended)",
        "```",
        "",
        (
            "I recommend `pip install phonometry[full]`, which enables every "
            "feature. The base install computes every metric on NumPy and SciPy "
            "alone; without the extras only the `.plot()` figures and the "
            "`.report()` PDF fiches (whose figure panel needs matplotlib as well "
            "as reportlab and svglib) are unavailable. The `impulse` time "
            "weighting still works without numba, on a pure-Python kernel that "
            "is only slower."
        ),
        "",
        (
            "One caveat about `[full]`: numba is the only extra that caps "
            "NumPy, and it raises that cap only once it supports a new NumPy "
            "minor, so in the weeks after a NumPy minor release "
            "`phonometry[full]` (like `phonometry[perf]`) can resolve one minor "
            "behind what a plain install gets. numba only makes the `impulse` "
            "time weighting faster, so install `phonometry[plot,report]` "
            "instead if you need the newest NumPy the day it ships."
        ),
        "",
        (
            "Minimal usage (all functions treat time as the LAST axis; 2D input is "
            "(channels, samples)):"
        ),
        "",
        "```python",
        "import numpy as np",
        "from phonometry import filters, signals",
        "",
        "fs = 48000",
        "x = np.random.randn(fs)              # 1 s of signal (pressure units)",
        "spl, freq = filters.octave_filter(x, fs, fraction=3)   # 1/3-octave bands",
        "la = signals.laeq(x, fs)                                 # A-weighted Leq",
        "stats = signals.ln_levels(x, fs, n=(10, 50, 90))         # statistical levels",
        "```",
        "",
        (
            "If you are an AI assistant setting this up for a user: install from PyPI "
            "(no system dependencies), remember integer audio (e.g. wavfile.read "
            "int16) is handled automatically, use `calibration_factor` from "
            "`sensitivity()` for real dB SPL, and prefer `OctaveFilterBank` over "
            "repeated `octave_filter()` calls in tight loops (although designs are "
            "cached either way). The library computes standardized quantities; it is "
            "not a certified instrument and does not acquire data from hardware."
        ),
        "",
    ]


def build_llms_txt(version: str, shard_slugs: tuple[str, ...]) -> str:
    pages = dict(_routes())
    route_to_file = {route: name for name, route in pages.items()}
    members = _area_members()
    labels = dict(AREAS)

    def link(route: str) -> str | None:
        name = route_to_file.get(route)
        if name is None:
            return None
        return f"- [{_page_title(DOCS / name)}]({SITE_URL}/{route}/)"

    lines = _summary(version)
    lines += ["## Start here", ""]
    for route in START_ROUTES:
        entry = link(route)
        if entry is None:
            # These are listed by hand, so an unresolvable one is always a
            # bug; the About page went missing this way once, silently.
            raise SystemExit(
                f"generate_llms.py: START_ROUTES names {route}, "
                "which has no mirror page"
            )
        lines.append(entry)
    lines += [
        f"- [Guides index]({SITE_URL}/start/guides/)",
        f"- [Glossary of quantities]({SITE_URL}/reference/glossary/)",
        "",
    ]

    shard_of = {
        route: slug for slug, routes in _shard_members().items() for route in routes
    }
    for slug, routes in members.items():
        lines += [f"## {labels.get(slug, 'Other guides')}", ""]
        if slug != "other":
            lines.append(f"- [Overview]({SITE_URL}/{slug}/)")
        # The shards are per subgroup, so the index says which subgroup, and
        # therefore which shard, each run of guides belongs to: a client that
        # wants the full text of one part should not have to infer the shard
        # from the URL path.
        seen_groups: set[str] = set()
        for route in routes:
            group = shard_of.get(route, slug)
            if group != slug and group not in seen_groups:
                seen_groups.add(group)
                folder = _shard_folders().get(group, group)
                lines += [
                    "",
                    f"### {_shard_label(group)}",
                    "",
                    (
                        f"- [Overview]({SITE_URL}/{folder}/) · "
                        f"[full text]({SITE_URL}/llms/llms-{group}.txt)"
                    ),
                ]
            if entry := link(route):
                lines.append(entry)
        lines.append("")

    lines += [
        "## Theory and reference",
        "",
        (
            "The theory pages travel in their own shard "
            f"({SITE_URL}/llms/llms-theory.txt); the rest of the reference in "
            f"{SITE_URL}/llms/llms-reference.txt."
        ),
        "",
    ]
    for route in sorted(
        route
        for route in pages.values()
        if route.startswith("reference/") and route != "reference/why-phonometry"
    ):
        if entry := link(route):
            lines.append(entry)
    lines.append("")

    lines += [
        "## Evidence and provenance",
        "",
        (
            f"- [Conformance report]({SITE_URL}/reference/conformance/): every check, "
            "with the standard, the clause, the normative expected value, the value "
            "the library computes and the delta"
        ),
        (
            f"- [Errata in published sources]({SITE_URL}/reference/errata/): defects "
            "found in published standards, each re-derived from that standard's own "
            "normative clauses"
        ),
        f"- [Bibliography]({SITE_URL}/reference/bibliography/)",
        f"- [About the author and the method]({SITE_URL}/start/about/)",
        "",
        "## Source and metadata",
        "",
        f"- [Repository]({REPO_URL})",
        "- [PyPI](https://pypi.org/project/phonometry/)",
        f"- [Changelog]({REPO_URL}/blob/main/CHANGELOG.md)",
        f"- [Cite this software]({REPO_URL}/blob/main/CITATION.cff)",
        f"- [Archived release (DOI)](https://doi.org/{DOI})",
        "- [Licence: MIT](https://opensource.org/licenses/MIT)",
        "",
        "## Full text",
        "",
        (
            "One file per area, each small enough to fetch whole. `llms-full.txt` is "
            "the complete concatenation and is far too large for a single fetch in "
            "most clients."
        ),
        "",
        f"- [Start here]({SITE_URL}/llms/llms-start.txt)",
    ]
    if "theory" in shard_slugs:
        lines.append(f"- [Theory and reference]({SITE_URL}/llms/llms-theory.txt)")
    # Grouped by topic in the order the site lists them, the topic's own shard
    # first, so the list reads like the navigation rather than like a sort key.
    # The slugs are the shards actually built this run: a hand-kept list here
    # published a link to a shard that a content move could stop producing.
    # "theory" is listed by hand just above, next to "start": neither is a
    # topic, so the loop below would otherwise never place them.
    listed: set[str] = {"start", "theory"}
    folders = _shard_folders()
    for topic, _label in AREAS:
        # Subgroups in the order the topic overview teaches them, matching the
        # navigation above; alphabetical-by-slug put Calibration before Spectra
        # in a topic whose overview teaches them the other way round.
        nav_order = {folder: rank for rank, folder in enumerate(_subgroups(topic))}
        for slug in [t for t in shard_slugs if t == topic] + sorted(
            (t for t in shard_slugs if t.startswith(f"{topic}-")),
            key=lambda t: (nav_order.get(folders.get(t, ""), len(nav_order)), t),
        ):
            listed.add(slug)
            lines.append(f"- [{_shard_label(slug)}]({SITE_URL}/llms/llms-{slug}.txt)")
    for slug in shard_slugs:
        if slug not in listed:
            label = "Other guides" if slug == "other" else _shard_label(slug)
            lines.append(f"- [{label}]({SITE_URL}/llms/llms-{slug}.txt)")
    lines += [
        f"- [Everything]({SITE_URL}/llms-full.txt)",
        "",
        "## Spanish",
        "",
        (
            "Every page above also exists in Spanish under the `/es/` prefix, for "
            f"example {SITE_URL}/es/start/getting-started/. The two trees are kept at "
            "parity by a build check."
        ),
        "",
        "## Optional",
        "",
        (
            "The generated API reference, one page per module. Fetch these only when "
            "a specific signature is needed; the guides above explain the same "
            "functions in context."
        ),
        "",
        f"- [API reference index]({SITE_URL}/reference/api/)",
    ]
    lines += [
        f"- [{route.removeprefix('reference/api/')}]({SITE_URL}/{route}/)"
        for route in _api_routes()
    ]
    lines.append("")
    return "\n".join(lines)


def _absolutize_links(content: str, md_name: str) -> str:
    """Rewrite mirror-relative markdown links to canonical absolute URLs.

    The shards are served as flat text, so a link relative to the page that
    holds it (``[x](../filters/filter-banks.md#anchor)``) would not resolve.
    The mirror is laid out like the site, so the target resolved against the
    page's own folder is the route, with the two generated documents at the
    repository root transplanted into the routes that publish them.
    """
    here = pathlib.PurePosixPath(md_name).parent
    #: Mirror page at the repository root -> the route that publishes it.
    transplanted = {
        "README": "",
        "CONFORMANCE": "reference/conformance",
        "ERRATA": "reference/errata",
    }

    def repl(match: re.Match[str]) -> str:
        target, anchor = match.group(1), match.group(2) or ""
        resolved = (here / target).as_posix()
        parts: list[str] = []
        for step in resolved.split("/"):
            if step == "..":
                if parts:
                    parts.pop()
            elif step not in (".", ""):
                parts.append(step)
        route = "/".join(parts).removesuffix(".md").removesuffix("/index")
        if route == "CONTRIBUTING":
            return f"]({REPO_URL}/blob/main/CONTRIBUTING.md{anchor})"
        route = transplanted.get(route, route)
        suffix = f"{route}/" if route else ""
        return f"]({SITE_URL}/{suffix}{anchor})"

    return re.sub(r"\]\(((?:\.\./)*[\w./-]+\.md)(#[\w-]+)?\)", repl, content)


def _body(md_name: str, route: str, pages: dict[str, str]) -> str:
    """One page, ready to concatenate: attribution header then verbatim text."""
    content = (DOCS / md_name).read_text(encoding="utf-8")
    # Drop the docs-index backlink line; keep everything else verbatim.
    content = "\n".join(
        line
        for line in content.splitlines()
        if not line.startswith("← [Documentation index]")
    ).strip()
    content = _absolutize_links(content, md_name)
    # A plain "Source:" line as well as the HTML comment: markdown-to-text
    # pipelines routinely strip comments, and a passage extracted with its
    # attribution stripped is a passage that cannot be cited back.
    header = (
        f"<!-- source: docs/{md_name} | canonical: {SITE_URL}/{route}/ -->\n"
        f"Source: {SITE_URL}/{route}/\n"
    )
    return f"\n{header}\n{content}\n\n---\n"


def build_shards() -> dict[str, str]:
    """One text artifact per area, keyed by shard slug."""
    pages = dict(_routes())
    route_to_file = {route: name for name, route in pages.items()}
    members = _shard_members()

    emitted: set[str] = set()

    def emit(title: str, routes: list[str]) -> str:
        parts = [
            f"# phonometry: {title}",
            "",
            f"Part of {SITE_URL}/llms.txt. Full text of the pages in this area.",
            "",
            "---",
        ]
        for route in routes:
            name = route_to_file.get(route)
            if name is None:
                continue
            emitted.add(route)
            parts.append(_body(name, route, pages))
        return "\n".join(parts)

    # Theory used to travel inside the start shard: six mirror pages with
    # eighteen figures pushed llms-start.txt itself over budget. It gets its
    # own shard instead of a cut, since a client after the getting-started
    # page has no reason to also fetch the physics.
    theory = sorted(
        route for route in pages.values() if route.startswith("reference/theory")
    )
    start_overview = ["start"] if "start" in route_to_file else []
    shards = {"start": emit("start here", [*start_overview, *START_ROUTES])}
    if theory:
        shards["theory"] = emit("Theory and reference", theory)

    manual_split_slugs = {slug for _folder, (slug, _label, _leaves) in MANUAL_SPLITS.items()}
    for slug, routes in members.items():
        overview = _shard_folders().get(slug, slug)
        # A carved-out shard (see MANUAL_SPLITS) has no overview page of its
        # own; the folder it names already opened its primary shard above.
        include_overview = overview in route_to_file and slug not in manual_split_slugs
        shards[slug] = emit(
            _shard_label(slug),
            ([overview] if include_overview else []) + routes,
        )

    # The reference pages that are neither theory (its own shard) nor a
    # topic's: today that is the bibliography, which was indexed in llms.txt
    # while its text sat in no shard at all.
    # The API index stays out: llms.txt lists the whole generated reference as
    # Optional, one page per module, and a shard duplicating that table would
    # be the largest file in the set saying the least.
    reference = sorted(
        route
        for route in pages.values()
        if route.startswith("reference/")
        and not route.startswith(("reference/theory", "reference/api"))
    )
    if reference:
        shards["reference"] = emit("Reference", reference)

    # Every mirror page belongs to a shard; a page that falls between the
    # buckets is how the bibliography went missing. The set compared against is
    # what emit() actually consumed, not a second prediction of it: predicting
    # exempted the overview of a subgroup with no guides, which no shard holds.
    allowed_out = {"reference/api"}
    dropped = sorted(set(pages.values()) - emitted - allowed_out)
    if dropped:
        raise SystemExit(
            "generate_llms.py: mirror page(s) in no shard: " + ", ".join(dropped)
        )
    return shards


def build_llms_full(llms_txt: str) -> str:
    pages = dict(_routes())
    parts = [llms_txt, "\n---\n"]
    parts += [_body(name, route, pages) for name, route in _routes()]
    return "\n".join(parts)


def _check_readme_covers_the_mirror() -> None:
    """docs/README.md links every mirror guide page.

    The README is the GitHub front door of the docs, kept by hand, and three
    pages went missing from it inside one week of merges: nothing looked. The
    overview index files are exempt because the README's section structure is
    its own index of the folders, which
    ``mirror_overviews.py --check`` asserts separately (it requires every
    generated ``index.md`` to be linked from the README).
    """
    readme = (DOCS / "README.md").read_text(encoding="utf-8")
    missing = [
        name
        for name, _route in _routes()
        if not name.endswith("/index.md")
        and not name.startswith("reference/api")
        and f"({name})" not in readme
    ]
    if missing:
        raise SystemExit(
            "generate_llms.py: docs/README.md does not link " + ", ".join(missing)
        )


def _check_shard_budget(sizes: dict[str, int]) -> None:
    """Fail the run if any shard exceeds :data:`SHARD_LIMIT_BYTES`.

    This used to be a printed note with exit 0, so llms-start.txt and
    llms-buildings-insulation.txt sat over budget for weeks with a green CI: a
    client that truncates at a few hundred kilobytes read a partial shard and
    nothing here ever said so. The budget is not negotiable (see its own
    docstring); the only fix for an oversized shard is a narrower split, never
    a higher SHARD_LIMIT_BYTES.
    """
    oversized = sorted(
        (slug, size) for slug, size in sizes.items() if size > SHARD_LIMIT_BYTES
    )
    if not oversized:
        return
    detail = "\n".join(
        f"  llms-{slug}.txt is {size:,} bytes, {size - SHARD_LIMIT_BYTES:,} over "
        f"the {SHARD_LIMIT_BYTES:,} byte budget"
        for slug, size in oversized
    )
    raise SystemExit(
        "generate_llms.py: shard(s) exceed the fetch budget:\n"
        + detail
        + "\nSplit the area into a narrower shard; do not raise SHARD_LIMIT_BYTES."
    )


def main() -> None:
    _check_areas_cover_the_tree()
    _check_readme_covers_the_mirror()
    version = _version()
    shards = build_shards()
    llms = build_llms_txt(version, tuple(shards))
    (ROOT / "llms.txt").write_text(llms, encoding="utf-8")
    (ROOT / "llms-full.txt").write_text(build_llms_full(llms), encoding="utf-8")

    # Remove shards this run no longer produces. Regrouping a section renames
    # its shard, and an orphan left behind would keep serving text that
    # llms.txt no longer indexes, silently and indefinitely. The root-level
    # sweep also retires shards left by the era when they lived next to
    # llms.txt.
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    keep = {f"llms-{slug}.txt" for slug in shards}
    for path in SHARDS_DIR.glob("llms-*.txt"):
        if path.name not in keep:
            path.unlink()
            print(f"  removed stale shard llms/{path.name}")
    for path in ROOT.glob("llms-*.txt"):
        if path.name != "llms-full.txt":
            path.unlink()
            print(f"  removed stale shard {path.name}")

    sizes: dict[str, int] = {}
    for slug, text in shards.items():
        (SHARDS_DIR / f"llms-{slug}.txt").write_text(text, encoding="utf-8")
        sizes[slug] = len(text.encode("utf-8"))

    print(
        f"llms.txt regenerated (v{version}): {len(_routes())} pages, "
        f"{len(_api_routes())} API pages listed as optional."
    )
    _check_shard_budget(sizes)


if __name__ == "__main__":
    main()
