#  Copyright (c) 2026. Jose M. Requena-Plens
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
``llms-<area>.txt``
    One shard per documented area, each small enough to survive a single fetch.
    ``llms-full.txt`` is roughly 291k tokens, so a client that truncates at a
    few hundred kilobytes (most of them do) reads the first few metrology
    guides and nothing else, which is the worst possible sample: it looks like
    the old narrow library.
``llms-full.txt``
    The complete concatenation, kept for clients that can take it.

All of them are published to the site by site/scripts/copy-llms.mjs (prebuild).
Deterministic: regenerate with ``make llms``.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CONTENT = ROOT / "site" / "src" / "content" / "docs"
SITE_URL = "https://jmrplens.github.io/phonometry"
REPO_URL = "https://github.com/jmrplens/phonometry"
DOI = "10.5281/zenodo.21215280"

#: Shard budget. Comfortably under the truncation limit of the AI fetch tools
#: that cap response size, with room for a page to grow.
SHARD_LIMIT_BYTES = 200_000

#: Markdown files under docs/ that are not documentation pages.
NOT_PAGES = frozenset({"README", "CONFORMANCE", "ERRATA"})

#: The handful of pages whose docs/ filename does not match its site route.
#: Everything else is matched by filename, so this table only grows when a page
#: is deliberately renamed on the way to the site.
ROUTE_OVERRIDES = {
    "api-reference": "reference/api",
    "references": "reference/bibliography",
    "theory": "reference/theory",
    "theory-signal-analysis": "reference/theory/signal-analysis",
    "theory-perception": "reference/theory/perception",
    "theory-rooms-buildings": "reference/theory/rooms-buildings",
    "theory-materials-surfaces": "reference/theory/materials-surfaces",
    "theory-environment-transport": "reference/theory/environment-transport",
    "theory-vibration": "reference/theory/vibration",
}

#: The nine documented areas, in the order the landing page lists them. Each is
#: a section overview page whose prose links every guide in that area, so the
#: grouping follows the content rather than a second hand-kept list.
AREAS: tuple[tuple[str, str], ...] = (
    ("core-signal-analysis", "Core signal analysis"),
    ("hearing-perception", "Hearing and perception"),
    ("rooms-buildings", "Rooms and buildings"),
    ("materials-surfaces", "Materials and surfaces"),
    ("vibration", "Vibration and structure-borne sound"),
    ("environment-transport", "Environment and transport"),
    ("underwater", "Underwater acoustics"),
    ("sources-devices", "Sources and devices"),
    ("simulation", "Wave simulation"),
)

#: Pages that open the documentation rather than belonging to an area.
START_ROUTES = ("getting-started", "reference/why-phonometry", "about")


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


def _site_routes() -> dict[str, str]:
    """Site page stem -> route, for the English tree, API pages excluded."""
    routes: dict[str, str] = {}
    for path in sorted(CONTENT.rglob("*.md*")):
        rel = path.relative_to(CONTENT)
        if rel.parts[0] == "es" or rel.parts[:2] == ("reference", "api"):
            continue
        if "sections" in rel.parts:
            # Area overviews share their stem with a guide (room-acoustics,
            # human-vibration); the guide is the docs/ page's counterpart.
            continue
        routes[path.stem] = rel.with_suffix("").as_posix()
    return routes


def _routes() -> list[tuple[str, str]]:
    """Every docs/ page as ``(filename, site route)``.

    Raises if a page cannot be placed: an unmapped page would otherwise vanish
    from llms.txt silently, which is exactly how the hand-kept list rotted.
    """
    site = _site_routes()
    pages: list[tuple[str, str]] = []
    unmapped: list[str] = []
    for path in sorted(DOCS.glob("*.md")):
        if path.stem in NOT_PAGES:
            continue
        route = ROUTE_OVERRIDES.get(path.stem) or site.get(path.stem)
        if route is None:
            unmapped.append(path.name)
            continue
        pages.append((path.name, route))
    if unmapped:
        raise SystemExit(
            "generate_llms.py: no site route for "
            + ", ".join(f"docs/{name}" for name in unmapped)
            + ". Add it to ROUTE_OVERRIDES if the page is renamed on the way to "
            "the site, or delete the file if the page is gone."
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


def _area_members() -> dict[str, list[str]]:
    """Area slug -> guide routes, read from each area overview page.

    Each overview page introduces the guides in its area by linking them, so
    membership is taken from the prose. A guide reachable from two areas is
    filed under the first one in :data:`AREAS`, and anything no overview links
    lands in a trailing group, so the partition is total either way.
    """
    sections = CONTENT / "guides" / "sections"
    guides = {path.stem for path in (CONTENT / "guides").glob("*.md*")}
    members: dict[str, list[str]] = {}
    claimed: set[str] = set()
    for slug, _label in AREAS:
        page = sections / f"{slug}.md"
        text = page.read_text(encoding="utf-8") if page.exists() else ""
        found = [
            name
            for name in dict.fromkeys(
                re.findall(r"/phonometry/guides/([a-z0-9-]+)/", text)
            )
            if name in guides and name not in claimed
        ]
        claimed.update(found)
        members[slug] = [f"guides/{name}" for name in found]
    leftover = sorted(guides - claimed)
    if leftover:
        members["other"] = [f"guides/{name}" for name in leftover]
    return members


def _shard_members() -> dict[str, list[str]]:
    """Shard slug -> guide routes, at the finest grouping the tree offers.

    The index groups by the nine areas a reader recognises, but two of those
    areas are large enough that a whole-area shard blows the fetch budget on
    its own. Several areas are already subdivided in the navigation (octave
    filtering, levels and weighting, and so on), each with its own overview
    page, so sharding follows those subdivisions where they exist and falls
    back to the area where they do not.
    """
    sections = CONTENT / "guides" / "sections"
    areas = {slug for slug, _ in AREAS}
    subsections = sorted(
        path.stem for path in sections.glob("*.md*") if path.stem not in areas
    )
    guides = {path.stem for path in (CONTENT / "guides").glob("*.md*")}

    def linked(slug: str) -> list[str]:
        page = sections / f"{slug}.md"
        text = page.read_text(encoding="utf-8") if page.exists() else ""
        return [
            name
            for name in dict.fromkeys(
                re.findall(r"/phonometry/guides/([a-z0-9-]+)/", text)
            )
            if name in guides
        ]

    shards: dict[str, list[str]] = {}
    claimed: set[str] = set()
    # Subsections first: they are the finer grain, and claiming there keeps the
    # parent area shard down to whatever it holds directly.
    for slug in subsections:
        found = [name for name in linked(slug) if name not in claimed]
        if not found:
            continue
        claimed.update(found)
        shards[slug] = [f"guides/{name}" for name in found]
    for slug, _label in AREAS:
        found = [name for name in linked(slug) if name not in claimed]
        claimed.update(found)
        if found:
            shards[slug] = [f"guides/{name}" for name in found]
    leftover = sorted(guides - claimed)
    if leftover:
        shards["other"] = [f"guides/{name}" for name in leftover]
    return shards


def _page_title(path: pathlib.Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


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
        "pip install phonometry[report]    # + reportlab, for .report() PDF fiches",
        "pip install phonometry[full]      # all of the above",
        "```",
        "",
        (
            "Minimal usage (all functions treat time as the LAST axis; 2D input is "
            "(channels, samples)):"
        ),
        "",
        "```python",
        "import numpy as np",
        "from phonometry import metrology",
        "",
        "fs = 48000",
        "x = np.random.randn(fs)              # 1 s of signal (pressure units)",
        "spl, freq = metrology.octave_filter(x, fs, fraction=3)   # 1/3-octave bands",
        "la = metrology.laeq(x, fs)                               # A-weighted Leq",
        "stats = metrology.ln_levels(x, fs, n=(10, 50, 90))       # statistical levels",
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


def build_llms_txt(version: str) -> str:
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
    lines += [entry for route in START_ROUTES if (entry := link(route))]
    lines += [
        f"- [Guides index]({SITE_URL}/guides/)",
        f"- [Glossary of quantities]({SITE_URL}/reference/glossary/)",
        "",
    ]

    for slug, routes in members.items():
        lines += [f"## {labels.get(slug, 'Other guides')}", ""]
        if slug != "other":
            lines.append(f"- [Overview]({SITE_URL}/guides/sections/{slug}/)")
        lines += [entry for route in routes if (entry := link(route))]
        lines.append("")

    lines += ["## Theory and reference", ""]
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
        f"- [About the author and the method]({SITE_URL}/about/)",
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
        f"- [Start here]({SITE_URL}/llms-start.txt)",
    ]
    for slug in _shard_members():
        label = labels.get(slug, slug.replace("-", " ").capitalize())
        lines.append(f"- [{label}]({SITE_URL}/llms-{slug}.txt)")
    lines += [
        f"- [Everything]({SITE_URL}/llms-full.txt)",
        "",
        "## Spanish",
        "",
        (
            "Every page above also exists in Spanish under the `/es/` prefix, for "
            f"example {SITE_URL}/es/getting-started/. The two trees are kept at "
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


def _absolutize_links(content: str, pages: dict[str, str]) -> str:
    """Rewrite docs-relative markdown links to canonical absolute URLs.

    The shards are served as flat text, so a link relative to docs/ (for
    example ``[x](filter-banks.md#anchor)``) would not resolve.
    """
    route_for = dict(pages)
    route_for["README.md"] = ""
    route_for["CONFORMANCE.md"] = "reference/conformance"
    route_for["ERRATA.md"] = "reference/errata"

    def repl(match: re.Match[str]) -> str:
        target, anchor = match.group(1), match.group(2) or ""
        if target == "../CONTRIBUTING.md":
            return f"]({REPO_URL}/blob/main/CONTRIBUTING.md{anchor})"
        route = route_for.get(target)
        if route is None:
            return match.group(0)
        suffix = f"{route}/" if route else ""
        return f"]({SITE_URL}/{suffix}{anchor})"

    return re.sub(r"\]\(((?:\.\./)?[\w.-]+\.md)(#[\w-]+)?\)", repl, content)


def _body(md_name: str, route: str, pages: dict[str, str]) -> str:
    """One page, ready to concatenate: attribution header then verbatim text."""
    content = (DOCS / md_name).read_text(encoding="utf-8")
    # Drop the docs-index backlink line; keep everything else verbatim.
    content = "\n".join(
        line
        for line in content.splitlines()
        if not line.startswith("← [Documentation index]")
    ).strip()
    content = _absolutize_links(content, pages)
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
    labels = dict(AREAS)

    def emit(title: str, routes: list[str]) -> str:
        parts = [
            f"# phonometry: {title}",
            "",
            f"Part of {SITE_URL}/llms.txt. Full text of the pages in this area.",
            "",
            "---",
        ]
        parts += [
            _body(name, route, pages)
            for route in routes
            if (name := route_to_file.get(route)) is not None
        ]
        return "\n".join(parts)

    theory = [route for route in pages.values() if route.startswith("reference/theory")]
    shards = {"start": emit("start here", [*START_ROUTES, *sorted(theory)])}
    for slug, routes in members.items():
        overview = f"guides/sections/{slug}"
        shards[slug] = emit(
            labels.get(slug, slug.replace("-", " ")),
            ([overview] if overview in route_to_file else []) + routes,
        )
    return shards


def build_llms_full(llms_txt: str) -> str:
    pages = dict(_routes())
    parts = [llms_txt, "\n---\n"]
    parts += [_body(name, route, pages) for name, route in _routes()]
    return "\n".join(parts)


def main() -> None:
    version = _version()
    llms = build_llms_txt(version)
    (ROOT / "llms.txt").write_text(llms, encoding="utf-8")
    (ROOT / "llms-full.txt").write_text(build_llms_full(llms), encoding="utf-8")

    shards = build_shards()

    # Remove shards this run no longer produces. Regrouping a section renames
    # its shard, and an orphan left behind would keep serving text that
    # llms.txt no longer indexes, silently and indefinitely.
    keep = {f"llms-{slug}.txt" for slug in shards} | {"llms.txt", "llms-full.txt"}
    for path in ROOT.glob("llms-*.txt"):
        if path.name not in keep:
            path.unlink()
            print(f"  removed stale shard {path.name}")

    oversized: list[tuple[str, int]] = []
    for slug, text in shards.items():
        (ROOT / f"llms-{slug}.txt").write_text(text, encoding="utf-8")
        size = len(text.encode("utf-8"))
        if size > SHARD_LIMIT_BYTES:
            oversized.append((slug, size))

    print(
        f"llms.txt regenerated (v{version}): {len(_routes())} pages, "
        f"{len(_api_routes())} API pages listed as optional."
    )
    for slug, size in oversized:
        print(
            f"  note: llms-{slug}.txt is {size / 1000:.0f} kB, over the "
            f"{SHARD_LIMIT_BYTES / 1000:.0f} kB budget; consider splitting the area."
        )


if __name__ == "__main__":
    main()
