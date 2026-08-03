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

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CONTENT = ROOT / "site" / "src" / "content" / "docs"
#: Where the per-area shards live. Everything under site/public is served
#: verbatim at the site root, so these publish at /llms/llms-<area>.txt and
#: the site root keeps exactly llms.txt and llms-full.txt.
SHARDS_DIR = ROOT / "site" / "public" / "llms"
SITE_URL = "https://jmrplens.github.io/phonometry"
REPO_URL = "https://github.com/jmrplens/phonometry"
DOI = "10.5281/zenodo.21215280"

#: Shard budget. Comfortably under the truncation limit of the AI fetch tools
#: that cap response size, with room for a page to grow.
SHARD_LIMIT_BYTES = 200_000

#: Markdown files under docs/ that are not documentation pages.
NOT_PAGES = frozenset({"README", "CONFORMANCE", "ERRATA"})

#: The documented topics, in the order the landing page lists them. Each is a
#: folder of `content/docs/` whose `index` page links the guides inside it, so
#: the grouping follows the content rather than a second hand-kept list.
AREAS: tuple[tuple[str, str], ...] = (
    ("signal", "Signal analysis"),
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

#: Pages that open the documentation rather than belonging to a topic.
START_ROUTES = ("start/getting-started", "start/why-phonometry", "start/about")


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


def _routes() -> list[tuple[str, str]]:
    """Every docs/ page as ``(path under docs/, site route)``.

    The mirror is laid out like the site, so the route is the path: the page
    at ``docs/signal/filters/filter-banks.md`` is published at
    ``/signal/filters/filter-banks/``. Raises if a mirror page has no page on
    the site, which would otherwise vanish from llms.txt silently, exactly how
    the hand-kept list rotted.
    """
    pages: list[tuple[str, str]] = []
    unmapped: list[str] = []
    for path in sorted(DOCS.rglob("*.md")):
        if path.parent == DOCS and path.stem in NOT_PAGES:
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
        shards[slug_of(folder)] = found
    for topic, _label in AREAS:
        found = [
            route
            for route in sorted(guides)
            if route.split("/")[0] == topic and route not in claimed
        ]
        if found:
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
            "One caveat about `[full]`: numba declares `numpy<2.5`, so "
            "`phonometry[full]` (like `phonometry[perf]`) resolves NumPy below "
            "2.5 while a plain install gets the newest release. numba only makes "
            "the `impulse` time weighting faster, so install "
            "`phonometry[plot,report]` instead if you would rather keep NumPy "
            "current."
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
        f"- [Guides index]({SITE_URL}/start/guides/)",
        f"- [Glossary of quantities]({SITE_URL}/reference/glossary/)",
        "",
    ]

    for slug, routes in members.items():
        lines += [f"## {labels.get(slug, 'Other guides')}", ""]
        if slug != "other":
            lines.append(f"- [Overview]({SITE_URL}/{slug}/)")
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
    for slug in _shard_members():
        label = labels.get(slug, slug.replace("-", " ").capitalize())
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
        overview = slug.replace("-", "/", 1) if "-" in slug else slug
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

    oversized: list[tuple[str, int]] = []
    for slug, text in shards.items():
        (SHARDS_DIR / f"llms-{slug}.txt").write_text(text, encoding="utf-8")
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
