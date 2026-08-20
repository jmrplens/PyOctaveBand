#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The repository's own configuration, checked against what it describes.

Three files here make claims about other files, and nothing compiled them, so
they can be wrong for as long as nobody rereads them. ``.coderabbit.yaml``
tells the review bot which pages under ``docs/`` are build output, and the bot
will act on that by rejecting a hand edit CI actually requires. The docs
workflow lists the paths that make the site rebuild, and the site build is
where the citation guards live, so a root file the build reads and the trigger
omits is a guard that cannot fail on the pull request that breaks it. And a
comment in the Python workflow quotes a dependency floor that lives in
``pyproject.toml``.

Each test below derives the truth from the thing being described rather than
from a second copy of the claim.
"""

from __future__ import annotations

import pathlib
import re
import sys
import tomllib

import pytest
import yaml

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPTS = str(_ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import mirror_overviews

_CODERABBIT = _ROOT / ".coderabbit.yaml"
_DOCS_WORKFLOW = _ROOT / ".github" / "workflows" / "docs.yml"
_PYTHON_WORKFLOW = _ROOT / ".github" / "workflows" / "python-app.yml"
_CITATION_MODULE = _ROOT / "site" / "src" / "data" / "citation.mjs"


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """A glob from ``.coderabbit.yaml`` as a regex over repository paths.

    Only the constructs that file uses are understood: ``**`` for any run of
    directories, ``*`` inside one name, and a whole-segment negation
    ``!(a|b)``. Anything else raises rather than being dropped, so a future
    pattern can never be judged by a translation that quietly ignored half of
    it.
    """
    segments = pattern.split("/")
    out = ""
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if segment == "**":
            out += ".+" if last else r"(?:[^/]+/)*"
            continue
        negated = re.fullmatch(r"!\(([^()]+)\)", segment)
        if negated:
            alternatives = "|".join(re.escape(a) for a in negated.group(1).split("|"))
            out += rf"(?!(?:{alternatives})(?:/|$))[^/]+"
        elif re.fullmatch(r"[\w.*-]+", segment):
            out += re.escape(segment).replace(r"\*", "[^/]*")
        else:
            msg = f"unsupported glob segment {segment!r} in {pattern!r}"
            raise AssertionError(msg)
        if not last:
            out += "/"
    return re.compile(out)


def _path_instructions() -> list[dict[str, str]]:
    config = yaml.safe_load(_CODERABBIT.read_text(encoding="utf-8"))
    rules = config["reviews"]["path_instructions"]
    assert rules, ".coderabbit.yaml carries no path instructions"
    return list(rules)


def _generated_overview_mirrors() -> set[str]:
    """The mirror indexes ``scripts/mirror_overviews.py`` actually writes."""
    return {
        mirror_overviews._mirror_path(
            page.parent.relative_to(mirror_overviews.SITE).as_posix()
        )
        .relative_to(_ROOT)
        .as_posix()
        for page in mirror_overviews._overviews()
    }


def test_the_generated_index_rule_covers_only_generated_indexes() -> None:
    """The rule that says "will be overwritten" must match nothing else.

    ``mirror_overviews.py`` skips ``reference/``, so ``docs/reference/api``
    and ``docs/reference/theory`` keep hand-written indexes. The API one is
    the sharper case: ``scripts/check_api_reference.py`` fails until a new
    public name is added to it by hand, so telling the bot the file is build
    output points it at the opposite of what CI demands.
    """
    generated = _generated_overview_mirrors()
    assert generated, "the overview generator claims to write nothing"

    on_disk = sorted(
        p.relative_to(_ROOT).as_posix() for p in _ROOT.glob("docs/**/*.md")
    )
    rules = [
        r for r in _path_instructions() if "mirror_overviews.py" in r["instructions"]
    ]
    generated_rules = [r for r in rules if "will be overwritten" in r["instructions"]]
    assert len(generated_rules) == 1, (
        "expected exactly one rule declaring the overview mirrors generated"
    )

    matcher = _glob_to_regex(generated_rules[0]["path"])
    matched = {path for path in on_disk if matcher.fullmatch(path)}
    assert matched == generated, (
        "the .coderabbit.yaml glob and the generator disagree; "
        f"claimed but hand-written: {sorted(matched - generated)}; "
        f"generated but unclaimed: {sorted(generated - matched)}"
    )


def test_the_readme_is_not_advertised_as_generated() -> None:
    """``docs/README.md`` is authored, and two gates require editing it.

    ``mirror_overviews.py`` and ``generate_llms.py`` both read it to assert it
    links every mirror page; neither writes it. Listing it among the generated
    files is the same false claim this configuration exists to correct.
    """
    claims = [
        rule["instructions"]
        for rule in _path_instructions()
        if "generated files under" in rule["instructions"]
    ]
    assert claims, (
        "no rule states which files under docs/ are generated; update this test "
        "with the phrasing that replaced it"
    )
    for text in claims:
        paragraph = text[text.index("generated files under") :]
        sentence = paragraph[: paragraph.index(".") + 1]
        assert "README" not in sentence, (
            f"the README is listed as generated output: {sentence!r}"
        )

    assert not any(
        path.endswith("README.md") for path in _generated_overview_mirrors()
    ), "the overview generator writes a README after all; the claim was right"


def _trigger_paths(workflow: pathlib.Path, event: str) -> set[str]:
    parsed = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    # ``on:`` is YAML 1.1's boolean true, which is what safe_load returns here.
    triggers = parsed[True] if True in parsed else parsed["on"]
    return set(triggers[event]["paths"])


def test_the_docs_workflow_triggers_on_every_root_file_the_site_build_reads() -> None:
    """A root file the site build reads has to be able to trigger that build.

    ``site/src/data/citation.mjs`` runs only inside ``astro build``, and that
    build runs only when the docs workflow's path filter matches. Every
    repository-root file the module reads therefore belongs in both filters:
    otherwise a pull request touching one of them merges without the guard
    that reads it ever running, and the failure surfaces later on an unrelated
    pull request that did not change the file at fault.
    """
    module = _CITATION_MODULE.read_text(encoding="utf-8")
    read = {
        match.group(1)
        for match in re.finditer(r"readFileSync\(\s*root\(\s*'([^']+)'", module)
    }
    root_files = sorted(name for name in read if "/" not in name)
    assert root_files, "citation.mjs reads no repository-root file; update this test"

    for event in ("push", "pull_request"):
        listed = _trigger_paths(_DOCS_WORKFLOW, event)
        missing = [name for name in root_files if name not in listed]
        assert not missing, (
            f"docs.yml {event} trigger does not list {missing}, "
            "so the site build that reads them cannot run on a change to them"
        )


def test_the_jit_job_comment_quotes_the_numba_floor_that_is_declared() -> None:
    """The 3.14 job explains itself by quoting pyproject's numba floor."""
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    declared = {
        spec
        for extra in pyproject["project"]["optional-dependencies"].values()
        for spec in extra
        if spec.replace(" ", "").startswith("numba")
    }
    assert declared, "pyproject.toml declares no numba requirement"

    quoted = set(
        re.findall(
            r"numba\s*>=\s*[0-9.]+", _PYTHON_WORKFLOW.read_text(encoding="utf-8")
        )
    )
    assert quoted, "the workflow quotes no numba floor; update this test"
    stale = {spec for spec in quoted if spec.replace(" ", "") not in declared}
    assert not stale, (
        f"python-app.yml quotes {sorted(stale)} but pyproject.toml declares {sorted(declared)}"
    )


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("docs/**/index.md", "docs/signals/index.md", True),
        ("docs/**/index.md", "docs/reference/api/index.md", True),
        ("docs/**/index.md", "docs/README.md", False),
        ("docs/!(reference)/**/index.md", "docs/signals/index.md", True),
        ("docs/!(reference)/**/index.md", "docs/signals/filters/index.md", True),
        ("docs/!(reference)/**/index.md", "docs/reference/api/index.md", False),
        ("docs/!(reference)/**/index.md", "docs/reference/theory/index.md", False),
        ("docs/**", "docs/start/about.md", True),
    ],
)
def test_the_glob_translation_agrees_with_the_matcher_coderabbit_uses(
    pattern: str, path: str, expected: bool
) -> None:
    """Pinned against minimatch and picomatch, run by hand on these cases."""
    assert bool(_glob_to_regex(pattern).fullmatch(path)) is expected
